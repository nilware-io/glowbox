"""Milestone 5, Task 1: the learner chooses the measurements (adaptive Bayesian design and an RL agent).

    uv run python -m glowbox.experiments.m5_adaptive_field [--config PATH] [--quick] [--report-dir DIR]

A calibrated device (EnvParams defaults) with a hidden field from milestone 2's prior. Every step is a block of
``shots`` shots of one pulsed-ODMR measurement, an action of :class:`~glowbox.learn.measurement.MeasurementSpec`
(MW frequency on a 2 MHz grid x pulse duration ~pi/2, pi, 2 pi, 4 pi). The counts come from the full surrogate
(precomputed noiseless means per true field + Poisson noise); every learner's model of the device is the
per-orientation emulator (:class:`~glowbox.learn.measurement.AxisTable`). Policies, all with the same particle
posterior (:mod:`glowbox.learn.adaptive`) for the estimate:

* ``fixed``: milestone 2/4's fixed sweep (every frequency once at the ~pi pulse) in an interleaved order, and the
  same sweep at the other durations; ``sweep_full``: the fixed sweep at fewer shots per point (every budget covers
  the whole band);
* ``eig``: greedy expected information gain over every (frequency, duration); ``eig_pi``: the same restricted to
  the pi pulse (does choosing the duration matter?); ``random``: uniformly random actions;
* ``rl``: a PPO agent (:mod:`glowbox.learn.rl`) trained on fields of the seed's training pool, greedy at test;
  ``rl_init``: the same network before training, greedy (its features already carry the posterior's predictive
  spread, so an untrained argmax is not random: this control measures what PPO adds);
* ``cnn``: milestone 2's CNN decoder on the fixed sweep (no posterior; trained on a bank of the training pool).

Quantum test: everything per seed with every MW frequency + ``detuned_offset_hz`` (the agent retrained there).
Timing: which durations (pulse areas) the adaptive learners pick, and every learner with all MW durations shifted
in the device while its model assumes the nominal ones. Check: the seed-0 agent run against the QuTiP reference
through ``Simulator.run()`` (:class:`~glowbox.learn.rl.SimulatorBackend`).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from glowbox.analysis.quantum_test import (
    RESONANT,
    Condition,
    QuantumTestResult,
    check_no_line_driven,
    detuned_condition,
    run_quantum_test,
    t_ci,
)
from glowbox.analysis.runlog import RunLog, write_report
from glowbox.experiments.m1_odmr_rabi import load_config
from glowbox.experiments.m2_field_estimation import _style
from glowbox.experiments.m3_bit_memory import _band, _ci
from glowbox.learn.adaptive import (
    EIGDesign,
    EpisodeLog,
    FieldPriorSpec,
    FixedSweep,
    Policy,
    PosteriorConfig,
    RandomDesign,
    TableBackend,
    run_episodes,
)
from glowbox.learn.decoder import ConvDecoder, DecoderConfig, TrainConfig, predict, train_decoder
from glowbox.learn.field_data import TargetScaler, features, log_shots, poisson_counts, sample_shots
from glowbox.learn.measurement import AxisTable, MeasurementSpec, emulator_error, measurement_means
from glowbox.learn.rl import FieldSearchEnv, PPOConfig, RLDesign, SetPolicy, SimulatorBackend, train_ppo
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "m5_adaptive_field.yaml"
REPORT_NAME = "m5-adaptive-field"
COLORS = {
    "fixed": "#8e44ad",
    "sweep_full": "#b07cc6",
    "eig": "#2a78d6",
    "eig_pi": "#7fb0ea",
    "rl": "#eb6834",
    "random": "#8a8985",
    "rl_init": "#f2b48f",
    "cnn": "#1baf7a",
    "prior": "#bdbcb6",
}
LABELS = {
    "fixed": "fixed sweep (interleaved, pi pulse)",
    "sweep_full": "fixed sweep, whole band at fewer shots",
    "eig": "adaptive: expected information gain",
    "eig_pi": "adaptive EIG, pi pulse only",
    "rl": "RL agent (PPO)",
    "random": "random measurements",
    "rl_init": "RL network before training (greedy)",
    "cnn": "M2 CNN on the fixed sweep",
}


def measurement_spec(cfg: dict[str, Any]) -> MeasurementSpec:
    m = cfg["measurement"]
    return MeasurementSpec(
        f_start_hz=float(m["f_start_hz"]),
        f_stop_hz=float(m["f_stop_hz"]),
        step_hz=float(m["step_hz"]),
        durations_ns=tuple(int(d) for d in m["durations_ns"]),
        amplitude=float(m["amplitude"]),
        init_ns=int(m["init_ns"]),
        settle_ns=int(m["settle_ns"]),
        windows_ns=tuple(int(w) for w in m["windows_ns"]),
    )


def model_options(cfg: dict[str, Any]) -> ModelOptions:
    n = cfg["numerics"]
    return ModelOptions(
        n_detuning=int(n["n_detuning"]),
        detuning_rule=str(n["detuning_rule"]),
        detuning_spacing=float(n["detuning_spacing"]),
    )


def posterior_config(cfg: dict[str, Any]) -> PosteriorConfig:
    p = cfg["posterior"]
    return PosteriorConfig(
        n_particles=int(p["n_particles"]),
        ess_frac=float(p["ess_frac"]),
        mh_steps=int(p["mh_steps"]),
        mh_scale=float(p["mh_scale"]),
        n_predict=int(p["n_predict"]),
    )


def shifted(spec: MeasurementSpec, shift_ns: int) -> MeasurementSpec:
    return dataclasses.replace(spec, durations_ns=tuple(max(spec.grid_ns, d + shift_ns) for d in spec.durations_ns))


class AdaptiveTask:
    """``task(condition, seed) -> metrics`` for the quantum-test harness; keeps curves, choices and agents."""

    def __init__(self, cfg: dict[str, Any], run: RunLog, device: torch.device) -> None:
        self.cfg, self.run, self.device = cfg, run, device
        self.base_seed = int(cfg["seed"])
        self.spec = measurement_spec(cfg)
        self.opt = model_options(cfg)
        self.post_cfg = posterior_config(cfg)
        self.env = EnvParams()
        self.prior = FieldPriorSpec(float(cfg["prior"]["b_min_t"]), float(cfg["prior"]["b_max_t"]))
        self.prior_rmse_t = self.prior.rmse_t()
        b = cfg["budget"]
        self.n_steps, self.shots = int(b["n_steps"]), int(b["shots"])
        self.curve_steps = [int(t) for t in b["curve_steps"]]
        self.tables: dict[str, AxisTable] = {}
        self.rows: list[dict[str, Any]] = []  # curves
        self.choice_rows: list[dict[str, Any]] = []
        self.timing_rows: list[dict[str, Any]] = []
        self.train_rows: list[dict[str, Any]] = []
        self.emu_rows: list[dict[str, Any]] = []
        self.agents: dict[tuple[int, str], SetPolicy] = {}
        self.test_fields: dict[int, torch.Tensor] = {}
        self.test_means: dict[tuple[int, str], torch.Tensor] = {}
        self.no_line_margin_hz = math.inf
        self.seconds: dict[str, float] = {"tables": 0.0, "truths": 0.0, "episodes": 0.0, "rl": 0.0, "cnn": 0.0}
        self.data_dir = run.dir / "data"
        self.data_dir.mkdir(exist_ok=True)

    def _gen(self, seed: int, salt: int) -> torch.Generator:
        g = torch.Generator(device=self.device)
        g.manual_seed((self.base_seed * 1_000_003 + seed * 7919 + salt) % (2**62))
        return g

    def cond_spec(self, cond: Condition) -> MeasurementSpec:
        return self.spec.detuned(cond.mw_offset_hz) if cond.detuned else self.spec

    def table(self, cond: Condition) -> AxisTable:
        if cond.name not in self.tables:
            n = self.cfg["numerics"]
            t0 = time.perf_counter()
            tab = AxisTable.build(
                self.cond_spec(cond),
                self.env,
                self.opt,
                self.device,
                b_max_t=float(n["table_b_max_t"]),
                d_par_t=float(n["table_d_par_t"]),
                d_perp_t=float(n["table_d_perp_t"]),
                n_az=int(n["table_n_az"]),
            )
            self.seconds["tables"] += time.perf_counter() - t0
            self.tables[cond.name] = tab
            print(f"[m5-field] table {cond.name} {tuple(tab.table.shape)} {time.perf_counter() - t0:.0f} s", flush=True)
        return self.tables[cond.name]

    def fields(self, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(test fields, training pool incl. validation), tesla, float64 on the device."""
        d = self.cfg["data"]
        g = self._gen(seed, 1)
        test = self.prior.sample((int(d["n_test"]),), g, self.device)
        pool = self.prior.sample((int(d["n_pool"]) + int(d["n_val"]),), g, self.device)
        return test, pool

    def truth(self, spec: MeasurementSpec, fields: torch.Tensor) -> torch.Tensor:
        t0 = time.perf_counter()
        m = measurement_means(spec, self.env, fields.cpu().numpy(), self.opt, self.device)
        self.seconds["truths"] += time.perf_counter() - t0
        return m

    def episodes(
        self,
        policy: Policy,
        means: torch.Tensor,
        truths: torch.Tensor,
        table: AxisTable,
        gen: torch.Generator,
        **kw: Any,
    ) -> EpisodeLog:
        t0 = time.perf_counter()
        n_steps = int(kw.get("n_steps", self.n_steps))
        shots = int(kw.get("shots", self.shots))
        log = run_episodes(
            policy, TableBackend(means, gen), truths, table, self.prior, self.post_cfg, n_steps, shots, gen
        )
        self.seconds["episodes"] += time.perf_counter() - t0
        return log

    # --- learners ----------------------------------------------------------------------------------------------------
    def train_agent(
        self, cond: Condition, seed: int, table: AxisTable, pool: torch.Tensor, pool_means: torch.Tensor
    ) -> tuple[SetPolicy, SetPolicy]:
        """(trained agent, the same network before training: the control for what PPO adds)."""
        rc = self.cfg["rl"]
        n_train = int(self.cfg["data"]["n_pool"])
        tr_f, tr_m = pool[:n_train], pool_means[:n_train]

        def sampler(n: int, gen: torch.Generator) -> tuple[torch.Tensor, TableBackend]:
            idx = torch.randint(0, tr_f.shape[0], (n,), generator=gen, device=self.device)
            return tr_f[idx], TableBackend(tr_m[idx], gen)

        train_post = dataclasses.replace(self.post_cfg, n_particles=int(rc["train_particles"]))
        env = FieldSearchEnv(
            table, self.prior, sampler, int(rc["num_envs"]), self.n_steps, self.shots, train_post, self._gen(seed, 20)
        )
        torch.manual_seed((self.base_seed + 31 * seed) % (2**31))
        pol = SetPolicy(env.n_features, 2, int(rc["hidden"])).to(self.device)
        pol0 = copy.deepcopy(pol).eval()
        pcfg = PPOConfig(
            iterations=int(rc["iterations"]),
            lr=float(rc["lr"]),
            epochs=int(rc["epochs"]),
            minibatches=int(rc["minibatches"]),
            clip=float(rc["clip"]),
            gae_lambda=float(rc["gae_lambda"]),
            ent_coef=float(rc["ent_coef"]),
            target_kl=float(rc["target_kl"]),
        )

        def log(it: int, st: dict[str, float]) -> None:
            row = {"cond": cond.name, "seed": seed, "iteration": it, **st}
            self.train_rows.append(row)
            if it % 5 == 0 or it == pcfg.iterations:
                rm = math.sqrt(st.get("sq_err", float("nan"))) * 1e6
                print(
                    f"[m5-field] rl {cond.name} s{seed} it {it}: return {st['return']:.2f} rmse {rm:.0f} uT", flush=True
                )

        t0 = time.perf_counter()
        train_ppo(env, pol, pcfg, self.n_steps, self._gen(seed, 21), log=log)
        self.seconds["rl"] += time.perf_counter() - t0
        pol.eval()
        return pol, pol0

    def cnn(
        self,
        cond: Condition,
        seed: int,
        pool: torch.Tensor,
        pool_means: torch.Tensor,
        test: torch.Tensor,
        test_means: torch.Tensor,
    ) -> dict[int, float]:
        """Milestone 2's CNN on the fixed sweep; RMSE (uT) on the test fields at every curve budget."""
        cc = self.cfg["cnn"]
        n_train = int(self.cfg["data"]["n_pool"])
        di = int(self.cfg["policies"]["fixed_dur_index"])
        sl = slice(di * self.spec.n_freq, (di + 1) * self.spec.n_freq)

        def feat_means(m: torch.Tensor) -> torch.Tensor:  # (N, A, W) -> (N, W, F) float32
            return m[:, sl].permute(0, 2, 1).float().contiguous()

        tr, va = feat_means(pool_means[:n_train]), feat_means(pool_means[n_train:])
        scaler = TargetScaler.fit(pool[:n_train].cpu().numpy())
        y_tr = torch.as_tensor(scaler.encode(pool[:n_train].cpu().numpy()), dtype=torch.float32, device=self.device)
        y_va = torch.as_tensor(scaler.encode(pool[n_train:].cpu().numpy()), dtype=torch.float32, device=self.device)
        gen = self._gen(seed, 40)
        lo, hi = float(cc["shots_min"]), float(cc["shots_max"])
        batch = int(cc["batch"])

        def sample(step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            idx = torch.randint(0, tr.shape[0], (batch,), generator=gen, device=self.device)
            s = sample_shots(batch, lo, hi, gen, self.device)
            return features(poisson_counts(tr[idx], s, gen)), log_shots(s), y_tr[idx]

        vgen = self._gen(seed, 41)
        vs = sample_shots(va.shape[0], lo, hi, vgen, self.device)
        val = [(features(poisson_counts(va, vs, vgen)), log_shots(vs), y_va)]
        dcfg = DecoderConfig(
            mix_channels=int(cc["mix_channels"]),
            channels=tuple(int(c) for c in cc["channels"]),
            kernel=int(cc["kernel"]),
            hidden=int(cc["hidden"]),
        )
        torch.manual_seed((self.base_seed + 17 * seed) % (2**31))
        model = ConvDecoder(self.spec.n_windows, self.spec.n_freq, 3, dcfg).to(self.device)
        t0 = time.perf_counter()
        train_decoder(
            model,
            sample,
            val,
            TrainConfig(
                steps=int(cc["steps"]),
                batch=batch,
                lr=float(cc["lr"]),
                weight_decay=float(cc["weight_decay"]),
                eval_every=int(cc["eval_every"]),
            ),
        )
        self.seconds["cnn"] += time.perf_counter() - t0
        te = feat_means(test_means)
        out = {}
        egen = self._gen(seed, 42)
        for t in self.curve_steps:
            per_point = max(1, round(t * self.shots / self.spec.n_freq))
            s = torch.full((te.shape[0],), float(per_point), device=self.device)
            m, _ = predict(model, features(poisson_counts(te, s, egen)), log_shots(s))
            b, _ = scaler.decode(m.cpu().numpy().astype(np.float64))
            out[t] = float(np.sqrt(np.mean(np.sum((b - test.cpu().numpy()) ** 2, axis=1)))) * 1e6
        return out

    # --- one condition and seed --------------------------------------------------------------------------------------
    def __call__(self, cond: Condition, seed: int) -> dict[str, float]:
        t_start = time.perf_counter()
        spec = self.cond_spec(cond)
        table = self.table(cond)
        test, pool = self.fields(seed)
        if cond.detuned:
            f_all, _ = spec.action_grid()
            for b in test[::7].cpu().numpy():
                ok, margin = check_no_line_driven(np.unique(f_all), b, self.env)
                self.no_line_margin_hz = min(self.no_line_margin_hz, margin)
                if not ok:
                    raise RuntimeError(f"detuned condition drives a line (margin {margin / 1e6:.1f} MHz)")
        test_means = self.truth(spec, test)
        pool_means = self.truth(spec, pool)
        self.test_fields[seed] = test
        self.test_means[(seed, cond.name)] = test_means
        emu = emulator_error(table, test_means, test.cpu().numpy())
        self.emu_rows.append({"cond": cond.name, "seed": seed, **emu})
        out: dict[str, float] = {}
        logs: dict[str, EpisodeLog] = {}
        di = int(self.cfg["policies"]["fixed_dur_index"])
        method = str(self.cfg["policies"]["eig_method"])
        allowed_pi = torch.zeros(spec.n_actions, dtype=torch.bool, device=self.device)
        allowed_pi[di * spec.n_freq : (di + 1) * spec.n_freq] = True
        agent, agent0 = self.train_agent(cond, seed, table, pool, pool_means)
        self.agents[(seed, cond.name)] = agent
        policies: dict[str, Policy] = {
            "fixed": FixedSweep(spec, di),
            "eig": EIGDesign(method),
            "eig_pi": EIGDesign(method, allowed_pi),
            "rl": RLDesign(agent, self.prior_rmse_t / math.sqrt(3.0)),
            "rl_init": RLDesign(agent0, self.prior_rmse_t / math.sqrt(3.0)),
            "random": RandomDesign(spec, self._gen(seed, 9)),
        }
        if not cond.detuned:  # the sweep at the other durations: resonant only
            for d in range(spec.n_dur):
                if d != di:
                    policies[f"fixed_d{d}"] = FixedSweep(spec, d)
        for k, (name, pol) in enumerate(policies.items()):
            logs[name] = self.episodes(pol, test_means, test, table, self._gen(seed, 100 + k))
        # the fixed sweep covering the whole band at each budget
        full: dict[int, float] = {}
        for t in self.curve_steps:
            per_point = max(1, round(t * self.shots / spec.n_freq))
            lg = self.episodes(
                FixedSweep(spec, di),
                test_means,
                test,
                table,
                self._gen(seed, 200 + t),
                n_steps=spec.n_freq,
                shots=per_point,
            )
            full[t] = lg.rmse_t() * 1e6
        cnn = self.cnn(cond, seed, pool, pool_means, test, test_means)
        prior_ut = self.prior_rmse_t * 1e6
        for name, lg in logs.items():
            rm = np.sqrt(lg.sq_err.mean(axis=1)) * 1e6
            for t in range(lg.sq_err.shape[0]):
                self.rows.append(
                    {
                        "cond": cond.name,
                        "seed": seed,
                        "policy": name,
                        "step": t,
                        "shots": t * self.shots,
                        "rmse_ut": rm[t],
                    }
                )
            out[f"{name}_rmse_ut"] = float(rm[-1])
            out[f"{name}_skill"] = 1.0 - float(rm[-1]) / prior_ut
            for t in self.curve_steps:
                out[f"{name}_rmse_ut_t{t}"] = float(rm[t])
            cal = float(lg.sq_err[-1].mean() / max(1e-30, (3.0 * lg.post_std[-1] ** 2).mean()))
            out[f"{name}_calibration"] = cal
            for thr in self.cfg["budget"]["thresholds_ut"]:
                hit = np.nonzero(rm <= float(thr))[0]
                out[f"{name}_shots_to_{int(thr)}ut"] = float(hit[0] * self.shots) if hit.size else math.nan
            if name in ("eig", "rl", "rl_init", "eig_pi", "random"):
                _, dj = spec.split(lg.actions.reshape(-1))
                for d in range(spec.n_dur):
                    self.choice_rows.append(
                        {
                            "cond": cond.name,
                            "seed": seed,
                            "policy": name,
                            "dur_index": d,
                            "frac": float(np.mean(dj == d)),
                        }
                    )
                if seed == 0:
                    np.save(self.data_dir / f"actions_s0_{cond.name}_{name}.npy", lg.actions)
        for t in self.curve_steps:
            out[f"sweep_full_rmse_ut_t{t}"] = full[t]
            out[f"cnn_rmse_ut_t{t}"] = cnn[t]
            self.rows.append(
                {
                    "cond": cond.name,
                    "seed": seed,
                    "policy": "sweep_full",
                    "step": t,
                    "shots": t * self.shots,
                    "rmse_ut": full[t],
                }
            )
            self.rows.append(
                {
                    "cond": cond.name,
                    "seed": seed,
                    "policy": "cnn",
                    "step": t,
                    "shots": t * self.shots,
                    "rmse_ut": cnn[t],
                }
            )
        out["cnn_rmse_ut"] = cnn[self.curve_steps[-1]]
        out["cnn_skill"] = 1.0 - out["cnn_rmse_ut"] / prior_ut
        out["prior_rmse_ut"] = prior_ut
        if not cond.detuned:
            self.timing_probe(seed, spec, table, test, policies, logs)
        print(
            f"[m5-field] {cond.name} s{seed}: "
            + ", ".join(f"{k}={out[k + '_rmse_ut']:.0f}" for k in ("fixed", "eig", "eig_pi", "rl", "rl_init", "cnn"))
            + f" uT ({time.perf_counter() - t_start:.0f} s)",
            flush=True,
        )
        return out

    def timing_probe(
        self,
        seed: int,
        spec: MeasurementSpec,
        table: AxisTable,
        test: torch.Tensor,
        policies: dict[str, Policy],
        logs: dict[str, EpisodeLog],
    ) -> None:
        """Every MW duration shifted in the device (truth), the learners' model unaware: RMSE change."""
        for shift in self.cfg["timing"]["shifts_ns"]:
            sh = int(shift)
            m = self.truth(shifted(spec, sh), test)
            for k, name in enumerate(("fixed", "eig", "rl")):
                lg = self.episodes(policies[name], m, test, table, self._gen(seed, 300 + 10 * k + sh))
                self.timing_rows.append(
                    {
                        "seed": seed,
                        "policy": name,
                        "shift_ns": sh,
                        "rmse_ut": lg.rmse_t() * 1e6,
                        "rmse0_ut": logs[name].rmse_t() * 1e6,
                    }
                )


# --- reference check -----------------------------------------------------------------------------------------------


def run_reference_check(task: AdaptiveTask) -> dict[str, Any]:
    """The seed-0 resonant agent and EIG design with the QuTiP reference as the device, through Simulator.run()."""
    from glowbox.sim.reference import ReferenceSimulator

    n = int(task.cfg["reference"]["n_fields"])
    o = task.opt
    rule: Any = o.detuning_rule
    ref = ReferenceSimulator(n_detuning=o.n_detuning, detuning_rule=rule, detuning_spacing=o.detuning_spacing)
    truths = task.test_fields[0][:n]
    sur_means = task.test_means[(0, "resonant")][:n]
    table = task.tables["resonant"]
    envs = [
        dataclasses.replace(task.env, b_field_t=(float(b[0]), float(b[1]), float(b[2]))) for b in truths.cpu().numpy()
    ]
    out: dict[str, Any] = {"n_fields": n}
    t0 = time.perf_counter()
    for name, pol in (
        ("rl", RLDesign(task.agents[(0, "resonant")], task.prior_rmse_t / math.sqrt(3.0))),
        ("eig", EIGDesign(str(task.cfg["policies"]["eig_method"]))),
    ):
        backend = SimulatorBackend(ref, task.spec, envs, np.random.default_rng(7))
        g = task._gen(0, 500)
        lg = run_episodes(pol, backend, truths, table, task.prior, task.post_cfg, task.n_steps, task.shots, g)
        diffs = []
        for (e, a), mean in backend._cache.items():
            diffs.append(np.abs(mean - sur_means[e, a].cpu().numpy()).max())
        spread = float(sur_means.std(dim=1).mean())
        out[f"{name}_rmse_ref_ut"] = lg.rmse_t() * 1e6
        g2 = task._gen(0, 500)
        lg2 = run_episodes(
            pol, TableBackend(sur_means, g2), truths, table, task.prior, task.post_cfg, task.n_steps, task.shots, g2
        )
        out[f"{name}_rmse_sur_ut"] = lg2.rmse_t() * 1e6
        out[f"{name}_ref_calls"] = backend.calls
        out[f"{name}_max_abs_mean_diff"] = float(max(diffs))
        out[f"{name}_max_diff_rel_signal"] = float(max(diffs)) / spread
    out["seconds"] = time.perf_counter() - t0
    return out


# --- plots -----------------------------------------------------------------------------------------------------------


def _sel(df: pd.DataFrame, **kw: Any) -> pd.DataFrame:
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def plot_curves(task: AdaptiveTask) -> Figure:
    df = pd.DataFrame(task.rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, cond in zip(axes, ("resonant", "detuned"), strict=True):
        d = _sel(df, cond=cond)
        for name in ("fixed", "eig", "eig_pi", "rl", "rl_init", "random"):
            dd = _sel(d, policy=name)
            steps = sorted(set(dd.step) - {0})
            xs = [s * task.shots for s in steps]
            vals = [_sel(dd, step=s).rmse_ut.to_numpy() for s in steps]
            _band(ax, xs, vals, COLORS[name], "-", LABELS[name], marker="")
        for name, mk in (("sweep_full", "s"), ("cnn", "D")):
            dd = _sel(d, policy=name)
            steps = sorted(set(dd.step))
            xs = [s * task.shots for s in steps]
            vals = [_sel(dd, step=s).rmse_ut.to_numpy() for s in steps]
            _band(ax, xs, vals, COLORS[name], "--", LABELS[name], marker=mk)
        ax.axhline(task.prior_rmse_t * 1e6, color=COLORS["prior"], ls=":", lw=1.5, label="prior only")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("total shots")
        ax.set_title(f"{cond} (95 % t CI over seeds)")
        _style(ax)
    axes[0].set_ylabel("RMSE of the field (uT)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig


def plot_choices(task: AdaptiveTask) -> Figure:
    spec = task.spec
    f = (spec.freqs_hz() - 2.87e9) / 1e6
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    areas = spec.pulse_area_pi(task.env)
    for ax, name in zip(axes[:2], ("eig", "rl"), strict=True):
        path = task.data_dir / f"actions_s0_resonant_{name}.npy"
        if not path.exists():
            continue
        acts = np.load(path)
        fi, di = spec.split(acts.reshape(-1))
        for d in range(spec.n_dur):
            h = np.bincount(fi[di == d], minlength=spec.n_freq) / acts.size
            ax.plot(f, h, lw=1.4, label=f"{spec.durations_ns[d]} ns ({areas[d]:.2f} pi)")
        ax.set_xlabel("MW frequency - 2.87 GHz (MHz)")
        ax.set_ylabel("fraction of all blocks")
        ax.set_title(f"{LABELS[name]}: choices (seed 0)")
        ax.legend(fontsize=7)
        _style(ax)
    ch = pd.DataFrame(task.choice_rows)
    ax = axes[2]
    if len(ch):
        width = 0.2
        for k, name in enumerate(("eig", "rl", "random")):
            dc = _sel(ch, cond="resonant", policy=name)
            mus = [t_ci(_sel(dc, dur_index=j).frac.to_numpy())[0] for j in range(spec.n_dur)]
            ax.bar(np.arange(spec.n_dur) + (k - 1) * width, mus, width, color=COLORS[name], label=LABELS[name])
        ax.set_xticks(range(spec.n_dur), [f"{d} ns\n{a:.2f} pi" for d, a in zip(spec.durations_ns, areas, strict=True)])
        ax.set_ylabel("fraction of blocks (mean over seeds)")
        ax.set_title("pulse durations chosen, resonant")
        ax.legend(fontsize=7)
        _style(ax)
    fig.tight_layout()
    return fig


def plot_quantum_test(task: AdaptiveTask, qt: QuantumTestResult) -> Figure:
    names = ["fixed", "eig", "eig_pi", "rl", "rl_init", "random", "cnn"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(names))
    for k, cond in enumerate(("resonant", "detuned")):
        mu, lo, hi = [], [], []
        for n in names:
            v, a, b = t_ci(qt.values[cond][f"{n}_rmse_ut"])
            mu.append(v)
            lo.append(v - a if math.isfinite(a) else 0)
            hi.append(b - v if math.isfinite(b) else 0)
        ax.bar(x + (k - 0.5) * 0.38, mu, 0.38, yerr=[lo, hi], label=cond, color=("#2a78d6", "#eb6834")[k], capsize=3)
    ax.axhline(task.prior_rmse_t * 1e6, color=COLORS["prior"], ls=":", label="prior only")
    ax.set_xticks(x, ["fixed", "EIG", "EIG pi", "RL", "RL untrained", "random", "CNN"])
    ax.set_yscale("log")
    ax.set_ylabel(f"RMSE at {task.n_steps * task.shots} shots (uT)")
    ax.legend(fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_training(task: AdaptiveTask) -> Figure:
    df = pd.DataFrame(task.train_rows)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for cond, color in (("resonant", "#2a78d6"), ("detuned", "#eb6834")):
        for s in sorted(set(df.seed)):
            d = _sel(df, cond=cond, seed=s)
            r = d["return"].rolling(10, min_periods=1).mean()
            ax.plot(d.iteration, r, color=color, lw=1, alpha=0.8, label=cond if s == 0 else None)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("episode return (log error reduction)")
    ax.legend(fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_timing(task: AdaptiveTask) -> Figure:
    df = pd.DataFrame(task.timing_rows)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for name in ("fixed", "eig", "rl"):
        d = _sel(df, policy=name)
        shifts = sorted(set(d.shift_ns))
        vals = [(_sel(d, shift_ns=s).rmse_ut / _sel(d, shift_ns=s).rmse0_ut - 1).to_numpy() for s in shifts]
        _band(ax, shifts, vals, COLORS[name], "-", LABELS[name])
    ax.axhline(0, color="#8a8985", lw=0.8)
    ax.set_xlabel("every MW duration shifted (ns), model unaware")
    ax.set_ylabel("relative RMSE increase")
    ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout()
    return fig


# --- report ----------------------------------------------------------------------------------------------------------


def timing_verdicts(task: AdaptiveTask) -> dict[str, Any]:
    df = pd.DataFrame(task.timing_rows)
    out: dict[str, Any] = {}
    shift = max(abs(int(s)) for s in task.cfg["timing"]["shifts_ns"])
    for name in ("fixed", "eig", "rl"):
        d = _sel(df, policy=name)
        rel = []
        for s in sorted(set(d.seed)):
            ds = _sel(d, seed=s)
            ds = ds[ds.shift_ns.abs() == shift]
            rel.append(float((ds.rmse_ut / ds.rmse0_ut - 1).mean()))
        v = t_ci(rel)
        out[name] = {"rel_at_max_shift": v, "relies_on_pulse_timing": bool(v[1] > 0.10)}
    ch = pd.DataFrame(task.choice_rows)
    areas = task.spec.pulse_area_pi(task.env)
    for name in ("eig", "rl"):
        d = _sel(ch, cond="resonant", policy=name)
        fr = {int(j): t_ci(_sel(d, dur_index=j).frac.to_numpy()) for j in range(task.spec.n_dur)}
        top = max(fr, key=lambda j: fr[j][0])
        out[name]["dur_fractions"] = fr
        out[name]["top_duration_ns"] = task.spec.durations_ns[top]
        out[name]["top_area_pi"] = float(areas[top])
        uniform = 1.0 / task.spec.n_dur
        out[name]["prefers_a_duration"] = bool(fr[top][1] > uniform)
    out["strategy_learns_timing"] = bool(out["rl"]["relies_on_pulse_timing"] and out["rl"]["prefers_a_duration"])
    return out


def _fmt(v: float) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:.3g}"


def report_body(task: AdaptiveTask, qt: QuantumTestResult, tv: dict[str, Any], ref: dict[str, Any]) -> str:
    T, S = task.n_steps, task.shots
    budget = T * S
    names = ["fixed", "eig", "eig_pi", "rl", "rl_init", "random", "cnn"]
    L: list[str] = []
    L.append(
        "**Task 1, the learner picks the measurements.** Each step is a block of "
        f"{S} shots of one pulsed-ODMR measurement (MW frequency on a {task.spec.step_hz / 1e6:g} MHz grid x pulse "
        f"duration {', '.join(str(d) for d in task.spec.durations_ns)} ns at amplitude {task.spec.amplitude:g}); "
        f"{T} steps = {budget} shots. Every policy updates the same particle posterior over the folded field; the "
        "estimate is its mean. Simulation only; no claim of quantum advantage: a gap only says that the performance "
        "depends on driving the spin resonance."
    )
    L += ["", "## Result", ""]
    L.append(f"RMSE of the folded field after {budget} shots, mean over {len(qt.seeds)} seeds [95 % Student-t CI]:")
    L += [
        "",
        "| policy | resonant | detuned | gap (detuned - resonant) | gap significant |",
        "| --- | --- | --- | --- | --- |",
    ]
    for n in names:
        s = qt.summary(f"{n}_rmse_ut")
        L.append(
            f"| {LABELS[n]} | {_ci(s['resonant'])} | {_ci(s['detuned'])} | {_ci(s['gap'])} | "
            f"{'yes' if qt.gap_significant(f'{n}_rmse_ut') else 'no'} |"
        )
    L.append(f"| prior only | {task.prior_rmse_t * 1e6:.0f} | {task.prior_rmse_t * 1e6:.0f} | | |")
    L.append("")

    # paired comparisons
    def paired(a: str, b: str, metric: str = "rmse_ut") -> str:
        va = np.asarray(qt.values["resonant"][f"{a}_{metric}"])
        vb = np.asarray(qt.values["resonant"][f"{b}_{metric}"])
        return _ci(t_ci(va - vb))

    L.append(
        "- Paired differences, resonant (positive = the second is better), uT: "
        f"fixed - EIG {paired('fixed', 'eig')}; fixed - RL {paired('fixed', 'rl')}; RL - EIG {paired('rl', 'eig')}; "
        f"RL before training - RL {paired('rl_init', 'rl')}; EIG pi only - EIG {paired('eig_pi', 'eig')}; "
        f"CNN - fixed {paired('cnn', 'fixed')}; "
        f"random - fixed {paired('random', 'fixed')}."
    )
    det = [t_ci(qt.values["detuned"][f"{n}_skill"])[0] for n in names]
    sig = [n for n in names if qt.gap_significant(f"{n}_rmse_ut")]
    L.append(
        f"- **Quantum test**: with every MW frequency shifted by {task.cfg['detuned_offset_hz'] / 1e6:.0f} MHz "
        f"(no line within {task.no_line_margin_hz / 1e6:.0f} MHz of any MW frequency; the agent and the CNN retrained "
        f"there) the detuned skill (1 - RMSE / prior RMSE) is between {min(det):.3g} and {max(det):.3g}; the gap is "
        f"significant for {len(sig)} of {len(names)} learners ({', '.join(sig) or 'none'})."
    )
    L.append(
        f"- Timing: the RL agent's strategy uses microsecond-scale (Rabi-scale) timing: "
        f"**{'yes' if tv['strategy_learns_timing'] else 'no'}** (details below)."
    )
    L += ["", "## Shots to accuracy", ""]
    L.append("RMSE (uT) vs total shots, resonant, mean [95 % t CI] over seeds:")
    hdr = ["total shots", *[LABELS[n] for n in ("fixed", "sweep_full", "eig", "eig_pi", "rl", "random", "cnn")]]
    L += ["", "| " + " | ".join(hdr) + " |", "|" + " --- |" * len(hdr)]
    for t in task.curve_steps:
        cells = [str(t * S)]
        for n in ("fixed", "sweep_full", "eig", "eig_pi", "rl", "random", "cnn"):
            cells.append(_ci(t_ci(qt.values["resonant"][f"{n}_rmse_ut_t{t}"])))
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append(
        "Shots until the RMSE over the test fields first falls below a threshold (mean over seeds; n/a = not "
        "within the budget):"
    )
    thr = [int(x) for x in task.cfg["budget"]["thresholds_ut"]]
    L += ["", "| policy | " + " | ".join(f"<= {x} uT" for x in thr) + " |", "|" + " --- |" * (len(thr) + 1)]
    for n in ("fixed", "eig", "eig_pi", "rl", "rl_init", "random"):
        cells = []
        for x in thr:
            v = np.asarray(qt.values["resonant"][f"{n}_shots_to_{x}ut"], dtype=float)
            cells.append(
                _fmt(float(np.mean(v)))
                if np.all(np.isfinite(v))
                else f"{_fmt(float(np.nanmean(v)))} ({int(np.isfinite(v).sum())}/{v.size} seeds)"
            )
        L.append(f"| {LABELS[n]} | " + " | ".join(cells) + " |")
    L.append("")
    L += ["## Fixed sweeps at each duration", ""]
    L.append("The interleaved fixed sweep at every pulse duration (resonant, full budget):")
    L += ["", "| duration (ns) | area (pi) | RMSE (uT) |", "| --- | --- | --- |"]
    areas = task.spec.pulse_area_pi(task.env)
    di = int(task.cfg["policies"]["fixed_dur_index"])
    for d in range(task.spec.n_dur):
        key = "fixed" if d == di else f"fixed_d{d}"
        L.append(
            f"| {task.spec.durations_ns[d]} | {areas[d]:.2f} | {_ci(t_ci(qt.values['resonant'][f'{key}_rmse_ut']))} |"
        )
    L.append("")
    L += ["## Timing", ""]
    L.append(
        "Durations (pulse areas at the nominal Rabi scale) chosen, fraction of all blocks, resonant [t CI over seeds]:"
    )
    L += [
        "",
        "| policy | "
        + " | ".join(f"{d} ns ({a:.2f} pi)" for d, a in zip(task.spec.durations_ns, areas, strict=True))
        + " |",
    ]
    L.append("|" + " --- |" * (task.spec.n_dur + 1))
    for n in ("eig", "rl"):
        fr = tv[n]["dur_fractions"]
        L.append(f"| {LABELS[n]} | " + " | ".join(_ci(fr[j], "{:.2f}") for j in range(task.spec.n_dur)) + " |")
    L.append("")
    df = pd.DataFrame(task.timing_rows)
    shifts = sorted(set(df.shift_ns))
    L.append(
        "Every MW duration shifted in the device while the learners' model assumes the nominal durations; "
        "relative RMSE increase [t CI]:"
    )
    L += ["", "| policy | " + " | ".join(f"{s:+d} ns" for s in shifts) + " |", "|" + " --- |" * (len(shifts) + 1)]
    for n in ("fixed", "eig", "rl"):
        cells = []
        for s in shifts:
            dt = _sel(df, policy=n, shift_ns=s)
            cells.append(_ci(t_ci((dt.rmse_ut / dt.rmse0_ut - 1).to_numpy()), "{:+.2f}"))
        L.append(f"| {LABELS[n]} | " + " | ".join(cells) + " |")
    L.append("")
    for n in ("fixed", "eig", "rl"):
        v = tv[n]
        extra = ""
        if "top_duration_ns" in v:
            pref = "yes" if v["prefers_a_duration"] else "no"
            extra = (
                f"; most chosen duration {v['top_duration_ns']} ns ({v['top_area_pi']:.2f} pi), preferred beyond "
                f"uniform: {pref}"
            )
        rel = "yes" if v["relies_on_pulse_timing"] else "no"
        L.append(f"- {LABELS[n]}: relies on its pulse timing (> 10 % worse at the largest shift, CI): {rel}{extra}.")
    L.append(
        "- `strategy_learns_timing` (RL: relies on pulse timing and prefers a pulse area) = "
        f"**{tv['strategy_learns_timing']}**."
    )
    L.append("")
    L += ["## Emulator and reference check", ""]
    emu = pd.DataFrame(task.emu_rows)
    e = _sel(emu, cond="resonant")
    L.append(
        "- Emulator (the learners' model) vs the full surrogate (the truth) on the test fields, resonant: RMS error "
        f"{e.rms_rel_signal.mean():.3g} of the signal spread over actions, max {e.max_rel_signal.max():.3g} "
        f"({e.max_abs.max():.3g} counts per shot; the shot noise of a block of {S} shots is ~{math.sqrt(3.4 / S):.3g} "
        "counts per shot)."
    )
    L.append(
        f"- QuTiP reference as the device through `Simulator.run()` (`SimulatorBackend`), seed-0 agent and EIG, "
        f"{ref['n_fields']} test fields: RL RMSE {ref['rl_rmse_ref_ut']:.1f} (reference) vs "
        f"{ref['rl_rmse_sur_ut']:.1f} uT (surrogate truth), EIG {ref['eig_rmse_ref_ut']:.1f} vs "
        f"{ref['eig_rmse_sur_ut']:.1f} uT; noiseless means of every action measured within "
        f"{ref['rl_max_diff_rel_signal']:.2g} (RL) and {ref['eig_max_diff_rel_signal']:.2g} (EIG) "
        f"of the signal spread ({ref['rl_ref_calls'] + ref['eig_ref_calls']} reference runs, {ref['seconds']:.0f} s)."
    )
    L.append("")
    L += ["## Quantum test", "", qt.to_markdown([f"{n}_rmse_ut" for n in names] + [f"{n}_skill" for n in names]), ""]
    L += ["## Setup", ""]
    c = task.cfg
    rc = c["rl"]
    L.append(
        f"- Device: `EnvParams()` defaults (T2* 1 us, Rabi full scale 10 MHz, beta 0.1, collection 3e-3, background "
        "1e4 counts/s, E = 0, 1000 NVs), calibrated (known). Hidden field: |B| uniform in "
        f"[{task.prior.b_min_t * 1e3:g}, {task.prior.b_max_t * 1e3:g}] mT, isotropic, folded into the wedge."
    )
    L.append(
        f"- Shot: init laser {task.spec.init_ns} ns, dark {task.spec.settle_ns} ns, MW pulse, readout windows "
        f"{task.spec.windows_ns} ns. Truths: surrogate, {task.opt.n_detuning} {task.opt.detuning_rule} T2* nodes "
        f"(spacing {task.opt.detuning_spacing:g})."
    )
    p = task.post_cfg
    L.append(
        f"- Posterior: {p.n_particles} particles, adaptive tempering, resample + {p.mh_steps} MH moves when ESS < "
        f"{p.ess_frac:g} N; EIG from {p.n_predict} posterior draws ({c['policies']['eig_method']} approximation)."
    )
    L.append(
        f"- RL: PPO, {rc['num_envs']} parallel episodes per iteration, {rc['iterations']} iterations, lr {rc['lr']}, "
        f"{rc['epochs']} epochs x {rc['minibatches']} minibatches, clip {rc['clip']}, entropy {rc['ent_coef']}, GAE "
        f"lambda {rc['gae_lambda']}, gamma 1; per-action set policy (hidden {rc['hidden']}); {rc['train_particles']} "
        "posterior particles while training; training fields from "
        f"the seed's pool of {c['data']['n_pool']}, test fields unseen."
    )
    cc = c["cnn"]
    L.append(
        f"- CNN: milestone 2's ConvDecoder on the fixed sweep (both windows per frequency), {cc['steps']} steps of "
        f"{cc['batch']}, shots per point log-uniform [{cc['shots_min']}, {cc['shots_max']}], bank = the training "
        f"pool, evaluated with the whole band at the budget's shots per point."
    )
    L.append(
        f"- {c['data']['n_test']} test fields per seed. Wall time: tables {task.seconds['tables'] / 60:.0f} min, "
        f"truths {task.seconds['truths'] / 60:.0f} min, episodes {task.seconds['episodes'] / 60:.0f} min, RL training "
        f"{task.seconds['rl'] / 60:.0f} min, CNN {task.seconds['cnn'] / 60:.0f} min."
    )
    L.append("")
    return "\n".join(L)


def _shrink_png(path: Path, colors: int = 128) -> None:
    from PIL import Image

    with Image.open(path) as im:
        q = im.convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q.save(path, optimize=True)


def main(
    cfg: dict[str, Any], output_root: Path | None = None, report_dir: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw: dict[str, Any] = {} if output_root is None else {"output_root": output_root}
    t_start = time.perf_counter()
    with RunLog(str(cfg["name"]), config=cfg, seed=int(cfg["seed"]), **kw) as run:
        task = AdaptiveTask(cfg, run, device)
        conds = (RESONANT, detuned_condition(float(cfg["detuned_offset_hz"])))
        qt = run_quantum_test(task, [int(s) for s in cfg["seeds"]], conds)
        qt.higher_is_better = {m: m.endswith("_skill") for m in qt.metrics()}
        print("[m5-field] reference check", flush=True)
        ref = run_reference_check(task)
        tv = timing_verdicts(task)
        for name, rows in (
            ("curves", task.rows),
            ("choices", task.choice_rows),
            ("timing", task.timing_rows),
            ("rl_training", task.train_rows),
            ("emulator", task.emu_rows),
        ):
            pd.DataFrame(rows).to_parquet(run.dir / f"{name}.parquet")
        (run.dir / "reference_check.json").write_text(json.dumps(ref), encoding="utf-8")
        (run.dir / "timing_verdicts.json").write_text(json.dumps(tv, default=str), encoding="utf-8")
        (run.dir / "quantum_test.json").write_text(
            json.dumps({"seeds": qt.seeds, "values": qt.values}), encoding="utf-8"
        )
        for (seed, cond), pol in task.agents.items():
            torch.save(pol.state_dict(), task.data_dir / f"agent_s{seed}_{cond}.pt")
        figs = {
            "shots-to-accuracy": plot_curves(task),
            "choices": plot_choices(task),
            "quantum-test": plot_quantum_test(task, qt),
            "rl-training": plot_training(task),
            "timing": plot_timing(task),
        }
        paths = [run.save_figure(f, name, dpi=80) for name, f in figs.items()]
        for pth in paths:
            _shrink_png(pth)
        for f in figs.values():
            plt.close(f)
        summary: dict[str, Any] = {
            "device": f"{device.type} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}".strip()
        }
        for n in ("fixed", "eig", "eig_pi", "rl", "rl_init", "random", "cnn"):
            sm = qt.summary(f"{n}_rmse_ut")
            summary[f"{n}_rmse_ut_resonant"] = sm["resonant"][0]
            summary[f"{n}_rmse_ut_detuned"] = sm["detuned"][0]
            summary[f"{n}_gap"], summary[f"{n}_gap_lo"], summary[f"{n}_gap_hi"] = sm["gap"]
        summary["strategy_learns_timing"] = float(tv["strategy_learns_timing"])
        summary["no_line_margin_mhz"] = task.no_line_margin_hz / 1e6
        summary["wall_seconds"] = time.perf_counter() - t_start
        run.log_metrics(step=0, **{k: v for k, v in summary.items() if isinstance(v, int | float)})
        body = report_body(task, qt, tv, ref)
    rdir = report_dir if report_dir is not None else run.dir / "report"
    title = "Milestone 5, Task 1: adaptive measurement and an RL agent choose the measurements"
    path = write_report(run.dir, title, body, figures=paths, report_dir=rdir, name=REPORT_NAME)
    return path, summary


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="small sizes and short training (smoke run)")
    p.add_argument("--report-dir", type=Path, default=None, help="where to write the report (e.g. docs/results)")
    p.add_argument("--output-root", type=Path, default=None, help="run directories (default: <repo>/outputs)")
    a = p.parse_args()
    path, summary = main(load_config(a.config, a.quick), a.output_root, a.report_dir)
    for k, v in summary.items():
        print(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}")
    print(f"report: {path}")


if __name__ == "__main__":
    cli()
