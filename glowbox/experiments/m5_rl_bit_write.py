"""Milestone 5, Task 2 with a learned write: an RL agent chooses the pulses that store a bit pattern.

    uv run python -m glowbox.experiments.m5_rl_bit_write [--config PATH] [--quick] [--report-dir DIR]

Milestone 3's write-delay-read protocol (glowbox.learn.bit_write): an L-bit pattern is written by a block of MW
segments (one per device line), held through a dark delay and recalled from the time-resolved readout by the
Bayes-optimal readout of the codebook. The write of each pattern is

* ``rl``: chosen by a PPO agent (glowbox.learn.rl.CodebookEnv: one-step episodes, reward = L + log2 P(pattern |
  readout) under the codebook's Bayes-optimal readout), the policy's mean action at test; ``rl_init``: the same
  policy before training (control);
* ``gradient``: the same parametrisation and objective trained by gradient through the differentiable surrogate
  (milestone 4's learner);
* ``random``: the parametrisation fixed at random (milestone 3/4's reservoir).

Evaluation at the converged quadrature, Poisson noise, the Bayes-optimal bitwise readout. Quantum test: everything
per seed with every MW frequency + ``detuned_offset_hz`` (retrained there). Timing: every duration shifted
(readout frozen, i.e. calibrated on the unshifted write, and recalibrated), the same area change through the
amplitude, scrambled phases, pulse areas. Check: the seed-0 RL codebook on the QuTiP reference.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import multiprocessing as mp
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
from glowbox.learn.autoencoder import DiamondChannel, SystemSpec
from glowbox.learn.bit_memory import all_patterns, device_lines_hz
from glowbox.learn.bit_write import (
    action_to_params,
    evaluate_codebook,
    template_separation_sigma,
    train_gradient,
)
from glowbox.learn.encoder import (
    BlockParams,
    Layout,
    PulseEncoder,
    area_matched_amplitude,
    pulse_areas_pi,
    readout_features,
    suffix_to_numpy,
)
from glowbox.learn.field_data import FieldPrior
from glowbox.learn.reservoir import full_sequence
from glowbox.learn.rl import CodebookEnv, GaussianPolicy, PPOConfig, train_ppo
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "m5_rl_bit_write.yaml"
REPORT_NAME = "m5-rl-bit-write"
ENCODERS = ("rl", "gradient", "random", "rl_init")
COLORS = {"rl": "#eb6834", "gradient": "#2a78d6", "random": "#8a8985", "rl_init": "#f2b48f"}
LABELS = {
    "rl": "RL agent (PPO, black box)",
    "gradient": "gradient through the surrogate (M4 learner)",
    "random": "fixed random encoder (M3/M4 reservoir)",
    "rl_init": "RL policy before training",
}


def _tgen(device: torch.device, *keys: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(np.random.default_rng(list(keys)).integers(2**62)))
    return g


class WriteTask:
    """``task(condition, seed) -> metrics``; keeps codebooks, histories and probe rows."""

    def __init__(self, cfg: dict[str, Any], run: RunLog, device: torch.device) -> None:
        self.cfg, self.run, self.device = cfg, run, device
        self.base_seed = int(cfg["seed"])
        lc = dict(cfg["layout"])
        self.layout = Layout(n_blocks=1, **lc)
        tc = cfg["task"]
        self.n_bits = int(tc["n_bits"])
        self.patterns = torch.as_tensor(all_patterns(self.n_bits), device=device)
        self.train_shots = float(tc["train_shots"])
        self.eval_shots = [float(s) for s in tc["eval_shots"]]
        self.headline = float(tc["headline_shots"])
        sc = cfg["system"]
        dc = cfg["device"]
        self.sys = SystemSpec(int(sc["init_ns"]), int(sc["settle_ns"]), ModelOptions(n_detuning=int(dc["n_detuning"])))
        self.sys_eval = SystemSpec(
            int(sc["init_ns"]), int(sc["settle_ns"]), ModelOptions(n_detuning=int(dc["eval_n_detuning"]))
        )
        self.rows: list[dict[str, Any]] = []
        self.train_rows: list[dict[str, Any]] = []
        self.timing_rows: list[dict[str, Any]] = []
        self.area_rows: list[dict[str, Any]] = []
        self.codebooks: dict[tuple[int, str, str], BlockParams] = {}
        self.templates: dict[tuple[int, str, str], torch.Tensor] = {}
        self.envs: dict[int, EnvParams] = {}
        self.no_line_margin_hz = math.inf
        self.seconds: dict[str, float] = {"rl": 0.0, "gradient": 0.0, "eval": 0.0, "timing": 0.0}
        self.data_dir = run.dir / "data"
        self.data_dir.mkdir(exist_ok=True)

    def env(self, seed: int) -> EnvParams:
        if seed not in self.envs:
            pc: dict[str, Any] = {k: tuple(v) for k, v in self.cfg["device"]["prior"].items()}
            env = FieldPrior(**pc).sample_env(np.random.default_rng([self.base_seed, seed, 1]))
            b = tuple(float(x) for x in self.cfg["device"]["bias_field_t"])
            self.envs[seed] = dataclasses.replace(env, b_field_t=b)  # type: ignore[arg-type]
        return self.envs[seed]

    def base_hz(self, env: EnvParams) -> torch.Tensor:
        lines = np.sort(device_lines_hz(env).ravel())
        s = self.layout.n_segments
        base = np.array([[lines[j % lines.size] for j in range(s)]])
        return torch.as_tensor(base, dtype=torch.float64, device=self.device)

    def n_action(self) -> int:
        return self.layout.n_blocks * self.layout.n_segments * 4

    # --- learners -----------------------------------------------------------------------------------------------
    def train_rl(
        self, cond: Condition, seed: int, ch: DiamondChannel, base: torch.Tensor
    ) -> tuple[GaussianPolicy, GaussianPolicy]:
        rc = self.cfg["rl"]
        torch.manual_seed(int(np.random.default_rng([self.base_seed, seed, 11]).integers(2**31)))
        pol = GaussianPolicy(
            self.n_bits, self.n_action(), tuple(int(h) for h in rc["hidden"]), init_log_std=float(rc["init_log_std"])
        ).to(self.device)
        pol0 = copy.deepcopy(pol)

        def counts_fn(z: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return ch.counts(action_to_params(z.float(), self.layout, base))

        env = CodebookEnv(
            self.patterns,
            int(rc["codebooks"]),
            counts_fn,
            self.train_shots,
            int(rc["n_noise"]),
            _tgen(self.device, self.base_seed, seed, 12),
        )
        pcfg = PPOConfig(
            iterations=int(rc["iterations"]),
            lr=float(rc["lr"]),
            epochs=int(rc["epochs"]),
            minibatches=int(rc["minibatches"]),
            clip=float(rc["clip"]),
            ent_coef=float(rc["ent_coef"]),
            target_kl=float(rc["target_kl"]),
            gamma=1.0,
            gae_lambda=1.0,
        )

        def log(it: int, st: dict[str, float]) -> None:
            self.train_rows.append({"cond": cond.name, "seed": seed, "learner": "rl", "step": it, "bits": st["return"]})
            if it % 50 == 0 or it == pcfg.iterations:
                print(f"[m5-write] rl {cond.name} s{seed} it {it}: bits {st['return']:.3f}", flush=True)

        t0 = time.perf_counter()
        train_ppo(env, pol, pcfg, 1, _tgen(self.device, self.base_seed, seed, 13), log=log)
        self.seconds["rl"] += time.perf_counter() - t0
        return pol, pol0

    def train_grad(self, cond: Condition, seed: int, ch: DiamondChannel, base: torch.Tensor) -> PulseEncoder:
        gc = self.cfg["gradient"]
        torch.manual_seed(int(np.random.default_rng([self.base_seed, seed, 21]).integers(2**31)))
        enc = PulseEncoder(
            self.layout,
            base.cpu().numpy(),
            self.n_bits,
            tuple(int(h) for h in gc["hidden"]),
            None,
            float(gc["out_scale"]),
        ).to(self.device)

        def log(step: int, bits: float) -> None:
            self.train_rows.append({"cond": cond.name, "seed": seed, "learner": "gradient", "step": step, "bits": bits})

        t0 = time.perf_counter()
        train_gradient(
            enc,
            ch,
            self.patterns,
            steps=int(gc["steps"]),
            lr=float(gc["lr"]),
            shots=self.train_shots,
            n_noise=int(gc["n_noise"]),
            generator=_tgen(self.device, self.base_seed, seed, 22),
            log=log,
            log_every=max(1, int(gc["steps"]) // 50),
        )
        self.seconds["gradient"] += time.perf_counter() - t0
        return enc

    # --- one condition and seed ----------------------------------------------------------------------------------
    def __call__(self, cond: Condition, seed: int) -> dict[str, float]:
        t_start = time.perf_counter()
        env = self.env(seed)
        base = self.base_hz(env)
        if cond.detuned:
            f = base.cpu().numpy().ravel() + cond.mw_offset_hz
            span = float(self.layout.span_hz)
            ok, margin = check_no_line_driven(np.concatenate([f - span, f + span]), np.zeros(3), env)
            self.no_line_margin_hz = min(self.no_line_margin_hz, margin)
            if not ok:
                raise RuntimeError(f"detuned condition drives a line (margin {margin / 1e6:.1f} MHz)")
        ch = DiamondChannel(env, self.layout, self.sys, self.device, offset_hz=cond.mw_offset_hz)
        ch_eval = DiamondChannel(env, self.layout, self.sys_eval, self.device, offset_hz=cond.mw_offset_hz)
        u = 2.0 * self.patterns.float() - 1.0
        pol, pol0 = self.train_rl(cond, seed, ch, base)
        enc = self.train_grad(cond, seed, ch, base)
        rnd = PulseEncoder(
            self.layout, base.cpu().numpy(), self.n_bits, tuple(int(h) for h in self.cfg["random"]["hidden"])
        ).to(self.device)
        rnd.randomize_(float(self.cfg["random"]["gain"]), _tgen(self.device, self.base_seed, seed, 31))
        with torch.no_grad():
            books = {
                "rl": action_to_params(pol.mean(u), self.layout, base),
                "rl_init": action_to_params(pol0.mean(u), self.layout, base),
                "gradient": enc(u),
                "random": rnd(u),
            }
        out: dict[str, float] = {}
        t0 = time.perf_counter()
        for name, params in books.items():
            params = params.detach()
            with torch.no_grad():
                tmpl = ch_eval.counts(params)
            tmpl = tmpl.double()
            self.codebooks[(seed, cond.name, name)] = params
            self.templates[(seed, cond.name, name)] = tmpl
            for s in self.eval_shots:
                met = evaluate_codebook(
                    tmpl,
                    self.patterns,
                    s,
                    int(self.cfg["task"]["draws_per_pattern"]),
                    _tgen(self.device, seed, int(s), 41),
                )
                tag = f"s{int(s)}"
                out[f"{name}_bits_{tag}"] = met["recalled_bits"]
                out[f"{name}_bitacc_{tag}"] = met["bit_acc"]
                out[f"{name}_patacc_{tag}"] = met["pattern_acc"]
                self.rows.append({"cond": cond.name, "seed": seed, "encoder": name, "shots": s, **met})
            out[f"{name}_sep_sigma"] = template_separation_sigma(tmpl, self.headline)
            if not cond.detuned:
                areas = pulse_areas_pi(params, env, self.layout.grid_ns).reshape(-1)
                on = areas[np.asarray(params.duration_ns.reshape(-1).cpu().numpy()) >= self.layout.grid_ns / 2]
                self.area_rows.append(
                    {
                        "seed": seed,
                        "encoder": name,
                        "median_area_pi": float(np.median(on)) if on.size else 0.0,
                        "frac_rabi": float(np.mean((on > 0.25) & (on < 4.0))) if on.size else 0.0,
                        "median_amp": float(params.amplitude.median()),
                        "median_dur_ns": float(params.duration_ns.median()),
                    }
                )
        self.seconds["eval"] += time.perf_counter() - t0
        h = f"s{int(self.headline)}"
        for name in books:
            out[f"{name}_bits"] = out[f"{name}_bits_{h}"]
            out[f"{name}_bitacc"] = out[f"{name}_bitacc_{h}"]
        if not cond.detuned:
            self.timing_probe(seed, ch_eval, books)
        print(
            f"[m5-write] {cond.name} s{seed}: "
            + ", ".join(f"{n}={out[n + '_bits']:.3f}" for n in books)
            + f" bits of {self.n_bits} at {int(self.headline)} shots ({time.perf_counter() - t_start:.0f} s)",
            flush=True,
        )
        return out

    def timing_probe(self, seed: int, ch: DiamondChannel, books: dict[str, BlockParams]) -> None:
        t0 = time.perf_counter()
        draws = int(self.cfg["task"]["draws_per_pattern"])
        for name in ("rl", "gradient", "random"):
            p = books[name].detach()
            tmpl = self.templates[(seed, "resonant", name)]
            base_bits = evaluate_codebook(tmpl, self.patterns, self.headline, draws, _tgen(self.device, seed, 51))[
                "recalled_bits"
            ]
            for sh in self.cfg["timing"]["shifts_ns"]:
                sh = float(sh)
                with torch.no_grad():
                    shifted = ch.counts(p, duration_shift_ns=sh).double()
                    area = ch.counts(area_matched_amplitude(p, self.layout, sh)).double()
                for kind, data, cal in (
                    ("frozen", shifted, tmpl),
                    ("recalibrated", shifted, shifted),
                    ("area", area, tmpl),
                ):
                    bits = evaluate_codebook(
                        cal, self.patterns, self.headline, draws, _tgen(self.device, seed, 52), data
                    )["recalled_bits"]
                    self.timing_rows.append(
                        {"seed": seed, "encoder": name, "probe": kind, "shift_ns": sh, "bits": bits, "bits0": base_bits}
                    )
            g = _tgen(self.device, seed, 53)
            ph = (torch.rand(p.phase_rad.shape, generator=g, device=self.device) * 2 - 1) * math.pi
            with torch.no_grad():
                scr = ch.counts(
                    BlockParams(p.amplitude, ph.to(p.phase_rad.dtype), p.frequency_hz, p.duration_ns)
                ).double()
            for kind, cal in (("phase_frozen", tmpl), ("phase_recalibrated", scr)):
                bits = evaluate_codebook(cal, self.patterns, self.headline, draws, _tgen(self.device, seed, 54), scr)[
                    "recalled_bits"
                ]
                self.timing_rows.append(
                    {"seed": seed, "encoder": name, "probe": kind, "shift_ns": 0.0, "bits": bits, "bits0": base_bits}
                )
        self.seconds["timing"] += time.perf_counter() - t0


# --- reference check -------------------------------------------------------------------------------------------------

_REF: dict[str, Any] = {}


def _ref_init(n_detuning: int) -> None:
    from glowbox.sim.reference import ReferenceSimulator

    _REF["sim"] = ReferenceSimulator(n_detuning=n_detuning)


def _ref_one(item: tuple[dict[str, Any], dict[str, Any]]) -> np.ndarray:
    from glowbox.control.pulse import PulseSequence

    seq, env = PulseSequence.from_dict(item[0]), EnvParams.from_dict(item[1])
    return np.asarray(_REF["sim"].expected_rate_cps(seq, env))


def run_reference_check(task: WriteTask) -> dict[str, Any]:
    """The seed-0 resonant RL codebook as full PulseSequences on the QuTiP reference (evaluation quadrature)."""
    from glowbox.learn.reservoir import init_prefix

    params = task.codebooks[(0, "resonant", "rl")]
    env = task.env(0)
    ly = task.layout
    sc = task.cfg["system"]
    pre = init_prefix(int(sc["init_ns"]), int(sc["settle_ns"]), grid_ns=ly.grid_ns)
    rows = suffix_to_numpy(params, ly).unbatch()
    items = [(full_sequence(pre, r).to_dict(), env.to_dict()) for r in rows]
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        int(task.cfg["reference"]["workers"]),
        initializer=_ref_init,
        initargs=(int(task.cfg["device"]["eval_n_detuning"]),),
    ) as pool:
        rates = pool.map(_ref_one, items)
    dt = ly.grid_ns * 1e-9
    counts = torch.as_tensor(np.stack([np.asarray(r[-ly.n_bins :]) * dt for r in rates]))
    feats = readout_features(counts, ly).double().to(task.device)
    sur = task.templates[(0, "resonant", "rl")]
    spread = float(sur.std(0).mean())
    draws = int(task.cfg["task"]["draws_per_pattern"])
    g = _tgen(task.device, 0, 61)
    ref_bits = evaluate_codebook(feats, task.patterns, task.headline, draws, g)["recalled_bits"]
    cross = evaluate_codebook(sur, task.patterns, task.headline, draws, _tgen(task.device, 0, 61), feats)[
        "recalled_bits"
    ]
    sur_bits = evaluate_codebook(sur, task.patterns, task.headline, draws, _tgen(task.device, 0, 61))["recalled_bits"]
    return {
        "n_patterns": len(items),
        "max_abs_feature_diff": float((feats - sur).abs().max()),
        "max_diff_rel_pattern_spread": float((feats - sur).abs().max()) / spread,
        "bits_reference": ref_bits,
        "bits_surrogate": sur_bits,
        "bits_reference_data_surrogate_calibration": cross,
        "seconds": time.perf_counter() - t0,
    }


# --- plots -----------------------------------------------------------------------------------------------------------


def _sel(df: pd.DataFrame, **kw: Any) -> pd.DataFrame:
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def plot_recall(task: WriteTask) -> Figure:
    df = pd.DataFrame(task.rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, lab in zip(axes, ("recalled_bits", "bit_acc"), ("recalled bits", "bit accuracy"), strict=True):
        for name in ENCODERS:
            for cond, ls in (("resonant", "-"), ("detuned", "--")):
                d = _sel(df, encoder=name, cond=cond)
                xs = sorted(set(d.shots))
                vals = [_sel(d, shots=s)[metric].to_numpy() for s in xs]
                _band(ax, xs, vals, COLORS[name], ls, f"{LABELS[name]}, {cond}" if metric == "recalled_bits" else "")
        ax.set_xscale("log")
        ax.set_xlabel("shots per recall")
        ax.set_ylabel(lab)
        _style(ax)
    axes[0].axhline(task.n_bits, color="#bdbcb6", ls=":", lw=1)
    axes[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    return fig


def plot_training(task: WriteTask) -> Figure:
    df = pd.DataFrame(task.train_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, learner in zip(axes, ("rl", "gradient"), strict=True):
        for cond, color in (("resonant", "#2a78d6"), ("detuned", "#eb6834")):
            for s in sorted(set(df.seed)):
                d = _sel(df, learner=learner, cond=cond, seed=s)
                y = d.bits.rolling(10 if learner == "rl" else 1, min_periods=1).mean()
                ax.plot(d.step, y, color=color, lw=1, label=cond if s == 0 else None)
        ax.set_xlabel("PPO iteration" if learner == "rl" else "gradient step")
        ax.set_ylabel("training objective (bits per pattern)")
        ax.set_title(LABELS[learner])
        ax.legend(fontsize=8)
        _style(ax)
    fig.tight_layout()
    return fig


def plot_codebooks(task: WriteTask) -> Figure:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    env = task.env(0)
    for ax, name in zip(axes[:3], ("rl", "gradient", "random"), strict=True):
        p = task.codebooks.get((0, "resonant", name))
        if p is None:
            continue
        a = pulse_areas_pi(p, env, task.layout.grid_ns)[:, 0]  # (P, S)
        im = ax.imshow(a, aspect="auto", cmap="viridis", vmin=0, vmax=max(1.0, float(a.max())))
        ax.set_xlabel("segment (device line)")
        ax.set_ylabel("pattern")
        ax.set_title(f"{name}: pulse areas (pi), seed 0", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[3]
    for k, name in enumerate(("rl", "gradient", "random")):
        t = task.templates.get((0, "resonant", name))
        if t is None:
            continue
        g = t[:, :3].sum(-1).cpu().numpy()
        ax.scatter(np.full(g.size, k), g, color=COLORS[name], s=12)
    ax.set_xticks(range(3), ["RL", "gradient", "random"])
    ax.set_ylabel("first 300 ns counts per shot")
    ax.set_title("templates (seed 0)", fontsize=9)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_timing(task: WriteTask) -> Figure:
    df = pd.DataFrame(task.timing_rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for ax, probe in zip(axes, ("frozen", "recalibrated", "area"), strict=True):
        for name in ("rl", "gradient", "random"):
            d = _sel(df, encoder=name, probe=probe)
            xs = sorted(set(d.shift_ns))
            vals = [(_sel(d, shift_ns=s).bits - _sel(d, shift_ns=s).bits0).to_numpy() for s in xs]
            _band(ax, xs, vals, COLORS[name], "-", LABELS[name])
        ax.axhline(0, color="#8a8985", lw=0.8)
        ax.set_xlabel("shift (ns)")
        ax.set_title(
            {
                "frozen": "durations shifted, readout frozen",
                "recalibrated": "durations shifted, recalibrated",
                "area": "same area change via amplitude",
            }[probe],
            fontsize=9,
        )
        _style(ax)
    axes[0].set_ylabel("change in recalled bits")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    return fig


# --- report ----------------------------------------------------------------------------------------------------------


def timing_verdicts(task: WriteTask) -> dict[str, Any]:
    df = pd.DataFrame(task.timing_rows)
    out: dict[str, Any] = {}
    big = max(abs(float(s)) for s in task.cfg["timing"]["shifts_ns"])
    for name in ("rl", "gradient", "random"):
        r: dict[str, Any] = {}
        base = float(_sel(df, encoder=name).groupby("seed").bits0.first().mean())
        for probe in ("frozen", "recalibrated", "area"):
            d = _sel(df, encoder=name, probe=probe)
            d = d[d.shift_ns.abs() == big]
            per = d.groupby("seed").apply(lambda x: float((x.bits - x.bits0).mean()))
            r[probe] = t_ci(per.to_numpy())
        for probe in ("phase_frozen", "phase_recalibrated"):
            d = _sel(df, encoder=name, probe=probe)
            r[probe] = t_ci((d.bits - d.bits0).to_numpy())
        # a probe "costs" when the CI of the change in recalled bits lies below -10 % of the unperturbed bits
        thr = -0.10 * base
        r["bits0"] = base
        r["relies_on_pulse_timing"] = bool(r["frozen"][2] < thr)
        r["information_needs_timing"] = bool(r["recalibrated"][2] < thr)
        r["area_explains"] = bool(r["area"][0] <= 0.5 * r["frozen"][0])
        r["uses_phases"] = bool(r["phase_frozen"][2] < thr)
        ar = pd.DataFrame(task.area_rows)
        a = _sel(ar, encoder=name)
        r["median_area_pi"] = t_ci(a.median_area_pi.to_numpy())
        r["frac_rabi"] = t_ci(a.frac_rabi.to_numpy())
        out[name] = r
    out["strategy_learns_timing"] = bool(out["rl"]["relies_on_pulse_timing"] and out["rl"]["median_area_pi"][0] > 0.25)
    return out


def report_body(task: WriteTask, qt: QuantumTestResult, tv: dict[str, Any], ref: dict[str, Any]) -> str:
    L: list[str] = []
    h = int(task.headline)
    ly = task.layout
    L.append(
        f"**Task 2 with a learned write.** {task.n_bits}-bit patterns are written by {ly.n_segments} MW segments (one "
        f"per device line, {ly.slot_ns} ns slots), held for a {ly.delay_ns} ns dark delay and recalled from a "
        f"{ly.readout_ns} ns time-resolved readout by the Bayes-optimal readout of the codebook. The write of every "
        "pattern is chosen by a PPO agent (black box), by gradient through the differentiable surrogate, or fixed at "
        "random. Simulation only; no claim of quantum advantage: a gap only says that the performance depends on "
        "driving the spin resonance."
    )
    L += ["", "## Result", ""]
    L.append(f"Recalled bits (of {task.n_bits}) at {h} shots per recall, mean over {len(qt.seeds)} seeds [95 % t CI]:")
    L += ["", "| write | resonant | detuned | gap (resonant - detuned) | gap significant | bit accuracy, resonant |"]
    L.append("| --- | --- | --- | --- | --- | --- |")
    for n in ENCODERS:
        s = qt.summary(f"{n}_bits")
        L.append(
            f"| {LABELS[n]} | {_ci(s['resonant'])} | {_ci(s['detuned'])} | {_ci(s['gap'])} | "
            f"{'yes' if qt.gap_significant(f'{n}_bits') else 'no'} | "
            f"{_ci(t_ci(qt.values['resonant'][f'{n}_bitacc']))} |"
        )
    L.append("")

    def paired(a: str, b: str) -> str:
        va = np.asarray(qt.values["resonant"][f"{a}_bits"])
        vb = np.asarray(qt.values["resonant"][f"{b}_bits"])
        return _ci(t_ci(va - vb))

    L.append(
        "- Paired differences in recalled bits, resonant: RL - random "
        f"{paired('rl', 'random')}; gradient - RL {paired('gradient', 'rl')}; RL - RL before training "
        f"{paired('rl', 'rl_init')}."
    )
    sig = [n for n in ENCODERS if qt.gap_significant(f"{n}_bits")]
    det = [t_ci(qt.values["detuned"][f"{n}_bits"])[0] for n in ENCODERS]
    L.append(
        f"- **Quantum test**: with every MW frequency shifted by {task.cfg['detuned_offset_hz'] / 1e6:.0f} MHz (no "
        f"line within {task.no_line_margin_hz / 1e6:.0f} MHz of any segment's frequency range; the learners retrained "
        f"there) the recalled bits are between {min(det):.3g} and {max(det):.3g}; the gap is significant for "
        f"{len(sig)} of {len(ENCODERS)} writes ({', '.join(sig) or 'none'})."
    )
    L.append(
        f"- Timing: the RL agent's strategy uses microsecond-scale (Rabi-scale) timing: "
        f"**{'yes' if tv['strategy_learns_timing'] else 'no'}** (details below)."
    )
    L += ["", "## Recall vs shots", ""]
    L.append("Recalled bits, resonant / detuned, mean [95 % t CI]:")
    hdr = ["shots", *[LABELS[n] for n in ENCODERS]]
    L += ["", "| " + " | ".join(hdr) + " |", "|" + " --- |" * len(hdr)]
    for shots in task.eval_shots:
        cells = [str(int(shots))]
        for n in ENCODERS:
            cr = _ci(t_ci(qt.values["resonant"][f"{n}_bits_s{int(shots)}"]))
            cd = _ci(t_ci(qt.values["detuned"][f"{n}_bits_s{int(shots)}"]))
            cells.append(f"{cr} / {cd}")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append(
        f"Template separation at {h} shots (median distance to the nearest other pattern in Poisson standard "
        "deviations, resonant): "
        + "; ".join(f"{n} {_ci(t_ci(qt.values['resonant'][f'{n}_sep_sigma']))}" for n in ENCODERS)
        + "."
    )
    L += ["", "## Timing", ""]
    L.append(
        "Change in recalled bits at the largest shift (+-"
        f"{max(abs(float(s)) for s in task.cfg['timing']['shifts_ns']):g} ns, mean of both signs) [t CI over seeds]: "
        "durations shifted with the readout calibrated on the unshifted write (frozen), recalibrated on the shifted "
        "write, the same area change through the amplitude (frozen), and every segment phase scrambled; 'bits' is the "
        "unperturbed recall:"
    )
    L += [
        "",
        "| write | bits | frozen | recalibrated | equal area | phases scrambled (frozen) | phases scrambled (recal.) "
        "| median area (pi) | areas in (pi/4, 4 pi) |",
    ]
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for n in ("rl", "gradient", "random"):
        r = tv[n]
        L.append(
            f"| {LABELS[n]} | {r['bits0']:.3g} | {_ci(r['frozen'], '{:+.2f}')} | {_ci(r['recalibrated'], '{:+.2f}')} | "
            f"{_ci(r['area'], '{:+.2f}')} | {_ci(r['phase_frozen'], '{:+.2f}')} | "
            f"{_ci(r['phase_recalibrated'], '{:+.2f}')} | {_ci(r['median_area_pi'], '{:.2f}')} | "
            f"{_ci(r['frac_rabi'], '{:.2f}')} |"
        )
    L.append("")
    for n in ("rl", "gradient"):
        r = tv[n]
        yn = {True: "yes", False: "no"}
        L.append(
            f"- {LABELS[n]}: relies on its pulse timing (frozen, > 10 % fewer bits, CI): "
            f"{yn[r['relies_on_pulse_timing']]}; the information needs it (recalibrated still > 10 % fewer): "
            f"{yn[r['information_needs_timing']]}; the equal-area control costs at least half as much (pulse area, "
            f"not timing as such): {yn[r['area_explains']]}; coherent phases matter (frozen, scrambled): "
            f"{yn[r['uses_phases']]}."
        )
    L.append(
        "- `strategy_learns_timing` (RL relies on its pulse timing, Rabi-scale areas) = "
        f"**{tv['strategy_learns_timing']}**."
    )
    L += ["", "## Reference check", ""]
    L.append(
        f"- The seed-0 RL codebook ({ref['n_patterns']} patterns) as full PulseSequences on the QuTiP reference: "
        f"templates within {ref['max_abs_feature_diff']:.2g} counts per shot "
        f"({ref['max_diff_rel_pattern_spread']:.2g} of the spread between patterns); recalled bits at {h} shots: "
        f"reference {ref['bits_reference']:.3f}, surrogate {ref['bits_surrogate']:.3f}, reference recalls decoded "
        "with the surrogate calibration "
        f"{ref['bits_reference_data_surrogate_calibration']:.3f} ({ref['seconds']:.0f} s)."
    )
    L += ["", "## Quantum test", ""]
    L.append(qt.to_markdown([f"{n}_bits" for n in ENCODERS] + [f"{n}_bitacc" for n in ENCODERS]))
    L += ["", "## Setup", ""]
    c = task.cfg
    rc, gc = c["rl"], c["gradient"]
    L.append(
        "- Device per seed: the static field (1.2, 0.7, 2.5) mT (eight lines), nuisance drawn from the milestone-2 "
        "ranges, calibrated. Shot: init 1500 ns, dark 1000 ns, the write block, the delay, the readout in "
        f"{ly.window_ns} ns windows. Segment parameters: amplitude [{ly.amp_lo:g}, {ly.amp_hi:g}], phase, frequency "
        f"within +-{ly.span_hz / 1e6:g} MHz of its line, duration [{ly.min_ns:g}, {ly.max_ns:g}] ns on the grid."
    )
    L.append(
        f"- RL: PPO, {rc['iterations']} iterations x {rc['codebooks']} codebooks ({rc['codebooks']} x "
        f"{2**task.n_bits} one-step episodes), {rc['n_noise']} noise draws per reward, lr {rc['lr']}, {rc['epochs']} "
        f"epochs x {rc['minibatches']} minibatches, clip {rc['clip']}, Gaussian policy over the {task.n_action()} "
        f"pre-activations (initial std {math.exp(float(rc['init_log_std'])):.2g}), MLP {rc['hidden']}; trained at "
        f"{int(task.train_shots)} shots with {c['device']['n_detuning']} T2* nodes."
    )
    L.append(
        f"- Gradient: {gc['steps']} Adam steps (lr {gc['lr']}, cosine) on the same objective with reparametrised "
        f"Gaussian shot noise ({gc['n_noise']} draws), encoder MLP {gc['hidden']}; random: gain {c['random']['gain']}."
    )
    L.append(
        f"- Evaluation: templates at {c['device']['eval_n_detuning']} T2* nodes, {c['task']['draws_per_pattern']} "
        f"Poisson recalls per pattern, bitwise MAP. Wall time: RL {task.seconds['rl'] / 60:.1f} min, gradient "
        f"{task.seconds['gradient'] / 60:.1f} min, evaluation {task.seconds['eval'] / 60:.1f} min, probes "
        f"{task.seconds['timing'] / 60:.1f} min."
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
        task = WriteTask(cfg, run, device)
        conds = (RESONANT, detuned_condition(float(cfg["detuned_offset_hz"])))
        qt = run_quantum_test(task, [int(s) for s in cfg["seeds"]], conds)
        qt.higher_is_better = {m: True for m in qt.metrics()}
        print("[m5-write] reference check", flush=True)
        ref = run_reference_check(task)
        tv = timing_verdicts(task)
        for name, rows in (
            ("recall", task.rows),
            ("training", task.train_rows),
            ("timing", task.timing_rows),
            ("areas", task.area_rows),
        ):
            pd.DataFrame(rows).to_parquet(run.dir / f"{name}.parquet")
        (run.dir / "reference_check.json").write_text(json.dumps(ref), encoding="utf-8")
        (run.dir / "timing_verdicts.json").write_text(json.dumps(tv, default=str), encoding="utf-8")
        (run.dir / "quantum_test.json").write_text(
            json.dumps({"seeds": qt.seeds, "values": qt.values}), encoding="utf-8"
        )
        torch.save(
            {f"{k[0]}_{k[1]}_{k[2]}": v.numpy() for k, v in task.codebooks.items()}, task.data_dir / "codebooks.pt"
        )
        figs = {
            "recall": plot_recall(task),
            "training": plot_training(task),
            "codebooks": plot_codebooks(task),
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
        for n in ENCODERS:
            sm = qt.summary(f"{n}_bits")
            summary[f"{n}_bits_resonant"] = sm["resonant"][0]
            summary[f"{n}_bits_detuned"] = sm["detuned"][0]
            summary[f"{n}_gap"], summary[f"{n}_gap_lo"], summary[f"{n}_gap_hi"] = sm["gap"]
        summary["strategy_learns_timing"] = float(tv["strategy_learns_timing"])
        summary["ref_bits"] = ref["bits_reference"]
        summary["wall_seconds"] = time.perf_counter() - t_start
        run.log_metrics(step=0, **{k: v for k, v in summary.items() if isinstance(v, int | float)})
        body = report_body(task, qt, tv, ref)
    rdir = report_dir if report_dir is not None else run.dir / "report"
    title = "Milestone 5, Task 2: an RL agent learns the write pulses of a bit pattern"
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
