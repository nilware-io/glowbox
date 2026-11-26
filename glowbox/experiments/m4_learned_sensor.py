"""Milestone 4, Task 1 re-run: estimate a hidden magnetic field with a sensor protocol the encoder chooses.

    uv run python -m glowbox.experiments.m4_learned_sensor [--config PATH] [--quick] [--report-dir DIR]

The learned protocol (glowbox.learn.sensor) is ``n_points`` measurements; each is the fixed initialisation, a MW
block of two segments whose amplitude, phase, frequency (within a few MHz of the point's grid frequency) and duration
(10 ns grid) are free parameters, and a time-resolved readout in 100 ns windows. It starts as a pulsed ODMR
protocol (by default the tuned baseline: amplitude and pulse area chosen by a small validation sweep for this grid;
optionally milestone 2's 310 ns pi pulse at amplitude 0.2) and is trained end to end with a CNN decoder through the
differentiable surrogate (phase 1, fresh fields from the prior at every step). Then the protocol is
frozen, banks are simulated with milestone 2's numerics and a fresh CNN is trained on them (phase 2), exactly as
for the fixed CW and pulsed protocols on the same grid, so the comparison isolates the protocol.

The quantum test runs everything per seed twice: as designed, and with every MW frequency shifted by
``detuned_offset_hz`` (the learned protocol is trained in that condition too). Timing: the learned pulse areas and
durations, the harness's timing probe (every MW duration shifted, decoder frozen) and a phase-scramble probe, with
the tuned pulsed protocol as the comparison, an equal-area (amplitude) control and CNNs retrained on the shifted
protocols. Check: the learned protocol on the QuTiP reference.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
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
from torch import nn

from glowbox.analysis.field_fit import transition_lines_hz
from glowbox.analysis.quantum_test import (
    RESONANT,
    Condition,
    QuantumTestResult,
    check_no_line_driven,
    detuned_condition,
    run_quantum_test,
    t_ci,
    timing_sensitivity,
)
from glowbox.analysis.runlog import RunLog, write_report
from glowbox.experiments.m1_odmr_rabi import load_config
from glowbox.experiments.m2_field_estimation import _style, field_metrics
from glowbox.experiments.m3_bit_memory import _band, _ci
from glowbox.learn.decoder import ConvDecoder, DecoderConfig, TrainConfig, predict, train_decoder
from glowbox.learn.encoder import BlockParams, Layout, area_matched_amplitude, pulse_areas_pi, suffix_to_numpy
from glowbox.learn.field_data import (
    BankSpec,
    FieldBank,
    FieldPrior,
    SweepProtocol,
    TargetScaler,
    draw_fields_and_envs,
    features,
    log_shots,
    poisson_counts,
    sample_shots,
    simulate_bank,
)
from glowbox.learn.reservoir import full_sequence, init_prefix
from glowbox.learn.sensor import (
    LearnedProtocol,
    SensorSpec,
    pulsed_init,
    simulate_learned_bank,
    train_protocol,
    window_bank,
)
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate import SurrogateSimulator
from glowbox.sim.surrogate.model import ModelOptions

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "m4_learned_sensor.yaml"
REPORT_NAME = "m4-learned-sensor"
PROTOCOLS = ("learned", "tuned", "pulsed", "cw")
COLORS = {"learned": "#2a78d6", "tuned": "#8e44ad", "pulsed": "#eb6834", "cw": "#1baf7a", "prior": "#8a8985"}
LABELS = {
    "learned": "learned protocol",
    "tuned": "tuned pulsed ODMR",
    "pulsed": "M2 pulsed ODMR (untuned)",
    "cw": "fixed CW ODMR",
    "prior": "prior only",
}
#: Milestone 2's CNN results on its 200-point sweeps (docs/results/m2-field-estimation.md), RMSE in uT at 1000 shots
#: per point, for context only (more points, more training).
M2_CONTEXT_UT = {"cw": 11.9, "pulsed": 26.4}


def condition_for(cfg: dict[str, Any], name: str) -> Condition:
    return RESONANT if name == "resonant" else detuned_condition(float(cfg["detuned_offset_hz"]))


def sensor_spec(cfg: dict[str, Any]) -> SensorSpec:
    sc = cfg["sensor"]
    return SensorSpec(
        n_points=int(sc["n_points"]),
        f_start_hz=float(sc["f_start_hz"]),
        f_stop_hz=float(sc["f_stop_hz"]),
        layout=Layout(**sc["layout"]),
        init_ns=int(sc["init_ns"]),
        settle_ns=int(sc["settle_ns"]),
    )


def bank_spec(cfg: dict[str, Any]) -> BankSpec:
    pc: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg["prior"].items()}
    b = cfg["bank"]
    return BankSpec(
        prior=FieldPrior(**pc),
        fields_per_env=int(b["fields_per_env"]),
        n_detuning=int(b["n_detuning"]),
        detuning_rule=str(b["detuning_rule"]),
        detuning_spacing=float(b["detuning_spacing"]),
    )


def nominal_pi_ns(amplitude: float, rabi_full_scale_hz: float, grid_ns: int = 10) -> int:
    """A nominal pi pulse at ``amplitude`` (MW along [100], |m_perp| = sqrt(2/3)), rounded to the grid."""
    t = 1.0 / (2.0 * rabi_full_scale_hz * math.sqrt(2.0 / 3.0) * amplitude) * 1e9
    return max(grid_ns, round(t / grid_ns) * grid_ns)


def fixed_protocol(
    cfg: dict[str, Any],
    spec: SensorSpec,
    kind: str,
    offset_hz: float = 0.0,
    pulse: tuple[float, int] | None = None,
) -> SweepProtocol:
    """A fixed sweep on the learned protocol's grid: ``cw``, ``pulsed`` (milestone 2's pi pulse) or a pulsed sweep
    with the given ``pulse`` = (amplitude, pi_ns) (the tuned baseline)."""
    fc = cfg["fixed"]
    amp, pi_ns = (float(fc["pulsed_amplitude"]), int(fc["pi_ns"])) if pulse is None else pulse
    return SweepProtocol(
        kind="cw" if kind == "cw" else "pulsed",
        f_start_hz=spec.f_start_hz,
        f_stop_hz=spec.f_stop_hz,
        step_hz=(spec.f_stop_hz - spec.f_start_hz) / spec.n_points,
        mw_offset_hz=offset_hz,
        cw_amplitude=float(fc["cw_amplitude"]),
        cw_shot_ns=int(fc["cw_shot_ns"]),
        pulsed_amplitude=float(amp),
        pi_ns=int(pi_ns),
        init_ns=spec.init_ns,
        settle_ns=spec.settle_ns,
        readout_ns=spec.layout.readout_ns,
    )


@dataclasses.dataclass
class SeedData:
    fields_t: np.ndarray
    envs: list[EnvParams]


class SensorTask:
    """The quantum-test task function plus everything recorded along the way."""

    def __init__(self, cfg: dict[str, Any], run: RunLog, device: torch.device) -> None:
        self.cfg, self.run, self.device = cfg, run, device
        self.spec = sensor_spec(cfg)
        self.bspec = bank_spec(cfg)
        b = cfg["bank"]
        self.n_train, self.n_val, self.n_test = int(b["n_train"]), int(b["n_val"]), int(b["n_test"])
        fpe = self.bspec.fields_per_env
        if self.n_train % fpe or self.n_val % fpe:
            raise ValueError("n_train and n_val must be multiples of fields_per_env")
        self.levels = [int(s) for s in cfg["eval"]["shots"]]
        self.headline = int(cfg["eval"]["headline_shots"])
        self.budget = int(cfg["eval"]["budget_shots"])
        self.levels = sorted({*self.levels, self.headline, self.budget})
        self.base_seed = int(cfg["seed"])
        pt = cfg["protocol_train"]
        self.train_opt = ModelOptions(
            n_detuning=int(pt["n_detuning"]),
            detuning_rule=str(pt["detuning_rule"]),
            detuning_spacing=float(pt["detuning_spacing"]),
        )
        self.window_bins = self.spec.layout.window_bins
        self.rows: list[dict[str, Any]] = []
        self.timing_rows: list[dict[str, Any]] = []
        self.protocol_rows: list[dict[str, Any]] = []
        self.histories: dict[str, Any] = {}
        self.seed_data: dict[int, SeedData] = {}
        self.scalers: dict[int, TargetScaler] = {}
        self.test_banks: dict[tuple[str, str], FieldBank] = {}  # seed 0
        self.models: dict[tuple[int, str, str], nn.Module] = {}  # (seed, condition, protocol), resonant seeds kept
        self.learned: dict[tuple[int, str], LearnedProtocol] = {}
        self.no_line_margin_hz = math.inf
        self.seconds: dict[str, float] = {"tune": 0.0, "phase1": 0.0, "banks": 0.0, "cnn": 0.0, "timing": 0.0}
        self.tune_rows: list[dict[str, Any]] = []
        self.eqtime_rows: list[dict[str, Any]] = []
        self.rabi_nominal_hz = float(np.mean(self.bspec.prior.rabi_full_scale_hz))
        self.tuned: tuple[float, int] | None = None
        self.data_dir = run.dir / "data"
        self.data_dir.mkdir(exist_ok=True)

    def _seed(self, seed: int, salt: int) -> int:
        return (self.base_seed * 1_000_003 + seed * 7919 + salt) % (2**62)

    def _gen(self, seed: int, salt: int) -> torch.Generator:
        g = torch.Generator(device=self.device)
        g.manual_seed(self._seed(seed, salt))
        return g

    def data(self, seed: int) -> SeedData:
        if seed not in self.seed_data:
            rng = np.random.default_rng([self.base_seed, seed])
            fields, envs = draw_fields_and_envs(self.n_train + self.n_val + self.n_test, self.bspec, rng)
            self.seed_data[seed] = SeedData(fields, envs)
            np.savez(self.data_dir / f"fields_s{seed}.npz", fields_t=fields)
        return self.seed_data[seed]

    def split(self, bank: FieldBank) -> tuple[FieldBank, FieldBank, FieldBank]:
        a, b = self.n_train, self.n_train + self.n_val
        return bank.subset(np.arange(a)), bank.subset(np.arange(a, b)), bank.subset(np.arange(b, len(bank)))

    def cnn(self, n_time: int) -> ConvDecoder:
        dc = self.cfg["decoder"]
        cfg = DecoderConfig(
            mix_channels=int(dc["mix_channels"]),
            channels=tuple(int(c) for c in dc["channels"]),
            kernel=int(dc["kernel"]),
            hidden=int(dc["hidden"]),
        )
        return ConvDecoder(n_time, self.spec.n_points, 3, cfg).to(self.device)

    def protocol_for(self, kind: str, offset_hz: float = 0.0) -> SweepProtocol:
        if kind == "tuned":
            assert self.tuned is not None, "tune_pulse() first"
            return fixed_protocol(self.cfg, self.spec, kind, offset_hz, self.tuned)
        return fixed_protocol(self.cfg, self.spec, kind, offset_hz)

    def shot_ns(self, kind: str) -> int:
        """Wall time of one shot of one point (CW counts photons for all of it; pulsed and learned for the readout)."""
        return self.spec.shot_ns if kind == "learned" else self.protocol_for(kind).shot_ns

    # --- banks ------------------------------------------------------------------------------------------------
    def bank(self, kind: str, cond: Condition, sd: SeedData, lp: LearnedProtocol | None = None) -> FieldBank:
        t0 = time.perf_counter()
        workers = int(self.cfg["bank"].get("workers", 4))
        if kind == "learned":
            assert lp is not None
            bank = simulate_learned_bank(
                lp, sd.fields_t, sd.envs, self.bspec, self.device, offset_hz=cond.mw_offset_hz, workers=workers
            )
        else:
            bank = simulate_bank(
                self.protocol_for(kind, cond.mw_offset_hz),
                sd.fields_t,
                sd.envs,
                self.bspec,
                self.device,
                workers=workers,
            )
            if kind != "cw":
                bank = window_bank(bank, self.window_bins)
        self.seconds["banks"] += time.perf_counter() - t0
        return bank

    def check_detuned(self, freqs: np.ndarray, sd: SeedData) -> None:
        for i in range(0, len(sd.fields_t), 97):
            env = sd.envs[i // self.bspec.fields_per_env]
            ok, margin = check_no_line_driven(freqs, sd.fields_t[i], env)
            self.no_line_margin_hz = min(self.no_line_margin_hz, margin)
            if not ok:
                raise RuntimeError(f"detuned condition drives a line (margin {margin / 1e6:.1f} MHz)")

    # --- training ---------------------------------------------------------------------------------------------
    def train_cnn(self, bank: FieldBank, scaler: TargetScaler, seed: int, salt: int) -> tuple[nn.Module, Any]:
        train, val, _ = self.split(bank)
        return self.train_cnn_on(train, val, scaler, seed, salt)

    def train_cnn_on(
        self, train: FieldBank, val: FieldBank, scaler: TargetScaler, seed: int, salt: int
    ) -> tuple[nn.Module, Any]:
        """Milestone 2's CNN recipe on a train bank, validated on ``val``."""
        tc = self.cfg["train"]
        torch.manual_seed(self._seed(seed, salt))
        model = self.cnn(train.means.shape[1])
        gen, vgen = self._gen(seed, salt + 1), self._gen(seed, salt + 2)
        means = torch.as_tensor(train.means, device=self.device)
        y = torch.as_tensor(scaler.encode(train.fields_t), dtype=torch.float32, device=self.device)
        lo, hi, batch = float(tc["shots_min"]), float(tc["shots_max"]), int(tc["batch"])

        def sample(step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            idx = torch.randint(0, means.shape[0], (batch,), generator=gen, device=self.device)
            shots = sample_shots(batch, lo, hi, gen, self.device)
            return features(poisson_counts(means[idx], shots, gen)), log_shots(shots), y[idx]

        vm = torch.as_tensor(val.means, device=self.device)
        vy = torch.as_tensor(scaler.encode(val.fields_t), dtype=torch.float32, device=self.device)
        vb = []
        for _ in range(2):
            vs = sample_shots(vm.shape[0], lo, hi, vgen, self.device)
            vb.append((features(poisson_counts(vm, vs, vgen)), log_shots(vs), vy))
        t0 = time.perf_counter()
        hist = train_decoder(
            model,
            sample,
            vb,
            TrainConfig(
                steps=int(tc["steps"]),
                batch=batch,
                lr=float(tc["lr"]),
                weight_decay=float(tc["weight_decay"]),
                eval_every=int(tc["eval_every"]),
            ),
        )
        self.seconds["cnn"] += time.perf_counter() - t0
        return model, hist

    def evaluate(
        self, model: nn.Module, means: np.ndarray, fields: np.ndarray, scaler: TargetScaler, seed: int, shots: int
    ) -> dict[str, float]:
        m = torch.as_tensor(means, device=self.device)
        g = self._gen(seed, 100_000 + shots)
        s = torch.full((m.shape[0],), float(shots), device=self.device)
        mean, sd = predict(model, features(poisson_counts(m, s, g)), log_shots(s))
        b, sig = scaler.decode(mean.cpu().numpy().astype(np.float64), sd.cpu().numpy().astype(np.float64))
        assert sig is not None
        return field_metrics(b, sig, fields)

    def tune_pulse(self) -> tuple[float, int]:
        """The tuned fixed pulsed baseline: a small sweep of the pulse amplitude (each with its nominal pi pulse),
        then of the pulse area at the best amplitude, scored by the CNN recipe's validation RMSE (seed ``tune.seed``,
        its train and validation fields only; the test fields are never seen)."""
        tc = self.cfg["tune"]
        seed = int(tc["seed"])
        sd = self.data(seed)
        scaler = self.scalers.setdefault(seed, TargetScaler.fit(sd.fields_t[: self.n_train]))
        a = self.n_train + self.n_val
        fpe = self.bspec.fields_per_env
        fields, envs = sd.fields_t[:a], sd.envs[: a // fpe]
        shots = int(tc.get("shots", self.headline))
        t0 = time.perf_counter()
        scores: dict[tuple[float, int], float] = {}

        def score(amp: float, pi_ns: int, stage: str) -> float:
            if (amp, pi_ns) in scores:
                return scores[(amp, pi_ns)]
            prot = fixed_protocol(self.cfg, self.spec, "tuned", 0.0, (amp, pi_ns))
            bank = window_bank(
                simulate_bank(prot, fields, envs, self.bspec, self.device, workers=int(self.cfg["bank"]["workers"])),
                self.window_bins,
            )
            tr, va = bank.subset(np.arange(self.n_train)), bank.subset(np.arange(self.n_train, a))
            model, _ = self.train_cnn_on(tr, va, scaler, seed, 700)  # same CNN seed for every candidate
            v = self.evaluate(model, va.means, va.fields_t, scaler, seed, shots)["rmse_ut"]
            scores[(amp, pi_ns)] = v
            area = 2 * self.rabi_nominal_hz * math.sqrt(2 / 3) * amp * pi_ns * 1e-9
            self.tune_rows.append({"stage": stage, "amplitude": amp, "pi_ns": pi_ns, "area_pi": area, "val_rmse_ut": v})
            print(f"[m4-sensor] tune {stage}: a={amp:.2f} {pi_ns} ns -> val {v:.1f} uT", flush=True)
            return v

        for amp in tc["amplitudes"]:
            score(float(amp), nominal_pi_ns(float(amp), self.rabi_nominal_hz), "amplitude")
        best = min(scores, key=lambda k: scores[k])
        amp0 = best[0]
        pi0 = nominal_pi_ns(amp0, self.rabi_nominal_hz)
        for f in tc["area_factors"]:
            score(amp0, max(10, round(pi0 * float(f) / 10) * 10), "area")
        best = min(scores, key=lambda k: scores[k])
        self.seconds["tune"] += time.perf_counter() - t0
        self.tuned = (float(best[0]), int(best[1]))
        return self.tuned

    def init_pulse(self) -> tuple[float, float]:
        """The learned protocol's starting pulse: milestone 2's (``sensor.init: m2``) or the tuned one (``tuned``),
        with the amplitude capped below ``amp_hi`` (a saturated sigmoid has no gradient), area kept."""
        sc = self.cfg["sensor"]
        if str(sc.get("init", "m2")) != "tuned":
            return float(sc["init_amplitude"]), float(sc["init_pi_ns"])
        assert self.tuned is not None
        amp, pi_ns = float(self.tuned[0]), float(self.tuned[1])
        cap = float(sc.get("init_amp_cap", 0.95)) * self.spec.layout.amp_hi
        if amp > cap:
            pi_ns, amp = pi_ns * amp / cap, cap
        return amp, pi_ns

    def train_learned(self, cond: Condition, seed: int, scaler: TargetScaler) -> LearnedProtocol:
        sc, pt = self.cfg["sensor"], self.cfg["protocol_train"]
        torch.manual_seed(self._seed(seed, 50))
        amp, pi_ns = self.init_pulse()
        init = pulsed_init(
            self.spec, amp, pi_ns, float(sc.get("init_rest_amplitude", 0.0)), float(sc.get("init_rest_ns", 0.0))
        )
        lp = LearnedProtocol(self.spec, init=init).to(self.device)
        dec = self.cnn(self.spec.n_time)
        t0 = time.perf_counter()
        hist = train_protocol(
            lp,
            dec,
            self.bspec.prior,
            scaler,
            self.train_opt,
            self.device,
            steps=int(pt["steps"]),
            fields_per_step=int(pt["fields_per_step"]),
            lr_protocol=float(pt["lr_protocol"]),
            lr_decoder=float(pt["lr_decoder"]),
            shots_range=(float(pt["shots"][0]), float(pt["shots"][1])),
            rng=np.random.default_rng([self.base_seed, seed, 51]),
            gen=self._gen(seed, 52),
            offset_hz=cond.mw_offset_hz,
            log_every=50,
        )
        self.seconds["phase1"] += time.perf_counter() - t0
        self.histories[f"s{seed}_{cond.name}_phase1"] = {"step": hist.step, "loss": hist.loss, "s": hist.seconds}
        for p in lp.parameters():
            p.requires_grad_(False)
        return lp

    def record_protocol(self, cond: str, seed: int, lp: LearnedProtocol) -> None:
        p = lp()
        nominal = EnvParams(rabi_full_scale_hz=float(np.mean(self.bspec.prior.rabi_full_scale_hz)))
        area = pulse_areas_pi(p, nominal, self.spec.layout.grid_ns)[:, 0]  # (P, S)
        d = p.quantized_ns(self.spec.layout.grid_ns).cpu().numpy()[:, 0]
        amp = p.amplitude.detach().cpu().numpy()[:, 0]
        f = p.frequency_hz.detach().cpu().numpy()[:, 0]
        ph = p.phase_rad.detach().cpu().numpy()[:, 0]
        base = self.spec.base_hz()
        for i in range(self.spec.n_points):
            for s in range(self.spec.layout.n_segments):
                self.protocol_rows.append(
                    {
                        "seed": seed,
                        "condition": cond,
                        "point": i,
                        "segment": s,
                        "base_hz": float(base[i]),
                        "frequency_hz": float(f[i, s]),
                        "amplitude": float(amp[i, s]),
                        "duration_ns": float(d[i, s]),
                        "phase_rad": float(ph[i, s]),
                        "area_pi": float(area[i, s]),
                    }
                )

    # --- the task ---------------------------------------------------------------------------------------------
    def __call__(self, cond: Condition, seed: int) -> dict[str, float]:
        t_seed = time.perf_counter()
        sd = self.data(seed)
        scaler = self.scalers.setdefault(seed, TargetScaler.fit(sd.fields_t[: self.n_train]))
        out: dict[str, float] = {}
        for pi_, kind in enumerate(PROTOCOLS):
            lp = None
            if kind == "learned":
                lp = self.train_learned(cond, seed, scaler)
                self.learned[(seed, cond.name)] = lp
                self.record_protocol(cond.name, seed, lp)
                freqs = lp().frequency_hz.detach().cpu().numpy().ravel() + cond.mw_offset_hz
                torch.save(lp.state_dict(), self.data_dir / f"protocol_s{seed}_{cond.name}.pt")
            else:
                freqs = self.protocol_for(kind, cond.mw_offset_hz).mw_freqs_hz()
            if cond.detuned:
                self.check_detuned(freqs, sd)
            bank = self.bank(kind, cond, sd, lp)
            _, _, test = self.split(bank)
            if seed == 0:
                self.test_banks[(cond.name, kind)] = test
                test.save(str(self.data_dir / f"test_bank_s0_{cond.name}_{kind}.npz"))
            prior = field_metrics(
                np.broadcast_to(scaler.mean_t, test.fields_t.shape),
                np.broadcast_to(scaler.std_t, test.fields_t.shape),
                test.fields_t,
            )
            model, hist = self.train_cnn(bank, scaler, seed, 10 * (pi_ + 1))
            self.histories[f"s{seed}_{cond.name}_{kind}_cnn"] = dataclasses.asdict(hist)
            for level in self.levels:
                met = self.evaluate(model, test.means, test.fields_t, scaler, seed, level)
                self.rows.append(
                    {
                        "seed": seed,
                        "condition": cond.name,
                        "protocol": kind,
                        "shots": level,
                        **met,
                        "prior_rmse_ut": prior["rmse_ut"],
                    }
                )
                if level == self.headline:
                    out[f"{kind}_rmse_ut"] = met["rmse_ut"]
                    out[f"{kind}_skill"] = 1.0 - met["rmse_ut"] / prior["rmse_ut"]
                if level == self.budget:
                    out[f"{kind}_rmse_ut_budget"] = met["rmse_ut"]
            # equal wall time per point: the learned protocol's shot count at each level, converted to this
            # protocol's shots by its shot duration
            for level in (self.headline, self.budget):
                sh = max(1, round(level * self.spec.shot_ns / self.shot_ns(kind)))
                met = self.evaluate(model, test.means, test.fields_t, scaler, seed, sh)
                self.eqtime_rows.append(
                    {
                        "seed": seed,
                        "condition": cond.name,
                        "protocol": kind,
                        "ref_shots": level,
                        "wall_us": level * self.spec.shot_ns * 1e-3,
                        "shots": sh,
                        "rmse_ut": met["rmse_ut"],
                    }
                )
                if level == self.headline:
                    out[f"{kind}_rmse_ut_eqtime"] = met["rmse_ut"]
            if not cond.detuned:
                self.models[(seed, cond.name, kind)] = model
            if seed == 0:
                torch.save(model.state_dict(), self.data_dir / f"cnn_s0_{cond.name}_{kind}.pt")
            del bank
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        if not cond.detuned:
            t0 = time.perf_counter()
            self.timing_probe(seed, scaler)
            self.seconds["timing"] += time.perf_counter() - t0
        print(
            f"[m4-sensor] seed {seed} {cond.name}: "
            + ", ".join(f"{k}={v:.3g}" for k, v in out.items() if k.endswith("_rmse_ut"))
            + f" ({time.perf_counter() - t_seed:.0f} s)",
            flush=True,
        )
        return out

    def timing_probe(self, seed: int, scaler: TargetScaler) -> None:
        """Resonant, learned and tuned pulsed protocols, on the first ``n_fields`` test fields:

        * ``shift``: every MW duration shifted (learned: every segment; tuned: the pulse), frozen CNN;
        * ``equal_area``: durations kept, amplitudes scaled so every pulse's area changes as under the shift
          (separates "timing" from "pulse area", i.e. the Rabi rotation angle), frozen CNN;
        * ``retrained``: a fresh CNN trained on a reduced bank of the shifted protocol (``retrain_n_train`` /
          ``retrain_n_val`` fields of the seed's own splits), with an unshifted reduced-bank CNN as its reference;
        * ``phase_scramble`` (learned): every segment's phase replaced by a random one, frozen CNN.
        """
        tc = self.cfg["timing"]
        n = int(tc["n_fields"])
        sd = self.data(seed)
        a = self.n_train + self.n_val
        fpe = self.bspec.fields_per_env
        n = (n // fpe) * fpe or fpe
        fields = sd.fields_t[a : a + n]
        envs = sd.envs[a // fpe : (a + n) // fpe]
        lp = self.learned[(seed, "resonant")]
        assert self.tuned is not None
        amp_t, pi_t = self.tuned
        ly = self.spec.layout

        def make_bank(kind: str, f: np.ndarray, e: list[EnvParams], shift: int, mode: str = "shift") -> FieldBank:
            if kind == "learned":
                if mode == "scramble":
                    g = torch.Generator(device=self.device)
                    g.manual_seed(self._seed(seed, 900))
                    p = lp()
                    ph = (torch.rand(p.phase_rad.shape, generator=g, device=self.device) * 2 - 1) * math.pi
                    fp = _FixedParams(lp, BlockParams(p.amplitude, ph, p.frequency_hz, p.duration_ns))
                    return simulate_learned_bank(fp, f, e, self.bspec, self.device)  # type: ignore[arg-type]
                if mode == "equal_area":
                    fp = _FixedParams(lp, area_matched_amplitude(lp(), ly, float(shift)))
                    return simulate_learned_bank(fp, f, e, self.bspec, self.device)  # type: ignore[arg-type]
                return simulate_learned_bank(lp, f, e, self.bspec, self.device, duration_shift_ns=shift)
            pi_s = max(ly.grid_ns, pi_t + shift)
            pulse = (amp_t * pi_s / pi_t, pi_t) if mode == "equal_area" else (amp_t, pi_s)
            prot = fixed_protocol(self.cfg, self.spec, "tuned", 0.0, pulse)
            return window_bank(simulate_bank(prot, f, e, self.bspec, self.device, workers=4), self.window_bins)

        def frozen(kind: str, shift: int, mode: str = "shift") -> float:
            bank = make_bank(kind, fields, envs, shift, mode)
            model = self.models[(seed, "resonant", kind)]
            return self.evaluate(model, bank.means, fields, scaler, seed, self.headline)["rmse_ut"]

        def row(kind: str, probe: str, shift: int, v: float) -> None:
            self.timing_rows.append(
                {"seed": seed, "protocol": kind, "probe": probe, "shift_ns": int(shift), "rmse_ut": float(v)}
            )

        for kind in ("learned", "tuned"):
            ts = timing_sensitivity(functools.partial(frozen, kind), [int(x) for x in tc["duration_shifts_ns"]])
            for sh, v in zip(ts.shifts_ns, ts.metric, strict=True):
                row(kind, "shift", int(sh), float(v))
            for sh in tc["area_shifts_ns"]:
                row(kind, "equal_area", int(sh), frozen(kind, int(sh), "equal_area"))
        row("learned", "phase_scramble", 0, frozen("learned", 0, "scramble"))

        # retrained CNNs on reduced banks (train / val fields from the seed's own splits, the same test fields)
        nr = (min(int(tc["retrain_n_train"]), self.n_train) // fpe) * fpe
        nv = (min(int(tc["retrain_n_val"]), self.n_val) // fpe) * fpe
        if nr <= 0:
            return
        idx = np.r_[0:nr, self.n_train : self.n_train + nv]
        f_all = np.concatenate([sd.fields_t[idx], fields])
        e_all = [*sd.envs[: nr // fpe], *sd.envs[self.n_train // fpe : (self.n_train + nv) // fpe], *envs]
        for kind in ("learned", "tuned"):
            for sh in sorted({0, *[int(x) for x in tc["retrain_shifts_ns"]]}):
                bank = make_bank(kind, f_all, e_all, sh)
                tr, va = bank.subset(np.arange(nr)), bank.subset(np.arange(nr, nr + nv))
                te = bank.subset(np.arange(nr + nv, len(bank)))
                model, _ = self.train_cnn_on(tr, va, scaler, seed, 800 + sh)
                row(
                    kind,
                    "retrained",
                    sh,
                    self.evaluate(model, te.means, te.fields_t, scaler, seed, self.headline)["rmse_ut"],
                )


class _FixedParams(torch.nn.Module):
    """A protocol stand-in that returns given parameters (the phase-scramble probe)."""

    def __init__(self, lp: LearnedProtocol, params: BlockParams) -> None:
        super().__init__()
        self.spec = lp.spec
        self.params = params

    def forward(self) -> BlockParams:
        return self.params


# --- QuTiP reference ------------------------------------------------------------------------------------------------

_REF: dict[str, Any] = {}


def _ref_init(n_detuning: int, rule: str, spacing: float) -> None:
    from glowbox.sim.reference import ReferenceSimulator

    _REF["sim"] = ReferenceSimulator(n_detuning=n_detuning, detuning_rule=rule, detuning_spacing=spacing)  # type: ignore[arg-type]


def _ref_one(item: tuple[dict[str, Any], dict[str, Any]]) -> np.ndarray:
    from glowbox.control.pulse import PulseSequence

    seq, env = PulseSequence.from_dict(item[0]), EnvParams.from_dict(item[1])
    return np.asarray(_REF["sim"].expected_rate_cps(seq, env))


def run_reference_check(task: SensorTask) -> dict[str, Any]:
    """The seed-0 resonant learned protocol for a few test fields on the QuTiP reference (bank numerics): features
    and the frozen CNN's estimates."""
    rc = task.cfg["reference"]
    n = int(rc["n_fields"])
    lp = task.learned[(0, "resonant")]
    bank = task.test_banks[("resonant", "learned")]
    spec = task.spec
    ly = spec.layout
    pre = init_prefix(spec.init_ns, spec.settle_ns, grid_ns=ly.grid_ns)
    sfx = suffix_to_numpy(lp(), ly)
    seqs = [full_sequence(pre, row) for row in sfx.unbatch()]
    items, owners = [], []
    for i in range(n):
        env = bank.env(i).to_dict()
        for s in seqs:
            items.append((s.to_dict(), env))
            owners.append(i)
    t0 = time.perf_counter()
    b = task.bspec
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        int(rc["workers"]), initializer=_ref_init, initargs=(b.n_detuning, b.detuning_rule, b.detuning_spacing)
    ) as pool:
        rates = pool.map(_ref_one, items, chunksize=2)
    dt = ly.grid_ns * 1e-9
    nro = ly.readout_bins
    feats = np.zeros((n, spec.n_time, spec.n_points), dtype=np.float32)
    for r, o, j in zip(rates, owners, list(range(len(seqs))) * n, strict=True):
        ro = np.asarray(r[-nro:]) * dt
        feats[o, :, j] = ro.reshape(spec.n_time, -1).sum(axis=1)
    sur = bank.means[:n]
    out: dict[str, Any] = {"n_fields": n, "max_rel_feature_diff": float(np.max(np.abs(feats - sur)) / np.max(sur))}
    model = task.models[(0, "resonant", "learned")]
    scaler = task.scalers[0]
    shots = task.headline
    c = log_shots(torch.full((n,), float(shots), device=task.device))
    preds = []
    for arr in (feats, sur):
        x = torch.as_tensor(arr * shots, device=task.device)
        preds.append(scaler.decode(predict(model, features(x), c)[0].cpu().numpy().astype(np.float64))[0])
    out["noiseless_max_pred_diff_ut"] = float(np.max(np.linalg.norm(preds[0] - preds[1], axis=1)) * 1e6)
    errs: dict[str, float] = {}
    for src, mean in (("ref", feats), ("sur", sur)):
        rep = np.repeat(mean, 64, axis=0)
        met = task.evaluate(model, rep, np.repeat(bank.fields_t[:n], 64, axis=0), scaler, 0, shots)
        errs[src] = met["rmse_ut"]
    out["noisy_rmse_ref_ut"], out["noisy_rmse_sur_ut"] = errs["ref"], errs["sur"]
    out["seconds"] = time.perf_counter() - t0
    return out


def log_example_traces(task: SensorTask) -> None:
    """Seed-0 test field 0, the learned protocol's point nearest its lowest line, both conditions, 10 shots."""
    sim = SurrogateSimulator(task.device, task.bspec.options())
    rng = np.random.default_rng([task.base_seed, 3])
    ly = task.spec.layout
    pre = init_prefix(task.spec.init_ns, task.spec.settle_ns, grid_ns=ly.grid_ns)
    for cname in ("resonant", "detuned"):
        lp = task.learned.get((0, cname))
        bank = task.test_banks.get((cname, "learned"))
        if lp is None or bank is None:
            continue
        env = bank.env(0)
        off = condition_for(task.cfg, cname).mw_offset_hz
        rows = suffix_to_numpy(lp(), ly, offset_hz=off).unbatch()
        line = float(transition_lines_hz(bank.fields_t[0]).min())
        i_on = int(np.argmin(np.abs(task.spec.base_hz() - line)))
        for i, which in ((i_on, "near_line"), (0, "first_point")):
            seq = full_sequence(pre, rows[i])
            tr = sim.run(seq, env, shots=10, rng=rng)
            task.run.log_trace(tr, seq, condition=cname, protocol="learned", point=which, field_t=bank.fields_t[0])


# --- plots ----------------------------------------------------------------------------------------------------------


def _sel(df: pd.DataFrame, **kw: Any) -> pd.DataFrame:
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def plot_protocol(task: SensorTask) -> Figure:
    pdf = pd.DataFrame(task.protocol_rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7))
    d = _sel(pdf, seed=0, condition="resonant")
    base = task.spec.base_hz()
    ax = axes[0]
    marks = ("o", "s")
    for s in sorted(d.segment.unique()):
        e = d[d.segment == s]
        ax.scatter(
            (e.frequency_hz - 2.87e9) / 1e6,
            e.area_pi,
            s=12 + 30 * e.amplitude,
            color=COLORS["learned"],
            alpha=0.75,
            marker=marks[s % 2],
            label=f"segment {s}",
        )
    ax.axhline(1.0, color=COLORS["pulsed"], ls="--", lw=1.2, label="fixed pulsed: pi pulse")
    ax.set_xlabel("segment frequency - 2.87 GHz (MHz)")
    ax.set_ylabel("pulse area (pi, nominal f_R)")
    ax.set_title("learned protocol (seed 0): area per segment (size = amplitude)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    ax = axes[1]
    for s in sorted(d.segment.unique()):
        e = d[d.segment == s].sort_values("point")
        ax.plot(
            (base - 2.87e9) / 1e6,
            (e.frequency_hz.to_numpy() - base) / 1e6,
            marks[s % 2],
            ms=4,
            color=COLORS["learned"],
            alpha=0.75,
            label=f"segment {s}",
        )
    ax.set_xlabel("point's grid frequency - 2.87 GHz (MHz)")
    ax.set_ylabel("learned offset from grid (MHz)")
    ax.set_title("where each segment drives (all within +-span)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    ax = axes[2]
    bank_l = task.test_banks.get(("resonant", "learned"))
    if bank_l is not None:
        f0 = bank_l.fields_t[0]
        for kind in PROTOCOLS:
            bk = task.test_banks.get(("resonant", kind))
            if bk is None:
                continue
            m = bk.means[0]
            spec = m[0] if m.shape[0] == 1 else m[0] / m[-1]
            ax.plot((base - 2.87e9) / 1e6, spec / np.median(spec), "-o", ms=3, color=COLORS[kind], label=LABELS[kind])
        for ln in transition_lines_hz(f0).ravel():
            ax.axvline((ln - 2.87e9) / 1e6, color="#8a8985", lw=0.6, ls=":")
        ax.set_title("noiseless features of test field 0 (first window / last; dotted: lines)", fontsize=8)
    ax.set_xlabel("point's grid frequency - 2.87 GHz (MHz)")
    ax.set_ylabel("relative signal")
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_error_vs_shots(task: SensorTask) -> Figure:
    df = pd.DataFrame(task.rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    lv = sorted(df.shots.unique())
    for kind in PROTOCOLS:
        for cname, ls in (("resonant", "-"), ("detuned", "--")):
            d = _sel(df, protocol=kind, condition=cname)
            if len(d):
                _band(
                    ax,
                    lv,
                    [d[d.shots == s].rmse_ut.to_numpy() for s in lv],
                    COLORS[kind],
                    ls,
                    f"{LABELS[kind]}, {cname}",
                )
    pr = _sel(df, protocol="learned", condition="resonant")
    ax.axhline(float(pr.prior_rmse_ut.mean()), color=COLORS["prior"], ls=":", lw=1.4, label="prior only")
    for kind, v in M2_CONTEXT_UT.items():
        ax.plot([task.headline], [v], "*", ms=10, color=COLORS[kind], label=f"M2 {kind}, 200 points (context)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"shots per point ({task.spec.n_points} points)")
    ax.set_ylabel("RMS field error (uT)")
    ax.set_title("field error vs shots, mean over seeds, 95 % t CI; dashed = detuned", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_quantum_test(task: SensorTask, qt: QuantumTestResult) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ticks, labels = [], []
    for x, kind in enumerate(PROTOCOLS):
        s = qt.summary(f"{kind}_skill")
        for j, (cname, alpha) in enumerate((("resonant", 1.0), ("detuned", 0.45))):
            v, lo, hi = s[cname]
            ax.bar(x + 0.36 * j, v, 0.32, color=COLORS[kind], alpha=alpha, edgecolor="none")
            ax.errorbar(x + 0.36 * j, v, yerr=[[v - lo], [hi - v]], color="#0b0b0b", lw=1, capsize=3)
        ticks.append(x + 0.18)
        labels.append(LABELS[kind])
    ax.axhline(0.0, color=COLORS["prior"], ls=":", lw=1.2)
    ax.set_xticks(ticks, labels, fontsize=8)
    ax.set_ylabel(f"skill 1 - RMSE / prior RMSE ({task.headline} shots)")
    ax.set_title("quantum test: resonant (solid) vs detuned +0.4 GHz (light), 95 % t CI", fontsize=9)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_training(task: SensorTask) -> Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for cname, ls in (("resonant", "-"), ("detuned", "--")):
        for key, h in task.histories.items():
            if key.endswith(f"_{cname}_phase1"):
                ax.plot(h["step"], h["loss"], ls, color=COLORS["learned"], alpha=0.6, lw=1.2)
        ax.plot([], [], ls, color=COLORS["learned"], label=cname)
    ax.set_xlabel("phase-1 step (protocol + CNN through the surrogate)")
    ax.set_ylabel("training loss (Gaussian NLL)")
    ax.set_title("end-to-end training, every seed", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_timing(task: SensorTask) -> Figure:
    df = pd.DataFrame(task.timing_rows)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for kind in ("learned", "tuned"):
        d = _sel(df, protocol=kind, probe="shift")
        xs = sorted(d.shift_ns.unique())
        _band(ax, xs, [d[d.shift_ns == x].rmse_ut.to_numpy() for x in xs], COLORS[kind], "-", f"{LABELS[kind]}, frozen")
        for probe, mk, lab in (("equal_area", "s", "equal area by amplitude, frozen"), ("retrained", "^", "retrained")):
            r = _sel(df, protocol=kind, probe=probe)
            if len(r):
                xr = sorted(r.shift_ns.unique())
                mus = [t_ci(r[r.shift_ns == x].rmse_ut.to_numpy()) for x in xr]
                ax.errorbar(
                    xr,
                    [m[0] for m in mus],
                    yerr=[[m[0] - m[1] for m in mus], [m[2] - m[0] for m in mus]] if math.isfinite(mus[0][1]) else None,
                    fmt=mk,
                    mfc="none",
                    color=COLORS[kind],
                    capsize=3,
                    label=f"{kind}, {lab}",
                )
    scr = _sel(df, protocol="learned", probe="phase_scramble").rmse_ut.to_numpy()
    if scr.size:
        v, lo, hi = t_ci(scr)
        ax.errorbar(
            [0],
            [v],
            yerr=[[v - lo], [hi - v]] if math.isfinite(lo) else None,
            fmt="D",
            color="#0b0b0b",
            capsize=3,
            label="learned, phases scrambled",
        )
    ax.set_yscale("log")
    ax.set_xlabel("shift of every MW duration (ns; equal area: the same area change by amplitude)")
    ax.set_ylabel(f"RMS field error (uT, {task.headline} shots)")
    ax.set_title("timing probe (resonant), mean over seeds, 95 % t CI", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


# --- report ---------------------------------------------------------------------------------------------------------


def timing_verdicts(task: SensorTask) -> dict[str, Any]:
    df = pd.DataFrame(task.timing_rows)
    pdf = pd.DataFrame(task.protocol_rows)
    out: dict[str, Any] = {}

    def rel_of(kind: str, probe: str, s: int, base_probe: str = "shift") -> tuple[float, float, float] | None:
        base = _sel(df, protocol=kind, probe=base_probe, shift_ns=0).sort_values("seed").rmse_ut.to_numpy()
        per = []
        for sg in (-s, s):
            d = _sel(df, protocol=kind, probe=probe, shift_ns=sg).sort_values("seed").rmse_ut.to_numpy()
            if d.size == base.size and d.size:
                per.append(d / base - 1.0)
        return t_ci(np.mean(per, axis=0)) if per else None

    for kind in ("learned", "tuned"):
        rel: dict[str, Any] = {}
        for s in (20, 50):
            for probe, key, bp in (
                ("shift", f"pm{s}", "shift"),
                ("equal_area", f"area_pm{s}", "shift"),
                ("retrained", f"retrained_pm{s}", "retrained"),
            ):
                v = rel_of(kind, probe, s, bp)
                if v is not None:
                    rel[key] = v
        out[kind] = {"relative_rmse_increase": rel}
        o = out[kind]
        o["relies_on_pulse_timing"] = bool("pm50" in rel and rel["pm50"][1] > 0.1)
        o["information_needs_exact_timing"] = bool("retrained_pm50" in rel and rel["retrained_pm50"][1] > 0.1)
        # the same area change made with the amplitude costs at least half as much: the probe measures pulse area
        o["area_explains_shift"] = bool(
            "area_pm50" in rel and "pm50" in rel and rel["area_pm50"][0] >= 0.5 * rel["pm50"][0]
        )
    scr = _sel(df, protocol="learned", probe="phase_scramble").sort_values("seed").rmse_ut.to_numpy()
    base = _sel(df, protocol="learned", probe="shift", shift_ns=0).sort_values("seed").rmse_ut.to_numpy()
    if scr.size and scr.size == base.size:
        out["learned"]["relative_rmse_increase"]["phase_scramble"] = t_ci(scr / base - 1.0)
    pr = _sel(pdf, condition="resonant")
    per_seed = pr.groupby("seed")
    pts = pr.groupby(["seed", "point"])
    on = pr[pr.duration_ns > 0]
    on_pts = on.groupby(["seed", "point"])
    split_pt = on_pts.frequency_hz.agg(lambda f: float(f.max() - f.min()) / 1e6)
    # two frequencies = both segments on and more than 0.5 MHz apart (a fraction of the excitation bandwidth)
    two = ((split_pt > 0.5) & (on_pts.size() > 1)).astype(float).groupby("seed").mean()
    split = split_pt.groupby("seed").median()
    out["learned"].update(
        {
            "point_total_area_median_pi": t_ci(pts.area_pi.sum().groupby("seed").median().to_numpy()),
            "segment_area_median_pi": t_ci(per_seed.area_pi.median().to_numpy()),
            "amplitude_median": t_ci(per_seed.amplitude.median().to_numpy()),
            "duration_total_median_ns": t_ci(pts.duration_ns.sum().groupby("seed").median().to_numpy()),
            "frac_points_two_frequencies": t_ci(two.to_numpy()),
            "segment_split_mhz_median": t_ci(split.to_numpy()),
        }
    )
    lv = out["learned"]
    rel = lv["relative_rmse_increase"]
    lv["uses_phase_coherence"] = bool("phase_scramble" in rel and rel["phase_scramble"][1] > 0.1)
    lv["rabi_scale_areas"] = bool(0.25 <= lv["point_total_area_median_pi"][0] <= 6.0)
    out["strategy_learns_timing"] = bool(lv["relies_on_pulse_timing"] and lv["rabi_scale_areas"])
    return out


def _bool(x: bool) -> str:
    return "yes" if x else "no"


def report_body(task: SensorTask, qt: QuantumTestResult, tv: dict[str, Any], ref: dict[str, Any]) -> str:
    cfg = task.cfg
    df = pd.DataFrame(task.rows)
    edf = pd.DataFrame(task.eqtime_rows)
    sp = task.spec
    ly = sp.layout
    hs, bs = task.headline, task.budget
    s = {k: qt.summary(f"{k}_rmse_ut") for k in PROTOCOLS}
    assert task.tuned is not None
    amp_t, pi_t = task.tuned

    def pair(a: str, b: str, shots: int) -> str:
        va = _sel(df, protocol=a, condition="resonant", shots=shots).sort_values("seed").rmse_ut.to_numpy()
        vb = _sel(df, protocol=b, condition="resonant", shots=shots).sort_values("seed").rmse_ut.to_numpy()
        return _ci(t_ci(va - vb))

    def area_pi(amp: float, ns: float) -> float:
        return 2 * task.rabi_nominal_hz * math.sqrt(2 / 3) * amp * ns * 1e-9

    lv = tv["learned"]
    rel = lv["relative_rmse_increase"]
    trel = tv["tuned"]["relative_rmse_increase"]
    nan3 = (float("nan"),) * 3
    init_amp, init_ns = task.init_pulse()
    lines = [
        "**Task 1 with a learned sensor protocol**: the encoder chooses the pulses of every measurement point, trained "
        "end to end with a CNN decoder through the differentiable surrogate, and compared with fixed protocols on the "
        "same frequency grid, shots and decoder recipe: a pulsed ODMR sweep whose pulse is tuned for this grid, "
        "milestone 2's untuned pulsed ODMR and fixed CW ODMR. Simulation only; no claim of quantum advantage: a gap "
        "only says that the performance depends on driving the spin resonance.",
        "",
        "## Result",
        "",
        f"RMS error of the folded field (|B| {task.bspec.prior.b_min_t * 1e3:.1f}-{task.bspec.prior.b_max_t * 1e3:.0f} "
        f"mT, any direction) with {sp.n_points} points, mean over {len(qt.seeds)} seeds [95 % Student-t CI]:",
        "",
        f"| protocol | shot (ns) | resonant, {hs} shots/point | detuned | gap (detuned - resonant) | resonant, {bs} "
        "shots/point (= M2's total shots) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for k in PROTOCOLS:
        b = t_ci(_sel(df, protocol=k, condition="resonant", shots=bs).sort_values("seed").rmse_ut.to_numpy())
        lines.append(
            f"| {LABELS[k]} | {task.shot_ns(k)} | {_ci(s[k]['resonant'])} | {_ci(s[k]['detuned'])} | "
            f"{_ci(s[k]['gap'])} | {_ci(b)} |"
        )
    lines += [
        "",
        f"- Tuned pulse (validation sweep below): amplitude {amp_t:g}, {pi_t} ns (nominal area "
        f"{area_pi(amp_t, pi_t):.2f} pi); milestone 2's: amplitude {cfg['fixed']['pulsed_amplitude']}, "
        f"{cfg['fixed']['pi_ns']} ns.",
        f"- Paired differences at {hs} shots/point (positive = the second is better): tuned - learned "
        f"{pair('tuned', 'learned', hs)} uT; CW - learned {pair('cw', 'learned', hs)} uT; CW - tuned "
        f"{pair('cw', 'tuned', hs)} uT; untuned - tuned {pair('pulsed', 'tuned', hs)} uT. At {bs} shots: tuned - "
        f"learned {pair('tuned', 'learned', bs)} uT; CW - learned {pair('cw', 'learned', bs)} uT.",
        f"- Context: milestone 2's CNN on its 200-point sweeps (1 MHz steps, 6000 training steps) reached "
        f"{M2_CONTEXT_UT['cw']} uT (CW) and {M2_CONTEXT_UT['pulsed']} uT (pulsed) at 1000 shots per point; at equal "
        f"total shots ({bs} per point here) compare with the last column.",
        "- **Quantum test**: with every MW frequency shifted by "
        f"{float(cfg['detuned_offset_hz']) / 1e6:.0f} MHz (no line within {task.no_line_margin_hz / 1e6:.0f} MHz) "
        "every protocol falls to the prior; the gaps are "
        + ", ".join(
            f"{k} {'significant' if qt.gap_significant(f'{k}_rmse_ut') else 'not significant'}" for k in PROTOCOLS
        )
        + ".",
        f"- Timing: the learned strategy uses microsecond-scale (Rabi-scale) timing: "
        f"**{_bool(tv['strategy_learns_timing'])}** (details below).",
        "",
        "## Equal wall time per point",
        "",
        "The protocols' shots differ in length (column `shot (ns)` above; CW counts photons for its whole shot, the "
        "pulsed and learned protocols for their 600 ns readout). RMSE at equal wall time per point, the learned "
        "protocol's shots converted by shot length, resonant:",
        "",
        "| wall time per point (us) | " + " | ".join(LABELS[k] for k in PROTOCOLS) + " |",
        "|" + " --- |" * (len(PROTOCOLS) + 1),
    ]
    if len(edf):
        for lvl in sorted(edf.ref_shots.unique()):
            d0 = _sel(edf, condition="resonant", ref_shots=lvl)
            cells = [f"{float(d0.wall_us.iloc[0]):.0f}"]
            for k in PROTOCOLS:
                d = _sel(d0, protocol=k)
                cells.append(f"{_ci(t_ci(d.rmse_ut.to_numpy()))} ({int(d.shots.iloc[0])} shots)" if len(d) else "")
            lines.append("| " + " | ".join(cells) + " |")
    tdf = pd.DataFrame(task.tune_rows)
    lines += [
        "",
        "## Tuning the fixed pulsed baseline",
        "",
        f"Seed {cfg['tune']['seed']}, its train / validation fields (test fields unseen), the CNN recipe below, "
        f"validation RMSE at {int(cfg['tune'].get('shots', hs))} shots/point. Stage 1: amplitudes with their nominal "
        "pi pulse; stage 2: pulse areas at the best amplitude.",
        "",
        "| stage | amplitude | pulse (ns) | nominal area (pi) | validation RMSE (uT) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, tr_ in tdf.iterrows():
        lines.append(
            f"| {tr_.stage} | {tr_.amplitude:g} | {int(tr_.pi_ns)} | {tr_.area_pi:.2f} | {tr_.val_rmse_ut:.1f} |"
        )
    lines += [
        "",
        "## Protocols and training",
        "",
        f"- Grid: {sp.n_points} points, {(sp.f_stop_hz - sp.f_start_hz) / sp.n_points / 1e6:.0f} MHz apart over "
        f"{sp.f_start_hz / 1e9:.2f}-{sp.f_stop_hz / 1e9:.2f} GHz. Every point: init laser {sp.init_ns} ns, dark "
        f"{sp.settle_ns} ns, MW, readout {ly.readout_ns} ns; features: readout counts in {ly.window_ns} ns windows "
        f"({sp.n_time} per point; the fixed pulsed protocols' 10 ns bins summed into the same windows); CW: one "
        f"steady-state count per {cfg['fixed']['cw_shot_ns']} ns shot.",
        f"- Learned protocol: per point {ly.n_segments} segments in {ly.slot_ns} ns slots (amplitude 0-1, phase, "
        f"frequency within +-{ly.span_hz / 1e6:.1f} MHz of the point's grid frequency, duration 0-{ly.max_ns:.0f} ns "
        f"on the grid); initialised at the {'tuned' if cfg['sensor'].get('init') == 'tuned' else 'M2'} pulse "
        f"(amplitude {init_amp:.3g}, {init_ns:.0f} ns; unused segments at amplitude "
        f"{cfg['sensor'].get('init_rest_amplitude', 0)}, {cfg['sensor'].get('init_rest_ns', 0)} ns). Phase 1: "
        f"{cfg['protocol_train']['steps']} Adam steps (lr {cfg['protocol_train']['lr_protocol']}, cosine), "
        f"{cfg['protocol_train']['fields_per_step']} fresh fields of one nuisance draw per step, Gaussian shot noise "
        f"(shots log-uniform {cfg['protocol_train']['shots']}), {task.train_opt.n_detuning} uniform quadrature "
        "nodes. Phase 2 and the fixed protocols: banks with milestone 2's numerics "
        f"({task.bspec.n_detuning} uniform nodes, spacing {task.bspec.detuning_spacing}), "
        f"{cfg['bank']['n_train']} / {cfg['bank']['n_val']} / {cfg['bank']['n_test']} fields per seed, CNN "
        f"{cfg['train']['steps']} steps with Poisson noise at shots {cfg['train']['shots_min']}-"
        f"{cfg['train']['shots_max']}.",
        f"- Wall time: tuning {task.seconds['tune'] / 60:.0f} min, phase 1 {task.seconds['phase1'] / 60:.0f} min, "
        f"banks {task.seconds['banks'] / 60:.0f} min, CNNs {task.seconds['cnn'] / 60:.0f} min, timing probes "
        f"{task.seconds['timing'] / 60:.1f} min.",
        "",
        "## Error vs shots",
        "",
        "| shots/point | " + " | ".join(f"{k}, resonant" for k in PROTOCOLS) + " | prior |",
        "|" + " --- |" * (len(PROTOCOLS) + 2),
    ]
    for sh in sorted(df.shots.unique()):
        cells = [str(sh)]
        for k in PROTOCOLS:
            cells.append(_ci(t_ci(_sel(df, protocol=k, condition="resonant", shots=sh).rmse_ut.to_numpy())))
        cells.append(f"{_sel(df, protocol='learned', condition='resonant', shots=sh).prior_rmse_ut.mean():.0f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Quantum test", "", qt.to_markdown(), ""]

    def r(d: dict[str, Any], key: str) -> str:
        return _ci(d.get(key, nan3), "{:+.2f}")

    lines += [
        "## What the protocol learned, and timing",
        "",
        f"- Pulse areas at the nominal Rabi full scale ({task.rabi_nominal_hz / 1e6:.0f} MHz, MW along [100]): per "
        f"point (both segments) median {_ci(lv['point_total_area_median_pi'])} pi (its start: "
        f"{area_pi(init_amp, init_ns):.2f} pi; tuned: {area_pi(amp_t, pi_t):.2f} pi); median amplitude "
        f"{_ci(lv['amplitude_median'], '{:.2f}')} (start {init_amp:.2f}); median total MW time per point "
        f"{_ci(lv['duration_total_median_ns'], '{:.0f}')} ns (start {init_ns:.0f}).",
        f"- Fraction of points whose two segments are both on and > 0.5 MHz apart: "
        f"{_ci(lv['frac_points_two_frequencies'], '{:.2f}')}; median split "
        f"{_ci(lv['segment_split_mhz_median'], '{:.1f}')} MHz.",
        "",
        "Timing probes (resonant, every seed, the first test fields), relative RMSE increase at "
        f"{hs} shots [95 % t CI]: every MW duration shifted (frozen CNN); the same area change made with the "
        "amplitude, durations kept (equal-area control, frozen CNN); a CNN retrained on a reduced bank of the shifted "
        f"protocol ({cfg['timing']['retrain_n_train']} train fields, relative to the same recipe unshifted):",
        "",
        "| protocol | +-20 ns, frozen | +-50 ns, frozen | equal area +-20, frozen | equal area +-50, frozen | "
        "+-50 ns, retrained | phases scrambled |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| learned | {r(rel, 'pm20')} | {r(rel, 'pm50')} | {r(rel, 'area_pm20')} | {r(rel, 'area_pm50')} | "
        f"{r(rel, 'retrained_pm50')} | {r(rel, 'phase_scramble')} |",
        f"| tuned pulsed | {r(trel, 'pm20')} | {r(trel, 'pm50')} | {r(trel, 'area_pm20')} | {r(trel, 'area_pm50')} | "
        f"{r(trel, 'retrained_pm50')} | |",
        "",
        f"- Learned: relies on its pulse timing (> 10 % worse at +-50 ns, frozen, CI): "
        f"{_bool(lv['relies_on_pulse_timing'])}; the information needs it (retrained CNN still > 10 % worse): "
        f"{_bool(lv['information_needs_exact_timing'])}; the equal-area control costs at least half as much (the "
        f"probe measures pulse area, the Rabi rotation angle, rather than timing as such): "
        f"{_bool(lv['area_explains_shift'])}; coherent phases between its segments (> 10 % worse scrambled): "
        f"{_bool(lv['uses_phase_coherence'])}; Rabi-scale areas: {_bool(lv['rabi_scale_areas'])}. "
        f"`strategy_learns_timing` = **{_bool(tv['strategy_learns_timing'])}**.",
        f"- Tuned pulsed: relies on its pulse timing (frozen): {_bool(tv['tuned']['relies_on_pulse_timing'])}; "
        f"retrained: {_bool(tv['tuned']['information_needs_exact_timing'])}; area explains it: "
        f"{_bool(tv['tuned']['area_explains_shift'])}.",
        "",
        "## Reference check",
        "",
        f"- The seed-0 learned protocol for {ref['n_fields']} test fields as full PulseSequences on the QuTiP "
        f"reference (bank numerics): features within {ref['max_rel_feature_diff']:.2g} of the peak; frozen CNN "
        f"estimates differ by at most {ref['noiseless_max_pred_diff_ut']:.3g} uT (noiseless, {hs} shots); noisy "
        f"RMSE {ref['noisy_rmse_ref_ut']:.1f} (reference) vs {ref['noisy_rmse_sur_ut']:.1f} uT (surrogate), 64 draws "
        f"per field ({ref['seconds']:.0f} s).",
        "",
        "## Assumptions and limits",
        "",
        f"- Fewer points ({sp.n_points}) than milestone 2's sweeps (200), and fewer CNN steps, to fit end-to-end "
        "training into the GPU budget: the absolute errors are not milestone 2's; the protocols are compared at equal "
        "points, decoder recipe and shots per point (shot lengths in the table above; equal wall time in its own "
        "table).",
        "- The tuned baseline is tuned on one seed's validation fields with the same CNN recipe; the learned protocol "
        "starts from it. Both tunings are coarse (a handful of candidates, one phase-1 schedule).",
        "- The learned protocol is trained with fewer quadrature nodes than its banks use; phase 2 retrains the CNN "
        "on the bank numerics, so the reported errors are consistent with the banks.",
        "- Segment frequencies are tied to the grid (+-span): a frequency far from every line has no gradient.",
        "- Field target and symmetry as in milestone 2 (fold into the wedge Bx >= By >= Bz >= 0).",
    ]
    return "\n".join(lines)


def _shrink_png(path: Path, colors: int = 128) -> None:
    """Re-save a PNG with a ``colors``-entry palette (figures in docs/ stay small)."""
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
        task = SensorTask(cfg, run, device)
        print("[m4-sensor] tuning the fixed pulsed baseline", flush=True)
        task.tune_pulse()
        conds = (RESONANT, detuned_condition(float(cfg["detuned_offset_hz"])))
        hib = {
            f"{k}_{m}": m == "skill"
            for k in PROTOCOLS
            for m in ("rmse_ut", "skill", "rmse_ut_budget", "rmse_ut_eqtime")
        }
        qt = run_quantum_test(task, [int(s) for s in cfg["seeds"]], conds, higher_is_better=hib)
        print("[m4-sensor] reference check", flush=True)
        ref = run_reference_check(task)
        log_example_traces(task)
        tv = timing_verdicts(task)
        pd.DataFrame(task.rows).to_parquet(run.dir / "results.parquet")
        pd.DataFrame(task.timing_rows).to_parquet(run.dir / "timing.parquet")
        pd.DataFrame(task.protocol_rows).to_parquet(run.dir / "protocols.parquet")
        pd.DataFrame(task.tune_rows).to_parquet(run.dir / "tuning.parquet")
        pd.DataFrame(task.eqtime_rows).to_parquet(run.dir / "equal_time.parquet")
        (run.dir / "train_history.json").write_text(json.dumps(task.histories), encoding="utf-8")
        (run.dir / "reference_check.json").write_text(json.dumps(ref), encoding="utf-8")
        (run.dir / "timing_verdicts.json").write_text(json.dumps(tv, default=str), encoding="utf-8")
        (run.dir / "quantum_test.json").write_text(
            json.dumps({"seeds": qt.seeds, "values": qt.values}), encoding="utf-8"
        )
        figs = {
            "protocol": plot_protocol(task),
            "error-vs-shots": plot_error_vs_shots(task),
            "quantum-test": plot_quantum_test(task, qt),
            "training": plot_training(task),
            "timing": plot_timing(task),
        }
        paths = [run.save_figure(f, name, dpi=72 if name == "protocol" else 90) for name, f in figs.items()]
        _shrink_png(paths[0])  # the dense scatter plot: a palette PNG keeps it small
        for f in figs.values():
            plt.close(f)
        summary: dict[str, Any] = {
            "device": f"{device.type} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}".strip()
        }
        for k in PROTOCOLS:
            sm = qt.summary(f"{k}_rmse_ut")
            summary[f"{k}_rmse_ut_resonant"] = sm["resonant"][0]
            summary[f"{k}_rmse_ut_detuned"] = sm["detuned"][0]
            summary[f"{k}_gap"], summary[f"{k}_gap_lo"], summary[f"{k}_gap_hi"] = sm["gap"]
        assert task.tuned is not None
        summary["tuned_amplitude"], summary["tuned_pi_ns"] = task.tuned
        summary["ref_max_rel_feature_diff"] = ref["max_rel_feature_diff"]
        summary["strategy_learns_timing"] = float(tv["strategy_learns_timing"])
        summary["learned_relies_on_pulse_timing"] = float(tv["learned"]["relies_on_pulse_timing"])
        summary["no_line_margin_mhz"] = task.no_line_margin_hz / 1e6
        summary["phase1_seconds"] = task.seconds["phase1"]
        summary["wall_seconds"] = time.perf_counter() - t_start
        run.log_metrics(step=0, **{k: v for k, v in summary.items() if isinstance(v, int | float)})
        body = report_body(task, qt, tv, ref)
    rdir = report_dir if report_dir is not None else run.dir / "report"
    title = "Milestone 4, Task 1 re-run: a learned sensor protocol for the hidden field"
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
