"""Milestone 2: Task 1, estimate a hidden magnetic field from the glow, with the quantum test.

    uv run python -m glowbox.experiments.m2_field_estimation [--config PATH] [--quick] [--report-dir DIR]

Two fixed sensor protocols (CW ODMR sweep, pulsed ODMR sweep with time-resolved readout) measure an unknown field
B drawn from a prior, with nuisance parameters drawn per group of fields (glowbox.learn.field_data). Three
estimators turn the photon counts into B (folded into the wedge Bx >= By >= Bz >= 0, the identifiable part) with an
uncertainty: a 1D CNN decoder (Gaussian NLL), a linear time-bin-weighting baseline (same loss), and a physics fit
(Lorentzian dips at the H0 transitions of a trial field). Everything learns from the surrogate; key results are
re-checked on the QuTiP reference.

The quantum test (glowbox.analysis.quantum_test) trains and evaluates the learned estimators for every seed twice:
MW on the sweep as designed, and every MW frequency shifted by ``detuned_offset_hz`` so that no line is driven. It
reports the gap with bootstrap CIs over seeds and probes whether the pulsed decoder relies on the Rabi timing of
its protocol (pi-pulse length perturbations, readout-bin saliency).

Outputs: ``outputs/<run-id>/`` (run.yaml, raw example traces, metrics, results.parquet with every per-shot-level
number, seed-0 test banks, models, plots) and a report (``outputs/<run-id>/report/`` unless ``--report-dir``).
"""

from __future__ import annotations

import argparse
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
from torch import nn

from glowbox.analysis.field_fit import FieldFitter, spectrum_from_counts, transition_lines_hz
from glowbox.analysis.quantum_test import (
    RESONANT,
    Condition,
    QuantumTestResult,
    bootstrap_ci,
    check_no_line_driven,
    detuned_condition,
    mw_timing_structure,
    rabi_band_fraction,
    run_quantum_test,
    time_saliency,
)
from glowbox.analysis.runlog import RunLog, write_report
from glowbox.control.pulse import MagnetSetting
from glowbox.experiments.m1_odmr_rabi import _merge, load_config
from glowbox.learn.decoder import (
    ConvDecoder,
    DecoderConfig,
    TimeBinLinear,
    TrainConfig,
    TrainHistory,
    predict,
    train_decoder,
)
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
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate import SurrogateSimulator

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "m2_field_estimation.yaml"
REPORT_NAME = "m2-field-estimation"
KINDS = ("cw", "pulsed")
MODELS = ("cnn", "linear")
# Fixed categorical order (one hue per estimator), resonant solid / detuned dashed.
COLORS = {"cnn": "#2a78d6", "linear": "#eb6834", "fit": "#1baf7a", "prior": "#8a8985"}
LABELS = {"cnn": "CNN decoder", "linear": "linear time-bin", "fit": "physics fit", "prior": "prior only"}

__all__ = ["DEFAULT_CONFIG", "_merge", "load_config", "main"]


# --- configuration helpers --------------------------------------------------------------------------------------------


def protocol_for(cfg: dict[str, Any], kind: str, offset_hz: float = 0.0) -> SweepProtocol:
    return SweepProtocol(kind=kind, mw_offset_hz=offset_hz, **cfg["sweep"])  # type: ignore[arg-type]


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


def condition_for(cfg: dict[str, Any], name: str) -> Condition:
    return RESONANT if name == "resonant" else detuned_condition(float(cfg["detuned_offset_hz"]))


# --- metrics ----------------------------------------------------------------------------------------------------------


def field_metrics(b_hat: np.ndarray, sigma: np.ndarray, b_true: np.ndarray) -> dict[str, float]:
    """Vector error (uT) and calibration of per-component Gaussian uncertainties."""
    err = b_hat - b_true
    norm = np.linalg.norm(err, axis=1)
    sig = np.maximum(sigma, 1e-12)
    z = err / sig
    s_ut = sig * 1e6
    return {
        "rmse_ut": float(np.sqrt(np.mean(norm**2)) * 1e6),
        "median_ut": float(np.median(norm) * 1e6),
        "nll": float(np.mean(0.5 * np.log(2 * np.pi * s_ut**2) + 0.5 * z**2)),
        "z_rms": float(np.sqrt(np.mean(z**2))),
        "cov68": float(np.mean(np.abs(z) < 1.0)),
        "cov95": float(np.mean(np.abs(z) < 1.96)),
        "sigma_ut": float(np.sqrt(np.mean(np.sum(s_ut**2, axis=1)))),
    }


# --- the task ---------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class SeedData:
    fields_t: np.ndarray
    envs: list[EnvParams]


class FieldTask:
    """Task 1 for the quantum-test harness: ``task(condition, seed) -> headline metrics``. Keeps every
    per-shot-level row in ``rows`` and the seed-0 test banks / models / predictions for the later probes."""

    def __init__(self, cfg: dict[str, Any], run: RunLog, device: torch.device) -> None:
        self.cfg, self.run, self.device = cfg, run, device
        self.spec = bank_spec(cfg)
        b = cfg["bank"]
        self.n_train, self.n_val, self.n_test = int(b["n_train"]), int(b["n_val"]), int(b["n_test"])
        fpe = self.spec.fields_per_env
        if self.n_train % fpe or self.n_val % fpe:
            raise ValueError("n_train and n_val must be multiples of fields_per_env (splits on group boundaries)")
        self.levels = [int(s) for s in cfg["eval"]["shots"]]
        self.headline = int(cfg["eval"]["headline_shots"])
        if self.headline not in self.levels:
            self.levels = sorted({*self.levels, self.headline})
        self.base_seed = int(cfg["seed"])
        self.rows: list[dict[str, Any]] = []
        self.histories: dict[str, dict[str, Any]] = {}
        self.z_headline: dict[tuple[str, str, str], list[np.ndarray]] = {}
        self.seed_data: dict[int, SeedData] = {}
        self.test_banks: dict[tuple[str, str], FieldBank] = {}  # seed 0: (condition, kind)
        self.models: dict[tuple[str, str, str], nn.Module] = {}  # seed 0: (condition, kind, model)
        self.preds: dict[tuple[str, str, str, int], tuple[np.ndarray, np.ndarray]] = {}  # seed 0
        self.scalers: dict[int, TargetScaler] = {}
        self.bank_seconds: dict[str, float] = {k: 0.0 for k in KINDS}
        self.no_line_margin_hz: float = math.inf
        self.data_dir = run.dir / "data"
        self.data_dir.mkdir(exist_ok=True)

    # data
    def data(self, seed: int) -> SeedData:
        if seed not in self.seed_data:
            rng = np.random.default_rng([self.base_seed, seed])
            n = self.n_train + self.n_val + self.n_test
            fields, envs = draw_fields_and_envs(n, self.spec, rng)
            self.seed_data[seed] = SeedData(fields, envs)
            np.savez(self.data_dir / f"fields_s{seed}.npz", fields_t=fields)
        return self.seed_data[seed]

    def split(self, bank: FieldBank) -> tuple[FieldBank, FieldBank, FieldBank]:
        a, b = self.n_train, self.n_train + self.n_val
        return bank.subset(np.arange(a)), bank.subset(np.arange(a, b)), bank.subset(np.arange(b, len(bank)))

    def simulate(self, protocol: SweepProtocol, fields: np.ndarray, envs: list[EnvParams]) -> FieldBank:
        return simulate_bank(
            protocol, fields, envs, self.spec, self.device, workers=int(self.cfg["bank"].get("workers", 4))
        )

    # training
    def _torch_seed(self, seed: int, salt: int) -> int:
        return (self.base_seed * 1_000_003 + seed * 7919 + salt) % (2**62)

    def sampler(self, train: FieldBank, scaler: TargetScaler, gen: torch.Generator, batch: int) -> Any:
        means = torch.as_tensor(train.means, device=self.device)
        y = torch.as_tensor(scaler.encode(train.fields_t), dtype=torch.float32, device=self.device)
        tc = self.cfg["train"]
        lo, hi = float(tc["shots_min"]), float(tc["shots_max"])

        def sample(step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            idx = torch.randint(0, means.shape[0], (batch,), generator=gen, device=self.device)
            shots = sample_shots(batch, lo, hi, gen, self.device)
            counts = poisson_counts(means[idx], shots, gen)
            return features(counts), log_shots(shots), y[idx]

        return sample

    def fixed_batches(
        self, bank: FieldBank, scaler: TargetScaler, gen: torch.Generator, repeats: int = 2, chunk: int = 512
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        tc = self.cfg["train"]
        means = torch.as_tensor(bank.means, device=self.device)
        y = torch.as_tensor(scaler.encode(bank.fields_t), dtype=torch.float32, device=self.device)
        out = []
        for _ in range(repeats):
            for i in range(0, means.shape[0], chunk):
                m = means[i : i + chunk]
                shots = sample_shots(m.shape[0], float(tc["shots_min"]), float(tc["shots_max"]), gen, self.device)
                out.append((features(poisson_counts(m, shots, gen)), log_shots(shots), y[i : i + chunk]))
        return out

    def build(self, model: str, protocol: SweepProtocol) -> nn.Module:
        if model == "cnn":
            dc = self.cfg["decoder"]
            cfg = DecoderConfig(
                mix_channels=int(dc["mix_channels"]),
                channels=tuple(int(c) for c in dc["channels"]),
                kernel=int(dc["kernel"]),
                hidden=int(dc["hidden"]),
            )
            return ConvDecoder(protocol.n_time, protocol.n_freq, 3, cfg)
        return TimeBinLinear(protocol.n_time, protocol.n_freq, 3)

    def train_config(self, model: str) -> TrainConfig:
        tc = dict(self.cfg["train"])
        if model == "linear":
            tc.update(self.cfg["linear_train"])
        return TrainConfig(
            steps=int(tc["steps"]),
            batch=int(tc["batch"]),
            lr=float(tc["lr"]),
            weight_decay=float(tc["weight_decay"]),
            eval_every=int(tc["eval_every"]),
        )

    # evaluation
    def evaluate(
        self, model: nn.Module, bank: FieldBank, scaler: TargetScaler, seed: int, shots: int, repeats: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(b_hat, sigma, b_true) in tesla, ``repeats`` noise draws stacked."""
        means = torch.as_tensor(bank.means, device=self.device)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self._torch_seed(seed, 100_000 + shots))
        bh, sg, bt = [], [], []
        for _ in range(repeats):
            s = torch.full((means.shape[0],), float(shots), device=self.device)
            x = features(poisson_counts(means, s, gen))
            m, sd = predict(model, x, log_shots(s))
            b, sigma = scaler.decode(m.cpu().numpy().astype(np.float64), sd.cpu().numpy().astype(np.float64))
            assert sigma is not None
            bh.append(b)
            sg.append(sigma)
            bt.append(bank.fields_t)
        return np.concatenate(bh), np.concatenate(sg), np.concatenate(bt)

    def __call__(self, cond: Condition, seed: int) -> dict[str, float]:
        t0 = time.perf_counter()
        sd = self.data(seed)
        scaler = self.scalers.setdefault(seed, TargetScaler.fit(sd.fields_t[: self.n_train]))
        out: dict[str, float] = {}
        for kind in KINDS:
            protocol = protocol_for(self.cfg, kind, cond.mw_offset_hz)
            if cond.detuned:
                for i in range(0, len(sd.fields_t), 97):  # spot check: nothing is driven
                    env = sd.envs[i // self.spec.fields_per_env]
                    ok, margin = check_no_line_driven(protocol.mw_freqs_hz(), sd.fields_t[i], env)
                    self.no_line_margin_hz = min(self.no_line_margin_hz, margin)
                    if not ok:
                        raise RuntimeError(f"detuned condition drives a line (margin {margin / 1e6:.1f} MHz)")
            bank = self.simulate(protocol, sd.fields_t, sd.envs)
            self.bank_seconds[kind] += bank.seconds
            train, val, test = self.split(bank)
            if seed == 0:
                self.test_banks[(cond.name, kind)] = test
                test.save(str(self.data_dir / f"test_bank_s0_{cond.name}_{kind}.npz"))
            # prior-only reference
            prior = field_metrics(
                np.broadcast_to(scaler.mean_t, test.fields_t.shape),
                np.broadcast_to(scaler.std_t, test.fields_t.shape),
                test.fields_t,
            )
            out[f"{kind}_prior_rmse_ut"] = prior["rmse_ut"]
            for mi, mname in enumerate(MODELS):
                torch.manual_seed(self._torch_seed(seed, mi))
                model = self.build(mname, protocol).to(self.device)
                gen = torch.Generator(device=self.device)
                gen.manual_seed(self._torch_seed(seed, 10 + mi))
                vgen = torch.Generator(device=self.device)
                vgen.manual_seed(self._torch_seed(seed, 20 + mi))
                tcfg = self.train_config(mname)
                hist: TrainHistory = train_decoder(
                    model, self.sampler(train, scaler, gen, tcfg.batch), self.fixed_batches(val, scaler, vgen), tcfg
                )
                key = f"s{seed}_{cond.name}_{kind}_{mname}"
                self.histories[key] = dataclasses.asdict(hist)
                torch.save(model.state_dict(), self.data_dir / f"model_{key}.pt")
                for level in self.levels:
                    b, s, t = self.evaluate(model, test, scaler, seed, level)
                    met = field_metrics(b, s, t)
                    row = {"seed": seed, "condition": cond.name, "protocol": kind, "model": mname, "shots": level}
                    self.rows.append({**row, **met, "prior_rmse_ut": prior["rmse_ut"]})
                    if level == self.headline:
                        self.z_headline.setdefault((cond.name, kind, mname), []).append((b - t) / s)
                        out[f"{kind}_{mname}_rmse_ut"] = met["rmse_ut"]
                        out[f"{kind}_{mname}_skill"] = 1.0 - met["rmse_ut"] / prior["rmse_ut"]
                    if seed == 0:
                        self.preds[(cond.name, kind, mname, level)] = (b, s)
                if seed == 0:
                    self.models[(cond.name, kind, mname)] = model
                else:
                    del model
            del bank, train, val
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        print(
            f"[m2] seed {seed} {cond.name}: "
            + ", ".join(f"{k}={v:.3g}" for k, v in out.items() if k.endswith("rmse_ut"))
            + f" ({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )
        return out


# --- physics fit (process pool) ---------------------------------------------------------------------------------------

_FITTERS: dict[str, FieldFitter] = {}


def _fit_init(freqs: dict[str, np.ndarray], prior_mean: np.ndarray, prior_std: np.ndarray) -> None:
    for name, f in freqs.items():
        _FITTERS[name] = FieldFitter(f, prior_mean, prior_std)


def _fit_one(item: tuple[str, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, bool]:
    name, y, s = item
    r = _FITTERS[name].fit(y, s)
    return r.b_t, r.sigma_t, r.ok


def run_physics_fit(task: FieldTask, workers: int, n_fields: int) -> list[dict[str, Any]]:
    """Physics-fit baseline on the first ``n_fields`` seed-0 test fields, every condition / protocol / shot level."""
    scaler = task.scalers[0]
    freqs = {
        f"{c}_{k}": protocol_for(task.cfg, k, condition_for(task.cfg, c).mw_offset_hz).mw_freqs_hz()
        for c in ("resonant", "detuned")
        for k in KINDS
    }
    rng = np.random.default_rng([task.base_seed, 99])
    items, keys = [], []
    for (cname, kind), bank in task.test_banks.items():
        m = bank.means[:n_fields].astype(np.float64)
        for level in task.levels:
            for i in range(m.shape[0]):
                y, s = spectrum_from_counts(rng.poisson(m[i] * level), kind)
                items.append((f"{cname}_{kind}", y, s))
                keys.append((cname, kind, level, i))
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_fit_init, initargs=(freqs, scaler.mean_t, scaler.std_t)) as pool:
        res = pool.map(_fit_one, items, chunksize=8)
    seconds = time.perf_counter() - t0
    rows = []
    grouped: dict[tuple[str, str, int], list[int]] = {}
    for j, (cname, kind, level, _) in enumerate(keys):
        grouped.setdefault((cname, kind, level), []).append(j)
    for (cname, kind, level), js in grouped.items():
        bank = task.test_banks[(cname, kind)]
        b = np.stack([res[j][0] for j in js])
        s = np.stack([res[j][1] for j in js])
        ok = np.array([res[j][2] for j in js])
        t = bank.fields_t[: len(js)]
        met = field_metrics(b, s, t)
        # the CNN on the same fields and level (seed 0), for a like-for-like comparison
        cb, cs = task.preds[(cname, kind, "cnn", level)]
        cnn = field_metrics(cb[: len(js)], cs[: len(js)], t)
        rows.append(
            {
                "condition": cname,
                "protocol": kind,
                "shots": level,
                **met,
                "ok_fraction": float(ok.mean()),
                "cnn_rmse_ut_same_fields": cnn["rmse_ut"],
                "n_fields": len(js),
                "seconds_total": seconds,
            }
        )
    return rows


# --- timing probe -----------------------------------------------------------------------------------------------------


def rmse_ci(b: np.ndarray, t: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    se = np.sum((b - t) ** 2, axis=1) * 1e12
    v, lo, hi = bootstrap_ci(se, rng=rng, n_boot=4000)
    return math.sqrt(v), math.sqrt(lo), math.sqrt(hi)


def run_timing_probe(task: FieldTask) -> dict[str, Any]:
    """Pulsed protocol, seed-0 decoders frozen: error vs pi-pulse length, readout-bin saliency, sequence timing."""
    tc = task.cfg["timing"]
    n = int(tc["n_fields"])
    fpe = task.spec.fields_per_env
    sd = task.seed_data[0]
    start = task.n_train + task.n_val
    fields = sd.fields_t[start : start + n]
    envs = sd.envs[start // fpe : -(-(start + n) // fpe)]
    scaler = task.scalers[0]
    rng = np.random.default_rng([task.base_seed, 7])
    base = protocol_for(task.cfg, "pulsed")
    out: dict[str, Any] = {"shifts_ns": [int(s) for s in tc["pi_shifts_ns"]], "curves": {}}
    for cname in ("resonant", "detuned"):
        cond = condition_for(task.cfg, cname)
        for mname in MODELS:
            out["curves"][f"{cname}_{mname}"] = []
        for shift in out["shifts_ns"]:
            prot = base.detuned(cond.mw_offset_hz).with_pi_ns(base.pi_ns + shift)
            bank = task.simulate(prot, fields, envs)
            for mname in MODELS:
                model = task.models[(cname, "pulsed", mname)]
                b, _, t = task.evaluate(model, bank, scaler, 0, task.headline, repeats=int(tc["noise_repeats"]))
                out["curves"][f"{cname}_{mname}"].append(rmse_ci(b, t, rng))
    # saliency of the readout bins (resonant and detuned CNN), and the physical spin contrast per bin
    m_res = task.test_banks[("resonant", "pulsed")]
    k = min(256, len(m_res))
    sal = {}
    mean_t = torch.as_tensor(scaler.mean_t, dtype=torch.float32, device=task.device)
    std_t = torch.as_tensor(scaler.std_t, dtype=torch.float32, device=task.device)
    for cname in ("resonant", "detuned"):
        bank = task.test_banks[(cname, "pulsed")]
        means = torch.as_tensor(bank.means[:k], device=task.device)
        gen = torch.Generator(device=task.device)
        gen.manual_seed(task._torch_seed(0, 555))
        s = torch.full((k,), float(task.headline), device=task.device)
        x = features(poisson_counts(means, s, gen))
        model = task.models[(cname, "pulsed", "cnn")]
        sal[cname] = time_saliency(model, x, log_shots(s), lambda m: torch.linalg.norm(m * std_t + mean_t, dim=1))
    mm = m_res.means[:k].astype(np.float64)  # (k, T, F)
    tot = mm.sum(axis=1)
    fdip = np.argmin(tot, axis=1)
    on = mm[np.arange(k), :, fdip]  # (k, T)
    off = np.median(mm, axis=2)
    contrast = np.mean(1.0 - on / off, axis=0)
    lin = task.models[("resonant", "pulsed", "linear")]
    assert isinstance(lin, TimeBinLinear)
    out["saliency"] = {c: v.tolist() for c, v in sal.items()}
    out["contrast_per_bin"] = contrast.tolist()
    out["linear_time_weights"] = lin.weights.detach().cpu().numpy().tolist()
    nt = len(contrast)
    w300 = 300 // base.grid_ns
    out["saliency_first_300ns_fraction"] = {c: float(np.sum(v[:w300]) / np.sum(v)) for c, v in sal.items()}
    out["uniform_first_300ns_fraction"] = w300 / nt
    # sequence-level timing structure of the two protocols (nominal environment)
    env = EnvParams()
    seq_p = base.sequences(MagnetSetting())[base.n_freq // 2]
    seq_c = protocol_for(task.cfg, "cw").sequences(MagnetSetting(), cw_ns=6000)[0]
    f_r = env.rabi_full_scale_hz * base.pulsed_amplitude * math.sqrt(2.0 / 3.0)
    out["pulsed_structure"] = mw_timing_structure(seq_p, env)
    out["cw_structure"] = mw_timing_structure(seq_c, env)
    out["pulsed_rabi_band_fraction"] = rabi_band_fraction(seq_p, f_r)
    out["cw_rabi_band_fraction"] = rabi_band_fraction(seq_c, env.rabi_full_scale_hz * 0.1 * math.sqrt(2.0 / 3.0))
    # verdict: does the resonant CNN degrade when the pulse is no longer a pi pulse (beyond its CI)?
    shifts = np.array(out["shifts_ns"])
    res = np.array(out["curves"]["resonant_cnn"])
    det = np.array(out["curves"]["detuned_cnn"])
    i0 = int(np.argmin(np.abs(shifts)))
    big = np.abs(shifts) >= 100
    out["resonant_degrades"] = bool(np.any(res[big, 1] > res[i0, 2])) if big.any() else False
    out["detuned_degrades"] = bool(np.any(det[big, 1] > det[i0, 2])) if big.any() else False
    out["uses_us_timing"] = out["resonant_degrades"] and not out["detuned_degrades"]
    return out


# --- QuTiP reference check --------------------------------------------------------------------------------------------

_REF: dict[str, Any] = {}


def _ref_init(n_detuning: int, rule: str, spacing: float) -> None:
    from glowbox.sim.reference import ReferenceSimulator

    _REF["sim"] = ReferenceSimulator(n_detuning=n_detuning, detuning_rule=rule, detuning_spacing=spacing)  # type: ignore[arg-type]


def _ref_one(item: tuple[dict[str, Any], dict[str, Any]]) -> np.ndarray:
    from glowbox.control.pulse import PulseSequence

    seq, env = PulseSequence.from_dict(item[0]), EnvParams.from_dict(item[1])
    return np.asarray(_REF["sim"].expected_rate_cps(seq, env))


def run_reference_check(task: FieldTask) -> dict[str, Any]:
    """Re-simulate a few seed-0 test fields on the QuTiP reference; compare features and decoder outputs."""
    rc = task.cfg["reference"]
    n = int(rc["n_fields"])
    scaler = task.scalers[0]
    out: dict[str, Any] = {}
    ctx = mp.get_context("spawn")
    spec = task.spec
    t0 = time.perf_counter()
    with ctx.Pool(
        int(rc["workers"]), initializer=_ref_init, initargs=(spec.n_detuning, spec.detuning_rule, spec.detuning_spacing)
    ) as pool:
        for kind in KINDS:
            bank = task.test_banks[("resonant", kind)]
            prot = protocol_for(task.cfg, kind)
            items, owners = [], []
            for i in range(n):
                env = bank.env(i).to_dict()
                for seq in prot.sequences(MagnetSetting(), cw_ns=int(rc["cw_ns"])):
                    items.append((seq.to_dict(), env))
                    owners.append(i)
            rates = pool.map(_ref_one, items, chunksize=4)
            feats = np.stack(
                [prot.features_from_rates([r for r, o in zip(rates, owners, strict=True) if o == i]) for i in range(n)]
            ).astype(np.float32)
            sur = bank.means[:n]
            rel = float(np.max(np.abs(feats - sur)) / np.max(sur))
            model = task.models[("resonant", kind, "cnn")]
            res: dict[str, Any] = {"max_rel_feature_diff": rel}
            for shots in (task.headline, 10_000):
                x_ref = torch.as_tensor(feats * shots, device=task.device)
                x_sur = torch.as_tensor(sur * shots, device=task.device)
                c = log_shots(torch.full((n,), float(shots), device=task.device))
                b_r = scaler.decode(predict(model, features(x_ref), c)[0].cpu().numpy().astype(np.float64))[0]
                b_s = scaler.decode(predict(model, features(x_sur), c)[0].cpu().numpy().astype(np.float64))[0]
                res[f"noiseless_{shots}_max_pred_diff_ut"] = float(np.max(np.linalg.norm(b_r - b_s, axis=1)) * 1e6)
                res[f"noiseless_{shots}_err_ref_ut"] = (np.linalg.norm(b_r - bank.fields_t[:n], axis=1) * 1e6).tolist()
                res[f"noiseless_{shots}_err_sur_ut"] = (np.linalg.norm(b_s - bank.fields_t[:n], axis=1) * 1e6).tolist()
            # noisy: 64 draws per field at the headline shots, same generator seed for both sources
            errs = {}
            for src, mean in (("ref", feats), ("sur", sur)):
                tb = FieldBank(prot, bank.fields_t[:n], bank.nuisance[:n], mean)
                b, _, t = task.evaluate(model, tb, scaler, 0, task.headline, repeats=64)
                errs[src] = float(np.sqrt(np.mean(np.sum((b - t) ** 2, axis=1))) * 1e6)
            res["noisy_rmse_ref_ut"], res["noisy_rmse_sur_ut"] = errs["ref"], errs["sur"]
            out[kind] = res
    out["seconds"] = time.perf_counter() - t0
    return out


# --- raw example traces -----------------------------------------------------------------------------------------------


def log_example_traces(task: FieldTask) -> None:
    """Time-domain traces of seed-0 test field 0 at an on-line and an off-line frequency, both protocols and
    conditions, 10 shots each (the banks hold the noiseless features of everything; data/*.npz)."""
    sim = SurrogateSimulator(task.device, task.spec.options())
    rng = np.random.default_rng([task.base_seed, 3])
    for cname in ("resonant", "detuned"):
        cond = condition_for(task.cfg, cname)
        for kind in KINDS:
            bank = task.test_banks[(cname, kind)]
            env = bank.env(0)
            prot = protocol_for(task.cfg, kind, cond.mw_offset_hz)
            seqs = prot.sequences(MagnetSetting(), cw_ns=6000)
            lines = transition_lines_hz(bank.fields_t[0])
            f = prot.freqs_hz()
            i_on = int(np.argmin(np.abs(f - lines.min())))
            for i, which in ((i_on, "on_line"), (0, "sweep_start")):
                tr = sim.run(seqs[i], env, shots=10, rng=rng)
                task.run.log_trace(tr, seqs[i], condition=cname, protocol=kind, point=which, field_t=bank.fields_t[0])


# --- plots ------------------------------------------------------------------------------------------------------------


def _style(ax: Any) -> None:
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_inputs(task: FieldTask) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    rng = np.random.default_rng(5)
    i = 0
    for ax, kind in zip(axes[:2], KINDS, strict=True):
        f = protocol_for(task.cfg, kind).freqs_hz() / 1e9
        for cname, shots, color, lw in (
            ("resonant", 100, "#9fc4ef", 0.8),
            ("resonant", 10_000, COLORS["cnn"], 1.4),
            ("detuned", 10_000, "#8a8985", 1.0),
        ):
            m = task.test_banks[(cname, kind)].means[i].sum(axis=0).astype(np.float64)
            y = rng.poisson(m * shots)
            ax.plot(f, y / y.mean(), color=color, lw=lw, label=f"{cname}, {shots} shots")
        for ln in transition_lines_hz(task.test_banks[("resonant", kind)].fields_t[i]):
            ax.axvline(ln / 1e9, color="#c3c2b7", lw=0.6, zorder=0)
        ax.set_xlabel("MW frequency (GHz; detuned: applied + 0.4 GHz)")
        ax.set_ylabel("counts / mean")
        ax.set_title(f"{'CW ODMR' if kind == 'cw' else 'pulsed ODMR (readout sum)'}")
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    ax = axes[2]
    bank = task.test_banks[("resonant", "pulsed")]
    m = bank.means[i].astype(np.float64)
    fdip = int(np.argmin(m.sum(axis=0)))
    t = np.arange(m.shape[0]) * 10
    ax.plot(t, m[:, fdip] * 1e3, color=COLORS["cnn"], lw=1.6, label="on a line")
    ax.plot(t, np.median(m, axis=1) * 1e3, color="#8a8985", lw=1.6, label="off resonance")
    ax.set_xlabel("time in readout pulse (ns)")
    ax.set_ylabel("mean counts per shot per bin (x1e-3)")
    ax.set_title("pulsed readout transient")
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_error_vs_shots(task: FieldTask, fit_rows: list[dict[str, Any]]) -> Figure:
    df = pd.DataFrame(task.rows)
    fdf = pd.DataFrame(fit_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    rng = np.random.default_rng(0)
    for ax, kind in zip(axes, KINDS, strict=True):
        for mname in MODELS:
            for cname, ls in (("resonant", "-"), ("detuned", "--")):
                d = df[(df.protocol == kind) & (df.model == mname) & (df.condition == cname)]
                xs, mu, lo, hi = [], [], [], []
                for level, g in d.groupby("shots"):
                    v, a, b = bootstrap_ci(g.rmse_ut.to_numpy(), rng=rng, n_boot=2000)
                    xs.append(level)
                    mu.append(v)
                    lo.append(a)
                    hi.append(b)
                ax.plot(xs, mu, ls, color=COLORS[mname], lw=2, marker="o", ms=4, label=f"{LABELS[mname]}, {cname}")
                ax.fill_between(xs, lo, hi, color=COLORS[mname], alpha=0.15, lw=0)
        for cname, ls in (("resonant", "-"), ("detuned", "--")):
            d = fdf[(fdf.protocol == kind) & (fdf.condition == cname)].sort_values("shots")
            ax.plot(d.shots, d.rmse_ut, ls, color=COLORS["fit"], lw=2, marker="s", ms=4, label=f"physics fit, {cname}")
        prior = df[df.protocol == kind].prior_rmse_ut.mean()
        ax.axhline(prior, color=COLORS["prior"], lw=1.2, ls=":", label="prior only")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("shots per frequency point")
        ax.set_title("CW ODMR" if kind == "cw" else "pulsed ODMR")
        _style(ax)
    axes[0].set_ylabel("field error, RMS |B_hat - B| (uT)")
    axes[1].legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
    return fig


def plot_calibration(task: FieldTask, fit_rows: list[dict[str, Any]]) -> Figure:
    df = pd.DataFrame(task.rows)
    fdf = pd.DataFrame(fit_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    for kind, marker in (("cw", "o"), ("pulsed", "^")):
        for mname in MODELS:
            d = df[(df.protocol == kind) & (df.model == mname) & (df.condition == "resonant")]
            g = d.groupby("shots").z_rms.mean()
            ax.plot(
                g.index, g.values, color=COLORS[mname], marker=marker, lw=1.6, ms=5, label=f"{LABELS[mname]}, {kind}"
            )
        d = fdf[(fdf.protocol == kind) & (fdf.condition == "resonant")].sort_values("shots")
        ax.plot(d.shots, d.z_rms, color=COLORS["fit"], marker=marker, lw=1.6, ms=5, label=f"physics fit, {kind}")
    ax.axhline(1.0, color="#52514e", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("shots per frequency point")
    ax.set_ylabel("RMS of error / predicted sigma (1 = calibrated)")
    ax.set_title("uncertainty calibration, resonant")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    _style(ax)
    ax = axes[1]
    from scipy.stats import norm

    levels = np.linspace(0.05, 0.99, 30)
    for kind, ls in (("cw", "-"), ("pulsed", "--")):
        z = np.abs(np.concatenate(task.z_headline[("resonant", kind, "cnn")]).ravel())
        emp = [np.mean(z < norm.ppf(0.5 + q / 2)) for q in levels]
        ax.plot(levels, emp, ls, color=COLORS["cnn"], lw=2, label=f"CNN, {kind}")
    ax.plot([0, 1], [0, 1], color="#52514e", lw=0.8)
    ax.set_xlabel("nominal coverage of central interval")
    ax.set_ylabel("empirical coverage")
    ax.set_title(f"CNN reliability at {task.headline} shots (all seeds)")
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_quantum_test(task: FieldTask, qt: QuantumTestResult) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = 0
    ticks, labels = [], []
    for kind in KINDS:
        for mname in MODELS:
            s = qt.summary(f"{kind}_{mname}_rmse_ut")
            for j, (cname, alpha) in enumerate((("resonant", 1.0), ("detuned", 0.45))):
                v, lo, hi = s[cname]
                ax.bar(x + 0.36 * j, v, 0.32, color=COLORS[mname], alpha=alpha, edgecolor="none")
                ax.errorbar(x + 0.36 * j, v, yerr=[[v - lo], [hi - v]], color="#0b0b0b", lw=1, capsize=3)
            ticks.append(x + 0.18)
            labels.append(f"{kind}\n{LABELS[mname]}")
            x += 1
        prior = np.mean(qt.values["resonant"][f"{kind}_prior_rmse_ut"])
        ax.hlines(prior, x - 2.2, x - 0.3, color=COLORS["prior"], ls=":", lw=1.4)
    ax.set_xticks(ticks, labels, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel(f"RMS field error at {task.headline} shots (uT)")
    ax.set_title("quantum test: resonant (solid) vs detuned +0.4 GHz (light); dotted = prior only; 95 % CI over seeds")
    ax.title.set_fontsize(9)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_timing(task: FieldTask, tp: dict[str, Any]) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    pi0 = protocol_for(task.cfg, "pulsed").pi_ns
    s = np.array(tp["shifts_ns"]) + pi0
    for mname in MODELS:
        for cname, ls in (("resonant", "-"), ("detuned", "--")):
            c = np.array(tp["curves"][f"{cname}_{mname}"])
            ax.errorbar(
                s,
                c[:, 0],
                yerr=[c[:, 0] - c[:, 1], c[:, 2] - c[:, 0]],
                fmt=ls,
                color=COLORS[mname],
                marker="o",
                ms=4,
                lw=1.6,
                capsize=2,
                label=f"{LABELS[mname]}, {cname}",
            )
    ax.axvline(pi0, color="#c3c2b7", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("MW pulse length (ns), decoder trained at the pi pulse (line)")
    ax.set_ylabel(f"RMS field error at {task.headline} shots (uT)")
    ax.set_title("frozen pulsed decoders vs pulse length (seed 0)")
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    ax = axes[1]
    t = np.arange(len(tp["contrast_per_bin"])) * 10
    for cname, color in (("resonant", COLORS["cnn"]), ("detuned", "#8a8985")):
        v = np.array(tp["saliency"][cname])
        ax.plot(t, v / v.max(), color=color, lw=1.8, label=f"CNN saliency, {cname}")
    c = np.array(tp["contrast_per_bin"])
    ax.plot(t, c / c.max(), color=COLORS["fit"], lw=1.8, label="spin contrast on a line (physics)")
    w = np.array(tp["linear_time_weights"])
    ax.plot(t, w / np.abs(w).max(), color=COLORS["linear"], lw=1.4, label="linear model time weights")
    ax.set_xlabel("time in readout pulse (ns)")
    ax.set_ylabel("normalised")
    ax.set_title("which readout bins matter")
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


# --- report -----------------------------------------------------------------------------------------------------------


def _ci(v: tuple[float, float, float], fmt: str = "{:.3g}") -> str:
    return f"{fmt.format(v[0])} [{fmt.format(v[1])}, {fmt.format(v[2])}]"


def report_body(
    task: FieldTask,
    qt: QuantumTestResult,
    fit_rows: list[dict[str, Any]],
    tp: dict[str, Any],
    ref: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    cfg = task.cfg
    df = pd.DataFrame(task.rows)
    fdf = pd.DataFrame(fit_rows)
    rng = np.random.default_rng(1)
    pr = protocol_for(cfg, "pulsed")
    out = [
        "Task 1 of the brief: estimate a hidden magnetic field from the glow. Two fixed sensor protocols, three "
        "estimators, all trained and tested on the fast surrogate, the decoders re-checked on the QuTiP reference, "
        "and the quantum test (every learned estimator retrained and re-evaluated with the microwaves detuned "
        f"{cfg['detuned_offset_hz'] / 1e6:.0f} MHz off resonance) over {len(qt.seeds)} seeds.",
        "",
        "## Task and data",
        "",
        f"- **Hidden field**: |B| uniform in [{cfg['prior']['b_min_t'] * 1e3:g}, {cfg['prior']['b_max_t'] * 1e3:g}] "
        "mT, isotropic direction. An ODMR spectrum cannot tell the sign of B along an NV axis nor which line belongs "
        "to which axis, so B is identifiable only up to the 48 signed permutations of the crystal axes; the target "
        "is the canonical representative, B folded into the wedge Bx >= By >= Bz >= 0 (generically unique). Errors "
        "are the vector error |B_hat - B| in µT.",
        "- **Nuisance** (one draw per group of "
        f"{task.spec.fields_per_env} fields): T2* log-uniform 0.5-2 µs, Rabi full scale 8-12 MHz, optical beta "
        "0.07-0.13, collection 2e-3-4e-3, background 5e3-2e4 counts/s, strain E 0-0.3 MHz. Decoders see counts "
        "normalised by their own mean, so absolute brightness is not a cue.",
        f"- **Protocols** (sweep {pr.f_start_hz / 1e9:.2f}-{pr.f_stop_hz / 1e9:.2f} GHz, {pr.n_freq} points, "
        f"{pr.step_hz / 1e6:g} MHz step): *CW ODMR*, laser + MW (amplitude {pr.cw_amplitude}) in the steady state, "
        f"{pr.cw_shot_ns / 1e3:g} µs dwell per shot per point, one number per point; *pulsed ODMR*, "
        f"{pr.init_ns / 1e3:g} µs init, {pr.settle_ns / 1e3:g} µs dark, a {pr.pi_ns} ns MW pulse at amplitude "
        f"{pr.pulsed_amplitude} (a pi pulse at the nominal Rabi full scale), {pr.readout_ns} ns readout recorded in "
        f"{pr.n_time} bins of 10 ns: a {pr.n_time} x {pr.n_freq} input. Shot lengths "
        f"{protocol_for(cfg, 'cw').shot_ns / 1e3:g} µs (CW) and {pr.shot_ns / 1e3:g} µs (pulsed), so equal shots "
        "are nearly equal time budgets.",
        f"- **Datasets**: per seed {task.n_train} train / {task.n_val} validation / {task.n_test} test fields; the "
        "surrogate gives the noiseless mean counts of every input cell (M = "
        f"{task.spec.n_detuning} weighted-uniform detuning nodes, spacing {task.spec.detuning_spacing:g} widths), "
        f"and Poisson noise for a log-uniform number of shots in [{cfg['train']['shots_min']}, "
        f"{cfg['train']['shots_max']}] is drawn fresh at every training step. CW points use the surrogate's "
        "steady-state solve, pulsed sweeps share their init + settle prefix (exact fast paths, "
        f"docs/notes/surrogate.md). Bank generation took {task.bank_seconds['cw']:.0f} s (CW) and "
        f"{task.bank_seconds['pulsed']:.0f} s (pulsed) in total on {summary['device']}.",
        "",
        "## Estimators",
        "",
        "- **CNN decoder** (`glowbox/learn/decoder.py`, `ConvDecoder`): learned 1x1 time-bin mixing (T -> 8), a "
        f"stem and {len(cfg['decoder']['channels'])} stages of residual 1D convolutions along the frequency axis "
        f"(channels {cfg['decoder']['channels']}, kernel {cfg['decoder']['kernel']}, stride-2 downsampling), "
        "flatten (dip positions matter, so no global pooling), log10(shots) appended, MLP to mean and log-variance "
        f"of the three field components; Gaussian NLL, AdamW, {cfg['train']['steps']} steps of "
        f"{cfg['train']['batch']}, best validation NLL kept.",
        "- **Linear time-bin baseline** (`TimeBinLinear`): one learned weighting of the readout bins (a weighted "
        "spectrum), an affine map of it to the field, a log-variance affine in log10(shots); same loss.",
        "- **Physics fit** (`glowbox/analysis/field_fit.py`): eight Lorentzian dips with shared width and contrast "
        "whose positions are the exact H0 transitions of a trial field (E = 0 assumed), coarse grid search over "
        "the wedge then least squares, covariance from the Jacobian; the prior when no dip is significant. Pulsed "
        "input is the sum over the readout bins (a late-window normalisation would cancel the signal here: with "
        "singlet shelving the whole 600 ns readout is spin dependent). Seed 0 only, on the first "
        f"{cfg['fit']['n_fields']} test fields.",
        "",
        "## Error vs shots",
        "",
        "RMS field error (µT), mean over seeds with 95 % bootstrap CI, resonant condition:",
        "",
        "| shots | " + " | ".join(f"{k} {LABELS[m]}" for k in KINDS for m in MODELS) + " | cw physics fit | "
        "pulsed physics fit |",
        "|" + " --- |" * (1 + 2 * len(MODELS) + 2),
    ]
    for level in task.levels:
        cells = [str(level)]
        for kind in KINDS:
            for mname in MODELS:
                d = df[(df.protocol == kind) & (df.model == mname) & (df.condition == "resonant") & (df.shots == level)]
                cells.append(_ci(bootstrap_ci(d.rmse_ut.to_numpy(), rng=rng, n_boot=4000)))
        for kind in KINDS:
            d = fdf[(fdf.protocol == kind) & (fdf.condition == "resonant") & (fdf.shots == level)]
            cells.append(f"{float(d.rmse_ut.iloc[0]):.3g}" if len(d) else "")
        out.append("| " + " | ".join(cells) + " |")
    prior = {k: float(df[df.protocol == k].prior_rmse_ut.mean()) for k in KINDS}
    out += [
        "",
        f"Prior only (predicting the training mean): {prior['cw']:.0f} µT. Physics fit vs CNN on the *same* "
        "fields and noise level (seed 0): "
        + "; ".join(
            f"{r['protocol']} {r['shots']} shots: fit {r['rmse_ut']:.3g} vs CNN {r['cnn_rmse_ut_same_fields']:.3g} µT "
            f"(fit found dips in {100 * r['ok_fraction']:.0f} %)"
            for r in fit_rows
            if r["condition"] == "resonant" and r["shots"] in (task.levels[0], task.headline, task.levels[-1])
        )
        + ".",
        "",
        "## Calibration",
        "",
        "Calibration of the predicted per-component sigma, resonant, mean over seeds: RMS of error / sigma "
        "(1 = calibrated) and coverage of the 68 % / 95 % intervals.",
        "",
        "| shots | " + " | ".join(f"{k} {LABELS[m]}" for k in KINDS for m in MODELS) + " | cw fit | pulsed fit |",
        "|" + " --- |" * (1 + 2 * len(MODELS) + 2),
    ]
    for level in task.levels:
        cells = [str(level)]
        for kind in KINDS:
            for mname in MODELS:
                d = df[(df.protocol == kind) & (df.model == mname) & (df.condition == "resonant") & (df.shots == level)]
                cells.append(f"{d.z_rms.mean():.2f} ({100 * d.cov68.mean():.0f} / {100 * d.cov95.mean():.0f} %)")
        for kind in KINDS:
            d = fdf[(fdf.protocol == kind) & (fdf.condition == "resonant") & (fdf.shots == level)]
            cells.append(
                f"{float(d.z_rms.iloc[0]):.2f} ({100 * float(d.cov68.iloc[0]):.0f} / "
                f"{100 * float(d.cov95.iloc[0]):.0f} %)"
                if len(d)
                else ""
            )
        out.append("| " + " | ".join(cells) + " |")
    out += [
        "",
        "## Quantum test",
        "",
        f"Every learned estimator was retrained from scratch and evaluated with every MW frequency shifted by "
        f"+{cfg['detuned_offset_hz'] / 1e6:.0f} MHz (no line within {task.no_line_margin_hz / 1e6:.0f} MHz of any "
        "MW frequency in the spot checks; `check_no_line_driven`), same fields, nuisance, initialisation and noise "
        f"seeds per seed (paired). Headline: RMS field error at {task.headline} shots; skill = 1 - error / "
        "prior-only error. 95 % percentile-bootstrap CIs over the seeds; gap > 0 means better on resonance.",
        "",
        qt.to_markdown(
            [f"{k}_{m}_rmse_ut" for k in KINDS for m in MODELS] + [f"{k}_{m}_skill" for k in KINDS for m in MODELS],
            fmt="{:.3g}",
        ),
        "",
    ]
    det_skill = [qt.summary(f"{k}_cnn_skill")["detuned"] for k in KINDS]
    out += [
        "Outcome: "
        + "; ".join(
            f"{k} CNN gap {_ci(qt.summary(f'{k}_cnn_rmse_ut')['gap'])} µT "
            f"({'significant' if qt.gap_significant(f'{k}_cnn_rmse_ut') else 'not significant'})"
            for k in KINDS
        )
        + ". Detuned, the CNN's skill is "
        + ", ".join(f"{k} {_ci(s)}" for k, s in zip(KINDS, det_skill, strict=True))
        + ": with no line driven the spectrum carries (almost) no field information and the decoders fall back "
        "to the prior. The physics fit on the detuned spectra found dips in "
        + ", ".join(
            f"{r['protocol']} {100 * r['ok_fraction']:.0f} %"
            for r in fit_rows
            if r["condition"] == "detuned" and r["shots"] == task.headline
        )
        + f" of the cases at {task.headline} shots.",
        "",
        "### Does the learned strategy use microsecond-scale timing?",
        "",
        "Task 1 uses fixed protocols, so the strategy is the protocol plus the learned decoder. The CW protocol has "
        "no timing structure by construction (MW and laser on together, steady state; Rabi-band fraction of its MW "
        f"envelope {tp['cw_rabi_band_fraction']:.2f}). The pulsed protocol's MW pulse has area "
        f"{tp['pulsed_structure']['segments'][0]['area_pi']:.2f} pi at the nominal Rabi frequency (Rabi-band "
        f"fraction {tp['pulsed_rabi_band_fraction']:.2f}). Probe: the seed-0 pulsed decoders, frozen, on test fields "
        "re-simulated with the MW pulse lengthened or shortened (RMS error at "
        f"{task.headline} shots, 95 % bootstrap CI over {cfg['timing']['n_fields']} fields x "
        f"{cfg['timing']['noise_repeats']} noise draws):",
        "",
        "| pulse length (ns) | "
        + " | ".join(f"{LABELS[m]}, {c}" for c in ("resonant", "detuned") for m in MODELS)
        + " |",
        "|" + " --- |" * (1 + 2 * len(MODELS)),
    ]
    for j, sh in enumerate(tp["shifts_ns"]):
        cells = [str(pr.pi_ns + sh)]
        for c in ("resonant", "detuned"):
            for m in MODELS:
                cells.append(_ci(tuple(tp["curves"][f"{c}_{m}"][j])))
        out.append("| " + " | ".join(cells) + " |")
    sal = tp["saliency_first_300ns_fraction"]
    out += [
        "",
        f"Readout saliency of the resonant CNN: {100 * sal['resonant']:.0f} % of |d|B_hat|/d input| falls in the "
        f"first 300 ns of the readout (uniform would be {100 * tp['uniform_first_300ns_fraction']:.0f} %); the "
        "physical spin contrast per readout bin and the linear model's learned time weights are in the timing "
        "figure. Logged verdict `uses_us_timing` (resonant error rises beyond its CI for |shift| >= 100 ns while "
        f"the detuned one does not): **{tp['uses_us_timing']}**"
        + (
            ": the resonant decoder depends on the MW pulse being a pi pulse, i.e. on Rabi timing at the ~100 ns scale."
            if tp["uses_us_timing"]
            else "."
        ),
        "",
        "## Reference check (QuTiP)",
        "",
        f"{cfg['reference']['n_fields']} seed-0 test fields re-simulated with the QuTiP reference (same detuning "
        f"nodes; CW points {cfg['reference']['cw_ns'] / 1e3:g} µs long, last 2 µs) and fed to the surrogate-trained "
        f"CNN ({ref['seconds']:.0f} s on the CPU pool):",
        "",
        "| protocol | max feature diff / peak | max pred. diff, noiseless (µT) | RMS error at "
        f"{task.headline} shots ref / surrogate (µT, 64 draws) |",
        "| --- | --- | --- | --- |",
    ]
    for k in KINDS:
        r = ref[k]
        out.append(
            f"| {k} | {r['max_rel_feature_diff']:.1e} | {r[f'noiseless_{task.headline}_max_pred_diff_ut']:.3g} "
            f"| {r['noisy_rmse_ref_ut']:.3g} / {r['noisy_rmse_sur_ut']:.3g} |"
        )
    out += [
        "",
        "## Notes and open issues",
        "",
        "- Field priors and nuisance ranges are assumptions (`FieldPrior`), not a characterised sample; the model "
        "excludes hyperfine structure and excited-state mixing (docs/physics.md section 10), which a real spectrum "
        "has. No claim of quantum advantage: the gap only shows the estimate needs the driven spin resonance.",
        "- The physics fit assumes E = 0 and a Lorentzian lineshape (the pulsed line is not exactly Lorentzian); "
        "it is evaluated for seed 0 only.",
        "- Five seeds make the bootstrap CIs coarse; per-seed values are in the table.",
    ]
    return "\n".join(out)


# --- main -------------------------------------------------------------------------------------------------------------


def main(
    cfg: dict[str, Any], output_root: Path | None = None, report_dir: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    """Run the milestone; returns (report path, summary numbers)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw: dict[str, Any] = {} if output_root is None else {"output_root": output_root}
    t_start = time.perf_counter()
    with RunLog(str(cfg["name"]), config=cfg, seed=int(cfg["seed"]), **kw) as run:
        task = FieldTask(cfg, run, device)
        conds = (RESONANT, detuned_condition(float(cfg["detuned_offset_hz"])))
        hib = {f"{k}_{m}_skill": True for k in KINDS for m in MODELS}
        qt = run_quantum_test(task, [int(s) for s in cfg["seeds"]], conds, higher_is_better=hib)
        print("[m2] physics fit", flush=True)
        fit_rows = run_physics_fit(task, int(cfg["fit"]["workers"]), int(cfg["fit"]["n_fields"]))
        print("[m2] timing probe", flush=True)
        tp = run_timing_probe(task)
        print("[m2] reference check", flush=True)
        ref = run_reference_check(task)
        log_example_traces(task)

        pd.DataFrame(task.rows).to_parquet(run.dir / "results.parquet")
        pd.DataFrame(fit_rows).to_parquet(run.dir / "fit_results.parquet")
        (run.dir / "train_history.json").write_text(json.dumps(task.histories), encoding="utf-8")
        (run.dir / "timing_probe.json").write_text(json.dumps(tp, default=float), encoding="utf-8")
        (run.dir / "reference_check.json").write_text(json.dumps(ref), encoding="utf-8")
        (run.dir / "quantum_test.json").write_text(
            json.dumps({"seeds": qt.seeds, "values": qt.values}), encoding="utf-8"
        )

        figs = {
            "inputs": plot_inputs(task),
            "error-vs-shots": plot_error_vs_shots(task, fit_rows),
            "calibration": plot_calibration(task, fit_rows),
            "quantum-test": plot_quantum_test(task, qt),
            "timing": plot_timing(task, tp),
        }
        paths = [run.save_figure(f, name, dpi=90) for name, f in figs.items()]
        for f in figs.values():
            plt.close(f)

        summary: dict[str, Any] = {
            "device": f"{device.type} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}".strip()
        }
        for k in KINDS:
            for m in MODELS:
                s = qt.summary(f"{k}_{m}_rmse_ut")
                summary[f"{k}_{m}_rmse_ut_resonant"] = s["resonant"][0]
                summary[f"{k}_{m}_rmse_ut_detuned"] = s["detuned"][0]
                summary[f"{k}_{m}_gap_ut"] = s["gap"][0]
                summary[f"{k}_{m}_gap_lo_ut"] = s["gap"][1]
                summary[f"{k}_{m}_gap_hi_ut"] = s["gap"][2]
            summary[f"{k}_prior_rmse_ut"] = float(np.mean(qt.values["resonant"][f"{k}_prior_rmse_ut"]))
            fr = [
                r
                for r in fit_rows
                if r["protocol"] == k and r["condition"] == "resonant" and r["shots"] == task.headline
            ]
            summary[f"{k}_fit_rmse_ut_resonant"] = fr[0]["rmse_ut"] if fr else float("nan")
            summary[f"{k}_ref_max_rel_feature_diff"] = ref[k]["max_rel_feature_diff"]
        summary["uses_us_timing_pulsed"] = float(tp["uses_us_timing"])
        summary["no_line_margin_mhz"] = task.no_line_margin_hz / 1e6
        summary["wall_seconds"] = time.perf_counter() - t_start
        run.log_metrics(step=0, **{k: v for k, v in summary.items() if isinstance(v, int | float)})
        body = report_body(task, qt, fit_rows, tp, ref, summary)
    rdir = report_dir if report_dir is not None else run.dir / "report"
    title = "Milestone 2: Task 1, field estimation from the glow, and the quantum test"
    path = write_report(run.dir, title, body, figures=paths, report_dir=rdir, name=REPORT_NAME)
    return path, summary


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="tiny banks and short training (smoke run)")
    p.add_argument("--report-dir", type=Path, default=None, help="where to write the report (e.g. docs/results)")
    p.add_argument("--output-root", type=Path, default=None, help="run directories (default: <repo>/outputs)")
    a = p.parse_args()
    path, summary = main(load_config(a.config, a.quick), a.output_root, a.report_dir)
    for k, v in summary.items():
        print(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}")
    print(f"report: {path}")


if __name__ == "__main__":
    cli()
