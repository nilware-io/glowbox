"""Milestone 3, Task 2: store a short bit pattern in the pulse response and recall it, with the quantum test.

    uv run python -m glowbox.experiments.m3_bit_memory [--config PATH] [--quick] [--report-dir DIR]

Protocol (glowbox.learn.bit_memory): init, settle, a write block whose MW segments are a fixed random function of
the L bits (glowbox.learn.reservoir.RandomPulseEncoder), a dark delay, optionally a fixed pi/2 read pulse, and a
time-resolved readout. Three readouts recall the bits from the readout counts: the reservoir's linear readout
(ridge), a CNN decoder (BCE, fresh shot noise every step) and the Bayes-optimal oracle (exact templates + Poisson
model). Recall is measured against pattern length, delay, shots and read arm.

The quantum test (glowbox.analysis.quantum_test) runs everything per seed twice: as designed, and with every MW
frequency shifted by ``detuned_offset_hz`` (no line driven). Timing: recall vs delay with and without the read
pulse (does the memory live in coherences, on the T2* scale?), and the write block's MW durations shifted with the
readouts frozen and retrained (does recall rely on the Rabi-scale pulse areas?). Nothing in the encoding is
learned, so ``strategy_learns_timing`` is False by construction.

Outputs: ``outputs/<run-id>/`` (run.yaml, raw example traces, metrics, results / timing parquet tables, seed-0
codebooks, plots) and a report (``outputs/<run-id>/report/`` unless ``--report-dir``).
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
from glowbox.experiments.m1_odmr_rabi import _merge, load_config
from glowbox.experiments.m2_field_estimation import _style
from glowbox.learn.bit_memory import (
    MemoryProtocol,
    bitwise_map,
    codebook,
    device_lines_hz,
    make_encoder,
    recall_metrics,
    sequences,
    template_separation,
)
from glowbox.learn.decoder import (
    ConvDecoder,
    DecoderConfig,
    TrainConfig,
    bce_logits_loss,
    predict,
    train_decoder,
)
from glowbox.learn.field_data import FieldPrior, log_shots, sample_shots
from glowbox.learn.reservoir import EncoderSpec, RidgeReadout, full_sequence
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate import SurrogateSimulator
from glowbox.sim.surrogate.model import ModelOptions

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "m3_bit_memory.yaml"
REPORT_NAME = "m3-bit-memory"
READOUTS = ("oracle", "ridge", "cnn")
COLORS = {"oracle": "#1baf7a", "ridge": "#eb6834", "cnn": "#2a78d6", "chance": "#8a8985"}
LABELS = {"oracle": "Bayes-optimal (oracle)", "ridge": "reservoir readout (ridge)", "cnn": "CNN decoder"}

__all__ = ["DEFAULT_CONFIG", "MemoryTask", "_merge", "load_config", "main"]


def input_saliency(model: torch.nn.Module, x: torch.Tensor, c: torch.Tensor, output: Any) -> np.ndarray:
    """Mean |d output / d x| over the batch, (T, F) (``time_saliency`` sums over F; here F is the time axis)."""
    model.eval()
    xg = x.detach().clone().requires_grad_(True)
    mean, _ = model(xg, c)
    output(mean).sum().backward()
    assert xg.grad is not None
    out: np.ndarray = xg.grad.abs().mean(dim=0).detach().cpu().numpy()
    return out


def condition_for(cfg: dict[str, Any], name: str) -> Condition:
    return RESONANT if name == "resonant" else detuned_condition(float(cfg["detuned_offset_hz"]))


def device_env(cfg: dict[str, Any], seed: int, base_seed: int) -> EnvParams:
    """The seed's device: nuisance draw from the prior, the known bias field."""
    pc: dict[str, Any] = {k: tuple(v) for k, v in cfg["device"]["prior"].items()}
    env = FieldPrior(**pc).sample_env(np.random.default_rng([base_seed, seed, 1]))
    return dataclasses.replace(env, b_field_t=tuple(float(x) for x in cfg["device"]["bias_field_t"]))  # type: ignore[arg-type]


def protocol_for(cfg: dict[str, Any], bits: int, delay_ns: int, arm: str) -> MemoryProtocol:
    p = dict(cfg["protocol"])
    line = int(p.pop("read_line"))
    read_line = line if arm == "pi2" else None
    enc = EncoderSpec(n_inputs=bits, **cfg["encoder"])
    return MemoryProtocol(n_bits=bits, delay_ns=int(delay_ns), read_line=read_line, encoder=enc, **p)


def _tgen(device: torch.device, *key: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(np.random.default_rng(list(key)).integers(2**62)))
    return g


class LevelDecoder(torch.nn.Module):
    """A :class:`ConvDecoder` on counts per shot relative to a fixed reference (the codebook's mean template):
    ``10 (x / ref - 1)``. Unlike ``field_data.features`` (normalised by the sample's own mean, M2, where brightness
    was a nuisance), the overall level is kept: each seed is one device, and the pattern changes the level."""

    def __init__(self, ref: torch.Tensor, net: ConvDecoder) -> None:
        super().__init__()
        self.register_buffer("ref", ref)
        self.net = net

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ref = self.get_buffer("ref")
        out: tuple[torch.Tensor, torch.Tensor] = self.net(10.0 * (x / ref - 1.0), c)
        return out


@dataclasses.dataclass
class Codebook:
    patterns: np.ndarray  # (P, L) {0, 1}
    templates: np.ndarray  # (P, T) mean counts per shot


class MemoryTask:
    """The quantum-test task function plus everything recorded along the way."""

    def __init__(self, cfg: dict[str, Any], run: RunLog, device: torch.device) -> None:
        self.cfg = cfg
        self.run = run
        self.device = device
        self.base_seed = int(cfg["seed"])
        self.options = ModelOptions(n_detuning=int(cfg["device"]["n_detuning"]))
        g = cfg["grid"]
        self.bits = [int(b) for b in g["bits"]]
        self.delays = [int(d) for d in g["delays_ns"]]
        self.arms = [str(a) for a in g["read_arms"]]
        self.shots = [int(s) for s in g["shots"]]
        h = cfg["headline"]
        self.h_bits, self.h_delay, self.h_shots, self.h_arm = (
            int(h["bits"]),
            int(h["delay_ns"]),
            int(h["shots"]),
            str(h["read"]),
        )
        self.cnn_pairs = [(int(a), int(b)) for a, b in cfg["cnn"]["pairs"]]
        self.rows: list[dict[str, Any]] = []
        self.timing_rows: list[dict[str, Any]] = []
        self.saliency: dict[str, Any] = {}
        self.encoding: list[dict[str, Any]] = []
        self.books: dict[tuple[str, int, str, int, int], Codebook] = {}  # seed 0 only
        self.models: dict[tuple[str, int, int], torch.nn.Module] = {}  # seed 0 resonant only
        self.call_models: dict[tuple[int, int], torch.nn.Module] = {}  # CNNs of the current task call
        self.ridges: dict[tuple[str, int], RidgeReadout] = {}  # seed 0 headline
        self.envs: dict[int, EnvParams] = {}
        self.histories: dict[str, Any] = {}
        self.no_line_margin_hz = math.inf
        self.seconds: dict[str, float] = {"codebooks": 0.0, "cnn": 0.0}

    # --- data -----------------------------------------------------------------------------------------------------

    def grid_pairs(self) -> list[tuple[int, int]]:
        """(bits, delay) pairs evaluated: every L at the headline delay and every delay at the headline L (the two
        sweeps of the report; the full product would spend most of its time on 4096-pattern codebooks)."""
        pairs = [(b, self.h_delay) for b in self.bits] + [(self.h_bits, d) for d in self.delays]
        return list(dict.fromkeys(pairs))

    def env(self, seed: int) -> EnvParams:
        if seed not in self.envs:
            self.envs[seed] = device_env(self.cfg, seed, self.base_seed)
        return self.envs[seed]

    def encoder(self, seed: int, bits: int, env: EnvParams) -> Any:
        return make_encoder(protocol_for(self.cfg, bits, self.h_delay, "none"), env, 1000 * seed + bits)

    def book(self, cond: Condition, seed: int, bits: int, arm: str, delay: int, shift_ns: int = 0) -> Codebook:
        env = self.env(seed)
        prot = protocol_for(self.cfg, bits, delay, arm)
        t0 = time.perf_counter()
        pats, tm = codebook(
            prot,
            self.encoder(seed, bits, env),
            env,
            offset_hz=cond.mw_offset_hz,
            duration_shift_ns=shift_ns,
            options=self.options,
            device=self.device,
        )
        self.seconds["codebooks"] += time.perf_counter() - t0
        cb = Codebook(pats, tm)
        if seed == 0 and shift_ns == 0:
            self.books[(cond.name, seed, arm, bits, delay)] = cb
        return cb

    def draws(self, cb: Codebook, n: int, shots: float, *key: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(pattern index (n,), counts (n, T)) at ``shots``; deterministic in ``key``."""
        g = _tgen(self.device, *key)
        idx = torch.randint(0, cb.patterns.shape[0], (n,), generator=g, device=self.device)
        tm = torch.as_tensor(cb.templates, dtype=torch.float32, device=self.device)
        return idx, torch.poisson(tm[idx] * float(shots), generator=g)

    # --- readouts -------------------------------------------------------------------------------------------------

    def oracle_bits(self, cb: Codebook, counts: torch.Tensor, shots: float) -> np.ndarray:
        tm = torch.as_tensor(cb.templates, dtype=torch.float64, device=self.device)
        pt = torch.as_tensor(cb.patterns, device=self.device)
        return (bitwise_map(counts, shots, tm, pt) > 0.5).cpu().numpy().astype(np.int64)

    def train_cnn(self, cb: Codebook, seed: int, tag: str) -> torch.nn.Module:
        c = self.cfg["cnn"]
        dc = self.cfg["decoder"]
        dcfg = DecoderConfig(
            mix_channels=int(dc["mix_channels"]),
            channels=tuple(int(x) for x in dc["channels"]),
            kernel=int(dc["kernel"]),
            hidden=int(dc["hidden"]),
        )
        t_bins = cb.templates.shape[1]
        torch.manual_seed(int(np.random.default_rng([self.base_seed, seed, cb.patterns.shape[1], 7]).integers(2**31)))
        tm = torch.as_tensor(cb.templates, dtype=torch.float32, device=self.device)[:, None, :]
        model = LevelDecoder(tm.mean(dim=0), ConvDecoder(1, t_bins, cb.patterns.shape[1], dcfg)).to(self.device)
        pt = torch.as_tensor(cb.patterns, dtype=torch.float32, device=self.device)
        lo, hi = float(c["shots_min"]), float(c["shots_max"])
        g = _tgen(self.device, self.base_seed, seed, cb.patterns.shape[1], 8)

        def batch(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            idx = torch.randint(0, tm.shape[0], (n,), generator=gen, device=self.device)
            shots = sample_shots(n, lo, hi, gen, self.device)
            counts = torch.poisson(tm[idx] * shots[:, None, None], generator=gen)
            return counts / shots[:, None, None], log_shots(shots), pt[idx]

        gv = _tgen(self.device, self.base_seed, seed, cb.patterns.shape[1], 9)
        val = [batch(256, gv) for _ in range(max(1, int(c["n_val"]) // 256))]
        tc = TrainConfig(
            steps=int(c["steps"]),
            batch=int(c["batch"]),
            lr=float(c["lr"]),
            weight_decay=float(c["weight_decay"]),
            eval_every=int(c["eval_every"]),
            mse_warmup=0.0,
        )
        t0 = time.perf_counter()
        hist = train_decoder(model, lambda _s: batch(tc.batch, g), val, tc, loss=bce_logits_loss)
        self.seconds["cnn"] += time.perf_counter() - t0
        self.histories[tag] = dataclasses.asdict(hist)
        return model

    def cnn_bits(self, model: torch.nn.Module, counts: torch.Tensor, shots: float) -> np.ndarray:
        x = counts[:, None, :] / float(shots)
        c = log_shots(torch.full((counts.shape[0],), float(shots), device=self.device))
        logits, _ = predict(model, x, c)
        return (logits > 0).cpu().numpy().astype(np.int64)

    # --- the task function ----------------------------------------------------------------------------------------

    def __call__(self, cond: Condition, seed: int) -> dict[str, float]:
        print(f"[m3-bits] {cond.name} seed {seed}", flush=True)
        env = self.env(seed)
        if cond.detuned:
            lines = device_lines_hz(env)
            enc = self.encoder(seed, self.h_bits, env)
            freqs = np.concatenate([enc.base_hz + s * enc.spec.freq_span_hz for s in (-1, 1)])
            freqs = np.concatenate([freqs, lines[:1]]) + cond.mw_offset_hz
            ok, margin = check_no_line_driven(freqs, np.asarray(env.b_field_t), env)
            if not ok:
                raise RuntimeError(f"detuned condition drives a line (margin {margin / 1e6:.1f} MHz)")
            self.no_line_margin_hz = min(self.no_line_margin_hz, margin)
        nr, nt = int(self.cfg["ridge"]["n_train"]), int(self.cfg["ridge"]["n_test"])
        head: dict[str, float] = {}
        self.call_models = {}
        for arm in self.arms:
            for bits, delay in self.grid_pairs():
                cb = self.book(cond, seed, bits, arm, delay)
                model = None
                if arm == "none" and (bits, delay) in self.cnn_pairs:
                    model = self.train_cnn(cb, seed, f"{cond.name}/{seed}/{bits}/{delay}")
                    self.call_models[(bits, delay)] = model
                    if seed == 0 and cond.name == "resonant":
                        self.models[(arm, bits, delay)] = model
                for shots in self.shots:
                    key = (self.base_seed, seed, bits, delay, self.arms.index(arm), shots)
                    idx_tr, c_tr = self.draws(cb, nr, shots, *key, 0)
                    idx_te, c_te = self.draws(cb, nt, shots, *key, 1)
                    truth = cb.patterns[idx_te.cpu().numpy()]
                    ridge = RidgeReadout().fit(
                        c_tr.cpu().numpy() / shots, 2.0 * cb.patterns[idx_tr.cpu().numpy()] - 1.0
                    )
                    if seed == 0 and cond.name == "resonant" and (bits, delay) == (self.h_bits, self.h_delay):
                        self.ridges[(arm, shots)] = ridge
                    preds = {
                        "oracle": self.oracle_bits(cb, c_te, shots),
                        "ridge": (ridge.predict(c_te.cpu().numpy() / shots) > 0).astype(np.int64),
                    }
                    if model is not None:
                        preds["cnn"] = self.cnn_bits(model, c_te, shots)
                    sep = template_separation(cb.templates, shots) if cb.patterns.shape[0] <= 4096 else math.nan
                    for rname, pb in preds.items():
                        m = recall_metrics(pb, truth)
                        self.rows.append(
                            {
                                "condition": cond.name,
                                "seed": seed,
                                "arm": arm,
                                "bits": bits,
                                "delay_ns": delay,
                                "shots": shots,
                                "readout": rname,
                                "separation": sep,
                                "ridge_lambda": ridge.lam,
                                **m,
                            }
                        )
                        hl = (bits, delay, shots) == (self.h_bits, self.h_delay, self.h_shots)
                        if hl and arm == self.h_arm:
                            head[f"{rname}_bit_acc"] = m["bit_acc"]
                            head[f"{rname}_recalled_bits"] = m["recalled_bits"]
                            head[f"{rname}_pattern_acc"] = m["pattern_acc"]
                        elif hl and arm == "pi2":
                            head[f"pi2_{rname}_bit_acc"] = m["bit_acc"]
        t0 = time.perf_counter()
        self.timing_probe(cond, seed)
        print(
            f"[m3-bits]   timing probe {time.perf_counter() - t0:.0f} s; so far codebooks "
            f"{self.seconds['codebooks']:.0f} s, CNN {self.seconds['cnn']:.0f} s",
            flush=True,
        )
        if cond.name == "resonant":
            self.encoding_stats(seed)
            if seed == 0:
                self.saliency_probe()
        return head

    # --- timing ---------------------------------------------------------------------------------------------------

    def timing_probe(self, cond: Condition, seed: int) -> None:
        """Every write-block MW duration shifted; readouts at the headline: oracle and ridge retrained on the shifted
        codebook, ridge and CNN frozen (trained at shift 0)."""
        bits, delay, shots, arm = self.h_bits, self.h_delay, self.h_shots, self.h_arm
        nr, nt = int(self.cfg["ridge"]["n_train"]), int(self.cfg["ridge"]["n_test"])
        base = self.book(cond, seed, bits, arm, delay) if seed != 0 else self.books[(cond.name, 0, arm, bits, delay)]
        key = (self.base_seed, seed, bits, delay, 99, shots)
        idx0, c0 = self.draws(base, nr, shots, *key, 0)
        frozen = RidgeReadout().fit(c0.cpu().numpy() / shots, 2.0 * base.patterns[idx0.cpu().numpy()] - 1.0)
        model = self.call_models.get((bits, delay)) if arm == "none" else None
        for shift in [int(s) for s in self.cfg["timing"]["duration_shifts_ns"]]:
            cb = base if shift == 0 else self.book(cond, seed, bits, arm, delay, shift_ns=shift)
            idx_tr, c_tr = self.draws(cb, nr, shots, *key, 500 + shift)
            idx_te, c_te = self.draws(cb, nt, shots, *key, 2000 + shift)
            truth = cb.patterns[idx_te.cpu().numpy()]
            retr = RidgeReadout().fit(c_tr.cpu().numpy() / shots, 2.0 * cb.patterns[idx_tr.cpu().numpy()] - 1.0)
            x = c_te.cpu().numpy() / shots
            preds = {
                "oracle": self.oracle_bits(cb, c_te, shots),
                "ridge_retrained": (retr.predict(x) > 0).astype(np.int64),
                "ridge_frozen": (frozen.predict(x) > 0).astype(np.int64),
            }
            if model is not None:
                preds["cnn_frozen"] = self.cnn_bits(model, c_te, shots)
            for rname, pb in preds.items():
                self.timing_rows.append(
                    {
                        "condition": cond.name,
                        "seed": seed,
                        "shift_ns": shift,
                        "readout": rname,
                        **recall_metrics(pb, truth),
                    }
                )

    def encoding_stats(self, seed: int) -> None:
        """Pulse areas of the headline encoder over all patterns (units of pi, |m_perp| = sqrt(2/3))."""
        env = self.env(seed)
        enc = self.encoder(seed, self.h_bits, env)
        from glowbox.learn.bit_memory import all_patterns

        p = enc.params(2.0 * all_patterns(self.h_bits) - 1.0)
        f_r = env.rabi_full_scale_hz * math.sqrt(2.0 / 3.0)
        area = 2.0 * f_r * p.amplitude * p.duration_ns * 1e-9
        total = area.sum(axis=1)
        self.encoding.append(
            {
                "seed": seed,
                "median_area_pi": float(np.median(area)),
                "frac_area_quarter_to_4pi": float(np.mean((area >= 0.25) & (area <= 4.0))),
                "median_block_area_pi": float(np.median(total)),
                "median_duration_ns": float(np.median(p.duration_ns)),
                "t2_star_us": env.t2_star_s * 1e6,
                "rabi_mhz": f_r / 1e6,
            }
        )

    def saliency_probe(self) -> None:
        """CNN saliency per readout bin vs the pattern contrast per bin (seed 0, resonant, headline)."""
        key = (self.h_arm, self.h_bits, self.h_delay)
        if key not in self.models:
            return
        cb = self.books[("resonant", 0, self.h_arm, self.h_bits, self.h_delay)]
        _, counts = self.draws(cb, 1024, self.h_shots, self.base_seed, 0, 55)
        x = counts[:, None, :] / float(self.h_shots)
        c = log_shots(torch.full((counts.shape[0],), float(self.h_shots), device=self.device))
        sal = input_saliency(self.models[key], x, c, lambda m: m.abs().sum(dim=1))[0]
        contrast = cb.templates.std(axis=0) / cb.templates.mean(axis=0)
        r = float(np.corrcoef(sal, contrast)[0, 1])
        n300 = 300 // 10
        self.saliency = {
            "saliency": sal.tolist(),
            "contrast": contrast.tolist(),
            "corr": r,
            "first_300ns_fraction": float(sal[:n300].sum() / sal.sum()),
        }


# --- QuTiP reference check ----------------------------------------------------------------------------------------

_REF: dict[str, Any] = {}


def _ref_init(n_detuning: int) -> None:
    from glowbox.sim.reference import ReferenceSimulator

    _REF["sim"] = ReferenceSimulator(n_detuning=n_detuning, detuning_rule="quantile")


def _ref_one(item: tuple[dict[str, Any], dict[str, Any], int]) -> np.ndarray:
    from glowbox.control.pulse import PulseSequence

    seq, env, n_ro = PulseSequence.from_dict(item[0]), EnvParams.from_dict(item[1]), item[2]
    rate = np.asarray(_REF["sim"].expected_rate_cps(seq, env))
    return rate[-n_ro:] * seq.grid_ns * 1e-9


def run_reference_check(task: MemoryTask) -> dict[str, Any]:
    """Seed-0 headline patterns on the QuTiP reference (both read arms): templates and the frozen readouts."""
    rc = task.cfg["reference"]
    n = int(rc["n_patterns"])
    env = task.env(0)
    out: dict[str, Any] = {}
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(int(rc["workers"]), initializer=_ref_init, initargs=(task.options.n_detuning,)) as pool:
        for arm in task.arms:
            prot = protocol_for(task.cfg, task.h_bits, task.h_delay, arm)
            cb = task.books[("resonant", 0, arm, task.h_bits, task.h_delay)]
            pats = cb.patterns[:n]
            pre, suf = sequences(prot, task.encoder(0, task.h_bits, env), env, pats)
            items = [(full_sequence(pre, row).to_dict(), env.to_dict(), prot.readout_bins) for row in suf.unbatch()]
            ref = np.stack(pool.map(_ref_one, items))
            sur = cb.templates[:n]
            res: dict[str, Any] = {"max_rel_template_diff": float(np.max(np.abs(ref - sur)) / np.max(sur))}
            # signal-relative: the pattern-dependent part of the templates
            dev = cb.templates - cb.templates.mean(axis=0)
            res["max_diff_over_signal"] = float(np.max(np.abs(ref - sur)) / max(np.max(np.abs(dev)), 1e-30))
            shots = task.h_shots
            ridge = task.ridges.get((arm, shots))
            if ridge is not None:
                res["ridge_noiseless_agree"] = float(np.mean((ridge.predict(ref) > 0) == (ridge.predict(sur) > 0)))
                res["ridge_max_score_diff"] = float(np.max(np.abs(ridge.predict(ref) - ridge.predict(sur))))
            # noisy: 64 draws per pattern, same generator for both sources
            accs: dict[str, dict[str, float]] = {}
            for src, tm in (("ref", ref), ("sur", sur)):
                g = torch.Generator(device=task.device)
                g.manual_seed(12345)
                t = torch.as_tensor(np.repeat(tm, 64, axis=0), dtype=torch.float32, device=task.device)
                counts = torch.poisson(t * shots, generator=g)
                truth = np.repeat(pats, 64, axis=0)
                accs[src] = {"oracle": recall_metrics(task.oracle_bits(cb, counts, shots), truth)["bit_acc"]}
                if ridge is not None:
                    pr = (ridge.predict(counts.cpu().numpy() / shots) > 0).astype(np.int64)
                    accs[src]["ridge"] = recall_metrics(pr, truth)["bit_acc"]
                model = task.models.get((arm, task.h_bits, task.h_delay))
                if model is not None:
                    accs[src]["cnn"] = recall_metrics(task.cnn_bits(model, counts, shots), truth)["bit_acc"]
            res["noisy_bit_acc"] = accs
            out[arm] = res
    out["seconds"] = time.perf_counter() - t0
    out["n_patterns"] = n
    return out


# --- raw example traces -------------------------------------------------------------------------------------------


def log_example_traces(task: MemoryTask) -> None:
    """Seed 0, headline protocol, the all-zeros and all-ones patterns, both conditions, 10 shots each (full
    sequences on the surrogate: init, write, delay, readout)."""
    env = task.env(0)
    sim = SurrogateSimulator(task.device, task.options)
    rng = np.random.default_rng([task.base_seed, 3])
    prot = protocol_for(task.cfg, task.h_bits, task.h_delay, task.h_arm)
    pats = np.stack([np.zeros(task.h_bits, np.int64), np.ones(task.h_bits, np.int64)])
    for cname in ("resonant", "detuned"):
        cond = condition_for(task.cfg, cname)
        pre, suf = sequences(prot, task.encoder(0, task.h_bits, env), env, pats, offset_hz=cond.mw_offset_hz)
        for j, row in enumerate(suf.unbatch()):
            seq = full_sequence(pre, row)
            trace = sim.run(seq, env, shots=10, rng=rng)
            task.run.log_trace(trace, seq, condition=cname, pattern="".join(map(str, pats[j])))


# --- plots --------------------------------------------------------------------------------------------------------


def _band(ax: Any, xs: list[Any], vals: list[np.ndarray], color: str, ls: str, label: str, marker: str = "o") -> None:
    st = [t_ci(v) for v in vals]
    mu = [s[0] for s in st]
    ax.plot(xs, mu, ls, color=color, lw=2, marker=marker, ms=4, label=label)
    lo = [s[1] if math.isfinite(s[1]) else s[0] for s in st]
    hi = [s[2] if math.isfinite(s[2]) else s[0] for s in st]
    ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0)


def _sel(df: pd.DataFrame, **kw: Any) -> pd.DataFrame:
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def plot_inputs(task: MemoryTask) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    env = task.env(0)
    bits = 2 if 2 in task.bits else task.bits[0]
    prot = protocol_for(task.cfg, bits, task.h_delay, "none")
    enc = task.encoder(0, bits, env)
    from glowbox.learn.bit_memory import all_patterns

    pats = all_patterns(bits)
    block = enc.block(2.0 * pats - 1.0)
    ax = axes[0]
    t = np.arange(block.n_bins) * 10
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#a35fd0"]
    amp = np.asarray(block.mw_amplitude)
    for j in range(min(4, len(pats))):
        ax.step(t, amp[j] + 1.1 * j, where="post", color=colors[j], lw=1.2, label="".join(map(str, pats[j])))
    ax.set_xlabel("time in write block (ns)")
    ax.set_ylabel("MW amplitude (offset per pattern)")
    ax.set_title(f"write blocks of the {len(pats)} patterns (L = {bits}, seed 0)")
    ax.legend(fontsize=7, frameon=False, title="bits", title_fontsize=7)
    _style(ax)
    tro = np.arange(prot.readout_bins) * 10
    for ax, cname in zip(axes[1:], ("resonant", "detuned"), strict=True):
        key = (cname, 0, "none", bits, task.h_delay)
        if key not in task.books:
            continue
        tm = task.books[key].templates
        mean = tm.mean(axis=0)
        for j in range(min(4, tm.shape[0])):
            ax.plot(tro, (tm[j] - mean) * 1e3, color=colors[j], lw=1.4)
        ax.set_xlabel("time in readout (ns)")
        ax.set_ylabel("template - mean (1e-3 counts/shot/bin)")
        ax.set_title(f"readout response, {cname} (delay {task.h_delay} ns)")
        _style(ax)
    lim = max(abs(a) for ax in axes[1:] for a in ax.get_ylim())
    for ax in axes[1:]:
        ax.set_ylim(-lim, lim)
    fig.tight_layout()
    return fig


def plot_recall_vs_bits(task: MemoryTask) -> Figure:
    df = pd.DataFrame(task.rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric in zip(axes, ("bit_acc", "recalled_bits"), strict=True):
        for r in READOUTS:
            for cname, ls in (("resonant", "-"), ("detuned", "--")):
                d = _sel(df, readout=r, condition=cname, arm=task.h_arm, delay_ns=task.h_delay, shots=task.h_shots)
                if not len(d):
                    continue
                xs = sorted(d.bits.unique())
                _band(ax, xs, [d[d.bits == b][metric].to_numpy() for b in xs], COLORS[r], ls, f"{LABELS[r]}, {cname}")
        if metric == "bit_acc":
            ax.axhline(0.5, color=COLORS["chance"], ls=":", lw=1.2, label="chance")
            ax.set_ylabel("bit accuracy")
        else:
            ax.plot(task.bits, task.bits, ":", color="#52514e", lw=1, label="all bits (ideal)")
            ax.set_ylabel("recalled bits, sum 1 - H2(error)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(task.bits, [str(b) for b in task.bits])
        ax.set_xlabel("pattern length L (bits)")
        _style(ax)
    axes[0].set_title(f"recall vs pattern length (delay {task.h_delay} ns, {task.h_shots} shots)", fontsize=9)
    axes[1].legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    return fig


def plot_recall_vs_delay(task: MemoryTask) -> Figure:
    df = pd.DataFrame(task.rows)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for arm, ls, mk in (("none", "-", "o"), ("pi2", "-.", "s")):
        for r in READOUTS:
            d = _sel(df, readout=r, condition="resonant", arm=arm, bits=task.h_bits, shots=task.h_shots)
            if not len(d):
                continue
            xs = sorted(d.delay_ns.unique())
            name = "no read pulse" if arm == "none" else "pi/2 read pulse"
            _band(
                ax,
                [x / 1e3 for x in xs],
                [d[d.delay_ns == x].bit_acc.to_numpy() for x in xs],
                COLORS[r],
                ls,
                f"{LABELS[r]}, {name}",
                mk,
            )
    d = _sel(df, readout="oracle", condition="detuned", arm="none", bits=task.h_bits, shots=task.h_shots)
    if len(d):
        xs = sorted(d.delay_ns.unique())
        _band(
            ax,
            [x / 1e3 for x in xs],
            [d[d.delay_ns == x].bit_acc.to_numpy() for x in xs],
            COLORS["chance"],
            "--",
            "oracle, detuned",
        )
    t2s = np.mean([task.env(s).t2_star_s for s in task.envs]) * 1e6
    ax.axvline(t2s, color="#c3c2b7", lw=1, zorder=0)
    ax.text(t2s, ax.get_ylim()[0], " mean T2*", fontsize=7, color="#52514e", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("delay between write and read (us)")
    ax.set_ylabel("bit accuracy")
    ax.set_title(f"recall vs delay (L = {task.h_bits}, {task.h_shots} shots; 95 % t CI over seeds)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_quantum_test(task: MemoryTask, qt: QuantumTestResult) -> Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ticks, labels = [], []
    for x, r in enumerate(READOUTS):
        if f"{r}_bit_acc" not in qt.metrics():
            continue
        s = qt.summary(f"{r}_bit_acc")
        for j, (cname, alpha) in enumerate((("resonant", 1.0), ("detuned", 0.45))):
            v, lo, hi = s[cname]
            ax.bar(x + 0.36 * j, v, 0.32, color=COLORS[r], alpha=alpha, edgecolor="none")
            ax.errorbar(x + 0.36 * j, v, yerr=[[v - lo], [hi - v]], color="#0b0b0b", lw=1, capsize=3)
        ticks.append(x + 0.18)
        labels.append(LABELS[r])
    ax.axhline(0.5, color=COLORS["chance"], ls=":", lw=1.4)
    ax.set_xticks(ticks, labels, fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel(f"bit accuracy, L = {task.h_bits}, {task.h_shots} shots")
    ax.set_title("quantum test: resonant (solid) vs detuned +0.4 GHz (light); dotted = chance; 95 % t CI", fontsize=9)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_timing(task: MemoryTask) -> Figure:
    df = pd.DataFrame(task.timing_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    styles = {
        "oracle": (COLORS["oracle"], "-", "oracle (retrained)"),
        "ridge_retrained": (COLORS["ridge"], "-", "ridge, retrained"),
        "ridge_frozen": (COLORS["ridge"], "--", "ridge, frozen"),
        "cnn_frozen": (COLORS["cnn"], "--", "CNN, frozen"),
    }
    ax = axes[0]
    for r, (col, ls, lab) in styles.items():
        d = _sel(df, readout=r, condition="resonant")
        if not len(d):
            continue
        xs = sorted(d.shift_ns.unique())
        _band(ax, xs, [d[d.shift_ns == x].bit_acc.to_numpy() for x in xs], col, ls, lab)
    ax.axhline(0.5, color=COLORS["chance"], ls=":", lw=1.2)
    ax.set_xlabel("shift of every write-block MW duration (ns)")
    ax.set_ylabel("bit accuracy")
    ax.set_title(f"timing probe, resonant (L = {task.h_bits}, delay {task.h_delay} ns)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    ax = axes[1]
    if task.saliency:
        s = np.asarray(task.saliency["saliency"])
        c = np.asarray(task.saliency["contrast"])
        t = np.arange(s.size) * 10
        ax.plot(t, s / s.max(), color=COLORS["cnn"], lw=1.8, label="CNN saliency")
        ax.plot(t, c / c.max(), color=COLORS["oracle"], lw=1.8, label="pattern contrast (template sd / mean)")
        ax.set_title(f"where the CNN looks (seed 0), r = {task.saliency['corr']:.2f}", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel("time in readout (ns)")
    ax.set_ylabel("normalised")
    _style(ax)
    fig.tight_layout()
    return fig


# --- report -------------------------------------------------------------------------------------------------------


def _ci(v: tuple[float, float, float], fmt: str = "{:.3g}") -> str:
    if not math.isfinite(v[1]):
        return fmt.format(v[0])
    return f"{fmt.format(v[0])} [{fmt.format(v[1])}, {fmt.format(v[2])}]"


def paired_diff(df: pd.DataFrame, metric: str, a: dict[str, Any], b: dict[str, Any]) -> tuple[float, float, float]:
    """t CI over seeds of metric(a) - metric(b), paired by seed."""
    da = _sel(df, **a).set_index("seed")[metric]
    db = _sel(df, **b).set_index("seed")[metric]
    common = sorted(set(da.index) & set(db.index))
    return t_ci(da.loc[common].to_numpy() - db.loc[common].to_numpy())


def timing_verdicts(task: MemoryTask) -> dict[str, Any]:
    df = pd.DataFrame(task.rows)
    tdf = pd.DataFrame(task.timing_rows)
    out: dict[str, Any] = {}
    lo_d, hi_d = min(task.delays), max(task.delays)
    base = {"condition": "resonant", "bits": task.h_bits, "shots": task.h_shots, "readout": "oracle"}
    for arm in task.arms:
        out[f"delay_drop_{arm}"] = paired_diff(
            df, "bit_acc", {**base, "arm": arm, "delay_ns": lo_d}, {**base, "arm": arm, "delay_ns": hi_d}
        )
    if "pi2" in task.arms:
        d = out["delay_drop_pi2"]
        out["memory_uses_coherence"] = bool(math.isfinite(d[1]) and (d[1] > 0 or d[2] < 0))
    shifts = sorted(tdf.shift_ns.unique()) if len(tdf) else []
    big = [s for s in shifts if abs(s) == max(abs(x) for x in shifts)] if shifts else []
    for r in ("oracle", "ridge_retrained", "ridge_frozen", "cnn_frozen"):
        drops = []
        for s in big:
            if len(_sel(tdf, readout=r, condition="resonant", shift_ns=s)):
                drops.append(
                    paired_diff(
                        tdf,
                        "bit_acc",
                        {"readout": r, "condition": "resonant", "shift_ns": 0},
                        {"readout": r, "condition": "resonant", "shift_ns": s},
                    )
                )
        out[f"shift_drop_{r}"] = drops
    fro = out.get("shift_drop_ridge_frozen", [])
    out["readout_relies_on_pulse_timing"] = bool(any(math.isfinite(d[1]) and d[1] > 0 for d in fro))
    ora = out.get("shift_drop_oracle", [])
    out["information_needs_exact_timing"] = bool(any(math.isfinite(d[1]) and d[1] > 0 for d in ora))
    out["strategy_learns_timing"] = False
    return out


def report_body(
    task: MemoryTask, qt: QuantumTestResult, tv: dict[str, Any], ref: dict[str, Any], summary: dict[str, Any]
) -> str:
    cfg = task.cfg
    df = pd.DataFrame(task.rows)
    p, e = cfg["protocol"], cfg["encoder"]
    hb, hd, hs = task.h_bits, task.h_delay, task.h_shots
    n_seeds = len(qt.seeds)
    lines: list[str] = []
    a = lines.append
    a(
        "Task 2 of the brief: store a short bit pattern in the pulse response and recall it. A fixed random pulse "
        "encoder writes the bits into a block of MW segments, the NV ensemble holds them through a dark delay, and "
        "the bits are recalled from the time-resolved fluorescence of one readout. Three readouts of the same "
        f"signal (a linear reservoir readout, a CNN decoder and the Bayes-optimal oracle), {n_seeds} seeds, all on "
        "the fast surrogate, the headline re-checked on the QuTiP reference; and the quantum test."
    )
    a("")
    a("## Protocol")
    a("")
    env0 = task.env(0)
    a(
        f"- **Device** (one per seed): static field {tuple(cfg['device']['bias_field_t'])} T (known, eight lines), "
        "nuisance drawn per seed from the milestone-2 ranges (T2* 0.5-2 µs log-uniform, Rabi full scale 8-12 MHz, "
        "beta 0.07-0.13, collection 2e-3-4e-3, background 5e3-2e4 counts/s, E 0-0.3 MHz); seed 0 has "
        f"T2* = {env0.t2_star_s * 1e6:.2f} µs, Rabi full scale {env0.rabi_full_scale_hz / 1e6:.2f} MHz. T2* "
        f"quadrature: M = {task.options.n_detuning} quantile nodes."
    )
    a(
        f"- **Shot**: {p['init_ns']} ns init laser, {p['settle_ns']} ns dark, the write block "
        f"({e['n_segments']} slots of {e['slot_ns']} ns), a dark delay, optionally a read pulse (line 0, amplitude "
        f"{p['read_amplitude']}, {p['read_ns']} ns: ~pi/2 at the nominal full scale), then {p['readout_ns']} ns of "
        "readout laser recorded in 10 ns bins. The write block has the same length for every L."
    )
    a(
        f"- **Encoder** (`RandomPulseEncoder`, fixed per seed and L, never trained): u = 2 b - 1; per segment "
        f"amplitude in [{e['amp_lo']}, {e['amp_hi']}], phase in (-pi, pi), frequency within "
        f"±{e['freq_span_hz'] / 1e6:g} MHz of one of the eight lines (fixed random choice), MW duration "
        f"{e['min_ns']}-{e['max_ns']} ns on the grid, "
        f"each a sigmoid / tanh of a random affine map of u (gain {e['gain']}). Every segment depends on every bit."
    )
    st = pd.DataFrame(task.encoding)
    if len(st):
        a(
            f"  Over the seeds' headline encoders (L = {hb}, all patterns): median segment area "
            f"{st.median_area_pi.median():.2f} pi, {100 * st.frac_area_quarter_to_4pi.mean():.0f} % of segments "
            f"between pi/4 and 4 pi (Rabi-scale pulses), median MW duration {st.median_duration_ns.median():.0f} ns."
        )
    a(
        "- **Codebook**: every one of the 2^L patterns (L <= 12) is simulated (noiseless mean counts per shot per "
        "readout bin, its template). Training and test draws pick patterns uniformly and add Poisson noise for the "
        "given number of shots (the pattern written and read that many times); test noise is fresh. So every readout "
        "is calibrated on all patterns, and recall is limited by shot noise and by how distinct the templates are."
    )
    a("")
    a("## Readouts")
    a("")
    a(
        "- **Oracle** (`bitwise_map`): exact templates, Poisson likelihood, uniform prior over patterns, posterior "
        "marginal per bit, threshold 1/2. Bayes-optimal bit decisions: an upper bound for any readout of this "
        "protocol and encoder."
    )
    a(
        f"- **Reservoir readout** (`RidgeReadout`): ridge regression from counts per shot (standardised) to u, one "
        f"model per shot level, {cfg['ridge']['n_train']} training draws, penalty by generalised cross-validation; "
        "decision = sign. The reservoir is encoder + spins + this linear map; nothing else is trained."
    )
    c = cfg["cnn"]
    a(
        "- **CNN decoder** (`ConvDecoder`, one logit per bit, BCE): input = counts per shot relative to the "
        "codebook's mean template (`LevelDecoder`: the level is kept, since each seed is one device and the pattern "
        "changes the level), "
        f"conditioned on log10(shots); trained {c['steps']} steps of {c['batch']} on fresh Poisson noise, shots "
        f"log-uniform in [{c['shots_min']}, {c['shots_max']}], best validation loss kept. Trained on the no-read-pulse "
        f"arm at (L, delay) = {', '.join(f'({b}, {d} ns)' for b, d in task.cnn_pairs)}."
    )
    a(
        "- Metrics: bit accuracy, pattern accuracy (all L bits right), and recalled bits = sum over bits of "
        "1 - H2(e_i) (the capacity of L binary symmetric channels with the measured per-bit error rates e_i; 0 at "
        "chance, L when perfect)."
    )
    a("")
    a("## Recall vs pattern length")
    a("")
    a(
        f"Delay {hd} ns, no read pulse, mean over {n_seeds} seeds with a 95 % Student-t CI over seeds; 'sep' is the "
        "median distance of a template to its nearest neighbour in Poisson standard deviations (seed mean)."
    )
    a("")
    head = ["L", "shots", "sep"] + [f"{r} bit acc" for r in READOUTS] + [f"{r} recalled bits" for r in READOUTS]
    a("| " + " | ".join(head) + " |")
    a("|" + " --- |" * len(head))
    for b in task.bits:
        for s in task.shots:
            d = _sel(df, condition="resonant", arm="none", delay_ns=hd, bits=b, shots=s)
            if not len(d):
                continue
            cells = [str(b), str(s), f"{d.separation.mean():.2g}"]
            for metric in ("bit_acc", "recalled_bits"):
                for r in READOUTS:
                    v = d[d.readout == r][metric].to_numpy()
                    cells.append(_ci(t_ci(v)) if v.size else "")
            a("| " + " | ".join(cells) + " |")
    a("")
    a(recall_summary(task, df))
    a("")
    a("## Recall vs delay")
    a("")
    a(
        f"L = {hb}, {hs} shots, bit accuracy (95 % t CI over seeds); 'none' reads populations only, 'pi2' applies "
        "the read pulse first."
    )
    a("")
    head = ["delay (ns)"] + [f"{arm} {r}" for arm in task.arms for r in READOUTS]
    a("| " + " | ".join(head) + " |")
    a("|" + " --- |" * len(head))
    for dl in task.delays:
        cells = [str(dl)]
        for arm in task.arms:
            for r in READOUTS:
                acc = _sel(df, condition="resonant", arm=arm, bits=hb, shots=hs, delay_ns=dl, readout=r).bit_acc
                cells.append(_ci(t_ci(acc.to_numpy())) if len(acc) else "")
        a("| " + " | ".join(cells) + " |")
    a("")
    lo_d, hi_d = min(task.delays), max(task.delays)
    for arm in task.arms:
        dd = tv[f"delay_drop_{arm}"]
        a(f"- Oracle bit accuracy at {lo_d} ns minus at {hi_d} ns, arm {arm}: {_ci(dd)} (paired over seeds).")
    if "memory_uses_coherence" in tv:
        verdict = "**True**" if tv["memory_uses_coherence"] else "**False**"
        a(
            f"- `memory_uses_coherence` = {verdict} (criterion: with the read pulse, recall changes between the "
            "shortest and longest delay beyond the CI; populations alone do not decay on this scale, T1 = 5 ms, so a "
            "change is the stored coherence dephasing on the T2* scale)."
        )
    a("")
    a("## Quantum test")
    a("")
    a(
        f"Every readout retrained and re-evaluated with every MW frequency (write block and read pulse) shifted by "
        f"+{cfg['detuned_offset_hz'] / 1e6:.0f} MHz (nearest line {task.no_line_margin_hz / 1e6:.0f} MHz away from "
        f"any MW frequency; `check_no_line_driven`), same devices, encoders and noise seeds (paired). Headline: L = "
        f"{hb}, delay {hd} ns, {hs} shots, no read pulse (pi2_* rows: with the read pulse). 95 % Student-t CIs over "
        f"the {n_seeds} seeds; gap > 0 means better on resonance."
    )
    a("")
    a(qt.to_markdown(fmt="{:.3g}"))
    a("")
    a(qt_outcome(qt))
    a("")
    a("### Does the strategy use microsecond-scale timing?")
    a("")
    a(
        "Nothing that sets timing is learned in Task 2: the encoder is a fixed random map and the readouts only weight "
        "readout bins, so `strategy_learns_timing` = **False**. What the log records is whether the fixed "
        "reservoir and the readouts *depend* on spin timing:"
    )
    a("")
    a(
        f"- **Pulse areas** (timing probe, every seed, resonant; every write-block MW duration shifted, clipped to "
        f"[10, {e['slot_ns']}] ns; L = {hb}, delay {hd} ns, {hs} shots). Bit accuracy at shift 0 minus at the largest "
        "shifts (paired over seeds):"
    )
    tdf = pd.DataFrame(task.timing_rows)
    shifts = sorted(tdf.shift_ns.unique()) if len(tdf) else []
    big = [s for s in shifts if abs(s) == max(abs(x) for x in shifts)] if shifts else []
    for r, lab in (
        ("oracle", "oracle, retrained"),
        ("ridge_retrained", "ridge, retrained"),
        ("ridge_frozen", "ridge, frozen"),
        ("cnn_frozen", "CNN, frozen"),
    ):
        drops = tv.get(f"shift_drop_{r}", [])
        if drops:
            a(f"  - {lab}: " + "; ".join(f"{s:+d} ns: {_ci(d)}" for s, d in zip(big, drops, strict=True)))
    a("")
    a(
        "  A frozen readout that degrades means it relies on the exact pulse areas the encoder produced at "
        "calibration (Rabi-scale structure: the Rabi period is ~120 ns at full amplitude and longer below, so a "
        "30 ns change is up to a quarter of a period); a "
        "retrained readout that does not degrade means the shifted encoding is just another random encoding, equally "
        "informative. Verdicts: `readout_relies_on_pulse_timing` = "
        f"**{tv['readout_relies_on_pulse_timing']}**, `information_needs_exact_timing` = "
        f"**{tv['information_needs_exact_timing']}**."
    )
    if "memory_uses_coherence" in tv:
        a(f"- **Storage time scale** (delay table above): `memory_uses_coherence` = **{tv['memory_uses_coherence']}**.")
    if task.saliency:
        a(
            f"- **Readout weighting** (seed 0 CNN): saliency per readout bin vs the pattern contrast per bin, r = "
            f"{task.saliency['corr']:.2f}; {100 * task.saliency['first_300ns_fraction']:.0f} % of the saliency in the "
            "first 300 ns of the readout."
        )
    a("")
    a("## Reference check (QuTiP)")
    a("")
    n = ref.get("n_patterns", 0)
    a(
        f"The first {n} patterns of the seed-0 headline codebook (L = {hb}, delay {hd} ns) re-simulated as full "
        "sequences (absolute-time phases, `full_sequence`) on the QuTiP reference with the same quadrature nodes, and "
        f"fed to the surrogate-trained readouts ({ref.get('seconds', 0):.0f} s on the CPU pool):"
    )
    a("")
    a(
        "| read arm | max template diff / peak | / max pattern signal | ridge decisions agree (noiseless) | bit acc "
        "ref / surrogate (64 draws each), oracle | ridge | CNN |"
    )
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for arm in task.arms:
        rr: dict[str, Any] | None = ref.get(arm)
        if not rr:
            continue
        na = rr["noisy_bit_acc"]

        def pair(k: str, na: dict[str, dict[str, float]] = na) -> str:
            return f"{na['ref'][k]:.3f} / {na['sur'][k]:.3f}" if k in na["ref"] else ""

        a(
            f"| {arm} | {rr['max_rel_template_diff']:.1e} | {rr['max_diff_over_signal']:.1e} | "
            f"{rr.get('ridge_noiseless_agree', float('nan')):.3f} | {pair('oracle')} | {pair('ridge')} | "
            f"{pair('cnn')} |"
        )
    a("")
    a("## Notes and open issues")
    a("")
    a(
        "- One readout per shot sees only the total ms = 0 population of all four axes (plus the singlet), a nearly "
        "one-dimensional signal: the readout's time shape adds little. That caps how much of a pattern one readout "
        "can return, whatever the readout: the oracle column is the cap for this encoder. A random encoding also "
        "spreads the 2^L patterns unevenly along that one dimension (many near-collisions); a designed encoding "
        "(e.g. binary-weighted pulse areas, or an encoder trained through the differentiable surrogate, the next "
        "phase-2 item) should do better."
    )
    a(
        "- The codebook contains every pattern, so the CNN is calibrated on all of them (a classifier under shot "
        "noise), not asked to generalise to unseen patterns; the ridge readout is linear, so it cannot memorise."
    )
    a(
        "- Model limits (docs/physics.md section 10): no hyperfine, no charge dynamics, no excited-state mixing; "
        "the T2* node set revives after ~2 M T2* (>= 32 µs here, longer than every sequence)."
    )
    a("- No claim of quantum advantage: the gap shows only that recall needs the driven spin resonance.")
    return "\n".join(lines)


def recall_summary(task: MemoryTask, df: pd.DataFrame) -> str:
    """One generated paragraph: recalled bits per L for each readout at the headline, and the readouts' gaps."""
    hs, hd = task.h_shots, task.h_delay
    parts = []
    for r in READOUTS:
        vals = []
        for b in task.bits:
            v = _sel(df, condition="resonant", arm="none", delay_ns=hd, bits=b, shots=hs, readout=r).recalled_bits
            if len(v):
                vals.append(f"{v.mean():.2f} of {b}")
        if vals:
            parts.append(f"{LABELS[r]} {', '.join(vals)}")
    best = _sel(df, condition="resonant", arm="none", delay_ns=hd, shots=hs, readout="oracle")
    cap = best.groupby("bits").recalled_bits.mean()
    if not len(cap):
        return ""
    ls = [int(x) for x in cap.index]
    peak = int(np.argmax(cap.to_numpy()))
    tail = (
        "and falls for longer patterns: they crowd the same readout signal"
        if peak < len(ls) - 1
        else "and is still growing at the longest pattern tested"
    )
    return (
        f"At {hs} shots the recalled bits are: " + "; ".join(parts) + f". The oracle's recall peaks at L = "
        f"{ls[peak]} ({cap.to_numpy()[peak]:.2f} bits) {tail}."
    )


def qt_outcome(qt: QuantumTestResult) -> str:
    out = []
    for r in READOUTS:
        m = f"{r}_bit_acc"
        if m not in qt.metrics():
            continue
        s = qt.summary(m)
        sig = "significant" if qt.gap_significant(m) else "not significant"
        out.append(f"{LABELS[r]} gap {_ci(s['gap'])} ({sig}; detuned {_ci(s['detuned'])})")
    return (
        "Outcome (bit accuracy): " + "; ".join(out) + ". Detuned, no MW frequency drives a line, the readout no "
        "longer depends on the pattern and every readout falls to chance (0.5)."
    )


# --- main ---------------------------------------------------------------------------------------------------------


def main(
    cfg: dict[str, Any], output_root: Path | None = None, report_dir: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    """Run the Task 2 milestone; returns (report path, summary numbers)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw: dict[str, Any] = {} if output_root is None else {"output_root": output_root}
    t_start = time.perf_counter()
    with RunLog(str(cfg["name"]), config=cfg, seed=int(cfg["seed"]), **kw) as run:
        task = MemoryTask(cfg, run, device)
        conds = (RESONANT, detuned_condition(float(cfg["detuned_offset_hz"])))
        hib = {
            f"{p}{r}_{m}": True
            for p in ("", "pi2_")
            for r in READOUTS
            for m in ("bit_acc", "recalled_bits", "pattern_acc")
        }
        qt = run_quantum_test(task, [int(s) for s in cfg["seeds"]], conds, higher_is_better=hib)
        print("[m3-bits] reference check", flush=True)
        ref = run_reference_check(task)
        log_example_traces(task)
        tv = timing_verdicts(task)

        pd.DataFrame(task.rows).to_parquet(run.dir / "results.parquet")
        pd.DataFrame(task.timing_rows).to_parquet(run.dir / "timing.parquet")
        pd.DataFrame(task.encoding).to_parquet(run.dir / "encoding.parquet")
        data = run.dir / "data"
        data.mkdir(exist_ok=True)
        for (cname, _s, arm, bits, delay), cb in task.books.items():
            np.savez_compressed(
                data / f"codebook-{cname}-{arm}-L{bits}-d{delay}.npz", patterns=cb.patterns, templates=cb.templates
            )
        (run.dir / "train_history.json").write_text(json.dumps(task.histories), encoding="utf-8")
        (run.dir / "reference_check.json").write_text(json.dumps(ref), encoding="utf-8")
        (run.dir / "timing_verdicts.json").write_text(json.dumps(tv, default=str), encoding="utf-8")
        (run.dir / "saliency.json").write_text(json.dumps(task.saliency), encoding="utf-8")
        (run.dir / "quantum_test.json").write_text(
            json.dumps({"seeds": qt.seeds, "values": qt.values}), encoding="utf-8"
        )

        figs = {
            "inputs": plot_inputs(task),
            "recall-vs-bits": plot_recall_vs_bits(task),
            "recall-vs-delay": plot_recall_vs_delay(task),
            "quantum-test": plot_quantum_test(task, qt),
            "timing": plot_timing(task),
        }
        paths = [run.save_figure(f, name, dpi=90) for name, f in figs.items()]
        for f in figs.values():
            plt.close(f)

        summary: dict[str, Any] = {
            "device": f"{device.type} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}".strip()
        }
        for m in qt.metrics():
            s = qt.summary(m)
            summary[f"{m}_resonant"] = s["resonant"][0]
            summary[f"{m}_detuned"] = s["detuned"][0]
            summary[f"{m}_gap"] = s["gap"][0]
            summary[f"{m}_gap_lo"] = s["gap"][1]
            summary[f"{m}_gap_hi"] = s["gap"][2]
        for arm in task.arms:
            r = ref.get(arm, {})
            summary[f"ref_{arm}_max_rel_template_diff"] = r.get("max_rel_template_diff", float("nan"))
        summary["memory_uses_coherence"] = float(tv.get("memory_uses_coherence", False))
        summary["readout_relies_on_pulse_timing"] = float(tv["readout_relies_on_pulse_timing"])
        summary["information_needs_exact_timing"] = float(tv["information_needs_exact_timing"])
        summary["strategy_learns_timing"] = 0.0
        summary["no_line_margin_mhz"] = task.no_line_margin_hz / 1e6
        summary["codebook_seconds"] = task.seconds["codebooks"]
        summary["cnn_seconds"] = task.seconds["cnn"]
        summary["wall_seconds"] = time.perf_counter() - t_start
        run.log_metrics(step=0, **{k: v for k, v in summary.items() if isinstance(v, int | float)})
        body = report_body(task, qt, tv, ref, summary)
    rdir = report_dir if report_dir is not None else run.dir / "report"
    title = "Milestone 3, Task 2: store a bit pattern in the pulse response and recall it"
    path = write_report(run.dir, title, body, figures=paths, report_dir=rdir, name=REPORT_NAME)
    return path, summary


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="small grid and short training (smoke run)")
    p.add_argument("--report-dir", type=Path, default=None, help="where to write the report (e.g. docs/results)")
    p.add_argument("--output-root", type=Path, default=None, help="run directories (default: <repo>/outputs)")
    a = p.parse_args()
    path, summary = main(load_config(a.config, a.quick), a.output_root, a.report_dir)
    for k, v in summary.items():
        print(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}")
    print(f"report: {path}")


if __name__ == "__main__":
    cli()
