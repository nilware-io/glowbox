"""Milestone 4, Task 4: compress-and-reconstruct through the NV ensemble, with a learned pulse encoder.

    uv run python -m glowbox.experiments.m4_autoencoder [--config PATH] [--quick] [--report-dir DIR]

A 4x4 image patch (a soft edge: two latent parameters, glowbox.learn.autoencoder.edge_patches) goes into a learned
encoder (glowbox.learn.encoder.PulseEncoder, an MLP) that writes K MW blocks of four segments (amplitude, phase,
frequency near a device line, duration on the 10 ns grid); after each block a gated readout returns one photon count.
A decoder (MLP) reconstructs the patch from the K counts at a given number of shots. Phase 1 trains encoder and
decoder together through the differentiable surrogate; phase 2 freezes the encoder and trains a fresh decoder on a
bank of its noiseless readouts with Poisson noise (the recipe every baseline gets).

Baselines and references: the same encoder parametrisation fixed at random (the reservoir of milestone 3, ridge
replaced by the same MLP decoder), linear PCA with K components and a noiseless K-number MLP autoencoder (what K
perfect numbers allow). The quantum test runs the learned and the random encoder per seed with every MW frequency
shifted by ``detuned_offset_hz``, trained the same way. Timing: pulse areas of the learned blocks, the harness's
timing probe (every MW duration shifted, decoder frozen and retrained) and a phase-scramble probe (coherent phase
relations between segments). Check: the learned sequences on the QuTiP reference and at 32 quadrature nodes.

Outputs: ``outputs/<run-id>/`` (run.yaml, raw example traces, metrics, results / timing / areas parquet tables,
seed-0 models and banks, plots) and a report (``outputs/<run-id>/report/`` unless ``--report-dir``).
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
    mw_timing_structure,
    run_quantum_test,
    t_ci,
    timing_sensitivity,
)
from glowbox.analysis.runlog import RunLog, write_report
from glowbox.experiments.m1_odmr_rabi import load_config
from glowbox.experiments.m2_field_estimation import _style
from glowbox.experiments.m3_bit_memory import _band, _ci
from glowbox.learn.autoencoder import (
    DiamondChannel,
    SystemSpec,
    VectorDecoder,
    edge_patches,
    nmse,
    pca_nmse,
    reconstruct,
    train_bank_decoder,
    train_end_to_end,
    train_ideal_bottleneck,
)
from glowbox.learn.bit_memory import device_lines_hz
from glowbox.learn.encoder import BlockParams, Layout, PulseEncoder, preactivations_for, pulse_areas_pi, suffix_to_numpy
from glowbox.learn.field_data import FieldPrior
from glowbox.learn.reservoir import full_sequence
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate import SurrogateSimulator
from glowbox.sim.surrogate.model import ModelOptions

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "m4_autoencoder.yaml"
REPORT_NAME = "m4-autoencoder"
ENCODERS = ("learned", "random")
COLORS = {"learned": "#2a78d6", "random": "#eb6834", "pca": "#8a8985", "ideal": "#1baf7a"}
LABELS = {
    "learned": "learned encoder",
    "random": "fixed random encoder (reservoir)",
    "pca": "PCA, K components (noiseless)",
    "ideal": "ideal K-number autoencoder (noiseless)",
}


def _tgen(device: torch.device, *key: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(np.random.default_rng(list(key)).integers(2**62)))
    return g


def condition_for(cfg: dict[str, Any], name: str) -> Condition:
    return RESONANT if name == "resonant" else detuned_condition(float(cfg["detuned_offset_hz"]))


@dataclasses.dataclass
class SeedData:
    xtr: np.ndarray
    xva: np.ndarray
    xte: np.ndarray
    ttr: torch.Tensor
    tva: torch.Tensor
    tte: torch.Tensor


@dataclasses.dataclass
class Trained:
    """One trained (or fixed) encoder with its phase-2 decoder and noiseless banks."""

    encoder: PulseEncoder
    decoder: VectorDecoder
    channel: DiamondChannel
    mu_te: torch.Tensor
    history: dict[str, Any]


class AutoencoderTask:
    """The quantum-test task function plus everything recorded along the way."""

    def __init__(self, cfg: dict[str, Any], run: RunLog, device: torch.device) -> None:
        self.cfg, self.run, self.device = cfg, run, device
        self.base_seed = int(cfg["seed"])
        self.sys = SystemSpec(
            init_ns=int(cfg["system"]["init_ns"]),
            settle_ns=int(cfg["system"]["settle_ns"]),
            options=ModelOptions(n_detuning=int(cfg["device"]["n_detuning"])),
        )
        self.blocks = [int(k) for k in cfg["grid"]["blocks"]]
        self.levels = [int(s) for s in cfg["grid"]["shots"]]
        self.h_blocks, self.h_shots = int(cfg["headline"]["blocks"]), int(cfg["headline"]["shots"])
        if self.h_blocks not in self.blocks:
            self.blocks = sorted({*self.blocks, self.h_blocks})
        if self.h_shots not in self.levels:
            self.levels = sorted({*self.levels, self.h_shots})
        self.shots_range = (float(cfg["train"]["shots"][0]), float(cfg["train"]["shots"][1]))
        self.rows: list[dict[str, Any]] = []
        self.timing_rows: list[dict[str, Any]] = []
        self.area_rows: list[dict[str, Any]] = []
        self.ref_rows: list[dict[str, Any]] = []
        self.histories: dict[str, Any] = {}
        self.seed0: dict[tuple[str, str], Trained] = {}  # (condition, encoder), headline K
        self.data_cache: dict[int, SeedData] = {}
        self.no_line_margin_hz = math.inf
        self.seconds: dict[str, float] = {"phase1": 0.0, "phase2": 0.0, "timing": 0.0}
        self.data_dir = run.dir / "data"
        self.data_dir.mkdir(exist_ok=True)

    # --- configuration ----------------------------------------------------------------------------------------
    def env(self, seed: int) -> EnvParams:
        pc: dict[str, Any] = {k: tuple(v) for k, v in self.cfg["device"]["prior"].items()}
        env = FieldPrior(**pc).sample_env(np.random.default_rng([self.base_seed, seed, 1]))
        b = tuple(float(x) for x in self.cfg["device"]["bias_field_t"])
        return dataclasses.replace(env, b_field_t=b)  # type: ignore[arg-type]

    def layout(self, k: int) -> Layout:
        lc = dict(self.cfg["layout"])
        return Layout(n_blocks=k, **lc)

    def base_hz(self, layout: Layout, env: EnvParams) -> np.ndarray:
        lines = np.sort(device_lines_hz(env).ravel())
        s = layout.n_segments
        return np.array([[lines[(k * s + j) % lines.size] for j in range(s)] for k in range(layout.n_blocks)])

    def data(self, seed: int) -> SeedData:
        if seed not in self.data_cache:
            dc = self.cfg["data"]
            rng = np.random.default_rng([self.base_seed, seed, 2])
            size, sharp, omax = int(dc["size"]), float(dc["sharpness"]), float(dc["offset_max"])
            xtr, xva, xte = (
                edge_patches(int(dc[n]), rng, size, sharp, omax)[0] for n in ("n_train", "n_val", "n_test")
            )
            dev = self.device
            self.data_cache[seed] = SeedData(
                xtr,
                xva,
                xte,
                torch.as_tensor(xtr, device=dev),
                torch.as_tensor(xva, device=dev),
                torch.as_tensor(xte, device=dev),
            )
        return self.data_cache[seed]

    def encoder(self, layout: Layout, env: EnvParams) -> PulseEncoder:
        ec = self.cfg["encoder"]
        init = preactivations_for(layout, float(ec["init_amplitude"]), float(ec["init_duration_ns"]))
        n_in = int(self.cfg["data"]["size"]) ** 2
        hidden = tuple(int(h) for h in ec["hidden"])
        return PulseEncoder(layout, self.base_hz(layout, env), n_in, hidden, init, float(ec["out_scale"])).to(
            self.device
        )

    def decoder(self, layout: Layout, ref: torch.Tensor) -> VectorDecoder:
        n_out = int(self.cfg["data"]["size"]) ** 2
        hidden = tuple(int(h) for h in self.cfg["decoder"]["hidden"])
        return VectorDecoder(layout.n_features, n_out, ref.float(), hidden).to(self.device)

    def var(self, seed: int) -> float:
        return float(np.mean(np.var(self.data(seed).xtr, axis=0)))

    # --- pieces -----------------------------------------------------------------------------------------------
    def bank_decoder(self, ch: DiamondChannel, enc: PulseEncoder, sd: SeedData, gen: torch.Generator) -> Trained:
        t0 = time.perf_counter()
        mu_tr, mu_va, mu_te = (ch.counts_for(enc, t).float() for t in (sd.ttr, sd.tva, sd.tte))
        dec = self.decoder(ch.layout, ch.reference())
        bc = self.cfg["bank_decoder"]
        h = train_bank_decoder(
            dec,
            mu_tr,
            sd.ttr,
            steps=int(bc["steps"]),
            batch=int(bc["batch"]),
            lr=float(bc["lr"]),
            shots_range=self.shots_range,
            gen=gen,
            val_mu=mu_va,
            val_u=sd.tva,
        )
        self.seconds["phase2"] += time.perf_counter() - t0
        return Trained(enc, dec, ch, mu_te, {"bank_step": h.step, "bank_val": h.val, "bank_seconds": h.seconds})

    def evaluate(self, tr: Trained, sd: SeedData, var: float, seed: int, shots: int) -> float:
        g = _tgen(self.device, self.base_seed, seed, 77, shots)
        rec = reconstruct(tr.decoder, tr.mu_te, shots, g)
        return nmse(rec.cpu().numpy(), sd.xte, var)

    def nmse_of_counts(self, tr: Trained, mu: torch.Tensor, sd: SeedData, var: float, seed: int, shots: int) -> float:
        g = _tgen(self.device, self.base_seed, seed, 77, shots)
        rec = reconstruct(tr.decoder, mu.float(), shots, g)
        return nmse(rec.cpu().numpy(), sd.xte, var)

    def check_detuned(self, layout: Layout, enc: PulseEncoder, env: EnvParams, offset: float, sd: SeedData) -> None:
        with torch.no_grad():
            f = enc(sd.tte[:256]).frequency_hz.cpu().numpy().ravel() + offset
        ok, margin = check_no_line_driven(f, np.asarray(env.b_field_t), env)
        self.no_line_margin_hz = min(self.no_line_margin_hz, margin)
        if not ok:
            raise RuntimeError(f"detuned condition drives a line (margin {margin / 1e6:.1f} MHz)")

    def record_areas(
        self, cond: str, seed: int, k: int, name: str, enc: PulseEncoder, env: EnvParams, sd: SeedData
    ) -> None:
        with torch.no_grad():
            p = enc(sd.tte[:512])
        a = pulse_areas_pi(p, env, self.layout(k).grid_ns)
        d = p.quantized_ns(10).cpu().numpy()
        amp = p.amplitude.cpu().numpy()
        self.area_rows.append(
            {
                "seed": seed,
                "condition": cond,
                "blocks": k,
                "encoder": name,
                "area_median_pi": float(np.median(a)),
                "area_q10_pi": float(np.quantile(a, 0.1)),
                "area_q90_pi": float(np.quantile(a, 0.9)),
                "frac_rabi_scale": float(np.mean((a >= 0.25) & (a <= 4.0))),
                "frac_off": float(np.mean(d == 0)),
                "duration_median_ns": float(np.median(d)),
                "amplitude_median": float(np.median(amp)),
                "areas": a.ravel()[:4096].tolist(),
            }
        )

    # --- the task ---------------------------------------------------------------------------------------------
    def __call__(self, cond: Condition, seed: int) -> dict[str, float]:
        t_seed = time.perf_counter()
        env = self.env(seed)
        sd = self.data(seed)
        var = self.var(seed)
        tc = self.cfg["train"]
        out: dict[str, float] = {}
        blocks = self.blocks if not cond.detuned else [self.h_blocks]
        if not cond.detuned:
            self.references(seed, sd, var)
        for k in blocks:
            ly = self.layout(k)
            ch = DiamondChannel(env, ly, self.sys, self.device, offset_hz=cond.mw_offset_hz)
            res: dict[str, Trained] = {}
            for ei, name in enumerate(ENCODERS):
                torch.manual_seed(int(np.random.default_rng([self.base_seed, seed, k, ei]).integers(2**31)))
                enc = self.encoder(ly, env)
                gen = _tgen(self.device, self.base_seed, seed, k, ei, 1)
                hist: dict[str, Any] = {}
                if name == "random":
                    enc.randomize_(
                        float(self.cfg["encoder"]["random_gain"]), _tgen(self.device, self.base_seed, seed, k, 9)
                    )
                else:
                    dec1 = self.decoder(ly, ch.reference())
                    t0 = time.perf_counter()

                    def sample(b: int, x: torch.Tensor = sd.ttr, g: torch.Generator = gen) -> torch.Tensor:
                        return x[torch.randint(0, x.shape[0], (b,), generator=g, device=self.device)]

                    h1 = train_end_to_end(
                        enc,
                        dec1,
                        ch,
                        sample,
                        steps=int(tc["steps"]),
                        batch=int(tc["batch"]),
                        lr_encoder=float(tc["lr_encoder"]),
                        lr_decoder=float(tc["lr_decoder"]),
                        shots_range=self.shots_range,
                        gen=gen,
                        val_u=sd.tva,
                        val_every=int(tc["val_every"]),
                    )
                    self.seconds["phase1"] += time.perf_counter() - t0
                    hist = {"e2e_step": h1.step, "e2e_val_nmse": [v / var for v in h1.val], "e2e_seconds": h1.seconds}
                for p in enc.parameters():
                    p.requires_grad_(False)
                if cond.detuned:
                    self.check_detuned(ly, enc, env, cond.mw_offset_hz, sd)
                tr = self.bank_decoder(ch, enc, sd, gen)
                tr.history.update(hist)
                self.histories[f"s{seed}_{cond.name}_K{k}_{name}"] = tr.history
                res[name] = tr
                for level in self.levels:
                    v = self.evaluate(tr, sd, var, seed, level)
                    self.rows.append(
                        {"seed": seed, "condition": cond.name, "encoder": name, "blocks": k, "shots": level, "nmse": v}
                    )
                    if k == self.h_blocks and level == self.h_shots:
                        out[f"{name}_nmse"] = v
                        out[f"{name}_skill"] = 1.0 - v
                self.record_areas(cond.name, seed, k, name, enc, env, sd)
                if seed == 0 and k == self.h_blocks:
                    self.seed0[(cond.name, name)] = tr
                    torch.save(
                        {"encoder": enc.state_dict(), "decoder": tr.decoder.state_dict()},
                        self.data_dir / f"model_s0_{cond.name}_{name}.pt",
                    )
                    np.save(self.data_dir / f"mu_test_s0_{cond.name}_{name}.npy", tr.mu_te.cpu().numpy())
            if k == self.h_blocks and not cond.detuned:
                t0 = time.perf_counter()
                for name, tr in res.items():
                    self.timing_probe(name, tr, sd, var, seed)
                self.seconds["timing"] += time.perf_counter() - t0
            del res
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        print(
            f"[m4-ae] seed {seed} {cond.name}: "
            + ", ".join(f"{k}={v:.3g}" for k, v in out.items() if k.endswith("nmse"))
            + f" ({time.perf_counter() - t_seed:.0f} s)",
            flush=True,
        )
        return out

    def references(self, seed: int, sd: SeedData, var: float) -> None:
        ic = self.cfg["ideal"]
        for k in self.blocks:
            g = _tgen(self.device, self.base_seed, seed, k, 55)
            torch.manual_seed(int(np.random.default_rng([self.base_seed, seed, k, 56]).integers(2**31)))
            model = train_ideal_bottleneck(
                sd.ttr, sd.tva, k, steps=int(ic["steps"]), batch=int(ic["batch"]), lr=float(ic["lr"]), gen=g
            )
            with torch.no_grad():
                ideal = nmse(model(sd.tte).cpu().numpy(), sd.xte, var)
            self.ref_rows.append({"seed": seed, "blocks": k, "pca": pca_nmse(sd.xtr, sd.xte, k), "ideal": ideal})

    def timing_probe(self, name: str, tr: Trained, sd: SeedData, var: float, seed: int) -> None:
        """Every MW duration shifted (frozen decoder; retrained decoder at a few shifts) and phases scrambled."""
        tcfg = self.cfg["timing"]
        shots = self.h_shots

        def frozen(shift: int) -> float:
            mu = tr.channel.counts_for(tr.encoder, sd.tte, duration_shift_ns=float(shift))
            return self.nmse_of_counts(tr, mu, sd, var, seed, shots)

        ts = timing_sensitivity(frozen, [int(s) for s in tcfg["duration_shifts_ns"]])
        for s, v in zip(ts.shifts_ns, ts.metric, strict=True):
            self.timing_rows.append(
                {"seed": seed, "encoder": name, "probe": "frozen", "shift_ns": int(s), "nmse": float(v)}
            )
        for s in tcfg["retrain_shifts_ns"]:
            shifted = _Shifted(tr.encoder, float(s))
            g = _tgen(self.device, self.base_seed, seed, 300 + int(s))
            t2 = self.bank_decoder(tr.channel, shifted, sd, g)  # type: ignore[arg-type]
            v = self.evaluate(t2, sd, var, seed, shots)
            self.timing_rows.append(
                {"seed": seed, "encoder": name, "probe": "retrained", "shift_ns": int(s), "nmse": v}
            )
        # phase scramble: every segment's phase replaced by an independent uniform random phase
        g = _tgen(self.device, self.base_seed, seed, 400)
        mus = []
        with torch.no_grad():
            for i in range(0, sd.tte.shape[0], tr.channel.chunk):
                p = tr.encoder(sd.tte[i : i + tr.channel.chunk])
                ph = (torch.rand(p.phase_rad.shape, generator=g, device=self.device) * 2 - 1) * math.pi
                mus.append(tr.channel.counts(BlockParams(p.amplitude, ph, p.frequency_hz, p.duration_ns)).double())
        v = self.nmse_of_counts(tr, torch.cat(mus), sd, var, seed, shots)
        self.timing_rows.append({"seed": seed, "encoder": name, "probe": "phase_scramble", "shift_ns": 0, "nmse": v})


class _Shifted(torch.nn.Module):
    """An encoder whose every duration is shifted (for the retrained timing probe: the bank sees shifted pulses)."""

    def __init__(self, enc: PulseEncoder, shift_ns: float) -> None:
        super().__init__()
        self.enc, self.shift = enc, shift_ns
        self.layout = enc.layout

    def forward(self, u: torch.Tensor) -> BlockParams:
        p = self.enc(u)
        d = torch.clamp(p.duration_ns + self.shift, 0.0, float(self.layout.slot_ns))
        return BlockParams(p.amplitude, p.phase_rad, p.frequency_hz, d)


# --- QuTiP reference and quadrature checks ------------------------------------------------------------------------

_REF: dict[str, Any] = {}


def _ref_init(n_detuning: int) -> None:
    from glowbox.sim.reference import ReferenceSimulator

    _REF["sim"] = ReferenceSimulator(n_detuning=n_detuning, detuning_rule="quantile")


def _ref_one(item: tuple[dict[str, Any], dict[str, Any]]) -> np.ndarray:
    from glowbox.control.pulse import PulseSequence

    seq, env = PulseSequence.from_dict(item[0]), EnvParams.from_dict(item[1])
    rate: np.ndarray = np.asarray(_REF["sim"].expected_rate_cps(seq, env))
    return rate


def gated_from_rate(rate: np.ndarray, layout: Layout, prefix_bins: int) -> np.ndarray:
    """Gated counts per shot per readout from a full-sequence rate (n_bins,)."""
    dt = layout.grid_ns * 1e-9
    out = []
    for k in range(layout.n_blocks):
        a = prefix_bins + layout.readout_start(k)
        out.append(float(np.sum(rate[a : a + layout.gate_bins]) * dt))
    return np.asarray(out)


def run_reference_check(task: AutoencoderTask) -> dict[str, Any]:
    rc = task.cfg["reference"]
    n = int(rc["n_patterns"])
    tr = task.seed0[("resonant", "learned")]
    sd, env, ly = task.data(0), task.env(0), tr.channel.layout
    t0 = time.perf_counter()
    with torch.no_grad():
        p = tr.encoder(sd.tte[:n])
    sfx = suffix_to_numpy(p, ly)
    pre = tr.channel.prefix
    seqs = [full_sequence(pre, row) for row in sfx.unbatch()]
    items = [(s.to_dict(), env.to_dict()) for s in seqs]
    ctx = mp.get_context("spawn")
    with ctx.Pool(int(rc["workers"]), initializer=_ref_init, initargs=(task.sys.options.n_detuning,)) as pool:
        rates = pool.map(_ref_one, items)
    ref = np.stack([gated_from_rate(r, ly, pre.n_bins) for r in rates])
    sur = tr.mu_te[:n].cpu().numpy()
    refm = tr.channel.reference().cpu().numpy()
    out: dict[str, Any] = {
        "n_patterns": n,
        "max_rel_diff": float(np.max(np.abs(ref - sur) / sur)),
        # relative to the input-dependent part of the counts (the signal the decoder reads)
        "max_diff_over_signal": float(np.max(np.abs(ref - sur)) / max(float(np.std(sur - refm, axis=0).max()), 1e-30)),
    }
    var = task.var(0)
    xs = sd.xte[:n]
    for src, mu in (("ref", ref), ("sur", sur)):
        g = _tgen(task.device, 4242)
        t = torch.as_tensor(np.repeat(mu, 64, axis=0), dtype=torch.float32, device=task.device)
        rec = reconstruct(tr.decoder, t, task.h_shots, g).cpu().numpy()
        out[f"nmse_noisy_{src}"] = nmse(rec, np.repeat(xs, 64, axis=0), var)
    with torch.no_grad():
        s = torch.full((n,), 1e9, device=task.device)
        a = tr.decoder(torch.as_tensor(ref, dtype=torch.float32, device=task.device), s).cpu().numpy()
        b = tr.decoder(torch.as_tensor(sur, dtype=torch.float32, device=task.device), s).cpu().numpy()
    out["noiseless_rec_rms_diff_over_sd"] = float(np.sqrt(np.mean((a - b) ** 2) / var))
    out["seconds"] = time.perf_counter() - t0
    out["example_timing"] = mw_timing_structure(seqs[0], env)
    return out


def run_quadrature_check(task: AutoencoderTask) -> dict[str, Any]:
    """Seed 0 resonant headline: the frozen system's counts at more quadrature nodes, decoded by the decoder trained
    at the working node count."""
    tr = task.seed0[("resonant", "learned")]
    m = int(task.cfg["quadrature_check"]["n_detuning"])
    spec = dataclasses.replace(task.sys, options=ModelOptions(n_detuning=m))
    ch = DiamondChannel(task.env(0), tr.channel.layout, spec, task.device)
    sd = task.data(0)
    mu = ch.counts_for(tr.encoder, sd.tte)
    var = task.var(0)
    base = tr.mu_te.double()
    sig = float((base - task.seed0[("resonant", "learned")].channel.reference()).std(0).max())
    return {
        "n_detuning": m,
        "max_rel_diff": float(((mu - base).abs() / base).max()),
        "max_diff_over_signal": float((mu - base).abs().max()) / max(sig, 1e-30),
        "nmse_working": task.nmse_of_counts(tr, base, sd, var, 0, task.h_shots),
        "nmse_more_nodes": task.nmse_of_counts(tr, mu, sd, var, 0, task.h_shots),
    }


def log_example_traces(task: AutoencoderTask) -> None:
    """Seed 0, headline K, two test patches, both conditions: full sequences on the surrogate, 10 shots each."""
    sim = SurrogateSimulator(task.device, task.sys.options)
    rng = np.random.default_rng([task.base_seed, 3])
    env, sd = task.env(0), task.data(0)
    for cname in ("resonant", "detuned"):
        tr = task.seed0.get((cname, "learned"))
        if tr is None:
            continue
        with torch.no_grad():
            p = tr.encoder(sd.tte[:2])
        sfx = suffix_to_numpy(p, tr.channel.layout, offset_hz=tr.channel.offset_hz)
        for j, row in enumerate(sfx.unbatch()):
            seq = full_sequence(tr.channel.prefix, row)
            trace = sim.run(seq, env, shots=10, rng=rng)
            task.run.log_trace(trace, seq, condition=cname, pattern=j, encoder="learned")


# --- plots --------------------------------------------------------------------------------------------------------


def _sel(df: pd.DataFrame, **kw: Any) -> pd.DataFrame:
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def plot_reconstructions(task: AutoencoderTask) -> Figure:
    sd = task.data(0)
    var = task.var(0)
    n = 8
    rows: list[tuple[str, np.ndarray]] = [("input", sd.xte[:n])]
    for cname, name in (("resonant", "learned"), ("resonant", "random"), ("detuned", "learned")):
        tr = task.seed0.get((cname, name))
        if tr is None:
            continue
        rec = reconstruct(tr.decoder, tr.mu_te[:n], task.h_shots, _tgen(task.device, 9)).cpu().numpy()
        rows.append((f"{LABELS[name]}\n{cname}, NMSE {nmse(rec, sd.xte[:n], var):.2f}", rec))
    size = int(task.cfg["data"]["size"])
    fig, axes = plt.subplots(len(rows), n, figsize=(1.1 * n + 1.6, 1.15 * len(rows)))
    for r, (lab, imgs) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[c].reshape(size, size), cmap="gray", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(lab, fontsize=6.5, rotation=0, ha="right", va="center")
    fig.suptitle(
        f"seed 0, K = {task.h_blocks} readouts, {task.h_shots} shots: test patches and reconstructions", fontsize=9
    )
    fig.tight_layout()
    return fig


def plot_nmse(task: AutoencoderTask) -> Figure:
    df = pd.DataFrame(task.rows)
    rdf = pd.DataFrame(task.ref_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    ax = axes[0]
    ks = sorted(df.blocks.unique())
    for name in ENCODERS:
        d = _sel(df, encoder=name, condition="resonant", shots=task.h_shots)
        _band(ax, ks, [d[d.blocks == k].nmse.to_numpy() for k in ks], COLORS[name], "-", LABELS[name])
        dd = _sel(df, encoder=name, condition="detuned", shots=task.h_shots)
        if len(dd):
            kd = sorted(dd.blocks.unique())
            _band(ax, kd, [dd[dd.blocks == k].nmse.to_numpy() for k in kd], COLORS[name], "--", f"{name}, detuned", "s")
    for ref in ("pca", "ideal"):
        _band(ax, ks, [rdf[rdf.blocks == k][ref].to_numpy() for k in ks], COLORS[ref], ":", LABELS[ref], "^")
    ax.set_xlabel("readouts K (numbers the decoder sees)")
    ax.set_ylabel(f"NMSE at {task.h_shots} shots")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.1)
    ax.set_title("reconstruction error vs bottleneck (mean over seeds, 95 % t CI)", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)
    _style(ax)
    ax = axes[1]
    lv = sorted(df.shots.unique())
    for name in ENCODERS:
        for cname, ls in (("resonant", "-"), ("detuned", "--")):
            d = _sel(df, encoder=name, condition=cname, blocks=task.h_blocks)
            if len(d):
                _band(ax, lv, [d[d.shots == s].nmse.to_numpy() for s in lv], COLORS[name], ls, f"{name}, {cname}")
    ax.set_xscale("log")
    ax.set_xlabel("shots per readout")
    ax.set_ylabel("NMSE")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"reconstruction error vs shots, K = {task.h_blocks}", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_quantum_test(task: AutoencoderTask, qt: QuantumTestResult) -> Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ticks, labels = [], []
    for x, name in enumerate(ENCODERS):
        s = qt.summary(f"{name}_nmse")
        for j, (cname, alpha) in enumerate((("resonant", 1.0), ("detuned", 0.45))):
            v, lo, hi = s[cname]
            ax.bar(x + 0.36 * j, v, 0.32, color=COLORS[name], alpha=alpha, edgecolor="none")
            ax.errorbar(x + 0.36 * j, v, yerr=[[v - lo], [hi - v]], color="#0b0b0b", lw=1, capsize=3)
        ticks.append(x + 0.18)
        labels.append(LABELS[name])
    ax.axhline(1.0, color="#8a8985", ls=":", lw=1.4)
    ax.set_xticks(ticks, labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(f"NMSE, K = {task.h_blocks}, {task.h_shots} shots")
    ax.set_title("quantum test: resonant (solid) vs detuned +0.4 GHz (light); dotted = mean patch", fontsize=9)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_sequences(task: AutoencoderTask) -> Figure:
    tr = task.seed0[("resonant", "learned")]
    ly = tr.channel.layout
    sd = task.data(0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    n = 4
    with torch.no_grad():
        p = tr.encoder(sd.tte[:n])
    amp = p.amplitude.cpu().numpy()
    ph = p.phase_rad.cpu().numpy()
    dur = p.quantized_ns(ly.grid_ns).cpu().numpy()
    cmap = plt.get_cmap("twilight")
    period = ly.period_bins * ly.grid_ns
    for i in range(n):
        y0 = (n - 1 - i) * 1.25
        for k in range(ly.n_blocks):
            t_blk = k * period
            for s in range(ly.n_segments):
                t0 = t_blk + s * ly.slot_ns
                d = dur[i, k, s]
                if d > 0:
                    ax.add_patch(
                        plt.Rectangle(
                            (t0, y0), d, amp[i, k, s], color=cmap((ph[i, k, s] + math.pi) / (2 * math.pi)), lw=0
                        )
                    )
            r0 = t_blk + ly.block_bins * ly.grid_ns
            ax.add_patch(plt.Rectangle((r0, y0), ly.readout_ns, 1.0, color="#1baf7a", alpha=0.12, lw=0))
            ax.add_patch(plt.Rectangle((r0, y0), ly.gate_ns, 1.0, color="#1baf7a", alpha=0.35, lw=0))
    ax.set_xlim(0, ly.n_bins * ly.grid_ns)
    ax.set_ylim(-0.1, n * 1.25)
    ax.set_yticks([(n - 1 - i) * 1.25 + 0.5 for i in range(n)], [f"patch {i}" for i in range(n)], fontsize=7)
    ax.set_xlabel("time after the initialisation (ns)")
    ax.set_title(
        "learned MW blocks (seed 0): height = amplitude, colour = phase; green = readout (dark = counted gate)",
        fontsize=8,
    )
    _style(ax)
    ax = axes[1]
    adf = pd.DataFrame(task.area_rows)
    bins = np.linspace(0, 6, 49)
    for name in ENCODERS:
        d = _sel(adf, encoder=name, condition="resonant", blocks=task.h_blocks)
        if len(d):
            a = np.concatenate([np.asarray(x) for x in d.areas])
            ax.hist(np.clip(a, 0, 6), bins=bins, color=COLORS[name], alpha=0.55, density=True, label=LABELS[name])
    for v in (0.5, 1.0, 2.0):
        ax.axvline(v, color="#8a8985", ls=":", lw=1)
    ax.set_xlabel("pulse area (units of pi; 0.5 = pi/2, 1 = pi)")
    ax.set_ylabel("density")
    ax.set_title(f"pulse areas, all seeds, K = {task.h_blocks}", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_timing(task: AutoencoderTask) -> Figure:
    df = pd.DataFrame(task.timing_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    for name in ENCODERS:
        d = _sel(df, encoder=name, probe="frozen")
        xs = sorted(d.shift_ns.unique())
        _band(ax, xs, [d[d.shift_ns == x].nmse.to_numpy() for x in xs], COLORS[name], "--", f"{name}, frozen decoder")
        r = _sel(df, encoder=name, probe="retrained")
        if len(r):
            xr = sorted(r.shift_ns.unique())
            mus = [t_ci(r[r.shift_ns == x].nmse.to_numpy()) for x in xr]
            ax.errorbar(
                xr,
                [m[0] for m in mus],
                yerr=[[m[0] - m[1] for m in mus], [m[2] - m[0] for m in mus]],
                fmt="o",
                color=COLORS[name],
                capsize=3,
                label=f"{name}, decoder retrained",
            )
    ax.set_xlabel("shift of every MW duration (ns)")
    ax.set_ylabel(f"NMSE (K = {task.h_blocks}, {task.h_shots} shots)")
    ax.set_title("timing probe (resonant), mean over seeds, 95 % t CI", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    ax = axes[1]
    for x, name in enumerate(ENCODERS):
        base = _sel(df, encoder=name, probe="frozen", shift_ns=0).sort_values("seed").nmse.to_numpy()
        scr = _sel(df, encoder=name, probe="phase_scramble").sort_values("seed").nmse.to_numpy()
        for j, (vals, alpha) in enumerate(((base, 1.0), (scr, 0.45))):
            v, lo, hi = t_ci(vals)
            ax.bar(x + 0.36 * j, v, 0.32, color=COLORS[name], alpha=alpha, edgecolor="none")
            if math.isfinite(lo):
                ax.errorbar(x + 0.36 * j, v, yerr=[[v - lo], [hi - v]], color="#0b0b0b", lw=1, capsize=3)
    ax.set_xticks([0.18, 1.18], list(ENCODERS), fontsize=8)
    ax.set_ylabel("NMSE")
    ax.set_title("phases as learned (solid) vs scrambled (light)", fontsize=9)
    _style(ax)
    fig.tight_layout()
    return fig


# --- report -------------------------------------------------------------------------------------------------------


def paired(df: pd.DataFrame, a: dict[str, Any], b: dict[str, Any], col: str = "nmse") -> tuple[float, float, float]:
    va = _sel(df, **a).sort_values("seed")[col].to_numpy()
    vb = _sel(df, **b).sort_values("seed")[col].to_numpy()
    return t_ci(va - vb)


def timing_verdicts(task: AutoencoderTask) -> dict[str, Any]:
    df = pd.DataFrame(task.timing_rows)
    adf = pd.DataFrame(task.area_rows)
    out: dict[str, Any] = {}
    for name in ENCODERS:
        base = _sel(df, encoder=name, probe="frozen", shift_ns=0).sort_values("seed").nmse.to_numpy()
        rel: dict[str, tuple[float, float, float]] = {}
        for probe, shifts in (("frozen", (-20, 20)), ("retrained", (-20, 20))):
            per = []
            for s in shifts:
                d = _sel(df, encoder=name, probe=probe, shift_ns=s).sort_values("seed").nmse.to_numpy()
                if d.size == base.size and d.size:
                    per.append(d / base - 1.0)
            if per:
                rel[probe] = t_ci(np.mean(per, axis=0))
        scr = _sel(df, encoder=name, probe="phase_scramble").sort_values("seed").nmse.to_numpy()
        if scr.size == base.size and scr.size:
            rel["phase_scramble"] = t_ci(scr / base - 1.0)
        a = _sel(adf, encoder=name, condition="resonant", blocks=task.h_blocks)
        out[name] = {
            "relative_nmse_increase": rel,
            "area_median_pi": t_ci(a.area_median_pi.to_numpy()),
            "frac_rabi_scale": t_ci(a.frac_rabi_scale.to_numpy()),
            "frac_off": t_ci(a.frac_off.to_numpy()),
            "duration_median_ns": t_ci(a.duration_median_ns.to_numpy()),
            "amplitude_median": t_ci(a.amplitude_median.to_numpy()),
        }
        o = out[name]
        o["relies_on_pulse_timing"] = bool("frozen" in rel and rel["frozen"][1] > 0.1)
        o["information_needs_exact_timing"] = bool("retrained" in rel and rel["retrained"][1] > 0.1)
        o["uses_phase_coherence"] = bool("phase_scramble" in rel and rel["phase_scramble"][1] > 0.1)
        o["rabi_scale_areas"] = bool(0.25 <= o["area_median_pi"][0] <= 4.0)
    lv = out["learned"]
    out["strategy_learns_timing"] = bool(lv["relies_on_pulse_timing"] and lv["rabi_scale_areas"])
    return out


def _bool(x: bool) -> str:
    return "yes" if x else "no"


def report_body(
    task: AutoencoderTask, qt: QuantumTestResult, tv: dict[str, Any], ref: dict[str, Any], quad: dict[str, Any]
) -> str:
    cfg = task.cfg
    df = pd.DataFrame(task.rows)
    rdf = pd.DataFrame(task.ref_rows)
    hk, hs = task.h_blocks, task.h_shots
    ly = task.layout(hk)
    s_l, s_r = qt.summary("learned_nmse"), qt.summary("random_nmse")
    sel = {"condition": "resonant", "blocks": hk, "shots": hs}
    lines = [
        "**Task 4, compress-and-reconstruct**: a learned pulse encoder writes 4x4 image patches (soft edges, two "
        "latent parameters) into MW blocks; the NV ensemble's gated readouts are the bottleneck (K photon counts, "
        "averaged over the shots); a decoder reconstructs the patch. Encoder and decoder are trained end to end "
        "through the differentiable surrogate. Simulation only; no claim of quantum advantage: a gap between the "
        "resonant and detuned arms only says that the performance depends on driving the spin resonance.",
        "",
        "## Result",
        "",
        f"- Headline (K = {hk} readouts, {hs} shots, mean over {len(qt.seeds)} seeds, 95 % Student-t CI): the learned "
        f"encoder reconstructs with NMSE {_ci(s_l['resonant'])}; the fixed random encoder (milestone-3 reservoir "
        f"parametrisation, same decoder recipe) {_ci(s_r['resonant'])}. Paired difference random - learned "
        f"{_ci(paired(df, {**sel, 'encoder': 'random'}, {**sel, 'encoder': 'learned'}))}.",
        f"- References at K = {hk} (noiseless numbers): PCA {_ci(t_ci(rdf[rdf.blocks == hk].pca.to_numpy()))}, "
        f"ideal K-number autoencoder {_ci(t_ci(rdf[rdf.blocks == hk].ideal.to_numpy()))}.",
        "- **Quantum test**: with every MW frequency shifted by "
        f"{float(cfg['detuned_offset_hz']) / 1e6:.0f} MHz (no line within "
        f"{task.no_line_margin_hz / 1e6:.0f} MHz of any MW frequency), learned NMSE {_ci(s_l['detuned'])}, random "
        f"{_ci(s_r['detuned'])}. Gaps (detuned - resonant): learned {_ci(s_l['gap'])} "
        f"({'significant' if qt.gap_significant('learned_nmse') else 'not significant'}), random {_ci(s_r['gap'])} "
        f"({'significant' if qt.gap_significant('random_nmse') else 'not significant'}).",
        f"- Timing: learned strategy uses microsecond-scale (Rabi-scale) timing: "
        f"**{_bool(tv['strategy_learns_timing'])}** (details below).",
        "",
        "## Task, bottleneck and training",
        "",
        f"- Data: 4x4 patches tanh({cfg['data']['sharpness']} (x cos theta + y sin theta - o)), theta uniform, "
        f"|o| <= {cfg['data']['offset_max']}; {cfg['data']['n_train']} train / {cfg['data']['n_val']} val / "
        f"{cfg['data']['n_test']} test patches per seed. NMSE = MSE / mean per-pixel variance (1 = mean patch).",
        f"- Sequence: init laser {task.sys.init_ns} ns, dark {task.sys.settle_ns} ns, then K x [{ly.n_segments} MW "
        f"segments in {ly.slot_ns} ns slots, a {ly.readout_ns} ns readout laser of which the first {ly.gate_ns} ns are "
        "counted (one number per readout)]. Each segment: amplitude in [0, 1], phase, frequency within "
        f"+-{ly.span_hz / 1e6:.0f} MHz of a device line (segment j of block k on line (4k + j) mod 8 of the known "
        f"bias field {cfg['device']['bias_field_t']} T), duration 0-{ly.max_ns:.0f} ns on the 10 ns grid "
        "(straight-through rounding; the surrogate's run-length path gives exact duration gradients).",
        f"- Device per seed: nuisance draw (T2*, Rabi full scale, beta, collection, background, strain) from the "
        f"milestone-3 prior; T2* quadrature {task.sys.options.n_detuning} quantile nodes.",
        f"- Phase 1: Adam, {cfg['train']['steps']} steps, batch {cfg['train']['batch']}, Gaussian shot noise "
        f"(Poisson variance), shots log-uniform {cfg['train']['shots']}. Phase 2: fresh decoder, "
        f"{cfg['bank_decoder']['steps']} steps on a bank with Poisson noise. Wall time: phase 1 "
        f"{task.seconds['phase1'] / 60:.0f} min, phase 2 {task.seconds['phase2'] / 60:.1f} min, timing probes "
        f"{task.seconds['timing'] / 60:.1f} min.",
        "",
        "## Error vs bottleneck and noise",
        "",
        f"NMSE at {hs} shots, resonant, mean [95 % t CI] over seeds:",
        "",
        "| K | learned | random | PCA (noiseless) | ideal K-number AE (noiseless) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for k in sorted(df.blocks.unique()):
        cells = [str(k)]
        for name in ENCODERS:
            d = _sel(df, encoder=name, condition="resonant", blocks=k, shots=hs)
            cells.append(_ci(t_ci(d.nmse.to_numpy())))
        r = rdf[rdf.blocks == k]
        cells += [_ci(t_ci(r.pca.to_numpy())), _ci(t_ci(r.ideal.to_numpy()))]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        f"NMSE vs shots at K = {hk}:",
        "",
        "| shots | " + " | ".join(f"{n}, {c}" for n in ENCODERS for c in ("resonant", "detuned")) + " |",
        "|" + " --- |" * 5,
    ]
    for s in sorted(df.shots.unique()):
        cells = [str(s)]
        for name in ENCODERS:
            for c in ("resonant", "detuned"):
                d = _sel(df, encoder=name, condition=c, blocks=hk, shots=s)
                cells.append(_ci(t_ci(d.nmse.to_numpy())) if len(d) else "")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Quantum test", "", qt.to_markdown(), ""]
    lines += [
        "## Learned strategies and timing",
        "",
        "Pulse areas are the resonant rotation angle 2 f_R a d (f_R the seed's full-scale Rabi frequency on a [100] "
        "MW field; units of pi). Probes (resonant, K = headline, every seed): every MW duration shifted with the "
        "decoder frozen (the harness's `timing_sensitivity`) and with a decoder retrained on the shifted encoding; "
        "every segment's phase replaced by an independent random phase (frozen decoder). Relative NMSE increase, "
        "mean [95 % t CI] over seeds:",
        "",
        "| encoder | median area (pi) | fraction of segments at 0.25-4 pi | fraction off | median duration (ns) | "
        "+-20 ns, frozen | +-20 ns, retrained | phases scrambled |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in ENCODERS:
        o = tv[name]
        rel = o["relative_nmse_increase"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _ci(o["area_median_pi"]),
                    _ci(o["frac_rabi_scale"], "{:.2f}"),
                    _ci(o["frac_off"], "{:.2f}"),
                    _ci(o["duration_median_ns"], "{:.0f}"),
                    *[_ci(rel[p], "{:+.2f}") if p in rel else "" for p in ("frozen", "retrained", "phase_scramble")],
                ]
            )
            + " |"
        )
    lv = tv["learned"]
    lines += [
        "",
        f"- Learned encoder relies on its pulse timing (frozen NMSE up > 10 % at +-20 ns, CI): "
        f"{_bool(lv['relies_on_pulse_timing'])}; the information needs the exact timing (retrained decoder still "
        f"worse > 10 %): {_bool(lv['information_needs_exact_timing'])}; it uses coherent phase relations between "
        f"segments (scrambling costs > 10 %): {_bool(lv['uses_phase_coherence'])}; its pulse areas are Rabi-scale "
        f"(median 0.25-4 pi): {_bool(lv['rabi_scale_areas'])}. `strategy_learns_timing` = relies on pulse timing "
        f"and Rabi-scale areas = **{_bool(tv['strategy_learns_timing'])}**.",
        "- The same probes on the random encoder show what an unlearned encoding of the same family does.",
        "",
        "## Checks",
        "",
        f"- QuTiP reference (seed 0, resonant, K = {hk}, {ref['n_patterns']} learned test sequences run as full "
        f"PulseSequences, {task.sys.options.n_detuning} quadrature nodes): gated counts within "
        f"{ref['max_rel_diff']:.2g} (relative) of the surrogate, {ref['max_diff_over_signal']:.2g} of the "
        "input-dependent signal; frozen decoder NMSE at "
        f"{hs} shots {ref['nmse_noisy_ref']:.4f} on reference counts vs {ref['nmse_noisy_sur']:.4f} on surrogate "
        f"counts (64 noise draws per patch, same draws); noiseless reconstructions differ by "
        f"{ref['noiseless_rec_rms_diff_over_sd']:.2g} of the pixel sd ({ref['seconds']:.0f} s).",
        f"- Quadrature: the frozen seed-0 system at {quad['n_detuning']} nodes: counts within "
        f"{quad['max_rel_diff']:.2g} (relative), {quad['max_diff_over_signal']:.2g} of the signal; NMSE "
        f"{quad['nmse_working']:.4f} -> {quad['nmse_more_nodes']:.4f} with the decoder trained at "
        f"{task.sys.options.n_detuning} nodes.",
        "",
        "## Assumptions and limits",
        "",
        "- One known device per seed (calibrated lines); the encoder's frequencies are tied to those lines. The "
        "readout is a gated photon count; only the counted gates reach the decoder (the rest of each laser pulse "
        "re-polarises).",
        "- Training uses Gaussian shot noise with the Poisson variance (at >= 10^3 shots the gated counts are "
        "> 10^3 photons); evaluation draws Poisson counts.",
        "- Durations are rounded to the 10 ns grid in the forward pass (straight-through gradient), so every learned "
        "sequence is a valid PulseSequence; the QuTiP check runs them as such.",
        "- The detuned arm is trained exactly like the resonant one (its encoder still learns, but its MW drives "
        "nothing).",
    ]
    return "\n".join(lines)


def main(
    cfg: dict[str, Any], output_root: Path | None = None, report_dir: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw: dict[str, Any] = {} if output_root is None else {"output_root": output_root}
    t_start = time.perf_counter()
    with RunLog(str(cfg["name"]), config=cfg, seed=int(cfg["seed"]), **kw) as run:
        task = AutoencoderTask(cfg, run, device)
        conds = (RESONANT, detuned_condition(float(cfg["detuned_offset_hz"])))
        hib = {f"{n}_{m}": m == "skill" for n in ENCODERS for m in ("nmse", "skill")}
        qt = run_quantum_test(task, [int(s) for s in cfg["seeds"]], conds, higher_is_better=hib)
        print("[m4-ae] reference check", flush=True)
        ref = run_reference_check(task)
        quad = run_quadrature_check(task)
        log_example_traces(task)
        tv = timing_verdicts(task)

        pd.DataFrame(task.rows).to_parquet(run.dir / "results.parquet")
        pd.DataFrame(task.timing_rows).to_parquet(run.dir / "timing.parquet")
        pd.DataFrame(task.area_rows).to_parquet(run.dir / "areas.parquet")
        pd.DataFrame(task.ref_rows).to_parquet(run.dir / "references.parquet")
        (run.dir / "train_history.json").write_text(json.dumps(task.histories), encoding="utf-8")
        (run.dir / "reference_check.json").write_text(json.dumps(ref, default=str), encoding="utf-8")
        (run.dir / "quadrature_check.json").write_text(json.dumps(quad), encoding="utf-8")
        (run.dir / "timing_verdicts.json").write_text(json.dumps(tv, default=str), encoding="utf-8")
        (run.dir / "quantum_test.json").write_text(
            json.dumps({"seeds": qt.seeds, "values": qt.values}), encoding="utf-8"
        )

        figs = {
            "reconstructions": plot_reconstructions(task),
            "nmse": plot_nmse(task),
            "quantum-test": plot_quantum_test(task, qt),
            "sequences": plot_sequences(task),
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
            for key, i in (("resonant", 0), ("detuned", 0)):
                summary[f"{m}_{key}"] = s[key][i]
            summary[f"{m}_gap"], summary[f"{m}_gap_lo"], summary[f"{m}_gap_hi"] = s["gap"]
        summary["ref_max_rel_diff"] = ref["max_rel_diff"]
        summary["quad_max_rel_diff"] = quad["max_rel_diff"]
        summary["strategy_learns_timing"] = float(tv["strategy_learns_timing"])
        summary["learned_relies_on_pulse_timing"] = float(tv["learned"]["relies_on_pulse_timing"])
        summary["learned_uses_phase_coherence"] = float(tv["learned"]["uses_phase_coherence"])
        summary["no_line_margin_mhz"] = task.no_line_margin_hz / 1e6
        summary["phase1_seconds"] = task.seconds["phase1"]
        summary["wall_seconds"] = time.perf_counter() - t_start
        run.log_metrics(step=0, **{k: v for k, v in summary.items() if isinstance(v, int | float)})
        body = report_body(task, qt, tv, ref, quad)
    rdir = report_dir if report_dir is not None else run.dir / "report"
    title = "Milestone 4, Task 4: compress-and-reconstruct through the NV ensemble (learned encoder)"
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
