"""Milestone 1: CW ODMR and pulsed Rabi for an NV ensemble on both simulators, plotted and compared.

    uv run python -m glowbox.experiments.m1_odmr_rabi [--config PATH] [--quick] [--report-dir DIR]

Runs the QuTiP reference and the fast surrogate on identical sequences, parameters and seeds (the same detuning
quadrature nodes too), logs everything with :class:`~glowbox.analysis.runlog.RunLog` (config, seed, git hash, raw
traces in Parquet, metrics, plots under ``outputs/<run-id>/``), measures the surrogate's shots/s per device, and
writes a Markdown report. The report goes to ``outputs/<run-id>/report/`` unless ``--report-dir`` is given
(``docs/results`` for the committed one; on the GPU machine keep the default so its checkout stays clean and copy
the report back).

Raw traces: each logged trace holds the noiseless ``rate_cps`` and the photon counts *summed over all shots* of the
point (tag ``summed_shots``): 20000 Rabi shots x 101 sequences x 660 bins would otherwise be ~10 GB.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.figure import Figure

from glowbox.analysis.protocols import Dip, RabiFit, find_dips, fit_rabi, late_mean, readout_mean
from glowbox.analysis.runlog import RunLog, write_report
from glowbox.control.pulse import PulseSequence, cw_odmr_point, pulsed_rabi
from glowbox.sim.interface import EnvParams, FluorescenceTrace, Simulator
from glowbox.sim.reference import ReferenceSimulator, transition_frequencies_hz
from glowbox.sim.surrogate import ModelOptions, SurrogateSimulator

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "m1_odmr_rabi.yaml"
REPORT_NAME = "m1-odmr-rabi"
#: Sequences per surrogate batch inside a sweep (memory: shots x n_bins x 8 B of counts per sequence).
SWEEP_CHUNK = 16


def load_config(path: Path, quick: bool) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    overrides = cfg.pop("quick", {}) or {}
    if quick:
        cfg = _merge(cfg, overrides)
        cfg["quick"] = True
    return cfg


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# --- running ------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class SweepResult:
    """One simulator's sweep: noiseless rows, shot-summed counts and wall time."""

    rates: list[np.ndarray]
    counts: list[np.ndarray]  # summed over shots, (n_bins,)
    shots: int
    seconds: float


def run_sweep(
    sim: Simulator, seqs: list[PulseSequence], env: EnvParams, shots: int, seed: int, run: RunLog, protocol: str
) -> SweepResult:
    """Run every sequence with ``shots`` shots; the generator is seeded identically for every simulator."""
    rng = np.random.default_rng(seed)
    res = SweepResult([], [], shots, 0.0)
    # Chunks keep the per-shot count arrays (shots x n_bins int64 each) of only a few sequences in memory.
    chunk = SWEEP_CHUNK if isinstance(sim, SurrogateSimulator) else 1
    for start in range(0, len(seqs), chunk):
        part = seqs[start : start + chunk]
        t0 = time.perf_counter()
        if isinstance(sim, SurrogateSimulator):
            traces = sim.run_batch(part, env, shots=shots, rng=rng)
        else:
            traces = [sim.run(s, env, shots=shots, rng=rng) for s in part]
        res.seconds += time.perf_counter() - t0
        for i, (s, tr) in enumerate(zip(part, traces, strict=True), start):
            assert tr.rate_cps is not None
            rate, counts = np.array(tr.rate_cps[0]), tr.counts.sum(axis=0)
            res.rates.append(rate)
            res.counts.append(counts)
            summed = FluorescenceTrace(counts[None], tr.bin_ns, rate[None], tr.metadata)
            run.log_trace(summed, s, protocol=protocol, index=i, summed_shots=shots)
    return res


def surrogate_throughput(
    devices: list[str], seqs: list[PulseSequence], env: EnvParams, shots: int, repeat: int, n_detuning: int
) -> list[dict[str, Any]]:
    """Best-of-``repeat`` wall time of ``run_batch`` (noiseless rates + Poisson counts for every shot) per device."""
    rows = []
    for dev in devices:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        sim = SurrogateSimulator(dev, ModelOptions(n_detuning=n_detuning))
        sim.run_batch(seqs[:2], env, shots=2, rng=np.random.default_rng(0))  # warm-up (CUDA context, kernels)
        best = float("inf")
        for r in range(repeat):
            t0 = time.perf_counter()
            sim.run_batch(seqs, env, shots=shots, rng=np.random.default_rng(r))
            best = min(best, time.perf_counter() - t0)
        name = torch.cuda.get_device_name(0) if dev == "cuda" else f"CPU ({torch.get_num_threads()} torch threads)"
        rows.append(
            {
                "device": dev,
                "name": name,
                "sequences": len(seqs),
                "n_bins": seqs[-1].n_bins,
                "shots_each": shots,
                "seconds": best,
                "shots_per_s": len(seqs) * shots / best,
                "sequences_per_s": len(seqs) / best,
            }
        )
    return rows


# --- analysis -----------------------------------------------------------------------------------------------------


def odmr_signal(res: SweepResult, grid_ns: int, late_ns: int) -> tuple[np.ndarray, np.ndarray]:
    """(noiseless steady-state rate, measured rate from the shot-summed counts), counts/s, per frequency."""
    clean = np.array([late_mean(r, grid_ns, late_ns) for r in res.rates])
    n = late_ns // grid_ns
    measured = np.array([c[-n:].sum() for c in res.counts]) / (res.shots * late_ns * 1e-9)
    return clean, measured


def rabi_signal(res: SweepResult, seqs: list[PulseSequence], window_ns: int) -> tuple[np.ndarray, np.ndarray]:
    """(noiseless, measured) contrast 1 - readout(t)/readout(0) per MW duration."""
    clean = np.array([readout_mean(r, s, window_ns) for r, s in zip(res.rates, seqs, strict=True)])
    measured = np.array(
        [readout_mean(c.astype(np.float64), s, window_ns) for c, s in zip(res.counts, seqs, strict=True)]
    )
    return 1.0 - clean / clean[0], 1.0 - measured / measured[0]


def safe_fit_rabi(t_s: np.ndarray, y: np.ndarray) -> RabiFit:
    """``fit_rabi`` for shot-noise data: a failed fit gives NaNs instead of aborting the run."""
    try:
        return fit_rabi(t_s, y)
    except RuntimeError:
        nan = float("nan")
        return RabiFit(nan, nan, nan, nan, nan, nan)


def match_dips(lines: np.ndarray, dips: list[Dip]) -> list[Dip | None]:
    """For each predicted line, the dip whose grid minimum is closest (within 2 MHz), else None."""
    out: list[Dip | None] = []
    for f in lines:
        best = min(dips, key=lambda d: abs(d.grid_frequency_hz - f), default=None)
        out.append(best if best is not None and abs(best.grid_frequency_hz - f) < 2e6 else None)
    return out


# --- plots --------------------------------------------------------------------------------------------------------

REF_COLOR, SUR_COLOR = "#1f4e9c", "#d0661c"


def plot_odmr(
    freqs: np.ndarray,
    ref: tuple[np.ndarray, np.ndarray, float],
    sur: tuple[np.ndarray, np.ndarray, float],
    lines: np.ndarray,
    shots: int,
) -> Figure:
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, height_ratios=[3, 1], layout="constrained")
    ghz = freqs / 1e9
    for f in lines.ravel():
        ax.axvline(f / 1e9, color="0.85", lw=0.8, zorder=0)
    (rc, rm, r0), (sc, sm, s0) = ref, sur
    ax.plot(ghz, rm / r0, ".", ms=3, color=REF_COLOR, alpha=0.45, label=f"reference, {shots} shots")
    ax.plot(ghz, sm / s0, ".", ms=3, color=SUR_COLOR, alpha=0.45, label=f"surrogate, {shots} shots")
    ax.plot(ghz, rc / r0, "-", lw=1.6, color=REF_COLOR, label="reference, noiseless")
    ax.plot(ghz, sc / s0, "--", lw=1.2, color=SUR_COLOR, label="surrogate, noiseless")
    ax.set_ylabel("steady-state fluorescence / MW off")
    ax.set_title("CW ODMR, NV ensemble, generic field: 4 axes x 2 lines (grey: H0 eigenvalues)")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    axr.plot(ghz, (sc / s0 - rc / r0) * 1e6, color="0.3", lw=1)
    axr.set_ylabel("surr. - ref.\n(ppm)")
    axr.set_xlabel("MW frequency (GHz)")
    return fig


def plot_rabi(
    t_ns: np.ndarray,
    ref: tuple[np.ndarray, np.ndarray, RabiFit],
    sur: tuple[np.ndarray, np.ndarray, RabiFit],
    shots: int,
    line_hz: float,
) -> Figure:
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, height_ratios=[3, 1], layout="constrained")
    (rc, rm, rf), (sc, sm, sf) = ref, sur
    ax.plot(t_ns, rm * 100, ".", ms=4, color=REF_COLOR, alpha=0.5, label=f"reference, {shots} shots")
    ax.plot(t_ns, sm * 100, ".", ms=4, color=SUR_COLOR, alpha=0.5, label=f"surrogate, {shots} shots")
    ax.plot(
        t_ns,
        rc * 100,
        "-",
        lw=1.6,
        color=REF_COLOR,
        label=f"reference, noiseless (fit {rf.frequency_hz / 1e6:.3f} MHz)",
    )
    ax.plot(
        t_ns,
        sc * 100,
        "--",
        lw=1.2,
        color=SUR_COLOR,
        label=f"surrogate, noiseless (fit {sf.frequency_hz / 1e6:.3f} MHz)",
    )
    ax.set_ylabel("readout contrast (%)")
    ax.set_title(f"Pulsed Rabi on the {line_hz / 1e9:.4f} GHz line (one NV axis of four)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    axr.plot(t_ns, (sc - rc) * 1e6, color="0.3", lw=1)
    axr.set_ylabel("surr. - ref.\n(contrast, ppm)")
    axr.set_xlabel("MW pulse duration (ns)")
    return fig


# --- main ---------------------------------------------------------------------------------------------------------


def main(
    cfg: dict[str, Any], output_root: Path | None = None, report_dir: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    """Run the milestone; returns (report path, summary numbers)."""
    env = EnvParams.from_dict(cfg.get("env", {}))
    m = int(cfg["n_detuning"])
    oc, rc = cfg["odmr"], cfg["rabi"]
    seed = int(cfg["seed"])
    lines = transition_frequencies_hz(env.d_hz, env.e_hz, np.asarray(env.b_field_t))  # (4, 2)
    ref = ReferenceSimulator(n_detuning=m)
    sur = SurrogateSimulator(None, ModelOptions(n_detuning=m))
    kw: dict[str, Any] = {} if output_root is None else {"output_root": output_root}
    with RunLog(str(cfg["name"]), config=cfg, seed=seed, **kw) as run:
        # CW ODMR: one sequence per frequency plus a MW-off baseline (last).
        freqs = np.arange(lines.min() - oc["margin_hz"], lines.max() + oc["margin_hz"] + 1, oc["step_hz"])
        odmr_seqs = [cw_odmr_point(float(f), duration_ns=oc["duration_ns"], amplitude=oc["amplitude"]) for f in freqs]
        odmr_seqs.append(dataclasses.replace(odmr_seqs[0], microwave=(), label={"protocol": "cw_odmr", "mw": "off"}))
        g = odmr_seqs[0].grid_ns
        o_ref = run_sweep(ref, odmr_seqs, env, oc["shots"], seed, run, "cw_odmr")
        o_sur = run_sweep(sur, odmr_seqs, env, oc["shots"], seed, run, "cw_odmr")
        (orc, orm), (osc, osm) = odmr_signal(o_ref, g, oc["late_ns"]), odmr_signal(o_sur, g, oc["late_ns"])
        ref_dips = match_dips(lines.ravel(), find_dips(freqs, orc[:-1], orc[-1]))
        sur_dips = match_dips(lines.ravel(), find_dips(freqs, osc[:-1], osc[-1]))

        # Pulsed Rabi on one line.
        axis, branch = rc["line"]
        f_line = float(lines[axis, branch])
        durations = np.arange(0, rc["max_ns"] + 1, rc["step_ns"])
        rabi_seqs = [
            pulsed_rabi(
                int(t),
                f_line,
                rc["amplitude"],
                init_ns=rc["init_ns"],
                settle_ns=rc["settle_ns"],
                readout_ns=rc["readout_ns"],
            )
            for t in durations
        ]
        r_ref = run_sweep(ref, rabi_seqs, env, rc["shots"], seed + 1, run, "rabi")
        r_sur = run_sweep(sur, rabi_seqs, env, rc["shots"], seed + 1, run, "rabi")
        (rrc, rrm), (rsc, rsm) = (
            rabi_signal(r_ref, rabi_seqs, rc["window_ns"]),
            rabi_signal(r_sur, rabi_seqs, rc["window_ns"]),
        )
        t_s = durations * 1e-9
        fit_ref, fit_sur, fit_noisy = fit_rabi(t_s, rrc), fit_rabi(t_s, rsc), safe_fit_rabi(t_s, rsm)

        # Agreement of the noiseless rows, all sequences.
        def max_rel(a: list[np.ndarray], b: list[np.ndarray]) -> float:
            return max(float(np.max(np.abs(x - y)) / np.max(y)) for x, y in zip(a, b, strict=True))

        rel_odmr, rel_rabi = max_rel(o_sur.rates, o_ref.rates), max_rel(r_sur.rates, r_ref.rates)

        # Throughput.
        tc = cfg["throughput"]
        n_tp = int(tc["n_sequences"])
        tp_seqs = [rabi_seqs[i % len(rabi_seqs)] for i in range(n_tp)]
        tp = surrogate_throughput(list(tc["devices"]), tp_seqs, env, int(tc["shots"]), int(tc["repeat"]), m)

        # Metrics.
        summary: dict[str, Any] = {
            "odmr_points": len(freqs),
            "odmr_max_rel_rate_diff": rel_odmr,
            "rabi_max_rel_rate_diff": rel_rabi,
            "rabi_freq_ref_hz": fit_ref.frequency_hz,
            "rabi_freq_sur_hz": fit_sur.frequency_hz,
            "rabi_freq_sur_noisy_hz": fit_noisy.frequency_hz,
            "rabi_decay_ref_s": fit_ref.decay_s,
            "rabi_decay_sur_s": fit_sur.decay_s,
            "rabi_amp_ref": fit_ref.amplitude,
            "rabi_amp_sur": fit_sur.amplitude,
            "rabi_freq_spec_hz": env.rabi_full_scale_hz * rc["amplitude"] * np.sqrt(2.0 / 3.0),
            "ref_seconds_odmr": o_ref.seconds,
            "sur_seconds_odmr": o_sur.seconds,
            "ref_seconds_rabi": r_ref.seconds,
            "sur_seconds_rabi": r_sur.seconds,
        }
        for row in tp:
            summary[f"shots_per_s_{row['device']}"] = row["shots_per_s"]
        run.log_metrics(0, **{k: float(v) for k, v in summary.items()})

        figs = [
            run.save_figure(
                plot_odmr(freqs, (orc[:-1], orm[:-1], orc[-1]), (osc[:-1], osm[:-1], osc[-1]), lines, oc["shots"]),
                "odmr",
                dpi=100,
            ),
            run.save_figure(
                plot_rabi(durations, (rrc, rrm, fit_ref), (rsc, rsm, fit_sur), rc["shots"], f_line), "rabi", dpi=100
            ),
        ]
        plt.close("all")
        text = report_text(cfg, env, lines, ref_dips, sur_dips, summary, fit_ref, fit_sur, fit_noisy, tp, f_line)
    out = report_dir if report_dir is not None else run.dir / "report"
    path = write_report(
        run.dir, "Milestone 1: CW ODMR and Rabi, surrogate vs QuTiP reference", text, figs, out, REPORT_NAME
    )
    return path, summary


def report_text(
    cfg: dict[str, Any],
    env: EnvParams,
    lines: np.ndarray,
    ref_dips: list[Dip | None],
    sur_dips: list[Dip | None],
    s: dict[str, Any],
    fit_ref: RabiFit,
    fit_sur: RabiFit,
    fit_noisy: RabiFit,
    tp: list[dict[str, Any]],
    f_line: float,
) -> str:
    oc, rc = cfg["odmr"], cfg["rabi"]
    bt = np.asarray(env.b_field_t) * 1e3
    out = [
        "Both simulators ran the same sequences with the same parameters, detuning nodes "
        f"(M = {cfg['n_detuning']}) and seeds: the QuTiP reference (seven-level Lindblad, CPU) and the fast surrogate "
        "(torch, float32, on the GPU when present). Field B = "
        f"({bt[0]:.2f}, {bt[1]:.2f}, {bt[2]:.2f}) mT in the crystal frame, so the four NV axes are inequivalent and "
        "CW ODMR shows eight lines; all other parameters are the `EnvParams` defaults "
        f"(T2* = {env.t2_star_s * 1e6:g} µs, T2 = {env.t2_s * 1e6:g} µs, β = {env.optical_beta_full_scale:g}, "
        f"{env.n_nv:g} NVs, collection {env.collection_efficiency:g}, background {env.background_cps:g} counts/s).",
        "",
        "## Plots",
        "",
        f"**CW ODMR** ({s['odmr_points']} frequencies, {oc['step_hz'] / 1e6:g} MHz step, "
        f"{oc['duration_ns'] / 1e3:g} µs "
        f"laser + MW at amplitude {oc['amplitude']:g}, steady state from the last {oc['late_ns']} ns): fluorescence "
        f"relative to MW off. Dots are {oc['shots']} shots of photon counts per point from each simulator, lines the "
        "noiseless rates; grey verticals are the transition frequencies from diagonalising H0 per axis. The lower "
        "panel is the surrogate minus the reference (noiseless), in parts per million.",
        "",
        f"**Pulsed Rabi** on the {f_line / 1e9:.4f} GHz line (axis {rc['line'][0]}), MW amplitude {rc['amplitude']:g}, "
        f"0 to {rc['max_ns']} ns in {rc['step_ns']} ns steps, {rc['init_ns'] / 1e3:g} µs init, "
        f"{rc['settle_ns'] / 1e3:g} µs dark, readout contrast from the first {rc['window_ns']} ns of the readout "
        f"pulse relative to no MW. Dots: {rc['shots']} shots per duration; lines: noiseless. Only one axis of four is "
        "resonant, so the contrast is a quarter of a single-axis contrast.",
        "",
        "## Comparison",
        "",
        "Noiseless rates, every bin of every sequence: max |surrogate − reference| / peak rate = "
        f"{s['odmr_max_rel_rate_diff']:.1e} (ODMR), {s['rabi_max_rel_rate_diff']:.1e} (Rabi). That is float32 "
        "round-off: in float64 the two agree to ~1e-9 (tests/sim/test_surrogate_vs_reference.py).",
        "",
        "CW ODMR lines (Lorentzian fit to the noiseless spectrum around each grid minimum):",
        "",
        "| axis | line | H0 eigenvalue (GHz) | grid min ref / surr (GHz) | fit centre ref − surr (Hz) "
        "| contrast ref / surr (%) | FWHM ref / surr (MHz) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, f in enumerate(lines.ravel()):
        a, b = ref_dips[i], sur_dips[i]
        name = ("f−", "f+")[i % 2]
        if a is None or b is None:
            out.append(f"| {i // 2} | {name} | {f / 1e9:.5f} | not resolved | | | |")
            continue
        out.append(
            f"| {i // 2} | {name} | {f / 1e9:.5f} "
            f"| {a.grid_frequency_hz / 1e9:.4f} / {b.grid_frequency_hz / 1e9:.4f} | "
            f"{a.center_hz - b.center_hz:+.0f} | {a.contrast * 100:.3f} / {b.contrast * 100:.3f} | "
            f"{a.fwhm_hz / 1e6:.3f} / {b.fwhm_hz / 1e6:.3f} |"
        )
    out += [
        "",
        "Rabi (fit offset + A·exp(−t/τ)·cos(2πft + φ) to the contrast):",
        "",
        "| | f (MHz) | τ (µs) | A (%) |",
        "| --- | --- | --- | --- |",
        fit_row("reference, noiseless", fit_ref),
        fit_row("surrogate, noiseless", fit_sur),
        fit_row(f"surrogate, {rc['shots']} shots", fit_noisy),
        "",
        f"Spec value f_R = rabi_full_scale · a · √(2/3) = {s['rabi_freq_spec_hz'] / 1e6:.4f} MHz. The fitted "
        "frequency is slightly higher because off-resonant spins of the Lorentzian T2* distribution oscillate at "
        "√(f_R² + δ²); τ is an effective exponential time (the T2*-averaged envelope is not exponential).",
        "",
        "## Speed",
        "",
        f"Wall time on this run: reference {s['ref_seconds_odmr']:.1f} s (ODMR) + {s['ref_seconds_rabi']:.1f} s "
        f"(Rabi) on the CPU; surrogate {s['sur_seconds_odmr']:.2f} s + {s['sur_seconds_rabi']:.2f} s.",
        "",
        "Surrogate throughput, `run_batch` of Rabi sequences (noiseless simulation plus Poisson counts for every "
        f"shot, M = {cfg['n_detuning']}, float32, no laser drift; best of {cfg['throughput']['repeat']}):",
        "",
        "| device | sequences x bins | shots each | time (s) | shots/s | sequences/s |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    out += [
        f"| {r['name']} | {r['sequences']} x {r['n_bins']} | {r['shots_each']} | {r['seconds']:.2f} | "
        f"{r['shots_per_s']:,.0f} | {r['sequences_per_s']:,.1f} |"
        for r in tp
    ]
    out += [
        "",
        "Cost is per distinct sequence; extra shots only add Poisson sampling. With laser drift every shot is its own "
        "simulation (docs/notes/surrogate.md).",
    ]
    return "\n".join(out)


def fit_row(name: str, f: RabiFit) -> str:
    return f"| {name} | {f.frequency_hz / 1e6:.4f} | {f.decay_s * 1e6:.3f} | {f.amplitude * 100:.3f} |"


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="coarse grids and few shots (smoke run)")
    p.add_argument("--report-dir", type=Path, default=None, help="where to write the report (e.g. docs/results)")
    p.add_argument("--output-root", type=Path, default=None, help="run directories (default: <repo>/outputs)")
    a = p.parse_args()
    path, summary = main(load_config(a.config, a.quick), a.output_root, a.report_dir)
    for k, v in summary.items():
        print(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}")
    print(f"report: {path}")


if __name__ == "__main__":
    cli()
