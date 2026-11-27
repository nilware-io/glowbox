"""PREREG-001, stage S: null calibration and positive control of the whole pre-registered pipeline.

    uv run python -m glowbox.experiments.prereg_calibration [--config PATH] [--quick] [--report-dir DIR]

The simulator has no decision or observer effect by construction, so nothing here tests the hypotheses. Each
replicate is one complete simulated experiment, run exactly as the plan prescribes:

1. Offline action sequences: the decision process run on a separate simulated device (not the measurement under
   test); they are what the calibration controls replay, and their marginal is the sham distribution.
2. CALIBRATION block (M_CAL valid blocks, arms C1-C4, every one a sealed replay with no decision applied and no live
   reader): :func:`~glowbox.analysis.prereg_analysis_001.derive_calibration` freezes the empirical null, the
   margins (smallest effect sizes of interest) and N.
3. Confirmatory run (N valid blocks, arms A1-A5 with the sealed controls A3, A3b, A3L), randomised order inside every
   block from the run's order seed, every outcome sealed, arms coded by the blinding key.
4. The frozen blinded analysis, then unblinding with the released key, then the pre-registered cells.

Phases: ``theta0`` (model runs that fix the selection effect theta0_D, before anything else), ``null`` (no effect:
false-positive rates, TOST equivalence rate, interval coverage, null-stability flags), ``positive`` (an effect of
exactly the replicate's frozen margin planted in the decision arm, or in every observed arm, for H_decision and
H_observer separately: power, attribution, non-equivalence) and ``half`` (margin / 2: the power curve).

Observations come from the surrogate's emission tables (noiseless means per action for a pool of true fields,
docs/physics.md readout windows) with Poisson shot noise and slow laser and contrast drift indexed by the run's
time slots. The decision process is a stand-in with the M5 learners' feedback structure (a threshold rule on the
previous step's referenced fluorescence that moves the MW frequency finely inside a dip and coarsely outside it);
the real M5 policy and the full sizes are the GPU command in docs/results/prereg-calibration.md.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from glowbox.analysis import prereg_analysis_001 as pa
from glowbox.analysis.arms import (
    ArmSpec,
    Episode,
    ProtocolDesign,
    StepContext,
    action_hash,
    empirical_sham,
    poisson_decoy,
    run_episode,
    run_protocol,
    unseal_outcomes,
)
from glowbox.analysis.field_fit import transition_lines_hz
from glowbox.analysis.prereg import (
    Lock,
    RunKeys,
    Sealer,
    arm_codes,
    block_schedule,
    module_files,
    prereg_record,
    unblind,
)
from glowbox.analysis.runlog import REPO_ROOT, RunLog, write_report
from glowbox.experiments.m1_odmr_rabi import load_config
from glowbox.learn.measurement import MW_PERP_DEFAULT, MeasurementSpec, measurement_means
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "prereg_001.yaml"
REPORT_NAME = "prereg-calibration"
PREREG_DOC = "docs/preregistration/PREREG-001-observer-decision.md"
DEVIATIONS = "docs/preregistration/deviations.md"

CAL_DESIGN_ARMS = (
    ArmSpec("C1", "replay", observed=False, source="offline", lag=0),
    ArmSpec("C2", "replay", observed=False, source="offline", lag=1),
    ArmSpec("C3", "replay", observed=False, source="offline", lag=1),
    ArmSpec("C3b", "replay", observed=False, source="offline", lag=1),
    ArmSpec("C3L", "replay", observed=False, source="offline", lag=2),
    ArmSpec("C4", "sham", observed=False),
    ArmSpec("C5", "replay", observed=False, source="offline", lag=1, variant="detuned"),
)
CONF_DESIGN_ARMS = (
    ArmSpec("A1", "decision", observed=True),
    ArmSpec("A2", "replay", observed=True, lag=1),
    ArmSpec("A3", "replay", observed=False, lag=1),
    ArmSpec("A3b", "replay", observed=False, lag=1),
    ArmSpec("A3L", "replay", observed=False, lag=2),
    ArmSpec("A4", "sham", observed=True),
    ArmSpec("A5", "replay", observed=True, lag=1, variant="detuned"),
)
OBSERVED_ARMS = tuple(a.name for a in CONF_DESIGN_ARMS if a.observed)
CAL_WARMUP = 2
CONF_WARMUP = 2


# --- The simulated device ---------------------------------------------------------------------------------------------


@dataclass
class Pool:
    """Noiseless per-shot means of every action for a pool of true fields, resonant (``nom``) and detuned (``det``,
    every MW frequency + the offset: no spin response), and what the frozen model derives from them."""

    nom: np.ndarray  # (P, A, W)
    det: np.ndarray  # (P, A, W)
    r_model_nom: np.ndarray  # (P, A)
    r_model_det: np.ndarray  # (P, A)
    off_nom: np.ndarray  # (P, A) bool
    off_det: np.ndarray  # (P, A) bool
    lines_hz: np.ndarray  # (P, 8)
    freqs_hz: np.ndarray  # (A,)
    fields_t: np.ndarray  # (P, 3)
    baseline_r: float
    sig: int
    ref: int


def build_pool(cfg: dict[str, Any], seed: int) -> Pool:
    spec = MeasurementSpec(**cfg["measurement"])
    env = EnvParams()
    n = cfg["numerics"]
    opt = ModelOptions(
        n_detuning=int(n["n_detuning"]),
        detuning_rule=str(n["detuning_rule"]),
        detuning_spacing=float(n["detuning_spacing"]),
    )
    rng = np.random.default_rng(seed)
    p = int(cfg["field_pool"])
    d = rng.normal(size=(p, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    mag = rng.uniform(cfg["prior"]["b_min_t"], cfg["prior"]["b_max_t"], size=(p, 1))
    fields = d * mag
    off = float(cfg["detuned_offset_hz"])
    nom = measurement_means(spec, env, fields, opt, "cpu").numpy()
    det = measurement_means(spec.detuned(off), env, fields, opt, "cpu").numpy()
    s, r = int(cfg["signal_window"]), int(cfg["reference_window"])
    lines = np.stack([transition_lines_hz(f, env.d_hz, env.e_hz).ravel() for f in fields])
    freqs = np.tile(spec.freqs_hz(), spec.n_dur)
    width = env.rabi_full_scale_hz * spec.amplitude * MW_PERP_DEFAULT + 1.0 / (math.pi * env.t2_star_s)
    lim = float(cfg["off_line_widths"]) * width

    def far(f: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.min(np.abs(f[None, :, None] - lines[:, None, :]), axis=2) > lim
        return out

    return Pool(
        nom,
        det,
        nom[..., s] / nom[..., r],
        det[..., s] / det[..., r],
        far(freqs),
        far(freqs + off),
        lines,
        freqs,
        fields,
        float(np.median(det[..., s] / det[..., r])),
        s,
        r,
    )


def ou(rng: np.random.Generator, batch: int, n: int, sigma: float, tau: float) -> np.ndarray:
    """(batch, n) stationary Ornstein-Uhlenbeck series with sd ``sigma`` and correlation time ``tau`` samples."""
    rho = math.exp(-1.0 / tau) if tau > 0 else 0.0
    x = np.empty((batch, n))
    x[:, 0] = rng.normal(0.0, sigma, batch)
    e = rng.normal(0.0, sigma * math.sqrt(1 - rho * rho), (batch, n))
    for i in range(1, n):
        x[:, i] = rho * x[:, i - 1] + e[:, i]
    return x


class TableMeasure:
    """The simulated device: Poisson counts of S shots per step from the pool means of each element's field, with
    the slow drifts of its time slot. ``plant`` (arm -> (B,) size) adds size x lambda_ref to the signal window of
    that arm, i.e. shifts its expected referenced fluorescence by the size (positive controls only)."""

    def __init__(
        self,
        pool: Pool,
        fidx: np.ndarray,
        shots: int,
        rng: np.random.Generator,
        laser: np.ndarray | None,
        contrast: np.ndarray | None,
        jitter: float,
        plant: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.pool, self.fidx, self.shots, self.rng = pool, fidx, shots, rng
        self.laser, self.contrast, self.jitter = laser, contrast, jitter
        self.plant = plant or {}

    def __call__(self, actions: np.ndarray, ctx: StepContext) -> np.ndarray:
        a = np.asarray(actions, dtype=np.int64)
        b = np.arange(a.size)
        det = self.pool.det[self.fidx, a]
        if ctx.variant == "detuned":
            lam = det.copy()
        else:
            c = 1.0 + (self.contrast[b, ctx.slot] if self.contrast is not None else 0.0)
            lam = det + np.asarray(c).reshape(-1, 1) * (self.pool.nom[self.fidx, a] - det)
        g = np.asarray(1.0 + (self.laser[b, ctx.slot] if self.laser is not None else 0.0), dtype=np.float64)
        if self.jitter:
            g = g * (1.0 + self.jitter * self.rng.standard_normal(a.size))
        lam = lam * np.asarray(g).reshape(-1, 1)
        if ctx.arm in self.plant:
            lam[:, self.pool.sig] += self.plant[ctx.arm] * lam[:, self.pool.ref]
        return self.rng.poisson(lam * self.shots).astype(np.int32)


class ThresholdController:
    """Stand-in decision process: the next frequency index moves by ``fine`` after a step whose referenced
    fluorescence was below ``threshold`` (inside a dip) and by ``coarse`` otherwise (wrapping around the grid)."""

    def __init__(self, n_freq: int, threshold: float, fine: int, coarse: int, sig: int, ref: int) -> None:
        self.n_freq, self.threshold, self.fine, self.coarse = n_freq, threshold, fine, coarse
        self.sig, self.ref = sig, ref
        self.f = np.zeros(1, dtype=np.int64)

    def reset(self, batch: int) -> None:
        self.f = np.zeros(batch, dtype=np.int64)

    def propose(self, step: int) -> np.ndarray:
        return self.f.copy()

    def observe(self, step: int, actions: np.ndarray, outcomes: np.ndarray) -> None:
        y = np.asarray(outcomes, dtype=np.float64)
        r = y[:, self.sig] / np.maximum(y[:, self.ref], 1.0)
        move = np.where(r < self.threshold, self.fine, self.coarse)
        self.f = (np.asarray(actions, dtype=np.int64) % self.n_freq + move) % self.n_freq


# --- One batch of replicates ------------------------------------------------------------------------------------------


@dataclass
class ArmData:
    actions: np.ndarray  # (n_blocks, T, B) int16
    r: np.ndarray  # (n_blocks, B, T) float32
    fallback: np.ndarray  # (n_blocks,) bool


def run_design(
    design: ProtocolDesign,
    n_blocks: int,
    schedules: np.ndarray,
    ctrl: ThresholdController,
    measure: TableMeasure,
    seal_key: bytes,
    decoy_rng: np.random.Generator,
    sham_rng: np.random.Generator,
    sham_source: np.ndarray,
    pool: Pool,
    offline: np.ndarray | None = None,
) -> dict[str, ArmData]:
    batch = schedules.shape[0]
    t_n = design.n_steps
    data = {
        a: ArmData(
            np.zeros((n_blocks, t_n, batch), np.int16),
            np.zeros((n_blocks, batch, t_n), np.float32),
            np.zeros(n_blocks, bool),
        )
        for a in design.names
    }

    def keep(ep: Episode) -> None:
        y = unseal_outcomes(ep, seal_key).astype(np.float64)  # (T, B, W)
        d = data[ep.arm]
        d.actions[ep.block] = ep.actions
        d.r[ep.block] = (y[..., pool.sig] / np.maximum(y[..., pool.ref], 1.0)).T
        d.fallback[ep.block] = ep.fallback

    mean_counts = float(pool.nom[..., pool.sig].mean() * measure.shots)
    run_protocol(
        design,
        schedules,
        ctrl,
        measure,
        Sealer(seal_key),
        poisson_decoy(decoy_rng, mean_counts),
        empirical_sham(sham_rng, sham_source),
        offline,
        on_episode=keep,
    )
    return data


def episode_table(
    data: dict[str, ArmData],
    design: ProtocolDesign,
    b: int,
    schedule: np.ndarray,
    codes: dict[str, str],
    pool: Pool,
    fidx: int,
    warmup: int,
    n_valid: int,
) -> pa.EpisodeTable:
    """Replicate b's episodes (warm-up blocks and blocks beyond warm-up + n_valid left out / flagged)."""
    rows: dict[str, list[Any]] = {k: [] for k in ("block", "pos", "code", "hash", "r", "rm", "off", "task")}
    n_blocks = warmup + n_valid
    lines = pool.lines_hz[fidx]
    for arm in design.arms:
        d = data[arm.name]
        detuned = arm.variant == "detuned"
        rm_tab = (pool.r_model_det if detuned else pool.r_model_nom)[fidx]
        off_tab = (pool.off_det if detuned else pool.off_nom)[fidx]
        for k in range(n_blocks):
            a = d.actions[k, :, b].astype(np.int64)
            r = d.r[k, b].astype(np.float64)
            f_est = pool.freqs_hz[a[int(np.argmin(r))]]
            rows["block"].append(k)
            rows["pos"].append(int(np.argmax(schedule[k] == arm.name)))
            rows["code"].append(codes[arm.name])
            rows["hash"].append(action_hash(a))
            rows["r"].append(r)
            rows["rm"].append(rm_tab[a])
            rows["off"].append(off_tab[a])
            rows["task"].append(float(np.min(np.abs(lines - f_est))) / 1e6)
    invalid = {k: "warm-up" for k in range(warmup)}
    return pa.EpisodeTable(
        np.array(rows["block"]),
        np.array(rows["pos"]),
        np.array(rows["code"], dtype=object),
        np.array(rows["hash"], dtype=object),
        np.stack(rows["r"]),
        np.stack(rows["rm"]),
        np.stack(rows["off"]),
        np.array(rows["task"]),
        invalid,
    )


def _analyse_rep(job: dict[str, Any]) -> dict[str, Any]:
    """Worker: the frozen analysis of one replicate (calibration, blinded analysis, unblinding)."""
    cal = job["cal"]
    ba = pa.analyse(job["table"], cal, n_boot=job["n_boot"])
    mapping = unblind(job["blind_key"], job["blind_commit"], pa.CONFIRMATORY_ARMS)
    rep = pa.unblind(ba, mapping, cal)
    blinded_text = repr(ba.to_dict())
    out: dict[str, Any] = {
        "names_hidden": not any(f"'{a}'" in blinded_text for a in pa.CONFIRMATORY_ARMS),
        "stable": rep.stability["stable"],
        "margin0": cal.margin[0],
        "margin1": cal.margin[1],
        "c_ref": cal.c_ref,
        "n_blocks": cal.n_blocks,
    }
    for h, r in rep.primary.items():
        out[f"{h}_p"] = r["p"]
        out[f"{h}_rej"] = bool(r["holm_rejected"])
        out[f"{h}_raw_rej"] = bool(r["p"] <= pa.ALPHA)
        out[f"{h}_eq"] = bool(r["equivalent"])
        out[f"{h}_d"] = r.get("d", float("nan"))
        out[f"{h}_cover"] = bool(r["ci95"][0] <= 0.0 <= r["ci95"][1]) if not r.get("missing") else False
        out[f"{h}_nw_rej"] = bool(r["nw_p"] <= pa.ALPHA)
        out[f"{h}_conclusion"] = r["conclusion"]
    for s, r in rep.secondary.items():
        out[f"{s}_rej"] = bool(r["holm_rejected"])
    return out


def simulate_batch(
    cfg: dict[str, Any],
    pool: Pool,
    rep_ids: np.ndarray,
    phase: str,
    plant: str | None,
    plant_frac: float,
    theta0: tuple[float, float],
    workers: int,
) -> list[dict[str, Any]]:
    """Run len(rep_ids) replicates in lock-step (vectorised simulation, one schedule and key set per replicate)."""
    batch = rep_ids.size
    seed = int(cfg["seed"])
    tag = hashlib.sha256(f"{seed}:{phase}:{plant}:{plant_frac}:{int(rep_ids[0])}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(tag[:8], "big"))
    keys = [RunKeys.from_seed(f"{seed}:{phase}:{plant}:{plant_frac}:{int(r)}") for r in rep_ids]
    seal_key = hashlib.sha256(b"batch-seal" + tag).digest()  # one sealing key per batch (see the report)
    fidx = rep_ids % pool.nom.shape[0]
    pr, nu, cc = cfg["protocol"], cfg["nuisance"], cfg["controller"]
    t_n, shots = int(pr["n_steps"]), int(pr["shots"])
    n_freq = pool.freqs_hz.size
    ctrl = ThresholdController(
        n_freq,
        pool.baseline_r - float(cc["threshold_below_baseline"]),
        int(cc["fine_step"]),
        int(cc["coarse_step"]),
        pool.sig,
        pool.ref,
    )

    def device(n_slots: int, plant_sizes: dict[str, np.ndarray] | None = None) -> TableMeasure:
        laser = ou(rng, batch, n_slots, float(nu["laser_rel_sigma"]), float(nu["laser_tau_slots"]))
        contrast = ou(rng, batch, n_slots, float(nu["contrast_rel_sigma"]), float(nu["contrast_tau_slots"]))
        return TableMeasure(pool, fidx, shots, rng, laser, contrast, float(nu["laser_jitter_rel"]), plant_sizes)

    m_cal = pa.M_CAL
    # 1. offline sequences: the decision process on a separate simulated device (the same nuisance model, its own
    # draws), before anything else
    n_off = m_cal + CAL_WARMUP
    off_dev = device(n_off)
    offline = []
    sealer = Sealer(seal_key)
    for k in range(n_off):
        ep = run_episode(ArmSpec("offline", "decision"), k, ctrl, off_dev, t_n, batch, sealer, np.full(batch, k))
        offline.append(ep.actions)
    off_actions = np.stack(offline)  # (M + warm-up, T, B)
    decoy_rng = np.random.default_rng(rng.integers(1 << 62))
    sham_rng = np.random.default_rng(rng.integers(1 << 62))

    conf_design = ProtocolDesign(CONF_DESIGN_ARMS, t_n)
    if phase == "theta0":
        n_conf = pa.N_MIN
        sch = np.stack([block_schedule(k.order, conf_design.names, n_conf + CONF_WARMUP) for k in keys])
        dev = device(sch.shape[1] * sch.shape[2])
        data = run_design(
            conf_design, n_conf + CONF_WARMUP, sch, ctrl, dev, seal_key, decoy_rng, sham_rng, off_actions, pool
        )
        dd = data["A1"].r[CONF_WARMUP:-1].mean(axis=2) - data["A2"].r[CONF_WARMUP + 1 :].mean(axis=2)
        do = data["A2"].r[CONF_WARMUP:].mean(axis=2) - data["A3"].r[CONF_WARMUP:].mean(axis=2)
        return [{"theta_d": float(dd[:, i].mean()), "theta_o": float(do[:, i].mean())} for i in range(batch)]

    # 2. calibration block
    cal_design = ProtocolDesign(CAL_DESIGN_ARMS, t_n)
    n_cal = m_cal + CAL_WARMUP
    cal_sch = np.stack(
        [block_schedule(hashlib.sha256(k.order + b"calibration").digest(), cal_design.names, n_cal) for k in keys]
    )
    cal_data = run_design(
        cal_design,
        n_cal,
        cal_sch,
        ctrl,
        device(n_cal * len(cal_design.names)),
        seal_key,
        decoy_rng,
        sham_rng,
        off_actions,
        pool,
        offline=off_actions,
    )
    cal_tables = [
        episode_table(
            cal_data, cal_design, i, cal_sch[i], {a: a for a in cal_design.names}, pool, int(fidx[i]), CAL_WARMUP, m_cal
        )
        for i in range(batch)
    ]
    with mp.get_context("fork").Pool(workers) as wp:
        cals = wp.starmap(pa.derive_calibration, [(t, *theta0) for t in cal_tables])
    del cal_data, cal_tables

    # 3. confirmatory run with the frozen N (and the frozen margins for the planted size)
    n_conf = max(c.n_blocks for c in cals)
    sch = np.stack([block_schedule(k.order, conf_design.names, n_conf + CONF_WARMUP) for k in keys])
    sizes: dict[str, np.ndarray] | None = None
    if plant == "decision":
        sizes = {"A1": plant_frac * np.array([c.margin[1] for c in cals])}
    elif plant == "observer":
        sizes = {a: plant_frac * np.array([c.margin[0] for c in cals]) for a in OBSERVED_ARMS}
    dev = device(sch.shape[1] * sch.shape[2], sizes)
    data = run_design(
        conf_design, n_conf + CONF_WARMUP, sch, ctrl, dev, seal_key, decoy_rng, sham_rng, off_actions, pool
    )
    jobs = []
    for i, kk in enumerate(keys):
        codes = arm_codes(kk.blind, conf_design.names)
        table = episode_table(data, conf_design, i, sch[i], codes, pool, int(fidx[i]), CONF_WARMUP, cals[i].n_blocks)
        jobs.append(
            {
                "cal": cals[i],
                "table": table,
                "n_boot": int(cfg["replicates"]["n_boot"]),
                "blind_key": kk.blind,
                "blind_commit": kk.commitments()["blind_sha256"],
            }
        )
    with mp.get_context("fork").Pool(workers) as wp:
        res = wp.map(_analyse_rep, jobs)
    for r in res:
        r["identified"] = (
            (r["H_decision_rej"] and not r["H_observer_rej"])
            if plant == "decision"
            else (r["H_observer_rej"] and not r["H_decision_rej"])
            if plant == "observer"
            else False
        )
    return res


# --- Driver -----------------------------------------------------------------------------------------------------------


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def rate(res: list[dict[str, Any]], key: str) -> tuple[float, float, float, int]:
    v = [bool(r[key]) for r in res]
    k, n = sum(v), len(v)
    lo, hi = clopper_pearson(k, n)
    return (k / n if n else float("nan")), lo, hi, n


def validation_lock(run_dir: Path) -> tuple[Lock, Path]:
    """A lock over the plan, the analysis script and everything this script reads (stage S validation: the same
    machinery as a confirmatory lock, written into the run directory)."""
    files = [
        REPO_ROOT / PREREG_DOC,
        *module_files("glowbox.experiments.prereg_calibration"),
        DEFAULT_CONFIG,
    ]
    lock = Lock.create(pa.PLAN, files, {"analysis": pa.frozen_constants()}, REPO_ROOT / DEVIATIONS)
    path = lock.write(run_dir / "validation.lock.yaml")
    pa.verify_lock(lock)
    return lock, path


def run_phase(
    cfg: dict[str, Any],
    pool: Pool,
    phase: str,
    n: int,
    plant: str | None = None,
    frac: float = 0.0,
    theta0: tuple[float, float] = (0.0, 0.0),
    start: int = 0,
) -> list[dict[str, Any]]:
    rc = cfg["replicates"]
    out: list[dict[str, Any]] = []
    for s in range(0, n, int(rc["batch"])):
        ids = np.arange(start + s, start + min(n, s + int(rc["batch"])))
        t0 = time.perf_counter()
        out += simulate_batch(cfg, pool, ids, phase, plant, frac, theta0, int(rc["workers"]))
        print(f"  {phase} {plant or ''} {frac:g}: {len(out)}/{n} ({time.perf_counter() - t0:.0f} s)", flush=True)
    return out


def fmt_rate(x: tuple[float, float, float, int]) -> str:
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}] (n = {x[3]})"


def summary_markdown(
    cfg: dict[str, Any],
    pool: Pool,
    theta0: tuple[float, float],
    theta_o: tuple[float, float],
    null: list[dict[str, Any]],
    pos: dict[str, list[dict[str, Any]]],
    seconds: float,
) -> tuple[str, dict[str, float]]:
    rc = cfg["replicates"]
    m0 = np.array([r["margin0"] for r in null])
    m1 = np.array([r["margin1"] for r in null])
    cref = np.array([r["c_ref"] for r in null])
    nb = sorted({int(r["n_blocks"]) for r in null})
    any_rej = [{"x": r["H_decision_rej"] or r["H_observer_rej"]} for r in null]
    sec_names = [k for k in pa.SECONDARY if k != "S4_quantum_test"]
    sec_any = [{"x": any(r[f"{s}_rej"] for s in sec_names)} for r in null]
    lines = [
        "Stage-S validation of the PREREG-001 pipeline (docs/preregistration/PREREG-001-observer-decision.md). The "
        "simulator has no decision or observer effect by construction: these numbers say whether the frozen procedure "
        "keeps its false-positive rate and has its planned power, nothing about the hypotheses. Status of the plan: "
        "DRAFT (not signed off, not locked).",
        "",
        "Every replicate is a complete simulated experiment: offline sequences, a calibration block of "
        f"{pa.M_CAL} valid blocks (seven sealed control arms, slot for slot like a confirmatory block), the frozen "
        "calibration (empirical null, margins, N), "
        f"a confirmatory run of N blocks x 7 arms in a random order per block, sealing of every outcome, arm codes, "
        f"the frozen blinded analysis ({rc['n_boot']} null draws per test), unblinding, the pre-registered cells. "
        f"T = {cfg['protocol']['n_steps']} steps x S = {cfg['protocol']['shots']} shots per episode; "
        f"{pool.nom.shape[0]} true fields cycled over replicates; slow laser and contrast drift (OU over episode "
        "slots), per-step laser jitter. Decision process: the stand-in threshold rule (see the script).",
        "",
        "## Frozen quantities (median over the null replicates, each replicate freezes its own)",
        "",
        f"- Margin (smallest effect size of interest, 0.95 quantile of |control-vs-control pair statistic|): "
        f"lag 0 (H_observer) {np.median(m0):.4g} [IQR {np.quantile(m0, 0.25):.4g}-{np.quantile(m0, 0.75):.4g}], "
        f"lag 1 (H_decision) {np.median(m1):.4g} [{np.quantile(m1, 0.25):.4g}-{np.quantile(m1, 0.75):.4g}] in "
        "units of the referenced fluorescence r = N_sig / N_ref.",
        f"- Spin signal of the replayed sequences, C_ref = mean(r_detuned - r_resonant): {np.median(cref):.4g}; "
        f"margins / C_ref: {np.median(m0 / cref):.3f} (lag 0), {np.median(m1 / cref):.3f} (lag 1).",
        f"- N (valid confirmatory blocks): {nb} (the floor N_MIN = {pa.N_MIN}; the power rule asks for fewer).",
        f"- theta0_D (selection effect of the decision arm under drift, from {rc['theta0']} model runs): "
        f"{theta0[0]:.3g} +- {theta0[1]:.2g} (MC SE; the plan requires SE <= margin / 10 = "
        f"{np.median(m1) / 10:.3g}). The observer contrast of the same runs: {theta_o[0]:.3g} +- {theta_o[1]:.2g} "
        "(expected 0).",
        "",
        "## Null calibration (no effect)",
        "",
        "| quantity | H_decision | H_observer |",
        "| --- | --- | --- |",
        f"| rejected after Holm (FWER 0.05) | {fmt_rate(rate(null, 'H_decision_rej'))} | "
        f"{fmt_rate(rate(null, 'H_observer_rej'))} |",
        f"| p <= 0.05 without correction | {fmt_rate(rate(null, 'H_decision_raw_rej'))} | "
        f"{fmt_rate(rate(null, 'H_observer_raw_rej'))} |",
        f"| TOST equivalent at the frozen margin | {fmt_rate(rate(null, 'H_decision_eq'))} | "
        f"{fmt_rate(rate(null, 'H_observer_eq'))} |",
        f"| 95 % interval covers 0 | {fmt_rate(rate(null, 'H_decision_cover'))} | "
        f"{fmt_rate(rate(null, 'H_observer_cover'))} |",
        f"| Newey-West check, p <= 0.05 | {fmt_rate(rate(null, 'H_decision_nw_rej'))} | "
        f"{fmt_rate(rate(null, 'H_observer_nw_rej'))} |",
        "",
        f"- Primary family, any Holm rejection: {fmt_rate(rate(any_rej, 'x'))}.",
        f"- Secondary family without the quantum test (S1-S3), any Holm rejection: {fmt_rate(rate(sec_any, 'x'))}; "
        "the quantum test S4 (detuned vs resonant task error) rejects in "
        f"{fmt_rate(rate(null, 'S4_quantum_test_rej'))}.",
        f"- Null-stability check flags the run (confirmatory controls differ from the calibration controls): "
        f"{fmt_rate(rate([{'x': not r['stable']} for r in null], 'x'))} (designed rate <= 0.05).",
        f"- Arm names absent from every blinded output: {all(r['names_hidden'] for r in null)}.",
        "",
        "## Positive control (effect planted in the simulated device)",
        "",
        "Decision: E[r] of the decision arm A1 shifted by the size. Observer: E[r] of every observed arm "
        "(A1, A2, A4, A5) shifted, the sealed arms not. Size = the replicate's own frozen margin (or half of it).",
        "",
        "| planted | size | power (Holm rejection of the planted hypothesis) "
        "| attributed (planted rejected, other not) | not equivalent |",
        "| --- | --- | --- | --- | --- |",
    ]
    metrics: dict[str, float] = {
        "fpr_decision": rate(null, "H_decision_rej")[0],
        "fpr_observer": rate(null, "H_observer_rej")[0],
        "fwer_primary": rate(any_rej, "x")[0],
        "margin_lag0_median": float(np.median(m0)),
        "margin_lag1_median": float(np.median(m1)),
        "theta0_d": theta0[0],
        "theta0_d_se": theta0[1],
    }
    for key, res in pos.items():
        plant, frac = key.split(":")
        h = "H_decision" if plant == "decision" else "H_observer"
        not_eq = [{"x": not r[f"{h}_eq"]} for r in res]
        lines.append(
            f"| {plant} | {frac} x margin | {fmt_rate(rate(res, f'{h}_rej'))} | {fmt_rate(rate(res, 'identified'))} "
            f"| {fmt_rate(rate(not_eq, 'x'))} |"
        )
        metrics[f"power_{plant}_{frac}"] = rate(res, f"{h}_rej")[0]
    lines += [
        "",
        "## Reading",
        "",
        "- Pass criteria of the plan: false-positive rate per hypothesis within the binomial interval of 0.05 "
        "(Holm makes it at most 0.05); TOST equivalence >= 0.90 under the null; power >= 0.90 and non-equivalence "
        ">= 0.90 at the margin; correct attribution. The margin / 2 rows are the power curve only.",
        "- The calibration block mirrors the confirmatory block slot for slot. A first version of this validation used "
        "a 4-arm calibration block: its paired controls were closer in time than the 7-arm confirmatory pairs, saw "
        "less drift, and the test was liberal (H_decision p <= 0.05 in 0.070 [0.055, 0.088] of 1000 null replicates, "
        "95 % coverage 0.930). The control runs must share the confirmatory schedule, not only its arms.",
        "- One sealing key per batch of replicates instead of one per replicate (speed); every outcome is still "
        "sealed and unsealed through the same code. Blinding keys and order seeds are per replicate.",
        "- The stand-in decision process is not the M5 policy, and theta0_D depends on the policy: the full-size run "
        "with the real policy is the command below (GPU machine), and its theta0_D is the one to freeze.",
        "",
        "Full size (R = 2000 null replicates; the null phase takes about twice as long as here):",
        "",
        "    uv run python -m glowbox.experiments.prereg_calibration --replicates 2000",
        "",
        "With the real M5 policy the decision process is a Controller (glowbox.analysis.arms) wrapping the M5 "
        "policy over its posterior; that adapter belongs to M7 and is not part of this validation.",
        "",
        f"Wall time here: {seconds / 60:.1f} min on a CPU.",
    ]
    return "\n".join(lines), metrics


def figure(null: list[dict[str, Any]], pos: dict[str, list[dict[str, Any]]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for h, c in (("H_decision", "C0"), ("H_observer", "C1")):
        p = np.array([r[f"{h}_p"] for r in null])
        axes[0].hist(p, bins=20, range=(0, 1), histtype="step", color=c, label=h, density=True)
    axes[0].axhline(1.0, color="0.5", lw=0.8, ls="--")
    axes[0].set(xlabel="empirical-null p (no effect)", ylabel="density", title="Null calibration: p-values")
    axes[0].legend(frameon=False)
    for plant, c in (("decision", "C0"), ("observer", "C1")):
        h = "H_decision" if plant == "decision" else "H_observer"
        xs, ys, lo, hi = [0.0], [rate(null, f"{h}_rej")[0]], [], []
        lo.append(rate(null, f"{h}_rej")[1])
        hi.append(rate(null, f"{h}_rej")[2])
        for key, res in sorted(pos.items()):
            if key.startswith(plant):
                xs.append(float(key.split(":")[1]))
                rr = rate(res, f"{h}_rej")
                ys.append(rr[0])
                lo.append(rr[1])
                hi.append(rr[2])
        o = np.argsort(xs)
        xa, ya, la, ha = (np.array(v)[o] for v in (xs, ys, lo, hi))
        axes[1].errorbar(xa, ya, yerr=[ya - la, ha - ya], marker="o", color=c, label=h, capsize=3)
    axes[1].axhline(0.9, color="0.5", lw=0.8, ls="--")
    axes[1].axhline(0.05, color="0.5", lw=0.8, ls=":")
    axes[1].set(xlabel="planted effect / frozen margin", ylabel="Holm rejection rate", title="Power", ylim=(0, 1.02))
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replicates", type=int, default=None, help="null replicates (default from the config)")
    ap.add_argument("--report-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.quick)
    if args.replicates:
        cfg["replicates"]["null_runs"] = args.replicates
    torch.set_num_threads(8)
    t_start = time.perf_counter()
    with RunLog(cfg["name"], config=cfg, seed=int(cfg["seed"])) as run:
        lock, lock_path = validation_lock(run.dir)
        keys = RunKeys.from_seed(f"{cfg['seed']}:run")
        run.annotate("prereg", prereg_record(lock, lock_path, keys, "simulation", {"positive_control": 1.0}))
        print("building the field pool ...", flush=True)
        pool = build_pool(cfg, int(cfg["seed"]))
        rc = cfg["replicates"]
        th = run_phase(cfg, pool, "theta0", int(rc["theta0"]))
        td = np.array([r["theta_d"] for r in th])
        to = np.array([r["theta_o"] for r in th])
        theta0 = (float(td.mean()), float(td.std(ddof=1) / math.sqrt(td.size)))
        theta_o = (float(to.mean()), float(to.std(ddof=1) / math.sqrt(to.size)))
        null = run_phase(cfg, pool, "null", int(rc["null_runs"]), theta0=theta0)
        pos: dict[str, list[dict[str, Any]]] = {}
        for plant in ("decision", "observer"):
            pos[f"{plant}:1"] = run_phase(cfg, pool, "positive", int(rc["positive"]), plant, 1.0, theta0)
            if int(rc["half"]):
                pos[f"{plant}:0.5"] = run_phase(cfg, pool, "positive", int(rc["half"]), plant, 0.5, theta0)
        seconds = time.perf_counter() - t_start
        md, metrics = summary_markdown(cfg, pool, theta0, theta_o, null, pos, seconds)
        run.log_metrics(0, **metrics)
        png = run.save_figure(figure(null, pos), "calibration")
        print(md)
    if args.report_dir is not None:
        path = write_report(
            run.dir,
            "PREREG-001 null calibration and positive control (stage S)",
            md,
            [png],
            report_dir=args.report_dir,
            name=REPORT_NAME,
        )
        print(f"report: {path}")


if __name__ == "__main__":
    main()
