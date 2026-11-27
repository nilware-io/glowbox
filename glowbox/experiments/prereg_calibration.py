"""PREREG-001, stage S: null calibration and positive control of the whole pre-registered pipeline.

    uv run python -m glowbox.experiments.prereg_calibration [--config PATH] [--quick] [--replicates N]
        [--report-dir DIR]

The simulator has no decision or observer effect by construction, so nothing here tests the hypotheses. Each
replicate is one complete simulated experiment, run exactly as the plan prescribes:

1. Model runs (phase ``model``, before anything else): confirmatory-design runs of each model of the theta0_D set
   (:func:`~glowbox.analysis.prereg_analysis_001.selection_model`): the predicted selection effect and the
   within-episode correlation of the sealed controls.
2. Offline action sequences: the decision process run on a separate simulated device (not the measurement under
   test); they are what the calibration controls replay, and their marginal is the sham distribution.
3. CALIBRATION block (M_CAL valid blocks, seven arms C1, C2, C3, C3b, C3L, C4, C5, slot for slot like a confirmatory
   block, every one a sealed replay with no decision applied and no live reader):
   :func:`~glowbox.analysis.prereg_analysis_001.derive_calibration` freezes the empirical null, the margins
   (smallest effect sizes of interest), the theta0_D set and the model check.
4. Confirmatory run (N_CONF valid blocks, arms A1-A5 with the sealed controls A3, A3b, A3L), randomised order inside
   every block from the run's order seed, every outcome sealed, records labelled by codes or random tokens, the
   arm-revealing fields sealed with the blinding key, decoys from the frozen model.
5. The locked reduction (:func:`~glowbox.analysis.prereg_analysis_001.build_episode_table`), the frozen blinded
   analysis, unblinding with the released key, the pre-registered cells and the complete decision rule.

Conditions (config ``conditions``): ``nominal`` (device = model: false-positive rates, power, power curve) and stress
conditions with no effect: within-episode drift that the model set covers (theta0_D clearly non-zero), the same
drift missing from the model set (the model check must refuse H_decision), and a confirmatory run whose drift is
larger and faster than the calibration block's (the concurrent nulls must hold the error rate). Phase
``engineering``: the M7 / E5 paired test at its fixed sizes.

Observations come from the surrogate's emission tables (noiseless means per action for a pool of true fields,
docs/physics.md readout windows) with Poisson shot noise and slow laser and contrast drift indexed by the run's
time slots. The decision process is a stand-in with the M5 learners' feedback structure (a threshold rule on the
previous step's referenced fluorescence that moves the MW frequency finely inside a dip and coarsely outside it).
Everything runs on the CPU; the documented full size is the same command with more replicates
(docs/results/prereg-calibration.md). The M5 policy as the decision process is M7's adapter (a Controller of
glowbox.analysis.arms) and needs its own model runs and validation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
import yaml
from scipy import stats

from glowbox.analysis import prereg_analysis_001 as pa
from glowbox.analysis.arms import (
    ArmSpec,
    Episode,
    ProtocolDesign,
    StepContext,
    empirical_sham,
    model_decoy,
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
#: Arms a model run needs (the H_decision cell and the model-check cell).
MODEL_ARMS = ("A1", "A2", "A3", "A3b")


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
    line_width_hz: float
    detuned_offset_hz: float
    sig: int = pa.SIGNAL_WINDOW
    ref: int = pa.REFERENCE_WINDOW


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
    s, r = pa.SIGNAL_WINDOW, pa.REFERENCE_WINDOW
    lines = np.stack([transition_lines_hz(f, env.d_hz, env.e_hz).ravel() for f in fields])
    freqs = np.tile(spec.freqs_hz(), spec.n_dur)
    width = env.rabi_full_scale_hz * spec.amplitude * MW_PERP_DEFAULT + 1.0 / (math.pi * env.t2_star_s)
    lim = pa.OFF_LINE_WIDTHS * width

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
        float(width),
        off,
    )


class PoolModel:
    """The frozen model of one simulated run for :func:`~glowbox.analysis.prereg_analysis_001.build_episode_table`
    (stage S: the pool's noiseless means of the run's field)."""

    def __init__(self, pool: Pool, f: int) -> None:
        self.pool, self.f = pool, f
        self.lines_hz = pool.lines_hz[f]
        self.line_width_hz = pool.line_width_hz

    def mw_hz(self, actions: np.ndarray) -> np.ndarray:
        out: np.ndarray = self.pool.freqs_hz[np.asarray(actions, dtype=np.int64)]
        return out

    def r_model(self, actions: np.ndarray, offset_hz: float) -> np.ndarray:
        if offset_hz == 0.0:
            tab = self.pool.r_model_nom[self.f]
        elif offset_hz == self.pool.detuned_offset_hz:
            tab = self.pool.r_model_det[self.f]
        else:
            raise ValueError(f"no model table for an MW offset of {offset_hz} Hz")
        out: np.ndarray = tab[np.asarray(actions, dtype=np.int64)]
        return out


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
    the slow drifts of its time slot and (``within_sigma`` > 0) a within-episode AR(1) drift of the spin contrast
    (sd ``within_sigma``, lag-one correlation ``within_rho`` per step, a fresh draw at every episode start).
    ``plant`` (arm -> (B,) size) adds size x lambda_ref to the signal window of that arm, i.e. shifts its expected
    referenced fluorescence by the size (positive controls only)."""

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
        within_sigma: float = 0.0,
        within_rho: float = 0.0,
    ) -> None:
        self.pool, self.fidx, self.shots, self.rng = pool, fidx, shots, rng
        self.laser, self.contrast, self.jitter = laser, contrast, jitter
        self.plant = plant or {}
        self.within_sigma, self.within_rho = within_sigma, within_rho
        self._w: np.ndarray | None = None

    def _within(self, n: int, step: int) -> np.ndarray | float:
        if self.within_sigma <= 0:
            return 0.0
        if step == 0 or self._w is None or self._w.size != n:
            self._w = self.rng.normal(0.0, self.within_sigma, n)
        else:
            rho = self.within_rho
            self._w = rho * self._w + math.sqrt(1 - rho * rho) * self.rng.normal(0.0, self.within_sigma, n)
        return self._w

    def __call__(self, actions: np.ndarray, ctx: StepContext) -> np.ndarray:
        a = np.asarray(actions, dtype=np.int64)
        b = np.arange(a.size)
        det = self.pool.det[self.fidx, a]
        w = self._within(a.size, ctx.step)
        if ctx.variant == "detuned":
            lam = det.copy()
        else:
            c = 1.0 + (self.contrast[b, ctx.slot] if self.contrast is not None else 0.0) + w
            lam = det + np.asarray(c).reshape(-1, 1) * (self.pool.nom[self.fidx, a] - det)
        g = np.asarray(1.0 + (self.laser[b, ctx.slot] if self.laser is not None else 0.0), dtype=np.float64)
        if self.jitter:
            g = g * (1.0 + self.jitter * self.rng.standard_normal(a.size))
        lam = lam * np.asarray(g).reshape(-1, 1)
        if ctx.arm in self.plant:
            lam[:, self.pool.sig] += self.plant[ctx.arm] * lam[:, self.pool.ref]
        return self.rng.poisson(np.maximum(lam, 0.0) * self.shots).astype(np.int32)


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
    counts: np.ndarray  # (n_blocks, T, B, W) int32, unsealed after the run


def run_design(
    design: ProtocolDesign,
    n_blocks: int,
    schedules: np.ndarray,
    ctrl: ThresholdController,
    measure: TableMeasure,
    seal_key: bytes,
    blind_key: bytes,
    decoy_rng: np.random.Generator,
    sham_rng: np.random.Generator,
    sham_source: np.ndarray,
    pool: Pool,
    offline: np.ndarray | None = None,
    arms: tuple[str, ...] | None = None,
) -> dict[str, ArmData]:
    """Run a design for a batch of replicates and unseal the outcomes after the run. This harness is the acquisition
    side (one process holds every key); the analysis side sees only :func:`replicate_table`'s coded tables."""
    batch = schedules.shape[0]
    t_n = design.n_steps
    w = pool.nom.shape[-1]
    keep_arms = arms or tuple(design.names)
    data = {
        a: ArmData(np.zeros((n_blocks, t_n, batch), np.int16), np.zeros((n_blocks, t_n, batch, w), np.int32))
        for a in keep_arms
    }

    def keep(ep: Episode) -> None:
        # the harness holds every key (simulation): each episode is unsealed as it is handed over
        if ep.arm in data:
            data[ep.arm].actions[ep.block] = ep.actions
            data[ep.arm].counts[ep.block] = unseal_outcomes(ep, seal_key)

    def expected(actions: np.ndarray, ctx: StepContext) -> np.ndarray:
        means = pool.det if ctx.variant == "detuned" else pool.nom
        out: np.ndarray = means[measure.fidx, np.asarray(actions, dtype=np.int64)] * measure.shots
        return out

    run_protocol(
        design,
        schedules,
        ctrl,
        measure,
        Sealer(seal_key),
        model_decoy(decoy_rng, expected),
        empirical_sham(sham_rng, sham_source),
        offline,
        on_episode=keep,
        blind_sealer=Sealer(blind_key),
    )
    return data


def replicate_table(
    data: dict[str, ArmData],
    i: int,
    schedule: np.ndarray,
    codes: dict[str, str],
    pool: Pool,
    fidx: int,
    n_blocks: int,
    variants: dict[str, str],
) -> pa.EpisodeTable:
    """Replicate i's episodes through the locked reduction (blocks beyond ``n_blocks`` were not collected)."""
    code, block, pos, acts, offs, cnts = [], [], [], [], [], []
    for arm, d in data.items():
        code += [codes[arm]] * n_blocks
        block.append(np.arange(n_blocks))
        pos.append(np.argmax(schedule[:n_blocks] == arm, axis=1))
        acts.append(d.actions[:n_blocks, :, i].astype(np.int64))
        offs.append(np.full(n_blocks, pool.detuned_offset_hz if variants[arm] == "detuned" else 0.0))
        cnts.append(d.counts[:n_blocks, :, i, :])
    return pa.build_episode_table(
        np.array(code, dtype=object),
        np.concatenate(block),
        np.concatenate(pos),
        np.concatenate(acts),
        np.concatenate(offs),
        np.concatenate(cnts),
        PoolModel(pool, fidx),
        pa.WARMUP_BLOCKS,
    )


def _analyse_rep(job: dict[str, Any]) -> dict[str, Any]:
    """Worker: the frozen analysis of one replicate (blinded analysis, unblinding, the complete decision rule)."""
    cal = job["cal"]
    ba = pa.analyse(job["table"], cal, n_boot=job["n_boot"])
    mapping = unblind(job["blind_key"], job["blind_commit"], pa.CONFIRMATORY_ARMS)
    rep = pa.unblind(ba, mapping, cal)
    blinded_text = repr(ba.to_dict())
    out: dict[str, Any] = {
        "names_hidden": not any(f"'{a}'" in blinded_text for a in pa.CONFIRMATORY_ARMS),
        "stable": rep.stability["stable"],
        "model_ok_cal": rep.model_check["calibration"],
        "model_ok_conf": rep.model_check["confirmatory"],
        "model_ok": rep.model_check["passed"],
        "consistent": ",".join(rep.model_check["consistent"]),
        "margin0": cal.margin[0],
        "margin1": cal.margin[1],
        "sd_mean0": cal.sd_mean[0],
        "sd_mean1": cal.sd_mean[1],
        "c_ref": cal.c_ref,
        "block_len0": cal.block_len.get(0, float("nan")),
    }
    for h, r in rep.primary.items():
        out[f"{h}_p"] = r["p"]
        out[f"{h}_p_conc"] = r.get("p_concurrent", float("nan"))
        out[f"{h}_rej"] = bool(r.get("rejected", False))
        out[f"{h}_rej_cal"] = bool(r.get("holm_rejected", False))
        out[f"{h}_raw_rej"] = bool(r["p"] <= pa.ALPHA)
        out[f"{h}_eq"] = bool(r.get("equivalent", False))
        out[f"{h}_null_claim"] = str(r["conclusion"]).startswith("null")
        out[f"{h}_d"] = r.get("d", float("nan"))
        # an interval exists only when the hypothesis is not inconclusive by the model check (H_decision: a
        # consistent member is left); coverage is over the runs that report one
        out[f"{h}_has_ci"] = not r.get("missing")
        out[f"{h}_cover"] = bool(r["ci95"][0] <= 0.0 <= r["ci95"][1]) if not r.get("missing") else False
        out[f"{h}_nw_rej"] = bool(r.get("nw_p", 1.0) <= pa.ALPHA)
        out[f"{h}_conclusion"] = r["conclusion"]
    for s, r in rep.secondary.items():
        out[f"{s}_rej"] = bool(r["rejected"])
        out[f"{s}_rej_cal"] = bool(r["holm_rejected"])
    for s, r in rep.manipulation.items():
        out[f"{s}_passed"] = bool(r["passed"])
    return out


#: Keys of a model's config entry that are not nuisance parameters: ``decision: offline`` makes the model's runs
#: replay offline sequences in the decision arm's slot (a model in which the decision process cannot select on the
#: drift: theta0_D = 0 whatever the drift).
MODEL_KEYS = ("decision",)


def nuisance(cfg: dict[str, Any], *overrides: dict[str, Any] | None) -> dict[str, float]:
    nu = {"within_sigma": 0.0, "within_rho": 0.0, **{k: float(v) for k, v in cfg["nuisance"].items()}}
    for o in overrides:
        nu.update({k: float(v) for k, v in (o or {}).items() if k not in MODEL_KEYS})
    return nu


#: The confirmatory design of a ``decision: offline`` model run: the decision arm's slot replays offline sequence k,
#: the yoked arms replay offline sequence k - 1 (the decision arm's sequence one block later, as in the design).
NOSELECT_MODEL_ARMS = (
    ArmSpec("A1", "replay", observed=True, source="offline", lag=0),
    ArmSpec("A2", "replay", observed=True, source="offline", lag=1),
    ArmSpec("A3", "replay", observed=False, source="offline", lag=1),
    ArmSpec("A3b", "replay", observed=False, source="offline", lag=1),
    ArmSpec("A3L", "replay", observed=False, source="offline", lag=2),
    ArmSpec("A4", "sham", observed=True),
    ArmSpec("A5", "replay", observed=True, source="offline", lag=1, variant="detuned"),
)


def simulate_batch(
    cfg: dict[str, Any],
    pool: Pool,
    rep_ids: np.ndarray,
    phase: str,
    condition: str,
    plant: str | None,
    plant_frac: float,
    models: dict[int, list[pa.SelectionModel]] | None,
    workers: int,
) -> list[dict[str, Any]]:
    """Run len(rep_ids) replicates in lock-step (vectorised simulation, one schedule and key set per replicate).
    ``phase`` "model" (``condition`` names the model): one model run per replicate, returns {"table", "field"};
    otherwise a complete experiment under ``condition`` (config ``conditions``) with the frozen model set of each
    replicate's field (``models[field]``: the model of a run is the model of its sample)."""
    batch = rep_ids.size
    seed = int(cfg["seed"])
    label = f"{seed}:{phase}:{condition}:{plant}:{plant_frac}"
    tag = hashlib.sha256(f"{label}:{int(rep_ids[0])}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(tag[:8], "big"))
    keys = [RunKeys.from_seed(f"{label}:{int(r)}") for r in rep_ids]
    seal_key = hashlib.sha256(b"batch-seal" + tag).digest()  # one sealing key per batch (see the report)
    blind_key = hashlib.sha256(b"batch-blind" + tag).digest()  # seals the arm-revealing episode fields
    fidx = rep_ids % pool.nom.shape[0]
    pr, cc = cfg["protocol"], cfg["controller"]
    if phase == "model":
        cond: dict[str, Any] = {}
        base = nuisance(cfg, cfg["models"][condition])
    else:
        cond = cfg["conditions"][condition]
        base = nuisance(cfg, cond.get("device"))
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

    def device(n_slots: int, nu: dict[str, float], plant_sizes: dict[str, np.ndarray] | None = None) -> TableMeasure:
        laser = ou(rng, batch, n_slots, nu["laser_rel_sigma"], nu["laser_tau_slots"])
        contrast = ou(rng, batch, n_slots, nu["contrast_rel_sigma"], nu["contrast_tau_slots"])
        return TableMeasure(
            pool,
            fidx,
            shots,
            rng,
            laser,
            contrast,
            nu["laser_jitter_rel"],
            plant_sizes,
            nu["within_sigma"],
            nu["within_rho"],
        )

    m_cal = pa.M_CAL
    wu = pa.WARMUP_BLOCKS
    # offline sequences: the decision process on a separate simulated device (the model's nuisance, its own draws);
    # the calibration block replays M_CAL + warm-up of them, a no-selection model member N_CONF + warm-up
    n_off = max(m_cal, pa.N_CONF) + wu
    off_dev = device(n_off, nuisance(cfg))
    offline = []
    sealer, meta = Sealer(seal_key), Sealer(blind_key)
    for k in range(n_off):
        ep = run_episode(
            ArmSpec("offline", "decision"), k, ctrl, off_dev, t_n, batch, sealer, np.full(batch, k), meta_sealer=meta
        )
        offline.append(ep.actions)
    off_actions = np.stack(offline)  # (max(M, N) + warm-up, T, B)
    decoy_rng = np.random.default_rng(rng.integers(1 << 62))
    sham_rng = np.random.default_rng(rng.integers(1 << 62))
    conf_design = ProtocolDesign(CONF_DESIGN_ARMS, t_n)
    conf_var = {a.name: a.variant for a in CONF_DESIGN_ARMS}
    n_conf = pa.N_CONF + wu

    if phase == "model":
        sch = np.stack([block_schedule(k.order, conf_design.names, n_conf) for k in keys])
        dev = device(sch.shape[1] * sch.shape[2], base)
        noselect = cfg["models"][condition].get("decision") == "offline"
        design = ProtocolDesign(NOSELECT_MODEL_ARMS, t_n) if noselect else conf_design
        data = run_design(
            design, n_conf, sch, ctrl, dev, seal_key, blind_key, decoy_rng, sham_rng, off_actions, pool,
            offline=off_actions if noselect else None, arms=MODEL_ARMS,
        )  # fmt: skip
        names = {a: a for a in MODEL_ARMS}
        return [
            {
                "table": replicate_table(data, i, sch[i], names, pool, int(fidx[i]), n_conf, conf_var),
                "field": int(fidx[i]),
            }
            for i in range(batch)
        ]

    assert models is not None
    # calibration block
    cal_design = ProtocolDesign(CAL_DESIGN_ARMS, t_n)
    n_cal = m_cal + wu
    cal_sch = np.stack(
        [block_schedule(hashlib.sha256(k.order + b"calibration").digest(), cal_design.names, n_cal) for k in keys]
    )
    cal_nu = nuisance(cfg, cond.get("device"), cond.get("calibration"))
    cal_data = run_design(
        cal_design,
        n_cal,
        cal_sch,
        ctrl,
        device(n_cal * len(cal_design.names), cal_nu),
        seal_key,
        blind_key,
        decoy_rng,
        sham_rng,
        off_actions,
        pool,
        offline=off_actions,
    )
    cal_var = {a.name: a.variant for a in CAL_DESIGN_ARMS}
    cal_tables = [
        replicate_table(cal_data, i, cal_sch[i], {a: a for a in cal_design.names}, pool, int(fidx[i]), n_cal, cal_var)
        for i in range(batch)
    ]
    del cal_data
    n_boot = int(cfg["replicates"]["n_boot"])
    with mp.get_context("fork").Pool(workers) as wp:
        cals = wp.starmap(pa.derive_calibration, [(t, models[int(fidx[i])], n_boot) for i, t in enumerate(cal_tables)])
    del cal_tables

    # confirmatory run: N_CONF valid blocks, the frozen margins for the planted size
    sch = np.stack([block_schedule(k.order, conf_design.names, n_conf) for k in keys])
    sizes: dict[str, np.ndarray] | None = None
    if plant == "decision":
        sizes = {"A1": plant_frac * np.array([c.margin[1] for c in cals])}
    elif plant == "observer":
        sizes = {a: plant_frac * np.array([c.margin[0] for c in cals]) for a in OBSERVED_ARMS}
    conf_nu = nuisance(cfg, cond.get("device"), cond.get("confirmatory"))
    dev = device(sch.shape[1] * sch.shape[2], conf_nu, sizes)
    data = run_design(conf_design, n_conf, sch, ctrl, dev, seal_key, blind_key, decoy_rng, sham_rng, off_actions, pool)
    jobs = []
    for i, kk in enumerate(keys):
        codes = arm_codes(kk.blind, conf_design.names)
        table = replicate_table(data, i, sch[i], codes, pool, int(fidx[i]), n_conf, conf_var)
        jobs.append(
            {
                "cal": cals[i],
                "table": table,
                "n_boot": n_boot,
                "blind_key": kk.blind,
                "blind_commit": kk.commitments()["blind_sha256"],
            }
        )
    del data
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


# --- Engineering comparisons (M7, E5) ---------------------------------------------------------------------------------


def engineering_validation(n_rep: int, seed: int, n_perm: int) -> dict[str, list[dict[str, Any]]]:
    """:func:`~glowbox.analysis.prereg_analysis_001.paired_margin_test` at its fixed sizes: null (normal and
    heavy-tailed t3 diffs, two-sided and one-sided), and an effect of the replicate's own control-derived margin."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[dict[str, Any]]] = {}
    for name, dist, alt, shift in (
        ("null, normal, two-sided", "normal", "two-sided", 0.0),
        ("null, t3, two-sided", "t3", "two-sided", 0.0),
        ("null, normal, one-sided", "normal", "greater", 0.0),
        ("effect = margin, normal, two-sided", "normal", "two-sided", 1.0),
        ("effect = margin, normal, one-sided", "normal", "greater", 1.0),
    ):
        res = []
        for i in range(n_rep):

            def draw(n: int, dist: str = dist) -> np.ndarray:
                out_: np.ndarray = rng.standard_t(3, n) if dist == "t3" else rng.normal(0.0, 1.0, n)
                return out_

            c = draw(pa.ENG_N_CONTROL)
            d0 = draw(pa.ENG_N)
            probe = pa.paired_margin_test(d0, c, alt, n_perm=16, seed=i)  # margin from the controls only
            d = d0 + shift * probe["margin"]
            t = pa.paired_margin_test(d, c, alt, n_perm=n_perm, seed=i)
            res.append({"rej": t["rejected"], "eq": t["tost"]["equivalent"], "margin": t["margin"]})
        out[name] = res
    return out


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


def validation_lock(run_dir: Path, config_path: Path, cfg: dict[str, Any]) -> tuple[Lock, Path]:
    """A plan lock over everything this script runs (stage S validation: the same machinery as a confirmatory lock),
    including the effective config (after --quick / --replicates), written into the run directory."""
    eff = run_dir / "config.effective.yaml"
    eff.write_text(yaml.safe_dump(cfg, sort_keys=True))
    configs: list[Path] = [eff]
    if config_path.resolve().is_relative_to(REPO_ROOT.resolve()):
        configs.append(config_path)
    lock = pa.create_plan_lock(["glowbox.experiments.prereg_calibration"], configs, model_set=cfg["models"])
    path = lock.write(run_dir / "validation.lock.yaml")
    pa.verify_lock(lock)
    return lock, path


def run_phase(
    cfg: dict[str, Any],
    pool: Pool,
    phase: str,
    condition: str,
    n: int,
    plant: str | None = None,
    frac: float = 0.0,
    models: dict[int, list[pa.SelectionModel]] | None = None,
) -> list[dict[str, Any]]:
    rc = cfg["replicates"]
    out: list[dict[str, Any]] = []
    for s in range(0, n, int(rc["batch"])):
        ids = np.arange(s, min(n, s + int(rc["batch"])))
        t0 = time.perf_counter()
        out += simulate_batch(cfg, pool, ids, phase, condition, plant, frac, models, int(rc["workers"]))
        what = f"{phase} {condition} {plant or ''} {frac:g}"
        print(f"  {what}: {len(out)}/{n} ({time.perf_counter() - t0:.0f} s)", flush=True)
    return out


def fmt_rate(x: tuple[float, float, float, int]) -> str:
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}] (n = {x[3]})"


#: Upper bound of the decisive error rate (the nominal primary family) at the documented full size.
DECISIVE_TOLERANCE = 0.06
#: The documented full size: nominal null replicates and replicates per stress condition.
FULL_NULL_RUNS = 4000
FULL_STRESS_RUNS = 1000


@dataclass
class Criterion:
    """A pass criterion of the plan (section 12), stated on the Clopper-Pearson 95 % interval of a rate.

    ``rule``: "lo<=x" (descriptive error rate: the interval is not above x), "hi>=x" (coverage, power or claim
    rate: the interval is not below x), "hi<=x" (decisive error rate: the interval's upper bound is at most x; it can
    show that a test is calibrated, and is judged only at the documented full size: ``full`` False -> PENDING)."""

    what: str
    value: tuple[float, float, float, int]
    rule: str
    x: float
    full: bool = True

    @property
    def pending(self) -> bool:
        return self.rule == "hi<=x" and not self.full

    @property
    def passed(self) -> bool:
        _, lo, hi, n = self.value
        if n == 0 or self.pending:
            return False
        if self.rule == "lo<=x":
            return lo <= self.x
        if self.rule == "hi<=x":
            return hi <= self.x
        return hi >= self.x

    @property
    def status(self) -> str:
        return "PENDING (full size)" if self.pending else ("PASS" if self.passed else "FAIL")

    def line(self) -> str:
        cmp = {
            "lo<=x": f"not above {self.x:g}",
            "hi>=x": f"not below {self.x:g}",
            "hi<=x": f"upper bound <= {self.x:g} at the full size",
        }[self.rule]
        return f"| {self.what} | {fmt_rate(self.value)} | {cmp} | {self.status} |"


def verdict(crit: list[Criterion]) -> str:
    """The report's overall line."""
    if any(not c.passed and not c.pending for c in crit):
        return "FAIL"
    if any(c.pending for c in crit):
        return "every criterion judged at this size passes; the decisive criterion needs the full-size run"
    return "PASS"


def _any(res: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    return [{"x": any(r[k] for k in keys)} for r in res]


def _not(res: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [{"x": not r[key]} for r in res]


def _with_ci(res: list[dict[str, Any]], h: str) -> list[dict[str, Any]]:
    """The runs that report an interval for ``h`` (H_decision without a consistent member reports none)."""
    return [r for r in res if r.get(f"{h}_has_ci", True)]


#: Stress conditions with a pass criterion (primary family, any positive claim); others are reported only.
STRESS_CRITERIA = ("drift_covered", "drift_unmodelled", "nonstationary", "drift_wrong_member")


def criteria(
    null: dict[str, list[dict[str, Any]]],
    pos: dict[str, list[dict[str, Any]]],
    eng: dict[str, list[dict[str, Any]]],
    full: bool = False,
) -> list[Criterion]:
    a, pw = pa.ALPHA, pa.POWER
    nom = null["nominal"]
    sec = [f"{k}_rej" for k in pa.SECONDARY]
    both = ["H_decision_rej", "H_observer_rej"]

    def err(what: str, res: list[dict[str, Any]], key: str, x: float = a) -> Criterion:
        return Criterion(what, rate(res, key), "lo<=x", x)

    def at_least(what: str, res: list[dict[str, Any]], key: str, x: float) -> Criterion:
        return Criterion(what, rate(res, key), "hi>=x", x)

    out = [
        Criterion(
            "nominal: primary family, any positive claim (complete rule) [decisive]",
            rate(_any(nom, both), "x"),
            "hi<=x",
            DECISIVE_TOLERANCE,
            full,
        ),
        err("nominal: H_decision, p <= 0.05 against the calibration null", nom, "H_decision_raw_rej"),
        err("nominal: H_observer, p <= 0.05 against the calibration null", nom, "H_observer_raw_rej"),
        err("nominal: primary family, any positive claim (complete rule)", _any(nom, both), "x"),
        err("nominal: secondary family S2-S3, any rejection (complete rule)", _any(nom, sec), "x"),
        at_least(
            "nominal: H_decision 95 % interval covers 0 (runs with an interval)",
            _with_ci(nom, "H_decision"),
            "H_decision_cover",
            0.95,
        ),
        at_least(
            "nominal: H_observer 95 % interval covers 0 (runs with an interval)",
            _with_ci(nom, "H_observer"),
            "H_observer_cover",
            0.95,
        ),
        at_least("nominal: H_decision null claim", nom, "H_decision_null_claim", pw),
        at_least("nominal: H_observer null claim", nom, "H_observer_null_claim", pw),
        err("nominal: null-stability check flags the run", _not(nom, "stable"), "x"),
        at_least("nominal: manipulation check S4 passes", nom, "S4_quantum_test_passed", 0.95),
        at_least("nominal: arm names absent from every blinded output", nom, "names_hidden", 1.0),
    ]
    for plant, h in (("decision", "H_decision"), ("observer", "H_observer")):
        res = pos.get(f"{plant}:1", [])
        out += [
            at_least(f"positive {plant}, 1 x margin: power of the complete rule", res, f"{h}_rej", pw),
            at_least(f"positive {plant}, 1 x margin: attributed", res, "identified", pw),
            at_least(f"positive {plant}, 1 x margin: not equivalent", _not(res, f"{h}_eq"), "x", pw),
        ]
    for cond in STRESS_CRITERIA:
        if null.get(cond):
            out.append(err(f"{cond}: primary family, any positive claim", _any(null[cond], both), "x"))
    for plant, h in (("decision", "H_decision"), ("observer", "H_observer")):
        res = pos.get(f"nonstationary:{plant}:1", [])
        if res:
            out.append(
                err(
                    f"nonstationary, {plant} effect = 1 x margin: false null claim", res, f"{h}_null_claim", pa.ALPHA_EQ
                )
            )
    for name, res in eng.items():
        if name.startswith("null"):
            out.append(err(f"engineering test, {name}: rejection rate", res, "rej"))
        else:
            out.append(at_least(f"engineering test, {name}: power", res, "rej", pw))
    return out


def _median_iqr(v: np.ndarray, f: str = ".4g") -> str:
    return f"{np.median(v):{f}} [IQR {np.quantile(v, 0.25):{f}}-{np.quantile(v, 0.75):{f}}]"


def summary_markdown(
    cfg: dict[str, Any],
    pool: Pool,
    models: dict[str, dict[int, pa.SelectionModel]],
    null: dict[str, list[dict[str, Any]]],
    pos: dict[str, list[dict[str, Any]]],
    eng: dict[str, list[dict[str, Any]]],
    seconds: float,
) -> tuple[str, dict[str, float]]:
    rc = cfg["replicates"]
    nom = null["nominal"]
    m0 = np.array([r["margin0"] for r in nom])
    m1 = np.array([r["margin1"] for r in nom])
    sd0 = np.array([r["sd_mean0"] for r in nom])
    cref = np.array([r["c_ref"] for r in nom])
    full = len(nom) >= FULL_NULL_RUNS and all(
        len(v) >= FULL_STRESS_RUNS for k, v in null.items() if k != "nominal" and v
    )
    crit = criteria(null, pos, eng, full)
    lines = [
        "Stage-S validation of the PREREG-001 pipeline (docs/preregistration/PREREG-001-observer-decision.md). The "
        "simulator has no decision or observer effect by construction: these numbers say whether the frozen procedure "
        "keeps its false-positive rate and has its planned power, nothing about the hypotheses. Status of the plan: "
        "DRAFT (not signed off, not locked).",
        "",
        "Every replicate is a complete simulated experiment: offline sequences, a calibration block of "
        f"{pa.M_CAL} valid blocks (seven sealed control arms, slot for slot like a confirmatory block), the frozen "
        f"calibration (empirical null, margins, theta0_D set, model check), a confirmatory run of N = {pa.N_CONF} "
        "valid blocks x 7 arms in a random order per block, sealing of every outcome, decoys from the frozen model, "
        "the arm-revealing fields sealed with the blinding key, the locked reduction to the episode table, the frozen "
        f"blinded analysis ({rc['n_boot']} null draws per test), unblinding, the complete decision rule. "
        f"T = {cfg['protocol']['n_steps']} steps x S = {cfg['protocol']['shots']} shots per episode; "
        f"{pool.nom.shape[0]} true fields cycled over replicates; slow laser and contrast drift (OU over episode "
        "slots), per-step laser jitter. Decision process: the stand-in threshold rule (see the script).",
        "",
        "## Frozen quantities (nominal condition; median over the null replicates, each replicate freezes its own)",
        "",
        "- Margin rule (section 5): Delta = Q_{1-alpha/2}(|D|) + Q_{2 MARGIN_POWER - 1}(|D|), D the "
        f"control-vs-control null of the mean of N = {pa.N_CONF} pairs (the shift detected with power "
        f"{pa.MARGIN_POWER} at Holm's first level; MARGIN_POWER = 1 - alpha_eq).",
        f"- Margin, lag 0 (H_observer): {_median_iqr(m0)}; lag 1 (H_decision): {_median_iqr(m1)}, in units of the "
        "referenced fluorescence r = N_sig / N_ref.",
        f"- Control-vs-control SD of the mean of N pairs (lag 0): {_median_iqr(sd0)}; margin / SD: "
        f"{np.median(m0 / sd0):.2f}.",
        f"- Spin signal of the replayed sequences, C_ref = mean(r_detuned - r_resonant): {np.median(cref):.4g}; "
        f"margin / C_ref: {np.median(m0 / cref):.3f} (lag 0), {np.median(m1 / cref):.3f} (lag 1).",
        "- theta0_D set (model runs, phase `model`, per model and per true field: the model of a run is the model "
        f"of its sample; the plan requires every SE <= margin / 10 = {np.median(m1) / 10:.3g}):",
    ]
    for name, per_field in models.items():
        th = ", ".join(f"{m.theta0:.3g} +- {m.theta0_se:.2g}" for m in per_field.values())
        wc = ", ".join(f"{m.wcorr_mean:.3g}" for m in per_field.values())
        n_r = sum(m.n_runs for m in per_field.values())
        params = cfg["models"][name] or "nominal"
        lines.append(f"  - {name} ({params}; {n_r} runs): theta0_D per field {th}; predicted wcorr per field {wc}.")
    lines += [
        "",
        "## Pass criteria (plan section 12)",
        "",
        "Each criterion is a rule on the Clopper-Pearson 95 % interval of a rate. The decisive error rate (the "
        f"nominal primary family) passes when the upper bound is at most {DECISIVE_TOLERANCE} at the documented full "
        f"size (R = {FULL_NULL_RUNS} nominal, {FULL_STRESS_RUNS} per stress condition): a rule that can show the test "
        "is calibrated; below the full size it is PENDING. Descriptive error rates pass when the interval is not "
        "above their level (lower bound <= level), coverage, power and claim rates when it is not below their "
        "target (upper bound >= target).",
        "",
        "| criterion | rate [95 % interval] | rule | result |",
        "| --- | --- | --- | --- |",
        *[c.line() for c in crit],
        "",
        f"All criteria: {verdict(crit)}.",
        "",
        "## Null calibration (no effect), nominal condition",
        "",
        "| quantity | H_decision | H_observer |",
        "| --- | --- | --- |",
    ]
    for label, key in (
        ("positive claim (complete rule)", "rej"),
        ("Holm rejection against the calibration null alone", "rej_cal"),
        ("p <= 0.05 against the calibration null (uncorrected)", "raw_rej"),
        ("TOST equivalent at the frozen margin", "eq"),
        ("null claim (equivalent, stable null; H_decision: model check)", "null_claim"),
        ("95 % interval covers 0 (runs with an interval)", "cover"),
        ("Newey-West check, p <= 0.05", "nw_rej"),
    ):
        dec, obs = (_with_ci(nom, h) if key == "cover" else nom for h in ("H_decision", "H_observer"))
        lines.append(
            f"| {label} | {fmt_rate(rate(dec, f'H_decision_{key}'))} | {fmt_rate(rate(obs, f'H_observer_{key}'))} |"
        )
    lines += [
        "",
        f"- Model check: the calibration block inside some member's range {fmt_rate(rate(nom, 'model_ok_cal'))}, the "
        f"confirmatory sealed controls {fmt_rate(rate(nom, 'model_ok_conf'))}, one member consistent with both (an "
        f"interval and a decision for H_decision) {fmt_rate(rate(nom, 'model_ok'))}.",
        "- Manipulation check S4 (the detuned arm loses the task): passes in "
        f"{fmt_rate(rate(nom, 'S4_quantum_test_passed'))}.",
        "- Secondary family (S2, S3: decision cells at every consistent member's theta0_D of their statistic, every "
        "cell against the calibration and the concurrent null): any rejection "
        f"{fmt_rate(rate(_any(nom, [f'{k}_rej' for k in pa.SECONDARY]), 'x'))}; against the calibration null alone "
        f"{fmt_rate(rate(_any(nom, [f'{k}_rej_cal' for k in pa.SECONDARY]), 'x'))}.",
        "",
        "## Stress conditions (no effect)",
        "",
        "| condition | what differs | model set | positive claim, H_decision | positive claim, H_observer | "
        "H_decision p <= 0.05 (calibration null alone) | H_observer p <= 0.05 (calibration null alone) | model "
        "check passed (a member reproduces both blocks) | null stable |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cond, res in null.items():
        if cond == "nominal" or not res:
            continue
        desc = cfg["conditions"][cond].get("description", "")
        mset = ", ".join(cfg["conditions"][cond]["models"])
        lines.append(
            f"| {cond} | {desc} | {mset} | {fmt_rate(rate(res, 'H_decision_rej'))} | "
            f"{fmt_rate(rate(res, 'H_observer_rej'))} | {fmt_rate(rate(res, 'H_decision_raw_rej'))} | "
            f"{fmt_rate(rate(res, 'H_observer_raw_rej'))} | {fmt_rate(rate(res, 'model_ok'))} | "
            f"{fmt_rate(rate(res, 'stable'))} |"
        )
    ns = {k.split(":")[1]: v for k, v in pos.items() if k.startswith("nonstationary:")}
    if ns:
        lines += [
            "",
            "Nonstationary confirmatory run with an effect of exactly the frozen margin (null claims under "
            "nonstationarity: a null claim here is a false equivalence):",
            "",
            "| planted | positive claim | false null claim | stability check flags the run |",
            "| --- | --- | --- | --- |",
        ]
        for plant, res in ns.items():
            h = "H_decision" if plant == "decision" else "H_observer"
            lines.append(
                f"| {plant} | {fmt_rate(rate(res, f'{h}_rej'))} | {fmt_rate(rate(res, f'{h}_null_claim'))} | "
                f"{fmt_rate(rate(_not(res, 'stable'), 'x'))} |"
            )
    lines += [
        "",
        "## Positive control (effect planted in the simulated device)",
        "",
        "Decision: E[r] of the decision arm A1 shifted by the size. Observer: E[r] of every observed arm "
        "(A1, A2, A4, A5) shifted, the sealed arms not. Size = the replicate's own frozen margin, or a fraction of it "
        "(power curve: no pass criterion).",
        "",
        "| planted | size | power (complete rule) | attributed (planted rejected, other not) | not equivalent |",
        "| --- | --- | --- | --- | --- |",
    ]
    metrics: dict[str, float] = {
        "fwer_primary_nominal": rate(_any(nom, ["H_decision_rej", "H_observer_rej"]), "x")[0],
        "margin_lag0_median": float(np.median(m0)),
        "margin_lag1_median": float(np.median(m1)),
        "margin_over_c_ref_lag0": float(np.median(m0 / cref)),
        "criteria_passed": float(sum(c.passed for c in crit)),
        "criteria_total": float(len(crit)),
    }
    for key, res in pos.items():
        if key.startswith("nonstationary:"):
            continue
        plant, frac = key.split(":")
        h = "H_decision" if plant == "decision" else "H_observer"
        lines.append(
            f"| {plant} | {frac} x margin | {fmt_rate(rate(res, f'{h}_rej'))} | {fmt_rate(rate(res, 'identified'))} "
            f"| {fmt_rate(rate(_not(res, f'{h}_eq'), 'x'))} |"
        )
        metrics[f"power_{plant}_{frac}"] = rate(res, f"{h}_rej")[0]
    lines += [
        "",
        f"## Engineering comparisons (M7 E-1 / E-2, E5): `paired_margin_test` at its fixed sizes "
        f"({pa.ENG_N} pairs, {pa.ENG_N_CONTROL} control repeats)",
        "",
        "| condition | rejection rate | TOST equivalent | median margin (units of the diff SD) |",
        "| --- | --- | --- | --- |",
    ]
    for name, res in eng.items():
        mm = np.median([r["margin"] for r in res])
        lines.append(f"| {name} | {fmt_rate(rate(res, 'rej'))} | {fmt_rate(rate(res, 'eq'))} | {mm:.3g} |")
    lines += [
        "",
        "## Reading",
        "",
        "- The margin is derived from the control runs by a fixed rule: the effect that the planned experiment "
        f"detects against the control-vs-control distribution with power {pa.MARGIN_POWER} at Holm's first level "
        "(1 - alpha_eq: the positive test and the equivalence test are dual at the margin). Version 1 used the 0.95 "
        "quantile of a single control pair (about 40 % of the spin signal, equivalence trivially easy); version 2 "
        "used power 0.95, which left the null claims below their 0.90 target.",
        "- No decision uses the calibration null alone: a positive claim needs the concurrent null too, and the "
        "decisive criterion is on the complete rule. The uncorrected rates of the calibration null alone are "
        "descriptive (the null-calibration table above).",
        "- A positive claim needs a Holm rejection against the frozen calibration controls and against the run's own "
        "sealed controls; the nonstationary condition shows why (the calibration null alone is too narrow when the "
        "confirmatory run drifts more).",
        "- H_decision depends on a model of the selection effect. The drift_unmodelled condition (within-episode "
        "drift missing from the model set) shows the calibration null alone rejecting too often, and the model check "
        "refusing H_decision instead. In drift_covered the same drift is in the model set: theta0_D is non-zero and "
        "the worst-case test keeps its level.",
        "- The model rule: members whose wcorr envelope excludes the sealed controls of the calibration block or of "
        "the confirmatory run are dropped, and the worst case is taken over the rest. drift_wrong_member adds a "
        "member whose drift matches the device (wcorr) but whose selection effect is wrong (its decision arm cannot "
        "select: theta0_D = 0); with the right member also in the set, the worst case keeps the level. "
        "drift_only_wrong_member (no criterion) shows the limit of any model check on wcorr: when the set holds only "
        "the wrong member and one the controls refute, H_decision is tested at a wrong centre. The set must contain "
        "the true selection effect; that is why it is locked with the plan and reviewed before sign-off.",
        "- The Newey-West robustness check (not decisive): p <= 0.05 in the nominal null "
        f"{fmt_rate(rate(nom, 'H_decision_nw_rej'))} (H_decision), {fmt_rate(rate(nom, 'H_observer_nw_rej'))} "
        "(H_observer).",
        f"- Sizes: N = {pa.N_CONF} confirmatory blocks (the owner's decision of 2026-11-27, was 100: the margin "
        f"halves), M_CAL = {pa.M_CAL} calibration blocks. M_CAL is not scaled with N: the calibration estimates the "
        "spread of one control pair, so the precision of the critical values and of the margin depends on M_CAL, not "
        "on N; the null of the N-pair mean is the block bootstrap of the M_CAL series. The model runs per member are "
        "not scaled either: theta0_D's Monte Carlo SE and the margin both fall as 1 / sqrt(N).",
        "- The bootstrap block length is the larger of the Newey-West rule and twice the integrated correlation time "
        "of the calibration series (Politis-White flat-top window). In this simulation the control pairs are "
        "uncorrelated (the slow drift cancels within a block), so it stays at the Newey-West rule: "
        f"median {np.median([r.get('block_len0', np.nan) for r in nom]):.0f} (lag 0).",
        "- One sealing key and one meta-sealing key per batch of replicates instead of per replicate (speed); every "
        "outcome is still sealed and unsealed through the same code. Blinding keys and order seeds are per replicate.",
        "- The stand-in decision process is not the M5 policy, and theta0_D depends on the policy: the M5 policy as a "
        "Controller (glowbox.analysis.arms) is M7's adapter, with its own model runs and this validation repeated.",
        "",
        "Run sizes here (CPU): "
        + ", ".join(f"{k} {len(v)}" for k, v in null.items())
        + f" null replicates; positive {rc['positive']} per hypothesis; curve {rc['curve']} per fraction; "
        f"{rc['model_runs']} model runs per model; engineering {rc['engineering']} per condition. The documented full "
        "size (several hours on this CPU; any machine, no GPU needed):",
        "",
        f"    uv run python -m glowbox.experiments.prereg_calibration --replicates {FULL_NULL_RUNS} "
        f"--stress-replicates {FULL_STRESS_RUNS}",
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
        r0 = rate(null, f"{h}_rej")
        xs, ys, lo, hi = [0.0], [r0[0]], [r0[1]], [r0[2]]
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
    axes[1].set(xlabel="planted effect / frozen margin", ylabel="positive-claim rate", title="Power", ylim=(0, 1.02))
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replicates", type=int, default=None, help="nominal null replicates (default from the config)")
    ap.add_argument("--stress-replicates", type=int, default=None, help="null replicates per stress condition")
    ap.add_argument("--report-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.quick)
    if args.replicates:
        cfg["replicates"]["null_runs"] = args.replicates
    if args.stress_replicates:
        cfg["replicates"]["stress_runs"] = args.stress_replicates
    torch.set_num_threads(8)
    t_start = time.perf_counter()
    with RunLog(cfg["name"], config=cfg, seed=int(cfg["seed"])) as run:
        lock, lock_path = validation_lock(run.dir, args.config, copy.deepcopy(cfg))
        keys = RunKeys.from_seed(f"{cfg['seed']}:run")
        run.annotate("prereg", prereg_record(lock, lock_path, keys, "simulation", {"positive_control": 1.0}))
        print("building the field pool ...", flush=True)
        pool = build_pool(cfg, int(cfg["seed"]))
        rc = cfg["replicates"]
        models: dict[str, dict[int, pa.SelectionModel]] = {}
        needed = sorted({m for c in cfg["conditions"].values() for m in c["models"]})
        for name in needed:
            runs = run_phase(cfg, pool, "model", name, int(rc["model_runs"]))
            fields = sorted({r["field"] for r in runs})
            params = cfg["models"][name]
            models[name] = {
                f: pa.selection_model(name, [r["table"] for r in runs if r["field"] == f], params=params)
                for f in fields
            }
            del runs
        null: dict[str, list[dict[str, Any]]] = {}
        pos: dict[str, list[dict[str, Any]]] = {}
        for cond, cc in cfg["conditions"].items():
            ms = {f: [models[m][f] for m in cc["models"]] for f in range(pool.nom.shape[0])}
            n = int(rc["null_runs"]) if cond == "nominal" else int(rc["stress_runs"])
            null[cond] = run_phase(cfg, pool, "null", cond, n, models=ms)
            if cond == "nonstationary" and int(rc.get("stress_positive", 0)):
                for plant in ("decision", "observer"):
                    n_p = int(rc["stress_positive"])
                    pos[f"nonstationary:{plant}:1"] = run_phase(cfg, pool, "positive", cond, n_p, plant, 1.0, ms)
            if cond != "nominal":
                continue
            for plant in ("decision", "observer"):
                pos[f"{plant}:1"] = run_phase(cfg, pool, "positive", cond, int(rc["positive"]), plant, 1.0, ms)
                for frac in rc.get("curve_fractions", []) if int(rc["curve"]) else []:
                    n_c = int(rc["curve"])
                    pos[f"{plant}:{frac:g}"] = run_phase(cfg, pool, "positive", cond, n_c, plant, float(frac), ms)
        eng = engineering_validation(int(rc["engineering"]), int(cfg["seed"]), int(rc["n_boot"]))
        seconds = time.perf_counter() - t_start
        md, metrics = summary_markdown(cfg, pool, models, null, pos, eng, seconds)
        (run.dir / "replicates.json").write_text(json.dumps({"null": null, "positive": pos}, default=str))
        run.log_metrics(0, **metrics)
        png = run.save_figure(figure(null["nominal"], pos), "calibration")
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
