"""Frozen analysis of PREREG-001 (docs/preregistration/PREREG-001-observer-decision.md). Do not edit once the plan
is locked: any change needs a new lock version and an entry in docs/preregistration/deviations.md.

Stages, in this order. Each refuses to start unless the plan lock verifies and is published (:func:`check_plan`),
the run registry is committed, and every run it reads is registered and its record matches the lock:

0. :func:`build_episode_table`: the frozen reduction of unsealed counts to per-step observables (r, r_model,
   off-line flags, task error, action-sequence hashes) and the arm-blind exclusion flags of every block.
1. :func:`freeze_calibration` on the CALIBRATION block (control arms only: no decision process applied, no live
   reader of the results) and on *all* registered model runs of the series (runs of the frozen drift and timing
   model that predict the selection effect theta0_D of each model in the pre-registered set). It derives the
   control-vs-control pair series (the empirical null of every test), the smallest effect size of interest per
   geometry (:func:`control_margin`), the theta0_D set and the model check, and writes the calibration and its
   calibration lock. Both are committed, and the calibration is registered, before any decision or observer arm
   runs.
2. :func:`analyse_run` on the confirmatory run, *blinded*: arms appear only as codes. Every pair of codes whose
   episodes replay the same action sequence (equal ``seq_hash``), in the same block (lag 0) or in consecutive
   blocks (lag 1), is a *cell*; every cell gets every statistic, its empirical-null p-value against the calibration
   controls, its 95 % interval, its equivalence (TOST) result, a Newey-West robustness check and, for the primary
   statistic, its p-value against every other cell of the same geometry *of the same run* (the concurrent nulls).
   The output does not depend on which code is which arm; its sha256 is registered and committed before the
   blinding key is released.
3. :func:`unblind_run`: maps codes to arms with the released blinding key and reads off the pre-registered cells:
   the two primary hypotheses (a positive claim needs Holm rejection against the calibration controls *and* against
   the run's own sealed controls; a null claim needs a stable null), the model check, the null-stability check, the
   secondary family (Holm), and every other cell as exploratory.

The empirical null: a cell's statistic is the mean over its pairs of a per-pair statistic; under the null
hypothesis the pairs are distributed like control-vs-control pairs of the same geometry, which are exchangeable
(random order inside blocks), so their distribution is symmetric about zero. The null distribution of the mean of n
pairs is a circular moving-block bootstrap of the control series with a random sign per bootstrap block
(:func:`null_means`): the spread and shape come from the control runs, the centre from the exchangeability. The one
term no control run can contain is the selection effect of a decision arm under drift (its choices depend on counts
that carry drift); it comes from the pre-registered set of drift and timing models (theta0_D per model), a decision
cell counts only if it rejects at *every* model's centre, and the model check refuses H_decision when the sealed
controls' within-episode correlation lies outside what the model set predicts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from scipy import stats

from glowbox.analysis.arms import action_hash
from glowbox.analysis.prereg import (
    Lock,
    LockMismatchError,
    Registry,
    check_run_record,
    committed_unchanged,
    plan_files,
    published_lock_problems,
    sha256_bytes,
    sha256_file,
)
from glowbox.analysis.prereg import unblind as unblind_codes
from glowbox.analysis.runlog import REPO_ROOT

PLAN = "PREREG-001"
SCRIPT_PATH = "glowbox/analysis/prereg_analysis_001.py"
SCRIPT_MODULE = "glowbox.analysis.prereg_analysis_001"
PLAN_DOC = "docs/preregistration/PREREG-001-observer-decision.md"
DEVIATIONS_PATH = "docs/preregistration/deviations.md"
REGISTRY_PATH = "docs/preregistration/PREREG-001.registry.yaml"
#: The git tag of the commit that holds the plan lock (pushed to the public repository: the timestamp).
LOCK_TAG = "prereg-001"

# --- Frozen constants of the plan ------------------------------------------------------------------------------------

#: Family-wise error rate of each family (primary, secondary), Holm within the family.
ALPHA = 0.05
#: TOST level per side for the two primaries (alpha / 2: Bonferroni over the two equivalence claims).
ALPHA_EQ = 0.025
#: Confirmatory valid blocks (fixed; the margin is derived for this N, :func:`control_margin`).
N_CONF = 100
#: Calibration valid blocks.
M_CAL = 200
#: Warm-up blocks at the start of the calibration block and of the confirmatory run (excluded from every analysis).
WARMUP_BLOCKS = 2
#: Collection stops at this multiple of the planned valid blocks attempted.
ATTEMPT_CAP = 1.2
#: The margin rule (smallest effect size of interest): Delta = Q_{1 - ALPHA/2}(|D|) + Q_{2 MARGIN_POWER - 1}(|D|),
#: D the empirical control-vs-control null of the mean of N_CONF pairs. That is the shift at which the
#: calibration-null test at Holm's first level (ALPHA / 2, two-sided) has power MARGIN_POWER. 0.95 leaves room for the
#: two further conditions of a positive claim (the concurrent null, the model check): the complete rule is validated
#: to keep power >= 0.90 at Delta.
MARGIN_POWER = 0.95
#: Planned power of the complete decision rule at the margin (validated in stage S).
POWER = 0.90
#: Bootstrap draws of the empirical null (confirmatory analysis).
N_BOOT = 10_000
#: Each of the four null-stability tests (location and scale, two geometries) at ALPHA / 4.
STABILITY_ALPHA = ALPHA / 4
#: Each of the two model checks (calibration block, confirmatory sealed controls), two-sided.
MODEL_CHECK_ALPHA = 0.005
#: The Monte Carlo SE of every theta0_D must be at most this fraction of the lag-1 margin.
THETA0_SE_MAX_FRACTION = 0.1
#: Within-episode correlation statistic: the mean autocorrelation of the per-step pair difference at lags 1..this.
WCORR_LAGS = 3
#: Readout windows of the counts (docs/physics.md section 8): signal, reference.
SIGNAL_WINDOW = 0
REFERENCE_WINDOW = 2
#: A block is excluded when any step of any arm has a reference-window count (summed over the shots) below this.
REF_COUNT_FLOOR = 100
#: "Far from every line": more than this many times (Rabi frequency at the pulse amplitude + T2* FWHM).
OFF_LINE_WIDTHS = 3.0
#: Engineering comparisons (M7, E5): exactly this many treatment pairs and control repeats.
ENG_N = 20
ENG_N_CONTROL = 20

CONFIRMATORY_ARMS = ("A1", "A2", "A3", "A3b", "A3L", "A4", "A5")
CALIBRATION_ARMS = ("C1", "C2", "C3", "C3b", "C3L", "C4", "C5")
#: Per-pair statistics: r (mean referenced fluorescence difference), r_off (the same on steps far from every line),
#: lvr (log ratio of residual variances), task (task-error difference), wcorr (within-episode autocorrelation of the
#: per-step difference: the model check), resid (difference of mean model residuals, any two episodes of a block).
HASH_STATS = ("r", "r_off", "lvr", "task", "wcorr")
BLOCK_STATS = ("resid",)
#: Statistics without an empirical null (descriptive in every cell; wcorr is read only by the model check).
UNTESTED_STATS = ("wcorr",)


@dataclass(frozen=True)
class CellSpec:
    """A pre-registered comparison: statistic of ``x`` minus ``y``; ``lag`` 1 = ``y`` one block after ``x``;
    ``geometry`` "hash" pairs episodes with identical action sequences, "block" any two episodes of a block."""

    x: str
    y: str
    lag: int
    stat: str
    geometry: str = "hash"


PRIMARY = {
    "H_decision": CellSpec("A1", "A2", 1, "r"),
    "H_observer": CellSpec("A2", "A3", 0, "r"),
}
#: The concurrent null of each primary: the cell of the run's own sealed controls with the same geometry.
CONCURRENT = {
    "H_decision": CellSpec("A3", "A3L", 1, "r"),
    "H_observer": CellSpec("A3", "A3b", 0, "r"),
}
STABILITY = {
    "lag0": CellSpec("A3", "A3b", 0, "r"),
    "lag1": CellSpec("A3", "A3L", 1, "r"),
}
MODEL_CHECK = CellSpec("A3", "A3b", 0, "wcorr")
SECONDARY = {
    "S1_sham_vs_yoked": CellSpec("A4", "A2", 0, "resid", "block"),
    "S2_dispersion_decision": CellSpec("A1", "A2", 1, "lvr"),
    "S2_dispersion_observer": CellSpec("A2", "A3", 0, "lvr"),
    "S3_offline_decision": CellSpec("A1", "A2", 1, "r_off"),
    "S3_offline_observer": CellSpec("A2", "A3", 0, "r_off"),
    "S4_quantum_test": CellSpec("A5", "A2", 0, "task"),
}
#: The calibration block mirrors a confirmatory block slot for slot, every arm replaced by a sealed replay without
#: an applied decision: C1 (offline sequence k, in A1's role), C2, C3, C3b (sequence k - 1), C3L (k - 2), C4 (sealed
#: sham), C5 (sequence k - 1 detuned). Its pairs give the null of lag-0 cells (C2, C3: the geometry of H_observer),
#: of lag-1 cells (C1 in block k, C2 in block k + 1: the geometry of H_decision) and of block cells (C4, C2: S1);
#: C_ref comes from the detuned control C5 against C2.
CAL_PAIRS = {0: ("C2", "C3"), 1: ("C1", "C2")}
CAL_BLOCK_PAIR = ("C4", "C2")
CAL_CREF = ("C5", "C2")
#: Model runs: confirmatory-design runs of a frozen model, arms named in the clear (a model run has no arm to hide).
MODEL_DECISION_PAIR = ("A1", "A2")
MODEL_WCORR_PAIR = ("A3", "A3b")


# --- Data -----------------------------------------------------------------------------------------------------------


@dataclass
class EpisodeTable:
    """One row per episode (unsealed outcomes reduced to per-step observables by :func:`build_episode_table`).

    ``r`` (n, T) referenced fluorescence N_sig / N_ref per step; ``r_model`` (n, T) the frozen model's prediction
    of r for the applied action; ``off_line`` (n, T) True where the applied MW frequency is far from every line
    (frozen rule of the plan); ``task_error`` (n,) the episode's task error; ``seq_hash`` the sha256 of the nominal
    applied action sequence; ``invalid_blocks`` block -> reason (excluded as a whole, arm-blind flags)."""

    block: np.ndarray
    position: np.ndarray
    code: np.ndarray
    seq_hash: np.ndarray
    r: np.ndarray
    r_model: np.ndarray
    off_line: np.ndarray
    task_error: np.ndarray
    invalid_blocks: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.block = np.asarray(self.block, dtype=np.int64)
        self.position = np.asarray(self.position, dtype=np.int64)
        self.code = np.asarray(self.code, dtype=object)
        self.seq_hash = np.asarray(self.seq_hash, dtype=object)
        self.r = np.asarray(self.r, dtype=np.float64)
        self.r_model = np.asarray(self.r_model, dtype=np.float64)
        self.off_line = np.asarray(self.off_line, dtype=bool)
        self.task_error = np.asarray(self.task_error, dtype=np.float64)
        n = self.block.size
        for name in ("position", "code", "seq_hash", "task_error"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have one entry per episode")
        if self.r.ndim != 2 or self.r.shape[0] != n or self.r_model.shape != self.r.shape:
            raise ValueError("r and r_model must be (n_episodes, T)")
        if self.off_line.shape != self.r.shape:
            raise ValueError("off_line must be (n_episodes, T)")

    @property
    def codes(self) -> list[str]:
        return sorted(set(map(str, self.code)))

    def valid(self, block: int) -> bool:
        return block not in self.invalid_blocks and block >= 0

    def to_arrow(self) -> pa.Table:
        meta = {b"invalid_blocks": json.dumps({str(k): v for k, v in self.invalid_blocks.items()}).encode()}
        t = pa.table(
            {
                "block": self.block,
                "position": self.position,
                "code": [str(c) for c in self.code],
                "seq_hash": [str(h) for h in self.seq_hash],
                "r": list(self.r),
                "r_model": list(self.r_model),
                "off_line": list(self.off_line),
                "task_error": self.task_error,
            }
        )
        return t.replace_schema_metadata(meta)

    def save(self, path: Path | str) -> Path:
        pq.write_table(self.to_arrow(), path)
        return Path(path)

    @classmethod
    def load(cls, path: Path | str) -> EpisodeTable:
        t = pq.read_table(path)
        meta = t.schema.metadata or {}
        inv = {int(k): str(v) for k, v in json.loads(meta.get(b"invalid_blocks", b"{}")).items()}
        d = t.to_pydict()
        return cls(
            np.array(d["block"]),
            np.array(d["position"]),
            np.array(d["code"], dtype=object),
            np.array(d["seq_hash"], dtype=object),
            np.array(d["r"], dtype=np.float64),
            np.array(d["r_model"], dtype=np.float64),
            np.array(d["off_line"], dtype=bool),
            np.array(d["task_error"], dtype=np.float64),
            inv,
        )

    def sha256(self) -> str:
        h = hashlib.sha256()
        for a in (self.block, self.position, self.r, self.r_model, self.off_line, self.task_error):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(json.dumps([list(map(str, self.code)), list(map(str, self.seq_hash))]).encode())
        h.update(json.dumps(sorted(self.invalid_blocks.items())).encode())
        return h.hexdigest()


class FrozenModel(Protocol):
    """The frozen physics model the reduction uses (its module must be in the plan lock): the nominal MW frequency of
    an action, the model's referenced fluorescence of an action at an MW offset, the line positions and widths."""

    #: (L,) transition frequencies of the sample (Hz).
    lines_hz: np.ndarray
    #: Rabi frequency at the pulse amplitude + T2* FWHM (Hz).
    line_width_hz: float

    def mw_hz(self, actions: np.ndarray) -> np.ndarray: ...

    def r_model(self, actions: np.ndarray, offset_hz: float) -> np.ndarray: ...


def build_episode_table(
    code: Sequence[str] | np.ndarray,
    block: Sequence[int] | np.ndarray,
    position: Sequence[int] | np.ndarray,
    actions: np.ndarray,
    offset_hz: Sequence[float] | np.ndarray,
    counts: np.ndarray,
    model: FrozenModel,
    warmup_blocks: int = WARMUP_BLOCKS,
    flags: Mapping[int, str] | None = None,
) -> EpisodeTable:
    """The frozen reduction (stage 0) of unsealed episodes to an :class:`EpisodeTable`.

    ``code`` (n,) the episode's arm code, ``block`` / ``position`` (n,), ``actions`` (n, T) the applied action
    indices, ``offset_hz`` (n,) the synthesiser offset of the episode (0 nominal; the detuned arm's offset), ``counts``
    (n, T, W) the unsealed counts per window summed over the shots. ``flags`` are acquisition flags per block
    (interlock trip, counter overflow, sequencer underrun or loss of sync), set by the acquisition code blind to
    outcomes.

    Rules: r = N_sig / max(N_ref, 1); r_model = the model at the applied action and offset; off_line = applied MW
    frequency more than :data:`OFF_LINE_WIDTHS` line widths from every line; task error = distance (MHz) from the
    nominal frequency of the step with the lowest r to the nearest line; seq_hash = :func:`action_hash` of the
    applied actions. A block is excluded as a whole (all arms) when it is a warm-up block, carries an acquisition
    flag, misses an arm or a step, has a reference count below :data:`REF_COUNT_FLOOR`, or a non-finite r."""
    code_a = np.asarray(code, dtype=object)
    block_a = np.asarray(block, dtype=np.int64)
    pos_a = np.asarray(position, dtype=np.int64)
    act = np.asarray(actions, dtype=np.int64)
    off = np.asarray(offset_hz, dtype=np.float64)
    cnt = np.asarray(counts, dtype=np.float64)
    n = block_a.size
    if act.ndim != 2 or act.shape[0] != n or cnt.shape[:2] != act.shape or off.shape != (n,):
        raise ValueError("actions (n, T), counts (n, T, W) and offset_hz (n,) must match the episodes")
    sig, ref = cnt[..., SIGNAL_WINDOW], cnt[..., REFERENCE_WINDOW]
    r = sig / np.maximum(ref, 1.0)
    r_model = np.empty_like(r)
    for o in np.unique(off):
        m = off == o
        r_model[m] = model.r_model(act[m], float(o))
    lines = np.asarray(model.lines_hz, dtype=np.float64).ravel()
    nominal = np.asarray(model.mw_hz(act), dtype=np.float64)
    applied = nominal + off[:, None]
    dist = np.min(np.abs(applied[..., None] - lines), axis=-1)
    off_line = dist > OFF_LINE_WIDTHS * float(model.line_width_hz)
    f_est = nominal[np.arange(n), np.argmin(r, axis=1)]
    task = np.min(np.abs(f_est[:, None] - lines[None, :]), axis=1) / 1e6
    seq = np.array([action_hash(a) for a in act], dtype=object)
    invalid: dict[int, str] = {}
    all_codes = set(map(str, code_a))
    for k in sorted(set(map(int, block_a))):
        rows = block_a == k
        reason = None
        if k < warmup_blocks:
            reason = "warm-up"
        elif flags and k in flags:
            reason = str(flags[k])
        elif set(map(str, code_a[rows])) != all_codes or int(rows.sum()) != len(all_codes):
            reason = "incomplete block"
        elif np.any(ref[rows] < REF_COUNT_FLOOR):
            reason = "reference count below the floor"
        elif not np.all(np.isfinite(r[rows])):
            reason = "non-finite values"
        if reason is not None:
            invalid[k] = reason
    return EpisodeTable(block_a, pos_a, code_a, seq, r, r_model, off_line, task, invalid)


# --- Pair statistics -------------------------------------------------------------------------------------------------


def _row_index(table: EpisodeTable) -> dict[tuple[str, int], int]:
    return {(str(c), int(b)): i for i, (c, b) in enumerate(zip(table.code, table.block, strict=True))}


def pair_rows(table: EpisodeTable, x: str, y: str, lag: int, geometry: str = "hash") -> tuple[np.ndarray, np.ndarray]:
    """Row indices (ix, iy) of the pairs of a cell, in block order: x in block k, y in block k + lag, both blocks
    valid, and (geometry "hash") identical action sequences."""
    idx = _row_index(table)
    ix, iy = [], []
    for k in sorted(set(map(int, table.block))):
        i, j = idx.get((x, k)), idx.get((y, k + lag))
        if i is None or j is None or not (table.valid(k) and table.valid(k + lag)):
            continue
        if geometry == "hash" and table.seq_hash[i] != table.seq_hash[j]:
            continue
        ix.append(i)
        iy.append(j)
    return np.asarray(ix, dtype=np.int64), np.asarray(iy, dtype=np.int64)


def within_episode_corr(z: np.ndarray, lags: int = WCORR_LAGS) -> np.ndarray:
    """(n,) mean autocorrelation at lags 1..``lags`` of each row of ``z`` (n, T), each row demeaned."""
    e = z - z.mean(axis=1, keepdims=True)
    den = np.sum(e * e, axis=1)
    acs = [np.sum(e[:, j:] * e[:, :-j], axis=1) for j in range(1, lags + 1)]
    out: np.ndarray = np.where(den > 0, np.mean(acs, axis=0) / np.where(den > 0, den, 1.0), np.nan)
    return out


def pair_stat(table: EpisodeTable, ix: np.ndarray, iy: np.ndarray, stat: str) -> np.ndarray:
    """Per-pair statistic (n_pairs,), NaN where undefined (e.g. no off-line step)."""
    if ix.size == 0:
        return np.zeros(0)
    rx, ry = table.r[ix], table.r[iy]
    if stat == "r":
        return (rx - ry).mean(axis=1)
    if stat == "r_off":
        m = table.off_line[ix] & table.off_line[iy]
        cnt = m.sum(axis=1)
        s = np.where(m, rx - ry, 0.0).sum(axis=1)
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    if stat == "wcorr":
        return within_episode_corr(rx - ry)
    ex, ey = rx - table.r_model[ix], ry - table.r_model[iy]
    if stat == "resid":
        return ex.mean(axis=1) - ey.mean(axis=1)
    if stat == "lvr":
        return np.log(ex.var(axis=1, ddof=1)) - np.log(ey.var(axis=1, ddof=1))
    if stat == "task":
        return table.task_error[ix] - table.task_error[iy]
    raise ValueError(f"unknown statistic {stat!r}")


# --- Empirical null, tests -------------------------------------------------------------------------------------------


def block_length(m: int) -> int:
    """Moving-block length for a series of m pairs: floor(4 (m / 100)^(2/9)), at least 1 (the Newey-West rule)."""
    return max(1, math.floor(4.0 * (m / 100.0) ** (2.0 / 9.0)))


def null_means(series: np.ndarray, n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot,) means of n pairs under the null: circular moving blocks of the control series (NaN dropped), each
    block multiplied by a random sign (exchangeable controls: symmetric about zero)."""
    s = np.asarray(series, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0 or n <= 0:
        return np.full(n_boot, np.nan)
    ln = min(block_length(s.size), s.size)
    nb = -(-n // ln)
    starts = rng.integers(0, s.size, size=(n_boot, nb))
    idx = (starts[..., None] + np.arange(ln)).reshape(n_boot, nb * ln)[:, :n] % s.size
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_boot, nb))
    sg = np.repeat(signs, ln, axis=1)[:, :n]
    out: np.ndarray = (s[idx] * sg).mean(axis=1)
    return out


def empirical_p(d: float, null: np.ndarray) -> float:
    """Two-sided p of a centred statistic d: (1 + #{|null| >= |d|}) / (B + 1)."""
    nb = null[np.isfinite(null)]
    if not np.isfinite(d) or nb.size == 0:
        return float("nan")
    return float((1 + np.sum(np.abs(nb) >= abs(d))) / (nb.size + 1))


def tost(d: float, null: np.ndarray, margin: float, alpha: float) -> dict[str, float | bool]:
    """Two one-sided tests of |theta| < margin at ``alpha`` per side, with the empirical null as the sampling
    distribution of the centred estimate: equivalent when d - q > -margin and d + q < margin, q the (1 - alpha)
    quantile of the null. Returns both one-sided p-values too (p_lower for H0: theta <= -margin)."""
    nb = null[np.isfinite(null)]
    if not np.isfinite(d) or nb.size == 0 or not np.isfinite(margin):
        return {"equivalent": False, "q": float("nan"), "p_lower": float("nan"), "p_upper": float("nan")}
    q = float(np.quantile(nb, 1.0 - alpha))
    p_lower = float((1 + np.sum(nb >= d + margin)) / (nb.size + 1))
    p_upper = float((1 + np.sum(nb <= d - margin)) / (nb.size + 1))
    return {"equivalent": bool(d - q > -margin and d + q < margin), "q": q, "p_lower": p_lower, "p_upper": p_upper}


def newey_west_se(y: np.ndarray) -> float:
    """Standard error of the mean of a series with Bartlett-kernel HAC variance, lag :func:`block_length`."""
    v = np.asarray(y, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return float("nan")
    e = v - v.mean()
    lag = min(block_length(n), n - 1)
    s = float(e @ e) / n
    for j in range(1, lag + 1):
        s += 2.0 * (1.0 - j / (lag + 1.0)) * float(e[j:] @ e[:-j]) / n
    return math.sqrt(max(s, 0.0) / n)


def holm(pvalues: Mapping[str, float], alpha: float) -> dict[str, bool]:
    """Holm step-down: name -> rejected. NaN p-values are never rejected and count in the family size."""
    names = sorted(pvalues, key=lambda k: (not np.isfinite(pvalues[k]), pvalues[k]))
    m = len(names)
    out = {k: False for k in names}
    for i, k in enumerate(names):
        p = pvalues[k]
        if not np.isfinite(p) or p > alpha / (m - i):
            break
        out[k] = True
    return out


def conclusion(rejected: bool, equivalent: bool, stable: bool = True, model_ok: bool = True) -> str:
    """The plan's decision table. ``rejected``: the complete positive rule (Holm against the calibration controls
    and against the run's own sealed controls); ``equivalent``: TOST at the frozen margin; ``stable``: the
    null-stability check; ``model_ok``: the model check (H_decision only)."""
    if not model_ok:
        return "inconclusive: the drift model does not reproduce the sealed controls (model check failed)"
    if rejected and not equivalent:
        return "effect detected"
    if rejected and equivalent:
        return "significant, but smaller than the margin (no effect of the pre-registered size)"
    if equivalent and stable:
        return "null: any effect is smaller than the margin"
    if equivalent:
        return "inconclusive: equivalent, but the null was not stable (no null claim)"
    return "inconclusive"


def control_margin(series: np.ndarray, n_pairs: int = N_CONF, seed: int = 0, n_boot: int = N_BOOT) -> float:
    """The margin rule (smallest effect size of interest, derived from the control runs): with D the empirical null
    of the mean of ``n_pairs`` control-vs-control pairs (:func:`null_means` of the calibration series),
    Delta = Q_{1 - ALPHA/2}(|D|) + Q_{2 MARGIN_POWER - 1}(|D|): the critical value of the test at Holm's first level
    plus the shift that puts MARGIN_POWER of the estimate's distribution beyond it. An effect of Delta is detected
    against the controls with power MARGIN_POWER at the planned N; smaller ones are below the resolution the plan
    is built for."""
    s = np.asarray(series, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    a = np.abs(null_means(s, n_pairs, n_boot, np.random.default_rng(seed)))
    return float(np.quantile(a, 1.0 - ALPHA / 2) + np.quantile(a, 2.0 * MARGIN_POWER - 1.0))


# --- Model predictions (theta0_D) -------------------------------------------------------------------------------------


@dataclass
class SelectionModel:
    """One member of the pre-registered set of drift and timing models, from its model runs: theta0 (the predicted
    selection effect of the H_decision cell) with its Monte Carlo SE, and the predicted within-episode correlation
    of the sealed-control pairs for the model check: the mean over runs of each run's mean wcorr, the SD of those
    run means (which carries any correlation between the pairs of a run), the mean number of pairs per run, and the
    SE of the mean."""

    name: str
    theta0: float
    theta0_se: float
    wcorr_mean: float
    wcorr_run_sd: float
    wcorr_run_pairs: float
    wcorr_se: float
    n_runs: int
    runs: list[dict[str, str]] = field(default_factory=list)

    def envelope(self, n: int) -> tuple[float, float]:
        """Two-sided (1 - MODEL_CHECK_ALPHA) range of the mean wcorr of a run of n pairs under this model (the
        run-to-run variance scaled from the model runs' size to n)."""
        var = self.wcorr_run_sd**2 * self.wcorr_run_pairs / n + self.wcorr_se**2
        half = float(stats.norm.ppf(1 - MODEL_CHECK_ALPHA / 2)) * math.sqrt(var)
        return self.wcorr_mean - half, self.wcorr_mean + half

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SelectionModel:
        return cls(
            str(d["name"]),
            float(d["theta0"]),
            float(d["theta0_se"]),
            float(d["wcorr_mean"]),
            float(d["wcorr_run_sd"]),
            float(d["wcorr_run_pairs"]),
            float(d["wcorr_se"]),
            int(d["n_runs"]),
            [dict(x) for x in d.get("runs", [])],
        )


def selection_model(name: str, tables: Sequence[EpisodeTable], run_ids: Sequence[str] | None = None) -> SelectionModel:
    """Aggregate the model runs of one model (arms named in the clear): theta per run = mean of the H_decision pair
    statistic (A1 in block k against A2 in block k + 1); wcorr from the (A3, A3b) pairs."""
    thetas, wmeans, npairs = [], [], []
    for t in tables:
        ix, iy = pair_rows(t, *MODEL_DECISION_PAIR, 1)
        v = pair_stat(t, ix, iy, "r")
        thetas.append(float(np.nanmean(v)) if v.size else np.nan)
        ix, iy = pair_rows(t, *MODEL_WCORR_PAIR, 0)
        w = pair_stat(t, ix, iy, "wcorr")
        w = w[np.isfinite(w)]
        wmeans.append(float(w.mean()) if w.size else np.nan)
        npairs.append(w.size)
    th = np.asarray(thetas)
    th = th[np.isfinite(th)]
    wm = np.asarray(wmeans)
    wm = wm[np.isfinite(wm)]
    if th.size < 2 or wm.size < 2:
        raise ValueError(f"model {name!r}: at least two usable model runs are needed")
    ids = list(run_ids) if run_ids is not None else [str(i) for i in range(len(tables))]
    return SelectionModel(
        name,
        float(th.mean()),
        float(th.std(ddof=1) / math.sqrt(th.size)),
        float(wm.mean()),
        float(wm.std(ddof=1)),
        float(np.mean(npairs)),
        float(wm.std(ddof=1) / math.sqrt(wm.size)),
        int(th.size),
        [{"run_id": i, "data_sha256": t.sha256()} for i, t in zip(ids, tables, strict=True)],
    )


def model_check(w_mean: float, n: int, models: Sequence[SelectionModel]) -> bool:
    """True when the observed mean wcorr of n sealed-control pairs lies inside the envelope of at least one model of
    the set (the model set reproduces the within-episode correlation of the controls)."""
    if not np.isfinite(w_mean) or n <= 0:
        return False
    return any(lo <= w_mean <= hi for lo, hi in (m.envelope(n) for m in models))


# --- Calibration ------------------------------------------------------------------------------------------------------


@dataclass
class Calibration:
    """The frozen output of the calibration block. ``series[(lag, stat)]`` are the control-vs-control pair series
    (block order); ``margin[lag]`` the smallest effect size of interest of the primary statistic per geometry;
    ``sd_mean[lag]`` the control-vs-control SD of the mean of N_CONF pairs; ``models`` the theta0_D set;
    ``model_ok`` the model check of the calibration block (wcorr of (C2, C3) inside the set's envelope)."""

    series: dict[tuple[int, str], np.ndarray]
    margin: dict[int, float]
    sd_mean: dict[int, float]
    sigma: dict[int, float]
    vif: dict[int, float]
    c_ref: float
    n_blocks: int
    boot_seed: int
    data_sha256: str
    models: list[SelectionModel]
    model_ok: bool
    wcorr: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": PLAN,
            "data_sha256": self.data_sha256,
            "margin": {str(k): v for k, v in self.margin.items()},
            "margin_over_c_ref": {str(k): v / self.c_ref for k, v in self.margin.items()} if self.c_ref else None,
            "sd_mean": {str(k): v for k, v in self.sd_mean.items()},
            "sigma": {str(k): v for k, v in self.sigma.items()},
            "vif": {str(k): v for k, v in self.vif.items()},
            "c_ref": self.c_ref,
            "n_blocks": self.n_blocks,
            "boot_seed": self.boot_seed,
            "models": [m.to_dict() for m in self.models],
            "model_ok": self.model_ok,
            "wcorr": self.wcorr,
            "series": {f"{k[0]}:{k[1]}": [float(x) for x in v] for k, v in self.series.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Calibration:
        series = {}
        for key, v in d["series"].items():
            lag, stat = key.split(":")
            series[(int(lag), stat)] = np.asarray(v, dtype=np.float64)
        return cls(
            series,
            {int(k): float(v) for k, v in d["margin"].items()},
            {int(k): float(v) for k, v in d["sd_mean"].items()},
            {int(k): float(v) for k, v in d["sigma"].items()},
            {int(k): float(v) for k, v in d["vif"].items()},
            float(d["c_ref"]),
            int(d["n_blocks"]),
            int(d["boot_seed"]),
            str(d["data_sha256"]),
            [SelectionModel.from_dict(m) for m in d["models"]],
            bool(d["model_ok"]),
            float(d["wcorr"]),
        )

    def sha256(self) -> str:
        return sha256_bytes(json.dumps(self.to_dict(), sort_keys=True).encode())


def derive_calibration(table: EpisodeTable, models: Sequence[SelectionModel], n_boot: int = N_BOOT) -> Calibration:
    """Freeze the null, the margins, the theta0_D set and the model check from the calibration block (codes = the
    calibration arm names: every calibration arm is a control, so there is nothing to blind). Raises ValueError when
    the model set is empty or a theta0_D's Monte Carlo SE exceeds THETA0_SE_MAX_FRACTION x the lag-1 margin."""
    if not models:
        raise ValueError("the theta0_D model set is empty")
    h = table.sha256()
    series: dict[tuple[int, str], np.ndarray] = {}
    margin, sd_mean, sigma, vif = {}, {}, {}, {}
    for lag, (x, y) in CAL_PAIRS.items():
        ix, iy = pair_rows(table, x, y, lag)
        for st in HASH_STATS:
            series[(lag, st)] = pair_stat(table, ix, iy, st)
        s = series[(lag, "r")]
        s = s[np.isfinite(s)]
        seed = int(h[:15], 16) + lag
        margin[lag] = control_margin(s, N_CONF, seed, n_boot)
        sd_mean[lag] = float(np.std(null_means(s, N_CONF, n_boot, np.random.default_rng(seed)))) if s.size else math.nan
        sd = float(s.std(ddof=1)) if s.size > 1 else float("nan")
        sigma[lag] = sd
        vif[lag] = float((newey_west_se(s) * math.sqrt(s.size) / sd) ** 2) if s.size > 1 and sd > 0 else 1.0
    bx, by = CAL_BLOCK_PAIR
    ix, iy = pair_rows(table, bx, by, 0, "block")
    for st in BLOCK_STATS:
        series[(0, st)] = pair_stat(table, ix, iy, st)
    cx, cy = CAL_CREF
    ix, iy = pair_rows(table, cx, cy, 0)
    c_ref = float(np.nanmean(pair_stat(table, ix, iy, "r"))) if ix.size else float("nan")
    for m in models:
        if not (m.theta0_se <= THETA0_SE_MAX_FRACTION * margin[1]):
            raise ValueError(
                f"model {m.name!r}: theta0_D SE {m.theta0_se:.3g} exceeds {THETA0_SE_MAX_FRACTION} x the lag-1 "
                f"margin {margin[1]:.3g}: more model runs are needed"
            )
    w = series[(0, "wcorr")]
    w = w[np.isfinite(w)]
    w_mean = float(w.mean()) if w.size else float("nan")
    ok = model_check(w_mean, int(w.size), models)
    return Calibration(series, margin, sd_mean, sigma, vif, c_ref, N_CONF, int(h[:15], 16), h, list(models), ok, w_mean)


# --- Blinded analysis -------------------------------------------------------------------------------------------------


@dataclass
class CellResult:
    x: str
    y: str
    lag: int
    stat: str
    geometry: str
    n: int
    mean: float
    #: Per centre ("0" and, for lag-1 cells, "theta0:<model>"): d, p, ci95, tost (margins of the primary statistic),
    #: Newey-West z and p, and (primary statistic) ``cross_p``: other cell "x|y" -> p against that cell's series.
    tests: dict[str, dict[str, Any]]
    #: Null-stability comparison of this cell's pair series with the calibration series (Brown-Forsythe scale p).
    scale_p: float

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("x", "y", "lag", "stat", "geometry", "n", "mean", "tests", "scale_p")}


@dataclass
class BlindedAnalysis:
    plan: str
    run_id: str
    data_sha256: str
    calibration_sha256: str
    calibration_lock_sha256: str
    lock_sha256: str
    script_sha256: str
    cells: list[CellResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan,
            "run_id": self.run_id,
            "data_sha256": self.data_sha256,
            "calibration_sha256": self.calibration_sha256,
            "calibration_lock_sha256": self.calibration_lock_sha256,
            "lock_sha256": self.lock_sha256,
            "script_sha256": self.script_sha256,
            "cells": [c.to_dict() for c in self.cells],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> BlindedAnalysis:
        return cls(
            str(d["plan"]),
            str(d["run_id"]),
            str(d["data_sha256"]),
            str(d["calibration_sha256"]),
            str(d["calibration_lock_sha256"]),
            str(d["lock_sha256"]),
            str(d["script_sha256"]),
            [CellResult(**cd) for cd in d["cells"]],
        )

    def sha256(self) -> str:
        return sha256_bytes(json.dumps(self.to_dict(), sort_keys=True, default=float).encode())

    def cell(self, x: str, y: str, lag: int, stat: str, geometry: str = "hash") -> tuple[CellResult, float] | None:
        """The cell of (x - y) and the sign to apply (-1 when it is stored as y - x, lag 0 only)."""
        for c in self.cells:
            if (c.lag, c.stat, c.geometry) != (lag, stat, geometry):
                continue
            if (c.x, c.y) == (x, y):
                return c, 1.0
            if lag == 0 and (c.x, c.y) == (y, x):
                return c, -1.0
        return None


def _centres(cal: Calibration, lag: int, n_boot: int) -> dict[str, tuple[float, np.ndarray]]:
    """Centre name -> (value, Monte Carlo noise added to the null draws)."""
    out: dict[str, tuple[float, np.ndarray]] = {"0": (0.0, np.zeros(n_boot))}
    if lag == 1:
        for i, m in enumerate(cal.models):
            rng = np.random.default_rng(cal.boot_seed + 1 + i)
            noise = rng.normal(0.0, m.theta0_se, size=n_boot) if m.theta0_se > 0 else np.zeros(n_boot)
            out[f"theta0:{m.name}"] = (m.theta0, noise)
    return out


def _cell_tests(
    vals: np.ndarray, null: np.ndarray, margin: float, centres: Mapping[str, tuple[float, np.ndarray]]
) -> dict[str, dict[str, Any]]:
    v = vals[np.isfinite(vals)]
    m = float(v.mean()) if v.size else float("nan")
    se = newey_west_se(v)
    out = {}
    for name, (c, extra) in centres.items():
        nl = null + extra
        d = m - c
        q95 = float(np.quantile(np.abs(nl[np.isfinite(nl)]), 0.95)) if np.isfinite(nl).any() else float("nan")
        z = d / se if se and np.isfinite(se) and se > 0 else float("nan")
        out[name] = {
            "d": d,
            "p": empirical_p(d, nl),
            "ci95": [d - q95, d + q95],
            "tost": tost(d, nl, margin, ALPHA_EQ),
            "nw_z": z,
            "nw_p": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan"),
        }
    return out


def _seed(*parts: Any) -> int:
    return int(sha256_bytes(":".join(map(str, parts)).encode())[:15], 16)


def analyse(
    table: EpisodeTable,
    cal: Calibration,
    n_boot: int = N_BOOT,
    provenance: Mapping[str, str] | None = None,
) -> BlindedAnalysis:
    """Every cell of every code pair, blinded (see the module docstring). Deterministic given the data, the frozen
    calibration and ``n_boot`` (bootstrap seeds: from the calibration's). ``provenance`` fills run_id and the
    lock, calibration-lock and script hashes (:func:`analyse_run`)."""
    codes = table.codes
    null_cache: dict[tuple[int, str, int], np.ndarray] = {}

    def null_for(lag: int, stat: str, n: int) -> np.ndarray:
        key = (lag, stat, n)
        if key not in null_cache:
            rng = np.random.default_rng(_seed(cal.boot_seed, lag, stat, n))
            null_cache[key] = null_means(cal.series[(lag, stat)], n, n_boot, rng)
        return null_cache[key]

    cells: list[CellResult] = []
    r_vals: dict[tuple[int, str], np.ndarray] = {}
    for lag in (0, 1):
        centres = _centres(cal, lag, n_boot)
        for x in codes:
            for y in codes:
                if x == y or (lag == 0 and x > y):
                    continue
                for geometry, stat_list in (("hash", HASH_STATS), ("block", BLOCK_STATS)):
                    if geometry == "block" and lag != 0:
                        continue
                    ix, iy = pair_rows(table, x, y, lag, geometry)
                    if ix.size == 0:
                        continue
                    for st in stat_list:
                        vals = pair_stat(table, ix, iy, st)
                        fv = vals[np.isfinite(vals)]
                        n = int(fv.size)
                        if n == 0:
                            continue
                        tests: dict[str, dict[str, Any]] = {}
                        scale_p = math.nan
                        if st not in UNTESTED_STATS:
                            margin = cal.margin[lag] if st == "r" else float("nan")
                            tests = _cell_tests(vals, null_for(lag, st, n), margin, centres)
                            cs = cal.series[(lag, st)]
                            cs = cs[np.isfinite(cs)]
                            scale_p = float(stats.levene(fv, cs, center="median").pvalue) if n > 2 else math.nan
                        if st == "r" and geometry == "hash":
                            r_vals[(lag, f"{x}|{y}")] = fv
                        cells.append(CellResult(x, y, lag, st, geometry, n, float(fv.mean()), tests, scale_p))
    # concurrent nulls: every primary-statistic cell against every other cell of the same geometry of this run
    cross_cache: dict[tuple[int, str, int], np.ndarray] = {}
    for c in cells:
        if c.stat != "r" or c.geometry != "hash":
            continue
        centres = _centres(cal, c.lag, n_boot)
        for name, (centre, extra) in centres.items():
            cp = {}
            for (lag, key), other in r_vals.items():
                if lag != c.lag or key == f"{c.x}|{c.y}":
                    continue
                ck = (lag, key, c.n)
                if ck not in cross_cache:
                    rng = np.random.default_rng(_seed(cal.boot_seed, "cross", lag, key, c.n))
                    cross_cache[ck] = null_means(other, c.n, n_boot, rng)
                cp[key] = empirical_p(c.mean - centre, cross_cache[ck] + extra)
            c.tests[name]["cross_p"] = cp
    prov = dict(provenance or {})
    return BlindedAnalysis(
        PLAN,
        prov.get("run_id", ""),
        table.sha256(),
        cal.sha256(),
        prov.get("calibration_lock_sha256", ""),
        prov.get("lock_sha256", ""),
        prov.get("script_sha256", ""),
        cells,
    )


# --- Unblinding -------------------------------------------------------------------------------------------------------


def _read(ba: BlindedAnalysis, inv: Mapping[str, str], spec: CellSpec, centre: str) -> dict[str, Any]:
    found = ba.cell(inv[spec.x], inv[spec.y], spec.lag, spec.stat, spec.geometry)
    if found is None:
        return {"cell": [spec.x, spec.y, spec.lag, spec.stat], "missing": True, "p": float("nan")}
    c, sign = found
    out: dict[str, Any] = {"cell": [spec.x, spec.y, spec.lag, spec.stat], "n": c.n, "mean": sign * c.mean}
    out["missing"] = False
    out["scale_p"] = c.scale_p
    if centre not in c.tests:
        out["p"] = float("nan")
        return out
    t = c.tests[centre]
    lo, hi = t["ci95"]
    out.update(
        {
            "d": sign * t["d"],
            "p": t["p"],
            "ci95": sorted([sign * lo, sign * hi]),
            "equivalent": bool(t["tost"]["equivalent"]),
            "nw_p": t["nw_p"],
            "cross_p": dict(t.get("cross_p", {})),
        }
    )
    return out


def _cross(r: Mapping[str, Any], inv: Mapping[str, str], spec: CellSpec) -> float:
    cp = r.get("cross_p", {})
    for key in (f"{inv[spec.x]}|{inv[spec.y]}", f"{inv[spec.y]}|{inv[spec.x]}"):
        if key in cp:
            return float(cp[key])
    return float("nan")


def _worst(ps: Sequence[float]) -> float:
    a = np.asarray(ps, dtype=np.float64)
    return float("nan") if a.size == 0 or not np.all(np.isfinite(a)) else float(a.max())


@dataclass
class Report:
    primary: dict[str, dict[str, Any]]
    stability: dict[str, Any]
    model_check: dict[str, Any]
    secondary: dict[str, dict[str, Any]]
    exploratory: list[dict[str, Any]]
    calibration: dict[str, Any]
    analysis_sha256: str
    registered_runs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def to_markdown(self) -> str:
        lines = [f"# {PLAN}: unblinded result", "", f"Blinded analysis sha256: `{self.analysis_sha256}`", ""]
        lines += ["## Primary hypotheses", ""]
        lines += [
            "Positive claim: Holm (FWER 0.05) against the calibration controls and against the run's own sealed "
            "controls. Null claim: TOST at the frozen margin and a stable null. H_decision: worst case over the "
            "theta0_D model set, and only with a passed model check.",
            "",
        ]
        lines += [
            "| hypothesis | n | estimate - theta0 | 95 % interval | p (calibration) | p (concurrent) | rejected "
            "| equivalent | margin (r) | margin / C_ref | conclusion |"
        ]
        lines += ["| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for k, r in self.primary.items():
            if r.get("missing"):
                lines.append(f"| {k} | missing | | | | | | | | | inconclusive |")
                continue
            lines.append(
                f"| {k} | {r.get('n', '')} | {r.get('d', float('nan')):.4g} | "
                f"[{r['ci95'][0]:.4g}, {r['ci95'][1]:.4g}] | {r['p']:.4g} | {r['p_concurrent']:.4g} | "
                f"{r['rejected']} | {r['equivalent']} | {r['margin']:.4g} | {r['margin_over_c_ref']:.3g} | "
                f"{r['conclusion']} |"
            )
        mc = self.model_check
        lines += [
            "",
            f"Model check (H_decision): calibration {'passed' if mc['calibration'] else 'FAILED'}, confirmatory "
            f"sealed controls {'passed' if mc['confirmatory'] else 'FAILED'} (wcorr {mc['wcorr']:.4g}).",
        ]
        lines += ["", f"Null stability: {'stable' if self.stability['stable'] else 'NOT stable'}", ""]
        for g, s in self.stability["cells"].items():
            lines.append(f"- {g}: location p {s['p']:.4g}, scale p {s['scale_p']:.4g}")
        lines += ["", "## Secondary family (Holm, FWER 0.05)", "", "| outcome | n | estimate | p | Holm |"]
        lines += ["| --- | --- | --- | --- | --- |"]
        for k, r in self.secondary.items():
            est = r.get("mean", float("nan"))
            lines.append(f"| {k} | {r.get('n', '')} | {est:.4g} | {r['p']:.4g} | {r['holm_rejected']} |")
        lines += ["", "## Every other cell (exploratory)", "", "| x | y | lag | stat | n | mean | p |"]
        lines += ["| --- | --- | --- | --- | --- | --- | --- |"]
        for e in self.exploratory:
            lines.append(
                f"| {e['x']} | {e['y']} | {e['lag']} | {e['stat']} | {e['n']} | {e['mean']:.4g} | {e['p']:.4g} |"
            )
        if self.registered_runs:
            lines += ["", "## Every registered run under the lock", "", "| run | mode | series | events |"]
            lines += ["| --- | --- | --- | --- |"]
            for rr in self.registered_runs:
                lines.append(f"| {rr['run_id']} | {rr['mode']} | {rr['series']} | {', '.join(rr['events'])} |")
        return "\n".join(lines) + "\n"


def unblind(
    ba: BlindedAnalysis,
    code_to_arm: Mapping[str, str],
    cal: Calibration,
    registered_runs: Sequence[Mapping[str, Any]] = (),
) -> Report:
    """Read off the pre-registered cells after the blinded output was committed (see the module docstring)."""
    inv = {a: c for c, a in code_to_arm.items()}
    stab = {g: _read(ba, inv, s, "0") for g, s in STABILITY.items()}
    stable = all(
        not s.get("missing") and s["p"] > STABILITY_ALPHA and not (s["scale_p"] <= STABILITY_ALPHA)
        for s in stab.values()
    )
    # wcorr is symmetric in the two arms: the stored orientation of the lag-0 cell does not matter (no sign)
    found = ba.cell(inv[MODEL_CHECK.x], inv[MODEL_CHECK.y], MODEL_CHECK.lag, MODEL_CHECK.stat)
    w_conf, n_w = (found[0].mean, found[0].n) if found is not None else (float("nan"), 0)
    conf_ok = model_check(w_conf, n_w, cal.models)
    mcheck = {"calibration": bool(cal.model_ok), "confirmatory": bool(conf_ok), "wcorr": w_conf}
    centres = {"H_decision": [f"theta0:{m.name}" for m in cal.models], "H_observer": ["0"]}
    primary: dict[str, dict[str, Any]] = {}
    for h, spec in PRIMARY.items():
        reads = [_read(ba, inv, spec, c) for c in centres[h]]
        if any(r.get("missing") or "d" not in r for r in reads):
            primary[h] = {**reads[0], "missing": True, "p": float("nan"), "p_concurrent": float("nan")}
            continue
        for r in reads:
            r["p_concurrent"] = _cross(r, inv, CONCURRENT[h])
        # worst case over the model set: the least favourable centre (largest p; NaN anywhere -> NaN)
        i_w = int(np.argmax([np.nan_to_num(r["p"], nan=np.inf) for r in reads]))
        res = {k: v for k, v in reads[i_w].items() if k != "cross_p"}
        res["p"] = _worst([r["p"] for r in reads])
        res["p_concurrent"] = _worst([r["p_concurrent"] for r in reads])
        res["equivalent"] = all(r["equivalent"] for r in reads)
        res["centre"] = centres[h][i_w]
        lag = spec.lag
        res["margin"] = cal.margin[lag]
        res["margin_over_c_ref"] = cal.margin[lag] / cal.c_ref if cal.c_ref else float("nan")
        primary[h] = res
    rej_cal = holm({k: v["p"] for k, v in primary.items()}, ALPHA)
    rej_conc = holm({k: v.get("p_concurrent", float("nan")) for k, v in primary.items()}, ALPHA)
    for k, v in primary.items():
        model_ok = bool(mcheck["calibration"] and mcheck["confirmatory"]) if k == "H_decision" else True
        v["holm_rejected"] = rej_cal[k]
        v["concurrent_rejected"] = rej_conc[k]
        v["model_ok"] = model_ok
        v["stable"] = stable
        v["rejected"] = bool(rej_cal[k] and rej_conc[k] and model_ok and not v.get("missing"))
        v.setdefault("equivalent", False)
        v["conclusion"] = (
            "inconclusive" if v.get("missing") else conclusion(v["rejected"], v["equivalent"], stable, model_ok)
        )
    # theta0_D is the model's prediction for the primary statistic only; secondary cells are centred at zero
    secondary = {k: _read(ba, inv, s, "0") for k, s in SECONDARY.items()}
    srej = holm({k: v["p"] for k, v in secondary.items()}, ALPHA)
    for k, v in secondary.items():
        v["holm_rejected"] = srej[k]
    explo = []
    for c in ba.cells:
        if c.stat in UNTESTED_STATS:
            continue
        explo.append(
            {
                "x": code_to_arm.get(c.x, c.x),
                "y": code_to_arm.get(c.y, c.y),
                "lag": c.lag,
                "stat": c.stat if c.geometry == "hash" else f"{c.stat} (block)",
                "n": c.n,
                "mean": c.mean,
                "p": c.tests["0"]["p"],
            }
        )
    cal_d = {k: v for k, v in cal.to_dict().items() if k != "series"}
    return Report(
        primary,
        {"stable": stable, "cells": stab},
        mcheck,
        secondary,
        explo,
        cal_d,
        ba.sha256(),
        [dict(r) for r in registered_runs],
    )


# --- Engineering comparisons (M7 controllers, E5) ---------------------------------------------------------------------


def paired_margin_test(
    diffs: np.ndarray,
    control_diffs: np.ndarray,
    alternative: str = "two-sided",
    alpha: float = ALPHA,
    n_perm: int = N_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    """A paired comparison judged against control repeats, at the fixed sizes ENG_N and ENG_N_CONTROL (refused
    otherwise: no sample size is chosen after seeing data).

    ``diffs`` (treatment - reference, per pair, e.g. per seed and field) and ``control_diffs`` (the same statistic
    between two repeats of the *same* controller under the same schedule; for E5, (estimate - estimate') / sqrt(2)).
    Under the null both are draws of the same distribution, so the test is a two-sample permutation test of
    mean(diffs) - mean(control_diffs): it carries the controls' own sampling uncertainty. ``alternative``:
    "two-sided", "greater" or "less". The margin comes from the controls only: Delta = (t_crit + t_power) x
    sd(control_diffs) x sqrt(1 / ENG_N + 1 / ENG_N_CONTROL), t_crit at alpha (two-sided: alpha / 2) and t_power at
    MARGIN_POWER, df = ENG_N_CONTROL - 1. TOST: the Welch (1 - 2 x alpha / 2) interval of the difference inside
    +-Delta."""
    d = np.asarray(diffs, dtype=np.float64)
    c = np.asarray(control_diffs, dtype=np.float64)
    if d.shape != (ENG_N,) or c.shape != (ENG_N_CONTROL,) or not (np.all(np.isfinite(d)) and np.all(np.isfinite(c))):
        raise ValueError(f"exactly {ENG_N} finite diffs and {ENG_N_CONTROL} finite control diffs are required")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unknown alternative {alternative!r}")
    obs = float(d.mean() - c.mean())
    pooled = np.concatenate([d, c])
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random((n_perm, pooled.size)), axis=1)
    pd_ = pooled[perm]
    null = pd_[:, :ENG_N].mean(axis=1) - pd_[:, ENG_N:].mean(axis=1)
    if alternative == "two-sided":
        p = float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-15)) / (n_perm + 1))
    elif alternative == "greater":
        p = float((1 + np.sum(null >= obs - 1e-15)) / (n_perm + 1))
    else:
        p = float((1 + np.sum(null <= obs + 1e-15)) / (n_perm + 1))
    df = ENG_N_CONTROL - 1
    t_crit = float(stats.t.ppf(1 - (alpha / 2 if alternative == "two-sided" else alpha), df))
    t_pow = float(stats.t.ppf(MARGIN_POWER, df))
    scale = math.sqrt(1.0 / ENG_N + 1.0 / ENG_N_CONTROL)
    margin = (t_crit + t_pow) * float(c.std(ddof=1)) * scale
    se = math.sqrt(d.var(ddof=1) / ENG_N + c.var(ddof=1) / ENG_N_CONTROL)
    w_df = se**4 / (
        (d.var(ddof=1) / ENG_N) ** 2 / (ENG_N - 1) + (c.var(ddof=1) / ENG_N_CONTROL) ** 2 / (ENG_N_CONTROL - 1)
    )
    q = float(stats.t.ppf(1 - alpha / 2, w_df)) * se
    return {
        "n": ENG_N,
        "n_control": ENG_N_CONTROL,
        "mean": obs,
        "p": p,
        "alternative": alternative,
        "margin": margin,
        "rejected": bool(p <= alpha),
        "tost": {"equivalent": bool(obs - q > -margin and obs + q < margin), "q": q},
    }


# --- Lock verification ------------------------------------------------------------------------------------------------


def frozen_constants() -> dict[str, Any]:
    """The constants this script freezes (recorded in the lock and checked against it)."""
    return {
        "ALPHA": ALPHA,
        "ALPHA_EQ": ALPHA_EQ,
        "N_CONF": N_CONF,
        "M_CAL": M_CAL,
        "WARMUP_BLOCKS": WARMUP_BLOCKS,
        "ATTEMPT_CAP": ATTEMPT_CAP,
        "MARGIN_POWER": MARGIN_POWER,
        "POWER": POWER,
        "N_BOOT": N_BOOT,
        "STABILITY_ALPHA": STABILITY_ALPHA,
        "MODEL_CHECK_ALPHA": MODEL_CHECK_ALPHA,
        "THETA0_SE_MAX_FRACTION": THETA0_SE_MAX_FRACTION,
        "WCORR_LAGS": WCORR_LAGS,
        "SIGNAL_WINDOW": SIGNAL_WINDOW,
        "REFERENCE_WINDOW": REFERENCE_WINDOW,
        "REF_COUNT_FLOOR": REF_COUNT_FLOOR,
        "OFF_LINE_WIDTHS": OFF_LINE_WIDTHS,
        "ENG_N": ENG_N,
        "ENG_N_CONTROL": ENG_N_CONTROL,
    }


def required_files(root: Path = REPO_ROOT) -> list[str]:
    """Files every PREREG-001 lock must contain: the plan, this script's module closure (package ``__init__`` files
    included) and the environment pins that exist under ``root``."""
    files = plan_files(PLAN_DOC, [SCRIPT_MODULE], root=root)
    return sorted(p.resolve().relative_to(root.resolve()).as_posix() for p in files)


def create_plan_lock(
    extra_modules: Sequence[str] = (),
    configs: Sequence[Path | str] = (),
    extra_files: Sequence[Path | str] = (),
    root: Path = REPO_ROOT,
) -> Lock:
    """The plan lock: the plan, the module closure of this script, of the arms and of ``extra_modules`` (the
    acquisition or runner script, the controller, the measurement and the frozen model a run uses), the
    ``configs`` and ``extra_files`` (e.g. the sham distribution), the environment pins, the frozen constants, and
    append-only pins of the deviations log and the run registry."""
    files = plan_files(
        PLAN_DOC, [SCRIPT_MODULE, "glowbox.analysis.arms", *extra_modules], configs, extra_files, root=root
    )
    return Lock.create(
        PLAN, files, {"analysis": frozen_constants()}, DEVIATIONS_PATH, root=root, registry=REGISTRY_PATH
    )


def verify_lock(lock: Lock, root: Path = REPO_ROOT) -> None:
    """Refuse to start unless the lock names this plan, contains every required file (:func:`required_files`) and
    pins the deviations log and the run registry, every file still matches, and the frozen constants are this
    script's."""
    probs = []
    if lock.name != PLAN:
        probs.append(f"lock is for {lock.name!r}, not {PLAN}")
    missing = [f for f in required_files(root) if f not in lock.files]
    if missing:
        probs.append(f"not in the lock: {', '.join(missing)}")
    if lock.deviations is None or lock.deviations.path != DEVIATIONS_PATH:
        probs.append("the lock does not pin the deviations log")
    if lock.registry is None or lock.registry.path != REGISTRY_PATH:
        probs.append("the lock does not pin the run registry")
    if probs:
        raise LockMismatchError(probs)
    lock.verify(root)
    if lock.constants.get("analysis") != frozen_constants():
        raise LockMismatchError(["the script's frozen constants differ from the locked ones"])


def check_plan(lock_path: Path, root: Path = REPO_ROOT) -> Lock:
    """Load and verify the plan lock (:func:`verify_lock`), require it committed and published under
    :data:`LOCK_TAG`, and require the run registry committed. Returns the lock."""
    lock = Lock.load(lock_path)
    verify_lock(lock, root)
    probs = published_lock_problems(lock_path, LOCK_TAG, root) + committed_unchanged(root / REGISTRY_PATH, root)
    if probs:
        raise LockMismatchError(probs)
    return lock


# --- Pipeline (each step refuses on any mismatch) ---------------------------------------------------------------------


def _load_info(run_dir: Path) -> dict[str, Any]:
    with open(run_dir / "run.yaml", encoding="utf-8") as f:
        info: dict[str, Any] = yaml.safe_load(f)
    return info


def _check_run(run_dir: Path, lock: Lock, mode: str, registry: Registry, root: Path) -> tuple[dict[str, Any], str]:
    info = _load_info(run_dir)
    check_run_record(info, lock, mode, root)
    run_id = str(info.get("run_id"))
    if probs := registry.run_problems(run_id, info, lock):
        raise LockMismatchError(probs)
    return info, run_id


CAL_LOCK_NAME = f"{PLAN}-calibration"


def freeze_calibration(
    cal_run: Path,
    model_runs: Mapping[str, Sequence[Path]],
    lock_path: Path,
    out_dir: Path,
    root: Path = REPO_ROOT,
    n_boot: int = N_BOOT,
) -> Lock:
    """Stage 1. ``model_runs`` maps each model of the pre-registered set to its run directories (mode "model");
    they must be *all* registered model runs of the calibration run's series, and every one must have started its
    data before the calibration block did. Writes ``<out_dir>/calibration.json`` and the calibration lock
    ``<out_dir>/calibration.lock.yaml`` (files: the calibration; constants: the plan lock sha256, the calibration
    run, its data sha256 and start time, the model runs and their data sha256), appends a ``calibration`` event to
    the registry, and returns the calibration lock. Commit both files and the registry before the confirmatory run
    starts."""
    lock = check_plan(lock_path, root)
    registry = Registry(root / REGISTRY_PATH)
    info, run_id = _check_run(cal_run, lock, "calibration", registry, root)
    rec = info["prereg"]
    series = rec["series"]
    registered = {e["run_id"] for e in registry.runs(lock.sha256(), "model", series)}
    models, given, model_prov = [], set(), []
    for name, dirs in model_runs.items():
        tables, ids = [], []
        for d in dirs:
            mi, mid = _check_run(d, lock, "model", registry, root)
            if mi["prereg"].get("model") != name:
                raise LockMismatchError([f"model run {mid} is of model {mi['prereg'].get('model')!r}, not {name!r}"])
            if str(mi["prereg"]["data_started_utc"]) >= str(rec["data_started_utc"]):
                raise LockMismatchError([f"model run {mid} did not start before the calibration block"])
            t = EpisodeTable.load(d / "episodes.parquet")
            tables.append(t)
            ids.append(mid)
            given.add(mid)
            model_prov.append({"run_id": mid, "model": name, "data_sha256": t.sha256()})
        models.append(selection_model(name, tables, ids))
    if given != registered:
        raise LockMismatchError(
            [f"model runs given {sorted(given)} are not all the registered model runs {sorted(registered)}"]
        )
    table = EpisodeTable.load(cal_run / "episodes.parquet")
    cal = derive_calibration(table, models, n_boot)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_path = out_dir / "calibration.json"
    cal_path.write_text(json.dumps(cal.to_dict(), sort_keys=True, indent=1))
    constants = {
        "plan_lock_sha256": lock.sha256(),
        "calibration_sha256": cal.sha256(),
        "calibration_run_id": run_id,
        "calibration_series": series,
        "calibration_data_sha256": table.sha256(),
        "calibration_data_started_utc": str(rec["data_started_utc"]),
        "model_runs": model_prov,
    }
    cal_lock = Lock.create(CAL_LOCK_NAME, [cal_path], constants, root=root)
    cal_lock.write(out_dir / "calibration.lock.yaml")
    registry.append("calibration", run_id, series=series, calibration_lock_sha256=cal_lock.sha256())
    return cal_lock


def _check_calibration(
    cal_lock_path: Path, lock: Lock, registry: Registry, root: Path
) -> tuple[Lock, Calibration, Path]:
    cal_lock = Lock.load(cal_lock_path)
    probs = []
    if cal_lock.name != CAL_LOCK_NAME:
        probs.append(f"{cal_lock_path} is not a {CAL_LOCK_NAME} lock")
    if cal_lock.constants.get("plan_lock_sha256") != lock.sha256():
        probs.append("the calibration lock was made under a different plan lock")
    probs += cal_lock.problems(root) + committed_unchanged(cal_lock_path, root)
    if len(cal_lock.files) != 1:
        probs.append("the calibration lock must hold exactly the calibration file")
    if probs:
        raise LockMismatchError(probs)
    (rel,) = cal_lock.files
    probs += committed_unchanged(root / rel, root)
    cal = Calibration.from_dict(json.loads((root / rel).read_text()))
    if cal.sha256() != cal_lock.constants.get("calibration_sha256"):
        probs.append("the calibration's content differs from the one frozen in its lock")
    if cal.data_sha256 != cal_lock.constants.get("calibration_data_sha256"):
        probs.append("the calibration was not derived from the calibration run's data")
    ev = [
        e
        for e in registry.events(str(cal_lock.constants.get("calibration_run_id")), "calibration")
        if e.get("calibration_lock_sha256") == cal_lock.sha256()
    ]
    if len(ev) != 1:
        probs.append("the calibration lock is not registered exactly once")
    if probs:
        raise LockMismatchError(probs)
    return cal_lock, cal, root / rel


def analyse_run(
    conf_run: Path,
    cal_run: Path,
    lock_path: Path,
    cal_lock_path: Path,
    out: Path,
    root: Path = REPO_ROOT,
    n_boot: int = N_BOOT,
) -> BlindedAnalysis:
    """Stage 2: the blinded analysis of a confirmatory run against its frozen calibration. Refuses unless the plan
    lock verifies and is published; the calibration lock is committed, registered, belongs to this plan lock and
    matches its calibration file; the calibration run passes its checks again and its data are the calibration's;
    the confirmatory run passes its checks, names this calibration lock, is in the calibration's series, and
    started after the calibration was registered. Writes the blinded output and appends its sha256 to the registry
    (commit both before the blinding key is released)."""
    lock = check_plan(lock_path, root)
    registry = Registry(root / REGISTRY_PATH)
    cal_lock, cal, _ = _check_calibration(cal_lock_path, lock, registry, root)
    cal_info, cal_id = _check_run(cal_run, lock, "calibration", registry, root)
    probs = []
    if cal_id != cal_lock.constants.get("calibration_run_id"):
        probs.append("the calibration run is not the one frozen in the calibration lock")
    if EpisodeTable.load(cal_run / "episodes.parquet").sha256() != cal.data_sha256:
        probs.append("the calibration run's data differ from the frozen calibration's")
    info, run_id = _check_run(conf_run, lock, "confirmatory", registry, root)
    rec = info["prereg"]
    if rec.get("calibration_lock_sha256") != cal_lock.sha256():
        probs.append("the confirmatory run was made under a different calibration")
    if rec.get("series") != cal_lock.constants.get("calibration_series"):
        probs.append("the confirmatory run is not in the calibration's series")
    cal_ev = registry.events(cal_id, "calibration")
    if not cal_ev or str(cal_ev[0]["utc"]) > str(rec.get("data_started_utc")):
        probs.append("the confirmatory run started before its calibration was frozen and registered")
    if str(cal_info["prereg"]["data_started_utc"]) >= str(rec.get("data_started_utc")):
        probs.append("the calibration block did not run before the confirmatory run")
    if registry.events(run_id, "blinded"):
        probs.append("this run already has a registered blinded analysis")
    if probs:
        raise LockMismatchError(probs)
    table = EpisodeTable.load(conf_run / "episodes.parquet")
    prov = {
        "run_id": run_id,
        "lock_sha256": lock.sha256(),
        "calibration_lock_sha256": cal_lock.sha256(),
        "script_sha256": sha256_file(root / SCRIPT_PATH),
    }
    ba = analyse(table, cal, n_boot, prov)
    out.write_text(json.dumps(ba.to_dict(), sort_keys=True, indent=1, default=float))
    registry.append("blinded", run_id, analysis_sha256=ba.sha256())
    return ba


def unblind_run(
    analysis: Path,
    conf_run: Path,
    lock_path: Path,
    cal_lock_path: Path,
    blind_key: bytes,
    out: Path,
    root: Path = REPO_ROOT,
) -> Report:
    """Stage 3. Refuses unless the plan lock verifies and is published, the blinded output is committed and its
    sha256 is the registered one (registry committed), it names this plan lock, calibration lock, script and run,
    and its data hash is the run's table. The report lists every registered run under the lock."""
    lock = check_plan(lock_path, root)
    registry = Registry(root / REGISTRY_PATH)
    cal_lock, cal, _ = _check_calibration(cal_lock_path, lock, registry, root)
    info, run_id = _check_run(conf_run, lock, "confirmatory", registry, root)
    probs = committed_unchanged(analysis, root)
    ba = BlindedAnalysis.from_dict(json.loads(analysis.read_text()))
    ev = registry.events(run_id, "blinded")
    if len(ev) != 1 or ev[0].get("analysis_sha256") != ba.sha256():
        probs.append("the blinded analysis is not the one registered for this run")
    if ba.run_id != run_id:
        probs.append("the blinded analysis is of another run")
    if ba.lock_sha256 != lock.sha256():
        probs.append("the blinded analysis was made under a different plan lock")
    if ba.calibration_lock_sha256 != cal_lock.sha256() or ba.calibration_sha256 != cal.sha256():
        probs.append("the blinded analysis was made against a different calibration")
    if ba.script_sha256 != sha256_file(root / SCRIPT_PATH):
        probs.append("the blinded analysis was made by a different script")
    if ba.data_sha256 != EpisodeTable.load(conf_run / "episodes.parquet").sha256():
        probs.append("the blinded analysis was made on different data")
    if probs:
        raise LockMismatchError(probs)
    mapping = unblind_codes(blind_key, info["prereg"]["keys"]["blind_sha256"], CONFIRMATORY_ARMS)
    rep = unblind(ba, mapping, cal, registry.summary(lock.sha256()))
    out.write_text(rep.to_markdown())
    return rep


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="freeze the calibration block and the theta0_D model set")
    c.add_argument("--run", required=True, type=Path)
    c.add_argument("--lock", required=True, type=Path)
    c.add_argument("--model-runs", required=True, nargs="+", metavar="MODEL=RUN_DIR", help="every registered model run")
    c.add_argument("--out-dir", required=True, type=Path)
    a = sub.add_parser("analyse", help="blinded analysis of a confirmatory run")
    a.add_argument("--run", required=True, type=Path)
    a.add_argument("--calibration-run", required=True, type=Path)
    a.add_argument("--lock", required=True, type=Path)
    a.add_argument("--calibration-lock", required=True, type=Path)
    a.add_argument("--out", required=True, type=Path)
    u = sub.add_parser("unblind", help="map codes to arms (after the blinded output is registered and committed)")
    u.add_argument("--analysis", required=True, type=Path)
    u.add_argument("--run", required=True, type=Path)
    u.add_argument("--lock", required=True, type=Path)
    u.add_argument("--calibration-lock", required=True, type=Path)
    u.add_argument("--blind-key-hex", required=True)
    u.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    if args.cmd == "calibrate":
        runs: dict[str, list[Path]] = {}
        for item in args.model_runs:
            name, _, d = item.partition("=")
            runs.setdefault(name, []).append(Path(d))
        cl = freeze_calibration(args.run, runs, args.lock, args.out_dir)
        print(f"calibration lock sha256 {cl.sha256()}: commit it, the calibration and the registry")
    elif args.cmd == "analyse":
        ba = analyse_run(args.run, args.calibration_run, args.lock, args.calibration_lock, args.out)
        print(f"blinded analysis sha256 {ba.sha256()}: commit it and the registry before the blinding key is released")
    else:
        unblind_run(
            args.analysis, args.run, args.lock, args.calibration_lock, bytes.fromhex(args.blind_key_hex), args.out
        )
        print(f"unblinded report: {args.out} (written {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')})")


if __name__ == "__main__":
    main()
