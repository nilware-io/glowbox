"""Frozen analysis of PREREG-001 (docs/preregistration/PREREG-001-observer-decision.md). Do not edit once the plan
is locked: any change needs a new lock version and an entry in docs/preregistration/deviations.md.

Stages, in this order (each refuses to start unless the lock verifies, :func:`verify_lock`):

1. :func:`derive_calibration` on the CALIBRATION block (control arms only: no decision process applied, no live
   reader of the results). It returns the control-vs-control pair series, which are the empirical null of every
   test, the smallest effect size of interest per comparison geometry (:data:`MARGIN_QUANTILE` quantile of
   |control-vs-control pair statistic|), the spin-signal scale C_ref (reporting only), the confirmatory sample size
   N and the bootstrap seed. Its output is frozen (hashed) before any decision or observer arm runs.
2. :func:`analyse` on the confirmatory run, *blinded*: arms appear only as codes. Every pair of codes whose episodes
   replay the same action sequence (equal ``seq_hash``), in the same block (lag 0) or in consecutive blocks
   (lag 1), is a *cell*; every cell gets every statistic, its empirical-null p-value, its 95 % interval, its
   equivalence (TOST) result and a parametric (Newey-West) robustness check. The output does not depend on which
   code is which arm.
3. :func:`unblind`: maps codes to arms with the released blinding key and reads off the pre-registered cells:
   the two primary hypotheses (Holm), the null-stability check, the secondary family (Holm), and every other cell
   as exploratory.

The empirical null: a cell's statistic is the mean over its pairs of a per-pair statistic; under the null
hypothesis the pairs are distributed like control-vs-control pairs of the same geometry, which are exchangeable
(random order inside blocks), so their distribution is symmetric about zero. The null distribution of the mean of n
pairs is a circular moving-block bootstrap of the calibration series with a random sign per bootstrap block
(:func:`null_means`): the spread and shape come from the control runs, the centre from the exchangeability. The one
term no control run can contain is the selection effect of a decision arm under drift (its choices depend on counts
that carry drift); it is the frozen model prediction theta0_D, and decision cells are tested at that centre (with its
Monte Carlo uncertainty added to the null draws).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from glowbox.analysis.prereg import Lock, LockMismatchError, sha256_bytes
from glowbox.analysis.runlog import REPO_ROOT

PLAN = "PREREG-001"
SCRIPT_PATH = "glowbox/analysis/prereg_analysis_001.py"

# --- Frozen constants of the plan ------------------------------------------------------------------------------------

#: Family-wise error rate of each family (primary, secondary), Holm within the family.
ALPHA = 0.05
#: TOST level per side for the two primaries (alpha / 2: Bonferroni over the two equivalence claims).
ALPHA_EQ = 0.025
#: Smallest effect size of interest: the MARGIN_QUANTILE quantile of |mean of MARGIN_PAIRS control-vs-control pair
#: statistics| in the calibration block (MARGIN_PAIRS = 1: of a single matched episode pair; :func:`control_margin`).
MARGIN_QUANTILE = 0.95
MARGIN_PAIRS = 1
#: Planned power (superiority at |theta - theta0| = margin; TOST at theta = theta0).
POWER = 0.90
#: Confirmatory valid blocks: at least this many, more when the power rule asks (:func:`required_blocks`).
N_MIN = 100
#: Calibration valid blocks.
M_CAL = 200
#: Allowance for excluded blocks in the sample size; collection stops at this multiple of N attempted blocks.
EXCLUSION_ALLOWANCE = 0.05
ATTEMPT_CAP = 1.2
#: Bootstrap draws of the empirical null (confirmatory analysis).
N_BOOT = 10_000
#: Each of the four null-stability tests (location and scale, two geometries) at ALPHA / 4.
STABILITY_ALPHA = ALPHA / 4

CONFIRMATORY_ARMS = ("A1", "A2", "A3", "A3b", "A3L", "A4", "A5")
CALIBRATION_ARMS = ("C1", "C2", "C3", "C4")
#: Per-pair statistics: r (mean referenced fluorescence difference), r_off (the same on steps far from every line),
#: resid (difference of mean model residuals), lvr (log ratio of residual variances), task (task-error difference).
HASH_STATS = ("r", "r_off", "lvr", "task")
BLOCK_STATS = ("resid",)


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
STABILITY = {
    "lag0": CellSpec("A3", "A3b", 0, "r"),
    "lag1": CellSpec("A3", "A3L", 1, "r"),
}
SECONDARY = {
    "S1_sham_vs_yoked": CellSpec("A4", "A2", 0, "resid", "block"),
    "S2_dispersion_decision": CellSpec("A1", "A2", 1, "lvr"),
    "S2_dispersion_observer": CellSpec("A2", "A3", 0, "lvr"),
    "S3_offline_decision": CellSpec("A1", "A2", 1, "r_off"),
    "S3_offline_observer": CellSpec("A2", "A3", 0, "r_off"),
    "S4_quantum_test": CellSpec("A5", "A2", 0, "task"),
}
#: Calibration pairs: the null of lag-0 cells (C1, C2: same block, same sequence) and lag-1 cells (C1 in block k,
#: C3 in block k + 1 replaying the same sequence); C_ref from the detuned control C4 against C1.
CAL_PAIRS = {0: ("C1", "C2"), 1: ("C1", "C3")}
CAL_CREF = ("C4", "C1")


# --- Data -----------------------------------------------------------------------------------------------------------


@dataclass
class EpisodeTable:
    """One row per episode (unsealed outcomes reduced to per-step observables).

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


def conclusion(rejected: bool, equivalent: bool) -> str:
    """The plan's decision table."""
    if rejected and not equivalent:
        return "effect detected"
    if rejected and equivalent:
        return "detectable but smaller than the margin: practically null"
    if equivalent:
        return "null: any effect is smaller than the margin"
    return "inconclusive"


def required_blocks(sigma: float, vif: float, margin: float) -> int:
    """Confirmatory valid blocks: max(N_MIN, analytic sizes for superiority power at the margin (two-sided,
    Bonferroni bound on Holm over two hypotheses) and TOST power at theta0), inflated by the exclusion allowance."""
    if not (np.isfinite(sigma) and np.isfinite(margin) and margin > 0):
        return N_MIN
    z = stats.norm.ppf
    n_sup = vif * sigma**2 * ((z(1 - ALPHA / 4) + z(POWER)) / margin) ** 2
    n_eq = vif * sigma**2 * ((z(1 - ALPHA_EQ) + z(1 - (1 - POWER) / 2)) / margin) ** 2
    return max(N_MIN, math.ceil(max(n_sup, n_eq) / (1 - EXCLUSION_ALLOWANCE)))


def control_margin(series: np.ndarray, n_pairs: int = MARGIN_PAIRS, seed: int = 0) -> float:
    """The margin rule: the :data:`MARGIN_QUANTILE` quantile of |mean of ``n_pairs`` control-vs-control pairs|.
    One pair: the empirical quantile of |series| itself; more: of the empirical null of the mean (seeded)."""
    s = np.asarray(series, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    if n_pairs == 1:
        return float(np.quantile(np.abs(s), MARGIN_QUANTILE))
    draws = null_means(s, n_pairs, N_BOOT, np.random.default_rng(seed))
    return float(np.quantile(np.abs(draws), MARGIN_QUANTILE))


# --- Calibration ------------------------------------------------------------------------------------------------------


@dataclass
class Calibration:
    """The frozen output of the calibration block. ``series[(lag, stat)]`` are the control-vs-control pair series
    (block order); ``margin[lag]`` the smallest effect size of interest of the primary statistic per geometry;
    ``theta0_d`` / ``theta0_d_se`` the model's selection-effect prediction for decision cells (stage S: from the
    null-calibration simulation; hardware: from the commissioned drift and timing models)."""

    series: dict[tuple[int, str], np.ndarray]
    margin: dict[int, float]
    sigma: dict[int, float]
    vif: dict[int, float]
    c_ref: float
    n_blocks: int
    boot_seed: int
    data_sha256: str
    theta0_d: float = 0.0
    theta0_d_se: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": PLAN,
            "data_sha256": self.data_sha256,
            "margin": {str(k): v for k, v in self.margin.items()},
            "margin_over_c_ref": {str(k): v / self.c_ref for k, v in self.margin.items()} if self.c_ref else None,
            "sigma": {str(k): v for k, v in self.sigma.items()},
            "vif": {str(k): v for k, v in self.vif.items()},
            "c_ref": self.c_ref,
            "n_blocks": self.n_blocks,
            "boot_seed": self.boot_seed,
            "theta0_d": self.theta0_d,
            "theta0_d_se": self.theta0_d_se,
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
            {int(k): float(v) for k, v in d["sigma"].items()},
            {int(k): float(v) for k, v in d["vif"].items()},
            float(d["c_ref"]),
            int(d["n_blocks"]),
            int(d["boot_seed"]),
            str(d["data_sha256"]),
            float(d.get("theta0_d", 0.0)),
            float(d.get("theta0_d_se", 0.0)),
        )

    def sha256(self) -> str:
        return sha256_bytes(json.dumps(self.to_dict(), sort_keys=True).encode())


def derive_calibration(table: EpisodeTable, theta0_d: float = 0.0, theta0_d_se: float = 0.0) -> Calibration:
    """Freeze the null, the margins and N from the calibration block (codes = the calibration arm names: every
    calibration arm is a control, so there is nothing to blind)."""
    series: dict[tuple[int, str], np.ndarray] = {}
    margin, sigma, vif = {}, {}, {}
    for lag, (x, y) in CAL_PAIRS.items():
        ix, iy = pair_rows(table, x, y, lag)
        for st in (*HASH_STATS, *BLOCK_STATS):
            series[(lag, st)] = pair_stat(table, ix, iy, st)
        s = series[(lag, "r")]
        s = s[np.isfinite(s)]
        margin[lag] = control_margin(s, MARGIN_PAIRS, int(table.sha256()[:15], 16) + lag)
        sd = float(s.std(ddof=1)) if s.size > 1 else float("nan")
        sigma[lag] = sd
        vif[lag] = float((newey_west_se(s) * math.sqrt(s.size) / sd) ** 2) if s.size > 1 and sd > 0 else 1.0
    cx, cy = CAL_CREF
    ix, iy = pair_rows(table, cx, cy, 0)
    c_ref = float(np.nanmean(pair_stat(table, ix, iy, "r"))) if ix.size else float("nan")
    n_blocks = max(required_blocks(sigma[g], max(vif[g], 1.0), margin[g]) for g in CAL_PAIRS)
    h = table.sha256()
    return Calibration(series, margin, sigma, vif, c_ref, n_blocks, int(h[:15], 16), h, theta0_d, theta0_d_se)


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
    #: Per centre ("0" and, for lag-1 cells, "theta0_d"): d, p, ci95, tost (margins of the primary statistic),
    #: Newey-West z and p.
    tests: dict[str, dict[str, Any]]
    #: Null-stability comparison of this cell's pair series with the calibration series: location p (empirical
    #: null) and scale p (Brown-Forsythe).
    scale_p: float

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("x", "y", "lag", "stat", "geometry", "n", "mean", "tests", "scale_p")}


@dataclass
class BlindedAnalysis:
    plan: str
    data_sha256: str
    calibration_sha256: str
    cells: list[CellResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan,
            "data_sha256": self.data_sha256,
            "calibration_sha256": self.calibration_sha256,
            "cells": [c.to_dict() for c in self.cells],
        }

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
        q975 = float(np.quantile(np.abs(nl[np.isfinite(nl)]), 0.95)) if np.isfinite(nl).any() else float("nan")
        z = d / se if se and np.isfinite(se) and se > 0 else float("nan")
        out[name] = {
            "d": d,
            "p": empirical_p(d, nl),
            "ci95": [d - q975, d + q975],
            "tost": tost(d, nl, margin, ALPHA_EQ),
            "nw_z": z,
            "nw_p": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan"),
        }
    return out


def analyse(table: EpisodeTable, cal: Calibration, n_boot: int = N_BOOT) -> BlindedAnalysis:
    """Every cell of every code pair, blinded (see the module docstring). Deterministic given the data, the frozen
    calibration and ``n_boot`` (bootstrap seed: the calibration's)."""
    codes = table.codes
    rng_extra = np.random.default_rng(cal.boot_seed + 1)
    null_cache: dict[tuple[int, str, int], np.ndarray] = {}

    def null_for(lag: int, stat: str, n: int) -> np.ndarray:
        key = (lag, stat, n)
        if key not in null_cache:
            seed = int(sha256_bytes(f"{cal.boot_seed}:{lag}:{stat}:{n}".encode())[:15], 16)
            null_cache[key] = null_means(cal.series[(lag, stat)], n, n_boot, np.random.default_rng(seed))
        return null_cache[key]

    theta_noise = rng_extra.normal(0.0, cal.theta0_d_se, size=n_boot) if cal.theta0_d_se > 0 else np.zeros(n_boot)
    zeros = np.zeros(n_boot)
    cells: list[CellResult] = []
    for lag in (0, 1):
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
                        n = int(np.isfinite(vals).sum())
                        if n == 0:
                            continue
                        null = null_for(lag, st, n)
                        centres: dict[str, tuple[float, np.ndarray]] = {"0": (0.0, zeros)}
                        if lag == 1:
                            centres["theta0_d"] = (cal.theta0_d, theta_noise)
                        margin = cal.margin[lag] if st == "r" else float("nan")
                        cs = cal.series[(lag, st)]
                        cs = cs[np.isfinite(cs)]
                        fv = vals[np.isfinite(vals)]
                        scale_p = float(stats.levene(fv, cs, center="median").pvalue) if fv.size > 2 else math.nan
                        cells.append(
                            CellResult(
                                x,
                                y,
                                lag,
                                st,
                                geometry,
                                n,
                                float(fv.mean()),
                                _cell_tests(vals, null, margin, centres),
                                scale_p,
                            )
                        )
    return BlindedAnalysis(PLAN, table.sha256(), cal.sha256(), cells)


# --- Unblinding -------------------------------------------------------------------------------------------------------


def _read(ba: BlindedAnalysis, inv: Mapping[str, str], spec: CellSpec, centre: str) -> dict[str, Any]:
    found = ba.cell(inv[spec.x], inv[spec.y], spec.lag, spec.stat, spec.geometry)
    if found is None:
        return {"cell": [spec.x, spec.y, spec.lag, spec.stat], "missing": True, "p": float("nan")}
    c, sign = found
    t = c.tests[centre]
    lo, hi = t["ci95"]
    return {
        "cell": [spec.x, spec.y, spec.lag, spec.stat],
        "n": c.n,
        "mean": sign * c.mean,
        "d": sign * t["d"],
        "p": t["p"],
        "ci95": sorted([sign * lo, sign * hi]),
        "equivalent": bool(t["tost"]["equivalent"]),
        "nw_p": t["nw_p"],
        "scale_p": c.scale_p,
        "missing": False,
    }


@dataclass
class Report:
    primary: dict[str, dict[str, Any]]
    stability: dict[str, Any]
    secondary: dict[str, dict[str, Any]]
    exploratory: list[dict[str, Any]]
    calibration: dict[str, Any]
    analysis_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary,
            "stability": self.stability,
            "secondary": self.secondary,
            "exploratory": self.exploratory,
            "calibration": self.calibration,
            "analysis_sha256": self.analysis_sha256,
        }

    def to_markdown(self) -> str:
        lines = [f"# {PLAN}: unblinded result", "", f"Blinded analysis sha256: `{self.analysis_sha256}`", ""]
        lines += ["## Primary hypotheses (Holm, FWER 0.05; TOST at the frozen margins)", ""]
        lines += ["| hypothesis | n | estimate - theta0 | 95 % interval | p | Holm | equivalent | conclusion |"]
        lines += ["| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for k, r in self.primary.items():
            lines.append(
                f"| {k} | {r.get('n', '')} | {r.get('d', float('nan')):.4g} | "
                f"[{r['ci95'][0]:.4g}, {r['ci95'][1]:.4g}] | {r['p']:.4g} | {r['holm_rejected']} | "
                f"{r['equivalent']} | {r['conclusion']} |"
                if not r.get("missing")
                else f"| {k} | missing | | | | | | inconclusive |"
            )
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
        return "\n".join(lines) + "\n"


def unblind(ba: BlindedAnalysis, code_to_arm: Mapping[str, str], cal: Calibration) -> Report:
    """Read off the pre-registered cells after the blinded output was committed."""
    inv = {a: c for c, a in code_to_arm.items()}
    primary = {
        "H_decision": _read(ba, inv, PRIMARY["H_decision"], "theta0_d"),
        "H_observer": _read(ba, inv, PRIMARY["H_observer"], "0"),
    }
    rej = holm({k: v["p"] for k, v in primary.items()}, ALPHA)
    for k, v in primary.items():
        v["holm_rejected"] = rej[k]
        v["conclusion"] = "inconclusive" if v.get("missing") else conclusion(rej[k], v["equivalent"])
    stab = {g: _read(ba, inv, s, "0") for g, s in STABILITY.items()}
    stable = all(
        not s.get("missing") and s["p"] > STABILITY_ALPHA and not (s["scale_p"] <= STABILITY_ALPHA)
        for s in stab.values()
    )
    secondary = {k: _read(ba, inv, s, "theta0_d" if s.lag == 1 and s.x == "A1" else "0") for k, s in SECONDARY.items()}
    srej = holm({k: v["p"] for k, v in secondary.items()}, ALPHA)
    for k, v in secondary.items():
        v["holm_rejected"] = srej[k]
    explo = []
    for c in ba.cells:
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
    return Report(primary, {"stable": stable, "cells": stab}, secondary, explo, cal_d, ba.sha256())


# --- Engineering comparisons (M7 controllers, E5) ---------------------------------------------------------------------


def paired_margin_test(
    diffs: np.ndarray, control_diffs: np.ndarray, alpha: float = ALPHA, n_boot: int = N_BOOT, seed: int = 0
) -> dict[str, Any]:
    """A paired comparison judged against control repeats: ``diffs`` (treatment - reference, per pair, e.g. per
    seed and field) and ``control_diffs`` (the same statistic between two repeats of the *same* controller under
    the same schedule). Margin = :func:`control_margin` of the control diffs; null of the mean = the
    sign-symmetric block bootstrap of the control diffs. Returns mean, p (two-sided), the margin and TOST at
    ``alpha / 2`` per side."""
    d = np.asarray(diffs, dtype=np.float64)
    d = d[np.isfinite(d)]
    c = np.asarray(control_diffs, dtype=np.float64)
    c = c[np.isfinite(c)]
    margin = control_margin(c, MARGIN_PAIRS, seed)
    null = null_means(c, d.size, n_boot, np.random.default_rng(seed))
    m = float(d.mean()) if d.size else float("nan")
    return {
        "n": int(d.size),
        "mean": m,
        "p": empirical_p(m, null),
        "margin": margin,
        "tost": tost(m, null, margin, alpha / 2),
    }


# --- Lock verification and command line -------------------------------------------------------------------------------


def frozen_constants() -> dict[str, Any]:
    """The constants this script freezes (recorded in the lock and checked against it)."""
    return {
        "ALPHA": ALPHA,
        "ALPHA_EQ": ALPHA_EQ,
        "MARGIN_QUANTILE": MARGIN_QUANTILE,
        "MARGIN_PAIRS": MARGIN_PAIRS,
        "POWER": POWER,
        "N_MIN": N_MIN,
        "M_CAL": M_CAL,
        "EXCLUSION_ALLOWANCE": EXCLUSION_ALLOWANCE,
        "ATTEMPT_CAP": ATTEMPT_CAP,
        "N_BOOT": N_BOOT,
        "STABILITY_ALPHA": STABILITY_ALPHA,
    }


def verify_lock(lock: Lock, root: Path = REPO_ROOT) -> None:
    """Refuse to start unless the lock names this plan, contains this script and every file still matches, and the
    frozen constants are this script's."""
    probs = []
    if lock.name != PLAN:
        probs.append(f"lock is for {lock.name!r}, not {PLAN}")
    if SCRIPT_PATH not in lock.files:
        probs.append(f"{SCRIPT_PATH} is not in the lock")
    if probs:
        raise LockMismatchError(probs)
    lock.verify(root)
    if lock.constants.get("analysis") != frozen_constants():
        raise LockMismatchError(["the script's frozen constants differ from the locked ones"])


def main(argv: Sequence[str] | None = None) -> None:
    from glowbox.analysis.prereg import check_run_record

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="freeze the calibration block")
    c.add_argument("--run", required=True, type=Path)
    c.add_argument("--lock", required=True, type=Path)
    c.add_argument("--theta0", nargs=2, type=float, default=(0.0, 0.0), metavar=("THETA0_D", "SE"))
    c.add_argument("--out", required=True, type=Path)
    a = sub.add_parser("analyse", help="blinded analysis of a confirmatory run")
    a.add_argument("--run", required=True, type=Path)
    a.add_argument("--lock", required=True, type=Path)
    a.add_argument("--calibration", required=True, type=Path)
    a.add_argument("--out", required=True, type=Path)
    u = sub.add_parser("unblind", help="map codes to arms (after the blinded output is committed)")
    u.add_argument("--analysis", required=True, type=Path)
    u.add_argument("--calibration", required=True, type=Path)
    u.add_argument("--run", required=True, type=Path)
    u.add_argument("--blind-key-hex", required=True)
    u.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    import yaml

    from glowbox.analysis.prereg import unblind as unblind_codes

    if args.cmd in ("calibrate", "analyse"):
        lock = Lock.load(args.lock)
        verify_lock(lock)
        with open(args.run / "run.yaml", encoding="utf-8") as f:
            info = yaml.safe_load(f)
        check_run_record(info, lock, "calibration" if args.cmd == "calibrate" else "confirmatory")
        table = EpisodeTable.load(args.run / "episodes.parquet")
        if args.cmd == "calibrate":
            cal = derive_calibration(table, *args.theta0)
            args.out.write_text(json.dumps(cal.to_dict(), sort_keys=True, indent=1))
            print(f"calibration sha256 {cal.sha256()}")
        else:
            cal = Calibration.from_dict(json.loads(args.calibration.read_text()))
            ba = analyse(table, cal)
            args.out.write_text(json.dumps(ba.to_dict(), sort_keys=True, indent=1, default=float))
            print(f"blinded analysis sha256 {ba.sha256()}: commit it before the blinding key is released")
    else:
        d = json.loads(args.analysis.read_text())
        cells = [CellResult(**cd) for cd in d["cells"]]
        ba = BlindedAnalysis(d["plan"], d["data_sha256"], d["calibration_sha256"], cells)
        cal = Calibration.from_dict(json.loads(args.calibration.read_text()))
        with open(args.run / "run.yaml", encoding="utf-8") as f:
            info = yaml.safe_load(f)
        commitment = info["prereg"]["keys"]["blind_sha256"]
        mapping = unblind_codes(bytes.fromhex(args.blind_key_hex), commitment, CONFIRMATORY_ARMS)
        rep = unblind(ba, mapping, cal)
        args.out.write_text(rep.to_markdown())


if __name__ == "__main__":
    main()
