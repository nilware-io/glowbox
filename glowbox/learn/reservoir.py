"""Reservoir baseline (project brief, phase 2): fixed random input encoding into pulses, the NV ensemble as the
reservoir, a linear readout trained by ridge regression. Nothing in the encoding is learned.

Pieces:

* :class:`EncoderSpec` / :class:`RandomPulseEncoder`: a fixed random map from an input vector u (n_inputs,) to a
  block of ``n_segments`` MW segments on the time grid, one per slot of ``slot_ns``. Per segment s, four
  pre-activations ``z = W_s u + b_s`` (W ~ N(0, gain^2 / n_inputs), b ~ N(0, 1), drawn once from the encoder seed)
  set

  - amplitude ``a = a_lo + (a_hi - a_lo) sigmoid(z_0)``,
  - phase ``phi = pi tanh(z_1)``,
  - frequency ``f = f_line(s) + span tanh(z_2)``, with ``f_line(s)`` one of the given ODMR lines (fixed random
    choice per segment: the device is calibrated, so the encoder knows where its lines are),
  - duration, on the grid, ``d_lo + round((d_hi - d_lo) sigmoid(z_3) / grid) grid``, MW from the slot start; the
    rest of the slot is dark.

  The spins are the reservoir: the readout fluorescence is a nonlinear (rotation, interference, saturation)
  function of the pulse parameters, and the spin state carries the information of earlier segments forward
  (populations for ~T1, coherences for ~T2*).
* :func:`multiplex_block`: time multiplexing, short readouts interleaved between groups of segments (virtual
  nodes; the spin state carries over between them); :func:`laser_bins` lists every readout bin of a suffix.
* :func:`compose_suffix`: a MW block (or nothing) + a dark delay + an optional fixed read pulse + a time-resolved
  readout laser, as :class:`~glowbox.control.pulse.SequenceArrays`; :func:`init_prefix`: the shared MW-free
  initialisation (laser + settle). :func:`simulate_suffixes` runs the prefix once and every suffix from its final
  state (``branched_rate_cps``, exact; see its note on phases) and returns the mean counts per shot per readout
  bin. :func:`full_sequence` writes a prefix + suffix as one :class:`PulseSequence` with the phases that make it
  physically identical to the branched simulation (for the QuTiP reference).
* :class:`RidgeReadout`: standardised features, closed-form ridge over a grid of penalties with the penalty chosen
  by generalised cross-validation (GCV, one SVD). The only trained part of the reservoir.

Phases and branching: MW phases in this module are referenced to the start of the suffix (the encoder does not
know how long the initialisation was). In absolute sequence time (docs/physics.md section 3), a suffix that starts
at T_pre with suffix-relative phase phi has the absolute phase ``phi - 2 pi f T_pre``; the state after a MW-free
prefix is stationary under H0, so the two descriptions are the same experiment (tests/learn/test_reservoir.py
checks the full-sequence surrogate against the branched one).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from glowbox.control.pulse import (
    LaserPulse,
    MagnetSetting,
    MicrowavePulse,
    PulseSequence,
    SequenceArrays,
    _runs,
)
from glowbox.sim.constants import GRID_DEFAULT_NS
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions
from glowbox.sim.surrogate.sweeps import branched_rate_cps

#: Sequences per surrogate call in :func:`simulate_suffixes` (memory: B x 4 axes x M nodes x 14 reals per bin).
DEFAULT_CHUNK = 1024


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = 1.0 / (1.0 + np.exp(-x))
    return out


@dataclass(frozen=True)
class EncoderSpec:
    """Shape and ranges of a :class:`RandomPulseEncoder` (assumptions of the baseline, not physics)."""

    n_inputs: int
    n_segments: int = 8
    slot_ns: int = 100
    #: MW duration range within a slot, ns (multiples of the grid; ``d_hi <= slot_ns``).
    min_ns: int = 10
    max_ns: int = 100
    amp_lo: float = 0.05
    amp_hi: float = 1.0
    #: Frequency offsets up to +-span around the segment's line (a few linewidths / Rabi frequencies).
    freq_span_hz: float = 4e6
    #: Input gain of the random map: larger = more saturated (more binary) pulse parameters.
    gain: float = 1.5
    grid_ns: int = GRID_DEFAULT_NS

    def __post_init__(self) -> None:
        g = self.grid_ns
        if self.n_inputs < 1 or self.n_segments < 1:
            raise ValueError("n_inputs and n_segments must be >= 1")
        if self.slot_ns % g or self.min_ns % g or self.max_ns % g:
            raise ValueError("slot and duration bounds must be multiples of the grid")
        if not 0 < self.min_ns <= self.max_ns <= self.slot_ns:
            raise ValueError("need 0 < min_ns <= max_ns <= slot_ns")
        if not 0 <= self.amp_lo <= self.amp_hi <= 1:
            raise ValueError("need 0 <= amp_lo <= amp_hi <= 1")

    @property
    def block_bins(self) -> int:
        return self.n_segments * self.slot_ns // self.grid_ns


@dataclass(frozen=True)
class SegmentParams:
    """Encoded pulse parameters, each (N, n_segments): amplitude, phase (rad), frequency (Hz), duration (ns)."""

    amplitude: np.ndarray
    phase_rad: np.ndarray
    frequency_hz: np.ndarray
    duration_ns: np.ndarray


class RandomPulseEncoder:
    """Fixed random map from inputs to a MW block (see the module docstring). Deterministic given ``seed``."""

    def __init__(self, spec: EncoderSpec, lines_hz: Sequence[float] | np.ndarray, seed: int) -> None:
        self.spec = spec
        lines = np.asarray(lines_hz, dtype=np.float64).ravel()
        if lines.size == 0:
            raise ValueError("need at least one line frequency")
        rng = np.random.default_rng([int(seed), 0x5E5E])
        s = spec
        self.weights = rng.standard_normal((s.n_segments, 4, s.n_inputs)) * (s.gain / math.sqrt(s.n_inputs))
        self.bias = rng.standard_normal((s.n_segments, 4))
        self.line_index = rng.integers(0, lines.size, size=s.n_segments)
        self.base_hz = lines[self.line_index]

    def params(self, u: np.ndarray, duration_shift_ns: int = 0) -> SegmentParams:
        """Pulse parameters for inputs ``u`` (N, n_inputs). ``duration_shift_ns`` adds a fixed shift (multiple of the
        grid) to every MW duration, clipped to [grid, slot] (the timing probe)."""
        s = self.spec
        x = np.asarray(u, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != s.n_inputs:
            raise ValueError(f"inputs must be (N, {s.n_inputs}), got {x.shape}")
        if duration_shift_ns % s.grid_ns:
            raise ValueError("duration shift must be a multiple of the grid")
        z = np.einsum("sci,ni->nsc", self.weights, x) + self.bias[None]
        amp = s.amp_lo + (s.amp_hi - s.amp_lo) * _sigmoid(z[..., 0])
        phase = np.pi * np.tanh(z[..., 1])
        freq = self.base_hz[None] + s.freq_span_hz * np.tanh(z[..., 2])
        steps = np.rint((s.max_ns - s.min_ns) / s.grid_ns * _sigmoid(z[..., 3]))
        dur = s.min_ns + steps * s.grid_ns + duration_shift_ns
        dur = np.clip(dur, s.grid_ns, s.slot_ns).astype(np.int64)
        return SegmentParams(amp, phase, freq, dur)

    def block(self, u: np.ndarray, offset_hz: float = 0.0, duration_shift_ns: int = 0) -> SequenceArrays:
        """The encoded MW block, (N, block_bins) arrays; laser off. ``offset_hz`` shifts every MW frequency (the
        quantum test's detuned arm)."""
        return params_to_block(self.params(u, duration_shift_ns), self.spec, offset_hz)


def params_to_block(p: SegmentParams, spec: EncoderSpec, offset_hz: float = 0.0) -> SequenceArrays:
    """Segment parameters -> per-bin arrays (N, n_segments x slot bins); MW from each slot's start."""
    n, ns = p.amplitude.shape
    g = spec.grid_ns
    sb = spec.slot_ns // g
    nb = ns * sb
    amp = np.zeros((n, nb), dtype=np.float32)
    freq = np.zeros((n, nb), dtype=np.float64)
    phase = np.zeros((n, nb), dtype=np.float32)
    on = np.arange(sb)[None, None, :] < (p.duration_ns // g)[..., None]  # (N, S, sb)
    amp[:] = np.where(on, p.amplitude[..., None], 0.0).reshape(n, nb)
    freq[:] = np.where(on, p.frequency_hz[..., None] + offset_hz, 0.0).reshape(n, nb)
    phase[:] = np.where(on, p.phase_rad[..., None], 0.0).reshape(n, nb)
    return SequenceArrays(
        laser=np.zeros((n, nb), dtype=np.float32),
        mw_amplitude=amp,
        mw_frequency_hz=freq,
        mw_phase_rad=phase,
        magnet_t=np.zeros((n, 3)),
        grid_ns=g,
    )


@dataclass(frozen=True)
class ReadPulse:
    """A fixed MW pulse between the delay and the readout (e.g. a pi/2 pulse that turns stored coherence into a
    population difference). Phase suffix-relative."""

    frequency_hz: float
    amplitude: float
    duration_ns: int
    phase_rad: float = 0.0


def compose_suffix(
    n: int,
    block: SequenceArrays | None,
    delay_ns: int,
    readout_ns: int,
    read_pulse: ReadPulse | None = None,
    offset_hz: float = 0.0,
    magnet_t: Sequence[float] = (0.0, 0.0, 0.0),
    grid_ns: int = GRID_DEFAULT_NS,
) -> SequenceArrays:
    """[MW block] + dark delay + [read pulse] + readout laser (power 1), as (n, bins) arrays."""
    g = grid_ns
    if delay_ns % g or readout_ns % g or (read_pulse is not None and read_pulse.duration_ns % g):
        raise ValueError("delay, readout and read-pulse lengths must be multiples of the grid")
    parts_amp, parts_freq, parts_phase, parts_laser = [], [], [], []
    if block is not None:
        b = block.to_numpy()
        if b.batch_shape != (n,) or b.grid_ns != g:
            raise ValueError(f"block must be a batch of {n} on a {g} ns grid")
        parts_amp.append(np.asarray(b.mw_amplitude, dtype=np.float32))
        parts_freq.append(np.asarray(b.mw_frequency_hz, dtype=np.float64))
        parts_phase.append(np.asarray(b.mw_phase_rad, dtype=np.float32))
        parts_laser.append(np.asarray(b.laser, dtype=np.float32))
    nd = delay_ns // g
    if nd:
        parts_amp.append(np.zeros((n, nd), np.float32))
        parts_freq.append(np.zeros((n, nd)))
        parts_phase.append(np.zeros((n, nd), np.float32))
        parts_laser.append(np.zeros((n, nd), np.float32))
    if read_pulse is not None and read_pulse.duration_ns > 0:
        k = read_pulse.duration_ns // g
        parts_amp.append(np.full((n, k), read_pulse.amplitude, np.float32))
        parts_freq.append(np.full((n, k), read_pulse.frequency_hz + offset_hz))
        parts_phase.append(np.full((n, k), math.remainder(read_pulse.phase_rad, 2 * math.pi), np.float32))
        parts_laser.append(np.zeros((n, k), np.float32))
    nr = readout_ns // g
    parts_amp.append(np.zeros((n, nr), np.float32))
    parts_freq.append(np.zeros((n, nr)))
    parts_phase.append(np.zeros((n, nr), np.float32))
    parts_laser.append(np.ones((n, nr), np.float32))
    return SequenceArrays(
        laser=np.concatenate(parts_laser, axis=1),
        mw_amplitude=np.concatenate(parts_amp, axis=1),
        mw_frequency_hz=np.concatenate(parts_freq, axis=1),
        mw_phase_rad=np.concatenate(parts_phase, axis=1),
        magnet_t=np.tile(np.asarray(magnet_t, dtype=np.float64), (n, 1)),
        grid_ns=g,
    )


def multiplex_block(block: SequenceArrays, spec: EncoderSpec, n_groups: int, readout_ns: int) -> SequenceArrays:
    """Time multiplexing: split the block's slots into ``n_groups`` consecutive groups and put a readout laser of
    ``readout_ns`` after every group but the last (the suffix's own readout follows the last one).

    Each interleaved readout is a *virtual node* of the reservoir: it reads the spins part-way through the encoding
    and only partly repolarises them, so later groups act on a state that still remembers earlier ones.
    """
    b = block.to_numpy()
    g = spec.grid_ns
    if spec.n_segments % n_groups:
        raise ValueError("n_segments must be divisible by n_groups")
    if readout_ns % g:
        raise ValueError("readout length must be a multiple of the grid")
    n = b.batch_shape[0]
    gb = spec.n_segments // n_groups * spec.slot_ns // g
    nr = readout_ns // g
    cols: dict[str, list[np.ndarray]] = {"laser": [], "amp": [], "freq": [], "phase": []}
    for k in range(n_groups):
        sl = slice(k * gb, (k + 1) * gb)
        cols["laser"].append(np.asarray(b.laser)[:, sl])
        cols["amp"].append(np.asarray(b.mw_amplitude)[:, sl])
        cols["freq"].append(np.asarray(b.mw_frequency_hz)[:, sl])
        cols["phase"].append(np.asarray(b.mw_phase_rad)[:, sl])
        if k < n_groups - 1:
            cols["laser"].append(np.ones((n, nr), np.float32))
            cols["amp"].append(np.zeros((n, nr), np.float32))
            cols["freq"].append(np.zeros((n, nr)))
            cols["phase"].append(np.zeros((n, nr), np.float32))
    return SequenceArrays(
        laser=np.concatenate(cols["laser"], axis=1),
        mw_amplitude=np.concatenate(cols["amp"], axis=1),
        mw_frequency_hz=np.concatenate(cols["freq"], axis=1),
        mw_phase_rad=np.concatenate(cols["phase"], axis=1),
        magnet_t=np.asarray(b.magnet_t),
        grid_ns=g,
    )


def laser_bins(suffix: SequenceArrays) -> np.ndarray:
    """Indices of the laser-on bins of a suffix (row 0): every readout window of a multiplexed reservoir."""
    out: np.ndarray = np.flatnonzero(np.asarray(suffix.to_numpy().laser).reshape(-1, suffix.n_bins)[0] > 0)
    return out


def init_prefix(
    init_ns: int, settle_ns: int, magnet_t: Sequence[float] = (0.0, 0.0, 0.0), grid_ns: int = GRID_DEFAULT_NS
) -> SequenceArrays:
    """MW-free initialisation: laser for ``init_ns`` then dark for ``settle_ns``, a batch of one."""
    g = grid_ns
    ni, ns = init_ns // g, settle_ns // g
    laser = np.concatenate([np.ones(ni, np.float32), np.zeros(ns, np.float32)])[None]
    z = np.zeros_like(laser)
    return SequenceArrays(
        laser=laser,
        mw_amplitude=z.copy(),
        mw_frequency_hz=z.astype(np.float64),
        mw_phase_rad=z.copy(),
        magnet_t=np.asarray(magnet_t, dtype=np.float64)[None],
        grid_ns=g,
    )


def _take(a: SequenceArrays, sl: slice) -> SequenceArrays:
    x = a.to_numpy()
    return SequenceArrays(
        laser=np.asarray(x.laser)[sl],
        mw_amplitude=np.asarray(x.mw_amplitude)[sl],
        mw_frequency_hz=np.asarray(x.mw_frequency_hz)[sl],
        mw_phase_rad=np.asarray(x.mw_phase_rad)[sl],
        magnet_t=np.asarray(x.magnet_t)[sl],
        grid_ns=x.grid_ns,
    )


def simulate_suffixes(
    prefix: SequenceArrays,
    suffix: SequenceArrays,
    env: EnvParams,
    readout_bins: int | np.ndarray,
    *,
    options: ModelOptions | None = None,
    device: torch.device | str | None = None,
    chunk: int = DEFAULT_CHUNK,
) -> np.ndarray:
    """Mean detected counts per shot in the last ``readout_bins`` bins (an int) or in the given suffix bins (an
    index array) of ``prefix + suffix[i]`` for every row of ``suffix`` (all rows share the one prefix), (N, bins)
    float64."""
    sfx = suffix.to_numpy()
    n = sfx.batch_shape[0]
    if isinstance(readout_bins, np.ndarray):
        sel = np.asarray(readout_bins, dtype=np.int64)
    else:
        sel = np.arange(sfx.n_bins - int(readout_bins), sfx.n_bins)
    pre = prefix.to_numpy()
    if pre.batch_shape != (1,):
        raise ValueError("the prefix must be a batch of one")
    dt_s = sfx.grid_ns * 1e-9
    out = np.empty((n, sel.size))
    for i in range(0, n, chunk):
        sub = _take(sfx, slice(i, i + chunk))
        m = sub.batch_shape[0]
        mag = np.asarray(sub.magnet_t)
        pre_m = SequenceArrays(
            laser=pre.laser,
            mw_amplitude=pre.mw_amplitude,
            mw_frequency_hz=pre.mw_frequency_hz,
            mw_phase_rad=pre.mw_phase_rad,
            magnet_t=mag[:1],
            grid_ns=pre.grid_ns,
        )
        if not np.all(mag == mag[:1]):
            raise ValueError("all suffixes must share the prefix's magnet setting")
        rate = branched_rate_cps(pre_m, sub, np.zeros(m, dtype=np.int64), env, options=options, device=device)
        out[i : i + m] = (rate[:, torch.as_tensor(sel, device=rate.device)] * dt_s).double().cpu().numpy()
    return out


def full_sequence(prefix: SequenceArrays, suffix_row: SequenceArrays) -> PulseSequence:
    """One prefix (batch of one) + one suffix row (unbatched arrays) as a validated :class:`PulseSequence` with
    absolute-time phases (``phi - 2 pi f T_pre``, see the module docstring). Adjacent equal bins merge."""
    pre, suf = prefix.to_numpy(), suffix_row.to_numpy()
    g = pre.grid_ns
    t_pre_s = pre.n_bins * g * 1e-9
    laser = np.concatenate([np.asarray(pre.laser)[0], np.asarray(suf.laser)]).astype(np.float64)
    amp = np.concatenate([np.asarray(pre.mw_amplitude)[0], np.asarray(suf.mw_amplitude)]).astype(np.float64)
    freq = np.concatenate([np.asarray(pre.mw_frequency_hz)[0], np.asarray(suf.mw_frequency_hz)])
    phase_s = np.asarray(suf.mw_phase_rad, dtype=np.float64)
    f_s = np.asarray(suf.mw_frequency_hz, dtype=np.float64)
    # 2 pi (f T_pre mod 1): the product is ~1e4 cycles, float64 keeps ~1e-12 of a cycle.
    shift = 2 * np.pi * np.mod(f_s * t_pre_s, 1.0)
    wrapped = np.remainder(phase_s - shift + np.pi, 2 * np.pi) - np.pi
    phase_abs = np.where(np.asarray(suf.mw_amplitude) > 0, wrapped, 0.0)
    phase = np.concatenate([np.zeros(pre.n_bins), phase_abs])
    lasers = [LaserPulse(s * g, (e - s) * g, float(laser[s])) for s, e in _runs((laser,), laser > 0)]
    mws = [
        MicrowavePulse(s * g, (e - s) * g, float(freq[s]), float(phase[s]), float(amp[s]))
        for s, e in _runs((amp, freq, phase), amp > 0)
    ]
    m = tuple(float(x) for x in np.asarray(suf.magnet_t).reshape(3))
    return PulseSequence(
        duration_ns=laser.size * g,
        laser=tuple(lasers),
        microwave=tuple(mws),
        magnet=MagnetSetting((m[0], m[1], m[2])),
        grid_ns=g,
    )


# --- Linear readout ---------------------------------------------------------------------------------------------------


@dataclass
class RidgeReadout:
    """Ridge regression on standardised features (the reservoir's only trained part).

    ``fit`` picks the penalty from ``lambdas`` (relative to the mean squared singular value, i.e. scale-free) by
    generalised cross-validation, averaged over the target columns; ``predict`` maps new features to targets.
    """

    lambdas: tuple[float, ...] = tuple(10.0 ** np.arange(-6.0, 2.01, 0.5))
    x_mean: np.ndarray | None = None
    x_std: np.ndarray | None = None
    y_mean: np.ndarray | None = None
    coef: np.ndarray | None = None
    lam: float = float("nan")
    gcv: np.ndarray | None = None
    #: The fitted target was 1-D: predictions are then 1-D as well.
    vector_target: bool = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> RidgeReadout:
        x = np.asarray(x, dtype=np.float64)
        y2 = np.asarray(y, dtype=np.float64)
        self.vector_target = y2.ndim == 1
        y2 = y2[:, None] if y2.ndim == 1 else y2
        n = x.shape[0]
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0) + 1e-12
        self.y_mean = y2.mean(axis=0)
        xs = (x - self.x_mean) / self.x_std
        yc = y2 - self.y_mean
        u, s, vt = np.linalg.svd(xs, full_matrices=False)
        uty = u.T @ yc
        scale = float(np.mean(s**2)) if s.size else 1.0
        gcv = []
        for lam in self.lambdas:
            f = s**2 / (s**2 + lam * scale)
            fit = u @ (f[:, None] * uty)
            rss = float(np.sum((yc - fit) ** 2))
            dof = float(f.sum())
            gcv.append(n * rss / max(n - dof, 1e-9) ** 2)
        self.gcv = np.asarray(gcv)
        k = int(np.argmin(self.gcv))
        self.lam = float(self.lambdas[k])
        d = s / (s**2 + self.lam * scale)
        self.coef = vt.T @ (d[:, None] * uty)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef is None or self.x_mean is None or self.x_std is None or self.y_mean is None:
            raise RuntimeError("fit first")
        xs = (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_std
        out: np.ndarray = xs @ self.coef + self.y_mean
        return out[:, 0] if self.vector_target else out
