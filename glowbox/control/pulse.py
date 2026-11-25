"""Pulse-sequence representation (Phase 2): a timeline on a fixed time grid.

A :class:`PulseSequence` is what a learner, a hand-written protocol or (later) the FPGA sequencer agrees on. It has

* a time grid ``grid_ns`` (default 10 ns) and a total ``duration_ns``, both integers, duration a multiple of the grid;
* laser pulses (on/off with a relative power in [0, 1]);
* microwave (MW) segments with carrier frequency (Hz), phase (rad), amplitude in [0, 1] and duration;
* one magnet setting for the whole sequence (magnets are slow; changing it means a new sequence).

Conventions (see docs/physics.md for how the simulator interprets them):

* All times are integer nanoseconds and every start/duration is a multiple of ``grid_ns``. Bin ``k`` covers
  ``[k*grid_ns, (k+1)*grid_ns)``.
* One laser channel and one MW channel: pulses on the same channel may not overlap. Laser and MW may overlap
  (CW ODMR does exactly that).
* Laser ``power`` is a fraction of the setup's full-scale power; the simulator maps 1.0 to
  ``EnvParams.optical_beta_full_scale``.
* MW ``amplitude`` is a fraction of full-scale field amplitude (linear in field, so Rabi frequency scales with it);
  the simulator maps 1.0 to ``EnvParams.rabi_full_scale_hz``.
* MW ``phase_rad`` is referenced to absolute sequence time: the field is ``A cos(2 pi f t + phi)`` with ``t`` measured
  from the start of the sequence. A phase-coherent synthesiser with IQ modulation behaves like this; it makes
  Ramsey/echo phase steps well defined independent of where a segment starts.
* ``MagnetSetting.field_t`` is the *applied* bias field (tesla, crystal frame); the simulator adds it to the hidden
  environmental field ``EnvParams.b_field_t``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from glowbox.sim.constants import GRID_DEFAULT_NS

#: Upper bound on MW carrier frequency accepted by validation, Hz (D + gamma_e * 0.25 T is ~9.9 GHz).
MW_FREQUENCY_MAX_HZ: float = 20e9


class PulseValidationError(ValueError):
    """A pulse sequence violates the grid, range or overlap rules."""


@dataclass(frozen=True)
class LaserPulse:
    """Laser on from ``start_ns`` for ``duration_ns`` at relative ``power`` (0..1 of full scale)."""

    start_ns: int
    duration_ns: int
    power: float = 1.0

    @property
    def end_ns(self) -> int:
        return self.start_ns + self.duration_ns


@dataclass(frozen=True)
class MicrowavePulse:
    """MW segment: field ``amplitude * A_full * cos(2 pi frequency_hz t + phase_rad)`` over its interval."""

    start_ns: int
    duration_ns: int
    frequency_hz: float
    phase_rad: float = 0.0
    amplitude: float = 1.0

    @property
    def end_ns(self) -> int:
        return self.start_ns + self.duration_ns


@dataclass(frozen=True)
class MagnetSetting:
    """Applied bias field in tesla, crystal frame (x,y,z = [100],[010],[001]); constant for the sequence."""

    field_t: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class PulseSequence:
    """A validated timeline. Construct directly, with :class:`SequenceBuilder` or a protocol constructor."""

    duration_ns: int
    laser: tuple[LaserPulse, ...] = ()
    microwave: tuple[MicrowavePulse, ...] = ()
    magnet: MagnetSetting = field(default_factory=MagnetSetting)
    grid_ns: int = GRID_DEFAULT_NS
    #: Free-form labels (protocol name, swept parameter); copied into trace/run metadata, ignored by physics.
    label: dict[str, float | int | str] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        # Sort pulses by start so equality and array conversion do not depend on construction order.
        object.__setattr__(self, "laser", tuple(sorted(self.laser, key=lambda p: p.start_ns)))
        object.__setattr__(self, "microwave", tuple(sorted(self.microwave, key=lambda p: p.start_ns)))
        self.validate()

    @property
    def n_bins(self) -> int:
        return self.duration_ns // self.grid_ns

    def times_ns(self) -> np.ndarray:
        """Start time of each bin, ns, shape (n_bins,)."""
        return np.arange(self.n_bins, dtype=np.int64) * self.grid_ns

    def validate(self) -> None:
        """Raise :class:`PulseValidationError` if any rule is broken."""
        g = self.grid_ns
        if not isinstance(g, int) or g <= 0:
            raise PulseValidationError(f"grid_ns must be a positive int, got {g!r}")
        if not isinstance(self.duration_ns, int) or self.duration_ns <= 0 or self.duration_ns % g:
            raise PulseValidationError(f"duration_ns={self.duration_ns!r} must be a positive multiple of {g}")
        for kind, pulses in (("laser", self.laser), ("microwave", self.microwave)):
            prev_end = 0
            for p in pulses:
                for name in ("start_ns", "duration_ns"):
                    v = getattr(p, name)
                    if not isinstance(v, int | np.integer) or v % g:
                        raise PulseValidationError(f"{kind} {name}={v!r} is not a multiple of grid {g} ns")
                if p.start_ns < 0 or p.duration_ns <= 0:
                    raise PulseValidationError(f"{kind} pulse {p} must have start >= 0 and duration > 0")
                if p.end_ns > self.duration_ns:
                    raise PulseValidationError(f"{kind} pulse {p} ends after the sequence ({self.duration_ns} ns)")
                if p.start_ns < prev_end:
                    raise PulseValidationError(f"{kind} pulses overlap at {p.start_ns} ns (one channel)")
                prev_end = p.end_ns
        for lp in self.laser:
            if not (0.0 <= lp.power <= 1.0) or not math.isfinite(lp.power):
                raise PulseValidationError(f"laser power {lp.power} outside [0, 1]")
        for mp in self.microwave:
            if not (0.0 <= mp.amplitude <= 1.0) or not math.isfinite(mp.amplitude):
                raise PulseValidationError(f"MW amplitude {mp.amplitude} outside [0, 1]")
            if not (0.0 < mp.frequency_hz <= MW_FREQUENCY_MAX_HZ):
                raise PulseValidationError(f"MW frequency {mp.frequency_hz} Hz outside (0, {MW_FREQUENCY_MAX_HZ}]")
            if not math.isfinite(mp.phase_rad):
                raise PulseValidationError("MW phase must be finite")
        if len(self.magnet.field_t) != 3 or not all(math.isfinite(x) for x in self.magnet.field_t):
            raise PulseValidationError(f"magnet field {self.magnet.field_t} must be 3 finite floats (tesla)")

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form (YAML/JSON/Parquet metadata); inverse of :meth:`from_dict`."""
        return {
            "duration_ns": self.duration_ns,
            "grid_ns": self.grid_ns,
            "laser": [asdict(p) for p in self.laser],
            "microwave": [asdict(p) for p in self.microwave],
            "magnet_t": list(self.magnet.field_t),
            "label": dict(self.label),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PulseSequence:
        m = tuple(float(x) for x in d.get("magnet_t", (0.0, 0.0, 0.0)))
        return cls(
            duration_ns=int(d["duration_ns"]),
            laser=tuple(LaserPulse(**p) for p in d.get("laser", ())),
            microwave=tuple(MicrowavePulse(**p) for p in d.get("microwave", ())),
            magnet=MagnetSetting((m[0], m[1], m[2])),
            grid_ns=int(d.get("grid_ns", GRID_DEFAULT_NS)),
            label=dict(d.get("label", {})),
        )

    def to_arrays(self) -> SequenceArrays:
        """Compact per-bin form for batching (see :class:`SequenceArrays`)."""
        n, g = self.n_bins, self.grid_ns
        laser = np.zeros(n, dtype=np.float32)
        mw_amp = np.zeros(n, dtype=np.float32)
        mw_freq = np.zeros(n, dtype=np.float64)
        mw_phase = np.zeros(n, dtype=np.float32)
        for lp in self.laser:
            laser[lp.start_ns // g : lp.end_ns // g] = lp.power
        for mp in self.microwave:
            s = slice(mp.start_ns // g, mp.end_ns // g)
            mw_amp[s] = mp.amplitude
            mw_freq[s] = mp.frequency_hz
            mw_phase[s] = math.remainder(mp.phase_rad, 2 * math.pi)
        return SequenceArrays(
            laser=laser,
            mw_amplitude=mw_amp,
            mw_frequency_hz=mw_freq,
            mw_phase_rad=mw_phase,
            magnet_t=np.asarray(self.magnet.field_t, dtype=np.float64),
            grid_ns=g,
        )


@dataclass
class SequenceArrays:
    """Per-bin array form of one sequence or (with a leading batch axis) a batch.

    Shapes: per-bin arrays ``(..., n_bins)``, ``magnet_t`` ``(..., 3)``. ``mw_amplitude == 0`` means MW off (frequency
    and phase are then 0 and meaningless). ``mw_frequency_hz == 0`` means "the previous bin's frequency": a consumer
    that drives an amplitude in such a bin (e.g. a gradient through the surrogate) uses the last set frequency.
    Frequencies are float64 because a float32 GHz carrier only resolves ~250 Hz, too coarse to integrate phase over
    microseconds. Phases are wrapped to [-pi, pi].
    Arrays are NumPy or torch tensors (see :meth:`to_torch`), never mixed.
    """

    laser: np.ndarray | torch.Tensor
    mw_amplitude: np.ndarray | torch.Tensor
    mw_frequency_hz: np.ndarray | torch.Tensor
    mw_phase_rad: np.ndarray | torch.Tensor
    magnet_t: np.ndarray | torch.Tensor
    grid_ns: int = GRID_DEFAULT_NS

    @property
    def n_bins(self) -> int:
        return int(self.laser.shape[-1])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.laser.shape[:-1])

    def to_torch(self, device: str | torch.device | None = None) -> SequenceArrays:
        """Return a copy with torch tensors on ``device`` (dtypes preserved: float32, float64 for frequency)."""

        def conv(a: np.ndarray | torch.Tensor) -> torch.Tensor:
            return torch.as_tensor(a, device=device)

        return SequenceArrays(
            laser=conv(self.laser),
            mw_amplitude=conv(self.mw_amplitude),
            mw_frequency_hz=conv(self.mw_frequency_hz),
            mw_phase_rad=conv(self.mw_phase_rad),
            magnet_t=conv(self.magnet_t),
            grid_ns=self.grid_ns,
        )

    def to_numpy(self) -> SequenceArrays:
        def conv(a: np.ndarray | torch.Tensor) -> np.ndarray:
            return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

        return SequenceArrays(
            laser=conv(self.laser),
            mw_amplitude=conv(self.mw_amplitude),
            mw_frequency_hz=conv(self.mw_frequency_hz),
            mw_phase_rad=conv(self.mw_phase_rad),
            magnet_t=conv(self.magnet_t),
            grid_ns=self.grid_ns,
        )

    def unbatch(self) -> list[SequenceArrays]:
        """Split a 1-D batch into single-sequence arrays (NumPy)."""
        a = self.to_numpy()
        if len(a.batch_shape) != 1:
            raise ValueError(f"unbatch needs exactly one batch axis, got batch shape {a.batch_shape}")
        return [
            SequenceArrays(
                np.asarray(a.laser[i]),
                np.asarray(a.mw_amplitude[i]),
                np.asarray(a.mw_frequency_hz[i]),
                np.asarray(a.mw_phase_rad[i]),
                np.asarray(a.magnet_t[i]),
                a.grid_ns,
            )
            for i in range(a.batch_shape[0])
        ]

    def to_sequence(self) -> PulseSequence:
        """Run-length decode one (unbatched) sequence back to a validated :class:`PulseSequence`.

        Adjacent bins with equal values become one pulse, so two back-to-back identical pulses come back merged.
        Phases come back wrapped to [-pi, pi]; amplitude/power are clipped to [0, 1].
        """
        a = self.to_numpy()
        if a.batch_shape:
            raise ValueError("to_sequence needs a single sequence; use unbatch() first")
        g = a.grid_ns
        laser_p = np.clip(np.asarray(a.laser, dtype=np.float64), 0.0, 1.0)
        amp = np.clip(np.asarray(a.mw_amplitude, dtype=np.float64), 0.0, 1.0)
        freq = np.asarray(a.mw_frequency_hz, dtype=np.float64)
        phase = np.asarray(a.mw_phase_rad, dtype=np.float64)
        lasers = [LaserPulse(s * g, (e - s) * g, float(laser_p[s])) for s, e in _runs((laser_p,), laser_p > 0)]
        mws = [
            MicrowavePulse(s * g, (e - s) * g, float(freq[s]), float(phase[s]), float(amp[s]))
            for s, e in _runs((amp, freq, phase), amp > 0)
        ]
        mag = tuple(float(x) for x in np.asarray(a.magnet_t).reshape(3))
        return PulseSequence(
            duration_ns=a.n_bins * g,
            laser=tuple(lasers),
            microwave=tuple(mws),
            magnet=MagnetSetting((mag[0], mag[1], mag[2])),
            grid_ns=g,
        )


def _runs(columns: Iterable[np.ndarray], active: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) bin runs where ``active`` is true and every column is constant."""
    cols = list(columns)
    n = active.shape[0]
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i + 1
        while j < n and active[j] and all(c[j] == c[i] for c in cols):
            j += 1
        runs.append((i, j))
        i = j
    return runs


def stack(sequences: Sequence[PulseSequence], pad_to_bins: int | None = None) -> SequenceArrays:
    """Batch sequences into arrays with a leading batch axis (NumPy).

    Sequences must share ``grid_ns``. Shorter ones are zero-padded (laser and MW off) to the longest, or to
    ``pad_to_bins``. Padding is "dark time": a simulator still evolves T1 relaxation through it, so compare traces
    only over each sequence's own ``n_bins``.
    """
    if not sequences:
        raise ValueError("stack needs at least one sequence")
    grids = {s.grid_ns for s in sequences}
    if len(grids) != 1:
        raise ValueError(f"all sequences must share one grid, got {sorted(grids)}")
    n = max(s.n_bins for s in sequences)
    if pad_to_bins is not None:
        if pad_to_bins < n:
            raise ValueError(f"pad_to_bins={pad_to_bins} shorter than the longest sequence ({n})")
        n = pad_to_bins
    parts = [s.to_arrays() for s in sequences]

    def pad(name: str) -> np.ndarray:
        out = []
        for p in parts:
            a = np.asarray(getattr(p, name))
            out.append(np.pad(a, (0, n - a.shape[0])))
        return np.stack(out)

    return SequenceArrays(
        laser=pad("laser"),
        mw_amplitude=pad("mw_amplitude"),
        mw_frequency_hz=pad("mw_frequency_hz"),
        mw_phase_rad=pad("mw_phase_rad"),
        magnet_t=np.stack([np.asarray(p.magnet_t) for p in parts]),
        grid_ns=grids.pop(),
    )


class SequenceBuilder:
    """Append-only builder with a time cursor, for writing protocols as "do this, then that".

    >>> seq = (SequenceBuilder().laser(3000).wait(1000).mw(50, 2.87e9).laser(3000)).build()
    """

    def __init__(self, grid_ns: int = GRID_DEFAULT_NS, magnet: MagnetSetting | None = None) -> None:
        self.grid_ns = grid_ns
        self.magnet = magnet or MagnetSetting()
        self.cursor_ns = 0
        self._laser: list[LaserPulse] = []
        self._mw: list[MicrowavePulse] = []
        self.label: dict[str, float | int | str] = {}

    def wait(self, duration_ns: int) -> SequenceBuilder:
        self.cursor_ns += int(duration_ns)
        return self

    def laser(self, duration_ns: int, power: float = 1.0) -> SequenceBuilder:
        self._laser.append(LaserPulse(self.cursor_ns, int(duration_ns), power))
        self.cursor_ns += int(duration_ns)
        return self

    def mw(
        self, duration_ns: int, frequency_hz: float, phase_rad: float = 0.0, amplitude: float = 1.0
    ) -> SequenceBuilder:
        if duration_ns > 0:  # a zero-length pulse (e.g. Rabi at t=0) is simply absent
            self._mw.append(MicrowavePulse(self.cursor_ns, int(duration_ns), frequency_hz, phase_rad, amplitude))
        self.cursor_ns += int(duration_ns)
        return self

    def laser_with_mw(
        self, duration_ns: int, frequency_hz: float, power: float = 1.0, amplitude: float = 1.0
    ) -> SequenceBuilder:
        """Laser and MW simultaneously (CW ODMR)."""
        self._laser.append(LaserPulse(self.cursor_ns, int(duration_ns), power))
        self._mw.append(MicrowavePulse(self.cursor_ns, int(duration_ns), frequency_hz, 0.0, amplitude))
        self.cursor_ns += int(duration_ns)
        return self

    def build(self, **label: float | int | str) -> PulseSequence:
        self.label.update(label)
        return PulseSequence(
            duration_ns=self.cursor_ns,
            laser=tuple(self._laser),
            microwave=tuple(self._mw),
            magnet=self.magnet,
            grid_ns=self.grid_ns,
            label=dict(self.label),
        )


# --- Standard protocols -----------------------------------------------------------------------------------------
# Default timings are typical for NV ensembles: ~1-3 us of laser re-polarises into ms=0; 3 us of dark time lets the
# singlet (lifetime ~ 1/(k71 + 2 k72) ~ 0.41 us with the Tetienne rates) empty before the MW (1 us would leave ~9 %
# of it, which drifts the baseline of sweeps with varying free time; docs/physics.md section 8); spin contrast lives
# in the first ~300 ns of the readout pulse (docs/physics.md, "Readout").

INIT_DEFAULT_NS = 3000
SETTLE_DEFAULT_NS = 3000
READOUT_DEFAULT_NS = 3000


def initialisation(
    duration_ns: int = INIT_DEFAULT_NS, power: float = 1.0, grid_ns: int = GRID_DEFAULT_NS
) -> PulseSequence:
    """A single laser pulse that pumps the spins into ms=0."""
    return SequenceBuilder(grid_ns).laser(duration_ns, power).build(protocol="initialisation")


def readout(duration_ns: int = READOUT_DEFAULT_NS, power: float = 1.0, grid_ns: int = GRID_DEFAULT_NS) -> PulseSequence:
    """A single laser readout pulse (same physics as initialisation; label differs)."""
    return SequenceBuilder(grid_ns).laser(duration_ns, power).build(protocol="readout")


def cw_odmr_point(
    frequency_hz: float,
    duration_ns: int = 20_000,
    amplitude: float = 0.1,
    laser_power: float = 1.0,
    magnet: MagnetSetting | None = None,
    grid_ns: int = GRID_DEFAULT_NS,
) -> PulseSequence:
    """One point of a CW ODMR sweep: laser and MW on together for ``duration_ns``.

    A sweep is ``[cw_odmr_point(f) for f in freqs]``; the steady-state count rate (late bins) dips at resonances.
    The default amplitude 0.1 (f_R = 0.82 MHz at the 10 MHz full scale) is *not* free of power broadening: with the
    default optical beta = 0.1 the line is strongly MW- and optically broadened (FWHM about 4.4 MHz vs the
    T2*-limited 1/(pi T2*) = 0.32 MHz at T2* = 1 us). A near-T2*-limited line needs amplitude <~ 0.01 and
    beta <~ 0.01 (FWHM about 0.7 MHz), and then the "uniform" detuning quadrature (docs/physics.md section 4).
    """
    b = SequenceBuilder(grid_ns, magnet).laser_with_mw(duration_ns, frequency_hz, laser_power, amplitude)
    return b.build(protocol="cw_odmr", frequency_hz=frequency_hz)


def pulsed_rabi(
    mw_duration_ns: int,
    frequency_hz: float,
    amplitude: float = 1.0,
    phase_rad: float = 0.0,
    init_ns: int = INIT_DEFAULT_NS,
    settle_ns: int = SETTLE_DEFAULT_NS,
    readout_ns: int = READOUT_DEFAULT_NS,
    magnet: MagnetSetting | None = None,
    grid_ns: int = GRID_DEFAULT_NS,
) -> PulseSequence:
    """init laser -> dark settle -> MW pulse of ``mw_duration_ns`` (may be 0) -> readout laser."""
    b = SequenceBuilder(grid_ns, magnet).laser(init_ns).wait(settle_ns)
    b.mw(mw_duration_ns, frequency_hz, phase_rad, amplitude).laser(readout_ns)
    return b.build(protocol="rabi", mw_duration_ns=mw_duration_ns, frequency_hz=frequency_hz)


def ramsey(
    free_evolution_ns: int,
    frequency_hz: float,
    pi_half_ns: int,
    amplitude: float = 1.0,
    second_phase_rad: float = 0.0,
    init_ns: int = INIT_DEFAULT_NS,
    settle_ns: int = SETTLE_DEFAULT_NS,
    readout_ns: int = READOUT_DEFAULT_NS,
    magnet: MagnetSetting | None = None,
    grid_ns: int = GRID_DEFAULT_NS,
) -> PulseSequence:
    """init -> pi/2 -> free evolution tau -> pi/2 (phase ``second_phase_rad``) -> readout.

    ``pi_half_ns`` must match the Rabi frequency at ``amplitude`` (1 / (4 f_Rabi)), rounded to the grid; at 10 ns
    grid this wants f_Rabi of order 10 MHz or a finer grid.
    """
    b = SequenceBuilder(grid_ns, magnet).laser(init_ns).wait(settle_ns)
    b.mw(pi_half_ns, frequency_hz, 0.0, amplitude).wait(free_evolution_ns)
    b.mw(pi_half_ns, frequency_hz, second_phase_rad, amplitude).laser(readout_ns)
    return b.build(protocol="ramsey", free_evolution_ns=free_evolution_ns, frequency_hz=frequency_hz)


def hahn_echo(
    tau_ns: int,
    frequency_hz: float,
    pi_ns: int,
    amplitude: float = 1.0,
    final_phase_rad: float = 0.0,
    init_ns: int = INIT_DEFAULT_NS,
    settle_ns: int = SETTLE_DEFAULT_NS,
    readout_ns: int = READOUT_DEFAULT_NS,
    magnet: MagnetSetting | None = None,
    grid_ns: int = GRID_DEFAULT_NS,
) -> PulseSequence:
    """init -> pi/2 -> tau -> pi -> tau -> pi/2 (phase ``final_phase_rad``; pi gives the inverted echo) -> readout.

    ``pi_ns`` is the pi-pulse length; the pi/2 pulses are ``pi_ns // 2`` rounded down to the grid, so ``pi_ns``
    should be an even number of grid steps. Total echo time is ``2 * tau_ns`` (free evolution only).
    """
    half = (pi_ns // 2) // grid_ns * grid_ns
    if half <= 0:
        raise PulseValidationError(f"pi_ns={pi_ns} too short for a pi/2 pulse on a {grid_ns} ns grid")
    b = SequenceBuilder(grid_ns, magnet).laser(init_ns).wait(settle_ns)
    b.mw(half, frequency_hz, 0.0, amplitude).wait(tau_ns)
    b.mw(pi_ns, frequency_hz, 0.0, amplitude).wait(tau_ns)
    b.mw(half, frequency_hz, final_phase_rad, amplitude).laser(readout_ns)
    return b.build(protocol="hahn_echo", tau_ns=tau_ns, frequency_hz=frequency_hz)
