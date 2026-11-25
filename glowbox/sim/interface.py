"""The simulator/hardware contract: ``run(pulse_sequence, env_params) -> FluorescenceTrace``.

Both simulator implementations (the QuTiP reference and the fast surrogate) and, in Phase 4, the hardware driver
implement :class:`Simulator`. Learners only ever see this interface. The physics behind every field is specified in
docs/physics.md; units are in field names (``_hz`` = cycles/s, ``_s`` = seconds, ``_t`` = tesla, ``_cps`` =
counts/s).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from glowbox.control.pulse import PulseSequence
from glowbox.sim import constants as c

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class EnvParams:
    """Everything about the (simulated) sample and setup that the pulse sequence does not control.

    Temperature-free: room temperature is baked into the defaults (D, T1, rates). Frame: the cubic crystal frame
    (x,y,z = [100],[010],[001]), assumed aligned with the lab frame.
    """

    # Field. Total field seen by the spins = b_field_t + PulseSequence.magnet.field_t.
    #: Hidden/environmental magnetic field vector, tesla (task 1 estimates this).
    b_field_t: Vec3 = (0.0, 0.0, 0.0)

    # Spin Hamiltonian (per NV, in the NV's own frame; see docs/physics.md section 1).
    #: Zero-field splitting D, Hz.
    d_hz: float = c.D_GS_HZ
    #: Transverse strain splitting E, Hz.
    e_hz: float = c.E_GS_DEFAULT_HZ

    # Ensemble.
    #: Number of NVs in the excitation/collection volume (density x volume). Float: it only scales signal.
    n_nv: float = c.N_NV_DEFAULT
    #: Fraction of NVs along each of the four ``constants.NV_AXES``; non-negative, sums to 1.
    orientation_populations: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    #: Lineshape of the static detuning distribution that produces T2* (docs/physics.md section 4).
    broadening_shape: Literal["lorentzian", "gaussian"] = "lorentzian"

    # Relaxation.
    #: Longitudinal relaxation time T1, s (ms=0 population excess decays as exp(-t/T1)).
    t1_s: float = c.T1_DEFAULT_S
    #: Inhomogeneous dephasing time T2*, s (sets the static detuning distribution width).
    t2_star_s: float = c.T2_STAR_DEFAULT_S
    #: Hahn-echo coherence time T2, s.
    t2_s: float = c.T2_DEFAULT_S
    #: Stretch exponent p of the echo decay exp(-(t/T2)^p); 1 = exponential.
    t2_stretch: float = 1.0

    # Drive and optics.
    #: Rabi frequency (Hz, population oscillation frequency) at MW amplitude 1 for an NV axis perpendicular to the
    #: MW field.
    rabi_full_scale_hz: float = c.RABI_FULL_SCALE_DEFAULT_HZ
    #: Unit vector of the linearly polarised MW magnetic field, crystal frame. Default [100] couples equally to all
    #: four axes (|sin| = sqrt(2/3)).
    mw_field_direction: Vec3 = (1.0, 0.0, 0.0)
    #: Optical pumping parameter beta = k_pump / k_r at laser power 1 (Tetienne et al. 2012).
    optical_beta_full_scale: float = c.OPTICAL_BETA_FULL_SCALE_DEFAULT
    #: Detected fraction of emitted photons, dimensionless.
    collection_efficiency: float = c.COLLECTION_EFFICIENCY_DEFAULT
    #: Background (laser leakage, fluorescence of other defects, dark counts) while the laser is on, counts/s at
    #: laser power 1; scales linearly with laser power.
    background_cps: float = c.BACKGROUND_DEFAULT_CPS
    #: Detector dark counts, counts/s, present with the laser off too.
    dark_cps: float = 0.0

    # Drift (docs/physics.md section 6).
    #: Relative laser power noise: stationary std of the Ornstein-Uhlenbeck factor (1 + x(t)). 0 disables.
    laser_drift_rel_sigma: float = 0.0
    #: Correlation time of the laser drift, s.
    laser_drift_tau_s: float = 1e-3

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if len(self.b_field_t) != 3 or len(self.mw_field_direction) != 3:
            raise ValueError("b_field_t and mw_field_direction must be 3-vectors")
        pops = np.asarray(self.orientation_populations, dtype=np.float64)
        if pops.shape != (4,) or np.any(pops < 0) or not math.isclose(float(pops.sum()), 1.0, abs_tol=1e-9):
            raise ValueError(f"orientation_populations must be 4 non-negative values summing to 1, got {pops}")
        if not math.isclose(float(np.linalg.norm(self.mw_field_direction)), 1.0, abs_tol=1e-6):
            raise ValueError("mw_field_direction must be a unit vector")
        positive = ("d_hz", "n_nv", "t1_s", "t2_star_s", "t2_s", "t2_stretch", "laser_drift_tau_s")
        for name in positive:
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0")
        non_negative = (
            "e_hz",
            "rabi_full_scale_hz",
            "optical_beta_full_scale",
            "background_cps",
            "dark_cps",
            "laser_drift_rel_sigma",
        )
        for name in non_negative:
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} must be >= 0")
        if not 0 <= self.collection_efficiency <= 1:
            raise ValueError("collection_efficiency must be in [0, 1]")
        if self.broadening_shape not in ("lorentzian", "gaussian"):
            raise ValueError(f"unknown broadening_shape {self.broadening_shape!r}")
        if self.t2_s < self.t2_star_s:
            # Physically T2 >= T2* (echo refocuses static broadening). Allow equality, reject inversion.
            raise ValueError("t2_s must be >= t2_star_s")

    def total_field_t(self, sequence: PulseSequence) -> np.ndarray:
        """Field seen by the spins during ``sequence``, tesla, shape (3,)."""
        return np.asarray(self.b_field_t, dtype=np.float64) + np.asarray(sequence.magnet.field_t, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnvParams:
        """Inverse of :meth:`to_dict`; lists (from YAML/JSON) become tuples. Unknown keys raise."""
        kwargs: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
        return cls(**kwargs)


@dataclass
class FluorescenceTrace:
    """Detector output of one sequence, ``shots`` repetitions, one bin per grid step.

    * ``counts``: detected photon counts, int64, shape (shots, n_bins). Includes background and dark counts and
      Poisson shot noise. This is the only field hardware is guaranteed to fill.
    * ``rate_cps``: expected detected count rate before shot noise (signal + background + dark, including that
      shot's laser drift), float64, same shape, or ``None`` (hardware). Simulators should fill it: it is the
      noiseless target for surrogates and tests.
    * ``metadata``: at least ``"source"`` (e.g. ``"qutip-reference"``, ``"surrogate"``, ``"hardware"``); simulators
      add their version, seed and anything useful. Must be JSON/YAML-serialisable.
    """

    counts: np.ndarray
    bin_ns: int
    rate_cps: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shots(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.counts.shape[1])

    def times_ns(self) -> np.ndarray:
        """Bin start times, ns."""
        return np.arange(self.n_bins, dtype=np.int64) * self.bin_ns

    def window_counts(self, start_ns: int, end_ns: int) -> np.ndarray:
        """Counts summed over bins fully inside [start_ns, end_ns), per shot, shape (shots,)."""
        a, b = -(-start_ns // self.bin_ns), end_ns // self.bin_ns
        return self.counts[:, a:b].sum(axis=1)

    def mean_counts(self) -> np.ndarray:
        """Shot-averaged counts per bin, shape (n_bins,)."""
        return self.counts.mean(axis=0)

    def check(self, sequence: PulseSequence, shots: int | None = None) -> None:
        """Raise ``ValueError`` if this trace does not match ``sequence`` (implementations call it in tests)."""
        if self.counts.ndim != 2 or self.counts.dtype != np.int64:
            raise ValueError(f"counts must be int64 (shots, n_bins), got {self.counts.dtype} {self.counts.shape}")
        if self.bin_ns != sequence.grid_ns or self.n_bins != sequence.n_bins:
            raise ValueError(
                f"trace {self.n_bins}x{self.bin_ns} ns does not match sequence {sequence.n_bins}x{sequence.grid_ns} ns"
            )
        if shots is not None and self.shots != shots:
            raise ValueError(f"expected {shots} shots, got {self.shots}")
        if np.any(self.counts < 0):
            raise ValueError("negative counts")
        if self.rate_cps is not None and (self.rate_cps.shape != self.counts.shape or np.any(self.rate_cps < 0)):
            raise ValueError("rate_cps must be non-negative with the same shape as counts")
        if "source" not in self.metadata:
            raise ValueError("metadata must name its 'source'")


@runtime_checkable
class Simulator(Protocol):
    """Anything that turns a pulse sequence into fluorescence: QuTiP reference, surrogate, hardware.

    Contract:

    * Each shot starts from the *thermal* (fully mixed) ground-state spin population, with nothing in the
      excited or singlet levels; only the sequence's own laser pulses polarise it. (Hardware approximates this by
      a long dark gap between shots; a sequence that needs polarisation must start with an initialisation pulse.)
    * Shots are independent repetitions; laser drift may differ between shots.
    * ``rng`` is the only source of randomness (shot noise, drift); the same generator state gives the same trace.
      ``None`` means a fresh unseeded generator.
    * The trace satisfies ``trace.check(sequence, shots)``.
    """

    #: Short identifier stored in trace metadata ``source``.
    name: str

    def run(
        self,
        sequence: PulseSequence,
        env: EnvParams,
        *,
        shots: int = 1,
        rng: np.random.Generator | None = None,
    ) -> FluorescenceTrace: ...
