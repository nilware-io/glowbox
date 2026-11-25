"""Task 1 data: hidden magnetic fields, nuisance priors, fixed sensor protocols and seeded datasets from the surrogate.

A *bank* holds, for N hidden fields, the noiseless mean photon counts per shot of every feature cell of a fixed
sensor protocol (docs/results/m2-field-estimation.md, "Task and data"). Training draws fresh Poisson noise from it
at every step (for any number of shots), so the bank fixes the physics and the noise stays on the fly.

**Target and its symmetry.** A spin-1 NV axis is blind to the sign of B along it, and an ODMR spectrum does not say
which line belongs to which of the four axes. The spectrum is therefore invariant under the 48 signed permutations
of the crystal axes (they permute the four <111> axes up to sign), and B is only identifiable modulo that group.
The task's target is the canonical representative: B folded into the wedge Bx >= By >= Bz >= 0 (:func:`fold_to_wedge`).
Generically the fold is unique: the three largest projections a >= b >= c of |B . n_k| (times sqrt 3) give
z = (a - b)/2, y = (a - c)/2, x = (b + c)/2.

**Priors** (assumptions, documented in :class:`FieldPrior`): |B| uniform in [0.5, 3] mT (the model's validity range
is B_perp <~ a few mT, docs/physics.md section 6; 0.5 mT keeps the lines resolvable), direction isotropic. Nuisance
parameters vary per group of fields: T2*, Rabi full scale, optical beta, collection efficiency, background, strain E.

**Batching.** Fields of one group share one nuisance draw; the simulator gets the hidden field as the sequence's
magnet setting with ``EnvParams.b_field_t = 0``. Physically identical (the spins see the sum), and it lets one
surrogate call cover a group. Each bank row still records its full ``EnvParams`` with the hidden field in
``b_field_t``.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch

from glowbox.control.pulse import MagnetSetting, PulseSequence, SequenceArrays, SequenceBuilder, cw_odmr_point
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions
from glowbox.sim.surrogate.sweeps import branched_rate_cps, cw_steady_rate_cps

ProtocolKind = Literal["cw", "pulsed"]


def fold_to_wedge(b: np.ndarray) -> np.ndarray:
    """Canonical representative of B modulo signed axis permutations: sorted |components|, descending. (..., 3)."""
    return -np.sort(-np.abs(np.asarray(b, dtype=np.float64)), axis=-1)


@dataclass(frozen=True)
class FieldPrior:
    """Prior over hidden fields and nuisance parameters (ranges are assumptions, not measurements).

    Uniform ranges unless marked log-uniform. Everything not listed stays at the ``EnvParams`` defaults (T2 =
    10 us, T1 = 5 ms, 1000 NVs, MW along [100], no laser drift).
    """

    b_min_t: float = 0.5e-3
    b_max_t: float = 3.0e-3
    #: log-uniform; ensemble T2* (Barry et al. 2020 range is 0.1-10 us; this is a moderate-density sample).
    t2_star_s: tuple[float, float] = (0.5e-6, 2.0e-6)
    #: MW coupling calibration uncertainty (+-20 % around the 10 MHz default).
    rabi_full_scale_hz: tuple[float, float] = (8e6, 12e6)
    optical_beta_full_scale: tuple[float, float] = (0.07, 0.13)
    collection_efficiency: tuple[float, float] = (2e-3, 4e-3)
    background_cps: tuple[float, float] = (5e3, 2e4)
    #: Transverse strain splitting; the physics fit baseline assumes E = 0, so this is a model mismatch for it.
    e_hz: tuple[float, float] = (0.0, 0.3e6)

    def sample_fields(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` fields (tesla, crystal frame) folded into the wedge, (n, 3)."""
        d = rng.standard_normal((n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        mag = rng.uniform(self.b_min_t, self.b_max_t, size=(n, 1))
        return fold_to_wedge(d * mag)

    def sample_env(self, rng: np.random.Generator) -> EnvParams:
        """One nuisance draw (``b_field_t`` = 0; the field is set per sequence)."""
        lo, hi = self.t2_star_s
        return EnvParams(
            t2_star_s=float(math.exp(rng.uniform(math.log(lo), math.log(hi)))),
            rabi_full_scale_hz=float(rng.uniform(*self.rabi_full_scale_hz)),
            optical_beta_full_scale=float(rng.uniform(*self.optical_beta_full_scale)),
            collection_efficiency=float(rng.uniform(*self.collection_efficiency)),
            background_cps=float(rng.uniform(*self.background_cps)),
            e_hz=float(rng.uniform(*self.e_hz)),
        )


#: Nuisance parameters recorded per bank row (column order of ``FieldBank.nuisance``).
NUISANCE_NAMES = (
    "t2_star_s",
    "rabi_full_scale_hz",
    "optical_beta_full_scale",
    "collection_efficiency",
    "background_cps",
    "e_hz",
)


@dataclass(frozen=True)
class SweepProtocol:
    """A fixed sensor protocol: one measurement per MW frequency of a sweep.

    * ``"cw"``: CW ODMR, laser and MW on together; the feature per frequency is the photon count over a dwell of
      ``cw_shot_ns`` per shot in the steady state (continuous integration, so a shot is just a unit of time).
    * ``"pulsed"``: pulsed ODMR, init laser -> dark settle -> MW pulse of ``pi_ns`` (a nominal pi pulse at the
      default Rabi full scale) -> readout laser; the features are the ``readout_ns / grid`` readout bins per
      frequency, so the decoder sees the time-resolved readout transient.

    ``mw_offset_hz`` shifts every MW frequency (the quantum test's detuned condition).
    """

    kind: ProtocolKind
    f_start_hz: float = 2.77e9
    f_stop_hz: float = 2.97e9
    step_hz: float = 1e6
    mw_offset_hz: float = 0.0
    cw_amplitude: float = 0.1
    cw_shot_ns: int = 3000
    pulsed_amplitude: float = 0.2
    pi_ns: int = 310
    init_ns: int = 1500
    settle_ns: int = 1000
    readout_ns: int = 600
    grid_ns: int = 10

    def freqs_hz(self) -> np.ndarray:
        """Nominal sweep frequencies (before ``mw_offset_hz``), (F,)."""
        n = round((self.f_stop_hz - self.f_start_hz) / self.step_hz)
        return self.f_start_hz + self.step_hz * np.arange(n)

    def mw_freqs_hz(self) -> np.ndarray:
        """Frequencies actually applied, (F,)."""
        return self.freqs_hz() + self.mw_offset_hz

    @property
    def n_freq(self) -> int:
        return int(self.freqs_hz().shape[0])

    @property
    def n_time(self) -> int:
        """Feature rows per frequency: 1 for CW, the readout bins for pulsed."""
        return 1 if self.kind == "cw" else self.readout_ns // self.grid_ns

    @property
    def shot_ns(self) -> int:
        """Duration of one shot of one frequency point (time budget)."""
        if self.kind == "cw":
            return self.cw_shot_ns
        return self.init_ns + self.settle_ns + self.pi_ns + self.readout_ns

    def detuned(self, offset_hz: float) -> SweepProtocol:
        return dataclasses.replace(self, mw_offset_hz=offset_hz)

    def with_pi_ns(self, pi_ns: int) -> SweepProtocol:
        return dataclasses.replace(self, pi_ns=pi_ns)

    # Sequences, for simulators that only take PulseSequence (the reference) and for logging.
    def prefix(self, magnet: MagnetSetting) -> PulseSequence:
        return SequenceBuilder(self.grid_ns, magnet).laser(self.init_ns).wait(self.settle_ns).build(protocol="init")

    def suffix(self, f_hz: float, magnet: MagnetSetting) -> PulseSequence:
        b = SequenceBuilder(self.grid_ns, magnet).mw(self.pi_ns, f_hz, 0.0, self.pulsed_amplitude)
        return b.laser(self.readout_ns).build(protocol="pulsed_odmr", frequency_hz=f_hz)

    def prefix_arrays(self, fields_t: np.ndarray) -> SequenceArrays:
        """``stack([prefix(m) for m in fields])`` built directly (vectorised; PulseSequence validation and padding
        of thousands of sequences cost more than simulating them)."""
        g = fields_t.shape[0]
        n_init, n = self.init_ns // self.grid_ns, (self.init_ns + self.settle_ns) // self.grid_ns
        laser = np.zeros((g, n), dtype=np.float32)
        laser[:, :n_init] = 1.0
        z32 = np.zeros((g, n), dtype=np.float32)
        return SequenceArrays(laser, z32, np.zeros((g, n)), z32.copy(), np.asarray(fields_t, np.float64), self.grid_ns)

    def suffix_arrays(self, fields_t: np.ndarray) -> SequenceArrays:
        """``stack([suffix(f, m) for m in fields for f in mw_freqs_hz()])`` built directly (field-major)."""
        f = self.mw_freqs_hz()
        g, nf = fields_t.shape[0], f.shape[0]
        n_mw, n = self.pi_ns // self.grid_ns, (self.pi_ns + self.readout_ns) // self.grid_ns
        laser = np.zeros((g * nf, n), dtype=np.float32)
        laser[:, n_mw:] = 1.0
        amp = np.zeros((g * nf, n), dtype=np.float32)
        amp[:, :n_mw] = self.pulsed_amplitude
        freq = np.zeros((g * nf, n))
        freq[:, :n_mw] = np.tile(f, g)[:, None]
        magnet = np.repeat(np.asarray(fields_t, np.float64), nf, axis=0)
        return SequenceArrays(laser, amp, freq, np.zeros((g * nf, n), dtype=np.float32), magnet, self.grid_ns)

    def sequences(self, magnet: MagnetSetting, cw_ns: int = 20_000) -> list[PulseSequence]:
        """Full time-domain sequences of the sweep. CW points last ``cw_ns`` (long enough to reach the steady
        state the surrogate path solves for directly: 20 us agrees to ~1e-9, 6 us to ~5e-4)."""
        if self.kind == "cw":
            return [cw_odmr_point(f, cw_ns, self.cw_amplitude, 1.0, magnet, self.grid_ns) for f in self.mw_freqs_hz()]
        out = []
        for f in self.mw_freqs_hz():
            b = SequenceBuilder(self.grid_ns, magnet).laser(self.init_ns).wait(self.settle_ns)
            b.mw(self.pi_ns, f, 0.0, self.pulsed_amplitude).laser(self.readout_ns)
            out.append(b.build(protocol="pulsed_odmr", frequency_hz=float(f)))
        return out

    def features_from_rates(self, rates_cps: list[np.ndarray]) -> np.ndarray:
        """Mean counts per shot (T, F) from time-domain rates of :meth:`sequences` (CW: last 2 us of each point)."""
        dt = self.grid_ns * 1e-9
        if self.kind == "cw":
            late = 2000 // self.grid_ns
            r = np.array([np.mean(x[-late:]) for x in rates_cps])
            return (r * self.cw_shot_ns * 1e-9)[None, :]
        n = self.readout_ns // self.grid_ns
        return np.stack([np.asarray(x[-n:]) * dt for x in rates_cps], axis=1)


@dataclass
class FieldBank:
    """Noiseless mean counts per shot for N fields under one protocol.

    ``means`` (N, T, F) float32; ``fields_t`` (N, 3) folded fields; ``nuisance`` (N, len(NUISANCE_NAMES)).
    """

    protocol: SweepProtocol
    fields_t: np.ndarray
    nuisance: np.ndarray
    means: np.ndarray
    seconds: float = 0.0
    options: dict[str, float | int | str] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.fields_t.shape[0])

    def subset(self, idx: np.ndarray) -> FieldBank:
        return FieldBank(
            self.protocol, self.fields_t[idx], self.nuisance[idx], self.means[idx], self.seconds, dict(self.options)
        )

    def env(self, i: int) -> EnvParams:
        """The full EnvParams of row ``i`` (hidden field in ``b_field_t``)."""
        kw: dict[str, Any] = {k: float(v) for k, v in zip(NUISANCE_NAMES, self.nuisance[i], strict=True)}
        b = self.fields_t[i]
        return EnvParams(b_field_t=(float(b[0]), float(b[1]), float(b[2])), **kw)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            fields_t=self.fields_t,
            nuisance=self.nuisance,
            means=self.means,
            protocol=np.array(repr(dataclasses.asdict(self.protocol))),
        )


@dataclass
class BankSpec:
    """How to draw a bank: prior, size, nuisance group size and surrogate numerics."""

    prior: FieldPrior = field(default_factory=FieldPrior)
    fields_per_env: int = 8
    n_detuning: int = 32
    detuning_rule: str = "uniform"
    detuning_spacing: float = 2.0

    def options(self) -> ModelOptions:
        return ModelOptions(
            n_detuning=self.n_detuning, detuning_rule=self.detuning_rule, detuning_spacing=self.detuning_spacing
        )


def draw_fields_and_envs(n: int, spec: BankSpec, rng: np.random.Generator) -> tuple[np.ndarray, list[EnvParams]]:
    """``n`` folded fields and one nuisance EnvParams per group of ``spec.fields_per_env`` consecutive fields."""
    fields = spec.prior.sample_fields(n, rng)
    n_env = -(-n // spec.fields_per_env)
    envs = [spec.prior.sample_env(rng) for _ in range(n_env)]
    return fields, envs


def simulate_bank(
    protocol: SweepProtocol,
    fields_t: np.ndarray,
    envs: list[EnvParams],
    spec: BankSpec,
    device: torch.device | str | None = None,
    progress: Callable[[int, int], None] | None = None,
    workers: int = 4,
) -> FieldBank:
    """Simulate ``protocol`` for every field (group g of ``fields_per_env`` fields uses ``envs[g]``).

    Groups run on ``workers`` threads: the NumPy part (diagonalisation and superoperators per group) releases the
    GIL and dominates short sweeps, so threads overlap it with the device work. Results do not depend on
    ``workers`` (each group is independent and deterministic).
    """
    opt = spec.options()
    n, fpe = fields_t.shape[0], spec.fields_per_env
    freqs = protocol.mw_freqs_hz()
    nf, nt = freqs.shape[0], protocol.n_time
    means = np.empty((n, nt, nf), dtype=np.float32)
    nuis = np.empty((n, len(NUISANCE_NAMES)))
    t0 = time.perf_counter()

    def one(g: int) -> int:
        env = envs[g]
        sl = slice(g * fpe, min(n, (g + 1) * fpe))
        sub = fields_t[sl]
        if sub.shape[0] == 0:
            return 0
        nuis[sl] = [getattr(env, k) for k in NUISANCE_NAMES]
        if protocol.kind == "cw":
            rate = cw_steady_rate_cps(env, sub, freqs, amplitude=protocol.cw_amplitude, options=opt, device=device)
            means[sl, 0, :] = (rate * (protocol.cw_shot_ns * 1e-9)).cpu().numpy()
        else:
            g_n = sub.shape[0]
            pre, suf = protocol.prefix_arrays(sub), protocol.suffix_arrays(sub)
            branch = np.repeat(np.arange(g_n), nf)
            rate = branched_rate_cps(pre, suf, branch, env, options=opt, device=device)
            ro = rate[:, -nt:] * (protocol.grid_ns * 1e-9)  # (g' * F, T)
            means[sl] = ro.reshape(g_n, nf, nt).transpose(1, 2).float().cpu().numpy()
        return int(sub.shape[0])

    done = 0
    if workers <= 1:
        for g in range(len(envs)):
            done += one(g)
            if progress is not None:
                progress(done, n)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for k in pool.map(one, range(len(envs))):
                done += k
                if progress is not None:
                    progress(done, n)
    opts: dict[str, float | int | str] = {
        "n_detuning": spec.n_detuning,
        "detuning_rule": spec.detuning_rule,
        "detuning_spacing": spec.detuning_spacing,
        "fields_per_env": fpe,
    }
    return FieldBank(protocol, fields_t.copy(), nuis, means, time.perf_counter() - t0, opts)


# --- Noise and features (torch, on the training device) ---------------------------------------------------------


def poisson_counts(means: torch.Tensor, shots: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Photon counts summed over ``shots`` shots: Poisson(shots * mean) per cell (a sum of Poissons is Poisson).

    ``means`` (B, T, F) mean counts per shot, ``shots`` (B,). Float32 counts (exact integers below 2^24).
    """
    lam = means * shots.to(means.dtype)[:, None, None]
    return torch.poisson(lam, generator=generator)


def features(counts: torch.Tensor) -> torch.Tensor:
    """Decoder input: counts relative to their per-sample mean, minus 1, times 10 (dips of a few % -> O(0.1-1)).

    Normalising by the sample's own mean removes the absolute brightness (collection, NV number, background are
    nuisance), so the decoder sees the relative spectrum / readout transient, as a fit would.
    """
    mean = counts.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return 10.0 * (counts / mean - 1.0)


def log_shots(shots: torch.Tensor) -> torch.Tensor:
    """Scalar conditioning input: log10(shots), centred on 2.5 (shots 10..1e4 -> -1.5..1.5)."""
    return torch.log10(shots.to(torch.float32)) - 2.5


def sample_shots(n: int, lo: float, hi: float, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    """Log-uniform integer shot counts in [lo, hi], (n,) float32."""
    u = torch.rand(n, generator=generator, device=device)
    return torch.round(torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))


@dataclass
class TargetScaler:
    """Fields (tesla) <-> network units: (B - mean) / std per component, computed on the training fields."""

    mean_t: np.ndarray
    std_t: np.ndarray

    @classmethod
    def fit(cls, fields_t: np.ndarray) -> TargetScaler:
        return cls(fields_t.mean(axis=0), fields_t.std(axis=0))

    def encode(self, fields_t: np.ndarray) -> np.ndarray:
        return (fields_t - self.mean_t) / self.std_t

    def decode(self, y: np.ndarray, sigma: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        b = y * self.std_t + self.mean_t
        return b, None if sigma is None else sigma * self.std_t
