"""QuTiP reference simulator: the ground truth that the fast surrogate is validated against.

Implements docs/physics.md as literally as possible:

* per NV axis, a 7x7 density matrix (ground triplet in the eigenbasis of H0, excited triplet, singlet) under a
  Lindblad master equation in the MW rotating frame (RWA), with optical pumping / emission / ISC / singlet decay
  (Tetienne 2012 rates), T1 and T2 as jump operators;
* the ensemble is the population-weighted sum over the four <111> axes and, per axis, a quadrature over the static
  detuning delta (equal-weight quantile midpoints, or uniform nodes with cell-probability weights; spec section 4);
* controls are piecewise constant per bin, so each bin's Liouvillian L is constant and the bin is propagated
  *exactly* with a matrix exponential. The photons emitted in the bin come from the same exponential (Van Loan
  augmentation): exp([[L, 0], [w, 0]] dt) has bottom-left row w int_0^dt exp(L s) ds. No sub-stepping or ODE
  tolerance is involved; the QuTiP ``mesolve`` cross-check in the tests confirms it agrees with QuTiP's integrator;
* MW frequency changes switch frames at the segment boundary (spec section 3, absolute-time phase);
* detected rate = eta N sum_k p_k e_k + background * laser * (1 + x) + dark; Poisson counts per bin; laser drift x
  is an exact Ornstein-Uhlenbeck sample on the grid, continuous across the shots of one call (spec section 7).

Accuracy over speed: one run costs roughly (distinct bin types) x (batched 50x50 expm over axes x detunings) plus
one batched 49x49 matrix-vector product per bin. With laser drift every lit bin is distinct, so drift is slow.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import qutip as qt
import scipy.signal
import scipy.stats
import torch

from glowbox.control.pulse import PulseSequence
from glowbox.sim import constants as c
from glowbox.sim.interface import EnvParams, FluorescenceTrace
from glowbox.sim.reference.model import (
    N_LEVELS,
    AxisModel,
    axis_geometry,
    build_axis_model,
    frame_phase_vec,
    initial_state_vec,
)

__all__ = [
    "BinControls",
    "DetuningRule",
    "ReferenceSimulator",
    "bin_controls",
    "detuning_quadrature_hz",
    "detuning_samples_hz",
]

VERSION = "1"

#: Validity limits from docs/physics.md section 6 (warn, do not refuse).
B_PERP_WARN_T = 5e-3  # excited-state mixing by B_perp is not modelled ("B_perp <~ a few mT")
B_PAR_ESLAC_WARN_T = 40e-3  # approaching the excited-state anticrossing at ~51 mT


DetuningRule = Literal["quantile", "uniform"]


def _width_and_cdf(env: EnvParams) -> tuple[float, Any]:
    """Width unit of the T2* distribution (Lorentzian HWHM Gamma or Gaussian sigma, Hz) and its CDF of delta/width."""
    if env.broadening_shape == "lorentzian":
        return 1.0 / (2.0 * math.pi * env.t2_star_s), lambda x: 0.5 + np.arctan(x) / math.pi
    return 1.0 / (math.sqrt(2.0) * math.pi * env.t2_star_s), scipy.stats.norm.cdf


def detuning_quadrature_hz(
    env: EnvParams, n: int, rule: DetuningRule = "quantile", spacing: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes (Hz) and weights (sum 1) of the static detuning delta (spec section 4).

    Lorentzian: HWHM Gamma = 1/(2 pi T2*); Gaussian: sigma = 1/(sqrt2 pi T2*). n = 1 gives delta = 0.

    * ``"quantile"``: equal weights 1/n at the quantile midpoints u_m = (m + 1/2)/n, delta = Gamma tan(pi (u - 1/2))
      or sigma Phi^-1(u). Tail nodes are sparse (spacing grows like delta^2), so a CW spectrum whose homogeneous
      width is below the tail spacing shows each tail node as a satellite line.
    * ``"uniform"``: n nodes evenly spaced by ``spacing`` x Gamma (or sigma), centred on 0; each weight is the
      distribution's probability in the node's cell, renormalised over the window (the distribution truncated at
      +-n spacing/2 widths; for the Lorentzian the dropped tail mass is about 4/(pi n spacing)). No comb while the
      homogeneous HWHM exceeds the spacing.
    """
    if n < 1:
        raise ValueError("n_detuning must be >= 1")
    if n == 1:
        return np.zeros(1), np.ones(1)
    width, cdf = _width_and_cdf(env)
    if rule == "quantile":
        u = (np.arange(n) + 0.5) / n
        if env.broadening_shape == "lorentzian":
            return np.asarray(width * np.tan(math.pi * (u - 0.5))), np.full(n, 1.0 / n)
        return np.asarray(width * scipy.stats.norm.ppf(u)), np.full(n, 1.0 / n)
    if rule == "uniform":
        if not spacing > 0:
            raise ValueError("detuning spacing must be > 0")
        x = spacing * (np.arange(n) + 0.5 - n / 2)  # node positions in widths
        p = np.asarray(cdf(x + spacing / 2) - cdf(x - spacing / 2), dtype=np.float64)
        return width * x, p / p.sum()
    raise ValueError(f"unknown detuning rule {rule!r}")


def detuning_samples_hz(env: EnvParams, n: int) -> np.ndarray:
    """Nodes of the equal-weight ``"quantile"`` rule (see ``detuning_quadrature_hz``), Hz."""
    return detuning_quadrature_hz(env, n, "quantile")[0]


@dataclass(frozen=True)
class BinControls:
    """Exact (float64, unquantised) per-bin controls taken from the PulseSequence objects."""

    laser: np.ndarray  # power 0..1
    mw_amplitude: np.ndarray  # 0..1, 0 = MW off
    mw_frequency_hz: np.ndarray
    mw_phase_rad: np.ndarray


def bin_controls(sequence: PulseSequence) -> BinControls:
    """Per-bin controls in float64 (``to_arrays`` stores float32, which would round e.g. a 0.1 amplitude)."""
    n, g = sequence.n_bins, sequence.grid_ns
    laser = np.zeros(n)
    amp = np.zeros(n)
    freq = np.zeros(n)
    phase = np.zeros(n)
    for lp in sequence.laser:
        laser[lp.start_ns // g : lp.end_ns // g] = lp.power
    for mp in sequence.microwave:
        s = slice(mp.start_ns // g, mp.end_ns // g)
        amp[s] = mp.amplitude
        freq[s] = mp.frequency_hz
        phase[s] = mp.phase_rad
    return BinControls(laser, amp, freq, phase)


@dataclass
class _Group:
    """Axes that are physically identical (same local field and drive), simulated once."""

    model: AxisModel
    axes: list[int]
    population: float


class ReferenceSimulator:
    """Seven-level Lindblad NV-ensemble simulator (QuTiP operators, exact piecewise-constant propagation).

    ``n_detuning``: number of quadrature nodes M over the T2* detuning distribution; ``detuning_rule`` and
    ``detuning_spacing`` choose the rule (``detuning_quadrature_hz``; spec section 4: "quantile" for pulsed
    protocols, "uniform" for CW lineshapes). ``dedupe_axes``: simulate symmetry-equivalent axes once (exact; only
    used when E = 0).
    """

    name = "qutip-reference"

    def __init__(
        self,
        n_detuning: int = 64,
        dedupe_axes: bool = True,
        detuning_rule: DetuningRule = "quantile",
        detuning_spacing: float = 0.5,
    ) -> None:
        if n_detuning < 1:
            raise ValueError("n_detuning must be >= 1")
        if detuning_rule not in ("quantile", "uniform"):
            raise ValueError(f"unknown detuning rule {detuning_rule!r}")
        self.n_detuning = n_detuning
        self.detuning_rule: DetuningRule = detuning_rule
        self.detuning_spacing = detuning_spacing
        self.dedupe_axes = dedupe_axes

    # --- public API ------------------------------------------------------------------------------------------

    def run(
        self,
        sequence: PulseSequence,
        env: EnvParams,
        *,
        shots: int = 1,
        rng: np.random.Generator | None = None,
    ) -> FluorescenceTrace:
        """Simulate ``shots`` back-to-back repetitions of ``sequence`` (Simulator protocol)."""
        if shots < 1:
            raise ValueError("shots must be >= 1")
        rng = rng if rng is not None else np.random.default_rng()
        n, g = sequence.n_bins, sequence.grid_ns
        ctrl = bin_controls(sequence)
        sigma = env.laser_drift_rel_sigma
        if sigma > 0:
            x = _ou_drift(rng, shots * n, g * 1e-9, sigma, env.laser_drift_tau_s).reshape(shots, n)
            power = np.maximum(0.0, 1.0 + x)
            rows = []
            for s in range(shots):
                emis, _ = self._emission(sequence, env, ctrl, power[s])
                rows.append(self._detected_rate(env, emis, ctrl, power[s]))
            rate = np.stack(rows)
        else:
            ones = np.ones(n)
            emis, _ = self._emission(sequence, env, ctrl, ones)
            rate = np.broadcast_to(self._detected_rate(env, emis, ctrl, ones), (shots, n)).copy()
        counts = rng.poisson(rate * g * 1e-9).astype(np.int64)
        meta: dict[str, Any] = {
            "source": self.name,
            "version": VERSION,
            "model": "seven-level Lindblad, RWA, exact piecewise-constant propagation",
            "n_detuning": self.n_detuning,
            "detuning_rule": self.detuning_rule,
            "detuning_spacing": self.detuning_spacing,
            "broadening_shape": env.broadening_shape,
            "qutip_version": qt.__version__,
            "label": dict(sequence.label),
        }
        return FluorescenceTrace(counts=counts, bin_ns=g, rate_cps=rate, metadata=meta)

    def emission_per_nv(
        self, sequence: PulseSequence, env: EnvParams, power_factor: np.ndarray | None = None
    ) -> np.ndarray:
        """Mean photon emission rate of one NV per axis, 1/s, shape (4, n_bins) (detuning-averaged, bin-averaged).

        Rows of axes with zero population are zero. ``power_factor`` (n_bins,) multiplies the laser (drift);
        default 1.
        """
        ctrl = bin_controls(sequence)
        pf = np.ones(sequence.n_bins) if power_factor is None else np.asarray(power_factor, dtype=np.float64)
        return self._emission(sequence, env, ctrl, pf)[0]

    def expected_rate_cps(self, sequence: PulseSequence, env: EnvParams) -> np.ndarray:
        """Noiseless detected count rate without drift, counts/s, shape (n_bins,)."""
        ctrl = bin_controls(sequence)
        ones = np.ones(sequence.n_bins)
        emis, _ = self._emission(sequence, env, ctrl, ones)
        return self._detected_rate(env, emis, ctrl, ones)

    def states(self, sequence: PulseSequence, env: EnvParams) -> tuple[list[AxisModel], list[list[int]], np.ndarray]:
        """Debug/test access: per group (axis models, axes), and the column-stacked rho after every bin.

        Returns (models, axes per model, states) with states of shape (n_groups, M, n_bins, 49) in each model's
        eigenbasis and the rotating frame current at the end of each bin. Node m carries the weight
        ``detuning_quadrature_hz(...)[1][m]`` (1/M for the default "quantile" rule).
        """
        ctrl = bin_controls(sequence)
        groups = self._groups(sequence, env)
        _, st = self._propagate(sequence, env, groups, ctrl, np.ones(sequence.n_bins), keep_states=True)
        assert st is not None
        return [gr.model for gr in groups], [gr.axes for gr in groups], st

    # --- internals ------------------------------------------------------------------------------------------

    def _detected_rate(self, env: EnvParams, emis: np.ndarray, ctrl: BinControls, power: np.ndarray) -> np.ndarray:
        pops = np.asarray(env.orientation_populations, dtype=np.float64)
        signal = env.collection_efficiency * env.n_nv * (pops @ emis)
        background = env.background_cps * ctrl.laser * power
        return np.asarray(signal + background + env.dark_cps)

    def _groups(self, sequence: PulseSequence, env: EnvParams) -> list[_Group]:
        if env.t2_stretch != 1.0:
            raise ValueError("the reference supports only t2_stretch = 1 (docs/physics.md section 5)")
        b = env.total_field_t(sequence)
        mw = np.asarray(env.mw_field_direction, dtype=np.float64)
        canonical = self.dedupe_axes and env.e_hz == 0.0
        proj_t = c.b_projections_hz(b) / c.GAMMA_E_HZ_PER_T
        groups: dict[tuple[float, ...], _Group] = {}
        for k, pop in enumerate(env.orientation_populations):
            if pop <= 0:
                continue
            if proj_t[k, 1] > B_PERP_WARN_T:
                warnings.warn(
                    f"axis {k}: B_perp = {proj_t[k, 1] * 1e3:.1f} mT; excited-state mixing is not modelled "
                    "(docs/physics.md section 6)",
                    stacklevel=3,
                )
            if abs(proj_t[k, 0]) > B_PAR_ESLAC_WARN_T:
                warnings.warn(
                    f"axis {k}: |B_par| = {abs(proj_t[k, 0]) * 1e3:.1f} mT near the excited-state anticrossing "
                    "(~51 mT), outside the model's validity",
                    stacklevel=3,
                )
            geom = axis_geometry(c.NV_AXES[k], b, mw, canonical)
            key = geom.key() if canonical else (float(k),)
            if key in groups:
                groups[key].axes.append(k)
                groups[key].population += pop
            else:
                model = build_axis_model(geom, env.d_hz, env.e_hz, env.t1_s, env.t2_s)
                groups[key] = _Group(model, [k], pop)
        return list(groups.values())

    def _emission(
        self, sequence: PulseSequence, env: EnvParams, ctrl: BinControls, power: np.ndarray
    ) -> tuple[np.ndarray, list[_Group]]:
        groups = self._groups(sequence, env)
        per_group, _ = self._propagate(sequence, env, groups, ctrl, power, keep_states=False)
        emis = np.zeros((4, sequence.n_bins))
        for gi, gr in enumerate(groups):
            for k in gr.axes:
                emis[k] = per_group[gi]
        return emis, groups

    def _propagate(
        self,
        sequence: PulseSequence,
        env: EnvParams,
        groups: list[_Group],
        ctrl: BinControls,
        power: np.ndarray,
        keep_states: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Propagate every (group, detuning) density matrix through the bins.

        Returns the detuning-averaged emission rate per NV (n_groups, n_bins) in 1/s, and optionally the states.
        """
        n, g = sequence.n_bins, sequence.grid_ns
        deltas, wq = detuning_quadrature_hz(env, self.n_detuning, self.detuning_rule, self.detuning_spacing)
        n_g, m = len(groups), deltas.size
        d2 = N_LEVELS * N_LEVELS
        # Stacked building blocks, (n_g, 49, 49).
        fixed = np.stack([gr.model.diss_fixed for gr in groups])
        pump = np.stack([gr.model.diss_pump for gr in groups])
        hm = np.stack([gr.model.ham_minus for gr in groups])
        hp = np.stack([gr.model.ham_plus for gr in groups])
        hk = np.stack([gr.model.ham_k for gr in groups])
        hkd = np.stack([gr.model.ham_kdag for gr in groups])
        fm = np.array([gr.model.f_minus_hz for gr in groups])
        fp = np.array([gr.model.f_plus_hz for gr in groups])
        fr_per_amp = env.rabi_full_scale_hz * np.array([gr.model.mperp for gr in groups])
        hd = np.stack([gr.model.ham_delta for gr in groups])
        w = np.stack([gr.model.emission for gr in groups])  # (n_g, 49)
        # delta enters as delta * Sz (spec section 4): (n_g, m, 49, 49)
        dpart = deltas[None, :, None, None] * hd[:, None]

        cache: dict[tuple[float, float, float, float], tuple[np.ndarray, np.ndarray]] = {}

        def propagator(beta: float, f_frame: float, amp: float, phase: float) -> tuple[np.ndarray, np.ndarray]:
            key = (beta, f_frame, amp, phase)
            hit = cache.get(key)
            if hit is not None:
                return hit
            base = fixed + beta * pump
            base = base + (fm - f_frame)[:, None, None] * hm + (fp - f_frame)[:, None, None] * hp
            if amp > 0:
                fr = (fr_per_amp * amp)[:, None, None]
                base = base + fr * (np.exp(-1j * phase) * hk + np.exp(1j * phase) * hkd)
            lio = base[:, None] + dpart  # (n_g, m, 49, 49)
            aug = np.zeros((n_g, m, d2 + 1, d2 + 1), dtype=np.complex128)
            aug[:, :, :d2, :d2] = lio
            aug[:, :, d2, :d2] = w[:, None, :]
            ex = _expm(aug * float(g))
            prop = np.ascontiguousarray(ex[:, :, :d2, :d2])
            photons = np.ascontiguousarray(ex[:, :, d2, :d2])  # photons emitted in the bin per unit state
            cache[key] = (prop, photons)
            return prop, photons

        beta_full = env.optical_beta_full_scale
        rho = np.broadcast_to(initial_state_vec(), (n_g, m, d2)).copy()
        emitted = np.zeros((n_g, n))
        states = np.zeros((n_g, m, n, d2), dtype=np.complex128) if keep_states else None
        mw_freqs = ctrl.mw_frequency_hz[ctrl.mw_amplitude > 0]
        f_cur = float(mw_freqs[0]) if mw_freqs.size else float(env.d_hz)
        for j in range(n):
            amp = float(ctrl.mw_amplitude[j])
            if amp > 0:
                f_new = float(ctrl.mw_frequency_hz[j])
                if f_new != f_cur:
                    # Frame switch at t = j * g ns (absolute-time phase convention, spec section 3).
                    cycles = (f_new - f_cur) * (j * g) * 1e-9
                    rho *= frame_phase_vec(cycles)
                    f_cur = f_new
                phase = float(ctrl.mw_phase_rad[j])
            else:
                phase = 0.0
            beta = beta_full * float(ctrl.laser[j]) * float(power[j])
            prop, photons = propagator(beta, f_cur, amp, phase)
            emitted[:, j] = ((photons * rho).sum(axis=2).real * wq).sum(axis=1)
            rho = np.matmul(prop, rho[..., None])[..., 0]
            if states is not None:
                states[:, :, j] = rho
        return emitted / (g * 1e-9), states


#: Thread cap for the batched matrix exponential. Small (50x50) matrices lose badly to oversubscription: on a
#: 32-thread box torch at 32 threads was ~20x slower than at 8 (scipy's OpenBLAS expm shows the same effect), and
#: 4 holds up best when the machine is busy.
EXPM_MAX_THREADS = 4


def _expm(a: np.ndarray) -> np.ndarray:
    """Batched matrix exponential (torch, float64/complex128 Pade scaling-and-squaring; agrees with
    scipy.linalg.expm to ~1e-13 on these matrices)."""
    prev = torch.get_num_threads()
    torch.set_num_threads(min(prev, EXPM_MAX_THREADS))
    try:
        return torch.linalg.matrix_exp(torch.from_numpy(a)).numpy()
    finally:
        torch.set_num_threads(prev)


def _ou_drift(rng: np.random.Generator, n: int, dt_s: float, sigma: float, tau_s: float) -> np.ndarray:
    """Exact Ornstein-Uhlenbeck samples on a grid (spec section 7), stationary start x0 ~ N(0, sigma^2)."""
    a = math.exp(-dt_s / tau_s)
    b = sigma * math.sqrt(1.0 - a * a)
    x0 = sigma * rng.standard_normal()
    if n == 1:
        return np.array([x0])
    xi = rng.standard_normal(n - 1)
    rest, _ = scipy.signal.lfilter([b], [1.0, -a], xi, zi=[a * x0])
    return np.concatenate([[x0], rest])
