"""Physics-fit baseline for Task 1: Lorentzian ODMR dips whose positions are set by the field -> field estimate.

The classical estimator a spectroscopist would use, done jointly: the spectrum is modelled as a baseline minus a
sum of eight Lorentzian dips (two per NV axis) with one shared contrast and width, the dip *positions* being the
exact transition frequencies of H0 (numerical diagonalisation, E = 0 assumed, docs/physics.md section 1) for a
trial field B. Fitting B directly (instead of fitting eight free dips and then solving for B) handles merged and
missing lines without an explicit line assignment.

1. Coarse search: for every field on a fixed grid over the wedge and two trial widths, the best baseline and
   contrast are a weighted linear least-squares solve (closed form); keep the best few grid points.
2. Refine each with ``scipy.optimize.least_squares`` over (B, log width, contrast, baseline); take the best.
3. If the dips are not significant (chi^2 gain over a flat line below ``min_delta_chi2``), report the prior: no
   dips means no information about B.

Uncertainty: from the Jacobian, cov = (J^T J)^-1 scaled by max(1, reduced chi^2). The result is folded into the
wedge Bx >= By >= Bz >= 0 (glowbox.learn.field_data.fold_to_wedge), as the decoder targets are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.optimize

from glowbox.sim import constants as c
from glowbox.sim.surrogate.model import nv_local_frames

_S = 1.0 / np.sqrt(2.0)
_SX = _S * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
_SY = _S * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex)
_SZ = np.diag([1.0, 0.0, -1.0]).astype(complex)
_FRAMES = nv_local_frames()  # (4, 3, 3) rows: local x, y, z in the crystal frame


def transition_lines_hz(b_t: np.ndarray, d_hz: float = c.D_GS_HZ, e_hz: float = 0.0) -> np.ndarray:
    """ODMR line frequencies (f-, f+ per axis) for fields ``b_t`` (..., 3) tesla -> (..., 8) Hz, axis-major.

    Lines are the two upper eigenvalues of H0 minus the lowest; valid while gamma_e |B| < D (|B| < ~100 mT).
    """
    b = np.asarray(b_t, dtype=np.float64)
    loc = np.einsum("kij,...j->...ki", _FRAMES, b) * c.GAMMA_E_HZ_PER_T  # (..., 4, 3)
    h = d_hz * (_SZ @ _SZ) + e_hz * (_SX @ _SX - _SY @ _SY)
    h = h + loc[..., 0, None, None] * _SX + loc[..., 1, None, None] * _SY + loc[..., 2, None, None] * _SZ
    ev = np.linalg.eigvalsh(h)  # (..., 4, 3) ascending
    lines = ev[..., 1:] - ev[..., :1]
    return lines.reshape(*b.shape[:-1], 8)


def _dips(freqs: np.ndarray, lines: np.ndarray, hwhm: float | np.ndarray) -> np.ndarray:
    """Sum of unit Lorentzians at ``lines`` (..., 8) evaluated on ``freqs`` (F,) -> (..., F)."""
    x = (freqs[None, :] - lines[..., :, None]) / np.asarray(hwhm)[..., None, None]
    return np.sum(1.0 / (1.0 + x * x), axis=-2)


def wedge_grid(n: int, b_min_t: float, b_max_t: float, seed: int = 0) -> np.ndarray:
    """A quasi-uniform set of ``n`` fields in the wedge (fixed seed: the same grid every call), (n, 3)."""
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    mag = np.sqrt(rng.uniform(b_min_t**2, b_max_t**2, size=(n, 1)))  # denser at large |B|, where lines move most
    return -np.sort(-np.abs(d * mag), axis=-1)


@dataclass(frozen=True)
class FitConfig:
    """Search ranges and thresholds (Hz, tesla)."""

    b_min_t: float = 0.3e-3
    b_max_t: float = 3.3e-3
    grid_size: int = 6000
    trial_hwhm_hz: tuple[float, ...] = (1.2e6, 2.4e6)
    n_refine: int = 4
    min_delta_chi2: float = 30.0


@dataclass(frozen=True)
class FieldFit:
    b_t: np.ndarray  # (3,) folded
    sigma_t: np.ndarray  # (3,)
    ok: bool  # False: no significant dips, prior returned
    hwhm_hz: float
    contrast: float
    delta_chi2: float


class FieldFitter:
    """Reusable fitter (precomputes the grid lines). ``prior_mean_t``/``prior_std_t`` are returned without dips."""

    def __init__(
        self,
        freqs_hz: np.ndarray,
        prior_mean_t: np.ndarray,
        prior_std_t: np.ndarray,
        config: FitConfig | None = None,
        d_hz: float = c.D_GS_HZ,
    ) -> None:
        self.cfg = config or FitConfig()
        self.f = np.asarray(freqs_hz, dtype=np.float64)
        self.d_hz = d_hz
        self.prior_mean = np.asarray(prior_mean_t, dtype=np.float64)
        self.prior_std = np.asarray(prior_std_t, dtype=np.float64)
        self.grid = wedge_grid(self.cfg.grid_size, self.cfg.b_min_t, self.cfg.b_max_t)
        lines = transition_lines_hz(self.grid, d_hz)
        self.grid_shapes = [_dips(self.f, lines, w) for w in self.cfg.trial_hwhm_hz]  # each (G, F)

    def _prior(self, dchi2: float) -> FieldFit:
        return FieldFit(self.prior_mean.copy(), self.prior_std.copy(), False, float("nan"), 0.0, dchi2)

    def _coarse(
        self,
        wi: int,
        shape: np.ndarray,
        w: np.ndarray,
        y: np.ndarray,
        sw: float,
        swy: float,
        chi2_flat: float,
        flat: float,
    ) -> list[tuple[float, int, int, float, float]]:
        with np.errstate(divide="ignore", invalid="ignore"):
            s1 = shape @ w
            s2 = (shape * shape) @ w
            sy = shape @ (w * y)
            det = sw * s2 - s1 * s1
            b0 = (s2 * swy - s1 * sy) / det
            a = (s1 * swy - sw * sy) / det  # coefficient of -shape
            chi2 = chi2_flat - (b0 * swy - a * sy - flat * swy)  # chi2 = sum w y^2 - (b0 swy - a sy) at the optimum
            chi2 = np.where(a > 0, chi2, np.inf)
            best = np.argsort(chi2)[: self.cfg.n_refine]
        return [(float(chi2[i]), wi, int(i), float(b0[i]), float(a[i])) for i in best]

    def fit(self, y: np.ndarray, sigma: np.ndarray) -> FieldFit:
        """Fit a spectrum ``y`` (F,), any scale (e.g. counts or a readout ratio), with 1-sigma errors ``sigma``."""
        y = np.asarray(y, dtype=np.float64)
        w = 1.0 / np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12) ** 2
        flat = float(np.sum(w * y) / np.sum(w))
        chi2_flat = float(np.sum(w * (y - flat) ** 2))
        # Coarse: model y = b0 - a * shape; weighted normal equations in closed form per grid point.
        cands = []
        sw, swy = float(np.sum(w)), float(np.sum(w * y))
        for wi, shape in enumerate(self.grid_shapes):
            cands += self._coarse(wi, shape, w, y, sw, swy, chi2_flat, flat)
        cands.sort(key=lambda t: t[0])
        if not cands or chi2_flat - cands[0][0] < self.cfg.min_delta_chi2:
            return self._prior(chi2_flat - (cands[0][0] if cands else chi2_flat))
        sq = np.sqrt(w)

        def resid(p: np.ndarray) -> np.ndarray:
            lines = transition_lines_hz(p[:3] * 1e-3, self.d_hz)
            return sq * (y - (p[5] - p[4] * _dips(self.f, lines, np.exp(p[3]))))

        best_res = None
        for _, wi, gi, b0, a in cands[: self.cfg.n_refine]:
            p0 = np.array([*(self.grid[gi] * 1e3), np.log(self.cfg.trial_hwhm_hz[wi]), a, b0])
            try:
                r = scipy.optimize.least_squares(resid, p0, x_scale=[0.01, 0.01, 0.01, 0.1, abs(a) + 1e-9, abs(b0)])
            except (ValueError, np.linalg.LinAlgError):
                continue
            if best_res is None or r.cost < best_res.cost:
                best_res = r
        if best_res is None:
            return self._prior(chi2_flat - cands[0][0])
        chi2 = 2.0 * float(best_res.cost)
        dof = max(1, y.size - 6)
        jac = best_res.jac
        try:
            cov = np.linalg.pinv(jac.T @ jac) * max(1.0, chi2 / dof)
            sig = np.sqrt(np.clip(np.diag(cov)[:3], 0, None)) * 1e-3
        except np.linalg.LinAlgError:
            sig = self.prior_std.copy()
        b = best_res.x[:3] * 1e-3
        order = np.argsort(-np.abs(b))
        return FieldFit(
            b_t=np.abs(b)[order],
            sigma_t=np.minimum(sig[order], self.prior_std * 3),
            ok=True,
            hwhm_hz=float(np.exp(best_res.x[3])),
            contrast=float(best_res.x[4] / best_res.x[5]),
            delta_chi2=chi2_flat - chi2,
        )


def spectrum_from_counts(counts: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Classical spectrum + 1-sigma (Poisson) error from feature counts (T, F): the counts summed over the T time
    bins (CW: T = 1).

    Pulsed: the whole 600 ns readout is spin dependent here. With singlet shelving (lifetime ~0.41 us) a spin in
    ms = +-1 after the pi pulse stays darker for the full window, and the contrast of the last 300 ns is even larger
    than that of the first 300 ns, so the hardware habit of normalising by a late "re-polarised" window would
    cancel the signal (the ratio turns the dips into small peaks). Without laser drift no normalisation is needed;
    the plain sum is the best classical time-bin weighting among the simple windows.
    """
    x = np.asarray(counts, dtype=np.float64)
    y = x.sum(axis=0)
    return y, np.sqrt(np.maximum(y, 1.0))
