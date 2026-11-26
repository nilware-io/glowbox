"""Task 3 (project brief): predict the next fluorescence reading of the driven NV ensemble.

The system (one sequence, repeated for ``shots`` shots, all times on the 10 ns grid)::

    init laser (init_ns) | cycle 0 | cycle 1 | ... | cycle K-1
    cycle k = MW pulse (mw_ns, amplitude a_k, on one ODMR line) | readout laser (readout_ns) | dark gap (gap_ns)

The drive a_k is a hidden, temporally correlated random process: z_k is a stationary AR(1) process with
correlation ``rho`` and a_k = amp_center + amp_scale tanh(z_k). The pulse area ~ a_k mw_ns is up to ~1.5 pi, so
the reading's response to the drive is non-monotonic (Rabi oscillation), and the short readout only partly
repolarises the spins, so each reading also depends on earlier cycles (spin memory). The reading y_k is the
photon count of cycle k's readout window summed over the shots; the task is to predict y_{k+1} from the past
(no access to the drive).

Predictors (window of the last ``window`` cycles):

* mean (skill 0 by definition), persistence (y_hat_{k+1} = y_k);
* linear AR(window): ridge on the past readings;
* CNN decoder (:class:`~glowbox.learn.decoder.ConvDecoder`): T = readout bins (rebinned), F = past cycles,
  Gaussian NLL (mean and uncertainty);
* NV reservoir: the past readings (standardised) go through a fixed random pulse encoder
  (:class:`~glowbox.learn.reservoir.RandomPulseEncoder`) into a separate sequence on the same diamond, whose
  readout counts feed a ridge regression; optionally the raw readings are appended (reservoir + direct input).

Metric: skill = 1 - MSE / Var(y) on the noisy test readings (the observable target). Two upper references:
the noise ceiling 1 - E[Poisson variance] / Var(y) (a predictor that knew the next noiseless reading) and the
hidden-state oracle (:func:`oracle_next_mean`): E[y_{k+1} | the true drive history and spin state after cycle k],
averaging over the unknown next drive innovation, the Bayes-optimal prediction for anyone who knew everything
hidden; no predictor that sees only past readings can beat it on average.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from glowbox.control.pulse import SequenceArrays
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions, emission_and_state, expected_rate_cps, statics_for


@dataclass(frozen=True)
class DrivenSeries:
    """The driven system's protocol and drive process (assumptions of the task, not measurements)."""

    n_cycles: int = 64
    burn_in: int = 8
    init_ns: int = 2000
    mw_ns: int = 200
    readout_ns: int = 400
    gap_ns: int = 100
    #: Index of the driven line (axis-major, glowbox.learn.bit_memory.device_lines_hz).
    line_index: int = 0
    rho: float = 0.8
    amp_center: float = 0.5
    amp_scale: float = 0.45
    #: Readout bins are summed in groups of ``rebin`` for the CNN input.
    rebin: int = 2
    grid_ns: int = 10

    def __post_init__(self) -> None:
        g = self.grid_ns
        if self.init_ns % g or self.mw_ns % g or self.readout_ns % g or self.gap_ns % g:
            raise ValueError("all durations must be multiples of the grid")
        if (self.readout_ns // g) % self.rebin:
            raise ValueError("readout bins must be divisible by rebin")
        if not -1 < self.rho < 1:
            raise ValueError("rho must be in (-1, 1)")

    @property
    def total_cycles(self) -> int:
        return self.burn_in + self.n_cycles

    @property
    def cycle_bins(self) -> int:
        return (self.mw_ns + self.readout_ns + self.gap_ns) // self.grid_ns

    @property
    def readout_bins(self) -> int:
        return self.readout_ns // self.grid_ns

    @property
    def n_rebinned(self) -> int:
        return self.readout_bins // self.rebin

    @property
    def n_bins(self) -> int:
        return self.init_ns // self.grid_ns + self.total_cycles * self.cycle_bins

    def drives(self, n_seq: int, rng: np.random.Generator) -> np.ndarray:
        """MW amplitudes a_k, (n_seq, total_cycles), from a stationary AR(1) z_k (unit variance)."""
        k = self.total_cycles
        z = np.empty((n_seq, k))
        z[:, 0] = rng.standard_normal(n_seq)
        s = np.sqrt(1.0 - self.rho**2)
        for i in range(1, k):
            z[:, i] = self.rho * z[:, i - 1] + s * rng.standard_normal(n_seq)
        out: np.ndarray = self.amp_center + self.amp_scale * np.tanh(z)
        return out

    def arrays(self, amps: np.ndarray, line_hz: float, offset_hz: float = 0.0) -> SequenceArrays:
        """Per-bin arrays of the sequences for drive amplitudes ``amps`` (n_seq, total_cycles)."""
        g = self.grid_ns
        n, k = amps.shape
        if k != self.total_cycles:
            raise ValueError(f"need {self.total_cycles} amplitudes per sequence, got {k}")
        ni, nm, nr = self.init_ns // g, self.mw_ns // g, self.readout_ns // g
        cb = self.cycle_bins
        laser_c = np.zeros(cb, np.float32)
        laser_c[nm : nm + nr] = 1.0
        mw_c = np.zeros(cb, bool)
        mw_c[:nm] = True
        laser = np.concatenate([np.ones(ni, np.float32), np.tile(laser_c, k)])
        mwmask = np.concatenate([np.zeros(ni, bool), np.tile(mw_c, k)])
        amp_bins = np.concatenate([np.zeros((n, ni)), np.repeat(amps, cb, axis=1)], axis=1)
        amp = np.where(mwmask[None], amp_bins, 0.0).astype(np.float32)
        freq = np.where(mwmask[None], line_hz + offset_hz, 0.0) * np.ones((n, 1))
        return SequenceArrays(
            laser=np.tile(laser, (n, 1)),
            mw_amplitude=amp,
            mw_frequency_hz=freq,
            mw_phase_rad=np.zeros((n, self.n_bins), np.float32),
            magnet_t=np.zeros((n, 3)),
            grid_ns=g,
        )

    def readout_index(self) -> np.ndarray:
        """Bin indices of every cycle's readout window, (total_cycles, readout_bins)."""
        g = self.grid_ns
        start = self.init_ns // g + np.arange(self.total_cycles) * self.cycle_bins + self.mw_ns // g
        out: np.ndarray = start[:, None] + np.arange(self.readout_bins)[None]
        return out

    def simulate(
        self,
        amps: np.ndarray,
        env: EnvParams,
        line_hz: float,
        *,
        offset_hz: float = 0.0,
        options: ModelOptions | None = None,
        device: torch.device | str | None = None,
        chunk: int = 64,
    ) -> np.ndarray:
        """Mean counts per shot of every readout bin, rebinned: (n_seq, total_cycles, n_rebinned) float64."""
        idx = self.readout_index()
        n = amps.shape[0]
        out = np.empty((n, self.total_cycles, self.n_rebinned))
        dt_s = self.grid_ns * 1e-9
        for i in range(0, n, chunk):
            arr = self.arrays(amps[i : i + chunk], line_hz, offset_hz)
            with torch.no_grad():
                rate = expected_rate_cps(arr, env, options=options, device=device).double().cpu().numpy()
            ro = rate[:, idx] * dt_s  # (m, K, readout_bins)
            out[i : i + chunk] = ro.reshape(ro.shape[0], self.total_cycles, self.n_rebinned, self.rebin).sum(-1)
        return out

    def cycle_arrays(self, amps: np.ndarray, line_hz: float, offset_hz: float = 0.0) -> SequenceArrays:
        """One cycle per row for amplitudes ``amps`` (n,): MW, readout laser, gap (for state continuation)."""
        g = self.grid_ns
        nm, nr, cb = self.mw_ns // g, self.readout_ns // g, self.cycle_bins
        n = amps.shape[0]
        laser = np.zeros((n, cb), np.float32)
        laser[:, nm : nm + nr] = 1.0
        amp = np.zeros((n, cb), np.float32)
        amp[:, :nm] = amps[:, None]
        freq = np.zeros((n, cb))
        freq[:, :nm] = line_hz + offset_hz
        return SequenceArrays(laser, amp, freq, np.zeros((n, cb), np.float32), np.zeros((n, 3)), g)

    def states_after_cycles(
        self,
        amps: np.ndarray,
        env: EnvParams,
        line_hz: float,
        *,
        offset_hz: float = 0.0,
        options: ModelOptions | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Simulate cycle by cycle, continuing the state: (readout means (n, K, n_rebinned), states (n, K, ...)
        after every cycle). Exact: one MW frequency and phase throughout, so the rotating frame is continuous
        across the cuts (tests/learn/test_forecast.py checks it against :meth:`simulate`)."""
        g = self.grid_ns
        n = amps.shape[0]
        dt_s = g * 1e-9
        opt = options or ModelOptions()
        init = self.arrays(amps, line_hz, offset_hz)
        ni = self.init_ns // g
        pre = SequenceArrays(
            np.asarray(init.laser)[:1, :ni],
            np.asarray(init.mw_amplitude)[:1, :ni],
            np.asarray(init.mw_frequency_hz)[:1, :ni],
            np.asarray(init.mw_phase_rad)[:1, :ni],
            np.zeros((1, 3)),
            g,
        )
        st = statics_for(pre, env, opt, device)
        _, r = emission_and_state(pre, env, options=opt, device=device, statics=st)
        r = r.expand(n, *r.shape[1:]).contiguous()
        nm = self.mw_ns // g
        means = np.empty((n, self.total_cycles, self.n_rebinned))
        states = []
        for k in range(self.total_cycles):
            ca = self.cycle_arrays(amps[:, k], line_hz, offset_hz)
            with torch.no_grad():
                em, r = emission_and_state(ca, env, options=opt, device=device, initial_state=r, statics=st)
            rate = (
                env.collection_efficiency * env.n_nv * em
                + env.background_cps * torch.as_tensor(np.asarray(ca.laser), device=em.device).to(em.dtype)
                + env.dark_cps
            )
            ro = (rate[:, nm : nm + self.readout_bins] * dt_s).double().cpu().numpy()
            means[:, k] = ro.reshape(n, self.n_rebinned, self.rebin).sum(-1)
            states.append(r)
        return means, torch.stack(states, dim=1)

    def oracle_next_mean(
        self,
        z_prev: np.ndarray,
        state: torch.Tensor,
        env: EnvParams,
        line_hz: float,
        *,
        offset_hz: float = 0.0,
        n_quad: int = 16,
        options: ModelOptions | None = None,
        device: torch.device | str | None = None,
        chunk: int = 256,
    ) -> np.ndarray:
        """E[next reading per shot | z_k, spin state after cycle k], (n,): Gauss-Hermite average over the drive
        innovation z_{k+1} = rho z_k + sqrt(1 - rho^2) xi, xi ~ N(0, 1). ``chunk`` windows (x ``n_quad`` one-cycle
        sequences, each with its own propagator) per surrogate call."""
        x, w = np.polynomial.hermite_e.hermegauss(n_quad)
        w = w / w.sum()
        n = z_prev.shape[0]
        opt = options or ModelOptions()
        nm = self.mw_ns // self.grid_ns
        out = np.empty(n)
        for i in range(0, n, chunk):
            zc = z_prev[i : i + chunk]
            m = zc.shape[0]
            z = self.rho * zc[:, None] + np.sqrt(1 - self.rho**2) * x[None]  # (m, J)
            amps = (self.amp_center + self.amp_scale * np.tanh(z)).ravel()
            ca = self.cycle_arrays(amps, line_hz, offset_hz)
            r0 = state[i : i + chunk].repeat_interleave(n_quad, dim=0)
            with torch.no_grad():
                em, _ = emission_and_state(ca, env, options=opt, device=device, initial_state=r0)
            em_ro = em[:, nm : nm + self.readout_bins]
            rate = env.collection_efficiency * env.n_nv * em_ro + env.background_cps + env.dark_cps  # laser on
            reading = (rate.sum(dim=1) * self.grid_ns * 1e-9).double().cpu().numpy().reshape(m, n_quad)
            out[i : i + m] = reading @ w
        return out

    def latent(self, amps: np.ndarray) -> np.ndarray:
        """z_k from the amplitudes (inverse of the tanh map)."""
        out: np.ndarray = np.arctanh(np.clip((amps - self.amp_center) / self.amp_scale, -1 + 1e-12, 1 - 1e-12))
        return out


def windows(n_seq: int, series: DrivenSeries, window: int) -> tuple[np.ndarray, np.ndarray]:
    """(sequence index, target cycle) of every prediction: targets k + 1 with the inputs k - window + 1 .. k all
    after the burn-in, (n,) each."""
    first = series.burn_in + window
    ks = np.arange(first, series.total_cycles)
    seq = np.repeat(np.arange(n_seq), ks.size)
    return seq, np.tile(ks, n_seq)


def gather_windows(cells: np.ndarray, seq: np.ndarray, target: np.ndarray, window: int) -> np.ndarray:
    """Input cells of each window: ``cells`` (n_seq, K, ...) -> (n, window, ...), oldest first."""
    lags = np.arange(-window, 0)
    out: np.ndarray = cells[seq[:, None], target[:, None] + lags[None]]
    return out


def skill(pred: np.ndarray, target: np.ndarray) -> float:
    """1 - MSE / Var(target): 0 for the mean, 1 for a perfect prediction, < 0 worse than the mean."""
    t = np.asarray(target, dtype=np.float64)
    return float(1.0 - np.mean((np.asarray(pred, dtype=np.float64) - t) ** 2) / np.var(t))


def noise_ceiling(mean_readings: np.ndarray, noisy_readings: np.ndarray) -> float:
    """1 - E[Poisson variance] / Var(noisy readings): the skill of an oracle that knows the next noiseless
    reading (Poisson variance = mean count)."""
    return float(1.0 - np.mean(mean_readings) / np.var(noisy_readings))
