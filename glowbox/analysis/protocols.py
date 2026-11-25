"""Read protocol features out of fluorescence: CW ODMR dips and pulsed-Rabi oscillations.

Shared by the surrogate-vs-reference tests and the milestone-1 experiment, so both judge the two simulators with the
same yardstick. Inputs are noiseless ``rate_cps`` rows or shot-averaged counts; all functions are pure NumPy/SciPy.

Conventions (docs/physics.md section 8): CW ODMR uses the steady-state rate of a long laser+MW pulse (late window);
pulsed readout uses the first ``window_ns`` of the last laser pulse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.optimize
import scipy.signal

from glowbox.control.pulse import PulseSequence


def late_mean(rate: np.ndarray, grid_ns: int, late_ns: int) -> float:
    """Mean of the last ``late_ns`` of a per-bin rate (steady state of a CW point)."""
    n = max(1, late_ns // grid_ns)
    return float(np.mean(np.asarray(rate)[..., -n:]))


def readout_mean(rate: np.ndarray, sequence: PulseSequence, window_ns: int = 300) -> float:
    """Mean rate over the first ``window_ns`` of the sequence's last laser pulse (spin-dependent readout window)."""
    if not sequence.laser:
        raise ValueError("sequence has no laser pulse to read out")
    lp = sequence.laser[-1]
    a = lp.start_ns // sequence.grid_ns
    b = a + min(window_ns, lp.duration_ns) // sequence.grid_ns
    return float(np.mean(np.asarray(rate)[..., a:b]))


@dataclass(frozen=True)
class Dip:
    """One ODMR dip: grid minimum, and a Lorentzian fit to the points around it (Hz, fractional contrast)."""

    grid_frequency_hz: float
    grid_contrast: float
    center_hz: float
    contrast: float
    fwhm_hz: float


def lorentzian_dip(f: np.ndarray, baseline: float, contrast: float, center: float, hwhm: float) -> np.ndarray:
    """baseline * (1 - contrast / (1 + ((f - center) / hwhm)^2))."""
    return baseline * (1.0 - contrast / (1.0 + ((f - center) / hwhm) ** 2))


def find_dips(
    freqs_hz: np.ndarray,
    signal: np.ndarray,
    baseline: float,
    min_contrast: float = 1e-3,
    fit_half_width_hz: float = 4e6,
) -> list[Dip]:
    """Local minima of a CW ODMR spectrum deeper than ``min_contrast`` (relative to ``baseline``), each refined by
    a Lorentzian fit over +-``fit_half_width_hz`` with the baseline fixed. Sorted by frequency.

    The CW line of a Lorentzian T2* ensemble, power-broadened by the MW and the optical pumping, is a Lorentzian
    (a convolution of Lorentzians), so the fit's FWHM is the line's FWHM; overlapping lines bias it.
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    c = 1.0 - np.asarray(signal, dtype=np.float64) / baseline
    peaks, _ = scipy.signal.find_peaks(np.concatenate([[-np.inf], c, [-np.inf]]), height=min_contrast)
    dips = []
    for p in peaks - 1:
        sel = np.abs(f - f[p]) <= fit_half_width_hz
        step = float(np.median(np.diff(f)))

        def model(x: np.ndarray, contrast: float, center: float, hwhm: float) -> np.ndarray:
            return lorentzian_dip(x, 1.0, contrast, center, hwhm)

        p0 = (float(c[p]), float(f[p]), max(step, 0.5e6))
        (con, cen, hw), _ = scipy.optimize.curve_fit(model, f[sel], 1.0 - c[sel], p0=p0, maxfev=10_000)
        dips.append(Dip(float(f[p]), float(c[p]), float(cen), float(con), 2.0 * abs(float(hw))))
    return dips


@dataclass(frozen=True)
class RabiFit:
    """signal(t) = offset + amplitude * exp(-t / decay_s) * cos(2 pi frequency_hz t + phase_rad)."""

    frequency_hz: float
    decay_s: float
    amplitude: float
    offset: float
    phase_rad: float
    rms_residual: float

    def __call__(self, t_s: np.ndarray) -> np.ndarray:
        return damped_cosine(t_s, self.offset, self.amplitude, self.frequency_hz, self.decay_s, self.phase_rad)


def damped_cosine(t: np.ndarray, offset: float, amp: float, freq: float, decay: float, phase: float) -> np.ndarray:
    return offset + amp * np.exp(-t / decay) * np.cos(2.0 * math.pi * freq * t + phase)


def fit_rabi(t_s: np.ndarray, signal: np.ndarray) -> RabiFit:
    """Fit a damped cosine to a Rabi curve (contrast or readout rate vs MW duration); start from the FFT peak.

    Exponential damping is a model choice: the T2*-averaged Rabi envelope of a Lorentzian ensemble at f_R >> Gamma
    decays slower than exponential, so ``decay_s`` is an effective number, good for comparing two simulators on
    the same durations, not a physical T2*.
    """
    t = np.asarray(t_s, dtype=np.float64)
    y = np.asarray(signal, dtype=np.float64)
    dt = float(t[1] - t[0])
    spec = np.abs(np.fft.rfft(y - y.mean(), n=16 * y.size))
    fq = np.fft.rfftfreq(16 * y.size, dt)
    f0 = float(fq[1 + np.argmax(spec[1:])])
    amp0 = 0.5 * float(y.max() - y.min())
    span = float(t[-1] - t[0])
    p0 = (float(y.mean()), amp0 if y[0] > y.mean() else -amp0, f0, span, 0.0)
    popt, _ = scipy.optimize.curve_fit(damped_cosine, t, y, p0=p0, maxfev=20_000)
    off, amp, freq, decay, phase = (float(v) for v in popt)
    if freq < 0:  # cos(-x + p) = cos(x - p)
        freq, phase = -freq, -phase
    if amp < 0:  # canonical sign: positive amplitude, phase absorbs the flip
        amp, phase = -amp, phase + math.pi
    phase = math.remainder(phase, 2 * math.pi)
    res = float(np.sqrt(np.mean((y - damped_cosine(t, *popt)) ** 2)))
    return RabiFit(freq, decay, amp, off, phase, res)
