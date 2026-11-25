"""The quantum test (project brief, required for every task): resonant vs far-detuned microwaves, over seeds.

Every task runs twice, once with its microwaves as designed and once with every MW frequency shifted far off
resonance, so that no spin transition is driven: the spins then do not respond to the MW at all and whatever the
learner still achieves comes from MW-independent fluorescence (brightness, laser-only transients, B_perp level
mixing). The test reports, over at least five seeds, the performance in each condition, the gap and its Student-t
confidence interval over seeds, and whether the learned strategy uses microsecond-scale timing (Rabi or echo
structure).

Pieces:

* :class:`Condition`, :data:`RESONANT`, :func:`detuned_condition`; :func:`detune_sequence` shifts a sequence's
  MW; :func:`min_line_distance_hz` / :func:`check_no_line_driven` verify the detuned condition drives nothing.
* :func:`run_quantum_test` calls a task function ``task(condition, seed) -> {metric: value}`` for every condition
  and seed (paired: the same seed draws the same fields, initialisation and noise in both conditions) and returns
  a :class:`QuantumTestResult` with Student-t CIs over seeds (:func:`t_ci`) of each condition's mean and of the
  paired gap. With n = 5 seeds a percentile bootstrap of the mean (:func:`bootstrap_ci`, kept for statistics over
  many samples, e.g. fields) covers only ~83 % at a nominal 95 %: it uses the plug-in sd and normal-like quantiles,
  where the t interval uses t_4 = 2.776.
* Timing probes: :func:`timing_sensitivity` (a metric as a function of timing perturbations of the protocol),
  :func:`mw_timing_structure` (pulse areas in units of pi, free-evolution gaps vs T2*) and
  :func:`mw_envelope_spectrum` / :func:`rabi_band_fraction` (spectral content of a MW envelope at the Rabi
  frequency), :func:`time_saliency` (where in a trace a differentiable model looks).

No claim of quantum advantage is made or implied by a gap: it only says the performance depends on driving the spin
resonance.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from glowbox.control.pulse import MicrowavePulse, PulseSequence
from glowbox.sim.interface import EnvParams

#: Default MW shift of the detuned condition, Hz: far beyond every ODMR line of the field priors used so far
#: (lines within D +- ~100 MHz up to ~3 mT) and >> Rabi frequencies (<~ 20 MHz) and linewidths (<~ 10 MHz).
DETUNED_OFFSET_HZ: float = 400e6


@dataclass(frozen=True)
class Condition:
    """One arm of the quantum test: a name and the MW frequency shift applied to every MW segment."""

    name: str
    mw_offset_hz: float = 0.0

    @property
    def detuned(self) -> bool:
        return self.mw_offset_hz != 0.0


RESONANT = Condition("resonant", 0.0)


def detuned_condition(offset_hz: float = DETUNED_OFFSET_HZ) -> Condition:
    return Condition("detuned", offset_hz)


def detune_sequence(seq: PulseSequence, offset_hz: float) -> PulseSequence:
    """The same sequence with every MW frequency shifted by ``offset_hz`` (timing, phases, amplitudes unchanged)."""
    mws = tuple(dataclasses.replace(m, frequency_hz=m.frequency_hz + offset_hz) for m in seq.microwave)
    label = {**seq.label, "mw_offset_hz": offset_hz}
    return dataclasses.replace(seq, microwave=mws, label=label)


def min_line_distance_hz(freqs_hz: np.ndarray, field_t: np.ndarray, env: EnvParams) -> float:
    """Smallest |f_MW - f_line| over the given MW frequencies and all eight ODMR lines of ``field_t`` (total)."""
    from glowbox.analysis.field_fit import transition_lines_hz

    lines = transition_lines_hz(np.asarray(field_t, dtype=np.float64), env.d_hz, env.e_hz)
    f = np.asarray(freqs_hz, dtype=np.float64).ravel()
    return float(np.min(np.abs(f[:, None] - lines.ravel()[None, :])))


def check_no_line_driven(
    freqs_hz: np.ndarray, field_t: np.ndarray, env: EnvParams, n_widths: float = 10.0
) -> tuple[bool, float]:
    """True when every MW frequency is more than ``n_widths`` x (full-scale Rabi frequency + T2* FWHM) from every
    line, and the margin in Hz. At that distance the off-resonant transfer, ~(f_R / delta)^2, is below 1e-2 even
    at full MW amplitude (4e-4 at amplitude 0.2): the spins do not respond."""
    width = env.rabi_full_scale_hz + 1.0 / (math.pi * env.t2_star_s)
    d = min_line_distance_hz(freqs_hz, field_t, env)
    return d > n_widths * width, d


def sequence_mw_freqs(seq: PulseSequence) -> np.ndarray:
    return np.array([m.frequency_hz for m in seq.microwave])


# --- Statistics ---------------------------------------------------------------------------------------------------


def t_ci(values: Sequence[float] | np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """(mean, lower, upper): Student-t interval mean +- t_{n-1} s / sqrt(n) (s the sample sd, ddof 1).

    The primary interval over seeds. Exact for normal per-seed values and close for any reasonably symmetric spread;
    with n = 5 the percentile bootstrap is ~0.63x as wide and covers ~83 % at a nominal 95 %
    (tests/analysis/test_quantum_test.py checks both coverages).
    """
    from scipy.stats import t as student_t

    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(v.mean())
    if v.size == 1:
        return m, float("nan"), float("nan")
    half = float(student_t.ppf((1 + level) / 2, v.size - 1)) * float(v.std(ddof=1)) / math.sqrt(v.size)
    return m, m - half, m + half


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """(statistic, lower, upper): percentile bootstrap over the values.

    For many samples (fields, shots). Not for a handful of seeds: with n = 5 the interval of the mean covers ~83 %
    at a nominal 95 %; use :func:`t_ci` there.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    gen = rng or np.random.default_rng(0)
    idx = gen.integers(0, v.size, size=(n_boot, v.size))
    boots = np.array([stat(v[i]) for i in idx]) if stat is not np.mean else v[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    return float(stat(v)), float(lo), float(hi)


@dataclass
class QuantumTestResult:
    """Per-seed metrics of every condition, and summaries.

    ``values[condition][metric]`` is the list over ``seeds`` (same order). ``higher_is_better[metric]`` sets the
    sign convention of the gap: gap = (resonant - detuned) when higher is better, (detuned - resonant) otherwise,
    so a positive gap always means "better on resonance".
    """

    seeds: list[int]
    conditions: list[Condition]
    values: dict[str, dict[str, list[float]]]
    higher_is_better: dict[str, bool] = field(default_factory=dict)
    timing: dict[str, object] = field(default_factory=dict)

    def metrics(self) -> list[str]:
        first = self.conditions[0].name
        return list(self.values[first])

    def gap_per_seed(self, metric: str, a: str = "resonant", b: str = "detuned") -> np.ndarray:
        va, vb = np.asarray(self.values[a][metric]), np.asarray(self.values[b][metric])
        return va - vb if self.higher_is_better.get(metric, False) else vb - va

    def summary(self, metric: str, level: float = 0.95) -> dict[str, tuple[float, float, float]]:
        """{condition: (mean, lo, hi), "gap": (mean, lo, hi)}: Student-t CIs over seeds (:func:`t_ci`), the gap
        paired per seed."""
        out = {c.name: t_ci(self.values[c.name][metric], level=level) for c in self.conditions}
        if {"resonant", "detuned"} <= set(self.values):
            out["gap"] = t_ci(self.gap_per_seed(metric), level=level)
        return out

    def gap_significant(self, metric: str, level: float = 0.95) -> bool:
        """True when the CI of the gap excludes zero (on the 'better on resonance' side)."""
        _, lo, _ = self.summary(metric, level=level)["gap"]
        return lo > 0

    def to_markdown(self, metrics: Sequence[str] | None = None, fmt: str = "{:.4g}") -> str:
        """A table: metric, each condition's mean [95 % t CI over seeds], gap [95 % t CI], per-seed values."""
        ms = list(metrics or self.metrics())
        names = [c.name for c in self.conditions]
        head = ["metric", *names, "gap (better on resonance > 0)", "per seed (" + " / ".join(names) + ")"]
        lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
        for m in ms:
            s = self.summary(m)
            cells = [m]
            for key in [*names, "gap"]:
                if key in s:
                    v, lo, hi = s[key]
                    cells.append(f"{fmt.format(v)} [{fmt.format(lo)}, {fmt.format(hi)}]")
                else:
                    cells.append("")
            per = "; ".join(", ".join(fmt.format(x) for x in self.values[n][m]) for n in names)
            cells.append(per)
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)


def run_quantum_test(
    task: Callable[[Condition, int], Mapping[str, float]],
    seeds: Sequence[int],
    conditions: Sequence[Condition] = (RESONANT, detuned_condition()),
    higher_is_better: Mapping[str, bool] | None = None,
    on_result: Callable[[Condition, int, Mapping[str, float]], None] | None = None,
) -> QuantumTestResult:
    """Run ``task(condition, seed)`` for every seed and condition (seed-major, so a seed's pair runs together)."""
    if len(seeds) < 5:
        raise ValueError("the quantum test needs at least 5 seeds (project brief)")
    values: dict[str, dict[str, list[float]]] = {c.name: {} for c in conditions}
    for s in seeds:
        for cond in conditions:
            res = task(cond, int(s))
            for k, v in res.items():
                values[cond.name].setdefault(k, []).append(float(v))
            if on_result is not None:
                on_result(cond, int(s), res)
    return QuantumTestResult(list(map(int, seeds)), list(conditions), values, dict(higher_is_better or {}))


# --- Timing-structure probes ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingSensitivity:
    """A metric evaluated at timing perturbations (ns); ``relative`` = metric / metric at 0 - 1."""

    shifts_ns: np.ndarray
    metric: np.ndarray

    @property
    def relative(self) -> np.ndarray:
        i0 = int(np.argmin(np.abs(self.shifts_ns)))
        return self.metric / self.metric[i0] - 1.0


def timing_sensitivity(score: Callable[[int], float], shifts_ns: Sequence[int]) -> TimingSensitivity:
    """Evaluate ``score(shift_ns)`` (e.g. a frozen decoder's error when a protocol's pulse length is changed by
    ``shift_ns``) over the shifts. A strategy that relies on Rabi/echo timing degrades for shifts that are a
    sizeable fraction of a Rabi period (~100 ns-1 us); one that does not is flat."""
    s = np.asarray(list(shifts_ns), dtype=np.int64)
    return TimingSensitivity(s, np.array([float(score(int(x))) for x in s]))


@dataclass(frozen=True)
class SegmentTiming:
    start_ns: int
    duration_ns: int
    frequency_hz: float
    #: Resonant rotation angle / pi of the pulse at the env's Rabi frequency (|m_perp| = sqrt(2/3) assumed: MW along
    #: [100], the default; 1 = pi pulse, 0.5 = pi/2).
    area_pi: float


def mw_timing_structure(seq: PulseSequence, env: EnvParams) -> dict[str, object]:
    """Pulse areas (units of pi) of every MW segment, the free-evolution gaps between them and their ratio to T2*.

    A sequence "uses microsecond-scale spin timing" when it has pulses whose areas are near multiples of pi/2 and/or
    MW-free gaps of order T2* (Ramsey) to T2 (echo) between MW pulses; CW drive has neither.
    """
    f_r = env.rabi_full_scale_hz * math.sqrt(2.0 / 3.0)
    segs = [
        SegmentTiming(m.start_ns, m.duration_ns, m.frequency_hz, 2.0 * f_r * m.amplitude * m.duration_ns * 1e-9)
        for m in seq.microwave
    ]
    gaps = [b.start_ns - (a.start_ns + a.duration_ns) for a, b in zip(seq.microwave, seq.microwave[1:], strict=False)]
    laser_with_mw = any(lp.start_ns < m.end_ns and m.start_ns < lp.end_ns for lp in seq.laser for m in seq.microwave)
    return {
        "segments": [dataclasses.asdict(s) for s in segs],
        "gaps_ns": gaps,
        "gaps_over_t2_star": [g * 1e-9 / env.t2_star_s for g in gaps],
        "cw_like": laser_with_mw,
    }


def mw_envelope(seq: PulseSequence) -> np.ndarray:
    """Complex MW envelope a e^{i phi} per bin (the rotating-frame drive), (n_bins,)."""
    env = np.zeros(seq.n_bins, dtype=complex)
    g = seq.grid_ns
    m: MicrowavePulse
    for m in seq.microwave:
        env[m.start_ns // g : m.end_ns // g] = m.amplitude * np.exp(1j * m.phase_rad)
    return env


def mw_envelope_spectrum(seq: PulseSequence) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies Hz, power) of the MW envelope's time structure (mean removed), one-sided magnitude."""
    e = mw_envelope(seq)
    e = e - e.mean()
    n = 8 * e.size
    p = np.abs(np.fft.fft(e, n)) ** 2
    f = np.abs(np.fft.fftfreq(n, seq.grid_ns * 1e-9))
    return f, p


def rabi_band_fraction(seq: PulseSequence, rabi_hz: float, rel_bandwidth: float = 0.25) -> float:
    """Fraction of the MW envelope's (mean-removed) power within +-``rel_bandwidth`` of ``rabi_hz`` and its first
    two harmonics. Near 0 for CW or slowly varying drive; large when the drive is switched on Rabi time scales."""
    f, p = mw_envelope_spectrum(seq)
    tot = float(p.sum())
    if tot <= 0:
        return 0.0
    band = np.zeros_like(f, dtype=bool)
    for h in (1, 2, 3):
        band |= np.abs(f - h * rabi_hz) <= rel_bandwidth * rabi_hz
    return float(p[band].sum() / tot)


def time_saliency(
    model: torch.nn.Module, x: torch.Tensor, c: torch.Tensor, output: Callable[[torch.Tensor], torch.Tensor]
) -> np.ndarray:
    """Mean |d output / d x| over a batch, summed over the last axis: saliency per time bin, (T,).

    ``x`` (B, T, F) decoder input, ``c`` its conditioning; ``output(mean)`` reduces the model's mean to a scalar per
    sample (e.g. the field magnitude). Tells which readout time bins the model relies on.
    """
    model.eval()
    xg = x.detach().clone().requires_grad_(True)
    mean, _ = model(xg, c)
    output(mean).sum().backward()
    assert xg.grad is not None
    return xg.grad.abs().mean(dim=0).sum(dim=-1).detach().cpu().numpy()
