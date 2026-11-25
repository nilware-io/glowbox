"""The fast surrogate against the QuTiP reference (first milestone, docs/physics.md section 9).

Both implement the same equations (docs/physics.md, including the δ·Sz detuning, the secular dissipator rule and the
frame bookkeeping) with independent code: the reference builds 49x49 Liouvillians from QuTiP operators, the
surrogate a reduced real 14x14 generator in torch. With the same quadrature size M they integrate over the *same*
detuning nodes, so any difference is implementation error, not physics or sampling. That is why the tolerances
below are tight; each states the numbers it was set from (measured on CPU, 2026-11-25).

Default surrogate dtype is float32 (what runs on the GPU); one test also checks float64 to show the residual is
float32 round-off. Runtime: ~30-60 s on CPU, dominated by the reference.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from glowbox.analysis.protocols import find_dips, fit_rabi, late_mean, readout_mean
from glowbox.control.pulse import PulseSequence, cw_odmr_point, pulsed_rabi
from glowbox.sim.interface import EnvParams
from glowbox.sim.reference import ReferenceSimulator, detuning_quadrature_hz, transition_frequencies_hz
from glowbox.sim.surrogate import ModelOptions, SurrogateSimulator, detuning_quadrature

M = 32  # quadrature nodes, the same for both simulators (identical nodes)
GRID_HZ = 0.25e6  # ODMR sweep step
CW_NS, CW_LATE_NS = 4000, 1500  # CW point length and the late (steady-state) window
# B along [001]: all four axes equivalent (one reference group, fast), two lines ~100 MHz apart.
ENV_ODMR = EnvParams(b_field_t=(0.0, 0.0, 3e-3))
# Rabi: shorter T2* so the decay is visible within the sweep, amplitude 0.5 (f_R = 4.08 MHz).
ENV_RABI = dataclasses.replace(ENV_ODMR, t2_star_s=0.3e-6)
RABI_AMP = 0.5
RABI_DURATIONS_NS = np.arange(0, 610, 20)
# Generic field: four inequivalent axes, eight lines (per-bin and detuned-control tests).
ENV_GENERIC = EnvParams(b_field_t=(1.2e-3, 0.7e-3, 2.5e-3))


def lines_hz(env: EnvParams) -> np.ndarray:
    return transition_frequencies_hz(env.d_hz, env.e_hz, np.asarray(env.b_field_t))


@pytest.fixture(scope="module")
def ref() -> ReferenceSimulator:
    return ReferenceSimulator(n_detuning=M)


@pytest.fixture(scope="module")
def sur() -> SurrogateSimulator:
    return SurrogateSimulator("cpu", ModelOptions(n_detuning=M))


@pytest.fixture(scope="module")
def odmr(ref: ReferenceSimulator, sur: SurrogateSimulator) -> dict[str, object]:
    """CW ODMR spectra (steady-state rate vs MW frequency) around both lines, from both simulators."""
    f_minus, f_plus = lines_hz(ENV_ODMR)[0]
    freqs = np.concatenate([f + np.arange(-4e6, 4e6 + 1, GRID_HZ) for f in (f_minus, f_plus)])
    seqs = [cw_odmr_point(float(f), duration_ns=CW_NS) for f in freqs]
    off = dataclasses.replace(seqs[0], microwave=())
    g = seqs[0].grid_ns
    ref_sig = np.array([late_mean(ref.expected_rate_cps(s, ENV_ODMR), g, CW_LATE_NS) for s in seqs])
    sur_sig = np.array([late_mean(r, g, CW_LATE_NS) for r in sur.expected_rates(seqs, ENV_ODMR)])
    return {
        "freqs": freqs,
        "lines": (f_minus, f_plus),
        "ref": (ref_sig, late_mean(ref.expected_rate_cps(off, ENV_ODMR), g, CW_LATE_NS)),
        "sur": (sur_sig, late_mean(sur.expected_rates([off], ENV_ODMR)[0], g, CW_LATE_NS)),
    }


def rabi_sequences(env: EnvParams) -> list[PulseSequence]:
    f_minus = float(lines_hz(env)[0, 0])
    return [pulsed_rabi(int(t), f_minus, RABI_AMP, init_ns=2000, readout_ns=400) for t in RABI_DURATIONS_NS]


@pytest.fixture(scope="module")
def rabi(ref: ReferenceSimulator) -> dict[str, object]:
    seqs = rabi_sequences(ENV_RABI)
    return {"seqs": seqs, "ref_rates": [ref.expected_rate_cps(s, ENV_RABI) for s in seqs]}


def rabi_contrast(rates: list[np.ndarray], seqs: list[PulseSequence]) -> np.ndarray:
    """1 - readout(t) / readout(0): fraction of the ms=0 brightness lost after a MW pulse of duration t."""
    r = np.array([readout_mean(x, s) for x, s in zip(rates, seqs, strict=True)])
    return 1.0 - r / r[0]


# --- per-bin noiseless rates --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_per_bin_rates_agree_generic_field(ref: ReferenceSimulator, dtype: torch.dtype) -> None:
    """Noiseless rate_cps per bin, eight-line generic field (B_perp 1.5-3 mT per axis), Rabi, CW, detuned, MW off.

    float64: measured <= 2e-9 of the peak rate. The two remaining differences are the delta-term treatment (the
    reference drops the 0~ <-> +-~ part of delta Sz, a shift ~delta^2 (gamma B_perp/D)^2/D) and expm round-off.
    Tolerance 1e-7. float32 (default surrogate): measured <= 8e-5 (round-off of the 14x14 propagators accumulated
    over ~400 bins); tolerance 5e-4.
    """
    sur = SurrogateSimulator("cpu", ModelOptions(n_detuning=M, dtype=dtype))
    f = float(lines_hz(ENV_GENERIC)[2, 0])
    seqs = [
        pulsed_rabi(60, f, init_ns=1500, settle_ns=1500, readout_ns=600),
        cw_odmr_point(f, duration_ns=3000),
        pulsed_rabi(200, f + 200e6, init_ns=1500, settle_ns=1500, readout_ns=600),
    ]
    tol = 1e-7 if dtype == torch.float64 else 5e-4
    for s, q in zip(seqs, sur.expected_rates(seqs, ENV_GENERIC), strict=True):
        r = ref.expected_rate_cps(s, ENV_GENERIC)
        assert np.max(np.abs(q - r)) / np.max(r) < tol, s.label


# --- CW ODMR ------------------------------------------------------------------------------------------------------


def test_cw_odmr_dips_contrast_linewidth(odmr: dict) -> None:
    freqs = odmr["freqs"]
    (ref_sig, ref_off), (sur_sig, sur_off) = odmr["ref"], odmr["sur"]
    # MW-off steady state: measured 9e-5 relative (float32); 5e-4 like the per-bin bound.
    assert abs(sur_off - ref_off) / ref_off < 5e-4
    rd, sd = find_dips(freqs, ref_sig, ref_off), find_dips(freqs, sur_sig, sur_off)
    assert len(rd) == len(sd) == 2
    for line, a, b in zip(odmr["lines"], rd, sd, strict=True):
        # Dip positions: the same grid point in both, and the predicted line (numerical H0 eigenvalues) within one
        # grid step of it (the symmetric T2* distribution does not shift the line).
        assert a.grid_frequency_hz == b.grid_frequency_hz
        assert abs(a.grid_frequency_hz - line) <= GRID_HZ / 2 + 1
        # Fitted centres: measured <= 140 Hz apart; the fit's own uncertainty from float32 noise is ~1e-4 of the
        # FWHM (~0.5 kHz). Tolerance 5 kHz (1/800 of the 4.4 MHz FWHM). Both within 20 kHz of the prediction.
        assert abs(a.center_hz - b.center_hz) < 5e3
        assert abs(a.center_hz - line) < 20e3
        # Contrast (~18 %) and FWHM (~4.4 MHz, power-broadened): measured 1e-5 / 4e-4 relative. A 1e-4 rate error
        # moves a contrast of 0.18 by ~5e-4 relative; tolerance 2e-3 for both.
        assert abs(a.contrast - b.contrast) / a.contrast < 2e-3
        assert abs(a.fwhm_hz - b.fwhm_hz) / a.fwhm_hz < 2e-3
        assert 0.05 < a.contrast < 0.4  # sanity: a real dip, not noise


@pytest.mark.parametrize("shape", ["lorentzian", "gaussian"])
@pytest.mark.parametrize("rule", ["quantile", "uniform"])
def test_detuning_quadrature_identical(shape: str, rule: str) -> None:
    """Both simulators build the same nodes and weights (independent code, docs/physics.md section 4)."""
    env = EnvParams(broadening_shape=shape)  # type: ignore[arg-type]
    for m in (1, 2, 7, 32):
        dr, wr = detuning_quadrature_hz(env, m, rule, 0.75)  # type: ignore[arg-type]
        ds, ws = detuning_quadrature(env, m, rule, 0.75)
        np.testing.assert_allclose(dr, ds, rtol=1e-12, atol=1e-6)
        np.testing.assert_allclose(wr, ws, rtol=1e-12)
        assert abs(wr.sum() - 1) < 1e-12


def test_cw_odmr_uniform_rule_agrees() -> None:
    """The weighted "uniform" rule gives the same rates in both simulators (float64; a few points over one line)."""
    env = dataclasses.replace(ENV_ODMR, orientation_populations=(1.0, 0.0, 0.0, 0.0))
    line = float(transition_frequencies_hz(env.d_hz, 0, np.array(env.b_field_t))[0, 0])
    seqs = [cw_odmr_point(line + d, duration_ns=2000, amplitude=0.05) for d in (-3e6, -0.5e6, 0.0, 1e6)]
    ref = ReferenceSimulator(n_detuning=12, detuning_rule="uniform", detuning_spacing=0.75)
    opts = ModelOptions(n_detuning=12, dtype=torch.float64, detuning_rule="uniform", detuning_spacing=0.75)
    a = np.array([ref.expected_rate_cps(s, env) for s in seqs])
    b = np.array(SurrogateSimulator("cpu", opts).expected_rates(seqs, env))
    # Same equations, nodes and weights: float64 agreement like the quantile rule (~1e-9 of the peak rate).
    assert np.abs(a - b).max() / a.max() < 1e-7


# --- pulsed Rabi --------------------------------------------------------------------------------------------------


def test_rabi_frequency_decay_contrast(rabi: dict, sur: SurrogateSimulator) -> None:
    seqs = rabi["seqs"]
    t_s = RABI_DURATIONS_NS * 1e-9
    c_ref = rabi_contrast(rabi["ref_rates"], seqs)
    c_sur = rabi_contrast(sur.expected_rates(seqs, ENV_RABI), seqs)
    # Point by point: measured 2.6e-5 in contrast units (contrast 0..0.20); tolerance 5e-4 as for the rates.
    assert np.max(np.abs(c_sur - c_ref)) < 5e-4
    fr, fs = fit_rabi(t_s, c_ref), fit_rabi(t_s, c_sur)
    # Fitted parameters: measured relative differences 1e-5 (frequency), 3e-4 (decay), 1e-7 (amplitude). The decay
    # is the least constrained parameter (effective exponential over 2.5 periods). Tolerances: 1e-3, 1e-2, 5e-3.
    assert abs(fs.frequency_hz - fr.frequency_hz) / fr.frequency_hz < 1e-3
    assert abs(fs.decay_s - fr.decay_s) / fr.decay_s < 1e-2
    assert abs(fs.amplitude - fr.amplitude) / fr.amplitude < 5e-3
    # Physics sanity: on resonance the oscillation is at f_R = rabi_full_scale * a * |m_perp| (spec section 3).
    # Measured +1.5 %: the Lorentzian tails (Gamma = 0.53 MHz) add off-resonant spins that oscillate at
    # sqrt(f_R^2 + delta^2); B_perp (2.4 mT) changes the drive matrix element by ~0.1 %. Tolerance 5 %.
    f_r = ENV_RABI.rabi_full_scale_hz * RABI_AMP * math.sqrt(2.0 / 3.0)
    assert abs(fr.frequency_hz / f_r - 1) < 0.05
    # It does decay: measured 2.7 us. The Lorentzian-averaged Rabi envelope at f_R / Gamma ~ 8 falls off slower than
    # an exponential, so an exponential fit over 0.6 us reads ~9 T2*.
    assert fr.decay_s < 5e-6


# --- detuned control and shot noise -------------------------------------------------------------------------------


def test_detuned_control_and_shot_noise_same_seed(ref: ReferenceSimulator, sur: SurrogateSimulator) -> None:
    """MW 200 MHz from every line = MW off within 1e-3 (spec section 9.5), in both simulators; the resonant pulse
    is clearly visible. Counts from the same seed are different Poisson draws (NumPy vs torch generator) of the
    same means: window totals agree within 4 sigma, and each simulator reproduces itself from the seed."""
    env = ENV_GENERIC
    lines = lines_hz(env)
    f_far = float(lines.max() + 200e6)
    assert np.min(np.abs(lines - f_far)) >= 200e6
    kw = {"init_ns": 1500, "settle_ns": 1500, "readout_ns": 600}
    det = pulsed_rabi(100, f_far, **kw)
    off = dataclasses.replace(det, microwave=())
    on = pulsed_rabi(100, float(lines[2, 0]), **kw)
    for sim in (ref, sur):
        r_off, r_det, r_on = (sim.run(s, env).rate_cps[0] for s in (off, det, on))  # type: ignore[index]
        # Measured: reference 1.3e-4 (the far-off-resonant drive still shifts the levels slightly), surrogate 2.2e-4
        # (the same plus float32 round-off).
        assert np.max(np.abs(r_det - r_off) / r_off.max()) < 1e-3
        assert abs(readout_mean(r_on, on) / readout_mean(r_off, off) - 1) > 0.01  # resonant: -2.3 % (one axis of 4)
    shots = 400
    seed = 1234
    tr_ref = ref.run(det, env, shots=shots, rng=np.random.default_rng(seed))
    tr_sur = sur.run(det, env, shots=shots, rng=np.random.default_rng(seed))
    again = sur.run(det, env, shots=shots, rng=np.random.default_rng(seed))
    np.testing.assert_array_equal(tr_sur.counts, again.counts)
    lp = det.laser[-1]
    n_ref = tr_ref.window_counts(lp.start_ns, lp.end_ns).sum()
    n_sur = tr_sur.window_counts(lp.start_ns, lp.end_ns).sum()
    assert abs(n_ref - n_sur) < 4 * math.sqrt(n_ref + n_sur)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_surrogate_matches_reference_rabi(rabi: dict) -> None:
    """The GPU surrogate (float32) on the same Rabi sweep: same 5e-4 bound as on CPU."""
    seqs = rabi["seqs"]
    sur = SurrogateSimulator("cuda", ModelOptions(n_detuning=M))
    for q, r in zip(sur.expected_rates(seqs, ENV_RABI), rabi["ref_rates"], strict=True):
        assert np.max(np.abs(q - r)) / np.max(r) < 5e-4
