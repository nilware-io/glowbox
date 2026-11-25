"""Validation suite of docs/physics.md section 9 on the QuTiP reference: CW ODMR, Rabi, Ramsey, Hahn echo and the
detuned control. Small, fast cases; every expected number is derived from the spec, not from the simulator.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import scipy.optimize

from glowbox.control.pulse import PulseSequence, SequenceBuilder, cw_odmr_point, hahn_echo, pulsed_rabi, ramsey
from glowbox.sim import constants as c
from glowbox.sim.interface import EnvParams
from glowbox.sim.reference import ReferenceSimulator, detuning_samples_hz
from glowbox.sim.reference.model import axis_geometry, diagonalise

N1 = c.NV_AXES[0]
MPERP_100 = math.sqrt(2.0 / 3.0)  # |m_perp| of every <111> axis for MW along [100]
# Protocol timings for tests: init long enough to polarise at beta = 0.1; settle = 3 us (~7 singlet lifetimes of
# 0.41 us), because with the default 1 us the ~3 % singlet remnant keeps decaying into the ground state during the
# swept free time and adds a slow baseline drift of ~10 % of the fringe amplitude; readout long enough for the
# 300 ns contrast window (spec section 8).
TIMING = {"init_ns": 2000, "settle_ns": 3000, "readout_ns": 400}
WINDOW_NS = 300


def single_axis_env(**kw: object) -> EnvParams:
    """All NVs on axis 1, 3 mT along it: f+- = D +- 84.1 MHz, B_perp = 0."""
    base = EnvParams(b_field_t=tuple(3e-3 * N1), orientation_populations=(1.0, 0.0, 0.0, 0.0))
    return dataclasses.replace(base, **kw)


def f_minus_single_axis(env: EnvParams) -> float:
    return env.d_hz - c.GAMMA_E_HZ_PER_T * 3e-3


def readout_signal(sim: ReferenceSimulator, seq: PulseSequence, env: EnvParams) -> float:
    """Noiseless detected counts in the first 300 ns of the last laser pulse (spin-state readout)."""
    rate = sim.expected_rate_cps(seq, env)
    a = seq.laser[-1].start_ns // seq.grid_ns
    return float(rate[a : a + WINDOW_NS // seq.grid_ns].sum() * seq.grid_ns * 1e-9)


def cw_level(sim: ReferenceSimulator, seq: PulseSequence, env: EnvParams, late_ns: int = 1000) -> float:
    """Mean detected rate over the last ``late_ns`` of a CW sequence (near steady state)."""
    rate = sim.expected_rate_cps(seq, env)
    return float(rate[-late_ns // seq.grid_ns :].mean())


# --- 1. CW ODMR -------------------------------------------------------------------------------------------------


def test_cw_odmr_eight_lines_at_spec_frequencies():
    """Generic field: up to 8 lines at f+- of each axis (numerical diagonalisation, spec section 1) and within the
    B_perp second-order shift of D +- gamma_e B.n; ensemble dip contrast of a few per cent at low MW power."""
    b = np.array([2.5e-3, 1.0e-3, 0.3e-3])
    env = EnvParams(b_field_t=tuple(b))
    sim = ReferenceSimulator(n_detuning=8)
    duration = 4000

    def level(f: float) -> float:
        return cw_level(sim, cw_odmr_point(f, duration_ns=duration, amplitude=0.1), env)

    off = cw_level(sim, dataclasses.replace(cw_odmr_point(3e9, duration_ns=duration), microwave=()), env)
    lines = []
    for k in range(4):
        geom = axis_geometry(c.NV_AXES[k], b, np.array(env.mw_field_direction), canonical=False)
        energies, _ = diagonalise(env.d_hz, env.e_hz, geom.b_local_hz)
        par_hz, perp_hz = c.b_projections_hz(b)[k]
        for sign, f_exact in ((-1, energies[1]), (1, energies[2])):
            lines.append((f_exact, env.d_hz + sign * abs(par_hz), perp_hz))
    for f_exact, f_first_order, perp_hz in lines:
        # Second-order B_perp shift is ~ (3/2) (gamma B_perp)^2 / D (Doherty 2013); allow twice that.
        assert abs(f_exact - f_first_order) < 3 * perp_hz**2 / env.d_hz + 1e3
        step = 0.25e6
        freqs = f_exact + step * np.arange(-3, 4)
        levels = np.array([level(f) for f in freqs])
        i = int(np.argmin(levels))
        assert 1 <= i <= 5, "dip must be near the predicted frequency"
        # parabola through the three lowest points: sub-step dip position
        y0, y1, y2 = levels[i - 1 : i + 2]
        f_dip = freqs[i] + step * 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        assert abs(f_dip - f_exact) < 0.05e6
        contrast = 1 - levels[i] / off
        assert 0.003 < contrast < 0.05, contrast  # one of four axes: ensemble contrast of a few per cent at most


def test_cw_odmr_low_power_width_and_power_dependence():
    """Single axis at low beta and low MW power: Lorentzian dip with FWHM -> 1/(pi T2*) (spec section 9.1);
    contrast grows with MW power."""
    t2s = 0.1e-6
    env = single_axis_env(t2_star_s=t2s, optical_beta_full_scale=0.01)
    sim = ReferenceSimulator(n_detuning=64)
    f0 = f_minus_single_axis(env)
    duration = 20_000

    def contrast(f: float, amp: float) -> float:
        on = cw_level(sim, cw_odmr_point(f, duration_ns=duration, amplitude=amp), env, late_ns=2000)
        return 1 - on / off

    off = cw_level(sim, dataclasses.replace(cw_odmr_point(f0, duration_ns=duration), microwave=()), env, 2000)
    freqs = f0 + 1e6 * np.arange(-6.0, 6.01, 0.5)
    cs = np.array([contrast(f, 0.01) for f in freqs])

    def lorentz(f: np.ndarray, a: float, x0: float, hwhm: float) -> np.ndarray:
        return a / (1 + ((f - x0) / hwhm) ** 2)

    (_amp, x0, hwhm), _ = scipy.optimize.curve_fit(lorentz, freqs, cs, p0=(cs.max(), f0, 1e6))
    fwhm_spec = 1 / (math.pi * t2s)
    # T2* sets the width; the optical pumping rate beta k_r = 6.6e5 1/s adds ~0.1 MHz of HWHM (~7 %).
    assert 1.0 < 2 * abs(hwhm) / fwhm_spec < 1.15
    assert abs(x0 - f0) < 0.02e6
    resonant = [contrast(f0, amp) for amp in (0.01, 0.03, 0.1, 0.3)]
    assert np.all(np.diff(resonant) > 0)


# --- 2. Pulsed Rabi ---------------------------------------------------------------------------------------------


def transfer(t_s: np.ndarray, f_rabi_hz: float, deltas_hz: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Two-level Rabi transfer averaged over detunings: sum_i w_i f_R^2 / W_i^2 sin^2(pi W_i t), W^2 = f_R^2 +
    delta^2 (population Rabi frequency f_R, spec section 3)."""
    w2 = f_rabi_hz**2 + deltas_hz * deltas_hz
    amp = weights * f_rabi_hz**2 / w2
    return np.array([np.sum(amp * np.sin(math.pi * np.sqrt(w2) * t) ** 2) for t in t_s])


def continuous_transfer(t_s: np.ndarray, f_rabi_hz: float, env: EnvParams) -> np.ndarray:
    """``transfer`` over the continuous T2* distribution of spec section 4, on a 200001-point grid (Lorentzian: in
    the angle delta = Gamma tan(theta); Gaussian: +-10 sigma). Independent of the simulator's quadrature."""
    if env.broadening_shape == "lorentzian":
        gamma = 1 / (2 * math.pi * env.t2_star_s)
        theta = np.linspace(-math.pi / 2, math.pi / 2, 200_001)[1:-1]
        return transfer(t_s, f_rabi_hz, gamma * np.tan(theta), np.full(theta.size, (theta[1] - theta[0]) / math.pi))
    sig = 1 / (math.sqrt(2) * math.pi * env.t2_star_s)
    d = np.linspace(-10 * sig, 10 * sig, 200_001)
    return transfer(t_s, f_rabi_hz, d, np.exp(-0.5 * (d / sig) ** 2) / (sig * math.sqrt(2 * math.pi)) * (d[1] - d[0]))


def fit_contrast(sig: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    """Least squares sig = s0 - kappa p; returns (s0, kappa, max |residual| / kappa)."""
    design = np.stack([np.ones_like(p), -p], axis=1)
    (s0, kappa), *_ = np.linalg.lstsq(design, sig, rcond=None)
    return float(s0), float(kappa), float(np.max(np.abs(sig - design @ np.array([s0, kappa])))) / float(kappa)


@pytest.mark.parametrize(
    ("amp", "t2_star_s", "shape"),
    [(1.0, 1e-6, "lorentzian"), (0.5, 1e-6, "lorentzian"), (0.25, 0.1e-6, "lorentzian"), (0.25, 0.1e-6, "gaussian")],
)
def test_pulsed_rabi_matches_two_level_ensemble(amp, t2_star_s, shape):
    """Readout contrast vs MW duration = S0 - kappa P(t) with P the T2*-damped Rabi transfer at
    f_R = rabi_full_scale * a * |m_perp| (spec sections 3, 4, 9.2). T1, T2 long so only T2* damps.

    Two comparisons: against P on the simulator's own detuning nodes (tests the dynamics, tight), and against P
    over the continuous distribution (adds the M = 64 quadrature error, ~1 % for Lorentzian tails at f_R >> Gamma).
    """
    n_det = 64
    env = single_axis_env(t2_star_s=t2_star_s, broadening_shape=shape, t1_s=1.0, t2_s=1e-3)
    sim = ReferenceSimulator(n_detuning=n_det)
    f_rabi = env.rabi_full_scale_hz * amp * MPERP_100
    durations = np.arange(0, 310, 10)
    f0 = f_minus_single_axis(env)
    # The MW pulse ends right before the readout and the dark time between the laser pulses is fixed (3 us, >= 6.5
    # singlet lifetimes before the MW starts), so the singlet left over from the init pulse neither varies with the
    # duration nor feeds unpolarised population into the ground state during the pulse.
    seqs = [
        SequenceBuilder().laser(2000).wait(3000 - int(t)).mw(int(t), f0, 0.0, amp).laser(400).build() for t in durations
    ]
    sig = np.array([readout_signal(sim, s, env) for s in seqs])
    t_s = durations * 1e-9
    nodes = detuning_samples_hz(env, n_det)
    s0, kappa, err_nodes = fit_contrast(sig, transfer(t_s, f_rabi, nodes, np.full(n_det, 1 / n_det)))
    assert 0.15 < kappa / s0 < 0.4  # single-axis spin contrast of the 300 ns window (~25 %)
    # Residual: off-resonant |0> <-> |+1> drive (168 MHz away), T2 = 1 ms, singlet remnant.
    assert err_nodes < 0.004
    _, _, err_cont = fit_contrast(sig, continuous_transfer(t_s, f_rabi, env))
    assert err_cont < 0.015
    # Not vacuous: a 3 % wrong Rabi frequency fits clearly worse.
    _, _, err_wrong = fit_contrast(sig, transfer(t_s, 1.03 * f_rabi, nodes, np.full(n_det, 1 / n_det)))
    assert err_wrong > 3 * err_nodes


# --- 3. Ramsey --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["lorentzian", "gaussian"])
def test_ramsey_fringes_at_detuning_with_t2star_envelope(shape):
    """Fringes at the drive detuning; envelope exp(-t/T2*) (Lorentzian) or exp(-(t/T2*)^2) (Gaussian), times
    exp(-t/T2) (spec sections 4, 5, 9.3)."""
    t2s, t2 = 1e-6, 10e-6
    env = single_axis_env(t2_star_s=t2s, t2_s=t2, broadening_shape=shape)
    sim = ReferenceSimulator(n_detuning=64)
    detuning = 2e6
    f = f_minus_single_axis(env) - detuning  # Delta_- = f_- - f = +2 MHz
    taus = np.arange(0, 1500, 30)
    sig = np.array([readout_signal(sim, ramsey(int(t), f, 30, **TIMING), env) for t in taus])
    t_s = taus * 1e-9

    if shape == "lorentzian":

        def model(t, a, b, rate, fr, ph):
            return a + b * np.exp(-rate * t) * np.cos(2 * np.pi * fr * t + ph)

        p0 = (sig.mean(), (sig.max() - sig.min()) / 2, 1e6, detuning, 0.0)
    else:

        def model(t, a, b, rate, fr, ph):
            return a + b * np.exp(-((rate * t) ** 2) - t / t2) * np.cos(2 * np.pi * fr * t + ph)

        p0 = (sig.mean(), (sig.max() - sig.min()) / 2, 1e6, detuning, 0.0)
    popt, _ = scipy.optimize.curve_fit(model, t_s, sig, p0=p0, maxfev=20000)
    _, _, rate, fr, _ = popt
    assert abs(fr) == pytest.approx(detuning, rel=0.005)
    if shape == "lorentzian":
        assert abs(rate) == pytest.approx(1 / t2s + 1 / t2, rel=0.1)
    else:
        assert abs(rate) == pytest.approx(1 / t2s, rel=0.1)


# --- 4. Hahn echo -----------------------------------------------------------------------------------------------


def test_hahn_echo_decays_with_t2_and_ignores_static_detuning():
    """Echo amplitude (final pi/2 phase pi minus phase 0) decays as exp(-2 tau / T2) and does not depend on a
    static detuning (spec sections 5, 9.4)."""
    t2 = 4e-6
    env = single_axis_env(t2_star_s=0.5e-6, t2_s=t2)
    sim = ReferenceSimulator(n_detuning=64)
    taus = np.array([500, 1000, 1500, 2000, 3000])

    def echo(f: float) -> np.ndarray:
        amp = []
        for tau in taus:
            s0 = readout_signal(sim, hahn_echo(int(tau), f, 60, final_phase_rad=0.0, **TIMING), env)
            s1 = readout_signal(sim, hahn_echo(int(tau), f, 60, final_phase_rad=math.pi, **TIMING), env)
            amp.append(s0 - s1)  # phase 0: net 2 pi rotation, bright; phase pi: net pi, dark
        return np.array(amp)

    f0 = f_minus_single_axis(env)
    on = echo(f0)
    assert np.all(on > 0)
    slope = np.polyfit(2 * taus * 1e-9, np.log(on), 1)[0]
    assert -slope == pytest.approx(1 / t2, rel=0.05)
    # Static detuning of 1 MHz: a Ramsey signal would swing through a full fringe every 1 us. The echo keeps the
    # same tau dependence; only a tau-independent factor from the slightly off-resonant pulses remains.
    ratio = echo(f0 - 1e6) / on
    assert np.ptp(ratio) < 0.03 * ratio.mean()
    assert 0.9 < ratio.mean() <= 1.0


# --- 5. Detuned control -----------------------------------------------------------------------------------------


def test_far_detuned_mw_equals_mw_off():
    """MW 200+ MHz from every line (|Delta| >> f_R, linewidth) gives the MW-off trace within shot noise
    (spec section 9.5); the same pulse on resonance does not."""
    env = EnvParams(b_field_t=tuple(3e-3 * N1))  # lines at D +- 84 MHz (axis 1) and D +- 28 MHz (axes 2-4)
    sim = ReferenceSimulator(n_detuning=32)
    far = env.d_hz + 300e6
    for seq in (pulsed_rabi(100, far, 1.0, **TIMING), cw_odmr_point(far, duration_ns=3000, amplitude=1.0)):
        off = dataclasses.replace(seq, microwave=())
        r_on, r_off = sim.expected_rate_cps(seq, env), sim.expected_rate_cps(off, env)
        np.testing.assert_allclose(r_on, r_off, rtol=2e-3, atol=1.0)
        # and at the level of counts: totals agree within Poisson error
        t_on = sim.run(seq, env, shots=200, rng=np.random.default_rng(1)).counts.sum()
        t_off = sim.run(off, env, shots=200, rng=np.random.default_rng(2)).counts.sum()
        assert abs(t_on - t_off) < 4 * math.sqrt(t_on + t_off)
    resonant = pulsed_rabi(60, env.d_hz - c.GAMMA_E_HZ_PER_T * 3e-3, 1.0, **TIMING)
    s_res = readout_signal(sim, resonant, env)
    s_off = readout_signal(sim, dataclasses.replace(resonant, microwave=()), env)
    assert 1 - s_res / s_off > 0.03  # a quarter of the NVs flipped: the control is sensitive
