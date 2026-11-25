"""QuTiP reference simulator: contract, solver cross-checks against independent QuTiP mesolve models, and
the rate-model numbers of docs/physics.md section 6.

The physics validation suite (spec section 9) is in test_reference_protocols.py.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import qutip as qt

from glowbox.control.pulse import LaserPulse, MicrowavePulse, PulseSequence, pulsed_rabi, readout
from glowbox.sim import constants as c
from glowbox.sim.interface import EnvParams, Simulator
from glowbox.sim.reference import ReferenceSimulator, detuning_samples_hz
from glowbox.sim.reference.model import frame_phase_vec

N1 = c.NV_AXES[0]
B_ALIGNED_T = 3e-3
TWO_PI_NS = 2 * math.pi * 1e-9  # Hz -> rad/ns


def single_axis_env(**kw: object) -> EnvParams:
    """All NVs on axis 1, 3 mT along it (f+- = D +- 84 MHz), MW along [100] (|m_perp| = sqrt(2/3))."""
    base = EnvParams(b_field_t=tuple(B_ALIGNED_T * N1), orientation_populations=(1.0, 0.0, 0.0, 0.0))
    return dataclasses.replace(base, **kw)


# --- contract ---------------------------------------------------------------------------------------------------


def test_protocol_and_trace_contract():
    sim = ReferenceSimulator(n_detuning=4)
    assert isinstance(sim, Simulator)
    seq = pulsed_rabi(60, 2.80e9, init_ns=500, settle_ns=200, readout_ns=300)
    env = single_axis_env()
    t1 = sim.run(seq, env, shots=5, rng=np.random.default_rng(3))
    t2 = sim.run(seq, env, shots=5, rng=np.random.default_rng(3))
    t1.check(seq, shots=5)
    assert t1.metadata["source"] == "qutip-reference"
    assert t1.metadata["n_detuning"] == 4
    np.testing.assert_array_equal(t1.counts, t2.counts)
    assert t1.rate_cps is not None
    np.testing.assert_array_equal(t1.rate_cps[0], t1.rate_cps[4])  # no drift: identical expected rates
    # dark settle window: no laser, no dark counts -> the excited states (lifetime ~12 ns) empty and the rate
    # falls to (essentially) zero
    r = t1.rate_cps[0, 50:70]
    assert np.all(np.diff(r) < 0) and r[-1] < 1e-5 * r[0]


def test_shot_noise_is_poisson():
    sim = ReferenceSimulator(n_detuning=1)
    env = single_axis_env(background_cps=1e8, dark_cps=1e6)
    seq = readout(200)
    tr = sim.run(seq, env, shots=4000, rng=np.random.default_rng(0))
    assert tr.rate_cps is not None
    expected_total = float(tr.rate_cps[0].sum() * seq.grid_ns * 1e-9)  # ~ 23 counts per shot
    totals = tr.counts.sum(axis=1)
    assert totals.mean() == pytest.approx(expected_total, rel=0.01)  # SE ~ 0.4 %
    assert totals.var() == pytest.approx(expected_total, rel=0.1)  # Poisson: variance = mean; SE ~ 2 %


def test_laser_drift_scales_pump_and_background():
    sim = ReferenceSimulator(n_detuning=1)
    seq = readout(100)
    env = single_axis_env(laser_drift_rel_sigma=0.05, laser_drift_tau_s=1e-6, background_cps=1e6)
    tr = sim.run(seq, env, shots=3, rng=np.random.default_rng(1))
    assert tr.rate_cps is not None
    assert not np.allclose(tr.rate_cps[0], tr.rate_cps[1])  # shots see different drift
    # Reproduce shot 0 from the same OU sample: x0 is the first standard normal drawn.
    x0 = 0.05 * np.random.default_rng(1).standard_normal()
    pf = np.full(seq.n_bins, max(0.0, 1 + x0))
    emis = sim.emission_per_nv(seq, env, power_factor=pf)
    rate0 = env.collection_efficiency * env.n_nv * emis[0, 0] + env.background_cps * pf[0]
    np.testing.assert_allclose(tr.rate_cps[0, 0], rate0, rtol=1e-12)


def test_t2_stretch_unsupported_and_validity_warning():
    sim = ReferenceSimulator(n_detuning=1)
    with pytest.raises(ValueError, match="t2_stretch"):
        sim.run(readout(100), EnvParams(t2_stretch=2.0))
    with pytest.warns(UserWarning, match="B_perp"):
        sim.run(readout(100), EnvParams(b_field_t=(0.0, 0.0, 20e-3)))


def test_detuning_quadrature():
    env = EnvParams(t2_star_s=1e-6)
    d = detuning_samples_hz(env, 64)
    gamma = 1 / (2 * math.pi * 1e-6)
    assert np.allclose(d, -d[::-1])  # symmetric
    assert np.sum(np.abs(d) < gamma) == 32  # half the mass inside the HWHM
    g = detuning_samples_hz(dataclasses.replace(env, broadening_shape="gaussian"), 2000)
    np.testing.assert_allclose(g.std(), 1 / (math.sqrt(2) * math.pi * 1e-6), rtol=2e-3)
    assert detuning_samples_hz(env, 1).tolist() == [0.0]


# --- independent QuTiP mesolve models ---------------------------------------------------------------------------


def _ms_basis_ops(t1_s: float, t2_s: float, beta: float) -> tuple[list[qt.Qobj], list[qt.Qobj]]:
    """Jump operators of spec sections 5-6 written out directly in the ms basis (rates in 1/ns).

    Levels 0,1,2 = ground ms 0,-1,+1; 3,4,5 = excited ms 0,-1,+1; 6 = singlet. Returns (always-on, pump).
    """

    def op(i: int, j: int, rate_per_s: float) -> qt.Qobj:
        return math.sqrt(rate_per_s * 1e-9) * qt.basis(7, i) * qt.basis(7, j).dag()

    kr, k47, k57, k71, k72 = (
        c.K_RADIATIVE_PER_S,
        c.K_ISC_MS0_PER_S,
        c.K_ISC_MS1_PER_S,
        c.K_SINGLET_TO_MS0_PER_S,
        c.K_SINGLET_TO_MS1_PER_S,
    )
    fixed = [op(0, 3, kr), op(1, 4, kr), op(2, 5, kr), op(6, 3, k47), op(6, 4, k57), op(6, 5, k57)]
    fixed += [op(0, 6, k71), op(1, 6, k72), op(2, 6, k72)]
    fixed += [op(i, j, 1 / (3 * t1_s)) for i in range(3) for j in range(3) if i != j]
    sz = qt.Qobj(np.diag([0, -1, 1, 0, 0, 0, 0]).astype(complex))
    fixed.append(math.sqrt(2 / t2_s * 1e-9) * sz)
    pump = [op(3, 0, beta * kr), op(4, 1, beta * kr), op(5, 2, beta * kr)]
    return fixed, pump


def _proj(i: int) -> qt.Qobj:
    return qt.basis(7, i) * qt.basis(7, i).dag()


def test_matches_independent_mesolve_with_frame_switch():
    """Rotating-frame model written independently (ms basis, B along the axis, frame fixed at f1, the second MW
    frequency carried as a phase ramp), integrated by qutip.mesolve. Pins the frame-switch sign (spec section 3).
    """
    env = single_axis_env(t1_s=20e-6, t2_star_s=1e-6, t2_s=2e-6, optical_beta_full_scale=0.5)
    sim = ReferenceSimulator(n_detuning=1, dedupe_axes=False)
    gb = c.GAMMA_E_HZ_PER_T * B_ALIGNED_T
    f_minus = env.d_hz - gb
    f1, f2 = f_minus + 2e6, f_minus - 3e6
    phi1, phi2 = 0.3, 1.1
    seq = PulseSequence(
        duration_ns=760,
        laser=(LaserPulse(0, 300, 1.0), LaserPulse(560, 200, 1.0)),
        microwave=(MicrowavePulse(400, 40, f1, phi1, 1.0), MicrowavePulse(490, 40, f2, phi2, 0.8)),
    )
    models, _, states = sim.states(seq, env)
    assert np.allclose(np.abs(models[0].v), np.eye(3))  # B along the axis: eigenbasis = ms basis

    fixed, pump = _ms_basis_ops(env.t1_s, env.t2_s, env.optical_beta_full_scale)
    h_det = TWO_PI_NS * ((f_minus - f1) * _proj(1) + (env.d_hz + gb - f1) * _proj(2))
    # u = MW component perpendicular to the axis, in the spec's local frame (x = [001] projected).
    z = N1
    x = np.array([0, 0, 1.0]) - N1[2] * N1
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.array(env.mw_field_direction)
    ux, uy = m @ x, m @ y
    # <-1|u.S|0> = (ux + i uy)/sqrt2, <+1|u.S|0> = (ux - i uy)/sqrt2 for S in the (0, -1, +1) ordering
    raise_op = (ux + 1j * uy) / math.sqrt(2) * qt.basis(7, 1) * qt.basis(7, 0).dag()
    raise_op += (ux - 1j * uy) / math.sqrt(2) * qt.basis(7, 2) * qt.basis(7, 0).dag()
    fr_full = env.rabi_full_scale_hz  # f_R = full scale * a * |m_perp|; |m_perp| is inside (ux, uy)
    k = (math.sqrt(2) / 2) * fr_full * TWO_PI_NS * raise_op

    def drive(amp: float, f: float, phi: float) -> list[object]:
        # In the frame at f1 a lab tone cos(2 pi f t + phi) gives exp(-i (2 pi (f - f1) t + phi)) on |+-><0|.
        def coef(t: float) -> complex:
            return amp * np.exp(-1j * (2 * math.pi * (f - f1) * t * 1e-9 + phi))

        def coef_c(t: float) -> complex:
            return np.conj(coef(t))

        return [h_det, [k, coef], [k.dag(), coef_c]]

    rho = qt.Qobj(np.diag([1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0]).astype(complex))
    segments = [(0, 300, True, None), (300, 400, False, None), (400, 440, False, (1.0, f1, phi1))]
    segments += [(440, 490, False, None), (490, 530, False, (0.8, f2, phi2)), (530, 560, False, None)]
    segments += [(560, 760, True, None)]
    opts = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 100000}
    ref_pops = []
    for start, end, laser_on, mw in segments:
        tl = np.arange(start, end + 1, 10.0)
        h = drive(*mw) if mw else h_det
        res = qt.mesolve(h, rho, tl, c_ops=fixed + (pump if laser_on else []), options=opts)
        ref_pops += [np.real(np.diag(s.full())) for s in res.states[1:]]
        rho = res.states[-1]
    ref_pops_a = np.array(ref_pops)
    rhos = np.stack([s.reshape(7, 7, order="F") for s in states[0, 0]])
    pops = np.real(np.einsum("nii->ni", rhos))
    np.testing.assert_allclose(pops, ref_pops_a, atol=2e-8)
    # Full final state, moved from the simulator's final frame (f2) back to f1.
    # The simulator works in the eigenbasis; here V is diagonal but eigh may flip eigenvector signs.
    final = (states[0, 0, -1] * frame_phase_vec((f1 - f2) * 760e-9)).reshape(7, 7, order="F")
    w = np.eye(7, dtype=complex)
    w[:3, :3] = models[0].v
    np.testing.assert_allclose(w @ final @ w.conj().T, rho.full(), atol=2e-8)


@pytest.mark.parametrize("b_dir", [(1.0, 0.0, 0.0), (0.3, 0.8, 0.52)])
def test_rwa_matches_lab_frame_mesolve_with_b_perp(b_dir):
    """Lab-frame (no RWA, no secular approximation) mesolve with B_perp != 0, against the reference's
    eigenbasis + RWA + secular dissipator. Agreement is limited by the RWA (Bloch-Siegert ~ f_R^2 / 4f)."""
    bvec = 3e-3 * np.asarray(b_dir) / np.linalg.norm(b_dir)
    env = single_axis_env(b_field_t=tuple(bvec), t1_s=20e-6, t2_s=2e-6, optical_beta_full_scale=0.5)
    sim = ReferenceSimulator(n_detuning=1, dedupe_axes=False)
    models, _, _ = sim.states(readout(10), env)
    v = models[0].v
    f = models[0].f_minus_hz
    seq = PulseSequence(
        duration_ns=260, laser=(LaserPulse(0, 150, 1.0),), microwave=(MicrowavePulse(160, 60, f, 0.4, 1.0),)
    )
    _, _, states = sim.states(seq, env)
    final = states[0, 0, -1].reshape(7, 7, order="F")

    # Lab frame, ms basis.
    from glowbox.sim.reference.model import local_frame, spin1_ms_basis

    fr = local_frame(N1)
    b_loc = fr @ bvec * c.GAMMA_E_HZ_PER_T
    u_loc = fr @ np.asarray(env.mw_field_direction)
    sx, sy, sz = spin1_ms_basis()
    h0 = env.d_hz * sz @ sz + b_loc[0] * sx + b_loc[1] * sy + b_loc[2] * sz
    us = u_loc[0] * sx + u_loc[1] * sy  # |m_perp| u . S
    e7 = np.zeros((7, 7), complex)
    e7[:3, :3] = h0
    h0q = qt.Qobj(TWO_PI_NS * e7)
    e7 = np.zeros((7, 7), complex)
    e7[:3, :3] = math.sqrt(2) * env.rabi_full_scale_hz * us
    hmw = qt.Qobj(TWO_PI_NS * e7)
    fixed, pump = _ms_basis_ops(env.t1_s, env.t2_s, env.optical_beta_full_scale)
    rho = qt.Qobj(np.diag([1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0]).astype(complex))
    opts = {"atol": 1e-11, "rtol": 1e-9, "nsteps": 10**7, "max_step": 0.02}
    rho = qt.mesolve(h0q, rho, [0.0, 150.0], c_ops=fixed + pump, options=opts).states[-1]
    rho = qt.mesolve(h0q, rho, [150.0, 160.0], c_ops=fixed, options=opts).states[-1]

    def cosine(t: float) -> float:
        return math.cos(2 * math.pi * f * t * 1e-9 + 0.4)

    rho = qt.mesolve([h0q, [hmw, cosine]], rho, [160.0, 220.0], c_ops=fixed, options=opts).states[-1]
    rho = qt.mesolve(h0q, rho, [220.0, 260.0], c_ops=fixed, options=opts).states[-1]
    lab = rho.full()
    w = np.eye(7, dtype=complex)
    w[:3, :3] = v
    lab_eig = w.conj().T @ lab @ w
    # Populations (frame independent) in the eigenbasis, excited and singlet levels.
    np.testing.assert_allclose(np.real(np.diag(final)), np.real(np.diag(lab_eig)), atol=2e-3)
    # The MW actually did something (otherwise the comparison would be vacuous).
    assert abs(np.real(final[1, 1]) - np.real(final[2, 2])) > 0.05


# --- rate-model numbers (spec section 6) -------------------------------------------------------------------------


def _rate_equation_steady_state(beta: float) -> np.ndarray:
    """Classical 7-level rate equations of Tetienne 2012 (spec table), independent of the simulator."""
    k = np.zeros((7, 7))  # k[i, j] = rate i -> j, 1/s
    kr = c.K_RADIATIVE_PER_S
    for g, e in ((0, 3), (1, 4), (2, 5)):
        k[g, e] = beta * kr
        k[e, g] = kr
    k[3, 6] = c.K_ISC_MS0_PER_S
    k[4, 6] = k[5, 6] = c.K_ISC_MS1_PER_S
    k[6, 0] = c.K_SINGLET_TO_MS0_PER_S
    k[6, 1] = k[6, 2] = c.K_SINGLET_TO_MS1_PER_S
    gen = k.T - np.diag(k.sum(1))
    gen[-1] = 1.0
    rhs = np.zeros(7)
    rhs[-1] = 1.0
    return np.linalg.solve(gen, rhs)


def test_cw_laser_steady_state_and_singlet_lifetime():
    """Late-time emission and polarisation under CW laser match the rate equations; the singlet empties with
    lifetime 1/(k71 + 2 k72) ~ 0.41 us once the laser is off."""
    beta = 0.3
    env = EnvParams(optical_beta_full_scale=beta, t1_s=1.0, t2_s=1e-3)  # B = 0; T1 long so it does not matter
    sim = ReferenceSimulator(n_detuning=1)
    seq = PulseSequence(duration_ns=12000, laser=(LaserPulse(0, 10000, 1.0),))
    _, _, states = sim.states(seq, env)
    rho_ss = states[0, 0, 999].reshape(7, 7, order="F")
    expected = _rate_equation_steady_state(beta)
    np.testing.assert_allclose(np.real(np.diag(rho_ss)), expected, atol=1e-6)
    emis = sim.emission_per_nv(seq, env)
    e_ss = c.K_RADIATIVE_PER_S * expected[3:6].sum()
    np.testing.assert_allclose(emis[0, 990:1000], e_ss, rtol=1e-5)
    assert expected[0] / expected[:3].sum() > 0.7  # optical pumping polarises into ms = 0
    # singlet decay after the laser: from t = 10.5 us to 11.5 us (excited states long gone)
    s = np.real(states[0, 0, :, 6 * 8])
    tau = 1.0 / (c.K_SINGLET_TO_MS0_PER_S + 2 * c.K_SINGLET_TO_MS1_PER_S)
    assert tau == pytest.approx(0.41e-6, rel=0.01)
    np.testing.assert_allclose(s[1149] / s[1049], math.exp(-1e-6 / tau), rtol=1e-6)
