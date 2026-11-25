"""Fast surrogate: contract, validation-suite physics (docs/physics.md section 9) and gradients. CPU, < ~1 min."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch
from scipy.optimize import curve_fit

from glowbox.analysis.protocols import find_dips, late_mean
from glowbox.control.pulse import (
    MagnetSetting,
    PulseSequence,
    SequenceBuilder,
    cw_odmr_point,
    hahn_echo,
    pulsed_rabi,
    stack,
)
from glowbox.sim import constants as c
from glowbox.sim.interface import EnvParams, Simulator
from glowbox.sim.reference import transition_frequencies_hz
from glowbox.sim.surrogate import (
    ModelOptions,
    SurrogateSimulator,
    build_statics,
    expected_rate_cps,
    mw_fill_for_duration,
)
from glowbox.sim.surrogate.model import nv_local_frames

CPU = torch.device("cpu")
FAST = ModelOptions(n_detuning=8)
DOUBLE = ModelOptions(n_detuning=4, dtype=torch.float64)
# B along [001]: all four axes see |B_par| = B/sqrt3 -> two ODMR lines; 3 mT puts them ~97 MHz apart (>> f_R).
B_Z = (0.0, 0.0, 3e-3)
INIT, SETTLE, READ = 2000, 500, 600


def transitions_hz(b_field_t: tuple[float, float, float], axis: int = 0, env: EnvParams | None = None) -> np.ndarray:
    """Independent check: f-, f+ from diagonalising H0 for one axis (own spin matrices, ms order +1, 0, -1)."""
    env = env or EnvParams()
    s2 = math.sqrt(2)
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / s2
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / s2
    sz = np.diag([1.0, 0.0, -1.0])
    fr = nv_local_frames()[axis]
    bl = fr @ np.asarray(b_field_t) * c.GAMMA_E_HZ_PER_T
    h = env.d_hz * sz @ sz + env.e_hz * (sx @ sx - sy @ sy) + bl[0] * sx + bl[1] * sy + bl[2] * sz
    e = np.linalg.eigvalsh(h)
    return np.array([e[1] - e[0], e[2] - e[0]])


def readout_contrast(rate: torch.Tensor, seqs: list[PulseSequence], window_ns: int = 300) -> np.ndarray:
    """Counts in the first ``window_ns`` of the last laser pulse of each sequence (noiseless)."""
    r = rate.detach().cpu().numpy()
    out = []
    for i, s in enumerate(seqs):
        b0 = s.laser[-1].start_ns // s.grid_ns
        out.append(r[i, b0 : b0 + window_ns // s.grid_ns].sum())
    return np.asarray(out)


def damped_cos(t: np.ndarray, a: float, b: float, f: float, tau: float, ph: float) -> np.ndarray:
    return a + b * np.cos(2 * np.pi * f * t + ph) * np.exp(-t / tau)


# --- contract ---------------------------------------------------------------------------------------------------


def test_protocol_and_trace_contract():
    sim = SurrogateSimulator(device="cpu", options=FAST)
    assert isinstance(sim, Simulator)
    env = EnvParams(b_field_t=B_Z)
    seq = pulsed_rabi(60, 2.9e9, init_ns=INIT, settle_ns=SETTLE, readout_ns=READ)
    tr = sim.run(seq, env, shots=5, rng=np.random.default_rng(1))
    tr.check(seq, shots=5)
    assert tr.metadata["source"] == "surrogate"
    tr2 = sim.run(seq, env, shots=5, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(tr.counts, tr2.counts)
    assert np.all(tr.rate_cps[0] == tr.rate_cps[-1])  # no drift -> identical expected rate per shot
    dark = tr.rate_cps[0, INIT // 10 + 30 : INIT // 10 + 45]  # excited tail (~13 ns) gone
    np.testing.assert_allclose(dark, env.dark_cps, atol=1.0)  # laser off, singlet does not emit


def test_drift_is_seeded_and_varies_between_shots():
    sim = SurrogateSimulator(device="cpu", options=ModelOptions(n_detuning=2))
    env = EnvParams(b_field_t=B_Z, laser_drift_rel_sigma=0.05, laser_drift_tau_s=1e-6)
    seq = SequenceBuilder().laser(500).build()
    a = sim.run(seq, env, shots=4, rng=np.random.default_rng(3))
    b = sim.run(seq, env, shots=4, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(a.rate_cps, b.rate_cps)
    assert np.std(a.rate_cps[:, -1]) > 0


def test_stretched_echo_refused():
    with pytest.raises(ValueError):
        build_statics(EnvParams(t2_stretch=2.0), np.zeros((1, 3)), 2, torch.float64, CPU)


def test_generators_are_real_and_trace_preserving():
    st = build_statics(EnvParams(), np.array([[1e-3, 2e-3, 3e-3]]), 4, torch.float64, CPU)
    # Total population is row 0+1+2 (ground) + 9..12; its time derivative must vanish for every generator piece.
    tr_row = torch.zeros(14, dtype=torch.float64)
    tr_row[[0, 1, 2, 9, 10, 11, 12]] = 1.0
    for piece in (st.l0, st.k_pump, st.drive_a, st.drive_b, st.k_m, st.k_p):
        torch.testing.assert_close(tr_row @ piece[..., :, :13], torch.zeros_like(piece[..., 0, :13]), atol=1e-3, rtol=0)


# --- validation suite -------------------------------------------------------------------------------------------


def test_odmr_dips_at_transitions():
    """Two lines for B || [001]; each dip minimum within 0.3 MHz of the diagonalisation."""
    env = EnvParams(b_field_t=B_Z)
    fm, fp = transitions_hz(B_Z, env=env)
    for f0 in (fm, fp):
        freqs = f0 + np.arange(-3e6, 3.01e6, 0.2e6)
        seqs = [cw_odmr_point(f, duration_ns=4000, amplitude=0.05) for f in freqs]
        rate = expected_rate_cps(stack(seqs), env, options=FAST, device=CPU)
        level = rate[:, -50:].mean(1).numpy()
        assert level.min() < 0.99 * level.max()  # a real dip
        # parabola through the three lowest points
        i = int(np.argmin(level))
        p = np.polyfit(freqs[i - 1 : i + 2] - f0, level[i - 1 : i + 2], 2)
        assert abs(-p[1] / (2 * p[0])) < 0.3e6


def test_odmr_eight_lines_for_generic_field():
    env = EnvParams(b_field_t=(2e-3, 5e-3, 9e-3), t2_star_s=2e-6)
    lines = np.sort(np.concatenate([transitions_hz(env.b_field_t, k, env) for k in range(4)]))
    assert np.min(np.diff(lines)) > 10e6
    on = [cw_odmr_point(f, duration_ns=3000, amplitude=0.05) for f in lines]
    off = [cw_odmr_point(f + 5e6, duration_ns=3000, amplitude=0.05) for f in lines]
    r = expected_rate_cps(stack(on + off), env, options=FAST, device=CPU)[:, -30:].mean(1).numpy()
    assert np.all(r[:8] < r[8:])


@pytest.mark.parametrize("amplitude", [1.0, 0.5])
def test_rabi_frequency_scales_with_amplitude(amplitude):
    env = EnvParams(b_field_t=B_Z)
    f_res = transitions_hz(B_Z, env=env)[1]
    durs = np.arange(0, 810, 10)
    seqs = [pulsed_rabi(int(t), f_res, amplitude, init_ns=INIT, settle_ns=SETTLE, readout_ns=READ) for t in durs]
    s = readout_contrast(expected_rate_cps(stack(seqs), env, options=FAST, device=CPU), seqs)
    f_expected = env.rabi_full_scale_hz * amplitude * math.sqrt(2 / 3)  # |m_perp| = sqrt(2/3) for m = [100]
    p0 = (s.mean(), (s[0] - s.min()) / 2, f_expected * 1e-9, 1e3, 0.0)
    popt, _ = curve_fit(damped_cos, durs.astype(float), s, p0=p0, maxfev=20000)
    assert abs(popt[2] * 1e9 / f_expected - 1) < 0.02
    assert (s.max() - s.min()) / s.max() > 0.1  # spin contrast of order 10-30 %


def test_detuned_mw_equals_mw_off():
    """Validation item 5: 200 MHz off resonance is the same trace as no MW (within 1e-3, far below shot noise)."""
    env = EnvParams(b_field_t=B_Z)
    f_res = transitions_hz(B_Z, env=env)[1]
    seqs = [
        pulsed_rabi(200, f_res + 200e6, init_ns=INIT, settle_ns=SETTLE, readout_ns=READ),
        SequenceBuilder().laser(INIT).wait(SETTLE + 200).laser(READ).build(),
        pulsed_rabi(60, f_res, init_ns=INIT, settle_ns=SETTLE, readout_ns=READ),
    ]
    r = expected_rate_cps(stack(seqs), env, options=FAST, device=CPU).numpy()
    np.testing.assert_allclose(r[0], r[1], rtol=1e-3, atol=1.0)
    assert np.max(np.abs(r[2] - r[1]) / (r[1] + 1)) > 0.05  # whereas resonance does respond


def test_ramsey_fringes_at_detuning():
    env = EnvParams(b_field_t=B_Z)
    f_res = transitions_hz(B_Z, env=env)[1]
    det = 5e6
    taus = np.arange(0, 1010, 20)
    seqs = []
    for tau in taus:
        b = SequenceBuilder().laser(INIT).wait(SETTLE).mw(30, f_res + det).wait(int(tau)).mw(30, f_res + det)
        seqs.append(b.laser(READ).build())
    s = readout_contrast(expected_rate_cps(stack(seqs), env, options=ModelOptions(n_detuning=32), device=CPU), seqs)
    fits = []
    for ph in np.linspace(-np.pi, np.pi, 8, endpoint=False):  # the fringe phase depends on the pulse timing
        p0 = (s.mean(), (s.max() - s.min()) / 2, det * 1e-9, 1e3, ph)
        with np.errstate(over="ignore"):  # bad starting phases may wander through tiny tau
            popt, _ = curve_fit(damped_cos, taus.astype(float), s, p0=p0, maxfev=20000)
        fits.append((float(np.sum((damped_cos(taus.astype(float), *popt) - s) ** 2)), popt))
    popt = min(fits, key=lambda x: x[0])[1]
    assert abs(abs(popt[2]) * 1e9 / det - 1) < 0.02
    assert 300 < popt[3] < 3000  # envelope ~ T2* = 1 us (exp(-t/T2*) x exp(-t/T2))


def test_hahn_echo_decay_and_detuning_insensitivity():
    env = EnvParams(b_field_t=B_Z)
    f_res = transitions_hz(B_Z, env=env)[1]

    def echo(tau: int, det: float) -> float:
        seqs = [
            hahn_echo(tau, f_res + det, 60, final_phase_rad=ph, init_ns=INIT, settle_ns=SETTLE, readout_ns=READ)
            for ph in (0.0, math.pi)
        ]
        s = readout_contrast(expected_rate_cps(stack(seqs), env, options=FAST, device=CPU), seqs)
        return float(s[1] - s[0])

    a1, a2 = echo(500, 0.0), echo(2000, 0.0)
    expected = math.exp(-2 * (2000 - 500) * 1e-9 / env.t2_s)
    assert abs(a2 / a1 - expected) < 0.05
    # a static 0.5 MHz detuning (larger than the T2* linewidth) is refocused
    assert abs(echo(2000, 0.5e6) / a2 - 1) < 0.05


# --- frames, precision, paths -----------------------------------------------------------------------------------


def test_frequency_switch_phase_sign():
    """A segment at f2 = f1 + df starting at t_s acts like a segment at f1 with phase + 2 pi df t_s (section 3)."""
    env = EnvParams(b_field_t=B_Z)
    f1 = transitions_hz(B_Z, env=env)[1]
    t_s = INIT + SETTLE + 30 + 500  # start of the second pi/2 pulse
    df = 1.25 / (t_s * 1e-9)  # ~0.41 MHz: 2 pi df t_s = 2 pi + pi/2, so +shift and -shift differ by pi
    shift = 2 * np.pi * df * t_s * 1e-9

    def ram(freq2: float, phase2: float) -> PulseSequence:
        b = SequenceBuilder().laser(INIT).wait(SETTLE).mw(30, f1).wait(500).mw(30, freq2, phase2)
        return b.laser(READ).build()

    # Base phase pi/2: the fringe is ~cos(phase), so +-shift are told apart only off the symmetric point.
    seqs = [ram(f1 + df, np.pi / 2), ram(f1, np.pi / 2 + shift), ram(f1, np.pi / 2 - shift)]
    s = readout_contrast(expected_rate_cps(stack(seqs), env, options=FAST, device=CPU), seqs)
    assert abs(s[0] - s[1]) < 0.2 * abs(s[0] - s[2])


def test_frame_switching_is_exact():
    """A negligible MW blip at a far frequency switches frames twice and must not change anything."""
    env = EnvParams(b_field_t=B_Z)
    f1 = transitions_hz(B_Z, env=env)[1]
    base = SequenceBuilder().laser(INIT).wait(SETTLE).mw(30, f1).wait(300)
    a = base.wait(10).wait(200).mw(30, f1).laser(READ).build()
    base = SequenceBuilder().laser(INIT).wait(SETTLE).mw(30, f1).wait(300)
    b = base.mw(10, 2.5e9, 0.0, 1e-9).wait(200).mw(30, f1).laser(READ).build()
    r = expected_rate_cps(stack([a, b]), env, options=DOUBLE, device=CPU).numpy()
    np.testing.assert_allclose(r[0], r[1], rtol=1e-7, atol=1e-6)


def test_single_matches_double_and_dense_matches_dedup():
    env = EnvParams(b_field_t=(1e-3, 0.0, 2e-3))
    f_res = transitions_hz(env.b_field_t, env=env)[1]
    seqs = [pulsed_rabi(t, f_res, 0.7, 0.3, init_ns=1000, settle_ns=300, readout_ns=500) for t in (0, 40, 90)]
    arr = stack(seqs)
    ref = expected_rate_cps(arr, env, options=ModelOptions(n_detuning=4, dtype=torch.float64), device=CPU)
    single = expected_rate_cps(arr, env, options=ModelOptions(n_detuning=4), device=CPU).double()
    dense = expected_rate_cps(
        arr, env, options=ModelOptions(n_detuning=4, dtype=torch.float64, dedup=False), device=CPU
    )
    torch.testing.assert_close(single, ref, rtol=2e-4, atol=1.0)
    torch.testing.assert_close(dense, ref, rtol=1e-10, atol=1e-6)


def test_taylor_expm_matches_torch():
    env = EnvParams(b_field_t=(1e-3, 0.0, 2e-3))
    seqs = [pulsed_rabi(t, 2.95e9, 0.7, 0.3, init_ns=500, settle_ns=300, readout_ns=300) for t in (0, 40)]
    arr = stack(seqs)
    for dtype, tol in ((torch.float64, 1e-9), (torch.float32, 1e-4)):
        a = expected_rate_cps(arr, env, options=ModelOptions(4, dtype, dedup=False, expm="torch"), device=CPU)
        b = expected_rate_cps(arr, env, options=ModelOptions(4, dtype, dedup=False, expm="taylor"), device=CPU)
        torch.testing.assert_close(b, a, rtol=tol, atol=tol * 1e3)


def test_poisson_counts_follow_rate():
    sim = SurrogateSimulator(device="cpu", options=FAST)
    env = EnvParams(b_field_t=B_Z)
    seq = SequenceBuilder().laser(1000).build()
    tr = sim.run(seq, env, shots=2000, rng=np.random.default_rng(5))
    mean = tr.rate_cps[0] * seq.grid_ns * 1e-9
    assert abs(tr.counts.mean() / mean.mean() - 1) < 0.02


# --- gradients --------------------------------------------------------------------------------------------------


def _small_controls() -> tuple[EnvParams, dict[str, torch.Tensor], float]:
    """Laser 100 ns, dark 30 ns, two MW bins + gap + two MW bins, laser 60 ns: 22 bins."""
    env = EnvParams(b_field_t=B_Z)
    f_res = float(transitions_hz(B_Z, env=env)[1])
    nb = 22
    laser = torch.zeros(nb, dtype=torch.float64)
    laser[:10] = 0.8
    laser[16:] = 1.0
    amp = torch.zeros(nb, dtype=torch.float64)
    amp[[13, 14]] = 0.6
    amp[10:12] = 0.9
    phase = torch.zeros(nb, dtype=torch.float64)
    phase[13:15] = 0.7
    dfreq = torch.zeros(nb, dtype=torch.float64)
    dfreq[10:12] = 1.0  # MHz offsets from f_res
    dfreq[13:15] = -2.0
    return env, {"laser": laser, "amp": amp, "phase": phase, "dfreq_mhz": dfreq}, f_res


def _rate(env: EnvParams, f_res: float, laser, amp, phase, dfreq_mhz, fill=None, opt=DOUBLE) -> torch.Tensor:
    from glowbox.control.pulse import SequenceArrays

    # Every bin carries a frequency (0 would mean "inherit"), so an amplitude perturbation never moves the frame.
    freq = f_res + dfreq_mhz * 1e6
    arr = SequenceArrays(laser, amp, freq, phase, torch.zeros(3, dtype=torch.float64), 10)
    return expected_rate_cps(arr, env, options=opt, mw_fill=fill, device=CPU) / 1e6


def test_gradients_match_finite_differences():
    env, x, f_res = _small_controls()
    leaves = {k: v.clone().requires_grad_(True) for k, v in x.items()}
    for name in leaves:

        def f(v: torch.Tensor, name: str = name) -> torch.Tensor:
            args = {k: (v if k == name else leaves[k].detach()) for k in leaves}
            return _rate(env, f_res, **args)[16:22].sum()

        # MW-off bins have exactly zero gradient for amp/phase/freq; perturb only bins in use there.
        torch.autograd.gradcheck(f, (leaves[name],), eps=1e-6, atol=1e-5, rtol=1e-4, fast_mode=True)


@pytest.mark.parametrize("with_fill", [False, True])
def test_shared_propagator_gradients_equal_dense(with_fill):
    """The dedup-with-derivatives path gives the same per-bin gradients as one propagator per bin."""
    env, x, f_res = _small_controls()
    fill0 = torch.ones(22, dtype=torch.float64)
    fill0[11] = 0.4
    grads = []
    for frac in (1.0, 0.0):  # 1.0: always shared; 0.0: never
        opt = ModelOptions(n_detuning=4, dtype=torch.float64, shared_grad_max_fraction=frac)
        leaves = {k: v.clone().requires_grad_(True) for k, v in x.items()}
        fill = fill0.clone().requires_grad_(True) if with_fill else None
        out = _rate(env, f_res, **leaves, fill=fill, opt=opt)
        (out * torch.linspace(0.5, 1.5, 22, dtype=torch.float64)).sum().backward()
        g = [v.grad for v in leaves.values()] + ([fill.grad] if with_fill else [])
        grads.append((out.detach(), g))
    torch.testing.assert_close(grads[0][0], grads[1][0], rtol=1e-10, atol=1e-10)
    for a, b in zip(grads[0][1], grads[1][1], strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-8)


def test_duration_gradient_via_fill():
    env, x, f_res = _small_controls()
    dur = torch.tensor(23.0, dtype=torch.float64, requires_grad=True)

    def f(d: torch.Tensor) -> torch.Tensor:
        fill = mw_fill_for_duration(100, d, 22, 10)
        amp = torch.where(fill > 0, torch.full_like(fill, 0.9), torch.zeros_like(fill)).detach()
        amp[:10] = 0.0
        return _rate(env, f_res, x["laser"], amp, x["phase"] * 0, x["dfreq_mhz"] * 0, fill=fill)[16:22].sum()

    torch.autograd.gradcheck(f, (dur,), eps=1e-4, atol=1e-5, rtol=1e-3)
    # The fill path at integer durations equals the plain grid path.
    with torch.no_grad():
        amp = torch.zeros(22, dtype=torch.float64)
        amp[10:13] = 0.9
        plain = _rate(env, f_res, x["laser"], amp, x["phase"] * 0, x["dfreq_mhz"] * 0)
        filled = f(torch.tensor(30.0, dtype=torch.float64))
    torch.testing.assert_close(filled, plain[16:22].sum(), rtol=1e-9, atol=1e-9)


def test_float32_gradients_finite():
    env = EnvParams(b_field_t=B_Z)
    seqs = [pulsed_rabi(t, 2.92e9, 0.5, init_ns=500, settle_ns=200, readout_ns=300) for t in (30, 60)]
    arr = stack(seqs).to_torch()
    amp = torch.as_tensor(arr.mw_amplitude).clone().requires_grad_(True)
    arr.mw_amplitude = amp
    rate = expected_rate_cps(arr, env, options=FAST, device=CPU)
    rate.sum().backward()
    assert amp.grad is not None and torch.isfinite(amp.grad).all() and amp.grad.abs().sum() > 0


def test_magnet_groups_in_one_batch():
    env = EnvParams()
    seqs = [
        cw_odmr_point(2.8e9, 1000, magnet=MagnetSetting((0.0, 0.0, 1e-3))),
        cw_odmr_point(2.8e9, 1000, magnet=MagnetSetting((0.0, 0.0, 2e-3))),
    ]
    batch = expected_rate_cps(stack(seqs), env, options=FAST, device=CPU)
    one = expected_rate_cps(seqs[1].to_arrays(), env, options=FAST, device=CPU)
    torch.testing.assert_close(batch[1], one)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu_and_differentiates():
    env = EnvParams(b_field_t=B_Z)
    f_res = transitions_hz(B_Z, env=env)[1]
    seqs = [pulsed_rabi(t, f_res, init_ns=INIT, settle_ns=SETTLE, readout_ns=READ) for t in (0, 30, 60)]
    arr = stack(seqs)
    cpu = expected_rate_cps(arr, env, options=FAST, device=CPU)
    gpu = expected_rate_cps(arr, env, options=FAST, device="cuda").cpu()
    torch.testing.assert_close(gpu, cpu, rtol=2e-4, atol=1.0)
    t = arr.to_torch("cuda")
    amp = torch.as_tensor(t.mw_amplitude).clone().requires_grad_(True)
    t.mw_amplitude = amp
    expected_rate_cps(t, env, options=FAST).sum().backward()
    assert amp.grad is not None and torch.isfinite(amp.grad).all()
    tr = SurrogateSimulator("cuda", FAST).run(seqs[1], env, shots=100, rng=np.random.default_rng(0))
    tr.check(seqs[1], shots=100)


def test_low_power_cw_odmr_one_dip_per_line() -> None:
    """Low MW power and low beta (spec 9.1 regime): the homogeneous HWHM (~0.2 MHz) is below the quantile rule's
    tail node spacing, so equal-weight quantile nodes turn the line into a comb (satellite dips at single tail
    nodes, e.g. Gamma tan(pi (31.5/32 - 1/2)) = 20.3 Gamma = 3.2 MHz for M = 32). The "uniform" rule
    (docs/physics.md section 4) gives exactly one dip. Measured 2026-11-25: quantile 3 dips, uniform 1."""
    env = EnvParams(
        b_field_t=tuple(3e-3 * np.ones(3) / math.sqrt(3)),
        orientation_populations=(1.0, 0.0, 0.0, 0.0),
        optical_beta_full_scale=0.01,
    )
    line = float(transition_frequencies_hz(env.d_hz, 0, np.array(env.b_field_t))[0, 0])
    freqs = line + np.arange(-6e6, 6e6 + 1, 0.1e6)
    seqs = [cw_odmr_point(float(f), duration_ns=20_000, amplitude=0.01) for f in freqs]
    off = dataclasses.replace(seqs[0], microwave=())

    def dips(rule: str) -> list:
        sim = SurrogateSimulator(CPU, ModelOptions(n_detuning=32, dtype=torch.float64, detuning_rule=rule))
        r = [late_mean(x, seqs[0].grid_ns, 2000) for x in sim.expected_rates([*seqs, off], env)]
        return find_dips(freqs, np.array(r[:-1]), r[-1], min_contrast=1e-3, fit_half_width_hz=3e6)

    assert len(dips("quantile")) == 3  # the documented artefact
    (d,) = dips("uniform")
    assert abs(d.grid_frequency_hz - line) <= 0.05e6 + 1
    # T2*-limited FWHM 1/(pi T2*) = 0.32 MHz plus residual MW / pump broadening: measured 0.70 MHz.
    assert 0.5e6 < d.fwhm_hz < 0.9e6
