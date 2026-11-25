"""Task 1 pieces: field prior and wedge fold, sweep protocols, banks, noise, decoders, physics fit. CPU, ~20 s."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from glowbox.analysis.field_fit import FieldFitter, spectrum_from_counts, transition_lines_hz
from glowbox.control.pulse import MagnetSetting, stack
from glowbox.learn.decoder import ConvDecoder, DecoderConfig, TimeBinLinear, TrainConfig, gaussian_nll, train_decoder
from glowbox.learn.field_data import (
    BankSpec,
    FieldPrior,
    SweepProtocol,
    TargetScaler,
    draw_fields_and_envs,
    features,
    fold_to_wedge,
    poisson_counts,
    simulate_bank,
)
from glowbox.sim import constants as c
from glowbox.sim.reference import transition_frequencies_hz
from glowbox.sim.surrogate import ModelOptions, SurrogateSimulator

SMALL = BankSpec(fields_per_env=2, n_detuning=4)
COARSE = {"step_hz": 4e6}


def test_prior_samples_lie_in_the_wedge_and_shell():
    b = FieldPrior().sample_fields(500, np.random.default_rng(0))
    assert np.all(b[:, 0] >= b[:, 1]) and np.all(b[:, 1] >= b[:, 2]) and np.all(b[:, 2] >= 0)
    mag = np.linalg.norm(b, axis=1)
    assert mag.min() >= 0.5e-3 - 1e-12 and mag.max() <= 3e-3 + 1e-12


def test_spectrum_is_invariant_under_signed_axis_permutations():
    """The symmetry behind the wedge target: every signed permutation of B gives the same set of ODMR lines."""
    b = np.array([1.3e-3, -0.4e-3, 2.1e-3])
    ref = np.sort(transition_lines_hz(b))
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            bp = np.array(signs) * b[list(perm)]
            np.testing.assert_allclose(np.sort(transition_lines_hz(bp)), ref, rtol=0, atol=1.0)
            np.testing.assert_allclose(fold_to_wedge(bp), fold_to_wedge(b))


def test_transition_lines_match_reference_diagonalisation():
    b = np.array([1.2e-3, 0.7e-3, 2.5e-3])
    ours = transition_lines_hz(b, c.D_GS_HZ, 0.2e6)
    ref = transition_frequencies_hz(c.D_GS_HZ, 0.2e6, b).ravel()
    np.testing.assert_allclose(ours, ref, rtol=0, atol=10.0)


def test_protocol_arrays_equal_the_sequences():
    p = SweepProtocol("pulsed", **COARSE)
    fields = np.array([[1e-3, 0.5e-3, 0.1e-3], [2e-3, 1e-3, 0.0]])
    mags = [MagnetSetting(tuple(b)) for b in fields]
    ref_pre = stack([p.prefix(m) for m in mags])
    ref_suf = stack([p.suffix(float(f), m) for m in mags for f in p.mw_freqs_hz()])
    for got, ref in ((p.prefix_arrays(fields), ref_pre), (p.suffix_arrays(fields), ref_suf)):
        for name in ("laser", "mw_amplitude", "mw_frequency_hz", "mw_phase_rad", "magnet_t"):
            np.testing.assert_array_equal(np.asarray(getattr(got, name)), np.asarray(getattr(ref, name)))
    assert p.n_time == 60 and p.shot_ns == 1500 + 1000 + 310 + 600
    assert SweepProtocol("cw").n_time == 1 and SweepProtocol("cw").n_freq == 200


@pytest.mark.parametrize("kind", ["cw", "pulsed"])
def test_bank_matches_the_simulator_on_full_sequences(kind):
    """The bank's fast paths reproduce SurrogateSimulator on the protocol's own time-domain sequences."""
    p = SweepProtocol(kind, f_start_hz=2.80e9, f_stop_hz=2.84e9, step_hz=10e6)
    rng = np.random.default_rng(1)
    fields, envs = draw_fields_and_envs(2, SMALL, rng)
    bank = simulate_bank(p, fields, envs, SMALL, device="cpu", workers=1)
    assert bank.means.shape == (2, p.n_time, p.n_freq)
    sim = SurrogateSimulator("cpu", ModelOptions(n_detuning=4, detuning_rule="uniform", detuning_spacing=2.0))
    env = bank.env(1)  # hidden field in b_field_t, no magnet: the same physics as the bank's magnet trick
    rates = sim.expected_rates(p.sequences(MagnetSetting(), cw_ns=20_000), env)
    np.testing.assert_allclose(p.features_from_rates(rates), bank.means[1], rtol=1e-3)  # float32 stepping


def test_threads_do_not_change_the_bank():
    p = SweepProtocol("cw", **COARSE)
    fields, envs = draw_fields_and_envs(6, SMALL, np.random.default_rng(2))
    a = simulate_bank(p, fields, envs, SMALL, device="cpu", workers=1)
    b = simulate_bank(p, fields, envs, SMALL, device="cpu", workers=3)
    np.testing.assert_array_equal(a.means, b.means)
    np.testing.assert_array_equal(a.nuisance, b.nuisance)


def test_detuned_protocol_shifts_every_frequency():
    p = SweepProtocol("cw", **COARSE)
    np.testing.assert_allclose(p.detuned(4e8).mw_freqs_hz(), p.freqs_hz() + 4e8)


def test_poisson_noise_and_features():
    gen = torch.Generator().manual_seed(0)
    means = torch.full((4000, 1, 2), 0.5)
    shots = torch.full((4000,), 20.0)
    counts = poisson_counts(means, shots, gen)
    assert abs(float(counts.mean()) - 10.0) < 0.2 and abs(float(counts.var()) - 10.0) < 0.6
    x = features(torch.tensor([[[2.0, 4.0, 6.0]]]))
    torch.testing.assert_close(x, torch.tensor([[[-5.0, 0.0, 5.0]]]))


def test_target_scaler_round_trip():
    f = FieldPrior().sample_fields(100, np.random.default_rng(3))
    s = TargetScaler.fit(f)
    y = s.encode(f)
    assert np.allclose(y.mean(0), 0, atol=1e-12) and np.allclose(y.std(0), 1)
    back, sig = s.decode(y, np.ones_like(y))
    np.testing.assert_allclose(back, f)
    assert sig is not None
    np.testing.assert_allclose(sig, np.broadcast_to(s.std_t, sig.shape))


def test_decoders_shapes_and_linearity():
    x = torch.randn(5, 60, 100)
    cvec = torch.randn(5)
    cnn = ConvDecoder(60, 100, 3, DecoderConfig(channels=(16, 32), hidden=32))
    m, lv = cnn(x, cvec)
    assert m.shape == (5, 3) and lv.shape == (5, 3)
    lin = TimeBinLinear(60, 100, 3)
    m1, _ = lin(x, cvec)
    m2, _ = lin(2 * x, cvec)
    m0, _ = lin(0 * x, cvec)
    torch.testing.assert_close(m2 - m0, 2 * (m1 - m0))  # affine in the input


def test_training_learns_a_toy_regression_with_calibrated_noise():
    """y = position of a dip on a 32-point axis, noisy: the CNN must locate it and report a sensible sigma."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    grid = torch.linspace(-1, 1, 32)

    def sample(step: int, n: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = torch.rand(n, 1, generator=gen) * 1.6 - 0.8
        x = -torch.exp(-((grid[None] - y) ** 2) / 0.01)[:, None] + 0.1 * torch.randn(n, 1, 32, generator=gen)
        return x, torch.zeros(n), y

    val = [sample(0, 256)]
    model = ConvDecoder(1, 32, 1, DecoderConfig(mix_channels=4, channels=(16, 16), hidden=32))
    hist = train_decoder(model, sample, val, TrainConfig(steps=300, lr=3e-3, eval_every=50))
    assert hist.val_rmse[-1] < 0.1  # targets span 1.6
    x, cc, y = val[0]
    mean, lv = model(x, cc)
    z = ((mean - y) / torch.exp(0.5 * lv)).detach()
    assert 0.4 < float(z.std()) < 2.5
    assert float(gaussian_nll(mean, lv, y)) < 0


def test_physics_fit_recovers_the_field_and_falls_back_without_dips():
    p = SweepProtocol("cw", step_hz=1e6)
    fields, envs = draw_fields_and_envs(4, BankSpec(fields_per_env=4, n_detuning=8), np.random.default_rng(4))
    bank = simulate_bank(p, fields, envs, BankSpec(fields_per_env=4, n_detuning=8), device="cpu", workers=1)
    fitter = FieldFitter(p.freqs_hz(), fields.mean(0), np.full(3, 1e-3))
    errs = []
    for i in range(4):
        y, s = spectrum_from_counts(bank.means[i] * 1e5, "cw")
        r = fitter.fit(y, s)
        assert r.ok
        errs.append(np.linalg.norm(r.b_t - fields[i]))
    assert np.median(errs) < 20e-6  # tesla; E != 0 in the nuisance prior biases the E = 0 fit slightly
    flat = np.full(p.n_freq, 1e5)
    r = fitter.fit(flat + np.random.default_rng(0).normal(0, 300, p.n_freq), np.full(p.n_freq, 316.0))
    assert not r.ok
    np.testing.assert_allclose(r.b_t, fields.mean(0))
