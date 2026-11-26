"""Reservoir pieces: random pulse encoder, suffix composition, branched vs full-sequence simulation, ridge. CPU."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from glowbox.analysis.field_fit import transition_lines_hz
from glowbox.learn.reservoir import (
    EncoderSpec,
    RandomPulseEncoder,
    ReadPulse,
    RidgeReadout,
    compose_suffix,
    full_sequence,
    init_prefix,
    laser_bins,
    multiplex_block,
    simulate_suffixes,
)
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions, expected_rate_cps

B = (1.2e-3, 0.7e-3, 2.5e-3)
LINES = transition_lines_hz(np.array(B))


def _inputs(n: int, k: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).choice([-1.0, 1.0], size=(n, k))


def test_encoder_is_deterministic_and_in_range():
    spec = EncoderSpec(n_inputs=3)
    u = _inputs(50, 3)
    p1 = RandomPulseEncoder(spec, LINES, seed=4).params(u)
    p2 = RandomPulseEncoder(spec, LINES, seed=4).params(u)
    p3 = RandomPulseEncoder(spec, LINES, seed=5).params(u)
    np.testing.assert_array_equal(p1.amplitude, p2.amplitude)
    assert not np.allclose(p1.amplitude, p3.amplitude)
    assert p1.amplitude.min() >= spec.amp_lo and p1.amplitude.max() <= spec.amp_hi
    assert np.all(np.abs(p1.phase_rad) <= np.pi)
    assert np.all(p1.duration_ns % spec.grid_ns == 0)
    assert p1.duration_ns.min() >= spec.min_ns and p1.duration_ns.max() <= spec.max_ns
    near = np.min(np.abs(p1.frequency_hz[..., None] - LINES[None, None]), axis=-1)
    assert near.max() <= spec.freq_span_hz + 1.0
    # duration shift: clipped to [grid, slot]
    p4 = RandomPulseEncoder(spec, LINES, seed=4).params(u, duration_shift_ns=-200)
    assert np.all(p4.duration_ns == spec.grid_ns)
    with pytest.raises(ValueError):
        RandomPulseEncoder(spec, LINES, seed=4).params(u, duration_shift_ns=5)


def test_block_layout_matches_parameters():
    spec = EncoderSpec(n_inputs=2, n_segments=3, slot_ns=50, max_ns=50)
    enc = RandomPulseEncoder(spec, LINES, seed=1)
    u = _inputs(4, 2)
    p = enc.params(u)
    blk = enc.block(u, offset_hz=1e6)
    amp, freq = np.asarray(blk.mw_amplitude), np.asarray(blk.mw_frequency_hz)
    assert amp.shape == (4, spec.block_bins)
    for i in range(4):
        for s in range(3):
            sl = amp[i, s * 5 : (s + 1) * 5]
            on = p.duration_ns[i, s] // 10
            assert np.all(sl[:on] > 0) and np.all(sl[on:] == 0)
            np.testing.assert_allclose(freq[i, s * 5], p.frequency_hz[i, s] + 1e6)
    assert np.all(freq[amp == 0] == 0)


def test_compose_suffix_layout():
    rp = ReadPulse(2.9e9, 0.5, 60)
    suf = compose_suffix(2, None, 100, 200, rp, offset_hz=3e6)
    laser, amp = np.asarray(suf.laser), np.asarray(suf.mw_amplitude)
    assert laser.shape == (2, 10 + 6 + 20)
    assert np.all(laser[:, :16] == 0) and np.all(laser[:, 16:] == 1)
    assert np.all(amp[:, 10:16] == 0.5) and np.all(amp[:, :10] == 0)
    np.testing.assert_allclose(np.asarray(suf.mw_frequency_hz)[:, 10], 2.903e9)


def test_branched_suffixes_equal_full_sequences_with_absolute_phases():
    """The phase rule of full_sequence: phi_abs = phi - 2 pi f T_pre makes the full sequence the same experiment."""
    env = EnvParams(b_field_t=B)
    opt = ModelOptions(n_detuning=4, dtype=torch.float64)
    enc = RandomPulseEncoder(EncoderSpec(n_inputs=3, n_segments=4), LINES, seed=2)
    u = _inputs(3, 3, seed=1)
    suf = compose_suffix(3, enc.block(u), 500, 300, ReadPulse(float(LINES[0]), 0.5, 60))
    pre = init_prefix(1500, 1000)
    branched = simulate_suffixes(pre, suf, env, 30, options=opt)
    for i, row in enumerate(suf.unbatch()):
        seq = full_sequence(pre, row)
        full = expected_rate_cps(seq.to_arrays(), env, options=opt).numpy()[-30:] * 1e-8
        np.testing.assert_allclose(full, branched[i], rtol=1e-7, atol=1e-12)
    # without the phase correction (suffix-relative phases used as absolute ones) the results differ
    seq = full_sequence(pre, suf.unbatch()[0])
    naive_mws = []
    sp = np.asarray(suf.unbatch()[0].mw_phase_rad)
    for m in seq.microwave:
        k = (m.start_ns - pre.n_bins * 10) // 10
        naive_mws.append(dataclasses.replace(m, phase_rad=float(sp[k])))
    naive = dataclasses.replace(seq, microwave=tuple(naive_mws))
    wrong = expected_rate_cps(naive.to_arrays(), env, options=opt).numpy()[-30:] * 1e-8
    assert np.max(np.abs(wrong - branched[0])) / np.ptp(branched[:, 0]) > 1e-3


def test_ridge_matches_least_squares_and_keeps_target_shape():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 6)) * 0.01 + 5.0
    w = rng.standard_normal(6)
    y = x @ w + 0.001 * rng.standard_normal(400)
    r = RidgeReadout(lambdas=(1e-10,)).fit(x, y)
    pred = r.predict(x)
    assert pred.shape == (400,)
    a = np.concatenate([x, np.ones((400, 1))], axis=1)
    ref = a @ np.linalg.lstsq(a, y, rcond=None)[0]
    np.testing.assert_allclose(pred, ref, atol=1e-6)
    y2 = np.stack([y, -y], axis=1)
    assert RidgeReadout().fit(x, y2).predict(x).shape == (400, 2)


def test_ridge_gcv_shrinks_on_pure_noise():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 50))
    noise = rng.standard_normal(200)
    signal = x[:, 0] * 3.0
    lam_noise = RidgeReadout().fit(x, noise).lam
    lam_signal = RidgeReadout().fit(x, signal + 0.1 * noise).lam
    assert lam_noise > lam_signal


def test_multiplexed_block_layout_and_readout_bins():
    spec = EncoderSpec(n_inputs=2, n_segments=4, slot_ns=50, max_ns=50)
    enc = RandomPulseEncoder(spec, LINES, seed=1)
    u = _inputs(3, 2)
    blk = multiplex_block(enc.block(u), spec, 2, 100)
    assert blk.n_bins == spec.block_bins + 10
    laser = np.asarray(blk.laser)
    assert np.all(laser[:, 10:20] == 1) and laser.sum() == 3 * 10
    assert np.all(np.asarray(blk.mw_amplitude)[:, 10:20] == 0)
    suf = compose_suffix(3, blk, 0, 100)
    idx = laser_bins(suf)
    np.testing.assert_array_equal(idx, np.concatenate([np.arange(10, 20), np.arange(30, 40)]))
    env = EnvParams(b_field_t=B)
    opt = ModelOptions(n_detuning=4)
    sel = simulate_suffixes(init_prefix(1000, 500), suf, env, idx, options=opt)
    last = simulate_suffixes(init_prefix(1000, 500), suf, env, 10, options=opt)
    np.testing.assert_allclose(sel[:, -10:], last)
    assert sel.shape == (3, 20)
