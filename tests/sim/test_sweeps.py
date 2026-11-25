"""Surrogate sweep fast paths: CW steady state and shared-prefix branching equal the time-stepped model. CPU, ~5 s."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from glowbox.control.pulse import MagnetSetting, SequenceBuilder, cw_odmr_point, pulsed_rabi, stack
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions, emission_and_state, expected_rate_cps
from glowbox.sim.surrogate.sweeps import branched_rate_cps, cw_steady_rate_cps

DOUBLE = ModelOptions(n_detuning=6, dtype=torch.float64, detuning_rule="uniform", detuning_spacing=2.0)
# Generic fields (B_perp != 0 on every axis) and strain, so eigenbases differ from the ms basis.
FIELDS = np.array([[1.2e-3, 0.7e-3, 2.5e-3], [0.3e-3, -1.0e-3, 0.2e-3]])
ENV = EnvParams(e_hz=0.2e6)
FREQS = np.array([2.7995e9, 2.80e9, 2.87e9, 2.942e9])


def test_cw_steady_state_is_the_long_time_limit():
    ss = cw_steady_rate_cps(ENV, FIELDS, FREQS, amplitude=0.1, options=DOUBLE, device="cpu").numpy()
    seqs = [cw_odmr_point(f, 20_000, 0.1, magnet=MagnetSetting(tuple(b))) for b in FIELDS for f in FREQS]
    r = expected_rate_cps(stack(seqs), ENV, options=DOUBLE, device="cpu").numpy()
    late = r[:, -100:].mean(axis=1).reshape(ss.shape)
    np.testing.assert_allclose(ss, late, rtol=1e-7)
    # and the dips are there: the on-line point is darker than the off-resonant one
    assert ss[0, 0] < 0.99 * ss[0, 2]


def test_cw_steady_state_shapes_and_per_field_freqs():
    f2 = np.stack([FREQS, FREQS + 1e6])
    ss = cw_steady_rate_cps(ENV, FIELDS, f2, amplitude=0.1, options=DOUBLE, device="cpu")
    assert ss.shape == (2, 4) and ss.dtype == torch.float64
    with pytest.raises(ValueError):
        cw_steady_rate_cps(ENV, FIELDS, np.stack([FREQS] * 3), amplitude=0.1, options=DOUBLE, device="cpu")


def test_branched_sweep_equals_full_sequences():
    full, pre, suf, branch = [], [], [], []
    for i, b in enumerate(FIELDS):
        m = MagnetSetting(tuple(b))
        pre.append(SequenceBuilder(magnet=m).laser(1500).wait(1000).build())
        for f in FREQS:
            full.append(pulsed_rabi(310, f, amplitude=0.2, init_ns=1500, settle_ns=1000, readout_ns=600, magnet=m))
            suf.append(SequenceBuilder(magnet=m).mw(310, f, 0.0, 0.2).laser(600).build())
            branch.append(i)
    rf = expected_rate_cps(stack(full), ENV, options=DOUBLE, device="cpu").numpy()[:, 250:]
    rb = branched_rate_cps(stack(pre), stack(suf), np.array(branch), ENV, options=DOUBLE, device="cpu").numpy()
    np.testing.assert_allclose(rb, rf, rtol=1e-10)


def test_branching_guards():
    m = MagnetSetting(tuple(FIELDS[0]))
    with_mw = stack([SequenceBuilder(magnet=m).laser(100).mw(50, 2.87e9).build()])
    suffix = stack([SequenceBuilder(magnet=m).laser(100).build()])
    with pytest.raises(ValueError, match="MW-free"):
        branched_rate_cps(with_mw, suffix, np.array([0]), ENV, options=DOUBLE, device="cpu")
    prefix = stack([SequenceBuilder(magnet=MagnetSetting((0.0, 0.0, 1e-3))).laser(100).build()])
    with pytest.raises(ValueError, match="static field"):
        branched_rate_cps(prefix, suffix, np.array([0]), ENV, options=DOUBLE, device="cpu")


def test_initial_state_shape_checked_and_default_is_thermal():
    seq = stack([SequenceBuilder().laser(200).build()])
    rate, state = emission_and_state(seq, ENV, options=DOUBLE, device="cpu")
    assert state.shape == (1, 4, 6, 14)
    with pytest.raises(ValueError, match="initial_state"):
        emission_and_state(seq, ENV, options=DOUBLE, device="cpu", initial_state=state[:, :2])
    thermal = torch.zeros_like(state)
    thermal[..., :3] = 1.0 / 3.0
    again, _ = emission_and_state(seq, ENV, options=DOUBLE, device="cpu", initial_state=thermal)
    torch.testing.assert_close(again, rate)
