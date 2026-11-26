"""Task 3 pieces: the driven series, cycle-wise simulation, the hidden-state oracle, windows and skill. CPU."""

from __future__ import annotations

import numpy as np
import torch

from glowbox.learn.bit_memory import device_lines_hz
from glowbox.learn.forecast import DrivenSeries, gather_windows, noise_ceiling, skill, windows
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions

ENV = EnvParams(b_field_t=(1.2e-3, 0.7e-3, 2.5e-3))
LINE = float(device_lines_hz(ENV)[0])
OPT64 = ModelOptions(n_detuning=4, dtype=torch.float64)
SHORT = DrivenSeries(n_cycles=6, burn_in=2, init_ns=500)


def test_drives_are_stationary_and_bounded():
    s = DrivenSeries(rho=0.9)
    a = s.drives(4000, np.random.default_rng(0))
    assert a.shape == (4000, s.total_cycles)
    assert a.min() >= s.amp_center - s.amp_scale and a.max() <= s.amp_center + s.amp_scale
    z = s.latent(a)
    assert abs(z[:, 0].std() - 1) < 0.05 and abs(z[:, -1].std() - 1) < 0.05
    assert abs(np.corrcoef(z[:, 10], z[:, 11])[0, 1] - 0.9) < 0.03


def test_arrays_layout_and_readout_index():
    a = SHORT.drives(2, np.random.default_rng(1))
    arr = SHORT.arrays(a, LINE)
    laser, amp = np.asarray(arr.laser), np.asarray(arr.mw_amplitude)
    assert laser.shape == (2, SHORT.n_bins)
    idx = SHORT.readout_index()
    assert np.all(laser[:, idx] == 1)
    assert int(laser.sum(axis=1)[0]) == SHORT.init_ns // 10 + SHORT.total_cycles * SHORT.readout_bins
    k0 = SHORT.init_ns // 10
    np.testing.assert_allclose(amp[:, k0 : k0 + SHORT.mw_ns // 10], a[:, :1].repeat(SHORT.mw_ns // 10, axis=1))
    assert np.all(amp[:, :k0] == 0)


def test_cycle_wise_states_equal_full_simulation():
    a = SHORT.drives(3, np.random.default_rng(2))
    full = SHORT.simulate(a, ENV, LINE, options=OPT64)
    cyc, states = SHORT.states_after_cycles(a, ENV, LINE, options=OPT64)
    np.testing.assert_allclose(cyc, full, rtol=1e-10, atol=1e-14)
    assert states.shape[:2] == (3, SHORT.total_cycles)


def test_oracle_matches_next_reading_when_the_drive_is_frozen():
    s = DrivenSeries(n_cycles=4, burn_in=1, init_ns=500, rho=0.999999)
    a = s.drives(2, np.random.default_rng(3))
    means, states = s.states_after_cycles(a, ENV, LINE, options=OPT64)
    z = s.latent(a)
    k = 2
    pred = s.oracle_next_mean(z[:, k - 1], states[:, k - 1], ENV, LINE, options=OPT64)
    np.testing.assert_allclose(pred, means[:, k].sum(-1), rtol=1e-4)


def test_detuned_series_carries_no_drive_information():
    a = SHORT.drives(4, np.random.default_rng(4))
    m = SHORT.simulate(a, ENV, LINE, offset_hz=400e6, options=ModelOptions(n_detuning=4))
    y = m[:, SHORT.burn_in :].sum(-1)
    # Across sequences (different drives) at each cycle; the laser-only transient along the cycles is the same.
    # What is left is off-resonant transfer, ~ (f_R / Delta)^2 with f_R <= 8 MHz and the nearest line 260 MHz away
    # (~4e-4 of the reading), against ~2e-2 on resonance.
    assert np.max(np.ptp(y, axis=0) / y.mean(axis=0)) < 1e-3
    r = SHORT.simulate(a, ENV, LINE, options=ModelOptions(n_detuning=4))[:, SHORT.burn_in :].sum(-1)
    assert np.max(np.ptp(r, axis=0) / r.mean(axis=0)) > 1e-2


def test_windows_and_skill():
    seq, tg = windows(3, SHORT, 2)
    assert np.all(tg - 2 >= SHORT.burn_in) and tg.max() == SHORT.total_cycles - 1
    cells = np.arange(3 * SHORT.total_cycles, dtype=float).reshape(3, SHORT.total_cycles)
    x = gather_windows(cells, seq, tg, 2)
    np.testing.assert_array_equal(x[:, -1], cells[seq, tg - 1])
    np.testing.assert_array_equal(x[:, 0], cells[seq, tg - 2])
    t = np.random.default_rng(0).standard_normal(1000)
    assert skill(t, t) == 1.0 and abs(skill(np.full(1000, t.mean()), t)) < 1e-12
    assert abs(noise_ceiling(np.full(1000, 1.0), t) - (1 - 1 / t.var())) < 1e-12
