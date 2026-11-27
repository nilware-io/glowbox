"""Milestone 5 measurement model: action grid, run-length means vs the run() interface, the per-axis emulator."""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from glowbox.control.pulse import MagnetSetting
from glowbox.learn.measurement import AxisTable, MeasurementSpec, axis_projections, measurement_means
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions, nv_local_frames
from glowbox.sim.surrogate.simulator import SurrogateSimulator

SPEC = MeasurementSpec(f_start_hz=2.80e9, f_stop_hz=2.94e9, step_hz=20e6, durations_ns=(50, 100), windows_ns=(300, 300))
ENV = EnvParams()
OPT = ModelOptions(n_detuning=4, dtype=torch.float64)
B = np.array([[1.2e-3, 0.7e-3, 2.5e-3], [0.3e-3, -1.1e-3, 0.4e-3]])


def test_action_grid_and_sequence():
    assert SPEC.n_freq == 7 and SPEC.n_actions == 14
    f, d = SPEC.action_grid()
    fi, di = SPEC.split(np.arange(14))
    np.testing.assert_array_equal(d, np.repeat([50, 100], 7))
    np.testing.assert_allclose(f[fi], SPEC.freqs_hz()[fi])
    assert np.all(di == np.repeat([0, 1], 7))
    seq = SPEC.sequence(9)
    assert seq.microwave[0].duration_ns == 100 and seq.microwave[0].frequency_hz == f[9]
    assert seq.duration_ns == int(SPEC.shot_ns(9))
    det = SPEC.detuned(400e6)
    assert det.action_grid()[0][0] == f[0] + 400e6
    np.testing.assert_allclose(SPEC.pulse_area_pi(ENV), [0.49, 0.98], atol=0.01)


def test_run_path_means_equal_the_run_interface():
    """measurement_means (run-length path, prefix once per field) == SurrogateSimulator.run's rate_cps summed over
    the windows of the same PulseSequence (bin path), to round-off."""
    m = measurement_means(SPEC, ENV, B, OPT, "cpu").numpy()
    sim = SurrogateSimulator(device="cpu", options=OPT)
    for i, b in enumerate(B):
        for a in (0, 3, 9, 13):
            seq = SPEC.sequence(a, MagnetSetting((float(b[0]), float(b[1]), float(b[2]))))
            tr = sim.run(seq, ENV, shots=1, rng=np.random.default_rng(0))
            assert tr.rate_cps is not None
            r = tr.rate_cps[0] * SPEC.grid_ns * 1e-9
            w = [r[s].sum() for s in SPEC.window_slices(a)]
            np.testing.assert_allclose(m[i, a], w, rtol=1e-8)
    # EnvParams.b_field_t is added to the rows
    env_b = dataclasses.replace(ENV, b_field_t=(1e-4, 0.0, 0.0))
    m2 = measurement_means(SPEC, env_b, B - np.array([1e-4, 0.0, 0.0]), OPT, "cpu").numpy()
    np.testing.assert_allclose(m2, m, rtol=1e-12)


def test_axis_table_sums_orientations_and_interpolates():
    """At the table's nodes the emulator reproduces single-orientation simulations exactly; for a full field it
    equals the sum of its four orientations (up to the neglected B_perp azimuth and the interpolation)."""
    tab = AxisTable.build(SPEC, ENV, OPT, "cpu", b_max_t=1.0e-3, d_par_t=100e-6, d_perp_t=500e-6)
    assert tab.table.shape == (11, 3, 14, 2)
    x1, _, n1 = nv_local_frames()[0]
    node = torch.as_tensor((0.5e-3 * n1 + 0.5e-3 * x1)[None])
    env1 = dataclasses.replace(ENV, orientation_populations=(1.0, 0.0, 0.0, 0.0))
    t1 = dataclasses.replace(tab, env=env1)
    exact = measurement_means(SPEC, env1, node.numpy(), OPT, "cpu")
    np.testing.assert_allclose(t1.means(node).double().numpy(), exact.numpy(), rtol=2e-6)
    # four-orientation sum for a field whose projections all sit on nodes: B along [100] (|B.n_k| equal)
    b = torch.tensor([[0.3e-3 * np.sqrt(3.0), 0.0, 0.0]], dtype=torch.float64)
    par, _ = axis_projections(b)
    np.testing.assert_allclose(par.numpy(), 0.3e-3, rtol=1e-12)
    full = measurement_means(SPEC, ENV, b.numpy(), OPT, "cpu").numpy()
    emu = tab.means(b).double().numpy()
    spread = full.std(axis=1).mean()
    assert np.abs(emu - full).max() < 0.1 * spread  # B_perp = 0.42 mT is between perp nodes: interpolation
    # means_at == means gathered
    a = torch.tensor([3, 11])
    bb = torch.as_tensor(B)
    np.testing.assert_allclose(tab.means_at(bb, a).numpy(), tab.means(bb)[torch.arange(2), a].numpy(), rtol=1e-6)
