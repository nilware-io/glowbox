import dataclasses

import numpy as np
import pytest

from glowbox.control.pulse import MagnetSetting, pulsed_rabi
from glowbox.sim import constants as c
from glowbox.sim.interface import EnvParams, FluorescenceTrace, Simulator


class BackgroundOnly:
    """Smallest Simulator: background while the laser is on, Poisson noise. Exercises the contract only."""

    name = "background-only"

    def run(self, sequence, env, *, shots=1, rng=None):
        rng = rng or np.random.default_rng()
        a = sequence.to_arrays()
        rate = env.background_cps * np.asarray(a.laser, dtype=np.float64) + env.dark_cps
        rate = np.broadcast_to(rate, (shots, sequence.n_bins)).copy()
        counts = rng.poisson(rate * sequence.grid_ns * 1e-9).astype(np.int64)
        return FluorescenceTrace(counts, sequence.grid_ns, rate, {"source": self.name})


def test_constants():
    assert pytest.approx(28.03e9, rel=1e-3) == c.GAMMA_E_HZ_PER_T
    np.testing.assert_allclose(np.linalg.norm(c.NV_AXES, axis=1), 1.0)
    # the four <111> axes meet at the tetrahedral angle: cos = -1/3
    g = c.NV_AXES @ c.NV_AXES.T
    np.testing.assert_allclose(g[~np.eye(4, dtype=bool)], -1 / 3)
    proj = c.b_projections_hz(np.array([0.0, 0.0, 1e-3]))
    assert proj.shape == (4, 2)
    np.testing.assert_allclose(np.abs(proj[:, 0]), c.GAMMA_E_HZ_PER_T * 1e-3 / np.sqrt(3))
    np.testing.assert_allclose(proj[:, 0] ** 2 + proj[:, 1] ** 2, (c.GAMMA_E_HZ_PER_T * 1e-3) ** 2)


def test_env_defaults_and_round_trip():
    env = EnvParams()
    assert env.d_hz == c.D_GS_HZ
    env2 = EnvParams.from_dict(env.to_dict())
    assert env2 == env
    seq = pulsed_rabi(0, 2.87e9, magnet=MagnetSetting((0.0, 0.0, 1e-3)))
    np.testing.assert_allclose(EnvParams(b_field_t=(1e-3, 0, 0)).total_field_t(seq), [1e-3, 0, 1e-3])


@pytest.mark.parametrize(
    "change",
    [
        {"orientation_populations": (0.5, 0.5, 0.5, 0.0)},
        {"orientation_populations": (1.0, 0.0, 0.0, -0.0001)},
        {"mw_field_direction": (1.0, 1.0, 0.0)},
        {"t2_star_s": 0.0},
        {"t2_s": 1e-7, "t2_star_s": 1e-6},
        {"collection_efficiency": 1.5},
        {"background_cps": -1.0},
        {"broadening_shape": "voigt"},
    ],
)
def test_env_validation(change):
    with pytest.raises(ValueError):
        dataclasses.replace(EnvParams(), **change)


def test_simulator_protocol_and_trace_contract():
    sim = BackgroundOnly()
    assert isinstance(sim, Simulator)
    seq = pulsed_rabi(40, 2.87e9)
    env = EnvParams(background_cps=1e7)
    t1 = sim.run(seq, env, shots=8, rng=np.random.default_rng(1))
    t2 = sim.run(seq, env, shots=8, rng=np.random.default_rng(1))
    t1.check(seq, shots=8)
    np.testing.assert_array_equal(t1.counts, t2.counts)  # same rng state, same trace
    assert t1.window_counts(0, 3000).shape == (8,)
    # dark between the pulses: no counts in the settle window
    assert t1.window_counts(3000, 4000).sum() == 0
    assert t1.times_ns()[-1] == seq.duration_ns - seq.grid_ns


def test_trace_check_rejects_mismatch():
    seq = pulsed_rabi(40, 2.87e9)
    bad = FluorescenceTrace(np.zeros((1, 3), dtype=np.int64), seq.grid_ns, None, {"source": "x"})
    with pytest.raises(ValueError):
        bad.check(seq)
    nosrc = FluorescenceTrace(np.zeros((1, seq.n_bins), dtype=np.int64), seq.grid_ns)
    with pytest.raises(ValueError, match="source"):
        nosrc.check(seq)
