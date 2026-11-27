"""Decision, yoked, sealed and sham arms and the block-randomised protocol (glowbox.analysis.arms)."""

import numpy as np
import pytest

from glowbox.analysis import arms as am
from glowbox.analysis.prereg import RunKeys, Sealer, block_schedule
from glowbox.control.pulse import SequenceBuilder
from glowbox.sim.interface import EnvParams, FluorescenceTrace


class Recorder:
    """A controller that proposes (last outcome sum) mod 50 and records everything it is shown."""

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []
        self.applied: list[np.ndarray] = []
        self.resets = 0
        self.last = np.zeros(1, dtype=np.int64)

    def reset(self, batch: int) -> None:
        self.resets += 1
        self.last = np.zeros(batch, dtype=np.int64)

    def propose(self, step: int) -> np.ndarray:
        return (self.last + step) % 50

    def observe(self, step: int, actions: np.ndarray, outcomes: np.ndarray) -> None:
        self.seen.append(np.array(outcomes))
        self.applied.append(np.array(actions))
        self.last = np.asarray(outcomes).sum(axis=1).astype(np.int64)


class Device:
    """Outcomes = (action, action + 1000 * variant flag, step) + Poisson noise; records every call."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.calls: list[am.StepContext] = []

    def __call__(self, actions: np.ndarray, ctx: am.StepContext) -> np.ndarray:
        self.calls.append(ctx)
        a = np.asarray(actions, dtype=np.int64)
        shift = 1000 if ctx.variant == "detuned" else 0
        base = np.stack([a, a + shift, np.full_like(a, ctx.step)], axis=1)
        return (base + self.rng.poisson(3.0, size=base.shape)).astype(np.int64)


def _decoy() -> am.Decoy:
    return am.poisson_decoy(np.random.default_rng(99), 5.0)


def test_decision_arm_applies_proposals():
    key = RunKeys.generate().seal
    ctrl, dev = Recorder(), Device()
    ep = am.run_episode(am.ArmSpec("A1", "decision"), 0, ctrl, dev, 6, 2, Sealer(key), np.zeros(2, int))
    np.testing.assert_array_equal(ep.actions, ep.proposals)
    assert ep.plaintext_reads == 6
    np.testing.assert_array_equal(np.stack(ctrl.seen), am.unseal_outcomes(ep, key))


def test_yoked_replay_is_exact_and_controller_still_runs():
    key = RunKeys.generate().seal
    table = np.random.default_rng(1).integers(0, 50, size=(8, 3))
    ctrl, dev = Recorder(), Device()
    ep = am.run_episode(
        am.ArmSpec("A2", "replay"), 1, ctrl, dev, 8, 3, Sealer(key), np.zeros(3, int), table=table, decoy=_decoy()
    )
    np.testing.assert_array_equal(ep.actions, table)
    assert am.action_hash(ep.actions[:, 0]) == am.action_hash(table[:, 0])
    assert len(ctrl.seen) == 8  # the decision process ran on the live outcomes, its output was discarded
    np.testing.assert_array_equal(np.stack(ctrl.seen), am.unseal_outcomes(ep, key))
    assert not np.array_equal(ep.proposals, table)


def test_sealed_arm_never_shows_outcomes():
    key = RunKeys.generate().seal
    table = np.tile(np.arange(10)[:, None], (1, 2))
    ctrl, dev = Recorder(), Device()
    shown: list[np.ndarray] = []
    ep = am.run_episode(
        am.ArmSpec("A3", "replay", observed=False),
        1,
        ctrl,
        dev,
        10,
        2,
        Sealer(key),
        np.zeros(2, int),
        table=table,
        decoy=_decoy(),
        monitors=[lambda ctx, a, y: shown.append(y)],
    )
    real = am.unseal_outcomes(ep, key)
    assert ep.plaintext_reads == 0
    np.testing.assert_array_equal(ep.actions, table)
    assert len(ctrl.seen) == 10 and len(shown) == 10
    # the controller and the monitor saw a decoy of the same shape, not the outcomes (the real step column is t)
    assert all(s.shape == r.shape for s, r in zip(ctrl.seen, real, strict=True))
    assert not any(np.array_equal(s, r) for s, r in zip(ctrl.seen, real, strict=True))
    np.testing.assert_array_equal(real[:, :, 2] - np.arange(10)[:, None] >= 0, True)


def test_arm_argument_checks():
    s = Sealer(RunKeys.generate().seal)
    with pytest.raises(ValueError, match="table"):
        am.run_episode(am.ArmSpec("A2", "replay"), 0, Recorder(), Device(), 2, 1, s, np.zeros(1, int))
    with pytest.raises(ValueError, match="sampler"):
        am.run_episode(am.ArmSpec("A4", "sham"), 0, Recorder(), Device(), 2, 1, s, np.zeros(1, int))
    with pytest.raises(ValueError, match="decoy"):
        am.run_episode(
            am.ArmSpec("A3", "replay", observed=False),
            0,
            Recorder(),
            Device(),
            2,
            1,
            s,
            np.zeros(1, int),
            table=np.zeros((2, 1), int),
        )
    with pytest.raises(ValueError, match="lag"):
        am.ProtocolDesign((am.ArmSpec("A1", "decision"), am.ArmSpec("A2", "replay", lag=0)), 3)
    with pytest.raises(ValueError, match="one decision"):
        am.ProtocolDesign((am.ArmSpec("A1", "decision"), am.ArmSpec("B1", "decision")), 3)


DESIGN = am.ProtocolDesign(
    (
        am.ArmSpec("A1", "decision"),
        am.ArmSpec("A2", "replay", lag=1),
        am.ArmSpec("A3", "replay", observed=False, lag=1),
        am.ArmSpec("A3L", "replay", observed=False, lag=2),
        am.ArmSpec("A4", "sham"),
        am.ArmSpec("A5", "replay", lag=1, variant="detuned"),
    ),
    n_steps=5,
)


def _protocol(batch: int, n_blocks: int = 5):
    key = RunKeys.generate().seal
    keys = [RunKeys.from_seed(i) for i in range(batch)]
    sch = np.stack([block_schedule(k.order, DESIGN.names, n_blocks) for k in keys])
    dev = Device()
    sham = am.empirical_sham(np.random.default_rng(3), np.arange(40, 50))
    res = am.run_protocol(DESIGN, sch if batch > 1 else sch[0], Recorder(), dev, Sealer(key), _decoy(), sham)
    return res, sch, dev, key


@pytest.mark.parametrize("batch", [1, 3])
def test_protocol_yoking_structure(batch):
    res, sch, dev, key = _protocol(batch)
    ep = {(e.arm, e.block): e for e in res.episodes}
    for k in range(5):
        a1 = ep[("A1", k)].actions
        for arm, lag in (("A2", 1), ("A3", 1), ("A5", 1), ("A3L", 2)):
            e = ep[(arm, k)]
            if k >= lag:
                np.testing.assert_array_equal(e.actions, ep[("A1", k - lag)].actions)
                assert not e.fallback
            else:
                assert e.fallback  # warm-up: no source yet
        assert np.all((ep[("A4", k)].actions >= 40) & (ep[("A4", k)].actions < 50))
        assert ep[("A3", k)].plaintext_reads == 0 and ep[("A3L", k)].plaintext_reads == 0
        assert ep[("A2", k)].plaintext_reads == 5
        # time slots follow each element's schedule
        for arm in DESIGN.names:
            want = k * len(DESIGN.names) + np.argmax(sch[:, k] == arm, axis=1)
            np.testing.assert_array_equal(ep[(arm, k)].slot, want)
        assert a1.shape == (5, batch)
    # the detuned arm was measured with its variant, the others nominal
    assert {c.variant for c in dev.calls if c.arm == "A5"} == {"detuned"}
    assert {c.variant for c in dev.calls if c.arm != "A5"} == {"nominal"}
    y5 = am.unseal_outcomes(ep[("A5", 2)], key)
    assert np.all(y5[..., 1] >= 1000)


def test_protocol_runs_in_schedule_order_for_one_device():
    res, sch, dev, _ = _protocol(1, 3)
    order = [c.arm for c in dev.calls if c.step == 0]
    assert order == [str(a) for row in sch[0] for a in row]
    assert [e.arm for e in res.episodes] == order


def test_protocol_streams_episodes():
    key = RunKeys.generate().seal
    sch = block_schedule(RunKeys.from_seed(0).order, DESIGN.names, 4)
    got: list[am.Episode] = []
    sham = am.empirical_sham(np.random.default_rng(3), np.arange(40, 50))
    res = am.run_protocol(DESIGN, sch, Recorder(), Device(), Sealer(key), _decoy(), sham, on_episode=got.append)
    assert not res.episodes and len(got) == 4 * len(DESIGN.names)


class FakeSim:
    """A Simulator stand-in: counts = MW frequency index in every bin."""

    name = "fake"

    def run(self, sequence, env, *, shots=1, rng=None):
        f = sequence.microwave[0].frequency_hz if sequence.microwave else 0.0
        counts = np.full((shots, sequence.n_bins), int((f - 2.8e9) / 1e6) % 7, dtype=np.int64)
        return FluorescenceTrace(counts, sequence.grid_ns, None, {"source": "fake"})


def test_simulator_measure_uses_run_interface():
    from glowbox.analysis.quantum_test import detune_sequence

    def seq_of(a):
        return SequenceBuilder().laser(100).mw(50, 2.8e9 + 1e6 * int(a), 0.5).laser(200).build()

    m = am.SimulatorMeasure(
        FakeSim(),
        EnvParams(),
        seq_of,
        lambda s: [(150, 250), (250, 350)],
        shots=3,
        rng=np.random.default_rng(0),
        variants={"detuned": lambda s: detune_sequence(s, 2e6)},
    )
    ctx = am.StepContext(0, "A1", "nominal", 0, np.zeros(2, int))
    y = m(np.array([1, 2]), ctx)
    assert y.shape == (2, 2) and y[0, 0] == 3 * 10 * 1 and y[1, 0] == 3 * 10 * 2
    yd = m(np.array([1, 2]), am.StepContext(0, "A5", "detuned", 0, np.zeros(2, int)))
    assert yd[0, 0] == 3 * 10 * 3
