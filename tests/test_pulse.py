import math

import numpy as np
import pytest
import torch

from glowbox.control.pulse import (
    LaserPulse,
    MagnetSetting,
    MicrowavePulse,
    PulseSequence,
    PulseValidationError,
    SequenceBuilder,
    cw_odmr_point,
    hahn_echo,
    initialisation,
    pulsed_rabi,
    ramsey,
    readout,
    stack,
)

F0 = 2.87e9


def test_basic_sequence_arrays():
    seq = PulseSequence(
        duration_ns=100,
        laser=(LaserPulse(0, 30, 0.5),),
        microwave=(MicrowavePulse(40, 20, F0, 0.25, 0.8),),
        magnet=MagnetSetting((0.0, 0.0, 1e-3)),
    )
    a = seq.to_arrays()
    assert a.n_bins == 10
    np.testing.assert_array_equal(a.laser, [0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(a.mw_amplitude, [0, 0, 0, 0, 0.8, 0.8, 0, 0, 0, 0])
    assert a.mw_frequency_hz.dtype == np.float64
    assert a.mw_frequency_hz[4] == F0 and a.mw_frequency_hz[3] == 0
    np.testing.assert_allclose(a.mw_phase_rad[4:6], 0.25)
    np.testing.assert_array_equal(a.magnet_t, [0, 0, 1e-3])
    np.testing.assert_array_equal(seq.times_ns(), np.arange(10) * 10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_ns": 105},  # not on grid
        {"duration_ns": 100, "laser": (LaserPulse(5, 10),)},  # start off grid
        {"duration_ns": 100, "laser": (LaserPulse(0, 0),)},  # zero length
        {"duration_ns": 100, "laser": (LaserPulse(90, 20),)},  # past the end
        {"duration_ns": 100, "laser": (LaserPulse(0, 30), LaserPulse(20, 10))},  # overlap
        {"duration_ns": 100, "laser": (LaserPulse(0, 30, 1.5),)},  # power > 1
        {"duration_ns": 100, "microwave": (MicrowavePulse(0, 10, F0, 0.0, -0.1),)},  # amp < 0
        {"duration_ns": 100, "microwave": (MicrowavePulse(0, 10, -1.0),)},  # bad frequency
        {"duration_ns": 100, "microwave": (MicrowavePulse(0, 10, F0, math.nan),)},  # bad phase
        {"duration_ns": 100, "microwave": (MicrowavePulse(0, 20, F0), MicrowavePulse(10, 20, F0))},
        {"duration_ns": 100, "magnet": MagnetSetting((0.0, math.inf, 0.0))},
    ],
)
def test_validation_rejects(kwargs):
    with pytest.raises(PulseValidationError):
        PulseSequence(**kwargs)


def test_laser_and_mw_may_overlap_and_order_is_normalised():
    a = PulseSequence(100, laser=(LaserPulse(50, 10), LaserPulse(0, 10)), microwave=(MicrowavePulse(0, 100, F0),))
    b = PulseSequence(100, laser=(LaserPulse(0, 10), LaserPulse(50, 10)), microwave=(MicrowavePulse(0, 100, F0),))
    assert a == b


def test_round_trip_arrays_and_dict():
    seq = hahn_echo(500, F0, 60, amplitude=0.7, final_phase_rad=math.pi, magnet=MagnetSetting((1e-3, 0.0, 0.0)))
    back = seq.to_arrays().to_sequence()
    assert back.duration_ns == seq.duration_ns
    assert [(p.start_ns, p.duration_ns, p.power) for p in back.laser] == [
        (p.start_ns, p.duration_ns, p.power) for p in seq.laser
    ]
    assert len(back.microwave) == 3
    for x, y in zip(back.microwave, seq.microwave, strict=True):
        assert (x.start_ns, x.duration_ns, x.frequency_hz) == (y.start_ns, y.duration_ns, y.frequency_hz)
        assert x.amplitude == pytest.approx(y.amplitude)
        assert math.cos(x.phase_rad) == pytest.approx(math.cos(y.phase_rad), abs=1e-6)
    assert back.magnet == seq.magnet
    assert PulseSequence.from_dict(seq.to_dict()) == seq


def test_stack_pads_and_converts_to_torch():
    seqs = [pulsed_rabi(t, F0) for t in (0, 20, 100)]
    batch = stack(seqs)
    assert batch.batch_shape == (3,)
    assert batch.n_bins == max(s.n_bins for s in seqs)
    # shorter sequences are zero-padded at the end
    assert batch.laser[0, -1] == 0 and batch.laser[2, -1] == 1
    tb = batch.to_torch("cpu")
    assert isinstance(tb.laser, torch.Tensor) and tb.mw_frequency_hz.dtype == torch.float64
    singles = tb.unbatch()
    assert singles[1].to_sequence().microwave[0].duration_ns == 20
    with pytest.raises(ValueError, match="grid"):
        stack([pulsed_rabi(0, F0), pulsed_rabi(0, F0, grid_ns=5)])


def test_protocol_shapes():
    r0 = pulsed_rabi(0, F0)
    assert r0.microwave == () and len(r0.laser) == 2
    r = pulsed_rabi(50, F0, init_ns=1000, settle_ns=500, readout_ns=1000)
    assert r.microwave[0].start_ns == 1500 and r.laser[1].start_ns == 1550 and r.duration_ns == 2550
    assert r.label["protocol"] == "rabi"

    cw = cw_odmr_point(2.8e9, duration_ns=1000)
    assert cw.laser[0].duration_ns == cw.microwave[0].duration_ns == 1000

    rm = ramsey(200, F0, 30, second_phase_rad=math.pi / 2)
    m = rm.microwave
    assert m[1].start_ns - m[0].end_ns == 200 and m[1].phase_rad == pytest.approx(math.pi / 2)

    he = hahn_echo(300, F0, 60)
    m = he.microwave
    assert [p.duration_ns for p in m] == [30, 60, 30]
    assert m[1].start_ns - m[0].end_ns == 300 and m[2].start_ns - m[1].end_ns == 300

    assert initialisation().laser[0].duration_ns == 3000
    assert readout(500).duration_ns == 500


def test_builder_magnet_and_grid():
    b = SequenceBuilder(grid_ns=5, magnet=MagnetSetting((0.0, 0.0, 2e-3)))
    seq = b.laser(15).mw(5, F0).build(note="x")
    assert seq.grid_ns == 5 and seq.n_bins == 4 and seq.magnet.field_t[2] == 2e-3 and seq.label["note"] == "x"
