"""Learned pulse encoder: parametrisation, straight-through durations, differentiable branched simulation vs the full
sequence, gradients vs finite differences, the detuned arm. CPU, float64, few quadrature nodes."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from glowbox.analysis.field_fit import transition_lines_hz
from glowbox.learn.encoder import (
    Layout,
    PulseEncoder,
    build_suffix,
    gaussian_shot_noise,
    preactivations_for,
    prefix_states,
    pulse_areas_pi,
    readout_features,
    suffix_counts,
    suffix_to_numpy,
)
from glowbox.learn.reservoir import full_sequence, init_prefix
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions, expected_rate_cps

B = (1.2e-3, 0.7e-3, 2.5e-3)
LINES = np.sort(transition_lines_hz(np.array(B)).ravel())
ENV = EnvParams(b_field_t=B)
OPT = ModelOptions(n_detuning=4, dtype=torch.float64)
LAYOUT = Layout(n_blocks=2, n_segments=2, slot_ns=60, max_ns=60.0, readout_ns=300, gate_ns=200, gap_ns=50)
BASE = np.array([[LINES[0], LINES[3]], [LINES[5], LINES[7]]])


def _run(enc: PulseEncoder, u: torch.Tensor | None, n: int, offset_hz: float = 0.0) -> torch.Tensor:
    pre = init_prefix(600, 400)
    st0 = prefix_states(pre, ENV, OPT, "cpu")
    p = enc(u, n)
    arr, fill = build_suffix(p, LAYOUT, offset_hz=offset_hz)
    c = suffix_counts(arr, fill, ENV, st0.expand(n, *st0.shape[1:]), OPT)
    return readout_features(c, LAYOUT)


def test_layout_bins_and_validation():
    ly = LAYOUT
    assert ly.period_bins == 12 + 30 + 5 and ly.n_bins == 2 * ly.period_bins
    assert ly.n_features == 2 and ly.readout_start(1) == ly.period_bins + 12
    assert Layout(gate_ns=0, readout_ns=300).n_features == 30
    with pytest.raises(ValueError):
        Layout(slot_ns=55)
    with pytest.raises(ValueError):
        Layout(max_ns=200.0, slot_ns=100)


def test_preactivations_reproduce_parameters():
    z = preactivations_for(LAYOUT, amplitude=0.3, duration_ns=40.0, phase_rad=0.5, freq_offset_hz=1e6)
    enc = PulseEncoder(LAYOUT, BASE, n_inputs=0, init=z).double()
    p = enc(None, 3)
    assert p.amplitude.shape == (3, 2, 2)
    np.testing.assert_allclose(p.amplitude.detach().numpy(), 0.3, rtol=1e-6)
    np.testing.assert_allclose(p.duration_ns.detach().numpy(), 40.0, rtol=1e-6)
    np.testing.assert_allclose(p.phase_rad.detach().numpy(), 0.5, rtol=1e-6)
    np.testing.assert_allclose(
        p.frequency_hz.detach().numpy(), np.broadcast_to(BASE[None] + 1e6, (3, 2, 2)), rtol=0, atol=10.0
    )


def test_straight_through_durations_on_grid():
    z = preactivations_for(LAYOUT, amplitude=0.5, duration_ns=np.array([[24.0, 36.0], [6.0, 60.0]]))
    enc = PulseEncoder(LAYOUT, BASE, n_inputs=0, init=z).double()
    p = enc(None, 1)
    _, fill = build_suffix(p, LAYOUT)
    f = fill.detach().numpy()[0]
    assert set(np.unique(f)) <= {0.0, 1.0}
    pb = LAYOUT.period_bins
    on = [int(f[k * pb + s * 6 : k * pb + (s + 1) * 6].sum()) for k in range(2) for s in range(2)]
    assert on == [2, 4, 1, 6]  # 24 -> 20 ns, 36 -> 40, 6 -> 10, 60 -> 60
    np.testing.assert_allclose(
        pulse_areas_pi(p, ENV, 10)[0, 0],
        [2 * ENV.rabi_full_scale_hz * math.sqrt(2 / 3) * 0.5 * d * 1e-9 for d in (20, 40)],
    )
    # the duration still gets a gradient (through the continuous fill)
    feats = _run(enc, None, 1)
    feats.sum().backward()
    assert enc.bias is not None and enc.bias.grad is not None
    assert torch.all(enc.bias.grad.reshape(2, 2, 4)[..., 3].abs() > 0)


def test_branched_differentiable_suffix_equals_full_sequence():
    torch.manual_seed(0)
    enc = PulseEncoder(LAYOUT, BASE, n_inputs=3, init=preactivations_for(LAYOUT, 0.5, 30.0)).double()
    enc.randomize_(1.5, torch.Generator().manual_seed(1))
    u = torch.randn(3, 3, dtype=torch.float64)
    pre = init_prefix(600, 400)
    st0 = prefix_states(pre, ENV, OPT, "cpu")
    p = enc(u)
    arr, fill = build_suffix(p, LAYOUT)
    c = suffix_counts(arr, fill, ENV, st0.expand(3, *st0.shape[1:]), OPT).detach()
    sfx = suffix_to_numpy(p, LAYOUT)
    for i, row in enumerate(sfx.unbatch()):
        seq = full_sequence(pre, row)
        r = expected_rate_cps(seq.to_arrays(), ENV, options=OPT, device="cpu") * 1e-8
        assert float((r[-LAYOUT.n_bins :] - c[i]).abs().max()) < 1e-8 * float(c[i].max())
    # input dependence of the random encoder: parameters differ between inputs
    assert float(p.amplitude.detach().std(0).mean()) > 0.05


def test_gradients_match_finite_differences():
    """Amplitude, phase and frequency pre-activations (continuous controls): autograd vs central differences."""
    z0 = preactivations_for(LAYOUT, amplitude=0.6, duration_ns=40.0, phase_rad=0.3, freq_offset_hz=0.5e6)
    enc = PulseEncoder(LAYOUT, BASE, n_inputs=0, init=z0).double()
    w = torch.tensor([1.0, -0.7], dtype=torch.float64)
    loss = (_run(enc, None, 1)[0] * w).sum()
    loss.backward()
    assert enc.bias is not None and enc.bias.grad is not None
    g = enc.bias.grad.reshape(2, 2, 4).clone()
    h = 1e-4
    for idx in [(0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1), (0, 0, 2), (1, 1, 2)]:
        flat = int(np.ravel_multi_index(idx, (2, 2, 4)))
        vals = []
        for sgn in (1, -1):
            with torch.no_grad():
                enc.bias[flat] += sgn * h
                vals.append(float((_run(enc, None, 1)[0] * w).sum()))
                enc.bias[flat] -= sgn * h
        fd = (vals[0] - vals[1]) / (2 * h)
        assert abs(float(g[idx]) - fd) <= 1e-5 * max(abs(fd), 1e-6) + 1e-12, (idx, float(g[idx]), fd)


def test_detuned_suffix_is_dark():
    z0 = preactivations_for(LAYOUT, amplitude=1.0, duration_ns=60.0)
    enc = PulseEncoder(LAYOUT, BASE, n_inputs=0, init=z0).double()
    with torch.no_grad():
        on = _run(enc, None, 1)[0]
        det = _run(enc, None, 1, offset_hz=400e6)[0]
        off = _run(PulseEncoder(LAYOUT, BASE, 0, init=preactivations_for(LAYOUT, 0.0, 0.0)).double(), None, 1)[0]
    assert float((on - off).abs().max()) > 1e-3 * float(off.max())  # the resonant block does something
    # off-resonant transfer ~ (f_R / 400 MHz)^2 ~ 4e-4 at full amplitude
    assert float((det - off).abs().max()) < 1e-3 * float(off.max())


def test_gaussian_shot_noise_moments():
    g = torch.Generator().manual_seed(0)
    mu = torch.full((20000, 2), 3.0, dtype=torch.float64)
    x = gaussian_shot_noise(mu, torch.full((20000,), 100.0), g)
    assert abs(float(x.mean()) - 3.0) < 3e-3
    assert abs(float(x.var()) - 0.03) < 2e-3


@pytest.mark.parametrize("gate", [200, 0])
def test_run_length_path_equals_bin_path(gate: int):
    import dataclasses

    from glowbox.learn.encoder import encoded_counts, prefix_states_and_statics

    ly = dataclasses.replace(LAYOUT, gate_ns=gate, window_ns=100 if gate == 0 else 0)
    torch.manual_seed(0)
    enc = PulseEncoder(ly, BASE, n_inputs=3, init=preactivations_for(ly, 0.5, 30.0)).double()
    enc.randomize_(1.5, torch.Generator().manual_seed(2))
    u = torch.randn(3, 3, dtype=torch.float64)
    pre = init_prefix(600, 400)
    st0, st, grp = prefix_states_and_statics(pre, ENV, OPT, "cpu")
    p = enc(u)
    arr, fill = build_suffix(p, ly)
    ref = readout_features(suffix_counts(arr, fill, ENV, st0.expand(3, *st0.shape[1:]), OPT), ly).detach()
    got = encoded_counts(p, ly, ENV, st0.expand(3, *st0.shape[1:]), st).detach()
    assert got.shape == (3, ly.n_features)
    np.testing.assert_allclose(got.numpy(), ref.numpy(), rtol=1e-9)
    assert grp.tolist() == [0]


def test_run_length_duration_gradient_is_exact():
    """Continuous durations (no rounding): autograd vs central differences of the duration pre-activation."""
    from glowbox.learn.encoder import encoded_counts, prefix_states_and_statics

    z0 = preactivations_for(LAYOUT, amplitude=0.7, duration_ns=np.array([[23.0, 41.0], [17.0, 52.0]]))
    enc = PulseEncoder(LAYOUT, BASE, n_inputs=0, init=z0).double()
    pre = init_prefix(600, 400)
    st0, st, _ = prefix_states_and_statics(pre, ENV, OPT, "cpu")
    w = torch.tensor([1.0, -0.4], dtype=torch.float64)

    def loss() -> torch.Tensor:
        return (encoded_counts(enc(None, 1), LAYOUT, ENV, st0, st, quantize=False)[0] * w).sum()

    loss().backward()
    assert enc.bias is not None and enc.bias.grad is not None
    g = enc.bias.grad.reshape(2, 2, 4).clone()
    h = 1e-5
    for idx in [(0, 0, 3), (0, 1, 3), (1, 0, 3), (1, 1, 3), (1, 0, 0), (0, 1, 2)]:
        flat = int(np.ravel_multi_index(idx, (2, 2, 4)))
        vals = []
        for sgn in (1, -1):
            with torch.no_grad():
                enc.bias[flat] += sgn * h
                vals.append(float(loss()))
                enc.bias[flat] -= sgn * h
        fd = (vals[0] - vals[1]) / (2 * h)
        assert abs(float(g[idx]) - fd) <= 1e-5 * max(abs(fd), 1e-8), (idx, float(g[idx]), fd)
