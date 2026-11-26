"""Task 4 pieces (data, metrics, channel, training loops) and the Task 1 learned sensor (protocol init equals the
fixed pulsed protocol; banks). CPU, few quadrature nodes, tiny sizes."""

from __future__ import annotations

import numpy as np
import torch

from glowbox.analysis.field_fit import transition_lines_hz
from glowbox.learn.autoencoder import (
    DiamondChannel,
    SystemSpec,
    VectorDecoder,
    edge_patches,
    nmse,
    pca_nmse,
    reconstruct,
    train_bank_decoder,
    train_end_to_end,
)
from glowbox.learn.encoder import Layout, PulseEncoder, preactivations_for
from glowbox.learn.field_data import BankSpec, FieldPrior, SweepProtocol, simulate_bank
from glowbox.learn.sensor import LearnedProtocol, SensorSpec, pulsed_init, simulate_learned_bank, window_bank
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions

B = (1.2e-3, 0.7e-3, 2.5e-3)
LINES = np.sort(transition_lines_hz(np.array(B)).ravel())
CPU = torch.device("cpu")


def test_edge_patches_and_metrics():
    x, lat = edge_patches(500, np.random.default_rng(0))
    assert x.shape == (500, 16) and lat.shape == (500, 2)
    assert np.abs(x).max() < 1.0
    assert nmse(np.broadcast_to(x.mean(0), x.shape), x) == np.float32(1.0) or abs(nmse(x.mean(0)[None], x) - 1) < 1e-6
    assert nmse(x, x) == 0.0
    # two latents: PCA errors fall with k, 16 components are exact
    e = [pca_nmse(x, x, k) for k in (1, 2, 4, 16)]
    assert e[0] > e[1] > e[2] > e[3] and e[3] < 1e-10


def _channel(offset_hz: float = 0.0) -> tuple[DiamondChannel, Layout, np.ndarray]:
    ly = Layout(n_blocks=2, n_segments=2, slot_ns=60, max_ns=60.0, readout_ns=300, gate_ns=200)
    base = np.array([[LINES[0], LINES[3]], [LINES[5], LINES[7]]])
    spec = SystemSpec(init_ns=600, settle_ns=400, options=ModelOptions(n_detuning=4))
    return DiamondChannel(EnvParams(b_field_t=B), ly, spec, CPU, offset_hz=offset_hz, chunk=7), ly, base


def test_channel_chunking_and_reference():
    ch, ly, base = _channel()
    torch.manual_seed(0)
    enc = PulseEncoder(ly, base, 16, init=preactivations_for(ly, 0.5, 30.0), out_scale=1.0)
    u = torch.as_tensor(edge_patches(20, np.random.default_rng(1))[0])
    a = ch.counts_for(enc, u)
    with torch.no_grad():
        b = ch.counts(enc(u)).double()
    torch.testing.assert_close(a, b, rtol=1e-6, atol=0)
    ref = ch.reference()
    assert ref.shape == (2,)
    # MW makes the gated readout darker than MW off (ms = 0 is the bright state)
    assert float(a.mean()) < float(ref.mean())
    det, _, _ = _channel(400e6)
    d = det.counts_for(enc, u)
    assert float((d - ref).abs().max()) < 1e-3 * float(ref.max())


def test_training_loops_run_and_reduce_error():
    ch, ly, base = _channel()
    torch.manual_seed(0)
    enc = PulseEncoder(ly, base, 16, init=preactivations_for(ly, 0.5, 30.0), out_scale=1.0)
    dec = VectorDecoder(ly.n_features, 16, ch.reference().float(), hidden=(32,))
    x = torch.as_tensor(edge_patches(256, np.random.default_rng(2))[0])
    gen = torch.Generator().manual_seed(0)

    def sample(b: int) -> torch.Tensor:
        return x[torch.randint(0, x.shape[0], (b,), generator=gen)]

    h = train_end_to_end(
        enc,
        dec,
        ch,
        sample,
        steps=4,
        batch=16,
        lr_encoder=1e-2,
        lr_decoder=1e-3,
        shots_range=(1e3, 1e5),
        gen=gen,
        val_u=x[:32],
        val_every=2,
    )
    assert len(h.val) == 2 and all(np.isfinite(h.val))
    mu = ch.counts_for(enc, x).float()
    dec2 = VectorDecoder(ly.n_features, 16, ch.reference().float(), hidden=(32,))
    before = nmse(reconstruct(dec2, mu, 1e5, torch.Generator().manual_seed(1)).numpy(), x.numpy())
    train_bank_decoder(
        dec2,
        mu,
        x,
        steps=300,
        batch=64,
        lr=3e-3,
        shots_range=(1e4, 1e5),
        gen=gen,
        val_mu=mu[:64],
        val_u=x[:64],
        val_every=50,
    )
    after = nmse(reconstruct(dec2, mu, 1e5, torch.Generator().manual_seed(1)).numpy(), x.numpy())
    assert after < before and after < 1.0


def test_learned_protocol_init_equals_fixed_pulsed_bank():
    """Initialised at a pulsed-ODMR pulse that fills both slots (400 ns), the learned protocol is the same sequence
    as milestone 2's pulsed protocol with ``pi_ns = 400``: the run-length bank equals the bin-stepped one, readout
    bins summed into the same windows. (With 310 ns the learned layout keeps 90 ns of dark before the readout.)"""
    import dataclasses

    spec = SensorSpec(n_points=4, f_start_hz=2.80e9, f_stop_hz=2.84e9)
    spec = dataclasses.replace(spec, layout=dataclasses.replace(spec.layout, readout_ns=200, window_ns=100))
    lp = LearnedProtocol(spec, init=pulsed_init(spec, 0.2, 400.0)).double()
    bspec = BankSpec(prior=FieldPrior(), fields_per_env=2, n_detuning=4, detuning_rule="uniform", detuning_spacing=2.0)
    rng = np.random.default_rng(0)
    fields = FieldPrior().sample_fields(4, rng)
    envs = [FieldPrior().sample_env(rng) for _ in range(2)]
    learned = simulate_learned_bank(lp, fields, envs, bspec, CPU, workers=1)
    fixed = SweepProtocol(kind="pulsed", f_start_hz=2.80e9, f_stop_hz=2.84e9, step_hz=1e7, readout_ns=200, pi_ns=400)
    ref = window_bank(simulate_bank(fixed, fields, envs, bspec, CPU, workers=1), 10)
    assert learned.means.shape == ref.means.shape == (4, 2, 4)
    # float32 banks: one propagator per run vs one per bin round differently (~5e-5)
    np.testing.assert_allclose(learned.means, ref.means, rtol=1e-4)
