"""Task 2 pieces: patterns, codebooks (resonant vs detuned), the Bayes-optimal readout, metrics, BCE training. CPU."""

from __future__ import annotations

import numpy as np
import torch

from glowbox.learn.bit_memory import (
    MemoryProtocol,
    all_patterns,
    bitwise_map,
    codebook,
    make_encoder,
    recall_metrics,
    template_separation,
)
from glowbox.learn.decoder import ConvDecoder, DecoderConfig, TrainConfig, bce_logits_loss, train_decoder
from glowbox.learn.reservoir import EncoderSpec
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions

ENV = EnvParams(b_field_t=(1.2e-3, 0.7e-3, 2.5e-3))
OPT = ModelOptions(n_detuning=4)
SMALL = EncoderSpec(n_inputs=2, n_segments=4)


def test_all_patterns():
    p = all_patterns(3)
    assert p.shape == (8, 3)
    assert len({tuple(r) for r in p}) == 8
    np.testing.assert_array_equal(p[5], [1, 0, 1])


def test_protocol_sets_encoder_inputs():
    p = MemoryProtocol(n_bits=3, delay_ns=100, encoder=SMALL)
    assert p.encoder.n_inputs == 3
    assert p.shot_ns == 1500 + 1000 + 400 + 100 + 1000


def test_codebook_resonant_differs_detuned_is_flat():
    prot = MemoryProtocol(n_bits=2, delay_ns=100, readout_ns=300, encoder=SMALL, read_line=0)
    enc = make_encoder(prot, ENV, seed=3)
    pats, res = codebook(prot, enc, ENV, options=OPT)
    _, det = codebook(prot, enc, ENV, offset_hz=400e6, options=OPT)
    assert pats.shape == (4, 2) and res.shape == (4, 30)
    spread_res = np.ptp(res.sum(axis=1)) / res.sum(axis=1).mean()
    spread_det = np.ptp(det.sum(axis=1)) / det.sum(axis=1).mean()
    assert spread_res > 1e-2
    assert spread_det < 1e-4
    assert template_separation(det, 1e4) < 0.05 < template_separation(res, 1e4)


def test_bitwise_map_is_near_perfect_on_separable_templates_and_half_on_identical():
    pats = torch.as_tensor(all_patterns(2))
    tm = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]], dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, 4, (500,), generator=g)
    counts = torch.poisson(tm[idx] * 1000, generator=g)
    post = bitwise_map(counts, 1000.0, tm, pats)
    acc = recall_metrics((post > 0.5).numpy().astype(int), pats[idx].numpy())
    assert acc["bit_acc"] > 0.999 and acc["recalled_bits"] > 1.95
    same = bitwise_map(counts, 1000.0, torch.ones(4, 2, dtype=torch.float64), pats)
    np.testing.assert_allclose(same.numpy(), 0.5, atol=1e-6)


def test_recall_metrics_extremes():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 2, size=(20000, 4))
    perfect = recall_metrics(truth, truth)
    assert perfect["bit_acc"] == 1 and perfect["pattern_acc"] == 1 and abs(perfect["recalled_bits"] - 4) < 1e-6
    rand = recall_metrics(rng.integers(0, 2, size=truth.shape), truth)
    assert abs(rand["bit_acc"] - 0.5) < 0.02 and rand["recalled_bits"] < 0.01


def test_bce_training_learns_a_bit():
    torch.manual_seed(0)
    model = ConvDecoder(1, 16, 1, DecoderConfig(channels=(8,), hidden=16, mix_channels=4, groups=4))

    def sample(_step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = torch.randint(0, 2, (64, 1)).float()
        x = torch.randn(64, 1, 16) * 0.3 + y[:, :, None]
        return x, torch.zeros(64), y

    val = [sample(0)]
    hist = train_decoder(model, sample, val, TrainConfig(steps=150, eval_every=50, mse_warmup=0.0), loss=bce_logits_loss)
    assert min(hist.val_nll) < 0.4  # BCE well below log 2 = 0.69
