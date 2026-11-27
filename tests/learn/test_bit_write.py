"""Milestone 5 Task 2 learned write: objective, gradient training step, evaluation."""

from __future__ import annotations

import numpy as np
import torch

from glowbox.analysis.field_fit import transition_lines_hz
from glowbox.learn.autoencoder import DiamondChannel, SystemSpec
from glowbox.learn.bit_memory import all_patterns
from glowbox.learn.bit_write import action_to_params, codebook_objective, evaluate_codebook, train_gradient
from glowbox.learn.encoder import Layout, PulseEncoder
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions

B = (1.2e-3, 0.7e-3, 2.5e-3)
ENV = EnvParams(b_field_t=B)
LAYOUT = Layout(n_segments=2, slot_ns=60, max_ns=60.0, delay_ns=200, readout_ns=300, gate_ns=0, window_ns=100)


def test_objective_is_bits_minus_l():
    tm = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [1.0, 3.0], [3.0, 3.0]]], dtype=torch.float64, requires_grad=True)
    obj = codebook_objective(tm, 1e4, 8, torch.Generator().manual_seed(0))
    assert abs(float(obj.detach())) < 1e-6  # well separated: log2 P(true) = 0
    same = torch.ones(1, 4, 2, dtype=torch.float64, requires_grad=True)
    obj2 = codebook_objective(same, 1e4, 8, torch.Generator().manual_seed(0))
    assert abs(float(obj2.detach()) + 2.0) < 1e-6
    obj2.backward()
    assert same.grad is not None


def test_evaluate_codebook_perfect_and_chance():
    pats = torch.as_tensor(all_patterns(2))
    tm = torch.tensor([[1.0, 1.0], [3.0, 1.0], [1.0, 3.0], [3.0, 3.0]], dtype=torch.float64)
    g = torch.Generator().manual_seed(1)
    assert evaluate_codebook(tm, pats, 1e4, 64, g)["bit_acc"] == 1.0
    met = evaluate_codebook(torch.ones(4, 2, dtype=torch.float64), pats, 1e4, 256, g)
    assert met["recalled_bits"] < 0.05


def test_gradient_training_improves_the_objective():
    lines = np.sort(transition_lines_hz(np.array(B)).ravel())
    base = np.array([[lines[0], lines[3]]])
    ch = DiamondChannel(ENV, LAYOUT, SystemSpec(600, 400, ModelOptions(n_detuning=2)), torch.device("cpu"))
    pats = torch.as_tensor(all_patterns(2))
    torch.manual_seed(0)
    enc = PulseEncoder(LAYOUT, base, 2, (16,), None, 1.0)
    log = train_gradient(
        enc, ch, pats, steps=30, lr=3e-2, shots=1e4, n_noise=4, generator=torch.Generator().manual_seed(2), log_every=10
    )
    assert log.bits[-1] > log.bits[0]
    # RL actions map to the same parametrisation
    z = torch.zeros(3, LAYOUT.n_segments * 4)
    p = action_to_params(z, LAYOUT, torch.as_tensor(base))
    assert p.amplitude.shape == (3, 1, 2) and float(p.amplitude[0, 0, 0]) == 0.5
