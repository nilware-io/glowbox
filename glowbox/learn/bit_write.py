"""Task 2 with a learned write (milestone 5): which pulses should write an L-bit pattern so that the readout after
a storage delay recalls it best?

Protocol (one shot, the encoder layout of :mod:`glowbox.learn.encoder` with a storage delay)::

    init laser | dark settle | WRITE: n_segments MW segments (slots of slot_ns) | dark DELAY | readout laser (windows)

As in milestone 3, the 2^L patterns form a codebook: each pattern's noiseless readout (its *template*) is known
(calibration), recalls draw Poisson counts for a number of shots, and the Bayes-optimal readout
(:func:`glowbox.learn.bit_memory.bitwise_map`) decides each bit. What changes is the write: instead of milestone 3's
fixed random encoder, the write block's pulse parameters (amplitude, phase, frequency within a few MHz of a device
line, duration, per segment) are chosen per pattern by

* an RL agent (:class:`~glowbox.learn.rl.CodebookEnv` + PPO): black box, reward = bits recovered per pattern;
* gradient descent through the differentiable surrogate (:func:`train_gradient`, the milestone-4 learner), on the
  same objective: the expected log posterior of the written pattern under the Bayes-optimal readout, with
  reparametrised Gaussian shot noise;
* the fixed random encoder in the same parametrisation (milestone 3/4's reservoir; :meth:`PulseEncoder.randomize_`).

Assumption: the device (its static field and nuisance parameters) is calibrated and fixed per seed; the codebook's
templates are measured once (here: computed by the surrogate at the evaluation quadrature).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from glowbox.learn.autoencoder import DiamondChannel
from glowbox.learn.bit_memory import bitwise_map, recall_metrics
from glowbox.learn.encoder import BlockParams, Layout, PulseEncoder, activate, gaussian_shot_noise
from glowbox.learn.rl import pattern_log_posterior


def action_to_params(z: torch.Tensor, layout: Layout, base_hz: torch.Tensor) -> BlockParams:
    """RL actions (N, n_blocks * n_segments * 4) -> pulse parameters (the encoder's bounded parametrisation)."""
    return activate(z.reshape(-1, layout.n_blocks, layout.n_segments, 4), layout, base_hz)


def codebook_objective(mu: torch.Tensor, shots: float, n_noise: int, generator: torch.Generator | None) -> torch.Tensor:
    """Mean log2 posterior of the written pattern, (R, P, K) templates -> scalar, differentiable in ``mu``
    (Gaussian shot noise with the Poisson variance, reparametrised; the posterior is the Poisson one)."""
    r, p, k = mu.shape
    mu_n = mu[:, :, None].expand(r, p, n_noise, k)
    s = torch.full((r * p * n_noise,), float(shots), device=mu.device)
    y = gaussian_shot_noise(mu_n.reshape(-1, k), s, generator).reshape(r, p * n_noise, k).clamp_min(0.0)
    logp = pattern_log_posterior(y, mu, shots).reshape(r, p, n_noise, p)
    idx = torch.arange(p, device=mu.device)
    lp_true = logp[:, idx, :, idx]  # (P, R, n_noise)
    return lp_true.mean() / math.log(2.0)


@dataclass
class GradientLog:
    step: list[int] = field(default_factory=list)
    bits: list[float] = field(default_factory=list)
    seconds: float = 0.0


def train_gradient(
    encoder: PulseEncoder,
    channel: DiamondChannel,
    patterns: torch.Tensor,
    *,
    steps: int,
    lr: float,
    shots: float,
    n_noise: int,
    generator: torch.Generator,
    log: Callable[[int, float], None] | None = None,
    log_every: int = 50,
) -> GradientLog:
    """Adam (cosine decay) on L + E[log2 P(pattern | readout)] through the surrogate: every step simulates the
    whole codebook (all patterns) with gradients."""
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps))))
    )
    u = 2.0 * patterns.float().to(channel.device) - 1.0
    n_bits = patterns.shape[1]
    hist = GradientLog()
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        mu = channel.counts(encoder(u))  # (P, K)
        obj = codebook_objective(mu[None], shots, n_noise, generator)
        loss = -obj
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(encoder.parameters()), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            bits = n_bits + float(obj.detach())
            hist.step.append(step)
            hist.bits.append(bits)
            if log is not None:
                log(step, bits)
    hist.seconds = time.perf_counter() - t0
    return hist


@torch.no_grad()
def evaluate_codebook(
    templates: torch.Tensor,
    patterns: torch.Tensor,
    shots: float,
    draws_per_pattern: int,
    generator: torch.Generator,
    data_means: torch.Tensor | None = None,
) -> dict[str, float]:
    """Recall with the Bayes-optimal bitwise readout on fresh Poisson noise.

    ``templates`` (P, K) mean counts per shot (the calibration the readout uses); ``data_means`` (P, K) the means
    the recalls are drawn from (default: the templates; a timing probe draws them from a perturbed write)."""
    src = templates if data_means is None else data_means
    p = templates.shape[0]
    idx = torch.arange(p, device=templates.device).repeat_interleave(draws_per_pattern)
    lam = (src[idx].double() * shots).clamp_min(0.0)
    counts = torch.poisson(lam, generator=generator)
    post = bitwise_map(counts, shots, templates.double(), patterns.double().to(templates.device))
    pred = (post > 0.5).long().cpu().numpy()
    true = patterns[idx.cpu()].cpu().numpy()
    out = recall_metrics(pred, true)
    # L + E[log2 P(written pattern | readout)]: the learners' training objective (an information lower bound on
    # the whole pattern; recalled_bits instead treats the bits as separate hard-decision channels)
    lp = pattern_log_posterior((counts / shots)[None], templates[None].double(), shots)[0]
    lp_true = lp[torch.arange(lp.shape[0], device=lp.device), idx]
    out["pattern_info_bits"] = float(patterns.shape[1] + (lp_true / math.log(2.0)).clamp_min(-20.0).mean())
    return out


def template_separation_sigma(templates: torch.Tensor, shots: float) -> float:
    """Median distance (in Poisson standard deviations at ``shots``) from each template to its nearest neighbour."""
    from glowbox.learn.bit_memory import template_separation

    return template_separation(np.asarray(templates.double().cpu().numpy()), shots)
