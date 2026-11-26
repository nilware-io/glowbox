"""Task 4, compress-and-reconstruct: encoder -> diamond (surrogate) -> decoder, an autoencoder whose bottleneck is the
NV ensemble's readout.

**Data** (:func:`edge_patches`): 4x4 image patches of a soft edge, ``tanh(sharpness (x cos theta + y sin theta - o))``
on pixel centres in [-1, 1]^2, theta uniform on the circle, offset o uniform in [-o_max, o_max]. 16 numbers on a
two-dimensional manifold (a cylinder): compressible to 2 numbers, not to 1 without loss.

**Bottleneck.** The encoder (:class:`~glowbox.learn.encoder.PulseEncoder`, an MLP of the patch) writes ``n_blocks``
MW blocks, each followed by a gated readout: the decoder sees ``n_blocks`` photon counts (one number per readout)
averaged over ``shots`` repetitions with Poisson shot noise. The number of readouts K and the shots are the
bottleneck: K numbers, each with a signal-to-noise ratio set by the spin contrast and the shots.

**Training.** Phase 1 trains encoder and decoder together through the differentiable surrogate (run-length path,
Gaussian shot noise with the Poisson variance, shots log-uniform in a range, MSE). Phase 2 freezes the encoder,
simulates a bank of noiseless readouts for a fixed training set and trains a fresh decoder on it with real Poisson
noise drawn at every step: the same recipe as every baseline (the fixed random encoder of the reservoir, the
detuned arm), so the comparison isolates the encoder.

**References.** :func:`pca_nmse` (linear K-component reconstruction, noiseless) and a noiseless K-number MLP
autoencoder (:func:`train_ideal_bottleneck`): what K perfect real numbers allow, a floor that photon-counting
readouts can only approach.

Metric: normalised MSE, ``mean((x_hat - x)^2) / mean(var(x))`` over pixels (1 = predicting the mean patch).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from glowbox.learn.encoder import (
    BlockParams,
    Layout,
    PulseEncoder,
    encoded_counts,
    gaussian_shot_noise,
    prefix_states_and_statics,
)
from glowbox.learn.reservoir import init_prefix
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions

# --- data ---------------------------------------------------------------------------------------------------------


def edge_patches(
    n: int, rng: np.random.Generator, size: int = 4, sharpness: float = 3.0, offset_max: float = 0.6
) -> tuple[np.ndarray, np.ndarray]:
    """(patches (n, size^2) in (-1, 1), latents (n, 2) = (theta, offset))."""
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    off = rng.uniform(-offset_max, offset_max, n)
    c = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    yy, xx = np.meshgrid(c, c, indexing="ij")
    proj = np.cos(theta)[:, None, None] * xx[None] + np.sin(theta)[:, None, None] * yy[None]
    x = np.tanh(sharpness * (proj - off[:, None, None])).reshape(n, size * size)
    return x.astype(np.float32), np.stack([theta, off], axis=1)


def nmse(x_hat: np.ndarray, x: np.ndarray, var: float | None = None) -> float:
    """Normalised MSE: mean squared error over the mean per-pixel variance of ``x`` (or the given ``var``)."""
    v = float(np.mean(np.var(x, axis=0))) if var is None else var
    return float(np.mean((x_hat - x) ** 2) / v)


def pca_nmse(train: np.ndarray, test: np.ndarray, k: int) -> float:
    """Linear reference: reconstruction of ``test`` from its first ``k`` principal components of ``train``."""
    mu = train.mean(axis=0)
    _, _, vt = np.linalg.svd(train - mu, full_matrices=False)
    w = vt[:k]
    rec = (test - mu) @ w.T @ w + mu
    return nmse(rec, test, float(np.mean(np.var(train, axis=0))))


# --- decoder ------------------------------------------------------------------------------------------------------


class VectorDecoder(nn.Module):
    """MLP from the K readout features (+ log shots) to the patch. Inputs are counts relative to the MW-off
    reference ``ref`` (per feature): ``10 (x / ref - 1)``, spin contrasts of a few % -> O(0.1-1)."""

    def __init__(self, n_features: int, n_out: int, ref: torch.Tensor, hidden: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        self.register_buffer("ref", ref.reshape(1, n_features).float())
        layers: list[nn.Module] = []
        w = n_features + 1
        for h in hidden:
            layers += [nn.Linear(w, h), nn.GELU()]
            w = h
        layers.append(nn.Linear(w, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, counts: torch.Tensor, shots: torch.Tensor) -> torch.Tensor:
        ref = self.get_buffer("ref")
        x = 10.0 * (counts.float() / ref - 1.0)
        c = (torch.log10(shots.float()) - 4.0)[:, None]
        out: torch.Tensor = self.net(torch.cat([x, c], dim=1))
        return out


def log_uniform_shots(n: int, lo: float, hi: float, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    u = torch.rand(n, generator=gen, device=device)
    return torch.round(torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))


# --- the system ---------------------------------------------------------------------------------------------------


@dataclass
class TrainLog:
    step: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    val: list[float] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class SystemSpec:
    """Device-side settings of the autoencoder (assumptions of the experiment)."""

    init_ns: int = 1500
    settle_ns: int = 1000
    options: ModelOptions = field(default_factory=lambda: ModelOptions(n_detuning=8))


class DiamondChannel:
    """The NV ensemble as a channel: encoder parameters -> noiseless mean counts per shot of each readout.

    Holds the initialised state and statics of one device (``env``; the known bias field is ``env.b_field_t``).
    ``offset_hz`` shifts every MW frequency (the quantum test's detuned arm). Calls are chunked (``chunk``
    sequences) to bound memory.
    """

    def __init__(
        self,
        env: EnvParams,
        layout: Layout,
        spec: SystemSpec,
        device: torch.device,
        offset_hz: float = 0.0,
        chunk: int = 256,
    ) -> None:
        self.env, self.layout, self.spec, self.device = env, layout, spec, device
        self.offset_hz = offset_hz
        self.chunk = chunk
        pre = init_prefix(spec.init_ns, spec.settle_ns, grid_ns=layout.grid_ns)
        self.prefix = pre
        self.state0, self.statics, _ = prefix_states_and_statics(pre, env, spec.options, device)

    def counts(self, params: BlockParams, duration_shift_ns: float = 0.0) -> torch.Tensor:
        """(N, n_features), differentiable when gradients are enabled (one call, no chunking)."""
        n = params.amplitude.shape[0]
        r0 = self.state0.expand(n, *self.state0.shape[1:])
        return encoded_counts(
            params,
            self.layout,
            self.env,
            r0,
            self.statics,
            offset_hz=self.offset_hz,
            duration_shift_ns=duration_shift_ns,
        )

    @torch.no_grad()
    def counts_for(self, encoder: PulseEncoder, u: torch.Tensor, duration_shift_ns: float = 0.0) -> torch.Tensor:
        """Noiseless counts of inputs ``u`` (N, n_inputs), chunked, no gradient, float64 on the device."""
        out = []
        for i in range(0, u.shape[0], self.chunk):
            out.append(self.counts(encoder(u[i : i + self.chunk]), duration_shift_ns).double())
        return torch.cat(out)

    @torch.no_grad()
    def reference(self) -> torch.Tensor:
        """Counts with every MW amplitude 0 (the MW-off baseline the decoder normalises by), (n_features,)."""
        ly = self.layout
        z = torch.zeros(1, ly.n_blocks, ly.n_segments, device=self.device)
        p = BlockParams(z, z, z.double() + self.env.d_hz, z)
        return self.counts(p)[0].double()


def train_end_to_end(
    encoder: PulseEncoder,
    decoder: VectorDecoder,
    channel: DiamondChannel,
    sample: Callable[[int], torch.Tensor],
    *,
    steps: int,
    batch: int,
    lr_encoder: float,
    lr_decoder: float,
    shots_range: tuple[float, float],
    gen: torch.Generator,
    val_u: torch.Tensor | None = None,
    val_every: int = 100,
    log: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainLog:
    """Phase 1: Adam on encoder and decoder through the surrogate, MSE on the reconstruction, Gaussian shot noise.

    ``sample(batch)`` returns input patches (batch, n_inputs) on the device (targets = inputs). Validation: MSE on
    ``val_u`` at fixed noise (Gaussian, shots = geometric mean of the range), every ``val_every`` steps.
    """
    trainable = [*encoder.parameters(), *decoder.parameters()]
    params = [
        {"params": list(encoder.parameters()), "lr": lr_encoder},
        {"params": list(decoder.parameters()), "lr": lr_decoder},
    ]
    opt = torch.optim.Adam(params)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps))))
    )
    hist = TrainLog()
    t0 = time.perf_counter()
    lo, hi = shots_range
    for step in range(1, steps + 1):
        encoder.train()
        decoder.train()
        u = sample(batch)
        shots = log_uniform_shots(batch, lo, hi, gen, channel.device)
        mu = channel.counts(encoder(u))
        x = gaussian_shot_noise(mu, shots, gen)
        loss = ((decoder(x, shots) - u) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        if val_u is not None and (step % val_every == 0 or step == steps):
            with torch.no_grad():
                g2 = torch.Generator(device=channel.device)
                g2.manual_seed(1234)
                s = torch.full((val_u.shape[0],), math.sqrt(lo * hi), device=channel.device)
                mu_v = channel.counts_for(encoder, val_u).float()
                v = float(((decoder(gaussian_shot_noise(mu_v, s, g2), s) - val_u) ** 2).mean())
            hist.step.append(step)
            hist.loss.append(float(loss.detach()))
            hist.val.append(v)
            if log is not None:
                log(step, {"loss": float(loss.detach()), "val_mse": v})
    hist.seconds = time.perf_counter() - t0
    return hist


def train_bank_decoder(
    decoder: VectorDecoder,
    bank_mu: torch.Tensor,
    bank_u: torch.Tensor,
    *,
    steps: int,
    batch: int,
    lr: float,
    shots_range: tuple[float, float],
    gen: torch.Generator,
    val_mu: torch.Tensor | None = None,
    val_u: torch.Tensor | None = None,
    val_every: int = 250,
) -> TrainLog:
    """Phase 2 / baselines: decoder only, on a bank of noiseless counts with fresh Poisson noise every step (AdamW,
    cosine decay); keeps the weights with the best validation MSE (Poisson noise, fixed draw)."""
    dev = bank_mu.device
    opt = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps))))
    )
    lo, hi = shots_range
    hist = TrainLog()
    best, best_state = math.inf, {k: v.detach().clone() for k, v in decoder.state_dict().items()}
    vfix: tuple[torch.Tensor, torch.Tensor] | None = None
    if val_mu is not None and val_u is not None:
        g2 = torch.Generator(device=dev)
        g2.manual_seed(4321)
        s = log_uniform_shots(val_mu.shape[0], lo, hi, g2, dev)
        vfix = (torch.poisson(val_mu * s[:, None], generator=g2) / s[:, None], s)
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        decoder.train()
        idx = torch.randint(0, bank_mu.shape[0], (batch,), generator=gen, device=dev)
        shots = log_uniform_shots(batch, lo, hi, gen, dev)
        x = torch.poisson(bank_mu[idx] * shots[:, None], generator=gen) / shots[:, None]
        loss = ((decoder(x, shots) - bank_u[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if vfix is not None and val_u is not None and (step % val_every == 0 or step == steps):
            decoder.eval()
            with torch.no_grad():
                v = float(((decoder(*vfix) - val_u) ** 2).mean())
            hist.step.append(step)
            hist.loss.append(float(loss.detach()))
            hist.val.append(v)
            if v < best:
                best = v
                best_state = {k: t.detach().clone() for k, t in decoder.state_dict().items()}
    decoder.load_state_dict(best_state)
    hist.seconds = time.perf_counter() - t0
    return hist


@torch.no_grad()
def reconstruct(
    decoder: VectorDecoder, mu: torch.Tensor, shots: float, gen: torch.Generator, repeats: int = 1
) -> torch.Tensor:
    """Decoder output for Poisson counts at ``shots`` from noiseless ``mu`` (N, K), ``repeats`` draws stacked."""
    decoder.eval()
    outs = []
    for _ in range(repeats):
        s = torch.full((mu.shape[0],), float(shots), device=mu.device)
        x = torch.poisson(mu * s[:, None], generator=gen) / s[:, None]
        outs.append(decoder(x, s))
    return torch.cat(outs)


def train_ideal_bottleneck(
    train: torch.Tensor,
    val: torch.Tensor,
    k: int,
    *,
    steps: int,
    batch: int,
    lr: float,
    gen: torch.Generator,
    hidden: tuple[int, ...] = (128, 128),
) -> nn.Module:
    """Noiseless K-number MLP autoencoder (tanh-bounded code): the reference for what K perfect numbers allow."""
    n_in = train.shape[1]

    def mlp(a: int, b: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        w = a
        for h in hidden:
            layers += [nn.Linear(w, h), nn.GELU()]
            w = h
        layers.append(nn.Linear(w, b))
        return nn.Sequential(*layers)

    model = nn.Sequential(mlp(n_in, k), nn.Tanh(), mlp(k, n_in)).to(train.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps))))
    )
    best, best_state = math.inf, {kk: v.detach().clone() for kk, v in model.state_dict().items()}
    for step in range(1, steps + 1):
        idx = torch.randint(0, train.shape[0], (batch,), generator=gen, device=train.device)
        loss = ((model(train[idx]) - train[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == steps:
            with torch.no_grad():
                v = float(((model(val) - val) ** 2).mean())
            if v < best:
                best, best_state = v, {kk: t.detach().clone() for kk, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model
