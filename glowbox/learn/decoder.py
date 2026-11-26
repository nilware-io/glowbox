"""Decoders from fluorescence features to task targets with an uncertainty, and the time-bin-weighting baseline.

Input convention: a batch of feature maps ``x`` of shape (B, T, F), T time bins (1 for a CW spectrum, the readout
bins of a pulsed protocol) per point of a 1D axis of length F (MW frequency for ODMR sweeps; for a single trace
F is the time axis and T = 1), plus a per-sample conditioning scalar ``c`` (B,) (here log10 of the shot number).
Outputs: mean and log-variance per target, (B, D) each, trained with the Gaussian negative log likelihood.

:class:`ConvDecoder` follows the idea of the 1D-CNN decoders of fluorescence traces and ODMR spectra cited in the
project brief (Yao et al., 2026): convolutions along the sweep axis find local features (dips, transients) wherever
they are. The concrete architecture is our choice, not a reproduction of that paper:

1. **time-bin mixing**: a 1x1 convolution T -> ``mix_channels``: learned linear weightings of the time bins, shared
   by every frequency (a learned version of the classic readout window / time-bin weighting);
2. **trunk**: a stem convolution, then ``len(channels)`` stages of two residual blocks (conv k, GroupNorm, GELU) and
   a stride-2 convolution that doubles the channels and halves F;
3. **head**: flatten (positions matter: a dip's place on the axis is the information, so no global pooling), concat
   ``c``, two GELU layers, linear to 2D outputs (mean, log-variance clamped to [-14, 6]).

:class:`TimeBinLinear` is the linear baseline: one learned time-bin weighting (T -> 1, i.e. a weighted spectrum),
then an affine map of that spectrum and ``c`` to the means; the log-variances are an affine function of ``c`` only
(the noise level), so the whole estimator is linear in the input features.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn

LOG_VAR_MIN = -14.0
LOG_VAR_MAX = 6.0


def gaussian_nll(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Gaussian negative log likelihood per target component (constant 0.5 log 2 pi dropped)."""
    return 0.5 * (log_var + (target - mean) ** 2 * torch.exp(-log_var)).mean()


@dataclass(frozen=True)
class DecoderConfig:
    mix_channels: int = 8
    channels: tuple[int, ...] = (32, 64, 128)
    kernel: int = 7
    hidden: int = 256
    groups: int = 8


class _ResBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, groups: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel, padding=kernel // 2),
            nn.GroupNorm(groups, ch),
            nn.GELU(),
            nn.Conv1d(ch, ch, kernel, padding=kernel // 2),
            nn.GroupNorm(groups, ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ConvDecoder(nn.Module):
    """1D CNN decoder (see module docstring). ``n_time`` T, ``n_points`` F, ``n_targets`` D."""

    def __init__(self, n_time: int, n_points: int, n_targets: int, config: DecoderConfig | None = None) -> None:
        super().__init__()
        cfg = config or DecoderConfig()
        self.config = cfg
        self.n_targets = n_targets
        self.mix = nn.Conv1d(n_time, cfg.mix_channels, 1)
        ch0 = cfg.channels[0]
        layers: list[nn.Module] = [
            nn.Conv1d(cfg.mix_channels, ch0, cfg.kernel, padding=cfg.kernel // 2),
            nn.GroupNorm(cfg.groups, ch0),
            nn.GELU(),
        ]
        length = n_points
        for i, ch in enumerate(cfg.channels):
            nxt = cfg.channels[i + 1] if i + 1 < len(cfg.channels) else ch
            layers += [_ResBlock(ch, cfg.kernel, cfg.groups), _ResBlock(ch, cfg.kernel, cfg.groups)]
            layers += [nn.Conv1d(ch, nxt, 4, stride=2, padding=1), nn.GELU()]
            length = length // 2
        self.trunk = nn.Sequential(*layers)
        self.flat = cfg.channels[-1] * length
        self.head = nn.Sequential(
            nn.Linear(self.flat + 1, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, 2 * n_targets),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.mix(x)).flatten(1)
        out = self.head(torch.cat([h, c[:, None]], dim=1))
        mean, log_var = out.chunk(2, dim=1)
        return mean, log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)


class TimeBinLinear(nn.Module):
    """Linear baseline: learned time-bin weighting -> weighted spectrum -> affine map to the targets."""

    def __init__(self, n_time: int, n_points: int, n_targets: int) -> None:
        super().__init__()
        self.n_targets = n_targets
        self.weights = nn.Parameter(torch.full((n_time,), 1.0 / n_time))
        self.readout = nn.Linear(n_points + 1, n_targets)
        self.noise = nn.Linear(1, n_targets)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spec = torch.einsum("btf,t->bf", x, self.weights)
        mean = self.readout(torch.cat([spec, c[:, None]], dim=1))
        log_var = self.noise(c[:, None]).clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        return mean, log_var


Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
#: ``loss(mean, log_var, target) -> scalar``; the default is :func:`gaussian_nll`.
LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def bce_logits_loss(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with the model's mean output as logits (bit targets in {0, 1}); log_var unused."""
    return nn.functional.binary_cross_entropy_with_logits(mean, target)


@dataclass
class TrainConfig:
    steps: int = 6000
    batch: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    #: Plain MSE on the means for this fraction of the steps first (stabilises the variance head).
    mse_warmup: float = 0.05
    eval_every: int = 250
    grad_clip: float = 1.0


@dataclass
class TrainHistory:
    step: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_nll: list[float] = field(default_factory=list)
    val_rmse: list[float] = field(default_factory=list)
    best_step: int = 0
    seconds: float = 0.0


def train_decoder(
    model: nn.Module,
    sample: Callable[[int], Batch],
    val_batches: list[Batch],
    config: TrainConfig,
    log: Callable[[int, dict[str, float]], None] | None = None,
    loss: LossFn | None = None,
) -> TrainHistory:
    """Train with AdamW + cosine decay on fresh batches ``sample(step) -> (x, c, y)``; keep the weights with the
    best validation loss (on the fixed ``val_batches``). Returns the history; ``model`` ends at the best weights.

    ``loss`` replaces the Gaussian NLL (e.g. :func:`bce_logits_loss` for bit recall); the MSE warm-up applies only
    to the default loss."""
    loss_fn = loss or gaussian_nll
    import time

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, config.steps))))
    )
    hist = TrainHistory()
    best = math.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    warm = int(config.mse_warmup * config.steps)
    t0 = time.perf_counter()
    running = 0.0
    for step in range(1, config.steps + 1):
        model.train()
        x, c, y = sample(step)
        mean, log_var = model(x, c)
        value = ((mean - y) ** 2).mean() if loss is None and step <= warm else loss_fn(mean, log_var, y)
        opt.zero_grad(set_to_none=True)
        value.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        opt.step()
        sched.step()
        running += float(value.detach())
        if step % config.eval_every == 0 or step == config.steps:
            nll, rmse = evaluate_nll(model, val_batches, loss)
            hist.step.append(step)
            hist.train_loss.append(running / config.eval_every)
            hist.val_nll.append(nll)
            hist.val_rmse.append(rmse)
            running = 0.0
            if (step > warm or loss is not None) and nll < best:
                best, hist.best_step = nll, step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if log is not None:
                log(step, {"train_loss": hist.train_loss[-1], "val_nll": nll, "val_rmse": rmse})
    model.load_state_dict(best_state)
    hist.seconds = time.perf_counter() - t0
    return hist


@torch.no_grad()
def evaluate_nll(model: nn.Module, batches: list[Batch], loss: LossFn | None = None) -> tuple[float, float]:
    """(mean loss (default: Gaussian NLL), RMSE of the means over components) in network units."""
    loss_fn = loss or gaussian_nll
    model.eval()
    nll, se, n = 0.0, 0.0, 0
    for x, c, y in batches:
        mean, log_var = model(x, c)
        nll += float(loss_fn(mean, log_var, y)) * y.shape[0]
        se += float(((mean - y) ** 2).mean()) * y.shape[0]
        n += y.shape[0]
    return nll / n, math.sqrt(se / n)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, c: torch.Tensor, chunk: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation (network units), chunked."""
    model.eval()
    ms, ss = [], []
    for i in range(0, x.shape[0], chunk):
        m, lv = model(x[i : i + chunk], c[i : i + chunk])
        ms.append(m)
        ss.append(torch.exp(0.5 * lv))
    return torch.cat(ms), torch.cat(ss)
