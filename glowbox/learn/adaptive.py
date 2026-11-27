"""Adaptive Bayesian measurement for Task 1 (project brief, phase 2): keep a posterior over the hidden field and
choose the next measurement from it.

**Posterior.** A batched particle filter (E independent episodes, N particles each) over the folded field (the
wedge Bx >= By >= Bz >= 0 of :func:`glowbox.learn.field_data.fold_to_wedge`; the likelihood is invariant under
the signed axis permutations, docs/results/m2-field-estimation.md). Prior: milestone 2's, |B| uniform in
[b_min, b_max], direction isotropic, i.e. a density proportional to 1/|B|^2 on that shell. Likelihood of a block of
S shots of action a: independent Poisson counts per readout window with means S * lambda_a(B) from the emulator
(:class:`~glowbox.learn.measurement.AxisTable`). The counts per action are sufficient statistics (C_a,w total
counts, S_a total shots), so the log-likelihood of any field given every block so far is
sum_a,w C_a,w log lambda_a,w(B) - S_a lambda_a,w(B), cheap to evaluate for new particles.

**Resample-move.** When the effective sample size drops below ``ess_frac`` N: systematic resampling, then
``mh_steps`` Metropolis-Hastings moves per particle with the exact posterior (prior x all data) as target and an
isotropic Gaussian random walk folded back into the wedge. The fold of an isotropic proposal is symmetric
(q(B'|B) = sum_g N(gB'; B, s^2) = q(B|B') because every g is orthogonal and the group is closed under inverses), so
the plain Metropolis ratio applies. The step s is ``mh_scale`` times the posterior's RMS standard deviation per
component (at least ``mh_min_t``). This is the standard sequential Monte Carlo design (e.g. Liu and West 2001;
Granade et al., New J. Phys. 14, 103013 (2012) for Bayesian experiment design in quantum sensing).

**Design.** :func:`expected_information_gain` scores every action by the mutual information between the next
block's counts and the field under the current posterior. Default: the Gaussian approximation per window,
I_a = 1/2 sum_w log(1 + S Var_post[lambda_a,w] / E_post[lambda_a,w]) (a Gaussian signal of the posterior's
predictive variance in Poisson noise of the predictive mean; the windows are treated as independent).
``method="mc"`` estimates the exact Poisson mutual information by Monte Carlo instead (tests compare both). The
predictive moments use ``n_predict`` particles drawn from the posterior (systematic resampling), so their cost does
not grow with N.

Policies (each ``select(state) -> actions (E,)``): :class:`FixedSweep` (the milestone 2/4 protocol: every frequency
once at one pulse, in an interleaved order so that a partial sweep still covers the band), :class:`RandomDesign`,
:class:`EIGDesign` (greedy expected information gain), and the RL agent in :mod:`glowbox.learn.rl`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from glowbox.learn.measurement import AxisTable, MeasurementSpec


def fold_wedge_t(b: torch.Tensor) -> torch.Tensor:
    """Torch version of :func:`glowbox.learn.field_data.fold_to_wedge` (sorted |components|, descending)."""
    return torch.sort(b.abs(), dim=-1, descending=True).values


@dataclass(frozen=True)
class FieldPriorSpec:
    """Milestone 2's field prior (|B| uniform in [b_min_t, b_max_t], isotropic direction), folded."""

    b_min_t: float = 0.5e-3
    b_max_t: float = 3.0e-3

    def sample(self, shape: tuple[int, ...], generator: torch.Generator, device: torch.device) -> torch.Tensor:
        d = torch.randn((*shape, 3), generator=generator, device=device, dtype=torch.float64)
        d = d / d.norm(dim=-1, keepdim=True)
        r = self.b_min_t + (self.b_max_t - self.b_min_t) * torch.rand(
            (*shape, 1), generator=generator, device=device, dtype=torch.float64
        )
        return fold_wedge_t(d * r)

    def log_density(self, b: torch.Tensor) -> torch.Tensor:
        """Log prior density up to a constant, -inf outside the shell (wedge points only)."""
        r = b.norm(dim=-1)
        inside = (r >= self.b_min_t) & (r <= self.b_max_t)
        return torch.where(inside, -2.0 * torch.log(r.clamp_min(1e-12)), torch.full_like(r, -math.inf))

    def rmse_t(self, n: int = 200_000, seed: int = 0) -> float:
        """RMS error of the prior mean (the 'no information' error), tesla."""
        g = torch.Generator().manual_seed(seed)
        b = self.sample((n,), g, torch.device("cpu"))
        return float(torch.sqrt(((b - b.mean(0)) ** 2).sum(-1).mean()))


@dataclass
class PosteriorConfig:
    n_particles: int = 512
    ess_frac: float = 0.5
    mh_steps: int = 4
    mh_scale: float = 0.6
    mh_min_t: float = 1e-6
    n_predict: int = 128
    #: Most tempering stages per block (see :meth:`FieldPosterior.update`).
    max_stages: int = 12


def _ess(logw: torch.Tensor) -> torch.Tensor:
    w = torch.softmax(logw, dim=-1)
    return 1.0 / (w**2).sum(-1)


def systematic_indices(logw: torch.Tensor, n: int, generator: torch.Generator) -> torch.Tensor:
    """Systematic resampling: (E, N) log weights -> (E, n) particle indices."""
    e = logw.shape[0]
    w = torch.softmax(logw.double(), dim=-1)
    cdf = torch.cumsum(w, dim=-1)
    cdf[:, -1] = 1.0
    u0 = torch.rand((e, 1), generator=generator, device=logw.device, dtype=torch.float64)
    u = (u0 + torch.arange(n, device=logw.device, dtype=torch.float64)[None]) / n
    return torch.searchsorted(cdf, u).clamp(max=logw.shape[1] - 1)


class FieldPosterior:
    """Batched particle posterior over the folded field for E episodes (see the module docstring)."""

    def __init__(
        self,
        table: AxisTable,
        prior: FieldPriorSpec,
        n_episodes: int,
        config: PosteriorConfig,
        generator: torch.Generator,
    ) -> None:
        self.table, self.prior, self.cfg, self.gen = table, prior, config, generator
        dev = table.device
        self.device = dev
        spec = table.spec
        self.particles = prior.sample((n_episodes, config.n_particles), generator, dev)  # (E, N, 3) float64
        self.logw = torch.zeros(n_episodes, config.n_particles, dtype=torch.float64, device=dev)
        self.counts = torch.zeros(n_episodes, spec.n_actions, spec.n_windows, dtype=torch.float64, device=dev)
        self.shots = torch.zeros(n_episodes, spec.n_actions, dtype=torch.float64, device=dev)
        self.n_moves = 0
        self.accept: list[float] = []
        #: The block being folded in (actions, counts, shots, phi) of the selected episodes during an update.
        self._pending: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    @property
    def n_episodes(self) -> int:
        return int(self.particles.shape[0])

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logw, dim=-1)

    def ess(self) -> torch.Tensor:
        w = self.weights
        return 1.0 / (w**2).sum(-1)

    def mean(self) -> torch.Tensor:
        """Posterior mean (E, 3), tesla."""
        return (self.weights[..., None] * self.particles).sum(1)

    def std(self) -> torch.Tensor:
        """RMS posterior standard deviation per component (E,), tesla."""
        w = self.weights[..., None]
        m = (w * self.particles).sum(1, keepdim=True)
        return torch.sqrt((w * (self.particles - m) ** 2).sum(1).mean(-1))

    def _block_ll(self, b: torch.Tensor, a: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Log-likelihood of one block per episode for fields b (E, n, 3): (E, n)."""
        lam = self.table.means_at(b, a[:, None].expand(-1, b.shape[1])).double().clamp_min(1e-12)
        return (y[:, None] * torch.log(lam) - s[:, None, None] * lam).sum(-1)

    def update(self, actions: torch.Tensor, counts: torch.Tensor, shots: torch.Tensor) -> None:
        """Bayes update with one block per episode: actions (E,), counts (E, W) summed over ``shots`` (E,).

        Adaptive tempering: a block that would drop the ESS below ``ess_frac`` N is folded in as L^phi in stages
        (phi found by bisection so that each stage keeps the ESS at the threshold), with a resample-move after
        each stage whose target includes the tempered block. Without it, one very informative block (a dip seen
        for the first time) can leave a handful of prior particles and lose the true mode for good.
        """
        a = actions.to(self.device).long()
        y = counts.to(self.device, torch.float64)
        s = shots.to(self.device, torch.float64)
        e_all = self.n_episodes
        thr = self.cfg.ess_frac * self.particles.shape[1]
        phi = torch.zeros(e_all, dtype=torch.float64, device=self.device)
        inc = self._block_ll(self.particles, a, y, s)
        for _ in range(self.cfg.max_stages):
            todo = phi < 1.0
            if not bool(todo.any()):
                break
            rem = 1.0 - phi
            full_ok = _ess(self.logw + rem[:, None] * inc) >= thr
            lo = torch.zeros_like(phi)
            hi = rem.clone()
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                ok = _ess(self.logw + mid[:, None] * inc) >= thr
                lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
            # Always make progress (a stage of at least 1e-4 of the block or the rest).
            d = torch.where(full_ok, rem, torch.maximum(lo, torch.minimum(rem, torch.full_like(rem, 1e-4))))
            d = torch.where(todo, d, torch.zeros_like(d))
            self.logw = self.logw + d[:, None] * inc
            self.logw = self.logw - self.logw.max(-1, keepdim=True).values
            phi = phi + d
            phi = torch.where(phi > 1.0 - 1e-12, torch.ones_like(phi), phi)
            low = (self.ess() < thr * 1.0001) & todo
            if bool(low.any()):
                self._pending = (a, y, s, phi)
                try:
                    self.resample_move(low)
                finally:
                    self._pending = None
                inc = self._block_ll(self.particles, a, y, s)
        if bool((phi < 1.0).any()):  # out of stages: take the rest in one go
            self.logw = self.logw + (1.0 - phi)[:, None] * inc
            self.logw = self.logw - self.logw.max(-1, keepdim=True).values
        e = torch.arange(a.shape[0], device=self.device)
        self.counts[e, a] += y
        self.shots[e, a] += s
        self.maybe_resample()

    def log_likelihood(self, b: torch.Tensor) -> torch.Tensor:
        """Log-likelihood of every block so far for fields b (E, n, 3), (E, n) (constants dropped), plus the
        tempered pending block during an update."""
        measured = self.shots > 0  # (E, A)
        n_used = int(measured.sum(-1).max())
        if n_used == 0:
            out = torch.zeros(b.shape[:2], dtype=torch.float64, device=self.device)
        else:
            # each episode's own measured actions, padded to the longest list (padding carries zero shots)
            act = torch.topk(measured.to(torch.int8), n_used, dim=-1).indices  # (E, U)
            lam = self.table.means_per_row(b, act).double().clamp_min(1e-12)  # (E, n, U, W)
            c = torch.gather(self.counts, 1, act[..., None].expand(-1, -1, self.counts.shape[-1]))[:, None]
            s = torch.gather(self.shots, 1, act)[:, None, :, None]
            out = (c * torch.log(lam) - s * lam).sum((-1, -2))
        if self._pending is not None:
            a, y, sh, phi = self._pending
            out = out + phi[:, None] * self._block_ll(b, a, y, sh)
        return out

    def maybe_resample(self) -> None:
        low = self.ess() < self.cfg.ess_frac * self.particles.shape[1]
        if bool(low.any()):
            self.resample_move(low)

    def resample_move(self, which: torch.Tensor) -> None:
        """Resample + MH-move the episodes flagged in ``which`` (E,) bool."""
        idx_e = torch.nonzero(which).reshape(-1)
        n = self.particles.shape[1]
        sub_logw = self.logw[idx_e]
        scale = (self.cfg.mh_scale * self._std_of(idx_e)).clamp_min(self.cfg.mh_min_t)  # (e,)
        pick = systematic_indices(sub_logw, n, self.gen)
        b = torch.gather(self.particles[idx_e], 1, pick[..., None].expand(-1, -1, 3))
        saved = self.counts, self.shots, self._pending
        self.counts, self.shots = self.counts[idx_e], self.shots[idx_e]
        if self._pending is not None:
            self._pending = tuple(x[idx_e] for x in self._pending)  # type: ignore[assignment]
        try:
            lp = self.prior.log_density(b) + self.log_likelihood(b)
            acc = 0.0
            for _ in range(self.cfg.mh_steps):
                step = torch.randn(b.shape, generator=self.gen, device=self.device, dtype=torch.float64)
                prop = fold_wedge_t(b + scale[:, None, None] * step)
                lq = self.prior.log_density(prop)
                ok = torch.isfinite(lq)
                lq = torch.where(ok, lq + self.log_likelihood(prop), lq)
                u = torch.rand(lq.shape, generator=self.gen, device=self.device, dtype=torch.float64)
                take = torch.log(u) < (lq - lp)
                b = torch.where(take[..., None], prop, b)
                lp = torch.where(take, lq, lp)
                acc += float(take.double().mean())
        finally:
            self.counts, self.shots, self._pending = saved
        self.particles[idx_e] = b
        self.logw[idx_e] = 0.0
        self.n_moves += 1
        self.accept.append(acc / max(1, self.cfg.mh_steps))

    def _std_of(self, idx_e: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.logw[idx_e], -1)[..., None]
        p = self.particles[idx_e]
        m = (w * p).sum(1, keepdim=True)
        return torch.sqrt((w * (p - m) ** 2).sum(1).mean(-1))

    def predictive_sample(self, n: int | None = None) -> torch.Tensor:
        """``n`` (default ``n_predict``) equally weighted particles drawn from the posterior, (E, n, 3)."""
        k = n or self.cfg.n_predict
        pick = systematic_indices(self.logw, k, self.gen)
        return torch.gather(self.particles, 1, pick[..., None].expand(-1, -1, 3))

    def predictive_moments(self, n: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior predictive mean and variance of the counts per shot of every action, each (E, A, W)."""
        lam = self.table.means(self.predictive_sample(n))  # (E, n, A, W) float32
        return lam.mean(1), lam.var(1, unbiased=False)


def expected_information_gain(
    post: FieldPosterior,
    shots: float,
    method: str = "gaussian",
    n_mc: int = 64,
    n_predict: int | None = None,
) -> torch.Tensor:
    """Expected information gain (nats) of a block of ``shots`` shots of every action, (E, A)."""
    if method == "gaussian":
        mu, var = post.predictive_moments(n_predict)
        return 0.5 * torch.log1p(shots * var / mu.clamp_min(1e-9)).sum(-1)
    if method != "mc":
        raise ValueError(f"unknown EIG method {method!r}")
    # Monte Carlo over the predictive sample: y ~ Poisson(S lambda_j), I = E[log p(y|j) - log mean_i p(y|i)].
    b = post.predictive_sample(n_predict)
    lam = (post.table.means(b).double() * shots).clamp_min(1e-12)  # (E, n, A, W)
    e, n = lam.shape[:2]
    j = torch.randint(0, n, (e, n_mc), generator=post.gen, device=post.device)
    lam_j = torch.gather(lam, 1, j[..., None, None].expand(-1, -1, *lam.shape[2:]))  # (E, n_mc, A, W)
    y = torch.poisson(lam_j, generator=post.gen)
    # log p(y | i) for every i: (E, n_mc, n, A)
    ll = (y[:, :, None] * torch.log(lam[:, None]) - lam[:, None]).sum(-1)
    ll_j = (y * torch.log(lam_j) - lam_j).sum(-1)
    return (ll_j - (torch.logsumexp(ll, dim=2) - math.log(n))).mean(1).float()


# --- Policies -------------------------------------------------------------------------------------------------------


@dataclass
class DesignState:
    """What a policy may look at: the posterior, the step, the block size and the step budget."""

    posterior: FieldPosterior
    step: int
    n_steps: int
    shots: float


class Policy(Protocol):
    name: str

    def select(self, state: DesignState) -> torch.Tensor: ...


def interleaved_order(n: int) -> np.ndarray:
    """A permutation of range(n) whose every prefix is spread over the range (golden-ratio sequence)."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    keys = np.mod(np.arange(n) * phi, 1.0)
    order: np.ndarray = np.argsort(np.argsort(keys))  # rank of each index
    out = np.empty(n, dtype=np.int64)
    out[order] = np.arange(n)
    return out


@dataclass
class FixedSweep:
    """Every frequency of the grid once per pass at duration index ``dur_index``, interleaved order."""

    spec: MeasurementSpec
    dur_index: int = 1
    name: str = "fixed_sweep"

    def select(self, state: DesignState) -> torch.Tensor:
        order = interleaved_order(self.spec.n_freq)
        f = int(order[state.step % self.spec.n_freq])
        a = self.dur_index * self.spec.n_freq + f
        return torch.full((state.posterior.n_episodes,), a, dtype=torch.long, device=state.posterior.device)


@dataclass
class RandomDesign:
    spec: MeasurementSpec
    generator: torch.Generator
    name: str = "random"

    def select(self, state: DesignState) -> torch.Tensor:
        p = state.posterior
        return torch.randint(0, self.spec.n_actions, (p.n_episodes,), generator=self.generator, device=p.device)


@dataclass
class EIGDesign:
    """Greedy expected information gain; ``allowed`` (A,) bool restricts the actions (e.g. one duration)."""

    method: str = "gaussian"
    allowed: torch.Tensor | None = None
    name: str = "eig"

    def select(self, state: DesignState) -> torch.Tensor:
        g = expected_information_gain(state.posterior, state.shots, self.method)
        if self.allowed is not None:
            g = torch.where(self.allowed.to(g.device)[None], g, torch.full_like(g, -math.inf))
        return torch.argmax(g, dim=-1)


# --- Backends (where the counts come from) and episodes ---------------------------------------------------------------


class Backend(Protocol):
    """Counts of one block per episode: actions (E,), shots per block -> (E, W) counts summed over the shots."""

    def measure(self, actions: torch.Tensor, shots: int) -> torch.Tensor: ...


@dataclass
class TableBackend:
    """Precomputed noiseless means (E, A, W) of each episode's true field (full surrogate) + Poisson noise."""

    means: torch.Tensor
    generator: torch.Generator

    def measure(self, actions: torch.Tensor, shots: int) -> torch.Tensor:
        e = torch.arange(self.means.shape[0], device=self.means.device)
        lam = self.means[e, actions.to(self.means.device)] * shots
        return torch.poisson(lam.double(), generator=self.generator)


@dataclass
class EpisodeLog:
    """Per step (T + 1 rows, row 0 = prior) and episode: squared error (T^2), posterior std (T), actions (T rows)."""

    sq_err: np.ndarray
    post_std: np.ndarray
    actions: np.ndarray
    seconds: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    def rmse_t(self, step: int = -1) -> float:
        return float(np.sqrt(self.sq_err[step].mean()))


def run_episodes(
    policy: Policy,
    backend: Backend,
    truths_t: torch.Tensor,
    table: AxisTable,
    prior: FieldPriorSpec,
    config: PosteriorConfig,
    n_steps: int,
    shots: int,
    generator: torch.Generator,
    on_step: Callable[[int, FieldPosterior], None] | None = None,
) -> EpisodeLog:
    """Run ``n_steps`` blocks of ``shots`` shots for every true field (E, 3) in parallel."""
    import time

    t0 = time.perf_counter()
    e = truths_t.shape[0]
    post = FieldPosterior(table, prior, e, config, generator)
    truths = truths_t.to(post.device, torch.float64)
    sq = [((post.mean() - truths) ** 2).sum(-1).cpu().numpy()]
    sd = [post.std().cpu().numpy()]
    acts = []
    for t in range(n_steps):
        a = policy.select(DesignState(post, t, n_steps, float(shots)))
        y = backend.measure(a, shots)
        post.update(a, y, torch.full((e,), float(shots), device=post.device))
        sq.append(((post.mean() - truths) ** 2).sum(-1).cpu().numpy())
        sd.append(post.std().cpu().numpy())
        acts.append(a.cpu().numpy())
        if on_step is not None:
            on_step(t, post)
    acc = float(np.mean(post.accept)) if post.accept else float("nan")
    return EpisodeLog(
        np.stack(sq),
        np.stack(sd),
        np.stack(acts) if acts else np.zeros((0, e), np.int64),
        time.perf_counter() - t0,
        {"moves": float(post.n_moves), "mh_accept": acc},
    )
