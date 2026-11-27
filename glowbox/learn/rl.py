"""RL agent (project brief, phase 2): PPO with free exploration over pulse parameters, reward from task performance.

**Algorithm.** Proximal policy optimisation (Schulman et al., arXiv:1707.06347) with a clipped surrogate objective,
generalised advantage estimation, an entropy bonus and a learned value baseline, written compactly in torch
(:func:`train_ppo`, :func:`ppo_update`). Why PPO and why no dependency: both tasks here are on-policy problems with
cheap, massively parallel simulated episodes (the surrogate runs every episode of a batch at once on the GPU), the
regime PPO is built for; SAC's replay buffer and twin critics buy sample efficiency that a simulator does not need,
and it is defined for continuous actions only (Task 1's actions are discrete). Libraries (stable-baselines3,
CleanRL) expect one environment step per call through gymnasium's NumPy interface, which would serialise the
batched surrogate; ~200 lines of torch keep everything on the device.

**Environments** follow gymnasium's *vector* API shape (:class:`VectorEnv`: ``reset(seed=...) -> (obs, info)``,
``step(action) -> (obs, reward, terminated, truncated, info)``, batched tensors instead of NumPy arrays) without
the dependency.

* :class:`FieldSearchEnv` (Task 1): the agent picks the next measurement (MW frequency x pulse duration, the action
  grid of :class:`~glowbox.learn.measurement.MeasurementSpec`) for E episodes at once; each step is a block of S
  shots. The counts come from a :class:`~glowbox.learn.adaptive.Backend`: precomputed surrogate means plus Poisson
  noise for training (:class:`~glowbox.learn.adaptive.TableBackend`) or any ``Simulator.run()`` - the QuTiP
  reference, the surrogate, later the hardware - through :class:`SimulatorBackend`. The environment keeps the
  Bayesian posterior of :mod:`glowbox.learn.adaptive` (the agent's memory of what it has measured) and shows the
  agent its posterior predictive per action: predicted dip depth, the spread of the prediction over the posterior
  in units of the block's shot noise (per window), how often the action was measured, its frequency and duration;
  plus the step fraction and the posterior width. The agent is not shown the information gain itself and has to
  learn what to do with these. Reward: the reduction of log(squared error + eps) of the posterior-mean estimate
  per step (sums to log(e_0^2 / e_T^2): every halving of the error is worth the same, early or late). A hardware
  run has no true field: ``reward="variance"`` uses the posterior variance instead (implemented, not used for the
  results here).
* :class:`CodebookEnv` (Task 2): a one-step contextual bandit. The context is a bit pattern, the action the
  pre-activations of the write block's MW segments (amplitude, phase, frequency offset, duration per segment, the
  bounded parametrisation of :mod:`glowbox.learn.encoder`). R codebooks (all 2^L patterns each) are sampled per
  step; the reward of each pattern is L + log2 P(pattern | readout) under the Bayes-optimal readout of *its*
  codebook (Poisson likelihood against the codebook's noiseless templates, averaged over noise draws): the bits
  recovered, the quantity that the milestone-3 'recalled bits' summarises.

**Policies.** :class:`SetPolicy` scores every action with one shared MLP of that action's features (permutation
equivariant over the action set) and a value head on the mean-pooled embedding; :class:`GaussianPolicy` is a
state-dependent mean with a learned, state-independent standard deviation and an MLP value baseline.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn

from glowbox.learn.adaptive import Backend, DesignState, FieldPosterior, FieldPriorSpec, PosteriorConfig
from glowbox.learn.measurement import AxisTable, MeasurementSpec
from glowbox.sim.interface import EnvParams, Simulator

Obs = dict[str, torch.Tensor]


class VectorEnv(Protocol):
    """gymnasium.vector-style batched environment (torch tensors)."""

    num_envs: int

    def reset(self, *, seed: int | None = None) -> tuple[Obs, dict[str, Any]]: ...

    def step(self, action: torch.Tensor) -> tuple[Obs, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]: ...


# --- Task 1: choose the next measurement ------------------------------------------------------------------------------


@dataclass
class SimulatorBackend:
    """Counts of one block per episode from ``simulator.run(spec.sequence(a), env_e, shots=S)``.

    ``envs[e]`` holds episode e's device with its true field in ``b_field_t``. With ``cache_rates`` (simulators
    only; hardware does not fill ``rate_cps``) each distinct (episode, action) is simulated once, with one shot,
    and later blocks draw Poisson counts from its noiseless rate: the same statistics, and the reference becomes
    affordable. ``calls`` counts ``run()`` calls."""

    simulator: Simulator
    spec: MeasurementSpec
    envs: list[EnvParams]
    rng: np.random.Generator
    cache_rates: bool = True
    calls: int = 0
    _cache: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    def mean_counts(self, episode: int, action: int) -> np.ndarray:
        """Noiseless counts per shot per window (W,) of one action (a simulator with ``rate_cps``)."""
        key = (episode, action)
        if key not in self._cache:
            tr = self.simulator.run(self.spec.sequence(action), self.envs[episode], shots=1, rng=self.rng)
            self.calls += 1
            if tr.rate_cps is None:
                raise ValueError("cache_rates needs a simulator that fills rate_cps")
            r = np.asarray(tr.rate_cps[0]) * tr.bin_ns * 1e-9
            self._cache[key] = np.array([r[s].sum() for s in self.spec.window_slices(action)])
        return self._cache[key]

    def measure(self, actions: torch.Tensor, shots: int) -> torch.Tensor:
        out = np.empty((len(self.envs), self.spec.n_windows))
        for e, a in enumerate(actions.tolist()):
            if self.cache_rates:
                out[e] = self.rng.poisson(shots * self.mean_counts(e, int(a)))
            else:
                tr = self.simulator.run(self.spec.sequence(int(a)), self.envs[e], shots=shots, rng=self.rng)
                self.calls += 1
                out[e] = [tr.counts[:, s].sum() for s in self.spec.window_slices(int(a))]
        return torch.as_tensor(out, dtype=torch.float64)


#: Per-action features of :class:`FieldSearchEnv` (+ one-hot duration).
N_BASE_FEATURES = 5
#: Global features (step fraction, log posterior width).
N_GLOBAL = 2


def observe_posterior(post: FieldPosterior, shots: float, t: int, n_steps: int, prior_std_t: float) -> Obs:
    """The agent's view of a posterior: per-action features x (E, A, F) and globals g (E, 2) (module docstring)."""
    spec = post.table.spec
    mu, var = post.predictive_moments()
    snr = torch.log1p(torch.sqrt(shots * var / mu.clamp_min(1e-9)))  # (E, A, W)
    if snr.shape[-1] == 1:
        snr = torch.cat([snr, snr], -1)
    ref = mu[..., 0].median(dim=-1, keepdim=True).values
    dip = 10.0 * (mu[..., 0] / ref - 1.0)
    meas = torch.log1p(post.shots.float() / shots)
    e, a = mu.shape[:2]
    fi, di = spec.split(np.arange(spec.n_actions))
    dev = mu.device
    fpos = torch.as_tensor(2.0 * fi / max(1, spec.n_freq - 1) - 1.0, dtype=torch.float32, device=dev)
    dur = torch.nn.functional.one_hot(torch.as_tensor(di, device=dev), spec.n_dur).float()
    x = torch.cat(
        [
            dip[..., None],
            snr[..., :2],
            meas[..., None],
            fpos[None, :, None].expand(e, a, 1),
            dur[None].expand(e, a, -1),
        ],
        dim=-1,
    )
    g = torch.stack(
        [torch.full((e,), t / n_steps, device=dev), torch.log(post.std().float() / prior_std_t)],
        dim=-1,
    )
    return {"x": x, "g": g}


class FieldSearchEnv:
    """Task 1 as a vector environment (see the module docstring). ``sampler(n, generator)`` draws the episodes'
    true fields (n, 3) and the backend that measures them."""

    def __init__(
        self,
        table: AxisTable,
        prior: FieldPriorSpec,
        sampler: Callable[[int, torch.Generator], tuple[torch.Tensor, Backend]],
        num_envs: int,
        n_steps: int,
        shots: int,
        posterior: PosteriorConfig,
        generator: torch.Generator,
        reward: str = "error",
        eps_t: float = 5e-6,
    ) -> None:
        if reward not in ("error", "variance"):
            raise ValueError(f"unknown reward {reward!r}")
        self.table, self.prior, self.sampler = table, prior, sampler
        self.num_envs, self.n_steps, self.shots = num_envs, n_steps, shots
        self.post_cfg, self.gen, self.reward_mode, self.eps2 = posterior, generator, reward, eps_t**2
        self.spec = table.spec
        self.prior_std = prior.rmse_t() / math.sqrt(3.0)
        self.post: FieldPosterior | None = None
        self.truths: torch.Tensor | None = None
        self.backend: Backend | None = None
        self.t = 0

    @property
    def n_features(self) -> int:
        return N_BASE_FEATURES + self.spec.n_dur

    def _loss(self) -> torch.Tensor:
        assert self.post is not None
        if self.reward_mode == "error":
            assert self.truths is not None
            e2 = ((self.post.mean() - self.truths) ** 2).sum(-1)
        else:
            e2 = 3.0 * self.post.std() ** 2
        return torch.log(e2 + self.eps2)

    def observe(self) -> Obs:
        assert self.post is not None
        return observe_posterior(self.post, float(self.shots), self.t, self.n_steps, self.prior_std)

    def reset(self, *, seed: int | None = None) -> tuple[Obs, dict[str, Any]]:
        if seed is not None:
            self.gen.manual_seed(seed)
        self.truths, self.backend = self.sampler(self.num_envs, self.gen)
        self.truths = self.truths.to(self.table.device, torch.float64)
        self.post = FieldPosterior(self.table, self.prior, self.num_envs, self.post_cfg, self.gen)
        self.t = 0
        self._last = self._loss()
        self._first = self._last
        return self.observe(), {}

    def step(self, action: torch.Tensor) -> tuple[Obs, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        assert self.post is not None and self.backend is not None
        a = action.to(self.table.device).long()
        y = self.backend.measure(a, self.shots)
        self.post.update(a, y, torch.full((self.num_envs,), float(self.shots), device=self.table.device))
        self.t += 1
        loss = self._loss()
        reward = (self._last - loss).float()
        self._last = loss
        done = torch.full((self.num_envs,), self.t >= self.n_steps, dtype=torch.bool, device=reward.device)
        info: dict[str, Any] = {}
        if self.truths is not None:
            info["sq_err"] = ((self.post.mean() - self.truths) ** 2).sum(-1)
        return self.observe(), reward, done, torch.zeros_like(done), info


class SetPolicy(nn.Module):
    """Logits over a set of actions from per-action features x (E, A, F) and globals g (E, G); value from the
    mean-pooled per-action embedding and g."""

    def __init__(self, n_features: int, n_global: int, hidden: int = 64) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_features + n_global, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.logit = nn.Linear(hidden, 1)
        self.value = nn.Sequential(nn.Linear(hidden + n_global, hidden), nn.GELU(), nn.Linear(hidden, 1))
        with torch.no_grad():
            self.logit.weight.mul_(0.1)
            self.logit.bias.zero_()

    def forward(self, obs: Obs) -> tuple[torch.distributions.Distribution, torch.Tensor]:
        x, g = obs["x"], obs["g"]
        h = self.enc(torch.cat([x, g[:, None].expand(-1, x.shape[1], -1)], dim=-1))
        logits = self.logit(h)[..., 0]
        v = self.value(torch.cat([h.mean(1), g], dim=-1))[..., 0]
        return torch.distributions.Categorical(logits=logits), v


@dataclass
class RLDesign:
    """A trained :class:`SetPolicy` as an :class:`~glowbox.learn.adaptive.Policy` for
    :func:`~glowbox.learn.adaptive.run_episodes` (greedy: the most probable action; else sampled)."""

    policy: SetPolicy
    prior_std_t: float
    greedy: bool = True
    generator: torch.Generator | None = None
    name: str = "rl"

    def select(self, state: DesignState) -> torch.Tensor:
        obs = observe_posterior(state.posterior, state.shots, state.step, state.n_steps, self.prior_std_t)
        with torch.no_grad():
            dist, _ = self.policy(obs)
        assert isinstance(dist, torch.distributions.Categorical)
        if self.greedy:
            return torch.argmax(dist.logits, dim=-1)
        return torch.multinomial(dist.probs, 1, generator=self.generator)[:, 0]


# --- Task 2: write a bit pattern (contextual bandit) ------------------------------------------------------------------


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian over R^n_out with a state-dependent mean and a learned global log std."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden: tuple[int, ...] = (64, 64),
        init_mean: torch.Tensor | None = None,
        init_log_std: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        w = n_in
        for h in hidden:
            layers += [nn.Linear(w, h), nn.GELU()]
            w = h
        last = nn.Linear(w, n_out)
        with torch.no_grad():
            last.weight.mul_(0.1)
            last.bias.copy_(torch.zeros(n_out) if init_mean is None else init_mean.reshape(-1))
        self.mean = nn.Sequential(*layers, last)
        self.log_std = nn.Parameter(torch.full((n_out,), float(init_log_std)))
        self.value = nn.Sequential(nn.Linear(n_in, hidden[0]), nn.GELU(), nn.Linear(hidden[0], 1))

    def forward(self, obs: Obs) -> tuple[torch.distributions.Distribution, torch.Tensor]:
        u = obs["u"]
        dist = torch.distributions.Normal(self.mean(u), torch.exp(self.log_std).expand(u.shape[0], -1))
        return torch.distributions.Independent(dist, 1), self.value(u)[..., 0]


def pattern_log_posterior(y: torch.Tensor, templates: torch.Tensor, shots: float) -> torch.Tensor:
    """log P(pattern j | counts) under the Poisson model and a uniform prior.

    ``y`` (R, n, K) counts per shot (averaged over ``shots``), ``templates`` (R, P, K) mean counts per shot;
    returns (R, n, P)."""
    lam = templates.clamp_min(1e-12).double()
    ll = shots * (torch.einsum("rnk,rpk->rnp", y.double(), torch.log(lam)) - lam.sum(-1)[:, None])
    return torch.log_softmax(ll, dim=-1)


class CodebookEnv:
    """Task 2 as a one-step vector environment (see the module docstring).

    ``counts_fn(z)`` maps actions (N, n_action) to noiseless mean counts per shot (N, K) (the surrogate channel,
    or a ``run()``-based one); ``patterns`` (P, L) in {0, 1}. Observation: u = 2 b - 1 of every pattern, repeated
    for ``n_codebooks`` codebooks."""

    def __init__(
        self,
        patterns: torch.Tensor,
        n_codebooks: int,
        counts_fn: Callable[[torch.Tensor], torch.Tensor],
        shots: float,
        n_noise: int,
        generator: torch.Generator,
        min_log2p: float = -20.0,
    ) -> None:
        self.patterns = patterns.float()
        self.n_codebooks, self.counts_fn, self.shots, self.n_noise = n_codebooks, counts_fn, shots, n_noise
        self.gen, self.min_log2p = generator, min_log2p
        self.n_patterns, self.n_bits = patterns.shape
        self.num_envs = n_codebooks * self.n_patterns

    def reset(self, *, seed: int | None = None) -> tuple[Obs, dict[str, Any]]:
        if seed is not None:
            self.gen.manual_seed(seed)
        u = (2.0 * self.patterns - 1.0).repeat(self.n_codebooks, 1)
        return {"u": u}, {}

    def step(self, action: torch.Tensor) -> tuple[Obs, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        mu = self.counts_fn(action).double()  # (R * P, K)
        r, p = self.n_codebooks, self.n_patterns
        tmpl = mu.reshape(r, p, -1)
        lam = (tmpl[:, :, None] * self.shots).expand(-1, -1, self.n_noise, -1)
        y = torch.poisson(lam, generator=self.gen) / self.shots  # (R, P, n_noise, K)
        logp = pattern_log_posterior(y.reshape(r, p * self.n_noise, -1), tmpl, self.shots)
        logp = logp.reshape(r, p, self.n_noise, p)
        lp_true = torch.stack([logp[:, j, :, j] for j in range(p)], dim=1)  # (R, P, n_noise)
        log2p = torch.clamp(lp_true / math.log(2.0), min=self.min_log2p)
        reward = (self.n_bits + log2p.mean(-1)).reshape(-1).float()
        # bitwise MAP decisions for monitoring
        post = torch.exp(logp)  # (R, P, n_noise, P)
        pbit = torch.einsum("rpnq,ql->rpnl", post, self.patterns.double().to(post.device))
        acc = ((pbit > 0.5).double() == self.patterns.double().to(post.device)[None, :, None]).double().mean()
        done = torch.ones(self.num_envs, dtype=torch.bool, device=reward.device)
        info = {"bit_acc": float(acc), "bits": float(reward.mean())}
        return {"u": torch.zeros(self.num_envs, self.n_bits)}, reward, done, torch.zeros_like(done), info


# --- PPO --------------------------------------------------------------------------------------------------------------


class ActorCritic(Protocol):
    def __call__(self, obs: Obs) -> tuple[torch.distributions.Distribution, torch.Tensor]: ...

    def parameters(self) -> Any: ...


@dataclass
class PPOConfig:
    iterations: int = 200
    lr: float = 3e-4
    epochs: int = 4
    minibatches: int = 4
    clip: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    #: Stop the epochs of an update early when the approximate KL exceeds this (0 = never).
    target_kl: float = 0.05


@dataclass
class Rollout:
    obs: dict[str, torch.Tensor]  # each (T, E, ...)
    actions: torch.Tensor  # (T, E, ...)
    logp: torch.Tensor  # (T, E)
    values: torch.Tensor  # (T, E)
    rewards: torch.Tensor  # (T, E)
    dones: torch.Tensor  # (T, E)
    infos: list[dict[str, Any]]


def collect_rollout(env: VectorEnv, policy: ActorCritic, n_steps: int, seed: int | None = None) -> Rollout:
    """One synchronous episode batch (``n_steps`` steps, or until every env is done)."""
    obs, _ = env.reset(seed=seed)
    keys = list(obs)
    buf: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    acts, logps, vals, rews, dones, infos = [], [], [], [], [], []
    for _ in range(n_steps):
        with torch.no_grad():
            dist, v = policy(obs)
            a = dist.sample()
            lp = dist.log_prob(a)
        for k in keys:
            buf[k].append(obs[k])
        obs, r, term, trunc, info = env.step(a)
        acts.append(a)
        logps.append(lp)
        vals.append(v)
        rews.append(r.to(v.device))
        dones.append((term | trunc).to(v.device))
        infos.append(info)
        if bool(dones[-1].all()):
            break
    return Rollout(
        {k: torch.stack(buf[k]) for k in keys},
        torch.stack(acts),
        torch.stack(logps),
        torch.stack(vals),
        torch.stack(rews),
        torch.stack(dones),
        infos,
    )


def gae(rollout: Rollout, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(advantages, returns), each (T, E); the episode ends with the rollout (terminal value 0)."""
    t_len = rollout.rewards.shape[0]
    adv = torch.zeros_like(rollout.rewards)
    last = torch.zeros_like(rollout.rewards[0])
    for t in reversed(range(t_len)):
        nonterm = 1.0 - rollout.dones[t].float()
        nxt = rollout.values[t + 1] if t + 1 < t_len else torch.zeros_like(last)
        delta = rollout.rewards[t] + gamma * nxt * nonterm - rollout.values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + rollout.values


def ppo_update(
    policy: ActorCritic, opt: torch.optim.Optimizer, rollout: Rollout, cfg: PPOConfig, generator: torch.Generator
) -> dict[str, float]:
    adv, ret = gae(rollout, cfg.gamma, cfg.gae_lambda)
    n = adv.numel()
    flat_obs = {k: v.reshape(n, *v.shape[2:]) for k, v in rollout.obs.items()}
    flat_a = rollout.actions.reshape(n, *rollout.actions.shape[2:])
    old_lp, old_v = rollout.logp.reshape(n), rollout.values.reshape(n)
    adv_f, ret_f = adv.reshape(n), ret.reshape(n)
    adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
    mb = max(1, n // cfg.minibatches)
    stats: dict[str, list[float]] = {"pg": [], "vf": [], "ent": [], "kl": [], "clipfrac": []}
    params = list(policy.parameters())
    stop = False
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, generator=generator, device=generator.device).to(adv_f.device)
        for i in range(0, n, mb):
            idx = perm[i : i + mb]
            dist, v = policy({k: x[idx] for k, x in flat_obs.items()})
            lp = dist.log_prob(flat_a[idx])
            ratio = torch.exp(lp - old_lp[idx])
            a_i = adv_f[idx]
            pg = -torch.min(ratio * a_i, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a_i).mean()
            vf = 0.5 * ((v - ret_f[idx]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                kl = float(((ratio - 1) - torch.log(ratio)).mean())
            stats["pg"].append(float(pg.detach()))
            stats["vf"].append(float(vf.detach()))
            stats["ent"].append(float(ent.detach()))
            stats["kl"].append(kl)
            stats["clipfrac"].append(float(((ratio - 1).abs() > cfg.clip).float().mean()))
            if cfg.target_kl and kl > 1.5 * cfg.target_kl:
                stop = True
                break
        if stop:
            break
    out = {k: float(np.mean(v)) for k, v in stats.items()}
    out["value_mean"] = float(old_v.mean())
    return out


@dataclass
class PPOHistory:
    iteration: list[int] = field(default_factory=list)
    episode_return: list[float] = field(default_factory=list)
    stats: list[dict[str, float]] = field(default_factory=list)
    seconds: float = 0.0


def train_ppo(
    env: VectorEnv,
    policy: ActorCritic,
    cfg: PPOConfig,
    n_steps: int,
    generator: torch.Generator,
    log: Callable[[int, dict[str, float]], None] | None = None,
) -> PPOHistory:
    """``cfg.iterations`` x (one rollout of ``n_steps``, one PPO update); Adam, linear learning-rate decay."""
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1.0 - i / max(1, cfg.iterations))
    hist = PPOHistory()
    t0 = time.perf_counter()
    for it in range(1, cfg.iterations + 1):
        ro = collect_rollout(env, policy, n_steps)
        st = ppo_update(policy, opt, ro, cfg, generator)
        sched.step()
        ret = float(ro.rewards.sum(0).mean())
        st["return"] = ret
        for k, v in ro.infos[-1].items():
            if isinstance(v, float):
                st[k] = v
            elif isinstance(v, torch.Tensor) and v.numel() > 1:
                st[k] = float(v.double().mean())
        hist.iteration.append(it)
        hist.episode_return.append(ret)
        hist.stats.append(st)
        if log is not None:
            log(it, st)
    hist.seconds = time.perf_counter() - t0
    return hist
