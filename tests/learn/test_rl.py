"""Milestone 5 RL: GAE, PPO on toy problems, the Task 1 environment (rewards, run() backend) and the Task 2 bandit."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch

from glowbox.learn.adaptive import FieldPriorSpec, PosteriorConfig, TableBackend
from glowbox.learn.measurement import AxisTable, MeasurementSpec, measurement_means
from glowbox.learn.rl import (
    CodebookEnv,
    FieldSearchEnv,
    GaussianPolicy,
    Obs,
    PPOConfig,
    Rollout,
    SetPolicy,
    SimulatorBackend,
    collect_rollout,
    gae,
    pattern_log_posterior,
    train_ppo,
)
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions
from glowbox.sim.surrogate.simulator import SurrogateSimulator


def test_gae_matches_hand_computation():
    r = torch.tensor([[1.0], [0.0], [2.0]])
    v = torch.tensor([[0.5], [0.2], [1.0]])
    done = torch.tensor([[False], [False], [True]])
    ro = Rollout({}, torch.zeros(3, 1), torch.zeros(3, 1), v, r, done, [])
    adv, ret = gae(ro, gamma=1.0, lam=1.0)
    np.testing.assert_allclose(ret[:, 0].numpy(), [3.0, 2.0, 2.0])  # Monte Carlo returns when lambda = 1
    np.testing.assert_allclose(adv[:, 0].numpy(), [2.5, 1.8, 1.0])


class _Bandit:
    """One-step continuous bandit: reward -(a - target)^2 per env."""

    def __init__(self, n: int, target: float) -> None:
        self.num_envs, self.target = n, target

    def reset(self, *, seed: int | None = None) -> tuple[Obs, dict[str, Any]]:
        return {"u": torch.ones(self.num_envs, 1)}, {}

    def step(self, action: torch.Tensor) -> tuple[Obs, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        r = -((action[:, 0] - self.target) ** 2)
        d = torch.ones(self.num_envs, dtype=torch.bool)
        return {"u": torch.ones(self.num_envs, 1)}, r, d, torch.zeros_like(d), {}


class _Pick:
    """T-step discrete env: reward 1 for choosing the action whose feature 0 is largest."""

    def __init__(self, n: int, a: int, t: int, gen: torch.Generator) -> None:
        self.num_envs, self.a, self.t, self.gen, self.k = n, a, t, gen, 0

    def _obs(self) -> Obs:
        self.x = torch.randn(self.num_envs, self.a, 2, generator=self.gen)
        return {"x": self.x, "g": torch.zeros(self.num_envs, 2)}

    def reset(self, *, seed: int | None = None) -> tuple[Obs, dict[str, Any]]:
        self.k = 0
        return self._obs(), {}

    def step(self, action: torch.Tensor) -> tuple[Obs, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        r = (action == self.x[..., 0].argmax(-1)).float()
        self.k += 1
        d = torch.full((self.num_envs,), self.k >= self.t, dtype=torch.bool)
        return self._obs(), r, d, torch.zeros_like(d), {}


def test_ppo_solves_toy_problems():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    pol = GaussianPolicy(1, 1, (16,), init_log_std=0.0)
    hist = train_ppo(_Bandit(256, 2.0), pol, PPOConfig(iterations=60, lr=3e-2, gae_lambda=1.0, ent_coef=0.0), 1, g)
    with torch.no_grad():
        mean = float(pol.mean(torch.ones(1, 1))[0, 0])
    assert abs(mean - 2.0) < 0.3 and hist.episode_return[-1] > hist.episode_return[0]
    sp = SetPolicy(2, 2, hidden=16)
    env = _Pick(64, 8, 4, torch.Generator().manual_seed(1))
    h2 = train_ppo(env, sp, PPOConfig(iterations=40, lr=1e-2, ent_coef=0.0), 4, g)
    assert np.mean(h2.episode_return[-5:]) > 3.0  # 4 steps, chance is 0.5


def test_pattern_log_posterior_and_codebook_reward():
    tm = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]]], dtype=torch.float64)
    lp = pattern_log_posterior(tm, tm, 1e4)  # noiseless counts at the templates themselves
    assert torch.allclose(lp.exp().sum(-1), torch.ones(1, 4, dtype=torch.float64))
    assert bool((lp.argmax(-1) == torch.arange(4)).all())
    pats = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
    g = torch.Generator().manual_seed(2)
    sep = CodebookEnv(pats, 2, lambda z: tm[0].repeat(2, 1), 1e4, 4, g)
    sep.reset()
    _, r, done, _, info = sep.step(torch.zeros(8, 1))
    assert bool(done.all()) and float(r.mean()) == pytest.approx(2.0, abs=1e-6) and info["bit_acc"] == 1.0
    same = CodebookEnv(pats, 2, lambda z: torch.ones(8, 2), 1e4, 4, g)
    same.reset()
    _, r, *_ = same.step(torch.zeros(8, 1))
    assert float(r.mean()) == pytest.approx(0.0, abs=1e-6)  # identical templates: log2(1/4) + 2


SPEC = MeasurementSpec(f_start_hz=2.80e9, f_stop_hz=2.94e9, step_hz=20e6, durations_ns=(100,))
OPT = ModelOptions(n_detuning=2, dtype=torch.float64)


@pytest.fixture(scope="module")
def table() -> AxisTable:
    return AxisTable.build(SPEC, EnvParams(), OPT, "cpu", d_par_t=50e-6, d_perp_t=500e-6, n_az=4)


def test_field_env_rewards_sum_to_the_log_error_reduction(table: AxisTable):
    prior = FieldPriorSpec()
    g = torch.Generator().manual_seed(3)
    pool = prior.sample((8,), g, torch.device("cpu"))
    means = table.means(pool).double()

    def sampler(n: int, gen: torch.Generator) -> tuple[torch.Tensor, TableBackend]:
        return pool[:n], TableBackend(means[:n], gen)

    env = FieldSearchEnv(table, prior, sampler, 4, 5, 500, PosteriorConfig(n_particles=64, n_predict=16), g)
    pol = SetPolicy(env.n_features, 2, 16)
    ro = collect_rollout(env, pol, 5)
    assert ro.obs["x"].shape == (5, 4, SPEC.n_actions, env.n_features) and bool(ro.dones[-1].all())
    e2 = ro.infos[-1]["sq_err"]
    total = ro.rewards.sum(0).double()
    # the per-step rewards telescope: log(e_0^2 + eps) - log(e_T^2 + eps)
    assert torch.allclose(total, (env._first - torch.log(e2 + env.eps2)).double(), atol=1e-4)


def test_simulator_backend_matches_the_table_backend():
    """The run()-based backend (here the surrogate's run, bin path) draws from the same means as the table."""
    env = EnvParams(b_field_t=(1.0e-3, 0.4e-3, 0.2e-3))
    sim = SurrogateSimulator(device="cpu", options=OPT)
    be = SimulatorBackend(sim, SPEC, [env], np.random.default_rng(0))
    m = measurement_means(SPEC, EnvParams(), np.array([env.b_field_t]), OPT, "cpu")[0].numpy()
    for a in range(SPEC.n_actions):
        np.testing.assert_allclose(be.mean_counts(0, a), m[a], rtol=1e-8)
    y = be.measure(torch.tensor([2]), 10_000)
    assert y.shape == (1, 2) and be.calls == SPEC.n_actions
    assert abs(float(y[0, 0]) / 10_000 - m[2, 0]) < 5 * math.sqrt(m[2, 0] / 10_000)
