"""Milestone 5 adaptive measurement: prior, particle posterior with tempering and MH moves, EIG, policies. A coarse
emulator table on the CPU; the 'truth' counts come from the emulator itself so the tests check the inference."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from glowbox.learn.adaptive import (
    DesignState,
    EIGDesign,
    FieldPosterior,
    FieldPriorSpec,
    FixedSweep,
    PosteriorConfig,
    TableBackend,
    expected_information_gain,
    fold_wedge_t,
    interleaved_order,
    run_episodes,
    systematic_indices,
)
from glowbox.learn.measurement import AxisTable, MeasurementSpec
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions

SPEC = MeasurementSpec(f_start_hz=2.78e9, f_stop_hz=2.96e9, step_hz=6e6, durations_ns=(100, 200))
PRIOR = FieldPriorSpec()


@pytest.fixture(scope="module")
def table() -> AxisTable:
    opt = ModelOptions(n_detuning=2, detuning_rule="uniform", detuning_spacing=2.0)
    return AxisTable.build(SPEC, EnvParams(), opt, "cpu", d_par_t=25e-6, d_perp_t=250e-6, n_az=4)


def test_prior_and_fold():
    g = torch.Generator().manual_seed(0)
    b = PRIOR.sample((5000,), g, torch.device("cpu"))
    r = b.norm(dim=-1)
    assert float(r.min()) >= PRIOR.b_min_t - 1e-12 and float(r.max()) <= PRIOR.b_max_t + 1e-12
    assert bool((b[:, 0] >= b[:, 1]).all() and (b[:, 1] >= b[:, 2]).all() and (b[:, 2] >= 0).all())
    assert torch.equal(fold_wedge_t(torch.tensor([[-1.0, 3.0, -2.0]])), torch.tensor([[3.0, 2.0, 1.0]]))
    ld = PRIOR.log_density(torch.tensor([[1e-3, 0.0, 0.0], [4e-3, 0.0, 0.0]], dtype=torch.float64))
    assert math.isfinite(float(ld[0])) and ld[1] == -math.inf
    assert 700e-6 < PRIOR.rmse_t(20_000) < 950e-6


def test_interleaved_order_and_systematic_resampling():
    o = interleaved_order(100)
    assert sorted(o.tolist()) == list(range(100))
    assert np.ptp(o[:10]) > 70  # the first ten points already span the band
    logw = torch.log(torch.tensor([[0.1, 0.0, 0.6, 0.3]], dtype=torch.float64).clamp_min(1e-300))
    idx = systematic_indices(logw, 1000, torch.Generator().manual_seed(1))
    counts = torch.bincount(idx[0], minlength=4).double() / 1000
    np.testing.assert_allclose(counts.numpy(), [0.1, 0.0, 0.6, 0.3], atol=2e-3)


def test_posterior_concentrates_on_the_truth(table: AxisTable):
    g = torch.Generator().manual_seed(2)
    truths = PRIOR.sample((8,), g, table.device)
    backend = TableBackend(table.means(truths).double(), g)
    cfg = PosteriorConfig(n_particles=256)
    log = run_episodes(EIGDesign(), backend, truths, table, PRIOR, cfg, 25, 2000, g)
    err = np.sqrt(log.sq_err[-1])
    assert log.rmse_t(-1) < 0.15 * log.rmse_t(0)
    # calibrated: the error is a few posterior standard deviations at most (3 components)
    assert np.all(err < 6 * math.sqrt(3) * log.post_std[-1] + 20e-6)
    assert log.extra["moves"] > 0 and 0.05 < log.extra["mh_accept"] < 0.95


def test_tempering_keeps_the_true_mode(table: AxisTable):
    """One huge block (10^6 shots) from the prior: without tempering a few hundred prior particles cannot hold the
    sharp likelihood; with it the posterior still ends near the truth."""
    g = torch.Generator().manual_seed(3)
    truths = PRIOR.sample((6,), g, table.device)
    post = FieldPosterior(table, PRIOR, 6, PosteriorConfig(n_particles=256), g)
    means = table.means(truths).double()
    for a in (5, 17, 25):  # three informative frequencies at once
        act = torch.full((6,), a)
        y = torch.poisson(means[torch.arange(6), act] * 1e6, generator=g)
        post.update(act, y, torch.full((6,), 1e6))
    assert bool((post.ess() >= 0.5 * 256 - 1).all() or post.n_moves > 0)
    lp_true = post.log_likelihood(truths[:, None])[:, 0]
    lp_post = post.log_likelihood(post.particles).max(-1).values
    # the particles reach the likelihood of the truth (within a few nats per episode)
    assert bool((lp_post > lp_true - 20.0).all())


def test_eig_gaussian_tracks_monte_carlo(table: AxisTable):
    g = torch.Generator().manual_seed(4)
    post = FieldPosterior(table, PRIOR, 2, PosteriorConfig(n_particles=256, n_predict=128), g)
    ga = expected_information_gain(post, 200.0, "gaussian")
    mc = expected_information_gain(post, 200.0, "mc", n_mc=128)
    assert ga.shape == (2, SPEC.n_actions)
    for e in range(2):
        r = np.corrcoef(np.argsort(np.argsort(ga[e].numpy())), np.argsort(np.argsort(mc[e].numpy())))[0, 1]
        assert r > 0.7


def test_fixed_sweep_cycles_the_grid(table: AxisTable):
    g = torch.Generator().manual_seed(5)
    post = FieldPosterior(table, PRIOR, 3, PosteriorConfig(n_particles=16), g)
    pol = FixedSweep(SPEC, dur_index=1)
    acts = [int(pol.select(DesignState(post, t, SPEC.n_freq, 100.0))[0]) for t in range(SPEC.n_freq)]
    assert sorted(acts) == list(range(SPEC.n_freq, 2 * SPEC.n_freq))


def test_restart_recovers_a_lost_mode(table: AxisTable):
    """Particles all stuck at a wrong field: the annealed restart from the prior through the same data finds a set
    with a better posterior density and replaces them; on a healthy posterior it changes nothing."""
    g = torch.Generator().manual_seed(6)
    truths = PRIOR.sample((4,), g, table.device)
    means = table.means(truths).double()
    cfg = PosteriorConfig(n_particles=256, restart_at=())
    post = FieldPosterior(table, PRIOR, 4, cfg, g)
    for a in (3, 9, 15, 21, 27, 33):
        act = torch.full((4,), a)
        post.update(act, torch.poisson(means[torch.arange(4), act] * 3000, generator=g), torch.full((4,), 3000.0))
    healthy = post.particles.clone()
    assert not bool(post.restart().any())  # nothing better to find
    wrong = PRIOR.sample((4,), g, table.device)
    post.particles = wrong[:, None].expand(-1, 256, -1).clone() + 1e-7 * torch.randn(4, 256, 3, generator=g)
    post.particles = fold_wedge_t(post.particles)
    post.logw.zero_()
    err_before = (post.mean() - truths).norm(dim=-1)
    replaced = post.restart()
    err_after = (post.mean() - truths).norm(dim=-1)
    healthy_err = (healthy.mean(1) - truths).norm(dim=-1)
    assert bool(replaced.any()) and float(err_after.mean()) < float(err_before.mean())
    # back to the healthy posterior's accuracy (the data are only six blocks: that is ~0.1-0.3 mT)
    assert bool((err_after <= 2 * healthy_err + 50e-6).all())
