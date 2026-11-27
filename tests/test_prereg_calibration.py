"""The PREREG-001 stage-S validation script (glowbox.experiments.prereg_calibration): device, controller, and a
miniature end-to-end run of the whole pipeline."""

import copy

import numpy as np
import pytest

from glowbox.analysis import prereg_analysis_001 as pa
from glowbox.analysis.arms import StepContext
from glowbox.experiments import prereg_calibration as pc
from glowbox.experiments.m1_odmr_rabi import load_config


@pytest.fixture(scope="module")
def cfg():
    c = load_config(pc.DEFAULT_CONFIG, quick=True)
    c["field_pool"] = 2
    c["numerics"]["n_detuning"] = 4
    c["protocol"]["n_steps"] = 12
    c["replicates"].update({"batch": 6, "n_boot": 300, "workers": 2})
    return c


@pytest.fixture(scope="module")
def pool(cfg):
    return pc.build_pool(cfg, 0)


def test_pool_has_spin_contrast_and_lines(pool):
    assert pool.nom.shape == (2, 100, 3) and pool.det.shape == pool.nom.shape
    gap = pool.r_model_det - pool.r_model_nom
    assert gap.max() > 0.03  # dips in the referenced fluorescence on resonance
    assert np.abs(gap[pool.off_nom]).max() < 0.01  # steps marked far from every line show no spin response
    assert pool.off_det.all()  # +400 MHz drives nothing
    assert 1.0 < pool.baseline_r < 1.5


def test_table_measure_plant_and_variants(pool):
    rng = np.random.default_rng(0)
    b = 4000
    fidx = np.zeros(b, dtype=np.int64)
    a = np.full(b, int(np.argmax(pool.r_model_det[0] - pool.r_model_nom[0])))
    ctx = StepContext(0, "A1", "nominal", 0, np.zeros(b, dtype=np.int64))
    dev = pc.TableMeasure(pool, fidx, 400, rng, None, None, 0.0, {"A1": np.full(b, 0.02)})
    y = dev(a, ctx).astype(float)
    r = y[:, pool.sig] / y[:, pool.ref]
    assert r.mean() == pytest.approx(pool.r_model_nom[0, a[0]] + 0.02, abs=2e-3)
    y2 = dev(a, StepContext(0, "A2", "nominal", 0, ctx.slot)).astype(float)
    assert (y2[:, pool.sig] / y2[:, pool.ref]).mean() == pytest.approx(pool.r_model_nom[0, a[0]], abs=2e-3)
    yd = dev(a, StepContext(0, "A5", "detuned", 0, ctx.slot)).astype(float)
    assert (yd[:, pool.sig] / yd[:, pool.ref]).mean() == pytest.approx(pool.r_model_det[0, a[0]], abs=2e-3)


def test_ou_statistics():
    x = pc.ou(np.random.default_rng(0), 200, 400, 0.05, 20.0)
    assert x.std() == pytest.approx(0.05, rel=0.1)
    lag1 = np.mean(x[:, 1:] * x[:, :-1]) / x.var()
    assert lag1 == pytest.approx(np.exp(-1 / 20), abs=0.03)


def test_threshold_controller_moves_finely_in_dips():
    c = pc.ThresholdController(100, 1.1, 1, 4, 0, 2)
    c.reset(2)
    a = c.propose(0)
    np.testing.assert_array_equal(a, [0, 0])
    c.observe(0, np.array([10, 98]), np.array([[100, 0, 100], [120, 0, 100]]))  # r = 1.0 (dip), 1.2 (no dip)
    np.testing.assert_array_equal(c.propose(1), [11, 2])


def test_clopper_pearson():
    lo, hi = pc.clopper_pearson(50, 1000)
    assert lo == pytest.approx(0.0373, abs=1e-3) and hi == pytest.approx(0.0654, abs=1e-3)
    assert pc.clopper_pearson(0, 10)[0] == 0.0


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_miniature_pipeline(cfg, pool, monkeypatch):
    """The whole pipeline at toy size (M_CAL = 30, N = 20, T = 12): runs, unblinds, attributes a planted effect."""
    monkeypatch.setattr(pa, "M_CAL", 30)
    monkeypatch.setattr(pa, "N_MIN", 20)
    c = copy.deepcopy(cfg)
    th = pc.simulate_batch(c, pool, np.arange(4), "theta0", None, 0.0, (0.0, 0.0), 2)
    assert len(th) == 4 and all(np.isfinite(r["theta_d"]) for r in th)
    null = pc.simulate_batch(c, pool, np.arange(6), "null", None, 0.0, (0.0, 0.0), 2)
    assert len(null) == 6
    assert all(r["names_hidden"] for r in null)
    assert all(r["n_blocks"] == 20 for r in null)
    pos = pc.simulate_batch(c, pool, np.arange(6), "positive", "decision", 3.0, (0.0, 0.0), 2)
    assert np.mean([r["H_decision_rej"] for r in pos]) >= 0.8
    assert np.mean([r["identified"] for r in pos]) >= 0.6
