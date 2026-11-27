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


def test_pool_model_matches_the_pool(pool):
    m = pc.PoolModel(pool, 1)
    a = np.array([[0, 5, 99]])
    np.testing.assert_allclose(m.r_model(a, 0.0), pool.r_model_nom[1, a])
    np.testing.assert_allclose(m.r_model(a, pool.detuned_offset_hz), pool.r_model_det[1, a])
    with pytest.raises(ValueError, match="offset"):
        m.r_model(a, 1.0)
    np.testing.assert_allclose(m.mw_hz(a), pool.freqs_hz[a])


def test_within_episode_drift_is_autocorrelated(pool):
    b = 3000
    dev = pc.TableMeasure(pool, np.zeros(b, np.int64), 400, np.random.default_rng(0), None, None, 0.0, None, 0.3, 0.9)
    ws = [np.array(dev._within(b, t)) for t in range(3)]
    assert np.std(ws[0]) == pytest.approx(0.3, rel=0.05)
    assert np.corrcoef(ws[0], ws[1])[0, 1] == pytest.approx(0.9, abs=0.03)
    assert abs(np.corrcoef(ws[2], np.array(dev._within(b, 0)))[0, 1]) < 0.06  # a new episode draws afresh


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_miniature_pipeline(cfg, pool, monkeypatch):
    """The whole pipeline at toy size (M_CAL = 30, N = 20, T = 12): model runs, calibration, confirmatory run,
    unblinding; a planted effect is found and attributed."""
    monkeypatch.setattr(pa, "M_CAL", 30)
    monkeypatch.setattr(pa, "N_CONF", 20)
    monkeypatch.setattr(pa, "THETA0_SE_MAX_FRACTION", 10.0)  # toy-size model runs
    c = copy.deepcopy(cfg)
    runs = pc.simulate_batch(c, pool, np.arange(16), "model", "nominal", None, 0.0, None, 2)
    assert len(runs) == 16 and {r["field"] for r in runs} == {0, 1}
    models = {f: [pa.selection_model("nominal", [r["table"] for r in runs if r["field"] == f])] for f in (0, 1)}
    assert all(np.isfinite(m[0].theta0) for m in models.values())
    assert set(runs[0]["table"].codes) == set(pc.MODEL_ARMS)
    null = pc.simulate_batch(c, pool, np.arange(6), "null", "nominal", None, 0.0, models, 2)
    assert len(null) == 6
    assert all(r["names_hidden"] for r in null)
    pos = pc.simulate_batch(c, pool, np.arange(6), "positive", "nominal", "decision", 3.0, models, 2)
    assert np.mean([r["H_decision_rej_cal"] for r in pos]) >= 0.8
    assert np.mean([r["identified"] for r in pos]) >= 0.5  # the toy model check (8 model runs per field) is noisy


def test_criteria_rules():
    lo = pc.Criterion("x", (0.06, 0.045, 0.08, 1000), "lo<=x", 0.05)
    assert lo.passed and "PASS" in lo.line()
    assert not pc.Criterion("x", (0.08, 0.065, 0.1, 1000), "lo<=x", 0.05).passed
    assert pc.Criterion("x", (0.88, 0.85, 0.91, 400), "hi>=x", 0.90).passed
    assert not pc.Criterion("x", (0.8, 0.76, 0.84, 400), "hi>=x", 0.90).passed
