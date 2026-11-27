"""The frozen PREREG-001 analysis (glowbox.analysis.prereg_analysis_001) on synthetic episode tables, and the locked
chain lock -> model runs -> calibration -> blinded analysis -> unblinding on a scratch git repository."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from glowbox.analysis import arms as am
from glowbox.analysis import prereg as pr
from glowbox.analysis import prereg_analysis_001 as pa

T = 20


def _table(arms_hash, n_blocks, rng, shift=None, codes=None, sd=0.05, invalid=None, ar=0.0):
    """arms_hash: arm -> lag of the sequence it replays (None = unique sequence). ``ar``: within-episode AR(1)
    correlation of the per-step noise of every arm (0: iid)."""
    shift = shift or {}
    rows = {k: [] for k in ("block", "pos", "code", "hash", "r", "rm", "off", "task")}
    for k in range(n_blocks):
        for p, (arm, lag) in enumerate(arms_hash.items()):
            rows["block"].append(k)
            rows["pos"].append(p)
            rows["code"].append((codes or {}).get(arm, arm))
            rows["hash"].append(f"seq{k - lag}" if lag is not None else f"{arm}-{k}")
            e = rng.normal(0, sd, T)
            if ar:
                for t in range(1, T):
                    e[t] = ar * e[t - 1] + np.sqrt(1 - ar * ar) * e[t]
            rows["r"].append(1.0 + shift.get(arm, 0.0) + e)
            rows["rm"].append(np.full(T, 1.0))
            rows["off"].append(np.arange(T) % 2 == 0)
            rows["task"].append(float(rng.normal(5.0, 1.0)) + (20.0 if arm in ("A5", "C5") else 0.0))
    return pa.EpisodeTable(
        np.array(rows["block"]),
        np.array(rows["pos"]),
        np.array(rows["code"], dtype=object),
        np.array(rows["hash"], dtype=object),
        np.stack(rows["r"]),
        np.stack(rows["rm"]),
        np.stack(rows["off"]),
        np.array(rows["task"]),
        dict(invalid or {}),
    )


CAL = {"C1": 0, "C2": 1, "C3": 1, "C3b": 1, "C3L": 2, "C4": None, "C5": 1}
CONF = {"A1": 0, "A2": 1, "A3": 1, "A3b": 1, "A3L": 2, "A4": None, "A5": 1}
MODEL = {"A1": 0, "A2": 1, "A3": 1, "A3b": 1}
WARM = {0: "warm-up", 1: "warm-up"}


def _model_tables(n=12, seed=500, shift=None, ar=0.0):
    rng = np.random.default_rng(seed)
    return [_table(MODEL, pa.N_CONF + 2, rng, shift=shift, invalid=WARM, ar=ar) for _ in range(n)]


def _model(name="nominal", **kw):
    return pa.selection_model(name, _model_tables(**kw))


def _cal(seed=0, n=pa.M_CAL + 2, models=None):
    t = _table(CAL, n, np.random.default_rng(seed), shift={"C5": 0.05}, invalid=WARM)
    return pa.derive_calibration(t, models or [_model()], n_boot=4000)


def test_tost_known_cases():
    null = np.random.default_rng(0).normal(0, 1.0, 100_000)
    q = float(np.quantile(null, 0.975))  # ~1.96
    assert pa.tost(0.0, null, 3.0, 0.025)["equivalent"]
    assert not pa.tost(0.0, null, 1.5, 0.025)["equivalent"]  # interval wider than the margin
    assert pa.tost(3.0 - q - 0.05, null, 3.0, 0.025)["equivalent"]
    assert not pa.tost(3.0 - q + 0.05, null, 3.0, 0.025)["equivalent"]
    assert not pa.tost(-(3.0 - q + 0.05), null, 3.0, 0.025)["equivalent"]
    t = pa.tost(0.5, null, 3.0, 0.025)
    assert t["p_lower"] < 0.025 and t["p_upper"] < 0.025


def test_empirical_p_and_holm():
    null = np.linspace(-1, 1, 1001)
    assert pa.empirical_p(0.0, null) == pytest.approx(1.0)
    assert pa.empirical_p(2.0, null) == pytest.approx(1 / 1002)
    assert pa.empirical_p(0.5, null) == pytest.approx((1 + 501) / 1002, abs=2e-3)
    assert pa.holm({"a": 0.01, "b": 0.04}, 0.05) == {"a": True, "b": True}
    assert pa.holm({"a": 0.03, "b": 0.04}, 0.05) == {"a": False, "b": False}
    assert pa.holm({"a": 0.02, "b": 0.30}, 0.05) == {"a": True, "b": False}
    assert pa.holm({"a": float("nan"), "b": 0.01}, 0.05) == {"a": False, "b": True}


def test_conclusion_table():
    assert pa.conclusion(True, False) == "effect detected"
    assert pa.conclusion(True, True).startswith("significant, but smaller than the margin")
    assert "practically null" not in pa.conclusion(True, True)
    assert pa.conclusion(False, True).startswith("null")
    assert pa.conclusion(False, True, stable=False).startswith("inconclusive")  # no null claim without a stable null
    assert pa.conclusion(False, False) == "inconclusive"
    assert "model check" in pa.conclusion(True, False, model_ok=False)


def test_null_means_symmetric_with_the_right_spread():
    s = np.random.default_rng(0).normal(0.3, 2.0, 400)  # an offset in the controls is not taken as the centre
    nm = pa.null_means(s, 100, 20_000, np.random.default_rng(2))
    assert abs(nm.mean()) < 0.01
    # sign-flipped blocks of 4: the variance of the mean is that of the (circular) window sums of 4, / 25 blocks
    w = np.convolve(np.r_[s, s[:3]], np.ones(4), "valid")
    assert nm.std() == pytest.approx(np.sqrt(np.mean(w**2) / 25) / 4, rel=0.05)
    assert pa.block_length(100) == 4 and pa.block_length(1) == 1


def test_margin_rule_is_the_detectable_effect_at_the_planned_n():
    """The margin rule: Q_{1-alpha/2}(|D|) + Q_{2 MARGIN_POWER - 1}(|D|) of the null of the mean of N_CONF pairs.
    For iid controls of sd s that is (z_{0.9875} + z_{0.975}) s / sqrt(N) ~ 4.20 s / 10, and an effect of that size
    is rejected at alpha / 2 with power MARGIN_POWER."""
    rng = np.random.default_rng(3)
    s = rng.normal(0, 1, 4000)
    m = pa.control_margin(s, pa.N_CONF, seed=1, n_boot=40_000)
    assert m == pytest.approx(4.201 / np.sqrt(pa.N_CONF), rel=0.04)
    null = pa.null_means(s, pa.N_CONF, 40_000, np.random.default_rng(2))
    crit = np.quantile(np.abs(null), 1 - pa.ALPHA / 2)
    est = m + rng.normal(0, 1, 20_000) / np.sqrt(pa.N_CONF)
    assert np.mean(np.abs(est) > crit) == pytest.approx(pa.MARGIN_POWER, abs=0.01)


def test_pairing_by_hash_lag_and_exclusions():
    t = _table(CONF, 6, np.random.default_rng(0), invalid={0: "warm-up", 1: "warm-up", 4: "laser fault"})
    ix, iy = pa.pair_rows(t, "A1", "A2", 1)
    blocks = [(int(t.block[i]), int(t.block[j])) for i, j in zip(ix, iy, strict=True)]
    assert blocks == [(2, 3)]  # (1,2) warm-up, (3,4) and (4,5) touch the excluded block
    assert all(t.seq_hash[i] == t.seq_hash[j] for i, j in zip(ix, iy, strict=True))
    ix, iy = pa.pair_rows(t, "A2", "A3", 0)
    assert [int(t.block[i]) for i in ix] == [2, 3, 5]
    assert pa.pair_rows(t, "A1", "A3", 0)[0].size == 0  # different sequences in the same block
    assert pa.pair_rows(t, "A4", "A2", 0, "block")[0].size == 3


def test_pair_statistics():
    t = _table(CONF, 4, np.random.default_rng(0))
    ix, iy = pa.pair_rows(t, "A2", "A3", 0)
    np.testing.assert_allclose(pa.pair_stat(t, ix, iy, "r"), (t.r[ix] - t.r[iy]).mean(1))
    off = pa.pair_stat(t, ix, iy, "r_off")
    np.testing.assert_allclose(off, (t.r[ix] - t.r[iy])[:, ::2].mean(1))
    np.testing.assert_allclose(pa.pair_stat(t, ix, iy, "task"), t.task_error[ix] - t.task_error[iy])
    np.testing.assert_allclose(pa.pair_stat(t, ix, iy, "wcorr"), pa.pair_stat(t, iy, ix, "wcorr"))  # symmetric
    with pytest.raises(ValueError):
        pa.pair_stat(t, ix, iy, "nope")


def test_within_episode_corr():
    rng = np.random.default_rng(0)
    iid = rng.normal(size=(2000, 100))
    assert pa.within_episode_corr(iid).mean() == pytest.approx(-1 / 100, abs=0.003)
    ar = iid.copy()
    for t in range(1, 100):
        ar[:, t] = 0.8 * ar[:, t - 1] + 0.6 * iid[:, t]
    assert pa.within_episode_corr(ar).mean() > 0.4


class _Model:
    lines_hz = np.array([2.80e9, 2.90e9])
    line_width_hz = 1e6

    def mw_hz(self, actions):
        return 2.79e9 + 1e6 * np.asarray(actions, dtype=float)

    def r_model(self, actions, offset_hz):
        return np.full(np.shape(actions), 1.0 if offset_hz == 0 else 1.2)


def test_build_episode_table_rules_and_exclusions():
    """The reduction from counts to the analysed table is locked code now (it lived in the stage-S script)."""
    rng = np.random.default_rng(0)
    codes = ["AAAA", "BBBB"]
    n_blocks, t_n = 6, 8
    rows = []
    for k in range(n_blocks):
        for i, c in enumerate(codes):
            y = rng.poisson(1000, size=(t_n, 3)).astype(float)
            y[3, 0] = 500  # the lowest r is at step 3
            if k == 4 and i == 1:
                y[2, pa.REFERENCE_WINDOW] = pa.REF_COUNT_FLOOR - 1
            if not (k == 5 and c == "BBBB"):  # block 5 misses an arm
                rows.append((c, k, i, np.arange(t_n) + 5 * i, 0.0 if i == 0 else 4e8, y))
    code, block, pos, acts, offs, cnts = (list(x) for x in zip(*rows, strict=True))
    t = pa.build_episode_table(code, block, pos, np.stack(acts), offs, np.stack(cnts), _Model(), flags={3: "interlock"})
    assert t.invalid_blocks == {
        0: "warm-up",
        1: "warm-up",
        3: "interlock",
        4: "reference count below the floor",
        5: "incomplete block",
    }
    c = np.stack(cnts)
    np.testing.assert_allclose(t.r, c[..., pa.SIGNAL_WINDOW] / c[..., pa.REFERENCE_WINDOW])
    assert np.all(t.r_model[t.code == "AAAA"] == 1.0) and np.all(t.r_model[t.code == "BBBB"] == 1.2)
    # step 3 of A is at 2.793 GHz: 7 MHz from the line at 2.8 GHz; off-line = more than 3 line widths from a line
    assert t.task_error[0] == pytest.approx(7.0)
    np.testing.assert_array_equal(t.off_line[0], np.abs(2.79e9 + 1e6 * np.arange(t_n) - 2.8e9) > 3e6)
    assert np.all(t.off_line[1])  # +400 MHz is far from every line
    assert t.seq_hash[0] == t.seq_hash[2] and t.seq_hash[0] != t.seq_hash[1]


def test_calibration_margins_and_models():
    cal = _cal()
    s1 = cal.series[(1, "r")]
    assert s1.size == pa.M_CAL - 1 and cal.series[(0, "r")].size == pa.M_CAL
    # pair sd = sqrt(2) * 0.05 / sqrt(T); the margin is ~4.20 x that / sqrt(N_CONF), well below one pair's spread
    pair_sd = np.sqrt(2) * 0.05 / np.sqrt(T)
    assert cal.sigma[0] == pytest.approx(pair_sd, rel=0.15)
    assert cal.margin[0] == pytest.approx(4.201 * pair_sd / np.sqrt(pa.N_CONF), rel=0.2)
    assert cal.margin[0] < 0.5 * np.quantile(np.abs(cal.series[(0, "r")]), 0.95)
    assert cal.c_ref == pytest.approx(0.05, abs=0.005)
    assert cal.n_blocks == pa.N_CONF and cal.model_ok
    again = pa.Calibration.from_dict(json.loads(json.dumps(cal.to_dict())))
    assert again.sha256() == cal.sha256()


def test_theta0_se_rule_is_enforced():
    """Regression: theta0_D and its SE were free command-line numbers; now they come from model runs and an SE above
    margin / 10 is refused."""
    noisy = pa.SelectionModel("noisy", 0.0, 1.0, -0.05, 0.01, pa.N_CONF, 0.001, 3)
    with pytest.raises(ValueError, match="more model runs"):
        _cal(models=[noisy])
    with pytest.raises(ValueError, match="empty"):
        pa.derive_calibration(_table(CAL, 10, np.random.default_rng(0), invalid=WARM), [])


def _analysed(shift, seed=0, cal=None, sd=0.05, ar=0.0):
    cal = cal or _cal(seed)
    keys = pr.RunKeys.from_seed(seed)
    codes = pr.arm_codes(keys.blind, pa.CONFIRMATORY_ARMS)
    rng = np.random.default_rng(seed + 100)
    t = _table(CONF, pa.N_CONF + 2, rng, shift=shift, codes=codes, invalid=WARM, sd=sd, ar=ar)
    ba = pa.analyse(t, cal, n_boot=2000)
    mapping = pr.unblind(keys.blind, pr.sha256_bytes(keys.blind), pa.CONFIRMATORY_ARMS)
    return ba, pa.unblind(ba, mapping, cal), cal


def test_null_analysis_and_blinding():
    ba, rep, cal = _analysed({})
    text = json.dumps(ba.to_dict(), default=float)
    for arm in pa.CONFIRMATORY_ARMS:
        assert f'"{arm}"' not in text  # blinded output carries codes only
    for h in ("H_decision", "H_observer"):
        r = rep.primary[h]
        assert not r["missing"] and r["n"] >= pa.N_CONF - 1
        assert r["equivalent"] and r["conclusion"].startswith("null")
        assert r["margin_over_c_ref"] == pytest.approx(r["margin"] / cal.c_ref)
    assert rep.stability["stable"] and rep.model_check["calibration"] and rep.model_check["confirmatory"]
    assert rep.model_check["passed"] and rep.model_check["consistent"] == ["nominal"]
    assert rep.manipulation["S4_quantum_test"]["passed"]  # the detuned arm's task error is 20 worse
    assert rep.manipulation["S4_quantum_test"]["mean"] == pytest.approx(20.0, abs=0.5)
    assert "S1_sham_vs_yoked" not in rep.secondary and "S4_quantum_test" not in rep.secondary
    assert next(e for e in rep.exploratory if e["name"] == "S1_sham_vs_yoked")["stat"] == "resid (block)"
    assert not any(v["rejected"] for v in rep.secondary.values())
    md = rep.to_markdown()
    assert "H_decision" in md and "inconclusive" not in md.split("## Secondary")[0]
    assert "margin / C_ref" in md


def test_planted_decision_effect_is_found_and_attributed():
    cal = _cal()
    _, rep, _ = _analysed({"A1": 1.5 * float(cal.margin[1])}, cal=cal)
    r = rep.primary["H_decision"]
    assert r["rejected"] and r["holm_rejected"] and r["concurrent_rejected"]
    assert r["d"] == pytest.approx(1.5 * cal.margin[1], rel=0.35)
    assert r["conclusion"] == "effect detected"
    assert not rep.primary["H_observer"]["rejected"]


def test_planted_observer_effect_is_found_and_attributed():
    cal = _cal()
    observed = {a: 1.5 * float(cal.margin[0]) for a in ("A1", "A2", "A4", "A5")}
    _, rep, _ = _analysed(observed, cal=cal)
    assert rep.primary["H_observer"]["rejected"]
    assert rep.primary["H_observer"]["d"] == pytest.approx(1.5 * cal.margin[0], rel=0.35)
    assert not rep.primary["H_decision"]["rejected"]
    assert rep.stability["stable"]  # the sealed controls are untouched


def test_unstable_null_blocks_null_claims():
    """Regression: a failed null-stability check had no consequence. Now no null claim comes from such a run."""
    _, rep, _ = _analysed({"A3b": 0.05})
    assert not rep.stability["stable"]
    for h in ("H_decision", "H_observer"):
        assert not rep.primary[h]["conclusion"].startswith("null")


def test_concurrent_null_stops_claims_from_a_noisier_run():
    """Regression: a confirmatory run noisier than its calibration block made the frozen null too narrow. A
    positive claim now needs a rejection against the run's own sealed controls as well."""
    cal = _cal()
    hits_cal, hits = 0, 0
    for seed in range(20):
        _, rep, _ = _analysed({}, seed=seed + 10, cal=cal, sd=0.15)
        r = rep.primary["H_observer"]
        hits_cal += r["holm_rejected"]
        hits += r["rejected"]
    # three times the noise: the calibration null alone rejects in about 40 % of runs (at Holm's first level)
    assert hits_cal >= 4
    assert hits <= 1


def test_lag0_orientation_does_not_depend_on_codes():
    cal = _cal()
    shift = {"A2": 0.01, "A1": 0.01, "A4": 0.01, "A5": 0.01}
    ds = []
    for seed in range(3):  # different keys: the stored orientation of the A2/A3 cell differs
        keys = pr.RunKeys.from_seed(seed)
        codes = pr.arm_codes(keys.blind, pa.CONFIRMATORY_ARMS)
        t = _table(CONF, pa.N_CONF + 2, np.random.default_rng(7), shift=shift, codes=codes, invalid=WARM)
        ba = pa.analyse(t, cal, n_boot=500)
        rep = pa.unblind(ba, {c: a for a, c in codes.items()}, cal)
        ds.append(rep.primary["H_observer"]["d"])
        assert rep.model_check["confirmatory"]  # wcorr has no orientation
    assert ds[0] > 0 and np.allclose(ds, ds[0])


def test_theta0_set_worst_case():
    """H_decision is tested at every model's centre and counts only if it rejects at all of them."""
    sel = _model("selection", shift={"A1": 0.004})
    assert sel.theta0 == pytest.approx(0.004, abs=0.001)
    cal = _cal(models=[sel])
    _, rep, _ = _analysed({"A1": 0.004}, cal=cal)
    r = rep.primary["H_decision"]
    assert r["d"] == pytest.approx(r["mean"] - sel.theta0)
    assert not r["rejected"]  # the selection effect the model predicts is not an effect
    both = _cal(models=[_model(), sel])
    _, rep2, _ = _analysed({"A1": 0.004}, cal=both)
    r2 = rep2.primary["H_decision"]
    assert r2["centre"] == "theta0:selection" and not r2["rejected"]  # rejects at theta0 = 0 only: no claim
    _, rep3, _ = _analysed({}, cal=both)
    assert rep3.primary["H_decision"]["centre"] == "theta0:nominal"


def test_model_check_refuses_h_decision_when_the_drift_model_is_wrong():
    """Regression: autocorrelated within-episode drift missing from the model made H_decision liberal. The model
    check compares the sealed controls' within-episode correlation with the model set's prediction."""
    _, rep, _ = _analysed({}, ar=0.6)  # the device has within-episode correlation, the model set does not
    assert not rep.model_check["confirmatory"]
    assert rep.primary["H_decision"]["conclusion"].startswith("inconclusive: the drift model")
    assert not rep.primary["H_decision"]["rejected"]
    assert not rep.primary["H_observer"]["conclusion"].startswith("inconclusive: the drift")  # model-free
    covered = _cal(models=[_model("ar", ar=0.6, n=40)])
    assert not covered.model_ok  # the calibration block (iid) is outside the AR model's envelope
    _, rep2, _ = _analysed({}, ar=0.6, cal=covered)
    assert rep2.model_check["confirmatory"]
    # a member must reproduce both blocks: the AR member matches the confirmatory run only, so no member is left
    assert not rep2.model_check["passed"] and rep2.primary["H_decision"]["conclusion"].startswith("inconclusive")


def test_model_rule_drops_members_the_controls_refute():
    """Regression: the model check passed when the observed wcorr lay in ANY member's envelope, and the worst case
    was taken over every member. A member that the sealed controls refute is now dropped before the worst case;
    the worst case is over the members that reproduce both the calibration block and the confirmatory run."""
    ar = _model("ar", ar=0.6, shift={"A1": 0.02}, n=40)  # refuted by iid controls; its theta0 would hide everything
    cal = _cal(models=[_model(), ar])
    assert cal.model_ok_members == {"nominal": True, "ar": False}
    cm = _cal()
    _, rep, _ = _analysed({"A1": 2.0 * float(cm.margin[1])}, cal=cal)
    r = rep.primary["H_decision"]
    assert rep.model_check["consistent"] == ["nominal"] and r["centre"] == "theta0:nominal"
    assert r["rejected"]  # the refuted member's theta0 (0.02) does not mask the effect


def test_wcorr_matched_member_with_a_wrong_theta0():
    """A member whose drift matches the controls (wcorr) but whose selection effect is wrong (theta0 = 0) stays in
    the set. With the right member in the set too, the worst case over both keeps the test at its level; the plan's
    model set must contain the right member (stage S, condition drift_wrong_member and drift_only_wrong_member)."""
    right = _model("right", shift={"A1": 0.004})
    wrong = _model("wrong")
    _, rep, _ = _analysed({"A1": 0.004}, cal=_cal(models=[right, wrong]))
    r = rep.primary["H_decision"]
    assert sorted(rep.model_check["consistent"]) == ["right", "wrong"]
    assert r["centre"] == "theta0:right" and not r["rejected"]


def test_decision_secondaries_are_centred_at_their_own_theta0():
    """Regression: S2 / S3 of the decision cell were centred at 0 although the decision process selects on the same
    noise (the stage-S secondary family was liberal). Each decision statistic now has its theta0_D per member."""
    sel = _model("selection", shift={"A1": 0.004})
    assert sel.centre("r_off")[0] == pytest.approx(0.004, abs=0.001)
    cal = _cal(models=[sel])
    _, rep, _ = _analysed({"A1": 0.004}, cal=cal)
    s3 = rep.secondary["S3_offline_decision"]
    assert s3["centre"] == "theta0:selection" and not s3["rejected"]


def test_block_length_follows_the_correlation_time():
    """The bootstrap block length is the Newey-West rule for uncorrelated pairs and grows with the correlation time
    of the control series (drift that correlates the pairs over several blocks)."""
    rng = np.random.default_rng(0)
    iid = rng.normal(size=400)
    assert pa.block_length_for(iid) == pa.block_length(400)
    ar = np.empty(400)
    ar[0] = 0.0
    for t in range(1, 400):
        ar[t] = 0.9 * ar[t - 1] + rng.normal()
    assert pa.correlation_time(ar) > 8
    assert pa.block_length_for(ar) >= 16
    nm = pa.null_means(ar, 100, 4000, np.random.default_rng(1))
    means = [ar[i : i + 100].mean() for i in range(0, 300)]
    assert nm.std() > 0.6 * np.std(means)  # NW blocks of 5 would give about a third of the spread


def test_series_claim_needs_every_series():
    """Regression: several series could be started under one lock and two agreeing ones picked. A claim now needs
    its rule in S1 and S2, and an aborted, missing or extra series means no claim."""
    _, null_rep, cal = _analysed({})
    _, pos_rep, _ = _analysed({"A1": 1.5 * float(cal.margin[1])}, cal=cal)
    _, pos2, _ = _analysed({"A1": 1.5 * float(cal.margin[1])}, seed=3, cal=cal)
    c = pa.series_claim({"S1": pos_rep, "S2": pos2})
    assert c["H_decision"].startswith("effect detected") and c["H_observer"].startswith("null")
    assert pa.series_claim({"S1": pos_rep, "S2": null_rep})["H_decision"].startswith("no claim")
    assert pa.series_claim({"S1": pos_rep, "S2": None})["H_decision"].startswith("no claim")
    assert pa.series_claim({"S1": pos_rep})["H_decision"].startswith("no claim")
    assert pa.series_claim({"S1": pos_rep, "S2": pos2, "S3": pos2})["H_decision"].startswith("no claim")


def test_paired_margin_test_fixed_sizes_and_calibration():
    """Regression: the M7 / E5 test bootstrapped a handful of control diffs as if they were the population and had
    an open sample size (5-10 seeds: false-positive rate 0.12-0.14). Now: fixed sizes, a permutation test that
    carries the controls' uncertainty, a margin from the controls only."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="exactly"):
        pa.paired_margin_test(rng.normal(size=5), rng.normal(size=pa.ENG_N_CONTROL))
    with pytest.raises(ValueError, match="exactly"):
        pa.paired_margin_test(rng.normal(size=pa.ENG_N), rng.normal(size=pa.ENG_N_CONTROL + 1))
    rej, power = [], []
    for i in range(300):
        c = rng.normal(size=pa.ENG_N_CONTROL)
        d = rng.normal(size=pa.ENG_N)
        t = pa.paired_margin_test(d, c, n_perm=400, seed=i)
        rej.append(t["rejected"])
        power.append(pa.paired_margin_test(d + t["margin"], c, n_perm=400, seed=i)["rejected"])
    assert np.mean(rej) < 0.09 and np.mean(power) > 0.85
    one = pa.paired_margin_test(rng.normal(1.5, 1, pa.ENG_N), rng.normal(size=pa.ENG_N_CONTROL), "greater")
    assert one["rejected"] and not one["tost"]["equivalent"]
    neg = pa.paired_margin_test(rng.normal(1.5, 1, pa.ENG_N), rng.normal(size=pa.ENG_N_CONTROL), "less")
    assert not neg["rejected"]


def test_episode_table_round_trip(tmp_path):
    t = _table(CONF, 3, np.random.default_rng(0), invalid={0: "warm-up"})
    t2 = pa.EpisodeTable.load(t.save(tmp_path / "episodes.parquet"))
    assert t2.sha256() == t.sha256() and t2.invalid_blocks == {0: "warm-up"}


def test_verify_lock_refuses():
    good = pa.create_plan_lock()
    pa.verify_lock(good)
    assert {"glowbox/__init__.py", "glowbox/analysis/__init__.py", "uv.lock", "pyproject.toml"} <= set(good.files)
    with pytest.raises(pr.LockMismatchError, match="frozen constants"):
        pa.verify_lock(pr.Lock.from_dict({**good.to_dict(), "constants": {"analysis": {"ALPHA": 0.1}}}))
    small = pr.Lock.create(pa.PLAN, [pa.SCRIPT_PATH], {"analysis": pa.frozen_constants()})
    with pytest.raises(pr.LockMismatchError, match="not in the lock") as e:
        pa.verify_lock(small)
    assert "glowbox/analysis/__init__.py" in str(e.value) and "uv.lock" in str(e.value)
    tampered = pr.Lock.from_dict({**good.to_dict(), "files": {**good.files, pa.SCRIPT_PATH: "0" * 64}})
    with pytest.raises(pr.LockMismatchError, match="differs"):
        pa.verify_lock(tampered)
    with pytest.raises(pr.LockMismatchError, match="PREREG-002"):
        pa.verify_lock(pr.Lock.from_dict({**good.to_dict(), "name": "PREREG-002"}))
    no_reg = pr.Lock.from_dict({k: v for k, v in good.to_dict().items() if k != "registry"})
    with pytest.raises(pr.LockMismatchError, match="registry"):
        pa.verify_lock(no_reg)


# --- The locked chain on a scratch git repository --------------------------------------------------------------------

FUTURE = "2099-01-01T00:00:{:02d}+00:00"  # data-start times after every registry entry of the test
REF = 10_000.0
DETUNED = 4e8
CHAIN_MODEL = """import numpy as np


class _Model:
    lines_hz = np.array([2.80e9, 2.90e9])
    line_width_hz = 1e6

    def mw_hz(self, actions):
        return 2.79e9 + 1e6 * np.asarray(actions, dtype=float)

    def r_model(self, actions, offset_hz):
        return np.full(np.shape(actions), 1.0 if offset_hz == 0 else 1.2)


def make(record):
    return _Model()
"""
CHAIN_MODEL_PATH = "prereg_models/chain_model.py"


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def _commit(root, msg):
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "--allow-empty", "-m", msg)


def _episodes(arms_hash, n_blocks, rng, keys, codes=None, shift=None, sd=0.05, scale=1.0):
    """Sealed episodes (B = 1) of a design: arm -> lag of the sequence it replays (None = its own sequence). The
    counts are REF reference photons and r x REF signal photons per step, r = 1 + shift + noise (scaled by
    ``scale`` around 1: an edit of the deviations)."""
    shift = shift or {}
    sealer, meta = pr.Sealer(keys.seal), pr.Sealer(keys.blind)
    names = list(arms_hash)
    sch = pr.block_schedule(keys.order, names, n_blocks)
    eps = []
    for k in range(n_blocks):
        for arm in names:
            lag = arms_hash[arm]
            seq_rng = np.random.default_rng(10_000 + k - lag) if lag is not None else rng
            acts = seq_rng.integers(0, 20, T)
            r = 1.0 + scale * (shift.get(arm, 0.0) + (0.05 if arm in ("C5", "A5") else 0.0) + rng.normal(0, sd, T))
            y = np.stack([r * REF, np.full(T, 5000.0), np.full(T, REF)], axis=1)
            label = (codes or {}).get(arm, arm)
            sealed = [sealer.seal(f"{k}/{label}/{t}", y[t][None, :]) for t in range(T)]
            m = meta.seal(f"{k}/{label}/meta", np.zeros(4, np.uint8))
            off = np.array([DETUNED if arm in ("C5", "A5") else 0.0])
            setting = sealer.seal(f"{k}/{label}/setting", off)
            slot = np.array([k * len(names) + int(np.argmax(sch[k] == arm))])
            eps.append(am.Episode(arm, k, acts[:, None], sealed, m, slot, label, setting))
    return eps


class Chain:
    """A scratch repository with the plan, the package, a frozen model file, a published plan lock, registered
    model, calibration and confirmatory runs (sealed artifacts, end entries), and a helper for each step."""

    def __init__(self, root: Path, model_set=None) -> None:
        self.root = root
        src = pr.REPO_ROOT
        shutil.copytree(src / "glowbox", root / "glowbox", ignore=shutil.ignore_patterns("__pycache__"))
        for f in ("pyproject.toml", "uv.lock", ".gitignore", pa.PLAN_DOC, pa.DEVIATIONS_PATH, pa.REGISTRY_PATH):
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / f, root / f)
        (root / CHAIN_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
        (root / CHAIN_MODEL_PATH).write_text(CHAIN_MODEL)
        _git(root, "init", "-q")
        _git(root, "config", "user.email", "test@example.invalid")
        _git(root, "config", "user.name", "test")
        _commit(root, "init")
        self.model_set = model_set or {"nominal": {}}
        self.lock = pa.create_plan_lock(
            root=root, model_set=self.model_set, frozen_model=f"{CHAIN_MODEL_PATH}:make"
        )  # fmt: skip
        self.lock_path = self.lock.write(root / "docs/preregistration/PREREG-001.lock.yaml")
        _commit(root, "lock")
        _git(root, "tag", pa.LOCK_TAG)
        self.registry = pr.Registry(root / pa.REGISTRY_PATH, root)
        self.keystore = pr.KeyStore(root / "keystore")
        self.clock = 0
        self.state: dict = {}

    def run(self, run_id, mode, eps, series="S1", calibration_lock=None, register=True, end=True, flags=None,
            commit_start=True, **extra):  # fmt: skip
        keys = pr.RunKeys.from_seed(run_id)
        rec = pr.prereg_record(
            self.lock, self.lock_path, keys, mode, root=self.root, series=series, schedule_sha256="cd" * 32,
            calibration_lock=calibration_lock, extra=extra,
        )  # fmt: skip
        if register:
            self.registry.register_start(run_id, rec)
            if commit_start:
                _commit(self.root, f"register {run_id}")
        self.keystore.save(run_id, keys)
        self.clock += 1
        rec["data_started_utc"] = FUTURE.format(self.clock)
        d = self.root / "outputs" / run_id
        d.mkdir(parents=True)
        info = {"run_id": run_id, "git_commit": rec["git_commit"], "git_dirty": False, "prereg": rec}
        (d / "run.yaml").write_text(yaml.safe_dump(info, sort_keys=False))
        w = am.ArtifactWriter(d / "sealed.jsonl")
        for e in eps:
            w(e)
        if register and end:
            self.registry.register_end(run_id, d, ["sealed.jsonl"], flags)
            _commit(self.root, f"end {run_id}")
        return d, keys

    def models(self, n=10, seed=500, name="nominal", params=None, start=0):
        rng = np.random.default_rng(seed)
        out = []
        for i in range(start, start + n):
            rid = f"model-{name}-{i}"
            eps = _episodes(MODEL, pa.N_CONF + 2, rng, pr.RunKeys.from_seed(rid))
            out.append(self.run(rid, "model", eps, model=name, model_params=params or {})[0])
        return {name: out}

    def calibration(self, seed=0, run_id="cal-1", series="S1", scale=1.0, **kw):
        keys = pr.RunKeys.from_seed(run_id)
        eps = _episodes(CAL, pa.M_CAL + 2, np.random.default_rng(seed), keys, scale=scale)
        return self.run(run_id, "calibration", eps, series=series, **kw)[0]

    def confirmatory(self, cal_lock, shift=None, run_id="conf-1", **kw):
        keys = pr.RunKeys.from_seed(run_id)
        codes = pr.arm_codes(keys.blind, pa.CONFIRMATORY_ARMS)
        eps = _episodes(CONF, pa.N_CONF + 2, np.random.default_rng(42), keys, codes=codes, shift=shift)
        return self.run(run_id, "confirmatory", eps, calibration_lock=cal_lock, **kw)

    def cal_dir_of(self, series="S1"):
        return self.root / f"docs/preregistration/calibration/{series}"

    @property
    def cal_dir(self):
        return self.cal_dir_of()

    @property
    def cal_lock_path(self):
        return self.cal_dir / "calibration.lock.yaml"

    def freeze(self, cal_dir, models, series="S1"):
        cl = pa.freeze_calibration(cal_dir, models, self.lock_path, self.cal_dir_of(series), self.root, n_boot=500,
                                   keys=self.keystore)  # fmt: skip
        _commit(self.root, "calibration")
        return cl

    def analyse(self, conf_dir, cal_dir, out=None, cal_lock_path=None):
        out = out or self.cal_dir / "blinded.json"
        ba = pa.analyse_run(conf_dir, cal_dir, self.lock_path, cal_lock_path or self.cal_lock_path, out, self.root,
                            n_boot=500, keys=self.keystore)  # fmt: skip
        _commit(self.root, "blinded")
        return ba, out

    def unblind(self, analysis, conf_dir, keys):
        out = self.root / "report.md"
        return pa.unblind_run(analysis, conf_dir, self.lock_path, self.cal_lock_path, keys.blind, out, self.root,
                              keys=self.keystore)  # fmt: skip

    def honest_until_calibration(self):
        models = self.models()
        cal_dir = self.calibration()
        return models, cal_dir


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """The honest chain, run once: model runs, calibration, confirmatory run, blinded analysis, unblinding."""
    c = Chain(tmp_path_factory.mktemp("chain"))
    models, cal_dir = c.honest_until_calibration()
    cal_lock = c.freeze(cal_dir, models)
    conf_dir, keys = c.confirmatory(cal_lock)
    ba, analysis = c.analyse(conf_dir, cal_dir)
    rep = c.unblind(analysis, conf_dir, keys)
    c.state = {"models": models, "cal_dir": cal_dir, "cal_lock": cal_lock, "conf_dir": conf_dir, "keys": keys}
    c.state.update({"ba": ba, "analysis": analysis, "rep": rep})
    return c


def test_chain_honest_run(chain):
    s = chain.state
    rep, ba = s["rep"], s["ba"]
    assert ba.lock_sha256 == chain.lock.sha256() and ba.calibration_lock_sha256 == s["cal_lock"].sha256()
    assert ba.script_sha256 == pr.sha256_file(chain.root / pa.SCRIPT_PATH) and ba.run_id == "conf-1"
    assert rep.primary["H_observer"]["conclusion"].startswith("null")
    runs = {r["run_id"]: r for r in rep.registered_runs}
    assert runs["cal-1"]["events"] == ["end", "calibration"] and runs["conf-1"]["events"] == ["end", "blinded"]
    assert len([r for r in runs.values() if r["mode"] == "model"]) == 10
    head = _git(chain.root, "rev-parse", "HEAD")
    assert all(len(c) == 40 for c in runs["conf-1"]["commits"].values()) and head
    text = (chain.root / "report.md").read_text()
    assert "Every registered run" in text and runs["conf-1"]["commits"]["blinded"][:12] in text
    assert json.loads((chain.root / "report.json").read_text())["analysis_sha256"] == rep.analysis_sha256
    rel = [r for r in yaml.safe_load((chain.root / "keystore/releases.yaml").read_text()) if r["run_id"] == "conf-1"]
    assert {r["key"] for r in rel} == {"seal"}  # the blinding key is passed by hand, after the blinded commit


def _expect(match, fn, *args, **kw):
    with pytest.raises(pr.LockMismatchError, match=match):
        fn(*args, **kw)


def test_chain_refuses_tampered_calibration(chain):
    """Regression: `analyse` accepted any calibration JSON (a series shrunk 10x made both primaries reject)."""
    s = chain.state
    cal_file = chain.cal_dir / "calibration.json"
    good = cal_file.read_text()
    d = json.loads(good)
    d["series"] = {k: [x / 10 for x in v] for k, v in d["series"].items()}
    cal_file.write_text(json.dumps(d))
    args = (s["conf_dir"], s["cal_dir"], chain.lock_path, chain.cal_lock_path, chain.root / "x.json", chain.root)
    try:
        _expect("differs", pa.analyse_run, *args, keys=chain.keystore)
        _commit(chain.root, "edited calibration")  # committing the edit does not help
        _expect("differs", pa.analyse_run, *args, keys=chain.keystore)
    finally:
        cal_file.write_text(good)
        _commit(chain.root, "restore")


def test_chain_refuses_tampered_blinded_analysis(chain):
    """Regression: `unblind` accepted a hand-edited blinded analysis (every p set to 1e-4)."""
    s = chain.state
    good = s["analysis"].read_text()
    d = json.loads(good)
    for c in d["cells"]:
        for t in c["tests"].values():
            t["p"] = 1e-4
    s["analysis"].write_text(json.dumps(d))
    _commit(chain.root, "edited analysis")
    try:
        _expect("not the one registered", chain.unblind, s["analysis"], s["conf_dir"], s["keys"])
    finally:
        s["analysis"].write_text(good)
        _commit(chain.root, "restore")
    chain.unblind(s["analysis"], s["conf_dir"], s["keys"])  # the committed original still unblinds


def test_chain_ignores_a_planted_episode_table(chain):
    """Regression: every step loaded `<run>/episodes.parquet` as it found it. The table is now always rebuilt from
    the registered sealed artifacts; a table in the run directory is never read."""
    s = chain.state
    t, _ = pa.run_table(s["conf_dir"], "conf-1", yaml.safe_load((s["conf_dir"] / "run.yaml").read_text())["prereg"],
                        chain.lock, chain.registry, chain.keystore, chain.root)  # fmt: skip
    assert t.sha256() == s["ba"].data_sha256
    fake = pa.EpisodeTable(t.block, t.position, t.code, t.seq_hash, t.r * 0 + 1, t.r_model, t.off_line, t.task_error)
    fake.save(s["conf_dir"] / "episodes.parquet")
    chain.unblind(s["analysis"], s["conf_dir"], s["keys"])  # unaffected


def test_chain_refuses_subset_of_model_runs_and_second_calibration(chain):
    """Regression: theta0_D was a free number; now it is all registered model runs or nothing. And one calibration
    block per series: a second registered calibration block makes the series unusable."""
    s = chain.state
    some = {"nominal": s["models"]["nominal"][:5]}
    out = chain.root / "tmp-cal"
    _expect("not all the registered", pa.freeze_calibration, s["cal_dir"], some, chain.lock_path, out, chain.root,
            keys=chain.keystore)  # fmt: skip
    second = chain.calibration(seed=1, run_id="cal-2", series="S2")
    chain.calibration(seed=2, run_id="cal-3", series="S2")
    _expect("2 registered calibration", pa.freeze_calibration, second, s["models"], chain.lock_path, out, chain.root,
            keys=chain.keystore)  # fmt: skip


def test_chain_refuses_unregistered_and_misbound_confirmatory_runs(chain):
    s = chain.state
    out = chain.root / "x.json"
    d, _ = chain.confirmatory(s["cal_lock"], run_id="conf-unreg", register=False)
    _expect("registered 0 times", pa.analyse_run, d, s["cal_dir"], chain.lock_path, chain.cal_lock_path, out,
            chain.root, keys=chain.keystore)  # fmt: skip
    wrong = pr.Lock.create("PREREG-001-calibration", [chain.cal_dir / "calibration.json"], {"x": 1}, root=chain.root)
    d2, _ = chain.confirmatory(wrong, run_id="conf-other-cal", series="S2")
    _expect("different calibration", pa.analyse_run, d2, s["cal_dir"], chain.lock_path, chain.cal_lock_path, out,
            chain.root, keys=chain.keystore)  # fmt: skip
    _expect("already has a registered blinded", pa.analyse_run, s["conf_dir"], s["cal_dir"], chain.lock_path,
            chain.cal_lock_path, out, chain.root, keys=chain.keystore)  # fmt: skip


def test_chain_refuses_changed_package_init_and_missing_tag(chain):
    """Regression: a monkeypatch in glowbox/analysis/__init__.py changed the analysis and the lock still verified."""
    s = chain.state
    init = chain.root / "glowbox/analysis/__init__.py"
    good = init.read_text()
    init.write_text(good + "import numpy\nnumpy.quantile = lambda *a, **k: 0.0\n")
    try:
        _expect(r"glowbox/analysis/__init__\.py", chain.unblind, s["analysis"], s["conf_dir"], s["keys"])
    finally:
        init.write_text(good)
    _git(chain.root, "tag", "-d", pa.LOCK_TAG)
    try:
        _expect("does not exist", pa.analyse_run, s["conf_dir"], s["cal_dir"], chain.lock_path, chain.cal_lock_path,
                chain.root / "x.json", chain.root, keys=chain.keystore)  # fmt: skip
    finally:
        _git(chain.root, "tag", pa.LOCK_TAG)


# --- Destructive scenarios: each on a fresh chain ---------------------------------------------------------------------


@pytest.fixture
def fresh(tmp_path):
    return Chain(tmp_path / "repo")


def _reseal(run_dir, keys, edit):
    """Rewrite a run's sealed artifacts with edited counts, sealed again with the (released) sealing key: valid
    records, so only the registered file hash can catch it."""
    path = run_dir / "sealed.jsonl"
    eps = am.read_artifacts(path)
    sealer = pr.Sealer(keys.seal)
    lines = []
    for e in eps:
        y = am.unseal_outcomes(e, keys.seal)
        y = edit(e, y)
        e.sealed = [sealer.seal(r.record_id, y[t]) for t, r in enumerate(e.sealed)]
        lines.append(json.dumps(am.sealed_artifact(e)))
    path.write_text("\n".join(lines) + "\n")


def _shrink(e, y):
    r = y[..., pa.SIGNAL_WINDOW] / y[..., pa.REFERENCE_WINDOW]
    y = y.copy()
    y[..., pa.SIGNAL_WINDOW] = (1.0 + (r - 1.0) / 1.5) * y[..., pa.REFERENCE_WINDOW]
    return y


def test_chain_refuses_edited_calibration_and_model_artifacts(fresh):
    """Regression: a calibration table with deviations shrunk 1.5x was accepted and lowered the margin. The table is
    rebuilt from the sealed artifacts whose sha256 was registered (and committed) at the end of the run, before the
    sealing key was released; an edit, even re-sealed with the released key, is refused. The same for a model run
    (theta0_D and the wcorr envelope come from them)."""
    models, cal_dir = fresh.honest_until_calibration()
    keys = pr.RunKeys.from_seed("cal-1")
    _reseal(cal_dir, keys, _shrink)
    out = fresh.cal_dir
    _expect("differs from the one registered", pa.freeze_calibration, cal_dir, models, fresh.lock_path, out,
            fresh.root, keys=fresh.keystore)  # fmt: skip
    _commit(fresh.root, "edited calibration artifacts")  # outputs are not in git; nothing changes
    fresh2 = models["nominal"][3]
    _reseal(fresh2, pr.RunKeys.from_seed("model-nominal-3"), _shrink)
    _expect("model-nominal-3: sealed artifact", pa.freeze_calibration, cal_dir, models, fresh.lock_path, out,
            fresh.root, keys=fresh.keystore)  # fmt: skip


def test_chain_refuses_edited_exclusions(fresh):
    """Regression: exclusions were parquet metadata, editable by hand (blocks 2-39 marked 'interlock' were
    accepted). They are now the acquisition flags of the registered end entry: a second end entry is refused, and
    so is an edit of the committed entry (the registry's git history must only append)."""
    models, cal_dir = fresh.honest_until_calibration()
    cal_lock = fresh.freeze(cal_dir, models)
    conf_dir, _ = fresh.confirmatory(cal_lock, flags={5: "interlock"})
    t, _ = pa.run_table(conf_dir, "conf-1", yaml.safe_load((conf_dir / "run.yaml").read_text())["prereg"],
                        fresh.lock, fresh.registry, fresh.keystore, fresh.root)  # fmt: skip
    assert t.invalid_blocks[5] == "interlock" and 6 not in t.invalid_blocks
    flags = {str(k): "interlock" for k in range(2, 40)}
    fresh.registry.append("end", "conf-1", artifacts={"sealed.jsonl": "0" * 64}, flags=flags)
    _commit(fresh.root, "second end entry")
    _expect("2 end entries", fresh.analyse, conf_dir, cal_dir)
    reg = fresh.root / pa.REGISTRY_PATH
    text = reg.read_text()
    first = text.index("'5': interlock")
    reg.write_text(text[:first] + "'5': interlock\n    '6': interlock\n    '7': interlock" + text[first + 14 :])
    _commit(fresh.root, "edited exclusions")
    _expect("rewrites earlier registry entries", fresh.analyse, conf_dir, cal_dir)
    reg.write_text(text)
    _commit(fresh.root, "restore")
    _expect("rewrites earlier registry entries", fresh.analyse, conf_dir, cal_dir)  # the history keeps the edit


def test_chain_one_confirmatory_run_per_series_and_plan_series_only(fresh):
    """Regression: several confirmatory runs could be analysed and unblinded against one calibration. A second
    registered confirmatory run makes the series unusable (as for calibration blocks), and a series outside S1 / S2
    is refused."""
    models, cal_dir = fresh.honest_until_calibration()
    cal_lock = fresh.freeze(cal_dir, models)
    conf_dir, _ = fresh.confirmatory(cal_lock)
    fresh.confirmatory(cal_lock, run_id="conf-2")
    _expect("2 registered confirmatory", fresh.analyse, conf_dir, cal_dir)
    d3 = fresh.calibration(run_id="cal-x", series="S3")
    _expect("not a series of the plan", pa.freeze_calibration, d3, models, fresh.lock_path, fresh.root / "o",
            fresh.root, keys=fresh.keystore)  # fmt: skip


def test_chain_model_set_is_locked(tmp_path):
    """Regression: any model name with registered runs joined the theta0_D set (40 runs of an invented 'made_up'
    member were accepted). The lock names the members and their parameters; the calibration refuses missing,
    extra and mismatched members."""
    c = Chain(tmp_path / "a", model_set={"nominal": {}, "drift": {"within_sigma": 0.3}})
    models = c.models()
    cal_dir = c.calibration()
    _expect("locked model set is", pa.freeze_calibration, cal_dir, models, c.lock_path, c.root / "o", c.root,
            keys=c.keystore)  # fmt: skip
    drift = c.models(n=10, seed=600, name="drift", params={"within_sigma": 0.1})
    _expect("locked parameters", pa.freeze_calibration, cal_dir, {**models, **drift}, c.lock_path, c.root / "o",
            c.root, keys=c.keystore)  # fmt: skip
    c2 = Chain(tmp_path / "b")
    m2 = c2.models()
    made_up = c2.models(n=3, seed=700, name="made_up")
    cal2 = c2.calibration()
    _expect("outside the locked model set", pa.freeze_calibration, cal2, {**m2, **made_up}, c2.lock_path,
            c2.root / "o", c2.root, keys=c2.keystore)  # fmt: skip


def test_chain_order_is_checked_by_git_ancestry(fresh):
    """Regression: the order checks compared self-reported local timestamps. The commits that added the registry
    entries are now the evidence: a run's end entry must be committed after its start entry, model runs must end
    before the calibration block is registered, and a confirmatory run must be registered after the calibration
    event."""
    models = fresh.models()
    d = fresh.calibration(commit_start=False)  # start and end entries land in the same commit
    _expect("not committed after its start", pa.freeze_calibration, d, models, fresh.lock_path, fresh.root / "o",
            fresh.root, keys=fresh.keystore)  # fmt: skip
    f2 = Chain(fresh.root.parent / "repo2")
    m2 = f2.models(n=9)
    cal = f2.calibration()
    late = f2.models(n=1, seed=900, start=9)  # a model run that ends after the calibration block started
    _expect("did not end before the calibration", pa.freeze_calibration, cal,
            {"nominal": m2["nominal"] + late["nominal"]}, f2.lock_path, f2.root / "o", f2.root,
            keys=f2.keystore)  # fmt: skip
