"""The frozen PREREG-001 analysis (glowbox.analysis.prereg_analysis_001) on synthetic episode tables."""

import json

import numpy as np
import pytest

from glowbox.analysis import prereg as pr
from glowbox.analysis import prereg_analysis_001 as pa

T = 20


def _table(arms_hash, n_blocks, rng, shift=None, codes=None, sd=0.05, invalid=None):
    """arms_hash: arm -> lag of the sequence it replays (None = unique sequence)."""
    shift = shift or {}
    rows = {k: [] for k in ("block", "pos", "code", "hash", "r", "rm", "off", "task")}
    for k in range(n_blocks):
        for p, (arm, lag) in enumerate(arms_hash.items()):
            rows["block"].append(k)
            rows["pos"].append(p)
            rows["code"].append((codes or {}).get(arm, arm))
            rows["hash"].append(f"seq{k - lag}" if lag is not None else f"{arm}-{k}")
            r = 1.0 + shift.get(arm, 0.0) + rng.normal(0, sd, T)
            rows["r"].append(r)
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


def _cal(seed=0, n=202):
    t = _table(CAL, n, np.random.default_rng(seed), shift={"C5": 0.05}, invalid={0: "warm-up", 1: "warm-up"})
    return pa.derive_calibration(t)


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
    assert pa.conclusion(True, True).startswith("detectable but smaller")
    assert pa.conclusion(False, True).startswith("null")
    assert pa.conclusion(False, False) == "inconclusive"


def test_null_means_symmetric_with_the_right_spread():
    s = np.random.default_rng(0).normal(0.3, 2.0, 400)  # an offset in the controls is not taken as the centre
    nm = pa.null_means(s, 100, 20_000, np.random.default_rng(2))
    assert abs(nm.mean()) < 0.01
    # sign-flipped blocks of 4: the variance of the mean is that of the (circular) window sums of 4, / 25 blocks
    w = np.convolve(np.r_[s, s[:3]], np.ones(4), "valid")
    assert nm.std() == pytest.approx(np.sqrt(np.mean(w**2) / 25) / 4, rel=0.05)
    assert pa.block_length(100) == 4 and pa.block_length(1) == 1


def test_control_margin_rule():
    s = np.random.default_rng(3).normal(0, 1, 5000)
    assert pa.control_margin(s, 1) == pytest.approx(np.quantile(np.abs(s), 0.95))
    assert pa.control_margin(s, 25) == pytest.approx(1.96 / 5, rel=0.08)


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
    with pytest.raises(ValueError):
        pa.pair_stat(t, ix, iy, "nope")


def test_calibration_margins_and_n():
    cal = _cal()
    s1 = cal.series[(1, "r")]
    assert s1.size == 199 and cal.series[(0, "r")].size == 200
    assert cal.margin[1] == pytest.approx(np.quantile(np.abs(s1), 0.95))
    # pair sd = sqrt(2) * 0.05 / sqrt(T)
    assert cal.sigma[0] == pytest.approx(np.sqrt(2) * 0.05 / np.sqrt(T), rel=0.15)
    assert cal.c_ref == pytest.approx(0.05, abs=0.005)
    assert cal.n_blocks == pa.N_MIN
    again = pa.Calibration.from_dict(json.loads(json.dumps(cal.to_dict())))
    assert again.sha256() == cal.sha256()


def _analysed(shift, seed=0):
    cal = _cal(seed)
    keys = pr.RunKeys.from_seed(seed)
    codes = pr.arm_codes(keys.blind, pa.CONFIRMATORY_ARMS)
    t = _table(
        CONF, 102, np.random.default_rng(seed + 100), shift=shift, codes=codes, invalid={0: "warm-up", 1: "warm-up"}
    )
    ba = pa.analyse(t, cal, n_boot=2000)
    mapping = pr.unblind(keys.blind, pr.sha256_bytes(keys.blind), pa.CONFIRMATORY_ARMS)
    return ba, pa.unblind(ba, mapping, cal), cal


def test_null_analysis_and_blinding():
    ba, rep, _ = _analysed({})
    text = json.dumps(ba.to_dict(), default=float)
    for arm in pa.CONFIRMATORY_ARMS:
        assert f'"{arm}"' not in text  # blinded output carries codes only
    for h in ("H_decision", "H_observer"):
        r = rep.primary[h]
        assert not r["missing"] and r["n"] >= 99
        assert r["equivalent"] and r["conclusion"].startswith("null")
    assert rep.stability["stable"]
    assert rep.secondary["S4_quantum_test"]["holm_rejected"]  # the detuned arm's task error is 20 worse
    assert rep.secondary["S4_quantum_test"]["mean"] == pytest.approx(20.0, abs=0.5)
    md = rep.to_markdown()
    assert "H_decision" in md and "inconclusive" not in md.split("## Secondary")[0]


def test_planted_decision_effect_is_found_and_attributed():
    cal = _cal()
    _, rep, _ = _analysed({"A1": float(cal.margin[1])})
    assert rep.primary["H_decision"]["holm_rejected"]
    assert rep.primary["H_decision"]["d"] == pytest.approx(cal.margin[1], rel=0.25)
    assert rep.primary["H_decision"]["conclusion"] == "effect detected"
    assert not rep.primary["H_observer"]["holm_rejected"]


def test_planted_observer_effect_is_found_and_attributed():
    cal = _cal()
    observed = {a: float(cal.margin[0]) for a in ("A1", "A2", "A4", "A5")}
    _, rep, _ = _analysed(observed)
    assert rep.primary["H_observer"]["holm_rejected"]
    assert rep.primary["H_observer"]["d"] == pytest.approx(cal.margin[0], rel=0.25)
    assert not rep.primary["H_decision"]["holm_rejected"]
    assert rep.stability["stable"]  # the sealed controls are untouched


def test_unstable_null_is_flagged():
    _, rep, _ = _analysed({"A3b": 0.05})
    assert not rep.stability["stable"]


def test_lag0_orientation_does_not_depend_on_codes():
    cal = _cal()
    shift = {"A2": 0.01, "A1": 0.01, "A4": 0.01, "A5": 0.01}
    ds = []
    for seed in range(3):  # different keys: the stored orientation of the A2/A3 cell differs
        keys = pr.RunKeys.from_seed(seed)
        codes = pr.arm_codes(keys.blind, pa.CONFIRMATORY_ARMS)
        t = _table(CONF, 102, np.random.default_rng(7), shift=shift, codes=codes, invalid={0: "w", 1: "w"})
        ba = pa.analyse(t, cal, n_boot=500)
        rep = pa.unblind(ba, {c: a for a, c in codes.items()}, cal)
        ds.append(rep.primary["H_observer"]["d"])
    assert ds[0] > 0 and np.allclose(ds, ds[0])


def test_theta0_centre_for_decision_cells():
    t = _table(CAL, 202, np.random.default_rng(0), shift={"C5": 0.05}, invalid={0: "w", 1: "w"})
    cal = pa.derive_calibration(t, theta0_d=0.004, theta0_d_se=0.0005)
    keys = pr.RunKeys.from_seed(0)
    codes = pr.arm_codes(keys.blind, pa.CONFIRMATORY_ARMS)
    tab = _table(CONF, 102, np.random.default_rng(5), shift={"A1": 0.004}, codes=codes, invalid={0: "w", 1: "w"})
    rep = pa.unblind(pa.analyse(tab, cal, n_boot=2000), {c: a for a, c in codes.items()}, cal)
    r = rep.primary["H_decision"]
    assert r["d"] == pytest.approx(r["mean"] - 0.004)
    assert not r["holm_rejected"]  # the selection effect the model predicts is not an effect


def test_paired_margin_test():
    rng = np.random.default_rng(0)
    ctrl = rng.normal(0, 1, 400)
    same = pa.paired_margin_test(rng.normal(0, 1, 100), ctrl, n_boot=4000)
    assert same["tost"]["equivalent"] and same["p"] > 0.01
    diff = pa.paired_margin_test(rng.normal(2.5, 1, 100), ctrl, n_boot=4000)
    assert diff["p"] < 0.001 and not diff["tost"]["equivalent"]
    assert same["margin"] == pytest.approx(1.96, rel=0.1)


def test_episode_table_round_trip(tmp_path):
    t = _table(CONF, 3, np.random.default_rng(0), invalid={0: "warm-up"})
    t2 = pa.EpisodeTable.load(t.save(tmp_path / "episodes.parquet"))
    assert t2.sha256() == t.sha256() and t2.invalid_blocks == {0: "warm-up"}


def test_verify_lock_refuses():
    files = [pa.SCRIPT_PATH, *pr.module_files("glowbox.analysis.prereg_analysis_001")]
    good = pr.Lock.create(pa.PLAN, files, {"analysis": pa.frozen_constants()})
    pa.verify_lock(good)
    with pytest.raises(pr.LockMismatchError, match="frozen constants"):
        pa.verify_lock(pr.Lock.create(pa.PLAN, files, {"analysis": {**pa.frozen_constants(), "ALPHA": 0.1}}))
    with pytest.raises(pr.LockMismatchError, match="not in the lock"):
        pa.verify_lock(pr.Lock.create(pa.PLAN, ["glowbox/analysis/prereg.py"], {"analysis": pa.frozen_constants()}))
    tampered = pr.Lock.from_dict({**good.to_dict(), "files": {**good.files, pa.SCRIPT_PATH: "0" * 64}})
    with pytest.raises(pr.LockMismatchError, match="differs"):
        pa.verify_lock(tampered)
    with pytest.raises(pr.LockMismatchError, match="PREREG-002"):
        pa.verify_lock(pr.Lock.from_dict({**good.to_dict(), "name": "PREREG-002"}))


def test_cli_refuses_a_run_without_record(tmp_path):
    files = [pa.SCRIPT_PATH, *pr.module_files("glowbox.analysis.prereg_analysis_001")]
    lock = pr.Lock.create(pa.PLAN, files, {"analysis": pa.frozen_constants()})
    lp = lock.write(tmp_path / "lock.yaml")
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.yaml").write_text("seed: 1\ngit_dirty: false\n")
    _table(CAL, 3, np.random.default_rng(0)).save(run / "episodes.parquet")
    with pytest.raises(pr.LockMismatchError, match="no prereg record"):
        pa.main(["calibrate", "--run", str(run), "--lock", str(lp), "--out", str(tmp_path / "cal.json")])
