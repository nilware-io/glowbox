"""Locking, run records, sealing, blinding and the block schedule (glowbox.analysis.prereg)."""

import numpy as np
import pytest

from glowbox.analysis import prereg as pr
from glowbox.analysis.runlog import RunLog, load_run


def _repo(tmp_path):
    (tmp_path / "plan.md").write_text("# plan\nalpha = 0.05\n")
    (tmp_path / "analysis.py").write_text("ALPHA = 0.05\n")
    (tmp_path / "deviations.md").write_text("# Deviations\n")
    return tmp_path


def test_lock_round_trip_and_verify(tmp_path):
    root = _repo(tmp_path)
    lock = pr.Lock.create("P-1", ["plan.md", root / "analysis.py"], {"a": 1}, "deviations.md", root=root)
    assert set(lock.files) == {"plan.md", "analysis.py"}
    lock.verify(root)
    path = lock.write(root / "locks" / "P-1.lock.yaml")
    again = pr.Lock.load(path)
    assert again.sha256() == lock.sha256() and again.files == lock.files
    again.verify(root)


@pytest.mark.parametrize("victim", ["plan.md", "analysis.py"])
def test_lock_refuses_tampering(tmp_path, victim):
    root = _repo(tmp_path)
    lock = pr.Lock.create("P-1", ["plan.md", "analysis.py"], root=root)
    (root / victim).write_text((root / victim).read_text() + "# one more line\n")
    with pytest.raises(pr.LockMismatchError) as e:
        lock.verify(root)
    assert victim in str(e.value)


def test_lock_refuses_missing_file(tmp_path):
    root = _repo(tmp_path)
    lock = pr.Lock.create("P-1", ["plan.md"], root=root)
    (root / "plan.md").unlink()
    with pytest.raises(pr.LockMismatchError, match="missing"):
        lock.verify(root)


def test_deviations_log_is_append_only(tmp_path):
    root = _repo(tmp_path)
    lock = pr.Lock.create("P-1", ["plan.md"], deviations="deviations.md", root=root)
    with open(root / "deviations.md", "a") as f:
        f.write("\n## 2026-12-01: an entry\n")
    lock.verify(root)  # appending is allowed
    (root / "deviations.md").write_text("# Deviations (edited)\n## 2026-12-01: an entry\n")
    with pytest.raises(pr.LockMismatchError, match="append-only"):
        lock.verify(root)


def test_lock_constants_and_identity(tmp_path):
    root = _repo(tmp_path)
    lock = pr.Lock.create("P-1", ["plan.md"], {"alpha": 0.05, "n": 100}, root=root)
    lock.check_constants({"n": 100, "alpha": 0.05})
    with pytest.raises(pr.LockMismatchError, match="alpha"):
        lock.check_constants({"n": 100, "alpha": 0.1})
    other = pr.Lock.create("P-1", ["plan.md"], {"alpha": 0.05, "n": 101}, root=root)
    assert other.sha256() != lock.sha256()


def test_module_files_follows_glowbox_imports():
    files = {p.name for p in pr.module_files("glowbox.analysis.prereg_analysis_001")}
    assert {"prereg_analysis_001.py", "prereg.py", "runlog.py"} <= files
    calib = {p.as_posix() for p in pr.module_files("glowbox.experiments.prereg_calibration")}
    assert any(p.endswith("glowbox/analysis/arms.py") for p in calib)
    assert any(p.endswith("glowbox/learn/measurement.py") for p in calib)


def test_run_record_refusals(tmp_path):
    root = _repo(tmp_path)
    lock = pr.Lock.create("P-1", ["plan.md"], root=root)
    keys = pr.RunKeys.generate()
    rec = pr.prereg_record(lock, root / "P-1.lock.yaml", keys, "confirmatory", root=root)
    assert "seal" not in str(rec["keys"]).replace("seal_sha256", "")  # commitments only
    assert keys.seal.hex() not in str(rec)
    info = {"prereg": rec, "git_dirty": False}
    pr.check_run_record(info, lock, "confirmatory")
    with pytest.raises(pr.LockMismatchError, match="no prereg record"):
        pr.check_run_record({"git_dirty": False}, lock, "confirmatory")
    with pytest.raises(pr.LockMismatchError, match="clean git tree"):
        pr.check_run_record({**info, "git_dirty": True}, lock, "confirmatory")
    with pytest.raises(pr.LockMismatchError, match="mode"):
        pr.check_run_record(info, lock, "calibration")
    other = pr.Lock.create("P-2", ["plan.md"], root=root)
    with pytest.raises(pr.LockMismatchError, match="different lock"):
        pr.check_run_record(info, other, "confirmatory")
    planted = {**info, "prereg": {**rec, "planted_effect": {"A1": 0.01}}}
    with pytest.raises(pr.LockMismatchError, match="planted"):
        pr.check_run_record(planted, lock, "confirmatory")
    with pytest.raises(ValueError, match="simulation"):
        pr.prereg_record(lock, root / "x", keys, "confirmatory", {"A1": 0.01}, root=root)


def test_runlog_annotate(tmp_path):
    with RunLog("t", config={"a": 1}, seed=1, output_root=tmp_path) as run:
        run.annotate("prereg", {"lock_sha256": "abc", "x": np.float64(1.5)})
        with pytest.raises(ValueError, match="reserved"):
            run.annotate("seed", 3)
    info = load_run(run.dir)
    assert info["prereg"] == {"lock_sha256": "abc", "x": 1.5} and info["seed"] == 1


def test_seal_round_trip_and_tamper():
    key = pr.RunKeys.generate().seal
    s = pr.Sealer(key)
    y = np.arange(12, dtype=np.int32).reshape(4, 3)
    rec = s.seal("0/A3/0", y)
    assert rec.ciphertext != y.tobytes() and s.n_sealed == 1
    np.testing.assert_array_equal(pr.unseal(key, rec), y)
    with pytest.raises(pr.SealError):
        pr.unseal(pr.RunKeys.generate().seal, rec)
    bad = pr.SealedRecord(
        rec.record_id, rec.nonce, rec.dtype, rec.shape, bytes([rec.ciphertext[0] ^ 1]) + rec.ciphertext[1:], rec.tag
    )
    with pytest.raises(pr.SealError):
        pr.unseal(key, bad)
    moved = pr.SealedRecord("0/A3/1", rec.nonce, rec.dtype, rec.shape, rec.ciphertext, rec.tag)
    with pytest.raises(pr.SealError):
        pr.unseal(key, moved)
    # the same plaintext twice gives different ciphertexts (fresh nonce)
    assert s.seal("x", y).ciphertext != s.seal("x", y).ciphertext


def test_keystore_logs_releases(tmp_path):
    ks = pr.KeyStore(tmp_path / "ks")
    keys = pr.RunKeys.generate()
    ks.save("run-1", keys)
    with pytest.raises(FileExistsError):
        ks.save("run-1", keys)
    assert ks.release("run-1", "blind", "analysis committed") == keys.blind
    assert "analysis committed" in (tmp_path / "ks" / "releases.yaml").read_text()


def test_arm_codes_hide_labels():
    arms = ["A1", "A2", "A3", "A3b", "A3L", "A4", "A5"]
    k1, k2 = pr.RunKeys.generate().blind, pr.RunKeys.generate().blind
    c1 = pr.arm_codes(k1, arms)
    assert len(set(c1.values())) == len(arms)
    assert all(len(c) == 4 and c.isalpha() and c.isupper() for c in c1.values())
    assert not set(c1.values()) & set(arms)
    assert pr.arm_codes(k1, arms) == c1  # deterministic given the key
    assert pr.arm_codes(k1, arms[::-1]) == c1  # independent of the arm order
    assert pr.arm_codes(k2, arms) != c1
    commit = pr.sha256_bytes(k1)
    assert pr.unblind(k1, commit, arms) == {c: a for a, c in c1.items()}
    with pytest.raises(pr.LockMismatchError):
        pr.unblind(k2, commit, arms)


def test_arm_codes_are_uniform_over_keys():
    # which arm gets the alphabetically first code is uniform over keys (codes say nothing about arms)
    arms = ["A1", "A2", "A3", "A4"]
    first = [
        min(pr.arm_codes(pr.RunKeys.from_seed(i).blind, arms).items(), key=lambda kv: kv[1])[0] for i in range(800)
    ]
    counts = np.array([first.count(a) for a in arms])
    assert counts.min() > 150 and counts.max() < 250


def test_block_schedule_deterministic_and_random():
    arms = ["A1", "A2", "A3", "A4", "A5"]
    seed = pr.RunKeys.from_seed(1).order
    s1 = pr.block_schedule(seed, arms, 400)
    assert s1.shape == (400, 5)
    np.testing.assert_array_equal(s1, pr.block_schedule(seed, arms, 400))
    assert pr.schedule_sha256(s1) == pr.schedule_sha256(pr.block_schedule(seed, arms, 400))
    assert all(sorted(row) == sorted(arms) for row in s1)  # every block holds each arm once
    # a longer table extends the shorter one (a resumed run continues, never redraws)
    np.testing.assert_array_equal(pr.block_schedule(seed, arms, 50), s1[:50])
    assert pr.schedule_sha256(pr.block_schedule(pr.RunKeys.from_seed(2).order, arms, 400)) != pr.schedule_sha256(s1)
    pos = pr.positions(s1)
    freq = np.array([np.bincount(pos[a], minlength=5) for a in arms])
    assert freq.min() > 50  # every arm in every position, roughly uniformly (expected 80)
