"""Locking, run records, sealing, blinding and the block schedule (glowbox.analysis.prereg)."""

import subprocess

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
    assert {"prereg_analysis_001.py", "prereg.py", "runlog.py", "arms.py"} <= files
    calib = {p.as_posix() for p in pr.module_files("glowbox.experiments.prereg_calibration")}
    assert any(p.endswith("glowbox/analysis/arms.py") for p in calib)
    assert any(p.endswith("glowbox/learn/measurement.py") for p in calib)


def test_module_files_include_package_inits():
    """Regression: the package __init__ files run on every import (a monkeypatch there changed the analysis) but
    were not in the closure."""
    rel = {p.relative_to(pr.REPO_ROOT).as_posix() for p in pr.module_files("glowbox.analysis.prereg_analysis_001")}
    assert {"glowbox/__init__.py", "glowbox/analysis/__init__.py", "glowbox/sim/__init__.py"} <= rel


def test_plan_files_cover_configs_environment_and_extra_modules():
    files = {
        p.relative_to(pr.REPO_ROOT).as_posix()
        for p in pr.plan_files(
            "docs/preregistration/PREREG-001-observer-decision.md",
            ["glowbox.analysis.prereg_analysis_001", "glowbox.experiments.prereg_calibration"],
            ["glowbox/experiments/configs/prereg_001.yaml"],
        )
    }
    assert {"pyproject.toml", "uv.lock", "glowbox/experiments/configs/prereg_001.yaml"} <= files
    assert "glowbox/experiments/prereg_calibration.py" in files and "glowbox/control/__init__.py" in files


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _git_repo(tmp_path):
    root = _repo(tmp_path)
    (root / ".gitignore").write_text("outputs/\n")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


def _strict_info(root, lock, mode="confirmatory", **kw):
    keys = pr.RunKeys.generate()
    cal = kw.pop("calibration_lock", pr.Lock.create("C", ["plan.md"], root=root) if mode == "confirmatory" else None)
    rec = pr.prereg_record(
        lock, root / "P-1.lock.yaml", keys, mode, root=root, series="S1", schedule_sha256="ab" * 32,
        calibration_lock=cal, **kw,
    )  # fmt: skip
    rec = pr.mark_data_start(rec)
    return {"run_id": "r1", "git_commit": rec["git_commit"], "git_dirty": rec["git_dirty"], "prereg": rec}, keys


def test_run_record_refusals(tmp_path):
    root = _git_repo(tmp_path)
    lock = pr.Lock.create("P-1", ["plan.md"], root=root)
    info, keys = _strict_info(root, lock)
    rec = info["prereg"]
    assert "seal" not in str(rec["keys"]).replace("seal_sha256", "")  # commitments only
    assert keys.seal.hex() not in str(rec)
    pr.check_run_record(info, lock, "confirmatory", root)
    with pytest.raises(pr.LockMismatchError, match="no prereg record"):
        pr.check_run_record({"git_dirty": False}, lock, "confirmatory", root)
    with pytest.raises(pr.LockMismatchError, match="clean git tree"):
        pr.check_run_record({**info, "git_dirty": True}, lock, "confirmatory", root)
    with pytest.raises(pr.LockMismatchError, match="mode"):
        pr.check_run_record(info, lock, "calibration", root)
    other = pr.Lock.create("P-2", ["plan.md"], root=root)
    with pytest.raises(pr.LockMismatchError, match="different lock"):
        pr.check_run_record(info, other, "confirmatory", root)
    planted = {**info, "prereg": {**rec, "planted_effect": {"A1": 0.01}}}
    with pytest.raises(pr.LockMismatchError, match="planted"):
        pr.check_run_record(planted, lock, "confirmatory", root)
    with pytest.raises(ValueError, match="simulation"):
        pr.prereg_record(lock, root / "x", keys, "confirmatory", {"A1": 0.01}, root=root)
    with pytest.raises(ValueError, match="calibration lock"):
        pr.prereg_record(lock, root / "x", keys, "confirmatory", root=root)
    for k, msg in (
        ("series", "no series"),
        ("schedule_sha256", "no schedule"),
        ("data_started_utc", "no data_started"),
        ("calibration_lock_sha256", "calibration lock"),
    ):
        with pytest.raises(pr.LockMismatchError, match=msg):
            pr.check_run_record({**info, "prereg": {**rec, k: None}}, lock, "confirmatory", root)


def test_run_record_recomputes_hashes_and_checks_the_commit(tmp_path):
    """Regression: the run record copied lock.files instead of re-hashing at run start, and the run's commit was
    never compared with the locked hashes."""
    root = _git_repo(tmp_path)
    lock = pr.Lock.create("P-1", ["plan.md", "analysis.py"], root=root)
    (root / "analysis.py").write_text("ALPHA = 0.1\n")
    with pytest.raises(pr.LockMismatchError, match=r"analysis\.py"):
        _strict_info(root, lock)  # the lock is verified at run start
    _git(root, "checkout", "-q", "--", "analysis.py")
    info, _ = _strict_info(root, lock)
    assert info["prereg"]["files"] == lock.files
    pr.check_run_record(info, lock, "confirmatory", root)
    # a run whose commit holds a different version of a locked file is refused, whatever its record says
    (root / "analysis.py").write_text("ALPHA = 0.1\n")
    _git(root, "commit", "-q", "-am", "edit")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    forged = {**info, "git_commit": head, "prereg": {**info["prereg"], "git_commit": head}}
    with pytest.raises(pr.LockMismatchError, match=r"does not hold the locked version of: analysis\.py"):
        pr.check_run_record(forged, lock, "confirmatory", root)


def test_published_lock_needs_commit_and_tag(tmp_path):
    root = _git_repo(tmp_path)
    lock = pr.Lock.create("P-1", ["plan.md"], root=root)
    lp = lock.write(root / "P-1.lock.yaml")
    assert any("not committed" in p for p in pr.published_lock_problems(lp, "p-1", root))
    _git(root, "add", "P-1.lock.yaml")
    _git(root, "commit", "-q", "-m", "lock")
    assert any("does not exist" in p for p in pr.published_lock_problems(lp, "p-1", root))
    _git(root, "tag", "p-1")
    assert pr.published_lock_problems(lp, "p-1", root) == []
    lp.write_text(lp.read_text() + "# edited\n")
    probs = pr.published_lock_problems(lp, "p-1", root)
    assert any("differs from its committed" in p for p in probs) and any("different" in p for p in probs)


def _commit_all(root, msg):
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "--allow-empty", "-m", msg)


def test_registry_rules(tmp_path):
    root = _git_repo(tmp_path)
    (root / "registry.yaml").write_text("# registry\n")
    _commit_all(root, "registry")
    lock = pr.Lock.create("P-1", ["plan.md"], root=root, registry="registry.yaml")
    reg = pr.Registry(root / "registry.yaml")
    assert reg.root.resolve() == root.resolve()
    keys = pr.RunKeys.generate()
    rec = pr.prereg_record(lock, root / "l.yaml", keys, "calibration", root=root, series="S1", schedule_sha256="x")
    reg.register_start("cal-1", rec)
    with pytest.raises(ValueError, match="already registered"):
        reg.register_start("cal-1", rec)
    rec = pr.mark_data_start(dict(rec))
    with pytest.raises(ValueError, match="before its first measurement"):
        reg.register_start("cal-x", rec)
    info = {"prereg": rec}
    assert any("not committed" in p for p in reg.run_problems("cal-1", info, lock))
    _commit_all(root, "register cal-1")
    assert reg.run_problems("cal-1", info, lock) == []
    assert any("registered 0 times" in p for p in reg.run_problems("cal-2", info, lock))
    assert any("key commitments" in p for p in reg.run_problems("cal-1", {"prereg": {**rec, "keys": {}}}, lock))
    assert any(
        "after its first" in p for p in reg.run_problems("cal-1", {"prereg": {**rec, "data_started_utc": "2000"}}, lock)
    )
    # a second calibration block in the same series is refused (no picking the most favourable null)
    rec2 = pr.prereg_record(lock, root / "l.yaml", keys, "calibration", root=root, series="S1", schedule_sha256="x")
    reg.register_start("cal-2", rec2)
    _commit_all(root, "register cal-2")
    assert any("2 registered calibration runs" in p for p in reg.run_problems("cal-1", info, lock))
    reg.append("abort", "cal-2", reason="test")
    _commit_all(root, "abort cal-2")
    summ = reg.summary(lock.sha256())
    assert [r["events"] for r in summ] == [[], ["abort"]]
    assert set(summ[1]["commits"]) == {"start", "abort"} and all(len(c) == 40 for c in summ[1]["commits"].values())
    lock.verify(root)  # appending keeps the lock
    (root / "registry.yaml").write_text("# rewritten\n")
    with pytest.raises(pr.LockMismatchError, match="append-only"):
        lock.verify(root)


def test_registry_end_entry_and_history(tmp_path):
    """The end entry carries the sealed artifacts' sha256 and the acquisition flags; the registry's git history
    must only append whole entries, and names the commit of every entry."""
    root = _git_repo(tmp_path)
    (root / "registry.yaml").write_text("# registry\n")
    _commit_all(root, "registry")
    lock = pr.Lock.create("P-1", ["plan.md"], root=root, registry="registry.yaml")
    reg = pr.Registry(root / "registry.yaml")
    rec = pr.prereg_record(lock, root / "l.yaml", pr.RunKeys.generate(), "model", root=root, series="S1",
                           schedule_sha256="x", extra={"model": "nominal"})  # fmt: skip
    with pytest.raises(ValueError, match="not registered"):
        reg.register_end("m-1", root, ["plan.md"], attempted_blocks=1)
    reg.register_start("m-1", rec)
    _commit_all(root, "start")
    run = root / "outputs" / "m-1"
    run.mkdir(parents=True)
    (run / "sealed.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="artifact"):
        reg.register_end("m-1", run, [], attempted_blocks=1)
    # regression: the attempted-block count was optional (and never checked): the stopping rule had no record
    with pytest.raises(TypeError):
        reg.register_end("m-1", run, ["sealed.jsonl"])  # type: ignore[call-arg]
    for bad in (-1, 1.5, True, None):
        with pytest.raises(ValueError, match="attempted_blocks"):
            reg.register_end("m-1", run, ["sealed.jsonl"], attempted_blocks=bad)  # type: ignore[arg-type]
    end = reg.register_end("m-1", run, ["sealed.jsonl"], {3: "interlock"}, attempted_blocks=12)
    assert end["attempted_blocks"] == 12
    assert end["artifacts"] == {"sealed.jsonl": pr.sha256_file(run / "sealed.jsonl")} and end["flags"] == {
        "3": "interlock"
    }
    with pytest.raises(ValueError, match="already has an end"):
        reg.register_end("m-1", run, ["sealed.jsonl"], attempted_blocks=12)
    probs, owner = reg.history()
    assert owner[-1] is None and any("differs from its last commit" in p for p in probs)
    _commit_all(root, "end")
    probs, owner = reg.history()
    assert probs == [] and owner[0] != owner[1] and pr.is_ancestor(owner[0], owner[1], root, proper=True)
    info = {"prereg": pr.mark_data_start(dict(rec))}
    assert reg.run_problems("m-1", info, lock) == []
    # appending lines to the last entry keeps the bytes before them: still a rewrite of that entry
    text = (root / "registry.yaml").read_text()
    (root / "registry.yaml").write_text(text + "  reason: added later\n")
    _commit_all(root, "extend the last entry")
    assert any("rewrites earlier" in p for p in reg.history()[0])


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
