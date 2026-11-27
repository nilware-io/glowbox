"""Pre-registration machinery: locking, run records, the run registry, sealing, blinding and the randomised block
schedule.

The rules these pieces enforce are in docs/preregistration/README.md. In short: an analysis plan, its analysis
script, every module (package ``__init__`` files included), config and data file the runs and the analysis read, the
dependency pins and the frozen constants are hashed into a *lock* before any confirmatory data exist; the lock is
committed and tagged; every run is registered before its first measurement and records the lock, the file hashes
recomputed at its start and its git commit; the analysis refuses to start when a locked file changed, and refuses
runs whose record, commit or registration does not match. Outcomes are sealed while the run is going on, arms are
labelled by random codes until the blinded analysis output is committed, and the order of the arms in every block
comes from a seed whose hash was committed before the run.

Pieces:

* Hashing: :func:`sha256_file`, :func:`sha256_bytes`, :func:`module_files` (the ``glowbox`` modules a module
  imports, transitively, with their parent packages: everything an analysis script can execute) and
  :func:`plan_files` (what a plan lock must cover).
* Git: :func:`git_blob_sha256`, :func:`committed_unchanged`, :func:`published_lock_problems` (the lock is committed
  and held by its tag).
* :class:`Lock`: create, write, load and verify a lock (files and their sha256, append-only pins of the deviations
  log and the run registry by their frozen prefixes, frozen constants, git commit). :class:`LockMismatchError`
  lists every problem found.
* Run records: :func:`prereg_record` (what goes into run.yaml under ``prereg``; see
  :meth:`glowbox.analysis.runlog.RunLog.annotate`), :func:`mark_data_start` and :func:`check_run_record` (the
  refusals).
* :class:`Registry`: the append-only list of every run started under a lock and of its later events.
* Keys: :class:`RunKeys` (a sealing key, a blinding key and the order seed, generated at run start; only their
  sha256 commitments are recorded) and :class:`KeyStore` (keys live outside ``outputs/<run-id>/``; every release
  is logged with its time).
* Sealing: :class:`Sealer` / :func:`unseal`: outcome records are encrypted and authenticated before they leave the
  acquisition code. Construction: keystream = SHAKE-256(key || nonce), ciphertext = plaintext XOR keystream,
  tag = HMAC-SHA256(key, nonce || header || ciphertext) (encrypt-then-MAC, standard library only). It protects
  against reading the outcomes by accident or by habit during a run, which is its purpose; it is not meant as a
  defence against an attacker with access to the acquisition process.
* Blinding: :func:`arm_codes` (a random 4-letter code per arm, derived from the blinding key) and
  :func:`unblind` (the inverse, after checking the key against its commitment).
* Schedule: :func:`block_schedule` (one random permutation of the arms per block, from the order seed) and
  :func:`schedule_sha256`.
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import hmac
import json
import secrets
import string
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from glowbox.analysis.runlog import REPO_ROOT, git_info

# --- Hashing --------------------------------------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _module_path(name: str, root: Path) -> Path | None:
    base = root.joinpath(*name.split("."))
    for p in (base.with_suffix(".py"), base / "__init__.py"):
        if p.is_file():
            return p
    return None


def module_files(module: str, root: Path = REPO_ROOT, package: str = "glowbox") -> list[Path]:
    """The source files of ``module`` and of every ``package`` module it imports, transitively (static ``import``
    statements, including those inside functions), sorted. The ``__init__.py`` of every parent package is included
    too: importing ``a.b.c`` runs ``a/__init__.py`` and ``a/b/__init__.py`` first. Paths are absolute;
    :class:`Lock` stores them relative to ``root``."""
    seen: dict[str, Path] = {}
    todo = [module]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        path = _module_path(name, root)
        if path is None:
            continue
        seen[name] = path
        parts = name.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                todo += [a.name for a in node.names if a.name.split(".")[0] == package]
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == package:
                todo.append(node.module)
                # ``from glowbox.analysis import prereg``: the names may be submodules
                todo += [f"{node.module}.{a.name}" for a in node.names]
    return sorted(set(seen.values()))


#: Files every lock must contain besides the plan's own: the dependency pins (numpy and scipy versions decide the
#: bootstrap streams and quantile rules) and the project definition.
ENVIRONMENT_FILES = ("pyproject.toml", "uv.lock")


def plan_files(
    plan_doc: Path | str,
    modules: Iterable[str],
    configs: Iterable[Path | str] = (),
    extra: Iterable[Path | str] = (),
    root: Path = REPO_ROOT,
) -> list[Path]:
    """Everything a plan lock must cover: the plan text, the module closure (:func:`module_files`, package
    ``__init__`` files included) of every module the runs and the analysis execute (the analysis script, the arms and
    protocol runner, the acquisition or runner script, the controller, the measurement, the frozen model), every
    config they read, any other data file (e.g. a sham distribution), and :data:`ENVIRONMENT_FILES`. Sorted,
    absolute, without duplicates."""
    out: set[Path] = {(root / _rel(plan_doc, root)).resolve()}
    for m in modules:
        out.update(p.resolve() for p in module_files(m, root))
    for p in (*configs, *extra):
        out.add((root / _rel(p, root)).resolve())
    for name in ENVIRONMENT_FILES:
        if (root / name).is_file():
            out.add((root / name).resolve())
    return sorted(out)


# --- Git ------------------------------------------------------------------------------------------------------------


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)


def git_blob_sha256(commit: str, rel: str, root: Path = REPO_ROOT) -> str | None:
    """sha256 of file ``rel`` as committed in ``commit`` (None when it is not in that commit or git fails)."""
    r = _git(root, "show", f"{commit}:{rel}")
    return sha256_bytes(r.stdout) if r.returncode == 0 else None


def committed_unchanged(path: Path | str, root: Path = REPO_ROOT) -> list[str]:
    """Problems unless ``path`` is tracked by git and identical to its committed version at HEAD."""
    rel = _rel(path, root)
    if _git(root, "ls-files", "--error-unmatch", rel).returncode != 0:
        return [f"{rel}: not committed"]
    if _git(root, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        return [f"{rel}: differs from its committed version"]
    return []


def published_lock_problems(lock_path: Path | str, tag: str, root: Path = REPO_ROOT) -> list[str]:
    """Problems unless the lock file is committed unchanged and the tag ``tag`` exists and holds the same bytes.
    (The tag pushed to the public repository is the timestamp; pushing is a step of the README, which git cannot
    verify offline.)"""
    rel = _rel(lock_path, root)
    probs = committed_unchanged(lock_path, root)
    r = _git(root, "show", f"refs/tags/{tag}:{rel}")
    if r.returncode != 0:
        probs.append(f"tag {tag!r} does not exist or does not contain {rel}")
    elif r.stdout != (root / rel).read_bytes():
        probs.append(f"tag {tag!r} holds a different {rel}")
    return probs


# --- Lock -----------------------------------------------------------------------------------------------------------


class LockMismatchError(RuntimeError):
    """A locked file, constant or record differs from the lock. ``problems`` lists every difference."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("pre-registration lock violated:\n  " + "\n  ".join(self.problems))


@dataclass(frozen=True)
class DeviationsPin:
    """An append-only file (the deviations log, the run registry): the lock pins its first ``prefix_bytes`` bytes
    (the file at freeze time). Appending entries keeps the lock valid; editing or deleting anything before them
    breaks it."""

    path: str
    prefix_bytes: int
    prefix_sha256: str


def _rel(path: Path | str, root: Path) -> str:
    p = Path(path)
    p = p if p.is_absolute() else root / p
    return p.resolve().relative_to(root.resolve()).as_posix()


@dataclass
class Lock:
    """A frozen set of files (sha256 each), optional append-only pins (deviations log, run registry) and frozen
    constants.

    ``files`` maps repository-relative POSIX paths to sha256. ``constants`` is any YAML-serialisable mapping (the
    plan's frozen numbers). ``git_commit`` is the commit the lock was created at (informative: the hashes are what is
    checked). The lock's own identity is :meth:`sha256` of its canonical YAML.
    """

    name: str
    files: dict[str, str]
    constants: dict[str, Any] = field(default_factory=dict)
    deviations: DeviationsPin | None = None
    git_commit: str | None = None
    created_utc: str | None = None
    registry: DeviationsPin | None = None

    @classmethod
    def create(
        cls,
        name: str,
        files: Iterable[Path | str],
        constants: Mapping[str, Any] | None = None,
        deviations: Path | str | None = None,
        root: Path = REPO_ROOT,
        registry: Path | str | None = None,
    ) -> Lock:
        hashes = {_rel(p, root): sha256_file(root / _rel(p, root)) for p in files}

        def pin(path: Path | str | None) -> DeviationsPin | None:
            if path is None:
                return None
            rel = _rel(path, root)
            data = (root / rel).read_bytes()
            return DeviationsPin(rel, len(data), sha256_bytes(data))

        return cls(
            name=name,
            files=dict(sorted(hashes.items())),
            constants=json.loads(json.dumps(dict(constants or {}), sort_keys=True)),
            deviations=pin(deviations),
            git_commit=git_info(root).commit,
            created_utc=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            registry=pin(registry),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "created_utc": self.created_utc,
            "git_commit": self.git_commit,
            "files": dict(self.files),
            "constants": self.constants,
        }
        for key in ("deviations", "registry"):
            pin = getattr(self, key)
            if pin is not None:
                d[key] = {"path": pin.path, "prefix_bytes": pin.prefix_bytes, "prefix_sha256": pin.prefix_sha256}
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Lock:
        def pin(v: Mapping[str, Any] | None) -> DeviationsPin | None:
            return DeviationsPin(str(v["path"]), int(v["prefix_bytes"]), str(v["prefix_sha256"])) if v else None

        return cls(
            name=str(d["name"]),
            files={str(k): str(v) for k, v in d["files"].items()},
            constants=dict(d.get("constants") or {}),
            deviations=pin(d.get("deviations")),
            git_commit=d.get("git_commit"),
            created_utc=d.get("created_utc"),
            registry=pin(d.get("registry")),
        )

    def canonical(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()

    def sha256(self) -> str:
        return sha256_bytes(self.canonical())

    def write(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"# Pre-registration lock {self.name}: sha256 {self.sha256()}. Do not edit; make a new version.\n")
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        return p

    @classmethod
    def load(cls, path: Path | str) -> Lock:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def problems(self, root: Path = REPO_ROOT) -> list[str]:
        """Every difference between the lock and the files under ``root`` (empty when the lock holds)."""
        out = []
        for rel, want in self.files.items():
            p = root / rel
            if not p.is_file():
                out.append(f"{rel}: missing")
            elif (got := sha256_file(p)) != want:
                out.append(f"{rel}: sha256 {got[:12]}... differs from the locked {want[:12]}...")
        for pin in (self.deviations, self.registry):
            if pin is None:
                continue
            p = root / pin.path
            if not p.is_file():
                out.append(f"{pin.path}: missing")
            else:
                head = p.read_bytes()[: pin.prefix_bytes]
                if len(head) < pin.prefix_bytes or sha256_bytes(head) != pin.prefix_sha256:
                    out.append(f"{pin.path}: the frozen part was edited (the log is append-only)")
        return out

    def current_hashes(self, root: Path = REPO_ROOT) -> dict[str, str]:
        """The locked files' sha256 recomputed from disk now (missing files are left out)."""
        return {rel: sha256_file(root / rel) for rel in self.files if (root / rel).is_file()}

    def verify(self, root: Path = REPO_ROOT) -> None:
        """Raise :class:`LockMismatchError` unless every locked file is unchanged and the deviations log only grew."""
        if probs := self.problems(root):
            raise LockMismatchError(probs)

    def check_constants(self, constants: Mapping[str, Any]) -> None:
        """Raise when ``constants`` differ from the frozen ones (keys and values, after a JSON round trip)."""
        got = json.loads(json.dumps(dict(constants), sort_keys=True))
        if got != self.constants:
            keys = sorted(set(got) | set(self.constants))
            diff = [k for k in keys if got.get(k) != self.constants.get(k)]
            msgs = [f"constant {k!r}: {got.get(k)!r} != locked {self.constants.get(k)!r}" for k in diff]
            raise LockMismatchError(msgs)


# --- Keys, sealing --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunKeys:
    """Per-run secrets, generated at run start. ``seal`` encrypts outcomes (released when the run has ended),
    ``blind`` derives the arm codes (released only after the blinded analysis output is committed), ``order`` seeds
    the block schedule (released with ``blind``). Only :meth:`commitments` go into run.yaml."""

    seal: bytes
    blind: bytes
    order: bytes

    @classmethod
    def generate(cls) -> RunKeys:
        return cls(secrets.token_bytes(32), secrets.token_bytes(32), secrets.token_bytes(32))

    @classmethod
    def from_seed(cls, seed: int | str) -> RunKeys:
        """Deterministic keys for *simulated* replicates and tests only (never for a confirmatory run)."""
        base = hashlib.sha256(f"glowbox-prereg-test-keys:{seed}".encode()).digest()
        return cls(*(hashlib.sha256(base + tag).digest() for tag in (b"seal", b"blind", b"order")))

    def commitments(self) -> dict[str, str]:
        return {
            "seal_sha256": sha256_bytes(self.seal),
            "blind_sha256": sha256_bytes(self.blind),
            "order_sha256": sha256_bytes(self.order),
        }


class KeyStore:
    """Keys of runs, kept outside the run's output directory (default ``<repo>/keystore/``, gitignored). Every
    release is appended to ``releases.yaml`` with its UTC time, so the order "analysis committed, then key released"
    can be audited."""

    def __init__(self, root: Path | str = REPO_ROOT / "keystore") -> None:
        self.root = Path(root)

    def save(self, run_id: str, keys: RunKeys) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.root / f"{run_id}.keys.json"
        if p.exists():
            raise FileExistsError(f"keys of {run_id} already stored")
        p.write_text(json.dumps({"seal": keys.seal.hex(), "blind": keys.blind.hex(), "order": keys.order.hex()}))
        return p

    def release(self, run_id: str, which: str, reason: str) -> bytes:
        if which not in ("seal", "blind", "order"):
            raise ValueError(f"unknown key {which!r}")
        d = json.loads((self.root / f"{run_id}.keys.json").read_text())
        utc = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        entry = {"run_id": run_id, "key": which, "reason": reason, "utc": utc}
        with open(self.root / "releases.yaml", "a", encoding="utf-8") as f:
            yaml.safe_dump([entry], f, sort_keys=False)
        return bytes.fromhex(d[which])


class SealError(RuntimeError):
    """A sealed record failed authentication (wrong key, or the record was altered)."""


@dataclass(frozen=True)
class SealedRecord:
    record_id: str
    nonce: bytes
    dtype: str
    shape: tuple[int, ...]
    ciphertext: bytes
    tag: bytes

    def header(self) -> bytes:
        return json.dumps([self.record_id, self.dtype, list(self.shape)]).encode()


def _keystream(key: bytes, nonce: bytes, n: int) -> np.ndarray:
    return np.frombuffer(hashlib.shake_256(b"glowbox-seal" + key + nonce).digest(n), dtype=np.uint8)


class Sealer:
    """Encrypts and authenticates outcome arrays with one key (see the module docstring). ``n_sealed`` counts the
    records written."""

    def __init__(self, key: bytes) -> None:
        if len(key) < 16:
            raise ValueError("sealing key too short")
        self._key = key
        self.n_sealed = 0

    @property
    def key_sha256(self) -> str:
        return sha256_bytes(self._key)

    def seal(self, record_id: str, array: np.ndarray) -> SealedRecord:
        a = np.ascontiguousarray(array)
        raw = np.frombuffer(a.tobytes(), dtype=np.uint8)
        nonce = secrets.token_bytes(16)
        ct = (raw ^ _keystream(self._key, nonce, raw.size)).tobytes()
        rec = SealedRecord(record_id, nonce, a.dtype.str, tuple(a.shape), ct, b"")
        tag = hmac.new(self._key, nonce + rec.header() + ct, hashlib.sha256).digest()
        self.n_sealed += 1
        return SealedRecord(record_id, nonce, a.dtype.str, tuple(a.shape), ct, tag)


def unseal(key: bytes, rec: SealedRecord) -> np.ndarray:
    """Decrypt a :class:`SealedRecord`; raises :class:`SealError` when the tag does not verify."""
    want = hmac.new(key, rec.nonce + rec.header() + rec.ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(want, rec.tag):
        raise SealError(f"record {rec.record_id!r} failed authentication")
    raw = np.frombuffer(rec.ciphertext, dtype=np.uint8) ^ _keystream(key, rec.nonce, len(rec.ciphertext))
    return np.frombuffer(raw.tobytes(), dtype=np.dtype(rec.dtype)).reshape(rec.shape).copy()


# --- Blinding -------------------------------------------------------------------------------------------------------


def arm_codes(blind_key: bytes, arms: Sequence[str]) -> dict[str, str]:
    """A distinct random 4-letter code per arm, derived from the blinding key (HMAC-SHA256 of the arm name).
    Without the key the codes carry no information about the arms (they do not depend on the arm order)."""
    letters = string.ascii_uppercase
    out: dict[str, str] = {}
    used: set[str] = set()
    for arm in arms:
        ctr = 0
        while True:
            d = hmac.new(blind_key, f"{arm}#{ctr}".encode(), hashlib.sha256).digest()
            code = "".join(letters[b % 26] for b in d[:4])
            if code not in used:
                break
            ctr += 1
        out[arm] = code
        used.add(code)
    return out


def unblind(blind_key: bytes, commitment: str, arms: Sequence[str]) -> dict[str, str]:
    """code -> arm, after checking the released key against the commitment recorded at run start."""
    if sha256_bytes(blind_key) != commitment:
        raise LockMismatchError(["released blinding key does not match its commitment"])
    return {c: a for a, c in arm_codes(blind_key, arms).items()}


# --- Schedule -------------------------------------------------------------------------------------------------------


def block_schedule(order_seed: bytes, arms: Sequence[str], n_blocks: int) -> np.ndarray:
    """(n_blocks, n_arms) array of arm names: row k is the order in which block k runs its arms, a uniformly random
    permutation drawn from ``order_seed`` alone. The whole table exists before any data; a resumed run continues
    at the next row and never redraws."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(b"glowbox-schedule" + order_seed).digest(), "big"))
    names = np.array(list(arms), dtype=object)
    return np.stack([names[rng.permutation(len(names))] for _ in range(n_blocks)]) if n_blocks else names[None][:0]


def schedule_sha256(schedule: np.ndarray) -> str:
    return sha256_bytes(json.dumps(schedule.tolist()).encode())


def positions(schedule: np.ndarray) -> dict[str, np.ndarray]:
    """arm -> (n_blocks,) position of that arm inside each block."""
    return {str(a): np.argmax(schedule == a, axis=1) for a in np.unique(schedule)}


# --- Run records ----------------------------------------------------------------------------------------------------

#: Modes whose runs count for a plan: they must come from a clean git tree whose commit holds exactly the locked
#: files, and they must be registered (:class:`Registry`) before their first data. ``model``: runs of the frozen
#: physics, drift and timing model that fix a model prediction (PREREG-001: theta0_D) before the calibration block.
STRICT_MODES = ("confirmatory", "calibration", "model")
MODES = (*STRICT_MODES, "simulation")


def prereg_record(
    lock: Lock,
    lock_path: Path | str,
    keys: RunKeys,
    mode: str,
    planted_effect: Mapping[str, float] | None = None,
    extra: Mapping[str, Any] | None = None,
    root: Path = REPO_ROOT,
    *,
    series: str | None = None,
    schedule_sha256: str | None = None,
    calibration_lock: Lock | None = None,
) -> dict[str, Any]:
    """The ``prereg`` section of run.yaml, made at run start. ``mode`` is one of :data:`MODES`: ``"confirmatory"``,
    ``"calibration"`` (the control block that fixes the null and the margins), ``"model"`` (model runs before the
    calibration block) or ``"simulation"`` (stage-S validation). ``planted_effect`` (arm -> size) is only allowed in
    simulation.

    The lock is verified first (raises :class:`LockMismatchError`), and the record holds the locked files' hashes
    *recomputed from disk* and the git commit the run starts from, so :func:`check_run_record` can check what ran.
    ``series`` names the run series (one calibration block per series), ``schedule_sha256`` the sha256 of the
    run's order table, ``calibration_lock`` (confirmatory runs) the calibration the run is analysed against.
    The acquisition code adds ``data_started_utc`` (:func:`mark_data_start`) just before its first measurement."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if planted_effect and mode != "simulation":
        raise ValueError("a planted effect is only allowed in simulation mode")
    if mode == "confirmatory" and calibration_lock is None:
        raise ValueError("a confirmatory run needs the calibration lock it will be analysed against")
    lock.verify(root)
    git = git_info(root)
    rec: dict[str, Any] = {
        "name": lock.name,
        "mode": mode,
        "lock_path": _rel(lock_path, root),
        "lock_sha256": lock.sha256(),
        "lock_git_commit": lock.git_commit,
        "files": lock.current_hashes(root),
        "git_commit": git.commit,
        "git_dirty": git.dirty,
        "keys": keys.commitments(),
        "planted_effect": dict(planted_effect or {}),
        "series": series,
        "schedule_sha256": schedule_sha256,
    }
    if calibration_lock is not None:
        rec["calibration_lock_sha256"] = calibration_lock.sha256()
    rec.update(dict(extra or {}))
    return rec


def mark_data_start(rec: dict[str, Any]) -> dict[str, Any]:
    """Record the UTC time of the first measurement in a prereg record (after the run was registered)."""
    rec["data_started_utc"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    return rec


def check_run_record(info: Mapping[str, Any], lock: Lock, mode: str, root: Path = REPO_ROOT) -> None:
    """Refuse a run (raise :class:`LockMismatchError`) whose run.yaml has no ``prereg`` section, was made under another
    lock or in another mode, carries a planted effect outside simulation, recorded file hashes at run start that
    differ from the lock, or (:data:`STRICT_MODES`) came from a dirty or unknown git tree, from a commit that does not
    hold exactly the locked files, or lacks its series, schedule hash or data-start time."""
    rec = info.get("prereg")
    if not rec:
        raise LockMismatchError(["run has no prereg record"])
    probs = []
    if rec.get("lock_sha256") != lock.sha256():
        probs.append("run was made under a different lock")
    if rec.get("files") != lock.files:
        probs.append("run's locked file hashes (recomputed at run start) differ from the lock")
    if rec.get("mode") != mode:
        probs.append(f"run mode {rec.get('mode')!r} is not {mode!r}")
    if mode != "simulation" and rec.get("planted_effect"):
        probs.append(f"planted effect in a {mode} run")
    for k in ("seal_sha256", "blind_sha256", "order_sha256"):
        if not (rec.get("keys") or {}).get(k):
            probs.append(f"key commitment {k} missing")
    if mode in STRICT_MODES:
        if info.get("git_dirty") is not False or rec.get("git_dirty") is not False:
            probs.append("run was not made from a clean git tree")
        commit = rec.get("git_commit")
        if not commit or commit != info.get("git_commit"):
            probs.append("run's git commit is unknown or differs between run.yaml and its prereg record")
        else:
            bad = [rel for rel, h in lock.files.items() if git_blob_sha256(commit, rel, root) != h]
            if bad:
                probs.append(f"run's commit {commit[:12]} does not hold the locked version of: {', '.join(bad)}")
        for k in ("series", "schedule_sha256", "data_started_utc"):
            if not rec.get(k):
                probs.append(f"run record has no {k}")
        if mode == "confirmatory" and not rec.get("calibration_lock_sha256"):
            probs.append("confirmatory run does not name its calibration lock")
    if probs:
        raise LockMismatchError(probs)


# --- Run registry ---------------------------------------------------------------------------------------------------


class Registry:
    """The append-only run registry of a plan (a YAML list, one item per event), pinned by the plan lock like the
    deviations log. Every run in a :data:`STRICT_MODES` mode is registered with :meth:`register_start` *before its
    first measurement*, and the entry is committed and pushed; later events (``calibration``: the calibration lock
    frozen; ``blinded``: the blinded analysis sha256 committed; ``end`` / ``abort``) are appended the same way.

    The analysis refuses an unregistered run, a second calibration block in a series, model runs that are not all
    the registered ones, and a blinded analysis whose sha256 is not the registered one; the report lists every
    registered run under the lock, completed or aborted."""

    EVENTS = ("start", "calibration", "blinded", "end", "abort")

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        with open(self.path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return list(data or [])

    def append(self, event: str, run_id: str, **fields: Any) -> dict[str, Any]:
        if event not in self.EVENTS:
            raise ValueError(f"unknown registry event {event!r}")
        entry = {"event": event, "run_id": run_id, "utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
        entry.update(json.loads(json.dumps(fields, sort_keys=True)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            yaml.safe_dump([entry], f, sort_keys=False)
        return entry

    def register_start(self, run_id: str, rec: Mapping[str, Any]) -> dict[str, Any]:
        """Register a run from its prereg record (call before the first measurement, then commit and push)."""
        if rec.get("data_started_utc"):
            raise ValueError("register the run before its first measurement")
        if run_id in {e["run_id"] for e in self.entries() if e["event"] == "start"}:
            raise ValueError(f"run {run_id} is already registered")
        return self.append(
            "start",
            run_id,
            plan=rec["name"],
            lock_sha256=rec["lock_sha256"],
            mode=rec["mode"],
            series=rec.get("series"),
            keys=dict(rec["keys"]),
            schedule_sha256=rec.get("schedule_sha256"),
        )

    def events(self, run_id: str, event: str) -> list[dict[str, Any]]:
        return [e for e in self.entries() if e["run_id"] == run_id and e["event"] == event]

    def runs(self, lock_sha256: str, mode: str | None = None, series: str | None = None) -> list[dict[str, Any]]:
        """Start entries under a lock (optionally of one mode and series), in registry order."""
        return [
            e
            for e in self.entries()
            if e["event"] == "start"
            and e.get("lock_sha256") == lock_sha256
            and (mode is None or e.get("mode") == mode)
            and (series is None or e.get("series") == series)
        ]

    def run_problems(self, run_id: str, info: Mapping[str, Any], lock: Lock) -> list[str]:
        """Problems unless the run is registered exactly once under ``lock``, before its first measurement, with
        the keys, schedule, mode and series of its record; and, for a calibration run, unless it is the only
        calibration block of its series."""
        rec = info.get("prereg") or {}
        starts = [e for e in self.runs(lock.sha256()) if e["run_id"] == run_id]
        if len(starts) != 1:
            return [f"run {run_id} is registered {len(starts)} times under this lock (needs exactly 1)"]
        e = starts[0]
        probs = []
        for k in ("mode", "series", "schedule_sha256"):
            if e.get(k) != rec.get(k):
                probs.append(f"registry {k} {e.get(k)!r} differs from the run's {rec.get(k)!r}")
        if e.get("keys") != rec.get("keys"):
            probs.append("registered key commitments differ from the run's")
        started = rec.get("data_started_utc")
        if not started or str(e["utc"]) > str(started):
            probs.append("run was registered after its first measurement")
        if rec.get("mode") == "calibration":
            n_cal = len(self.runs(lock.sha256(), "calibration", rec.get("series")))
            if n_cal != 1:
                probs.append(f"series {rec.get('series')!r} has {n_cal} registered calibration blocks (needs 1)")
        return probs

    def summary(self, lock_sha256: str) -> list[dict[str, Any]]:
        """Every registered run under the lock with its later events (for the report: all runs are reported)."""
        out = []
        for e in self.runs(lock_sha256):
            later = [x["event"] for x in self.entries() if x["run_id"] == e["run_id"] and x["event"] != "start"]
            out.append({"run_id": e["run_id"], "mode": e["mode"], "series": e.get("series"), "events": later})
        return out
