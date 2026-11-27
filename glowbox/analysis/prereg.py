"""Pre-registration machinery: locking, run records, sealing, blinding and the randomised block schedule.

The rules these pieces enforce are in docs/preregistration/README.md. In short: an analysis plan, its analysis
script, every module and config the script reads, and the frozen constants are hashed into a *lock* before any
confirmatory data exist; every run records the lock it ran under; the analysis refuses to start when a locked file
changed, and refuses runs whose record does not match. Outcomes are sealed while the run is going on, arms are
labelled by random codes until the blinded analysis output is committed, and the order of the arms in every block
comes from a seed whose hash was committed before the run.

Pieces:

* Hashing: :func:`sha256_file`, :func:`sha256_bytes`, :func:`module_files` (the ``glowbox`` modules a module
  imports, transitively: everything an analysis script can read).
* :class:`Lock`: create, write, load and verify a lock (files and their sha256, an append-only deviations log pinned
  by its frozen prefix, frozen constants, git commit). :class:`LockMismatchError` lists every problem found.
* Run records: :func:`prereg_record` (what goes into run.yaml under ``prereg``; see
  :meth:`glowbox.analysis.runlog.RunLog.annotate`) and :func:`check_run_record` (the refusals).
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
    statements, including those inside functions), sorted. Paths are absolute; :class:`Lock` stores them relative
    to ``root``."""
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
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                todo += [a.name for a in node.names if a.name.split(".")[0] == package]
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == package:
                todo.append(node.module)
                # ``from glowbox.analysis import prereg``: the names may be submodules
                todo += [f"{node.module}.{a.name}" for a in node.names]
    return sorted(set(seen.values()))


# --- Lock -----------------------------------------------------------------------------------------------------------


class LockMismatchError(RuntimeError):
    """A locked file, constant or record differs from the lock. ``problems`` lists every difference."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("pre-registration lock violated:\n  " + "\n  ".join(self.problems))


@dataclass(frozen=True)
class DeviationsPin:
    """The deviations log is append-only: the lock pins its first ``prefix_bytes`` bytes (the file at freeze time).
    Appending entries keeps the lock valid; editing or deleting anything before them breaks it."""

    path: str
    prefix_bytes: int
    prefix_sha256: str


def _rel(path: Path | str, root: Path) -> str:
    p = Path(path)
    p = p if p.is_absolute() else root / p
    return p.resolve().relative_to(root.resolve()).as_posix()


@dataclass
class Lock:
    """A frozen set of files (sha256 each), an optional deviations pin and frozen constants.

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

    @classmethod
    def create(
        cls,
        name: str,
        files: Iterable[Path | str],
        constants: Mapping[str, Any] | None = None,
        deviations: Path | str | None = None,
        root: Path = REPO_ROOT,
    ) -> Lock:
        hashes = {_rel(p, root): sha256_file(root / _rel(p, root)) for p in files}
        pin = None
        if deviations is not None:
            rel = _rel(deviations, root)
            data = (root / rel).read_bytes()
            pin = DeviationsPin(rel, len(data), sha256_bytes(data))
        return cls(
            name=name,
            files=dict(sorted(hashes.items())),
            constants=json.loads(json.dumps(dict(constants or {}), sort_keys=True)),
            deviations=pin,
            git_commit=git_info(root).commit,
            created_utc=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "created_utc": self.created_utc,
            "git_commit": self.git_commit,
            "files": dict(self.files),
            "constants": self.constants,
        }
        if self.deviations is not None:
            d["deviations"] = {
                "path": self.deviations.path,
                "prefix_bytes": self.deviations.prefix_bytes,
                "prefix_sha256": self.deviations.prefix_sha256,
            }
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Lock:
        dev = d.get("deviations")
        return cls(
            name=str(d["name"]),
            files={str(k): str(v) for k, v in d["files"].items()},
            constants=dict(d.get("constants") or {}),
            deviations=DeviationsPin(str(dev["path"]), int(dev["prefix_bytes"]), str(dev["prefix_sha256"]))
            if dev
            else None,
            git_commit=d.get("git_commit"),
            created_utc=d.get("created_utc"),
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
        if self.deviations is not None:
            p = root / self.deviations.path
            if not p.is_file():
                out.append(f"{self.deviations.path}: missing")
            else:
                data = p.read_bytes()
                head = data[: self.deviations.prefix_bytes]
                if len(head) < self.deviations.prefix_bytes or sha256_bytes(head) != self.deviations.prefix_sha256:
                    out.append(f"{self.deviations.path}: the frozen part was edited (the log is append-only)")
        return out

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


def prereg_record(
    lock: Lock,
    lock_path: Path | str,
    keys: RunKeys,
    mode: str,
    planted_effect: Mapping[str, float] | None = None,
    extra: Mapping[str, Any] | None = None,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """The ``prereg`` section of run.yaml. ``mode`` is ``"confirmatory"``, ``"calibration"`` (the control block that
    fixes the null and the margins) or ``"simulation"`` (stage-S validation). ``planted_effect`` (arm -> size) is
    only allowed in simulation."""
    if mode not in ("confirmatory", "calibration", "simulation"):
        raise ValueError(f"unknown mode {mode!r}")
    if planted_effect and mode != "simulation":
        raise ValueError("a planted effect is only allowed in simulation mode")
    return {
        "name": lock.name,
        "mode": mode,
        "lock_path": _rel(lock_path, root),
        "lock_sha256": lock.sha256(),
        "lock_git_commit": lock.git_commit,
        "files": dict(lock.files),
        "keys": keys.commitments(),
        "planted_effect": dict(planted_effect or {}),
        **dict(extra or {}),
    }


def check_run_record(info: Mapping[str, Any], lock: Lock, mode: str) -> None:
    """Refuse a run (raise :class:`LockMismatchError`) whose run.yaml has no ``prereg`` section, was made under another
    lock or in another mode, carries a planted effect while ``mode`` is confirmatory, or (confirmatory and
    calibration) came from a dirty or unknown git tree."""
    rec = info.get("prereg")
    if not rec:
        raise LockMismatchError(["run has no prereg record"])
    probs = []
    if rec.get("lock_sha256") != lock.sha256():
        probs.append("run was made under a different lock")
    if rec.get("files") != lock.files:
        probs.append("run's locked file hashes differ from the lock")
    if rec.get("mode") != mode:
        probs.append(f"run mode {rec.get('mode')!r} is not {mode!r}")
    if mode == "confirmatory" and rec.get("planted_effect"):
        probs.append("planted effect in a confirmatory run")
    if mode in ("confirmatory", "calibration") and info.get("git_dirty") is not False:
        probs.append("run was not made from a clean git tree")
    for k in ("seal_sha256", "blind_sha256", "order_sha256"):
        if not (rec.get("keys") or {}).get(k):
            probs.append(f"key commitment {k} missing")
    if probs:
        raise LockMismatchError(probs)
