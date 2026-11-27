"""Control arms for decision and observer comparisons: the closed-loop decision arm, yoked replays (no decision),
sealed replays (no observer), sham decisions, and the block-randomised protocol that runs them.

Everything is generic over a *controller* (the decision process) and a *measurement* (anything that turns actions
into outcomes: :class:`SimulatorMeasure` wraps ``Simulator.run()``, so the same code later drives hardware):

* :class:`Controller`: ``reset(batch)``, ``propose(step) -> actions``, ``observe(step, actions, outcomes)``. Batched:
  actions are arrays with a leading batch axis (B = 1 for one physical device; B > 1 for simulated replicates).
* :class:`Measure`: ``measure(actions, ctx) -> outcomes`` (B, W), e.g. photon counts per readout window. The
  :class:`StepContext` names the arm's measurement ``variant`` (``"nominal"``, ``"detuned"``) and the time slot of
  each batch element, so simulated nuisance drift can follow the run's real time order.

Arm kinds (:class:`ArmSpec`), all running the same controller code with the same inputs at the same steps:

* ``decision``: the controller's proposal is applied (an outcome-contingent choice reaches the control).
* ``replay``: the applied actions are a recorded table (the decision arm's actions ``lag`` blocks earlier, or an
  offline table); the controller still runs on the outcomes and its proposals are logged and discarded.
* ``sham``: applied actions are drawn from a fixed distribution (``sham(step, batch)``), proposals discarded.

``observed=False`` makes an arm *sealed*: its outcomes go only to the :class:`~glowbox.analysis.prereg.Sealer`;
the controller and every monitor receive a decoy from a separate generator (:func:`model_decoy`: counts drawn from
the frozen model at the applied action, so the controller's inputs have the structure of real outcomes), and the
episode's count of plaintext deliveries stays 0. Every arm seals every outcome (the same workload in every arm); the
analysis gets the outcomes only by unsealing them after the run (:func:`unseal_outcomes`).

Nothing that tells the arms apart travels in the clear. Outcome records are labelled by the arm's blinding *code*
(or a random token), never by its name. What differs between observed and sealed arms by construction (the count
of plaintext deliveries, the controller's proposals, which in an observed arm are a function of the real outcomes,
the warm-up fallback flag and the controller's compute time per step) is sealed with the *blinding* key
(:class:`EpisodeMeta`), released only after the blinded analysis is committed. :func:`sealed_artifact` is what an
episode writes to disk.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

from glowbox.analysis.prereg import SealedRecord, Sealer, sha256_bytes, unseal
from glowbox.control.pulse import PulseSequence
from glowbox.sim.interface import EnvParams, Simulator


@dataclass(frozen=True)
class StepContext:
    block: int
    arm: str
    variant: str
    step: int
    #: (B,) time slot of the episode per batch element: block * n_arms + position in the block.
    slot: np.ndarray


class Controller(Protocol):
    """A decision process. It is reset at the start of every episode, in every arm."""

    def reset(self, batch: int) -> None: ...

    def propose(self, step: int) -> np.ndarray: ...

    def observe(self, step: int, actions: np.ndarray, outcomes: np.ndarray) -> None: ...


class Measure(Protocol):
    def __call__(self, actions: np.ndarray, ctx: StepContext) -> np.ndarray: ...


#: A live reader of results (a display, a monitor, a person-facing log): ``monitor(ctx, actions, outcomes)``.
Monitor = Callable[[StepContext, np.ndarray, np.ndarray], None]
#: Decoy outcomes for sealed arms: ``decoy(actions, ctx, shape, dtype) -> array`` (the applied actions and the step
#: context are given so the decoy can follow the frozen model; it never sees the real outcomes).
Decoy = Callable[[np.ndarray, StepContext, tuple[int, ...], np.dtype], np.ndarray]
#: Sham actions: ``sham(step, batch) -> actions``.
Sham = Callable[[int, int], np.ndarray]


def poisson_decoy(rng: np.random.Generator, mean: float) -> Decoy:
    """Decoy outcomes: Poisson counts of one fixed mean in every window (unit tests only: the structure differs from
    real outcomes, so a controller's code paths could differ between observed and sealed arms; use
    :func:`model_decoy` in a plan)."""

    def decoy(actions: np.ndarray, ctx: StepContext, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        return rng.poisson(mean, size=shape).astype(dtype)

    return decoy


def model_decoy(rng: np.random.Generator, expected: Callable[[np.ndarray, StepContext], np.ndarray]) -> Decoy:
    """Decoy outcomes from the frozen model: Poisson counts with the model's expected counts per window at the
    applied action (``expected(actions, ctx) -> (B, W)``), drawn from ``rng`` (a generator of its own)."""

    def decoy(actions: np.ndarray, ctx: StepContext, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        lam = np.broadcast_to(np.asarray(expected(actions, ctx), dtype=np.float64), shape)
        return rng.poisson(lam).astype(dtype)

    return decoy


def empirical_sham(rng: np.random.Generator, actions: np.ndarray) -> Sham:
    """Sham actions drawn iid from the marginal distribution of ``actions`` (any shape; flattened, per batch
    element when ``actions`` is (..., B))."""
    a = np.asarray(actions)

    def sham(step: int, batch: int) -> np.ndarray:
        if a.ndim >= 2 and a.shape[-1] == batch:
            flat = a.reshape(-1, batch)
            idx = rng.integers(0, flat.shape[0], size=batch)
            return flat[idx, np.arange(batch)]
        return rng.choice(a.ravel(), size=batch)

    return sham


@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: Literal["decision", "replay", "sham"]
    observed: bool = True
    #: Replays: where the applied table comes from ("decision": the decision arm's actions ``lag`` blocks earlier;
    #: "offline": the offline table of block ``k - lag``).
    source: Literal["decision", "offline"] = "decision"
    lag: int = 1
    variant: str = "nominal"


@dataclass(frozen=True)
class EpisodeMeta:
    """What differs between observed and sealed arms by construction: sealed with the blinding key."""

    #: (T, B, ...) the controller's own proposals (applied only in the decision arm).
    proposals: np.ndarray
    #: Deliveries of real outcomes to the controller or a monitor (0 in a sealed arm).
    plaintext_reads: int
    #: True when the replay source did not exist yet (warm-up) and the sham supplied the actions.
    fallback: bool
    #: (T,) seconds the controller spent per step (propose + observe).
    compute_s: np.ndarray

    def to_bytes(self) -> bytes:
        p = np.asarray(self.proposals)
        d = {
            "proposals": p.tolist(),
            "proposals_dtype": p.dtype.str,
            "plaintext_reads": int(self.plaintext_reads),
            "fallback": bool(self.fallback),
            "compute_s": [float(x) for x in self.compute_s],
        }
        return json.dumps(d).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> EpisodeMeta:
        d = json.loads(data)
        return cls(
            np.asarray(d["proposals"], dtype=np.dtype(d["proposals_dtype"])),
            int(d["plaintext_reads"]),
            bool(d["fallback"]),
            np.asarray(d["compute_s"], dtype=np.float64),
        )


@dataclass
class Episode:
    """One episode of one arm in one block (all batch elements). ``arm`` is the arm's name, known only inside the
    acquisition process; nothing written to disk carries it (:func:`sealed_artifact`)."""

    arm: str
    block: int
    #: (T, B, ...) actions applied.
    actions: np.ndarray
    #: One sealed record per step: the (B, W) outcomes (sealing key).
    sealed: list[SealedRecord]
    #: :class:`EpisodeMeta`, sealed with the blinding key.
    meta: SealedRecord
    #: (B,) time slot of the episode.
    slot: np.ndarray
    #: The label of the episode's records: the arm's blinding code, or a random token.
    label: str = ""

    def outcomes(self, key: bytes) -> np.ndarray:
        return unseal_outcomes(self, key)

    def unseal_meta(self, blind_key: bytes) -> EpisodeMeta:
        return EpisodeMeta.from_bytes(unseal(blind_key, self.meta).tobytes())


def sealed_artifact(ep: Episode) -> dict[str, Any]:
    """The on-disk form of an episode: its label (code or token), block, time slots, applied actions, the sealed
    outcome records and the sealed meta record. No arm name and no field that differs between observed and sealed
    arms is in the clear."""

    def rec(r: SealedRecord) -> dict[str, Any]:
        return {
            "id": r.record_id,
            "nonce": r.nonce.hex(),
            "dtype": r.dtype,
            "shape": list(r.shape),
            "ct": r.ciphertext.hex(),
            "tag": r.tag.hex(),
        }

    return {
        "label": ep.label,
        "block": int(ep.block),
        "slot": np.asarray(ep.slot).tolist(),
        "actions": np.asarray(ep.actions).tolist(),
        "outcomes": [rec(r) for r in ep.sealed],
        "meta": rec(ep.meta),
    }


def unseal_outcomes(ep: Episode, key: bytes) -> np.ndarray:
    """(T, B, W) outcomes of an episode, decrypted and authenticated (raises on a wrong key or altered records)."""
    return np.stack([unseal(key, r) for r in ep.sealed])


def action_hash(actions: np.ndarray) -> str:
    """sha256 of an action sequence (T, ...) as int64/float64 bytes: equal hashes = identical applied sequences."""
    a = np.asarray(actions)
    a = a.astype(np.int64) if np.issubdtype(a.dtype, np.integer) else a.astype(np.float64)
    return sha256_bytes(np.ascontiguousarray(a).tobytes() + str(a.shape).encode())


def run_episode(
    arm: ArmSpec,
    block: int,
    controller: Controller,
    measure: Measure,
    n_steps: int,
    batch: int,
    sealer: Sealer,
    slot: np.ndarray,
    table: np.ndarray | None = None,
    sham: Sham | None = None,
    decoy: Decoy | None = None,
    monitors: Sequence[Monitor] = (),
    *,
    meta_sealer: Sealer,
    label: str | None = None,
    fallback: bool = False,
) -> Episode:
    """Run one episode. ``table`` (T, B, ...) is required for replays, ``sham`` for sham arms, ``decoy`` for sealed
    arms. ``meta_sealer`` holds the blinding key; ``label`` (the arm's blinding code) labels the sealed records, a
    random token when None."""
    if arm.kind == "replay" and table is None:
        raise ValueError(f"{arm.name}: a replay needs a table")
    if arm.kind == "sham" and sham is None:
        raise ValueError(f"{arm.name}: a sham arm needs a sampler")
    if not arm.observed and decoy is None:
        raise ValueError(f"{arm.name}: a sealed arm needs a decoy source")
    tag = label if label is not None else secrets.token_hex(4)
    controller.reset(batch)
    acts, props, sealed, compute = [], [], [], []
    reads = 0
    for t in range(n_steps):
        t0 = time.perf_counter()
        proposal = np.asarray(controller.propose(t))
        c_s = time.perf_counter() - t0
        if arm.kind == "decision":
            applied = proposal
        elif arm.kind == "replay":
            assert table is not None
            applied = np.asarray(table[t])
        else:
            assert sham is not None
            applied = np.asarray(sham(t, batch))
        ctx = StepContext(block, arm.name, arm.variant, t, slot)
        y = np.asarray(measure(applied, ctx))
        sealed.append(sealer.seal(f"{block}/{tag}/{t}", y))
        if arm.observed:
            seen = y
            reads += 1 + len(monitors)
        else:
            assert decoy is not None
            seen = decoy(applied, ctx, tuple(y.shape), y.dtype)
        del y
        t0 = time.perf_counter()
        controller.observe(t, applied, seen)
        compute.append(c_s + time.perf_counter() - t0)
        for mon in monitors:
            mon(ctx, applied, seen)
        acts.append(applied)
        props.append(proposal)
    meta = EpisodeMeta(np.stack(props), reads, fallback, np.asarray(compute))
    meta_rec = meta_sealer.seal(f"{block}/{tag}/meta", np.frombuffer(meta.to_bytes(), dtype=np.uint8))
    return Episode(arm.name, block, np.stack(acts), sealed, meta_rec, np.asarray(slot), tag)


@dataclass(frozen=True)
class ProtocolDesign:
    """The arms of one block and the episode length. At most one decision arm; replays from the decision arm need
    ``lag >= 1`` (so the order inside a block can be fully random)."""

    arms: tuple[ArmSpec, ...]
    n_steps: int

    def __post_init__(self) -> None:
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            raise ValueError("arm names must be unique")
        if sum(a.kind == "decision" for a in self.arms) > 1:
            raise ValueError("at most one decision arm")
        for a in self.arms:
            if a.kind == "replay" and a.source == "decision" and a.lag < 1:
                raise ValueError(f"{a.name}: replays of the decision arm need lag >= 1")
            if a.kind == "replay" and a.lag < 0:
                raise ValueError(f"{a.name}: negative lag")

    @property
    def names(self) -> list[str]:
        return [a.name for a in self.arms]

    def arm(self, name: str) -> ArmSpec:
        return next(a for a in self.arms if a.name == name)


@dataclass
class ProtocolResult:
    episodes: list[Episode]
    #: block -> (T, B, ...) actions of the decision arm (the last ``max lag + 1`` blocks).
    decision_tables: dict[int, np.ndarray] = field(default_factory=dict)


def run_protocol(
    design: ProtocolDesign,
    schedules: np.ndarray,
    controller: Controller,
    measure: Measure,
    sealer: Sealer,
    decoy: Decoy,
    sham: Sham,
    offline: np.ndarray | None = None,
    monitors: Sequence[Monitor] = (),
    on_episode: Callable[[Episode], None] | None = None,
    *,
    blind_sealer: Sealer,
    codes: Mapping[str, str] | None = None,
) -> ProtocolResult:
    """Run every block of ``schedules`` ((B, n_blocks, n_arms) arm names, one schedule per batch element, or
    (n_blocks, n_arms) for B = 1).

    With B = 1 the arms run in the schedule's order (real time). With B > 1 (simulated replicates, each with its own
    schedule) they run arm by arm inside each block, decision arm first, and each element's time slot is passed to
    the measurement; that is equivalent only for simulators whose state between episodes is exogenous drift indexed
    by the slot. Replays whose source does not exist yet (the first ``lag`` blocks) replay a sham table and are
    marked ``fallback`` in their sealed meta record (those blocks are warm-up blocks and excluded from analysis).
    ``offline`` is (n_blocks, T, B, ...) for replays with ``source="offline"``. With ``on_episode`` every finished
    episode is handed to it (e.g. written to disk with :func:`sealed_artifact`) instead of being kept in the result.
    ``blind_sealer`` (the blinding key) seals each episode's :class:`EpisodeMeta`; ``codes`` (arm -> blinding code,
    B = 1) labels the records, random tokens otherwise.
    """
    sch = np.asarray(schedules, dtype=object)
    if sch.ndim == 2:
        sch = sch[None]
    batch, n_blocks, n_arms = sch.shape
    if sorted(design.names) != sorted(map(str, sch[0, 0])):
        raise ValueError("schedule arms differ from the design's arms")
    pos = {a: np.argmax(sch == a, axis=2) for a in design.names}  # (B, n_blocks)
    out = ProtocolResult([])
    t_n = design.n_steps
    max_lag = max([a.lag for a in design.arms if a.kind == "replay" and a.source == "decision"], default=0)
    for k in range(n_blocks):
        if batch == 1:
            order = [str(a) for a in sch[0, k]]
        else:
            order = sorted(design.names, key=lambda a: (design.arm(a).kind != "decision", design.names.index(a)))
        for name in order:
            arm = design.arm(name)
            table = None
            fallback = False
            if arm.kind == "replay":
                if arm.source == "decision":
                    table = out.decision_tables.get(k - arm.lag)
                elif offline is not None and k - arm.lag >= 0:
                    table = offline[k - arm.lag]
                if table is None:
                    table = np.stack([np.asarray(sham(t, batch)) for t in range(t_n)])
                    fallback = True
            slot = k * n_arms + pos[name][:, k]
            ep = run_episode(
                arm,
                k,
                controller,
                measure,
                t_n,
                batch,
                sealer,
                slot,
                table,
                sham,
                decoy,
                monitors,
                meta_sealer=blind_sealer,
                label=codes.get(name) if codes is not None else None,
                fallback=fallback,
            )
            if arm.kind == "decision":
                out.decision_tables[k] = ep.actions
                for old in [b for b in out.decision_tables if b < k - max_lag]:
                    del out.decision_tables[old]
            if on_episode is not None:
                on_episode(ep)
            else:
                out.episodes.append(ep)
    return out


class SimulatorMeasure:
    """A :class:`Measure` over ``Simulator.run()``: one pulse sequence per action, ``shots`` shots, outcomes = counts
    summed over the shots in each readout window. ``variants`` map a variant name to a sequence transform (e.g.
    ``{"detuned": lambda s: detune_sequence(s, 400e6)}``)."""

    def __init__(
        self,
        sim: Simulator,
        env: EnvParams,
        sequence_of: Callable[[np.ndarray], PulseSequence],
        windows_ns: Callable[[PulseSequence], Sequence[tuple[int, int]]],
        shots: int,
        rng: np.random.Generator,
        variants: dict[str, Callable[[PulseSequence], PulseSequence]] | None = None,
    ) -> None:
        self.sim, self.env, self.sequence_of, self.windows_ns = sim, env, sequence_of, windows_ns
        self.shots, self.rng = shots, rng
        self.variants = dict(variants or {})

    def __call__(self, actions: np.ndarray, ctx: StepContext) -> np.ndarray:
        rows = []
        for a in np.asarray(actions):
            seq = self.sequence_of(a)
            if ctx.variant != "nominal":
                seq = self.variants[ctx.variant](seq)
            tr = self.sim.run(seq, self.env, shots=self.shots, rng=self.rng)
            rows.append([int(tr.window_counts(s, e).sum()) for s, e in self.windows_ns(seq)])
        return np.asarray(rows, dtype=np.int64)
