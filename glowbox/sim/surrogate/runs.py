"""Run-length surrogate path: a sequence as a fixed list of *runs* (stretches of constant controls) with per-sequence
durations, one propagator per run.

The bin-by-bin surrogate (:mod:`glowbox.sim.surrogate.model`) steps every 10 ns bin. A pulse-level sequence (an
encoder's MW segments, a readout laser) is a handful of runs of identical bins, and within a run the generator L is
constant, so the run's propagator is exactly ``exp(L T)`` for its duration T: the same physics with one step per
run instead of one per bin. Two things come for free:

* **Durations are continuous** and differentiable: T enters as ``exp(L T)``, whose derivative with respect to T is
  exact (no fill fractions, no partial bins). On the grid (T a multiple of 10 ns) the result equals the bin-stepped
  surrogate up to round-off (tests/learn/test_encoder.py, run path vs bin path).
* **Summed emission** over a run (a gated photon counter's window) comes from the photon accumulator, which is not
  reset inside the run: set it to 0 at the run start, propagate with ``exp(L T)`` (L has no accumulator column, so the
  accumulator integrates k_r (p4 + p5 + p6) over the run), read it at the end.

Frame bookkeeping follows docs/physics.md section 3: a run with MW is integrated in the frame of its MW frequency; at
the start of a run whose frame frequency differs from the previous run's, the coherences rho_{0~,+-~} get the phase
exp(-i 2 pi (F_new - F_old) t_start), t_start in the sequence's own time origin (the caller passes it). Runs without
MW keep the previous frame (the caller passes the same F). Like :func:`~glowbox.sim.surrogate.model.emission_and_state`
with an ``initial_state``, the time origin is unobservable when the start state has no spin coherence (after a
MW-free prefix).

Matrix exponentials: Taylor scaling-and-squaring (backward ~2x forward) when differentiating, with the squaring
count chosen per run from a norm bound and runs of equal count batched together; ``torch.linalg.matrix_exp``
otherwise. Long runs (|L T| ~ 100-1000 for a microsecond of laser with the frame 100 MHz from a line) cost a few
extra squarings, not more steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from glowbox.sim.surrogate.model import (
    _KEEP_COLS,
    ACC,
    N_AUG,
    Statics,
    _generator,
    _rotate,
)

_TWO_PI = 2.0 * math.pi
#: 14x14 matrices per exponential call (one chunk of runs; memory bound when differentiating: the Taylor
#: intermediates of one chunk, ~(deg + s) x 2^17 x 784 B ~ 2 GB float32).
MAX_RUN_BATCH = 1 << 17


@dataclass
class Run:
    """One run, per sequence of the batch (B,): tensors or scalars broadcast to (B,).

    ``duration_s`` may require grad (continuous duration); ``frame_hz`` float64 (the run's MW frequency, or the
    previous run's when MW is off); ``start_s`` the run's start time (float64, only used when the frame changes);
    ``record`` = return the photons per NV emitted during this run.
    """

    duration_s: torch.Tensor | float
    beta: torch.Tensor | float = 0.0
    ac: torch.Tensor | float = 0.0
    as_: torch.Tensor | float = 0.0
    frame_hz: torch.Tensor | float = 0.0
    start_s: torch.Tensor | float = 0.0
    record: bool = False


def _vec(x: torch.Tensor | float, b: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=dtype, device=device)
    return t.expand(b) if t.ndim == 0 else t


def _expm_taylor_fixed(a: torch.Tensor, s: int) -> torch.Tensor:
    """exp(a) by Taylor (degree 10 float32 / 18 float64) after scaling by 2^-s, then s squarings; ``s`` given by the
    caller (from a norm bound, :func:`_scaling`), so there is no host sync."""
    x = a / (2.0**s)
    deg = 18 if a.dtype == torch.float64 else 10
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    e = eye + x / deg
    for k in range(deg - 1, 0, -1):
        e = eye + (x @ e) / k
    for _ in range(s):
        e = e @ e
    return e


def _run_propagators(
    st: Statics,
    group: torch.Tensor,
    beta: torch.Tensor,
    ac: torch.Tensor,
    as_: torch.Tensor,
    frame: torch.Tensor,
    dur: torch.Tensor,
    taylor: bool,
    s: int,
) -> torch.Tensor:
    """exp(L T) for a chunk of runs, controls (B, R): (B, R, K, M, 14, 14)."""
    lgen = _generator(st, group[:, None].expand_as(beta), beta, ac, as_, frame)
    x = lgen * dur[..., None, None, None, None]
    flat = x.reshape(-1, N_AUG, N_AUG)
    out = _expm_taylor_fixed(flat, s) if taylor else torch.linalg.matrix_exp(flat)
    return out.reshape(x.shape)


def _scaling(
    st: Statics, beta: torch.Tensor, ac: torch.Tensor, as_: torch.Tensor, frame: torch.Tensor, dur: torch.Tensor
) -> list[int]:
    """Per run (columns of the (B, R) controls): the squaring count s that brings an upper bound of the 1-norm of
    L T below 1/2 (triangle inequality over the generator's terms; detunings bounded by the extreme e+- of the
    statics)."""

    def n1(x: torch.Tensor) -> float:
        return float(x.abs().sum(-2).amax())

    c0, cp = n1(st.l0), n1(st.k_pump)
    cd = max(n1(st.drive_a), n1(st.drive_b))
    ck = max(n1(st.k_m), n1(st.k_p))
    e = torch.stack([st.e_minus_hz.amin(), st.e_minus_hz.amax(), st.e_plus_hz.amin(), st.e_plus_hz.amax()])
    dmax = (e[None, None] - frame[..., None]).abs().amax(-1)  # (B, R) Hz
    with torch.no_grad():
        bound = dur.detach().abs().double() * (
            c0
            + beta.detach().abs().double() * cp
            + (ac.detach().abs() + as_.detach().abs()).double() * cd
            + 2.0 * ck * dmax.detach()
        )
        nb = bound.amax(0).clamp_min(1e-30)
        s = torch.ceil(torch.log2(nb / 0.5)).clamp_min(0).long()
    return [int(x) for x in s.tolist()]


def run_emission(
    st: Statics,
    group: torch.Tensor,
    runs: list[Run],
    r0: torch.Tensor,
    *,
    taylor: bool | None = None,
    use_checkpoint: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate (B, K, M, 14) states ``r0`` through ``runs``; returns (photons per NV of the recorded runs, averaged
    over orientations and detuning nodes with the quadrature weights, (B, n_recorded); final states).

    ``group`` (B,) is each sequence's static-field index into ``st``. ``taylor`` defaults to "when gradients are
    enabled". The propagators of all runs are built in a few large batched exponentials (chunks of runs; with
    gradients each chunk is checkpointed: backward recomputes it instead of storing the Taylor intermediates), then
    the state steps through them run by run.
    """
    b = r0.shape[0]
    dev, dtype = r0.device, st.l0.dtype
    grad = torch.is_grad_enabled()
    use_taylor = grad if taylor is None else taylor
    keep = _KEEP_COLS.to(dtype).to(dev)
    n_runs = len(runs)
    if n_runs == 0:
        return torch.zeros(b, 0, dtype=dtype, device=dev), r0

    def col(name: str, dt: torch.dtype) -> torch.Tensor:
        return torch.stack([_vec(getattr(r, name), b, dt, dev) for r in runs], dim=1)  # (B, R)

    beta, ac, as_ = col("beta", dtype), col("ac", dtype), col("as_", dtype)
    frame, start = col("frame_hz", torch.float64), col("start_s", torch.float64)
    dur = col("duration_s", dtype)
    df = torch.diff(frame, dim=1, prepend=frame[:, :1])
    theta = torch.remainder(_TWO_PI * df * start, _TWO_PI)
    rot = (theta != 0).any(0).tolist()
    cos_t, sin_t = torch.cos(theta).to(dtype), torch.sin(theta).to(dtype)
    k, m = st.weights.shape
    per_run = max(1, b * k * m)
    chunk = max(1, MAX_RUN_BATCH // per_run)
    scal = _scaling(st, beta, ac, as_, frame, dur) if use_taylor else [0] * n_runs
    # Runs with equal scaling share exponential calls (a long readout no longer forces its squarings on every
    # short MW segment).
    props: list[torch.Tensor | None] = [None] * n_runs
    for sv in sorted(set(scal)):
        idx = [j for j in range(n_runs) if scal[j] == sv]
        for i in range(0, len(idx), chunk):
            cols = torch.as_tensor(idx[i : i + chunk], device=dev)
            args = (
                st,
                group,
                beta[:, cols],
                ac[:, cols],
                as_[:, cols],
                frame[:, cols],
                dur[:, cols],
                use_taylor,
                sv,
            )
            if grad and use_checkpoint:
                pc = checkpoint(_run_propagators, *args, use_reentrant=False)
            else:
                pc = _run_propagators(*args)
            for c_i, j in enumerate(idx[i : i + chunk]):
                props[j] = pc[:, c_i]
    r = r0.to(dtype)
    out = []
    for j, run in enumerate(runs):
        if rot[j]:
            r = _rotate(r, cos_t[:, j, None, None], sin_t[:, j, None, None])
        p = props[j]
        assert p is not None
        if run.record:
            # Reset the accumulator, then integrate it over the whole run (accumulator column kept).
            r = (p @ (r * keep)[..., None])[..., 0]
            out.append(torch.einsum("bkm,km->b", r[..., ACC], st.weights))
        else:
            r = ((p * keep) @ r[..., None])[..., 0]
    em = torch.stack(out, dim=1) if out else torch.zeros(b, 0, dtype=dtype, device=dev)
    return em, r
