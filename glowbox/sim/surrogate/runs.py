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

Matrix exponentials: Taylor scaling-and-squaring (:func:`~glowbox.sim.surrogate.model._expm_taylor`, backward ~2x
forward) when differentiating, ``torch.linalg.matrix_exp`` otherwise. The scaling is chosen per call from the largest
norm; long runs (|L T| ~ 100 for a microsecond of laser) cost a few extra squarings, not more steps.
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
    _expm_taylor,
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


def _run_propagators(
    st: Statics,
    group: torch.Tensor,
    beta: torch.Tensor,
    ac: torch.Tensor,
    as_: torch.Tensor,
    frame: torch.Tensor,
    dur: torch.Tensor,
    taylor: bool,
) -> torch.Tensor:
    """exp(L T) for a chunk of runs, controls (B, R): (B, R, K, M, 14, 14)."""
    lgen = _generator(st, group[:, None].expand_as(beta), beta, ac, as_, frame)
    x = lgen * dur[..., None, None, None, None]
    flat = x.reshape(-1, N_AUG, N_AUG)
    fn = _expm_taylor if taylor else torch.linalg.matrix_exp
    return fn(flat).reshape(x.shape)


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
    props = []
    for i in range(0, n_runs, chunk):
        sl = slice(i, min(n_runs, i + chunk))
        args = (st, group, beta[:, sl], ac[:, sl], as_[:, sl], frame[:, sl], dur[:, sl], use_taylor)
        if grad and use_checkpoint:
            props.append(checkpoint(_run_propagators, *args, use_reentrant=False))
        else:
            props.append(_run_propagators(*args))
    p_all = torch.cat(props, dim=1)  # (B, R, K, M, 14, 14)
    r = r0.to(dtype)
    out = []
    for j, run in enumerate(runs):
        if rot[j]:
            r = _rotate(r, cos_t[:, j, None, None], sin_t[:, j, None, None])
        p = p_all[:, j]
        if run.record:
            # Reset the accumulator, then integrate it over the whole run (accumulator column kept).
            r = (p @ (r * keep)[..., None])[..., 0]
            out.append(torch.einsum("bkm,km->b", r[..., ACC], st.weights))
        else:
            r = ((p * keep) @ r[..., None])[..., 0]
    em = torch.stack(out, dim=1) if out else torch.zeros(b, 0, dtype=dtype, device=dev)
    return em, r
