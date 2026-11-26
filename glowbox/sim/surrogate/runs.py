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
#: Matrices per exponential call (memory bound, as MAX_EXP_BATCH_GRAD of the bin path).
MAX_RUN_BATCH = 1 << 16


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


def _expm_batched(x: torch.Tensor, taylor: bool) -> torch.Tensor:
    flat = x.reshape(-1, N_AUG, N_AUG)
    fn = _expm_taylor if taylor else torch.linalg.matrix_exp
    if flat.shape[0] <= MAX_RUN_BATCH:
        return fn(flat).reshape(x.shape)
    return torch.cat([fn(p) for p in torch.split(flat, MAX_RUN_BATCH)]).reshape(x.shape)


def run_emission(
    st: Statics,
    group: torch.Tensor,
    runs: list[Run],
    r0: torch.Tensor,
    *,
    taylor: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate (B, K, M, 14) states ``r0`` through ``runs``; returns (photons per NV of the recorded runs, averaged
    over orientations and detuning nodes with the quadrature weights, (B, n_recorded); final states).

    ``group`` (B,) is each sequence's static-field index into ``st``. ``taylor`` defaults to "when gradients are
    enabled".
    """
    b = r0.shape[0]
    dev, dtype = r0.device, st.l0.dtype
    use_taylor = torch.is_grad_enabled() if taylor is None else taylor
    keep = _KEEP_COLS.to(dtype).to(dev)
    r = r0.to(dtype)
    out = []
    prev_f: torch.Tensor | None = None
    for run in runs:
        f = _vec(run.frame_hz, b, torch.float64, dev)
        if prev_f is not None:
            t0 = _vec(run.start_s, b, torch.float64, dev)
            theta = torch.remainder(_TWO_PI * (f - prev_f) * t0, _TWO_PI)
            if bool((theta != 0).any()):
                r = _rotate(r, torch.cos(theta).to(dtype)[:, None, None], torch.sin(theta).to(dtype)[:, None, None])
        prev_f = f
        lgen = _generator(
            st,
            group,
            _vec(run.beta, b, dtype, dev),
            _vec(run.ac, b, dtype, dev),
            _vec(run.as_, b, dtype, dev),
            f,
        )  # (B, K, M, 14, 14)
        dur = _vec(run.duration_s, b, dtype, dev)
        p = _expm_batched(lgen * dur[:, None, None, None, None], use_taylor)
        if run.record:
            # Reset the accumulator, then integrate it over the whole run (column kept).
            r = r * keep
            r = (p @ r[..., None])[..., 0]
            out.append(torch.einsum("bkm,km->b", r[..., ACC], st.weights))
        else:
            r = ((p * keep) @ r[..., None])[..., 0]
    em = torch.stack(out, dim=1) if out else torch.zeros(b, 0, dtype=dtype, device=dev)
    return em, r
