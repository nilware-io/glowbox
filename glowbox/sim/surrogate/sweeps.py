"""Fast paths for frequency sweeps on the surrogate: CW steady state and shared-prefix (branched) pulsed sweeps.

Both are exact reformulations of :func:`~glowbox.sim.surrogate.model.expected_rate_cps`, not approximations of
the physics; they only skip work that a sweep repeats:

* :func:`cw_steady_rate_cps`: a CW ODMR point (laser and MW on together, constant controls) relaxes to the
  steady state of its constant generator L. Instead of stepping bins until it gets there, solve L x = 0 with unit
  total population. This is the infinite-duration limit of a CW point; the time-stepped late window of a CW
  sequence converges to it within a few microseconds at the default optical pumping (tests/sim/test_sweeps.py).
* :func:`branched_rate_cps`: the sequences of a pulsed sweep share their MW-free start (initialisation laser and
  dark settle). Simulate that prefix once per static field and continue every sweep point from its final state.
  Exact because a MW-free prefix never creates 0~/+-~ spin coherence (see
  :func:`~glowbox.sim.surrogate.model.emission_and_state`).

Laser drift is not supported here (it makes every shot a different simulation anyway).
"""

from __future__ import annotations

import numpy as np
import torch

from glowbox.control.pulse import SequenceArrays
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import (
    ACC,
    N_STATE,
    ModelOptions,
    _generator,
    build_statics,
    emission_and_state,
    statics_for,
)

#: Indices of populations in the real state layout (ground 0~, -~, +~; excited ms=0, -1, +1; singlet).
_POPULATIONS = (0, 1, 2, 9, 10, 11, 12)
#: Fields per batched solve in :func:`cw_steady_rate_cps` (memory: fields x freqs x K x M x 14 x 14 floats).
_CW_FIELD_CHUNK = 8


def _device(device: torch.device | str | None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


@torch.no_grad()
def cw_steady_emission_per_nv(
    env: EnvParams,
    fields_t: np.ndarray,
    freqs_hz: np.ndarray,
    *,
    amplitude: float,
    laser_power: float = 1.0,
    options: ModelOptions | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Steady-state photon emission per NV (1/s) under CW laser + MW, (G, F) float64.

    ``fields_t`` (G, 3) total static fields (tesla, crystal frame; ``env.b_field_t`` is *not* added), ``freqs_hz``
    (F,) MW frequencies shared by every field, or (G, F). MW phase 0 (irrelevant for one CW tone). The solve runs
    in float64 whatever ``options.dtype`` says: a steady state needs no speed from float32 and the 13x13 systems
    are mildly ill-conditioned (rates span 1/T1 = 200 s^-1 to k_r = 6.6e7 s^-1).
    """
    opt = options or ModelOptions()
    dev = _device(device)
    fields = np.atleast_2d(np.asarray(fields_t, dtype=np.float64))
    g = fields.shape[0]
    f = np.asarray(freqs_hz, dtype=np.float64)
    f = np.broadcast_to(f, (g, f.shape[-1])) if f.ndim == 1 else f
    if f.shape[0] != g:
        raise ValueError(f"freqs_hz has {f.shape[0]} rows for {g} fields")
    beta = laser_power * env.optical_beta_full_scale
    out = []
    for s in range(0, g, _CW_FIELD_CHUNK):
        sub = fields[s : s + _CW_FIELD_CHUNK]
        st = build_statics(
            env,
            sub,
            opt.n_detuning,
            torch.float64,
            dev,
            detuning_rule=opt.detuning_rule,
            detuning_spacing=opt.detuning_spacing,
        )
        fr = torch.as_tensor(np.array(f[s : s + _CW_FIELD_CHUNK]), dtype=torch.float64, device=dev)  # (g', F)
        grp = torch.arange(sub.shape[0], device=dev)[:, None].expand(fr.shape)
        full = torch.full(fr.shape, 1.0, dtype=torch.float64, device=dev)
        lmat = _generator(st, grp, beta * full, amplitude * full, 0.0 * full, fr)  # (g', F, K, M, 14, 14)
        a = lmat[..., :N_STATE, :N_STATE].clone()
        # Trace conservation makes the population rows dependent: replace row 0 (d rho00/dt) by the normalisation.
        norm = torch.zeros(N_STATE, dtype=torch.float64, device=dev)
        norm[list(_POPULATIONS)] = 1.0
        a[..., 0, :] = norm
        rhs = torch.zeros(N_STATE, dtype=torch.float64, device=dev)
        rhs[0] = 1.0
        x = torch.linalg.solve(a, rhs.expand(a.shape[:-1]))  # (g', F, K, M, 13)
        em = torch.einsum("gfkmi,gfkmi->gfkm", lmat[..., ACC, :N_STATE], x)  # k_r * excited population
        out.append(torch.einsum("gfkm,km->gf", em, st.weights))
    return torch.cat(out)


def cw_steady_rate_cps(
    env: EnvParams,
    fields_t: np.ndarray,
    freqs_hz: np.ndarray,
    *,
    amplitude: float,
    laser_power: float = 1.0,
    options: ModelOptions | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Steady-state detected count rate (counts/s, docs/physics.md section 7) of CW ODMR points, (G, F) float64."""
    em = cw_steady_emission_per_nv(
        env, fields_t, freqs_hz, amplitude=amplitude, laser_power=laser_power, options=options, device=device
    )
    return env.collection_efficiency * env.n_nv * em + env.background_cps * laser_power + env.dark_cps


@torch.no_grad()
def branched_rate_cps(
    prefix: SequenceArrays,
    suffix: SequenceArrays,
    branch: np.ndarray,
    env: EnvParams,
    *,
    options: ModelOptions | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Noiseless detected rate (counts/s) of the suffix bins of sequences ``prefix[branch[i]] + suffix[i]``.

    ``prefix`` is a batch of P MW-free sequences (e.g. init laser + settle, one per static field), ``suffix`` a
    batch of S sequences that each continue prefix row ``branch[i]``; the suffix magnet must equal its prefix's.
    Returns (S, suffix.n_bins); equal to the corresponding bins of the full sequences (exactly, up to round-off).
    """
    opt = options or ModelOptions()
    dev = _device(device)
    pre = prefix.to_numpy()
    suf = suffix.to_numpy()
    if np.any(np.asarray(pre.mw_amplitude) != 0):
        raise ValueError("the prefix must be MW-free (it may not carry spin coherence into the suffix)")
    idx = np.asarray(branch, dtype=np.int64)
    if not np.array_equal(np.asarray(pre.magnet_t).reshape(-1, 3)[idx], np.asarray(suf.magnet_t).reshape(-1, 3)):
        raise ValueError("each suffix must have the static field (magnet) of its prefix")
    # The suffix fields are a subset of the prefix fields (checked above); when they are the same set (every
    # prefix row is used), one Statics serves both calls.
    st = statics_for(pre, env, opt, dev)
    n_pre = np.unique(np.asarray(pre.magnet_t).reshape(-1, 3), axis=0).shape[0]
    n_suf = np.unique(np.asarray(suf.magnet_t).reshape(-1, 3), axis=0).shape[0]
    _, state = emission_and_state(pre, env, options=opt, device=dev, statics=st)
    r0 = state[torch.as_tensor(idx, device=state.device)]
    st_suf = st if n_suf == n_pre else None
    em, _ = emission_and_state(suf, env, options=opt, device=dev, initial_state=r0, statics=st_suf)
    laser = torch.as_tensor(np.asarray(suf.laser), device=em.device).to(em.dtype).reshape(em.shape)
    return env.collection_efficiency * env.n_nv * em + env.background_cps * laser + env.dark_cps
