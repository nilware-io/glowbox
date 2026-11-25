"""Fast, batched, differentiable NV-ensemble model (the physics core of the surrogate).

Implements docs/physics.md with PyTorch on any device. The state of one NV (one orientation k, one static detuning
sample j) is the seven-level model of section 6 written in its exact reduced form:

* the ground-state 3x3 density matrix (spin coherences live only here), in the eigenbasis {0~, -~, +~} of H0 and in
  the frame rotating at the current MW frequency (section 3), stored as 9 real numbers;
* the populations of the three excited levels and of the singlet (excited/singlet levels carry populations only:
  every jump of section 6 is a separate operator |j><i|, so no coherence is ever created there; this is exact, not
  the three-level simplification);
* one photon accumulator (integral of k_r * excited population over the current bin).

Controls are piecewise constant per grid bin, so within a bin the generator is a constant real 14x14 matrix L and
the bin propagator is ``matrix_exp(L * dt)``: no sub-stepping error and the bin-integrated emission comes out
exactly (accumulator row). Batches run over (sequence, orientation, detuning sample); with no gradient needed,
identical bins share one propagator (deduplicated), which makes standard protocols cheap.

Rotating frame and RWA (section 3): in bins with MW on, the frame rotates at that bin's frequency; in MW-off bins it
keeps the last frequency. A frequency change at bin start t_s applies the phase 2 pi (f2 - f1) t_s to the upper
states (absolute-time phase rule). Dissipators are fixed in the lab eigenbasis; in the rotating frame their
elements that couple coherences of different "frame order" oscillate at ~f and are dropped (secular
approximation, the dissipative half of the RWA; exact when the 0~ state is pure ms=0, error ~ (gamma_e B_perp / D)^2
otherwise).

Inhomogeneous broadening (section 4): each detuning sample delta adds ``delta * Sz`` (a static field shift along
the NV axis) to H0 before diagonalisation, so f+- -> f+- +- delta whenever gamma_e B_par >> E, gamma_e B_perp (and
the degenerate low-field case is well defined). Samples are equal-probability quantile midpoints (M of them).

Everything static (eigenbases, superoperators) is built in float64 NumPy and cast to the working dtype; frequencies
and frame phases stay float64 until the (small) detunings are formed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import ndtr, ndtri
from torch.utils.checkpoint import checkpoint

from glowbox.control.pulse import SequenceArrays
from glowbox.sim import constants as c
from glowbox.sim.interface import EnvParams

#: Real state layout: rho00, rho--, rho++, Re/Im rho0-, Re/Im rho0+, Re/Im rho-+, e(ms0), e(-1), e(+1), singlet.
N_STATE = 13
#: Augmented with the photon accumulator (index 13).
N_AUG = 14
ACC = 13
_TWO_PI = 2.0 * math.pi

#: Maximum number of 14x14 matrices exponentiated in one call (memory bound: 2^17 * 196 * 4 B ~ 100 MB float32).
MAX_EXP_BATCH = 1 << 17
#: The same when differentiating: autograd keeps ~deg + s intermediates per matrix (one chunk at a time).
MAX_EXP_BATCH_GRAD = 1 << 16
#: Smallest chunk of bins (overrides the budgets above for large batches).
MIN_CHUNK_BINS = 16


# --- Static (control-independent) part --------------------------------------------------------------------------


def detuning_quadrature(
    env: EnvParams, m: int, rule: str = "quantile", spacing: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Static detunings delta (Hz) and weights (sum 1) for the T2* distribution (docs/physics.md section 4).

    Width unit w: Lorentzian HWHM 1/(2 pi T2*), Gaussian sigma = 1/(sqrt(2) pi T2*). ``"quantile"``: M
    equal-probability quantile midpoints, weights 1/M (pulsed protocols). ``"uniform"``: M nodes spaced
    ``spacing`` x w around 0, weight = probability of the node's cell, renormalised over the window (CW
    lineshapes: no satellite comb while the homogeneous HWHM exceeds the spacing). M = 1 means delta = 0.
    """
    if m < 1:
        raise ValueError("need at least one detuning sample")
    if m == 1:
        return np.zeros(1), np.ones(1)
    lor = env.broadening_shape == "lorentzian"
    w = 1.0 / (2.0 * np.pi * env.t2_star_s) if lor else 1.0 / (np.sqrt(2.0) * np.pi * env.t2_star_s)
    if rule == "quantile":
        q = (np.arange(m) + 0.5) / m
        x = np.tan(np.pi * (q - 0.5)) if lor else ndtri(q)
        return x * w, np.full(m, 1.0 / m)
    if rule == "uniform":
        if not spacing > 0:
            raise ValueError("detuning spacing must be > 0")
        x = spacing * (np.arange(m) - (m - 1) / 2)
        edges = np.concatenate([x - spacing / 2, x[-1:] + spacing / 2])
        cdf = 0.5 + np.arctan(edges) / np.pi if lor else ndtr(edges)
        p = np.diff(cdf)
        return x * w, p / p.sum()
    raise ValueError(f"unknown detuning rule {rule!r}")


def _spin1() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spin-1 operators in the ms basis ordered (0, -1, +1) (Tetienne levels |1>, |2>, |3>)."""
    s2 = math.sqrt(2.0)
    sp = np.zeros((3, 3), dtype=complex)
    sp[0, 1] = s2  # S+ |-1> = sqrt2 |0>
    sp[2, 0] = s2  # S+ |0> = sqrt2 |+1>
    sm = sp.conj().T
    return (sp + sm) / 2, (sp - sm) / 2j, np.diag([0.0, -1.0, 1.0]).astype(complex)


def nv_local_frames() -> np.ndarray:
    """(4, 3, 3): rows x, y, z of each NV's local frame (z = axis, x = projection of [001]; docs/physics.md 1)."""
    out = np.empty((4, 3, 3))
    ez = np.array([0.0, 0.0, 1.0])
    for k, n in enumerate(c.NV_AXES):
        x = ez - (ez @ n) * n
        x /= np.linalg.norm(x)
        out[k] = (x, np.cross(n, x), n)
    return out


# Superoperators on row-major vec(rho): vec(A rho B) = (A kron B^T) vec(rho).
def _left(a: np.ndarray) -> np.ndarray:
    return np.einsum("...ac,bd->...abcd", a, np.eye(3)).reshape(*a.shape[:-2], 9, 9)


def _right(b: np.ndarray) -> np.ndarray:
    return np.einsum("ac,...db->...abcd", np.eye(3), b).reshape(*b.shape[:-2], 9, 9)


def _sandwich(j: np.ndarray) -> np.ndarray:
    return np.einsum("...ac,...bd->...abcd", j, j.conj()).reshape(*j.shape[:-2], 9, 9)


def _dissipator(j: np.ndarray) -> np.ndarray:
    jd = np.swapaxes(j.conj(), -1, -2) @ j
    return _sandwich(j) - 0.5 * (_left(jd) + _right(jd))


def _commutator(h: np.ndarray) -> np.ndarray:
    """Superoperator of -i 2 pi [H, .] for H in Hz."""
    return -1j * _TWO_PI * (_left(h) - _right(h))


def _real_transform() -> tuple[np.ndarray, np.ndarray]:
    """T (complex vec -> real coordinates) and its inverse, 14x14."""
    t = np.zeros((N_AUG, N_AUG), dtype=complex)
    for i, d in enumerate((0, 4, 8)):
        t[i, d] = 1.0
    for i, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        ab, ba = 3 * a + b, 3 * b + a
        t[3 + 2 * i, ab], t[3 + 2 * i, ba] = 0.5, 0.5
        t[4 + 2 * i, ab], t[4 + 2 * i, ba] = -0.5j, 0.5j
    for i in range(9, N_AUG):
        t[i, i] = 1.0
    return t, np.linalg.inv(t)


_T, _TINV = _real_transform()
#: Frame order n of each complex-vec index: n(0~)=0, n(-~)=n(+~)=1; element (a,b) carries n_a - n_b.
_N_LEVEL = np.array([0, 1, 1])
_NDIFF = np.concatenate([(_N_LEVEL[:, None] - _N_LEVEL[None, :]).ravel(), np.zeros(N_AUG - 9, dtype=int)])
_SECULAR = _NDIFF[:, None] == _NDIFF[None, :]


def _to_real(lc: np.ndarray) -> np.ndarray:
    return _T @ lc @ _TINV


@dataclass
class Statics:
    """Control-independent model pieces for G distinct static fields, K orientations, M detuning samples.

    Generator of one bin: L = L0 + beta * K_pump + a (cos phi A + sin phi B) + Delta- K_m + Delta+ K_p, in 1/s,
    with Delta+- = e+- - F the detunings from the frame frequency F (Hz). Tensors are on the working device/dtype.
    """

    l0: torch.Tensor  # (G, K, M, 14, 14)
    k_pump: torch.Tensor  # (G, K, M, 14, 14), per unit beta
    drive_a: torch.Tensor  # (G, K, M, 14, 14), per unit MW amplitude, cos(phi) part
    drive_b: torch.Tensor  # (G, K, M, 14, 14), sin(phi) part
    k_m: torch.Tensor  # (14, 14), per Hz of Delta-
    k_p: torch.Tensor  # (14, 14), per Hz of Delta+
    e_minus_hz: torch.Tensor  # (G, K, M) float64, energy of -~ above 0~
    e_plus_hz: torch.Tensor  # (G, K, M) float64
    weights: torch.Tensor  # (K, M): orientation population x quadrature weight
    orientations: tuple[int, ...]  # which of the four NV_AXES are kept (population > 0)


def build_statics(
    env: EnvParams,
    fields_t: np.ndarray,
    n_detuning: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    detuning_rule: str = "quantile",
    detuning_spacing: float = 0.5,
) -> Statics:
    """Diagonalise H0 per (field, orientation, detuning) and assemble the superoperators (docs/physics.md 1-6)."""
    if env.t2_stretch != 1.0:
        # Stretched echo decay is not a Lindblad process; like the reference, refuse rather than fake it.
        raise ValueError("the surrogate supports only t2_stretch == 1 (docs/physics.md section 5)")
    fields_t = np.atleast_2d(np.asarray(fields_t, dtype=np.float64))
    pops = np.asarray(env.orientation_populations, dtype=np.float64)
    keep = tuple(int(k) for k in np.nonzero(pops > 0)[0])
    frames = nv_local_frames()[list(keep)]  # (K, 3, 3)
    delta, wq = detuning_quadrature(env, n_detuning, detuning_rule, detuning_spacing)
    sx, sy, sz = _spin1()

    # H0 in the ms basis, (G, K, M, 3, 3), Hz.
    b_loc = np.einsum("kij,gj->gki", frames, fields_t) * c.GAMMA_E_HZ_PER_T  # (G, K, 3) local x, y, z in Hz
    h_static = env.d_hz * (sz @ sz) + env.e_hz * (sx @ sx - sy @ sy)
    h_b = h_static + b_loc[..., 0, None, None] * sx + b_loc[..., 1, None, None] * sy + b_loc[..., 2, None, None] * sz
    h = h_b[:, :, None] + delta[None, None, :, None, None] * sz
    evals, evecs = np.linalg.eigh(h)
    # 0~ = eigenvector with the largest ms=0 weight; -~, +~ = the other two by ascending energy.
    i0 = np.argmax(np.abs(evecs[..., 0, :]) ** 2, axis=-1)
    others = np.sort(np.stack([(i0 + 1) % 3, (i0 + 2) % 3], axis=-1), axis=-1)  # eigh is ascending
    perm = np.concatenate([i0[..., None], others], axis=-1)
    evals = np.take_along_axis(evals, perm, axis=-1)
    v = np.take_along_axis(evecs, perm[..., None, :], axis=-1)  # columns: |0~>, |-~>, |+~> in the ms basis
    vd = np.swapaxes(v.conj(), -1, -2)
    e_minus, e_plus = evals[..., 1] - evals[..., 0], evals[..., 2] - evals[..., 0]

    def eig(op: np.ndarray) -> np.ndarray:
        return vd @ op @ v

    proj_ms = [np.zeros((3, 3), dtype=complex) for _ in range(3)]
    for m in range(3):
        proj_ms[m][m, m] = 1.0
    proj = [eig(p) for p in proj_ms]  # ms projectors in the eigenbasis, (G, K, M, 3, 3)
    shape = (*v.shape[:-2], N_AUG, N_AUG)

    # Static generator: T1, T2, excited/singlet decay, returns, accumulator.
    l0 = np.zeros(shape, dtype=complex)
    g1 = 1.0 / (3.0 * env.t1_s)
    gnd = np.zeros((*v.shape[:-2], 9, 9), dtype=complex)
    for m in range(3):
        for mp in range(3):
            if m != mp:
                jump = np.zeros((3, 3), dtype=complex)
                jump[mp, m] = math.sqrt(g1)
                gnd += _dissipator(eig(jump))
    gnd += _dissipator(eig(math.sqrt(2.0 / env.t2_s) * sz))
    l0[..., :9, :9] = gnd
    k_isc = (c.K_ISC_MS0_PER_S, c.K_ISC_MS1_PER_S, c.K_ISC_MS1_PER_S)
    k_back = (c.K_SINGLET_TO_MS0_PER_S, c.K_SINGLET_TO_MS1_PER_S, c.K_SINGLET_TO_MS1_PER_S)
    kr = c.K_RADIATIVE_PER_S
    for m in range(3):
        l0[..., 9 + m, 9 + m] = -(kr + k_isc[m])
        l0[..., 12, 9 + m] = k_isc[m]
        l0[..., :9, 9 + m] = kr * proj[m].reshape(*v.shape[:-2], 9)  # e(m) -> ground ms=m
        l0[..., :9, 12] += k_back[m] * proj[m].reshape(*v.shape[:-2], 9)  # singlet -> ground ms=m
        l0[..., ACC, 9 + m] = kr
    l0[..., 12, 12] = -sum(k_back)

    # Optical pumping per unit beta: ground ms=m -> excited ms=m at beta * k_r.
    kp = np.zeros(shape, dtype=complex)
    for m in range(3):
        kp[..., :9, :9] += -0.5 * kr * (_left(proj[m]) + _right(proj[m]))
        # <m|rho_ms|m> = sum_ab V_ma rho~_ab conj(V_mb)
        kp[..., 9 + m, :9] = kr * (v[..., m, :, None] * v[..., m, None, :].conj()).reshape(*v.shape[:-2], 9)
    l0 = np.where(_SECULAR, l0, 0.0)
    kp = np.where(_SECULAR, kp, 0.0)

    # MW drive (already in the rotating frame, section 3): (sqrt2/2) f_R,k [e^{-i phi} M+- |+-~><0~| + h.c.].
    m_dir = np.asarray(env.mw_field_direction, dtype=np.float64)
    m_loc = np.einsum("kij,j->ki", frames, m_dir)  # (K, 3) local components
    m_perp = np.hypot(m_loc[:, 0], m_loc[:, 1])
    u = np.where(m_perp[:, None] > 0, m_loc[:, :2] / np.where(m_perp > 0, m_perp, 1.0)[:, None], 0.0)
    us = u[:, 0, None, None] * sx + u[:, 1, None, None] * sy  # (K, 3, 3)
    scale = (math.sqrt(2.0) / 2.0) * env.rabi_full_scale_hz * m_perp  # (K,)
    us_e = eig(us[None, :, None])
    a1 = np.zeros_like(us_e)
    a1[..., 1:, 0] = us_e[..., 1:, 0]  # M+- |+-~><0~|
    a1 = a1 * scale[None, :, None, None, None]
    k1 = np.zeros(shape, dtype=complex)
    k2 = np.zeros(shape, dtype=complex)
    k1[..., :9, :9] = _commutator(a1)
    k2[..., :9, :9] = _commutator(np.swapaxes(a1.conj(), -1, -2))
    r1, r2 = _to_real(k1), _to_real(k2)
    # e^{-i phi} R1 + e^{i phi} R2 = cos(phi) (R1 + R2) + sin(phi) i (R2 - R1); both real.
    drive_a = (r1 + r2).real
    drive_b = (1j * (r2 - r1)).real

    def unit(diag: list[float]) -> np.ndarray:
        lc = np.zeros((N_AUG, N_AUG), dtype=complex)
        lc[:9, :9] = _commutator(np.diag(diag).astype(complex))
        return _to_real(lc).real

    w = pops[list(keep), None] * wq[None, :]

    def t(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)

    return Statics(
        l0=t(_to_real(l0).real),
        k_pump=t(_to_real(kp).real),
        drive_a=t(drive_a),
        drive_b=t(drive_b),
        k_m=t(unit([0.0, 1.0, 0.0])),
        k_p=t(unit([0.0, 0.0, 1.0])),
        e_minus_hz=torch.as_tensor(e_minus, dtype=torch.float64, device=device),
        e_plus_hz=torch.as_tensor(e_plus, dtype=torch.float64, device=device),
        weights=t(w),
        orientations=keep,
    )


# --- Controls and frames ----------------------------------------------------------------------------------------


def frame_frequency(freq: torch.Tensor, default_hz: float) -> torch.Tensor:
    """Frame frequency per bin (float64, (B, nb)): the bin's MW frequency where one is set (> 0), else the last one.

    A bin's MW frequency is ``freq`` where it is > 0; ``SequenceArrays`` stores 0 for MW-off bins, which therefore
    inherit the previous frequency (an amplitude there drives at it, so the amplitude gradient of an off bin is
    "switch on at the current frequency"). Before the first set bin the frame already sits at the first frequency
    (no switch); with none at all it is ``default_hz``. The frame choice is exact, it does not change the physics;
    in bins with MW the frame equals the MW frequency, as the time-independent drive requires. Differentiable with
    respect to ``freq`` (a gather).
    """
    nb = freq.shape[-1]
    on = (freq.detach() > 0).cpu()
    idx = torch.where(on, torch.arange(nb), torch.full((nb,), -1, dtype=torch.long))
    idx = torch.cummax(idx, dim=-1).values
    first = torch.where(on.any(-1), on.to(torch.int8).argmax(-1), torch.zeros(on.shape[0], dtype=torch.long))
    idx = torch.where(idx < 0, first[:, None], idx).to(freq.device)
    f = torch.gather(freq, -1, idx)
    has = on.any(-1).to(freq.device)[:, None]
    return torch.where(has, f, torch.full_like(f, default_hz))


def frame_phase(frame_hz: torch.Tensor, grid_ns: int) -> torch.Tensor:
    """Phase 2 pi (F_b - F_{b-1}) t_b (rad, float64, (B, nb), wrapped) applied at the start of bin b."""
    t_s = torch.arange(frame_hz.shape[-1], dtype=torch.float64, device=frame_hz.device) * (grid_ns * 1e-9)
    df = torch.diff(frame_hz, dim=-1, prepend=frame_hz[..., :1])
    return torch.remainder(_TWO_PI * df * t_s, _TWO_PI)


def _rotate(r: torch.Tensor, cos_t: torch.Tensor, sin_t: torch.Tensor) -> torch.Tensor:
    """Frame switch: rho_{0+-} *= exp(-i theta). ``cos_t``/``sin_t`` broadcast against r[..., 0]."""
    re1, im1, re2, im2 = r[..., 3], r[..., 4], r[..., 5], r[..., 6]
    return torch.cat(
        [
            r[..., :3],
            torch.stack(
                [
                    re1 * cos_t + im1 * sin_t,
                    im1 * cos_t - re1 * sin_t,
                    re2 * cos_t + im2 * sin_t,
                    im2 * cos_t - re2 * sin_t,
                ],
                dim=-1,
            ),
            r[..., 7:],
        ],
        dim=-1,
    )


def _generator(
    st: Statics,
    group: torch.Tensor,
    beta: torch.Tensor,
    ac: torch.Tensor,
    as_: torch.Tensor,
    frame_hz: torch.Tensor,
) -> torch.Tensor:
    """L (1/s) for controls of shape S (broadcastable; ``group`` long), output (*S, K, M, 14, 14).

    ``ac``, ``as_`` = a cos(phi), a sin(phi) (MW amplitude and phase), ``frame_hz`` float64.
    """
    dtype = st.l0.dtype
    dm = (st.e_minus_hz[group] - frame_hz[..., None, None]).to(dtype)  # (*S, K, M)
    dp = (st.e_plus_hz[group] - frame_hz[..., None, None]).to(dtype)

    def e(x: torch.Tensor) -> torch.Tensor:
        return x[..., None, None, None, None]

    return (
        st.l0[group]
        + e(beta) * st.k_pump[group]
        + e(ac) * st.drive_a[group]
        + e(as_) * st.drive_b[group]
        + dm[..., None, None] * st.k_m
        + dp[..., None, None] * st.k_p
    )


def _directions(st: Statics, group: torch.Tensor) -> list[torch.Tensor]:
    """dL/d(beta, ac, as, frame_hz) for unique bins of field group ``group`` (U,), each (U, K, M, 14, 14)."""
    k, m = st.weights.shape
    df = -(st.k_m + st.k_p)  # Delta+- = e+- - F
    return [st.k_pump[group], st.drive_a[group], st.drive_b[group], df.expand(group.shape[0], k, m, N_AUG, N_AUG)]


def _expm_taylor(a: torch.Tensor) -> torch.Tensor:
    """exp(a) by scaling and squaring with a Taylor polynomial (Horner), for a batch (N, n, n).

    One scaling s for the whole batch brings every 1-norm to <= 1/2; degree 10 (float32) / 18 (float64) then
    truncates below the working precision (0.5^11/11! ~ 1e-11, 0.5^19/19! ~ 2e-23). Plain matmuls: its autograd
    backward costs ~2x the forward, against ~8x for ``torch.linalg.matrix_exp`` (which exponentiates a 2n x 2n
    block matrix in backward), which matters for training through the surrogate.
    """
    norm = float(a.detach().abs().sum(-2).amax()) if a.numel() else 0.0
    s = max(0, math.ceil(math.log2(norm / 0.5))) if norm > 0.5 else 0
    x = a / (2.0**s)
    deg = 18 if a.dtype == torch.float64 else 10
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    e = eye + x / deg
    for k in range(deg - 1, 0, -1):
        e = eye + (x @ e) / k
    for _ in range(s):
        e = e @ e
    return e


def _expm(x: torch.Tensor, taylor: bool = False, max_batch: int = MAX_EXP_BATCH) -> torch.Tensor:
    flat = x.reshape(-1, N_AUG, N_AUG)
    fn = _expm_taylor if taylor else torch.linalg.matrix_exp
    if flat.shape[0] <= max_batch:
        return fn(flat).reshape(x.shape)
    return torch.cat([fn(p) for p in torch.split(flat, max_batch)]).reshape(x.shape)


def _propagator(
    st: Statics,
    group: torch.Tensor,
    beta: torch.Tensor,
    ac: torch.Tensor,
    as_: torch.Tensor,
    frame_hz: torch.Tensor,
    fill: torch.Tensor | None,
    dt_s: float,
    taylor: bool = False,
) -> torch.Tensor:
    """Bin propagator, accumulator column zeroed (each bin's photon count starts from 0).

    With ``fill`` f, MW is on for the first f*dt of the bin and off for the rest.
    """
    if fill is None:
        p = _expm(_generator(st, group, beta, ac, as_, frame_hz) * dt_s, taylor)
    else:
        on_l = _generator(st, group, beta, ac, as_, frame_hz) * (dt_s * fill)[..., None, None, None, None]
        z = torch.zeros_like(ac)
        off_l = _generator(st, group, beta, z, z, frame_hz) * (dt_s * (1 - fill))[..., None, None, None, None]
        p = _expm(off_l, taylor) @ _expm(on_l, taylor)
    return p * _KEEP_COLS.to(p.dtype).to(p.device)


def _frechet(a: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Directional derivative of exp at ``a`` along ``e`` (both (..., n, n)): upper block of exp([[a, e], [0, a]])."""
    n = a.shape[-1]
    shape = a.shape
    a2, e2 = a.reshape(-1, n, n), e.expand(shape).reshape(-1, n, n)
    out = []
    for i in range(0, a2.shape[0], MAX_EXP_BATCH // 4):
        aa, ee = a2[i : i + MAX_EXP_BATCH // 4], e2[i : i + MAX_EXP_BATCH // 4]
        blk = torch.zeros(aa.shape[0], 2 * n, 2 * n, dtype=a.dtype, device=a.device)
        blk[:, :n, :n] = aa
        blk[:, n:, n:] = aa
        blk[:, :n, n:] = ee
        out.append(torch.linalg.matrix_exp(blk)[:, :n, n:])
    return torch.cat(out).reshape(shape)


@torch.no_grad()
def _propagator_and_derivatives(
    st: Statics,
    group: torch.Tensor,
    beta: torch.Tensor,
    ac: torch.Tensor,
    as_: torch.Tensor,
    frame_hz: torch.Tensor,
    fill: torch.Tensor | None,
    dt_s: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """P and dP/d(beta, ac, as, frame_hz[, fill]) for unique bins (U,), exact (Frechet derivatives)."""
    dirs = _directions(st, group)
    keep = _KEEP_COLS.to(st.l0.dtype).to(st.l0.device)
    if fill is None:
        x = _generator(st, group, beta, ac, as_, frame_hz) * dt_s
        p = torch.linalg.matrix_exp(x.reshape(-1, N_AUG, N_AUG)).reshape(x.shape)
        ds = [_frechet(x, d * dt_s) for d in dirs]
    else:
        f = fill[..., None, None, None, None]
        l_on = _generator(st, group, beta, ac, as_, frame_hz)
        z = torch.zeros_like(ac)
        l_off = _generator(st, group, beta, z, z, frame_hz)
        x_on, x_off = l_on * (dt_s * f), l_off * (dt_s * (1 - f))
        e_on = torch.linalg.matrix_exp(x_on.reshape(-1, N_AUG, N_AUG)).reshape(x_on.shape)
        e_off = torch.linalg.matrix_exp(x_off.reshape(-1, N_AUG, N_AUG)).reshape(x_off.shape)
        p = e_off @ e_on
        ds = []
        for i, d in enumerate(dirs):
            on_part = e_off @ _frechet(x_on, d * (dt_s * f))
            if i in (1, 2):  # the drive is off in the second part
                ds.append(on_part)
            else:
                ds.append(_frechet(x_off, d * (dt_s * (1 - f))) @ e_on + on_part)
        ds.append(dt_s * (e_off @ (l_on - l_off) @ e_on))
    return p * keep, [d * keep for d in ds]


_KEEP_COLS = torch.ones(N_AUG)
_KEEP_COLS[ACC] = 0.0


def _initial_state(batch: int, st: Statics) -> torch.Tensor:
    """Fully mixed ground spin, nothing excited (docs/physics.md section 0); (B, K, M, 14) with accumulator."""
    k, m = st.weights.shape
    r = torch.zeros(batch, k, m, N_AUG, dtype=st.l0.dtype, device=st.l0.device)
    r[..., :3] = 1.0 / 3.0
    return r


def _step(r: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """One bin: the new state; its accumulator entry ([..., ACC]) is that bin's photons per NV."""
    return (p @ r[..., None])[..., 0]


def _fold_rotation(p: torch.Tensor, cos_t: torch.Tensor, sin_t: torch.Tensor) -> torch.Tensor:
    """P @ R(theta): the frame switch at the start of the bin folded into its propagator (columns 3-6)."""
    c_, s_ = cos_t[..., None], sin_t[..., None]
    cols = [
        p[..., :3],
        p[..., 3:4] * c_ - p[..., 4:5] * s_,
        p[..., 3:4] * s_ + p[..., 4:5] * c_,
        p[..., 5:6] * c_ - p[..., 6:7] * s_,
        p[..., 5:6] * s_ + p[..., 6:7] * c_,
        p[..., 7:],
    ]
    return torch.cat(cols, dim=-1)


# --- Public entry point -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelOptions:
    """Numerical options of the surrogate (not physics)."""

    #: Number of static-detuning quadrature samples M (docs/physics.md section 4).
    n_detuning: int = 32
    #: Working dtype of the propagators (float32 on GPUs; float64 for gradient checks).
    dtype: torch.dtype = torch.float32
    #: Bins per chunk when gradients are needed (memory bound, checkpointed). 0 = choose from MAX_EXP_BATCH.
    chunk_bins: int = 0
    #: Share propagators between identical bins when no gradient is needed.
    dedup: bool = True
    #: Matrix exponential: "torch" (torch.linalg.matrix_exp), "taylor" (_expm_taylor, cheap backward) or "auto"
    #: (taylor when differentiating, torch otherwise).
    expm: str = "auto"
    #: Recompute each chunk in backward instead of storing its propagators (less memory, ~1.5x time).
    checkpoint: bool = True
    #: With gradients, share propagators (plus exact first derivatives) between identical bins when the distinct
    #: bins are at most this fraction of all bins; otherwise every bin gets its own propagator.
    shared_grad_max_fraction: float = 0.25
    #: Detuning quadrature rule, "quantile" (equal weights; pulsed protocols) or "uniform" (weighted, evenly
    #: spaced; CW lineshapes), and the "uniform" node spacing in units of the distribution width (section 4).
    detuning_rule: str = "quantile"
    detuning_spacing: float = 0.5


def emission_per_nv(
    arrays: SequenceArrays,
    env: EnvParams,
    *,
    options: ModelOptions | None = None,
    mw_fill: torch.Tensor | None = None,
    laser_factor: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Mean photon emission rate per NV, averaged over orientations and detunings, per bin (1/s), (B, nb).

    ``arrays`` is a (batched or single) :class:`SequenceArrays`, NumPy or torch; tensors that require grad are
    differentiated through (laser power, MW amplitude, frequency, phase). ``mw_fill`` (same shape, in [0, 1]) is the
    fraction of each MW bin during which the MW is actually on (at the start of the bin): continuous pulse durations
    for gradients. ``laser_factor`` multiplies the pump rate per bin (laser drift, max(0, 1 + x)).
    """
    opt = options or ModelOptions()
    dev = torch.device(device) if device is not None else _device_of(arrays)
    a = arrays.to_torch(dev)
    assert isinstance(a.laser, torch.Tensor)
    single = a.laser.ndim == 1

    def b2(x: torch.Tensor) -> torch.Tensor:
        return x[None] if single else x

    laser = b2(torch.as_tensor(a.laser)).to(opt.dtype)
    amp = b2(torch.as_tensor(a.mw_amplitude)).to(opt.dtype)
    freq = b2(torch.as_tensor(a.mw_frequency_hz)).to(torch.float64)
    phase = b2(torch.as_tensor(a.mw_phase_rad)).to(opt.dtype)
    magnet = torch.as_tensor(a.magnet_t).reshape(-1, 3)
    if laser.ndim != 2:
        raise ValueError(f"expected one batch axis, got laser shape {tuple(laser.shape)}")
    bsz, nb = laser.shape
    fill = None if mw_fill is None else b2(torch.as_tensor(mw_fill, device=dev)).to(opt.dtype)
    if laser_factor is not None:
        laser = laser * b2(torch.as_tensor(laser_factor, device=dev)).to(opt.dtype)

    # Static fields per sequence -> groups.
    fields = np.asarray(env.b_field_t, dtype=np.float64)[None] + magnet.detach().cpu().numpy().astype(np.float64)
    ufields, ginv = np.unique(fields, axis=0, return_inverse=True)
    st = build_statics(
        env,
        ufields,
        opt.n_detuning,
        opt.dtype,
        dev,
        detuning_rule=opt.detuning_rule,
        detuning_spacing=opt.detuning_spacing,
    )
    group = torch.as_tensor(ginv.reshape(-1), device=dev)
    if group.shape[0] == 1 and bsz > 1:
        group = group.expand(bsz)

    beta = laser * env.optical_beta_full_scale
    ac, as_ = amp * torch.cos(phase), amp * torch.sin(phase)
    frame = frame_frequency(freq, env.d_hz)
    theta = frame_phase(frame, a.grid_ns)
    dt_s = a.grid_ns * 1e-9
    g2 = group[:, None].expand(bsz, nb)
    controls = [beta, ac, as_, frame] + ([] if fill is None else [fill])

    need_grad = torch.is_grad_enabled() and any(x.requires_grad for x in controls)
    ukeys: torch.Tensor | None = None
    if opt.dedup:
        ukeys, inv = _unique_bins([g2, *controls])
    if ukeys is not None and not need_grad and ukeys.shape[0] <= 0.5 * bsz * nb:
        with torch.no_grad():
            em = _run_dedup(st, ukeys, inv, theta, fill is not None, dt_s)
    elif ukeys is not None and ukeys.shape[0] <= opt.shared_grad_max_fraction * bsz * nb:
        em = _run_shared_grad(st, ukeys, inv, controls, theta, dt_s, opt)
    else:
        taylor = opt.expm == "taylor" or (opt.expm == "auto" and need_grad)
        cb = _chunk_bins(opt.chunk_bins, bsz, st, 2 if fill is not None else 1)

        def make_p(sl: slice) -> torch.Tensor:
            fl = None if fill is None else fill[:, sl]
            return _propagator(st, g2[:, sl], beta[:, sl], ac[:, sl], as_[:, sl], frame[:, sl], fl, dt_s, taylor)

        em = _run_chunks(st, bsz, nb, make_p, theta, cb, opt.checkpoint)
    rate = torch.einsum("bnkm,km->bn", em, st.weights) / dt_s
    return rate[0] if single else rate


def expected_rate_cps(
    arrays: SequenceArrays,
    env: EnvParams,
    *,
    options: ModelOptions | None = None,
    mw_fill: torch.Tensor | None = None,
    laser_factor: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Noiseless detected count rate (counts/s) per bin, docs/physics.md section 7; differentiable, (B, nb)."""
    em = emission_per_nv(arrays, env, options=options, mw_fill=mw_fill, laser_factor=laser_factor, device=device)
    laser = torch.as_tensor(arrays.laser, device=em.device).to(em.dtype)
    if laser_factor is not None:
        laser = laser * torch.as_tensor(laser_factor, device=em.device).to(em.dtype)
    if laser.shape != em.shape:
        laser = laser.reshape(em.shape)
    return env.collection_efficiency * env.n_nv * em + env.background_cps * laser + env.dark_cps


def _device_of(arrays: SequenceArrays) -> torch.device:
    if isinstance(arrays.laser, torch.Tensor):
        return arrays.laser.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _unique_bins(keys: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Distinct rows of the per-bin control keys (each (B, nb), detached): (U, n_keys) float64 and inverse (B, nb)."""
    k = torch.stack([x.detach().to(torch.float64) for x in keys], dim=-1)
    ukeys, inv = torch.unique(k.reshape(-1, len(keys)), dim=0, return_inverse=True)
    return ukeys, inv.reshape(k.shape[:2])


def _run_dedup(
    st: Statics,
    ukeys: torch.Tensor,
    inv: torch.Tensor,
    theta: torch.Tensor,
    has_fill: bool,
    dt_s: float,
) -> torch.Tensor:
    """Forward only: one propagator per distinct (field, beta, ac, as, frame, fill) bin."""
    bsz, nb = inv.shape
    dtype = st.l0.dtype
    fill = ukeys[:, 5].to(dtype) if has_fill else None
    p_u = _propagator(
        st,
        ukeys[:, 0].long(),
        ukeys[:, 1].to(dtype),
        ukeys[:, 2].to(dtype),
        ukeys[:, 3].to(dtype),
        ukeys[:, 4],
        fill,
        dt_s,
    )  # (U, K, M, 14, 14)
    cos_t, sin_t = torch.cos(theta).to(dtype), torch.sin(theta).to(dtype)
    rot = (theta != 0).any(0).tolist()
    same = (inv == inv[:1]).all(0).tolist()
    r = _initial_state(bsz, st)
    out = []
    for b in range(nb):
        if rot[b]:
            r = _rotate(r, cos_t[:, b, None, None], sin_t[:, b, None, None])
        p = p_u[inv[0, b]] if same[b] else p_u[inv[:, b]]
        r = _step(r, p)
        out.append(r[..., ACC])
    return torch.stack(out, dim=1)


def _run_chunks(
    st: Statics,
    bsz: int,
    nb: int,
    make_p: Callable[[slice], torch.Tensor],
    theta: torch.Tensor,
    chunk_bins: int,
    use_checkpoint: bool,
) -> torch.Tensor:
    """Step through the bins chunk by chunk; ``make_p(slice)`` gives the chunk's propagators (B, c, K, M, 14, 14).

    The frame switch of each bin is folded into its propagator. With gradients, each chunk is checkpointed:
    backward recomputes its propagators instead of storing them.
    """
    dtype = st.l0.dtype
    cos_t, sin_t = torch.cos(theta).to(dtype), torch.sin(theta).to(dtype)

    def chunk(r: torch.Tensor, start: int, stop: int) -> tuple[torch.Tensor, torch.Tensor]:
        sl = slice(start, stop)
        p = _fold_rotation(make_p(sl), cos_t[:, sl, None, None, None], sin_t[:, sl, None, None, None])
        states = []
        for i in range(stop - start):
            r = _step(r, p[:, i])
            states.append(r)
        # One stack + one slice (not a slice per step): far fewer autograd nodes in backward.
        return r, torch.stack(states, dim=1)[..., ACC]

    fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = chunk
    r = _initial_state(bsz, st)
    out = []
    for s in range(0, nb, chunk_bins):
        e = min(nb, s + chunk_bins)
        if torch.is_grad_enabled() and use_checkpoint:
            r, acc = checkpoint(fn, r, s, e, use_reentrant=False)
        else:
            r, acc = fn(r, s, e)
        out.append(acc)
    return torch.cat(out, dim=1)


def _chunk_bins(opt_chunk: int, bsz: int, st: Statics, per_bin: int) -> int:
    k, m = st.weights.shape
    budget = MAX_EXP_BATCH_GRAD if torch.is_grad_enabled() else MAX_EXP_BATCH
    # At least MIN_CHUNK_BINS: tiny chunks make the run launch-bound (each chunk is a checkpoint region).
    return opt_chunk or max(MIN_CHUNK_BINS, budget // max(1, bsz * k * m * per_bin))


def _run_shared_grad(
    st: Statics,
    ukeys: torch.Tensor,
    inv: torch.Tensor,
    controls: list[torch.Tensor],
    theta: torch.Tensor,
    dt_s: float,
    opt: ModelOptions,
) -> torch.Tensor:
    """Differentiable with shared propagators: P_b = P_u + sum_i (c_bi - c_bi.detach()) dP_u/dc_i.

    The added term is exactly zero in value, and its gradient with respect to each bin's own controls c_b is the
    exact first derivative <dLoss/dP_b, dP_u/dc_i>: per-bin gradients stay exact (no averaging over the bins that
    share u). Second derivatives are not available on this path.
    """
    bsz, nb = inv.shape
    dtype = st.l0.dtype
    has_fill = len(controls) == 5
    p_u, d_u = _propagator_and_derivatives(
        st,
        ukeys[:, 0].long(),
        ukeys[:, 1].to(dtype),
        ukeys[:, 2].to(dtype),
        ukeys[:, 3].to(dtype),
        ukeys[:, 4],
        ukeys[:, 5].to(dtype) if has_fill else None,
        dt_s,
    )
    deltas = [(c - c.detach()).to(dtype) for c in controls]
    cb = _chunk_bins(opt.chunk_bins, bsz, st, 1 + len(deltas))

    def make_p(sl: slice) -> torch.Tensor:
        idx = inv[:, sl]
        p = p_u[idx]
        for dlt, d in zip(deltas, d_u, strict=True):
            if dlt.requires_grad:
                p = p + dlt[:, sl, None, None, None, None] * d[idx]
        return p

    return _run_chunks(st, bsz, nb, make_p, theta, cb, opt.checkpoint)


def mw_fill_for_duration(start_ns: int, duration_ns: torch.Tensor, n_bins: int, grid_ns: int) -> torch.Tensor:
    """Per-bin MW fill fractions (n_bins,) for a pulse starting on the grid with a continuous duration (ns).

    Bin b gets clamp((start + duration - b * grid) / grid, 0, 1) from the start bin on: the pulse covers whole bins
    and the last bin partly, its on-part at the start of that bin (the ``mw_fill`` convention). Differentiable in
    ``duration_ns``; pass it as ``mw_fill`` with the MW amplitude set on every bin the pulse may reach.
    """
    if start_ns % grid_ns:
        raise ValueError("start_ns must be on the grid")
    t0 = torch.arange(n_bins, dtype=duration_ns.dtype, device=duration_ns.device) * grid_ns
    fill = torch.clamp((start_ns + duration_ns - t0) / grid_ns, 0.0, 1.0)
    return torch.where(t0 >= start_ns, fill, torch.zeros_like(fill))
