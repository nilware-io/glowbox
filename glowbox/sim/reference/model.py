"""Per-orientation operators of the seven-level NV model (docs/physics.md sections 1-6), built with QuTiP.

Everything here is for one NV axis. The state space is 7-dimensional, ordered

    0, 1, 2  ground triplet in the eigenbasis of H0: |0~>, |-~>, |+~> (|0~> = the state with most ms=0 weight,
             |-~> / |+~> = the lower / upper of the two others)
    3, 4, 5  excited triplet, ms = 0, -1, +1 of the NV axis (Tetienne |4>, |5>, |6>)
    6        metastable singlet (Tetienne |7>)

Ground-state operators of the spec are defined in the ms basis (0, -1, +1) and transformed to the eigenbasis with
the eigenvector matrix V (columns = eigenvectors in the ms basis). The rotating frame (spec section 3) rotates the
two upper ground eigenstates, indices 1 and 2, at the MW frame frequency.

Units inside the solver: time in ns, so Hamiltonian terms are in rad/ns (2 pi * Hz * 1e-9) and rates in 1/ns
(1/s * 1e-9). Superoperators act on column-stacked density matrices (QuTiP's convention: index i + 7 j for
rho[i, j]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import qutip as qt

from glowbox.sim import constants as c

#: Dimension of the seven-level model.
N_LEVELS = 7
#: Hz -> rad/ns.
HZ_TO_RAD_PER_NS = 2.0 * math.pi * 1e-9
#: 1/s -> 1/ns.
PER_S_TO_PER_NS = 1e-9
#: Rotation class of each level in the MW rotating frame: 1 = rotates at the frame frequency (upper ground states).
ROTATING = np.array([0, 1, 1, 0, 0, 0, 0])
#: Excited-state index of ground ms index m (ms order 0, -1, +1).
EXCITED = (3, 4, 5)
SINGLET = 6


def spin1_ms_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spin-1 operators (Sx, Sy, Sz) in the basis ordered ms = (0, -1, +1), dimensionless."""
    s_plus = np.zeros((3, 3), dtype=np.complex128)
    s_plus[0, 1] = math.sqrt(2.0)  # S+ |-1> = sqrt2 |0>
    s_plus[2, 0] = math.sqrt(2.0)  # S+ |0>  = sqrt2 |+1>
    s_minus = s_plus.conj().T
    sx = (s_plus + s_minus) / 2.0
    sy = (s_plus - s_minus) / 2.0j
    sz = np.diag([0.0, -1.0, 1.0]).astype(np.complex128)
    return sx, sy, sz


def local_frame(axis: np.ndarray) -> np.ndarray:
    """Rows x, y, z of the NV-local frame (spec section 1): z = axis, x = normalised projection of [001]."""
    z = np.asarray(axis, dtype=np.float64)
    z = z / np.linalg.norm(z)
    e001 = np.array([0.0, 0.0, 1.0])
    p = e001 - (e001 @ z) * z
    x = p / np.linalg.norm(p)  # never degenerate for <111> axes
    y = np.cross(z, x)
    return np.stack([x, y, z])


@dataclass(frozen=True)
class AxisGeometry:
    """Field and MW drive of one NV axis, in its local frame.

    ``b_local_hz`` = gamma_e * (Bx, By, Bz) local, Hz. ``u_local`` = component of the MW field direction
    perpendicular to the axis, local (x, y), *not* normalised: its norm is |m_perp| (spec section 3).
    """

    b_local_hz: tuple[float, float, float]
    u_local: tuple[float, float]

    def key(self) -> tuple[float, ...]:
        """Rounded values for detecting physically identical axes (dedupe)."""
        return (*(round(v, 3) for v in self.b_local_hz), *(round(v, 12) for v in self.u_local))


def axis_geometry(axis: np.ndarray, b_field_t: np.ndarray, mw_dir: np.ndarray, canonical: bool) -> AxisGeometry:
    """Project the crystal-frame field and MW direction onto one axis' local frame.

    ``canonical=True`` (only valid when E = 0) rotates the local x, y about the axis so that B_perp lies along x
    (or, with B_perp = 0, the MW perpendicular component does). With E = 0 the model is invariant under such a
    rotation: it is the unitary exp(-i a Sz), which leaves every jump operator (all diagonal or single-element in
    the ms basis) unchanged up to a global phase. It lets axes that are equivalent by symmetry share one
    simulation.
    """
    frame = local_frame(axis)
    b = frame @ np.asarray(b_field_t, dtype=np.float64) * c.GAMMA_E_HZ_PER_T
    u = frame @ np.asarray(mw_dir, dtype=np.float64)
    bx, by, bz = float(b[0]), float(b[1]), float(b[2])
    ux, uy = float(u[0]), float(u[1])
    if canonical:
        if math.hypot(bx, by) > 1e-3:  # 1 mHz: B_perp present
            ang = math.atan2(by, bx)
        elif math.hypot(ux, uy) > 1e-12:
            ang = math.atan2(uy, ux)
        else:
            ang = 0.0
        ca, sa = math.cos(ang), math.sin(ang)
        bx, by = ca * bx + sa * by, -sa * bx + ca * by
        ux, uy = ca * ux + sa * uy, -sa * ux + ca * uy
    return AxisGeometry((bx, by, bz), (ux, uy))


@dataclass
class AxisModel:
    """Liouvillian building blocks for one axis. All superoperators are dense (49, 49) complex arrays, 1/ns.

    L(bin) = diss_fixed + beta * diss_pump
             + Delta_minus * ham_minus + Delta_plus * ham_plus + delta * ham_delta
             + f_R * (e^{-i phi} ham_k + e^{+i phi} ham_kdag)

    with Delta_pm = E_pm - f_frame in Hz, delta the static broadening shift in Hz (spec section 4: a term
    delta * Sz in H0, which gives f+- -> f+- +- delta when gamma_e B_par >> E, gamma_e B_perp), f_R =
    rabi_full_scale_hz * a * |m_perp| in Hz.
    """

    #: Eigenvectors of H0 (columns, ms basis (0, -1, +1)) ordered |0~>, |-~>, |+~>.
    v: np.ndarray
    #: Energies of |-~>, |+~> above |0~>, Hz (the transition frequencies f-, f+).
    f_minus_hz: float
    f_plus_hz: float
    #: |m_perp| of this axis (Rabi frequency per unit amplitude = rabi_full_scale_hz * mperp).
    mperp: float
    diss_fixed: np.ndarray
    diss_pump: np.ndarray
    ham_minus: np.ndarray
    ham_plus: np.ndarray
    #: -i [Sz, .] per Hz of static detuning delta, Sz written in the H0 eigenbasis and rotating frame (RWA: its
    #: 0~ <-> +-~ elements oscillate at the frame frequency and are dropped, spec section 4).
    ham_delta: np.ndarray
    ham_k: np.ndarray
    ham_kdag: np.ndarray
    #: Row vector w with w . vec(rho) = photon emission rate, photons per ns (k_r * excited population).
    emission: np.ndarray


def diagonalise(d_hz: float, e_hz: float, b_local_hz: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose H0/h = D Sz^2 + E (Sx^2 - Sy^2) + gamma_e B.S (Hz) in the ms basis.

    Returns (energies relative to |0~>, Hz, shape (3,)), (V, columns ordered |0~>, |-~>, |+~>).
    """
    sx, sy, sz = spin1_ms_basis()
    bx, by, bz = b_local_hz
    h0 = d_hz * sz @ sz + e_hz * (sx @ sx - sy @ sy) + bx * sx + by * sy + bz * sz
    w, vec = np.linalg.eigh(h0)
    i0 = int(np.argmax(np.abs(vec[0, :]) ** 2))
    rest = sorted((j for j in range(3) if j != i0), key=lambda j: w[j])
    order = [i0, *rest]
    return w[order] - w[i0], vec[:, order]


def _embed_ground(op3: np.ndarray) -> np.ndarray:
    out = np.zeros((N_LEVELS, N_LEVELS), dtype=np.complex128)
    out[:3, :3] = op3
    return out


def _ket(i: int) -> np.ndarray:
    k = np.zeros(N_LEVELS, dtype=np.complex128)
    k[i] = 1.0
    return k


def secular_parts(op: np.ndarray) -> list[np.ndarray]:
    """Split a jump operator into its components of fixed rotating-frame frequency (-f, 0, +f).

    In the frame rotating at f the component with r(a) - r(b) = w picks up exp(i w 2 pi f t). Averaging the
    dissipator over the ~GHz oscillation (the same approximation as the RWA) leaves sum_w D[L_w]. When V is the
    identity (B along the axis, E = 0) every operator is a single component and this is exact.
    """
    diff = ROTATING[:, None] - ROTATING[None, :]
    parts = []
    for w in (-1, 0, 1):
        p = np.where(diff == w, op, 0.0)
        if np.any(np.abs(p) > 0):
            parts.append(p)
    return parts


def _dissipator(ops: list[np.ndarray]) -> np.ndarray:
    total = np.zeros((N_LEVELS**2, N_LEVELS**2), dtype=np.complex128)
    for op in ops:
        for part in secular_parts(op):
            total += qt.lindblad_dissipator(qt.Qobj(part)).full()
    return total


def _commutator(h: np.ndarray) -> np.ndarray:
    """Superoperator of -i [h, .] with h in rad/ns."""
    q = qt.Qobj(h)
    return np.asarray((-1j * (qt.spre(q) - qt.spost(q))).full())


def build_axis_model(
    geom: AxisGeometry,
    d_hz: float,
    e_hz: float,
    t1_s: float,
    t2_s: float,
) -> AxisModel:
    """Assemble the superoperator pieces for one axis (spec sections 1, 3, 5, 6)."""
    energies, v = diagonalise(d_hz, e_hz, geom.b_local_hz)
    sx, sy, sz = spin1_ms_basis()
    vh = v.conj().T

    def ground_ket(m: int) -> np.ndarray:
        """|ms=m> (m = ms index 0, 1, 2 for ms 0, -1, +1) in the 7-level eigen coordinates."""
        k = np.zeros(N_LEVELS, dtype=np.complex128)
        k[:3] = vh[:, m]  # <j~|m> = conj(V[m, j])
        return k

    k_r = c.K_RADIATIVE_PER_S * PER_S_TO_PER_NS
    k_isc = (c.K_ISC_MS0_PER_S, c.K_ISC_MS1_PER_S, c.K_ISC_MS1_PER_S)
    k_sing = (c.K_SINGLET_TO_MS0_PER_S, c.K_SINGLET_TO_MS1_PER_S, c.K_SINGLET_TO_MS1_PER_S)

    fixed: list[np.ndarray] = []
    pump: list[np.ndarray] = []
    for m in range(3):
        g, e = ground_ket(m), _ket(EXCITED[m])
        # Optical pumping g_m -> e_m at beta * k_r (beta multiplies this block at run time). Spin-conserving.
        pump.append(math.sqrt(k_r) * np.outer(e, g.conj()))
        # Radiative decay e_m -> g_m (spin-conserving), ISC e_m -> singlet, singlet -> g_m.
        fixed.append(math.sqrt(k_r) * np.outer(g, e.conj()))
        fixed.append(math.sqrt(k_isc[m] * PER_S_TO_PER_NS) * np.outer(_ket(SINGLET), e.conj()))
        fixed.append(math.sqrt(k_sing[m] * PER_S_TO_PER_NS) * np.outer(g, _ket(SINGLET).conj()))
    # T1: every ordered pair of ground ms levels at 1/(3 T1).
    g1 = 1.0 / (3.0 * t1_s) * PER_S_TO_PER_NS
    for m in range(3):
        for mp in range(3):
            if m != mp:
                fixed.append(math.sqrt(g1) * np.outer(ground_ket(mp), ground_ket(m).conj()))
    # T2: sqrt(2/T2) Sz (ms basis), coherences 0 <-> +-1 decay as exp(-t/T2).
    fixed.append(math.sqrt(2.0 / t2_s * PER_S_TO_PER_NS) * _embed_ground(vh @ sz @ v))

    p_minus = np.zeros((N_LEVELS, N_LEVELS), dtype=np.complex128)
    p_minus[1, 1] = HZ_TO_RAD_PER_NS
    p_plus = np.zeros((N_LEVELS, N_LEVELS), dtype=np.complex128)
    p_plus[2, 2] = HZ_TO_RAD_PER_NS
    # Static detuning delta * Sz (spec section 4) in the eigenbasis. Its part that connects |0~> with |+-~> (non-zero
    # only when B_perp or E mix the ms states) rotates at the frame frequency and is dropped like the RWA drops the
    # counter-rotating drive; the -~/+~ block is kept whole. With V = 1 this is diag(0, -1, +1): f+- -> f+- +- delta.
    # The surrogate diagonalises H0 + delta Sz instead; the two differ only by the dropped 0~/+-~ part, a level shift
    # of order delta^2 (gamma_e B_perp / D)^2 / D, far below any linewidth.
    same_class = ROTATING[:, None] == ROTATING[None, :]
    p_delta = np.where(same_class, _embed_ground(vh @ sz @ v), 0.0) * HZ_TO_RAD_PER_NS

    # MW drive in the RWA (spec section 3): (sqrt2/2) f_R e^{-i phi} <j~|u.S|0~> |j~><0~| + h.c., j = -~, +~,
    # with u the unit perpendicular direction. The f_R factor (incl. |m_perp|) is applied at run time.
    ux, uy = geom.u_local
    mperp = math.hypot(ux, uy)
    k_op = np.zeros((N_LEVELS, N_LEVELS), dtype=np.complex128)
    if mperp > 0:
        mat = vh @ ((ux / mperp) * sx + (uy / mperp) * sy) @ v
        for j in (1, 2):
            k_op[j, 0] = math.sqrt(2.0) / 2.0 * mat[j, 0] * HZ_TO_RAD_PER_NS

    emission = np.zeros(N_LEVELS**2, dtype=np.complex128)
    for i in EXCITED:
        emission[i + N_LEVELS * i] = k_r

    return AxisModel(
        v=v,
        f_minus_hz=float(energies[1]),
        f_plus_hz=float(energies[2]),
        mperp=mperp,
        diss_fixed=_dissipator(fixed),
        diss_pump=_dissipator(pump),
        ham_minus=_commutator(p_minus),
        ham_plus=_commutator(p_plus),
        ham_delta=_commutator(p_delta),
        ham_k=_commutator(k_op),
        ham_kdag=_commutator(k_op.conj().T),
        emission=emission,
    )


def initial_state_vec() -> np.ndarray:
    """Column-stacked rho0: ground spin fully mixed (identity/3 in any ground basis), nothing elsewhere."""
    rho = np.zeros((N_LEVELS, N_LEVELS), dtype=np.complex128)
    rho[0, 0] = rho[1, 1] = rho[2, 2] = 1.0 / 3.0
    return rho.reshape(-1, order="F")


def frame_phase_vec(cycles: float) -> np.ndarray:
    """Elementwise factor that moves a column-stacked rho from frame f1 to f2, cycles = (f2 - f1) * t_s (spec 3).

    rho_ab -> exp(i 2 pi cycles (r_a - r_b)) rho_ab, with r the rotation class.
    """
    theta = 2.0 * math.pi * (cycles % 1.0)
    ph = np.exp(1j * theta * ROTATING)
    return np.outer(ph, ph.conj()).reshape(-1, order="F")


def transition_frequencies_hz(d_hz: float, e_hz: float, b_field_t: np.ndarray) -> np.ndarray:
    """ODMR line positions (f-, f+) of each of the four NV axes, Hz, shape (4, 2), from the numerical
    diagonalisation of H0 (spec section 1; includes the second-order shift by B_perp, no T2* detuning).
    ``b_field_t`` is the total crystal-frame field in tesla."""
    out = np.empty((4, 2))
    for k, axis in enumerate(c.NV_AXES):
        geom = axis_geometry(axis, np.asarray(b_field_t, dtype=np.float64), np.array([1.0, 0.0, 0.0]), False)
        energies, _ = diagonalise(d_hz, e_hz, geom.b_local_hz)
        out[k] = energies[1], energies[2]
    return out
