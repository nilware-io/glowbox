"""Single pulsed-ODMR measurements as discrete actions, for the adaptive-measurement and RL learners (milestone 5).

One *measurement* is one shot of

    init laser (init_ns) | dark settle (settle_ns) | MW pulse (amplitude, frequency f, duration d) | readout laser

and returns the photon counts in a few readout windows (default: the first 300 ns, where the spin contrast lives,
and the next 300 ns, mostly a brightness reference). An *action* picks (f, d) from a fixed grid: ``n_freq``
frequencies times the listed durations, index ``a = d_index * n_freq + f_index``. With the default amplitude 0.6
(f_R = 4.9 MHz at the nominal 10 MHz full scale, MW along [100]) the durations 50, 100, 200, 400 ns are pulse areas
of about pi/2, pi, 2 pi and 4 pi: a learner that picks durations picks Rabi rotation angles. A block of S shots of
one action gives Poisson counts with S times the per-shot mean.

Three ways to get the per-shot means:

* :meth:`MeasurementSpec.sequence`: the measurement as a :class:`~glowbox.control.pulse.PulseSequence`, for any
  :class:`~glowbox.sim.interface.Simulator` (``run()``: the QuTiP reference, the surrogate, later hardware).
* :func:`measurement_means`: the surrogate's physics for many fields and every action at once, with the
  shortcuts of :func:`emission_table` (equal to the bin path to round-off, tests/learn).
* :class:`AxisTable`: a fast *emulator* of :func:`measurement_means` for arbitrary fields, used as the likelihood
  of the Bayesian learners. The ensemble signal is a sum over the four NV orientations (docs/physics.md 7), and
  each orientation's response depends on the field only through B_par, |B_perp| and the azimuth of B_perp
  relative to the MW's transverse direction (:func:`axis_coordinates`; the azimuth matters where an axis is
  nearly perpendicular to B and its levels mix). So one single-orientation table on that 3-D grid, interpolated
  and summed over the four axes, gives the means of any field. Its error against the full surrogate is measured
  (:func:`emulator_error`) and reported; the learners' *observations* always come from the full surrogate (or
  another ``run()``), so the emulator only enters as the learners' model, never as the truth.

Assumption: the device is calibrated (known ``EnvParams`` apart from the hidden field: T2*, Rabi scale, optics,
background, E = 0). Milestones 2-4 marginalised over a nuisance prior; milestone 5 does not (docs/results/m5-*).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch

from glowbox.control.pulse import MagnetSetting, PulseSequence, SequenceBuilder
from glowbox.learn.reservoir import init_prefix
from glowbox.sim import constants as c
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import (
    ACC,
    N_AUG,
    ModelOptions,
    Statics,
    _generator,
    build_statics,
    emission_and_state,
)

#: MW direction perpendicular fraction for every axis with the default MW along [100].
MW_PERP_DEFAULT = math.sqrt(2.0 / 3.0)


@dataclass(frozen=True)
class MeasurementSpec:
    """The action grid and the shot layout (assumptions of the experiment, not physics)."""

    f_start_hz: float = 2.77e9
    f_stop_hz: float = 2.97e9
    step_hz: float = 2e6
    durations_ns: tuple[int, ...] = (50, 100, 200, 400)
    amplitude: float = 0.6
    init_ns: int = 1500
    settle_ns: int = 1000
    #: Readout windows, ns, back to back from the start of the readout laser.
    windows_ns: tuple[int, ...] = (300, 300)
    #: Every MW frequency is shifted by this (the quantum test's detuned arm).
    mw_offset_hz: float = 0.0
    grid_ns: int = 10

    def __post_init__(self) -> None:
        g = self.grid_ns
        for name, vals in (("durations_ns", self.durations_ns), ("windows_ns", self.windows_ns)):
            if not vals or any(v <= 0 or v % g for v in vals):
                raise ValueError(f"{name} must be positive multiples of the grid")
        if self.init_ns % g or self.settle_ns % g:
            raise ValueError("init_ns and settle_ns must be multiples of the grid")
        if not 0 < self.amplitude <= 1:
            raise ValueError("amplitude must be in (0, 1]")

    def freqs_hz(self) -> np.ndarray:
        """Nominal grid frequencies (before ``mw_offset_hz``), (F,)."""
        n = round((self.f_stop_hz - self.f_start_hz) / self.step_hz)
        return self.f_start_hz + self.step_hz * np.arange(n)

    @property
    def n_freq(self) -> int:
        return int(self.freqs_hz().shape[0])

    @property
    def n_dur(self) -> int:
        return len(self.durations_ns)

    @property
    def n_actions(self) -> int:
        return self.n_freq * self.n_dur

    @property
    def n_windows(self) -> int:
        return len(self.windows_ns)

    @property
    def readout_ns(self) -> int:
        return int(sum(self.windows_ns))

    def detuned(self, offset_hz: float) -> MeasurementSpec:
        return dataclasses.replace(self, mw_offset_hz=offset_hz)

    def action_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """(applied MW frequency (A,) Hz incl. the offset, duration (A,) ns) of every action."""
        f = self.freqs_hz() + self.mw_offset_hz
        d = np.asarray(self.durations_ns, dtype=np.int64)
        return np.tile(f, self.n_dur), np.repeat(d, self.n_freq)

    def split(self, action: np.ndarray | int) -> tuple[np.ndarray, np.ndarray]:
        """(frequency index, duration index) of action indices."""
        a = np.asarray(action)
        return a % self.n_freq, a // self.n_freq

    def shot_ns(self, action: np.ndarray | int) -> np.ndarray:
        """Length of one shot of each action, ns (the MW pulse differs)."""
        _, di = self.split(action)
        d = np.asarray(self.durations_ns)[di]
        out: np.ndarray = self.init_ns + self.settle_ns + d + self.readout_ns
        return out

    def pulse_area_pi(self, env: EnvParams, mw_perp: float = MW_PERP_DEFAULT) -> np.ndarray:
        """Resonant rotation angle / pi of each duration at the device's Rabi scale, (n_dur,)."""
        f_r = env.rabi_full_scale_hz * mw_perp * self.amplitude
        out: np.ndarray = 2.0 * f_r * np.asarray(self.durations_ns, dtype=np.float64) * 1e-9
        return out

    def sequence(self, action: int, magnet: MagnetSetting | None = None) -> PulseSequence:
        """The measurement as a validated PulseSequence (for any ``Simulator.run``)."""
        f, d = self.action_grid()
        b = SequenceBuilder(self.grid_ns, magnet).laser(self.init_ns).wait(self.settle_ns)
        b.mw(int(d[action]), float(f[action]), 0.0, self.amplitude).laser(self.readout_ns)
        return b.build(protocol="m5_measurement", action=int(action), frequency_hz=float(f[action]))

    def window_slices(self, action: int) -> list[slice]:
        """Bin slices of the readout windows in :meth:`sequence` (for counts from a trace)."""
        _, d = self.action_grid()
        start = (self.init_ns + self.settle_ns + int(d[action])) // self.grid_ns
        out = []
        for w in self.windows_ns:
            n = w // self.grid_ns
            out.append(slice(start, start + n))
            start += n
        return out

    def background_counts(self, env: EnvParams) -> np.ndarray:
        """MW- and field-independent counts per shot per window (background + dark), (W,)."""
        w_s = np.asarray(self.windows_ns, dtype=np.float64) * 1e-9
        out: np.ndarray = (env.background_cps + env.dark_cps) * w_s
        return out


def _readout_functionals(st: Statics, spec: MeasurementSpec, beta: float) -> torch.Tensor:
    """Linear maps from the post-MW state to each window's photons per NV, (G, K, M, W, 14).

    The readout is MW-free: its generator couples the populations (and the frame-independent -~/+~ coherence) only
    among themselves (secular rule, docs/physics.md 3), so the emission does not depend on the frame frequency and
    one map per field and detuning node serves every action (checked against the run path in tests/learn)."""
    g, k, m = st.l0.shape[:3]
    grp = torch.arange(g, device=st.l0.device)
    dt, dev = st.l0.dtype, st.l0.device
    z = torch.zeros(g, dtype=dt, device=dev)
    # Any frame gives the same emission; one near the lines keeps the exponentials' norms small.
    frame = torch.full((g,), float(spec.freqs_hz().mean()), dtype=torch.float64, device=dev)
    lro = _generator(st, grp, z + beta, z, z, frame)
    lro = lro.to(torch.float64)
    keep = torch.ones(N_AUG, dtype=torch.float64, device=st.l0.device)
    keep[ACC] = 0.0
    eye_k = torch.diag(keep)
    e_acc = torch.zeros(N_AUG, dtype=torch.float64, device=st.l0.device)
    e_acc[ACC] = 1.0
    chain = eye_k.expand(g, k, m, N_AUG, N_AUG)  # maps the post-MW state to the state at the current window start
    out = []
    for w in spec.windows_ns:
        p = torch.linalg.matrix_exp(lro * (w * 1e-9))
        out.append(torch.einsum("i,gkmij,gkmjl->gkml", e_acc, p, chain))
        chain = eye_k @ p @ chain
    return torch.stack(out, dim=3).to(st.l0.dtype)


@torch.no_grad()
def emission_table(
    spec: MeasurementSpec,
    env: EnvParams,
    fields_t: np.ndarray,
    options: ModelOptions,
    device: torch.device | str,
    chunk: int = 0,
) -> torch.Tensor:
    """Photons emitted per NV per shot in each readout window, (N, A, W) float64 on ``device``, for every field
    (total field = ``env.b_field_t`` + row) and every action, averaged over the orientations kept by
    ``env.orientation_populations`` with their weights.

    Exact surrogate physics with three shortcuts (all checked against ``SurrogateSimulator.run`` in tests/learn):
    the initialisation runs once per field (MW-free, so no coherence enters the measurement); the MW pulse is one
    propagator exp(L d) per (field, frequency, node), the longer durations by squaring when they double; the
    readout is a fixed linear functional of the post-MW state (:func:`_readout_functionals`). ``chunk`` bounds
    the (field, frequency) pairs per batch (0: from the memory budget)."""
    dev = torch.device(device)
    fields = np.atleast_2d(np.asarray(fields_t, dtype=np.float64)) + np.asarray(env.b_field_t)[None]
    env0 = dataclasses.replace(env, b_field_t=(0.0, 0.0, 0.0))
    # Statics for the distinct fields in np.unique order (what emission_and_state expects); inv maps rows to them.
    ufields, inv = np.unique(fields, axis=0, return_inverse=True)
    inv_t = torch.as_tensor(inv.reshape(-1), device=dev)
    st = build_statics(
        env0,
        ufields,
        options.n_detuning,
        options.dtype,
        dev,
        detuning_rule=options.detuning_rule,
        detuning_spacing=options.detuning_spacing,
    )
    pre = init_prefix(spec.init_ns, spec.settle_ns, grid_ns=spec.grid_ns)
    k = pre.n_bins
    ng = ufields.shape[0]
    pre_b = dataclasses.replace(
        pre,
        laser=np.repeat(np.asarray(pre.laser), ng, axis=0),
        mw_amplitude=np.zeros((ng, k), np.float32),
        mw_frequency_hz=np.zeros((ng, k)),
        mw_phase_rad=np.zeros((ng, k), np.float32),
        magnet_t=ufields,
    )
    _, state = emission_and_state(pre_b, env0, options=options, device=dev, statics=st)  # (G, K, M, 14)
    ro = _readout_functionals(st, spec, env.optical_beta_full_scale)  # (G, K, M, W, 14)
    freqs = torch.as_tensor(spec.freqs_hz() + spec.mw_offset_hz, dtype=torch.float64, device=dev)
    durs = sorted(set(spec.durations_ns))
    nf, nk, nm = freqs.shape[0], st.weights.shape[0], st.weights.shape[1]
    if chunk <= 0:
        chunk = max(1, (1 << 18) // (nk * nm))
    keep = torch.ones(N_AUG, dtype=st.l0.dtype, device=dev)
    keep[ACC] = 0.0
    em_u = torch.empty(ng, nf, len(durs), spec.n_windows, dtype=torch.float64, device=dev)
    pairs = ng * nf
    for i in range(0, pairs, chunk):
        idx = torch.arange(i, min(pairs, i + chunk), device=dev)
        g, fi = idx // nf, idx % nf
        b = idx.shape[0]
        zb = torch.zeros(b, dtype=st.l0.dtype, device=dev)
        lmw = _generator(st, g, zb, zb + spec.amplitude, zb, freqs[fi])
        r0 = (state[g] * keep)[..., None]
        prev_d, prop = 0, None
        for di, d in enumerate(durs):
            doubled = prop is not None and d == 2 * prev_d
            prop = prop @ prop if doubled and prop is not None else torch.linalg.matrix_exp(lmw * (d * 1e-9))
            prev_d = d
            r = (prop @ r0)[..., 0]  # (b, K, M, 14)
            em = torch.einsum("bkmwi,bkmi,km->bw", ro[g], r, st.weights)
            em_u[g, fi, di] = em.double()
    # actions are duration-major: a = d_index * n_freq + f_index (durations in the spec's order)
    order = [durs.index(d) for d in spec.durations_ns]
    em_all = em_u[:, :, order].permute(0, 2, 1, 3).reshape(ng, spec.n_actions, spec.n_windows)
    return em_all[inv_t]


def measurement_means(
    spec: MeasurementSpec,
    env: EnvParams,
    fields_t: np.ndarray,
    options: ModelOptions,
    device: torch.device | str,
    chunk: int = 0,
    field_chunk: int = 512,
) -> torch.Tensor:
    """Mean detected counts per shot, (N, A, W) float64: collection x NVs x emission + background (the truth
    tables of the learners' environments). Fields go in chunks of ``field_chunk`` (host memory of the statics)."""
    f = np.atleast_2d(np.asarray(fields_t, dtype=np.float64))
    em = torch.cat(
        [
            emission_table(spec, env, f[i : i + field_chunk], options, device, chunk)
            for i in range(0, f.shape[0], field_chunk)
        ]
    )
    bg = torch.as_tensor(spec.background_counts(env), dtype=torch.float64, device=em.device)
    return env.collection_efficiency * env.n_nv * em + bg


def axis_projections(fields_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(|B . n_k|, |B_perp,k|) in tesla for the four NV axes, each (..., 4)."""
    axes = torch.as_tensor(np.array(c.NV_AXES), dtype=fields_t.dtype, device=fields_t.device)
    par = fields_t @ axes.T
    tot2 = (fields_t**2).sum(-1, keepdim=True)
    perp = torch.sqrt(torch.clamp(tot2 - par**2, min=0.0))
    return par.abs(), perp


def mw_azimuths(mw_direction: tuple[float, float, float]) -> np.ndarray:
    """Azimuth (rad) of the MW field's transverse part in each axis's local frame, (4,)."""
    from glowbox.sim.surrogate.model import nv_local_frames

    m = np.einsum("kij,j->ki", nv_local_frames(), np.asarray(mw_direction, dtype=np.float64))
    out: np.ndarray = np.arctan2(m[:, 1], m[:, 0])
    return out


def axis_coordinates(fields_t: torch.Tensor, mw_direction: tuple[float, float, float]) -> torch.Tensor:
    """Per axis k the coordinates of the single-orientation table, (..., 4, 3): |B_par|, |B_perp| and the table
    azimuth theta in [0, pi).

    Every orientation's response depends on the field through B_par, |B_perp| and the angle between B_perp and the
    MW's transverse direction (the model is symmetric under rotations about the axis). Two more symmetries of the
    model (D S_z^2, isotropic rates and dephasing): a pi rotation about a transverse axis flips B_par and that
    relative angle, and a linear MW drive is unchanged by u -> -u, so the angle has period pi. Hence theta =
    sign(B_par) (psi_k - chi_k) + chi_1 mod pi, with psi_k the azimuth of B in axis k's local frame and chi_k the
    MW's: the angle at which orientation 1 (the table's) sees the same relative geometry."""
    from glowbox.sim.surrogate.model import nv_local_frames

    fr = torch.as_tensor(nv_local_frames(), dtype=fields_t.dtype, device=fields_t.device)  # (4, 3, 3)
    loc = torch.einsum("kij,...j->...ki", fr, fields_t)  # (..., 4, 3)
    par = loc[..., 2]
    perp = torch.hypot(loc[..., 0], loc[..., 1])
    psi = torch.atan2(loc[..., 1], loc[..., 0])
    chi = torch.as_tensor(mw_azimuths(mw_direction), dtype=fields_t.dtype, device=fields_t.device)
    sgn = torch.where(par < 0, -1.0, 1.0).to(fields_t.dtype)
    theta = torch.remainder(sgn * (psi - chi) + chi[0], math.pi)
    return torch.stack([par.abs(), perp, theta], dim=-1)


def _catmull_rom(t: torch.Tensor) -> torch.Tensor:
    """Weights (..., 4) of the nodes i-1, i, i+1, i+2 at fraction t in [0, 1) past node i."""
    t2, t3 = t * t, t * t * t
    return torch.stack(
        [0.5 * (-t + 2 * t2 - t3), 0.5 * (2 - 5 * t2 + 3 * t3), 0.5 * (t + 4 * t2 - 3 * t3), 0.5 * (-t2 + t3)], -1
    )


@dataclass
class AxisTable:
    """Single-orientation emission per NV, ``table`` (n_par, n_perp, n_az, A, W), on a regular grid of
    (B_par, |B_perp|, azimuth theta of B_perp in orientation 1's frame, see :func:`axis_coordinates`).

    :meth:`means` interpolates it for the four axes of each field (Catmull-Rom cubic along B_par, where the lines
    are sharp, and along the periodic azimuth; linear along |B_perp|, where the lines only shift quadratically) and
    returns counts per shot. The B_par grid starts two steps below 0 so the cubic stencil never leaves it; beyond
    the far edges it is clamped (the prior stays inside; jittered particles may step out by a few uT). The
    measured error against the full surrogate is reported per run (:func:`emulator_error`).
    """

    spec: MeasurementSpec
    env: EnvParams
    b_par_t: np.ndarray
    b_perp_t: np.ndarray
    n_az: int
    table: torch.Tensor

    @classmethod
    def build(
        cls,
        spec: MeasurementSpec,
        env: EnvParams,
        options: ModelOptions,
        device: torch.device | str,
        b_max_t: float = 3.1e-3,
        d_par_t: float = 10e-6,
        d_perp_t: float = 100e-6,
        n_az: int = 8,
        chunk: int = 0,
        field_chunk: int = 2048,
    ) -> AxisTable:
        """Simulate orientation 1 alone on the grid."""
        from glowbox.sim.surrogate.model import nv_local_frames

        b_par = np.arange(-2, round(b_max_t / d_par_t) + 1) * d_par_t
        b_perp = np.arange(0, round(b_max_t / d_perp_t) + 1) * d_perp_t
        az = np.arange(n_az) * (math.pi / n_az)
        x1, y1, n1 = nv_local_frames()[0]
        pp, qq, aa = np.meshgrid(b_par, b_perp, az, indexing="ij")
        dirs = np.cos(aa).reshape(-1, 1) * x1[None] + np.sin(aa).reshape(-1, 1) * y1[None]
        fields = pp.reshape(-1, 1) * n1[None] + qq.reshape(-1, 1) * dirs
        env1 = dataclasses.replace(env, b_field_t=(0.0, 0.0, 0.0), orientation_populations=(1.0, 0.0, 0.0, 0.0))
        # Field chunks bound the host memory of the statics (complex 14x14 superoperators per field and node).
        parts = [
            emission_table(spec, env1, fields[i : i + field_chunk], options, device, chunk=chunk).float()
            for i in range(0, fields.shape[0], field_chunk)
        ]
        tab = torch.cat(parts).reshape(b_par.size, b_perp.size, n_az, spec.n_actions, spec.n_windows)
        return cls(spec, env, b_par, b_perp, n_az, tab)

    def to(self, device: torch.device | str) -> AxisTable:
        return dataclasses.replace(self, table=self.table.to(device))

    @property
    def device(self) -> torch.device:
        return self.table.device

    def _stencil(self, fields_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat grid indices (..., 4 axes, 32) and weights (..., 4, 32) (orientation population included)."""
        # float32 is ample (fields ~1e-3 T, grid steps >= 1e-5 T) and much faster than float64 on consumer GPUs
        q = axis_coordinates(fields_t.to(self.device, torch.float32), self.env.mw_field_direction)
        p0, dp = float(self.b_par_t[0]), float(self.b_par_t[1] - self.b_par_t[0])
        dq = float(self.b_perp_t[1] - self.b_perp_t[0])
        npar, nperp, naz = self.b_par_t.size, self.b_perp_t.size, self.n_az
        dev = self.device
        u = torch.clamp((q[..., 0] - p0) / dp, 1.0, npar - 3 - 1e-9)
        v = torch.clamp(q[..., 1] / dq, 0.0, nperp - 1 - 1e-9)
        w = q[..., 2] / (math.pi / naz)
        i1, j0, k1 = torch.floor(u).long(), torch.floor(v).long(), torch.floor(w).long()
        wp = _catmull_rom(u - i1)  # (..., 4, 4)
        fv = v - j0
        wq = torch.stack([1 - fv, fv], -1)  # (..., 4, 2)
        wa = _catmull_rom(w - k1)  # (..., 4, 4)
        ip = i1[..., None] + torch.arange(-1, 3, device=dev)
        jq = torch.clamp(j0[..., None] + torch.arange(2, device=dev), max=nperp - 1)
        ka = torch.remainder(k1[..., None] + torch.arange(-1, 3, device=dev), naz)
        idx = (ip[..., :, None, None] * nperp + jq[..., None, :, None]) * naz + ka[..., None, None, :]
        wt = wp[..., :, None, None] * wq[..., None, :, None] * wa[..., None, None, :]
        pops = torch.as_tensor(self.env.orientation_populations, dtype=torch.float32, device=dev)
        return idx.flatten(-3), wt.flatten(-3) * pops[:, None]

    def emission(self, fields_t: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        """Orientation-averaged emission per NV per shot, (..., A or A', W) float32; ``actions`` (A',) restricts
        the actions (the same for every field)."""
        flat = self.table.reshape(-1, self.spec.n_actions, self.spec.n_windows)
        if actions is not None:
            flat = flat[:, actions]
        idx, w = self._stencil(fields_t)
        out = torch.zeros((*idx.shape[:-2], *flat.shape[1:]), device=self.device)
        for k in range(4):
            if self.env.orientation_populations[k] == 0:
                continue
            for q in range(idx.shape[-1]):
                out = out + flat[idx[..., k, q]] * w[..., k, q, None, None]
        return out

    def means(self, fields_t: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        """Counts per shot (..., A or A', W) float32 for fields (..., 3) in tesla."""
        e = self.env
        bg = torch.as_tensor(self.spec.background_counts(e), dtype=torch.float32, device=self.device)
        return e.collection_efficiency * e.n_nv * self.emission(fields_t, actions) + bg

    def means_per_row(self, fields_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Counts per shot of a few actions per row: fields (E, n, 3), actions (E, U) -> (E, n, U, W) (each row's
        particles at that row's actions; cheaper than :meth:`means` when U << A)."""
        e = self.env
        flat = self.table.reshape(-1, self.spec.n_actions, self.spec.n_windows)
        idx, w = self._stencil(fields_t)  # (E, n, 4, 32)
        ne, n = idx.shape[:2]
        act = actions.to(self.device)[:, None, None, :]  # (E, 1, 1, U)
        out = torch.empty((ne, n, actions.shape[1], self.spec.n_windows), device=self.device)
        step = max(1, (1 << 24) // max(1, 128 * actions.shape[1] * self.spec.n_windows))
        for i in range(0, n, step):  # one gather per chunk of particles (memory bound)
            sl = slice(i, i + step)
            g = flat[idx[:, sl].flatten(2)[..., None], act]  # (E, m, 128, U, W)
            out[:, sl] = torch.einsum("emquw,emq->emuw", g, w[:, sl].flatten(2))
        bg = torch.as_tensor(self.spec.background_counts(e), dtype=torch.float32, device=self.device)
        return e.collection_efficiency * e.n_nv * out + bg

    def means_at(self, fields_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Counts per shot of one action per field: fields (..., 3), actions (...) -> (..., W)."""
        e = self.env
        flat = self.table.reshape(-1, self.spec.n_actions, self.spec.n_windows)
        idx, w = self._stencil(fields_t)
        act = actions.to(self.device)
        g = flat[idx.flatten(-2), act[..., None]]  # (..., 128, W): one gather
        out = torch.einsum("...qw,...q->...w", g, w.flatten(-2))
        bg = torch.as_tensor(self.spec.background_counts(e), dtype=torch.float32, device=self.device)
        return e.collection_efficiency * e.n_nv * out + bg


def emulator_error(table: AxisTable, truth_means: torch.Tensor, fields_t: np.ndarray) -> dict[str, float]:
    """Emulator vs full surrogate on the same fields: error relative to the spin contrast scale.

    ``truth_means`` (N, A, W) from :func:`measurement_means`. Reports the max and RMS absolute error in counts
    per shot, and both relative to the RMS spread of the true means over actions (the signal the learners use)."""
    f = torch.as_tensor(np.asarray(fields_t), dtype=torch.float64, device=table.device)
    emu = table.means(f).double()
    tru = truth_means.to(table.device).double()
    err = emu - tru
    spread = tru.std(dim=1).mean()
    rms = torch.sqrt((err**2).mean())
    return {
        "max_abs": float(err.abs().max()),
        "rms_abs": float(rms),
        "signal_rms": float(spread),
        "rms_rel_signal": float(rms / spread),
        "max_rel_signal": float(err.abs().max() / spread),
    }
