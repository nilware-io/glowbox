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
* :func:`measurement_means`: the surrogate's run-length path (:mod:`glowbox.sim.surrogate.runs`) for many fields
  and every action at once (one propagator per run; equal to the bin path on the grid, tests/learn).
* :class:`AxisTable`: a fast *emulator* of :func:`measurement_means` for arbitrary fields, used as the likelihood
  of the Bayesian learners. The ensemble signal is a sum over the four NV orientations (docs/physics.md 7), and
  each orientation's response depends on the field only through B_par = |B . n_k| and B_perp = |B - (B . n_k) n_k|,
  up to the azimuth of B_perp relative to the MW direction (it changes the drive matrix elements at order
  (gamma_e B_perp / D)^2 ~ 1e-3 at 3 mT). So one single-orientation table on a (B_par, B_perp) grid, bilinearly
  interpolated and summed over the four axes, gives the means of any field. Its error against the full surrogate
  is measured (:func:`emulator_error`) and reported; the learners' *observations* always come from the full
  surrogate (or another ``run()``), so the emulator only enters as the learners' model, never as the truth.

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
from glowbox.sim.surrogate.model import ModelOptions, build_statics, emission_and_state
from glowbox.sim.surrogate.runs import Run, run_emission

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


def _runs(spec: MeasurementSpec, env: EnvParams, freq_hz: torch.Tensor, dur_ns: torch.Tensor) -> list[Run]:
    ns = 1e-9
    beta = env.optical_beta_full_scale
    runs = [Run(dur_ns * ns, 0.0, spec.amplitude, 0.0, freq_hz, 0.0)]
    runs += [Run(w * ns, beta, frame_hz=freq_hz, record=True) for w in spec.windows_ns]
    return runs


@torch.no_grad()
def emission_table(
    spec: MeasurementSpec,
    env: EnvParams,
    fields_t: np.ndarray,
    options: ModelOptions,
    device: torch.device | str,
    actions: np.ndarray | None = None,
    chunk: int = 2048,
) -> torch.Tensor:
    """Photons emitted per NV per shot in each readout window, (N, A', W) float64 on ``device``, for every field
    (total field = ``env.b_field_t`` + row) and every action (or the listed ``actions``), averaged over the
    orientations kept by ``env.orientation_populations`` with their weights. The run-length surrogate path."""
    dev = torch.device(device)
    fields = np.atleast_2d(np.asarray(fields_t, dtype=np.float64)) + np.asarray(env.b_field_t)[None]
    n = fields.shape[0]
    f_all, d_all = spec.action_grid()
    act = np.arange(spec.n_actions) if actions is None else np.asarray(actions, dtype=np.int64)
    f_a = torch.as_tensor(f_all[act], dtype=torch.float64, device=dev)
    d_a = torch.as_tensor(d_all[act], dtype=torch.float64, device=dev)
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
    # Initialisation, once per field (MW-free: no coherence enters the measurement).
    pre = init_prefix(spec.init_ns, spec.settle_ns, grid_ns=spec.grid_ns)
    k = pre.n_bins
    pre_b = dataclasses.replace(
        pre,
        laser=np.repeat(np.asarray(pre.laser), n, axis=0),
        mw_amplitude=np.zeros((n, k), np.float32),
        mw_frequency_hz=np.zeros((n, k)),
        mw_phase_rad=np.zeros((n, k), np.float32),
        magnet_t=fields,
    )
    _, state = emission_and_state(pre_b, env0, options=options, device=dev, statics=st)
    na = act.shape[0]
    out = torch.empty(n, na, spec.n_windows, dtype=torch.float64, device=dev)
    pairs = n * na
    for i in range(0, pairs, chunk):
        idx = torch.arange(i, min(pairs, i + chunk), device=dev)
        g, a = idx // na, idx % na
        em, _ = run_emission(st, inv_t[g], _runs(spec, env0, f_a[a], d_a[a]), state[g])
        out[g, a] = em.double()
    return out


def measurement_means(
    spec: MeasurementSpec,
    env: EnvParams,
    fields_t: np.ndarray,
    options: ModelOptions,
    device: torch.device | str,
    actions: np.ndarray | None = None,
    chunk: int = 2048,
) -> torch.Tensor:
    """Mean detected counts per shot, (N, A', W) float64: collection x NVs x emission + background (the truth
    tables of the learners' environments)."""
    em = emission_table(spec, env, fields_t, options, device, actions, chunk)
    bg = torch.as_tensor(spec.background_counts(env), dtype=torch.float64, device=em.device)
    return env.collection_efficiency * env.n_nv * em + bg


def axis_projections(fields_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(|B . n_k|, |B_perp,k|) in tesla for the four NV axes, each (..., 4)."""
    axes = torch.as_tensor(np.array(c.NV_AXES), dtype=fields_t.dtype, device=fields_t.device)
    par = fields_t @ axes.T
    tot2 = (fields_t**2).sum(-1, keepdim=True)
    perp = torch.sqrt(torch.clamp(tot2 - par**2, min=0.0))
    return par.abs(), perp


@dataclass
class AxisTable:
    """Single-orientation emission per NV, ``table`` (n_par, n_perp, A, W), on a regular (B_par, B_perp) grid.

    :meth:`means` interpolates it for the four axes of each field (Catmull-Rom cubic along B_par, where the lines
    are sharp; linear along B_perp, where they only shift quadratically) and returns counts per shot. Measured
    interpolation error at the default steps (10 uT, 100 uT), single axis, relative to its spread over actions:
    max 0.3 % along B_par, 2 % along B_perp (scratch check recorded in docs/results/m5-adaptive-field.md). The
    response is even in B_par (spin-1, E = 0), so the grid starts at 0 and reflects there; beyond the far edge it
    is clamped (the prior stays inside; jittered particles may step out by a few uT).
    """

    spec: MeasurementSpec
    env: EnvParams
    b_par_t: np.ndarray
    b_perp_t: np.ndarray
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
        chunk: int = 2048,
    ) -> AxisTable:
        """Simulate orientation 1 alone on the grid (B_perp along its local x; the azimuth of B_perp is
        neglected, see the module docstring)."""
        b_par = np.arange(0.0, b_max_t + 0.5 * d_par_t, d_par_t)
        b_perp = np.arange(0.0, b_max_t + 0.5 * d_perp_t, d_perp_t)
        from glowbox.sim.surrogate.model import nv_local_frames

        x1, _, n1 = nv_local_frames()[0]
        pp, qq = np.meshgrid(b_par, b_perp, indexing="ij")
        fields = pp.reshape(-1, 1) * n1[None] + qq.reshape(-1, 1) * x1[None]
        env1 = dataclasses.replace(env, b_field_t=(0.0, 0.0, 0.0), orientation_populations=(1.0, 0.0, 0.0, 0.0))
        em = emission_table(spec, env1, fields, options, device, chunk=chunk)
        tab = em.reshape(b_par.size, b_perp.size, spec.n_actions, spec.n_windows).float()
        return cls(spec, env, b_par, b_perp, tab)

    def to(self, device: torch.device | str) -> AxisTable:
        return dataclasses.replace(self, table=self.table.to(device))

    @property
    def device(self) -> torch.device:
        return self.table.device

    def _stencil(self, fields_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat grid indices (..., 4 axes, 8) and weights (..., 4, 8) (orientation population included)."""
        par, perp = axis_projections(fields_t.to(self.device, torch.float64))
        dp = float(self.b_par_t[1] - self.b_par_t[0])
        dq = float(self.b_perp_t[1] - self.b_perp_t[0])
        npar, nperp = self.b_par_t.size, self.b_perp_t.size
        u = torch.clamp(par / dp, 0.0, npar - 1 - 1e-9)
        v = torch.clamp(perp / dq, 0.0, nperp - 1 - 1e-9)
        i1, j0 = torch.floor(u).long(), torch.floor(v).long()
        t, fv = (u - i1).float(), (v - j0).float()
        # Catmull-Rom weights of the nodes i1-1, i1, i1+1, i1+2.
        t2, t3 = t * t, t * t * t
        wp = torch.stack(
            [
                0.5 * (-t + 2 * t2 - t3),
                0.5 * (2 - 5 * t2 + 3 * t3),
                0.5 * (t + 4 * t2 - 3 * t3),
                0.5 * (-t2 + t3),
            ],
            -1,
        )
        ip = i1[..., None] + torch.arange(-1, 3, device=self.device)
        ip = torch.where(ip < 0, -ip, ip)  # even in B_par: reflect at 0
        ip = torch.clamp(ip, max=npar - 1)
        wq = torch.stack([1 - fv, fv], -1)
        jq = torch.clamp(j0[..., None] + torch.arange(2, device=self.device), max=nperp - 1)
        idx = (ip[..., :, None] * nperp + jq[..., None, :]).flatten(-2)  # (..., 4, 8)
        pops = torch.as_tensor(self.env.orientation_populations, dtype=torch.float32, device=self.device)
        w = (wp[..., :, None] * wq[..., None, :]).flatten(-2) * pops[:, None]
        return idx, w

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
            for q in range(8):
                out = out + flat[idx[..., k, q]] * w[..., k, q, None, None]
        return out

    def means(self, fields_t: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        """Counts per shot (..., A or A', W) float32 for fields (..., 3) in tesla."""
        e = self.env
        bg = torch.as_tensor(self.spec.background_counts(e), dtype=torch.float32, device=self.device)
        return e.collection_efficiency * e.n_nv * self.emission(fields_t, actions) + bg

    def means_at(self, fields_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Counts per shot of one action per field: fields (..., 3), actions (...) -> (..., W)."""
        e = self.env
        flat = self.table.reshape(-1, self.spec.n_actions, self.spec.n_windows)
        idx, w = self._stencil(fields_t)
        act = actions.to(self.device)
        out = torch.zeros((*act.shape, self.spec.n_windows), device=self.device)
        for k in range(4):
            if e.orientation_populations[k] == 0:
                continue
            for q in range(8):
                out = out + flat[idx[..., k, q], act] * w[..., k, q, None]
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
