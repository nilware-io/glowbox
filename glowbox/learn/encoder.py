"""Learned pulse encoder (project brief, phase 2): input data -> pulse sequence, trained end to end through the
differentiable surrogate together with a decoder.

The encoder writes the *suffix* of a sequence that starts with a fixed MW-free initialisation (laser + dark settle,
:func:`glowbox.learn.reservoir.init_prefix`). The suffix is ``n_blocks`` repetitions of

    [MW block: n_segments slots of slot_ns, one MW segment from each slot's start] [readout laser] [dark gap]

and the decoder sees the readout counts: either one gated number per readout (the first ``gate_ns`` of each
readout laser, what a gated photon counter returns; the rest of the laser re-polarises the spins) or every readout
bin (time-resolved). The block layout is :class:`Layout`.

**Parametrisation (smooth and bounded).** Per segment, four pre-activations z (from a network of the input, or
free parameters when the encoder has no input) set

* amplitude ``a = amp_lo + (amp_hi - amp_lo) sigmoid(z_0)``,
* phase ``phi = pi tanh(z_1)`` (suffix-relative, as in the reservoir; :func:`~glowbox.learn.reservoir.full_sequence`
  converts to absolute time),
* frequency ``f = base_hz[block, segment] + span_hz tanh(z_2)``: each segment is tied to a base frequency (a device
  line, or a point of a sweep grid) because the drive only acts within a few linewidths of a line, so a frequency
  far from every line has no gradient to follow,
* duration ``d = min_ns + (max_ns - min_ns) sigmoid(z_3)``, MW from the slot start.

**Durations on the grid (straight-through).** The forward pass rounds every duration to the 10 ns grid, so a
trained sequence is exactly a :class:`~glowbox.control.pulse.PulseSequence` (hardware and the QuTiP reference can
run it). The backward pass uses the continuous fill fractions of
:func:`glowbox.sim.surrogate.model.mw_fill_for_duration` (the MW covers the first ``d / grid - j`` of bin j of the
slot): ``fill = fill_q + (fill_c - fill_c.detach())``. The
gradient with respect to a duration is then the derivative of the propagator of the boundary bin with respect to
its fill, an approximation (the true function of d is piecewise constant).

**Simulation.** :func:`prefix_states` runs the initialisation once per static field (no gradient); every suffix
continues from its prefix's final state (:func:`suffix_counts`, exact because a MW-free prefix leaves no spin
coherence, docs/notes/surrogate.md "Sweep fast paths"). The suffix runs with gradients through
:func:`~glowbox.sim.surrogate.model.emission_and_state`; with pulse-level controls it takes the surrogate's shared-
propagator gradient path (exact first derivatives per bin).

**Noise.** Training adds shot noise as a Gaussian with the Poisson variance, ``x = mu + sqrt(mu / shots) eps``
(:func:`gaussian_shot_noise`), differentiable in mu (the reparametrisation trick; at >= 10^3 shots the counts per
readout are >> 10, where the Gaussian is a good approximation). Evaluation draws real Poisson counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from glowbox.control.pulse import SequenceArrays
from glowbox.sim.constants import GRID_DEFAULT_NS
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions, Statics, emission_and_state, statics_for
from glowbox.sim.surrogate.runs import Run, run_emission


@dataclass(frozen=True)
class Layout:
    """Shape and ranges of the encoder's suffix (assumptions of the experiment, not physics)."""

    n_blocks: int = 1
    n_segments: int = 4
    slot_ns: int = 100
    min_ns: float = 0.0
    max_ns: float = 100.0
    amp_lo: float = 0.0
    amp_hi: float = 1.0
    #: Frequency freedom around each segment's base frequency, Hz.
    span_hz: float = 4e6
    #: Readout laser after every block (power 1), ns.
    readout_ns: int = 1000
    #: Counted window at the start of each readout, ns; 0 = time-resolved: the readout split into windows of
    #: ``window_ns`` (0 = the grid, every bin).
    gate_ns: int = 300
    window_ns: int = 0
    #: Dark time after each readout (lets the singlet empty), ns.
    gap_ns: int = 0
    grid_ns: int = GRID_DEFAULT_NS

    def __post_init__(self) -> None:
        g = self.grid_ns
        if self.n_blocks < 1 or self.n_segments < 1:
            raise ValueError("need n_blocks >= 1 and n_segments >= 1")
        for name in ("slot_ns", "readout_ns", "gate_ns", "gap_ns", "window_ns"):
            if getattr(self, name) % g:
                raise ValueError(f"{name} must be a multiple of the grid")
        if not 0 <= self.min_ns <= self.max_ns <= self.slot_ns:
            raise ValueError("need 0 <= min_ns <= max_ns <= slot_ns")
        if not 0 <= self.amp_lo <= self.amp_hi <= 1:
            raise ValueError("need 0 <= amp_lo <= amp_hi <= 1")
        if not 0 <= self.gate_ns <= self.readout_ns or self.readout_ns <= 0:
            raise ValueError("need 0 <= gate_ns <= readout_ns, readout_ns > 0")
        if self.window_ns and self.readout_ns % self.window_ns:
            raise ValueError("readout_ns must be a multiple of window_ns")

    @property
    def slot_bins(self) -> int:
        return self.slot_ns // self.grid_ns

    @property
    def block_bins(self) -> int:
        return self.n_segments * self.slot_bins

    @property
    def readout_bins(self) -> int:
        return self.readout_ns // self.grid_ns

    @property
    def gate_bins(self) -> int:
        return self.gate_ns // self.grid_ns

    @property
    def period_bins(self) -> int:
        return self.block_bins + self.readout_bins + self.gap_ns // self.grid_ns

    @property
    def n_bins(self) -> int:
        return self.n_blocks * self.period_bins

    @property
    def window_bins(self) -> int:
        return (self.window_ns or self.grid_ns) // self.grid_ns

    @property
    def features_per_readout(self) -> int:
        return 1 if self.gate_ns else self.readout_bins // self.window_bins

    @property
    def n_features(self) -> int:
        """Numbers the decoder sees per sequence: one per readout (gated) or one per readout window."""
        return self.n_blocks * self.features_per_readout

    def readout_start(self, k: int) -> int:
        """First bin of readout ``k``."""
        return k * self.period_bins + self.block_bins


@dataclass
class BlockParams:
    """Pulse parameters, each (N, n_blocks, n_segments): amplitude, phase (rad), frequency (Hz, float64), duration
    (ns, continuous; the sequence uses it rounded to the grid)."""

    amplitude: torch.Tensor
    phase_rad: torch.Tensor
    frequency_hz: torch.Tensor
    duration_ns: torch.Tensor

    def detach(self) -> BlockParams:
        return BlockParams(*(x.detach() for x in (self.amplitude, self.phase_rad, self.frequency_hz, self.duration_ns)))

    def numpy(self) -> dict[str, np.ndarray]:
        d = self.detach()
        return {
            "amplitude": d.amplitude.cpu().numpy(),
            "phase_rad": d.phase_rad.cpu().numpy(),
            "frequency_hz": d.frequency_hz.cpu().numpy(),
            "duration_ns": d.duration_ns.cpu().numpy(),
        }

    def quantized_ns(self, grid_ns: int) -> torch.Tensor:
        return torch.round(self.duration_ns.detach() / grid_ns) * grid_ns


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _atanh(x: float) -> float:
    x = min(max(x, -0.999), 0.999)
    return math.atanh(x)


def preactivations_for(
    layout: Layout,
    amplitude: float | np.ndarray,
    duration_ns: float | np.ndarray,
    phase_rad: float | np.ndarray = 0.0,
    freq_offset_hz: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Pre-activations (n_blocks, n_segments, 4) that produce the given pulse parameters (to initialise an encoder
    at a hand-designed protocol). Values are clipped into the open ranges."""
    shape = (layout.n_blocks, layout.n_segments)
    a = np.broadcast_to(np.asarray(amplitude, dtype=np.float64), shape)
    d = np.broadcast_to(np.asarray(duration_ns, dtype=np.float64), shape)
    p = np.broadcast_to(np.asarray(phase_rad, dtype=np.float64), shape)
    f = np.broadcast_to(np.asarray(freq_offset_hz, dtype=np.float64), shape)
    z = np.empty((*shape, 4))
    da = max(layout.amp_hi - layout.amp_lo, 1e-12)
    dd = max(layout.max_ns - layout.min_ns, 1e-12)
    for i in range(shape[0]):
        for j in range(shape[1]):
            z[i, j, 0] = _logit((a[i, j] - layout.amp_lo) / da)
            z[i, j, 1] = _atanh(p[i, j] / math.pi)
            z[i, j, 2] = _atanh(f[i, j] / layout.span_hz) if layout.span_hz > 0 else 0.0
            z[i, j, 3] = _logit((d[i, j] - layout.min_ns) / dd)
    return z


class PulseEncoder(nn.Module):
    """Input (N, n_inputs) -> :class:`BlockParams`. ``n_inputs = 0``: the sequence is a free parameter (the same for
    every sample; a learned *protocol*, e.g. a sensor protocol for Task 1).

    With inputs, an MLP (GELU) maps the input to the pre-activations; its last layer's bias is the input-independent
    part (initialised from ``init`` if given) and its weights start small (``out_scale``), so training starts near
    that protocol and learns how the input should modulate it.
    """

    def __init__(
        self,
        layout: Layout,
        base_hz: np.ndarray,
        n_inputs: int,
        hidden: tuple[int, ...] = (64, 64),
        init: np.ndarray | None = None,
        out_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.n_inputs = n_inputs
        base = np.broadcast_to(np.asarray(base_hz, dtype=np.float64), (layout.n_blocks, layout.n_segments)).copy()
        self.register_buffer("base_hz", torch.as_tensor(base, dtype=torch.float64))
        n_out = layout.n_blocks * layout.n_segments * 4
        z0 = torch.zeros(n_out) if init is None else torch.as_tensor(np.asarray(init), dtype=torch.float32).reshape(-1)
        if n_inputs == 0:
            self.net: nn.Module | None = None
            self.bias = nn.Parameter(z0.clone())
        else:
            layers: list[nn.Module] = []
            width = n_inputs
            for h in hidden:
                layers += [nn.Linear(width, h), nn.GELU()]
                width = h
            last = nn.Linear(width, n_out)
            with torch.no_grad():
                last.weight.mul_(out_scale)
                last.bias.copy_(z0)
            layers.append(last)
            self.net = nn.Sequential(*layers)
            self.register_parameter("bias", None)

    def preactivations(self, u: torch.Tensor | None, n: int | None = None) -> torch.Tensor:
        """(N, n_blocks, n_segments, 4)."""
        ly = self.layout
        if self.net is None:
            if n is None:
                n = 1 if u is None else int(u.shape[0])
            assert self.bias is not None
            z = self.bias[None].expand(n, -1)
        else:
            if u is None:
                raise ValueError("this encoder needs inputs")
            z = self.net(u)
        return z.reshape(-1, ly.n_blocks, ly.n_segments, 4)

    def forward(self, u: torch.Tensor | None, n: int | None = None) -> BlockParams:
        ly = self.layout
        z = self.preactivations(u, n)
        amp = ly.amp_lo + (ly.amp_hi - ly.amp_lo) * torch.sigmoid(z[..., 0])
        phase = math.pi * torch.tanh(z[..., 1])
        base = self.get_buffer("base_hz")
        freq = base[None] + ly.span_hz * torch.tanh(z[..., 2]).to(torch.float64)
        dur = ly.min_ns + (ly.max_ns - ly.min_ns) * torch.sigmoid(z[..., 3])
        return BlockParams(amp, phase, freq, dur)

    @torch.no_grad()
    def randomize_(self, gain: float, generator: torch.Generator) -> PulseEncoder:
        """Fixed random encoder (the reservoir baseline in this parametrisation): every layer N(0, 1/fan_in), the
        last layer N(0, gain^2 / fan_in) with N(0, 1) biases, as :class:`~glowbox.learn.reservoir.RandomPulseEncoder`
        (pre-activations of order ``gain``, so pulse parameters depend strongly on the input)."""
        if self.net is None:
            raise ValueError("a free-parameter encoder has no input map to randomise")
        lin = [m for m in self.net.modules() if isinstance(m, nn.Linear)]
        dev = lin[0].weight.device
        for i, m in enumerate(lin):
            w = torch.randn(m.weight.shape, generator=generator, device=dev) / math.sqrt(m.in_features)
            if i == len(lin) - 1:
                m.weight.copy_(w * gain)
                m.bias.copy_(torch.randn(m.bias.shape, generator=generator, device=dev))
            else:
                m.weight.copy_(w)
                m.bias.zero_()
        return self


def _fill(duration_ns: torch.Tensor, layout: Layout, shift_ns: float = 0.0) -> torch.Tensor:
    """Straight-through fill fractions (N, B, S, slot_bins): forward on the grid, backward continuous."""
    g = float(layout.grid_ns)
    d = torch.clamp(duration_ns + shift_ns, 0.0, float(layout.slot_ns))
    j = torch.arange(layout.slot_bins, device=d.device, dtype=d.dtype)
    fc = torch.clamp(d[..., None] / g - j, 0.0, 1.0)
    dq = torch.round(d.detach() / g)
    fq = (j < dq[..., None]).to(d.dtype)
    return fq + (fc - fc.detach())  # exact grid values forward (x - x == 0)


def build_suffix(
    params: BlockParams,
    layout: Layout,
    *,
    offset_hz: float = 0.0,
    magnet_t: torch.Tensor | np.ndarray | None = None,
    duration_shift_ns: float = 0.0,
) -> tuple[SequenceArrays, torch.Tensor]:
    """Per-bin torch arrays of the suffix (N, n_bins) and the MW fill (N, n_bins); differentiable in ``params``.

    Every bin of a slot carries its segment's amplitude, phase and frequency; the fill switches the MW off after
    the (grid-rounded) duration. ``offset_hz`` shifts every MW frequency (the quantum test's detuned arm);
    ``duration_shift_ns`` adds a fixed shift to every duration before rounding (the timing probe). ``magnet_t``
    (N, 3) sets the static field per sequence (default zero).
    """
    ly = layout
    amp, phase, freq = params.amplitude, params.phase_rad, params.frequency_hz
    n = amp.shape[0]
    dev, dt = amp.device, amp.dtype
    sb = ly.slot_bins
    fill = _fill(params.duration_ns, ly, duration_shift_ns)  # (N, B, S, sb)

    def rep(x: torch.Tensor) -> torch.Tensor:  # (N, B, S) -> (N, B, S*sb)
        return x[..., None].expand(*x.shape, sb).reshape(n, ly.n_blocks, ly.block_bins)

    nr, ngap = ly.readout_bins, ly.gap_ns // ly.grid_ns
    tail = nr + ngap

    def with_tail(x: torch.Tensor, value: float) -> torch.Tensor:
        pad = torch.full((n, ly.n_blocks, tail), value, dtype=x.dtype, device=dev)
        return torch.cat([x, pad], dim=-1).reshape(n, ly.n_bins)

    laser_blk = torch.zeros(n, ly.n_blocks, ly.block_bins, dtype=dt, device=dev)
    laser_tail = torch.cat([torch.ones(nr, dtype=dt, device=dev), torch.zeros(ngap, dtype=dt, device=dev)])
    laser = torch.cat([laser_blk, laser_tail.expand(n, ly.n_blocks, tail)], dim=-1).reshape(n, ly.n_bins)
    arrays = SequenceArrays(
        laser=laser,
        mw_amplitude=with_tail(rep(amp), 0.0),
        mw_frequency_hz=with_tail(rep(freq + offset_hz), 0.0),
        mw_phase_rad=with_tail(rep(phase), 0.0),
        magnet_t=(
            torch.zeros(n, 3, dtype=torch.float64, device=dev)
            if magnet_t is None
            else torch.as_tensor(magnet_t, dtype=torch.float64, device=dev)
        ),
        grid_ns=ly.grid_ns,
    )
    fill_bins = with_tail(fill.reshape(n, ly.n_blocks, ly.block_bins), 0.0)
    return arrays, fill_bins


def suffix_to_numpy(
    params: BlockParams,
    layout: Layout,
    *,
    offset_hz: float = 0.0,
    magnet_t: np.ndarray | None = None,
    duration_shift_ns: float = 0.0,
) -> SequenceArrays:
    """The same suffix as plain NumPy arrays with grid-rounded durations (MW amplitude 0 where the fill is 0), the
    form :func:`~glowbox.learn.reservoir.full_sequence` and :func:`~glowbox.control.pulse.SequenceArrays.to_sequence`
    take. Phases wrapped to [-pi, pi]."""
    with torch.no_grad():
        arr, fill = build_suffix(
            params.detach(),
            layout,
            offset_hz=offset_hz,
            magnet_t=None if magnet_t is None else torch.as_tensor(magnet_t),
            duration_shift_ns=duration_shift_ns,
        )
    on = fill.cpu().numpy() > 0.5
    amp = np.where(on, np.asarray(arr.mw_amplitude.cpu().numpy(), np.float32), 0.0).astype(np.float32)  # type: ignore[union-attr]
    freq = np.where(on, arr.mw_frequency_hz.cpu().numpy(), 0.0)  # type: ignore[union-attr]
    ph = np.remainder(arr.mw_phase_rad.cpu().numpy().astype(np.float64) + np.pi, 2 * np.pi) - np.pi  # type: ignore[union-attr]
    phase = np.where(on, ph, 0.0).astype(np.float32)
    return SequenceArrays(
        laser=arr.laser.cpu().numpy().astype(np.float32),  # type: ignore[union-attr]
        mw_amplitude=amp,
        mw_frequency_hz=freq,
        mw_phase_rad=phase,
        magnet_t=arr.magnet_t.cpu().numpy(),  # type: ignore[union-attr]
        grid_ns=layout.grid_ns,
    )


# --- simulation -------------------------------------------------------------------------------------------------


@torch.no_grad()
def prefix_states(
    prefix: SequenceArrays,
    env: EnvParams,
    options: ModelOptions,
    device: torch.device | str,
) -> torch.Tensor:
    """Final per-NV states (P, K, M, 14) of a batch of MW-free prefixes (one per static field)."""
    pre = prefix.to_numpy()
    if np.any(np.asarray(pre.mw_amplitude) != 0):
        raise ValueError("the prefix must be MW-free")
    _, state = emission_and_state(pre, env, options=options, device=device)
    return state


@torch.no_grad()
def prefix_states_and_statics(
    prefix: SequenceArrays,
    env: EnvParams,
    options: ModelOptions,
    device: torch.device | str,
) -> tuple[torch.Tensor, Statics, torch.Tensor]:
    """(final states (P, K, M, 14), statics of the prefixes' distinct static fields, group index (P,) of each
    prefix row into them): everything :func:`encoded_counts` needs to continue every prefix."""
    pre = prefix.to_numpy()
    if np.any(np.asarray(pre.mw_amplitude) != 0):
        raise ValueError("the prefix must be MW-free")
    st = statics_for(pre, env, options, device)
    fields = np.asarray(env.b_field_t, dtype=np.float64)[None] + np.asarray(pre.magnet_t, np.float64).reshape(-1, 3)
    _, inv = np.unique(fields, axis=0, return_inverse=True)
    _, state = emission_and_state(pre, env, options=options, device=device, statics=st)
    return state, st, torch.as_tensor(inv.reshape(-1), device=torch.device(device))


def suffix_counts(
    arrays: SequenceArrays,
    fill: torch.Tensor,
    env: EnvParams,
    initial_state: torch.Tensor,
    options: ModelOptions,
    *,
    statics: Statics | None = None,
) -> torch.Tensor:
    """Noiseless mean detected counts per shot per suffix bin, (N, n_bins), differentiable in the controls."""
    dev = initial_state.device
    st = statics if statics is not None else statics_for(arrays, env, options, dev)
    em, _ = emission_and_state(
        arrays, env, options=options, mw_fill=fill, initial_state=initial_state, device=dev, statics=st
    )
    laser = torch.as_tensor(arrays.laser, device=dev).to(em.dtype)
    rate = env.collection_efficiency * env.n_nv * em + env.background_cps * laser + env.dark_cps
    return rate * (arrays.grid_ns * 1e-9)


def readout_features(counts: torch.Tensor, layout: Layout) -> torch.Tensor:
    """The decoder's numbers from per-bin counts (N, n_bins): gated sums or window sums, (N, n_features) in
    block-major order."""
    ly = layout
    n = counts.shape[0]
    per = counts.reshape(n, ly.n_blocks, ly.period_bins)
    ro = per[..., ly.block_bins : ly.block_bins + ly.readout_bins]
    if ly.gate_ns:
        return ro[..., : ly.gate_bins].sum(-1)
    return ro.reshape(n, ly.n_blocks, -1, ly.window_bins).sum(-1).reshape(n, ly.n_features)


def _ste_duration_ns(d: torch.Tensor, layout: Layout, shift_ns: float, quantize: bool) -> torch.Tensor:
    """Duration clamped to the slot; forward rounded to the grid, backward continuous (straight-through)."""
    g = float(layout.grid_ns)
    dc = torch.clamp(d + shift_ns, 0.0, float(layout.slot_ns))
    if not quantize:
        return dc
    return torch.round(dc.detach() / g) * g + (dc - dc.detach())


def layout_runs(
    params: BlockParams,
    layout: Layout,
    env: EnvParams,
    *,
    offset_hz: float = 0.0,
    duration_shift_ns: float = 0.0,
    quantize: bool = True,
) -> list[Run]:
    """The suffix of :func:`build_suffix` as runs (:mod:`glowbox.sim.surrogate.runs`): per segment an MW-on run
    (its duration exact and differentiable) and an MW-off rest of the slot; per readout the recorded gate or
    windows, the rest of the laser and the gap. Frames as the bin path: off runs and readouts keep the last MW
    frequency."""
    ly = layout
    ns = 1e-9
    beta_on = env.optical_beta_full_scale
    dur = _ste_duration_ns(params.duration_ns, ly, duration_shift_ns, quantize)
    amp, ph = params.amplitude, params.phase_rad
    freq = params.frequency_hz + offset_hz
    ac, as_ = amp * torch.cos(ph), amp * torch.sin(ph)
    runs: list[Run] = []
    for k in range(ly.n_blocks):
        t_blk = k * ly.period_bins * ly.grid_ns
        f_last = freq[:, k, -1]
        for s in range(ly.n_segments):
            t0 = (t_blk + s * ly.slot_ns) * ns
            f = freq[:, k, s]
            d = dur[:, k, s]
            runs.append(Run(d * ns, 0.0, ac[:, k, s], as_[:, k, s], f, t0))
            runs.append(Run((ly.slot_ns - d) * ns, 0.0, 0.0, 0.0, f, t0))
        if ly.gate_ns:
            runs.append(Run(ly.gate_ns * ns, beta_on, frame_hz=f_last, record=True))
            rest = ly.readout_ns - ly.gate_ns
            if rest:
                runs.append(Run(rest * ns, beta_on, frame_hz=f_last))
        else:
            w = ly.window_bins * ly.grid_ns
            for _ in range(ly.features_per_readout):
                runs.append(Run(w * ns, beta_on, frame_hz=f_last, record=True))
        if ly.gap_ns:
            runs.append(Run(ly.gap_ns * ns, 0.0, frame_hz=f_last))
    return runs


def encoded_counts(
    params: BlockParams,
    layout: Layout,
    env: EnvParams,
    initial_state: torch.Tensor,
    statics: Statics,
    *,
    group: torch.Tensor | None = None,
    offset_hz: float = 0.0,
    duration_shift_ns: float = 0.0,
    quantize: bool = True,
) -> torch.Tensor:
    """Noiseless mean detected counts per shot of every feature (gate or window), (N, n_features); the run-length
    path (one propagator per run, exact and differentiable durations). ``group`` (N,) indexes the static fields of
    ``statics`` (default: all 0, one field)."""
    n = params.amplitude.shape[0]
    dev = initial_state.device
    g = torch.zeros(n, dtype=torch.long, device=dev) if group is None else group
    runs = layout_runs(params, layout, env, offset_hz=offset_hz, duration_shift_ns=duration_shift_ns, quantize=quantize)
    em, _ = run_emission(statics, g, runs, initial_state)
    t_s = layout.gate_ns * 1e-9 if layout.gate_ns else layout.window_bins * layout.grid_ns * 1e-9
    per_nv = em  # photons per NV per recorded run
    return env.collection_efficiency * env.n_nv * per_nv + (env.background_cps + env.dark_cps) * t_s


def gaussian_shot_noise(mean: torch.Tensor, shots: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    """Mean counts per shot -> noisy counts per shot averaged over ``shots`` (B,) shots: Gaussian with the Poisson
    variance mu / shots, differentiable in mu (reparametrisation)."""
    s = shots.to(mean.dtype).reshape(-1, *([1] * (mean.ndim - 1)))
    eps = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
    return mean + torch.sqrt(mean.clamp_min(0.0) / s) * eps


def poisson_shot_noise(mean: torch.Tensor, shots: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    """Real shot noise: Poisson(shots mu) / shots (no gradient)."""
    s = shots.to(mean.dtype).reshape(-1, *([1] * (mean.ndim - 1)))
    return torch.poisson((mean.detach() * s).clamp_min(0.0), generator=generator) / s


# --- analysis ---------------------------------------------------------------------------------------------------


def pulse_areas_pi(
    params: BlockParams, env: EnvParams, grid_ns: int, mw_perp: float = math.sqrt(2.0 / 3.0)
) -> np.ndarray:
    """Resonant rotation angle / pi of every segment (grid-rounded durations), (N, B, S): 2 f_R a d with
    f_R = rabi_full_scale |m_perp| (MW along [100]: |m_perp| = sqrt(2/3) on every axis). 1 = a pi pulse."""
    f_r = env.rabi_full_scale_hz * mw_perp
    d = params.quantized_ns(grid_ns).cpu().numpy() * 1e-9
    out: np.ndarray = 2.0 * f_r * params.amplitude.detach().cpu().numpy() * d
    return out
