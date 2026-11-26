"""Task 1 with a learned sensor protocol: the encoder chooses the pulses of every measurement point.

A *learned protocol* is ``n_points`` measurements, each its own sequence: the fixed initialisation (laser, dark
settle), a MW block of ``layout.n_segments`` segments whose amplitude, phase, frequency and duration are free
parameters, and a time-resolved readout (``layout.window_ns`` windows). It has no input: the "data" is the hidden
field, so the encoder is a free-parameter :class:`~glowbox.learn.encoder.PulseEncoder` (one block per point; the
points are separate sequences, each continuing the same initialisation). Point p's segments are tied to base
frequency ``base_hz[p]`` (a sweep grid) with ``layout.span_hz`` of freedom, and the protocol starts at a
hand-designed one (:func:`pulsed_init`, the pulsed-ODMR pi pulse of milestone 2).

Features have the milestone-2 layout (T, F): T readout windows per point, F points; the CNN decoder
(:class:`~glowbox.learn.decoder.ConvDecoder`) and the normalisation by each sample's own mean
(:func:`~glowbox.learn.field_data.features`) are milestone 2's.

:class:`LearnedSensor` simulates the protocol for a batch of fields of one nuisance draw (differentiable, run-length
path) and builds :class:`~glowbox.learn.field_data.FieldBank` s for the frozen protocol (no gradient), so phase 2
(decoder training, evaluation) is milestone 2's code path.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from glowbox.control.pulse import SequenceArrays
from glowbox.learn.decoder import gaussian_nll
from glowbox.learn.encoder import (
    BlockParams,
    Layout,
    PulseEncoder,
    encoded_counts,
    gaussian_shot_noise,
    preactivations_for,
    prefix_states_and_statics,
    suffix_to_numpy,
)
from glowbox.learn.field_data import (
    NUISANCE_NAMES,
    BankSpec,
    FieldBank,
    FieldPrior,
    SweepProtocol,
    TargetScaler,
    features,
    log_shots,
    sample_shots,
)
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions


@dataclass(frozen=True)
class SensorSpec:
    """Shape of a learned protocol (assumptions of the experiment)."""

    n_points: int = 40
    f_start_hz: float = 2.77e9
    f_stop_hz: float = 2.97e9
    layout: Layout = dataclasses.field(
        default_factory=lambda: Layout(
            n_blocks=1,
            n_segments=2,
            slot_ns=200,
            min_ns=0.0,
            max_ns=200.0,
            amp_lo=0.0,
            amp_hi=1.0,
            span_hz=7.5e6,
            readout_ns=600,
            gate_ns=0,
            window_ns=100,
        )
    )
    init_ns: int = 1500
    settle_ns: int = 1000

    def base_hz(self) -> np.ndarray:
        """Base frequency of every point, (n_points,): the sweep grid of the matching fixed protocol."""
        step = (self.f_stop_hz - self.f_start_hz) / self.n_points
        return self.f_start_hz + step * np.arange(self.n_points)

    @property
    def n_time(self) -> int:
        return self.layout.features_per_readout

    @property
    def shot_ns(self) -> int:
        return self.init_ns + self.settle_ns + self.layout.n_bins * self.layout.grid_ns


def pulsed_init(
    spec: SensorSpec, amplitude: float, pi_ns: float, rest_amplitude: float = 0.0, rest_ns: float = 0.0
) -> np.ndarray:
    """Pre-activations (n_points, n_segments, 4) of a pulsed ODMR protocol: one pulse of ``pi_ns`` at
    ``amplitude`` on each point's base frequency, split over the slots (the first slots full, the rest in the
    next one). Unused slots get ``rest_amplitude`` for ``rest_ns`` (default off; a small nonzero rest keeps their
    bounded parameters away from saturation so training can still switch them on)."""
    ly = spec.layout
    durs = np.zeros(ly.n_segments)
    left = pi_ns
    for s in range(ly.n_segments):
        durs[s] = min(left, ly.max_ns)
        left -= durs[s]
    big = dataclasses.replace(ly, n_blocks=spec.n_points)
    amp = np.where(durs > 0, amplitude, rest_amplitude)
    durs = np.where(durs > 0, durs, rest_ns)
    z = preactivations_for(big, amplitude=np.broadcast_to(amp, (spec.n_points, ly.n_segments)), duration_ns=durs)
    return z


class LearnedProtocol(torch.nn.Module):
    """The protocol's free parameters: a :class:`PulseEncoder` without inputs whose ``n_blocks`` are the points."""

    def __init__(self, spec: SensorSpec, init: np.ndarray | None = None) -> None:
        super().__init__()
        self.spec = spec
        ly = spec.layout
        self.big = dataclasses.replace(ly, n_blocks=spec.n_points)
        base = np.repeat(spec.base_hz()[:, None], ly.n_segments, axis=1)
        self.encoder = PulseEncoder(self.big, base, n_inputs=0, init=init)

    def forward(self) -> BlockParams:
        """Per-point parameters, (n_points, 1, n_segments) each."""
        p = self.encoder(None, 1)
        return BlockParams(
            *(x.reshape(self.spec.n_points, 1, -1) for x in (p.amplitude, p.phase_rad, p.frequency_hz, p.duration_ns))
        )

    def suffixes(self, offset_hz: float = 0.0) -> SequenceArrays:
        """The points' suffixes as NumPy arrays (grid durations), (n_points, bins)."""
        return suffix_to_numpy(self(), self.spec.layout, offset_hz=offset_hz)


def _prefix(spec: SensorSpec, fields_t: np.ndarray) -> SequenceArrays:
    g = spec.layout.grid_ns
    n_init, n = spec.init_ns // g, (spec.init_ns + spec.settle_ns) // g
    b = fields_t.shape[0]
    laser = np.zeros((b, n), dtype=np.float32)
    laser[:, :n_init] = 1.0
    z = np.zeros((b, n), dtype=np.float32)
    return SequenceArrays(laser, z, np.zeros((b, n)), z.copy(), np.asarray(fields_t, np.float64), g)


def _repeat(p: BlockParams, b: int) -> BlockParams:
    return BlockParams(*(x.repeat(b, 1, 1) for x in (p.amplitude, p.phase_rad, p.frequency_hz, p.duration_ns)))


def protocol_means(
    protocol: LearnedProtocol,
    fields_t: np.ndarray,
    env: EnvParams,
    options: ModelOptions,
    device: torch.device,
    *,
    offset_hz: float = 0.0,
    duration_shift_ns: float = 0.0,
    params: BlockParams | None = None,
) -> torch.Tensor:
    """Noiseless mean counts per shot (B, T, F) of the protocol for B fields sharing one nuisance draw ``env``
    (hidden field as the magnet, ``env.b_field_t`` = 0, as milestone 2's banks); differentiable in the protocol."""
    spec = protocol.spec
    b, p_n = fields_t.shape[0], spec.n_points
    st0, st, grp = prefix_states_and_statics(_prefix(spec, fields_t), env, options, device)
    prm = protocol() if params is None else params
    rows = _repeat(prm, b)  # field-major: field i, point p -> row i * P + p
    counts = encoded_counts(
        rows,
        spec.layout,
        env,
        st0.repeat_interleave(p_n, dim=0),
        st,
        group=grp.repeat_interleave(p_n),
        offset_hz=offset_hz,
        duration_shift_ns=duration_shift_ns,
    )  # (B * P, T)
    return counts.reshape(b, p_n, spec.n_time).transpose(1, 2)


@torch.no_grad()
def simulate_learned_bank(
    protocol: LearnedProtocol,
    fields_t: np.ndarray,
    envs: list[EnvParams],
    spec: BankSpec,
    device: torch.device,
    *,
    offset_hz: float = 0.0,
    duration_shift_ns: float = 0.0,
    workers: int = 2,
    progress: Callable[[int, int], None] | None = None,
) -> FieldBank:
    """A :class:`FieldBank` (means (N, T, F)) of the frozen protocol; group g of ``fields_per_env`` fields uses
    ``envs[g]`` (as :func:`glowbox.learn.field_data.simulate_bank`)."""
    opt = spec.options()
    n, fpe = fields_t.shape[0], spec.fields_per_env
    sp = protocol.spec
    means = np.empty((n, sp.n_time, sp.n_points), dtype=np.float32)
    nuis = np.empty((n, len(NUISANCE_NAMES)))
    params = protocol()
    params = BlockParams(
        *(x.detach() for x in (params.amplitude, params.phase_rad, params.frequency_hz, params.duration_ns))
    )
    t0 = time.perf_counter()

    def one(g: int) -> int:
        sl = slice(g * fpe, min(n, (g + 1) * fpe))
        sub = fields_t[sl]
        if sub.shape[0] == 0:
            return 0
        env = envs[g]
        nuis[sl] = [getattr(env, k) for k in NUISANCE_NAMES]
        m = protocol_means(
            protocol, sub, env, opt, device, offset_hz=offset_hz, duration_shift_ns=duration_shift_ns, params=params
        )
        means[sl] = m.float().cpu().numpy()
        return int(sub.shape[0])

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for k in pool.map(one, range(len(envs))):
            done += k
            if progress is not None:
                progress(done, n)
    proto = SweepProtocol(
        kind="pulsed",
        f_start_hz=sp.f_start_hz,
        f_stop_hz=sp.f_stop_hz,
        step_hz=(sp.f_stop_hz - sp.f_start_hz) / sp.n_points,
        mw_offset_hz=offset_hz,
        init_ns=sp.init_ns,
        settle_ns=sp.settle_ns,
        readout_ns=sp.layout.readout_ns,
    )
    opts: dict[str, float | int | str] = {
        "n_detuning": spec.n_detuning,
        "detuning_rule": spec.detuning_rule,
        "detuning_spacing": spec.detuning_spacing,
        "fields_per_env": fpe,
        "learned": 1,
    }
    return FieldBank(proto, fields_t.copy(), nuis, means, time.perf_counter() - t0, opts)


def window_bank(bank: FieldBank, window_bins: int) -> FieldBank:
    """Sum a time-resolved bank's readout bins into windows of ``window_bins`` (counts add, so means add)."""
    n, t, f = bank.means.shape
    if t % window_bins:
        raise ValueError("readout bins must be a multiple of the window")
    m = bank.means.reshape(n, t // window_bins, window_bins, f).sum(axis=2)
    return FieldBank(bank.protocol, bank.fields_t, bank.nuisance, m, bank.seconds, dict(bank.options))


@dataclass
class ProtocolTrainLog:
    step: list[int] = dataclasses.field(default_factory=list)
    loss: list[float] = dataclasses.field(default_factory=list)
    seconds: float = 0.0


def train_protocol(
    protocol: LearnedProtocol,
    decoder: torch.nn.Module,
    prior: FieldPrior,
    scaler: TargetScaler,
    options: ModelOptions,
    device: torch.device,
    *,
    steps: int,
    fields_per_step: int,
    lr_protocol: float,
    lr_decoder: float,
    shots_range: tuple[float, float],
    rng: np.random.Generator,
    gen: torch.Generator,
    offset_hz: float = 0.0,
    mse_warmup: float = 0.05,
    log_every: int = 50,
    log: Callable[[int, dict[str, float]], None] | None = None,
) -> ProtocolTrainLog:
    """Phase 1: protocol and decoder together through the surrogate. Every step draws one nuisance env and
    ``fields_per_step`` fresh fields from the prior, simulates the protocol for them (differentiable), adds Gaussian
    shot noise with the Poisson variance at log-uniform shots and takes an Adam step on the Gaussian NLL of the
    folded field (plain MSE for the first ``mse_warmup`` of the steps, as milestone 2)."""
    trainable = [*protocol.parameters(), *decoder.parameters()]
    groups = [
        {"params": list(protocol.parameters()), "lr": lr_protocol},
        {"params": list(decoder.parameters()), "lr": lr_decoder},
    ]
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps))))
    )
    warm = int(mse_warmup * steps)
    hist = ProtocolTrainLog()
    t0 = time.perf_counter()
    run = 0.0
    lo, hi = shots_range
    for step in range(1, steps + 1):
        protocol.train()
        decoder.train()
        env = prior.sample_env(rng)
        fields = prior.sample_fields(fields_per_step, rng)
        mu = protocol_means(protocol, fields, env, options, device, offset_hz=offset_hz)
        shots = sample_shots(fields_per_step, lo, hi, gen, device)
        counts = gaussian_shot_noise(mu, shots, gen) * shots[:, None, None]
        y = torch.as_tensor(scaler.encode(fields), dtype=torch.float32, device=device)
        mean, log_var = decoder(features(counts.float()), log_shots(shots))
        loss = ((mean - y) ** 2).mean() if step <= warm else gaussian_nll(mean, log_var, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        run += float(loss.detach())
        if step % log_every == 0 or step == steps:
            k = log_every if step % log_every == 0 else step % log_every
            hist.step.append(step)
            hist.loss.append(run / k)
            if log is not None:
                log(step, {"loss": run / k})
            run = 0.0
    hist.seconds = time.perf_counter() - t0
    return hist
