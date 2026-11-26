"""Task 2 (project brief): store a short bit pattern in the pulse response and recall it.

Protocol (one shot, all times on the 10 ns grid)::

    init laser (init_ns) | dark settle (settle_ns) | WRITE: MW block (n_segments x slot_ns) | dark DELAY |
    READ pulse (fixed, optional) | readout laser (readout_ns, recorded per 10 ns bin)

* **Write.** The L bits b in {0, 1}^L enter as u = 2 b - 1 into a fixed random pulse encoder
  (:class:`~glowbox.learn.reservoir.RandomPulseEncoder`, one instance per seed): every MW segment's amplitude,
  phase, frequency (near one of the device's ODMR lines) and duration depend on all bits. The write block has the
  same length for every L (the comparison over L is at equal write time).
* **Store.** During the dark delay the spins keep what the write did to them: population changes for ~T1 (5 ms),
  coherences for ~T2* (~1 us, static detuning spread; no echo pulse).
* **Read.** The readout laser; its time-resolved fluorescence is the "pulse response" from which the pattern is
  recalled. It sees populations only. Optionally a fixed pi/2-like read pulse on one line comes first: it turns
  part of the stored coherence into population, so what is read then depends on the delay on the T2* scale.

Every pattern's noiseless readout (its *template*) is computed by the surrogate; noisy recalls draw Poisson counts
for a number of shots (the same pattern repeated). All 2^L patterns are in the codebook (L <= 12), so readouts are
calibrated on every pattern and tested on fresh shot noise. Readouts:

* :func:`bitwise_map` (oracle): the Bayes-optimal bit decision given the exact templates and the Poisson model
  (posterior over patterns, marginalised per bit) - the best any readout of this protocol can do;
* the reservoir readout: :class:`~glowbox.learn.reservoir.RidgeReadout` on counts per shot, targets u, decision
  sign (linear);
* the CNN decoder (:class:`~glowbox.learn.decoder.ConvDecoder` with one logit per bit, BCE), trained on fresh
  noise over a log-uniform range of shots.

Metrics (:func:`recall_metrics`): bit accuracy, pattern accuracy (every bit right) and *recalled bits*
sum_i (1 - H2(e_i)), the capacity of L binary symmetric channels with the measured per-bit error rates e_i (a
lower-bound-style summary of how much of the pattern comes back).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from glowbox.analysis.field_fit import transition_lines_hz
from glowbox.control.pulse import SequenceArrays
from glowbox.learn.reservoir import (
    EncoderSpec,
    RandomPulseEncoder,
    ReadPulse,
    compose_suffix,
    init_prefix,
    simulate_suffixes,
)
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions

#: Largest pattern length with an enumerated codebook (2^12 = 4096 templates).
MAX_BITS = 12


@dataclass(frozen=True)
class MemoryProtocol:
    """Timing of the write-delay-read protocol (ns) and the fixed read pulse (None: no read pulse)."""

    n_bits: int
    delay_ns: int
    init_ns: int = 1500
    settle_ns: int = 1000
    readout_ns: int = 1000
    #: Optional read pulse: index of the line it drives (axis-major, :func:`device_lines_hz`; None = no read
    #: pulse), amplitude, duration.
    read_line: int | None = None
    read_amplitude: float = 0.5
    read_ns: int = 60
    encoder: EncoderSpec = field(default_factory=lambda: EncoderSpec(n_inputs=1))
    grid_ns: int = 10

    def __post_init__(self) -> None:
        if not 1 <= self.n_bits <= MAX_BITS:
            raise ValueError(f"n_bits must be in [1, {MAX_BITS}]")
        if self.encoder.n_inputs != self.n_bits:
            object.__setattr__(self, "encoder", _with_inputs(self.encoder, self.n_bits))

    @property
    def readout_bins(self) -> int:
        return self.readout_ns // self.grid_ns

    @property
    def shot_ns(self) -> int:
        read = self.read_ns if self.read_line is not None else 0
        return (
            self.init_ns
            + self.settle_ns
            + self.encoder.n_segments * self.encoder.slot_ns
            + self.delay_ns
            + read
            + (self.readout_ns)
        )


def _with_inputs(spec: EncoderSpec, n: int) -> EncoderSpec:
    from dataclasses import replace

    return replace(spec, n_inputs=n)


def device_lines_hz(env: EnvParams) -> np.ndarray:
    """The eight ODMR lines (Hz, axis-major f-, f+) of the device's static field and strain: what a calibrated
    setup knows about its own sample."""
    return np.asarray(transition_lines_hz(np.asarray(env.b_field_t), env.d_hz, env.e_hz), dtype=np.float64)


def all_patterns(n_bits: int) -> np.ndarray:
    """Every bit pattern of length ``n_bits``, (2^L, L) in {0, 1}; row j is the binary expansion of j (LSB first)."""
    j = np.arange(2**n_bits)[:, None]
    return ((j >> np.arange(n_bits)[None]) & 1).astype(np.int64)


def make_encoder(protocol: MemoryProtocol, env: EnvParams, seed: int) -> RandomPulseEncoder:
    return RandomPulseEncoder(protocol.encoder, device_lines_hz(env), seed)


def codebook(
    protocol: MemoryProtocol,
    encoder: RandomPulseEncoder,
    env: EnvParams,
    *,
    offset_hz: float = 0.0,
    duration_shift_ns: int = 0,
    options: ModelOptions | None = None,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(patterns (2^L, L), templates (2^L, readout_bins)): mean counts per shot per readout bin of every pattern.

    ``offset_hz`` shifts every MW frequency (write and read pulse): the quantum test's detuned arm.
    """
    pats = all_patterns(protocol.n_bits)
    pre, suf = sequences(protocol, encoder, env, pats, offset_hz=offset_hz, duration_shift_ns=duration_shift_ns)
    return pats, simulate_suffixes(pre, suf, env, protocol.readout_bins, options=options, device=device)


def sequences(
    protocol: MemoryProtocol,
    encoder: RandomPulseEncoder,
    env: EnvParams,
    patterns: np.ndarray,
    *,
    offset_hz: float = 0.0,
    duration_shift_ns: int = 0,
) -> tuple[SequenceArrays, SequenceArrays]:
    """(prefix, suffixes) arrays of ``patterns`` (N, L) for :func:`~glowbox.learn.reservoir.simulate_suffixes`."""
    p = protocol
    u = 2.0 * np.asarray(patterns, dtype=np.float64) - 1.0
    block = encoder.block(u, offset_hz=offset_hz, duration_shift_ns=duration_shift_ns)
    rp = None
    if p.read_line is not None:
        rp = ReadPulse(float(device_lines_hz(env)[p.read_line]), p.read_amplitude, p.read_ns)
    suf = compose_suffix(u.shape[0], block, p.delay_ns, p.readout_ns, rp, offset_hz=offset_hz, grid_ns=p.grid_ns)
    pre = init_prefix(p.init_ns, p.settle_ns, grid_ns=p.grid_ns)
    return pre, suf


# --- Readouts ---------------------------------------------------------------------------------------------------------


@torch.no_grad()
def bitwise_map(
    counts: torch.Tensor, shots: torch.Tensor | float, templates: torch.Tensor, patterns: torch.Tensor, chunk: int = 512
) -> torch.Tensor:
    """Posterior P(b_i = 1 | counts) per bit under the Poisson model with uniform pattern prior, (N, L).

    ``counts`` (N, T) summed over ``shots`` (scalar or (N,)), ``templates`` (P, T) mean counts per shot,
    ``patterns`` (P, L) in {0, 1}. Thresholding at 1/2 is the Bayes-optimal bitwise decision.
    """
    lam = templates.clamp_min(1e-12).to(torch.float64)
    log_lam = torch.log(lam)
    pats = patterns.to(torch.float64)
    n = counts.shape[0]
    s = torch.as_tensor(shots, dtype=torch.float64, device=counts.device).expand(n)
    out = []
    for i in range(0, n, chunk):
        c = counts[i : i + chunk].to(torch.float64)
        si = s[i : i + chunk, None]
        ll = c @ log_lam.T - si * lam.sum(dim=1)[None]  # (n, P); log(shots) and log(c!) terms are constant
        post = torch.softmax(ll, dim=1)
        out.append(post @ pats)
    return torch.cat(out).to(torch.float32)


def binary_entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    out: np.ndarray = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    return out


def recall_metrics(pred_bits: np.ndarray, true_bits: np.ndarray) -> dict[str, float]:
    """Bit accuracy, pattern accuracy, recalled bits (sum over bits of 1 - H2(error rate)), per-bit min accuracy."""
    pb, tb = np.asarray(pred_bits), np.asarray(true_bits)
    correct = pb == tb
    e = 1.0 - correct.mean(axis=0)  # per-bit error rate
    return {
        "bit_acc": float(correct.mean()),
        "pattern_acc": float(correct.all(axis=1).mean()),
        "recalled_bits": float(np.sum(1.0 - binary_entropy(np.minimum(e, 0.5)))),
        "worst_bit_acc": float(1.0 - e.max()),
    }


def template_separation(templates: np.ndarray, shots: float) -> float:
    """Median over patterns of the distance to the nearest other template in units of the Poisson noise (sqrt of
    the mean counts) at ``shots``: >~ 3 means patterns are distinguishable."""
    t = np.asarray(templates, dtype=np.float64) * shots
    if t.shape[0] < 2:
        return float("inf")
    sig = np.sqrt(np.maximum(t.mean(axis=0), 1e-12))
    z = t / sig
    d2 = np.sum(z**2, axis=1)[:, None] + np.sum(z**2, axis=1)[None] - 2 * z @ z.T
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(np.maximum(d2.min(axis=1), 0.0))))
