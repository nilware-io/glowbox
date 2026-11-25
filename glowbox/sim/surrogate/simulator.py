"""The fast surrogate as a :class:`glowbox.sim.interface.Simulator`."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from scipy.signal import lfilter

from glowbox.control.pulse import PulseSequence, stack
from glowbox.sim.interface import EnvParams, FluorescenceTrace
from glowbox.sim.surrogate.model import ModelOptions, expected_rate_cps

SURROGATE_VERSION = "0.1"


def laser_drift(env: EnvParams, n_steps: int, grid_ns: int, rng: np.random.Generator) -> np.ndarray:
    """Ornstein-Uhlenbeck relative laser drift x on the grid, exact discretisation (docs/physics.md section 7).

    Starts from the stationary distribution N(0, sigma^2); returns x, shape (n_steps,).
    """
    sigma = env.laser_drift_rel_sigma
    if sigma == 0:
        return np.zeros(n_steps)
    a = np.exp(-grid_ns * 1e-9 / env.laser_drift_tau_s)
    xi = rng.standard_normal(n_steps)
    drive = sigma * np.sqrt(1.0 - a * a) * xi
    drive[0] = sigma * xi[0]  # x_0 ~ N(0, sigma^2)
    return np.asarray(lfilter([1.0], [1.0, -a], drive))


class SurrogateSimulator:
    """Batched PyTorch implementation of docs/physics.md (seven-level model, exact per-bin propagators).

    ``device`` defaults to CUDA when available. ``run_batch`` simulates many sequences in one pass (the fast path);
    ``run`` is the :class:`~glowbox.sim.interface.Simulator` contract for one sequence.
    """

    name = "surrogate"

    def __init__(self, device: str | torch.device | None = None, options: ModelOptions | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.options = options or ModelOptions()

    def run(
        self,
        sequence: PulseSequence,
        env: EnvParams,
        *,
        shots: int = 1,
        rng: np.random.Generator | None = None,
    ) -> FluorescenceTrace:
        return self.run_batch([sequence], env, shots=shots, rng=rng)[0]

    def expected_rates(self, sequences: Sequence[PulseSequence], env: EnvParams) -> list[np.ndarray]:
        """Noiseless rate_cps (no drift) per sequence, each of shape (n_bins,)."""
        arrays = stack(sequences)
        with torch.no_grad():
            rate = expected_rate_cps(arrays, env, options=self.options, device=self.device)
        r = rate.detach().cpu().numpy().astype(np.float64)
        return [np.clip(r[i, : s.n_bins], 0.0, None) for i, s in enumerate(sequences)]

    def run_batch(
        self,
        sequences: Sequence[PulseSequence],
        env: EnvParams,
        *,
        shots: int = 1,
        rng: np.random.Generator | None = None,
    ) -> list[FluorescenceTrace]:
        """Simulate each sequence ``shots`` times. Laser drift runs as one chain per sequence (one run() call)."""
        rng = rng or np.random.default_rng()
        if shots < 1:
            raise ValueError("shots must be >= 1")
        if env.laser_drift_rel_sigma == 0:
            # Without drift every shot has the same expected rate: a read-only broadcast view, no copy.
            rates = [np.broadcast_to(r, (shots, r.shape[0])) for r in self.expected_rates(sequences, env)]
        else:
            rates = [self._drifting_rate(s, env, shots, rng) for s in sequences]
        traces = []
        for seq, rate in zip(sequences, rates, strict=True):
            dt_s = seq.grid_ns * 1e-9
            mean = rate * dt_s if rate.strides[0] else np.broadcast_to(rate[0] * dt_s, rate.shape)
            counts = self._poisson(mean, rng)
            meta = {
                "source": self.name,
                "surrogate_version": SURROGATE_VERSION,
                "n_detuning": self.options.n_detuning,
                "detuning_rule": self.options.detuning_rule,
                "detuning_spacing": self.options.detuning_spacing,
                "dtype": str(self.options.dtype).removeprefix("torch."),
                "device": self.device.type,
                "label": dict(seq.label),
            }
            traces.append(FluorescenceTrace(counts, seq.grid_ns, rate, meta))
        return traces

    def _poisson(self, mean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Poisson counts, sampled on the device with a torch generator seeded from ``rng``.

        ``rng`` stays the only source of randomness (same state, same device -> same counts); sampling on the GPU
        is what makes millions of shot-bins per second possible.
        """
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(rng.integers(0, 2**63 - 1)))
        if mean.ndim == 2 and mean.strides[0] == 0:  # broadcast view (no drift): expand on the device, no copy
            lam = torch.as_tensor(np.array(mean[0]), dtype=torch.float32, device=self.device)
            lam = lam.expand(mean.shape)
        else:
            lam = torch.as_tensor(np.ascontiguousarray(mean), dtype=torch.float32, device=self.device)
        return torch.poisson(lam, generator=gen).to(torch.int64).cpu().numpy()

    def _drifting_rate(self, seq: PulseSequence, env: EnvParams, shots: int, rng: np.random.Generator) -> np.ndarray:
        """With drift every shot has its own pump rate history: simulate the shots as a batch (exact, slower)."""
        x = laser_drift(env, shots * seq.n_bins, seq.grid_ns, rng).reshape(shots, seq.n_bins)
        factor = np.clip(1.0 + x, 0.0, None)
        arrays = stack([seq] * shots)
        with torch.no_grad():
            rate = expected_rate_cps(
                arrays,
                env,
                options=self.options,
                laser_factor=torch.as_tensor(factor),
                device=self.device,
            )
        return np.clip(rate.detach().cpu().numpy().astype(np.float64), 0.0, None)
