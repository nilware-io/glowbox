"""Throughput benchmark of the surrogate: ``python -m glowbox.sim.surrogate.bench [--device cuda] [--quick]``.

Reports, per workload, wall time, distinct sequences per second and shots per second (shots are cheap once the
noiseless rate of a sequence is known, unless laser drift is on, which makes every shot a distinct simulation).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import numpy as np
import torch

from glowbox.control.pulse import SequenceArrays, cw_odmr_point, pulsed_rabi, stack
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate.model import ModelOptions, expected_rate_cps
from glowbox.sim.surrogate.simulator import SurrogateSimulator

ENV = EnvParams(b_field_t=(0.0, 0.0, 3e-3))
F_RES_HZ = 2.921e9
#: Segment layout of the pulse-level gradient workload.
N_SEG, SEG_BINS = 20, 20


def _timed(fn: Callable[[], object], device: torch.device, repeat: int = 3) -> float:
    fn()  # warm-up (allocator, kernels)
    best = float("inf")
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quick", action="store_true", help="smaller batches (CPU smoke run)")
    args = ap.parse_args()
    dev = torch.device(args.device)
    q = args.quick
    results = []

    def report(name: str, seconds: float, sequences: int, shots: int, **extra: object) -> None:
        row = {
            "workload": name,
            "device": dev.type,
            "seconds": round(seconds, 4),
            "sequences_per_s": round(sequences / seconds, 1),
            "shots_per_s": round(shots / seconds, 1),
            **extra,
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    opts = ModelOptions(n_detuning=32)
    sim = SurrogateSimulator(dev, opts)

    # 1. Rabi sweep (init 3 us, settle 3 us, MW 0..T, readout 3 us), 1000 shots each, full Simulator path.
    n_rabi = 64 if q else 256
    rabi = [pulsed_rabi(10 * i, F_RES_HZ) for i in range(n_rabi)]
    shots = 1000
    rng = np.random.default_rng(0)
    t = _timed(lambda: sim.run_batch(rabi, ENV, shots=shots, rng=rng), dev, repeat=2)
    report("rabi_run_batch", t, n_rabi, n_rabi * shots, n_bins=rabi[-1].n_bins, n_detuning=32, shots_each=shots)

    # 2. CW ODMR sweep, 5 us per point, noiseless rates only.
    n_odmr = 128 if q else 512
    odmr = stack([cw_odmr_point(f, duration_ns=5000) for f in np.linspace(2.80e9, 2.94e9, n_odmr)])
    t = _timed(lambda: expected_rate_cps(odmr, ENV, options=opts, device=dev), dev)
    report("odmr_rates", t, n_odmr, n_odmr, n_bins=odmr.n_bins, n_detuning=32)

    # 3. Differentiable path: random per-bin controls (encoder-like), forward + backward.
    for bsz, nb, m in ((16, 200, 8),) if q else ((64, 400, 8), (256, 400, 8), (64, 400, 32)):
        g = torch.Generator().manual_seed(0)
        laser = (torch.rand(bsz, nb, generator=g) > 0.5).float()
        amp = torch.rand(bsz, nb, generator=g).requires_grad_(True)
        phase = (torch.rand(bsz, nb, generator=g) * 6.28 - 3.14).requires_grad_(True)
        freq = (F_RES_HZ + 5e6 * torch.randn(bsz, nb, generator=g, dtype=torch.float64)).requires_grad_(True)

        def fwd_bwd(
            laser: torch.Tensor = laser,
            amp: torch.Tensor = amp,
            phase: torch.Tensor = phase,
            freq: torch.Tensor = freq,
            m: int = m,
            bsz: int = bsz,
        ) -> None:
            arr = SequenceArrays(laser, amp, freq, phase, torch.zeros(bsz, 3, dtype=torch.float64), 10)
            rate = expected_rate_cps(arr, ENV, options=ModelOptions(n_detuning=m), device=dev)
            rate.sum().backward()

        t = _timed(fwd_bwd, dev, repeat=2)
        report("grad_fwd_bwd", t, bsz, bsz, n_bins=nb, n_detuning=m)

    # 3b. Differentiable, pulse-level controls (encoder-like): 20 segments x 20 bins, one (laser, amp, phase,
    # freq) per segment and sequence. Identical bins share propagators (with exact per-bin derivatives).
    for bsz, m in ((16, 8),) if q else ((64, 8), (256, 8), (256, 32)):
        g = torch.Generator().manual_seed(1)
        seg_laser = (torch.rand(bsz, N_SEG, generator=g) > 0.5).float()
        seg_amp = torch.rand(bsz, N_SEG, generator=g).requires_grad_(True)
        seg_phase = (torch.rand(bsz, N_SEG, generator=g) * 6.28 - 3.14).requires_grad_(True)
        seg_freq = (F_RES_HZ + 5e6 * torch.randn(bsz, N_SEG, generator=g, dtype=torch.float64)).requires_grad_(True)

        def fwd_bwd_seg(
            sl: torch.Tensor = seg_laser,
            sa: torch.Tensor = seg_amp,
            sp: torch.Tensor = seg_phase,
            sf: torch.Tensor = seg_freq,
            m: int = m,
            bsz: int = bsz,
        ) -> None:
            def rep(x: torch.Tensor) -> torch.Tensor:
                return x.repeat_interleave(SEG_BINS, dim=1)

            arr = SequenceArrays(rep(sl), rep(sa), rep(sf), rep(sp), torch.zeros(bsz, 3, dtype=torch.float64), 10)
            rate = expected_rate_cps(arr, ENV, options=ModelOptions(n_detuning=m), device=dev)
            rate.sum().backward()

        t = _timed(fwd_bwd_seg, dev, repeat=2)
        report("grad_fwd_bwd_segments", t, bsz, bsz, n_bins=N_SEG * SEG_BINS, n_detuning=m)

    # 4. Laser drift on: every shot simulated separately (exact).
    shots_d = 32 if q else 256
    env_d = EnvParams(b_field_t=ENV.b_field_t, laser_drift_rel_sigma=0.02)
    seq = pulsed_rabi(60, F_RES_HZ)
    sim8 = SurrogateSimulator(dev, ModelOptions(n_detuning=8))
    t = _timed(lambda: sim8.run(seq, env_d, shots=shots_d, rng=rng), dev, repeat=1)
    report("rabi_with_drift", t, 1, shots_d, n_bins=seq.n_bins, n_detuning=8)

    if dev.type == "cuda":
        print(json.dumps({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}))


if __name__ == "__main__":
    main()
