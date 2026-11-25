# Fast surrogate (`glowbox/sim/surrogate/`)

PyTorch implementation of docs/physics.md, batched over sequences, orientations and static-detuning samples,
device-agnostic (CUDA when available), differentiable with respect to the continuous controls. It implements the
`Simulator` protocol (`SurrogateSimulator.run`), plus `run_batch` and the differentiable `expected_rate_cps`.

## Model

Not a reduction: the seven-level model of section 6 in its exact reduced form. Every jump of section 6 is its own
operator |j><i|, so excited and singlet levels never carry coherence; the state of one NV (orientation k, detuning
sample j) is the ground 3x3 density matrix (9 reals, eigenbasis of H0, rotating frame) + 3 excited populations + the
singlet population + a photon accumulator: 14 reals. Controls are constant per grid bin, so each bin's propagator
is `matrix_exp(L dt)` of a constant 14x14 generator: no sub-stepping error, and the accumulator row gives the
bin-integrated emission k_r (p4 + p5 + p6) exactly, including the ~12 ns excited-state rise and tail.

Generator per bin: `L = L0 + beta K_pump + a (cos(phi) A + sin(phi) B) + Delta- K_m + Delta+ K_p`, with
Delta+- = e+- - F (detunings from the frame frequency F, formed in float64). L0 holds T1, T2, excited and singlet
decays and returns; all pieces are built once per static field in float64 NumPy (numerical 3x3 diagonalisation)
and cast to the working dtype (float32 by default).

## Choices the spec left open (should be written into docs/physics.md, the reference must match)

1. **Static detuning is `delta * Sz` added to H0 before diagonalisation** (a field shift along the NV axis,
   section 4's mechanism). This gives f+- -> f+- +- delta whenever gamma_e B_par >> E, gamma_e B_perp, and is well
   defined at zero field (degenerate ms=+-1). It differs from a literal "shift the two eigenvalues by +-delta" when
   E or B_perp dominate (there delta enters quadratically). Samples: M equal-probability quantile midpoints
   (default M = 32, `ModelOptions.n_detuning`), same as the reference's quadrature rule.
2. **Dissipators in the rotating frame: secular approximation.** Jump operators are defined in the ms basis (lab).
   In the frame rotating at f on the two upper eigenstates, superoperator elements that couple coherences of
   different frame order (0~-+~ vs populations / -~+~) oscillate at ~f and are dropped (the dissipative half of the
   RWA). Exact when 0~ is pure ms=0; otherwise the error is ~(gamma_e B_perp / D)^2 (1e-4 per mT^2).
3. **Which eigenstate is 0~:** the one with the largest ms=0 weight; -~, +~ are the others by energy.
4. **Frame bookkeeping:** in MW bins the frame frequency is the bin's MW frequency; in MW-off bins it is held at
   the last one (before the first MW bin: the first MW frequency; no MW at all: D). A frequency change at bin
   start t_s applies exp(-i 2 pi (f2 - f1) t_s) to rho_{0~,+-~} (section 3's absolute-time rule). The frame choice
   is exact, it does not change any result (`test_frame_switching_is_exact`); `test_frequency_switch_phase_sign`
   pins the sign against the lab definition.
5. **`t2_stretch != 1` is refused** (ValueError), like the reference, rather than faked with a non-Markovian rate.
6. **Drift with `run_batch`:** one OU chain per sequence (each sequence is its own `run()` call).

## Requested spec / interface changes (resolved at integration)

- docs/physics.md now states the delta * Sz detuning (section 4), the secular dissipator rule, the 0~ rule and the
  frame bookkeeping (section 3); the reference was changed to delta * Sz.
- `SequenceArrays` documents that `mw_frequency_hz == 0` inherits the previous frequency. An optional per-bin
  `mw_fill` in `SequenceArrays` is deferred to the encoder work (phase 2), where differentiable durations are first
  needed; until then it stays a separate argument of `expected_rate_cps`.
- `FluorescenceTrace.rate_cps` may be a read-only view; the interface docstring says to copy before writing.

## Differentiability

`expected_rate_cps(arrays, env, options=..., mw_fill=..., laser_factor=...)` takes torch tensors that require grad
(laser power, MW amplitude, phase, frequency as float64, `mw_fill`) and returns (B, n_bins) counts/s. Three paths:

- **forward only, deduplicated**: one propagator per distinct bin (field, beta, a cos phi, a sin phi, frame
  frequency, fill), `torch.linalg.matrix_exp`;
- **shared with gradients** (distinct bins <= 25 % of bins, e.g. pulse-level controls): P_b = P_u + sum_i
  (c_bi - c_bi.detach()) dP_u/dc_i with exact Frechet derivatives dP_u/dc_i. Values unchanged, per-bin first
  derivatives exact (`test_shared_propagator_gradients_equal_dense`); no second derivatives on this path;
- **dense**: one propagator per (sequence, bin) by a Taylor scaling-and-squaring exponential (backward ~2x forward
  instead of ~8x for `matrix_exp`), chunked and checkpointed.

Durations: integer on the grid in `PulseSequence`; for gradients, `mw_fill_for_duration(start_ns, duration_ns,
n_bins, grid_ns)` turns a continuous duration into fill fractions (exact two-part propagator in the partial bin).
Gradients are checked against finite differences in float64 (`gradcheck`) for every control and the duration.

## Validation (tests/sim/test_surrogate.py, ~10-20 s CPU)

ODMR dip positions within 0.3 MHz of an independent diagonalisation (2 lines for B || [001], 8 for a generic
field); Rabi frequency = rabi_full_scale * a * |m_perp| within 2 % at a = 1 and 0.5; detuned 200 MHz = MW off within
1e-3; Ramsey fringe frequency within 2 % of the detuning and envelope of order T2*; Hahn echo decay exp(-2 tau/T2)
within 0.05 and refocusing a 0.5 MHz detuning; frame switch sign and exactness; float32 vs float64, dedup vs dense,
Taylor vs `matrix_exp`; Poisson mean; drift seeding; CUDA vs CPU (gpu-marked). Surrogate-vs-reference agreement: tests/sim/test_surrogate_vs_reference.py.

## Throughput (2026-11-25)

GPU machine (RTX 5070 Ti, torch 2.11.0+cu128, float32), `python -m glowbox.sim.surrogate.bench`:

| workload | size | time | rate |
| --- | --- | --- | --- |
| Rabi sweep, `run_batch`, 1000 shots each | 256 seqs x 955 bins, M = 32 | 0.69 s | 369 seq/s, 369 k shots/s |
| CW ODMR rates | 512 seqs x 500 bins, M = 32 | 0.18 s | 2870 seq/s |
| grad, pulse-level controls (20 segments) | 256 x 400 bins, M = 8 | 1.15 s | 222 seq/s fwd+bwd |
| grad, pulse-level controls | 256 x 400 bins, M = 32 | 4.5 s | 57 seq/s |
| grad, independent per-bin controls | 256 x 400 bins, M = 8 | 4.6 s | 56 seq/s |
| Rabi with laser drift (every shot simulated) | 256 shots x 706 bins, M = 8 | 1.56 s | 164 shots/s |

Local CPU (WSL, under heavy load from other jobs, `--quick`): Rabi 64 seqs x 1000 shots 2.7 s (24 k shots/s),
ODMR 128 seqs 1.5 s (88 seq/s), per-bin grad 16 x 200 bins 7.3 s, pulse-level grad 16 x 400 bins 1.7 s.

Shots are cheap once a sequence's noiseless rate is known (Poisson sampling on the device, seeded from the caller's
`rng`); the cost is per distinct sequence. Laser drift makes every shot a distinct simulation. The per-bin
time loop is launch-bound for small batches; batch sequences.
