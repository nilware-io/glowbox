# QuTiP reference simulator: implementation notes

Code: `glowbox/sim/reference/` (`model.py` builds the per-axis operators, `simulator.py` propagates and adds
detection). Tests: `tests/sim/test_reference.py` (contract, solver cross-checks, rate-model numbers) and
`tests/sim/test_reference_protocols.py` (the validation suite of docs/physics.md section 9). The spec is
docs/physics.md; this file records how it was implemented, the interpretations made where the spec is silent, and
what the tests measured.

## What it computes

For every populated NV axis (axes that are equivalent by symmetry are simulated once, see below) and every node
δ_m of the T2\* detuning quadrature, a 7×7 density matrix evolves under a Lindblad master equation in the MW
rotating frame:

- levels: ground triplet in the eigenbasis of H0 (|0̃⟩ = most ms=0 weight, |−̃⟩/|+̃⟩ = lower/upper of the other two),
  excited ms = 0, −1, +1 of the axis, singlet;
- H_rot = (f− − f)|−̃⟩⟨−̃| + (f+ − f)|+̃⟩⟨+̃| + δ·Sz (RWA part, item 2 below) + the RWA drive of spec section 3 (matrix elements
  ⟨±̃|u·S|0̃⟩ from the numerical eigenvectors);
- jump operators exactly as in the spec table (pump β k_r, radiative k_r, ISC, singlet returns), T1 (all six ground
  pairs at 1/(3 T1)) and T2 (√(2/T2) Sz), all defined in the ms basis of the axis and transformed to the eigenbasis;
- operators are QuTiP objects (`qutip.lindblad_dissipator`, `spre`/`spost`); the superoperators are then summed
  per bin as dense 49×49 arrays.

Controls are piecewise constant per 10 ns bin, so each bin has a constant Liouvillian L and is propagated **exactly**
by a matrix exponential. The photons emitted in the bin come out of the same exponential with Van Loan's
augmentation, exp([[L, 0], [w, 0]] Δt), whose bottom-left row is w ∫₀^Δt e^{Ls} ds with w·vec(ρ) = k_r (ρ44 + ρ55 +
ρ66). No sub-stepping and no ODE tolerance: the result is exact up to the RWA / secular approximations of the spec
and floating point. Propagators are cached per distinct bin type within a call.

Then e_k = detuning-average per axis; detected rate r = η N Σ p_k e_k + background · laser · max(0, 1 + x) + dark;
counts ~ Poisson(r Δt) per bin and shot; x = exact Ornstein–Uhlenbeck samples on the grid, continuous through the
shots of one call (each shot is then propagated with its own pump rates).

## Public API

```python
from glowbox.sim.reference import ReferenceSimulator, bin_controls, detuning_samples_hz

sim = ReferenceSimulator(n_detuning: int = 64, dedupe_axes: bool = True)
sim.name == "qutip-reference"
sim.run(sequence, env, *, shots=1, rng=None) -> FluorescenceTrace      # Simulator protocol
sim.expected_rate_cps(sequence, env) -> np.ndarray                     # (n_bins,), noiseless, no drift
sim.emission_per_nv(sequence, env, power_factor=None) -> np.ndarray    # (4, n_bins) photons/s per NV per axis
sim.states(sequence, env) -> (models, axes_per_model, states)          # states: (groups, M, n_bins, 49), debug
detuning_samples_hz(env, n) -> np.ndarray                              # the quadrature nodes, Hz
bin_controls(sequence) -> BinControls                                  # float64 per-bin controls
```

`trace.metadata`: `source`, `version`, `model`, `n_detuning`, `broadening_shape`, `qutip_version`, `label`.

## Interpretations and approximations (where the spec is silent)

1. **Secular dissipator for B⊥ ≠ 0 or E ≠ 0.** The spec defines jump operators in the ms basis and the rotating
   frame in the eigenbasis. Written in the rotating frame, a jump operator that mixes |0̃⟩ with |±̃⟩ (possible only
   when the eigenbasis differs from the ms basis) acquires parts oscillating at ±f ≈ 3 GHz. These are split off and
   averaged (Σ_ω D[L_ω] for ω ∈ {−f, 0, +f}), the same order of approximation as the RWA. With B along the axis
   and E = 0 every operator is a single component and nothing is dropped. Validated against a lab-frame
   `qutip.mesolve` without RWA or secular approximation, with B⊥ ≈ 2.4–2.8 mT: populations agree to 2e-3.
2. **Detuning δ·Sz** (spec section 4, reconciled with the surrogate at integration): δ Sz is written in the H0
   eigenbasis and its |0̃⟩ ↔ |±̃⟩ part (rotating at the frame frequency) is dropped, the −̃/+̃ block kept whole.
   With B along the axis and E = 0 this is f± → f± ± δ exactly. (The first version shifted the eigenenergies
   directly, which misses the strain protection at E ≠ 0 and low field.) The surrogate diagonalises H0 + δ Sz per
   node; the per-bin rates of the two agree to 2e-9 at a generic 8-line field (float64).
3. **Frame bookkeeping.** The integration frame starts at the sequence's first MW frequency (D if there is no MW),
   is kept through MW-off bins, and switches at the start of a MW bin with a different frequency by multiplying the
   upper-state coherences by exp(i 2π (f₂ − f₁) t_s) (spec section 3). Validated against an independent mesolve
   that keeps one frame and carries the second frequency as a phase ramp; flipping the sign makes that test fail.
4. **Controls in float64.** The simulator reads pulse values from the `PulseSequence` objects, not from
   `to_arrays()`, whose float32 storage rounds e.g. amplitude 0.1 by about 1.5e-8 relative. Irrelevant in practice;
   mentioned because surrogate-vs-reference comparisons at the 1e-7 level would see it.
5. **Symmetry dedupe** (`dedupe_axes`, only when E = 0): rotating the local x, y about the axis is the unitary
   exp(−iαSz), which leaves every jump operator unchanged up to a phase. Each axis is therefore rotated so that B⊥
   (or, if B⊥ = 0, the MW perpendicular component) lies along x, and axes with identical local field and drive are
   simulated once. For B ∥ [100] (or B = 0 with MW ∥ [100]) the four axes collapse to one group, which makes the
   run four times faster.
6. **Validity warnings** (`UserWarning`, not errors) for B⊥ > 5 mT on a populated axis (excited-state mixing is not
   modelled) and |B∥| > 40 mT (approaching the excited-state anticrossing at about 51 mT).
7. `t2_stretch ≠ 1` raises `ValueError`, as the spec requires.

## Quadrature (spec section 4)

Equal-weight quantile midpoints; the default is M = 64. The Lorentzian's heavy tails are the weak spot. For pulsed
Rabi at f_R/Γ ≈ 50 (T2\* = 1 µs, a = 1), the M = 64 transfer curve deviates from the continuous-distribution
integral by up to 1.2 % of the contrast; at M = 256 the deviation is 0.2 %. The dynamics on the nodes themselves match the
two-level formula to 0.16 %. Use M ≥ 128 when sub-percent agreement with the continuous Lorentzian matters
(long Ramsey or echo times, strong drive). The revival time of the node set is about 2 M T2\*, far beyond any test
sequence.

## Validation results (CPU, this repo's tests)

| check | expected (spec) | measured |
| --- | --- | --- |
| independent rotating-frame mesolve incl. frame switch (populations, full ρ) | equal | ≤ 2e-8 |
| lab-frame mesolve with B⊥ (no RWA, no secular approximation) | equal up to RWA | ≤ 2e-3 in populations |
| CW laser steady state vs classical 7-level rate equations (β = 0.3) | equal | ≤ 1e-6 |
| singlet lifetime 1/(k71 + 2 k72) | 0.41 µs | exact (1e-6) |
| CW ODMR, generic B: 8 lines vs numerical f± per axis | equal | within 50 kHz (parabolic dip fit) |
| same, vs D ± γe B·n | within B⊥ second-order shift | yes |
| ensemble ODMR contrast per line at a = 0.1, β = 0.1 | a few % | 0.3–5 % (4.6 % for the [111]-aligned line at 3 mT) |
| low-power CW FWHM (T2\* = 100 ns, β = 0.01, a = 0.01) | 1/(π T2\*) | 1.11 × (the pump adds about 0.1 MHz of HWHM) |
| CW contrast vs MW power | increasing | monotonic for a = 0.01 … 0.3 |
| pulsed Rabi vs two-level T2\*-damped formula, 4 cases (Lorentzian/Gaussian, a = 1 … 0.25) | equal | ≤ 0.4 % of contrast on nodes, ≤ 1.5 % vs continuous |
| Ramsey fringe frequency (Δ = 2 MHz) | Δ | −0.10 % (Lorentzian), +0.03 % (Gaussian) |
| Ramsey decay, Lorentzian | 1/T2\* + 1/T2 = 1.1e6 s⁻¹ | 1.090e6 (−0.9 %) |
| Ramsey decay, Gaussian (fit with exp(−(t/T2\*)²) · exp(−t/T2)) | 1/T2\* = 1e6 s⁻¹ | 1.038e6 (+3.8 %) |
| Hahn echo decay exp(−2τ/T2), T2 = 4 µs | 2.5e5 s⁻¹ | 2.54e5 (+1.6 %) |
| echo with 1 MHz static detuning | same τ dependence | ratio constant within 3 % (factor about 0.95 from off-resonant pulses) |
| MW 300 MHz off every line vs MW off | identical within shot noise | rate_cps within 2e-3; counts within 4σ |

## Timing

`expected_rate_cps` for a default `pulsed_rabi` (7.06 µs, 706 bins) and a default `cw_odmr_point` (20 µs, 2000
bins):

| M | field | local WSL box (32 threads, heavily loaded) rabi / cw | GPU machine CPU rabi / cw |
| --- | --- | --- | --- |
| 32 | B ∥ [100] (1 group) | 0.19 s / 0.30 s | 0.27 s / 0.50 s |
| 32 | generic (4 groups) | 0.35 s / 0.41 s | 0.54 s / 0.80 s |
| 64 | B ∥ [100] (1 group) | 0.34 s / 0.43 s | 0.44 s / 0.74 s |
| 64 | generic (4 groups) | 0.64 s / 0.83 s | 0.83 s / 1.51 s |

With laser drift on, every lit bin has its own pump rate and needs its own exponential: about 10 s per shot for a
Rabi sequence at M = 32. Use drift with the reference only for short checks, or let the surrogate handle drift.
The reference is CPU-only (dense 50×50 exponentials in complex128 via `torch.linalg.matrix_exp`, which agrees with
`scipy.linalg.expm` to 1e-13). The torch thread count is capped at 4 while it runs, because small matrices
oversubscribed with 32 threads were 20× slower. The cap changes the global torch thread setting for the duration of
the call (restored afterwards), so the simulator is not meant to be called from several Python threads at once.

The full reference test set takes about 60 s locally under load, and 98 s on the GPU machine. The GPU machine run
used a scratch copy outside its clone, via `uv run --project <clone>` with `PYTHONPATH` pointing at the copy; the
clone stayed clean.

## Requested spec / interface changes (resolved at integration)

1. Secular split of the jump operators: now prescribed in docs/physics.md section 3 (the surrogate's element mask
   is the same rule).
2. Detuning: docs/physics.md section 4 now defines δ as δ·Sz in H0; the reference was changed accordingly (item 2).
3. Quadrature numbers (1.2 % at M = 64, 0.2 % at M = 256 for f_R/Γ ≈ 50): in docs/physics.md section 4.
4. `SETTLE_DEFAULT_NS` is now 3000 ns, with the reason in docs/physics.md section 8.
