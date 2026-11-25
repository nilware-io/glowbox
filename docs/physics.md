# Glowbox physics spec

This is the single specification that both simulator implementations follow: the QuTiP reference
(`glowbox/sim/reference*`) and the fast surrogate (`glowbox/sim/surrogate*`). Numbers live in
`glowbox/sim/constants.py` (each with its unit and literature source); the inputs are `PulseSequence`
(`glowbox/control/pulse.py`) and `EnvParams`, the output is `FluorescenceTrace` (`glowbox/sim/interface.py`).
If an implementation needs to deviate, change this document first.

## 0. Units, frames, conventions

- **Frequencies are in Hz (cycles/s).** A Hamiltonian written `H/h` is in Hz; multiply by 2π to get rad/s for a
  solver that integrates `dψ/dt = -i (H/ħ) ψ`. Names carry units: `_hz`, `_s`, `_ns`, `_t` (tesla), `_cps`
  (counts/s), `_per_s` (incoherent rates, 1/s). Tetienne et al. quote rates in "MHz" meaning 10⁶ s⁻¹: these are
  **not** angular frequencies and are never multiplied by 2π.
- **Time grid:** sequences and fluorescence bins share `grid_ns` (default 10 ns). Bin `k` is
  `[k·grid_ns, (k+1)·grid_ns)`. Controls are piecewise constant on bins. Solvers may sub-step inside a bin
  (they must: the excited-state lifetime is ~12 ns and Rabi periods are ~100 ns).
- **Frames:** vectors (`EnvParams.b_field_t`, `MagnetSetting.field_t`, `EnvParams.mw_field_direction`) are in the
  cubic crystal frame x, y, z = [100], [010], [001], assumed aligned with the lab (a (100)-cut plate). The total
  static field is `B = EnvParams.b_field_t + PulseSequence.magnet.field_t`, constant during a sequence.
- **Initial state of every shot:** all NVs in the ground triplet, spin fully mixed (ρ = 𝟙/3 on ms = 0, ±1), no
  excited-state or singlet population. Only the sequence's laser polarises the spins.

## 1. Ground-state spin Hamiltonian (per NV)

For an NV whose symmetry axis is the unit vector **n** (one of the four in §2), with spin-1 operators
S = (Sx, Sy, Sz) quantised along **n**:

    H0/h = D Sz² + E (Sx² − Sy²) + γe B·S

- D = 2.870 GHz (`D_GS_HZ`; Doherty et al., Phys. Rep. 528, 1 (2013)). Temperature dependence
  (dD/dT ≈ −74 kHz/K, Acosta et al., PRL 104, 070801 (2010)) is ignored; treat it as drift if ever needed.
- E: strain splitting, default 0 (`E_GS_DEFAULT_HZ`); ensembles show 0.1 to a few MHz (Barry et al.,
  Rev. Mod. Phys. 92, 015004 (2020)).
- γe = g μB/h = 28.03 GHz/T with g = 2.0028 (`GAMMA_E_HZ_PER_T`; Doherty 2013; μB/h from CODATA 2018). Isotropic
  g assumed.
- **NV-local frame.** z = **n**; x = the normalised projection of the crystal [001] direction onto the plane
  normal to **n** (never degenerate for <111> axes); y = z × x. The strain axis is taken along this x (a
  convention: when E ≪ γe B⊥ or E = 0 it does not matter).
- Then B·S = B∥ Sz + Bx Sx + By Sy with B∥ = B·**n** and (Bx, By) the components in the local x, y.
- With B ∥ **n**: transition frequencies f± = D ± √(E² + (γe B∥)²). In general both implementations
  **diagonalise H0 numerically** (3×3); the surrogate may use closed forms only where they agree with the
  diagonalisation to well below the linewidth.
- Omitted: ¹⁴N hyperfine (A∥ ≈ −2.1 MHz, a three-line splitting; Felton et al., PRB 79, 075203 (2009)) and ¹³C
  hyperfine. With ensemble T2* ≲ 1 µs the 2 MHz triplet is mostly hidden in the broadening; add it as a later
  refinement if fits demand it.

## 2. The four orientations

NV axes are the four <111> body diagonals (`NV_AXES`):
**n**₁ = (1,1,1)/√3, **n**₂ = (1,−1,−1)/√3, **n**₃ = (−1,1,−1)/√3, **n**₄ = (−1,−1,1)/√3.
Any two meet at the tetrahedral angle (cos = −1/3). `EnvParams.orientation_populations` gives the fraction of
NVs on each axis (default 1/4 each). For each axis: B∥ = B·**n**ₖ, B⊥ = |B − (B·**n**ₖ)**n**ₖ|
(`constants.b_projections_hz` returns γe·(B∥, B⊥) in Hz). A generic field splits the ODMR spectrum into up to
8 lines (4 axes × ms = ±1); B along [100] makes all four axes equivalent (2 lines); B along one [111] gives
1 aligned + 3 degenerate axes.

## 3. Microwave drive and the rotating frame

**Lab frame.** A MW segment with amplitude a ∈ [0, 1], frequency f, phase φ produces a linearly polarised field
along the unit vector **m** = `mw_field_direction`, oscillating as cos(2π f t + φ) with t the absolute time since
the start of the sequence (phase-coherent synthesiser; see `pulse.py`). For axis k let
**m**⊥ₖ = **m** − (**m**·**n**ₖ)**n**ₖ, u_k = **m**⊥ₖ/|**m**⊥ₖ|. Define

    f_R,k = rabi_full_scale_hz · a · |m⊥ₖ|                                   (Hz)
    H_mw/h = √2 · f_R,k · cos(2π f t + φ) · (u_k · S)

The √2 makes f_R,k the **population Rabi frequency** of an isolated, resonant 0 ↔ ±1 transition:
P(ms=0) = cos²(π f_R t). (⟨±1|S⊥|0⟩ = 1/√2, and the RWA keeps half of the cosine.) With the default
**m** = [100], |**m**⊥ₖ| = √(2/3) for all four axes. The component of the MW field along **n**ₖ (Sz coupling)
is neglected (it only phase-modulates at the MW frequency).

**Rotating frame / RWA.** In the eigenbasis {|0̃⟩, |−̃⟩, |+̃⟩} of H0 (energies 0, f−, f+ after subtracting the
|0̃⟩ energy; |0̃⟩ is the eigenvector with the largest ms = 0 weight, |−̃⟩ and |+̃⟩ are the other two in ascending
energy), go to the frame rotating at f on both upper states, U = exp(i 2π f t (|−̃⟩⟨−̃| + |+̃⟩⟨+̃|)), and drop
terms oscillating at ≈ 2f. Result (Hz):

    H_rot/h = Δ− |−̃⟩⟨−̃| + Δ+ |+̃⟩⟨+̃| + Σ_± (√2/2) f_R,k [e^{−iφ} ⟨±̃|u_k·S|0̃⟩ |±̃⟩⟨0̃| + h.c.],   Δ± = f± − f

(the ±̃ ↔ ∓̃ matrix elements oscillate at ≈ f and are dropped as well). The RWA holds when

- f_R ≪ f (here f_R ≲ 20 MHz vs f ≈ 3 GHz; the Bloch–Siegert shift f_R²/(4f) ≈ 10 kHz is below any T2*
  linewidth);
- both transition frequencies f± ≫ f_R, i.e. away from the ground-state level anticrossing (B∥ ≈ D/γe ≈ 102 mT,
  where f− → 0);
- the drive is not near the ±1 ↔ ∓1 splitting (2γe B∥ ≈ f, also only near ~100 mT).

The 2-level picture (only one of the two transitions) additionally needs |Δ_other| ≫ f_R and ≫ the linewidth. At
low field (γe B∥ ≲ f_R) both transitions are driven at once; the reference handles this automatically, the
surrogate must not assume the 2-level picture there.

**Phase and frequency changes.** In a frame rotating at a fixed f, a segment's phase enters as the constant φ.
If a sequence uses several frequencies, the lab definition (absolute time t) is the truth: either integrate in a
frame at one reference frequency f_ref (each segment then carries a detuning-dependent phase ramp
2π (f − f_ref) t + φ), or switch frames at the segment boundary t_s by applying U₂(t_s) U₁(t_s)† to the state (a phase
2π (f₂ − f₁) t_s on the upper states). A test with two segments at different frequencies pins the sign. MW is not applied to the excited state (its zero-field splitting 1.42 GHz is far from the ~2.87 GHz drive;
valid unless f approaches D_ES ± γe B).

**Frame bookkeeping (both implementations).** In MW-on bins the frame frequency is that bin's MW frequency. In
MW-off bins it is held at the last MW frequency; before the first MW bin it already sits at the first MW frequency,
and with no MW at all it is D. A frequency change at bin start t_s multiplies ρ_{0̃,±̃} by exp(−i 2π (f₂ − f₁) t_s).
The frame choice is exact (it changes no observable); only the sign of the phase matters and is pinned by tests.

**Dissipators in the rotating frame (secular rule).** The jump operators of §5 and §6 are defined in the ms basis,
the frame rotates the eigenstates |±̃⟩. When the eigenbasis differs from the ms basis (B⊥ ≠ 0 or E ≠ 0) a jump
operator that connects |0̃⟩ with |±̃⟩ acquires parts oscillating at ±f ≈ 3 GHz in the rotating frame. Both
implementations drop the oscillating cross terms: with L = Σ_ω L_ω split by frame frequency ω ∈ {−f, 0, +f}, the
dissipator is Σ_ω D[L_ω] (equivalently: superoperator elements that couple coherences of different frame order are
zeroed). This is the dissipative half of the RWA, exact for B ∥ **n** and E = 0, with an error of order
(γe B⊥ / D)² otherwise (the reference checks it against a lab-frame `mesolve` without RWA at B⊥ ≈ 2.5 mT: 2e-3 in
populations).

## 4. Inhomogeneous broadening (T2*)

Each NV sees a static random shift δ of its local field along its axis (dominant ensemble mechanism: magnetic
noise from P1 centres and ¹³C, Barry et al. 2020): a term **δ·Sz (Hz) added to H0**. When γe B∥ ≫ E, γe B⊥ this
is f± → f± ± δ; at zero field it is well defined (degenerate ms = ±1), and at E ≠ 0 with γe B∥ ≲ E it gives the
strain-protected (quadratic) response a magnetic shift really has. The surrogate diagonalises H0 + δ Sz per sample;
the reference writes δ Sz in the eigenbasis of H0 and drops its |0̃⟩ ↔ |±̃⟩ part (it rotates at the frame
frequency, RWA). The two differ by a level shift of order δ² (γe B⊥/D)²/D, far below any linewidth. δ is drawn
from

- **Lorentzian** (default, `broadening_shape="lorentzian"`): half width at half maximum Γ = 1/(2π T2*) (Hz), FWHM
  1/(π T2*). Free-induction (Ramsey) envelope exp(−t/T2*).
- **Gaussian**: standard deviation σ = 1/(√2 π T2*) (Hz). Envelope exp(−(t/T2*)²).

Implementation: both integrate over δ with the same deterministic quadrature: M equal-probability quantile midpoints
u_m = (m + ½)/M, δ_m = Γ tan(π(u_m − ½)) (Lorentzian) or σ Φ⁻¹(u_m) (Gaussian), equal weights 1/M; M = 1 means
δ = 0. Report M. The Lorentzian's heavy tails set the quadrature error: for pulsed Rabi at f_R/Γ ≈ 50 the transfer
curve on M nodes deviates from the continuous-distribution integral by up to 1.2 % of the contrast at M = 64 and
0.2 % at M = 256 (reference measurement). M ≥ 32 is fine for comparisons at equal M (identical nodes); use
M ≥ 128 when sub-percent agreement with the continuous Lorentzian matters. The node set revives after about
2 M T2\*. Strain-type shifts (moving f+ and f− together) are not modelled separately.

## 5. Relaxation: T1, T2

Ground-state Lindblad terms, per NV, in the ms basis of its axis:

- **T1:** jumps |ms⟩ → |ms'⟩ for every ordered pair of ground sublevels, each at rate Γ1 = 1/(3 T1). Population
  imbalance (e.g. the ms=0 excess) then relaxes as exp(−t/T1). Default T1 = 5 ms (Jarmola et al., PRL 108, 197601
  (2012)): irrelevant on µs sequences but keeps steady states honest.
- **T2 (homogeneous dephasing):** L = √γφ Sz with γφ = 2/T2, so 0 ↔ ±1 coherences decay as exp(−t/T2). A Hahn
  echo refocuses δ (§4), leaving the echo amplitude ∝ exp(−(2τ/T2)^p) for total free time 2τ, with p = 1.
- `t2_stretch` p ≠ 1 (stretched exponential, typical for spin-bath echo decay) is phenomenological and not a
  Lindblad process: the reference supports only p = 1 (it must raise for p ≠ 1); the surrogate may apply it.
- Consequence: the Ramsey envelope is the §4 envelope times exp(−t/T2). For T2 ≫ T2* (the usual case,
  `EnvParams` enforces T2 ≥ T2*) the difference is small; keep it, both implementations do the same.

## 6. Optical cycle: seven-level rate model

Levels (Tetienne et al., New J. Phys. 14, 103033 (2012)): |1⟩ g ms=0, |2⟩ g ms=−1, |3⟩ g ms=+1, |4⟩ e ms=0,
|5⟩ e ms=−1, |6⟩ e ms=+1, |7⟩ metastable singlet (¹E/¹A₁ merged). Rates (1/s, mean of the four NVs in their
Table 1; `constants.py`):

| transition | rate | value |
| --- | --- | --- |
| \|4⟩→\|1⟩, \|5⟩→\|2⟩, \|6⟩→\|3⟩ radiative, spin-conserving | k_r | 65.9 × 10⁶ s⁻¹ |
| \|4⟩→\|7⟩ ISC from e ms=0 | k47 | 7.9 × 10⁶ s⁻¹ |
| \|5⟩→\|7⟩, \|6⟩→\|7⟩ ISC from e ms=±1 (each) | k57 = k67 | 53.3 × 10⁶ s⁻¹ |
| \|7⟩→\|1⟩ singlet to g ms=0 | k71 | 0.98 × 10⁶ s⁻¹ |
| \|7⟩→\|2⟩, \|7⟩→\|3⟩ singlet to g ms=±1 (each) | k72 = k73 | 0.73 × 10⁶ s⁻¹ |
| \|1⟩→\|4⟩, \|2⟩→\|5⟩, \|3⟩→\|6⟩ optical pumping | β(t) k_r | β(t) = optical_beta_full_scale · laser(t) · (1 + x(t)) |

- Spin-flip optical transitions are neglected (≲ 2 % of spin-conserving ones, Robledo et al., New J. Phys. 13,
  025013 (2011), as used by Tetienne). β is Tetienne's optical pumping parameter; β ≈ 1 is near saturation.
- The spin-dependent ISC (k57 ≫ k47) plus the singlet's preferential return to ms=0 is what pumps the spins into
  ms=0 under illumination and makes ms=±1 darker for the first ~300 ns of a readout pulse. The singlet lifetime is
  1/(k71 + 2 k72) ≈ 0.41 µs.
- **As a master equation:** a 7×7 density matrix; the coherent part (H0 + H_mw, §1/§3) acts on the ground
  3×3 block; every rate above is a jump operator √k |j⟩⟨i|, in the ms basis of the NV's own axis; plus §5. The
  pumping jumps also destroy ground-state spin coherence at the pump rate, as they should. Excited and singlet
  levels carry populations only. Excited-state spin mixing by B⊥ (Tetienne's main subject: PL drops > 30 % for
  B⊥ > 20 mT) is not modelled: the model is valid for B⊥ ≲ a few mT per axis and away from the excited-state
  anticrossing (B∥ ≈ D_ES/γe ≈ 51 mT, Fuchs et al., PRL 101, 117601 (2008)). Ground-state mixing by B⊥ *is*
  captured, because the rates act in the ms basis while H0 is diagonalised with B⊥.
- **Photon emission rate** of one NV: k_r (ρ44 + ρ55 + ρ66).

### Three-level simplification (surrogate option)

Adiabatically eliminate the excited states (lifetime 1/(k_r + k_ISC) ≈ 13 ns for ms=0, 8 ns for ms=±1). From
ground level m, under pump rate β k_r: the NV emits at R_m = β k_r · k_r/(k_r + k_ISC,m) and transfers to the
singlet at β k_r · k_ISC,m/(k_r + k_ISC,m); the singlet returns as in the table. Keeping g ms=0, the driven
ms=±1 level (lumping or dropping the other) and the singlet gives a three-level model. Acceptable when

- β ≲ 0.3 (excited-state population small, no saturation subtleties),
- the quantity of interest varies slowly compared with ~20 ns (true for the 10 ns grid only after binning a few
  bins; check the first bins of each laser pulse),
- only one MW transition is near resonance (|Δ_other| ≫ f_R and linewidth), B⊥ small.

The surrogate may use it only where it passes the validation suite (§9) against the seven-level reference.

## 7. Detected signal, background, shot noise, drift

Per NV of orientation k (after the §4 average), emission rate e_k(t). Expected detected rate (counts/s):

    r(t) = η · N · Σ_k p_k e_k(t)  +  background_cps · laser(t) · (1 + x(t))  +  dark_cps

with η = `collection_efficiency`, N = `n_nv`, p_k = `orientation_populations`. Counts in bin j:
`counts[j] ~ Poisson(∫_bin r dt)`, independently per bin and shot (`rate_cps` stores the bin-average r). Spin
projection noise is neglected: with η ≪ 1 the detected photon count is far below one per NV per shot, so photon
shot noise dominates.

**Laser drift.** x(t) is an Ornstein–Uhlenbeck process with stationary std `laser_drift_rel_sigma` and
correlation time `laser_drift_tau_s`, sampled exactly on the grid:
x_{j+1} = x_j e^{−Δ/τ} + σ √(1 − e^{−2Δ/τ}) ξ_j. It runs continuously through the shots of one `run()` call (shots
are back to back), starting from x ~ N(0, σ²). It scales the pump rate and the background together
(effective power factor max(0, 1 + x)). σ = 0 gives a deterministic `rate_cps`.

## 8. Readout convention

Spin-state contrast is read from the first ~300 ns of a readout laser pulse, normalised by a late window of the
same pulse (after ~2 µs the spins are re-polarised and the rate is spin-independent). Analysis code defines its
windows explicitly; the simulator only returns per-bin counts.

**Dark time after the initialisation pulse.** The singlet empties with lifetime 1/(k71 + 2 k72) ≈ 0.41 µs and feeds
ms = 0 preferentially. After 1 µs of dark about 9 % of the end-of-laser singlet population is still there (≈ 3 % of
all NVs at β = 0.1); in sweeps whose free time varies it keeps relaxing into the ground state and adds a baseline
drift (≈ 2–3 % of the Rabi contrast, ≈ 10 % of a Ramsey fringe). The protocol default is therefore a 3 µs settle
(`SETTLE_DEFAULT_NS`, ≈ 7 lifetimes, residue < 0.1 %).

## 9. Validation suite (what both implementations must reproduce)

With the defaults of `EnvParams`:

1. **CW ODMR:** dips at f± of each orientation (8 lines max, 2 for B ∥ [100]), contrast increasing with MW power
   until power-broadened, width → ~1/(π T2*) at low power and low β.
2. **Pulsed Rabi:** oscillation of the readout contrast at f_R,k vs MW duration, damped by the §4 detuning spread.
3. **Ramsey:** fringes at the detuning, envelope per §4 × exp(−t/T2).
4. **Hahn echo:** decay exp(−(2τ/T2)^p), insensitive to the static detuning.
5. **Detuned control:** MW detuned far off resonance (|Δ| ≫ f_R and linewidth, e.g. 200 MHz) gives a trace
   identical to MW off within shot noise.

Surrogate-vs-reference agreement is judged on noiseless `rate_cps` (relative error per bin) and on the protocol
features above; tolerances and their justification are in `tests/sim/test_surrogate_vs_reference.py`.

## 10. Assumptions summary

Temperature-free (room temperature constants); no hyperfine; no NV⁰/charge dynamics (ionisation appears only as
background); no excited-state spin mixing; MW couples only to the ground state, linear polarisation, ⊥ component
only; static broadening is magnetic-like (opposite shifts of f±); shots start from a fully mixed spin state; shots
are back to back for drift purposes.
