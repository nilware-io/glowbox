# Glowbox

An open-source nilware.io lab project: neural networks get the controls and sensors of a room-temperature quantum
system (NV centres in diamond), explore it, and we test whether they learn to use the quantum spin dynamics.
Simulation first; hardware later.

The goal, scope, phases, tasks and working agreement are in [project-definition.md](project-definition.md).

![CW ODMR of an NV ensemble: eight dips, two per crystal axis, simulated by the QuTiP reference and the GPU surrogate](docs/results/figures/m1-odmr-rabi-odmr.png)

*Continuous-wave ODMR of an NV ensemble in a generic magnetic field: four crystal axes, two lines each. The exact
QuTiP reference and the fast GPU surrogate agree to within 50 ppm (bottom panel).*

![Pulsed Rabi oscillations on one NV line, reference and surrogate](docs/results/figures/m1-odmr-rabi-rabi.png)

*Pulsed Rabi oscillations on the 2.7995 GHz line, with T2\* decay: both simulators fit 4.094 MHz and differ by under
10 ppm.*

Licence: MIT (LICENSE).

Layout: `glowbox/` (sim, control, learn, dream, experiments, analysis, hw), `tests/`, `docs/`. Setup and the
GPU-machine workflow: [docs/dev.md](docs/dev.md). Physics spec: [docs/physics.md](docs/physics.md).

## Status

**Milestone 2 done** (Task 1, estimate a hidden magnetic field from the glow; decoders, baselines and the quantum
test). Report: [docs/results/m2-field-estimation.md](docs/results/m2-field-estimation.md).

![Field error vs shots for the CNN decoder, the linear time-bin baseline and the physics fit, resonant and detuned](docs/results/figures/m2-field-estimation-error-vs-shots.png)

*RMS error of the estimated field (|B| 0.5-3 mT, any direction, folded into the identifiable wedge) vs shots per
sweep point, CW and pulsed ODMR, mean over 5 seeds with 95 % CIs. Detuned (dashed): every estimator falls to the
prior.*

- At 1000 shots per point the CNN decoder reaches 11.9 [11.0, 12.9] µT (CW) and 26.4 [21.1, 31.8] µT (pulsed;
  95 % Student-t CIs over 5 seeds); the linear time-bin baseline ~400 µT; prior only 830 µT. The physics fit
  (Lorentzian dips at the H0 transitions of a trial field, seed 0, same 300 fields) is as good or better than the
  CNN at 100-300 shots (CW) and 300-1000 shots (pulsed, typical case; 18.6 vs 35.9 µT RMSE at 1000); the CNN wins
  where the fit finds few dips (<= 30 shots CW, <= 100 pulsed) and on RMSE at >= 1000 (CW) / 3000 (pulsed) shots.
  The CNN's uncertainty is well calibrated below 1000 shots and 5-16 % over-confident at >= 1000.
- Quantum test: with the microwaves 400 MHz off resonance every learned estimator drops to the prior (skill
  ~0); gap 819 [816, 822] µT (CW CNN), 804 [798, 810] µT (pulsed CNN). Timing: no timing is learned in M2 (fixed
  protocols). The hand-designed pulsed protocol relies on its pi pulse (`protocol_uses_rabi_timing`: frozen-decoder
  error 8-14x higher at 110 / 510 ns, which mixes contrast loss and distribution shift); what the decoder learned
  is a readout time-bin weighting that follows the spin-contrast transient (saliency vs contrast r = 0.98). The
  decoders agree on QuTiP-reference inputs to < 0.02 µT.
- Code: `glowbox/learn/field_data.py` (priors, protocols, surrogate datasets), `glowbox/learn/decoder.py`
  (1D CNN + Gaussian NLL, linear time-bin baseline), `glowbox/analysis/field_fit.py` (physics fit),
  `glowbox/analysis/quantum_test.py` (the reusable quantum-test harness), surrogate fast paths for sweeps
  (`glowbox/sim/surrogate/sweeps.py`).

Run (the full config trains 20 decoders: ~100 min on an RTX 5070 Ti; `--quick` ~8 min on a CPU):

    uv run python -m glowbox.experiments.m2_field_estimation --quick
    uv run python -m glowbox.experiments.m2_field_estimation --report-dir docs/results

**Milestone 1 done** (simulate CW ODMR and Rabi oscillations for an NV ensemble, plot both, test the fast surrogate
against the QuTiP reference). Report with plots and numbers: [docs/results/m1-odmr-rabi.md](docs/results/m1-odmr-rabi.md).

What works:

- `glowbox/control/pulse.py`: pulse sequences on a 10 ns grid (laser, MW segments, magnet), standard protocols.
- `glowbox/sim/reference/`: QuTiP reference, seven-level Lindblad model per NV axis, four orientations, T2*
  quadrature, T1/T2, shot noise, laser drift; exact per-bin propagation. CPU, about 0.2-1 s per sequence.
- `glowbox/sim/surrogate/`: the same physics in batched, differentiable torch (CPU or CUDA). Agrees with the
  reference to ~1e-9 per bin in float64 and ~1e-4 in float32; hundreds of thousands of shots per second on the GPU.
- Both implement `run(sequence, env) -> FluorescenceTrace` (`glowbox/sim/interface.py`) and reproduce the
  validation suite of [docs/physics.md](docs/physics.md) section 9 (ODMR, Rabi, Ramsey, Hahn echo, detuned control).
- `glowbox/analysis/runlog.py`: per-run config, seed, git hash, Parquet traces and metrics, Markdown reports.

Run milestone 1:

    uv run python -m glowbox.experiments.m1_odmr_rabi --quick                      # ~30 s smoke run
    uv run python -m glowbox.experiments.m1_odmr_rabi --report-dir docs/results    # full run, committed report

Tests: `uv run pytest` (about 2-3 minutes; CUDA tests are skipped without a GPU, slow ones need `-m slow`).

Next: the remaining phase-2 learners and tasks (reservoir baseline, encoder through the differentiable surrogate,
RL agent; tasks 2-4), each with the quantum test.
