# Glowbox

An open-source nilware.io lab project: neural networks get the controls and sensors of a room-temperature quantum
system (NV centres in diamond), explore it, and we test whether they learn to use the quantum spin dynamics.
Simulation first; hardware later.

The goal, scope, phases, tasks and working agreement are in [project-definition.md](project-definition.md).

Licence: MIT (LICENSE).

Layout: `glowbox/` (sim, control, learn, dream, experiments, analysis, hw), `tests/`, `docs/`. Setup and the
GPU-machine workflow: [docs/dev.md](docs/dev.md). Physics spec: [docs/physics.md](docs/physics.md).

## Status

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

Run the milestone (CPU works; the surrogate uses CUDA when present; the full config takes a few minutes, mostly the
reference):

    uv run python -m glowbox.experiments.m1_odmr_rabi --quick                      # ~30 s smoke run
    uv run python -m glowbox.experiments.m1_odmr_rabi --report-dir docs/results    # full run, committed report

Tests: `uv run pytest` (about 2-3 minutes; CUDA tests are skipped without a GPU).

Next: phase 2 learners (decoder, reservoir baseline, encoder through the differentiable surrogate).
