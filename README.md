# Glowbox

An open-source nilware.io lab project: neural networks get the controls and sensors of a room-temperature quantum
system (NV centres in diamond), explore it, and we test whether they learn to use the quantum spin dynamics.
Simulation first; hardware later.

The goal, scope, phases, tasks and working agreement are in [project-definition.md](project-definition.md).

Licence: MIT (LICENSE).

Layout: `glowbox/` (sim, control, learn, dream, experiments, analysis, hw), `tests/`, `docs/`. Setup and the
GPU-machine workflow: [docs/dev.md](docs/dev.md). Physics spec: [docs/physics.md](docs/physics.md).

Status: scaffolded (pulse representation, simulator interface, physics spec, run logging). First milestone: simulate continuous-wave ODMR and Rabi oscillations for an NV ensemble, plot
both, and pass tests comparing the fast surrogate with the QuTiP reference.
