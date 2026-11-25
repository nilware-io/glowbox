"""Fast surrogate NV simulator (PyTorch, batched, differentiable). See docs/notes/surrogate.md."""

from glowbox.sim.surrogate.model import (
    ModelOptions,
    build_statics,
    detuning_quadrature,
    emission_and_state,
    emission_per_nv,
    expected_rate_cps,
    mw_fill_for_duration,
)
from glowbox.sim.surrogate.simulator import SurrogateSimulator, laser_drift
from glowbox.sim.surrogate.sweeps import branched_rate_cps, cw_steady_rate_cps

__all__ = [
    "ModelOptions",
    "SurrogateSimulator",
    "branched_rate_cps",
    "build_statics",
    "cw_steady_rate_cps",
    "detuning_quadrature",
    "emission_and_state",
    "emission_per_nv",
    "expected_rate_cps",
    "laser_drift",
    "mw_fill_for_duration",
]
