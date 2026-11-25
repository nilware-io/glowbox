"""Fast surrogate NV simulator (PyTorch, batched, differentiable). See docs/notes/surrogate.md."""

from glowbox.sim.surrogate.model import (
    ModelOptions,
    build_statics,
    detuning_quadrature,
    emission_per_nv,
    expected_rate_cps,
    mw_fill_for_duration,
)
from glowbox.sim.surrogate.simulator import SurrogateSimulator, laser_drift

__all__ = [
    "ModelOptions",
    "SurrogateSimulator",
    "build_statics",
    "detuning_quadrature",
    "emission_per_nv",
    "expected_rate_cps",
    "laser_drift",
    "mw_fill_for_duration",
]
