"""QuTiP reference simulator (ground truth). See docs/physics.md and docs/notes/reference.md."""

from glowbox.sim.reference.model import transition_frequencies_hz
from glowbox.sim.reference.simulator import (
    ReferenceSimulator,
    bin_controls,
    detuning_quadrature_hz,
    detuning_samples_hz,
)

__all__ = [
    "ReferenceSimulator",
    "bin_controls",
    "detuning_quadrature_hz",
    "detuning_samples_hz",
    "transition_frequencies_hz",
]
