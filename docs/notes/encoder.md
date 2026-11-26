# Learned pulse encoder (`glowbox/learn/encoder.py`, milestone 4)

The encoder maps input data to a pulse sequence on the 10 ns grid and is trained end to end through the
differentiable surrogate together with a decoder. This note lists its parametrisation and the assumptions behind
it; results are in docs/results/m4-autoencoder.md (Task 4) and docs/results/m4-learned-sensor.md (Task 1 with a
learned sensor protocol).

## Sequence layout (`Layout`)

Every sequence starts with the fixed MW-free initialisation (laser, dark settle), simulated once per static field.
The encoder writes the suffix: `n_blocks` x [MW block of `n_segments` slots] [readout laser] [optional dark gap].
The decoder sees either one gated count per readout (the first `gate_ns` of the laser, a gated photon counter; the
rest of the laser re-polarises the spins) or the readout split into `window_ns` windows (time-resolved).

## Parametrisation (smooth and bounded)

Per segment, four pre-activations z set: amplitude `amp_lo + (amp_hi - amp_lo) sigmoid(z0)`; phase `pi tanh(z1)`
(relative to the suffix start; `glowbox.learn.reservoir.full_sequence` converts to absolute time); frequency
`base_hz + span_hz tanh(z2)`; duration `min_ns + (max_ns - min_ns) sigmoid(z3)`, MW from the slot start.

- Frequencies are tied to a base frequency per segment (a device line for Task 4, a sweep-grid point for Task 1):
  the drive only acts within a few linewidths / Rabi frequencies of a line, so a frequency far from every line has
  no gradient and could not be learned. The ranges are assumptions of each experiment (configs).
- Durations are rounded to the grid in the forward pass (straight-through): every learned sequence is a valid
  `PulseSequence` (hardware and the QuTiP reference run it). The gradient is the exact derivative of the continuous
  duration's propagator (run-length path, docs/notes/surrogate.md) evaluated at the rounded duration.
- With inputs, an MLP produces the pre-activations; the bias of its last layer is the input-independent protocol
  (initialised at a hand-designed one) and its weights start scaled by `out_scale`. Without inputs
  (`n_inputs = 0`) the pre-activations are free parameters: a learned protocol (Task 1).
- `randomize_` turns the same parametrisation into a fixed random encoder (the reservoir baseline of milestone 3,
  N(0, gain^2 / fan_in) last layer, N(0, 1) biases).

## Training noise

Training adds shot noise as a Gaussian with the Poisson variance, `mu + sqrt(mu / shots) eps`, differentiable in mu.
At the shot levels used (>= 10^3 for gated counts, >= 10 shots x tens of photons per window for Task 1) the counts
per feature are large enough for the Gaussian to be a good approximation. Evaluation always draws Poisson counts.

## Two-phase training

Phase 1 trains encoder and decoder together through the surrogate (few quadrature nodes, fresh data every step).
Phase 2 freezes the encoder, simulates a bank of its noiseless readouts (no gradient, more accurate numerics) and
trains a fresh decoder on it with Poisson noise: the same recipe every baseline gets (fixed random encoder, fixed
protocols, the detuned arm), so the comparisons isolate the encoder.

## Quantum test

The detuned arm shifts every MW frequency by 400 MHz during training and evaluation (`check_no_line_driven` is
asserted on the learned frequencies). Its encoder still trains, but its MW drives nothing.
