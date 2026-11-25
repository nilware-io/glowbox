# Glowbox: project definition

Hand-over brief for Claude Code. Glowbox is an open-source nilware.io lab project: neural networks get the controls and sensors of a room-temperature quantum system (NV centres in diamond), explore it, and we test whether they learn to use the quantum spin dynamics. Build simulation-first; hardware comes later.

## Goal

- Build a closed loop: encoder/agent → pulse controls → (simulated, later real) NV diamond → fluorescence → decoder.
- Let learners discover useful control strategies with minimal hand-design.
- Test whether performance depends on spin resonance, using a detuned control.
- Run DeepDream-style experiments: invert the trained networks to see what they "imagine".

## Non-goals (for now)

- No claim of quantum advantage.
- No hardware drivers until the simulator and learners work end to end.
- No cloud dependence; everything runs locally.

## Hardware context

| Tier | Hardware | Role |
| --- | --- | --- |
| Reflex | Terasic DE10-Nano (Cyclone V SoC FPGA + dual ARM A9) | Later: pulse sequencer (ns timing), photon counting, tiny on-chip readout network |
| Training | NVIDIA RTX 5070 Ti | Train decoder, encoder, RL agent |
| Big memory | AMD Ryzen AI Max, 128 GB unified memory | Large simulations, data store, experiment-planning agents |

## Stack

- Python 3.12, PyTorch (CUDA on the 5070 Ti; keep code device-agnostic so it also runs on the Ryzen box).
- NumPy/SciPy; QuTiP for the reference spin simulation.
- Hydra or plain YAML for configs; Weights & Biases optional, local logging required (Parquet or HDF5).
- pytest; ruff; type hints throughout.
- Later: Verilog/SystemVerilog for the FPGA (Quartus), Linux on the ARM side, a simple TCP or USB link to the PC.

## Repository layout

```
glowbox/
  sim/          NV simulator (physics core, noise, fast surrogate)
  control/      pulse-sequence representation and compiler
  learn/        decoder, encoder, RL agent, reservoir baseline
  dream/        DeepDream experiments (gradient and black-box)
  experiments/  configs and run scripts
  analysis/     metrics, plots, notebooks
  hw/           placeholder: FPGA and drivers (phase 4)
  tests/
  docs/
```

## Phase 1: the simulator

- **Physics core:** NV ground-state spin-1 Hamiltonian with zero-field splitting D ≈ 2.87 GHz, Zeeman term from a controllable magnetic field, microwave drive (frequency, phase, amplitude, timing), optical pumping and spin-dependent fluorescence (three/seven-level rate model). Ensemble of NVs with inhomogeneous broadening and four crystal orientations.
- **Noise:** photon shot noise, laser intensity drift, T1, T2\* and T2 decay, background light.
- **Must reproduce:** continuous-wave ODMR dips; Rabi oscillations; Ramsey fringes; Hahn echo; detuned microwaves giving no resonance.
- **Fast surrogate:** a vectorised or learned surrogate that runs thousands of shots per second on the GPU, validated against the QuTiP reference.
- **Interface:** `run(pulse_sequence, env_params) -> fluorescence_trace`; identical interface later for hardware.

## Phase 2: learners

- **Pulse representation:** a timeline of laser on/off, microwave segments (freq, phase, amplitude, duration) and magnet setting, on a fixed time grid (start with 10 ns resolution).
- **Decoder:** 1D CNN on fluorescence traces and ODMR spectra (after Yao et al., 2026), plus a time-bin-weighting baseline; outputs task targets and an uncertainty.
- **Reservoir baseline:** fixed random input encoding into pulses, train only a linear readout.
- **Encoder:** learns to map input data to pulse sequences; trained end to end through the differentiable surrogate.
- **RL agent:** PPO or SAC with free exploration over pulse parameters; reward from task performance.
- **Adaptive measurement:** Bayesian experiment design option, choosing the next measurement from the current posterior.

## Tasks

1. Estimate a hidden magnetic field from the glow.
2. Store a short bit pattern in the pulse response and recall it.
3. Predict the next fluorescence reading.
4. Compress-and-reconstruct: encoder → diamond → decoder autoencoder.

## The quantum test (required for every task)

- Run each task twice: on resonance, and with microwaves detuned far off resonance (spins stop responding).
- Report the performance gap with confidence intervals over seeds.
- Log whether learned strategies use microsecond-scale timing (Rabi or echo structure).

## Phase 3: DeepDream experiments

1. **Dream the diamond:** freeze the decoder, gradient-ascend on control inputs to maximise a chosen neuron or output; then run the imagined sequence through the simulator (later hardware) and measure the mismatch.
2. **Self-dreaming:** CMA-ES or evolution strategies over pulse sequences to amplify a chosen glow feature, using the simulator or hardware as a black box; repeat with detuning.
3. **Dream of nothing:** optimise for inputs driving outputs or activations to zero; analyse the structure found.

## Phase 4: hardware (later)

- FPGA pulse sequencer on the DE10-Nano driving laser TTL and the microwave switch; photon counter or ADC readout; same `run()` interface as the simulator.
- Tiny quantised readout network on the FPGA; big models stay on the PC.
- Laser safety interlock in software and documented in `docs/safety.md`.

## Metrics and outputs

- Task accuracy, resonance-vs-detuned gap, sample efficiency, wall-clock per step.
- Every run saves config, seed, git hash, raw traces and plots.
- A short Markdown report per experiment in `docs/results/`.

## Working agreement

- Simulation first; never block on hardware.
- Small, tested steps; each phase ends with a runnable demo script.
- Document assumptions in code comments and `docs/`.
- Open source under nilware.io once incorporated (licence to be decided, e.g. BSD-3 or MIT).

## First milestone

A script that simulates continuous-wave ODMR and Rabi oscillations for an NV ensemble, plots both, and passes tests comparing the fast surrogate with the QuTiP reference.
