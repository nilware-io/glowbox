# Adaptive measurement and the RL agent (milestone 5)

Code: `glowbox/learn/measurement.py` (measurements as actions, the emulator), `glowbox/learn/adaptive.py`
(posterior, expected information gain, fixed and greedy designs), `glowbox/learn/rl.py` (PPO, environments),
`glowbox/learn/bit_write.py` (Task 2 learned write). Results: docs/results/m5-adaptive-field.md (Task 1),
docs/results/m5-rl-bit-write.md (Task 2). This note lists the design choices and the assumptions behind them.

## Task 1: one measurement per step

- **Action grid.** A measurement is one pulsed-ODMR shot (init laser, dark settle, one MW pulse, readout in two
  300 ns windows) repeated for a block of shots. Actions are a MW frequency (2 MHz grid over 2.77-2.97 GHz) times a
  pulse duration (50, 100, 200, 400 ns at amplitude 0.6: about pi/2, pi, 2 pi, 4 pi at the nominal Rabi scale).
  Discrete because the posterior's information gain is evaluated for every action at every step, and a frequency
  far from every line carries no signal (no gradient to follow in a continuous parametrisation either). The grid is
  fine compared with the pulse-broadened lines (~10 MHz for 100 ns) and the RL agent explores it freely.
- **Calibrated device.** Every learner knows the device parameters (`EnvParams` defaults: the milestone-2 prior
  midpoints, E = 0); only the field is hidden. Milestones 2-4 marginalised over nuisance parameters; here the
  question is how to choose measurements, and a nuisance-aware posterior (more particle dimensions) is left for
  later. The fixed sweeps use the same posterior, so the comparison is fair within milestone 5.
- **Truth vs model.** Observations always come from the full surrogate (precomputed noiseless means per true field
  for every action, then Poisson noise), or from any `Simulator.run()` (QuTiP reference, later hardware) through
  `rl.SimulatorBackend`. The learners' model is the emulator `AxisTable`: one orientation's response tabulated on
  (B_par, |B_perp|, azimuth of B_perp relative to the MW) and summed over the four axes. The symmetries this needs
  (rotation about the axis; a pi rotation about a transverse axis flips B_par and the azimuth; a linear drive is
  unchanged by u -> -u) are properties of the model of docs/physics.md and are tested exactly
  (tests/learn/test_measurement.py). Without the azimuth the emulator was off by up to 1.6x the signal spread on
  fields with one axis nearly perpendicular to B (mixed levels), enough for the information-greedy design to chase
  model errors; with it the error is ~0.2 % RMS of the signal spread (numbers per run in the report).
- **Fast truths.** `measurement.emission_table` computes every action for a field with three exact shortcuts:
  the MW-free initialisation once per field; one MW propagator per (field, frequency, detuning node), longer
  durations by squaring when they double; the readout as a frame-independent linear functional of the post-MW
  state (secular rule: MW-off generators do not couple frame-rotating coherences to populations). Checked against
  the bin-stepped `SurrogateSimulator.run` to 1e-8.

## Posterior and design

- **Particles** over the folded field (the wedge Bx >= By >= Bz >= 0, milestone 2's target); prior |B| uniform in
  [0.5, 3] mT, isotropic. Counts per action are sufficient statistics, so the exact log-likelihood of any field
  given all data is cheap, which makes Metropolis-Hastings moves exact.
- **Adaptive tempering.** A block that would drop the effective sample size below half is folded in as L^phi in
  stages with a resample-move after each. Without it the first informative block (a dip found) left a handful of
  prior particles and the filter could lose the true mode for good (seen as 1 mT outliers in early tests).
- **Expected information gain** per action from 128 posterior draws, Gaussian approximation per window
  (1/2 log(1 + S Var / E)); a Monte Carlo estimate of the exact Poisson mutual information is available and ranks
  actions alike (tests). Greedy (one step ahead).
- **Fixed sweep**: every frequency once per pass at one duration, in a golden-ratio interleaved order so that a
  partial pass still covers the band; "whole band at fewer shots" runs the full sweep at the budget's shots per
  point.

## RL agent

- **PPO, written in torch** (`rl.train_ppo`): on-policy with many cheap parallel simulated episodes is PPO's home
  ground; SAC needs continuous actions (Task 1's are discrete) and its replay buffer buys sample efficiency a
  simulator does not need; library implementations step one environment at a time through NumPy and would
  serialise the batched GPU surrogate.
- **Environments** have gymnasium's vector-env shape (`reset(seed) -> (obs, info)`, `step(a) -> (obs, reward,
  terminated, truncated, info)`) with torch tensors, without the dependency.
- **Task 1 observation**: the environment keeps the posterior (the agent's memory) and shows, per action, the
  posterior-predictive dip depth, the spread of the prediction over the posterior in units of the block's shot
  noise (both windows), how often it was measured, its frequency and duration; plus the step fraction and the
  posterior width. It does not show the information gain itself. Policy: one MLP scores each action from its
  features (permutation-equivariant set policy), value from the mean-pooled embedding.
- **Task 1 reward**: per-step reduction of log(squared error + (5 uT)^2) of the posterior mean; the episode sum is
  log(e_0^2 / e_T^2), so halving the error counts the same early and late (a plain squared-error reward would make
  late steps worthless). It needs the true field: simulation only. `reward="variance"` (the posterior's own
  variance) is the hardware-compatible alternative, implemented but not used for the results.
- **Control**: the agent's network before training, run greedily. Its features already carry the predictive
  spread, so an untrained argmax is not random; the report shows what PPO adds on top.
- **Task 2** is a one-step contextual bandit: context = bit pattern, action = the write block's 32 pre-activations
  (Gaussian policy), reward per pattern = L + log2 P(pattern | readout) under the Bayes-optimal readout of the
  sampled codebook (averaged over noise draws). The same objective is optimised by gradient through the surrogate
  for the white-box comparison.
