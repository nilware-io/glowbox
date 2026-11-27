# PREREG-001 null calibration and positive control (stage S)

Stage-S validation of the PREREG-001 pipeline (docs/preregistration/PREREG-001-observer-decision.md). The simulator has no decision or observer effect by construction: these numbers say whether the frozen procedure keeps its false-positive rate and has its planned power, nothing about the hypotheses. Status of the plan: DRAFT (not signed off, not locked).

Every replicate is a complete simulated experiment: offline sequences, a calibration block of 200 valid blocks (seven sealed control arms, slot for slot like a confirmatory block), the frozen calibration (empirical null, margins, N), a confirmatory run of N blocks x 7 arms in a random order per block, sealing of every outcome, arm codes, the frozen blinded analysis (2000 null draws per test), unblinding, the pre-registered cells. T = 100 steps x S = 400 shots per episode; 4 true fields cycled over replicates; slow laser and contrast drift (OU over episode slots), per-step laser jitter. Decision process: the stand-in threshold rule (see the script).

## Frozen quantities (median over the null replicates, each replicate freezes its own)

- Margin (smallest effect size of interest, 0.95 quantile of |control-vs-control pair statistic|): lag 0 (H_observer) 0.01264 [IQR 0.0121-0.01323], lag 1 (H_decision) 0.01264 [0.01205-0.01319] in units of the referenced fluorescence r = N_sig / N_ref.
- Spin signal of the replayed sequences, C_ref = mean(r_detuned - r_resonant): 0.03111; margins / C_ref: 0.402 (lag 0), 0.401 (lag 1).
- N (valid confirmatory blocks): [100] (the floor N_MIN = 100; the power rule asks for fewer).
- theta0_D (selection effect of the decision arm under drift, from 500 model runs): -1.3e-05 +- 3e-05 (MC SE; the plan requires SE <= margin / 10 = 0.00126). The observer contrast of the same runs: -4.95e-06 +- 2.9e-05 (expected 0).

## Null calibration (no effect)

| quantity | H_decision | H_observer |
| --- | --- | --- |
| rejected after Holm (FWER 0.05) | 0.033 [0.023, 0.046] (n = 1000) | 0.029 [0.020, 0.041] (n = 1000) |
| p <= 0.05 without correction | 0.052 [0.039, 0.068] (n = 1000) | 0.053 [0.040, 0.069] (n = 1000) |
| TOST equivalent at the frozen margin | 1.000 [0.996, 1.000] (n = 1000) | 1.000 [0.996, 1.000] (n = 1000) |
| 95 % interval covers 0 | 0.948 [0.932, 0.961] (n = 1000) | 0.947 [0.931, 0.960] (n = 1000) |
| Newey-West check, p <= 0.05 | 0.055 [0.042, 0.071] (n = 1000) | 0.077 [0.061, 0.095] (n = 1000) |

- Primary family, any Holm rejection: 0.056 [0.043, 0.072] (n = 1000).
- Secondary family without the quantum test (S1-S3), any Holm rejection: 0.058 [0.044, 0.074] (n = 1000); the quantum test S4 (detuned vs resonant task error) rejects in 1.000 [0.996, 1.000] (n = 1000).
- Null-stability check flags the run (confirmatory controls differ from the calibration controls): 0.038 [0.027, 0.052] (n = 1000) (designed rate <= 0.05).
- Arm names absent from every blinded output: True.

## Positive control (effect planted in the simulated device)

Decision: E[r] of the decision arm A1 shifted by the size. Observer: E[r] of every observed arm (A1, A2, A4, A5) shifted, the sealed arms not. Size = the replicate's own frozen margin, or a fraction of it (power curve).

| planted | size | power (Holm rejection of the planted hypothesis) | attributed (planted rejected, other not) | not equivalent |
| --- | --- | --- | --- | --- |
| decision | 1 x margin | 1.000 [0.991, 1.000] (n = 400) | 0.948 [0.921, 0.967] (n = 400) | 0.990 [0.975, 0.997] (n = 400) |
| decision | 0.5 x margin | 1.000 [0.982, 1.000] (n = 200) | 0.955 [0.916, 0.979] (n = 200) | 0.000 [0.000, 0.018] (n = 200) |
| decision | 0.25 x margin | 0.985 [0.957, 0.997] (n = 200) | 0.910 [0.861, 0.946] (n = 200) | 0.000 [0.000, 0.018] (n = 200) |
| decision | 0.125 x margin | 0.530 [0.458, 0.601] (n = 200) | 0.495 [0.424, 0.566] (n = 200) | 0.000 [0.000, 0.018] (n = 200) |
| observer | 1 x margin | 1.000 [0.991, 1.000] (n = 400) | 0.945 [0.918, 0.965] (n = 400) | 0.973 [0.951, 0.986] (n = 400) |
| observer | 0.5 x margin | 1.000 [0.982, 1.000] (n = 200) | 0.950 [0.910, 0.976] (n = 200) | 0.000 [0.000, 0.018] (n = 200) |
| observer | 0.25 x margin | 0.990 [0.964, 0.999] (n = 200) | 0.955 [0.916, 0.979] (n = 200) | 0.000 [0.000, 0.018] (n = 200) |
| observer | 0.125 x margin | 0.535 [0.463, 0.606] (n = 200) | 0.505 [0.434, 0.576] (n = 200) | 0.000 [0.000, 0.018] (n = 200) |

## Reading

- Pass criteria of the plan: false-positive rate per hypothesis within the binomial interval of 0.05 (Holm makes it at most 0.05); TOST equivalence >= 0.90 under the null; power >= 0.90 and non-equivalence >= 0.90 at the margin; correct attribution. The rows below 1 x margin are the power curve only (no pass criterion): they show how far below the margin the plan still detects an effect (the margin governs the equivalence claims and the power target).
- The Newey-West robustness check is liberal at N = 100 (above 0.05 in the null), one reason it is not the decisive test.
- The calibration block mirrors the confirmatory block slot for slot. A first version of this validation used a 4-arm calibration block: its paired controls were closer in time than the 7-arm confirmatory pairs, saw less drift, and the test was liberal (H_decision p <= 0.05 in 0.070 [0.055, 0.088] of 1000 null replicates, 95 % coverage 0.930). The control runs must share the confirmatory schedule, not only its arms.
- One sealing key per batch of replicates instead of one per replicate (speed); every outcome is still sealed and unsealed through the same code. Blinding keys and order seeds are per replicate.
- The stand-in decision process is not the M5 policy, and theta0_D depends on the policy. The command below is the full size with the stand-in (R = 2000 null replicates; the null phase takes about twice as long as here). With the real policy the decision process is a Controller (glowbox.analysis.arms) wrapping the M5 policy over its posterior, run on the GPU machine; that adapter belongs to M7, and its theta0_D is the one to freeze.

    uv run python -m glowbox.experiments.prereg_calibration --replicates 2000

Wall time here: 13.7 min on a CPU.

![calibration](figures/prereg-calibration-calibration.png)

## Provenance

- run id: `20261127-084033-prereg-001-calibration-a74f`
- git: `5ea2e41abe285a1ee08ff3ceeb3fa8ac04ff9fbd`
- seed: 20261127
- device: cpu
- python 3.12.3, glowbox 0.1.0, torch 2.11.0+cu128, qutip 5.3.1, numpy 2.5.5, scipy 1.18.1, pyarrow 26.0.1
- started 2026-11-27T08:40:33+00:00, finished 2026-11-27T08:54:17+00:00

## Config

```yaml
controller:
  coarse_step: 4
  fine_step: 1
  threshold_below_baseline: 0.03
detuned_offset_hz: 400000000.0
field_pool: 4
measurement:
  amplitude: 0.6
  durations_ns:
  - 100
  f_start_hz: 2770000000.0
  f_stop_hz: 2970000000.0
  init_ns: 1500
  settle_ns: 3000
  step_hz: 2000000.0
  windows_ns:
  - 300
  - 1700
  - 300
name: prereg-001-calibration
nuisance:
  contrast_rel_sigma: 0.05
  contrast_tau_slots: 70
  laser_jitter_rel: 0.01
  laser_rel_sigma: 0.02
  laser_tau_slots: 20
numerics:
  detuning_rule: uniform
  detuning_spacing: 2.0
  n_detuning: 8
off_line_widths: 3.0
prior:
  b_max_t: 0.003
  b_min_t: 0.0005
protocol:
  n_steps: 100
  shots: 400
reference_window: 2
replicates:
  batch: 500
  curve: 200
  curve_fractions:
  - 0.5
  - 0.25
  - 0.125
  n_boot: 2000
  null_runs: 1000
  positive: 400
  theta0: 500
  workers: 8
seed: 20261127
signal_window: 0
```

## Final metrics

| metric | value |
| --- | --- |
| fpr_decision | 0.033 |
| fpr_observer | 0.029 |
| fwer_primary | 0.056 |
| margin_lag0_median | 0.0126388 |
| margin_lag1_median | 0.0126441 |
| theta0_d | -1.30012e-05 |
| theta0_d_se | 2.99913e-05 |
| power_decision_1 | 1 |
| power_decision_0.5 | 1 |
| power_decision_0.25 | 0.985 |
| power_decision_0.125 | 0.53 |
| power_observer_1 | 1 |
| power_observer_0.5 | 1 |
| power_observer_0.25 | 0.99 |
| power_observer_0.125 | 0.535 |
