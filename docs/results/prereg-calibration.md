# PREREG-001 null calibration and positive control (stage S)

Stage-S validation of the PREREG-001 pipeline (docs/preregistration/PREREG-001-observer-decision.md). The simulator has no decision or observer effect by construction: these numbers say whether the frozen procedure keeps its false-positive rate and has its planned power, nothing about the hypotheses. Status of the plan: DRAFT (not signed off, not locked).

Every replicate is a complete simulated experiment: offline sequences, a calibration block of 200 valid blocks (seven sealed control arms, slot for slot like a confirmatory block), the frozen calibration (empirical null, margins, theta0_D set, model check), a confirmatory run of N = 100 valid blocks x 7 arms in a random order per block, sealing of every outcome, decoys from the frozen model, the arm-revealing fields sealed with the blinding key, the locked reduction to the episode table, the frozen blinded analysis (2000 null draws per test), unblinding, the complete decision rule. T = 100 steps x S = 400 shots per episode; 4 true fields cycled over replicates; slow laser and contrast drift (OU over episode slots), per-step laser jitter. Decision process: the stand-in threshold rule (see the script).

## Frozen quantities (nominal condition; median over the null replicates, each replicate freezes its own)

- Margin rule (section 5): Delta = Q_{1-alpha/2}(|D|) + Q_{0.90}(|D|), D the control-vs-control null of the mean of N = 100 pairs (the shift detected with power 0.95 at Holm's first level).
- Margin, lag 0 (H_observer): 0.00252 [IQR 0.002377-0.002663]; lag 1 (H_decision): 0.002514 [IQR 0.002386-0.002661], in units of the referenced fluorescence r = N_sig / N_ref.
- Control-vs-control SD of the mean of N pairs (lag 0): 0.0006505 [IQR 0.0006118-0.0006845]; margin / SD: 3.88.
- Spin signal of the replayed sequences, C_ref = mean(r_detuned - r_resonant): 0.03119; margin / C_ref: 0.079 (lag 0), 0.080 (lag 1).
- theta0_D set (model runs, phase `model`, per model and per true field: the model of a run is the model of its sample; the plan requires every SE <= margin / 10 = 0.000251):
  - nominal (500 runs): theta0_D per field 1.83e-05 +- 5.4e-05, 1.66e-05 +- 6.1e-05, -4.21e-05 +- 6.1e-05, 5.54e-05 +- 5.4e-05; predicted wcorr per field -0.0104, -0.0094, -0.0103, -0.0101.
  - within_drift (500 runs): theta0_D per field -0.000458 +- 8e-05, -0.000668 +- 9e-05, -0.000323 +- 7.3e-05, -0.000606 +- 9.5e-05; predicted wcorr per field 0.0157, 0.049, 0.0302, 0.0635.

## Pass criteria (plan section 12)

Each criterion is a rule on the Clopper-Pearson 95 % interval of a rate: an error rate passes when the interval is not above alpha (its lower bound <= alpha), a coverage, power or claim rate when the interval is not below its target (upper bound >= target).

| criterion | rate [95 % interval] | rule | result |
| --- | --- | --- | --- |
| nominal: H_decision, p <= 0.05 against the calibration null | 0.057 [0.043, 0.073] (n = 1000) | not above 0.05 | PASS |
| nominal: H_observer, p <= 0.05 against the calibration null | 0.065 [0.051, 0.082] (n = 1000) | not above 0.05 | FAIL |
| nominal: primary family, any positive claim (complete rule) | 0.045 [0.033, 0.060] (n = 1000) | not above 0.05 | PASS |
| nominal: secondary family S1-S3, any Holm rejection | 0.075 [0.059, 0.093] (n = 1000) | not above 0.05 | FAIL |
| nominal: H_decision 95 % interval covers 0 | 0.942 [0.926, 0.956] (n = 1000) | not below 0.95 | PASS |
| nominal: H_observer 95 % interval covers 0 | 0.935 [0.918, 0.949] (n = 1000) | not below 0.95 | FAIL |
| nominal: H_decision null claim | 0.877 [0.855, 0.897] (n = 1000) | not below 0.9 | FAIL |
| nominal: H_observer null claim | 0.889 [0.868, 0.908] (n = 1000) | not below 0.9 | PASS |
| nominal: null-stability check flags the run | 0.052 [0.039, 0.068] (n = 1000) | not above 0.05 | PASS |
| nominal: arm names absent from every blinded output | 1.000 [0.996, 1.000] (n = 1000) | not below 1 | PASS |
| positive decision, 1 x margin: power of the complete rule | 0.925 [0.895, 0.949] (n = 400) | not below 0.9 | PASS |
| positive decision, 1 x margin: attributed | 0.890 [0.855, 0.919] (n = 400) | not below 0.9 | PASS |
| positive decision, 1 x margin: not equivalent | 0.978 [0.958, 0.990] (n = 400) | not below 0.9 | PASS |
| positive observer, 1 x margin: power of the complete rule | 0.907 [0.875, 0.934] (n = 400) | not below 0.9 | PASS |
| positive observer, 1 x margin: attributed | 0.875 [0.839, 0.906] (n = 400) | not below 0.9 | PASS |
| positive observer, 1 x margin: not equivalent | 0.965 [0.942, 0.981] (n = 400) | not below 0.9 | PASS |
| drift_covered: primary family, any positive claim | 0.033 [0.016, 0.060] (n = 300) | not above 0.05 | PASS |
| drift_unmodelled: primary family, any positive claim | 0.023 [0.009, 0.047] (n = 300) | not above 0.05 | PASS |
| nonstationary: primary family, any positive claim | 0.027 [0.012, 0.052] (n = 300) | not above 0.05 | PASS |
| engineering test, null, normal, two-sided: rejection rate | 0.049 [0.040, 0.059] (n = 2000) | not above 0.05 | PASS |
| engineering test, null, t3, two-sided: rejection rate | 0.051 [0.042, 0.062] (n = 2000) | not above 0.05 | PASS |
| engineering test, null, normal, one-sided: rejection rate | 0.046 [0.037, 0.056] (n = 2000) | not above 0.05 | PASS |
| engineering test, effect = margin, normal, two-sided: power | 0.944 [0.933, 0.954] (n = 2000) | not below 0.9 | PASS |
| engineering test, effect = margin, normal, one-sided: power | 0.947 [0.936, 0.956] (n = 2000) | not below 0.9 | PASS |

All criteria: FAIL.

## Null calibration (no effect), nominal condition

| quantity | H_decision | H_observer |
| --- | --- | --- |
| positive claim (complete rule) | 0.022 [0.014, 0.033] (n = 1000) | 0.027 [0.018, 0.039] (n = 1000) |
| Holm rejection against the calibration null alone | 0.026 [0.017, 0.038] (n = 1000) | 0.040 [0.029, 0.054] (n = 1000) |
| p <= 0.05 against the calibration null (uncorrected) | 0.057 [0.043, 0.073] (n = 1000) | 0.065 [0.051, 0.082] (n = 1000) |
| TOST equivalent at the frozen margin | 0.930 [0.912, 0.945] (n = 1000) | 0.930 [0.912, 0.945] (n = 1000) |
| null claim (equivalent, stable null; H_decision: model check) | 0.877 [0.855, 0.897] (n = 1000) | 0.889 [0.868, 0.908] (n = 1000) |
| 95 % interval covers 0 | 0.942 [0.926, 0.956] (n = 1000) | 0.935 [0.918, 0.949] (n = 1000) |
| Newey-West check, p <= 0.05 | 0.059 [0.045, 0.075] (n = 1000) | 0.076 [0.060, 0.094] (n = 1000) |

- Model check passed: calibration block 0.995 [0.988, 0.998] (n = 1000), confirmatory sealed controls 0.996 [0.990, 0.999] (n = 1000).
- The quantum test S4 (detuned vs resonant task error) rejects in 1.000 [0.996, 1.000] (n = 1000).

## Stress conditions (no effect)

| condition | what differs | positive claim, H_decision | positive claim, H_observer | H_decision p <= 0.05 (calibration null alone) | H_observer p <= 0.05 (calibration null alone) | model check passed (both) | null stable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| drift_covered | within-episode contrast drift (sd 0.3, rho 0.9 per step), in the model set | 0.020 [0.007, 0.043] (n = 300) | 0.027 [0.012, 0.052] (n = 300) | 0.037 [0.018, 0.065] (n = 300) | 0.063 [0.039, 0.097] (n = 300) | 0.987 [0.966, 0.996] (n = 300) | 0.940 [0.907, 0.964] (n = 300) |
| drift_unmodelled | the same drift, missing from the model set | 0.000 [0.000, 0.012] (n = 300) | 0.023 [0.009, 0.047] (n = 300) | 0.087 [0.057, 0.124] (n = 300) | 0.043 [0.023, 0.073] (n = 300) | 0.000 [0.000, 0.012] (n = 300) | 0.933 [0.899, 0.959] (n = 300) |
| nonstationary | confirmatory drift 3x larger and 3x faster than in the calibration block | 0.003 [0.000, 0.018] (n = 300) | 0.023 [0.009, 0.047] (n = 300) | 0.050 [0.028, 0.081] (n = 300) | 0.073 [0.047, 0.109] (n = 300) | 0.980 [0.957, 0.993] (n = 300) | 0.803 [0.754, 0.847] (n = 300) |

## Positive control (effect planted in the simulated device)

Decision: E[r] of the decision arm A1 shifted by the size. Observer: E[r] of every observed arm (A1, A2, A4, A5) shifted, the sealed arms not. Size = the replicate's own frozen margin, or a fraction of it (power curve: no pass criterion).

| planted | size | power (complete rule) | attributed (planted rejected, other not) | not equivalent |
| --- | --- | --- | --- | --- |
| decision | 1 x margin | 0.925 [0.895, 0.949] (n = 400) | 0.890 [0.855, 0.919] (n = 400) | 0.978 [0.958, 0.990] (n = 400) |
| decision | 0.5 x margin | 0.310 [0.221, 0.410] (n = 100) | 0.290 [0.204, 0.389] (n = 100) | 0.480 [0.379, 0.582] (n = 100) |
| decision | 0.75 x margin | 0.740 [0.643, 0.823] (n = 100) | 0.730 [0.632, 0.814] (n = 100) | 0.830 [0.742, 0.898] (n = 100) |
| observer | 1 x margin | 0.907 [0.875, 0.934] (n = 400) | 0.875 [0.839, 0.906] (n = 400) | 0.965 [0.942, 0.981] (n = 400) |
| observer | 0.5 x margin | 0.340 [0.248, 0.442] (n = 100) | 0.310 [0.221, 0.410] (n = 100) | 0.520 [0.418, 0.621] (n = 100) |
| observer | 0.75 x margin | 0.710 [0.611, 0.796] (n = 100) | 0.700 [0.600, 0.788] (n = 100) | 0.840 [0.753, 0.906] (n = 100) |

## Engineering comparisons (M7 E-1 / E-2, E5): `paired_margin_test` at its fixed sizes (20 pairs, 20 control repeats)

| condition | rejection rate | TOST equivalent | median margin (units of the diff SD) |
| --- | --- | --- | --- |
| null, normal, two-sided | 0.049 [0.040, 0.059] (n = 2000) | 0.885 [0.870, 0.899] (n = 2000) | 1.19 |
| null, t3, two-sided | 0.051 [0.042, 0.062] (n = 2000) | 0.803 [0.785, 0.820] (n = 2000) | 1.72 |
| null, normal, one-sided | 0.046 [0.037, 0.056] (n = 2000) | 0.807 [0.789, 0.824] (n = 2000) | 1.09 |
| effect = margin, normal, two-sided | 0.944 [0.933, 0.954] (n = 2000) | 0.024 [0.018, 0.032] (n = 2000) | 1.18 |
| effect = margin, normal, one-sided | 0.947 [0.936, 0.956] (n = 2000) | 0.026 [0.019, 0.034] (n = 2000) | 1.07 |

## Reading

- The margin is derived from the control runs by a fixed rule: the effect that the planned experiment detects against the control-vs-control distribution with power 0.95 at Holm's first level. It replaces the earlier rule (the 0.95 quantile of a single control pair), which made the margin about 40 % of the spin signal and its equivalence claims trivially easy.
- A positive claim needs a Holm rejection against the frozen calibration controls and against the run's own sealed controls; the nonstationary condition shows why (the calibration null alone is too narrow when the confirmatory run drifts more).
- H_decision depends on a model of the selection effect. The drift_unmodelled condition (within-episode drift missing from the model set) shows the calibration null alone rejecting too often, and the model check refusing H_decision instead. In drift_covered the same drift is in the model set: theta0_D is non-zero and the worst-case test keeps its level.
- The Newey-West robustness check is liberal at N = 100 (above 0.05 in the null), one reason it is not the decisive test.
- One sealing key and one meta-sealing key per batch of replicates instead of per replicate (speed); every outcome is still sealed and unsealed through the same code. Blinding keys and order seeds are per replicate.
- The stand-in decision process is not the M5 policy, and theta0_D depends on the policy: the M5 policy as a Controller (glowbox.analysis.arms) is M7's adapter, with its own model runs and this validation repeated.

Run sizes here (CPU): nominal 1000, drift_covered 300, drift_unmodelled 300, nonstationary 300 null replicates; positive 400 per hypothesis; curve 100 per fraction; 500 model runs per model; engineering 2000 per condition. The documented full size (several hours on this CPU; any machine, no GPU needed):

    uv run python -m glowbox.experiments.prereg_calibration --replicates 4000 --stress-replicates 1000

Wall time here: 17.6 min on a CPU.

![calibration](figures/prereg-calibration-calibration.png)

## Provenance

- run id: `20261127-100225-prereg-001-calibration-a8f4`
- git: `feb71252c5a5e901e5b808279bdb835f2a1ee5fe`
- seed: 20261127
- device: cpu
- python 3.12.3, glowbox 0.1.0, torch 2.11.0+cu128, qutip 5.3.1, numpy 2.5.5, scipy 1.18.1, pyarrow 26.0.1
- started 2026-11-27T10:02:25+00:00, finished 2026-11-27T10:20:04+00:00

## Config

```yaml
conditions:
  drift_covered:
    description: within-episode contrast drift (sd 0.3, rho 0.9 per step), in the
      model set
    device:
      within_rho: 0.9
      within_sigma: 0.3
    models:
    - within_drift
  drift_unmodelled:
    description: the same drift, missing from the model set
    device:
      within_rho: 0.9
      within_sigma: 0.3
    models:
    - nominal
  nominal:
    description: device = the nominal model
    models:
    - nominal
  nonstationary:
    confirmatory:
      contrast_rel_sigma: 0.15
      contrast_tau_slots: 23.3
      laser_rel_sigma: 0.06
      laser_tau_slots: 6.67
    description: confirmatory drift 3x larger and 3x faster than in the calibration
      block
    models:
    - nominal
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
models:
  nominal: {}
  within_drift:
    within_rho: 0.9
    within_sigma: 0.3
name: prereg-001-calibration
nuisance:
  contrast_rel_sigma: 0.05
  contrast_tau_slots: 70
  laser_jitter_rel: 0.01
  laser_rel_sigma: 0.02
  laser_tau_slots: 20
  within_rho: 0.0
  within_sigma: 0.0
numerics:
  detuning_rule: uniform
  detuning_spacing: 2.0
  n_detuning: 8
prior:
  b_max_t: 0.003
  b_min_t: 0.0005
protocol:
  n_steps: 100
  shots: 400
replicates:
  batch: 500
  curve: 100
  curve_fractions:
  - 0.5
  - 0.75
  engineering: 2000
  model_runs: 500
  n_boot: 2000
  null_runs: 1000
  positive: 400
  stress_runs: 300
  workers: 8
seed: 20261127
```

## Final metrics

| metric | value |
| --- | --- |
| fwer_primary_nominal | 0.045 |
| margin_lag0_median | 0.00252014 |
| margin_lag1_median | 0.00251381 |
| margin_over_c_ref_lag0 | 0.0791802 |
| criteria_passed | 20 |
| criteria_total | 24 |
| power_decision_1 | 0.925 |
| power_decision_0.5 | 0.31 |
| power_decision_0.75 | 0.74 |
| power_observer_1 | 0.9075 |
| power_observer_0.5 | 0.34 |
| power_observer_0.75 | 0.71 |
