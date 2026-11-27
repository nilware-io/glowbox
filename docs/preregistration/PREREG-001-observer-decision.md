# PREREG-001: decision and observer coupling, simulation stage of M7 and E5

**Status: DRAFT.** Not signed off and not locked. Nothing in this plan is frozen until the owner signs the final
text and the lock is committed (docs/preregistration/README.md, "Freezing"). Until then no run counts as
confirmatory, whatever its result.

- Plan: PREREG-001, version 2 (draft of 2026-11-27, revised after the red-team and engineering review).
- Analysis script: `glowbox/analysis/prereg_analysis_001.py` (frozen constants at its top; the reduction of counts
  to the episode table; the calibration, blinded analysis and unblinding steps).
- Arms and protocol runner: `glowbox/analysis/arms.py`; locking, run records, the run registry, sealing, blinding,
  schedule: `glowbox/analysis/prereg.py`.
- Run registry: `docs/preregistration/PREREG-001.registry.yaml` (append-only).
- Stage-S validation: `glowbox/experiments/prereg_calibration.py`, config
  `glowbox/experiments/configs/prereg_001.yaml`, report `docs/results/prereg-calibration.md`.
- Deviations: `docs/preregistration/deviations.md` (append-only).

## 1. Question and scope

Does coupling a decision-making process or an observing process to the measurement change the measured outcome
distribution beyond what the physics model and the timing model predict?

- A **decision-making process** chooses the next control parameter from the outcomes so far, and its choice is
  applied.
- An **observing process** receives the outcomes, or any function of them, while the run is going on.

The comparisons are built so that, without such an effect, the physics gives the same distribution in the compared
arms. A difference counts only when it lies beyond the control-run variability and the model's predicted selection
effect.

**Two stages with different evidential status.**

- **Stage S (simulation, this plan).** The simulator has no decision or observer effect by construction. Stage S
  only validates the pipeline: arms, randomisation, sealing, blinding, locking, the frozen analysis, its
  false-positive rate and its power. No stage-S result counts as evidence for or against either hypothesis.
- **Stage H (hardware).** Confirmatory tests happen only on hardware, under a successor plan (PREREG-002) frozen
  after commissioning (section 11). PREREG-002 reuses this plan's machinery and rules and adds the hardware
  conditions.

This plan also fixes how the simulation stage of milestone M7 (the three-controller comparison) and of the E5
spins-as-clock run are analysed (section 10).

## 2. Measurement task and observable

- **Task.** The milestone-5 adaptive field measurement (Task 1). One **episode** has T steps; each step is one
  pulsed-ODMR action (an MW frequency on a grid, times a pulse duration) repeated for S shots. Stage-S validation
  values: T = 100, S = 400, 100 frequencies on a 2 MHz grid over 2.77-2.97 GHz, a ~pi pulse of 100 ns.
- **Readout windows** (docs/physics.md section 8): signal = the first 300 ns of the readout pulse; reference = the
  window 2.0-2.3 us into the same pulse, after re-polarisation (spin-independent). A 3 us settle follows the
  initialisation pulse.
- **Per-step observable.** r_t = N_sig,t / N_ref,t, the photon counts summed over the S shots in the two windows.
  Laser-power drift cancels to first order.
- The sample and its field stay fixed for a whole run (on hardware by nature; in simulation one field per run).
  The decision process restarts from its prior at every episode.

## 3. Arms

All arms run the same code path, the same decision-process computation and the same step timing; only the switches
below differ. Every outcome of every arm is sealed by the acquisition code (section 7).

**Confirmatory arms** (one episode of each per block):

| Code | Arm | Applied actions | Decision process | Live readers of the outcomes |
| --- | --- | --- | --- | --- |
| A1 | DECISION | the decision process's choices | runs, applied | decision process and monitors |
| A2 | YOKED, OBSERVED | replay of A1's actions of the previous block | runs on the live outcomes, output discarded | decision process and monitors |
| A3 | YOKED, SEALED | the same replay | runs on a decoy of the same shape, output discarded | none |
| A3b | YOKED, SEALED (second copy) | the same replay | as A3 | none |
| A3L | YOKED, SEALED, LAG 2 | replay of A1's actions of two blocks earlier | as A3 | none |
| A4 | SHAM DECISION | iid draws from the frozen sham distribution | runs on the live outcomes, output discarded | decision process and monitors |
| A5 | DETUNED | the A2 replay with every MW frequency + 400 MHz | as A2 | decision process and monitors |

**Calibration arms** (the CALIBRATION block, section 5). The calibration block mirrors a confirmatory block slot
for slot (the same number of arms and episodes, so the same drift between paired slots), with every arm replaced by
a sealed replay: no decision process applied, no live reader.

| Code | In the role of | Applied actions |
| --- | --- | --- |
| C1 | A1 | offline sequence k (in block k) |
| C2, C3, C3b | A2, A3, A3b | offline sequence k - 1 |
| C3L | A3L | offline sequence k - 2 |
| C4 | A4 | iid draws from the sham distribution (sealed) |
| C5 | A5 | offline sequence k - 1 with every MW frequency + 400 MHz |

**Definitions.**

- **Offline sequences**: action sequences produced by the frozen decision process run on a separate simulated
  device, not on the measurement under test, before the calibration block. Their marginal action distribution is
  the **sham distribution**.
- **Yoked replay**: the applied actions are exactly the recorded table (tested in `tests/analysis/test_arms.py`).
  The decision process still runs, on the same inputs at the same steps, and its proposals are logged and
  discarded.
- **Sealed**: the outcomes go only to the sealed writer. The decision process and every monitor receive a decoy:
  counts drawn from the frozen model's expected counts at the applied action (Poisson, a generator of its own), so
  the decision process runs on inputs with the structure of real outcomes. The episode's count of plaintext
  deliveries is logged and must be 0.
- **What travels in the clear**: an episode's arm code (or a random token), block, time slots and applied actions.
  Outcome records are labelled by the code, never the arm name. The fields that differ between observed and sealed
  arms by construction (count of plaintext deliveries, the decision process's proposals, which in an observed arm
  are a function of the real outcomes, the warm-up fallback flag, the decision process's compute time per step)
  are sealed with the blinding key and released with it.
- **Lag-one yoking** lets the order of the arms inside a block be fully random, because replays never depend on an
  episode of the same block. The first two blocks of the confirmatory run and of the calibration block are warm-up
  blocks (their lag-2 replays have no source yet) and are excluded from every analysis.
- **Order inside a block**: a uniformly random permutation per block, drawn from the run's order seed (section 7).
  The whole order table exists before data; a resumed run continues at the next row and never redraws.

## 4. Hypotheses, statistics and comparison cells

A **cell** is a pair of arms compared episode by episode: x in block k against y in block k + lag, over the pairs
whose applied (nominal) action sequences are identical. The **pair statistic** of the primaries is

    y_p = mean over t of (r_x,t - r_y,t)

and a cell's estimate is the mean of y_p over its valid pairs.

**H_decision.** Cell (A1 in block k, A2 in block k + 1): identical actions, both observed. The two differ only in
whether the outcome-contingent choice reached the control.

- Null: the cell's mean equals theta0_D, the selection effect that the frozen physics, drift and timing model
  predicts for this protocol. A decision arm's choices depend on counts that carry drift, so the replay one block
  later can differ without any new effect.
- **The model set.** theta0_D is not a single number. The plan names a set of drift and timing models (stage S:
  the nominal model; stage H: the commissioned model and its variants with within-episode correlated drift, bounded
  by commissioning data). Each member gets its theta0_D and Monte Carlo SE from **model runs**: registered runs of
  that model (mode `model`) with the confirmatory design, made before the calibration block; all registered model
  runs are used. Each SE must be at most margin / 10 (enforced by the script).
- **Worst case.** H_decision counts only if it rejects at every member's theta0_D (the reported p is the largest),
  and it is equivalent only if it is equivalent at every member's theta0_D.
- **Model check.** A misspecified drift model (above all, drift correlated within an episode that an adaptive
  decision process can exploit) would show up as a decision effect. The model runs therefore also predict the
  within-episode correlation wcorr of sealed-control pairs (the mean autocorrelation at lags 1-3 of the per-step
  difference of two sealed replays of the same sequence). The calibration block's (C2, C3) pairs and the
  confirmatory run's (A3, A3b) pairs must each lie inside the prediction range of at least one member of the set
  (two-sided, 0.005 each). If either fails, H_decision is **inconclusive** ("the drift model does not reproduce
  the sealed controls") and no claim is made either way.
- Alternative: two-sided. No direction is predicted.

**H_observer.** Cell (A2, A3) in the same block: identical actions, and neither applies a decision. They differ only
in whether the outcomes were read while the run was going on.

- Null: mean 0 (identical inputs adjacent in time, random order). Alternative: two-sided.

**What the hypotheses are not.** They say nothing about how well the decision process estimates the field. Adaptive
design is expected to do better than a replay; that is a physics and statistics result, not an effect.

## 5. Empirical null and the smallest effect size of interest

The owner decided on 2026-11-27 that the smallest effect size of interest is defined from the control runs, as
statistical significance relative to the control runs. The null is therefore **empirical, from control runs**.

**Calibration block.** Before any decision or observer arm runs, a separate CALIBRATION block of M_CAL = 200 valid
blocks of the calibration arms (section 3) is run under its own random order table. Its control-vs-control pairs
come in the same two geometries as the primaries:

- lag 0: (C2, C3), same block, same sequence (the geometry of H_observer);
- lag 1: (C1 in block k, C2 in block k + 1), same sequence (the geometry of H_decision);
- block: (C4, C2), same block, different actions (the geometry of the secondary S1).

The calibration run and all model runs are registered before their first measurement. The locked script
(`calibrate`) then checks every run against the plan lock and the registry, computes the following, writes the
calibration file and a **calibration lock** (the calibration's sha256, the plan lock's sha256, the calibration run
and its data sha256 and start time, every model run and its data sha256), and appends a `calibration` event to the
registry. All are committed before the confirmatory run starts, and the confirmatory run records the calibration
lock it will be analysed against.

1. **The empirical null of every test.**
   - The null distribution of a cell's mean over n pairs is a circular moving-block bootstrap of the calibration
     series of the same geometry and statistic, with block length floor(4 (M / 100)^(2/9)) and a random sign per
     bootstrap block.
   - The spread and shape come from the controls. The centre is zero because the controls are exchangeable (random
     order inside blocks).
   - Two-sided p = (1 + #{|null| >= |d|}) / (B + 1), d the cell's centred estimate.
2. **The margin (smallest effect size of interest)**, per geometry, fixed rule:

       Delta = Q_{1 - alpha/2}(|D|) + Q_{0.90}(|D|),

   where D is the empirical control-vs-control null of the mean of N = 100 pairs (item 1) and the quantiles are of
   |D|. For a symmetric null, Q_{1 - alpha/2}(|D|) is the critical value of the test at Holm's first level (alpha / 2,
   two-sided) and Q_{0.90}(|D|) = Q_{0.95}(D), so an effect of Delta is detected against the controls with power
   0.95 (MARGIN_POWER) at the planned N. For roughly normal controls Delta = 3.9 x the control-vs-control SD of the
   N-pair mean.
   - Meaning: the smallest effect that this experiment, at its fixed size, separates from the control-vs-control
     variability with high power. Smaller effects are not "absent": they are below the resolution the plan is
     built for, and the plan makes no claim about them.
   - Why 0.95 and not 0.90: a positive claim has two further conditions (the concurrent null and, for H_decision,
     the model check). The validation shows that the complete rule keeps power >= 0.90 at Delta.
   - Why not the earlier rule (the 0.95 quantile of a single control pair, MARGIN_PAIRS = 1): that is about
     2 sigma of a single pair, 20 SE of the cell mean and about 40 % of the spin signal C_ref; the power rule never
     bound, equivalence held in every null run, and an effect of a third of the spin signal would have been called
     "practically null".
   - Delta is also reported in units of the spin signal of the replayed sequences, C_ref = the mean over
     calibration blocks of (r of C5 - r of C2), next to every equivalence claim.
3. **N = 100 valid confirmatory blocks, fixed.** The margin is derived for this N; there is no data-dependent sample
   size. Stage S gives Delta / C_ref of about 0.08 (section 13).
4. **The theta0_D set and the model check** (section 4).
5. **The bootstrap seed**, from the sha256 of the calibration data (no analyst choice).

**Equivalence (TOST).** A primary is equivalent when d - q > -Delta and d + q < Delta, where q is the 0.975 quantile
of the empirical null (alpha_eq = 0.025 per side: Bonferroni over the two equivalence claims).

**The concurrent null.** The calibration null is frozen before the confirmatory run; if the device then drifts
faster or is noisier, the frozen null is too narrow. Every primary is therefore also tested against the run's own
sealed controls of the same geometry (computed blind, for every cell against every other cell of its geometry):
H_observer against the (A3, A3b) series, H_decision against the (A3 in block k, A3L in block k + 1) series, centred
at each theta0_D.

**Decision rule** (per primary):

- **Positive claim** ("effect detected"): Holm (FWER 0.05 over the two primaries) rejects against the calibration
  null, Holm rejects against the concurrent null, and (H_decision) the model check passed.
- **Null claim**: TOST equivalent at the frozen margin, the null-stability check passed, and (H_decision) the model
  check passed.

| Positive rule | Equivalence (stable null) | Conclusion |
| --- | --- | --- |
| rejects | not equivalent | effect detected |
| rejects | equivalent | significant, but smaller than the margin (no effect of the pre-registered size) |
| does not reject | equivalent | null: any effect is smaller than the margin |
| does not reject | not equivalent, or the null not stable | inconclusive; no claim either way |
| (H_decision) model check failed | | inconclusive: the drift model does not reproduce the sealed controls |

With the margin rule above the second row needs an estimate both beyond the critical value and inside
Delta - q; it is kept so that the table is complete.

**Null stability.** Every confirmatory block also carries sealed controls whose cells mirror the calibration pairs:
(A3, A3b) at lag 0, and (A3 in block k, A3L in block k + 1) at lag 1.
- Each is compared with the frozen calibration series by location (empirical-null p) and scale (Brown-Forsythe):
  four tests at 0.05 / 4 each.
- A failure means the null of the confirmatory run differs from the frozen one. It is **reported** with the
  primaries, and nothing is re-fitted.
- Consequence: no null claim comes from an unstable run, and an unstable run cannot serve as the original or the
  replication of a null claim. A positive claim does not rest on the frozen null alone (the concurrent null).

**Robustness check** (pre-specified, not decisive): a parametric z with a Newey-West standard error per cell. If it
disagrees with the empirical-null test, both are reported and the empirical test governs.

## 6. Secondary outcomes

One family, Holm at FWER 0.05. No claims come from them if the primaries are null. Every secondary uses the empirical
null of the same calibration geometry and statistic, centred at zero (theta0_D is the model prediction for the
primary statistic only).

- **S1, sham vs yoked**: the mean model residual (r - r_model(action)) of A4 against A2 in the same block. The
  actions differ, hence residuals. r_model is the frozen model's prediction (stage S: the simulator's noiseless
  means).
- **S2, dispersion**: the log ratio of the residual variances, A1 vs A2 (lag 1) and A2 vs A3 (lag 0).
- **S3, spin specificity**: the primary statistics restricted to steps whose applied frequency is more than
  3 x (Rabi frequency at the pulse amplitude + T2* FWHM) from every line. A spin-coupled effect must vanish there.
- **S4, quantum test**: the task error of A5 against A2 (the detuned arm must lose the task).

## 7. Sealing, blinding, locking

- **Keys per run** (`RunKeys`): a sealing key, a blinding key and an order seed.
  - They are generated at run start and kept in a key store outside the run's output directory.
  - run.yaml records only their sha256 commitments.
  - Every release is logged with its time.
- **Sealing**: every outcome record of every arm is encrypted and authenticated before it leaves the acquisition
  code (SHAKE-256 keystream, HMAC-SHA256 tag). The sealing key is released when the run has ended, so the analysis
  only ever sees unsealed data after that.
- **Blinding**: arms appear in every analyst-facing output as random 4-letter codes derived from the blinding key.
  The blinded analysis computes every cell of every code pair (the replay structure is visible from the action
  hashes) and does not need to know the arms.
- **Order of steps after a run**:
  1. Release the sealing key, unseal, and build the episode table and the arm-blind exclusion flags with the locked
     reduction (`build_episode_table`: r, r_model, off-line flags, task error, sequence hashes and the block
     exclusions of section 8). The table carries arm codes only.
  2. Run the locked script (`analyse`) on all valid data. It refuses unless the plan lock, the calibration lock, the
     calibration run and the confirmatory run all check out (README, rules 2-5). Its output records the run, the
     plan lock, the calibration lock and the script sha256. Commit it and the registry entry of its sha256, with the
     deviations log.
  3. Only then release the blinding key, run the locked `unblind` step (it refuses an output whose sha256 is not the
     registered one), and read off the pre-registered cells.
- **What blinding protects.** An analyst can guess A5 (its large detuning gap) and the replay structure. Blinding
  guards the decisions that could still be influenced (exclusions, deviations, failure handling), all made before
  unblinding. Unblinding after the analysis is a commit, so its time is auditable.
- **Locking**: see README, "Freezing". The analysis refuses to start when any locked file differs or the lock is not
  held by its pushed tag. It refuses runs whose record is missing, belongs to another lock, mode or calibration,
  carries a planted effect outside simulation, came from a dirty git tree or from a commit that does not hold the
  locked files, or is not registered.
- **Registry**: every model, calibration and confirmatory run is registered before its first measurement, and every
  registered run under the lock is reported (completed or aborted). One calibration block per series.

## 8. Sample size, stopping, exclusions

- **Fixed sample size, no interim looks.**
  - N = 100 valid confirmatory blocks, M_CAL = 200 valid calibration blocks, both fixed.
  - Collection stops at N valid blocks or at 1.2 N attempted blocks, whichever comes first. The calibration block
    stops at M_CAL valid or 1.2 M_CAL attempted blocks.
  - Stopping depends only on validity flags, computed blind to arm and outcome.
  - Peeking is impossible by construction: the outcomes and the arm map are sealed.
- **Shortfall**: if the cap is reached with fewer valid blocks, the analysis runs on what exists and the shortfall
  is a deviation.
- **Exclusion of a block as a whole** (all its arms; automated, arm-blind):
  - a warm-up block;
  - an interlock trip;
  - counter overflow or saturation;
  - sequencer underrun or loss of sync;
  - a reference-window count below the frozen floor (REF_COUNT_FLOOR = 100 counts per step summed over the shots;
    PREREG-002 refixes it for the hardware count rates) (laser fault);
  - non-finite values;
  - an incomplete episode in any arm.

  A pair is used only when both of its blocks are valid.
- **Not excluded** (intention to treat): slot overruns (their rate per arm is reported, and above 1 % is a
  deviation), outliers in r, unusual task outcomes.
- **Failed runs**: every completed, flag-clean block is kept, and a restart continues the sealed order table. No data
  are ever discarded because of their values.

## 9. Replication and exploratory results

- A **replication** is a new run series under the same lock, with:
  - new keys and a new order seed;
  - a new calibration block;
  - a start at least a week after the original;
  - in stage H, ideally a second diamond or a second chip, started by a different person.
- A positive claim needs a positive claim (section 5, decision rule) in the same direction in the original and in
  the replication. A null claim needs a null claim in both; an unstable run counts as neither.
- A replication series registers its own model runs (if the model set is refitted, that needs a new lock),
  calibration block and confirmatory run; every registered series is reported.
- Everything else (any other cell of the matrix, any post-hoc pattern) is **exploratory** and labelled so. It
  becomes a claim only after a new pre-registration and a successful confirmatory run.

## 10. Milestone M7 and E5 (simulation stage)

**M7, three controllers**: (a) ideal clocked, (b) drifting chip-timed, (c) (b) plus event-driven feedback (the next
pulse conditioned on the previous window's photons).

- **Engineering family** (separate from the primaries; Holm at FWER 0.05 within it):
  - E-1: task error (b) - (a), the drift cost, direction > 0;
  - E-2: task error (c) - (b), the feedback benefit, direction < 0.

  Both are paired over seeds and fields, with exactly ENG_N = 20 pairs and ENG_N_CONTROL = 20 **control repeats**
  (the same controller run twice with independent noise under the same schedule); the script refuses other sizes.
  The test (`paired_margin_test`) is a two-sample permutation test of mean(diffs) - mean(control diffs), one-sided
  in the stated direction, so it carries the control sample's own uncertainty. The margin comes from the controls
  only, by the rule of section 5 in its two-sample t form: Delta = (t_{1-alpha, 19} + t_{0.95, 19}) x sd(control
  diffs) x sqrt(1/20 + 1/20). The validation reports its false-positive rate (normal and heavy-tailed diffs) and its
  power at Delta at exactly these sizes.
- **Controller (c) is a decision process**, so the H_decision and H_observer design applies to it:
  - (c) as A1; its yoked replays under the same timing-model realisation class (A2, A3, A3b, A3L); the sham and
    detuned arms; a calibration block of its offline sequences.
  - M7 supplies the controller through the `Controller` interface of `glowbox/analysis/arms.py`, and the measurement
    through `SimulatorMeasure` (or a fast table device).
  - theta0_D is frozen from model runs with that controller before its null calibration.
- **Exploratory**: which strategies the learners find, timing structure (`mw_timing_structure`,
  `rabi_band_fraction`), robustness curves against the timing noise, anything not listed above.

**E5, spins as the clock.**
- **Confirmatory in simulation**: the two-detuning estimate of the timing scale epsilon agrees with the injected
  value: `paired_margin_test` (two-sided) of 20 diffs (estimate - injected) against 20 control diffs ((estimate -
  estimate') / sqrt(2), two independent runs at the same injected epsilon), equivalence by its TOST at the
  control-derived margin.
- **Exploratory**: the delta separation, the echo variants, the shot-budget fraction.
- E5 has no decision or observer arm unless its locking loop is closed. A closed loop is a decision process and gets
  the arm structure above.

## 11. What changes on hardware (PREREG-002, frozen after commissioning)

- **Interleaving** at block level as above, with the detuning set only at the synthesiser, and shot-level alternation
  of resonant and detuned inside a block where the sequencer allows it.
- **Constant workload**: branch-free, fixed-length playback tables in every arm, and the decision and monitor
  computations running in every arm. In sealed arms the counts leave the counter board only encrypted, with the same
  byte count and transfer times as in the other arms.
- **Display**: in observing arms the display shows the live results. In sealed arms it shows a decoy stream with the
  same pixel activity and mean luminance; it stays on, because switching it off changes power and EMI. Whether a
  person was present is logged per block and used only in exploratory analysis.
- **Telemetry per block** in the trace metadata: the timing scale against a crystal reference, a package thermistor,
  VDD and the shunt currents. Their balance across arms is a secondary outcome.
- **Leak test** (a precondition, run before the primaries): a decoder that sees only telemetry, only laser-only
  shots, or the decision process's compute time per step (sealed in every episode, section 3) must classify arm
  codes at chance (TOST against chance, with the margin from the same rule on shuffled labels). If it fails, the
  primaries are reported as "confounded" and no claim is made.
- **Model set** for theta0_D: the commissioned drift and timing model and its within-episode correlated variants,
  bounded by commissioning data, each with its registered model runs.
- **Equal MW power**, measured across arms, and the antenna thermally separated from the controller board.
- **Delta-shift test** for any fringe-based strategy: a real fringe moves with the synthesiser's detuning.
- **Offline sequences** for the calibration block come from the frozen controller run on the commissioned model.
- **theta0_D recomputed and frozen** from the commissioned drift and timing models before any confirmatory data.
- **Replay test** (confirmatory for any claim that depends on the chip):
  - the chip's recorded edges are replayed by a crystal-timed sequencer with the same injected classical drift;
  - estimand: the difference in task error, and in Fisher information per detected photon, between the chip-timed
    arm and the replay;
  - prediction: equivalence;
  - in addition, a one-sided test that the measured Fisher information per photon exceeds the independent-spin
    Cramer-Rao bound computed from the measured photon numbers, contrast and visibility. The prediction is that it
    does not.
- **Scaling with the number of NVs** (focal volume or diamond), checked against 1/sqrt(N_NV): secondary.
- **Required order before any claim of a chip-specific or non-classical effect**:
  1. the leak test passes;
  2. heating and sampling artefacts are excluded;
  3. the replay test is run;
  4. the bound test is run.

## 12. Stage-S validation (pass criteria)

The calibration script runs the whole procedure on simulated data, R replicates per condition. Each replicate is a
complete experiment with its own keys, order tables and calibration block, analysed with the frozen model set of
its condition (model runs per model and per true field).

Every criterion is a rule on the Clopper-Pearson 95 % interval of a rate, and the report prints PASS or FAIL for
each. An error rate passes when its interval is **not above** alpha (lower bound <= 0.05); a coverage, power or
claim rate when its interval is **not below** its target (upper bound >= target). With R = 1000 a rate of 0.07 is
excluded; the full size (R = 4000) narrows that to about 0.06.

- **Null, nominal condition** (the device is the model):
  - each primary's uncorrected rate p <= 0.05 against the calibration null: not above 0.05;
  - the primary family (any positive claim under the complete rule): not above 0.05;
  - the secondary family S1-S3 (any Holm rejection): not above 0.05;
  - each primary's 95 % interval covers zero: not below 0.95;
  - each primary's null claim: not below 0.90;
  - the null-stability flag rate: not above 0.05;
  - arm names absent from every blinded output: in every replicate.
- **Null, stress conditions** (primary family, any positive claim: not above 0.05 in each):
  - drift_covered: within-episode contrast drift (AR(1), sd 0.3, correlation 0.9 per step), in the model set
    (theta0_D clearly non-zero);
  - drift_unmodelled: the same drift, missing from the model set (the model check must refuse H_decision);
  - nonstationary: confirmatory drift 3x larger and 3x faster than in the calibration block.
- **Positive control**: an effect of exactly the replicate's frozen margin, planted in the decision arm (H_decision)
  or in every observed arm (H_observer). Power of the complete rule, attribution (the planted hypothesis claimed,
  the other not) and non-equivalence: each not below 0.90. Below the margin only the power curve is reported.
- **Engineering test** (`paired_margin_test` at ENG_N = ENG_N_CONTROL = 20): rejection rate not above 0.05 under
  the null (normal and t3 diffs, two- and one-sided); power at the control-derived margin not below 0.90.
- Results: `docs/results/prereg-calibration.md`.

## 13. Open for the owner before sign-off

- **The margin rule** (section 5): Delta = the effect detected against the controls with power 0.95 at N = 100.
  Stage S (stand-in decision process, docs/results/prereg-calibration.md): Delta is about 0.0025 in r, that is
  about 8 % of the spin signal C_ref, about 3.9 x the control-vs-control SD of the 100-pair mean. A larger N
  gives a smaller margin (Delta falls as 1 / sqrt(N)); the hardware rates change it again (PREREG-002).
- **The sizes** M_CAL = 200, N = 100, the model runs per model (stage S: 500 over 4 true fields), ENG_N =
  ENG_N_CONTROL = 20.
- **The model set** for theta0_D and the model check's statistic (within-episode correlation at lags 1-3).
- **The decision process** of the stage-S validation (a stand-in) versus the M5 policy for the full-size run.
