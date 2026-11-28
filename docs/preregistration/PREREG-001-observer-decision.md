# PREREG-001: decision and observer coupling, simulation stage of M7 and E5

**Status: DRAFT.** Not signed off and not locked. Nothing in this plan is frozen until the owner signs the final
text and the lock is committed (docs/preregistration/README.md, "Freezing"). Until then no run counts as
confirmatory, whatever its result.

- Plan: PREREG-001, version 3 (draft of 2026-11-27, revised after the second review: the analysed data bound to
  the registered sealed artifacts, one confirmatory run per series and two fixed series, the model set locked, the
  secondary family under the complete rule, the decisive pass criterion, order checked by git ancestry; N = 400
  confirmatory blocks by the owner's decision of 2026-11-27, revalidated at the full size).
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
  the members of `models` in the stage-S config; stage H: the commissioned model and its variants with
  within-episode correlated drift, bounded by commissioning data).
  - The set is **locked with the plan**: the plan lock's constants list every member by name with its parameters
    (`create_plan_lock(model_set=...)`).
  - Each member gets its theta0_D and Monte Carlo SE from **model runs**: registered runs of that member (mode
    `model`) with the confirmatory design. Every model run records its member and the parameters it ran with, and
    must end (its end entry committed) before the calibration block is registered.
  - The calibration step refuses a missing member, an extra member (a registered model run of a name outside the
    locked set), a model run whose parameters differ from its member's locked ones, and a subset of the registered
    model runs. Each SE must be at most margin / 10 (enforced by the script).
- **Model check, member by member.** A misspecified drift model (above all, drift correlated within an episode that
  an adaptive decision process can exploit) would show up as a decision effect. The model runs therefore also
  predict the within-episode correlation wcorr of sealed-control pairs (the mean autocorrelation at lags 1-3 of the
  per-step difference of two sealed replays of the same sequence), as a two-sided prediction range per member
  (0.005).
  - A member is **consistent** when both the calibration block's (C2, C3) pairs and the confirmatory run's
    (A3, A3b) pairs lie inside its range. Members that are not consistent are dropped: their theta0_D comes from a
    drift model the sealed controls refute.
  - If no member is consistent, H_decision (and the decision cells of the secondary family) is **inconclusive**
    ("the drift model does not reproduce the sealed controls") and no claim is made either way.
- **Worst case.** H_decision counts only if it rejects at every consistent member's theta0_D (the reported p is the
  largest), and it is equivalent only if it is equivalent at every consistent member's theta0_D.
- **What the model check cannot do.** It tests the drift, not the selection effect. A member whose drift matches
  the controls but whose selection effect is wrong passes it. With the right member also in the set the worst case
  still holds the level (stage S, condition drift_wrong_member); a set that holds only such a member does not
  (condition drift_only_wrong_member, reported without a criterion). The set must therefore contain the true
  selection effect, which is why it is locked with the plan and reviewed before sign-off.
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

The calibration run and all model runs are registered before their first measurement, and each registers its end
(the sha256 of its sealed artifact files and its acquisition flags) before its sealing key is released (section
7). The locked script (`calibrate`) then checks every run against the plan lock and the registry, rebuilds every
episode table from the registered sealed artifacts with the locked reduction, computes the following, writes the
calibration file and a **calibration lock** (the calibration's sha256, the plan lock's sha256, the calibration run,
its artifact hashes and data sha256, every model run and its data sha256), and appends a `calibration` event to the
registry. All are committed before the confirmatory run is registered, and the confirmatory run records the
calibration lock it will be analysed against.

1. **The empirical null of every test.**
   - The null distribution of a cell's mean over n pairs is a circular moving-block bootstrap of the calibration
     series of the same geometry and statistic, with a random sign per bootstrap block.
   - Block length: the larger of floor(4 (M / 100)^(2/9)) (the Newey-West rule) and twice the series' integrated
     correlation time, at most M / 4. The correlation time is tau = 1 + 2 sum_k lambda(k / B) rho(k) with the
     flat-top window lambda and the bandwidth B = 2 m* of Politis and White (2004) (m*: the smallest lag after which
     max(5, ceil(sqrt(log10 M))) consecutive autocorrelations are below 2 sqrt(log10 M / M) in size). Drift that
     correlates the pairs over several blocks lengthens the blocks; without it the rule is the Newey-West one.
   - The spread and shape come from the controls. The centre is zero because the controls are exchangeable (random
     order inside blocks).
   - Two-sided p = (1 + #{|null| >= |d|}) / (B + 1), d the cell's centred estimate.
2. **The margin (smallest effect size of interest)**, per geometry, fixed rule:

       Delta = Q_{1 - alpha/2}(|D|) + Q_{0.95}(|D|),

   where D is the empirical control-vs-control null of the mean of N = 400 pairs (item 1) and the quantiles are of
   |D|. For a symmetric null, Q_{1 - alpha/2}(|D|) is the critical value of the test at Holm's first level (alpha / 2,
   two-sided) and Q_{0.95}(|D|) = Q_{0.975}(D), so an effect of Delta is detected against the controls with power
   0.975 (MARGIN_POWER) at the planned N. For roughly normal controls Delta = 4.2 x the control-vs-control SD of the
   N-pair mean.
   - Meaning: the smallest effect that this experiment, at its fixed size, separates from the control-vs-control
     variability with high power. Smaller effects are not "absent": they are below the resolution the plan is
     built for, and the plan makes no claim about them.
   - Why MARGIN_POWER = 0.975 = 1 - alpha_eq: the positive test and the equivalence test are then dual at the
     margin. An effect of exactly Delta escapes the positive test, and a zero effect fails the equivalence test,
     each with probability about alpha_eq = 0.025. Version 2 used 0.95: its null claims (equivalence, a stable null
     and the model check together) came out at 0.877 against the 0.90 target, because TOST alone was at 0.93. The
     complete positive rule (with the concurrent null and the model check) is validated to keep power >= 0.90 at
     Delta.
   - Why not the earlier rule (the 0.95 quantile of a single control pair, MARGIN_PAIRS = 1): that is about
     2 sigma of a single pair, 20 SE of the cell mean and about 40 % of the spin signal C_ref; the power rule never
     bound, equivalence held in every null run, and an effect of a third of the spin signal would have been called
     "practically null".
   - Delta is also reported in units of the spin signal of the replayed sequences, C_ref = the mean over
     calibration blocks of (r of C5 - r of C2), next to every equivalence claim.
3. **N = 400 valid confirmatory blocks, fixed.** The margin is derived for this N; there is no data-dependent sample
   size. The owner set N = 400 on 2026-11-27 (was 100), before any confirmatory data, so that the margin falls below
   5 % of the spin signal: Delta falls as 1 / sqrt(N), and stage S gives Delta / C_ref of about 0.043 at N = 400
   (0.086 at N = 100; section 13, deviations log).
   - M_CAL = 200 is not scaled with N. The calibration block estimates the spread (and shape) of one control pair;
     the null of the N-pair mean is the block bootstrap of the M_CAL series, so the relative precision of the
     critical values and of the margin (about 1 / sqrt(2 M_CAL), 5 %) depends on M_CAL and not on N. Stage S
     checks this at N = 400 (the decisive criterion, coverage and power, section 12).
   - The model runs per member are not scaled: the Monte Carlo SE of theta0_D and the margin both fall as
     1 / sqrt(N), so the rule SE <= margin / 10 holds with the same number of runs.
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

One family of four cells, Holm at FWER 0.05, under the complete rule of the primaries: a secondary rejects only
when Holm rejects against the calibration null *and* Holm rejects against the concurrent null (the run's own sealed
controls of the same geometry and statistic: (A3, A3b) at lag 0, (A3 in block k, A3L in block k + 1) at lag 1).
The decision cells are centred at every consistent member's theta0_D *of their own statistic* (the model runs
predict the selection effect of every statistic of the decision cell, not only of r; the worst case is taken), and
they are refused, like H_decision, when no member is consistent. The observer cells are centred at zero. No claims
come from the secondaries if the primaries are null.

- **S2, dispersion**: the log ratio of the residual variances (r - r_model(action)), A1 vs A2 (lag 1) and A2 vs A3
  (lag 0). r_model is the frozen model's prediction (stage S: the simulator's noiseless means).
- **S3, spin specificity**: the primary statistics restricted to steps whose applied frequency is more than
  3 x (Rabi frequency at the pulse amplitude + T2* FWHM) from every line, A1 vs A2 (lag 1) and A2 vs A3 (lag 0). A
  spin-coupled effect must vanish there.

Outside the family:

- **Manipulation check (S4, quantum test)**: the task error of A5 against A2, one-sided at 0.05 (the detuned arm
  must lose the task). It is a check of the setup, not a hypothesis; a failure is reported with the primaries.
- **S1, sham vs yoked, exploratory**: the mean model residual of A4 against A2 in the same block. The two arms apply
  different action sequences, so their residuals differ wherever the frozen model is imperfect, and no sealed
  control of the design has the same structure (version 2 had it in the family, which was liberal: stage S 0.075
  for S1-S3). It is reported with the exploratory cells.

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
- **Sealed artifacts**: the acquisition code writes every episode as it finishes to the run's artifact file (JSON
  lines: the code or token, block, time slots, applied actions, the sealed outcome records, the sealed setting
  record, the sealed meta record; `arms.ArtifactWriter`). The episode's synthesiser offset (the detuned arm's
  400 MHz) is sealed with the sealing key, because in the clear it would name the detuned arm during the run.
- **Order of steps after a run**:
  1. Register the end of acquisition (`Registry.register_end`): the sha256 of every sealed artifact file and the
     acquisition flags per block (section 8). Commit and push it.
  2. Only then release the sealing key. The locked script never reads an episode table from the run directory: every
     step (`calibrate`, `analyse`, `unblind`, and the model-run aggregation) rebuilds the table with the locked
     reduction (`build_episode_table`: r, r_model, off-line flags, task error, sequence hashes and the block
     exclusions of section 8) from exactly the registered files, with the released sealing key (checked against
     its commitment; every release is logged) and the frozen model executed from its locked file. It refuses when a
     file's sha256 differs from the registered one or the run has no single committed end entry. The table carries
     arm codes only.
  3. Run the locked script (`analyse`) on all valid data. It refuses unless the plan lock, the calibration lock, the
     calibration run and the confirmatory run all check out (README, rules 2-5). Its output records the run, the
     plan lock, the calibration lock and the script sha256. Commit it and the registry entry of its sha256, with the
     deviations log.
  4. Only then release the blinding key, run the locked `unblind` step (it refuses an output whose sha256 is not the
     registered one), and read off the pre-registered cells.
- **What blinding protects.** An analyst can guess A5 (its large detuning gap) and the replay structure. Blinding
  guards the decisions that could still be influenced (exclusions, deviations, failure handling), all made before
  unblinding. Unblinding after the analysis is a commit, so its time is auditable.
- **Locking**: see README, "Freezing". The analysis refuses to start when any locked file differs or the lock is not
  held by its pushed tag. It refuses runs whose record is missing, belongs to another lock, mode or calibration,
  carries a planted effect outside simulation, came from a dirty git tree or from a commit that does not hold the
  locked files, or is not registered.
- **Registry**: every model, calibration and confirmatory run is registered before its first measurement, and every
  registered run under the lock is reported (completed or aborted) with the commit of each of its events. One
  calibration block and one confirmatory run per series, two series (section 9).
- **Order is checked by git ancestry**, not by local clocks. The registry's committed history must only append
  whole entries (no commit may change or extend an earlier entry), and the commit that added each entry is its
  evidence of time: a run's start entry must be on a commit that descends from the commit the run started from;
  its end entry on a later commit; every model run's end entry before the calibration block's start entry; the
  calibration block's end entry before the calibration event; the calibration event before the confirmatory run's
  start entry; the confirmatory run's end entry before its blinded entry. The commits pushed to the public
  repository at each registry event are the timestamps (README, rule 3); the `utc` fields and `data_started_utc`
  are informative only.

## 8. Sample size, stopping, exclusions

- **Fixed sample size, no interim looks.**
  - N = 400 valid confirmatory blocks, M_CAL = 200 valid calibration blocks, both fixed (N = 400 by the owner's
    decision of 2026-11-27, before any confirmatory data; section 5, item 3).
  - Collection stops at N valid blocks or at 1.2 N attempted blocks, whichever comes first. The calibration block
    stops at M_CAL valid or 1.2 M_CAL attempted blocks. The warm-up blocks come on top of the cap (cap = 2 +
    ceil(1.2 N): 482 for the confirmatory run, 242 for the calibration block).
  - Stopping depends only on validity flags, computed blind to arm and outcome: the warm-up blocks, the
    acquisition flags and complete blocks. The outcome-based exclusions below (reference floor, non-finite values)
    are not visible during acquisition and do not count for stopping.
  - The end entry records the number of attempted blocks (required). The script enforces the rule against it
    (`sample_size`): it refuses a run with episodes of blocks beyond the attempted ones, more attempts than the
    cap, more valid blocks than planned, blocks attempted after the N-th (M_CAL-th) valid one, or a stop before
    both N valid blocks and the cap; the calibration's sample is frozen in its lock.
  - Peeking is impossible by construction: the outcomes and the arm map are sealed.
- **Shortfall**: if the cap is reached with fewer valid blocks, or outcome-based exclusions leave fewer than N
  valid blocks, the analysis runs on what exists and the shortfall is a deviation (recorded in the blinded output
  and reported).
- **Exclusion of a block as a whole** (all its arms; automated, arm-blind). The acquisition flags (interlock trip,
  counter overflow or saturation, sequencer underrun or loss of sync) are set by the acquisition code blind to
  outcomes and are part of the registered end entry; the other reasons are computed by the locked reduction. No
  exclusion can be added after the end entry is committed:
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

- **Two series, fixed**: series S1 is the original and series S2 the replication. The script refuses any other
  series name, and each series has exactly one calibration block and one confirmatory run (a second registered one
  makes the series unusable, as for calibration blocks).
- A **replication** is the series S2 under the same lock, with:
  - new keys and a new order seed;
  - a new calibration block;
  - a start at least a week after the original;
  - in stage H, ideally a second diamond or a second chip, started by a different person.
- A positive claim needs a positive claim (section 5, decision rule) in the same direction in S1 and in S2. A null
  claim needs a null claim in both; an unstable run counts as neither (`series_claim`).
- A series that is aborted, never completed or unusable counts against every claim: without a completed analysis of
  both series there is no claim. No third series can be added under the lock, so there is no choice of two agreeing
  series out of several.
- A replication series registers its own model runs (if the model set is refitted, that needs a new lock),
  calibration block and confirmatory run; every registered series is reported.
- Everything else (any other cell of the matrix, any post-hoc pattern) is **exploratory** and labelled so. It
  becomes a claim only after a new pre-registration and a successful confirmatory run.

## 10. Milestone M7 and E5 (simulation stage)

**M7, three controllers**: (a) ideal clocked, (b) drifting chip-timed, (c) (b) plus event-driven feedback (the next
pulse conditioned on the previous window's photons).

The analog readout and drive work (ADC and DAC) is a separate milestone, M8, by the owner's decision of 2026-11-27.
It is not part of M7 and not covered by this plan; if it adds a decision or observer arm, it gets its own plan.

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
complete experiment with its own keys, order tables and calibration block (M_CAL = 200), and a confirmatory run of
the plan's N = 400 blocks, analysed with the frozen model set of its condition (model runs per model and per true
field). A change of N or M_CAL needs this validation again at the full size.

Every criterion is a rule on the Clopper-Pearson 95 % interval of a rate, and the report prints PASS, FAIL or
PENDING for each.

- **The decisive criterion**: the nominal primary family (any positive claim under the complete rule) passes when
  the interval's **upper bound is at most 0.06** at the documented full size (R = 4000 nominal null replicates,
  1000 per stress condition). This rule can show that the test is calibrated: a true rate of 0.05 gives an upper
  bound of about 0.057 at R = 4000. Below the full size it is PENDING, and the plan is not signed off until the
  full-size run has passed it.
- **Descriptive error rates** pass when their interval is **not above** their level (lower bound <= level). This rule
  only fails a test proven liberal (at R = 300 a true rate of 0.07 still passes); it is kept for the checks that are
  not decisive.
- Coverage, power and claim rates pass when their interval is **not below** their target (upper bound >= target).

- **Null, nominal condition** (the device is the model):
  - the primary family (any positive claim under the complete rule): decisive, and descriptive (not above 0.05);
  - each primary's uncorrected rate p <= 0.05 against the calibration null: not above 0.05;
  - the secondary family S2-S3 (any rejection under the complete rule): not above 0.05;
  - the manipulation check S4 passes: not below 0.95;
  - each primary's 95 % interval covers zero, over the runs that report an interval (H_decision without a
    consistent member is inconclusive and reports none; that rate is the model check's): not below 0.95;
  - each primary's null claim: not below 0.90;
  - the null-stability flag rate: not above 0.05;
  - arm names absent from every blinded output: in every replicate.
- **Null, stress conditions** (primary family, any positive claim: not above 0.05 in each):
  - drift_covered: within-episode contrast drift (AR(1), sd 0.3, correlation 0.9 per step), in the model set
    (theta0_D clearly non-zero);
  - drift_unmodelled: the same drift, missing from the model set (the model check must refuse H_decision);
  - nonstationary: confirmatory drift 3x larger and 3x faster than in the calibration block;
  - drift_wrong_member: the same drift; the set holds the right member and a member whose drift matches but whose
    decision slot replays offline sequences (theta0_D = 0: a wcorr-matched member with a wrong selection effect).
- **Reported without a criterion**: drift_only_wrong_member (the set holds that wrong member and the nominal model,
  which the controls refute): the limit of the model check (section 4).
- **Null claims under nonstationarity**: in the nonstationary condition, an effect of exactly the replicate's
  frozen margin planted in the decision arm or in every observed arm; the rate of (false) null claims: not above
  alpha_eq = 0.025 for each primary.
- **Positive control**: an effect of exactly the replicate's frozen margin, planted in the decision arm (H_decision)
  or in every observed arm (H_observer). Power of the complete rule, attribution (the planted hypothesis claimed,
  the other not) and non-equivalence: each not below 0.90. Below the margin only the power curve is reported.
- **Engineering test** (`paired_margin_test` at ENG_N = ENG_N_CONTROL = 20): rejection rate not above 0.05 under
  the null (normal and t3 diffs, two- and one-sided); power at the control-derived margin not below 0.90.
- Results: `docs/results/prereg-calibration.md`.

## 13. Open for the owner before sign-off

Stage-S numbers below are from docs/results/prereg-calibration.md, run at the documented full size with N = 400
(R = 4000 nominal null replicates, 1000 per stress condition; stand-in decision process; 4.7 h on a shared CPU).

- **The sample size** (owner's decision of 2026-11-27, section 5 item 3): N = 400 valid confirmatory blocks (was
  100), M_CAL = 200, the model runs per member (500 over 4 true fields), ENG_N = ENG_N_CONTROL = 20.
- **The margin rule** (section 5): Delta = the effect detected against the controls with power 0.975 at N = 400.
  Stage S: Delta is about 0.00136 in r, **4.3 % of the spin signal C_ref** (8.6 % at N = 100), 4.19 x the
  control-vs-control SD of the 400-pair mean. The hardware rates change it again (PREREG-002).
- **Stage-S criteria at N = 400.** Every criterion passes at the full size (the report gives each with its interval):
  - decisive: the primary family 0.039 [0.033, 0.046] (upper bound 0.046 <= 0.06);
  - the calibration null alone, uncorrected: H_decision 0.054 [0.048, 0.062], H_observer 0.047 [0.041, 0.054];
  - secondary family 0.033 [0.027, 0.038]; coverage 0.945 [0.937, 0.952] (H_decision, runs with an interval) and
    0.952 [0.945, 0.958]; null claims 0.904 [0.895, 0.913] and 0.921 [0.912, 0.929]; null-stability flags 0.055
    [0.048, 0.063];
  - power of the complete rule at the margin 0.943 [0.915, 0.963] (decision) and 0.953 [0.927, 0.971] (observer),
    attribution 0.912 and 0.917, non-equivalence 0.970 and 0.970;
  - stress conditions (primary family): drift_covered 0.048 [0.036, 0.063], drift_unmodelled 0.014, nonstationary
    0.042 [0.030, 0.056], drift_wrong_member 0.018; false null claims under nonstationarity 0.010 [0.002, 0.029]
    for each primary (300 replicates per hypothesis);
  - the engineering test (unchanged by N): 0.049, 0.051, 0.046 under the null, power 0.969 and 0.967 at its margin.
- **The model set** for theta0_D and the model check's statistic (within-episode correlation at lags 1-3). The
  member-by-member rule protects only if the set contains the true selection effect. **A larger N makes this
  matter more**: a wrong theta0_D is a bias that does not shrink with N, while the test's resolution does. With a
  wcorr-matched member whose theta0_D is wrong as the only consistent one (stage S, drift_only_wrong_member, no
  criterion) the primary family's rate is 0.125 [0.105, 0.147] at N = 400 (0.046 at N = 100), and H_decision
  against the calibration null alone 0.191 (0.095 at N = 100). H_observer does not depend on the model set (0.022).
  The owner signs the members and their parameters with the plan.
- **The decision process** of the stage-S validation (a stand-in) versus the M5 policy for the full-size run of M7.
- **What changed from version 2** (which failed four criteria) is unchanged in kind: S1 exploratory, S2 and S3 under
  the complete rule with the decision cells at their own theta0_D; MARGIN_POWER = 0.975; coverage counted over the
  runs that report an interval (H_decision without a consistent member reports none, 1.0 % of nominal runs).
- **Recorded owner decisions of 2026-11-27**: N = 400 (above); the analog ADC and DAC work is milestone M8, outside
  M7 and this plan (section 10).
