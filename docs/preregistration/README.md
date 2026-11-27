# Pre-registration

Confirmatory comparisons of **decision arms or observer arms** go through this directory. A decision arm applies an
outcome-contingent choice to the control; an observer arm has a process reading the results while the run is going
on. Any result about such arms that is not produced under a locked plan here is exploratory, and is labelled so.

The purpose: the analysis is fixed before the data exist, and every comparison is judged against control runs
that have no decision and no observer. Nobody can then say that a pattern was found in the noise.

## Plans

| Plan | Scope | Status |
| --- | --- | --- |
| [PREREG-001](PREREG-001-observer-decision.md) | decision and observer coupling; simulation stage of M7 and E5 | DRAFT (not signed off, not locked) |

Deviations from any locked plan: [deviations.md](deviations.md) (append-only).

## How it works

- **The plan** (`PREREG-<n>-*.md`): hypotheses, arms, statistics, the null, the margins (smallest effect sizes of
  interest), the sample size, the stopping rule, exclusions, multiplicity and replication. It is written in
  operational terms only.
- **The analysis script** (`glowbox/analysis/prereg_analysis_<n>.py`): the plan as code, with the frozen constants
  at its top.
- **Machinery** shared by every plan:
  - `glowbox/analysis/prereg.py`: locks, run records, sealing, blinding, the block schedule;
  - `glowbox/analysis/arms.py`: the decision, yoked-replay, sealed and sham arms, and the protocol runner, generic
    over any controller and any `run()` measurement.
- **The empirical null.** A plan's tests compare decision or observer arms against the distribution of the same
  statistic between **control runs**: arms with no decision applied and no live reader, run under the same schedule.
  The control runs that define the null and the margins are a separate **calibration block**. It is run, hashed and
  frozen before any decision or observer arm runs.
- **Stage-S validation.** Before a plan is signed off, the whole procedure is run on simulated data with no effect
  (the null calibration: false-positive rate) and with planted effects (the positive control: power). For PREREG-001
  this is `glowbox/experiments/prereg_calibration.py`, reported in `docs/results/prereg-calibration.md`. A
  simulation has no decision or observer effect by construction, so these runs validate the pipeline and are never
  evidence about the hypotheses.

## Rules

1. **No confirmatory data before the lock.** Every constant the analysis uses is frozen in the plan, in the script
   or in the calibration lock before the confirmatory run starts.
2. **The calibration block comes first.** It runs after the plan lock and before any decision or observer arm. Its
   data hash and the derived quantities (null series, margins, N, bootstrap seed) are committed before the
   confirmatory run starts.
3. **Every run records its lock.** `run.yaml` gets a `prereg` section (`prereg.prereg_record`, written with
   `RunLog.annotate`) holding:
   - the lock sha256 and its locked file hashes;
   - the mode (confirmatory, calibration or simulation);
   - the sha256 commitments of the run's keys;
   - any planted effect (allowed in simulation only).

   Confirmatory and calibration runs must come from a clean git tree.
4. **The analysis refuses to start** when any locked file changed: the plan, the script, every `glowbox` module the
   script imports (`prereg.module_files`), the configs, or the frozen part of the deviations log. It refuses runs
   whose record is missing, belongs to another lock or mode, carries a planted effect in confirmatory mode, or came
   from a dirty tree.
5. **Outcomes are sealed during the run.** Every outcome record is encrypted and authenticated by the acquisition
   code; sealed arms show the decision process and the monitors a decoy. The sealing key is released after the run
   has ended.
6. **Blinded analysis first.** The script analyses arm *codes*. Its full output and sha256 are committed before the
   blinding key is released. Unblinding is a separate locked step that only reads off pre-registered cells.
7. **Randomised, interleaved blocks.** Every block holds one episode of each arm, in a random order from the run's
   order seed. The seed's sha256 is recorded at run start, and the whole order table exists before data.
8. **Fixed sample size, no interim looks, arm-blind exclusions.** Data are never discarded because of their values.
9. **Every outcome is reported**, nulls and inconclusive results included, with the decision table of the plan.
10. **Changes need a new lock.** A changed locked file needs a new lock version and an entry in `deviations.md`
    giving the date, what changed, why, whether it was before or after unblinding, and the commit. Nothing is changed
    silently.
11. **Claims need replication.** A positive claim needs the same result in an independent replication under the same
    lock. Anything not pre-registered is exploratory.

## Freezing a plan

Done by the owner or the lead, never by an agent on its own:

1. The owner signs the final text of the plan. The lead records the sign-off with the date and the sha256 of the
   text in `deviations.md` ("sign-off" entry) and sets the plan's status line to LOCKED.
2. Create the plan lock over the plan, the analysis script and its module closure, the configs, and the deviations
   log (pinned by its current content, which may only grow), with the script's frozen constants:

       uv run python -c "from glowbox.analysis import prereg as p, prereg_analysis_001 as a; \
       files = ['docs/preregistration/PREREG-001-observer-decision.md', \
       *p.module_files('glowbox.analysis.prereg_analysis_001'), *p.module_files('glowbox.analysis.arms')]; \
       p.Lock.create(a.PLAN, files, {'analysis': a.frozen_constants()}, \
       'docs/preregistration/deviations.md').write('docs/preregistration/PREREG-001.lock.yaml')"

   Commit the lock file on its own and tag the commit (e.g. `prereg-001`). Pushing the tag to the public repository
   gives the public timestamp.
3. Run the calibration block (mode `calibration`). Freeze it with
   `python -m glowbox.analysis.prereg_analysis_001 calibrate --run <run dir> --lock <lock> --out <calibration.json>`
   and commit the output and its sha256.
4. Run the confirmatory series (mode `confirmatory`). Then:
   1. release the sealing key;
   2. run `... analyse` and commit the blinded output and its sha256;
   3. release the blinding key;
   4. run `... unblind`;
   5. commit the unblinded report.
