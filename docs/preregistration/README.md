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
  - `glowbox/analysis/prereg.py`: locks, run records, the run registry, sealing, blinding, the block schedule;
  - `glowbox/analysis/arms.py`: the decision, yoked-replay, sealed and sham arms, and the protocol runner, generic
    over any controller and any `run()` measurement.
- **The empirical null.** A plan's tests compare decision or observer arms against the distribution of the same
  statistic between **control runs**: arms with no decision applied and no live reader, run under the same schedule.
  The control runs that define the null and the margins are a separate **calibration block**. It is run, hashed and
  frozen before any decision or observer arm runs. A positive claim must also hold against the confirmatory run's
  own sealed controls (the concurrent null).
- **Stage-S validation.** Before a plan is signed off, the whole procedure is run on simulated data with no effect
  (the null calibration: false-positive rate) and with planted effects (the positive control: power). For PREREG-001
  this is `glowbox/experiments/prereg_calibration.py`, reported in `docs/results/prereg-calibration.md`. A
  simulation has no decision or observer effect by construction, so these runs validate the pipeline and are never
  evidence about the hypotheses.

## Rules

1. **No confirmatory data before the lock.** Every constant the analysis uses is frozen in the plan, in the script
   or in the calibration lock before the confirmatory run starts.
2. **The calibration block comes first.** It runs after the plan lock and after the model runs (runs of the frozen
   drift and timing model that predict the selection effect), and before any decision or observer arm. The locked
   script freezes it (null series, margins, model predictions, model check) into a calibration file and a
   calibration lock, which are committed and registered before the confirmatory run starts. One calibration block
   per run series.
3. **Every run is registered before its first measurement.** `prereg.Registry.register_start` appends the run id,
   mode, series, key commitments and schedule sha256 to the plan's append-only registry
   (`PREREG-<n>.registry.yaml`, pinned by the lock like the deviations log); the entry is committed and pushed
   before data collection. The analysis refuses unregistered runs, a calibration block that is not the only one of
   its series, and model runs that are not all the registered ones; the report lists every registered run under
   the lock, completed or aborted.
4. **Every run records its lock.** `run.yaml` gets a `prereg` section (`prereg.prereg_record`, written with
   `RunLog.annotate`), made at run start after the lock verifies, holding:
   - the lock sha256 and the locked files' hashes *recomputed from disk*, and the git commit;
   - the mode (confirmatory, calibration, model or simulation), the series and the schedule sha256;
   - the sha256 commitments of the run's keys, and (confirmatory) the calibration lock sha256;
   - the time of the first measurement (`prereg.mark_data_start`);
   - any planted effect (allowed in simulation only).

   Confirmatory, calibration and model runs must come from a clean git tree whose commit holds exactly the locked
   files.
5. **The analysis refuses to start** when any locked file changed: the plan, the script and its module closure
   (`prereg.plan_files`: every `glowbox` module it imports, package `__init__` files included), the arms and runner,
   the controller, measurement and frozen model of the run, the configs, data files such as the sham distribution,
   `pyproject.toml` and `uv.lock`, or the frozen parts of the deviations log and the registry. It refuses a lock
   that is not committed and held by its tag. It refuses runs whose record is missing, belongs to another lock,
   mode or calibration, carries a planted effect outside simulation, came from a dirty tree or from a commit that
   does not hold the locked files, or is not registered.
6. **Outcomes are sealed during the run.** Every outcome record is encrypted and authenticated by the acquisition
   code and labelled by the arm's code (or a random token), never its name. Sealed arms show the decision process
   and the monitors a decoy drawn from the frozen model at the applied action. What differs between observed and
   sealed arms by construction (plaintext deliveries, the controller's proposals, fallback flags, compute time) is
   sealed with the blinding key. The sealing key is released after the run has ended.
7. **Blinded analysis first.** The locked reduction builds the episode table and the arm-blind exclusion flags; the
   script analyses arm *codes*. Its full output is committed and its sha256 registered before the blinding key is
   released. Unblinding is a separate locked step that refuses an output whose sha256 is not the registered one.
8. **Randomised, interleaved blocks.** Every block holds one episode of each arm, in a random order from the run's
   order seed. The seed's sha256 is recorded at run start, and the whole order table exists before data.
9. **Fixed sample size, no interim looks, arm-blind exclusions.** Data are never discarded because of their values.
10. **Every run and every outcome is reported**, nulls, inconclusive results and aborted runs included, with the
    decision table of the plan.
11. **Changes need a new lock.** A changed locked file needs a new lock version and an entry in `deviations.md`
    giving the date, what changed, why, whether it was before or after unblinding, and the commit. Nothing is changed
    silently.
12. **Claims need replication.** A positive claim needs the same result in an independent replication under the same
    lock. Anything not pre-registered is exploratory.

## Freezing a plan

Done by the owner or the lead, never by an agent on its own:

1. Bring the branch up to date with `main` first, so that the lock's git commit is on the line `main` will hold.
2. The owner signs the final text of the plan. The lead records the sign-off with the date and the sha256 of the
   text in `deviations.md` ("sign-off" entry) and sets the plan's status line to LOCKED.
3. Create the plan lock with the locked helper. It covers the plan, the analysis script and the arms with their
   module closures, the modules a run executes (runner, controller, measurement, frozen model: pass them as
   `extra_modules`), the configs and data files, `pyproject.toml` and `uv.lock`, the frozen constants, and pins the
   deviations log and the run registry:

       uv run python -c "from glowbox.analysis import prereg_analysis_001 as a; \
       a.create_plan_lock(extra_modules=['<runner module>', '<controller module>'], \
       configs=['<config.yaml>']).write('docs/preregistration/PREREG-001.lock.yaml')"

   Commit the lock file on its own, tag the commit `prereg-001` and push the tag to the public repository: the
   pushed tag is the public timestamp. The analysis refuses a lock that the tag does not hold.
4. Model runs (mode `model`, one `model` name per member of the plan's model set) and the calibration block (mode
   `calibration`), each registered before its first measurement (rule 3). Freeze them with

       python -m glowbox.analysis.prereg_analysis_001 calibrate --run <calibration run> --lock <lock> \
           --model-runs <model>=<run dir> ... --out-dir docs/preregistration/calibration/<series>

   and commit the calibration, its calibration lock and the registry.
5. Run the confirmatory series (mode `confirmatory`, recording the calibration lock; registered first). Then:
   1. release the sealing key;
   2. run `... analyse --run <run> --calibration-run <calibration run> --lock <lock> --calibration-lock <...>
      --out <blinded.json>`, and commit the blinded output and the registry;
   3. release the blinding key;
   4. run `... unblind`;
   5. commit the unblinded report.
