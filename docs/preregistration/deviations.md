# Deviations log

Append-only. A lock pins this file's content at freeze time: entries may only be added below. Each entry gives the
date, the plan and lock version, what changed, why, whether it was before or after unblinding, and the commit.
Sign-offs of plans are recorded here too, with the date and the sha256 of the signed text.

## PREREG-001

- 2026-11-27: draft written; not signed off, not locked. No confirmatory data exist. Commit 3c51b07.
- 2026-11-27: version 3 draft (second review: analysed data bound to the registered sealed artifacts, one
  confirmatory run per series and two fixed series, locked model set with a member-by-member model check, secondary
  family under the complete rule, MARGIN_POWER = 1 - ALPHA_EQ, decisive pass criterion at the full size, order
  checked by git ancestry); not signed off, not locked. No confirmatory data exist. Commit 0707b24.
- 2026-11-27: version 3 draft, sample size: N = 400 valid confirmatory blocks (was 100), by the owner's decision of
  2026-11-27 ("if reasonable, use more runs, e.g. 400, so that the margin falls below the criterion"). Made before
  any confirmatory data and before sign-off and locking, so it is a change of the draft, not of a locked plan. The
  control-derived margin falls as 1 / sqrt(N): from about 8.6 % to about 4.3 % of the spin signal C_ref. M_CAL = 200
  and the model runs per member are unchanged (PREREG-001 section 5, item 3). The stage-S validation was rerun at the
  full size with N = 400 (docs/results/prereg-calibration.md). The change has a measured cost: a wrong model-set
  member's bias does not shrink with N, so in the stage-S condition drift_only_wrong_member (no criterion) the
  primary family's false-positive rate went from 0.046 to 0.125 [0.105, 0.147], and H_decision against the
  calibration null alone from 0.095 to 0.191 (PREREG-001 section 13); every criterion still passes. Not signed off,
  not locked. Commits a04141b (plan) and 2743ea0 (validation).
- 2026-11-27: version 3 draft, stopping rule enforced by the script (review finding): the end entry's number of
  attempted blocks is required, and `sample_size` refuses a calibration block or confirmatory run of the wrong size
  (more valid blocks than planned, more attempts than the cap, blocks after the N-th valid one, a stop before both);
  section 8 now states that the warm-up blocks come on top of the cap and that the outcome-based exclusions do not
  count for stopping (a shortfall instead). Before any data, sign-off and locking. Commit 727a7cf.
