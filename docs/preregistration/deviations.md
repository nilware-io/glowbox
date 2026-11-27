# Deviations log

Append-only. A lock pins this file's content at freeze time: entries may only be added below. Each entry gives the
date, the plan and lock version, what changed, why, whether it was before or after unblinding, and the commit.
Sign-offs of plans are recorded here too, with the date and the sha256 of the signed text.

## PREREG-001

- 2026-11-27: draft written; not signed off, not locked. No confirmatory data exist.
- 2026-11-27: version 3 draft (second review: analysed data bound to the registered sealed artifacts, one
  confirmatory run per series and two fixed series, locked model set with a member-by-member model check, secondary
  family under the complete rule, MARGIN_POWER = 1 - ALPHA_EQ, decisive pass criterion at the full size, order
  checked by git ancestry); not signed off, not locked. No confirmatory data exist.
- 2026-11-27: version 3 draft, sample size: N = 400 valid confirmatory blocks (was 100), by the owner's decision of
  2026-11-27 ("if reasonable, use more runs, e.g. 400, so that the margin falls below the criterion"). Made before
  any confirmatory data and before sign-off and locking, so it is a change of the draft, not of a locked plan. The
  control-derived margin falls as 1 / sqrt(N): from about 8.6 % to about 4.3 % of the spin signal C_ref. M_CAL = 200
  and the model runs per member are unchanged (PREREG-001 section 5, item 3). The stage-S validation was rerun at the
  full size with N = 400 (docs/results/prereg-calibration.md). Not signed off, not locked.
