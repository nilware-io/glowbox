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
