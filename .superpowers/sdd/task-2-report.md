## Task 2 Report

- Implemented bounded personalized offset calibration and food-group direction pass-fraction validation.
- Added the specified calibration objective tests.
- Exported the new scoring interfaces from `gmnps.scoring`.
- Runner wiring was not touched by assignment, per the task request; `run_personalized_calibration_experiments.py` remains for the main controller.

## Review Fix

- Added and exported `calibration_objective`, including population rank agreement, population shift, food-group direction pass fraction, and the weighted group-direction penalty.
- Added regression tests proving that food groups affect the objective and that increasing `group_penalty` worsens an objective with imperfect group-direction compliance.
- Renamed the misleading rank-change test to describe its directional-change assertions.

Test output:

```text
.......                                                                  [100%]
7 passed in 0.50s
```

## Re-review Fix

- Accepted food-group labels as either a Series or a DataFrame with required
  `food_group` and optional `food_subgroup`; group and subgroup direction
  checks now contribute independently to the objective.
- Added `mean_individual_rank_shift` to reward genuine per-person reranking,
  preventing near-zero offsets from being preferred when population consensus
  is preserved.
- Added strict missing-column and finite-score validation for raw offsets and
  personalized scores, with regression coverage for subgroup cancellation and
  near-zero offsets.

Test output:

```text
............                                                             [100%]
12 passed in 0.44s
```

## Second Fix

- Made `expected_group_direction` and `expected_subgroup_direction` authoritative
  metadata fields, with `expected_direction` as a shared fallback.
- Missing direction metadata now fails by default; the existing name-based
  mapping is retained only as the explicitly named, versioned
  `allow_legacy_name_direction_policy=True` compatibility mode. Invalid
  directions always fail fast.
- Added default 90% group/subgroup consensus gates and a 100-point failure
  penalty, plus returned gate status and penalty fields from the objective.
- Rejected zero-row, zero-column, and zero-aligned-column offset matrices.
- Added regression coverage for explicit-direction naming quirks, fail-fast
  direction validation, empty offsets, and consensus failures with larger rank
  shifts.

Test output:

```text
..................                                                       [100%]
18 passed in 0.53s
```
