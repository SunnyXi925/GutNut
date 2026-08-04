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
