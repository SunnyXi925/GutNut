# Task 7 Report

## Summary

Added submission-readiness aggregation to the personalized calibration runner. It writes CSV and JSON reports with all five blocking reference targets and only accepts submission readiness when each target passes. The official GMWI2 external balanced-accuracy target is explicitly recorded as missing and failing because the current Section 2 outputs do not compute that benchmark.

## Changed Files

- `code/src/scripts/run_personalized_calibration_experiments.py`
- `code/src/tests/test_personalized_calibration_runner.py`

## Tests

- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_reference_targets.py code/src/tests/test_calibration_objective.py code/src/tests/test_microbiome_health.py code/src/tests/test_signed_paths.py code/src/tests/test_adjudicator_eval.py code/src/tests/test_response_benchmark.py code/src/tests/test_fcs_style_supplement.py -q` - 57 passed
- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests -q` - 123 passed
- `PYTHONPATH=code/src .venv/bin/python -c "import importlib.util; spec=importlib.util.spec_from_file_location('runner','code/src/scripts/run_personalized_calibration_experiments.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('ok')"` - ok

## Hygiene

- `git ls-files --others --exclude-standard outputs | sed -n '1,20p'` printed nothing.
- Staged-artifact guard exited 0; no prohibited generated artifacts were staged.

## Concern

The runner does not currently calculate official GMWI2 external balanced accuracy. The readiness gate deliberately emits a missing/NaN failed target and keeps `ready_for_submission` false until a valid official external evaluation is added.

## Readiness Provenance Fix

The submission-readiness response target now reads only the FDR pass fraction derived from an explicit `--response-outcome-list`. Without that list, the target is recorded as missing/NaN and fails safely. The existing validation-only added-value outputs remain unchanged.

## Fix Verification Outputs

```text
$ PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_personalized_calibration_runner.py code/src/tests/test_response_benchmark.py code/src/tests/test_reference_targets.py -q
.................                                                        [100%]
17 passed in 0.42s
```

```text
$ PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests -q
........................................................................ [ 57%]
.....................................................                    [100%]
125 passed in 5.25s
```

```text
$ PYTHONPATH=code/src .venv/bin/python -c "import importlib.util; spec=importlib.util.spec_from_file_location('runner','code/src/scripts/run_personalized_calibration_experiments.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('ok')"
ok
```
