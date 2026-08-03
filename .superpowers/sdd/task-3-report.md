# Task 3: Individual Beta-I Estimator Report

## What I Implemented

- Added `BetaEstimatorConfig` with configurable finite-difference dose, absolute beta clipping limit, and nutrient batch size.
- Added `compute_beta_matrix`, which calculates the microbiome-derived nutrient-response calibration weight using the specified forward finite difference:
  `beta_i,k = (H(M_i + dose * P_k) - H(M_i)) / dose`.
- The estimator aligns CLR and perturbation inputs to the health-index model genera, fills missing CLR values with model means and missing perturbation values with zero, returns a samples-by-nutrients `float32` beta matrix, and supplies per-sample baseline-health and beta summary diagnostics.
- Exported the estimator API from `gmnps.beta_i`.

## Tests And Exact Results

```text
PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_estimator.py -q
2 passed in 0.48s

PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_perturbation.py -q
2 passed in 0.29s

PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_health_index.py -q
9 passed in 0.35s

PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_article_scoring.py -q
9 passed in 1.40s

git diff --check -- code/src/gmnps/beta_i/__init__.py code/src/gmnps/beta_i/beta_estimator.py code/src/tests/test_beta_i_estimator.py
No output; passed.
```

## TDD Evidence

The new estimator test was added before the implementation. Its initial execution failed during collection with the expected error:

```text
ModuleNotFoundError: No module named 'gmnps.beta_i.beta_estimator'
```

After implementation, the focused test passed with two passing tests.

## Files Changed

- `code/src/gmnps/beta_i/__init__.py`
- `code/src/gmnps/beta_i/beta_estimator.py`
- `code/src/tests/test_beta_i_estimator.py`
- `.superpowers/sdd/task-3-report.md`

## Self-Review Findings

- The implementation uses the requested primary forward finite-difference formula.
- Alignment, null handling, score calculation, clipping, output dtypes, and diagnostics were reviewed against the task brief.
- No defects found in the scoped changes. `git diff --check` is clean.

## Issues Or Concerns

- `beta_i` remains a microbiome-derived nutrient-response calibration weight and is not presented or validated as a causal nutrient effect.
- Existing unrelated dirty and untracked validation work was left untouched and will not be staged or committed.

## Fixes After Review

- Added an exact forward finite-difference assertion against `score_health_index` in the estimator test.
- Preserved original nutrient index labels for perturbation lookup and stringified only beta output column labels, with coverage for integer nutrient labels.
- Added validation that `batch_size` is positive and removed the unused NumPy import.

## Review-Fix Test Results

```text
PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_estimator.py -q
4 passed in 1.03s

PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_perturbation.py -q
2 passed in 1.02s

PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_health_index.py -q
9 passed in 1.14s

PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_article_scoring.py -q
9 passed in 4.32s

git diff --check -- code/src/gmnps/beta_i/beta_estimator.py code/src/tests/test_beta_i_estimator.py
No output; passed.
```
