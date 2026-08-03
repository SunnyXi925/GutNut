# Task 1: Health Index Core Report

## Status

Completed and committed as `9cbc4be` (`feat: add gmwi2-style health index`).

## Changes

- Added `HealthIndexConfig` and immutable `HealthIndexModel` dataclasses.
- Added binary health-label derivation from `sample_id`, `phenotype_label`, and `disease` metadata.
- Added CLR standardization, health-index fitting, probability scoring, and model persistence APIs.
- Uses L1 logistic regression through scikit-learn when available.
- Uses a deterministic NumPy logistic-regression fallback when scikit-learn/SciPy are unavailable.
- Uses joblib persistence when available and pickle persistence otherwise.
- Added the required focused tests for label derivation, healthy-vs-nonhealthy scoring, score bounds, model fields, and round-trip persistence.

## TDD and Verification

- Initial focused command: failed during collection because `code/src/tests/test_beta_i_health_index.py` did not yet exist (`no tests ran`).
- Focused suite after implementation: `2 passed`.
- Existing article suite: `9 passed`.
- `git diff --cached --check`: passed before commit.
- Environment confirmed NumPy/pandas available; scikit-learn, SciPy, and joblib unavailable, so the fallback paths were exercised.

## Scope

Only the three Task 1 files were staged and committed. Existing unrelated worktree changes were left untouched and unstaged.

## Fixes After Review

- Replaced the no-scikit-learn L2 fallback with deterministic proximal-gradient L1 logistic training using soft-thresholding, and added a direct sparsity test.
- Updated `save_health_index` to write a sibling `*.manifest.json` containing method, config, training summary, and the SHA256 digest of the persisted model bytes.
- Changed metadata derivation to exclude unknown, missing, empty, and non-classified phenotype/disease values; contradictory healthy/nonhealthy metadata now raises `ValueError`.
- Added tests covering unknown metadata, contradictory metadata, fallback sparsity, and manifest integrity.

Exact verification outputs:

- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_health_index.py -q`: `4 passed in 0.28s`
- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_article_scoring.py -q`: `9 passed in 1.39s`
- `git diff --check -- code/src/gmnps/beta_i/__init__.py code/src/gmnps/beta_i/health_index.py code/src/tests/test_beta_i_health_index.py`: passed (no output)

## Second Review Fixes

- Corrected the NumPy fallback L1 penalty to `1 / (C * n_samples)` for the mean logistic-loss objective.
- Added pre-deduplication validation for conflicting labels on duplicate `sample_id` records.
- Excluded null, blank, and literal `nan` sample IDs instead of converting them into usable labels.
- Added default-config fallback coverage for nonzero coefficients and higher healthy-sample scores.

Exact verification outputs:

- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_health_index.py -q`: `6 passed in 0.52s`
- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_article_scoring.py -q`: `9 passed in 1.59s`
- `git diff --check -- code/src/gmnps/beta_i/__init__.py code/src/gmnps/beta_i/health_index.py code/src/tests/test_beta_i_health_index.py`: passed (no output)

## Third Review Fixes

- Normalized CLR feature column names to strings during both health-index fitting and scoring, keeping model statistics, coefficients, and feature alignment consistent for non-string columns.
- Added an integer-feature-column regression test covering fit, score, canonical string genus names, score index preservation, and probability bounds.

Exact verification outputs:

- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_health_index.py -q`: `7 passed in 0.74s`
- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_article_scoring.py -q`: `9 passed in 2.20s`
- `git diff --check -- code/src/gmnps/beta_i/health_index.py code/src/tests/test_beta_i_health_index.py`: passed (no output)

## Fourth Review Fixes

- Fixed the NumPy fallback Lipschitz bound to use the augmented design matrix, including the unregularized intercept column. This keeps the learning rate finite for constant or zero-valued standardized features.
- Added a forced-fallback regression test with two degenerate feature columns and imbalanced labels `[1, 1, 1, 1, 1, 0]`, verifying the fitted probability is approximately the observed healthy prevalence (`5/6`).

Exact verification outputs:

- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_health_index.py -q`: `8 passed in 0.62s`
- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_article_scoring.py -q`: `9 passed in 1.39s`
- `git diff --check -- code/src/gmnps/beta_i/health_index.py code/src/tests/test_beta_i_health_index.py`: passed (no output)
