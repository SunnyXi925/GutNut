# Task 2: Nutrient Perturbation Builder

## What I Implemented

- Added `NutrientPerturbationConfig` with MAC, LIPID, OTHER, bridge-threshold, and L2-scale parameters.
- Added `build_nutrient_perturbations`, which aligns bridge columns to the Task 1 health-index genera, removes sub-threshold bridge values, uses absolute health-index coefficients as microbiome-derived calibration weights, applies expert MAC/LIPID/OTHER channel weights, and returns float32 nutrient-by-genus perturbations.
- Added `summarize_perturbations`, reporting nutrient, channel, nonzero genus count, L1 norm, and L2 norm.
- Exported the new public API from `gmnps.beta_i`.
- Added focused tests for bridge sign preservation, channel attenuation, and MAC summary metadata.

The implementation treats beta_i only as a microbiome-derived nutrient-response calibration weight. It does not represent a validated causal nutrient effect.

## Tests

All required verification commands passed:

```text
PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_perturbation.py -q
2 passed in 0.51s

PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_health_index.py -q
9 passed in 0.34s

PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_article_scoring.py -q
9 passed in 1.42s

git diff --check -- code/src/gmnps/beta_i/__init__.py code/src/gmnps/beta_i/nutrient_perturbation.py code/src/tests/test_beta_i_perturbation.py
passed with no output
```

## TDD Evidence

- Added the specified focused tests before the module existed.
- Initial test execution failed as expected with `ModuleNotFoundError: No module named 'gmnps.beta_i.nutrient_perturbation'`.
- The focused test suite passed after implementation.

## Files Changed

- `code/src/gmnps/beta_i/__init__.py`
- `code/src/gmnps/beta_i/nutrient_perturbation.py`
- `code/src/tests/test_beta_i_perturbation.py`
- `.superpowers/sdd/task-2-report.md`

## Self-Review Findings

- Verified genus columns are deterministically reindexed to `HealthIndexModel.genus_names` and missing values are treated as zero.
- Verified the primary channel masks are obtained solely from `build_channel_vectors`.
- The brief's literal sample implementation conflicts with its supplied assertions in two ways: coefficient-sign multiplication reverses the asserted negative bridge sign, and normalization after channel weighting removes OTHER attenuation. The delivered implementation preserves bridge direction, uses coefficient magnitude as the calibration weight, and applies the channel multiplier after direction-vector normalization so the acceptance tests and stated channel behavior hold.
- No unrelated dirty or untracked worktree files were staged or modified.

## Issues or Concerns

- None for the implemented scope. This task intentionally creates in-memory perturbations only; later artifact-producing work remains responsible for parameter manifests and SHA256 checksums.

## Fixes After Review

- Oriented each nutrient-genus bridge by `np.sign(model.coefficients)` so negative health-index coefficients flip the corresponding genus direction.
- Applied channel weights once, before row-wise L2 normalization; every nonzero perturbation row now has norm `config.l2_norm`.
- Updated the focused tests to assert negative-coefficient sign orientation and post-normalization L2 norms.

Exact verification outputs:

```text
PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_perturbation.py -q
..                                                                       [100%]
2 passed in 0.51s

PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_health_index.py -q
.........                                                                [100%]
9 passed in 0.59s

PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_article_scoring.py -q
.........                                                                [100%]
9 passed in 1.66s

git diff --check -- code/src/gmnps/beta_i/nutrient_perturbation.py code/src/tests/test_beta_i_perturbation.py
passed with no output
```
