# Task 4 Report: End-To-End Beta-I Builder CLI

## What I Implemented

- Added `code/src/scripts/build_beta_i_weights.py`, a CLI and importable `run()` entry point that reads the four specified local inputs and regenerates `W_personalized.parquet` from a GMWI2-style health-index finite-difference method.
- The runner trains the health index from `M_clr.parquet` and classified metadata, builds nutrient perturbations from `B_nutrient_genus.parquet`, and uses `nutrient_index.json` as the output nutrient order.
- It writes the requested bundle: `W_personalized.parquet`, `health_index.joblib`, `nutrient_perturbations.parquet`, `sample_beta_diagnostics.csv`, `nutrient_perturbation_summary.csv`, and `beta_i_manifest.json`.
- The manifest records method, formula, dimensions, model training summary, estimator parameters, input paths, SHA256 checksums for generated artifacts, Git commit, Python version, and the non-causal interpretation boundary.
- The implementation never reads `W_personalized.parquet` as an input or training target.

## Tests

- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_build_beta_i_weights_script.py -q`
  - Initial TDD run: failed as expected with `ModuleNotFoundError: No module named 'scripts.build_beta_i_weights'`.
  - Final run: `1 passed in 0.62s`.
- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_estimator.py code/src/tests/test_beta_i_perturbation.py code/src/tests/test_beta_i_health_index.py -q`
  - `15 passed in 0.47s`.
- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_article_scoring.py -q`
  - `9 passed in 1.58s`.
- `git diff --check -- code/src/scripts/build_beta_i_weights.py code/src/tests/test_build_beta_i_weights_script.py`
  - Passed with no output.

## TDD Evidence

The focused test was added before the CLI module existed and failed at collection with the expected missing-module error. After the minimal runner implementation, the same test passed.

## Files Changed

- `code/src/scripts/build_beta_i_weights.py`
- `code/src/tests/test_build_beta_i_weights_script.py`
- `.superpowers/sdd/task-4-report.md`

## Self-Review Findings

No findings. The runner maintains the specified nutrient order, consumes only the designated source inputs, keeps beta_i interpretation bounded, and includes reproducibility metadata and artifact checksums.

## Issues or Concerns

None. The full production output directory was intentionally not run, per the task instruction; the temporary-repository integration test exercised the complete pipeline.

## Fixes After Review

- Added a CLR-first metadata audit before health-index fitting. Sample IDs with contradictory healthy/nonhealthy evidence are excluded rather than adjudicated, using the approved policy: "Exclude CLR-overlapping sample IDs with contradictory healthy and nonhealthy metadata evidence before deriving binary health labels or fitting the health index; do not adjudicate them to either class."
- Added health-label diagnostics to `beta_i_manifest.json`: the policy, CLR-overlapping metadata-row count, excluded contradictory-ID count, derived labeled-sample count, healthy count, and nonhealthy count.
- Added complete reproducibility configuration for the health index, nutrient perturbations, and beta estimator, including `batch_size`; linked the generated `health_index.joblib.manifest.json`; and extended artifact checksums to cover that manifest plus every non-self-referential bundle artifact.
- Extended the temporary-repository integration test with contradictory and out-of-CLR metadata, summary-output existence, byte-level checksum validation, full configuration assertions, the health-model-manifest link, and the non-causal interpretation boundary.

## Verification After Review

- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_build_beta_i_weights_script.py -q`
  - `1 passed in 0.67s`
- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_estimator.py code/src/tests/test_beta_i_perturbation.py code/src/tests/test_beta_i_health_index.py -q`
  - `15 passed in 0.64s`
- `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_article_scoring.py -q`
  - `9 passed in 1.74s`
- `git diff --check -- code/src/scripts/build_beta_i_weights.py code/src/tests/test_build_beta_i_weights_script.py`
  - Passed with no output.
