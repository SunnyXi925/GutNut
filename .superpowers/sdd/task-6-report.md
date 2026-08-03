# Task 6 Report: Provenance Audit Integration

Status: DONE

## Scope completed

- Added `W_personalized_beta_i` to the input-data provenance audit.
- Named the source `GMWI2-style health-index finite-difference beta_i`.
- Kept the claim bounded: beta_i is a finite-difference improvement in a fitted gut microbiome health index after a nutrient-linked perturbation, not a validated causal nutrient effect.
- Added the focused provenance regression test.
- Rebuilt `outputs/provenance_audit` from the real Task 5 bundle.

## Test-first result

Before the audit row was added, the new test failed as expected with:

```text
E   KeyError: 'W_personalized_beta_i'
FAILED code/src/tests/test_beta_i_provenance.py::test_provenance_prefers_reproducible_beta_i_bundle
1 failed in 0.65s
```

## Verification commands and exact outputs

### Focused provenance test

Command:

```bash
PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_provenance.py -q
```

Output:

```text
.                                                                        [100%]
1 passed in 0.52s
```

### beta_i regression tests

Command:

```bash
PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_build_beta_i_weights_script.py code/src/tests/test_beta_i_health_index.py code/src/tests/test_beta_i_perturbation.py code/src/tests/test_beta_i_estimator.py -q
```

Output:

```text
................                                                         [100%]
16 passed in 0.45s
```

### Rebuild provenance audit

Command:

```bash
PYTHONPATH=code/src .venv/bin/python code/src/scripts/build_provenance_audit_tables.py --root /Users/fengxi.25/Desktop/GMNPS --output-dir /Users/fengxi.25/Desktop/GMNPS/outputs/provenance_audit
```

Output:

```text
```

Exit status: `0`

### Generated provenance row

Command:

```bash
rg -n -F 'W_personalized_beta_i' outputs/provenance_audit/input_data_provenance.csv
```

Output:

```text
9:W_personalized_beta_i,microbiome_derived_nutrient_response_weights,data/project_data/predict_multi/L7_nutrient_bridge_beta_i/W_personalized.parquet,GMWI2-style health-index finite-difference beta_i,local reproducible beta_i output,local_file_present,5145210,8123e6f280b4078da3799ee45b3d27c9c1e7b7e154ba167bb97af287cbabea1d,"beta_i,k = finite-difference improvement in fitted gut microbiome health index after nutrient-linked perturbation."
```

### Whitespace check

Command:

```bash
git diff --check -- code/src/scripts/build_provenance_audit_tables.py code/src/tests/test_beta_i_provenance.py outputs/provenance_audit
```

Output:

```text
```

Exit status: `0`

## Self-review

- The new row is placed immediately after the historical `W_personalized` row, as specified.
- The audit identifies the Task 5 reproducible bundle; it does not change historical L7-derived audit calculations or use the old `W_personalized.parquet` as a beta_i training target.
- Generated provenance files were rebuilt from the real repository bundle. `outputs/` is ignored by the repository-wide `.gitignore`, so the requested generated audit files must be force-added intentionally.
- No unrelated dirty files were changed, staged, or included in the task commit.

## Fixes After Review

- Updated `provenance_audit_report.json` to read the Task 5 `beta_i_manifest.json` when present.
- Added explicit `beta_i_method`, `beta_i_formula`, `beta_i_profiles`, `beta_i_nutrients`, `beta_i_health_training_summary`, `beta_i_health_label_diagnostics`, `beta_i_interpretation_boundary`, and `beta_i_manifest_path` fields.
- Renamed historical beta audit metrics to `historical_bridge_reconstruction_*` so they are not confused with the GMWI2-style health-index finite-difference beta_i method.
- Extended the focused test with a minimal summary-report fixture covering the method, bounded interpretation, manifest path, and historical metric naming.

### Verification commands and exact outputs

Focused provenance test:

```text
Command: PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_beta_i_provenance.py -q
..                                                                       [100%]
2 passed in 0.51s
```

beta_i regression tests:

```text
Command: PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_build_beta_i_weights_script.py code/src/tests/test_beta_i_health_index.py code/src/tests/test_beta_i_perturbation.py code/src/tests/test_beta_i_estimator.py -q
................                                                         [100%]
16 passed in 0.42s
```

Rebuild:

```text
Command: PYTHONPATH=code/src .venv/bin/python code/src/scripts/build_provenance_audit_tables.py --root /Users/fengxi.25/Desktop/GMNPS --output-dir /Users/fengxi.25/Desktop/GMNPS/outputs/provenance_audit
Output: (empty)
Exit status: 0
```

Generated report fields include:

```text
beta_i_method: GMWI2-style health-index finite-difference beta_i
beta_i_formula: beta_i,k = (H(M_i + dose * P_k) - H(M_i)) / dose
beta_i_profiles: 15492
beta_i_nutrients: 65
beta_i_interpretation_boundary: beta_i estimates health-index finite-difference response to nutrient-linked microbiome perturbations; it is not a causal nutrient effect.
beta_i_manifest_path: data/project_data/predict_multi/L7_nutrient_bridge_beta_i/beta_i_manifest.json
```

Whitespace check:

```text
Command: git diff --check -- code/src/scripts/build_provenance_audit_tables.py code/src/tests/test_beta_i_provenance.py outputs/provenance_audit
Output: (empty)
Exit status: 0
```
