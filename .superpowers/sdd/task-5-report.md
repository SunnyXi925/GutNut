# Task 5: Full Local Smoke Run And Output Audit

Date: 2026-08-03

## Build

Command:

```bash
PYTHONPATH=code/src .venv/bin/python code/src/scripts/build_beta_i_weights.py \
  --root /Users/fengxi.25/Desktop/GMNPS \
  --output-dir /Users/fengxi.25/Desktop/GMNPS/data/project_data/predict_multi/L7_nutrient_bridge_beta_i
```

Exact command output: no stdout or stderr. Exit code: 0.

## Artifact Audit

Command:

```bash
PYTHONPATH=code/src .venv/bin/python - <<'PY'
import json
import pandas as pd
root = "/Users/fengxi.25/Desktop/GMNPS/data/project_data/predict_multi/L7_nutrient_bridge_beta_i"
w = pd.read_parquet(f"{root}/W_personalized.parquet")
manifest = json.load(open(f"{root}/beta_i_manifest.json"))
print(w.shape)
print(w.index[:3].tolist())
print(w.columns[:5].tolist())
print(manifest["method"])
print(manifest["health_model_training_summary"])
assert w.shape[1] == 65
assert manifest["n_nutrients"] == 65
assert manifest["method"] == "GMWI2-style health-index finite-difference beta_i"
assert manifest["health_model_training_summary"]["n_nonzero_coefficients"] > 0
PY
```

Exact command output:

```text
(15492, 65)
['SID31004', 'SID31030', 'SID31137']
['Energy (kcal)', 'Protein (g)', 'Carbohydrate (g)', 'Sugars, total (g)', 'Fiber, total dietary (g)']
GMWI2-style health-index finite-difference beta_i
{'n_samples': 15350, 'n_healthy': 11717, 'n_nonhealthy': 3633, 'n_genera': 115, 'n_nonzero_coefficients': 114, 'train_auc': 0.8473103922229026}
```

All assertions passed. The manifest method is `GMWI2-style health-index finite-difference beta_i`; beta_i is a microbiome-derived nutrient-response calibration weight, not a validated causal nutrient effect. The manifest records `health_label_diagnostics.n_excluded_contradictory_sample_ids: 142`.

## Tests

Command:

```bash
PYTHONPATH=code/src .venv/bin/python -m pytest \
  code/src/tests/test_beta_i_health_index.py \
  code/src/tests/test_beta_i_perturbation.py \
  code/src/tests/test_beta_i_estimator.py \
  code/src/tests/test_build_beta_i_weights_script.py \
  -q
```

Exact command output:

```text
................                                                         [100%]
16 passed in 0.39s
```

## Generated Files

- `W_personalized.parquet` (4.9M)
- `beta_i_manifest.json` (2.7K)
- `health_index.joblib` (15K)
- `health_index.joblib.manifest.json` (448B)
- `nutrient_perturbation_summary.csv` (3.7K)
- `nutrient_perturbations.parquet` (84K)
- `sample_beta_diagnostics.csv` (1.4M)

## Bundle Size And Commit Decision

Command:

```bash
du -sh data/project_data/predict_multi/L7_nutrient_bridge_beta_i
```

Exact command output:

```text
6.4M    data/project_data/predict_multi/L7_nutrient_bridge_beta_i
```

The 6.4M reproducibility bundle is acceptable for normal repository practice. Although `data/` is ignored by `.gitignore`, the seven generated files and this report are intentionally force-staged and committed. No unrelated dirty or untracked worktree files are staged. There are no uncommitted beta_i artifact paths.
