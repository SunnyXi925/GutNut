# Phase 2 Task 5 implementation report

## Status

**DONE_WITH_CONCERNS.** The Task 5 implementation, focused tests, related attribute-scoring tests, evidence matrix and deterministic synthetic source-data freeze are complete. The repository-wide suite retains 13 pre-existing/out-of-scope Task 3 failures caused by test split fixtures that contain fewer independent family/twin components than requested folds.

## Deliberate plan-safe divergence

The plan named `code/src/gmnps/validation/synthetic_twin.py`. That file already contained uncommitted user work based on the former final-score/centering design. It was not edited, staged or committed. The attribute-level implementation was instead created as `code/src/gmnps/validation/attribute_synthetic_twin.py`. This prevents the prior centered method from contaminating the locked attribute-recomposition stress test and preserves the user's work. `code/src/gmnps/scoring/__init__.py` was likewise not edited, staged or committed; the new module imports the locked scorer components directly from concrete modules.

The historical scorer performs per-food population centering. Because Task 5 prohibits population/per-food centering, `legacy_final_score_offset` is implemented only as a no-centering final-offset-form sensitivity comparator. It is not represented as an exact rerun of the historical centered algorithm. Original-versus-expert mask comparisons use this same no-centering legacy form; the locked attribute-level primary scorer retains the expert-revised mask.

## RED and GREEN

### RED

Command:

```text
uv run pytest -q src/tests/test_attribute_synthetic_twin.py src/tests/test_evidence_gate.py
```

Expected collection failures were observed for the absent modules:

```text
ModuleNotFoundError: No module named 'gmnps.validation.attribute_synthetic_twin'
ModuleNotFoundError: No module named 'gmnps.validation.evidence_gate'
```

### GREEN

Focused command after fixture optimization:

```text
uv run pytest -q code/src/tests/test_attribute_synthetic_twin.py code/src/tests/test_evidence_gate.py
```

Result: **13 passed in 37.02 s**. A module-scoped fixture now shares the default deterministic benchmark; the formal five-seed run remains separate.

Related locked-scoring command:

```text
uv run pytest -q code/src/tests/test_attribute_calibration.py code/src/tests/test_attribute_recomposition.py code/src/tests/test_attribute_gmnps.py code/src/tests/test_attribute_method_lock.py code/src/tests/test_method_lock_gate.py code/src/tests/test_attribute_synthetic_twin.py code/src/tests/test_evidence_gate.py
```

Result: **182 passed in 44.83 s**.

Repository-wide command:

```text
uv run pytest -q --tb=short code/src/tests
```

Result: **630 passed, 13 failed, 1 warning in 80.43 s**. The failures are restricted to `test_cohort_split.py` and `test_person_meal_benchmark.py`; all report `number of independent groups (1-3) is smaller than folds (3-5)` or the resulting `PrimaryAnalysisNotEstimable`. Those files are outside Task 5 ownership and were not modified.

## Synthetic design and inputs

The simulator defines truth before any GMNPS scoring call:

1. A universal FCS2 food-quality component.
2. Individual MAC and LIPID capacities generated from an explicit independent RNG stream.
3. Prespecified per-attribute effects from physical exposure normalization, the locked 20% attribute-range perturbation, and published Food Compass domain aggregation, including top-k membership.
4. A leave-one-attribute-out decomposition that allocates top-k interaction residuals while preserving the exact prespecified total effect.
5. Independent additive noise from a separate RNG stream.

Excluded carbohydrate, zinc, copper and vitamin A proxies are generated but do not enter truth. They are available only to the original-mask sensitivity comparator. The simulator records separate stream seeds for food quality, exposures, capacities, excluded proxies, noise, random microbiome, Sattolo derangement and bootstrap sampling.

Formal frozen configuration:

| Field | Value |
| --- | ---: |
| Independent simulation seeds | 1701, 1702, 1703, 1704, 1705 |
| Individuals per replicate | 24 |
| Foods per replicate | 12 |
| Person-food observations per replicate | 288 |
| Noise standard deviation | 0.35 |
| Bootstrap replicates per comparator and seed | 200 |
| Attribute cap modes tested | 10%, 20% primary, 30% |
| Final cap modes tested | +/-8, +/-12 primary, +/-15 |

The stress test uses no observed participant outcome, GMrepo label, ZOE rank or knowledge-graph label. No real validation directory was created or modified.

## Synthetic results

Intervals below are empirical 2.5th-97.5th percentiles across the five independent simulation seeds; `valid n = 5/5` for every row. Per-seed participant-bootstrap intervals and valid counts are retained in source data.

| Comparator | Residual RMSE (replicate interval) | Residual Spearman (replicate interval) |
| --- | ---: | ---: |
| FCS baseline | 3.541 (3.062-3.843) | 0.000 (0.000-0.000) |
| Locked attribute GMNPS | 0.352 (0.346-0.365) | 0.988 (0.985-0.990) |
| Random microbiome | 2.825 (2.518-3.253) | 0.157 (0.087-0.277) |
| Sattolo-deranged microbiome | 3.654 (3.116-4.458) | -0.054 (-0.260-0.114) |
| Expert-mask legacy offset form | 3.931 (3.446-4.546) | 0.912 (0.887-0.935) |
| Original-mask legacy offset form | 5.272 (4.835-5.625) | 0.502 (0.346-0.646) |

Locked universal-rank preservation was 0.999 (0.994-1.000). The excluded-proxy contribution fraction was exactly 0 under the expert mask and 0.749 (0.636-0.869) under the original mask. All five prespecified success criteria passed. Sattolo permutations were deterministic, complete permutations with no fixed points.

These results establish identifiability and pipeline behavior only under the stated synthetic data-generating process. They are not clinical, construct or external validation.

## Frozen outputs

All files are under `results/phase2/source-data/synthetic_identifiability_stress_test/` and carry `evidence_role=synthetic_identifiability_stress_test`, `data_class=synthetic`, seeds and payload hashes.

| File | SHA-256 |
| --- | --- |
| `synthetic_attribute_twin_config.json` | `ea51c95b7d888498b2cd73e1e904fe5af443ee6a96c5ff4b15128002c2d8ac2d` |
| `synthetic_attribute_twin_manifest.json` | `006233f6ffea3fe789aa809411adbb7a516419dc898eed7a8feb5fcadb99cce6` |
| `synthetic_attribute_twin_replicate_metrics.csv` | `250198dac1de6ac740cf7ba71805af86b4f5919fcf92ba3c75feb28bb60d2084` |
| `synthetic_attribute_twin_success_checks.csv` | `48ffe7f1c4ad551bb92c0a9c3ca9b09a77d830831bbf8ad9c6ae69262a4fa6cf` |
| `synthetic_attribute_twin_summary.csv` | `376e6d3ee3bb59b2e4a1b9a6fa9477940427df92e414a8b2055e512e2ccbbbe9` |

## Evidence gate

The gate fails closed to **`computational_feasibility`**. Current blockers are exactly:

```text
eligible real observed participant-by-meal outcome manifest is missing
locked subject-held-out primary paired-results table is missing
```

`direct_external_validity` requires eligible observed participant-by-meal outcomes, verified microbiome linkage, a passed run-level method lock, disjoint development/test subjects and exactly one completed subject-held-out RMSE comparison for each of `glucose_iAUC_2h` and `tg_6h_rise`. Both comparisons must show negative model-minus-reference RMSE, a 95% paired CI entirely below zero, Holm-adjusted P <= 0.05, and 2,000 bootstrap plus 2,000 permutation replicates with at least 90% valid replicates.

Synthetic, aggregate ZOE, GMrepo, knowledge-graph and predictor-only evidence cannot upgrade the gate. Current evidence cannot support claims of transformed postprandial response, precision readiness, clinical validity, direct response validity, external validity or causal dietary effects.

## Limitations and concerns

1. The synthetic DGP deliberately shares the locked mapping and published domain aggregation structure with the evaluated model. This is appropriate for an identifiability stress test but creates a correctly specified setting and cannot establish robustness to biological model misspecification.
2. Five replicate seeds yield coarse across-replicate percentile intervals. They are reported transparently and must not be interpreted as clinical uncertainty intervals.
3. The no-centering legacy comparator preserves the requested final-offset form but is not numerically identical to the prohibited historical centered implementation.
4. The full repository suite is not green because of 13 out-of-scope Task 3 fixture/fold incompatibilities. Task 5 focused and related suites are green.
5. The SciPy/scikit-learn import path emits an environment warning because installed NumPy 2.4.6 is outside SciPy's declared `<2.3.0` range. Task 5 focused tests nevertheless pass; environment pinning should be repaired separately.
