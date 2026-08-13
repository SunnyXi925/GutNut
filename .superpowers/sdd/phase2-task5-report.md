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
3. Programmed per-attribute effects from physical exposure normalization, the locked 20% attribute-range perturbation, and published Food Compass domain aggregation, including top-k membership.
4. A leave-one-attribute-out decomposition that allocates top-k interaction residuals while preserving the exact programmed total effect.
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
| Formal attribute cap mode | 20% primary |
| Formal final cap mode | +/-12 primary |

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

Locked universal-rank preservation was 0.999 (0.994-1.000). The excluded-proxy contribution fraction was exactly 0 under the expert mask and 0.749 (0.636-0.869) under the original mask. All five code-fixed checks passed. Sattolo permutations were deterministic, complete permutations with no fixed points.

The locked implementation recovered the programmed mapping in this correctly specified synthetic positive-control and lost recovery after random or Sattolo-deranged assignment. This is not clinical, construct or external validation.

## Frozen outputs

The following hashes record the historical files produced by the original Task 5 commit; those files were deleted and replaced by the review-closure outputs below.

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

1. The synthetic DGP deliberately shares the locked mapping and published domain aggregation structure with the evaluated model. This creates a correctly specified positive-control and cannot establish robustness to biological model misspecification.
2. Five replicate seeds yield coarse across-replicate percentile intervals. They are reported transparently and must not be interpreted as clinical uncertainty intervals.
3. The no-centering legacy comparator preserves the requested final-offset form but is not numerically identical to the prohibited historical centered implementation.
4. The full repository suite is not green because of 13 out-of-scope Task 3 fixture/fold incompatibilities. Task 5 focused and related suites are green.
5. The SciPy/scikit-learn import path emits an environment warning because installed NumPy 2.4.6 is outside SciPy's declared `<2.3.0` range. Task 5 focused tests nevertheless pass; environment pinning should be repaired separately.

## Review closure (2026-08-14; supersedes the status, wording and test counts above)

### Closure status

**DONE.** All Task 5 review findings are closed on top of `9d3da7a`. The
historical implementation record above is retained for auditability, but its
former evidence-role wording, source-data paths, gate API and full-suite result
are superseded by this section. The parallel grouped-split fixture files were not
modified.

The deliberate plan-safe divergence remains in force:
`code/src/gmnps/validation/synthetic_twin.py` still contains user work and was not
edited, staged or committed. The review changes remain in
`attribute_synthetic_twin.py`. The manuscript and `scoring/__init__.py` were not
edited, staged or committed.

### Review RED and GREEN

Review RED was captured with:

```text
uv run pytest -q code/src/tests/test_evidence_gate.py --maxfail=1
```

Collection failed because the required `gmnps.validation.claim_policy` module did
not exist. After implementation, the final test results were:

| Scope | Result | Time |
| --- | ---: | ---: |
| Task 5 focused (`test_attribute_synthetic_twin.py`, `test_evidence_gate.py`) | 24 passed | 36.62 s |
| Related attribute and method-lock tests before four final negative gate cases were added | 189 passed | 45.54 s |
| Full `code/src/tests` suite after all final cases | 654 passed | 193.85 s |

All three runs emitted one environment warning: installed NumPy 2.4.6 is outside
SciPy 1.13.1's declared `<2.3.0` range. No test failed.

A subsequent related-only rerun was intentionally stopped at the coordinator's
request after 192 passing tests and no failures; the complete final full-suite run
above includes and passes every related test.

### Critical gate closure

The production API now accepts only `EvidenceGateArtifactPaths`. It no longer
accepts a caller mapping or data frame. In one fail-closed call it rereads regular
non-symlink artifacts, invokes the Task 3 trusted method-lock/outcome loader (which
revalidates Task 1/2/3 contracts), verifies the independently approved result-run
manifest digest from the repository-fixed registry, and verifies hashes for paired
results, detailed split audit, analysis status, outcome source, predictor frame,
feature contract, validation config, Task 3 implementations and benchmark
specification.

The gate independently recomputes the run binding, requires all result and audit
rows to share it, requires exactly the two frozen subject-held-out primary RMSE
rows, recomputes Holm adjustment from raw paired permutation P values, enforces the
exact paired family/twin-component bootstrap method, 2,000 bootstrap and
permutation replicates, and at least 90% valid replicates. Participant and
family/twin-component disjointness is recomputed from the hashed detailed split
audit, not accepted from manifest lists.

The lower-level in-memory evaluator always returns
`testing_only_no_claim_upgrade`. Tests cover self-signing, fake digests, supporting
evidence relabelling, byte tamper, mismatched run bindings, forged Holm values, and
participant/component leakage. Synthetic and supporting evidence cannot upgrade
the production gate.

The fixed registry currently has no approved runs. The deterministically generated
decision is therefore `computational_feasibility` with these blockers:

```text
eligible observed participant-by-meal outcomes are unavailable: predict_controlled_clinical_zenodo access_status=controlled_not_granted
run-level method-lock manifest path is missing
result-run manifest path is missing
paired primary results path is missing
hashed detailed split audit path is missing
analysis status path is missing
```

`results/phase2/evidence-gate/current_gate_decision.json` is bound by SHA-256 to
`claim_policy.json`. `code/src/scripts/check_claim_policy.py` exits nonzero for a
forbidden phrase or tampered decision/policy. Phase 3 must run it against every
manuscript/build input. Task 5 did not check or revise the current manuscript.

### Correctly specified synthetic positive-control

The former general-role synthetic files were deleted and replaced by
`results/phase2/source-data/correctly_specified_synthetic_positive_control/` with
`evidence_role=correctly_specified_synthetic_positive_control`.

Locked implementation recovered the programmed mapping in a correctly specified
synthetic positive-control and lost recovery after random or Sattolo-deranged
assignment.

The checks are code-fixed for reproducibility in the same release and were not
independently preregistered. The formal five-seed run uses only the primary 20%
attribute cap and +/-12 final cap. The 10/30% and +/-8/15 cap cases are unit
boundary tests only and are not formal-run sensitivity results. The bundle retains
raw Gaussian noise, effective post-clipping noise and a clipping indicator; stream
independence is asserted for the raw noise, without claiming independence after
clipping.

Formal configuration remained five seeds (1701-1705), 24 individuals, 12 foods,
noise SD 0.35 and 200 participant bootstraps per comparator/seed. No parameter was
changed after observing results. Selected five-replicate means and empirical
2.5th-97.5th percentiles are:

| Comparator/metric | Estimate (interval) |
| --- | ---: |
| Locked attribute GMNPS residual RMSE | 0.3522 (0.3461-0.3647) |
| FCS residual RMSE | 3.5411 (3.0619-3.8431) |
| Random assignment residual RMSE | 2.8253 (2.5182-3.2532) |
| Sattolo assignment residual RMSE | 3.6541 (3.1163-4.4582) |
| Locked residual Spearman | 0.9876 (0.9848-0.9896) |
| Locked universal-rank Spearman | 0.9986 (0.9937-1.0000) |
| Expert-mask legacy-form residual RMSE | 3.9315 (3.4457-4.5456) |
| Original-mask legacy-form residual RMSE | 5.2724 (4.8346-5.6255) |

This is a correctly specified synthetic positive-control only. It does not support
general model superiority, biological validity of either mask, clinical validity,
external validity or a precision-ready claim.

The manifest records simulator and concrete scorer source hashes, all RNG streams,
CPython 3.11.15, NumPy 2.4.6, pandas 3.0.5, SciPy 1.13.1 and scikit-learn 1.5.2.
Frozen file SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| `synthetic_attribute_twin_config.json` | `fd5a21d815406225f3fe5a6222cd39dafed7b98014dbc25190b61399fdb97c35` |
| `synthetic_attribute_twin_manifest.json` | `7e765b23bfc2aaeb3f6e04cac7cbd7da36855f2a31af8a744c8283aeb6b610a4` |
| `synthetic_attribute_twin_replicate_metrics.csv` | `b0a0e379e8654ef1c370d614abe932a2a7cd0cedb4ad13ccf52ef6db0a7e70e6` |
| `synthetic_attribute_twin_success_checks.csv` | `091bea04c4baed04fa4f4898d8c980cac081e3bc7dfb58435b5799fdfac32719` |
| `synthetic_attribute_twin_summary.csv` | `b78ab05011a7c5f9bb150b22d50d72ff84797721c6a5c1d3bc8c37a3edbf9bdd` |

### Remaining limitations

The positive-control deliberately shares the programmed mapping with the locked
implementation and cannot establish robustness to misspecification. Five seeds
give coarse replicate intervals. Direct observed-response validation remains
blocked by controlled access and absent immutable production artifacts; this is an
expected gate outcome, not evidence that a real validation run was completed.
