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
non-symlink artifacts and invokes the Task 3 trusted method-lock/outcome loader.
This v2 closure did not yet export the real `BenchmarkResult` schema or independently
recompute production splits; those claims are corrected and superseded by the v3
re-review closure below.

The v2 gate independently recomputed the run binding and primary statistical
checks, but its uploaded detailed split table was not an independent split
recomputation. Independent component and fold verification begins only with the v3
implementation documented below.

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

## Re-review closure: Task 3-integrated evidence gate v3

### Status and correction boundary

**DONE.** All four Major re-review findings are closed. This section supersedes
the v2 gate-integration and split-verification descriptions above. Independent
split/component recomputation was not present in v2; it begins with this v3
implementation. No manuscript, grouped-split test, `scoring/__init__.py`, or the
user-owned legacy `synthetic_twin.py` was modified.

### M1: actual Task 3 integration

`export_benchmark_result` now accepts the real
`person_meal_benchmark.BenchmarkResult` dataclass and a `VerifiedRunBinding`. It
writes canonical CSV artifacts for the actual Task 3 `paired_metrics`,
`predictions`, `analysis_status` DataFrame and split-audit summary, then writes a
result-run manifest containing each artifact SHA-256. The path-only gate reads
exactly those artifacts and preserves Task 3 fields including `comparator`,
`reference` and `adjusted_p_value`; the former parallel `model` and
`holm_adjusted_p_value` contract was removed. Predictions are now a required,
hash-bound artifact.

The integration test constructs an actual `BenchmarkResult`, exports it, registers
the immutable manifest digest and roundtrips all artifacts through the production
gate contract.

### M2: independently recomputed split semantics

After the trusted Task 3 loader has revalidated the live method lock and outcomes,
the v3 gate obtains the verified predictor frame and frozen validation config. It
calls production `family_twin_component_ids` and `make_nested_group_splits` using
the frozen five-fold settings and seeds. It then verifies:

- development/test component disjointness from the recomputed transitive graph;
- one consistent test fold per predictor row, participant and component;
- every prediction `row_id`, participant, meal, recomputed component and
  `outer_fold`;
- exact finite-outcome row coverage for every primary endpoint and locked Task 3
  comparator; and
- all five nonempty split-summary rows and their recomputed train/test/drop counts.

Uploaded component labels are not trusted. Predictor opportunities and entire
participants with no outcome row, as well as individual non-finite endpoint
values, are permitted and omitted only from the applicable prediction coverage.
The gate no longer requires the predictor/audit universe to equal the outcome
participant universe.

### M3: strict primary statistical domains

Raw permutation P values and adjusted values must be finite and in [0,1]. Confidence
interval bounds must be finite, ordered and in the required improvement direction,
with the exact Task 3 paired component-bootstrap method. Requested and valid
resample counts must be strict non-boolean integers: requested counts equal 2,000,
counts are nonnegative, valid never exceeds requested, and valid fractions are at
least 90% and consistent with Task 3's recorded fraction columns. No integer
truncation is performed. Holm adjustment is recomputed from the two raw Task 3 P
values and compared with `adjusted_p_value`.

### M4: dynamic claim policy

The generated policy now derives `tier`, `source_state`, allowed claims, forbidden
claims and tier-specific forbidden patterns from the actual gate outcome. A direct
tier remains conservative and continues to prohibit clinical utility/validity,
causal dietary-effect, precision-ready and transformation claims, while it no
longer contradicts a verified endpoint-scoped direct-validity decision by globally
forbidding the phrase “external validity”. The checker recognizes explicit
negative limitation constructions such as “does not establish external validity”
but still rejects positive assertions.

The current artifacts remain fail-closed because no real production artifacts are
available:

```text
tier=computational_feasibility
source_state=absent_real_validation_artifacts
```

Current artifact SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| `claim_policy.json` | `9eac4dac7085834757068dc145015279bb88fba90151ee8e7fb128e81570be51` |
| `current_gate_decision.json` | `74765d4461365ae1da5d4aa719c28ef053dc9ebe884c7e88b95fb799b1a74dd8` |

No current manuscript was checked or revised.

### Final tests

| Scope | Result | Time |
| --- | ---: | ---: |
| Focused exporter/integration/tamper plus synthetic tests | 30 passed | 37.54 s |
| Related attribute, lock, loader, benchmark and split tests before the final provenance-tamper case | 289 passed | 170.70 s |
| Full `code/src/tests` suite after all cases | 660 passed | 208.25 s |

All runs had zero failures. The sole warning remains the pre-existing environment
mismatch: NumPy 2.4.6 is outside SciPy 1.13.1's declared `<2.3.0` range.

## Final review closure: producer integration and policy trust

### M1: actual producer integration

This section supersedes the earlier statement that constructing a
`BenchmarkResult` dataclass alone demonstrated producer integration. The final
module-scoped integration fixture builds coherent testing-only Task 3 lock,
predictor and outcome artifacts and actually calls `run_person_meal_benchmark`
once. It preserves the producer's real column names, `analysis_status` DataFrame,
prediction rows and four split mappings. The resulting real `BenchmarkResult` is
passed to `export_benchmark_result`; canonical paired metrics, predictions,
analysis status and split summary are then consumed by the path-only gate with a
test-controlled trusted result registry and outcome loader. The roundtrip reaches
the direct tier only inside this testing-only controlled fixture.

The exporter now rejects empty/non-DataFrame producer artifacts, incomplete real
producer columns, missing subject-held-out split objects and incomplete primary
comparator prediction coverage. The inexpensive tamper/domain tests continue to
use small hand-built fixtures; they are not described as producer integration.

### M4: fixed production claim authority

Production policy generation no longer accepts an arbitrary in-memory
`EvidenceGateOutcome`. `build_claim_policy_from_evidence_gate` reruns the
path-only production gate and writes only the fixed repository decision/policy
location. The lower-level fixture writer is private and stamps its output
`testing_only_unauthorized`, which the production checker refuses.

`code/src/configs/claim_policy_registry.json` fixes the approved current
decision/policy pair. The checker reads only the repository-fixed paths, validates
the registry schema and approved entry, verifies both exact file SHA-256 values,
then verifies the decision-to-policy binding and payload hashes. The CLI accepts
only manuscript/build inputs; `--decision`, `--policy` and public testing bypasses
do not exist.

The current path-only gate was rerun without real validation artifacts and remains:

```text
tier=computational_feasibility
source_state=absent_real_validation_artifacts
```

Current approved hashes are:

| Artifact | SHA-256 |
| --- | --- |
| `claim_policy.json` | `a437c42a5e91ddcc9eae64f463a0b79882f4fc2997c28e510ef520c4e6114d2f` |
| `current_gate_decision.json` | `b0ab73b94a102dfa3a0ad272b0c4ec401bb20f6c60d387008ede153fa8981c64` |

Generic positive “external validity” wording remains forbidden at every tier. A
future verified direct tier permits only the exact frozen formulation scoped to
both locked primary endpoints, `glucose_iAUC_2h` and `tg_6h_rise`. Explicit
negative limitation sentences are allowed; clinical, causal, general
precision-ready and postprandial-transformation claims remain forbidden. Task 5
did not inspect, check or revise the manuscript.

### Final-review tests

Final test results were:

| Scope | Result | Time |
| --- | ---: | ---: |
| Focused Task 5 | 35 passed, 1 warning | 60.01 s |
| Related Task 1/2/3/5 validation | 295 passed, 1 warning | 185.93 s |
| Full `code/src/tests` | 665 passed, 1 warning | 202.86 s |

There were no failures. The one warning is the unchanged NumPy 2.4.6 / SciPy
1.13.1 declared-version mismatch. The real producer fixture ran once per test
module; no benchmark result cache or generated real-validation artifact was
introduced.

## Whole-review final closure: independent statistic recomputation and tiered claims

### Status

**DONE.** This closure supersedes the preceding v3 statistical and claim-control
descriptions. The production gate is now `direct-response-evidence-gate-v4`.
The user-owned legacy `synthetic_twin.py`, the manuscript and
`scoring/__init__.py` were not edited or staged.

### M1: prediction-derived production-statistic recomputation

After live method-lock and trusted outcome loading, and after production split and
family/twin-component recomputation, the gate now joins each frozen primary endpoint
to trusted outcomes by the official participant-meal keys. It requires the
`locked_attribute_gmnps` and `fcs_microbiome` rows, folds and recomputed clusters to
align exactly, requires exported `y_true` to equal the trusted endpoint value and
rejects every non-finite primary `y_pred`.

For each of `glucose_iAUC_2h` and `tg_6h_rise`, the gate independently recomputes
both RMSE values and their difference from predictions. It verifies the Task 3 seed
derivation, then calls the Task 3 production component-cluster percentile bootstrap
and complete-cluster paired permutation algorithms with the exact artifact seeds and
2,000 repeats each. Recomputed estimates, confidence limits, raw permutation *P*
values, methods and valid counts must match the paired artifact within a `1e-12`
relative/absolute floating-point tolerance. Holm adjustment is then recomputed from
the two reproduced raw *P* values.

Negative tests fail closed for identical model/reference predictions paired with a
declared improvement, incorrect exported truth, non-finite primary predictions and
seed/statistic mismatch. Broad tamper cases use deterministic test doubles; one small
gate fixture executes the actual Task 3 algorithms at 2,000 bootstrap and 2,000
permutation repeats.

### M2: tier-aware sentence-level claim control

The current fail-closed allowed-claim set no longer contains
`biological_consistency` or `mechanistic_consistency`. No independent,
registry-bound Task 4 production evidence currently authorizes either claim.

The fixed-path, registry-bound checker now treats each sentence containing validation,
external or independent-cohort wording, generalization, held-out wording, prediction
improvement, metabolic response, guidance, recommendation, biological consistency,
mechanistic validation, clinical, causal or precision-ready wording as claim-bearing.
Such a sentence must be an explicit negative limitation or exactly match a positive
template authorized for its tier. The computational tier has only the correctly
specified synthetic programmed-mapping, audited-data-unavailable and computational
fail-closed templates. A future direct tier adds only the exact statement naming both
locked primary endpoints and their subject-held-out RMSE comparison; it does not
authorize generic external-validity, clinical, causal or dietary-guidance wording.

The current production bundle remains fail closed:

```text
tier=computational_feasibility
source_state=absent_real_validation_artifacts
```

| Artifact | SHA-256 |
| --- | --- |
| `claim_policy.json` | `ede519199becc7416a73d8f78bfdd82d07f5475d99bc5c31a382f56a4ecd11af` |
| `current_gate_decision.json` | `3398cda302524393eca7dbe0dcbba431b0d3aaaaa66e6d62078da3cbdd39e8a1` |
| `claim_policy_registry.json` | `ddf04f82ca51262377ffcce8b516d21d470c63a05ae5f47573019fa431dddb29` |

### Minor findings

The Task 2 report now retains the former `12bbe...` validation-config digest as
historical and records current `a8208e...` as superseding it. The historical Task 2
body was not rewritten.

The formal synthetic experiment now refuses to choose a truth digest unless all
replicates yield exactly one unique truth definition. The full formal experiment was
rerun and frozen; all code-fixed checks passed.

| Frozen synthetic artifact | SHA-256 |
| --- | --- |
| `synthetic_attribute_twin_config.json` | `fd5a21d815406225f3fe5a6222cd39dafed7b98014dbc25190b61399fdb97c35` |
| `synthetic_attribute_twin_manifest.json` | `df0684efa66377bdbdaec77182cdcbadc03b35cd4c5ec0403febf3c7d1096263` |
| `synthetic_attribute_twin_replicate_metrics.csv` | `b0a0e379e8654ef1c370d614abe932a2a7cd0cedb4ad13ccf52ef6db0a7e70e6` |
| `synthetic_attribute_twin_success_checks.csv` | `091bea04c4baed04fa4f4898d8c980cac081e3bc7dfb58435b5799fdfac32719` |
| `synthetic_attribute_twin_summary.csv` | `b78ab05011a7c5f9bb150b22d50d72ff84797721c6a5c1d3bc8c37a3edbf9bdd` |

### Final verification

| Scope | Result | Time |
| --- | ---: | ---: |
| Focused Task 5 | 53 passed, 1 warning | 54.04 s |
| Related Task 1/2/3/5 validation | 352 passed, 1 warning | 208.16 s |
| Full `code/src/tests` | 683 passed, 1 warning | 205.55 s |

The full suite exceeds the 665-test baseline by 18 tests and has no failures. The sole
warning is the pre-existing environment mismatch: installed NumPy 2.4.6 is outside
SciPy 1.13.1's declared `<2.3.0` range. This remains the only known concern.

## Final claim-policy Major closure (2026-08-14)

### Status and enforcement

**DONE.** This section supersedes the sentence-detection and negative-limitation
descriptions above. The fixed policy is now `claim-policy-v3`. In addition to the
existing forbidden families, it records claim-bearing subject patterns for GMNPS,
models, frameworks, approaches, personalization, personalized scores,
microbiome-informed scores and related scientific subjects, plus assertion families
covering prediction, superiority, improvement, generalization, validation, accuracy,
establishment, demonstration, support, enablement, guidance, recommendation,
association, recovery and response.

Any subject-plus-assertion sentence supplied to the checker fails closed unless it
exactly matches a positive template authorized for the current tier or is an explicit
negative limitation. This applies to every supplied plain-text input, so a caller
cannot bypass enforcement by labelling content as Methods or another section. A
narrow infrastructure exception covers support/enable statements whose objects are
only hashes, manifests or artifacts and contain no scientific outcome object.

Negative cues are bound to the specific forbidden/assertion match in the same local
clause, with a maximum six-word window. Commas, semicolons, contrast boundaries and
coordinating predicates terminate the exemption. Consequently, both `Although
external validation was not performed, external validity is established.` and
`does not fail and accurately predicts` constructions are rejected, while `does not
establish external validity` remains allowed.

No Task 4 production evidence has been executed and registry-bound. Biological
consistency remains a forbidden positive claim at the current computational tier.
The current approved hashes are:

| Artifact | SHA-256 |
| --- | --- |
| `claim_policy.json` | `43b9781f19f99d1bc84210e0f0f53784a54ded6cc3b230a9678ed5cd97eaf7cc` |
| `current_gate_decision.json` | `e2ae3020bfcea4b208d1ac3afd50987d8b2f943632df31afa306c05e929e18eb` |
| `claim_policy_registry.json` | `4b15b40cb58f21f0cdded7c02d609195b8cd970fc31a9109a7a091719c4c862c` |

### Verification

| Scope | Result | Time |
| --- | ---: | ---: |
| Focused `test_evidence_gate.py` | 56 passed, 1 warning | 13.95 s |
| Related Task 1/2/3/5 claim/gate validation | 338 passed, 1 warning | 169.23 s |
| Full `code/src/tests` | 694 passed, 1 warning | 185.50 s |

The full suite exceeds the 683-test baseline by 11 regression cases and has no
failures. The sole warning remains the pre-existing NumPy 2.4.6 / SciPy 1.13.1
declared-version mismatch.

## Default-deny claim-policy Major closure (2026-08-14)

### Enforcement correction

**DONE.** This section supersedes the subject-plus-assertion rule immediately
above. The fixed policy is now `claim-policy-v4`. A sentence that matches a
claim subject is denied by default; an assertion-vocabulary match is no longer
required to classify it as a claim. Consequently, all of the following fail
closed at the current tier:

```text
GMNPS yields lower RMSE than Food Compass.
GMNPS achieves superior glycaemic forecasting.
GMNPS stratifies individuals by glycaemic excursion.
```

Only three subject-bearing sentence classes can pass: an exact tier-authorized
positive template; a single-clause negative limitation whose cue is locally
bound to every concrete assertion/forbidden match; or an enumerated
infrastructure Methods template. The Methods templates are anchored full-sentence
forms for `implemented as`, `computes`, `loads`, `verifies`, `hash-binds`, `uses
bounded attribute calibration`, and the existing source-hash artifact-verification
form. They reject outcome, performance, validation, guidance, prediction,
forecasting, stratification and related scientific semantics. Section labels do
not alter enforcement.

The regenerated production bundle remains fail closed:

```text
tier=computational_feasibility
source_state=absent_real_validation_artifacts
```

| Artifact | SHA-256 |
| --- | --- |
| `claim_policy.json` | `0e29b08359d3a335db05c613540149640127842e133bb8e582f920428645a7d5` |
| `current_gate_decision.json` | `da0739f4df4896ff54b03cef3ba8600218ad9acb2d0286b30579f220a48ec085` |
| `claim_policy_registry.json` | `90bb3828d1f62e160d3cc90309da4520b85f453f2eb03401ca7cfba6004a1311` |

### Verification

| Scope | Result | Time |
| --- | ---: | ---: |
| Focused `test_evidence_gate.py` | 71 passed, 1 warning | 13.20 s |
| Full `code/src/tests` | 709 passed, 1 warning | 181.91 s |

The full suite exceeds the 694-test baseline by 15 regression cases and has no
failures. The warning is the unchanged NumPy 2.4.6 / SciPy 1.13.1 declared-version
mismatch.
