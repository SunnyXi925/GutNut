# Validation evidence matrix and claim gate

## Evidence boundary

This matrix separates direct response validity from predictor reconstruction,
aggregate or mechanistic context and synthetic pipeline stress testing. Only
verified, observed participant-by-meal outcomes can enter the direct-validity gate.
A locally available file is not eligible merely because it is accessible. No current
Task 4 production registry binds aggregate biological or mechanistic evidence to an
authorized positive claim.

| Evidence | Data class | Access status | Allowed role | Execution status | Claims it can support | Claims it cannot support |
| --- | --- | --- | --- | --- | --- | --- |
| PREDICT-1 microbiome resources (`PRJEB39223`, `EH5458`) | Real public microbiome predictor data | Public or locally cached predictor metadata; participant-to-meal linkage is not verified | Predictor reconstruction only | Provenance audited; no direct outcome analysis executed | Audited predictor availability only | Biological consistency, person-by-meal response validity, clinical validity, causal dietary effects |
| PREDICT-2/3 sequencing (`PRJEB75460`, `PRJEB75462`, `PRJEB75463`, `PRJEB75464`) | Real public sequencing predictor data | Public; no eligible response table identified | Predictor reconstruction only | Provenance audited; no direct outcome analysis executed | Audited sequencing availability only | Biological consistency, person-by-meal response validity or independent endpoint performance |
| Nature supplementary workbook (`10.1038/s41586-025-09854-7:MOESM3`) | Aggregate public supplementary evidence | Public | Context only; no current claim authorization | Workbook structure audited; no participant-by-meal endpoint established and no registry-bound Task 4 production evidence | Audited workbook availability only | Biological consistency, direct, construct, clinical or external response validity |
| ZOE public microbiome ranks and GMrepo disease labels | Aggregate or cohort-level supporting evidence | Public or repository-mediated | Context only; no current claim authorization | Supporting pathways are separate from this gate; no registry-bound Task 4 production evidence | Audited resource availability only | Biological consistency, individual food-response validity, clinical utility or causal effects |
| Knowledge-graph paths | Curated mechanistic supporting evidence | Repository-local/public-source dependent | Context only; no current claim authorization | Separate supporting analysis; no registry-bound Task 4 production evidence | Audited graph availability only | Mechanistic validation, construct, clinical or external response validity |
| Controlled clinical archive (`10.5281/zenodo.17236382`; version `10.5281/zenodo.17236383`) | Controlled real participant clinical resource | Access not granted; encrypted payload; endpoint schema, meal keys, units and microbiome linkage remain unverified | Controlled eligibility assessment; future direct validation only after approval and verification | Metadata audited; outcomes not opened; direct benchmark not executed | Existence of a controlled access route | Any observed endpoint performance or completed external validation |
| Local `glucose_iAUC_2h`, `tg_6h_rise` and `c_peptide_iAUC_2h` fields | Synthetic local response data | Local | Synthetic stress testing only | Classified in the prior provenance audit; not used as real outcomes | Computational pipeline stress testing | Observed PREDICT response validity, retrospective external validation or clinical validity |
| Attribute-level digital gut twin generated in Task 5 | Correctly specified synthetic positive-control with programmed truth | Deterministically generated in `results/phase2/source-data/correctly_specified_synthetic_positive_control/` | `correctly_specified_synthetic_positive_control` | Executed; code-fixed checks, metrics, seeds, source hashes and environment versions are frozen in the accompanying manifest | The locked implementation recovered the programmed mapping in this correctly specified synthetic positive-control and lost recovery after random or Sattolo-deranged assignment; cap and pipeline reproducibility checks | General model superiority, biological validity of a mask, clinical validity, external validity, precision readiness, or real-world transformation of postprandial responses |
| Task 1 method-lock contract and Task 3 benchmark implementation | Method/provenance infrastructure | Repository-local implementation | Computational feasibility | Implementation exists; no eligible real run-level outcome manifest and paired primary result table are available | Fail-closed readiness and reproducible analysis design | Completed external validation or empirical endpoint improvement |

## Frozen direct-validity requirements

The gate returns `direct_external_validity` only when one eligible run jointly
establishes all of the following:

1. Observed participant-by-meal outcomes with verified provenance and microbiome linkage.
2. A production call using only `EvidenceGateArtifactPaths`; no caller-provided mapping or data frame can upgrade claims.
3. Live Task 1/2/3 method-lock and trusted outcome-loader revalidation, plus an independently reviewed result-run manifest whose exact digest appears in the repository-fixed registry.
4. The formal Task 5 exporter consumes the real Task 3 `BenchmarkResult` schema and hashes canonical `paired_metrics`, `predictions`, `analysis_status`, split-audit summary and result-run manifest artifacts. A module-scoped integration fixture actually invokes `run_person_meal_benchmark`, retains its produced predictions, status DataFrame and split objects, then roundtrips that result through the exporter and path-only gate. It preserves Task 3 `comparator`, `reference` and `adjusted_p_value` fields; no parallel hand-made result schema is accepted.
5. After live method-lock/outcome-loader validation, the gate recomputes the frozen primary nested subject-held-out splits and transitive family/twin components from the verified predictor frame and config using the production cohort-split functions. Every prediction `row_id`, participant, component and outer fold, and every split-summary count, must match that recomputation. Predictor opportunities and participants without a finite outcome are allowed and are not required to appear in predictions.
6. Exactly two subject-held-out primary RMSE rows, one each for `glucose_iAUC_2h` and `tg_6h_rise`, comparing locked attribute GMNPS with `fcs_microbiome`.
7. For each endpoint, `locked_attribute_gmnps` and `fcs_microbiome` predictions must have identical official person-meal rows, folds and recomputed family/twin-component clusters. Exported `y_true` must equal the trusted outcome and every `y_pred` must be finite.
8. The gate independently recomputes both RMSE values and their difference from predictions, then reruns Task 3's production component-cluster percentile bootstrap and complete-cluster paired permutation using the exact frozen artifact seeds and 2,000 repeats each.
9. Recomputed differences, confidence limits, raw permutation *P* values, methods and valid counts must match the paired artifact within floating-point tolerance. At least 90% of replicates must be valid.
10. Holm values are recomputed from the two independently reproduced raw *P* values, must pass at 0.05, and are cross-checked against supplied adjusted values.

GMrepo, ZOE ranks, knowledge graphs, simulations and predictor-only resources are
supporting evidence and cannot satisfy or upgrade any of these requirements.

## Current gate outcome

**Tier: `computational_feasibility` (fail closed).**

The machine-readable decision at
`results/phase2/evidence-gate/current_gate_decision.json` was generated from the
absence of real production artifacts and reports these blockers:

- `eligible observed participant-by-meal outcomes are unavailable: predict_controlled_clinical_zenodo access_status=controlled_not_granted`
- `run-level method-lock manifest path is missing`
- `result-run manifest path is missing`
- `paired metrics path is missing`
- `predictions path is missing`
- `split audit summary path is missing`
- `analysis status path is missing`

Therefore, no positive claim of biological consistency, mechanistic validation,
transformation of postprandial response, precision readiness, clinical validity,
direct response validity or external validity is permitted. No real
participant-by-meal validation experiment is represented as completed. Phase 3 must
run `code/src/scripts/check_claim_policy.py` against every manuscript/build input. The
CLI accepts input files only: it loads the fixed decision/policy paths and verifies their
exact hashes against `code/src/configs/claim_policy_registry.json`. Arbitrary decision
or policy paths cannot authorize a claim. The production policy builder reruns the
path-only evidence gate; an in-memory outcome can create testing-only bytes but cannot
create an authorized bundle. Claim-bearing sentences are checked individually. Any
sentence matching GMNPS, model, framework, approach, method, system, platform,
algorithm, implementation, personalization, personalized score,
microbiome-informed score, score, finding, result or analysis subjects is denied by
default; classification no longer depends on also finding an assertion-vocabulary
term. Every supplied plain-text input is scanned, so section labelling cannot bypass
enforcement. The only Methods exception is a full-sentence match to an enumerated,
anchored infrastructure template for `implemented as`, `computes`, `loads`,
`verifies`, `hash-binds`, `uses bounded attribute calibration`, or reproducible
source-hash artifact verification. These templates cannot contain outcome,
performance, validation, guidance, prediction, forecasting, stratification or related
scientific semantics.

At the current computational tier, only exact templates for the correctly specified
synthetic programmed mapping, audited outcome unavailability and the computational
fail-closed design are eligible. An explicit negative limitation is permitted only
when its cue is bound to every specific forbidden/assertion match within one local
clause and at most six words; punctuation, contrast and a new coordinated predicate
end that scope. Thus `does not establish external validity` is allowed, but
`Although external validation was not performed, external validity is established.`
is rejected. Every other subject-bearing sentence fails closed, including claims that
use previously unseen verbs such as `yields`, `achieves` or `stratifies`.

Even at a future direct tier, generic external-validity, clinical, causal, guidance
and recommendation wording remains forbidden. The sole additional positive template
is explicitly scoped to both locked primary endpoints, `glucose_iAUC_2h` and
`tg_6h_rise`, and their subject-held-out RMSE comparison. No Task 4 production
evidence has been executed and registry-bound, so biological consistency remains a
forbidden positive claim. Task 5 did not check or revise the current manuscript.

Current approved claim-control hashes:

| Artifact | SHA-256 |
| --- | --- |
| `claim_policy.json` | `0e29b08359d3a335db05c613540149640127842e133bb8e582f920428645a7d5` |
| `current_gate_decision.json` | `da0739f4df4896ff54b03cef3ba8600218ad9acb2d0286b30579f220a48ec085` |
| `claim_policy_registry.json` | `90bb3828d1f62e160d3cc90309da4520b85f453f2eb03401ca7cfba6004a1311` |
