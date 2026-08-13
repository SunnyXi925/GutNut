# Validation evidence matrix and claim gate

## Evidence boundary

This matrix separates direct response validity from predictor reconstruction,
aggregate biological consistency, mechanistic consistency and synthetic pipeline
stress testing. Only verified, observed participant-by-meal outcomes can enter the
direct-validity gate. A locally available file is not eligible merely because it is
accessible.

| Evidence | Data class | Access status | Allowed role | Execution status | Claims it can support | Claims it cannot support |
| --- | --- | --- | --- | --- | --- | --- |
| PREDICT-1 microbiome resources (`PRJEB39223`, `EH5458`) | Real public microbiome predictor data | Public or locally cached predictor metadata; participant-to-meal linkage is not verified | Predictor reconstruction; biological consistency | Provenance audited; no direct outcome analysis executed | Predictor availability and microbiome-profile feasibility | Person-by-meal response validity, clinical validity, causal dietary effects |
| PREDICT-2/3 sequencing (`PRJEB75460`, `PRJEB75462`, `PRJEB75463`, `PRJEB75464`) | Real public sequencing predictor data | Public; no eligible response table identified | Predictor reconstruction; biological consistency | Provenance audited; no direct outcome analysis executed | Public sequencing availability | Person-by-meal response validity or independent endpoint performance |
| Nature supplementary workbook (`10.1038/s41586-025-09854-7:MOESM3`) | Aggregate public supplementary evidence | Public | Aggregate biological consistency | Workbook structure audited; no participant-by-meal endpoint established | Published aggregate rank and consistency context | Direct, construct, clinical or external response validity |
| ZOE public microbiome ranks and GMrepo disease labels | Aggregate or cohort-level supporting evidence | Public or repository-mediated | Biological consistency | Supporting pathways are separate from this gate; no direct response upgrade executed | Concordance or disease-stratified biological consistency when analysed | Individual food-response validity, clinical utility or causal effects |
| Knowledge-graph paths | Curated mechanistic supporting evidence | Repository-local/public-source dependent | Mechanistic consistency | Separate supporting analysis; not a direct response test | Plausibility of nutrient–microbiome–host paths | Construct, clinical or external response validity |
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
4. Result-manifest hashes for paired results, detailed split audit, analysis status, outcome source, predictor frame, feature contract, validation config, Task 3 implementations and benchmark specification; all rows must bind to one run and one independently recomputed run binding.
5. Disjoint development/test participant and transitive family/twin component sets recomputed from the hashed detailed split audit, with the audited participant universe matching the loaded outcomes.
6. Exactly two subject-held-out primary RMSE rows, one each for `glucose_iAUC_2h` and `tg_6h_rise`, comparing locked attribute GMNPS with `fcs_microbiome`.
7. RMSE differences in the favourable direction and the exact paired family/twin-component bootstrap method, with 95% intervals excluding zero.
8. Holm values recomputed from raw paired permutation *P* values, passing at 0.05 in the frozen two-endpoint family; supplied adjusted values are cross-checked but never trusted.
9. Exactly 2,000 bootstrap and 2,000 permutation replicates, with at least 90% valid replicates for each procedure and endpoint.

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
- `paired primary results path is missing`
- `hashed detailed split audit path is missing`
- `analysis status path is missing`

Therefore, no claim of transformation of postprandial response, precision readiness,
clinical validity, direct response validity or external validity is permitted. No real
participant-by-meal validation experiment is represented as completed. Phase 3 must
run `code/src/scripts/check_claim_policy.py` against every manuscript/build input using
the bound decision and `claim_policy.json`; a violation or artifact-hash mismatch fails
the check. Task 5 did not check or revise the current manuscript.
