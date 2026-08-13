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
| Attribute-level digital gut twin generated in Task 5 | Synthetic data with prespecified truth | Deterministically generated in `results/phase2/source-data/synthetic_identifiability_stress_test/` | `synthetic_identifiability_stress_test` | Executed after implementation and test review; metrics and seeds are frozen in the accompanying manifest | Identifiability under the stated data-generating process, cap enforcement, comparator behavior and pipeline reproducibility | Clinical validity, external validity, precision readiness, real-world transformation of postprandial responses |
| Task 1 method-lock contract and Task 3 benchmark implementation | Method/provenance infrastructure | Repository-local implementation | Computational feasibility | Implementation exists; no eligible real run-level outcome manifest and paired primary result table are available | Fail-closed readiness and reproducible analysis design | Completed external validation or empirical endpoint improvement |

## Frozen direct-validity requirements

The gate returns `direct_external_validity` only when one eligible run jointly
establishes all of the following:

1. Observed participant-by-meal outcomes with verified provenance and microbiome linkage.
2. A passed run-level method lock and lowercase SHA-256 digests for the method-lock and outcome manifests.
3. Disjoint development and test subject identifiers.
4. Subject-held-out primary analyses for both `glucose_iAUC_2h` and `tg_6h_rise`.
5. Locked attribute GMNPS versus `fcs_microbiome` RMSE improvements in the favourable direction, with 95% paired confidence intervals excluding zero.
6. Completed Holm-adjusted tests at adjusted *P* <= 0.05 within the frozen two-endpoint family.
7. Exactly 2,000 bootstrap and 2,000 permutation replicates, with at least 90% valid replicates for each procedure and endpoint.

GMrepo, ZOE ranks, knowledge graphs, simulations and predictor-only resources are
supporting evidence and cannot satisfy or upgrade any of these requirements.

## Current gate outcome

**Tier: `computational_feasibility` (fail closed).**

The executable gate reports the following blockers:

- `eligible real observed participant-by-meal outcome manifest is missing`
- `locked subject-held-out primary paired-results table is missing`

Therefore, no claim of transformation of postprandial response, precision readiness,
clinical validity, direct response validity or external validity is permitted. No real
participant-by-meal validation experiment is represented as completed.
