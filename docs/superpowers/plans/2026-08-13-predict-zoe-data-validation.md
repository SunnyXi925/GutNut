# PREDICT/ZOE Data Audit and Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish an auditable data boundary and test the locked attribute-level GMNPS against independent person-by-meal metabolic responses when eligible real data are available, while clearly separating biological consistency and simulation evidence.

**Architecture:** Every local and external dataset is classified before analysis as real individual-level validation data, aggregate biological-consistency data, controlled/restricted data, or synthetic data. Model fitting, safety-parameter selection and endpoint evaluation use non-overlapping subject partitions. Direct validity is assessed only on real person-by-meal outcomes; GMrepo, ZOE ranks, knowledge graphs and simulations remain supporting analyses.

**Tech Stack:** Python, pandas, pyarrow, scikit-learn, scipy, official Nature supplementary files, repository/accession metadata, ENA-compatible microbiome files and the locked Phase 1 scoring API.

## Global Constraints

- The Phase 1 scoring rule and primary endpoints must be frozen before any external test labels are inspected.
- Phase 2 may inventory metadata and reconstruct predictor-only resources before a run instance exists, but it must fail closed before any outcome or label-bearing table is opened unless a real run-level method-lock instance has been generated from frozen fit/scoring inputs and independently verified.
- Local fields documented as synthetic or statistically anchored must never be presented as observed PREDICT outcomes.
- The uploaded `41586_2025_9854_MOESM3_ESM.xlsx` is aggregate supplementary evidence unless a sheet demonstrably contains participant-level outcomes.
- GMrepo disease labels are an external biological consistency analysis, not external validation of person-food scores.
- Knowledge-graph label recovery is mechanistic consistency, not construct or clinical validity.
- Direct validation requires real `participant × meal/food × postprandial response` records and subject-level microbiome profiles linkable without leakage.
- Primary endpoints are glucose 2-h incremental area under the curve and triglyceride 6-h rise; C-peptide 2-h incremental area under the curve is secondary if officially available.
- Primary generalization uses subject-held-out evaluation; food/meal-held-out and cohort-held-out tests are reported when the data support them.
- Candidate selection, missingness rules, endpoint selection and hyperparameter tuning occur only within development/training folds.
- Report absolute performance, paired incremental effect versus every comparator, 95% confidence intervals, calibration and permutation nulls.
- Do not claim clinical utility, causal effects or real-time dietary guidance.
- As of 2026-08-13, public resources do not contain eligible person-by-meal response values. `glucose_iAUC_2h`, `tg_6h_rise` and `c_peptide_iAUC_2h` in the local PREDICT directory are synthetic and remain stress-test data.
- The controlled participant-level clinical resource is Zenodo concept DOI `10.5281/zenodo.17236382`, version DOI `10.5281/zenodo.17236383`; direct validation remains blocked until access is granted and the requested endpoints, meal identifiers, units and microbiome linkage keys are confirmed.
- Public PREDICT-1 microbiome evidence includes ENA `PRJEB39223` and ExperimentHub `EH5458`; PREDICT-2/3 sequencing includes `PRJEB75460`, `PRJEB75462`, `PRJEB75463` and `PRJEB75464`. These support biological consistency, not person-meal response validity.
- Synthetic and aggregate supplementary labels cannot bypass the direct-validation method-lock gate. They remain stress-test or supporting evidence even when their files are locally available.
- When controlled outcome resources are unavailable, record the direct-validation path as blocked with its access barrier; code and documentation must not fabricate values.

---

### Task 1: Provenance Inventory and Method-Lock Contract

**Files:**
- Create: `code/src/gmnps/validation/method_lock_gate.py`
- Create: `docs/data/data_availability_audit.md`
- Create: `docs/data/predict_zoe_source_manifest.csv`
- Create: `docs/data/predict1_real_vs_synthetic_audit.csv`
- Create: `docs/data/eligible_for_main_validation.csv`
- Create: `code/src/gmnps/data_sources/predict_zoe_registry.py`
- Test: `code/src/tests/test_method_lock_gate.py`
- Test: `code/src/tests/test_predict_zoe_registry.py`

- [ ] **Step 1: Write failing provenance and contract tests** requiring stable source URL or accession, owner, cohort, unit of observation, sample size, data modality, declared response endpoint, access route, local path, checksum, real/synthetic status, allowed analytical role and explicit exclusion reason. The gate tests define the generator/validator contract for artifact paths, canonical hashes, fit/scoring ID disjointness, schema validation and fail-closed errors.
- [ ] **Step 2: Verify RED** with the focused pytest command.
- [ ] **Step 3: Build the provenance inventory without opening outcome values**. Audit Data Availability and Code Availability statements, supplementary workbook metadata, referenced repositories and local provenance. Record inaccessible controlled data as controlled; do not invent identifiers or response values.
- [ ] **Step 3a: Verify and record known identifiers**: official supplementary workbook checksum, `EH5458`, `PRJEB39223`, PREDICT-2/3 ENA accessions, Zenodo `10.5281/zenodo.15307999`, code/archive `10.5281/zenodo.17236261`, controlled data `10.5281/zenodo.17236382`, and fixed code release `SegataLab/inverse_var_weight` v1.0.0 commit `ab1a974bf1abd66175d190b55183d2957224b39f`.
- [ ] **Step 4: Implement the registry, four audit artifacts and method-lock generator/validator contract**. Task 1 does not generate or require a run-level instance because the real held-out scoring beta and food bundle do not exist yet. The implementation accepts explicit artifact paths later in Task 2 and has no permission to open outcome-bearing tables itself.
- [ ] **Step 5: Commit** `data: audit PREDICT and ZOE validation eligibility`.

### Task 2: Predictor-First Reconstruction and Outcome Gate

**Files:**
- Create: `code/src/scripts/fetch_official_predict_zoe.py`
- Create: `code/src/gmnps/data_sources/predict_zoe_loader.py`
- Create: `code/src/configs/person_meal_validation.yaml`
- Create: `docs/data/predict_zoe_reconstruction_log.md`
- Generate: `results/phase2/method_lock_manifest.json` (real run artifact; never a placeholder)
- Test: `code/src/tests/test_predict_zoe_loader.py`
- Test: `code/src/tests/test_method_lock_gate.py`

- [ ] **Step 1: Write failing staged-DAG tests** for predictor/outcome access separation, checksum verification, schema validation, subject/meal key uniqueness, units, time windows, frozen endpoint configuration, fit/scoring participant disjointness, manifest generation and refusal to open outcome-bearing tables before the gate.
- [ ] **Step 2: Verify RED** with the focused pytest command.
- [ ] **Stage 2A: predictor-only acquisition and reconstruction**. Fetch only official resources needed for microbiome, participant, meal and food predictors. This stage must not read outcome or label tables, including local synthetic columns and aggregate supplementary labels.
- [ ] **Stage 2B: freeze scoring inputs and analysis config**. From predictor-only resources, create the held-out scoring beta and production-eligible food bundle, prove development/scoring participant disjointness, and create `person_meal_validation.yaml` with endpoint names, units, time windows, missingness rules, split policy and deterministic seeds. Freeze and hash this config before any outcome access.
- [ ] **Stage 2C: generate and validate the real method-lock instance**. Use the Task 1 contract and actual development beta, frozen normalization state, held-out scoring beta, food bundle, implementation bytes and frozen config to write `method_lock_manifest.json`. Record real `development_beta_sha256`, `normalization_state_fingerprint` and `scoring_beta_sha256`; recompute all digests and fail closed on any mismatch or overlap.
- [ ] **Stage 2D: unlock outcome loader after gate success**. Only after the gate succeeds may the loader open eligible real outcome-bearing tables and compare reconstructed cohort counts and descriptive statistics. Synthetic and aggregate supplementary labels cannot bypass the direct-validation gate. If controlled outcomes remain unavailable, record the direct-validation path as blocked with the access barrier and stop; the loader must not fabricate values.
- [ ] **Step 5: Commit** `data: reconstruct eligible official PREDICT ZOE resources`.

### Task 3: Locked Cohort Split and Baseline Benchmark

**Files:**
- Create: `code/src/gmnps/validation/cohort_split.py`
- Create: `code/src/gmnps/validation/person_meal_benchmark.py`
- Consume: frozen `code/src/configs/person_meal_validation.yaml` from Task 2
- Test: `code/src/tests/test_cohort_split.py`
- Test: `code/src/tests/test_person_meal_benchmark.py`

- [ ] **Step 1: Consume the frozen `person_meal_validation.yaml` and write failing benchmark tests** for family/twin-aware subject grouping, no participant overlap, fit-state isolation, pre-specified endpoints, nested tuning and deterministic seeds. Task 3 may not define, rename or tune endpoints for the first time.
- [ ] **Step 2: Verify RED** with focused pytest commands.
- [ ] **Step 3: Implement comparators on identical splits:** clinical/demographic/diet baseline; Food Compass only; microbiome only; Food Compass plus microbiome; legacy final-score offset; locked attribute-level GMNPS; random microbiome; shuffled attribute mapping.
- [ ] **Step 4: Implement regression and ranking metrics** including MAE/RMSE, Spearman correlation, calibration slope/intercept, paired bootstrap confidence intervals and participant-level permutation tests.
- [ ] **Step 5: Commit** `feat: add leakage-safe person meal validation benchmark`.

### Task 4: Population Safety, Biological Consistency and Robustness

**Files:**
- Create: `code/src/gmnps/validation/attribute_population_safety.py`
- Create: `code/src/gmnps/validation/biological_consistency.py`
- Create: `code/src/scripts/run_attribute_validation.py`
- Test: `code/src/tests/test_attribute_population_safety.py`
- Test: `code/src/tests/test_biological_consistency.py`

- [ ] **Step 1: Write failing tests** for FCS food-group/subgroup convergence, between-group discrimination, category transition matrices, disease/cohort stratification, shuffled controls and bootstrap intervals.
- [ ] **Step 2: Verify RED** with focused pytest commands.
- [ ] **Step 3: Implement population safety as a design audit**, not the main empirical finding. Report global rank correlation but emphasize group distributions and clinically implausible reversals.
- [ ] **Step 4: Implement supporting GMrepo, ZOE rank and knowledge-path analyses** with explicit labels `biological_consistency` or `mechanistic_consistency`.
- [ ] **Step 5: Commit** `feat: validate population safety and biological consistency`.

### Task 5: Digital Gut Twin Stress Test and Evidence Gate

**Files:**
- Modify: `code/src/gmnps/validation/synthetic_twin.py`
- Create: `code/src/gmnps/validation/evidence_gate.py`
- Create: `docs/data/validation_evidence_matrix.md`
- Test: `code/src/tests/test_attribute_synthetic_twin.py`
- Test: `code/src/tests/test_evidence_gate.py`

- [ ] **Step 1: Write failing tests** for pre-specified synthetic truth, independent noise, recovery metrics, shuffled-control failure and evidence-tier labels.
- [ ] **Step 2: Verify RED** with focused pytest commands.
- [ ] **Step 3: Extend the simulator to generate attribute-level microbiome-conditioned responses** and benchmark the same locked comparators. Simulation demonstrates identifiability and pipeline behavior only.
- [ ] **Step 4: Implement the evidence gate:** `direct_external_validity` only when eligible real response data pass provenance, independence and incremental-performance checks; otherwise the manuscript is automatically restricted to `computational_feasibility`.
- [ ] **Step 5: Run the full validation suite, freeze source-data tables and commit** `feat: gate claims on direct response validity`.

## Phase Exit Criteria

- Every analysed field has a real/synthetic/access classification and stable provenance.
- Direct response validation is either completed with real independent outcomes or explicitly marked unavailable with the exact access barrier.
- All comparisons use the same splits and input opportunities, with confidence intervals and permutation controls.
- The evidence gate determines title, abstract and Discussion claim strength before writing begins.
