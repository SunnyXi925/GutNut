# Attribute-level GMNPS Method Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the primary final-score-offset formulation with a locked, Food Compass 2.0-compatible attribute-level personalized calibration that preserves the full population-level NPS baseline while recalculating only evidence-supported attributes.

**Architecture:** The official Food Compass 2.0 score is converted to its native unscaled score. Food nutrient amounts are expressed per 100 kcal and scored with the published attribute rules. Because the current microbiome-derived beta values are finite-difference health-index responses rather than measured bioavailability, standardized beta values produce bounded adjustments to Food Compass attribute points, modulated by the food's normalized nutrient exposure and an explicit nutrient-to-attribute allocation matrix. The affected domains are recomputed using the original top-5, top-3, half-weight and truncation rules. Unavailable or non-personalized Food Compass attributes remain in a fixed residual, so zero calibration reproduces the official baseline exactly. The existing final-score-offset model remains a labelled sensitivity comparator.

**Tech Stack:** Python 3.9+, pandas, numpy, PyYAML, pytest, existing `gmnps.beta_i`, `gmnps.scoring`, FNDDS/FoodData Central inputs and Food Compass 2.0 supplementary rules.

## Global Constraints

- Food Compass 2.0 is a multi-attribute population-level baseline NPS, not the scientific outcome and not a final score to which an arbitrary offset is added.
- The primary score recalculates Food Compass-compatible attribute and domain contributions before final scaling.
- Personalization is restricted to attributes with literature support and the expert-reviewed dual-channel mapping; zinc, copper and vitamin A RAE are not microbiome calibration variables, while retinol is retained as a component-level lipid marker.
- Carbohydrate is fixed in the primary fiber-to-carbohydrate ratio and is allowed as a calibration proxy only in sensitivity analysis.
- No per-food or per-population score centering is permitted in the primary method.
- Beta normalization parameters and calibration strength are learned or selected on development data only, then frozen before validation.
- Calibration strength may be selected by population-safety constraints only; no response endpoint, disease label, desired non-zero drift or desired rank shift may enter candidate selection.
- Primary attribute-point adjustment is bounded to 20% of each attribute's published point range; 10% and 30% are sensitivity analyses. Final individual score deviation is bounded to `[-12, 12]`, with `[-8, 8]` and `[-15, 15]` sensitivity analyses.
- The native Food Compass unscaled range is truncated to `[-12.1, 35.0]` and scaled to `[1, 100]` using the published equation.
- Zero personalized calibration must reproduce the supplied official Food Compass 2.0 score to numerical tolerance `1e-8` before integer display rounding.
- All method manifests must record source file hashes, FNDDS release, nutrient units, missingness, mapping version, beta normalization fit cohort and software version.
- Existing user changes in the working tree must be preserved; do not revert or rewrite unrelated modules.
- Per-food effective attribute weights are first-class provenance. Dairy
  `unsaturated_to_saturated_fat_ratio` uses 0.5, making the all-calculated
  nutrient-ratio denominator 2.5; non-dairy uses 1.0 and denominator 3.0.
- Ratio applicability is recomputed from the same food exposure bytes used for
  calibration; CLI serialization masks cannot determine gate status.
- Production food bundles are created only by the same-immutable-bytes verified
  loader. Direct DataFrame construction is development/non-production only.
- The method implementation lock and run-level release-registry snapshot are
  separate. Registry population in Phase 2 does not change the method version.
- Primary food composition must use FNDDS/FoodData Central releases aligned with the FCS2 food table (FNDDS 2001-2018); the local FNDDS 2021-2023 matrix is allowed only for development smoke tests until release alignment is complete.
- Nutrient values are converted from per 100 g to per 100 kcal before attribute scoring. Total sugars must not substitute for added sugar, rounded NOVA classes must not substitute for the original energy-weighted mixed-dish NOVA value, and `18:3` must not be assumed to be ALA unless its source definition is verified.

---

## File Structure

- Create `code/src/gmnps/scoring/fcs2_attribute_rules.py`: published attribute point functions, domain aggregation and FCS scaling.
- Create `code/src/gmnps/scoring/fcs2_attribute_mapping.py`: nutrient-to-attribute mappings, channel labels and primary/sensitivity eligibility.
- Create `code/src/gmnps/scoring/attribute_calibration.py`: development-fitted beta normalization, exposure modulation, allocation and bounded attribute-point adjustment.
- Create `code/src/gmnps/scoring/attribute_recomposition.py`: fixed-residual reconstruction and personalized domain recomposition.
- Create `code/src/gmnps/scoring/attribute_gmnps.py`: public scoring API and output tables.
- Create `code/src/scripts/run_attribute_gmnps.py`: chunked reproducible runner and manifests.
- Create `code/src/configs/attribute_gmnps.yaml`: locked primary and sensitivity parameters.
- Modify `code/src/gmnps/scoring/__init__.py`: export the new primary API without removing legacy exports.
- Create focused tests under `code/src/tests/test_fcs2_attribute_*.py` and `code/src/tests/test_attribute_gmnps.py`.

### Task 1: Published Food Compass Attribute Engine

**Files:**
- Create: `code/src/gmnps/scoring/fcs2_attribute_rules.py`
- Test: `code/src/tests/test_fcs2_attribute_rules.py`

**Interfaces:**
- Produces `AttributeRule`, `FCS2_RULES`, `to_per_100_kcal`, `score_attribute`, `aggregate_domains`, `fcs_to_unscaled`, and `unscaled_to_fcs`.
- The registry represents 54 conceptual FCS2 attributes as 56 operational Table S10 rows because fruits and non-starchy vegetables each have separate dried and non-dried thresholds. Iodine and trans fat are explicitly inactive in the reported USDA analysis, leaving 54 active operational scoring rows.

- [ ] **Step 1: Write failing tests** for the published low/high targets, added-sugar bins, NOVA ordering, ratio gates, dairy half-weight, top-5 vitamins/minerals, top-3 specific lipids, half-weighted attributes/domains, truncation and exact inverse scaling. Represent the Table S10 nitrite inconsistency explicitly as primary `footnote_50` and sensitivity `table_25` rules.
- [ ] **Step 2: Verify RED** with `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests/test_fcs2_attribute_rules.py -q`; expected failure is `ModuleNotFoundError`.
- [ ] **Step 3: Implement the immutable registry and pure scoring functions**. Linear rules must clip to their stated point range; ratio rules must use the published log scale and minimum-exposure gates; aggregation must preserve signed absolute-value selection for top-k domains.
- [ ] **Step 4: Verify GREEN** with the focused test and then `PYTHONPATH=code/src .venv/bin/python -m pytest code/src/tests -q`; expected result is all tests passing with no warnings.
- [ ] **Step 5: Commit** `feat: implement Food Compass attribute scoring rules`.

### Task 2: Expert-reviewed Attribute Mapping

**Files:**
- Create: `code/src/gmnps/scoring/fcs2_attribute_mapping.py`
- Test: `code/src/tests/test_fcs2_attribute_mapping.py`

**Interfaces:**
- Produces `AttributeCalibrationMapping`, `PRIMARY_ATTRIBUTE_MAPPINGS`, `SENSITIVITY_ATTRIBUTE_MAPPINGS`, `validate_attribute_mappings`, `build_allocation_matrix`, and `build_food_specific_response`.
- Direct mappings: vitamin C, vitamin E, vitamin K1, magnesium, potassium, cobalamin, choline, cholesterol and total fiber.
- Composite mappings: carotenoid components to total carotenoids; food folate to the food-folate fraction of folate DFE; retinol to the retinol fraction of vitamin A RAE; C8:0+C10:0+C12:0 to MCFA.
- Ratio mappings: fiber modifies the numerator of fiber:carbohydrate; potassium modifies the numerator of potassium:sodium; saturated-fat response modifies the denominator of unsaturated:saturated fat. The unsaturated side remains fixed in the primary analysis because MUFA/PUFA were not included in the expert-reviewed calibration mask; adding an unreviewed unsaturated beta for mathematical symmetry is not permitted.
- Total fat supplies the published ratio eligibility gate but is not an invented independent Food Compass attribute. C4:0, C6:0, C14:0, C16:0 and C18:0 remain explanatory subcomponents of the saturated-fat response and do not add duplicate points.

- [ ] **Step 1: Write failing tests** asserting that every primary mapping targets an existing Food Compass attribute, primary mappings exclude carbohydrate, zinc, copper and vitamin A RAE as calibration variables, retinol is present only as a component mapping, and each nutrient's allocation weights sum to one across all target attributes. For fiber and potassium, primary allocation is split equally between ratio and absolute attributes; all-to-ratio and all-to-absolute variants are sensitivity analyses.
- [ ] **Step 2: Verify RED** with the focused pytest command.
- [ ] **Step 3: Implement the mapping registry and validation**. Food-specific responses must use per-100-kcal exposure normalized to the published attribute threshold. Zero-total components return zero effect with an explicit diagnostic flag. A nutrient may influence several native FCS attributes only through its allocation weights; it must never receive full weight in every representation.
- [ ] **Step 3a: Emit a reconstruction-status table** separating nutrient-derived attributes, attributes requiring ingredient/processing data, and fixed residual attributes. Missing inputs must remain unavailable or residualized; they must not be silently set to zero.
- [ ] **Step 4: Verify GREEN** with focused and full tests.
- [ ] **Step 5: Commit** `feat: map microbiome channels to Food Compass attributes`.

### Task 3: Locked Beta Normalization and Attribute-point Calibration

**Files:**
- Create: `code/src/gmnps/scoring/attribute_calibration.py`
- Test: `code/src/tests/test_attribute_calibration.py`

**Interfaces:**
- Produces `BetaNormalizationState`, `fit_beta_normalization`, `transform_beta`, `attribute_response`, and `calibrate_attribute_points`.
- Development normalization is nutrient-wise median/MAD with IQR and standard-deviation fallback. The transformed response is `tanh(z/2)`.
- For attribute `a`, `r_ija = sum_k(pi_ka * q_jk * z_ik)`, where `pi_ka` is the locked allocation weight, `q_jk` is per-100-kcal food exposure normalized to the relevant FCS threshold, and `z_ik` is the frozen beta transform. The personalized point is `clip(p0_ja + lambda_a * tanh(r_ija / 2), L_a, U_a)`. Primary `lambda_a` is 20% of `U_a-L_a`.

- [ ] **Step 1: Write failing tests** for train-only fit state, deterministic transform on held-out rows, fallback scales, finite output, point-adjustment bounds, locked nutrient allocation, food-dose modulation, identity at beta zero and absence of cohort-dependent per-food centering.
- [ ] **Step 2: Verify RED** with the focused pytest command.
- [ ] **Step 3: Implement normalization and attribute calibration** without reading disease labels, response outcomes or validation data. Do not describe beta as bioavailability or alter measured nutrient amounts.
- [ ] **Step 4: Verify GREEN** with focused and full tests.
- [ ] **Step 5: Commit** `feat: add bounded microbiome effective exposure`.

### Task 4: Native-domain Recomposition with Fixed Residual

**Files:**
- Create: `code/src/gmnps/scoring/attribute_recomposition.py`
- Test: `code/src/tests/test_attribute_recomposition.py`

**Interfaces:**
- Produces `BaselineDecomposition`, `decompose_official_baseline`, `recompute_personalized_domains`, and `compose_personalized_fcs`.
- For food `j`, calculate `U0_j = fcs_to_unscaled(FCS2_j)`, local baseline domain sum `L0_j`, and fixed residual `Q_j = U0_j - L0_j`.
- For individual `i`, recompute affected attribute points and all affected top-k domains to obtain `Lij`, then calculate `Uij = clip(Q_j + Lij, -12.1, 35.0)` and scale to `[1,100]`.
- The final deviation safety bound is applied relative to `FCS2_j` after native recomposition and before `[1,100]` clipping; no centering is applied.

- [ ] **Step 1: Write failing tests** proving exact zero-effect recovery, top-5/top-3 membership changes when calibrated points cross, fixed-domain invariance, published scaling, deviation bounds and absence of forced population centering.
- [ ] **Step 2: Verify RED** with the focused pytest command.
- [ ] **Step 3: Implement decomposition and recomposition** with per-attribute and per-domain audit tables.
- [ ] **Step 4: Verify GREEN** with focused and full tests.
- [ ] **Step 5: Commit** `feat: recompose personalized scores in native FCS domains`.

### Task 5: Primary Scoring API, Chunked Runner and Legacy Comparator

**Files:**
- Create: `code/src/gmnps/scoring/attribute_gmnps.py`
- Create: `code/src/scripts/run_attribute_gmnps.py`
- Create: `code/src/configs/attribute_gmnps.yaml`
- Modify: `code/src/gmnps/scoring/__init__.py`
- Test: `code/src/tests/test_attribute_gmnps.py`
- Test: `code/src/tests/test_run_attribute_gmnps.py`

**Interfaces:**
- Produces `AttributeGMNPSConfig`, `fit_attribute_gmnps`, `score_attribute_gmnps`, and the required individual-food, food-summary, attribute-attribution, domain-attribution and run-manifest outputs.
- Required individual-food fields retain `individual_id`, `food_id`, `food_name`, `food_group`, `FCS2`, `GMNPS_delta`, `GMNPS_score`, `MAC_delta`, `LIPID_delta`, drivers, mapping version and scoring version.
- The runner exposes `attribute_recomposition` as primary and `legacy_final_score_offset` as a sensitivity comparator only.

- [ ] **Step 1: Write failing API and CLI tests** using a tiny synthetic nutrient table and beta matrix; test chunk equivalence, deterministic ordering, required columns, provenance fields, score bounds, zero-effect recovery and legacy labelling.
- [ ] **Step 2: Verify RED** with focused pytest commands.
- [ ] **Step 3: Implement the API and runner**. The YAML primary values are `attribute_point_fraction_cap: 0.20`, `attribute_point_fraction_sensitivity: [0.10, 0.30]`, `final_delta_cap: 12`, `beta_temperature: 2`, `carbohydrate_policy: sensitivity_proxy_only`, `primary_method: attribute_recomposition` and final-cap sensitivities `8` and `15`.
- [ ] **Step 4: Verify GREEN** with focused and full tests.
- [ ] **Step 5: Commit** `feat: expose attribute-level GMNPS scoring pipeline`.

### Task 6: Method Lock, Reproducibility Audit and Migration Note

**Files:**
- Create: `docs/methods/attribute_level_gmnps_spec.md`
- Create: `docs/methods/attribute_mapping_table.csv`
- Create: `docs/methods/method_lock_manifest.schema.json`
- Create: `code/src/tests/test_attribute_method_lock.py`

**Interfaces:**
- Documents the frozen mathematical definitions, all FCS2 source-table values, mappings, unavailable attributes, development-only fit boundary, sensitivity variants and the reason the legacy final-score-offset formulation is non-primary.

- [ ] **Step 1: Write a failing lock-manifest test** requiring method version, source hashes, FNDDS release, beta fit cohort, calibration fit IDs hash, mapping version, all fixed parameter values and validation embargo flag.
- [ ] **Step 2: Verify RED** with the focused pytest command.
- [ ] **Step 3: Add the method specification, mapping table and JSON schema** with no outcome-dependent acceptance criteria.
- [ ] **Step 4: Run all 204 pre-existing tests plus new tests**, then execute a small scoring smoke test and compare zero-effect output with official FCS2 to `1e-8`.
- [ ] **Step 5: Commit** `docs: lock attribute-level GMNPS method`.

### Final review amendments

- [ ] Carry `effective_attribute_weights` through Task 1, the food bundle,
  Task 3 fingerprints, Task 4 decomposition/recomposition, API, CLI and audits;
  verify zero and non-zero dairy oracles independently.
- [ ] Reuse the Task 1 ratio-gate implementation at the bundle boundary and
  enforce gate fail iff canonical `NOT_CALCULATED`, gate pass iff finite point.
- [ ] Replace caller-authored production DataFrames with a verified byte loader
  binding source-byte digests, canonical parsed-content fingerprints and one
  approved registry entry; retain empty-registry fail-closed behavior.
- [ ] Treat the release-registry snapshot SHA-256, canonical set and approved
  entry as run provenance, while locking the runner as implementation source.
- [ ] Require Phase 2 to bind and recompute the external method-lock schema hash,
  registry snapshot hash and future `method_lock_gate.py` implementation hash
  at both the outcome loader and Task 3 entry.
- [ ] Use the unique Task 1 top-k selector in Task 4 and record
  `score_centering: none` separately from
  `beta_centering: development_median`.

## Phase Exit Criteria

- The primary code path no longer computes an arbitrary raw response followed by `FCS2 + centered offset`.
- Zero effect exactly recovers official FCS2, while non-zero effects are generated inside published attribute/domain logic.
- No test or parameter selection forces a target rank shift, mean shift or response advantage.
- All tests pass, task reviews are clean and a whole-phase scientific/code review approves progression to data validation.
