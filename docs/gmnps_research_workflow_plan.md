# GMNPS Nature Food Research Workflow Plan

Working title: **Personalized calibration transforms nutrient profiling systems for precision nutrition**

This plan reorganizes the GMNPS project around the central scientific claim:

> Nutrient profiling systems can meet precision-nutrition needs by retaining a population-level food-quality prior while adding a bounded, interpretable, microbiome-informed personalized deviation.

Food Compass 2.0 is used only as the anchored baseline NPS prior. The article should not be framed as an evaluation or replacement of Food Compass 2.0. GMNPS is the proposed personalized calibration framework.

## Material Passport

- Origin Skill: academic-research-suite
- Workflow Route: academic-pipeline plus experiment-agent plan mode
- Date: 2026-08-02
- Verification Status: PLANNED
- Primary Workspace: `.`
- Data Root: `./data`
- Expert Review Root: `./expert_review`
- Current Stage: Research design and evidence-chain reconstruction before manuscript drafting

## 1. Article Logic

### 1.1 Core Research Question

Can a nutrient profiling system be transformed for precision nutrition by adding a bounded personalized calibration term that preserves population-level NPS consensus while revealing biologically interpretable individual heterogeneity?

### 1.2 Primary Hypothesis

An anchored score of the form

```text
GMNPS_ij = clip_1-100(NPS_j + D_ij)
```

will preserve food-level NPS ranking at the population mean while allowing individual-food deviations that are structured by gut microbiome-derived nutrient response profiles.

### 1.3 Secondary Hypotheses

1. Population-mean GMNPS will remain highly concordant with the baseline NPS prior.
2. GMNPS will show individual-level variance that static NPS scores cannot express.
3. MAC and LIPID channels will show food-group-specific heterogeneity patterns.
4. External microbiome-health and microbiome-diet resources will support retrospective biological plausibility, especially for cardiometabolic phenotypes.
5. Expert review will support the biological interpretability and guideline consistency of channel definitions, food-group labels and case-level outputs.

### 1.4 Claim Boundary

Supported claim:

```text
GMNPS demonstrates computational feasibility and biological plausibility for transforming NPS into a precision-nutrition-ready framework.
```

Unsupported claims unless future clinical or controlled retrospective data are obtained:

- GMNPS improves clinical outcomes.
- GMNPS is a disease-agnostic general health predictor.
- GMNPS provides clinical dietary prescriptions without clinician review.
- Gut microbiome causally determines the effect of each food.

## 2. Evidence Architecture

The article should be organized as a sequence of evidence modules.

| Evidence module | Main question | Main data | Manuscript position |
| --- | --- | --- | --- |
| E0. Provenance and data audit | Which assets are credible enough for claims? | FCS2, FNDDS, L9, GMrepo, ZOE, CRA013939, expert review | Methods and Supplement |
| E1. Anchored score definition | What is personalized calibration? | Scoring equations and masks | Results 1, Fig. 1 |
| E2. NPS preservation | Does GMNPS remain an NPS? | L9 food summary and individual-food scores | Results 2, Fig. 2 |
| E3. Personalized heterogeneity | Does calibration reveal non-random individual variation? | L9 individual-food matrices, MAC/LIPID channels | Results 3, Fig. 3 |
| E4. External biological plausibility | Does the microbiome layer align with external health/diet signals? | GMrepo v2, ZOE 2025, CRA013939, AGP stress test | Results 4, Fig. 4 or Supplement |
| E5. Digital-gut-twin validation | Does the design work under known ground truth? | Synthetic benchmark | Results 5, Fig. 5 or Fig. 4 panel |
| E6. Expert consensus review | Are channels and outputs interpretable to domain experts? | C1, C3, E4 expert review | Results 6 or Supplement; Discussion |

## 3. Data Asset Triage

### 3.1 Main-Article-Ready or Near-Ready Assets

#### Food Compass 2.0 and FCS2 Table S5

Path:

- `./FCS2.0_data.pdf`
- `./outputs/fcs2/table_s5.csv`
- `./outputs/fcs2/table_s5_audit.json`

Use:

- Baseline NPS prior.
- Food universe and food-code metadata.
- QC and provenance in Supplement.

Risk:

- PDF extraction should not occupy main-figure space.
- If possible, replace or cross-check with an official spreadsheet.

#### L9 Food Compass GMNPS Outputs

Path:

- `./data/project_data/predict_multi/L9_food_compass/S_food_summary_v3.csv`
- `./data/project_data/predict_multi/L9_food_compass/S_score_individual_x_food_v3.parquet`
- `./data/project_data/predict_multi/L9_food_compass/S_mac_individual_x_food_v3.parquet`
- `./data/project_data/predict_multi/L9_food_compass/S_lipid_individual_x_food_v3.parquet`
- `./data/project_data/predict_multi/L9_food_compass/scoring_manifest_v3.json`

Use:

- Core evidence for NPS preservation and individual heterogeneity.
- 15,492 individuals by 9,234 foods can support high-density main figures.

Critical limitation:

- `scoring_manifest_v3.json` uses the old 15+15 nutrient masks and still includes carbohydrate, zinc, copper and vitamin A RAE.
- Therefore v3 must become sensitivity or historical comparison.
- Primary analysis must be rerun as **v4 expert-revised masks**:
  - Carbohydrate: sensitivity proxy only.
  - Zinc and copper: removed from primary MAC.
  - Vitamin A RAE: removed from LIPID.
  - Retinol: retained in LIPID.

#### GMrepo External Validation v2

Path:

- `./data/project_data/verification/e1_global_v2/`

Use:

- External microbiome plausibility for metabolic phenotypes.
- Strongest reported signals are CAD and T2D.

Claim boundary:

- Supports a cardiometabolic microbiome signal.
- Does not support a general health predictor.
- IBD/CD/UC reversal must be stated as a target-design limitation, not hidden.

#### ZOE 2025 Public Ranking Tables

Path:

- `./data/project_data/verification/zoe2025/`

Use:

- External directionality anchor for microbiome health and diet ranks.
- Support biological plausibility of microbiome-informed calibration.

Claim boundary:

- Public rankings do not replace restricted individual-level diet-response data.
- Use for directionality and plausibility, not personal response prediction.

#### CRA013939 Processed Tables

Path:

- `./data/project_data/CRA013939/processed/`

Use:

- Real public or project-level aggregated evidence for food-genus-metabolite-host linkages.
- Good for mechanism anchoring and China-cohort relevance.

Claim boundary:

- Processed aggregate or linkage tables are not raw individual-level clinical validation.
- Avoid species- or gene-level claims if the underlying data are genus-level or 16S-derived.

### 3.2 Supplementary or Stress-Test Assets

#### American Gut Project

Path:

- `./data/project_data/external/american_gut/`

Use:

- Noisy external stress test.

Boundary:

- Self-reported diet and health labels limit causal or clinical interpretation.

#### Predict1-Style Local Data

Path:

- `./data/predict1_data/`

Use:

- Method demonstration or simulation if provenance remains uncertain.

Boundary:

- Do not present as real PREDICT1 individual-level validation unless provenance is independently verified.

## 4. Expert Review Integration

### 4.1 Expert Panel

Six experts completed two rounds of structured review:

- One expert with clinical background.
- Two experts with food analysis backgrounds:
  - One focused on bioactive compounds.
  - One focused on food biomarkers.
- One expert in population nutrition.
- One expert in nutritional bioactive compounds.
- One expert in probiotics.

Round 1:

- C3 food-group consensus review.
- Result: passed.

Round 2:

- C1 nutrient mask revision.
- Result: passed.

E4:

- Individual recommendation review.
- Result: supports clinical translation potential, while noting that clinical application requires additional safety, disease-stage and implementation considerations.

### 4.2 How To Use Expert Review in the Manuscript

Use expert review as:

- Content validity evidence.
- Construct validity evidence.
- Biological interpretability evidence.
- Guideline-consistency review.
- Clinical translation plausibility.

Do not use expert review as:

- Clinical efficacy evidence.
- Prospective intervention evidence.
- Proof of outcome improvement.
- A substitute for retrospective individual-level validation.

### 4.3 Required Expert Review Outputs

Before manuscript writing, generate a clean expert-review summary table:

```text
expert_id
expertise_domain
round
module
item_id
item_label
rating
comment_category
free_text_comment_redacted
revision_action
final_status
```

Main text should only report:

- Six-expert panel composition.
- Two-round structured expert consensus review.
- Predefined consensus threshold, if confirmed.
- Round 1 C3 passed.
- Round 2 C1 passed after specified mask changes.
- E4 indicated clinical translation potential but not clinical readiness.

Supplementary material should include:

- C1 nutrient list and revisions.
- C3 food-group labels.
- E4 sample-card structure and anonymized examples.
- Agreement rates.
- Disagreement categories.
- Revision log.

### 4.4 Expert-Driven Mask Decision

Primary v4 masks:

MAC:

- Keep: fiber, alpha-carotene, beta-carotene, beta-cryptoxanthin, lycopene, lutein and zeaxanthin, vitamin C, food folate, magnesium, potassium, vitamin K1/phylloquinone, vitamin E.
- Carbohydrate: sensitivity proxy only.
- Remove: zinc, copper.

LIPID:

- Keep: total fat, saturated fat, cholesterol, retinol, C4:0, C6:0, C8:0, C10:0, C12:0, C14:0, C16:0, C18:0, choline, vitamin B12.
- Remove: vitamin A RAE.

Original 15+15 masks:

- Use only as sensitivity analysis.

## 5. Experiment Package

### E0. Data and Provenance Audit

Objective:

Confirm which datasets can support main-text claims.

Inputs:

- `data/`
- `outputs/fcs2/`
- `expert_review/`
- Existing manifests and reports.

Methods:

1. Create a dataset registry with source, access type, rows, columns, provenance, leakage risk and claim level.
2. Separate real public data, project processed data, synthetic data and reconstructed data.
3. Audit all candidate validation datasets for overlap with training datasets.
4. Mark assets as main, supplement, stress test, or excluded.

Outputs:

- `docs/data_claim_registry.md`
- `outputs/audit/data_asset_audit.csv`
- `outputs/audit/leakage_and_provenance_report.md`

Acceptance criteria:

- No dataset is used for a stronger claim than its provenance supports.
- PREDICT-like local data are not presented as real PREDICT validation without proof.
- GMrepo final sample count and leakage-filter version are unified.

### E1. Rebuild Expert-Revised GMNPS v4

Objective:

Rerun primary scoring using expert-revised nutrient masks.

Inputs:

- L8 inference bundle v2.
- L9 food/nutrient matrices.
- FCS2 baseline scores.
- Expert-revised C1 masks.

Methods:

1. Implement `expert_revised_v4` mask config.
2. Recompute total, MAC and LIPID deviations.
3. Apply food-level centering and bounded transform.
4. Export individual-food and food-summary tables.
5. Keep v3 original mask as sensitivity.

Outputs:

- `outputs/gmnps_v4/tables/individual_food_scores.parquet`
- `outputs/gmnps_v4/tables/food_summary.csv`
- `outputs/gmnps_v4/tables/channel_variance.csv`
- `outputs/gmnps_v4/manifests/scoring_manifest_v4.json`
- `outputs/gmnps_v4/qc/mask_revision_audit.csv`

Acceptance criteria:

- All scores in 1-100.
- Mean personalized deviation per food approximately zero before clipping.
- Zinc, copper and vitamin A RAE absent from primary masks.
- Retinol retained.
- v4 preservation remains high.

### E2. NPS Preservation Analysis

Objective:

Show GMNPS remains an NPS after personalization.

Inputs:

- v4 `food_summary.csv`
- FCS2 baseline scores.
- Food-group labels.

Methods:

1. Compute Spearman and Kendall correlation between FCS2 and population-mean GMNPS.
2. Compute category transition matrix.
3. Compute food-group ordering and compare with expected nutritional consensus.
4. Identify any category shifts and manually annotate whether they are interpretable.
5. Compare v4 with v3 original mask and cap sensitivity.

Outputs:

- `outputs/gmnps_v4/validation/nps_preservation_metrics.json`
- `outputs/gmnps_v4/validation/category_transition.csv`
- `outputs/gmnps_v4/validation/food_group_ordering.csv`

Primary success criteria:

- Spearman FCS2 versus GMNPS mean >= 0.90.
- Category shifts are rare and interpretable.
- Population-level ranking does not invert established nutrition consensus.

Main figure:

- Scatter of FCS2 versus population-mean GMNPS.
- Category transition heatmap.
- Food-group ordering panel.

### E3. Personalized Heterogeneity Analysis

Objective:

Show that GMNPS expresses meaningful heterogeneity unavailable to static NPS.

Inputs:

- v4 individual-food score matrix.
- v4 MAC and LIPID channel matrices.
- Food groups and FCS2 baseline.

Methods:

1. Quantify within-food individual variance of GMNPS delta.
2. Compare static FCS2 variance with GMNPS variance.
3. Partition variance into MAC and LIPID channel components.
4. Compare channel dominance by food group.
5. Select representative foods with high heterogeneity but stable population mean.

Outputs:

- `outputs/gmnps_v4/validation/heterogeneity_by_food.csv`
- `outputs/gmnps_v4/validation/heterogeneity_by_group.csv`
- `outputs/gmnps_v4/validation/example_food_intervals.csv`

Acceptance criteria:

- FCS2 has no individual variance by design.
- GMNPS has substantial bounded within-food variance.
- MAC-dominant heterogeneity is enriched in plant-food groups.
- LIPID-dominant heterogeneity is enriched in animal-fat, egg, meat, dairy or mixed groups.

Main figure:

- Heatmap or density plot of individual-food deltas.
- MAC versus LIPID variance by food group.
- Example food intervals around the population prior.

### E4. External Microbiome Plausibility Analysis

Objective:

Test whether the microbiome-derived calibration layer aligns with independent microbiome-health or microbiome-diet signals.

Inputs:

- GMrepo v2 predictions and AUROC outputs.
- ZOE 2025 public health and diet rank tables.
- CRA013939 processed linkage tables.
- American Gut stress-test outputs.

Methods:

GMrepo:

1. Use final v2 leakage-filtered pool only.
2. Report AUROC for cardiometabolic phenotypes separately from inflammatory bowel disease.
3. Compare against Shannon, Simpson, richness and random baselines.
4. Stratify by country and experiment type where sample size permits.

ZOE:

1. Map ZOE SGB rankings to available genus-level features only when taxonomy mapping is auditable.
2. Test whether GMNPS beneficial or adverse microbial directions align with ZOE diet/health rankings.
3. Use rank correlation and sign concordance.

CRA013939:

1. Use processed food-genus-metabolite-host linkage tables as mechanism anchors.
2. Test sign agreement between GMNPS nutrient-microbe directions and observed food-metabolite or genus-metabolite associations.

American Gut:

1. Use as noisy stress test.
2. Report null or weak results honestly.

Outputs:

- `outputs/gmnps_v4/validation/gmrepo_metabolic_validation.csv`
- `outputs/gmnps_v4/validation/zoe_rank_concordance.csv`
- `outputs/gmnps_v4/validation/cra013939_mechanism_concordance.csv`
- `outputs/gmnps_v4/validation/external_validation_limitations.md`

Acceptance criteria:

- GMrepo supports metabolic-axis plausibility, not general disease prediction.
- ZOE rank concordance is directionally positive versus shuffled taxa.
- CRA013939 mechanism anchors are consistent with pre-specified MAC/LIPID interpretation.
- Negative or phenotype-specific findings are preserved in Results or Supplement.

Main or extended-data figure:

- Preservation-personalization-external plausibility panel.
- GMrepo metabolic AUROC panel.
- ZOE/CRA concordance panel if robust.

### E5. Digital-Gut-Twin Benchmark

Objective:

Demonstrate performance under known ground truth.

Inputs:

- Synthetic simulator.
- Real FNDDS/FCS2 nutrient vectors where possible.
- v4 expert-revised masks.

Methods:

1. Simulate microbial MAC and LIPID capacities.
2. Use food nutrient exposures from real food vectors if available.
3. Define host-response ground truth as universal food quality plus microbiome-conditioned effect plus noise.
4. Compare:
   - FCS2 only.
   - Anchored GMNPS.
   - Unanchored microbiome score.
   - Random microbiome.
   - Shuffled microbiome.
   - Shuffled masks.
   - v3 original masks.
   - v4 expert-revised masks.
5. Treat FCS2-only personalized residual recovery as not estimable, not zero.

Outputs:

- `outputs/gmnps_v4/simulation/synthetic_benchmark.csv`
- `outputs/gmnps_v4/simulation/design_ablation.csv`
- `outputs/gmnps_v4/simulation/delta_cap_sensitivity.csv`

Acceptance criteria:

- Anchored GMNPS preserves NPS ranking better than unanchored microbiome score.
- Anchored GMNPS recovers more personalized residual signal than random or shuffled controls.
- v4 expert-revised masks perform stably relative to v3 original masks.

Main figure:

- Trade-off plot: NPS preservation versus personalized recovery.
- Model comparison bars.
- Cap sensitivity inset.

### E6. Structured Expert Consensus Review Analysis

Objective:

Convert C1, C3 and E4 reviews into manuscript-grade evidence.

Inputs:

- `./expert_review/C1_nutrient_masks.xlsx`
- `./expert_review/C3_food_group_consensus.xlsx`
- `./expert_review/E4_individual_recommendations.docx`
- `./expert_review/README_for_experts.pdf`
- Any missing anonymized ratings or summary sheets provided by the team.

Methods:

1. Create anonymized expert metadata table.
2. Create item-level agreement matrix.
3. Summarize round 1 C3 agreement and passed status.
4. Summarize round 2 C1 revision and passed status.
5. Summarize E4 clinical translation potential and clinical-readiness caveats.
6. Report percent agreement and, if feasible, Fleiss' kappa.

Outputs:

- `outputs/expert_review/expert_panel_summary.csv`
- `outputs/expert_review/c1_mask_agreement.csv`
- `outputs/expert_review/c3_food_group_agreement.csv`
- `outputs/expert_review/e4_case_review_summary.csv`
- `outputs/expert_review/expert_review_methods_summary.md`

Acceptance criteria:

- Expert ratings are auditable.
- Main manuscript reports only concise expert-review results.
- Supplement includes complete item-level details.
- E4 is framed as plausibility and translational potential, not clinical efficacy.

## 6. Figure Plan

### Figure 1. Personalized Calibration Framework

Purpose:

Show the conceptual transformation from static NPS to anchored personalized calibration.

Panels:

- a. Population prior plus bounded personalized deviation.
- b. MAC and LIPID microbiome-informed calibration channels.
- c. Expert-reviewed food groups: universally healthful, universally unhealthful, microbiome-conditioned.
- d. Score interpretation: food-level prior retained, individual deviations shown around it.

Data status:

- Schematic plus expert-reviewed categories.

### Figure 2. Population-Level NPS Preservation

Purpose:

Prove GMNPS remains an NPS.

Panels:

- a. FCS2 versus population-mean GMNPS.
- b. Category transition matrix.
- c. Food-group ordering.
- d. Sensitivity across delta caps and v3/v4 masks.

Data source:

- v4 L9 output.

### Figure 3. Personalized Heterogeneity

Purpose:

Show that personalization adds structured individual heterogeneity.

Panels:

- a. Individual-food GMNPS delta distribution.
- b. MAC and LIPID channel variance by food group.
- c. Food-level MAC versus LIPID variance.
- d. Example foods with score intervals.

Data source:

- v4 L9 individual-food and channel matrices.

### Figure 4. External Plausibility and Digital-Twin Validation

Purpose:

Show that microbiome-informed calibration is biologically plausible and computationally necessary.

Panels:

- a. GMrepo metabolic AUROC versus baselines.
- b. ZOE or CRA concordance if robust.
- c. Digital-gut-twin benchmark.
- d. Preservation-personalization trade-off.

Data source:

- GMrepo v2, ZOE 2025, CRA013939 processed tables, synthetic benchmark.

### Extended Data and Supplementary Figures

- FCS2/FNDDS extraction and join audit.
- Full expert-review agreement table.
- v3 original mask sensitivity.
- American Gut stress test.
- Negative controls and shuffled masks.
- Clinical-readiness caveat diagram for E4.

## 7. Manuscript Structure

### Abstract

Logic:

1. Static NPSs support population-level nutrition guidance.
2. Precision nutrition requires individual calibration.
3. GMNPS adds a bounded personalized deviation while preserving the NPS prior.
4. Results show preserved NPS consensus, structured individual heterogeneity and external microbiome plausibility.
5. Expert review supports interpretability and translation potential, but clinical efficacy remains future work.

### Introduction

Paragraph logic:

1. NPSs are useful because they provide stable population-level food-healthfulness scores.
2. Precision nutrition exposes a limitation: the same food can produce different metabolic responses across individuals.
3. Replacing NPSs with fully personalized models would lose public-health interpretability.
4. The proposed solution is personalized calibration: population prior plus bounded individual deviation.
5. The gut microbiome is a biologically plausible calibration layer.
6. This study tests GMNPS as a demonstration framework.

### Results

Recommended sequence:

1. GMNPS defines personalized calibration as an anchored NPS transform.
2. Expert review defines biologically interpretable channels and microbiome-conditioned food groups.
3. GMNPS preserves population-level NPS consensus.
4. GMNPS reveals bounded personalized heterogeneity.
5. External microbiome data support metabolic-axis plausibility.
6. Digital-gut-twin simulations show that anchoring balances preservation and personalization.

### Methods

Recommended subsections:

- Baseline NPS prior.
- Food and nutrient vector construction.
- Microbiome-derived nutrient weights.
- Expert-revised MAC and LIPID masks.
- Bounded personalized deviation transform.
- NPS preservation analysis.
- Heterogeneity analysis.
- External validation datasets.
- Digital-gut-twin simulation.
- Structured expert consensus review.
- Statistical analysis.
- Data and code availability.

### Discussion

Logic:

1. Main contribution: NPS can be transformed by calibration rather than replaced.
2. Population prior and personalized heterogeneity can coexist.
3. The gut microbiome is a feasible and biologically plausible calibration layer.
4. Expert review supports interpretability and translation potential.
5. Limitations: no clinical efficacy, no causal intervention, external data are retrospective and phenotype-specific.
6. Future work: restricted individual-level PREDICT/ZOE validation, prospective feeding studies, clinical safety layer.

## 8. Parallel Workstreams

### Workstream A. Data Provenance and Registry

Owner:

- Data audit subagent.

Tasks:

- Build claim-level registry.
- Resolve GMrepo final sample count and leakage version.
- Mark PREDICT-like data as simulation unless provenance is verified.
- Identify missing code required to reproduce existing verification outputs.

Deliverable:

- `docs/data_claim_registry.md`

### Workstream B. Scoring v4 Implementation

Owner:

- Scoring/code worker.

Tasks:

- Implement expert-revised v4 masks.
- Rerun L9 scoring.
- Produce v4 outputs and QC.
- Preserve v3 as sensitivity.

Deliverable:

- `outputs/gmnps_v4/`

### Workstream C. Preservation and Heterogeneity Analysis

Owner:

- Validation/statistics worker.

Tasks:

- Run E2 and E3 metrics.
- Generate figure source data.
- Identify representative foods and food groups.

Deliverable:

- `outputs/gmnps_v4/validation/`

### Workstream D. External Plausibility

Owner:

- Microbiome validation worker.

Tasks:

- Consolidate GMrepo v2 results.
- Recheck ZOE rank concordance.
- Use CRA013939 processed tables for mechanism anchors.
- Move American Gut to stress-test supplement.

Deliverable:

- `outputs/gmnps_v4/external_validation/`

### Workstream E. Expert Review Evidence

Owner:

- Expert-review synthesis worker.

Tasks:

- Create anonymized expert background table.
- Convert C1/C3/E4 into manuscript-grade summary.
- Request or compile missing item-level ratings if not in current files.
- Generate supplement-ready tables.

Deliverable:

- `outputs/expert_review/`

### Workstream F. Manuscript and Figures

Owner:

- Writing and visualization worker.

Tasks:

- Draft Results only after E2-E6 tables exist.
- Generate figure panels from source data.
- Avoid internal process language such as PDF extraction, Zotero workflow or script details in main text.
- Keep Food Compass 2.0 as baseline prior, not the narrative center.

Deliverable:

- `manuscript/nature_food_submission/`

## 9. Subagent Design

### Subagent 1. Data Evidence Auditor

Purpose:

Validate data provenance and claim level.

Input:

- `data/`
- `outputs/`
- Existing verification reports.

Output:

- Data claim registry.
- Leakage and provenance risks.
- Exclusion list.

### Subagent 2. Scoring Engineer

Purpose:

Implement and rerun expert-revised v4 scoring.

Input:

- L8 bundle v2.
- L9 data.
- C1 expert-revised masks.

Output:

- v4 scoring tables.
- v3 versus v4 comparison.
- tests for score range, centering and mask membership.

### Subagent 3. Validation Statistician

Purpose:

Run NPS preservation, heterogeneity, sensitivity and negative controls.

Input:

- v4 scoring outputs.

Output:

- Validation metrics.
- Figure source data.

### Subagent 4. Microbiome External Validator

Purpose:

Review and rerun external plausibility checks.

Input:

- GMrepo v2.
- ZOE 2025.
- CRA013939.
- American Gut.

Output:

- External validation report and figure source tables.

### Subagent 5. Expert Review Synthesizer

Purpose:

Convert expert review into manuscript-grade evidence.

Input:

- C1, C3, E4, README and any missing rating summaries.

Output:

- Expert review methods paragraph.
- Results summary.
- Supplementary tables.

### Subagent 6. Manuscript Architect

Purpose:

Rebuild manuscript after data results are complete.

Input:

- E1-E6 outputs.
- Figure source data.
- Literature matrix.

Output:

- Nature Food style main text.
- Figure legends.
- Methods.
- Supplementary structure.

## 10. Execution Order

### Phase 0. Freeze Claims and Exclusions

Actions:

1. Confirm PREDICT-like local data provenance.
2. Confirm whether expert-review filled rating sheets exist beyond current files.
3. Freeze the primary claim boundary.

Gate:

- No writing of Results before data claim levels are fixed.

### Phase 1. Recompute Core GMNPS v4

Actions:

1. Implement expert-revised masks.
2. Rerun L9 scoring.
3. Run QC tests.

Gate:

- v4 outputs pass mask, range, centering and preservation tests.

### Phase 2. Main Evidence Analyses

Actions:

1. NPS preservation.
2. Personalized heterogeneity.
3. v3/v4 sensitivity.
4. delta-cap sensitivity.

Gate:

- Main figures 2 and 3 have source data from v4 real food outputs.

### Phase 3. External Plausibility

Actions:

1. Repackage GMrepo v2 metabolic-axis evidence.
2. Run or summarize ZOE rank concordance.
3. Run or summarize CRA013939 mechanism concordance.
4. Move American Gut to supplement.

Gate:

- Main text avoids general-health claims.

### Phase 4. Digital-Twin Benchmark

Actions:

1. Update simulator to v4 masks.
2. Use real food nutrient vectors where possible.
3. Rerun benchmark and negative controls.

Gate:

- FCS2-only residual is reported as not estimable if constant.

### Phase 5. Expert Review Synthesis

Actions:

1. Compile expert review summary.
2. Report C3 and C1 passed status.
3. Summarize E4 translation potential and clinical-readiness caveats.

Gate:

- Expert review is framed as plausibility and interpretability validation.

### Phase 6. Manuscript Reconstruction

Actions:

1. Rewrite Introduction around population prior plus personalized calibration.
2. Write Results in evidence order.
3. Write Methods from reproducible pipeline.
4. Write Discussion with limitations explicit.
5. Prepare Nature Food compatible LaTeX only after content is scientifically stable.

Gate:

- No internal workflow terms in the manuscript narrative.

## 11. Immediate To-Do List

High priority:

1. Build v4 expert-revised mask config.
2. Rerun L9 scoring with v4.
3. Generate NPS preservation and heterogeneity source data from v4.
4. Compile expert-review agreement summary; request missing filled rating sheets if needed.
5. Normalize GMrepo v2 sample-count and leakage-filter descriptions.

Medium priority:

1. Rerun digital-gut-twin benchmark with v4 masks.
2. Run ZOE rank concordance if taxonomy mapping is auditable.
3. Build CRA013939 mechanism-concordance table.
4. Create Nature-style figure source-data directory.

Low priority:

1. Polish schematic panels with image tools after numeric panels are final.
2. Re-upload to CSTCloud only after the manuscript passes internal evidence review.

## 12. Writing Guardrails

- Use "personalized calibration" rather than "replacement score".
- Use "baseline NPS prior" rather than over-centering Food Compass 2.0.
- Use "retrospective plausibility" for GMrepo, ZOE and CRA013939.
- Use "computational feasibility" for digital-gut-twin results.
- Use "structured expert consensus review" for C1/C3/E4.
- Do not use "clinical efficacy", "clinical validation" or "causal effect" unless a valid study supports it.
- Avoid exposing internal artifacts in main text: PDF extraction, Zotero, local paths, script names and code-debug history.
- Avoid overusing quotation marks, em dashes and nominalized language.

## 13. Expected Final Deliverables

Code and data outputs:

- `outputs/gmnps_v4/`
- `outputs/expert_review/`
- `outputs/main_figures/source_data/`
- `docs/data_claim_registry.md`

Manuscript outputs:

- Nature Food main text.
- Four high-density main figures.
- Figure legends.
- Methods.
- Supplementary methods and tables.
- Data availability and code availability statements.

Decision point after Phase 3:

- If external validation is strong enough, submit as an empirical framework Article.
- If external validation remains mostly plausibility-level, frame as a computational framework Article with explicit retrospective and simulation validation.
