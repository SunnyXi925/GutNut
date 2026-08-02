# GMNPS Data Registry

This registry separates data sources that can support article claims from
sources that only support plausibility checks. GMNPS uses Food Compass 2.0 as
an anchored baseline prior, then adds bounded personalized calibration from
microbiome-derived nutrient response profiles.

## Immediate Sources

| Source | Local status | Planned use | Claim level | Access notes |
| --- | --- | --- | --- | --- |
| Food Compass 2.0 Supplementary Table S5 | User provided `FCS2.0_data.pdf`; parsed via `scripts/extract_fcs2_table_s5.py` | Baseline NPS prior `FCS2_j`; food-code keyed metadata | Baseline prior | PDF extraction must be audited; an author spreadsheet is preferable if available. |
| USDA FNDDS 2001-2002 through 2017-2018 | Public download required from FoodData Central and USDA ARS FNDDS pages | Cycle-specific food nutrient vectors aligned to the Food Compass 2.0 food universe | Data engineering input | Preserve `Foodcode`, description and FNDDS cycle; do not replace historical cycles with current FoodData Central releases. |
| FPED 2001-2002 through 2017-2018 | Public download required if food-pattern vectors are used | Sensitivity or food-pattern vector construction | Data engineering input | Must follow the same NHANES/FNDDS cycle range as the baseline prior. |

## Retrospective Plausibility Sources

| Source | Planned use | Claim level | Access notes |
| --- | --- | --- | --- |
| PREDICT 1 / ZOE-style diet, microbiome and postprandial response data | Direct retrospective check of microbiome-informed deviations against host metabolic response | Retrospective plausibility, potentially direct validation if individual-level access is approved | Individual-level clinical, dietary and intervention data require ZOE approval and a data-sharing agreement. |
| ZOE Microbiome Health Ranking 2025 | External biological plausibility anchor for pre-specified microbiome feature directionality | Retrospective biological plausibility | Public ranking and public metagenomic profiles can support plausibility; restricted host data are not public. |
| American Gut Project | Diet-microbiome plausibility and transportability stress tests | Retrospective plausibility | Public sequence and questionnaire resources; nutrient-level intake resolution is limited. |
| curatedMetagenomicData | Cross-cohort microbiome health-state transfer and confounding checks | Retrospective plausibility | Public Bioconductor package; not a direct diet-response validation set. |
| GMrepo | Microbiome-health-state plausibility and disease-control gradients | Retrospective plausibility | Public database; phenotype harmonization must be audited. |

## Synthetic Validation

| Source | Planned use | Claim level | Access notes |
| --- | --- | --- | --- |
| Digital gut twin simulator using real FNDDS food vectors | Known-ground-truth benchmark of anchored GMNPS against FCS2-only, unanchored microbiome score and shuffled controls | Computational feasibility | Simulator parameters must be pre-specified before outcome inspection. |

## Claim Boundaries

- Do not claim clinical intervention efficacy.
- Do not claim causal dietary treatment effects from retrospective or synthetic
  analyses.
- Do not present Food Compass 2.0 as the article's main contribution.
- Do present Food Compass 2.0 as the selected universal NPS prior because it
  offers broad and current attribute coverage.
- Do present gut microbiome information as a proof-of-concept personalized
  calibration layer, not as the only possible precision-nutrition layer.
