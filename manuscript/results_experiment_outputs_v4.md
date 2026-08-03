# GMNPS v4 Experiment Outputs for Manuscript Drafting

This document records the current manuscript-facing evidence base for
Personalized calibration transforms nutrient profiling systems for precision
nutrition. It is intended as a source document for Results, Methods and figure
legends. It should not be copied verbatim into the manuscript.

## Evidence Positioning

The primary empirical claim is computational and methodological:

> GMNPS preserves the population-level nutrient-profiling prior while adding a
> bounded, channel-attributed personalized calibration layer.

The current evidence supports computational feasibility, NPS preservation,
expert-reviewed content validity, external cardiometabolic plausibility and
digital-gut-twin validation under known ground truth. It does not support
clinical efficacy, causal dietary intervention claims or disease-agnostic
diagnosis.

## Main Figure Bundle

Generated command:

```bash
.venv/bin/python code/src/scripts/make_v4_main_figures.py \
  --v4-dir outputs/gmnps_v4 \
  --simulation-dir outputs/gmnps_v4_simulation \
  --external-dir outputs/external_validation \
  --output-dir outputs/main_figures_v4
```

Generated figures:

| Figure | File stem | Main role |
| --- | --- | --- |
| Fig. 1 | `figure_1_framework` | Conceptual scoring architecture: NPS prior plus bounded personalized deviation. |
| Fig. 2 | `figure_2_preservation` | NPS preservation across the 9,234-food v4 scoring run. |
| Fig. 3 | `figure_3_heterogeneity` | Personalized heterogeneity and MAC/LIPID channel attribution. |
| Fig. 4 | `figure_4_validation` | GMrepo plausibility plus digital-gut-twin benchmark, ablations and cap sensitivity. |

Exports are available as SVG, PDF and 600-dpi PNG. Quantitative source data are
stored under `outputs/main_figures_v4/source_data/`.

Figure QA:

- Automated source preflight: ready, 0 FAIL.
- Remaining warnings: PNG rather than TIFF export, explicit `dropna` after
  fixed group reindexing, and illustrative synthetic points in Fig. 1d.
- Fig. 1d is schematic only and should not be described as quantitative
  evidence.

## NPS Preservation

Source files:

- `outputs/gmnps_v4/validation/nps_preservation_metrics.json`
- `outputs/gmnps_v4/validation/category_transition.csv`
- `outputs/gmnps_v4/validation/food_group_summary.csv`
- `outputs/gmnps_v4/validation/v3_vs_v4_summary.json`

Key results:

| Metric | Value |
| --- | ---: |
| Foods scored | 9,234 |
| Spearman correlation, baseline NPS vs population-mean GMNPS | 0.9998387709 |
| Category shifts at population mean | 79 |
| Category shift fraction | 0.008555339 |
| Preservation criterion, rho >= 0.90 | Pass |

Category transition counts:

| Baseline category | GMNPS minimize | GMNPS moderate | GMNPS encourage |
| --- | ---: | ---: | ---: |
| Minimize | 2,887 | 0 | 0 |
| Moderate | 38 | 4,178 | 0 |
| Encourage | 0 | 41 | 2,090 |

Interpretation for Results:

Anchored calibration preserved the population-level NPS ranking. The
population mean of GMNPS was almost identical to the baseline prior across
9,234 foods, and fewer than 1% of foods crossed the broad minimize, moderate or
encourage category boundaries at the population mean.

Writing boundary:

Do not frame this as Food Compass 2.0 validation. The point is that GMNPS keeps
the selected population prior intact after personalized calibration.

## Expert-Revised v4 Mask Versus Old v3 Mask

Source file:

- `outputs/gmnps_v4/validation/v3_vs_v4_summary.json`

Key comparison:

| Metric | Old v3 mask | Expert-revised v4 |
| --- | ---: | ---: |
| Spearman with baseline NPS mean | 0.8087356097 | 0.9998387709 |
| Mean total or delta variance | 417.3822078455 | 26.2671070099 |
| Mean MAC variance | 639.0527803223 | 3.0726783276 |
| Mean LIPID variance | 781.1337425710 | 5.1514563560 |

Interpretation for Results:

The expert-revised v4 mask improved anchoring and variance control relative to
the old 15+15 aggregate mask output. This supports the decision to treat the
old v3 mask as a sensitivity design rather than the primary manuscript model.

Writing boundary:

The v3-v4 comparison is a design audit, not proof that the v4 biological mask
is universally optimal.

## Personalized Heterogeneity

Source files:

- `outputs/gmnps_v4/tables/S_food_summary_v4.csv`
- `outputs/gmnps_v4/validation/food_group_summary.csv`

Key food-group pattern:

| Food group | n foods | FCS2 mean | GMNPS mean | Mean delta variance | MAC variance mean | LIPID variance mean | Dominant channel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Beverages | 406 | 32.342 | 32.989 | 27.139 | 2.864 | 2.229 | MAC |
| Grains | 995 | 35.771 | 35.848 | 26.330 | 1.283 | 5.661 | LIPID |
| Fruit | 242 | 70.537 | 70.156 | 26.602 | 4.654 | 2.421 | MAC |
| Vegetables | 1,464 | 76.149 | 75.636 | 26.219 | 5.057 | 4.455 | MAC |
| Legumes/nuts | 312 | 80.843 | 80.632 | 26.357 | 1.572 | 4.425 | LIPID |
| Meat/poultry/eggs | 724 | 44.186 | 44.278 | 25.539 | 0.884 | 3.378 | LIPID |
| Seafood | 241 | 80.320 | 80.015 | 25.992 | 0.284 | 2.306 | LIPID |
| Dairy | 474 | 44.604 | 44.743 | 26.864 | 0.333 | 7.456 | LIPID |
| Fats/oils | 309 | 25.054 | 25.244 | 26.683 | 1.029 | 6.495 | LIPID |
| Mixed dishes | 2,365 | 44.078 | 44.487 | 26.453 | 5.058 | 5.557 | LIPID |
| Sauces/condiments | 888 | 46.941 | 47.488 | 26.472 | 3.834 | 4.823 | LIPID |
| Savory/sweet | 814 | 18.494 | 18.540 | 26.510 | 1.686 | 8.126 | LIPID |

Interpretation for Results:

GMNPS expressed non-zero individual variability for each food while preserving
the population mean. The food-group summary showed channel-attributed
differences: fruit and vegetables were MAC-dominant, whereas dairy, fats/oils,
savory/sweet foods and several animal or mixed groups were LIPID-dominant.

Writing boundary:

These are model-attributed heterogeneity patterns. They should be described as
channel attribution in the GMNPS framework, not as measured mechanisms in a
human intervention.

## Expert Review

Source files:

- `outputs/expert_review/expert_panel_summary.csv`
- `outputs/expert_review/expert_review_methods_summary.md`
- `outputs/expert_review/c1_v4_mask_revision_audit.csv`

Panel composition:

- Six experts.
- Expertise areas: clinical nutrition, food analysis, food biomarkers,
  population nutrition, nutritional bioactive compounds and probiotics.
- Two structured review rounds.
- C3 food-group consensus passed in the first round.
- C1 nutrient mask revisions passed in the second round.
- E4 individual recommendation review supported clinical translation potential
  but indicated that clinical use would require additional constraints.

Primary expert-driven revisions:

- Carbohydrate downgraded to sensitivity/proxy status.
- Zinc and copper removed from the primary MAC channel.
- Vitamin A RAE removed from the lipid channel.
- Retinol retained as the lipid-channel vitamin A marker.

Interpretation for Methods:

The review can be written as a structured expert elicitation for content
validity and clinical interpretability. It should not be called clinical
validation or efficacy testing.

## GMrepo External Plausibility

Source file:

- `outputs/external_validation/gmrepo_v2_gmnps_auroc.csv`

Main-text eligible results:

| Comparison | Disease n | AUROC | 95% CI | p value | Recommended use |
| --- | ---: | ---: | --- | ---: | --- |
| Healthy vs T2D | 771 | 0.6332 | 0.6115-0.6528 | 1.08e-35 | Main text plausibility |
| Healthy vs CAD | 47 | 0.6686 | 0.5823-0.7516 | 3.23e-05 | Main text with small-sample caveat |

Supplementary or boundary-condition results:

- CRC AUROC 0.5174: weak signal, supplementary only.
- CD, UC and IBD AUROCs below 0.5: boundary conditions, not positive results.

Interpretation for Results:

GMrepo provides independent microbiome-health plausibility for the
cardiometabolic axis, especially T2D and CAD. It does not validate GMNPS as a
general health predictor.

Safe sentence:

In an independent GMrepo validation set, the microbiome-derived health-state
component showed modest discrimination for cardiometabolic phenotypes,
including T2D and CAD, while inflammatory bowel phenotypes did not show a
positive signal and therefore defined a boundary condition of the current
target.

## Digital-Gut-Twin Benchmark

Source files:

- `outputs/gmnps_v4_simulation/figure_source_data/synthetic_benchmark.csv`
- `outputs/gmnps_v4_simulation/figure_source_data/design_ablation.csv`
- `outputs/gmnps_v4_simulation/figure_source_data/delta_cap_sensitivity.csv`

Synthetic benchmark:

| Model | Individual response Spearman | Personalized residual Spearman | NPS preservation Spearman |
| --- | ---: | ---: | ---: |
| FCS2 only | 0.910854 | NA | 1.000000 |
| Unanchored microbiome score | 0.206337 | 0.247912 | 0.932292 |
| Anchored GMNPS | 0.940040 | 0.395680 | 0.999965 |
| Random microbiome | 0.900962 | 0.047131 | 0.999937 |
| Shuffled microbiome | 0.897531 | 0.008768 | 0.999965 |
| Original mask | 0.940040 | 0.395680 | 0.999965 |
| Expert-revised mask | 0.940040 | 0.395680 | 0.999965 |

Delta-cap sensitivity:

| Delta cap | Individual response Spearman | Personalized residual Spearman | NPS preservation Spearman |
| ---: | ---: | ---: | ---: |
| 8 | 0.936205 | 0.396858 | 0.999965 |
| 12 | 0.940040 | 0.395680 | 0.999965 |
| 15 | 0.940512 | 0.393889 | 0.999965 |

Interpretation for Results:

The synthetic benchmark supported the intended trade-off. FCS2-only preserved
population ranking but had no individual residual variation. Unanchored
microbiome scoring recovered some residual signal but reduced NPS preservation.
Anchored GMNPS recovered more personalized residual signal than the unanchored,
random and shuffled controls while keeping the population prior intact.

Writing boundary:

The simulator tests computational behaviour under known ground truth. It does
not replace retrospective or prospective human validation.

## Retrospective Human Data Boundary

Available public resources support plausibility analyses but not individual
clinical claims in the current manuscript package.

- ZOE/PREDICT public tables can anchor directionality of microbiome-health and
  microbiome-diet rankings.
- Restricted individual-level host-response, dietary and intervention data
  require data access approval.
- CRA013939 processed tables can support food-microbiome-metabolite-host
  mechanism plausibility, but reconstructed individual tables should not be
  treated as raw individual data.
- American Gut is best used as a noisy self-report stress test in
  supplementary material.

## Claims Supported Now

- GMNPS is an anchored NPS transform, not a replacement for the baseline NPS.
- Population-mean GMNPS preserves the selected NPS prior across 9,234 foods.
- Bounded deviations express individual heterogeneity that static NPS cannot
  represent.
- Expert-reviewed v4 masks improve anchoring relative to the old aggregate
  mask output and support content validity.
- Synthetic digital-gut-twin experiments support the preservation-personalized
  recovery trade-off under known ground truth.
- GMrepo provides modest external plausibility for cardiometabolic microbiome
  signals, especially T2D and CAD.

## Claims Not Supported Now

- GMNPS improves clinical outcomes.
- GMNPS is a clinical diagnostic model.
- GMNPS predicts general health status across diseases.
- Microbiome-derived deviations are causal effects of microbes or foods.
- ZOE public tables validate individual GMNPS predictions.
- CRA013939 processed or reconstructed tables prove causal mechanisms.
