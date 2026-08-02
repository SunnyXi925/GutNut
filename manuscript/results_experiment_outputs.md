# Results Output Snapshot

This file records the current reproducible outputs for the GMNPS manuscript
draft. Generated CSV, JSON and LaTeX source tables are written under `outputs/`
and remain outside Git.

## Baseline Extraction Audit

Source: `FCS2.0_data.pdf`, Food Compass 2.0 Supplementary Table S5.

Command:

```bash
.venv/bin/python code/src/scripts/extract_fcs2_table_s5.py \
  --text .codex_work/fcs2/FCS2.0_data.txt \
  --output-csv outputs/fcs2/table_s5.csv \
  --audit-json outputs/fcs2/table_s5_audit.json \
  --strict
```

Audit results:

| Metric | Value |
| --- | ---: |
| Extracted rows | 9,273 |
| Expected rows | 9,273 |
| Unique Foodcode values | 9,251 |
| Duplicate Foodcode values | 22 |
| Missing Nutri-Score labels | 24 |
| Score range check | Pass |
| FCS difference check | Pass |
| Malformed food-code rows | 0 |

Interpretation: the PDF extraction recovers the expected Food Compass 2.0 food
universe and passes numeric checks. It is not yet join-ready as a unique
Foodcode table because some food codes repeat and some Nutri-Score labels are
missing in the PDF text extraction. Primary FNDDS matching should preserve
cycle, food code and description.

## Digital-Gut-Twin Benchmark

Command:

```bash
.venv/bin/python code/src/scripts/run_nature_food_article.py synthetic \
  --output-dir outputs/nature_food_article_synthetic \
  --n-individuals 240 \
  --n-foods 120 \
  --seed 42
```

NPS preservation:

| Metric | Value |
| --- | ---: |
| Foods | 120 |
| Spearman correlation between FCS2 and population-mean GMNPS | 0.999965 |
| Category shifts at population mean | 0 |
| Category shift fraction | 0.000 |

Model benchmark:

| Model | Individual response Spearman | Personalized residual Spearman | NPS preservation Spearman |
| --- | ---: | ---: | ---: |
| FCS2 only | 0.910854 | 0.000000 | 1.000000 |
| Unanchored microbiome score | 0.206337 | 0.247912 | 0.932292 |
| Anchored GMNPS | 0.940040 | 0.395680 | 0.999965 |
| Random microbiome | 0.900962 | 0.047131 | 0.999937 |
| Shuffled microbiome | 0.897531 | 0.008768 | 0.999965 |
| Original mask | 0.940040 | 0.395680 | 0.999965 |
| Expert-revised mask | 0.940040 | 0.395680 | 0.999965 |

Food-group heterogeneity:

| Food group | Delta variance | MAC variance | LIPID variance | Dominant channel |
| --- | ---: | ---: | ---: | --- |
| animal | 28.114080 | 0.618420 | 22.769373 | LIPID |
| mixed | 26.768541 | 8.667231 | 8.810633 | LIPID |
| plant | 26.604651 | 20.889483 | 0.760930 | MAC |

Interpretation: in a known-ground-truth synthetic benchmark, anchored GMNPS
preserves the population-level NPS prior while recovering more simulated
personalized residual signal than baseline or shuffled controls. The simulated
channel pattern is aligned with the design: plant foods show MAC-dominant
heterogeneity, whereas animal foods show lipid-dominant heterogeneity. Random
and shuffled microbiome controls lose most of the simulated personalized
residual signal.

## Gated Retrospective Validation

The Nature 2025 ZOE/PREDICT data availability statement makes public
metagenomic reads, age, sex, BMI, country metadata and taxonomic profiles
available for retrospective plausibility analyses. Restricted individual-level
clinical, dietary and intervention data require a proposal to ZOE and a
data-sharing agreement. Therefore, current manuscript wording should describe
ZOE-based analyses as external biological anchoring or retrospective
plausibility unless restricted host-response data are obtained.
