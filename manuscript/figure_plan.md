# Main Figure Plan

The manuscript uses four high-density display items, consistent with a Nature
Food Article-style limit of a small number of main figures. Data panels must be
generated from code and source data. GPT-image or design tools may polish only
schematic styling, not numerical panels.

## Figure 1. Framework And Anchoring Principle

Purpose: show that GMNPS is an anchored NPS transformation rather than a
replacement of the baseline score.

Panels:

- a. Static NPS prior plus microbiome-calibrated deviation.
- b. Bounded deviation transform \(12 \tanh(Z/2)\).
- c. Mechanistic reporting channels.
- d. Schematic individual scores varying around a food-level prior.

## Figure 2. Baseline Integrity And Population Preservation

Purpose: prove that personalized calibration preserves population-level NPS
consensus.

Panels:

- a. Food Compass 2.0 Table S5 extraction audit.
- b. Baseline NPS versus population-mean GMNPS scatter.
- c. Population category transition matrix.

## Figure 3. Personalized Heterogeneity

Purpose: show that GMNPS reveals bounded, interpretable heterogeneity that a
static NPS cannot represent.

Panels:

- a. Heat map of individual-food GMNPS deviations.
- b. MAC and lipid variance by food group.
- c. Food-level MAC versus lipid variance.
- d. Example personalized score intervals.

## Figure 4. Synthetic Digital-Gut-Twin Benchmark

Purpose: test the preservation-personalization trade-off under known ground
truth.

Panels:

- a. Individual response recovery.
- b. Personalized residual recovery, with FCS2-only marked as not estimable.
- c. NPS preservation versus residual recovery trade-off.

## Current Generated Drafts

Run:

```bash
.venv/bin/python code/src/scripts/make_main_figures.py \
  --synthetic-dir outputs/nature_food_article_synthetic \
  --fcs-audit outputs/fcs2/table_s5_audit.json \
  --output-dir outputs/manuscript_figures
```

The command writes PNG, SVG and PDF drafts under `outputs/manuscript_figures/`.
