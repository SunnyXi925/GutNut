# GutNut

A microbiome-informed Bayesian adaptation framework for personalized nutrient profiling.

This repository contains the analysis code for the framework. A fixed population predictor is
adapted through three personal numbers, obtained in closed form from a small number of measured
meals: an overall response level and one slope for each of two named nutrient axes. The code
reproduces every quantity reported in the manuscript, including the negative results.

No study data, fitted artifacts, manuscript sources or generated tables are stored here.

## Layout

```
analysis/
  lib/            orchestration, evaluation protocols, prior arms, statistics
    model/        the model package (masked operator, conditional adapter, nutrient masks)
  experiments/    one script per experiment; each writes a table and an input hash manifest
  figures/        R script for the two revised figures
```

Each experiment writes its outputs to `analysis/results/` and the SHA-256 of its inputs to
`analysis/provenance/`. Both directories are created on first run and are not tracked.

## Inputs

Paths are read from the environment, so nothing machine-specific is committed.

| Variable | Default | Contents |
|---|---|---|
| `GUTNUT_MODEL_ROOT` | `analysis/lib/model` | the model package; the default already holds it |
| `GUTNUT_SOURCE_DATA` | `./source_data` | `module_predictions.csv`, `CG_future_predictions.csv`, `microbial_history_predictions.csv`, `microbial_history_coefficients.csv`, `predict.parquet` |
| `GUTNUT_COHORT_DATA` | `GUTNUT_SOURCE_DATA` | `cgmacros_functional.parquet`, `cgmacros.parquet` |
| `GUTNUT_FOOD_LIBRARY` | `./food_library` | `metadata.parquet`, `food_exposures.parquet` |

The free-living cohort is the public CGMacros dataset (doi:10.1038/s41597-025-05851-7). The
standardized-meal cohort is available under the original study's access conditions and is not
redistributed. The food library uses FNDDS 2017-18 food codes.

## Running

```sh
python3 analysis/experiments/t0_reproduce.py        # regression lock on the published numbers
python3 analysis/experiments/t1_decomposition.py    # three-layer decomposition of the gain
python3 analysis/experiments/t1b_runner_check.py    # runner agrees with the saved predictions
python3 analysis/experiments/t3_permutation.py      # microbial negative control (200 permutations)
python3 analysis/experiments/t4_external.py         # external replication and frozen transport
python3 analysis/experiments/t4b_arm_contrasts.py   # equivalence between prior arms
python3 analysis/experiments/t5_channels.py         # channel instantiation comparison
python3 analysis/experiments/t6_protocols.py        # fixed, deployment and random protocols
python3 analysis/experiments/t7_baselines.py        # linear mixed model and ridge head
python3 analysis/experiments/t8_calibration.py      # coverage, conditioning, second sensor
python3 analysis/experiments/t9_simulation.py       # identifiability simulation
python3 analysis/experiments/t10_cohorts.py         # cohort description and responders
python3 analysis/experiments/t11_foodscore.py       # food scores against observed responses
python3 analysis/experiments/t13_calibration_curve.py  # calibration curve to eleven meals
python3 analysis/experiments/t14_figure_data.py     # source data for the figures
Rscript  analysis/figures/make_figures_v38.R        # figures 2 and 3
```

`t3_permutation.py` is the slowest script at roughly thirty minutes on one machine; the others
finish in seconds to a few minutes. Two scripts consume another script's output: `t4b` reads the
predictions written by `t4`, and the R script reads the table written by `t14`. Run those pairs in
order.

## Conventions

Grouped cross-validation uses family membership where available and participant identity
otherwise. Confidence intervals are percentile intervals from 2,000 paired participant-bootstrap
resamples. Permutations number 200 and random calibration draws number five. Sign tests exclude
participants whose value is identical under both models being compared. All units are
mmol l⁻¹ h.

## Requirements

Python 3.11 with pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1, scipy 1.17.1, statsmodels 0.14.6
and pyarrow. R with ggplot2, patchwork, ggbeeswarm, RColorBrewer and hexbin for the figures.

## Scope

The code establishes that calibration improves predicted response magnitude beyond a personal
offset, that the channel slopes transport across cohorts while the personal level does not, and
that microbial conditioning contributes nothing detectable. It does not establish that personal
scores reorder foods, and the food-library output is a projection of the fitted model rather than
a validated recommendation.
