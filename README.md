# GutNut

Analysis code for **A microbiome-informed Bayesian adaptation framework for personalized nutrient profiling**.

The method adapts a fixed nonlinear population predictor of postprandial glucose response to an
individual through three numbers obtained in closed form from a handful of measured meals: an
overall response level and one slope for each of two named nutrient channels, microbiota-accessible
carbohydrate represented by total dietary fibre and lipid represented by total fat. Because the
channels are coordinates that every food shares, a profile estimated from three meals produces
predictions for foods the person has never eaten.

This repository contains only the code needed to reproduce the reported analyses. Manuscript
sources, figures, supplementary documents and result files are not part of it.

## What the code establishes

- **The adaptation gain belongs to the representation, not to one estimator.** With the population
  predictor, the calibration meals, the query meals, the design matrix and the residual variance
  held fixed, nine adaptation rules are compared on identical residuals
  (`t16_fair_baselines.py`). Fitting the three coefficients without shrinkage is far worse than no
  personalization; isotropic shrinkage recovers only part of the loss; four different ways of
  obtaining a correlated three-coordinate prior agree on the nutrient-channel increment.
- **Choosing the calibration meals is part of the method.** Nutrient composition is known before a
  meal is eaten, so the calibration set can be selected by the same posterior algebra that performs
  the update (`t17_calibration_design.py`, `lib/design.py`). Seven deployable selectors and an
  oracle upper bound are scored on a query set that does not move.
- **One tolerance for every equality-based test.** `lib/stats.py` defines
  `COMPARISON_TOLERANCE = 1e-10` once; every sign test and better-or-worse count in the project
  uses it, reports the number of ties next to the number of informative pairs, and is recounted at
  `1e-12`, `1e-10` and `1e-8`.
- **Negative controls under five exchangeability assumptions.** The microbiome-to-participant link
  is broken within fold, within glycaemic stratum, within blocks of host-similar participants, in
  the conditioning step alone and in the prior-estimation step alone
  (`t18_permutation_nulls.py`).
- **Audits rather than assertions.** `t22_figure_audit.py` checks every figure file for content
  outside its visible box, `t24_registry.py` builds the statistical registry and the
  headline-number provenance map, and `t26_consistency.py` recomputes every headline number and
  fails if the manuscript and the result files disagree.

## Layout

```
analysis/lib/            environment resolution, statistics, protocols, prior arms, runner
analysis/lib/baselines.py  adaptation rules under one fixed population predictor
analysis/lib/design.py     calibration-set selection criteria
analysis/lib/model/      the model package: masked operator, conditional adapter, masks
analysis/experiments/    one script per reported experiment
analysis/figures/        R scripts for the revised figures
requirements.txt         pinned Python environment
```

## Inputs

No machine-specific path is committed. Four environment variables resolve the input locations, and
every script fails early with the list of files it could not find.

| Variable | Contents |
|---|---|
| `GUTNUT_COHORT_DATA` | `cgmacros_functional.parquet`, `predict.parquet` |
| `GUTNUT_SOURCE_DATA` | `module_predictions.csv`, `CG_future_predictions.csv`, `microbial_history_predictions.csv`, `microbial_history_coefficients.csv` |
| `GUTNUT_FOOD_LIBRARY` | `metadata.parquet`, `food_exposures.parquet` |
| `GUTNUT_MODEL_ROOT` | the model package; defaults to `analysis/lib/model` |

The cohort data are governed by the access conditions of the original studies: the standardized-meal
measurements are available under the source study's conditions, and the free-living measurements are
available through the published dataset record. The food library is derived from FNDDS 2017-18 food
codes with a published nutritional-quality reference.

## Running

```bash
python -m pip install -r requirements.txt
export GUTNUT_COHORT_DATA=/path/to/cohorts
export GUTNUT_SOURCE_DATA=/path/to/saved_predictions
export GUTNUT_FOOD_LIBRARY=/path/to/food_library

cd analysis
python experiments/t0_reproduce.py          # regression lock on the published values
python experiments/t1_decomposition.py      # three-layer decomposition, primary estimand
python experiments/t16_fair_baselines.py    # adaptation rules under a fixed shared predictor
python experiments/t17_calibration_design.py  # calibration-set design
python experiments/t26_consistency.py       # numerical consistency of every headline number
```

Every script writes a JSON summary and its tables to `analysis/results/`, and an input hash
manifest to `analysis/provenance/`. Seeds are fixed in the scripts and recorded in the output.
Run `t0_reproduce.py` first: it is the regression lock and will fail if the inputs are not the ones
the paper used.

Scripts in order: `t0` regression lock, `t1`/`t1b` decomposition and runner check, `t3` permutation
negative control, `t4`/`t4b` external replication, frozen transport and prior-arm contrasts,
`t5` channel instantiation, `t6` evaluation protocols, `t7` classical baselines, `t8` uncertainty
calibration and design conditioning, `t9` identifiability simulation, `t10` cohort description,
`t11` food scores and ordering, `t13` calibration curve, `t14`/`t25` figure source data,
`t16` adaptation rules, `t17` calibration-set design, `t18` alternative permutation nulls,
`t19` captured fraction, `t20` sensitivity analyses, `t21` cohort representativeness,
`t22`/`t23` figure audit and re-export, `t24` statistical registry and provenance map,
`t26` consistency check.

Figures are produced with R and `ggplot2`:

```bash
GUTNUT_FIGURE_DATA=figure_data GUTNUT_FIGURE_OUT=figure Rscript figures/make_figures_v38.R
GUTNUT_FIGURE_DATA=figure_data GUTNUT_FIGURE_OUT=figure Rscript figures/make_figure6_v39.R
```

## Environment

Python 3.11 with the pinned versions in `requirements.txt`. R with `ggplot2`, `patchwork`,
`ggbeeswarm`, `RColorBrewer` and `hexbin` for the figures.

## Citation

Please cite the manuscript. A persistent archive with a digital object identifier will be deposited
on acceptance.
