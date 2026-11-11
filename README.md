# GutNut

**A microbiome-informed Bayesian adaptation framework for personalized nutrient profiling**

Code and verified result tables for the manuscript. The framework adapts a fixed population
predictor through three personal numbers, obtained in closed form from three measured meals: an
overall response level and one slope for each of two named nutrient axes.

## What the analysis establishes

| Claim | Evidence |
|---|---|
| The two channel slopes improve prediction beyond a personal offset | 3.1% (95% CI 1.1-4.9) in 44 free-living participants; 4.5% (3.7-5.4) in 998 participants eating standardized meals |
| The gain exceeds a classical mixed model given the same calibration meals | 8.2% and 16.3% against the best linear unbiased predictor |
| Channel slopes transport across cohorts; the personal level does not | frozen transport: 6.9% (P = 0.005) versus 0.1% (P = 0.46) |
| Three meals are enough | the calibration curve is reported to eleven meals and flattens after three |
| The microbial component contributes nothing detectable | equivalent to host conditioning within a margin eight times smaller than the channel effect; permuting the microbiome-to-participant link reproduces its apparent advantage (P = 0.73) |
| Personal scores do not reorder foods | within-person ordering of observed responses is unchanged in both cohorts |

The last two are negative results and are reported in the main text.

## Layout

```
analysis/lib/            model orchestration, evaluation protocols, prior arms, statistics
analysis/experiments/    one script per experiment, each writing a table and a provenance record
analysis/figures/        R script for Figures 2 and 3
manuscript/              article and supplementary LaTeX sources, bibliography, rendered tables
supplementary/           28 supplementary tables as CSV, plus the claim-to-evidence map
docs/                    methods-gap closure and annotation dispositions
build/                   compiled article and supplementary information
beta_i/, response/, scoring/   modules retained from the earlier version of this project
```

## Reproducing

```sh
python3 analysis/experiments/t0_reproduce.py        # regression lock on published numbers
python3 analysis/experiments/t1_decomposition.py    # three-layer decomposition
python3 analysis/experiments/t4_external.py         # external replication and frozen transport
python3 analysis/experiments/t3_permutation.py      # microbial negative control
python3 analysis/experiments/t9_simulation.py       # identifiability simulation
Rscript  analysis/figures/make_figures_v38.R
tectonic -X compile manuscript/sn-article.tex --outdir build
```

Requires Python 3.11 with pandas, numpy, scikit-learn, scipy and statsmodels, and R with ggplot2,
patchwork, ggbeeswarm, RColorBrewer and hexbin. Each script writes the SHA-256 of its inputs to
`analysis/provenance/`.

Grouped cross-validation uses family membership where available and participant identity
otherwise. Confidence intervals are percentile intervals from 2,000 paired participant-bootstrap
resamples. Permutations number 200. Sign tests exclude participants tied under both models.

## Data

The free-living cohort is the public CGMacros dataset (doi:10.1038/s41597-025-05851-7). The
standardized-meal cohort is used under the original study's access conditions. The food library
uses FNDDS 2017-18 food codes. Analysis code and fitted parameters are in this repository;
cohort data are not redistributed here.

## Status

Prediction is improved; food ordering and prospective food choice are not yet demonstrated. The
food-library scores are a projection of the fitted model and supply candidates for prospective
testing rather than validated recommendations.
