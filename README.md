# GMNPS

The source code of the article framework **Personalized calibration transforms nutrient profiling systems for precision nutrition**.

Gut Microbiome-informed Nutrient Profiling System (GMNPS) is implemented here
as a proof-of-concept for anchored personalized calibration in nutrient
profiling:

```text
GMNPS_ij = clip_1-100(FCS2_j + D_ij)
```

The selected anchored baseline prior is Food Compass 2.0 because of its
broad and up-to-date NPS attribute coverage. Gut microbiome information contributes a bounded
personalized deviation, allowing GMNPS to preserve population-level NPS
consensus while revealing personalized metabolic heterogeneity.

## Article Workflow

The reproducible article-facing implementation lives under `code/src`:

- `gmnps.scoring`: Food Compass 2.0-anchored scoring and expert-revised masks.
- `gmnps.validation`: NPS preservation, heterogeneity and digital-gut-twin benchmarks.
- `gmnps.manuscript`: CSV, figure-source-data and LaTeX table exports.
- `scripts/run_nature_food_article.py`: command-line runner.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[article]"

python code/src/scripts/run_nature_food_article.py synthetic \
  --output-dir outputs/nature_food_article_synthetic \
  --n-individuals 120 \
  --n-foods 80 \
  --seed 42
```

For retrospective/public data scoring, provide:

- `weights.csv`: `individual_id` plus nutrient beta columns.
- `food_nutrients.csv`: `food_id` plus standardized food nutrient columns.
- `food_metadata.csv`: `food_id`, `food_name`, `food_group`, `FCS2`.

```bash
python code/src/scripts/run_nature_food_article.py score \
  --weights data/weights.csv \
  --nutrients data/food_nutrients.csv \
  --food-metadata data/food_metadata.csv \
  --output-dir outputs/nature_food_article
```

## Scientific Scope

GMNPS is not framed as replacing Food Compass 2.0. It demonstrates a
computationally reproducible path for transforming static NPS into precision
nutrition by adding gut microbiome-informed calibration.

The current validation strategy is retrospective and simulation-based. It does
not claim clinical intervention efficacy.
