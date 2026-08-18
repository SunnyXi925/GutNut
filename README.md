# GMNPS

Minimal, reproducible source code for the Gut Microbiome-informed Nutrient
Profiling System (GMNPS).

The repository intentionally contains method code only. Manuscripts, review
documents, figures, generated results, raw data, and trained models are not
versioned.

## Layout

- `code/src/gmnps/scoring`: anchored and attribute-level scoring.
- `code/src/gmnps/beta_i`: microbiome health-index and nutrient-weight estimation.
- `code/src/gmnps/data_sources`: input loaders and public-resource adapters.
- `code/src/gmnps/knowledge_graph`: signed-path and label-adjudication utilities.
- `code/src/gmnps/validation`: scientific validation and synthetic controls.
- `code/src/scripts`: command-line entry points for data preparation, scoring,
  and validation.
- `code/src/configs`: versioned method and resource configuration.
- `code/src/tests`: unit and integration tests.

## Installation

Python 3.9 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For a dependency set with conservative version bounds:

```bash
pip install -r code/requirements.txt
pip install -e . --no-deps
```

## Reproduction

Run the complete test suite:

```bash
pytest -q code/src/tests
```

Run the attribute-level scoring pipeline with the versioned default
configuration:

```bash
python code/src/scripts/run_attribute_gmnps.py --help
python code/src/scripts/run_attribute_validation.py --help
```

Build individual nutrient weights or audited public-resource inputs:

```bash
python code/src/scripts/build_beta_i_weights.py --help
python code/src/scripts/build_official_gmwi2_scores.py --help
```

Input datasets are supplied by the user and remain outside Git. Commands write
generated artifacts beneath ignored output directories.

## Reproducibility policy

- Method defaults and resource identifiers are versioned under
  `code/src/configs`.
- Tests include deterministic synthetic controls and fixed random seeds.
- Raw or derived study data, model binaries, article text, citations, figures,
  and submission assets are excluded from the repository.
