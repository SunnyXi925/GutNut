# GMNPS Reproducible Pipeline

## Overview

This repository contains the reproducible implementation of the Gut
Microbiome-informed Nutrient Profiling System (GMNPS). The current
article-facing framework treats **Food Compass 2.0 as the universal
population-level nutrient profiling prior** and adds a **bounded,
microbiome-defined personalized deviation**:

```text
GMNPS_ij = clip_1-100(FCS2_j + D_ij)
```

This makes GMNPS an anchored nutrient profiling system rather than an
unanchored diet recommender or clinical intervention model. Legacy modules are
kept for provenance and possible reuse, but the Nature Food article workflow
should start from the interfaces below.

## Nature Food Article Workflow

Primary implementation:

- `src/gmnps/scoring/` — Food Compass 2.0-anchored scoring, expert-revised
  dual-channel masks, individual-food output tables and food summaries.
- `src/gmnps/validation/` — NPS preservation, personalized heterogeneity and
  synthetic digital-gut-twin benchmarks.
- `src/gmnps/manuscript/` — manuscript-ready CSV, figure source data and LaTeX
  table exports.
- `src/scripts/run_nature_food_article.py` — command-line entry point.
- `src/configs/nature_food_article.yaml` — default policy/configuration record.

### Score Retrospective Or Public Data

Expected CSV inputs:

- `weights.csv`: `individual_id` plus nutrient beta columns.
- `food_nutrients.csv`: `food_id` plus standardized food nutrient columns.
- `food_metadata.csv`: `food_id`, `food_name`, `food_group`, `FCS2`.

```bash
cd code/src
python scripts/run_nature_food_article.py score \
  --weights ../../data/example_weights.csv \
  --nutrients ../../data/example_food_nutrients.csv \
  --food-metadata ../../data/example_food_metadata.csv \
  --output-dir ../../outputs/nature_food_article
```

The run writes:

- `tables/individual_food_scores.csv`
- `tables/food_summary.csv`
- `figure_source_data/food_group_heterogeneity.csv`
- `latex/*.tex`
- `article_export_manifest.json`

### Run Digital-Gut-Twin Validation

```bash
cd code/src
python scripts/run_nature_food_article.py synthetic \
  --output-dir ../../outputs/nature_food_article_synthetic \
  --n-individuals 120 \
  --n-foods 80 \
  --seed 42
```

The benchmark compares `FCS2 only`, `unanchored microbiome score`, `anchored
GMNPS`, `random microbiome`, `shuffled microbiome`, `original mask` and
`expert-revised mask`. The intended claim is computational feasibility and
biological plausibility, not clinical efficacy.

### Primary Expert-Revised Masks

The primary mask is `expert_revised_dual_channel`.

- MAC/SCFA/plant-matrix axis keeps fiber, carotenoids, vitamin C, food folate,
  magnesium, potassium, phylloquinone and vitamin E.
- Lipid/bile-acid/TMAO axis keeps total fat, saturated fat, cholesterol,
  retinol, short/medium/long saturated fatty acids, choline and vitamin B12.
- Carbohydrate is sensitivity/proxy only.
- Zinc, copper and vitamin A RAE are removed from primary microbiome channels.
- Retinol remains in the lipid/bile-acid/TMAO channel.

## Legacy Pipeline

The older pipeline is designed with emphasis on:

- **Reproducibility**: Fixed random seeds, version-controlled dependencies
- **Transparency**: Clear documentation of each step
- **Prevention of data leakage**: Strict separation of training and test data
- **Modular design**: Easy to extend and modify components

## Companion documents

The paper-citable framework lives entirely under [`src/`](src/). Two companion
documents describe what to cite and why the structure is sufficient:

- [`CODE_AVAILABILITY.md`](CODE_AVAILABILITY.md) — the canonical *Code
  Availability* text referenced from the manuscript, with the minimal
  reproduction recipe.
- [`SYSTEM_STRUCTURE_REPORT.md`](SYSTEM_STRUCTURE_REPORT.md) — full
  module-by-module mapping to manuscript equations / results, the data
  dependencies of each module, and an assessment of whether the structure
  is adequate given the datasets currently under `../datasets/project_data/`.

Code outside `src/` (under [`exploratory/`](exploratory/)) is **frozen,
not maintained** — preserved only for reviewer provenance and not part of
the Code Availability claim.

## Installation

```bash
# Create conda environment
conda create -n gmnps python=3.9
conda activate gmnps

# Install required packages
pip install -r requirements.txt
```

## Requirements

- Python 3.9+
- scikit-learn >= 1.0
- xgboost >= 1.5
- scikit-optimize >= 0.9
- numpy >= 1.21
- pandas >= 1.3
- scipy >= 1.7

## Pipeline Architecture

The GMNPS pipeline consists of several modular components:

### 1. Data Preprocessing (`DataPreprocessor`)
- **CLR Transformation**: Proper handling of compositional microbiome data
- **Quality Control**: Filtering low-abundance and low-prevalence features
- **Normalization**: Standardization of nutrient data

### 2. Feature Selection (`FeatureSelector`)
- **Multiple Methods**: XGBoost, Elastic Net, and Ensemble approaches
- **Cross-validation**: Prevents overfitting in feature selection
- **Stability Assessment**: Robust feature selection across folds

### 3. Model Training (`GMNPSModel`)
- **SVM with Bayesian Optimization**: Efficient hyperparameter tuning
- **Nested Cross-validation**: Prevents optimistic bias in performance estimation
- **Comprehensive Evaluation**: Multiple metrics including R², RMSE, and Spearman correlation

### 4. Monte Carlo Simulation (`MonteCarloSimulator`)
- **Community Dynamics**: Simulates microbiome perturbations
- **Health Outcome Assessment**: Evaluates simulated scenarios

### 5. Nutrient Weight Estimation (`NutrientWeightEstimator`)
- **Elastic Net Regularization**: Balances model complexity and interpretability
- **Personalized Weights**: Individual-specific nutrient valuations

## Usage

### Basic Usage

```python
from GMNPS_pipeline_reproducible import GMNPSModel, DataPreprocessor

# Initialize model
gmnps = GMNPSModel(cv_folds=5, n_bayes_iter=50)

# Fit model
gmnps.fit(X_microbiome_train, X_nutrients_train, y_train)

# Make predictions
predictions = gmnps.predict(X_microbiome_test, X_nutrients_test)

# Evaluate performance
metrics = gmnps.evaluate(X_microbiome_test, X_nutrients_test, y_test)
```

### Running the Complete Pipeline

```bash
python GMNPS_pipeline_reproducible.py
```

This will:
1. Generate synthetic data for demonstration
2. Train the GMNPS model
3. Evaluate performance on test set
4. Run Monte Carlo simulations
5. Estimate nutrient weights

## Data Format

### Input Data

**Microbiome Data**: 
- Format: CSV or NumPy array
- Shape: (n_samples, n_microbial_features)
- Values: Relative abundances (compositional data)

**Nutrient Data**:
- Format: CSV or NumPy array  
- Shape: (n_samples, n_nutrient_features)
- Values: Absolute amounts or concentrations

**Food Composition Data**:
- Format: CSV or NumPy array
- Shape: (n_foods, n_nutrient_features)
- Values: Nutrient content per food item

### Output Data

**Predictions**:
- Personalized health scores for each food item
- Range: 0-100 (higher = healthier for individual)

**Nutrient Weights**:
- Individual-specific nutrient valuations
- Positive weights = beneficial, negative = detrimental

## Data Leakage Prevention

The pipeline implements several safeguards against data leakage:

1. **Strict Temporal Separation**: Test data never used in training
2. **Nested Cross-validation**: Hyperparameter tuning within cross-validation
3. **Feature Selection**: Performed only on training folds
4. **Preprocessing**: Fit on training data only, applied to test data

## Overfitting Mitigation

1. **Regularization**: Elastic Net and SVM with proper regularization
2. **Cross-validation**: Multiple folds for robust performance estimation
3. **Feature Selection Stability**: Ensemble methods reduce selection bias
4. **Bayesian Optimization**: Efficient hyperparameter search with uncertainty quantification

## Reproducibility Features

1. **Fixed Random Seeds**: All random processes use seed 42
2. **Version Control**: All dependencies specified in requirements.txt
3. **Comprehensive Logging**: Detailed output of all processing steps
4. **Synthetic Data**: Complete pipeline can be tested without real data

## Extending the Pipeline

### Adding New Feature Selection Methods

```python
class NewFeatureSelector:
    def select_features(self, X, y, feature_names):
        # Implement your feature selection logic
        # Return selected indices and importance scores
        pass
```

### Adding New Models

```python
class NewModel:
    def fit(self, X, y):
        # Implement training logic
        pass
    
    def predict(self, X):
        # Implement prediction logic
        pass
```

## Performance Benchmarks

On synthetic data (n=500 samples):
- Training R²: ~0.85
- Test R²: ~0.75
- Spearman correlation: ~0.85
- RMSE: ~5.0

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `n_bayes_iter` or use fewer features
2. **Convergence Warnings**: Increase `max_iter` in ElasticNet
3. **Poor Performance**: Check data quality and feature engineering

### Debugging Tips

1. **Start Small**: Use synthetic data first
2. **Check Data**: Verify data formats and ranges
3. **Monitor Logs**: Detailed output helps identify issues
4. **Cross-validation**: Always validate with multiple folds

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

```
[To be added after publication]
```

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
