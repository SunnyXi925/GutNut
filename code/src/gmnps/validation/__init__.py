"""GMNPS validation interfaces.

The new Nature Food article framework uses preservation, heterogeneity and
digital-gut-twin benchmarks. Legacy independent-cohort scripts remain available
as direct module imports.
"""

from gmnps.validation.preservation import (
    fcs_category,
    heterogeneity_metrics,
    nps_preservation_metrics,
)
from gmnps.validation.clinical_correlation import (
    ClinicalVariableSpec,
    audit_clinical_completeness,
    choose_clinical_cohort,
    evaluate_gmwi2_retention,
    spearman_clinical_correlations,
)
from gmnps.validation.rank_shift import (
    RankShiftThresholds,
    evaluate_rank_shift_thresholds,
    individual_rank_shift_metrics,
    population_consensus_metrics,
)
from gmnps.validation.response_prediction import (
    fit_ridge_predict,
    group_folds,
    paired_bootstrap_delta,
    regression_metrics,
    required_ablation_models,
)
from gmnps.validation.synthetic_twin import (
    SyntheticTwinBundle,
    run_synthetic_benchmark,
    simulate_synthetic_twin,
)

__all__ = [
    "ClinicalVariableSpec",
    "RankShiftThresholds",
    "SyntheticTwinBundle",
    "audit_clinical_completeness",
    "choose_clinical_cohort",
    "evaluate_gmwi2_retention",
    "evaluate_rank_shift_thresholds",
    "fcs_category",
    "fit_ridge_predict",
    "group_folds",
    "heterogeneity_metrics",
    "individual_rank_shift_metrics",
    "nps_preservation_metrics",
    "paired_bootstrap_delta",
    "population_consensus_metrics",
    "regression_metrics",
    "required_ablation_models",
    "run_synthetic_benchmark",
    "simulate_synthetic_twin",
    "spearman_clinical_correlations",
]
