"""Public interfaces for scientific validation and synthetic controls."""

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
from gmnps.validation.gmwi2_dual_channel import (
    DualChannelFeatureSet,
    build_dual_channel_features,
    clinical_correlation_table,
    compare_clinical_retention,
    evaluate_official_vs_dual_channel,
    summarize_dual_channel_superiority,
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
from gmnps.validation.response_registry import (
    read_response_registry,
    summarize_pre_registered_added_value,
    validate_response_registry,
)
from gmnps.validation.synthetic_twin import (
    SyntheticTwinBundle,
    run_synthetic_benchmark,
    simulate_synthetic_twin,
)

__all__ = [
    "ClinicalVariableSpec",
    "DualChannelFeatureSet",
    "RankShiftThresholds",
    "SyntheticTwinBundle",
    "audit_clinical_completeness",
    "build_dual_channel_features",
    "choose_clinical_cohort",
    "clinical_correlation_table",
    "compare_clinical_retention",
    "evaluate_gmwi2_retention",
    "evaluate_official_vs_dual_channel",
    "evaluate_rank_shift_thresholds",
    "fcs_category",
    "fit_ridge_predict",
    "group_folds",
    "heterogeneity_metrics",
    "individual_rank_shift_metrics",
    "nps_preservation_metrics",
    "paired_bootstrap_delta",
    "population_consensus_metrics",
    "read_response_registry",
    "regression_metrics",
    "required_ablation_models",
    "run_synthetic_benchmark",
    "simulate_synthetic_twin",
    "spearman_clinical_correlations",
    "summarize_dual_channel_superiority",
    "summarize_pre_registered_added_value",
    "validate_response_registry",
]
