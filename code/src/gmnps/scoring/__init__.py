"""Article-facing GMNPS scoring interfaces."""

from gmnps.scoring.anchored import (
    AnchoredScoringConfig,
    attributed_channel_delta,
    bounded_centered_delta,
    raw_channel_responses,
    robust_zscore_by_food,
    score_individual_foods,
    summarize_foods,
)
from gmnps.scoring.masks import (
    EXPERT_REVISED_LIPID,
    EXPERT_REVISED_MAC,
    EXPERT_REVISED_V4_MASK_VERSION,
    LEGACY_EXPERT_REVISED_MASK_VERSION,
    MASK_POLICY,
    ORIGINAL_LIPID,
    ORIGINAL_MAC,
    PRIMARY_MASK_VERSION,
    SCORING_VERSION,
    audit_primary_mask,
    build_channel_vectors,
    get_mask_definition,
)
from gmnps.scoring.calibration_objective import (
    CalibrationParams,
    apply_personalized_offset,
    calibration_objective,
    food_group_direction_pass_fraction,
)

__all__ = [
    "AnchoredScoringConfig",
    "attributed_channel_delta",
    "bounded_centered_delta",
    "raw_channel_responses",
    "robust_zscore_by_food",
    "score_individual_foods",
    "summarize_foods",
    "EXPERT_REVISED_LIPID",
    "EXPERT_REVISED_MAC",
    "EXPERT_REVISED_V4_MASK_VERSION",
    "LEGACY_EXPERT_REVISED_MASK_VERSION",
    "MASK_POLICY",
    "ORIGINAL_LIPID",
    "ORIGINAL_MAC",
    "PRIMARY_MASK_VERSION",
    "SCORING_VERSION",
    "audit_primary_mask",
    "build_channel_vectors",
    "get_mask_definition",
    "CalibrationParams",
    "apply_personalized_offset",
    "calibration_objective",
    "food_group_direction_pass_fraction",
]

from gmnps.scoring.attribute_gmnps import (
    ATTRIBUTE_GMNPS_SCORING_VERSION,
    AttributeGMNPSConfig,
    AttributeGMNPSModel,
    AttributeGMNPSResult,
    FoodAttributeBundle,
    ProductionBundleAttestation,
    ReleaseRegistrySnapshot,
    fit_attribute_gmnps,
    load_food_attribute_bundle_from_bytes,
    load_release_registry_snapshot,
    score_attribute_gmnps,
    summarize_attribute_gmnps_foods,
)

__all__ += [
    "ATTRIBUTE_GMNPS_SCORING_VERSION",
    "AttributeGMNPSConfig",
    "AttributeGMNPSModel",
    "AttributeGMNPSResult",
    "FoodAttributeBundle",
    "ProductionBundleAttestation",
    "ReleaseRegistrySnapshot",
    "fit_attribute_gmnps",
    "load_food_attribute_bundle_from_bytes",
    "load_release_registry_snapshot",
    "score_attribute_gmnps",
    "summarize_attribute_gmnps_foods",
]
