from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceTarget:
    name: str
    threshold: float
    direction: str
    rationale: str


def reference_targets() -> dict[str, ReferenceTarget]:
    return {
        "fcs2_population_spearman": ReferenceTarget(
            "fcs2_population_spearman",
            0.95,
            ">=",
            "Population-average personalized scores should remain strongly aligned with FCS2.0.",
        ),
        "food_group_direction_pass_fraction": ReferenceTarget(
            "food_group_direction_pass_fraction",
            0.90,
            ">=",
            "Major food-group shifts should not contradict nutrition consensus.",
        ),
        "gmwi2_external_balanced_accuracy": ReferenceTarget(
            "gmwi2_external_balanced_accuracy",
            0.72,
            ">=",
            "Reference-only benchmark based on official GMWI2 external validation around this level for compatible taxonomic profiles using the official GMWI2 model; this project has not run official GMWI2 validation.",
        ),
        "kg_binary_balanced_accuracy": ReferenceTarget(
            "kg_binary_balanced_accuracy",
            0.65,
            ">=",
            "KG binary health/disease adjudication should exceed a weak exploratory signal.",
        ),
        "response_delta_spearman_fdr_pass_fraction": ReferenceTarget(
            "response_delta_spearman_fdr_pass_fraction",
            0.50,
            ">=",
            "At least half of pre-registered response outcomes should show FDR-supported added value.",
        ),
    }


def evaluate_target(name: str, observed: float) -> dict[str, object]:
    targets = reference_targets()
    if name not in targets:
        raise KeyError(f"unknown reference target: {name}")
    target = targets[name]
    if target.direction == ">=":
        passes = observed >= target.threshold
        margin = observed - target.threshold
    elif target.direction == "<=":
        passes = observed <= target.threshold
        margin = target.threshold - observed
    else:
        raise ValueError(f"unsupported target direction: {target.direction}")
    return {
        "name": name,
        "observed": float(observed),
        "threshold": target.threshold,
        "direction": target.direction,
        "passes": bool(passes),
        "margin": round(float(margin), 12),
        "rationale": target.rationale,
    }
