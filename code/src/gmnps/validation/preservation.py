"""Validation metrics for Food Compass 2.0-anchored GMNPS."""
from __future__ import annotations

import numpy as np
import pandas as pd


def fcs_category(score: float) -> str:
    """Food Compass interpretation category."""

    if score >= 70:
        return "encourage"
    if score >= 31:
        return "moderate"
    return "minimize"


def nps_preservation_metrics(
    individual_food: pd.DataFrame,
    min_rho: float = 0.90,
) -> dict[str, float | int | bool]:
    """Compute NPS-preservation metrics from an individual-food score table."""

    required = {"food_id", "FCS2", "GMNPS_score"}
    missing = required - set(individual_food.columns)
    if missing:
        raise ValueError(f"individual_food missing required columns: {sorted(missing)}")
    food = individual_food.groupby("food_id", sort=False).agg(
        FCS2=("FCS2", "first"),
        GMNPS_mean=("GMNPS_score", "mean"),
    )
    rho = food["FCS2"].rank().corr(food["GMNPS_mean"].rank())
    fcs_cat = food["FCS2"].map(fcs_category)
    gmnps_cat = food["GMNPS_mean"].map(fcs_category)
    shifts = (fcs_cat != gmnps_cat)
    return {
        "n_foods": int(len(food)),
        "spearman_fcs2_gmnps_mean": float(rho),
        "passes_min_rho": bool(rho >= min_rho),
        "category_shift_count": int(shifts.sum()),
        "category_shift_fraction": float(shifts.mean()),
    }


def heterogeneity_metrics(individual_food: pd.DataFrame) -> pd.DataFrame:
    """Summarize personalized heterogeneity by food group."""

    required = {"food_group", "GMNPS_delta", "MAC_delta", "LIPID_delta"}
    missing = required - set(individual_food.columns)
    if missing:
        raise ValueError(f"individual_food missing required columns: {sorted(missing)}")
    return (
        individual_food.groupby("food_group")
        .agg(
            n_rows=("GMNPS_delta", "size"),
            delta_variance=("GMNPS_delta", "var"),
            mac_variance=("MAC_delta", "var"),
            lipid_variance=("LIPID_delta", "var"),
        )
        .reset_index()
        .assign(
            dominant_channel=lambda d: np.where(
                d["mac_variance"] >= d["lipid_variance"], "MAC", "LIPID"
            )
        )
    )
