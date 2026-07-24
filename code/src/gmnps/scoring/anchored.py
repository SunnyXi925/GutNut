"""Food Compass 2.0-anchored GMNPS scoring.

This module implements the article-facing score:

    GMNPS_ij = clip(FCS2_j + D_ij, 1, 100)

where D_ij is a bounded, centered microbiome-defined personalized deviation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from gmnps.scoring.masks import (
    PRIMARY_MASK_VERSION,
    SCORING_VERSION,
    build_channel_vectors,
)


@dataclass(frozen=True)
class AnchoredScoringConfig:
    """Config for Food Compass 2.0-anchored GMNPS scoring."""

    score_min: float = 1.0
    score_max: float = 100.0
    delta_cap: float = 12.0
    z_scale: float = 2.0
    mask_version: str = PRIMARY_MASK_VERSION
    scoring_version: str = SCORING_VERSION
    top_n_drivers: int = 3


def _require_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def align_weight_and_nutrient_tables(
    weights: pd.DataFrame,
    nutrients: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Align beta weights and food nutrient vectors on shared nutrient columns."""

    common = [c for c in weights.columns if c in nutrients.columns]
    if not common:
        raise ValueError("No shared nutrient columns between weights and nutrients.")
    return weights[common].astype(float), nutrients[common].astype(float), common


def raw_channel_responses(
    weights: pd.DataFrame,
    nutrients: pd.DataFrame,
    mask_version: str = PRIMARY_MASK_VERSION,
) -> dict[str, pd.DataFrame | dict[str, np.ndarray | list[str] | str]]:
    """Compute raw total, MAC, lipid and OTHER responses for all individual-food pairs."""

    w, x, common = align_weight_and_nutrient_tables(weights, nutrients)
    masks = build_channel_vectors(common, mask_version)
    w_arr = w.to_numpy(dtype=float)
    x_arr = x.to_numpy(dtype=float)

    def response(mask: np.ndarray) -> pd.DataFrame:
        arr = (w_arr * mask.astype(float)) @ x_arr.T
        return pd.DataFrame(arr, index=w.index, columns=x.index)

    mac = response(masks["mac"])
    lipid = response(masks["lipid"])
    other = response(masks["other"])
    total = mac + lipid + other
    return {
        "total": total,
        "mac": mac,
        "lipid": lipid,
        "other": other,
        "masks": masks,
    }


def robust_zscore_by_food(raw_response: pd.DataFrame) -> pd.DataFrame:
    """Robustly z-score each food column across individuals."""

    values = raw_response.to_numpy(dtype=float)
    med = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - med), axis=0) * 1.4826
    q75 = np.nanpercentile(values, 75, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    iqr_scale = (q75 - q25) / 1.349
    std = np.nanstd(values, axis=0)
    scale = np.where(mad > 1e-9, mad, np.where(iqr_scale > 1e-9, iqr_scale, std))
    scale = np.where(scale > 1e-9, scale, 1.0)
    z = (values - med) / scale
    return pd.DataFrame(z, index=raw_response.index, columns=raw_response.columns)


def bounded_centered_delta(
    raw_response: pd.DataFrame,
    config: AnchoredScoringConfig | None = None,
) -> pd.DataFrame:
    """Transform raw response to centered, bounded personalized deviation."""

    cfg = config or AnchoredScoringConfig()
    z = robust_zscore_by_food(raw_response).to_numpy(dtype=float)
    delta = cfg.delta_cap * np.tanh(z / cfg.z_scale)
    delta = delta - np.nanmean(delta, axis=0, keepdims=True)
    delta = np.clip(delta, -cfg.delta_cap, cfg.delta_cap)
    delta = delta - np.nanmean(delta, axis=0, keepdims=True)
    delta = np.clip(delta, -cfg.delta_cap, cfg.delta_cap)
    return pd.DataFrame(delta, index=raw_response.index, columns=raw_response.columns)


def attributed_channel_delta(
    raw_channel: pd.DataFrame,
    raw_channels: Sequence[pd.DataFrame],
    total_delta: pd.DataFrame,
) -> pd.DataFrame:
    """Attribute bounded total deviation to one mechanistic channel.

    The article score is defined by the total raw response, so reported channel
    deltas should not be independently rescaled to the full cap. This allocates
    the absolute GMNPS deviation across raw MAC/LIPID/OTHER magnitudes while
    retaining each channel's raw sign.
    """

    channel = raw_channel.to_numpy(dtype=float)
    denom = np.zeros_like(channel, dtype=float)
    for raw in raw_channels:
        denom = denom + np.abs(raw.to_numpy(dtype=float))
    share = np.divide(np.abs(channel), denom, out=np.zeros_like(channel), where=denom > 1e-12)
    attributed = np.sign(channel) * np.abs(total_delta.to_numpy(dtype=float)) * share
    return pd.DataFrame(attributed, index=raw_channel.index, columns=raw_channel.columns)


def _driver_strings(
    weights: pd.DataFrame,
    nutrients: pd.DataFrame,
    common: list[str],
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top positive and negative nutrient drivers per individual-food pair."""

    w = weights[common].to_numpy(dtype=float)
    x = nutrients[common].to_numpy(dtype=float)
    names = np.asarray(common, dtype=object)
    positive = []
    negative = []
    for i in range(w.shape[0]):
        contrib = w[i, None, :] * x[None, :, :]
        pos_rows = []
        neg_rows = []
        for j in range(x.shape[0]):
            row = contrib[0, j]
            pos_idx = np.argsort(-row)[:top_n]
            neg_idx = np.argsort(row)[:top_n]
            pos_rows.append("; ".join(f"{names[k]}={row[k]:.3g}" for k in pos_idx if row[k] > 0))
            neg_rows.append("; ".join(f"{names[k]}={row[k]:.3g}" for k in neg_idx if row[k] < 0))
        positive.append(pos_rows)
        negative.append(neg_rows)
    return (
        pd.DataFrame(positive, index=weights.index, columns=nutrients.index),
        pd.DataFrame(negative, index=weights.index, columns=nutrients.index),
    )


def score_individual_foods(
    weights: pd.DataFrame,
    nutrients: pd.DataFrame,
    food_metadata: pd.DataFrame,
    config: AnchoredScoringConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Score all individual-food pairs and summarize foods.

    food_metadata must be indexed by food id and contain:
    food_name, food_group, FCS2.
    """

    cfg = config or AnchoredScoringConfig()
    _require_columns(food_metadata, ["food_name", "food_group", "FCS2"], "food_metadata")
    shared_foods = [f for f in nutrients.index if f in food_metadata.index]
    if not shared_foods:
        raise ValueError("No overlapping food ids between nutrients and food_metadata.")
    nutrients = nutrients.loc[shared_foods]
    food_metadata = food_metadata.loc[shared_foods]

    responses = raw_channel_responses(weights, nutrients, cfg.mask_version)
    total_raw = responses["total"]
    delta = bounded_centered_delta(total_raw, cfg)
    fcs = food_metadata["FCS2"].astype(float)
    score = delta.add(fcs, axis=1).clip(cfg.score_min, cfg.score_max)

    # Channel attribution is shown on the same point scale as total GMNPS_delta,
    # not as a separate score. OTHER is used for allocation but not reported.
    raw_channels = [responses["mac"], responses["lipid"], responses["other"]]
    mac_delta = attributed_channel_delta(responses["mac"], raw_channels, delta)
    lipid_delta = attributed_channel_delta(responses["lipid"], raw_channels, delta)

    _, _, common = align_weight_and_nutrient_tables(weights, nutrients)
    pos, neg = _driver_strings(weights, nutrients, common, cfg.top_n_drivers)

    rows = []
    for individual_id in weights.index:
        for food_id in shared_foods:
            rows.append(
                {
                    "individual_id": individual_id,
                    "food_id": food_id,
                    "food_name": food_metadata.at[food_id, "food_name"],
                    "food_group": food_metadata.at[food_id, "food_group"],
                    "FCS2": float(fcs.at[food_id]),
                    "GMNPS_delta": float(delta.at[individual_id, food_id]),
                    "GMNPS_score": float(score.at[individual_id, food_id]),
                    "MAC_delta": float(mac_delta.at[individual_id, food_id]),
                    "LIPID_delta": float(lipid_delta.at[individual_id, food_id]),
                    "top_positive_drivers": pos.at[individual_id, food_id],
                    "top_negative_drivers": neg.at[individual_id, food_id],
                    "mask_version": cfg.mask_version,
                    "scoring_version": cfg.scoring_version,
                }
            )
    individual_food = pd.DataFrame(rows)
    food_summary = summarize_foods(individual_food)
    manifest = {
        "scoring_version": cfg.scoring_version,
        "mask_version": cfg.mask_version,
        "delta_cap": cfg.delta_cap,
        "score_range": [cfg.score_min, cfg.score_max],
        "mac_nutrients": responses["masks"]["mac_nutrients"],
        "lipid_nutrients": responses["masks"]["lipid_nutrients"],
        "other_nutrients": responses["masks"]["other_nutrients"],
    }
    return individual_food, food_summary, manifest


def summarize_foods(individual_food: pd.DataFrame) -> pd.DataFrame:
    """Create the article-facing per-food summary table."""

    grouped = individual_food.groupby(["food_id", "food_name", "food_group"], sort=False)
    summary = grouped.agg(
        FCS2=("FCS2", "first"),
        GMNPS_mean=("GMNPS_score", "mean"),
        GMNPS_sd=("GMNPS_score", "std"),
        GMNPS_p05=("GMNPS_score", lambda x: float(np.percentile(x, 5))),
        GMNPS_p95=("GMNPS_score", lambda x: float(np.percentile(x, 95))),
        MAC_variance=("MAC_delta", "var"),
        LIPID_variance=("LIPID_delta", "var"),
    ).reset_index()
    summary["GMNPS_sd"] = summary["GMNPS_sd"].fillna(0.0)
    summary["MAC_variance"] = summary["MAC_variance"].fillna(0.0)
    summary["LIPID_variance"] = summary["LIPID_variance"].fillna(0.0)
    summary["dominant_channel"] = np.where(
        summary["MAC_variance"] >= summary["LIPID_variance"],
        "MAC",
        "LIPID",
    )
    return summary
