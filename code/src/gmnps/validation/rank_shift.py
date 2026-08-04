"""Rank-consensus and personalized rank-shift diagnostics for GMNPS."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankShiftThresholds:
    """Default article thresholds for useful but bounded personalization."""

    disease_max_jaccard: float = 0.70
    disease_max_rank_spearman: float = 0.75
    healthy_min_rank_spearman: float = 0.60
    min_delta_span: float = 12.0


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _rank_spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left.astype(float), right.astype(float)], axis=1).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].rank(method="average").corr(pair.iloc[:, 1].rank(method="average")))


def _top_k(values: pd.Series, k: int) -> set[str]:
    if k <= 0:
        return set()
    return set(values.dropna().sort_values(ascending=False).head(k).index.astype(str))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return float(len(left & right) / len(left | right))


def individual_rank_shift_metrics(individual_food: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    """Compute per-individual rank preservation, top-k overlap and delta span.

    ``individual_food`` must contain one row per individual-food score with
    ``individual_id``, ``food_id``, ``FCS2`` and ``GMNPS_score``. Optional cohort
    or phenotype columns are carried through as individual-level annotations.
    """

    _require_columns(individual_food, {"individual_id", "food_id", "FCS2", "GMNPS_score"}, "individual_food")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    fcs = individual_food.drop_duplicates("food_id").set_index("food_id")["FCS2"].astype(float)
    fcs_top = _top_k(fcs, min(top_k, len(fcs)))
    rows: list[dict[str, object]] = []

    for individual_id, group in individual_food.groupby("individual_id", sort=False):
        scored = group.set_index("food_id").reindex(fcs.index)
        gmnps = scored["GMNPS_score"].astype(float)
        delta = gmnps - fcs
        delta_nonmissing = delta.dropna().to_numpy(dtype=float)
        span = (
            float(np.percentile(delta_nonmissing, 95) - np.percentile(delta_nonmissing, 5))
            if len(delta_nonmissing)
            else float("nan")
        )
        row: dict[str, object] = {
            "individual_id": str(individual_id),
            "n_foods": int(gmnps.notna().sum()),
            "rank_spearman_vs_fcs2": _rank_spearman(fcs, gmnps),
            "top_k_jaccard_vs_fcs2": _jaccard(fcs_top, _top_k(gmnps, min(top_k, len(gmnps)))),
            "delta_span_p95_p05": span,
        }
        for col in ("phenotype_label", "disease", "cohort"):
            if col in group.columns:
                values = group[col].dropna().astype(str)
                row[col] = values.iloc[0] if len(values) else "missing"
        rows.append(row)

    return pd.DataFrame(rows)


def population_consensus_metrics(individual_food: pd.DataFrame) -> dict[str, float | int]:
    """Compare mean personalized food scores against Food Compass 2.0 ranks."""

    _require_columns(individual_food, {"food_id", "FCS2", "GMNPS_score"}, "individual_food")
    food = individual_food.groupby("food_id", sort=False).agg(
        FCS2=("FCS2", "first"),
        GMNPS_mean=("GMNPS_score", "mean"),
    )
    return {
        "n_foods": int(len(food)),
        "spearman_fcs2_gmnps_mean": _rank_spearman(food["FCS2"], food["GMNPS_mean"]),
        "mean_absolute_population_shift": float((food["GMNPS_mean"] - food["FCS2"]).abs().mean()),
    }


def _is_healthy(group_name: str) -> bool:
    return group_name.strip().lower() in {"health", "healthy", "control", "controls"}


def evaluate_rank_shift_thresholds(
    metrics: pd.DataFrame,
    thresholds: RankShiftThresholds,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Summarize threshold pass/fail calls by phenotype group or overall."""

    _require_columns(
        metrics,
        {"rank_spearman_vs_fcs2", "top_k_jaccard_vs_fcs2", "delta_span_p95_p05"},
        "metrics",
    )
    if group_col is None or group_col not in metrics.columns:
        grouped = metrics.assign(group="all").groupby("group", sort=False)
    else:
        grouped = metrics.groupby(group_col, sort=False)

    rows: list[dict[str, object]] = []
    for group, frame in grouped:
        healthy = _is_healthy(str(group))
        median_rank = float(frame["rank_spearman_vs_fcs2"].median())
        median_jaccard = float(frame["top_k_jaccard_vs_fcs2"].median())
        median_span = float(frame["delta_span_p95_p05"].median())
        passes_healthy = healthy and median_rank >= thresholds.healthy_min_rank_spearman
        passes_disease = (
            (not healthy)
            and median_rank <= thresholds.disease_max_rank_spearman
            and median_jaccard <= thresholds.disease_max_jaccard
        )
        rows.append(
            {
                "group": str(group),
                "n_individuals": int(len(frame)),
                "median_rank_spearman_vs_fcs2": median_rank,
                "median_top_k_jaccard_vs_fcs2": median_jaccard,
                "median_delta_span_p95_p05": median_span,
                "passes_healthy_rank_preservation": bool(passes_healthy),
                "passes_disease_reranking": bool(passes_disease),
                "passes_delta_span": bool(median_span >= thresholds.min_delta_span),
            }
        )
    return pd.DataFrame(rows)
