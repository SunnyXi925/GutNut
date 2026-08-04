"""Leakage-safe summaries for dietary-response benchmarking."""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_outcome_discovery_validation(outcomes: list[str], seed: int) -> dict[str, list[str]]:
    """Deterministically assign unique outcomes to discovery or validation."""

    unique_outcomes = sorted(dict.fromkeys(str(outcome) for outcome in outcomes))
    rng = np.random.default_rng(seed)
    shuffled = [unique_outcomes[index] for index in rng.permutation(len(unique_outcomes))]
    n_validation = (len(shuffled) + 1) // 2
    return {
        "discovery": sorted(shuffled[n_validation:]),
        "validation": sorted(shuffled[:n_validation]),
    }


def fdr_bh(p_values: pd.Series) -> pd.Series:
    """Apply Benjamini-Hochberg correction while preserving missing values."""

    values = pd.to_numeric(p_values, errors="coerce")
    valid = values.notna()
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    if not valid.any():
        return result
    observed = values.loc[valid]
    if ((observed < 0) | (observed > 1)).any():
        raise ValueError("p_values must lie between zero and one")
    order = np.argsort(observed.to_numpy(dtype=float), kind="mergesort")
    ranked = observed.to_numpy(dtype=float)[order]
    ranks = np.arange(1, len(ranked) + 1, dtype=float)
    adjusted = np.minimum.accumulate((ranked * len(ranked) / ranks)[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    corrected = np.empty(len(observed), dtype=float)
    corrected[order] = adjusted
    result.loc[observed.index] = corrected
    return result


def summarize_added_value(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Summarize positive GMNPS added value using validation outcomes only."""

    required = {"split", "metric", "estimate_model_minus_baseline", "p_value"}
    missing = required.difference(comparisons.columns)
    if missing:
        raise ValueError(f"comparisons missing required columns: {sorted(missing)}")

    validation = comparisons.loc[
        comparisons["split"].eq("validation")
        & comparisons["metric"].eq("delta_spearman")
    ].copy()
    group_columns = [column for column in ["model", "baseline", "metric"] if column in validation.columns]
    if not group_columns:
        group_columns = ["metric"]
    rows = []
    for keys, frame in validation.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        frame = frame.copy()
        frame["fdr_q_value"] = fdr_bh(frame["p_value"])
        estimates = pd.to_numeric(frame["estimate_model_minus_baseline"], errors="coerce")
        positive_fdr = estimates.gt(0) & frame["fdr_q_value"].le(0.05)
        row.update(
            {
                "n_validation_outcomes": int(len(frame)),
                "n_validation_positive": int(estimates.gt(0).sum()),
                "n_validation_positive_fdr": int(positive_fdr.sum()),
                "mean_validation_estimate": float(estimates.mean()) if len(frame) else np.nan,
                "median_validation_estimate": float(estimates.median()) if len(frame) else np.nan,
            }
        )
        rows.append(row)
    columns = [
        *group_columns,
        "n_validation_outcomes",
        "n_validation_positive",
        "n_validation_positive_fdr",
        "mean_validation_estimate",
        "median_validation_estimate",
    ]
    return pd.DataFrame(rows, columns=columns)
