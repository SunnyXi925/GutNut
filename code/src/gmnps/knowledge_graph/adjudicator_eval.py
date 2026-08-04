from __future__ import annotations

import pandas as pd


def class_balanced_metrics(predictions: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float | int | bool]:
    """Evaluate predictions with equal weight for each observed true class."""
    _require_columns(predictions, {"sample_id", "predicted_label"}, "predictions")
    _require_columns(truth, {"sample_id", "true_label"}, "truth")

    pred = predictions.loc[:, ["sample_id", "predicted_label"]].copy()
    labels = truth.loc[:, ["sample_id", "true_label"]].copy()
    _require_unique_sample_ids(pred, "predictions")
    _require_unique_sample_ids(labels, "truth")
    if labels.empty:
        raise ValueError("truth must contain at least one labeled sample")

    pred["sample_id"] = pred["sample_id"].astype(str)
    labels["sample_id"] = labels["sample_id"].astype(str)
    merged = labels.merge(pred, on="sample_id", how="left", validate="one_to_one")
    correct = merged["predicted_label"].eq(merged["true_label"])
    recalls = [
        float(merged.loc[merged["true_label"].eq(label), "predicted_label"].eq(label).mean())
        for label in sorted(merged["true_label"].dropna().unique())
    ]

    predicted = pred["predicted_label"].dropna()
    largest_prediction_share = float(predicted.value_counts(normalize=True).iloc[0]) if not predicted.empty else 0.0
    return {
        "n_samples": int(len(merged)),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "single_class_collapse": largest_prediction_share >= 0.95,
    }


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _require_unique_sample_ids(frame: pd.DataFrame, name: str) -> None:
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"{name} sample_id values must be unique")
