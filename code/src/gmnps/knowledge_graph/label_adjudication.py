from __future__ import annotations

from collections import Counter

import pandas as pd


def build_adjudication_prompt_table(path_scores: pd.DataFrame, leakage_columns: list[str]) -> pd.DataFrame:
    """Prepare KG evidence for deterministic label adjudication with leakage checks."""
    present = [column for column in leakage_columns if column in path_scores.columns]
    if present:
        raise ValueError(f"leakage columns present in adjudication inputs: {present}")
    required = {"sample_id", "disease", "evidence_score"}
    missing = required.difference(path_scores.columns)
    if missing:
        raise ValueError(f"path_scores missing required columns: {sorted(missing)}")
    prompt = path_scores.copy()
    prompt["sample_id"] = prompt["sample_id"].astype(str)
    prompt["disease"] = prompt["disease"].astype(str)
    prompt["evidence_score"] = pd.to_numeric(prompt["evidence_score"], errors="raise").astype(float)
    return prompt


def adjudicate_labels_rule_ai(prompt_table: pd.DataFrame, label_space: list[str]) -> pd.DataFrame:
    """Deterministic AI-style surrogate that picks the strongest KG evidence label."""
    if not label_space:
        raise ValueError("label_space must contain at least one label")
    required = {"sample_id", "disease", "evidence_score"}
    missing = required.difference(prompt_table.columns)
    if missing:
        raise ValueError(f"prompt_table missing required columns: {sorted(missing)}")

    label_order = {label: idx for idx, label in enumerate(label_space)}
    frame = prompt_table.loc[prompt_table["disease"].isin(label_space)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["sample_id", "predicted_label", "decision_score", "adjudicator"])

    frame["_label_order"] = frame["disease"].map(label_order)
    frame = frame.sort_values(
        ["sample_id", "evidence_score", "_label_order"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    pred = (
        frame.groupby("sample_id", as_index=False, group_keys=False)
        .head(1)
        .loc[:, ["sample_id", "disease", "evidence_score"]]
        .rename(columns={"disease": "predicted_label", "evidence_score": "decision_score"})
    )
    pred["adjudicator"] = "kg_rule_ai_surrogate"
    return pred.reset_index(drop=True)


def label_metrics(predictions: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float | int]:
    """Compute label-hit metrics and simple baseline metrics."""
    _require_columns(predictions, {"sample_id", "predicted_label"}, "predictions")
    _require_columns(truth, {"sample_id", "true_label"}, "truth")

    pred = predictions.loc[:, ["sample_id", "predicted_label"]].copy()
    labels = truth.loc[:, ["sample_id", "true_label"]].copy()
    pred["sample_id"] = pred["sample_id"].astype(str)
    labels["sample_id"] = labels["sample_id"].astype(str)
    merged = labels.merge(pred, on="sample_id", how="left")
    n = int(len(merged))
    if n == 0:
        raise ValueError("truth must contain at least one labeled sample")

    correct = merged["predicted_label"].eq(merged["true_label"])
    accuracy = float(correct.mean())
    label_space = sorted(set(merged["true_label"].dropna()) | set(merged["predicted_label"].dropna()))
    balanced_accuracy = _balanced_accuracy(merged, label_space)
    macro_f1 = _macro_f1(merged, label_space)
    majority_accuracy = _majority_baseline_accuracy(merged["true_label"].tolist())
    random_expected_accuracy = _random_baseline_expected_accuracy(merged["true_label"].tolist(), label_space)

    return {
        "n_samples": n,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "majority_baseline_accuracy": majority_accuracy,
        "random_baseline_expected_accuracy": random_expected_accuracy,
        "no_kg_baseline_accuracy": majority_accuracy,
        "rule_only_baseline_accuracy": accuracy,
    }


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _balanced_accuracy(frame: pd.DataFrame, label_space: list[object]) -> float:
    recalls = []
    for label in label_space:
        actual = frame["true_label"].eq(label)
        if actual.any():
            recalls.append(float(frame.loc[actual, "predicted_label"].eq(label).mean()))
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def _macro_f1(frame: pd.DataFrame, label_space: list[object]) -> float:
    scores = []
    for label in label_space:
        true_positive = int(frame["true_label"].eq(label).mul(frame["predicted_label"].eq(label)).sum())
        false_positive = int(frame["true_label"].ne(label).mul(frame["predicted_label"].eq(label)).sum())
        false_negative = int(frame["true_label"].eq(label).mul(frame["predicted_label"].ne(label)).sum())
        denom = (2 * true_positive) + false_positive + false_negative
        scores.append((2 * true_positive / denom) if denom else 0.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


def _majority_baseline_accuracy(true_labels: list[object]) -> float:
    counts = Counter(true_labels)
    return float(max(counts.values()) / len(true_labels))


def _random_baseline_expected_accuracy(true_labels: list[object], label_space: list[object]) -> float:
    if not label_space:
        return 0.0
    counts = Counter(true_labels)
    return float(sum(count / len(true_labels) for count in counts.values()) / len(label_space))
