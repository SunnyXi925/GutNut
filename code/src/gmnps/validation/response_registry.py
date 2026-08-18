"""Pre-registered dietary-response outcome registry helpers."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def read_response_registry(path: Path) -> list[str]:
    """Read one pre-registered response outcome per line.

    Text after ``#`` is treated as a comment. Duplicate outcomes are removed
    while preserving the first registered order.
    """

    outcomes: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        outcome = line.split("#", maxsplit=1)[0].strip()
        if outcome:
            outcomes.append(outcome)
    return list(dict.fromkeys(outcomes))


def validate_response_registry(
    outcomes: list[str],
    available_columns: Iterable[str],
) -> None:
    """Validate that every registered outcome is present in available columns."""

    if not outcomes:
        raise ValueError("response outcome registry must contain at least one outcome")

    available = set(map(str, available_columns))
    missing = [outcome for outcome in outcomes if outcome not in available]
    if missing:
        raise ValueError("unavailable outcomes: " + ", ".join(missing))


def summarize_pre_registered_added_value(
    model_comparisons: pd.DataFrame,
) -> dict[str, float | int | bool]:
    """Summarize validation-only added value for pre-registered outcomes.

    An outcome passes when it improves over the food-plus-microbiome baseline
    and its FDR-adjusted q-value is at most 0.10. The overall gate passes when
    at least half of validation outcomes pass.
    """

    required = {
        "split",
        "outcome_source",
        "improved_over_food_plus_microbiome",
    }
    missing = required.difference(model_comparisons.columns)
    if missing:
        raise ValueError(f"model_comparisons missing columns: {sorted(missing)}")
    q_column = "q_value" if "q_value" in model_comparisons.columns else "fdr_q_value"
    if q_column not in model_comparisons.columns:
        raise ValueError("model_comparisons missing columns: ['q_value']")

    frame = model_comparisons.loc[
        model_comparisons["split"].eq("validation")
        & model_comparisons["outcome_source"].eq("prespecified_list")
    ].copy()
    if frame.empty:
        return {
            "n_pre_registered_validation_outcomes": 0,
            "pre_registered_response_delta_spearman_fdr_pass_fraction": 0.0,
            "response_added_value_gate_passed": False,
        }

    q_values = pd.to_numeric(frame[q_column], errors="coerce")
    improved = frame["improved_over_food_plus_microbiome"].eq(True)
    fdr_supported = improved & q_values.le(0.10)
    pass_fraction = float(fdr_supported.mean())
    return {
        "n_pre_registered_validation_outcomes": int(len(frame)),
        "pre_registered_response_delta_spearman_fdr_pass_fraction": pass_fraction,
        "response_added_value_gate_passed": bool(pass_fraction >= 0.5),
    }
