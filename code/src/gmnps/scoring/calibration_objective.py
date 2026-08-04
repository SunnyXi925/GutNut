from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationParams:
    delta_cap: float = 20.0
    temperature: float = 1.0
    group_penalty: float = 0.0


def _rank_spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left.astype(float), right.astype(float)], axis=1).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return 0.0
    return float(pair.iloc[:, 0].rank(method="average").corr(pair.iloc[:, 1].rank(method="average")))


def _expected_direction(group_name: str) -> str:
    normalized = group_name.strip().lower()
    if any(token in normalized for token in ("sweet", "sugar", "fat", "oil", "processed")):
        return "stable_or_down"
    if any(token in normalized for token in ("vegetable", "fruit", "seafood", "fish", "whole grain", "legume")):
        return "stable_or_up"
    return "stable"


def apply_personalized_offset(
    fcs: pd.Series,
    raw_offset: pd.DataFrame,
    params: CalibrationParams,
) -> pd.DataFrame:
    if params.delta_cap <= 0:
        raise ValueError("delta_cap must be positive")
    if params.temperature <= 0:
        raise ValueError("temperature must be positive")
    aligned = raw_offset.reindex(columns=fcs.index).astype(float)
    scaled = np.tanh(aligned / params.temperature) * params.delta_cap
    return scaled.add(fcs.astype(float), axis=1).clip(1.0, 100.0)


def food_group_direction_pass_fraction(summary: pd.DataFrame) -> float:
    required = {"expected_direction", "delta_group_mean"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"summary missing required columns: {sorted(missing)}")
    passes = []
    for _, row in summary.iterrows():
        direction = str(row["expected_direction"])
        delta = float(row["delta_group_mean"])
        if direction == "stable_or_up":
            passes.append(delta >= -2.0)
        elif direction == "stable_or_down":
            passes.append(delta <= 2.0)
        elif direction == "stable":
            passes.append(abs(delta) <= 2.0)
        else:
            raise ValueError(f"unknown expected_direction: {direction}")
    return float(np.mean(passes)) if passes else float("nan")


def calibration_objective(
    fcs: pd.Series,
    raw_offset: pd.DataFrame,
    food_groups: pd.Series,
    params: CalibrationParams,
) -> dict[str, float]:
    """Calculate the minimization objective for a calibration parameter set.

    Food groups are aligned to the score columns and used to assess whether
    each group's mean score shift follows its nutrition-consistent direction.
    The objective is ``1 - population_spearman`` plus normalized population
    shift and the weighted fraction of failed group-direction checks.
    """
    if params.group_penalty < 0:
        raise ValueError("group_penalty must be non-negative")
    missing_groups = fcs.index.difference(food_groups.index)
    if len(missing_groups):
        raise ValueError(f"food_groups missing required foods: {list(missing_groups)}")

    scores = apply_personalized_offset(fcs, raw_offset, params)
    mean_scores = scores.mean(axis=0).reindex(fcs.index)
    population_spearman = _rank_spearman(fcs, mean_scores)
    mean_absolute_population_shift = float((mean_scores - fcs.astype(float)).abs().mean())

    group_rows = []
    aligned_groups = food_groups.reindex(fcs.index)
    for group, foods in aligned_groups.groupby(aligned_groups, sort=False):
        group_rows.append(
            {
                "food_group": group,
                "expected_direction": _expected_direction(str(group)),
                "delta_group_mean": float((mean_scores.loc[foods.index] - fcs.loc[foods.index]).mean()),
            }
        )
    pass_fraction = food_group_direction_pass_fraction(pd.DataFrame(group_rows))
    group_penalty = float(params.group_penalty * (1.0 - pass_fraction))
    objective_value = float(
        (1.0 - population_spearman)
        + (mean_absolute_population_shift / 100.0)
        + group_penalty
    )
    return {
        "population_spearman": float(population_spearman),
        "mean_absolute_population_shift": mean_absolute_population_shift,
        "food_group_direction_pass_fraction": float(pass_fraction),
        "group_penalty": group_penalty,
        "objective_value": objective_value,
    }
