from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


_VALID_EXPECTED_DIRECTIONS = frozenset({"stable_or_up", "stable_or_down", "stable"})
_LEGACY_NAME_DIRECTION_POLICY_VERSION = "v1"


@dataclass(frozen=True)
class CalibrationParams:
    delta_cap: float = 20.0
    temperature: float = 1.0
    group_penalty: float = 0.0
    min_group_pass_fraction: float = 0.90
    min_subgroup_pass_fraction: float = 0.90
    consensus_failure_penalty: float = 100.0
    allow_legacy_name_direction_policy: bool = False


def _validate_params(params: CalibrationParams) -> None:
    numeric_parameters = {
        "delta_cap": params.delta_cap,
        "temperature": params.temperature,
        "group_penalty": params.group_penalty,
        "min_group_pass_fraction": params.min_group_pass_fraction,
        "min_subgroup_pass_fraction": params.min_subgroup_pass_fraction,
        "consensus_failure_penalty": params.consensus_failure_penalty,
    }
    for name, value in numeric_parameters.items():
        try:
            finite = bool(np.isfinite(value))
        except TypeError as error:
            raise ValueError(f"{name} must be finite") from error
        if not finite:
            raise ValueError(f"{name} must be finite")
    if params.delta_cap <= 0:
        raise ValueError("delta_cap must be positive")
    if params.temperature <= 0:
        raise ValueError("temperature must be positive")
    if params.group_penalty < 0:
        raise ValueError("group_penalty must be non-negative")
    if not 0.0 <= params.min_group_pass_fraction <= 1.0:
        raise ValueError("min_group_pass_fraction must be between 0 and 1")
    if not 0.0 <= params.min_subgroup_pass_fraction <= 1.0:
        raise ValueError("min_subgroup_pass_fraction must be between 0 and 1")
    if params.consensus_failure_penalty < 0:
        raise ValueError("consensus_failure_penalty must be non-negative")


def _rank_spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left.astype(float), right.astype(float)], axis=1).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return 0.0
    return float(pair.iloc[:, 0].rank(method="average").corr(pair.iloc[:, 1].rank(method="average")))


def _legacy_expected_direction_v1(group_name: str) -> str:
    """Compatibility-only direction policy for metadata without expectations."""
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
    _validate_params(params)
    if raw_offset.shape[0] == 0:
        raise ValueError("raw_offset must contain at least one row")
    if raw_offset.shape[1] == 0:
        raise ValueError("raw_offset must contain at least one food column")
    missing_foods = fcs.index.difference(raw_offset.columns)
    if len(missing_foods):
        raise ValueError(f"raw_offset missing required foods: {list(missing_foods)}")
    aligned = raw_offset.reindex(columns=fcs.index).astype(float)
    if aligned.shape[1] == 0:
        raise ValueError("raw_offset must contain at least one aligned food column")
    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("raw_offset must contain only finite values for FCS foods")
    scaled = np.tanh(aligned / params.temperature) * params.delta_cap
    unbounded_scores = scaled.add(fcs.astype(float), axis=1)
    if not np.isfinite(unbounded_scores.to_numpy()).all():
        raise ValueError("personalized scores must contain only finite values")
    return unbounded_scores.clip(1.0, 100.0)


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


def _food_group_frame(food_groups: pd.Series | pd.DataFrame, foods: pd.Index) -> pd.DataFrame:
    if isinstance(food_groups, pd.Series):
        labels = food_groups.rename("food_group").to_frame()
    elif isinstance(food_groups, pd.DataFrame):
        if "food_group" not in food_groups.columns:
            raise ValueError("food_groups DataFrame missing required column: food_group")
        allowed_columns = [
            "food_group",
            "food_subgroup",
            "expected_group_direction",
            "expected_subgroup_direction",
            "expected_direction",
        ]
        labels = food_groups.loc[:, [column for column in allowed_columns if column in food_groups]]
    else:
        raise TypeError("food_groups must be a pandas Series or DataFrame")

    missing_foods = foods.difference(labels.index)
    if len(missing_foods):
        raise ValueError(f"food_groups missing required foods: {list(missing_foods)}")
    labels = labels.reindex(foods)
    if labels["food_group"].isna().any():
        raise ValueError("food_groups contains missing food_group labels")
    if "food_subgroup" in labels and labels["food_subgroup"].isna().any():
        raise ValueError("food_groups contains missing food_subgroup labels")
    return labels


def _resolved_expected_directions(
    labels: pd.DataFrame,
    primary_column: str,
    *,
    allow_legacy_name_direction_policy: bool,
) -> pd.Series:
    directions = pd.Series(index=labels.index, dtype=object)
    if primary_column in labels:
        directions = labels[primary_column].copy()
    if "expected_direction" in labels:
        directions = directions.where(directions.notna(), labels["expected_direction"])

    missing = directions.isna()
    if missing.any():
        if not allow_legacy_name_direction_policy:
            fallback = "expected_direction"
            raise ValueError(
                f"food metadata requires {primary_column} or {fallback}; "
                f"set allow_legacy_name_direction_policy=True to use "
                f"legacy name policy {_LEGACY_NAME_DIRECTION_POLICY_VERSION}"
            )
        directions.loc[missing] = labels.loc[missing, "food_group"].map(
            _legacy_expected_direction_v1
        )

    unknown = ~directions.isin(_VALID_EXPECTED_DIRECTIONS)
    if unknown.any():
        unknown_values = sorted(map(str, directions.loc[unknown].unique()))
        raise ValueError(f"unknown expected_direction: {unknown_values}")
    return directions


def _direction_summary(
    labels: pd.DataFrame,
    mean_scores: pd.Series,
    fcs: pd.Series,
    columns: list[str],
    direction_column: str,
    *,
    allow_legacy_name_direction_policy: bool,
) -> pd.DataFrame:
    directions = _resolved_expected_directions(
        labels,
        direction_column,
        allow_legacy_name_direction_policy=allow_legacy_name_direction_policy,
    )
    rows = []
    for keys, foods in labels.groupby(columns, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        group_directions = directions.loc[foods.index].unique()
        if len(group_directions) != 1:
            raise ValueError(
                f"metadata has conflicting {direction_column} values for {keys}"
            )
        row = dict(zip(columns, keys, strict=True))
        row.update(
            expected_direction=group_directions[0],
            delta_group_mean=float((mean_scores.loc[foods.index] - fcs.loc[foods.index]).mean()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_individual_rank_shift(fcs: pd.Series, scores: pd.DataFrame) -> float:
    """Return 0 for unchanged ranks and 1 for complete rank reversals."""
    shifts = []
    for _, personalized in scores.iterrows():
        if personalized.nunique() < 2:
            shifts.append(0.0)
            continue
        spearman = _rank_spearman(fcs, personalized)
        shifts.append((1.0 - spearman) / 2.0)
    return float(np.mean(shifts)) if shifts else float("nan")


def calibration_objective(
    fcs: pd.Series,
    raw_offset: pd.DataFrame,
    food_groups: pd.Series | pd.DataFrame,
    params: CalibrationParams,
) -> dict[str, float | bool]:
    """Calculate the minimization objective for a calibration parameter set.

    Food groups and optional subgroups are aligned to the score columns and
    assessed independently, so opposing subgroup shifts cannot cancel at the
    food-group level. The objective rewards individual rank changes while
    penalizing population and direction-consensus failures. Expected directions
    come from metadata; the legacy label-name policy is available only through
    ``allow_legacy_name_direction_policy`` and is versioned as ``v1``.
    """
    _validate_params(params)
    scores = apply_personalized_offset(fcs, raw_offset, params)
    mean_scores = scores.mean(axis=0).reindex(fcs.index)
    population_spearman = _rank_spearman(fcs, mean_scores)
    mean_absolute_population_shift = float((mean_scores - fcs.astype(float)).abs().mean())
    mean_individual_rank_shift = _mean_individual_rank_shift(fcs, scores)

    labels = _food_group_frame(food_groups, fcs.index)
    group_pass_fraction = food_group_direction_pass_fraction(
        _direction_summary(
            labels,
            mean_scores,
            fcs,
            ["food_group"],
            "expected_group_direction",
            allow_legacy_name_direction_policy=params.allow_legacy_name_direction_policy,
        )
    )
    subgroup_pass_fraction = float("nan")
    if "food_subgroup" in labels:
        subgroup_pass_fraction = food_group_direction_pass_fraction(
            _direction_summary(
                labels,
                mean_scores,
                fcs,
                ["food_group", "food_subgroup"],
                "expected_subgroup_direction",
                allow_legacy_name_direction_policy=params.allow_legacy_name_direction_policy,
            )
        )
    group_consensus_gate_passed = group_pass_fraction >= params.min_group_pass_fraction
    subgroup_consensus_gate_passed = (
        np.isnan(subgroup_pass_fraction)
        or subgroup_pass_fraction >= params.min_subgroup_pass_fraction
    )
    subgroup_failure = 0.0 if np.isnan(subgroup_pass_fraction) else 1.0 - subgroup_pass_fraction
    group_penalty = float(params.group_penalty * ((1.0 - group_pass_fraction) + subgroup_failure))
    consensus_gate_penalty = float(
        params.consensus_failure_penalty
        * (
            int(not group_consensus_gate_passed)
            + int(not subgroup_consensus_gate_passed)
        )
    )
    objective_value = float("inf") if not (
        group_consensus_gate_passed and subgroup_consensus_gate_passed
    ) else float(
        (1.0 - population_spearman)
        + (mean_absolute_population_shift / 100.0)
        + group_penalty
        + consensus_gate_penalty
        - mean_individual_rank_shift
    )
    return {
        "population_spearman": float(population_spearman),
        "mean_absolute_population_shift": mean_absolute_population_shift,
        "mean_individual_rank_shift": mean_individual_rank_shift,
        "food_group_direction_pass_fraction": float(group_pass_fraction),
        "food_subgroup_direction_pass_fraction": float(subgroup_pass_fraction),
        "group_penalty": group_penalty,
        "group_consensus_gate_passed": bool(group_consensus_gate_passed),
        "subgroup_consensus_gate_passed": bool(subgroup_consensus_gate_passed),
        "consensus_gates_passed": bool(
            group_consensus_gate_passed and subgroup_consensus_gate_passed
        ),
        "consensus_gate_penalty": consensus_gate_penalty,
        "objective_value": objective_value,
    }
