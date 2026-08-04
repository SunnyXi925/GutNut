from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationParams:
    delta_cap: float = 20.0
    temperature: float = 1.0
    group_penalty: float = 0.0


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
