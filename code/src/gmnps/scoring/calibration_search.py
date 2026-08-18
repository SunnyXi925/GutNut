from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gmnps.scoring.calibration_objective import CalibrationParams, calibration_objective


@dataclass(frozen=True)
class CalibrationCandidate:
    params: CalibrationParams
    diagnostics: dict[str, float | bool]


def generate_candidate_grid(
    delta_caps: list[float],
    temperatures: list[float],
    min_rank_shifts: list[float],
) -> list[CalibrationParams]:
    candidates: list[CalibrationParams] = []
    for delta_cap in delta_caps:
        for temperature in temperatures:
            for min_rank_shift in min_rank_shifts:
                candidates.append(
                    CalibrationParams(
                        delta_cap=float(delta_cap),
                        temperature=float(temperature),
                        group_penalty=0.0,
                        min_group_pass_fraction=0.90,
                        min_subgroup_pass_fraction=0.90,
                        min_individual_rank_shift=float(min_rank_shift),
                        require_subgroup_consensus=True,
                        consensus_failure_penalty=100.0,
                    )
                )
    return candidates


def select_calibration_candidate(
    fcs: pd.Series,
    raw_offset: pd.DataFrame,
    food_groups: pd.DataFrame,
    candidates: list[CalibrationParams],
) -> CalibrationCandidate:
    if not candidates:
        raise ValueError("at least one calibration candidate is required")

    evaluated: list[CalibrationCandidate] = []
    for params in candidates:
        diagnostics = calibration_objective(fcs, raw_offset, food_groups, params)
        evaluated.append(CalibrationCandidate(params=params, diagnostics=diagnostics))

    eligible = [
        candidate
        for candidate in evaluated
        if bool(candidate.diagnostics.get("calibration_eligible"))
        and np.isfinite(float(candidate.diagnostics.get("objective_value", np.inf)))
    ]
    if not eligible:
        best_failed = min(
            evaluated,
            key=lambda candidate: float(candidate.diagnostics.get("objective_value", np.inf)),
        )
        raise RuntimeError(f"no eligible calibration candidate; best_status={best_failed.diagnostics}")
    return min(eligible, key=lambda candidate: float(candidate.diagnostics["objective_value"]))
