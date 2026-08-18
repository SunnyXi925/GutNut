import pandas as pd
import pytest

from gmnps.scoring.calibration_objective import CalibrationParams
from gmnps.scoring.calibration_search import generate_candidate_grid, select_calibration_candidate


def test_generate_candidate_grid_requires_subgroup_consensus():
    candidates = generate_candidate_grid([8.0, 16.0], [0.35], [0.02])

    assert [candidate.delta_cap for candidate in candidates] == [8.0, 16.0]
    assert all(candidate.require_subgroup_consensus for candidate in candidates)
    assert all(candidate.min_group_pass_fraction == 0.90 for candidate in candidates)
    assert all(candidate.min_subgroup_pass_fraction == 0.90 for candidate in candidates)


def test_select_calibration_candidate_rejects_overcompressed_offsets():
    fcs = pd.Series({"veg": 90.0, "fish": 89.0, "sweet": 15.0, "candy": 14.0})
    raw = pd.DataFrame(
        {
            "veg": [2.0, -2.5, 2.0],
            "fish": [0.2, 2.5, -2.0],
            "sweet": [-2.0, 2.5, -2.0],
            "candy": [-0.2, -2.5, 2.0],
        },
        index=["healthy", "disease_a", "disease_b"],
    )
    groups = pd.DataFrame(
        {
            "food_group": ["Vegetables", "Seafood", "Sweets", "Sweets"],
            "food_subgroup": ["Leafy", "Fish", "Candy", "Candy"],
            "expected_group_direction": [
                "stable_or_up",
                "stable_or_up",
                "stable_or_down",
                "stable_or_down",
            ],
            "expected_subgroup_direction": [
                "stable_or_up",
                "stable_or_up",
                "stable_or_down",
                "stable_or_down",
            ],
        },
        index=fcs.index,
    )
    candidates = [
        CalibrationParams(
            delta_cap=0.1,
            temperature=10.0,
            min_individual_rank_shift=0.02,
            require_subgroup_consensus=True,
        ),
        CalibrationParams(
            delta_cap=12.0,
            temperature=0.5,
            min_individual_rank_shift=0.02,
            require_subgroup_consensus=True,
        ),
    ]

    selected = select_calibration_candidate(fcs, raw, groups, candidates)

    assert selected.params.delta_cap == 12.0
    assert selected.diagnostics["calibration_eligible"] is True
    assert selected.diagnostics["mean_individual_rank_shift"] >= 0.02


def test_select_calibration_candidate_raises_when_all_candidates_fail():
    fcs = pd.Series({"veg": 90.0, "sweet": 15.0})
    raw = pd.DataFrame({"veg": [-1.0], "sweet": [1.0]}, index=["s1"])
    groups = pd.DataFrame(
        {
            "food_group": ["Vegetables", "Sweets"],
            "food_subgroup": ["Leafy", "Candy"],
            "expected_group_direction": ["stable_or_up", "stable_or_down"],
            "expected_subgroup_direction": ["stable_or_up", "stable_or_down"],
        },
        index=fcs.index,
    )

    with pytest.raises(RuntimeError, match="no eligible calibration candidate"):
        select_calibration_candidate(
            fcs,
            raw,
            groups,
            [
                CalibrationParams(
                    delta_cap=12.0,
                    temperature=0.5,
                    min_individual_rank_shift=0.02,
                    require_subgroup_consensus=True,
                )
            ],
        )
