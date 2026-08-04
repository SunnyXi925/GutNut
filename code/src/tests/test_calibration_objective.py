import numpy as np
import pandas as pd
import pytest

from gmnps.scoring.calibration_objective import (
    CalibrationParams,
    apply_personalized_offset,
    calibration_objective,
    food_group_direction_pass_fraction,
)


def test_apply_personalized_offset_preserves_bounds_and_applies_directional_changes():
    fcs = pd.Series({"veg": 90.0, "sweet": 10.0, "fish": 80.0})
    raw = pd.DataFrame(
        {"veg": [-1.0, 0.5], "sweet": [2.0, -0.5], "fish": [-0.2, 0.3]},
        index=["s1", "s2"],
    )
    out = apply_personalized_offset(
        fcs,
        raw,
        CalibrationParams(delta_cap=20, temperature=1.0, group_penalty=0.0),
    )
    assert out.min().min() >= 1.0
    assert out.max().max() <= 100.0
    assert out.loc["s1", "sweet"] > fcs["sweet"]
    assert out.loc["s1", "veg"] < fcs["veg"]


def test_calibration_objective_uses_food_groups():
    fcs = pd.Series({"veg": 60.0, "sweet": 40.0})
    raw = pd.DataFrame({"veg": [0.5, 0.5], "sweet": [-0.5, -0.5]}, index=["s1", "s2"])
    result = calibration_objective(
        fcs,
        raw,
        pd.Series({"veg": "Vegetables", "sweet": "SavorySweet"}),
        CalibrationParams(delta_cap=20.0, temperature=1.0, group_penalty=2.0),
    )
    assert result["food_group_direction_pass_fraction"] == 1.0
    assert result["group_penalty"] == 0.0

    reversed_groups = pd.Series({"veg": "SavorySweet", "sweet": "Vegetables"})
    reversed_result = calibration_objective(
        fcs, raw, reversed_groups, CalibrationParams(group_penalty=2.0)
    )
    assert reversed_result["food_group_direction_pass_fraction"] == 0.0
    assert reversed_result["objective_value"] > result["objective_value"]


def test_increasing_group_penalty_worsens_imperfect_direction_objective():
    fcs = pd.Series({"veg": 60.0, "sweet": 40.0})
    raw = pd.DataFrame({"veg": [-0.5, -0.5], "sweet": [0.5, 0.5]}, index=["s1", "s2"])
    groups = pd.Series({"veg": "Vegetables", "sweet": "SavorySweet"})
    low = calibration_objective(fcs, raw, groups, CalibrationParams(group_penalty=1.0))
    high = calibration_objective(fcs, raw, groups, CalibrationParams(group_penalty=3.0))
    assert low["food_group_direction_pass_fraction"] < 1.0
    assert high["group_penalty"] > low["group_penalty"]
    assert high["objective_value"] > low["objective_value"]


def test_food_group_direction_pass_fraction_detects_wrong_group_direction():
    summary = pd.DataFrame(
        {
            "food_group": ["Vegetables", "SavorySweet", "Seafood", "Fats Oils"],
            "expected_direction": [
                "stable_or_up",
                "stable_or_down",
                "stable_or_up",
                "stable_or_down",
            ],
            "delta_group_mean": [-5.0, 3.0, 1.0, -2.0],
        }
    )
    assert food_group_direction_pass_fraction(summary) == 0.5


def test_subgroup_direction_check_catches_within_group_cancellation():
    fcs = pd.Series({"leafy": 60.0, "root": 60.0})
    raw = pd.DataFrame({"leafy": [1.0], "root": [-1.0]}, index=["s1"])
    labels = pd.DataFrame(
        {
            "food_group": ["Vegetables", "Vegetables"],
            "food_subgroup": ["Leafy vegetables", "Root vegetables"],
        },
        index=fcs.index,
    )

    result = calibration_objective(fcs, raw, labels, CalibrationParams(group_penalty=2.0))

    assert result["food_group_direction_pass_fraction"] == 1.0
    assert result["food_subgroup_direction_pass_fraction"] == 0.5
    assert result["group_penalty"] == 1.0


def test_individual_rank_shift_reward_prevents_near_zero_offsets_from_winning():
    fcs = pd.Series({"a": 65.0, "b": 60.0, "c": 45.0, "d": 40.0})
    labels = pd.Series("Neutral", index=fcs.index)
    meaningful = pd.DataFrame(
        {
            "a": [-3.0, 3.0],
            "b": [3.0, -3.0],
            "c": [-3.0, 3.0],
            "d": [3.0, -3.0],
        },
        index=["s1", "s2"],
    )
    near_zero = meaningful * 1e-6

    shifted = calibration_objective(
        fcs, meaningful, labels, CalibrationParams(delta_cap=10.0)
    )
    unchanged = calibration_objective(
        fcs, near_zero, labels, CalibrationParams(delta_cap=10.0)
    )

    assert shifted["mean_individual_rank_shift"] > 0.0
    assert unchanged["mean_individual_rank_shift"] == 0.0
    assert shifted["objective_value"] < unchanged["objective_value"]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (pd.DataFrame({"veg": [0.1]}), "missing required foods"),
        (pd.DataFrame({"veg": [np.nan], "sweet": [0.1]}), "only finite values"),
    ],
)
def test_apply_personalized_offset_rejects_missing_or_nonfinite_offsets(raw, message):
    fcs = pd.Series({"veg": 60.0, "sweet": 40.0})

    with pytest.raises(ValueError, match=message):
        apply_personalized_offset(fcs, raw, CalibrationParams())


def test_apply_personalized_offset_rejects_nonfinite_personalized_scores():
    fcs = pd.Series({"veg": np.nan})
    raw = pd.DataFrame({"veg": [0.1]})

    with pytest.raises(ValueError, match="personalized scores must contain only finite values"):
        apply_personalized_offset(fcs, raw, CalibrationParams())
