import pandas as pd

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
