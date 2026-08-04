from dataclasses import replace

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
    groups = pd.DataFrame(
        {
            "food_group": ["Vegetables", "SavorySweet"],
            "expected_direction": ["stable_or_up", "stable_or_down"],
        },
        index=fcs.index,
    )
    result = calibration_objective(
        fcs,
        raw,
        groups,
        CalibrationParams(
            delta_cap=20.0,
            temperature=1.0,
            group_penalty=2.0,
            min_individual_rank_shift=0.0,
        ),
    )
    assert result["food_group_direction_pass_fraction"] == 1.0
    assert result["group_penalty"] == 0.0

    opposed_expectations = groups.assign(
        expected_direction=["stable_or_down", "stable_or_up"]
    )
    reversed_result = calibration_objective(
        fcs,
        raw,
        opposed_expectations,
        CalibrationParams(group_penalty=2.0, min_individual_rank_shift=0.0),
    )
    assert reversed_result["food_group_direction_pass_fraction"] == 0.0
    assert reversed_result["objective_value"] > result["objective_value"]


def test_increasing_group_penalty_worsens_imperfect_direction_objective():
    fcs = pd.Series({"veg": 60.0, "sweet": 40.0})
    raw = pd.DataFrame({"veg": [-0.5, -0.5], "sweet": [0.5, 0.5]}, index=["s1", "s2"])
    groups = pd.DataFrame(
        {
            "food_group": ["Vegetables", "SavorySweet"],
            "expected_group_direction": ["stable_or_up", "stable_or_down"],
        },
        index=fcs.index,
    )
    low = calibration_objective(fcs, raw, groups, CalibrationParams(group_penalty=1.0))
    high = calibration_objective(fcs, raw, groups, CalibrationParams(group_penalty=3.0))
    assert low["food_group_direction_pass_fraction"] < 1.0
    assert high["group_penalty"] > low["group_penalty"]
    assert np.isinf(low["objective_value"])
    assert np.isinf(high["objective_value"])


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
            "expected_group_direction": ["stable_or_up", "stable_or_up"],
            "expected_subgroup_direction": ["stable_or_up", "stable_or_up"],
        },
        index=fcs.index,
    )

    result = calibration_objective(fcs, raw, labels, CalibrationParams(group_penalty=2.0))

    assert result["food_group_direction_pass_fraction"] == 1.0
    assert result["food_subgroup_direction_pass_fraction"] == 0.5
    assert result["group_penalty"] == 1.0


def test_individual_rank_shift_reward_prevents_near_zero_offsets_from_winning():
    fcs = pd.Series({"a": 65.0, "b": 60.0, "c": 45.0, "d": 40.0})
    labels = pd.DataFrame(
        {"food_group": "Neutral", "expected_direction": "stable"}, index=fcs.index
    )
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


@pytest.mark.parametrize(
    "parameter",
    [
        "delta_cap",
        "temperature",
        "group_penalty",
        "min_group_pass_fraction",
        "min_subgroup_pass_fraction",
        "min_individual_rank_shift",
        "consensus_failure_penalty",
    ],
)
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_calibration_objective_rejects_nonfinite_numeric_parameters(parameter, nonfinite):
    fcs = pd.Series({"veg": 60.0})
    raw = pd.DataFrame({"veg": [0.1]})
    labels = pd.DataFrame(
        {"food_group": ["Vegetables"], "expected_group_direction": ["stable_or_up"]},
        index=fcs.index,
    )

    with pytest.raises(ValueError, match=rf"{parameter} must be finite"):
        calibration_objective(fcs, raw, labels, replace(CalibrationParams(), **{parameter: nonfinite}))


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (pd.DataFrame(columns=["veg"]), "at least one row"),
        (pd.DataFrame(index=["s1"]), "at least one food column"),
    ],
)
def test_apply_personalized_offset_rejects_empty_offsets(raw, message):
    with pytest.raises(ValueError, match=message):
        apply_personalized_offset(pd.Series({"veg": 60.0}), raw, CalibrationParams())


def test_explicit_directions_override_food_group_name_quirks():
    fcs = pd.Series({"oddly_named": 60.0})
    raw = pd.DataFrame({"oddly_named": [1.0]}, index=["s1"])
    labels = pd.DataFrame(
        {
            "food_group": ["Sugary Fats"],
            "expected_group_direction": ["stable_or_up"],
        },
        index=fcs.index,
    )

    result = calibration_objective(fcs, raw, labels, CalibrationParams())

    assert result["food_group_direction_pass_fraction"] == 1.0
    assert result["group_consensus_gate_passed"]


def test_missing_direction_requires_explicit_legacy_compatibility_option():
    fcs = pd.Series({"veg": 60.0})
    raw = pd.DataFrame({"veg": [0.1]})
    labels = pd.DataFrame({"food_group": ["Vegetables"]}, index=fcs.index)

    with pytest.raises(ValueError, match="expected_group_direction or expected_direction"):
        calibration_objective(fcs, raw, labels, CalibrationParams())

    result = calibration_objective(
        fcs,
        raw,
        labels,
        CalibrationParams(allow_legacy_name_direction_policy=True),
    )
    assert result["food_group_direction_pass_fraction"] == 1.0


def test_unknown_explicit_direction_fails_fast():
    fcs = pd.Series({"veg": 60.0})
    raw = pd.DataFrame({"veg": [0.1]})
    labels = pd.DataFrame(
        {"food_group": ["Vegetables"], "expected_group_direction": ["sideways"]},
        index=fcs.index,
    )

    with pytest.raises(ValueError, match="unknown expected_direction"):
        calibration_objective(fcs, raw, labels, CalibrationParams())


def test_consensus_gate_penalty_outweighs_larger_rank_shift():
    fcs = pd.Series({"a": 80.0, "b": 60.0, "c": 40.0, "d": 20.0})
    labels = pd.DataFrame(
        {
            "food_group": ["A", "B", "C", "D"],
            "expected_group_direction": ["stable_or_up"] * 4,
        },
        index=fcs.index,
    )
    higher_shift_with_failed_consensus = pd.DataFrame(
        {"a": [-5.0], "b": [5.0], "c": [-5.0], "d": [5.0]}, index=["s1"]
    )
    lower_shift_with_consensus = pd.DataFrame(
        {"a": [0.1], "b": [0.1], "c": [0.1], "d": [0.1]}, index=["s1"]
    )

    failed = calibration_objective(
        fcs, higher_shift_with_failed_consensus, labels, CalibrationParams()
    )
    passed = calibration_objective(
        fcs,
        lower_shift_with_consensus,
        labels,
        CalibrationParams(min_individual_rank_shift=0.0),
    )

    assert failed["mean_individual_rank_shift"] > passed["mean_individual_rank_shift"]
    assert not failed["group_consensus_gate_passed"]
    assert failed["consensus_gate_penalty"] == 100.0
    assert failed["objective_value"] > passed["objective_value"]


@pytest.mark.parametrize("failed_gate", ["group", "subgroup"])
def test_failed_consensus_gate_is_infinite_with_zero_penalties(failed_gate):
    fcs = pd.Series({"a": 60.0, "b": 60.0})
    raw = pd.DataFrame({"a": [-1.0], "b": [-1.0]}, index=["s1"])
    if failed_gate == "group":
        labels = pd.DataFrame(
            {
                "food_group": ["A", "B"],
                "expected_group_direction": ["stable_or_up", "stable_or_up"],
            },
            index=fcs.index,
        )
    else:
        labels = pd.DataFrame(
            {
                "food_group": ["A", "A"],
                "food_subgroup": ["A1", "A2"],
                "expected_group_direction": ["stable_or_down", "stable_or_down"],
                "expected_subgroup_direction": ["stable_or_up", "stable_or_up"],
            },
            index=fcs.index,
        )

    result = calibration_objective(
        fcs,
        raw,
        labels,
        CalibrationParams(
            consensus_failure_penalty=0.0,
            group_penalty=0.0,
            require_subgroup_consensus=failed_gate == "subgroup",
        ),
    )

    assert not result[f"{failed_gate}_consensus_gate_passed"]
    assert not result["consensus_gates_passed"]
    assert np.isinf(result["objective_value"])


def test_near_zero_reranking_is_ineligible_even_when_consensus_passes():
    fcs = pd.Series({"a": 80.0, "b": 60.0, "c": 40.0, "d": 20.0})
    raw = pd.DataFrame({column: [1e-9, -1e-9] for column in fcs.index})
    labels = pd.DataFrame(
        {
            "food_group": ["A", "B", "C", "D"],
            "expected_group_direction": ["stable"] * 4,
        },
        index=fcs.index,
    )

    result = calibration_objective(fcs, raw, labels, CalibrationParams())

    assert result["consensus_gates_passed"] is True
    assert result["individual_rank_shift_gate_passed"] is False
    assert result["calibration_eligible"] is False
    assert np.isinf(result["objective_value"])


def test_required_subgroup_consensus_marks_missing_metadata_as_failure():
    fcs = pd.Series({"a": 60.0, "b": 40.0})
    raw = pd.DataFrame({"a": [-1.0], "b": [1.0]})
    labels = pd.DataFrame(
        {
            "food_group": ["A", "B"],
            "expected_group_direction": ["stable", "stable"],
        },
        index=fcs.index,
    )

    result = calibration_objective(
        fcs,
        raw,
        labels,
        CalibrationParams(require_subgroup_consensus=True),
    )

    assert result["subgroup_metadata_available"] is False
    assert result["subgroup_consensus_gate_passed"] is False
    assert result["consensus_gates_passed"] is False


@pytest.mark.parametrize("missing_subgroup", [None, "", "nan", "None"])
def test_required_subgroup_consensus_rejects_missing_subgroup_values(missing_subgroup):
    fcs = pd.Series({"a": 60.0, "b": 40.0})
    raw = pd.DataFrame({"a": [-1.0], "b": [1.0]})
    labels = pd.DataFrame(
        {
            "food_group": ["A", "A"],
            "food_subgroup": ["A1", missing_subgroup],
            "expected_group_direction": ["stable", "stable"],
            "expected_subgroup_direction": ["stable", "stable"],
        },
        index=fcs.index,
    )

    with pytest.raises(ValueError, match="missing food_subgroup labels"):
        calibration_objective(
            fcs,
            raw,
            labels,
            CalibrationParams(require_subgroup_consensus=True),
        )
