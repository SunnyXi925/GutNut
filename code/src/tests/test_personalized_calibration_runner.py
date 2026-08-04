import pandas as pd
import pytest

from scripts.run_personalized_calibration_experiments import (
    FCS2_CONSENSUS_DIRECTION_POLICY_VERSION,
    calibration_candidate_status,
    fcs2_consensus_direction_policy_v1,
)


def test_fcs2_direction_policy_builds_explicit_objective_metadata():
    food = pd.DataFrame(
        {
            "food_group": ["Vegetables", "Sweets", "Unlisted group"],
            "food_subgroup": ["Leafy vegetables", "Candy", "Other"],
        },
        index=["veg", "sweet", "other"],
    )

    metadata = fcs2_consensus_direction_policy_v1(food)

    assert FCS2_CONSENSUS_DIRECTION_POLICY_VERSION == "fcs2_consensus_direction_policy_v1"
    assert list(metadata.columns) == [
        "food_group",
        "food_subgroup",
        "expected_group_direction",
        "expected_subgroup_direction",
    ]
    assert metadata.loc["veg", "expected_group_direction"] == "stable_or_up"
    assert metadata.loc["sweet", "expected_group_direction"] == "stable_or_down"
    assert metadata.loc["other", "expected_group_direction"] == "stable"
    assert metadata["expected_subgroup_direction"].notna().all()


@pytest.mark.parametrize(
    "diagnostics",
    [
        {"objective_value": float("inf"), "consensus_gates_passed": False},
        {"objective_value": float("nan"), "consensus_gates_passed": True},
        {"objective_value": 0.25, "consensus_gates_passed": False},
    ],
)
def test_calibration_candidate_status_rejects_ineligible_candidates(diagnostics):
    assert calibration_candidate_status(diagnostics) == "rejected_consensus_gate"


def test_calibration_candidate_status_accepts_finite_consensus_passing_candidate():
    assert calibration_candidate_status({"objective_value": 0.25, "consensus_gates_passed": True}) == "eligible"
