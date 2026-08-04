import pandas as pd
import pytest

from scripts.run_personalized_calibration_experiments import (
    FCS2_CONSENSUS_DIRECTION_POLICY_VERSION,
    SECTION1_FOOD_SUMMARY_COLUMNS,
    SECTION1_GROUP_CONSENSUS_COLUMNS,
    SECTION1_INDIVIDUAL_RANK_COLUMNS,
    SECTION1_POPULATION_COLUMNS,
    SECTION1_RANK_THRESHOLD_COLUMNS,
    calibration_candidate_status,
    fcs2_consensus_direction_policy_v1,
    section1_output_frame,
    write_section1_outputs,
    write_supplement,
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


def test_all_rejected_section1_writes_headers_and_supplement(tmp_path):
    section1 = tmp_path / "section1_population_consensus"
    rejected = section1_output_frame(
        [
            {
                "amplification": 1.0,
                "n_individuals": 2,
                "n_foods": 3,
                "calibration_status": "rejected_consensus_gate",
                "objective_value": float("inf"),
                "consensus_gates_passed": False,
            }
        ],
        SECTION1_POPULATION_COLUMNS,
    )
    write_section1_outputs(
        section1,
        rejected,
        section1_output_frame([], SECTION1_RANK_THRESHOLD_COLUMNS),
        section1_output_frame([], SECTION1_INDIVIDUAL_RANK_COLUMNS),
        section1_output_frame([], SECTION1_GROUP_CONSENSUS_COLUMNS),
        section1_output_frame([], SECTION1_FOOD_SUMMARY_COLUMNS),
    )

    for name, columns in {
        "section2_clinical_consistency/gmwi2_retention.csv": [
            "clinical_variable", "baseline_rho", "candidate_rho", "retained_fraction", "absolute_loss", "passes_retention"
        ],
        "section4_response_prediction/model_metrics.csv": ["outcome", "model", "n", "rmse", "r2", "spearman"],
        "section4_response_prediction/gmnps_added_value_summary.csv": [
            "outcome", "estimate_model_minus_baseline", "ci_low", "ci_high", "improved_over_food_plus_microbiome"
        ],
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=columns).to_csv(path, index=False)
    kg_dir = tmp_path / "section3_kg_label_adjudication"
    kg_dir.mkdir()
    (kg_dir / "kg_label_metrics.json").write_text('{"accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0, "majority_baseline_accuracy": 0.0}')
    (kg_dir / "kg_binary_label_metrics.json").write_text('{"accuracy": 0.0, "balanced_accuracy": 0.0}')

    write_supplement(tmp_path, {"sections": {}})

    assert pd.read_csv(section1 / "rank_shift_thresholds.csv").columns.tolist() == SECTION1_RANK_THRESHOLD_COLUMNS
    assert pd.read_csv(section1 / "food_summary_by_amplification.csv").empty
    supplement = (tmp_path / "supplementary_material.tex").read_text()
    assert "rejected\\_consensus\\_gate" in supplement
    assert "Calibration candidate diagnostics" in supplement
