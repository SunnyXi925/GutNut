import json

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
    build_submission_readiness_report,
    fcs2_consensus_direction_policy_v1,
    section1_output_frame,
    write_submission_readiness_report,
    write_section1_outputs,
    write_supplement,
    validate_registered_outcomes,
)


def test_submission_readiness_uses_section_metrics_and_fails_missing_official_gmwi2_target():
    report = build_submission_readiness_report(
        {
            "population_consensus": [
                {
                    "amplification": 1.0,
                    "calibration_status": "eligible",
                    "spearman_fcs2_gmnps_mean": 0.97,
                    "food_group_direction_pass_fraction": 0.95,
                },
                {
                    "amplification": 2.0,
                    "calibration_status": "eligible",
                    "spearman_fcs2_gmnps_mean": 1.0,
                    "food_group_direction_pass_fraction": 1.0,
                },
            ]
        },
        {"microbiome_health_mode": "official_gmwi2"},
        {
            "binary_balanced_accuracy": 0.70,
            "binary_single_class_collapse": False,
            "binary_metrics_kind": "class_balanced",
            "signed_edges_validated": True,
        },
        {"response_delta_spearman_fdr_pass_fraction": 0.60},
    )

    rows = {row["name"]: row for row in report["blocking_targets"]}
    assert rows["fcs2_population_spearman"]["observed"] == 0.97
    assert rows["gmwi2_external_balanced_accuracy"]["status"] == "missing"
    assert rows["gmwi2_external_balanced_accuracy"]["passes"] is False
    assert rows["response_delta_spearman_fdr_pass_fraction"]["status"] == "missing"
    assert rows["response_delta_spearman_fdr_pass_fraction"]["passes"] is False
    assert report["n_blocking_targets_passing"] == 3
    assert report["ready_for_submission"] is False


def test_submission_readiness_rejects_collapsed_signed_kg_predictions():
    report = build_submission_readiness_report(
        {"population_consensus": []},
        {},
        {
            "binary_balanced_accuracy": 1.0,
            "binary_single_class_collapse": True,
            "binary_metrics_kind": "class_balanced",
            "signed_edges_validated": True,
        },
        {},
    )

    rows = {row["name"]: row for row in report["blocking_targets"]}
    kg = rows["kg_binary_balanced_accuracy"]
    assert kg["status"] == "missing"
    assert kg["passes"] is False
    assert "collapsed" in kg["observed_source"]


def test_write_submission_readiness_report_writes_csv_and_json(tmp_path):
    report = {
        "ready_for_submission": False,
        "n_blocking_targets": 2,
        "n_blocking_targets_passing": 1,
        "blocking_targets": [
            {"name": "passing_target", "observed": 1.0, "passes": True},
            {"name": "missing_target", "observed": float("nan"), "passes": False},
        ],
    }

    write_submission_readiness_report(tmp_path, report)

    csv_path = tmp_path / "submission_readiness_report.csv"
    json_path = tmp_path / "submission_readiness_report.json"
    assert csv_path.exists()
    assert json_path.exists()
    assert json.loads(json_path.read_text())["ready_for_submission"] is False
    rows = pd.read_csv(csv_path)
    assert rows["name"].tolist() == ["passing_target", "missing_target"]
    assert rows["passes"].tolist() == [True, False]


def test_submission_readiness_uses_only_pre_registered_response_outcomes():
    report = build_submission_readiness_report(
        {"population_consensus": []},
        {},
        {},
        {
            "response_delta_spearman_fdr_pass_fraction": 1.0,
            "response_outcomes_pre_registered": True,
            "pre_registered_response_delta_spearman_fdr_pass_fraction": 0.25,
        },
    )

    rows = {row["name"]: row for row in report["blocking_targets"]}
    response = rows["response_delta_spearman_fdr_pass_fraction"]
    assert response["observed"] == 0.25
    assert response["passes"] is False
    assert response["observed_source"] == "section4.pre_registered_response_delta_spearman_fdr_pass_fraction"


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
    ("diagnostics", "expected"),
    [
        ({"objective_value": float("inf"), "consensus_gates_passed": False}, "rejected_consensus_gate"),
        ({"objective_value": float("nan"), "consensus_gates_passed": True}, "rejected_rank_shift_gate"),
        ({"objective_value": 0.25, "consensus_gates_passed": False}, "rejected_consensus_gate"),
        (
            {
                "objective_value": float("inf"),
                "consensus_gates_passed": True,
                "individual_rank_shift_gate_passed": False,
                "calibration_eligible": False,
            },
            "rejected_rank_shift_gate",
        ),
    ],
)
def test_calibration_candidate_status_rejects_ineligible_candidates(diagnostics, expected):
    assert calibration_candidate_status(diagnostics) == expected


def test_calibration_candidate_status_accepts_finite_consensus_passing_candidate():
    assert calibration_candidate_status(
        {
            "objective_value": 0.25,
            "consensus_gates_passed": True,
            "individual_rank_shift_gate_passed": True,
            "calibration_eligible": True,
        }
    ) == "eligible"


def test_response_registry_rejects_partially_unavailable_outcomes():
    with pytest.raises(ValueError, match="unavailable outcomes: missing_target"):
        validate_registered_outcomes(
            ["glucose", "missing_target"],
            ["subject_id", "glucose"],
        )


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
