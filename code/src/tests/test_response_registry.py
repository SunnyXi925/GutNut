import pandas as pd
import pytest

from gmnps.validation.response_registry import (
    read_response_registry,
    summarize_pre_registered_added_value,
    validate_response_registry,
)


def test_read_response_registry_removes_comments_and_duplicates(tmp_path):
    path = tmp_path / "outcomes.txt"
    path.write_text("glucose\nhdl # lipid\n\nhdl\n", encoding="utf-8")

    assert read_response_registry(path) == ["glucose", "hdl"]


def test_validate_response_registry_fails_missing_outcomes():
    with pytest.raises(ValueError, match="unavailable outcomes: ldl"):
        validate_response_registry(["glucose", "ldl"], ["glucose"])


def test_validate_response_registry_requires_at_least_one_outcome():
    with pytest.raises(ValueError, match="must contain at least one outcome"):
        validate_response_registry([], ["glucose"])


def test_summarize_pre_registered_added_value_uses_validation_only_rows():
    comparisons = pd.DataFrame(
        {
            "outcome": ["glucose", "hdl", "discovery_metab"],
            "split": ["validation", "validation", "discovery"],
            "outcome_source": ["prespecified_list", "prespecified_list", "screened_metabolite"],
            "comparison": ["gmnps_full_vs_food_plus_microbiome"] * 3,
            "improved_over_food_plus_microbiome": [True, False, True],
            "q_value": [0.01, 0.20, 0.01],
        }
    )

    summary = summarize_pre_registered_added_value(comparisons)

    assert summary["n_pre_registered_validation_outcomes"] == 2
    assert summary["pre_registered_response_delta_spearman_fdr_pass_fraction"] == 0.5
    assert summary["response_added_value_gate_passed"] is True


def test_summarize_pre_registered_added_value_requires_half_to_pass_gate():
    comparisons = pd.DataFrame(
        {
            "outcome": ["glucose", "hdl", "ldl"],
            "split": ["validation", "validation", "validation"],
            "outcome_source": ["prespecified_list", "prespecified_list", "prespecified_list"],
            "improved_over_food_plus_microbiome": [True, True, False],
            "q_value": [0.09, 0.11, 0.01],
        }
    )

    summary = summarize_pre_registered_added_value(comparisons)

    assert summary["pre_registered_response_delta_spearman_fdr_pass_fraction"] == pytest.approx(1 / 3)
    assert summary["response_added_value_gate_passed"] is False
