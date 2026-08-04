import pandas as pd

from gmnps.validation.response_benchmark import (
    fdr_bh,
    split_outcome_discovery_validation,
    summarize_added_value,
)


def test_split_outcome_discovery_validation_is_deterministic_and_disjoint():
    split = split_outcome_discovery_validation(["a", "b", "c", "d"], seed=7)
    assert set(split["discovery"]).isdisjoint(split["validation"])
    assert sorted(split["discovery"] + split["validation"]) == ["a", "b", "c", "d"]
    assert split == split_outcome_discovery_validation(["a", "b", "c", "d"], seed=7)


def test_fdr_bh_preserves_index_and_handles_missing_values():
    p_values = pd.Series([0.01, 0.04, None], index=["a", "b", "c"])
    adjusted = fdr_bh(p_values)
    assert adjusted.index.tolist() == ["a", "b", "c"]
    assert adjusted["a"] == 0.02
    assert adjusted["b"] == 0.04
    assert pd.isna(adjusted["c"])


def test_summarize_added_value_requires_validation_outcomes():
    comparisons = pd.DataFrame(
        {
            "outcome": ["a", "b"],
            "split": ["validation", "discovery"],
            "metric": ["delta_spearman", "delta_spearman"],
            "estimate_model_minus_baseline": [0.05, 0.20],
            "p_value": [0.01, 0.001],
        }
    )
    summary = summarize_added_value(comparisons)
    assert summary.loc[0, "n_validation_outcomes"] == 1
    assert summary.loc[0, "n_validation_positive_fdr"] == 1
    assert summary.loc[0, "mean_validation_estimate"] == 0.05
