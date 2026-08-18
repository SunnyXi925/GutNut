import numpy as np
import pandas as pd
import pytest

from gmnps.scoring.masks import build_channel_vectors
from gmnps.validation.gmwi2_dual_channel import (
    DualChannelFeatureSet,
    build_dual_channel_features,
    clinical_correlation_table,
    compare_clinical_retention,
    evaluate_official_vs_dual_channel,
    summarize_dual_channel_superiority,
)


def _synthetic_inputs(n=80):
    rng = np.random.default_rng(20260804)
    sample_ids = [f"s{i}" for i in range(n)]
    abundance = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "Akkermansia": rng.normal(0.0, 1.0, n),
            "Faecalibacterium": rng.normal(0.0, 1.0, n),
            "Bilophila": rng.normal(0.0, 1.0, n),
            "Klebsiella": rng.normal(0.0, 1.0, n),
        }
    )
    official = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "official_gmwi2_score": abundance["Akkermansia"] * 0.35 - abundance["Klebsiella"] * 0.35,
        }
    )
    nutrient_bridge = pd.DataFrame(
        {
            "Akkermansia": [0.8, -0.2],
            "Faecalibacterium": [0.6, -0.1],
            "Bilophila": [-0.2, 0.7],
            "Klebsiella": [-0.3, 0.8],
        },
        index=["Fiber, total dietary (g)", "Fatty acids, total saturated (g)"],
    )
    latent = (
        official["official_gmwi2_score"].to_numpy()
        + abundance["Faecalibacterium"].to_numpy() * 0.8
        - abundance["Bilophila"].to_numpy() * 0.8
    )
    labels = pd.Series(np.where(latent > np.median(latent), "Health", "Disease"), index=sample_ids)
    groups = pd.Series([f"subject_{i // 2}" for i in range(n)], index=sample_ids)
    masks = build_channel_vectors(nutrient_bridge.index)
    return abundance, nutrient_bridge, official, labels, groups, masks


def test_build_dual_channel_features_aligns_official_scores_and_channels():
    abundance, nutrient_bridge, official, labels, groups, masks = _synthetic_inputs()

    features = build_dual_channel_features(abundance, nutrient_bridge, official, labels, masks, groups)

    assert isinstance(features, DualChannelFeatureSet)
    assert features.feature_matrix.index.equals(features.labels.index)
    assert "official_gmwi2_z" in features.feature_matrix.columns
    assert "mac_capacity_z" in features.feature_matrix.columns
    assert "lipid_risk_capacity_z" in features.feature_matrix.columns
    assert features.official_score.name == "official_gmwi2_score"
    assert features.groups is not None


def test_dual_channel_outperforms_official_gmwi2_on_held_out_folds():
    abundance, nutrient_bridge, official, labels, groups, masks = _synthetic_inputs()
    features = build_dual_channel_features(abundance, nutrient_bridge, official, labels, masks, groups)

    fold_metrics = evaluate_official_vs_dual_channel(features, n_splits=5, seed=11)
    summary = summarize_dual_channel_superiority(fold_metrics)

    assert set(fold_metrics["model"]) == {"official_gmwi2", "gmwi2_dual_channel"}
    assert summary["delta_balanced_accuracy_mean"] > 0.05
    assert summary["delta_auroc_mean"] > 0.03
    assert summary["dual_channel_superiority_passed"] is True


def test_build_dual_channel_features_rejects_proxy_gmwi2_scores():
    abundance, nutrient_bridge, official, labels, groups, masks = _synthetic_inputs()
    proxy = official.rename(columns={"official_gmwi2_score": "genus_proxy_health_score"})

    with pytest.raises(ValueError, match="not genus_proxy_health_score"):
        build_dual_channel_features(abundance, nutrient_bridge, proxy, labels, masks, groups)


def test_evaluate_official_vs_dual_channel_keeps_groups_out_of_same_fold():
    abundance, nutrient_bridge, official, labels, groups, masks = _synthetic_inputs()
    features = build_dual_channel_features(abundance, nutrient_bridge, official, labels, masks, groups)

    fold_metrics = evaluate_official_vs_dual_channel(features, n_splits=4, seed=17)

    assert fold_metrics.groupby("fold")["n_test"].first().sum() == len(features.labels)


def test_clinical_correlation_table_reports_score_variable_pairs():
    clinical = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2", "s3"],
            "age": [20, 30, 40, 50],
            "bmi": [21, 23, 25, 29],
        }
    )
    scores = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2", "s3"],
            "official_gmwi2_score": [2.0, 1.0, 0.0, -1.0],
            "gmwi2_dual_channel_score": [2.2, 1.2, -0.2, -1.2],
        }
    )

    table = clinical_correlation_table(scores, clinical, ["age", "bmi"])

    assert set(table["score"]) == {"official_gmwi2_score", "gmwi2_dual_channel_score"}
    assert set(table["variable"]) == {"age", "bmi"}
    assert table["n"].eq(4).all()


def test_compare_clinical_retention_requires_dual_channel_to_match_gmwi2_signal():
    clinical = pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70],
            "bmi": [21, 22, 25, 28, 31, 34],
        },
        index=[f"s{i}" for i in range(6)],
    )
    official = pd.DataFrame({"official_gmwi2_score": [2, 1, 0, -1, -2, -3]}, index=clinical.index)
    dual = pd.DataFrame(
        {"gmwi2_dual_channel_score": [2.4, 1.1, 0.1, -1.0, -2.1, -3.2]},
        index=clinical.index,
    )

    retention = compare_clinical_retention(official, dual, clinical, ["age", "bmi"])

    assert retention["retains_gmwi2_direction"].all()
    assert retention["abs_delta_spearman"].max() <= 0.2
