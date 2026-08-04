import numpy as np
import pandas as pd

from gmnps.validation.response_prediction import (
    fold_local_design_matrices,
    paired_bootstrap_delta,
    regression_metrics,
    required_ablation_models,
)


def test_required_ablation_models_cover_personalized_offset_claim():
    models = set(required_ablation_models())
    assert {
        "FCS2_only",
        "food_nutrients",
        "microbiome_only",
        "food_plus_microbiome",
        "GMNPS_full",
        "population_beta",
        "shuffled_beta",
        "random_beta",
        "no_personalized_offset",
        "no_KG",
        "high_compression",
    }.issubset(models)


def test_regression_metrics_and_paired_bootstrap_delta():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred_good = np.array([1.1, 2.1, 2.9, 4.1])
    met = regression_metrics(y, pred_good)
    assert met["rmse"] < 0.2
    assert met["r2"] > 0.95

    rows = []
    for i, true in enumerate(y):
        rows.append(
            {
                "row_id": i,
                "group": f"g{i}",
                "model": "GMNPS_full",
                "y_true": true,
                "y_pred": pred_good[i],
            }
        )
        rows.append(
            {
                "row_id": i,
                "group": f"g{i}",
                "model": "food_plus_microbiome",
                "y_true": true,
                "y_pred": 2.5,
            }
        )
    delta, lo, hi = paired_bootstrap_delta(
        pd.DataFrame(rows),
        "GMNPS_full",
        "food_plus_microbiome",
        "rmse",
        "group",
        50,
        7,
    )
    assert delta < 0
    assert lo <= delta <= hi


def test_fold_local_transforms_apply_training_statistics_to_test_data():
    train = pd.DataFrame({"feature": [0.0, 2.0]}, index=["train_a", "train_b"])
    test = pd.DataFrame({"feature": [100.0]}, index=["test"])

    train_design, test_design = fold_local_design_matrices(
        train,
        test,
        ["feature"],
        {},
    )

    assert train_design["feature"].tolist() == [-1.0, 1.0]
    assert test_design.loc["test", "feature"] == 99.0
