import pytest
import pandas as pd

from gmnps.beta_i.health_index import HealthIndexConfig, HealthIndexModel
from gmnps.beta_i.nutrient_perturbation import (
    NutrientPerturbationConfig,
    build_nutrient_perturbations,
    summarize_perturbations,
)


def _model():
    return HealthIndexModel(
        genus_names=("Akkermansia", "Escherichia"),
        mean_=pd.Series({"Akkermansia": 0.0, "Escherichia": 0.0}),
        scale_=pd.Series({"Akkermansia": 1.0, "Escherichia": 1.0}),
        coefficients=pd.Series({"Akkermansia": 2.0, "Escherichia": -1.0}),
        intercept=0.0,
        config=HealthIndexConfig(),
        training_summary={
            "n_samples": 6,
            "n_healthy": 3,
            "n_nonhealthy": 3,
            "n_genera": 2,
            "n_nonzero_coefficients": 2,
            "train_auc": 1.0,
        },
    )


def test_build_nutrient_perturbations_aligns_to_health_direction_and_masks():
    nutrient_genus = pd.DataFrame(
        {
            "Akkermansia": {
                "Fiber, total dietary (g)": 0.8,
                "Fatty acids, total saturated (g)": 0.5,
                "Water (g)": 0.4,
            },
            "Escherichia": {
                "Fiber, total dietary (g)": -0.2,
                "Fatty acids, total saturated (g)": 0.6,
                "Water (g)": 0.4,
            },
        }
    )
    perturb = build_nutrient_perturbations(nutrient_genus, _model(), NutrientPerturbationConfig())
    assert list(perturb.columns) == ["Akkermansia", "Escherichia"]
    assert perturb.loc["Fiber, total dietary (g)", "Akkermansia"] > 0
    assert perturb.loc["Fiber, total dietary (g)", "Escherichia"] > 0
    assert perturb.loc["Fatty acids, total saturated (g)"].abs().sum() > 0
    config = NutrientPerturbationConfig()
    for _, row in perturb.iterrows():
        if (row != 0).any():
            assert (row.astype(float).pow(2).sum() ** 0.5) == pytest.approx(config.l2_norm)


def test_summarize_perturbations_reports_norm_and_channel():
    nutrient_genus = pd.DataFrame(
        {
            "Akkermansia": {"Fiber, total dietary (g)": 1.0},
            "Escherichia": {"Fiber, total dietary (g)": -1.0},
        }
    )
    perturb = build_nutrient_perturbations(nutrient_genus, _model(), NutrientPerturbationConfig())
    summary = summarize_perturbations(perturb)
    row = summary.set_index("nutrient").loc["Fiber, total dietary (g)"]
    assert row["channel"] == "MAC"
    assert row["l2_norm"] > 0
