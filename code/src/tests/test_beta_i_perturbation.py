import numpy as np
import pandas as pd
import pytest

from gmnps.beta_i.beta_estimator import BetaEstimatorConfig, compute_beta_matrix
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
    config = NutrientPerturbationConfig(
        mac_channel_weight=0.8,
        lipid_channel_weight=0.6,
        other_channel_weight=0.2,
        l2_norm=2.0,
    )
    perturb = build_nutrient_perturbations(nutrient_genus, _model(), config)
    assert list(perturb.columns) == ["Akkermansia", "Escherichia"]
    assert perturb.loc["Fiber, total dietary (g)", "Akkermansia"] > 0
    assert perturb.loc["Fiber, total dietary (g)", "Escherichia"] < 0
    assert perturb.loc["Fatty acids, total saturated (g)"].abs().sum() > 0
    assert np.allclose(perturb.sum(axis=1), 0.0, atol=1e-7)
    expected_norms = {
        "Fiber, total dietary (g)": config.l2_norm * config.mac_channel_weight,
        "Fatty acids, total saturated (g)": config.l2_norm * config.lipid_channel_weight,
        "Water (g)": config.l2_norm * config.other_channel_weight,
    }
    for nutrient, expected_norm in expected_norms.items():
        norm = perturb.loc[nutrient].astype(float).pow(2).sum() ** 0.5
        assert norm == pytest.approx(expected_norm)


def test_default_perturbations_stay_in_clr_space_with_channel_norms():
    model = HealthIndexModel(
        genus_names=("g1", "g2", "g3", "g4"),
        mean_=pd.Series(0.0, index=["g1", "g2", "g3", "g4"]),
        scale_=pd.Series(1.0, index=["g1", "g2", "g3", "g4"]),
        coefficients=pd.Series({"g1": 2.0, "g2": -1.0, "g3": 0.5, "g4": -0.25}),
        intercept=0.0,
        config=HealthIndexConfig(),
        training_summary={},
    )
    nutrient_genus = pd.DataFrame(
        {
            "g1": [0.8, 0.2, 0.4, 1.0],
            "g2": [-0.3, 0.7, 0.1, -1.0],
            "g3": [0.5, -0.4, 0.9, 2.0],
            "g4": [0.1, 0.6, -0.2, -2.0],
        },
        index=[
            "Fiber, total dietary (g)",
            "Fatty acids, total saturated (g)",
            "Water (g)",
            "Protein (g)",
        ],
    )

    perturb = build_nutrient_perturbations(
        nutrient_genus,
        model,
        NutrientPerturbationConfig(l2_norm=1.0, coefficient_power=0.0),
    )

    assert np.allclose(perturb.sum(axis=1), 0.0, atol=1e-7)
    norms = np.linalg.norm(perturb.to_numpy(dtype=float), axis=1)
    assert norms == pytest.approx([1.0, 1.0, 0.25, 0.25])


def test_constant_oriented_perturbation_becomes_zero_after_clr_centering():
    nutrient_genus = pd.DataFrame(
        {"Akkermansia": [1.0], "Escherichia": [-1.0]},
        index=["Fiber, total dietary (g)"],
    )

    perturb = build_nutrient_perturbations(
        nutrient_genus,
        _model(),
        NutrientPerturbationConfig(coefficient_power=0.0, l2_norm=1.0),
    )

    assert np.allclose(perturb.to_numpy(), 0.0)


def test_mac_and_lipid_evidence_directions_produce_opposite_beta_signs():
    nutrient_genus = pd.DataFrame(
        {
            "Akkermansia": {
                "Fiber, total dietary (g)": 1.0,
                "Fatty acids, total saturated (g)": 1.0,
            },
            "Escherichia": {
                "Fiber, total dietary (g)": 0.5,
                "Fatty acids, total saturated (g)": 0.5,
            },
        }
    )
    model = _model()
    perturb = build_nutrient_perturbations(
        nutrient_genus,
        model,
        NutrientPerturbationConfig(coefficient_power=0.0, l2_norm=1.0),
    )
    clr = pd.DataFrame({"Akkermansia": [0.0], "Escherichia": [0.0]}, index=["s1"])

    beta, _ = compute_beta_matrix(
        clr,
        perturb,
        model,
        BetaEstimatorConfig(dose=0.1, clip_abs_beta=100.0),
    )

    assert beta.loc["s1", "Fiber, total dietary (g)"] > 0
    assert beta.loc["s1", "Fatty acids, total saturated (g)"] < 0


def test_summarize_perturbations_reports_norm_and_channel():
    nutrient_genus = pd.DataFrame(
        {
            "Akkermansia": {"Fiber, total dietary (g)": 1.0},
            "Escherichia": {"Fiber, total dietary (g)": -0.5},
        }
    )
    perturb = build_nutrient_perturbations(
        nutrient_genus,
        _model(),
        NutrientPerturbationConfig(coefficient_power=0.0, l2_norm=1.0),
    )
    summary = summarize_perturbations(perturb)
    row = summary.set_index("nutrient").loc["Fiber, total dietary (g)"]
    assert row["channel"] == "MAC"
    assert row["channel_weight"] == 1.0
    assert row["evidence_direction"] == 1.0
    assert row["l2_norm"] > 0


def test_perturbations_can_retain_health_coefficient_magnitude():
    model = HealthIndexModel(
        genus_names=("g_strong", "g_weak", "g_bad"),
        mean_=pd.Series(0.0, index=["g_strong", "g_weak", "g_bad"]),
        scale_=pd.Series(1.0, index=["g_strong", "g_weak", "g_bad"]),
        coefficients=pd.Series({"g_strong": 9.0, "g_weak": 1.0, "g_bad": -1.0}),
        intercept=0.0,
        config=HealthIndexConfig(),
        training_summary={},
    )
    bridge = pd.DataFrame(
        {"g_strong": [1.0], "g_weak": [1.0], "g_bad": [1.0]},
        index=["Fiber, total dietary (g)"],
    )

    perturb = build_nutrient_perturbations(
        bridge,
        model,
        NutrientPerturbationConfig(coefficient_power=1.0, l2_norm=3.0),
    )

    assert abs(perturb.loc["Fiber, total dietary (g)", "g_strong"]) > abs(
        perturb.loc["Fiber, total dietary (g)", "g_weak"]
    )
    assert np.linalg.norm(perturb.loc["Fiber, total dietary (g)"].to_numpy(dtype=float)) == pytest.approx(3.0)
