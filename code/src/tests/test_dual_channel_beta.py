import numpy as np
import pandas as pd
import pytest

from gmnps.beta_i.dual_channel_beta import DualChannelBetaConfig, compute_dual_channel_beta
from gmnps.scoring.masks import build_channel_vectors


def _synthetic_inputs():
    clr = pd.DataFrame(
        {
            "Akkermansia": [0.8, -0.3],
            "Faecalibacterium": [0.5, -0.2],
            "Bilophila": [-0.4, 0.9],
            "Klebsiella": [-0.2, 1.1],
        },
        index=["healthy_like", "disease_like"],
    )
    perturb = pd.DataFrame(
        {
            "Akkermansia": [1.0, -0.2],
            "Faecalibacterium": [0.8, -0.1],
            "Bilophila": [-0.2, 1.0],
            "Klebsiella": [-0.1, 0.9],
        },
        index=["Fiber, total dietary (g)", "Fatty acids, total saturated (g)"],
    )
    weights = pd.Series(
        {
            "official_gmwi2_z": 1.0,
            "mac_capacity_z": 0.8,
            "lipid_risk_capacity_z": -0.8,
            "mac_x_gmwi2": 0.1,
            "lipid_x_gmwi2": -0.1,
        }
    )
    return clr, perturb, weights, build_channel_vectors(perturb.index)


def test_dual_channel_beta_preserves_channel_specific_directionality():
    clr, perturb, weights, masks = _synthetic_inputs()

    beta, diagnostics = compute_dual_channel_beta(
        clr,
        perturb,
        weights,
        masks,
        DualChannelBetaConfig(dose=0.25, compression_temperature=0.35),
    )

    assert beta.loc["disease_like", "Fiber, total dietary (g)"] > 0
    assert beta.loc["disease_like", "Fatty acids, total saturated (g)"] < 0
    assert diagnostics["beta_delta_span_p95_p05"].min() > 0.05
    assert diagnostics["beta_model"].eq("gmwi2_dual_channel").all()


def test_dual_channel_beta_aligns_to_sample_index_and_nutrient_columns():
    clr, perturb, weights, masks = _synthetic_inputs()

    beta, diagnostics = compute_dual_channel_beta(
        clr,
        perturb.reindex(columns=["Klebsiella", "Akkermansia", "missing_genus"]),
        weights,
        masks,
        DualChannelBetaConfig(dose=0.25, compression_temperature=0.35),
    )

    assert beta.index.tolist() == ["healthy_like", "disease_like"]
    assert beta.columns.tolist() == ["Fiber, total dietary (g)", "Fatty acids, total saturated (g)"]
    assert diagnostics["sample_id"].tolist() == ["healthy_like", "disease_like"]
    assert np.isfinite(beta.to_numpy(dtype=float)).all()
    assert np.isfinite(diagnostics["beta_mean"].to_numpy(dtype=float)).all()


def test_dual_channel_beta_rejects_invalid_config():
    clr, perturb, weights, masks = _synthetic_inputs()

    with pytest.raises(ValueError, match="response_scale"):
        compute_dual_channel_beta(
            clr,
            perturb,
            weights,
            masks,
            DualChannelBetaConfig(response_scale="odds"),
        )
