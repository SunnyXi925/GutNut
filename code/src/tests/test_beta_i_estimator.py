import pandas as pd
import pytest

from gmnps.beta_i.beta_estimator import BetaEstimatorConfig, compute_beta_matrix
from gmnps.beta_i.health_index import HealthIndexConfig, HealthIndexModel, score_health_index


def _model():
    return HealthIndexModel(
        genus_names=("Akkermansia", "Escherichia"),
        mean_=pd.Series({"Akkermansia": 0.0, "Escherichia": 0.0}),
        scale_=pd.Series({"Akkermansia": 1.0, "Escherichia": 1.0}),
        coefficients=pd.Series({"Akkermansia": 4.0, "Escherichia": -4.0}),
        intercept=0.0,
        config=HealthIndexConfig(),
        training_summary={"n_samples": 4, "n_healthy": 2, "n_nonhealthy": 2, "n_genera": 2, "n_nonzero_coefficients": 2, "train_auc": 1.0},
    )


def test_compute_beta_matrix_is_finite_difference_of_health_index():
    clr = pd.DataFrame(
        {"Akkermansia": [0.0, 1.0], "Escherichia": [0.0, -1.0]},
        index=["s1", "s2"],
    )
    perturb = pd.DataFrame(
        {"Akkermansia": {"Fiber, total dietary (g)": 1.0}, "Escherichia": {"Fiber, total dietary (g)": -1.0}}
    )
    beta, diag = compute_beta_matrix(
        clr,
        perturb,
        _model(),
        BetaEstimatorConfig(
            dose=0.1,
            clip_abs_beta=100.0,
            response_scale="probability",
            difference="forward",
        ),
    )
    model = _model()
    baseline = score_health_index(model, clr)
    expected = (
        score_health_index(model, clr + 0.1 * perturb.loc["Fiber, total dietary (g)"]) - baseline
    ) / 0.1
    assert beta.shape == (2, 1)
    pd.testing.assert_series_equal(
        beta["Fiber, total dietary (g)"].astype(float), expected, check_names=False, rtol=1e-6, atol=1e-6
    )
    assert beta.loc["s1", "Fiber, total dietary (g)"] > 0
    assert beta.loc["s1", "Fiber, total dietary (g)"] > beta.loc["s2", "Fiber, total dietary (g)"]
    assert diag.set_index("sample_id").loc["s1", "baseline_health_index"] == baseline.loc["s1"]


def test_compute_beta_matrix_clips_large_values():
    clr = pd.DataFrame({"Akkermansia": [0.0], "Escherichia": [0.0]}, index=["s1"])
    perturb = pd.DataFrame(
        {"Akkermansia": {"Fiber, total dietary (g)": 50.0}, "Escherichia": {"Fiber, total dietary (g)": -50.0}}
    )
    beta, _ = compute_beta_matrix(
        clr,
        perturb,
        _model(),
        BetaEstimatorConfig(
            dose=1.0,
            clip_abs_beta=0.5,
            response_scale="probability",
            difference="forward",
        ),
    )
    assert beta.abs().max().max() <= 0.5


def test_compute_beta_matrix_preserves_non_string_nutrient_labels_for_lookup():
    clr = pd.DataFrame({"Akkermansia": [0.0], "Escherichia": [0.0]}, index=["s1"])
    perturb = pd.DataFrame(
        {"Akkermansia": {101: 1.0}, "Escherichia": {101: -1.0}}
    )

    beta, _ = compute_beta_matrix(
        clr,
        perturb,
        _model(),
        BetaEstimatorConfig(
            dose=0.1,
            clip_abs_beta=100.0,
            response_scale="probability",
            difference="forward",
        ),
    )
    model = _model()
    expected = (
        score_health_index(model, clr + 0.1 * perturb.loc[101]) - score_health_index(model, clr)
    ) / 0.1

    assert list(beta.columns) == ["101"]
    assert beta.loc["s1", "101"] == pytest.approx(expected.loc["s1"], abs=1e-6)


def test_compute_beta_matrix_rejects_non_positive_batch_size():
    clr = pd.DataFrame({"Akkermansia": [0.0], "Escherichia": [0.0]}, index=["s1"])
    perturb = pd.DataFrame(
        {"Akkermansia": {"fiber": 1.0}, "Escherichia": {"fiber": -1.0}}
    )

    with pytest.raises(ValueError, match="batch_size must be positive"):
        compute_beta_matrix(clr, perturb, _model(), BetaEstimatorConfig(batch_size=0))


def test_logit_scale_central_difference_recovers_noncompressed_slope():
    clr = pd.DataFrame(
        {"Akkermansia": [4.0], "Escherichia": [-4.0]},
        index=["disease_like"],
    )
    perturb = pd.DataFrame(
        {"Akkermansia": {"Fiber, total dietary (g)": 1.0}, "Escherichia": {"Fiber, total dietary (g)": -1.0}}
    )
    model = _model()

    probability_beta, _ = compute_beta_matrix(
        clr,
        perturb,
        model,
        BetaEstimatorConfig(dose=0.25, clip_abs_beta=100.0, response_scale="probability", difference="forward"),
    )
    logit_beta, diag = compute_beta_matrix(
        clr,
        perturb,
        model,
        BetaEstimatorConfig(dose=0.25, clip_abs_beta=100.0, response_scale="logit", difference="central"),
    )

    assert logit_beta.loc["disease_like", "Fiber, total dietary (g)"] > 6.0
    assert logit_beta.loc["disease_like", "Fiber, total dietary (g)"] > probability_beta.loc[
        "disease_like", "Fiber, total dietary (g)"
    ] * 10
    assert diag.set_index("sample_id").loc["disease_like", "beta_response_scale"] == "logit"
    assert diag.set_index("sample_id").loc["disease_like", "beta_difference"] == "central"
