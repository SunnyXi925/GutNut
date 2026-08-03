import pandas as pd

from gmnps.beta_i.beta_estimator import BetaEstimatorConfig, compute_beta_matrix
from gmnps.beta_i.health_index import HealthIndexConfig, HealthIndexModel, score_health_index


def _model():
    return HealthIndexModel(
        genus_names=("Akkermansia", "Escherichia"),
        mean_=pd.Series({"Akkermansia": 0.0, "Escherichia": 0.0}),
        scale_=pd.Series({"Akkermansia": 1.0, "Escherichia": 1.0}),
        coefficients=pd.Series({"Akkermansia": 2.0, "Escherichia": -2.0}),
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
    beta, diag = compute_beta_matrix(clr, perturb, _model(), BetaEstimatorConfig(dose=0.1, clip_abs_beta=100.0))
    baseline = score_health_index(_model(), clr)
    assert beta.shape == (2, 1)
    assert beta.loc["s1", "Fiber, total dietary (g)"] > 0
    assert beta.loc["s1", "Fiber, total dietary (g)"] > beta.loc["s2", "Fiber, total dietary (g)"]
    assert diag.set_index("sample_id").loc["s1", "baseline_health_index"] == baseline.loc["s1"]


def test_compute_beta_matrix_clips_large_values():
    clr = pd.DataFrame({"Akkermansia": [0.0], "Escherichia": [0.0]}, index=["s1"])
    perturb = pd.DataFrame(
        {"Akkermansia": {"Fiber, total dietary (g)": 50.0}, "Escherichia": {"Fiber, total dietary (g)": -50.0}}
    )
    beta, _ = compute_beta_matrix(clr, perturb, _model(), BetaEstimatorConfig(dose=1.0, clip_abs_beta=0.5))
    assert beta.abs().max().max() <= 0.5
