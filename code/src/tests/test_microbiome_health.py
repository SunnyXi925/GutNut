import pandas as pd
import pytest

from gmnps.validation.microbiome_health import (
    genus_proxy_health_score,
    health_nonhealth_metrics,
    load_official_gmwi2_scores,
)


def test_genus_proxy_health_score_orders_health_like_samples():
    abundance = pd.DataFrame(
        {
            "sample_id": ["h", "d"],
            "Akkermansia": [10.0, 0.0],
            "Bifidobacterium": [8.0, 0.0],
            "Escherichia": [0.0, 10.0],
            "Klebsiella": [0.0, 8.0],
        }
    )
    score = genus_proxy_health_score(abundance)

    assert score.loc["h"] > score.loc["d"]
    assert score.attrs["mode"] == "genus_proxy_not_official_gmwi2"


def test_health_nonhealth_metrics_reports_balanced_accuracy():
    scores = pd.Series({"h1": 1.0, "h2": 0.8, "d1": -1.0, "d2": -0.5})
    labels = pd.Series({"h1": "Health", "h2": "Health", "d1": "Disease", "d2": "Disease"})

    metrics = health_nonhealth_metrics(scores, labels)

    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["n_samples"] == 4


def test_load_official_gmwi2_scores_preserves_explicit_official_label(tmp_path):
    path = tmp_path / "official_gmwi2.csv"
    pd.DataFrame({"sample_id": ["s1", "s2"], "gmwi2": [0.4, -0.3]}).to_csv(path, index=False)

    scores = load_official_gmwi2_scores(path)

    assert scores.columns.tolist() == ["sample_id", "official_gmwi2_score"]
    assert scores.attrs["mode"] == "official_gmwi2"


def test_runner_rejects_official_mode_without_score_file(tmp_path):
    from scripts.run_personalized_calibration_experiments import build_parser, run

    args = build_parser().parse_args(
        ["--root", str(tmp_path), "--microbiome-health-mode", "official_gmwi2"]
    )

    with pytest.raises(ValueError, match="official_gmwi2 mode requires --official-gmwi2-scores"):
        run(args)
