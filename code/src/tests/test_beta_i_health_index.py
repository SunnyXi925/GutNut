import numpy as np
import pandas as pd
import json
import hashlib
import pytest
import sys

from gmnps.beta_i.health_index import (
    HealthIndexConfig,
    derive_binary_health_labels,
    fit_health_index,
    load_health_index,
    save_health_index,
    score_health_index,
    _numpy_logistic_fit,
)


def _toy_clr():
    return pd.DataFrame(
        {
            "Akkermansia": [2.0, 1.8, 1.5, -1.0, -1.2, -1.4],
            "Faecalibacterium": [1.5, 1.2, 1.0, -1.1, -0.9, -1.3],
            "Escherichia": [-1.2, -1.0, -0.8, 1.4, 1.6, 1.2],
        },
        index=["s1", "s2", "s3", "s4", "s5", "s6"],
    )


def test_derive_binary_health_labels_from_cmd_metadata():
    metadata = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "phenotype_label": ["Health", "IBD", "CRC", "T2D"],
            "disease": ["healthy", "IBD", "CRC", "T2D"],
        }
    )
    labels = derive_binary_health_labels(metadata)
    assert labels.to_dict() == {"s1": 1, "s2": 0, "s3": 0, "s4": 0}


def test_unknown_metadata_is_excluded_and_contradictions_raise():
    metadata = pd.DataFrame({
        "sample_id": ["healthy", "ibd", "unknown", "empty", "conflict"],
        "phenotype_label": ["Health", "IBD", "Other", "", "IBD"],
        "disease": ["healthy", "IBD", "unknown", "", "healthy"],
    })
    with pytest.raises(ValueError, match="contradictory"):
        derive_binary_health_labels(metadata)
    labels = derive_binary_health_labels(metadata.drop(index=4))
    assert labels.to_dict() == {"healthy": 1, "ibd": 0}


def test_missing_sample_ids_are_excluded_and_duplicate_conflicts_raise():
    metadata = pd.DataFrame({
        "sample_id": [None, "", "  ", "s1", "s1"],
        "phenotype_label": ["Health", "Health", "Health", "Health", "IBD"],
        "disease": ["healthy", "healthy", "healthy", "healthy", "IBD"],
    })
    with pytest.raises(ValueError, match="conflicting health labels"):
        derive_binary_health_labels(metadata)

    labels = derive_binary_health_labels(metadata.iloc[:3])
    assert labels.empty


def test_numpy_fallback_is_l1_sparse():
    x = np.array([[2.0, 0.1, 0.0], [1.8, -0.1, 0.0], [-2.0, 0.1, 0.0], [-1.7, -0.1, 0.0]])
    y = np.array([1, 1, 0, 0])
    coefficients, _ = _numpy_logistic_fit(x, y, HealthIndexConfig(c_value=0.25, max_iter=1000))
    assert np.count_nonzero(coefficients) < len(coefficients)


def test_default_config_fallback_keeps_signal_and_scores_healthy_higher(monkeypatch):
    base = _toy_clr()
    clr = pd.concat([base.iloc[:3], base.iloc[:3], base.iloc[3:], base.iloc[3:]], ignore_index=True)
    clr.index = [f"s{i}" for i in range(len(clr))]
    labels = np.array([1] * 6 + [0] * 6)
    coefficients, intercept = _numpy_logistic_fit(clr.to_numpy(), labels, HealthIndexConfig())
    assert np.count_nonzero(coefficients) > 0
    scores = 1.0 / (1.0 + np.exp(-(intercept + clr.to_numpy() @ coefficients)))
    assert scores[:3].mean() > scores[3:].mean()

    monkeypatch.setitem(sys.modules, "sklearn", None)
    model = fit_health_index(clr, pd.Series(labels, index=clr.index), HealthIndexConfig())
    assert model.training_summary["n_nonzero_coefficients"] > 0
    assert score_health_index(model, clr).iloc[:3].mean() > score_health_index(model, clr).iloc[3:].mean()


def test_fit_health_index_scores_healthy_samples_higher(tmp_path):
    clr = _toy_clr()
    labels = pd.Series([1, 1, 1, 0, 0, 0], index=clr.index, name="health_label")
    model = fit_health_index(clr, labels, HealthIndexConfig(c_value=10.0, max_iter=500, random_state=7))
    scores = score_health_index(model, clr)
    assert scores.loc[["s1", "s2", "s3"]].mean() > scores.loc[["s4", "s5", "s6"]].mean()
    assert scores.between(0.0, 1.0).all()
    assert set(model.genus_names) == {"Akkermansia", "Faecalibacterium", "Escherichia"}
    assert np.isfinite(model.intercept)

    path = tmp_path / "health_index.joblib"
    save_health_index(model, path)
    manifest_path = path.with_name(f"{path.name}.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["method"] == "l1_logistic_gmwi2_style"
    assert manifest["config"]["c_value"] == 10.0
    assert manifest["training_summary"] == model.training_summary
    assert manifest["model_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_health_index(path)
    loaded_scores = score_health_index(loaded, clr)
    assert np.allclose(scores.to_numpy(), loaded_scores.to_numpy())


def test_integer_feature_columns_round_trip_score():
    clr = _toy_clr()
    clr.columns = [101, 202, 303]
    labels = pd.Series([1, 1, 1, 0, 0, 0], index=clr.index, name="health_label")

    model = fit_health_index(clr, labels, HealthIndexConfig(c_value=10.0, max_iter=500, random_state=7))
    scores = score_health_index(model, clr)

    assert model.genus_names == ("101", "202", "303")
    assert scores.index.equals(clr.index)
    assert scores.between(0.0, 1.0).all()
