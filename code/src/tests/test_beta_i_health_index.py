import numpy as np
import pandas as pd

from gmnps.beta_i.health_index import (
    HealthIndexConfig,
    derive_binary_health_labels,
    fit_health_index,
    load_health_index,
    save_health_index,
    score_health_index,
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
    loaded = load_health_index(path)
    loaded_scores = score_health_index(loaded, clr)
    assert np.allclose(scores.to_numpy(), loaded_scores.to_numpy())
