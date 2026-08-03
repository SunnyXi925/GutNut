from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HealthIndexConfig:
    c_value: float = 0.25
    max_iter: int = 2000
    random_state: int = 20260803
    min_abs_coefficient: float = 1e-12


@dataclass(frozen=True)
class HealthIndexModel:
    genus_names: tuple[str, ...]
    mean_: pd.Series
    scale_: pd.Series
    coefficients: pd.Series
    intercept: float
    config: HealthIndexConfig
    training_summary: dict[str, float | int]


def derive_binary_health_labels(metadata: pd.DataFrame) -> pd.Series:
    required = {"sample_id", "phenotype_label", "disease"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")
    frame = metadata[["sample_id", "phenotype_label", "disease"]].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    phenotype = frame["phenotype_label"].fillna("").astype(str).str.lower()
    disease = frame["disease"].fillna("").astype(str).str.lower()
    healthy = phenotype.eq("health") | disease.eq("healthy")
    disease_named = phenotype.isin({"ibd", "crc", "cvd", "t2d"}) | disease.ne("healthy")
    labels = pd.Series(np.where(healthy, 1, np.where(disease_named, 0, np.nan)), index=frame["sample_id"])
    labels = labels.dropna().astype(int)
    labels.name = "health_label"
    return labels[~labels.index.duplicated(keep="first")]


def _standardize_fit(clr: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    x = clr.astype(float).copy()
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0).replace(0.0, 1.0)
    return (x - mean) / scale, mean, scale


def _standardize_apply(clr: pd.DataFrame, model: HealthIndexModel) -> pd.DataFrame:
    x = clr.reindex(columns=model.genus_names).astype(float)
    x = x.fillna(model.mean_)
    return (x - model.mean_) / model.scale_


def _numpy_logistic_fit(x: np.ndarray, y: np.ndarray, config: HealthIndexConfig) -> tuple[np.ndarray, float]:
    """Fit a small deterministic L2-regularized logistic model without SciPy."""
    weights = np.zeros(x.shape[1], dtype=float)
    intercept = 0.0
    learning_rate = 0.5
    penalty = 1.0 / max(float(config.c_value), np.finfo(float).eps)
    for _ in range(config.max_iter):
        linear = np.clip(intercept + x @ weights, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-linear))
        error = probabilities - y
        grad_intercept = float(error.mean())
        grad_weights = (x.T @ error) / len(y) + penalty * weights / len(y)
        step = np.concatenate(([grad_intercept], grad_weights))
        intercept -= learning_rate * grad_intercept
        weights -= learning_rate * grad_weights
        if np.max(np.abs(step)) < 1e-9:
            break
    return weights, intercept


def fit_health_index(clr: pd.DataFrame, labels: pd.Series, config: HealthIndexConfig) -> HealthIndexModel:
    if config.c_value <= 0 or config.max_iter <= 0:
        raise ValueError("c_value and max_iter must be positive")
    x = clr.copy()
    x.index = x.index.astype(str)
    y = labels.copy()
    y.index = y.index.astype(str)
    common = x.index.intersection(y.index)
    if len(common) < 6:
        raise ValueError(f"need at least 6 labelled samples, found {len(common)}")
    x = x.loc[common]
    y = y.loc[common].astype(int)
    if not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError("health labels must contain both 0 and 1")
    if x.isna().any().any():
        x = x.fillna(x.mean(axis=0))
    x_std, mean, scale = _standardize_fit(x)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ModuleNotFoundError:
        weights, intercept = _numpy_logistic_fit(x_std.to_numpy(float), y.to_numpy(int), config)
        raw_scores = 1.0 / (1.0 + np.exp(-np.clip(intercept + x_std.to_numpy(float) @ weights, -40.0, 40.0)))
        train_auc = _roc_auc(y.to_numpy(int), raw_scores)
    else:
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=config.c_value, max_iter=config.max_iter, random_state=config.random_state)
        clf.fit(x_std.to_numpy(float), y.to_numpy(int))
        weights, intercept = clf.coef_[0], float(clf.intercept_[0])
        raw_scores = clf.predict_proba(x_std.to_numpy(float))[:, 1]
        train_auc = float(roc_auc_score(y, raw_scores))

    coefficients = pd.Series(weights, index=x_std.columns, name="health_coefficient")
    coefficients = coefficients.where(coefficients.abs() >= config.min_abs_coefficient, 0.0)
    summary = {"n_samples": int(len(x_std)), "n_healthy": int(y.sum()), "n_nonhealthy": int((1 - y).sum()), "n_genera": int(x_std.shape[1]), "n_nonzero_coefficients": int((coefficients.abs() > 0).sum()), "train_auc": train_auc}
    return HealthIndexModel(tuple(x_std.columns.astype(str)), mean, scale, coefficients, intercept, config, summary)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    comparisons = (positives[:, None] > negatives[None, :]).sum() + 0.5 * (positives[:, None] == negatives[None, :]).sum()
    return float(comparisons / (len(positives) * len(negatives)))


def score_health_index(model: HealthIndexModel, clr: pd.DataFrame) -> pd.Series:
    x_std = _standardize_apply(clr, model)
    linear = model.intercept + x_std.to_numpy(float) @ model.coefficients.to_numpy(float)
    probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
    return pd.Series(probability, index=clr.index, name="gut_microbiome_health_index")


def save_health_index(model: HealthIndexModel, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
    except ModuleNotFoundError:
        with path.open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        joblib.dump(model, path)


def load_health_index(path: Path) -> HealthIndexModel:
    path = Path(path)
    try:
        import joblib
    except ModuleNotFoundError:
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
    else:
        loaded = joblib.load(path)
    if not isinstance(loaded, HealthIndexModel):
        raise TypeError(f"expected HealthIndexModel in {path}, found {type(loaded)!r}")
    return loaded
