"""Reusable dietary-response prediction validation utilities."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def train_fitted_standardize(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute and standardize both partitions using training statistics only."""

    train_numeric = train.apply(pd.to_numeric, errors="coerce")
    test_numeric = test.reindex(columns=train.columns).apply(pd.to_numeric, errors="coerce")
    medians = train_numeric.median(axis=0).fillna(0.0)
    train_filled = train_numeric.fillna(medians)
    test_filled = test_numeric.fillna(medians)
    means = train_filled.mean(axis=0)
    scales = train_filled.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    return (train_filled - means) / scales, (test_filled - means) / scales


def train_fitted_pca(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_components: int,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit PCA on a training partition and project its held-out partition."""

    if n_components <= 0:
        raise ValueError("n_components must be positive")
    train_scaled, test_scaled = train_fitted_standardize(train, test)
    _, _, components = np.linalg.svd(train_scaled.to_numpy(dtype=float), full_matrices=False)
    n = min(n_components, components.shape[0])
    columns = [f"{prefix}{index + 1}" for index in range(n)]
    basis = components[:n].T
    return (
        pd.DataFrame(train_scaled.to_numpy(dtype=float) @ basis, index=train.index, columns=columns),
        pd.DataFrame(test_scaled.to_numpy(dtype=float) @ basis, index=test.index, columns=columns),
    )


def fold_local_design_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    pca_blocks: dict[str, tuple[list[str], int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build matched train/test matrices with fold-isolated PCA and scaling."""

    missing = set(columns).difference(train.columns)
    if missing:
        raise ValueError(f"training data missing feature columns: {sorted(missing)}")
    remaining = list(columns)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for prefix, (block_columns, n_components) in pca_blocks.items():
        selected = [column for column in block_columns if column in remaining]
        if not selected:
            continue
        if len(selected) != len(block_columns):
            raise ValueError(f"PCA block {prefix} must be included as a complete feature block")
        train_pca, test_pca = train_fitted_pca(
            train[selected], test[selected], n_components, prefix
        )
        train_parts.append(train_pca)
        test_parts.append(test_pca)
        remaining = [column for column in remaining if column not in selected]

    if remaining:
        train_scaled, test_scaled = train_fitted_standardize(
            train[remaining], test[remaining]
        )
        train_parts.append(train_scaled)
        test_parts.append(test_scaled)

    train_design = pd.concat(train_parts, axis=1) if train_parts else pd.DataFrame(index=train.index)
    test_design = pd.concat(test_parts, axis=1) if test_parts else pd.DataFrame(index=test.index)
    train_design.insert(0, "intercept", 1.0)
    test_design.insert(0, "intercept", 1.0)
    return train_design, test_design.reindex(columns=train_design.columns)


def group_folds(groups: pd.Series, n_folds: int, seed: int) -> list[np.ndarray]:
    """Create deterministic folds that keep all rows from a group together."""

    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    groups_as_str = groups.astype(str)
    unique = pd.Series(groups_as_str.unique())
    rng = np.random.default_rng(seed)
    unique = unique.iloc[rng.permutation(len(unique))].tolist()
    fold_groups: list[list[str]] = [[] for _ in range(n_folds)]
    fold_sizes = np.zeros(n_folds, dtype=int)
    counts = groups_as_str.value_counts().to_dict()
    for group in unique:
        fold = int(np.argmin(fold_sizes))
        fold_groups[fold].append(group)
        fold_sizes[fold] += int(counts[group])
    return [np.where(groups_as_str.isin(fold_groups[fold]).to_numpy())[0] for fold in range(n_folds)]


def fit_ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    """Fit ridge regression with an unpenalized intercept in column zero."""

    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    penalty = np.eye(x_train.shape[1], dtype=float) * float(alpha)
    if penalty.size:
        penalty[0, 0] = 0.0
    lhs = x_train.T @ x_train + penalty
    rhs = x_train.T @ y_train
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
    return x_test @ beta


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Return RMSE, MAE, R2, Pearson and Spearman metrics."""

    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if y.shape != pred.shape:
        raise ValueError("y and pred must have the same shape")
    resid = y - pred
    sst = float(np.sum((y - np.mean(y)) ** 2))
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    r2 = float(1 - np.sum(resid**2) / sst) if sst > 1e-12 else math.nan
    y_series = pd.Series(y)
    pred_series = pd.Series(pred)
    if y_series.nunique() < 2 or pred_series.nunique() < 2:
        pearson = math.nan
        spearman = math.nan
    else:
        pearson = float(y_series.corr(pred_series, method="pearson"))
        spearman = float(
            y_series.rank(method="average").corr(
                pred_series.rank(method="average"),
                method="pearson",
            )
        )
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": pearson, "spearman": spearman}


def paired_bootstrap_delta(
    predictions: pd.DataFrame,
    model: str,
    baseline: str,
    metric: str,
    group_col: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap paired model-minus-baseline metric deltas by independent group."""

    required = {"row_id", "model", "y_true", "y_pred", group_col}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")
    if metric not in {"rmse", "mae", "r2", "pearson", "spearman"}:
        raise ValueError("metric must be one of rmse, mae, r2, pearson, spearman")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    model_frame = predictions.loc[
        predictions["model"].eq(model),
        ["row_id", group_col, "y_true", "y_pred"],
    ].rename(columns={"y_pred": "pred_model", group_col: "group", "y_true": "y"})
    baseline_frame = predictions.loc[
        predictions["model"].eq(baseline),
        ["row_id", "y_pred"],
    ].rename(columns={"y_pred": "pred_baseline"})
    paired = model_frame.merge(baseline_frame, on="row_id", how="inner").dropna()
    if paired.empty:
        return (math.nan, math.nan, math.nan)

    y = paired["y"].to_numpy(dtype=float)
    pred_model = paired["pred_model"].to_numpy(dtype=float)
    pred_baseline = paired["pred_baseline"].to_numpy(dtype=float)

    def score(indices: np.ndarray) -> float:
        model_score = regression_metrics(y[indices], pred_model[indices])[metric]
        baseline_score = regression_metrics(y[indices], pred_baseline[indices])[metric]
        return float(model_score - baseline_score)

    observed = score(np.arange(len(paired)))
    rng = np.random.default_rng(seed)
    group_values = paired["group"].astype(str).to_numpy()
    unique = pd.unique(group_values)
    group_indices = {group: np.where(group_values == group)[0] for group in unique}
    reps = []
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(unique, size=len(unique), replace=True)
        sampled_indices = np.concatenate([group_indices[group] for group in sampled_groups])
        reps.append(score(sampled_indices))
    return observed, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def required_ablation_models() -> list[str]:
    """Return the article-required dietary-response model and ablation names."""

    return [
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
    ]
