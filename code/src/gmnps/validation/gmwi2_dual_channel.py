"""Official GMWI2 versus dual-channel microbiome-health validation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
import pandas as pd


_OFFICIAL_SCORE_COLUMNS = ("official_gmwi2_score", "gmwi2_score", "gmwi2")


@dataclass(frozen=True)
class DualChannelFeatureSet:
    """Aligned official GMWI2 baseline, dual-channel features, labels and groups."""

    feature_matrix: pd.DataFrame
    labels: pd.Series
    official_score: pd.Series
    groups: pd.Series | None = None


def _sample_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "sample_id" in result.columns:
        result = result.set_index("sample_id", drop=True)
    result.index = result.index.astype(str)
    if result.index.has_duplicates:
        raise ValueError("sample identifiers must be unique")
    return result


def _series_index(series: pd.Series, name: str) -> pd.Series:
    result = series.copy()
    result.index = result.index.astype(str)
    if result.index.has_duplicates:
        raise ValueError(f"{name} sample identifiers must be unique")
    return result


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return (values * 0.0).rename(series.name)
    return ((values - values.mean()) / sd).rename(series.name)


def _mask_vector(
    channel_masks: dict[str, np.ndarray | list[str] | str],
    key: str,
    nutrient_index: pd.Index,
) -> np.ndarray:
    if key not in channel_masks:
        raise ValueError(f"channel_masks must contain {key!r}")
    mask = channel_masks[key]
    if isinstance(mask, np.ndarray):
        result = mask.astype(bool)
    elif isinstance(mask, str):
        result = nutrient_index.astype(str).eq(mask).to_numpy(dtype=bool)
    elif isinstance(mask, Iterable):
        names = {str(value) for value in mask}
        result = nutrient_index.astype(str).isin(names).to_numpy(dtype=bool)
    else:
        raise TypeError(f"unsupported {key!r} mask type: {type(mask).__name__}")
    if len(result) != len(nutrient_index):
        raise ValueError(f"{key!r} mask length must match nutrient_bridge rows")
    return result


def _official_column(official_scores: pd.DataFrame) -> str:
    for column in _OFFICIAL_SCORE_COLUMNS:
        if column in official_scores.columns:
            return column
    if "genus_proxy_health_score" in official_scores.columns:
        raise ValueError("official_scores must use official GMWI2 output, not genus_proxy_health_score")
    raise ValueError("official_scores must contain one of: " + ", ".join(_OFFICIAL_SCORE_COLUMNS))


def _capacity(
    abundance: pd.DataFrame,
    bridge: pd.DataFrame,
    mask: np.ndarray,
    name: str,
) -> pd.Series:
    if not mask.any():
        return pd.Series(0.0, index=abundance.index, name=name)
    sub = bridge.loc[mask].reindex(columns=abundance.columns).fillna(0.0)
    weights = sub.mean(axis=0)
    score = abundance.mul(weights, axis=1).sum(axis=1)
    return _zscore(score.rename(name))


def _safe_feature_name(prefix: str, nutrient: object) -> str:
    text = str(nutrient)
    safe = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()
    return f"{prefix}__{safe[:80]}"


def _nutrient_axis_features(
    abundance: pd.DataFrame,
    bridge: pd.DataFrame,
    mask: np.ndarray,
    prefix: str,
) -> pd.DataFrame:
    rows = bridge.loc[mask]
    if rows.empty:
        return pd.DataFrame(index=abundance.index)
    data = {}
    for nutrient, weights in rows.iterrows():
        score = abundance.mul(weights.reindex(abundance.columns).fillna(0.0), axis=1).sum(axis=1)
        data[_safe_feature_name(prefix, nutrient)] = _zscore(score.rename(str(nutrient)))
    return pd.DataFrame(data, index=abundance.index)


def build_dual_channel_features(
    abundance: pd.DataFrame,
    nutrient_bridge: pd.DataFrame,
    official_scores: pd.DataFrame,
    labels: pd.Series,
    channel_masks: dict[str, np.ndarray | list[str] | str],
    groups: pd.Series | None = None,
) -> DualChannelFeatureSet:
    """Construct leakage-safe dual-channel features around immutable official GMWI2."""

    abundance_i = _sample_index(abundance).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    official_i = _sample_index(official_scores)
    official_col = _official_column(official_i)
    labels_i = _series_index(labels, "labels")

    common = abundance_i.index.intersection(official_i.index).intersection(labels_i.index)
    if len(common) < 10:
        raise ValueError("at least 10 aligned samples are required")

    abundance_i = abundance_i.reindex(common)
    bridge = nutrient_bridge.reindex(columns=abundance_i.columns).fillna(0.0)
    bridge.index = bridge.index.astype(str)
    mac = _mask_vector(channel_masks, "mac", bridge.index)
    lipid = _mask_vector(channel_masks, "lipid", bridge.index)

    official = pd.to_numeric(official_i.reindex(common)[official_col], errors="coerce")
    if official.isna().any():
        raise ValueError("official GMWI2 scores must be numeric for all aligned samples")
    official = official.rename("official_gmwi2_score")

    feature_matrix = pd.DataFrame(
        {
            "official_gmwi2_z": _zscore(official).rename("official_gmwi2_z"),
            "mac_capacity_z": _capacity(abundance_i, bridge, mac, "mac_capacity_z"),
            "lipid_risk_capacity_z": _capacity(abundance_i, bridge, lipid, "lipid_risk_capacity_z"),
        },
        index=common,
    )
    feature_matrix["mac_x_gmwi2"] = feature_matrix["mac_capacity_z"] * feature_matrix["official_gmwi2_z"]
    feature_matrix["lipid_x_gmwi2"] = (
        feature_matrix["lipid_risk_capacity_z"] * feature_matrix["official_gmwi2_z"]
    )
    nutrient_axes = pd.concat(
        [
            _nutrient_axis_features(abundance_i, bridge, mac, "mac_nutrient"),
            _nutrient_axis_features(abundance_i, bridge, lipid, "lipid_nutrient"),
        ],
        axis=1,
    )
    if not nutrient_axes.empty:
        feature_matrix = pd.concat([feature_matrix, nutrient_axes], axis=1)

    aligned_labels = labels_i.reindex(common)
    if aligned_labels.isna().any():
        raise ValueError("labels must be present for all aligned samples")

    group_out = None
    if groups is not None:
        groups_i = _series_index(groups, "groups").reindex(common)
        group_out = groups_i.where(groups_i.notna(), pd.Series(common, index=common)).astype(str)

    return DualChannelFeatureSet(
        feature_matrix=feature_matrix.astype(float),
        labels=aligned_labels,
        official_score=official,
        groups=group_out,
    )


def _binary_labels(labels: pd.Series) -> np.ndarray:
    normalized = labels.astype(str).str.strip().str.lower()
    y = normalized.isin({"health", "healthy"}).astype(int).to_numpy()
    if y.min() == y.max():
        raise ValueError("labels must contain health and non-health samples")
    return y


def _balanced_accuracy_at_threshold(y_true: np.ndarray, score: np.ndarray, threshold: float) -> float:
    pred = (score >= threshold).astype(int)
    positive = y_true == 1
    negative = y_true == 0
    sensitivity = float((pred[positive] == 1).mean())
    specificity = float((pred[negative] == 0).mean())
    return (sensitivity + specificity) / 2.0


def _best_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(score) == 0:
        return float("nan")
    candidates = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 97)))
    best = float(np.median(score))
    best_ba = -np.inf
    for threshold in candidates:
        ba = _balanced_accuracy_at_threshold(y_true, score, float(threshold))
        if ba > best_ba:
            best = float(threshold)
            best_ba = ba
    return best


def _metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {"balanced_accuracy": float("nan"), "auroc": float("nan")}
    return {
        "balanced_accuracy": _balanced_accuracy_at_threshold(y_true, score, threshold),
        "auroc": _auroc(y_true, score),
    }


def _auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score_series = pd.Series(np.asarray(score, dtype=float))
    ranks = score_series.rank(method="average").to_numpy(dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = float(ranks[y_true == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _fit_ridge_discriminant(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    xz = (x_train - center) / scale
    design = np.column_stack([np.ones(len(xz)), xz])
    target = y_train.astype(float) * 2.0 - 1.0
    penalty = np.eye(design.shape[1]) * 1e-3
    penalty[0, 0] = 0.0
    weights = np.linalg.pinv(design.T @ design + penalty) @ design.T @ target
    return center, scale, weights


def _predict_ridge_discriminant(
    x_test: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    xz = (x_test - center) / scale
    design = np.column_stack([np.ones(len(xz)), xz])
    return design @ weights


def _plain_stratified_splits(y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    fold_bins: list[list[int]] = [[] for _ in range(n_splits)]
    for label in sorted(np.unique(y)):
        indices = np.flatnonzero(y == label)
        shuffled = indices[rng.permutation(len(indices))]
        for position, index in enumerate(shuffled):
            fold_bins[position % n_splits].append(int(index))

    all_indices = np.arange(len(y))
    splits = []
    for fold in fold_bins:
        test_idx = np.array(sorted(fold), dtype=int)
        train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=False)
        splits.append((train_idx, test_idx))
    return splits


def _group_stratified_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    group_values = np.array(sorted(pd.unique(groups).astype(str)))
    group_stats = []
    for group in group_values:
        idx = np.flatnonzero(groups == group)
        group_stats.append(
            {
                "group": group,
                "indices": idx,
                "size": len(idx),
                "positive": int(y[idx].sum()),
                "negative": int(len(idx) - y[idx].sum()),
                "jitter": float(rng.random()),
            }
        )
    group_stats.sort(key=lambda row: (-int(row["size"]), -abs(int(row["positive"]) - int(row["negative"])), row["jitter"]))

    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    fold_counts = np.zeros((n_splits, 2), dtype=float)
    target = np.array([(y == 0).sum(), (y == 1).sum()], dtype=float) / n_splits
    for row in group_stats:
        group_count = np.array([row["negative"], row["positive"]], dtype=float)
        costs = []
        for fold in range(n_splits):
            projected = fold_counts[fold] + group_count
            costs.append(float(np.square(projected - target).sum() + 0.01 * projected.sum()))
        fold = int(np.argmin(costs))
        fold_groups[fold].append(str(row["group"]))
        fold_counts[fold] += group_count

    all_indices = np.arange(len(y))
    splits = []
    for fold in fold_groups:
        test_mask = np.isin(groups, fold)
        test_idx = all_indices[test_mask]
        train_idx = all_indices[~test_mask]
        if len(np.unique(y[test_idx])) < 2 or len(np.unique(y[train_idx])) < 2:
            return _plain_stratified_splits(y, n_splits, seed)
        splits.append((train_idx, test_idx))
    return splits


def _fold_splits(
    features: DualChannelFeatureSet,
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    class_counts = np.bincount(y, minlength=2)
    effective_splits = min(n_splits, int(class_counts.min()))
    if effective_splits < 2:
        raise ValueError("each label class must have at least two samples")

    if features.groups is not None:
        groups = features.groups.reindex(features.feature_matrix.index).astype(str).to_numpy()
        if len(np.unique(groups)) >= effective_splits:
            return _group_stratified_splits(y, groups, effective_splits, seed)

    return _plain_stratified_splits(y, effective_splits, seed)


def evaluate_official_vs_dual_channel(
    features: DualChannelFeatureSet,
    n_splits: int = 5,
    seed: int = 20260804,
) -> pd.DataFrame:
    """Compare official GMWI2 and fitted dual-channel scores on identical held-out folds."""

    y = _binary_labels(features.labels)
    x = features.feature_matrix.to_numpy(dtype=float)
    official = features.official_score.reindex(features.feature_matrix.index).to_numpy(dtype=float)
    rows: list[dict[str, float | int | str]] = []

    for fold, (train_idx, test_idx) in enumerate(_fold_splits(features, y, n_splits, seed)):
        center, scale, weights = _fit_ridge_discriminant(x[train_idx], y[train_idx])
        dual_train_score = _predict_ridge_discriminant(x[train_idx], center, scale, weights)
        dual_score = _predict_ridge_discriminant(x[test_idx], center, scale, weights)
        model_scores = {
            "official_gmwi2": (official[train_idx], official[test_idx]),
            "gmwi2_dual_channel": (dual_train_score, dual_score),
        }
        for model_name, (train_score, test_score) in model_scores.items():
            threshold = _best_threshold(y[train_idx], np.asarray(train_score, dtype=float))
            metrics = _metrics(y[test_idx], np.asarray(test_score, dtype=float), threshold)
            rows.append(
                {
                    "fold": int(fold),
                    "model": model_name,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "auroc": metrics["auroc"],
                    "threshold": threshold,
                    "n_test": int(len(test_idx)),
                }
            )
    return pd.DataFrame(rows)


def _paired_pvalue(delta: pd.Series) -> float:
    observed = pd.to_numeric(delta, errors="coerce").dropna()
    if observed.empty:
        return float("nan")
    values = observed.to_numpy(dtype=float)
    if np.allclose(values, 0.0):
        return 1.0
    values = values[~np.isclose(values, 0.0)]
    if len(values) == 0:
        return 1.0
    ranks = pd.Series(np.abs(values)).rank(method="average").to_numpy(dtype=float)
    observed_rank_sum = float(ranks[values > 0].sum())
    rank_sums = np.array([0.0])
    for rank in ranks:
        rank_sums = np.concatenate([rank_sums, rank_sums + rank])
    return float((rank_sums >= observed_rank_sum - 1e-12).mean())


def summarize_dual_channel_superiority(fold_metrics: pd.DataFrame) -> dict[str, float | bool]:
    """Summarize paired fold-level official-versus-dual-channel superiority."""

    required = {"fold", "model", "balanced_accuracy", "auroc"}
    missing = required.difference(fold_metrics.columns)
    if missing:
        raise ValueError(f"fold_metrics missing required columns: {sorted(missing)}")

    pivot_ba = fold_metrics.pivot(index="fold", columns="model", values="balanced_accuracy")
    pivot_auc = fold_metrics.pivot(index="fold", columns="model", values="auroc")
    for model_name in ("official_gmwi2", "gmwi2_dual_channel"):
        if model_name not in pivot_ba.columns or model_name not in pivot_auc.columns:
            raise ValueError(f"fold_metrics must include model {model_name!r}")

    delta_ba = pivot_ba["gmwi2_dual_channel"] - pivot_ba["official_gmwi2"]
    delta_auc = pivot_auc["gmwi2_dual_channel"] - pivot_auc["official_gmwi2"]
    p_ba = _paired_pvalue(delta_ba)
    p_auc = _paired_pvalue(delta_auc)

    ba_mean = float(delta_ba.mean())
    auc_mean = float(delta_auc.mean())
    ba_noninferiority_margin = -0.01
    return {
        "official_balanced_accuracy_mean": float(pivot_ba["official_gmwi2"].mean()),
        "dual_channel_balanced_accuracy_mean": float(pivot_ba["gmwi2_dual_channel"].mean()),
        "delta_balanced_accuracy_mean": ba_mean,
        "official_auroc_mean": float(pivot_auc["official_gmwi2"].mean()),
        "dual_channel_auroc_mean": float(pivot_auc["gmwi2_dual_channel"].mean()),
        "delta_auroc_mean": auc_mean,
        "paired_balanced_accuracy_p": p_ba,
        "paired_auroc_p": p_auc,
        "balanced_accuracy_noninferiority_margin": ba_noninferiority_margin,
        "balanced_accuracy_noninferiority_passed": bool(ba_mean >= ba_noninferiority_margin),
        "dual_channel_superiority_passed": bool(
            ba_mean >= ba_noninferiority_margin and auc_mean > 0.0 and p_auc < 0.05
        ),
    }


def clinical_correlation_table(
    scores: pd.DataFrame,
    clinical: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """Compute score-by-clinical-variable Spearman correlations after sample alignment."""

    rows: list[dict[str, float | str | int]] = []
    score_frame = _sample_index(scores).apply(pd.to_numeric, errors="coerce")
    clinical_frame = _sample_index(clinical)
    for score_name in score_frame.columns:
        for variable in variables:
            if variable not in clinical_frame.columns:
                rows.append(
                    {
                        "score": str(score_name),
                        "variable": str(variable),
                        "spearman_rho": float("nan"),
                        "n": 0,
                    }
                )
                continue
            pair = pd.concat(
                [score_frame[score_name], pd.to_numeric(clinical_frame[variable], errors="coerce")],
                axis=1,
                join="inner",
            ).dropna()
            rho = (
                pair.iloc[:, 0].rank(method="average").corr(pair.iloc[:, 1].rank(method="average"))
                if len(pair) >= 3
                else float("nan")
            )
            rows.append(
                {
                    "score": str(score_name),
                    "variable": str(variable),
                    "spearman_rho": float(rho),
                    "n": int(len(pair)),
                }
            )
    return pd.DataFrame(rows)


def compare_clinical_retention(
    official: pd.DataFrame,
    dual: pd.DataFrame,
    clinical: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """Check whether dual-channel clinical correlations retain official GMWI2 signal."""

    official_table = clinical_correlation_table(official, clinical, variables).rename(
        columns={"score": "official_score", "spearman_rho": "official_spearman", "n": "official_n"}
    )
    dual_table = clinical_correlation_table(dual, clinical, variables).rename(
        columns={"score": "dual_score", "spearman_rho": "dual_spearman", "n": "dual_n"}
    )
    official_table = official_table.loc[:, ["variable", "official_score", "official_spearman", "official_n"]]
    dual_table = dual_table.loc[:, ["variable", "dual_score", "dual_spearman", "dual_n"]]
    merged = official_table.merge(dual_table, on="variable", how="inner")
    merged["abs_delta_spearman"] = (
        merged["dual_spearman"].abs() - merged["official_spearman"].abs()
    ).abs()
    merged["retains_gmwi2_direction"] = np.sign(merged["dual_spearman"].fillna(0.0)).eq(
        np.sign(merged["official_spearman"].fillna(0.0))
    )
    return merged
