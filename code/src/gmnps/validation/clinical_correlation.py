"""Clinical-metadata completeness and health-index consistency checks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClinicalVariableSpec:
    """Default clinical variables and non-inferiority thresholds."""

    variables: tuple[str, ...] = ("age", "bmi", "fbg", "cholesterol", "hdl", "ldl", "tg")
    max_missing_fraction: float = 0.40
    min_n: int = 50
    min_retained_fraction: float = 0.80
    max_abs_loss: float = 0.05


def audit_clinical_completeness(
    metadata: pd.DataFrame,
    variables: list[str],
    max_missing_fraction: float,
    min_n: int,
) -> pd.DataFrame:
    """Report variable-level nonmissing counts and cohort usability flags."""

    rows: list[dict[str, object]] = []
    n_total = int(len(metadata))
    for variable in variables:
        if variable not in metadata.columns:
            rows.append(
                {
                    "variable": variable,
                    "n_total": n_total,
                    "n_nonmissing": 0,
                    "missing_fraction": 1.0,
                    "passes": False,
                }
            )
            continue

        nonmissing = int(pd.to_numeric(metadata[variable], errors="coerce").notna().sum())
        missing_fraction = 1.0 - nonmissing / max(n_total, 1)
        rows.append(
            {
                "variable": variable,
                "n_total": n_total,
                "n_nonmissing": nonmissing,
                "missing_fraction": float(missing_fraction),
                "passes": bool(nonmissing >= min_n and missing_fraction <= max_missing_fraction),
            }
        )
    return pd.DataFrame(rows)


def choose_clinical_cohort(gmrepo_audit: pd.DataFrame, cra_audit: pd.DataFrame) -> str:
    """Prefer GMrepo unless CRA013939 has better passing clinical coverage."""

    for name, audit in {"gmrepo_audit": gmrepo_audit, "cra_audit": cra_audit}.items():
        if "passes" not in audit.columns:
            raise ValueError(f"{name} must contain passes")
    gmrepo_pass = int(gmrepo_audit["passes"].sum())
    cra_pass = int(cra_audit["passes"].sum())
    if gmrepo_pass >= cra_pass and gmrepo_pass > 0:
        return "GMrepo"
    if cra_pass > 0:
        return "CRA013939"
    return "insufficient_clinical_metadata"


def _spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].rank(method="average").corr(pair.iloc[:, 1].rank(method="average")))


def _permutation_pvalue(x: pd.Series, y: pd.Series, n_perm: int = 999, seed: int = 20260804) -> float:
    observed = abs(_spearman(x, y))
    if np.isnan(observed):
        return float("nan")
    x_rank = pd.Series(pd.to_numeric(x, errors="coerce")).rank(method="average")
    y_values = pd.Series(pd.to_numeric(y, errors="coerce")).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        permuted = pd.Series(rng.permutation(y_values)).rank(method="average")
        corr = float(x_rank.corr(permuted))
        count += bool(abs(corr) >= observed)
    return float((count + 1) / (n_perm + 1))


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p values."""

    p = p_values.astype(float).fillna(1.0).to_numpy()
    n = len(p)
    if n == 0:
        return pd.Series(dtype=float, index=p_values.index)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for rank in range(n, 0, -1):
        idx = order[rank - 1]
        running = min(running, p[idx] * n / rank)
        adjusted[idx] = running
    return pd.Series(adjusted, index=p_values.index)


def spearman_clinical_correlations(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    variables: list[str],
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compute feature-by-clinical Spearman correlations after sample alignment."""

    if "sample_id" not in features.columns or "sample_id" not in metadata.columns:
        raise ValueError("features and metadata must contain sample_id")
    merged = features.merge(metadata, on="sample_id", how="inner")
    rows: list[dict[str, object]] = []
    for feature in feature_cols:
        for variable in variables:
            if feature not in merged.columns or variable not in merged.columns:
                continue
            pair = merged[[feature, variable]].apply(pd.to_numeric, errors="coerce").dropna()
            rho = _spearman(pair[feature], pair[variable])
            p_value = _permutation_pvalue(pair[feature], pair[variable]) if len(pair) >= 3 else float("nan")
            rows.append(
                {
                    "feature": feature,
                    "clinical_variable": variable,
                    "n": int(len(pair)),
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_value"] = _bh_fdr(result["p_value"])
    return result


def evaluate_gmwi2_retention(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    min_retained_fraction: float,
    max_abs_loss: float,
) -> pd.DataFrame:
    """Evaluate whether candidate correlations retain baseline GMWI2 direction and magnitude."""

    required = {"feature", "clinical_variable", "spearman_rho"}
    for name, frame in {"baseline": baseline, "candidate": candidate}.items():
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")

    base = baseline.rename(columns={"feature": "baseline_feature", "spearman_rho": "baseline_rho"})
    cand = candidate.rename(columns={"feature": "candidate_feature", "spearman_rho": "candidate_rho"})
    merged = base[["clinical_variable", "baseline_feature", "baseline_rho"]].merge(
        cand[["clinical_variable", "candidate_feature", "candidate_rho"]],
        on="clinical_variable",
        how="inner",
    )
    eps = 1e-12
    baseline_abs = merged["baseline_rho"].abs()
    candidate_abs = merged["candidate_rho"].abs()
    merged["retained_fraction"] = candidate_abs / baseline_abs.clip(lower=eps)
    merged["absolute_loss"] = baseline_abs - candidate_abs
    merged["same_direction"] = np.sign(merged["baseline_rho"]) == np.sign(merged["candidate_rho"])
    merged["passes_retention"] = merged["same_direction"] & (
        (merged["retained_fraction"] >= min_retained_fraction)
        | (merged["absolute_loss"] <= max_abs_loss)
    )
    return merged
