"""Explicit official and proxy microbiome-health validation utilities."""
from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd


class MicrobiomeHealthMode(str, Enum):
    """Supported microbiome-health score sources."""

    OFFICIAL_GMWI2 = "official_gmwi2"
    GENUS_PROXY = "genus_proxy"


_SAMPLE_ID_COLUMNS = ("sample_id", "subject_id")
_OFFICIAL_SCORE_COLUMNS = ("official_gmwi2_score", "gmwi2_score", "gmwi2", "score")
_HEALTH_ASSOCIATED_GENERA = ("Akkermansia", "Bifidobacterium", "Faecalibacterium", "Roseburia")
_DISEASE_ASSOCIATED_GENERA = ("Escherichia", "Klebsiella", "Enterococcus", "Bilophila")


def _sample_index(abundance: pd.DataFrame) -> pd.DataFrame:
    """Return an abundance matrix indexed by a supported sample identifier."""

    frame = abundance.copy()
    sample_column = next((column for column in _SAMPLE_ID_COLUMNS if column in frame.columns), None)
    if sample_column is not None:
        frame = frame.set_index(sample_column, drop=True)
    frame.index = frame.index.astype(str)
    if frame.index.has_duplicates:
        raise ValueError("abundance sample identifiers must be unique")
    return frame


def load_official_gmwi2_scores(path: Path) -> pd.DataFrame:
    """Load externally generated official GMWI2 scores without reimplementing GMWI2.

    The input must provide ``sample_id`` and one recognized GMWI2 score column.
    The returned score name makes the provenance explicit for downstream joins.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        scores = pd.read_parquet(path)
    elif path.suffix.lower() in {".tsv", ".tab"}:
        scores = pd.read_csv(path, sep="\t")
    else:
        scores = pd.read_csv(path)

    if "sample_id" not in scores.columns:
        raise ValueError("official GMWI2 score file must contain sample_id")
    score_column = next((column for column in _OFFICIAL_SCORE_COLUMNS if column in scores.columns), None)
    if score_column is None:
        raise ValueError(
            "official GMWI2 score file must contain one of: " + ", ".join(_OFFICIAL_SCORE_COLUMNS)
        )

    result = scores.loc[:, ["sample_id", score_column]].copy()
    result["sample_id"] = result["sample_id"].astype(str)
    result["official_gmwi2_score"] = pd.to_numeric(result.pop(score_column), errors="coerce")
    result = result.dropna(subset=["official_gmwi2_score"])
    if result["sample_id"].duplicated().any():
        raise ValueError("official GMWI2 score file contains duplicate sample_id values")
    result.attrs["mode"] = MicrobiomeHealthMode.OFFICIAL_GMWI2.value
    return result


def genus_proxy_health_score(abundance: pd.DataFrame) -> pd.Series:
    """Calculate a deterministic genus-level health proxy, not an official GMWI2 score."""

    frame = _sample_index(abundance)
    numeric = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    positive = [genus for genus in _HEALTH_ASSOCIATED_GENERA if genus in numeric.columns]
    negative = [genus for genus in _DISEASE_ASSOCIATED_GENERA if genus in numeric.columns]
    pos = np.log1p(numeric[positive]).mean(axis=1) if positive else pd.Series(0.0, index=numeric.index)
    neg = np.log1p(numeric[negative]).mean(axis=1) if negative else pd.Series(0.0, index=numeric.index)
    raw_score = pos - neg
    std = raw_score.std(ddof=0)
    score = raw_score * 0.0 if not np.isfinite(std) or std <= 1e-12 else (raw_score - raw_score.mean()) / std
    score = score.rename("genus_proxy_health_score")
    score.attrs["mode"] = "genus_proxy_not_official_gmwi2"
    return score


def health_nonhealth_metrics(scores: pd.Series, labels: pd.Series) -> dict[str, float]:
    """Report threshold-optimized health-versus-non-health classification metrics."""

    paired = pd.concat(
        [pd.to_numeric(scores, errors="coerce").rename("score"), labels.rename("label")], axis=1, join="inner"
    ).dropna()
    is_health = paired["label"].astype(str).str.strip().str.lower().isin({"health", "healthy"})
    n_samples = len(paired)
    n_health = int(is_health.sum())
    n_nonhealth = n_samples - n_health
    if n_health == 0 or n_nonhealth == 0:
        return {
            "balanced_accuracy": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "threshold": float("nan"),
            "n_samples": float(n_samples),
        }

    values = paired["score"].to_numpy(dtype=float)
    thresholds = np.r_[np.nextafter(values.min(), -np.inf), np.unique(values), np.nextafter(values.max(), np.inf)]
    best: tuple[float, float, float, float] | None = None
    for threshold in thresholds:
        predicted_health = values >= threshold
        sensitivity = float(predicted_health[is_health.to_numpy()].mean())
        specificity = float((~predicted_health[~is_health.to_numpy()]).mean())
        candidate = ((sensitivity + specificity) / 2.0, sensitivity, specificity, float(threshold))
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None
    return {
        "balanced_accuracy": best[0],
        "sensitivity": best[1],
        "specificity": best[2],
        "threshold": best[3],
        "n_samples": float(n_samples),
    }
