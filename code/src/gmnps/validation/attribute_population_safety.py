"""Population-safety design audits for locked attribute-level GMNPS scores.

The outputs in this module are diagnostics of preservation and bounded score
movement.  They are not person-food outcome validation and they never recenter
scores to manufacture agreement with Food Compass 2.0 (FCS2).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Mapping

import numpy as np
import pandas as pd


POPULATION_EVIDENCE_ROLE = "population_safety_design_audit"
POPULATION_SCHEMA_VERSION = "gmnps-attribute-validation-input-v1"
_CATEGORY_ORDER = ("minimize", "moderate", "encourage")
_CATEGORY_INDEX = {value: index for index, value in enumerate(_CATEGORY_ORDER)}
_REQUIRED_COLUMNS = {
    "individual_id",
    "food_id",
    "food_group",
    "food_subgroup",
    "FCS2",
    "GMNPS_delta",
    "GMNPS_score",
}
_FORBIDDEN_POPULATION_COLUMNS = frozenset(
    {
        "disease",
        "disease_label",
        "phenotype_label",
        "outcome",
        "y_true",
        "glucose_iauc_2h",
        "tg_6h_rise",
        "c_peptide_iauc_2h",
        "postprandial_response",
    }
)


@dataclass(frozen=True)
class PopulationSafetyConfig:
    """Pre-specified statistical configuration for the design audit."""

    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 20260813
    confidence_level: float = 0.95
    minimum_valid_replicates: int = 800

    def __post_init__(self) -> None:
        if isinstance(self.bootstrap_replicates, bool) or self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be a positive integer")
        if isinstance(self.minimum_valid_replicates, bool) or not (
            1 <= self.minimum_valid_replicates <= self.bootstrap_replicates
        ):
            raise ValueError(
                "minimum_valid_replicates must be between 1 and bootstrap_replicates"
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if isinstance(self.bootstrap_seed, bool) or not isinstance(
            self.bootstrap_seed, int
        ):
            raise ValueError("bootstrap_seed must be an integer")


@dataclass(frozen=True)
class PopulationSafetyAudit:
    """Complete collection of population-safety source-data tables."""

    evidence_role: str
    global_summary: dict[str, object]
    group_convergence: pd.DataFrame
    subgroup_convergence: pd.DataFrame
    between_group_discrimination: pd.DataFrame
    transition_matrix: pd.DataFrame
    reversal_audit: pd.DataFrame
    bootstrap_summary: pd.DataFrame


def _require_nonempty_string(manifest: Mapping[str, object], field: str) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"provenance {field} must be a nonempty string")
    return value.strip()


def validate_population_provenance(
    manifest: Mapping[str, object], *, allow_test_data: bool = False
) -> None:
    """Fail closed unless provenance identifies the locked uncentered method.

    Test fixtures require an explicit caller opt-in and remain ineligible for
    result writing.  Production evidence additionally requires the declared
    9,234-food bundle.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("provenance manifest must be an object")
    expected = {
        "schema_version": POPULATION_SCHEMA_VERSION,
        "analysis_kind": "population_safety",
        "evidence_role": POPULATION_EVIDENCE_ROLE,
        "method": "attribute_recomposition",
        "method_role": "primary",
        "score_centering": "none",
    }
    for field, required in expected.items():
        observed = manifest.get(field)
        if observed != required:
            if field == "method":
                raise ValueError(
                    "provenance method must be attribute_recomposition; the legacy "
                    "final-score offset is a comparator only"
                )
            raise ValueError(f"provenance {field} must equal {required!r}")
    if manifest.get("scoring_frozen_before_labels") is not True:
        raise ValueError("scoring must be frozen before any labels are inspected")

    independent_unit = _require_nonempty_string(manifest, "independent_unit")
    if independent_unit not in {
        "individual_id",
        "component_id",
        "cohort_component_id",
    }:
        raise ValueError("provenance independent_unit is not an allowed cluster unit")

    synthetic = manifest.get("synthetic") is True
    testing_only = manifest.get("testing_only") is True
    if synthetic or testing_only:
        if not allow_test_data:
            raise ValueError("testing/synthetic provenance is rejected by default")
        if manifest.get("data_class") != "synthetic_test_fixture":
            raise ValueError("testing provenance data_class must be synthetic_test_fixture")
        if manifest.get("production_label") != "non-production":
            raise ValueError("testing provenance must be non-production")
    else:
        if manifest.get("data_class") != "locked_attribute_level_gmnps":
            raise ValueError("production data_class must be locked_attribute_level_gmnps")
        if manifest.get("production_label") != "production":
            raise ValueError("real population-safety input must be production")
        if manifest.get("n_foods") != 9234:
            raise ValueError("production population-safety input must declare 9,234 foods")


def _normalized_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().lower(): str(column) for column in frame.columns}


def _check_invariant(frame: pd.DataFrame, unit: str, column: str) -> None:
    counts = frame.groupby(unit, dropna=False)[column].nunique(dropna=False)
    if (counts > 1).any():
        bad = str(counts[counts > 1].index[0])
        raise ValueError(f"{column} is inconsistent within {unit}={bad}")


def _validate_population_frame(
    frame: pd.DataFrame, manifest: Mapping[str, object]
) -> tuple[pd.DataFrame, str]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("individual_food must be a nonempty pandas DataFrame")
    missing = _REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"individual_food missing required columns: {sorted(missing)}")
    normalized = _normalized_columns(frame)
    leaked = sorted(_FORBIDDEN_POPULATION_COLUMNS.intersection(normalized))
    if leaked:
        raise ValueError(f"population input contains outcome/label leakage columns: {leaked}")
    if frame.duplicated(["individual_id", "food_id"]).any():
        raise ValueError("individual_food must contain one row per individual_id-food_id")

    working = frame.copy()
    for column in ("individual_id", "food_id", "food_group", "food_subgroup"):
        if working[column].isna().any() or working[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"individual_food {column} contains missing or empty values")
        working[column] = working[column].astype(str).str.strip()
    for column in ("FCS2", "GMNPS_delta", "GMNPS_score"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    if working["FCS2"].isna().any() or not np.isfinite(working["FCS2"]).all():
        raise ValueError("FCS2 must contain finite values")
    if not working["FCS2"].between(0.0, 100.0).all():
        raise ValueError("FCS2 must lie between 0 and 100")
    finite_score = working["GMNPS_score"].notna() & np.isfinite(working["GMNPS_score"])
    if not working.loc[finite_score, "GMNPS_score"].between(0.0, 100.0).all():
        raise ValueError("finite GMNPS_score values must lie between 0 and 100")
    complete = finite_score & working["GMNPS_delta"].notna()
    observed_delta = working.loc[complete, "GMNPS_score"] - working.loc[complete, "FCS2"]
    if not np.allclose(
        working.loc[complete, "GMNPS_delta"], observed_delta, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            "GMNPS_delta must equal GMNPS_score - FCS2; post-hoc centering is forbidden"
        )
    if "method_role" in working and not working["method_role"].eq("primary").all():
        raise ValueError("individual_food method_role must be primary")

    for column in ("FCS2", "food_group", "food_subgroup"):
        _check_invariant(working, "food_id", column)
    _check_invariant(working, "individual_id", "individual_id")

    cluster_column = str(manifest["independent_unit"])
    if cluster_column != "individual_id":
        if cluster_column not in working.columns:
            raise ValueError(
                f"declared independent_unit column is missing: {cluster_column}"
            )
        if working[cluster_column].isna().any():
            raise ValueError(f"{cluster_column} contains missing values")
        working[cluster_column] = working[cluster_column].astype(str).str.strip()
        _check_invariant(working, "individual_id", cluster_column)

    observed_foods = int(working["food_id"].nunique())
    if manifest.get("n_foods") != observed_foods:
        raise ValueError(
            f"provenance n_foods={manifest.get('n_foods')!r} does not match "
            f"observed n_foods={observed_foods}"
        )
    return working, cluster_column


def _fcs_category(value: float) -> str:
    if value >= 70.0:
        return "encourage"
    if value >= 31.0:
        return "moderate"
    return "minimize"


def _spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")],
        axis=1,
    ).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(
        pair.iloc[:, 0]
        .rank(method="average")
        .corr(pair.iloc[:, 1].rank(method="average"))
    )


def _food_level(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.loc[np.isfinite(frame["GMNPS_score"])].copy()
    return (
        valid.groupby("food_id", sort=True)
        .agg(
            FCS2=("FCS2", "first"),
            GMNPS_population_mean=("GMNPS_score", "mean"),
            food_group=("food_group", "first"),
            food_subgroup=("food_subgroup", "first"),
            n_individuals=("individual_id", "nunique"),
        )
        .reset_index()
    )


def _iqr(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(finite):
        return float("nan")
    return float(np.percentile(finite, 75) - np.percentile(finite, 25))


def _convergence_table(
    food: pd.DataFrame, raw: pd.DataFrame, column: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stratum, group in food.groupby(column, sort=True):
        baseline = group["FCS2"].to_numpy(dtype=float)
        personalized = group["GMNPS_population_mean"].to_numpy(dtype=float)
        raw_group = raw[raw[column].astype(str) == str(stratum)]
        rows.append(
            {
                "stratum_level": column,
                "stratum": str(stratum),
                "n_foods": int(group["food_id"].nunique()),
                "n_individuals": int(raw_group["individual_id"].nunique()),
                "fcs2_median": float(np.median(baseline)),
                "gmnps_population_median": float(np.median(personalized)),
                "median_shift": float(np.median(personalized - baseline)),
                "mean_absolute_food_shift": float(np.mean(np.abs(personalized - baseline))),
                "fcs2_iqr": _iqr(group["FCS2"]),
                "gmnps_population_iqr": _iqr(group["GMNPS_population_mean"]),
                "distribution_l1_quantile_distance": float(
                    np.mean(np.abs(np.sort(personalized) - np.sort(baseline)))
                ),
                "within_stratum_spearman": _spearman(
                    group["FCS2"], group["GMNPS_population_mean"]
                ),
                "evidence_role": POPULATION_EVIDENCE_ROLE,
            }
        )
    return pd.DataFrame(rows)


def _between_group_table(food: pd.DataFrame) -> pd.DataFrame:
    means = food.groupby("food_group", sort=True).agg(
        FCS2=("FCS2", "mean"),
        GMNPS_population_mean=("GMNPS_population_mean", "mean"),
        n_foods=("food_id", "nunique"),
    )
    rows: list[dict[str, object]] = []
    for left, right in combinations(means.index.astype(str), 2):
        baseline = float(means.loc[left, "FCS2"] - means.loc[right, "FCS2"])
        personalized = float(
            means.loc[left, "GMNPS_population_mean"]
            - means.loc[right, "GMNPS_population_mean"]
        )
        ratio = (
            abs(personalized) / abs(baseline)
            if not np.isclose(baseline, 0.0)
            else float("nan")
        )
        rows.append(
            {
                "stratum_a": left,
                "stratum_b": right,
                "n_foods_a": int(means.loc[left, "n_foods"]),
                "n_foods_b": int(means.loc[right, "n_foods"]),
                "fcs2_mean_difference": baseline,
                "gmnps_population_mean_difference": personalized,
                "direction_preserved": bool(
                    np.isclose(baseline, 0.0)
                    or np.isclose(personalized, 0.0)
                    or np.sign(baseline) == np.sign(personalized)
                ),
                "absolute_difference_ratio": float(ratio),
                "evidence_role": POPULATION_EVIDENCE_ROLE,
            }
        )
    return pd.DataFrame(rows)


def _transition_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = frame.loc[
        np.isfinite(frame["GMNPS_score"]) & np.isfinite(frame["GMNPS_delta"])
    ].copy()
    valid["fcs2_category"] = valid["FCS2"].map(_fcs_category)
    valid["gmnps_category"] = valid["GMNPS_score"].map(_fcs_category)
    rows: list[dict[str, object]] = []
    for origin, destination in product(_CATEGORY_ORDER, repeat=2):
        selected = valid[
            valid["fcs2_category"].eq(origin)
            & valid["gmnps_category"].eq(destination)
        ]
        denominator = int(valid["fcs2_category"].eq(origin).sum())
        movement = _CATEGORY_INDEX[destination] - _CATEGORY_INDEX[origin]
        rows.append(
            {
                "fcs2_category": origin,
                "gmnps_category": destination,
                "direction": "up" if movement > 0 else "down" if movement < 0 else "stable",
                "category_steps": int(abs(movement)),
                "n_individual_foods": int(len(selected)),
                "origin_fraction": float(len(selected) / denominator) if denominator else 0.0,
                "mean_delta": float(selected["GMNPS_delta"].mean()) if len(selected) else float("nan"),
                "median_absolute_delta": (
                    float(selected["GMNPS_delta"].abs().median())
                    if len(selected)
                    else float("nan")
                ),
                "evidence_role": POPULATION_EVIDENCE_ROLE,
            }
        )
    transitions = pd.DataFrame(rows)

    extreme = valid[
        (
            valid["fcs2_category"].eq("encourage")
            & valid["gmnps_category"].eq("minimize")
        )
        | (
            valid["fcs2_category"].eq("minimize")
            & valid["gmnps_category"].eq("encourage")
        )
    ].copy()
    reversal_columns = [
        "individual_id",
        "food_id",
        "food_name",
        "food_group",
        "food_subgroup",
        "FCS2",
        "GMNPS_score",
        "GMNPS_delta",
        "fcs2_category",
        "gmnps_category",
        "reversal_type",
        "absolute_delta",
        "audit_status",
        "evidence_role",
    ]
    if extreme.empty:
        return transitions, pd.DataFrame(columns=reversal_columns)
    if "food_name" not in extreme:
        extreme["food_name"] = extreme["food_id"]
    extreme["reversal_type"] = (
        extreme["fcs2_category"].astype(str)
        + "_to_"
        + extreme["gmnps_category"].astype(str)
    )
    extreme["absolute_delta"] = extreme["GMNPS_delta"].abs()
    extreme["audit_status"] = "flagged_for_clinical_nutritional_review"
    extreme["evidence_role"] = POPULATION_EVIDENCE_ROLE
    return transitions, extreme.loc[:, reversal_columns].sort_values(
        ["individual_id", "food_id"], kind="mergesort"
    ).reset_index(drop=True)


def _bootstrap_summary(
    frame: pd.DataFrame,
    cluster_column: str,
    config: PopulationSafetyConfig,
) -> pd.DataFrame:
    valid = frame.loc[np.isfinite(frame["GMNPS_score"])].copy()
    food = _food_level(valid)
    estimate = _spearman(food["FCS2"], food["GMNPS_population_mean"])
    clusters = sorted(valid[cluster_column].astype(str).unique())
    rng = np.random.default_rng(config.bootstrap_seed)
    values: list[float] = []
    for _ in range(config.bootstrap_replicates):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        chunks = [valid[valid[cluster_column].astype(str).eq(cluster)] for cluster in sampled]
        replicate = pd.concat(chunks, ignore_index=True)
        replicate_food = _food_level(replicate)
        value = _spearman(
            replicate_food["FCS2"], replicate_food["GMNPS_population_mean"]
        )
        if np.isfinite(value):
            values.append(value)
    if len(values) < config.minimum_valid_replicates:
        raise ValueError(
            "cluster bootstrap produced fewer valid replicates than the pre-specified minimum"
        )
    alpha = (1.0 - config.confidence_level) / 2.0
    lower, upper = np.quantile(np.asarray(values), [alpha, 1.0 - alpha])
    missing = int((~np.isfinite(frame["GMNPS_score"])).sum())
    return pd.DataFrame(
        [
            {
                "metric": "spearman_fcs2_vs_population_gmnps",
                "estimand": "food-level population mean",
                "estimate": float(estimate),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "confidence_level": float(config.confidence_level),
                "ci_method": "cluster_percentile",
                "cluster_column": cluster_column,
                "n_clusters": int(len(clusters)),
                "n_individuals": int(frame["individual_id"].nunique()),
                "replicates_requested": int(config.bootstrap_replicates),
                "replicates_valid": int(len(values)),
                "bootstrap_seed": int(config.bootstrap_seed),
                "missing_n_rows": missing,
                "excluded_n_rows": missing,
                "exclusion_rule": "nonfinite_GMNPS_score",
                "evidence_role": POPULATION_EVIDENCE_ROLE,
            }
        ]
    )


def audit_population_safety(
    individual_food: pd.DataFrame,
    provenance_manifest: Mapping[str, object],
    *,
    config: PopulationSafetyConfig | None = None,
    allow_test_data: bool = False,
) -> PopulationSafetyAudit:
    """Run the complete FCS2-anchored population-safety design audit."""

    validate_population_provenance(
        provenance_manifest, allow_test_data=allow_test_data
    )
    frame, cluster_column = _validate_population_frame(
        individual_food, provenance_manifest
    )
    resolved = config or PopulationSafetyConfig()
    food = _food_level(frame)
    transitions, reversals = _transition_tables(frame)
    global_summary: dict[str, object] = {
        "evidence_role": POPULATION_EVIDENCE_ROLE,
        "analysis_boundary": "design_audit_not_empirical_person_food_validation",
        "estimand": "food-level population mean",
        "score_centering": "none",
        "n_rows": int(len(frame)),
        "n_individuals": int(frame["individual_id"].nunique()),
        "n_foods": int(frame["food_id"].nunique()),
        "n_groups": int(frame["food_group"].nunique()),
        "n_subgroups": int(frame["food_subgroup"].nunique()),
        "missing_n_rows": int((~np.isfinite(frame["GMNPS_score"])).sum()),
        "excluded_n_rows": int((~np.isfinite(frame["GMNPS_score"])).sum()),
        "exclusion_rule": "nonfinite_GMNPS_score",
        "spearman_fcs2_vs_population_gmnps": _spearman(
            food["FCS2"], food["GMNPS_population_mean"]
        ),
        "mean_absolute_food_shift": float(
            (food["GMNPS_population_mean"] - food["FCS2"]).abs().mean()
        ),
        "implausible_reversal_count": int(len(reversals)),
        "implausible_reversal_fraction": float(
            len(reversals) / int(np.isfinite(frame["GMNPS_score"]).sum())
        ),
    }
    return PopulationSafetyAudit(
        evidence_role=POPULATION_EVIDENCE_ROLE,
        global_summary=global_summary,
        group_convergence=_convergence_table(food, frame, "food_group"),
        subgroup_convergence=_convergence_table(food, frame, "food_subgroup"),
        between_group_discrimination=_between_group_table(food),
        transition_matrix=transitions,
        reversal_audit=reversals,
        bootstrap_summary=_bootstrap_summary(frame, cluster_column, resolved),
    )


__all__ = [
    "POPULATION_EVIDENCE_ROLE",
    "PopulationSafetyAudit",
    "PopulationSafetyConfig",
    "audit_population_safety",
    "validate_population_provenance",
]
