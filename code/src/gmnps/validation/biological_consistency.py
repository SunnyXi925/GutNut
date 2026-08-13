"""Supporting biological and mechanistic consistency analyses.

GMrepo disease labels and ZOE aggregate ranks are biological-consistency
evidence.  Knowledge paths are mechanistic-consistency evidence.  None of these
inputs validates person-food responses, clinical utility, or causal effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

import numpy as np
import pandas as pd


SUPPORTING_SCHEMA_VERSION = "gmnps-supporting-evidence-v1"
BIOLOGICAL_CONSISTENCY = "biological_consistency"
MECHANISTIC_CONSISTENCY = "mechanistic_consistency"
_ROLE_BY_SOURCE = {
    "gmrepo_disease_labels": BIOLOGICAL_CONSISTENCY,
    "zoe_aggregate_ranks": BIOLOGICAL_CONSISTENCY,
    "knowledge_paths": MECHANISTIC_CONSISTENCY,
}
_OUTCOME_COLUMNS = frozenset(
    {
        "outcome",
        "y_true",
        "glucose_iauc_2h",
        "tg_6h_rise",
        "c_peptide_iauc_2h",
        "postprandial_response",
    }
)


@dataclass(frozen=True)
class BiologicalConsistencyConfig:
    """Pre-specified clustered inference and negative-control configuration."""

    bootstrap_replicates: int = 1000
    shuffle_replicates: int = 1000
    seed: int = 20260813
    confidence_level: float = 0.95
    minimum_valid_replicates: int = 800

    def __post_init__(self) -> None:
        for name in ("bootstrap_replicates", "shuffle_replicates"):
            value = getattr(self, name)
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 1 <= self.minimum_valid_replicates <= self.bootstrap_replicates:
            raise ValueError(
                "minimum_valid_replicates must be between 1 and bootstrap_replicates"
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class DiseaseConsistencyResult:
    """Cohort-stratified estimates, cluster CIs, and shuffled controls."""

    evidence_role: str
    estimates: pd.DataFrame
    inference: pd.DataFrame
    shuffle_controls: pd.DataFrame


def validate_evidence_provenance(
    manifest: Mapping[str, object], *, allow_test_data: bool = False
) -> None:
    """Validate source-specific evidence roles before any label table is read."""

    if not isinstance(manifest, Mapping):
        raise ValueError("provenance manifest must be an object")
    if manifest.get("schema_version") != SUPPORTING_SCHEMA_VERSION:
        raise ValueError(
            f"provenance schema_version must equal {SUPPORTING_SCHEMA_VERSION!r}"
        )
    source_kind = manifest.get("source_kind")
    if source_kind not in _ROLE_BY_SOURCE:
        raise ValueError("provenance source_kind is not supported")
    expected_role = _ROLE_BY_SOURCE[str(source_kind)]
    if manifest.get("evidence_role") != expected_role:
        raise ValueError(
            f"provenance evidence_role for {source_kind} must equal {expected_role!r}"
        )
    if manifest.get("source_verified") is not True:
        raise ValueError("supporting evidence source must be verified")
    if manifest.get("scoring_frozen_before_labels") is not True:
        raise ValueError("scoring must be frozen before supporting labels are inspected")
    if source_kind == "gmrepo_disease_labels" and manifest.get(
        "label_access_authorized"
    ) is not True:
        raise ValueError("GMrepo label access must be explicitly authorized")
    outcome_columns = manifest.get("outcome_columns")
    if outcome_columns != []:
        raise ValueError("supporting evidence outcome_columns must be an empty list")

    synthetic = manifest.get("synthetic") is True
    testing_only = manifest.get("testing_only") is True
    if synthetic or testing_only:
        if not allow_test_data:
            raise ValueError("testing/synthetic supporting evidence is rejected by default")
        if manifest.get("data_class") != "synthetic_test_fixture":
            raise ValueError("testing data_class must be synthetic_test_fixture")
        if manifest.get("production_label") != "non-production":
            raise ValueError("testing supporting evidence must be non-production")
    else:
        if manifest.get("production_label") != "production":
            raise ValueError("real supporting evidence must be production")
        if manifest.get("data_class") not in {
            "observed_supporting_labels",
            "published_aggregate_ranks",
            "curated_knowledge_paths",
        }:
            raise ValueError("real supporting evidence has an invalid data_class")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{label} must be a nonempty pandas DataFrame")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")


def _reject_outcomes(frame: pd.DataFrame) -> None:
    normalized = {str(column).strip().lower() for column in frame.columns}
    leaked = sorted(_OUTCOME_COLUMNS.intersection(normalized))
    if leaked:
        raise ValueError(f"supporting input contains outcome/label leakage columns: {leaked}")


def _check_person_invariant(frame: pd.DataFrame, column: str) -> None:
    counts = frame.groupby("individual_id", dropna=False)[column].nunique(dropna=False)
    if (counts > 1).any():
        bad = str(counts[counts > 1].index[0])
        raise ValueError(f"inconsistent {column} for individual_id={bad}")


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big") % (2**32)


def _difference(units: pd.DataFrame, disease: str, reference: str) -> float:
    disease_values = units.loc[units["disease_label"].eq(disease), "value"]
    reference_values = units.loc[units["disease_label"].eq(reference), "value"]
    if disease_values.empty or reference_values.empty:
        return float("nan")
    return float(disease_values.mean() - reference_values.mean())


def _bootstrap_difference(
    units: pd.DataFrame,
    disease: str,
    reference: str,
    config: BiologicalConsistencyConfig,
) -> tuple[list[float], float, float]:
    rng = np.random.default_rng(config.seed)
    disease_units = units[units["disease_label"].eq(disease)]
    reference_units = units[units["disease_label"].eq(reference)]
    values: list[float] = []
    for _ in range(config.bootstrap_replicates):
        disease_sample = disease_units.iloc[
            rng.integers(0, len(disease_units), size=len(disease_units))
        ]
        reference_sample = reference_units.iloc[
            rng.integers(0, len(reference_units), size=len(reference_units))
        ]
        value = float(disease_sample["value"].mean() - reference_sample["value"].mean())
        if np.isfinite(value):
            values.append(value)
    if len(values) < config.minimum_valid_replicates:
        raise ValueError(
            "cluster bootstrap produced fewer valid replicates than the pre-specified minimum"
        )
    alpha = (1.0 - config.confidence_level) / 2.0
    lower, upper = np.quantile(np.asarray(values), [alpha, 1.0 - alpha])
    return values, float(lower), float(upper)


def _shuffle_control(
    units: pd.DataFrame,
    disease: str,
    reference: str,
    config: BiologicalConsistencyConfig,
    *,
    control: str,
    cohort: str,
) -> dict[str, object]:
    seed = _stable_seed(config.seed, cohort, disease, control)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(config.shuffle_replicates):
        shuffled = units.copy()
        if control == "shuffled_label":
            shuffled["disease_label"] = rng.permutation(
                shuffled["disease_label"].to_numpy()
            )
        elif control == "shuffled_microbiome":
            shuffled["value"] = rng.permutation(shuffled["value"].to_numpy())
        else:  # pragma: no cover - internal guard
            raise ValueError(f"unknown control: {control}")
        value = _difference(shuffled, disease, reference)
        if np.isfinite(value):
            values.append(value)
    if not values:
        lower = upper = null_mean = float("nan")
    else:
        alpha = (1.0 - config.confidence_level) / 2.0
        lower, upper = np.quantile(np.asarray(values), [alpha, 1.0 - alpha])
        null_mean = float(np.mean(values))
    return {
        "cohort_id": cohort,
        "disease_label": disease,
        "reference_label": reference,
        "control": control,
        "null_mean_difference": null_mean,
        "null_ci_lower": float(lower),
        "null_ci_upper": float(upper),
        "ci_method": "independent_unit_permutation_percentile",
        "shuffle_replicates_requested": int(config.shuffle_replicates),
        "shuffle_replicates_valid": int(len(values)),
        "seed": int(seed),
        "evidence_role": BIOLOGICAL_CONSISTENCY,
    }


def disease_cohort_consistency(
    individual_food: pd.DataFrame,
    provenance_manifest: Mapping[str, object],
    *,
    config: BiologicalConsistencyConfig | None = None,
    value_column: str = "GMNPS_delta",
    allow_test_data: bool = False,
) -> DiseaseConsistencyResult:
    """Estimate disease-reference differences within cohort at person/component level."""

    validate_evidence_provenance(
        provenance_manifest, allow_test_data=allow_test_data
    )
    if provenance_manifest.get("source_kind") != "gmrepo_disease_labels":
        raise ValueError("disease analysis requires GMrepo disease-label provenance")
    required = {
        "individual_id",
        "cohort_id",
        "disease_label",
        "label_source",
        "label_provenance",
        value_column,
    }
    _require_columns(individual_food, required, "GMrepo individual_food")
    _reject_outcomes(individual_food)
    frame = individual_food.copy()
    for column in (
        "individual_id",
        "cohort_id",
        "disease_label",
        "label_source",
        "label_provenance",
    ):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} contains missing or empty values")
        frame[column] = frame[column].astype(str).str.strip()
        _check_person_invariant(frame, column)
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    if np.isinf(frame[value_column]).any():
        raise ValueError(f"{value_column} cannot contain infinite values")
    frame["__value_finite"] = frame[value_column].notna()

    cluster_column = "component_id" if "component_id" in frame.columns else "individual_id"
    if cluster_column == "component_id":
        if frame[cluster_column].isna().any():
            raise ValueError("component_id contains missing values")
        frame[cluster_column] = frame[cluster_column].astype(str).str.strip()
        _check_person_invariant(frame, cluster_column)

    person = (
        frame.groupby("individual_id", sort=True)
        .agg(
            cohort_id=("cohort_id", "first"),
            disease_label=("disease_label", "first"),
            label_source=("label_source", "first"),
            label_provenance=("label_provenance", "first"),
            cluster_id=(cluster_column, "first"),
            value=(value_column, "mean"),
            n_rows=(value_column, "size"),
            n_analysis_rows=("__value_finite", "sum"),
        )
        .reset_index()
    )
    cluster_metadata = person.groupby("cluster_id", dropna=False).agg(
        n_cohorts=("cohort_id", "nunique"),
        n_labels=("disease_label", "nunique"),
    )
    if (cluster_metadata["n_cohorts"] > 1).any():
        raise ValueError("one independent component cannot span cohorts")
    if (cluster_metadata["n_labels"] > 1).any():
        raise ValueError(
            "one independent component cannot carry discordant disease labels for this audit"
        )
    analysis_person = person.loc[np.isfinite(person["value"])].copy()
    units = (
        analysis_person.groupby("cluster_id", sort=True)
        .agg(
            cohort_id=("cohort_id", "first"),
            disease_label=("disease_label", "first"),
            value=("value", "mean"),
            n_individuals=("individual_id", "nunique"),
            n_rows=("n_rows", "sum"),
            n_analysis_rows=("n_analysis_rows", "sum"),
        )
        .reset_index()
    )

    reference = provenance_manifest.get("reference_label")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("provenance reference_label must be a nonempty string")
    reference = reference.strip()
    resolved = config or BiologicalConsistencyConfig()
    estimate_rows: list[dict[str, object]] = []
    inference_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []

    for cohort, cohort_units in units.groupby("cohort_id", sort=True):
        labels = sorted(set(cohort_units["disease_label"]) - {reference})
        if reference not in set(cohort_units["disease_label"]):
            raise ValueError(f"cohort {cohort} does not contain reference_label={reference}")
        for disease in labels:
            selected = cohort_units[
                cohort_units["disease_label"].isin([reference, disease])
            ].copy()
            eligible_people = person[
                (person["cohort_id"].eq(cohort))
                & person["disease_label"].isin([reference, disease])
            ]
            selected_people = eligible_people.loc[
                np.isfinite(eligible_people["value"])
            ]
            original = frame[
                (frame["cohort_id"].eq(cohort))
                & frame["disease_label"].isin([reference, disease])
            ]
            estimate = _difference(selected, disease, reference)
            provenance_values = sorted(original["label_provenance"].unique())
            source_values = sorted(original["label_source"].unique())
            base = {
                "cohort_id": str(cohort),
                "disease_label": disease,
                "reference_label": reference,
                "estimate": float(estimate),
                "estimand": "mean independent-unit GMNPS_delta difference",
                "n_individuals": int(selected_people["individual_id"].nunique()),
                "n_eligible_individuals": int(
                    eligible_people["individual_id"].nunique()
                ),
                "n_rows": int(len(original)),
                "n_analysis_rows": int(original["__value_finite"].sum()),
                "n_independent_units": int(selected["cluster_id"].nunique()),
                "label_source": "|".join(source_values),
                "label_provenance": "|".join(provenance_values),
                "evidence_role": BIOLOGICAL_CONSISTENCY,
                "analysis_boundary": "supporting_not_person_food_response_validation",
            }
            estimate_rows.append(base)
            boot, lower, upper = _bootstrap_difference(
                selected, disease, reference, resolved
            )
            inference_rows.append(
                {
                    **base,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": resolved.confidence_level,
                    "ci_method": "cluster_percentile",
                    "cluster_column": cluster_column,
                    "bootstrap_seed": resolved.seed,
                    "bootstrap_replicates_requested": resolved.bootstrap_replicates,
                    "bootstrap_replicates_valid": len(boot),
                    "missing_n_individuals": int(
                        eligible_people["value"].isna().sum()
                    ),
                    "excluded_n_individuals": int(
                        eligible_people["value"].isna().sum()
                    ),
                    "missing_n_rows": int((~original["__value_finite"]).sum()),
                    "excluded_n_rows": int((~original["__value_finite"]).sum()),
                    "exclusion_rule": f"nonfinite_{value_column}",
                }
            )
            for control in ("shuffled_label", "shuffled_microbiome"):
                control_rows.append(
                    _shuffle_control(
                        selected,
                        disease,
                        reference,
                        resolved,
                        control=control,
                        cohort=str(cohort),
                    )
                )
    if not estimate_rows:
        raise ValueError("no within-cohort disease-versus-reference contrast is estimable")
    return DiseaseConsistencyResult(
        evidence_role=BIOLOGICAL_CONSISTENCY,
        estimates=pd.DataFrame(estimate_rows),
        inference=pd.DataFrame(inference_rows),
        shuffle_controls=pd.DataFrame(control_rows),
    )


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


def zoe_aggregate_rank_consistency(
    aggregate_ranks: pd.DataFrame,
    provenance_manifest: Mapping[str, object],
    *,
    allow_test_data: bool = False,
) -> dict[str, object]:
    """Compare published aggregate ranks without person-level interpretation."""

    validate_evidence_provenance(
        provenance_manifest, allow_test_data=allow_test_data
    )
    if provenance_manifest.get("source_kind") != "zoe_aggregate_ranks":
        raise ValueError("ZOE rank analysis requires zoe_aggregate_ranks provenance")
    _require_columns(
        aggregate_ranks,
        {"food_id", "gmnps_rank", "zoe_rank", "rank_source"},
        "ZOE aggregate ranks",
    )
    _reject_outcomes(aggregate_ranks)
    if aggregate_ranks["food_id"].duplicated().any():
        raise ValueError("ZOE aggregate ranks must contain one row per food_id")
    return {
        "n_foods": int(len(aggregate_ranks)),
        "spearman_rank_correlation": _spearman(
            aggregate_ranks["gmnps_rank"], aggregate_ranks["zoe_rank"]
        ),
        "rank_sources": sorted(aggregate_ranks["rank_source"].astype(str).unique()),
        "evidence_role": BIOLOGICAL_CONSISTENCY,
        "analysis_boundary": "aggregate_ranks_not_person_food_responses",
    }


def knowledge_path_consistency(
    paths: pd.DataFrame,
    provenance_manifest: Mapping[str, object],
    *,
    allow_test_data: bool = False,
) -> dict[str, object]:
    """Summarize pre-adjudicated signed knowledge paths."""

    validate_evidence_provenance(
        provenance_manifest, allow_test_data=allow_test_data
    )
    if provenance_manifest.get("source_kind") != "knowledge_paths":
        raise ValueError("knowledge-path analysis requires knowledge_paths provenance")
    _require_columns(
        paths,
        {
            "path_id",
            "source_node",
            "target_node",
            "direction",
            "is_direction_consistent",
            "path_provenance",
        },
        "knowledge paths",
    )
    _reject_outcomes(paths)
    if paths["path_id"].duplicated().any():
        raise ValueError("knowledge paths must contain unique path_id values")
    if not paths["direction"].isin(["positive", "negative"]).all():
        raise ValueError("knowledge path direction must be positive or negative")
    consistent = paths["is_direction_consistent"]
    if not consistent.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("is_direction_consistent must be boolean")
    return {
        "n_paths": int(len(paths)),
        "n_direction_consistent": int(consistent.sum()),
        "direction_consistent_fraction": float(consistent.mean()),
        "n_path_provenance_sources": int(paths["path_provenance"].nunique()),
        "evidence_role": MECHANISTIC_CONSISTENCY,
        "analysis_boundary": "mechanistic_paths_only",
    }


__all__ = [
    "BIOLOGICAL_CONSISTENCY",
    "MECHANISTIC_CONSISTENCY",
    "BiologicalConsistencyConfig",
    "DiseaseConsistencyResult",
    "disease_cohort_consistency",
    "knowledge_path_consistency",
    "validate_evidence_provenance",
    "zoe_aggregate_rank_consistency",
]
