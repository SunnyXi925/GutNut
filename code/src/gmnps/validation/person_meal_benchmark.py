"""Leakage-safe nested validation for locked person-by-meal comparators.

The production entry point reuses the Task 2 method-lock validator, then
independently rereads all four live trust roots before delegating to the Task 2
outcome loader.  This module never discovers endpoints or comparator candidates
from outcomes and never writes benchmark results.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from gmnps.data_sources.predict_zoe_loader import (
    FrozenValidationConfig,
    LoadedTable,
    load_frozen_validation_config,
    load_outcome_table_after_gate,
)
from gmnps.scoring.attribute_gmnps import (
    FCS2_FNDDS_REGISTRY_DIGEST_ALGORITHM,
    FCS2_FNDDS_REGISTRY_SCHEMA_VERSION,
    FCS2_FNDDS_REGISTRY_VERSION,
    load_release_registry_snapshot,
)
from gmnps.validation.cohort_split import NestedGroupSplit, make_nested_group_splits
from gmnps.validation.method_lock_gate import validate_method_lock_manifest


REQUIRED_COMPARATORS = (
    "clinical_demographic_diet",
    "fcs_only",
    "microbiome_only",
    "fcs_microbiome",
    "legacy_final_score_offset",
    "locked_attribute_gmnps",
    "random_microbiome",
    "shuffled_mapping",
)
_METRICS = (
    "mae",
    "rmse",
    "spearman",
    "calibration_slope",
    "calibration_intercept",
)
_ALPHA_GRID = (0.1, 1.0, 10.0)
_TUNING_METRIC = "mae"
BOOTSTRAP_REPLICATES = 2_000
PERMUTATION_REPLICATES = 2_000
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_KEY_COLUMNS = ("participant_id", "meal_id")
_NONPREDICTOR_IDENTIFIERS = frozenset(
    {
        "participant_id",
        "meal_id",
        "food_id",
        "family_id",
        "twin_id",
        "cohort_id",
        "row_id",
        "outer_fold",
        "inner_fold",
    }
)
_LEAKAGE_TOKENS = ("response", "outcome", "target", "label", "post_split")


class BenchmarkAccessBlocked(PermissionError):
    """Raised before benchmark outcome access when trust cannot be proven."""


@dataclass(frozen=True)
class FeatureContract:
    """Predeclared feature blocks for the eight locked comparators."""

    clinical_demographic_diet: tuple[str, ...]
    fcs: tuple[str, ...]
    microbiome: tuple[str, ...]
    legacy_final_score_offset: tuple[str, ...]
    locked_attribute_gmnps: tuple[str, ...]
    mapping_unit: str

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "mapping_unit":
                if not isinstance(value, str) or not value:
                    raise TypeError("mapping_unit must be a nonempty string")
                continue
            if (
                not isinstance(value, tuple)
                or not value
                or any(not isinstance(column, str) or not column for column in value)
                or len(set(value)) != len(value)
            ):
                raise TypeError(f"{field.name} must contain unique feature names")


@dataclass(frozen=True)
class VerifiedBenchmarkLock:
    """Task 3 verification result bound to the live frozen configuration."""

    manifest: Mapping[str, object]
    config: FrozenValidationConfig


@dataclass(frozen=True)
class LockedOutcomeInput:
    """Outcome table returned only after both Task 3 and Task 2 gates pass."""

    outcome: LoadedTable
    verified_lock: VerifiedBenchmarkLock


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    n_participants: int


@dataclass(frozen=True)
class PermutationTestResult:
    estimate_delta: float
    p_value: float
    null_distribution: tuple[float, ...]
    n_participants: int


@dataclass(frozen=True)
class BenchmarkResult:
    predictions: pd.DataFrame
    absolute_metrics: pd.DataFrame
    paired_metrics: pd.DataFrame
    fit_audit: pd.DataFrame
    splits: tuple[NestedGroupSplit, ...]


def _read_regular_bytes(path: Path, label: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise BenchmarkAccessBlocked(f"benchmark method-lock {label} is missing")
        return path.read_bytes()
    except OSError as error:
        raise BenchmarkAccessBlocked(
            f"benchmark method-lock {label} is missing or unreadable"
        ) from error


def _read_manifest(path: Path) -> dict[str, object]:
    raw = _read_regular_bytes(path, "manifest")
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkAccessBlocked("benchmark method-lock manifest is invalid") from error
    if not isinstance(payload, dict):
        raise BenchmarkAccessBlocked("benchmark method-lock manifest must be an object")
    return payload


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise BenchmarkAccessBlocked(f"benchmark method-lock {label} hash is invalid")
    return value


def _path_from_lock(method_lock_paths: object, field: str) -> Path:
    value = getattr(method_lock_paths, field, None)
    if not isinstance(value, (str, Path)):
        raise BenchmarkAccessBlocked(f"benchmark method-lock paths omit {field}")
    return Path(value)


def validate_benchmark_method_lock(
    *,
    manifest_path: Path,
    method_lock_paths: object,
) -> VerifiedBenchmarkLock:
    """Reuse Task 2 validation, then independently recheck live trust bytes."""

    manifest = _read_manifest(Path(manifest_path))
    snapshot = manifest.get("release_registry_snapshot")
    if not isinstance(snapshot, dict):
        raise BenchmarkAccessBlocked(
            "benchmark method-lock manifest omits release registry"
        )
    expected_registry_sha256 = _require_digest(
        snapshot.get("snapshot_sha256"),
        "release-registry snapshot",
    )

    try:
        validate_method_lock_manifest(
            manifest,
            method_lock_paths,
            expected_release_registry_sha256=expected_registry_sha256,
        )
    except Exception as error:
        raise BenchmarkAccessBlocked(
            "Task 2 method-lock validation failed before benchmark outcome access"
        ) from error

    config_path = _path_from_lock(method_lock_paths, "person_meal_validation_config")
    schema_path = _path_from_lock(method_lock_paths, "method_lock_schema")
    registry_path = _path_from_lock(method_lock_paths, "release_registry")
    gate_path = _path_from_lock(method_lock_paths, "gate_implementation")
    config_bytes = _read_regular_bytes(config_path, "validation config")
    schema_bytes = _read_regular_bytes(schema_path, "schema")
    registry_bytes = _read_regular_bytes(registry_path, "release registry")
    gate_bytes = _read_regular_bytes(gate_path, "gate implementation")

    live_hashes = {
        "person_meal_validation_config_sha256": sha256(config_bytes).hexdigest(),
        "method_lock_schema_sha256": sha256(schema_bytes).hexdigest(),
        "method_lock_gate_implementation_sha256": sha256(gate_bytes).hexdigest(),
    }
    for field, observed in live_hashes.items():
        expected = _require_digest(manifest.get(field), field)
        if observed != expected:
            raise BenchmarkAccessBlocked(
                f"benchmark method-lock {field} hash does not match live bytes"
            )
    if sha256(registry_bytes).hexdigest() != expected_registry_sha256:
        raise BenchmarkAccessBlocked(
            "benchmark method-lock release registry hash does not match live bytes"
        )

    try:
        registry = load_release_registry_snapshot(registry_bytes)
    except (TypeError, ValueError) as error:
        raise BenchmarkAccessBlocked(
            "benchmark method-lock release registry constants are invalid"
        ) from error
    expected_registry_metadata = {
        "schema_version": FCS2_FNDDS_REGISTRY_SCHEMA_VERSION,
        "registry_version": FCS2_FNDDS_REGISTRY_VERSION,
        "digest_algorithm": FCS2_FNDDS_REGISTRY_DIGEST_ALGORITHM,
        "canonical_release_set": list(registry.canonical_release_set),
    }
    observed_registry_metadata = {
        "schema_version": registry.schema_version,
        "registry_version": registry.registry_version,
        "digest_algorithm": registry.digest_algorithm,
        "canonical_release_set": list(registry.canonical_release_set),
    }
    if observed_registry_metadata != expected_registry_metadata:
        raise BenchmarkAccessBlocked(
            "benchmark method-lock live release registry constants changed"
        )
    for field, expected in expected_registry_metadata.items():
        if snapshot.get(field) != expected:
            raise BenchmarkAccessBlocked(
                f"benchmark method-lock manifest registry {field} is not constant"
            )

    try:
        config = load_frozen_validation_config(
            config_path,
            expected_sha256=live_hashes["person_meal_validation_config_sha256"],
        )
    except Exception as error:
        raise BenchmarkAccessBlocked(
            "benchmark method-lock frozen validation config is invalid"
        ) from error
    return VerifiedBenchmarkLock(manifest=manifest, config=config)


def load_locked_outcomes_for_benchmark(
    source_id: str,
    *,
    manifest_path: Path,
    method_lock_paths: object,
) -> LockedOutcomeInput:
    """Open outcomes only after Task 3 independently revalidates Task 2 lock."""

    verified = validate_benchmark_method_lock(
        manifest_path=manifest_path,
        method_lock_paths=method_lock_paths,
    )
    outcome = load_outcome_table_after_gate(
        source_id,
        manifest_path=manifest_path,
        method_lock_paths=method_lock_paths,
    )
    if not isinstance(outcome, LoadedTable):
        raise BenchmarkAccessBlocked("benchmark outcome loader returned an invalid table")
    return LockedOutcomeInput(outcome=outcome, verified_lock=verified)


def validate_feature_contract(
    frame: pd.DataFrame,
    feature_contract: FeatureContract,
    *,
    endpoint_names: Sequence[str],
) -> None:
    """Reject identifiers, outcomes, and post-split statistics as predictors."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("predictors must be a pandas DataFrame")
    if not isinstance(feature_contract, FeatureContract):
        raise TypeError("feature_contract must be a FeatureContract")
    endpoints = set(endpoint_names)
    feature_blocks = (
        feature_contract.clinical_demographic_diet,
        feature_contract.fcs,
        feature_contract.microbiome,
        feature_contract.legacy_final_score_offset,
        feature_contract.locked_attribute_gmnps,
    )
    feature_columns = tuple(column for block in feature_blocks for column in block)
    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"feature contract references missing columns: {missing}")
    if feature_contract.mapping_unit not in frame.columns:
        raise ValueError("feature contract mapping unit is missing")
    for column in feature_columns:
        normalized = column.lower()
        if (
            column in endpoints
            or column in _NONPREDICTOR_IDENTIFIERS
            or column.endswith("_id")
            or any(token in normalized for token in _LEAKAGE_TOKENS)
        ):
            raise ValueError(
                f"feature contract contains forbidden leakage/identifier column {column}"
            )


def _finite_vectors(
    y_true: np.ndarray,
    prediction: np.ndarray,
    *additional: np.ndarray,
) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(array) for array in (y_true, prediction, *additional))
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("metric inputs must be one-dimensional")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("metric inputs must have the same shape")
    numeric = tuple(np.asarray(array, dtype=float) for array in arrays)
    finite = np.logical_and.reduce([np.isfinite(array) for array in numeric])
    return tuple(array[finite] for array in numeric)


def compute_regression_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Compute locked absolute regression, rank, and calibration metrics."""

    y, pred = _finite_vectors(y_true, prediction)
    if len(y) == 0:
        return {metric: math.nan for metric in _METRICS}
    residual = y - pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    if len(y) < 2 or len(np.unique(y)) < 2 or len(np.unique(pred)) < 2:
        spearman = math.nan
    else:
        spearman = float(
            pd.Series(y).rank(method="average").corr(
                pd.Series(pred).rank(method="average"), method="pearson"
            )
        )
    if len(y) < 2 or float(np.var(pred)) <= np.finfo(float).eps:
        calibration_slope = math.nan
        calibration_intercept = float(np.mean(y))
    else:
        design = np.column_stack([np.ones(len(pred), dtype=float), pred])
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        calibration_intercept = float(coefficients[0])
        calibration_slope = float(coefficients[1])
    return {
        "mae": mae,
        "rmse": rmse,
        "spearman": spearman,
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
    }


def _metric_delta(
    y_true: np.ndarray,
    prediction: np.ndarray,
    reference_prediction: np.ndarray | None,
    metric: str,
) -> float:
    estimate = compute_regression_metrics(y_true, prediction)[metric]
    if reference_prediction is None:
        return estimate
    reference = compute_regression_metrics(y_true, reference_prediction)[metric]
    return float(estimate - reference)


def participant_bootstrap_ci(
    y_true: np.ndarray,
    prediction: np.ndarray,
    participant_ids: np.ndarray,
    *,
    metric: str,
    n_bootstrap: int,
    seed: int,
    reference_prediction: np.ndarray | None = None,
) -> BootstrapInterval:
    """Bootstrap absolute or paired metrics by participant clusters."""

    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    identifiers = np.asarray(participant_ids)
    arrays = [np.asarray(y_true), np.asarray(prediction)]
    if reference_prediction is not None:
        arrays.append(np.asarray(reference_prediction))
    if identifiers.ndim != 1 or any(array.shape != identifiers.shape for array in arrays):
        raise ValueError("bootstrap inputs must have the same shape")
    numeric = [np.asarray(array, dtype=float) for array in arrays]
    finite = np.logical_and.reduce([np.isfinite(array) for array in numeric])
    if not finite.any():
        return BootstrapInterval(math.nan, math.nan, math.nan, 0)
    y = numeric[0][finite]
    pred = numeric[1][finite]
    reference = numeric[2][finite] if reference_prediction is not None else None
    ids = identifiers[finite].astype(str)
    unique = np.asarray(sorted(set(ids)))
    group_indices = {value: np.flatnonzero(ids == value) for value in unique}
    estimate = _metric_delta(y, pred, reference, metric)
    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([group_indices[value] for value in sampled])
        sampled_reference = reference[positions] if reference is not None else None
        replicates.append(
            _metric_delta(y[positions], pred[positions], sampled_reference, metric)
        )
    finite_replicates = np.asarray(replicates, dtype=float)
    finite_replicates = finite_replicates[np.isfinite(finite_replicates)]
    if len(finite_replicates) == 0:
        lower = upper = math.nan
    else:
        lower, upper = np.percentile(finite_replicates, [2.5, 97.5]).tolist()
    return BootstrapInterval(
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
        n_participants=len(unique),
    )


def participant_permutation_test(
    y_true: np.ndarray,
    prediction: np.ndarray,
    reference_prediction: np.ndarray,
    participant_ids: np.ndarray,
    *,
    metric: str,
    n_permutations: int,
    seed: int,
) -> PermutationTestResult:
    """Swap model labels for complete participant clusters under the paired null."""

    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}")
    if not isinstance(n_permutations, int) or n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    identifiers = np.asarray(participant_ids)
    if identifiers.ndim != 1:
        raise ValueError("participant_ids must be one-dimensional")
    y, pred, reference = _finite_vectors(
        y_true,
        prediction,
        reference_prediction,
    )
    original_shape = np.asarray(y_true).shape
    finite = np.logical_and.reduce(
        [
            np.isfinite(np.asarray(y_true, dtype=float)),
            np.isfinite(np.asarray(prediction, dtype=float)),
            np.isfinite(np.asarray(reference_prediction, dtype=float)),
        ]
    )
    if identifiers.shape != original_shape:
        raise ValueError("permutation inputs must have the same shape")
    ids = identifiers[finite].astype(str)
    unique = np.asarray(sorted(set(ids)))
    observed = _metric_delta(y, pred, reference, metric)
    rng = np.random.default_rng(seed)
    null_statistics = []
    for _ in range(n_permutations):
        swaps = dict(zip(unique, rng.integers(0, 2, size=len(unique)).astype(bool)))
        permuted_prediction = pred.copy()
        permuted_reference = reference.copy()
        for participant_id, swap in swaps.items():
            if swap:
                mask = ids == participant_id
                permuted_prediction[mask] = reference[mask]
                permuted_reference[mask] = pred[mask]
        null_statistics.append(
            _metric_delta(
                y,
                permuted_prediction,
                permuted_reference,
                metric,
            )
        )
    null = np.asarray(null_statistics, dtype=float)
    if not np.isfinite(observed):
        p_value = math.nan
    else:
        finite_null = null[np.isfinite(null)]
        p_value = float(
            (1 + np.sum(np.abs(finite_null) >= abs(observed)))
            / (1 + len(finite_null))
        )
    return PermutationTestResult(
        estimate_delta=float(observed),
        p_value=p_value,
        null_distribution=tuple(float(value) for value in null),
        n_participants=len(unique),
    )


def _canonical_hash(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _endpoint_specs(config: FrozenValidationConfig) -> tuple[dict[str, object], ...]:
    contract = config.payload.get("outcome_contract")
    if not isinstance(contract, dict) or not isinstance(contract.get("endpoints"), list):
        raise ValueError("frozen validation config omits endpoint contract")
    endpoints = tuple(dict(value) for value in contract["endpoints"])
    if tuple(value.get("name") for value in endpoints) != (
        "glucose_iAUC_2h",
        "tg_6h_rise",
        "c_peptide_iAUC_2h",
    ):
        raise ValueError("frozen endpoint names or order changed")
    return endpoints


def _comparator_columns(
    contract: FeatureContract,
) -> dict[str, tuple[str, ...]]:
    return {
        "clinical_demographic_diet": contract.clinical_demographic_diet,
        "fcs_only": contract.fcs,
        "microbiome_only": contract.microbiome,
        "fcs_microbiome": contract.fcs + contract.microbiome,
        "legacy_final_score_offset": contract.legacy_final_score_offset,
        "locked_attribute_gmnps": contract.locked_attribute_gmnps,
        "random_microbiome": contract.microbiome,
        "shuffled_mapping": contract.locked_attribute_gmnps,
    }


def _pipeline(frame: pd.DataFrame, alpha: float) -> Pipeline:
    numeric = [column for column in frame.columns if is_numeric_dtype(frame[column])]
    categorical = [column for column in frame.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("comparator has no usable feature columns")
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(transformers, remainder="drop"),
            ),
            ("model", Ridge(alpha=float(alpha))),
        ]
    )


def _random_microbiome(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    columns = list(train.columns)
    return (
        pd.DataFrame(
            rng.standard_normal((len(train), len(columns))),
            index=train.index,
            columns=columns,
        ),
        pd.DataFrame(
            rng.standard_normal((len(evaluation), len(columns))),
            index=evaluation.index,
            columns=columns,
        ),
    )


def _shuffled_mapping(
    full_frame: pd.DataFrame,
    train_positions: np.ndarray,
    evaluation_positions: np.ndarray,
    columns: tuple[str, ...],
    mapping_unit: str,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = full_frame.iloc[train_positions]
    evaluation = full_frame.iloc[evaluation_positions]
    train_numeric = train.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    unit_profiles = train_numeric.assign(
        __mapping_unit__=train[mapping_unit].astype(str).to_numpy()
    ).groupby("__mapping_unit__", sort=True).median()
    if unit_profiles.empty:
        raise ValueError("shuffled mapping has no development mapping units")
    units = unit_profiles.index.to_numpy(dtype=str)
    rng = np.random.default_rng(seed)
    permuted_units = units[rng.permutation(len(units))]
    mapping = dict(zip(units, permuted_units))

    def project(rows: pd.DataFrame) -> pd.DataFrame:
        projected = []
        for value in rows[mapping_unit].astype(str):
            source = mapping.get(value)
            if source is None:
                digest = sha256(f"{seed}:{value}".encode("utf-8")).digest()
                source = units[int.from_bytes(digest[:8], "big") % len(units)]
            projected.append(unit_profiles.loc[source].to_numpy(dtype=float))
        return pd.DataFrame(projected, index=rows.index, columns=list(columns))

    return project(train), project(evaluation)


def _feature_pair(
    frame: pd.DataFrame,
    comparator: str,
    columns: tuple[str, ...],
    feature_contract: FeatureContract,
    train_positions: np.ndarray,
    evaluation_positions: np.ndarray,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame.iloc[train_positions].loc[:, list(columns)].copy()
    evaluation = frame.iloc[evaluation_positions].loc[:, list(columns)].copy()
    if comparator == "random_microbiome":
        return _random_microbiome(train, evaluation, seed=seed)
    if comparator == "shuffled_mapping":
        return _shuffled_mapping(
            frame,
            train_positions,
            evaluation_positions,
            columns,
            feature_contract.mapping_unit,
            seed=seed,
        )
    return train, evaluation


def _fit_participant_ids(frame: pd.DataFrame, positions: np.ndarray) -> tuple[str, ...]:
    return tuple(sorted(frame.iloc[positions]["participant_id"].astype(str).unique()))


def _prepare_joined_frame(
    predictors: pd.DataFrame,
    outcome: pd.DataFrame,
    endpoint_names: tuple[str, ...],
) -> pd.DataFrame:
    for label, frame in (("predictors", predictors), ("outcomes", outcome)):
        missing = set(_KEY_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"{label} missing person-meal keys: {sorted(missing)}")
        if frame.duplicated(list(_KEY_COLUMNS)).any():
            raise ValueError(f"{label} person-meal keys must be unique")
    forbidden_predictor_columns = set(endpoint_names).intersection(predictors.columns)
    forbidden_predictor_columns.update(
        column
        for column in predictors.columns
        if column not in _KEY_COLUMNS
        and any(token in column.lower() for token in _LEAKAGE_TOKENS)
    )
    if forbidden_predictor_columns:
        raise ValueError(
            "predictor table contains response/post-split leakage columns: "
            f"{sorted(forbidden_predictor_columns)}"
        )
    outcome_projection = outcome.loc[
        :, [*_KEY_COLUMNS, *[name for name in endpoint_names if name in outcome.columns]]
    ]
    joined = outcome_projection.merge(
        predictors,
        on=list(_KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("one or more outcome rows lack the same predictor opportunity")
    joined = joined.drop(columns="_merge").reset_index(drop=True)
    joined.insert(
        0,
        "row_id",
        joined["participant_id"].astype(str)
        + "::"
        + joined["meal_id"].astype(str),
    )
    return joined


def _benchmark_predictions(
    frame: pd.DataFrame,
    config: FrozenValidationConfig,
    feature_contract: FeatureContract,
    endpoints: tuple[dict[str, object], ...],
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[NestedGroupSplit, ...]]:
    split_config = config.payload["split"]
    seeds = config.payload["seeds"]
    splits = make_nested_group_splits(
        frame,
        outer_folds=int(split_config["outer_folds"]),
        inner_folds=int(split_config["inner_folds"]),
        outer_seed=int(seeds["outer_split"]),
        inner_seed=int(seeds["inner_cv"]),
    )
    comparator_columns = _comparator_columns(feature_contract)
    prediction_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    endpoint_seed_offset = {
        str(endpoint["name"]): index * 1_000_003
        for index, endpoint in enumerate(endpoints)
    }
    comparator_seed_offset = {
        comparator: index * 10_007
        for index, comparator in enumerate(REQUIRED_COMPARATORS)
    }

    for endpoint in endpoints:
        endpoint_name = str(endpoint["name"])
        if endpoint_name not in frame.columns:
            if endpoint.get("role") == "primary":
                raise ValueError(f"required primary endpoint {endpoint_name} is missing")
            continue
        for comparator in REQUIRED_COMPARATORS:
            columns = comparator_columns[comparator]
            for nested in splits:
                inner_scores: dict[float, list[float]] = {
                    alpha: [] for alpha in _ALPHA_GRID
                }
                for inner in nested.inner:
                    inner_train = np.asarray(inner.train_positions, dtype=int)
                    inner_validation = np.asarray(inner.test_positions, dtype=int)
                    train_available = frame.iloc[inner_train][endpoint_name].notna().to_numpy()
                    validation_available = frame.iloc[inner_validation][
                        endpoint_name
                    ].notna().to_numpy()
                    inner_train = inner_train[train_available]
                    inner_validation = inner_validation[validation_available]
                    if len(inner_train) == 0 or len(inner_validation) == 0:
                        raise ValueError(
                            f"endpoint {endpoint_name} has an empty nested development fold"
                        )
                    stage_seed = (
                        int(seeds["inner_cv"])
                        + endpoint_seed_offset[endpoint_name]
                        + comparator_seed_offset[comparator]
                        + nested.outer.fold_id * 101
                        + inner.fold_id
                    )
                    x_train, x_validation = _feature_pair(
                        frame,
                        comparator,
                        columns,
                        feature_contract,
                        inner_train,
                        inner_validation,
                        seed=stage_seed,
                    )
                    y_train = frame.iloc[inner_train][endpoint_name].to_numpy(dtype=float)
                    y_validation = frame.iloc[inner_validation][endpoint_name].to_numpy(
                        dtype=float
                    )
                    for alpha in _ALPHA_GRID:
                        estimator = _pipeline(x_train, alpha)
                        estimator.fit(x_train, y_train)
                        prediction = estimator.predict(x_validation)
                        inner_scores[alpha].append(
                            compute_regression_metrics(y_validation, prediction)[
                                _TUNING_METRIC
                            ]
                        )
                        audit_rows.append(
                            {
                                "endpoint": endpoint_name,
                                "comparator": comparator,
                                "outer_fold": nested.outer.fold_id,
                                "stage": "inner_tuning",
                                "inner_fold": inner.fold_id,
                                "alpha": alpha,
                                "fit_participant_ids": _fit_participant_ids(
                                    frame, inner_train
                                ),
                                "validation_participant_ids": _fit_participant_ids(
                                    frame, inner_validation
                                ),
                            }
                        )
                selected_alpha = min(
                    _ALPHA_GRID,
                    key=lambda alpha: (
                        float(np.mean(inner_scores[alpha])),
                        alpha,
                    ),
                )
                outer_train = np.asarray(nested.outer.train_positions, dtype=int)
                outer_test = np.asarray(nested.outer.test_positions, dtype=int)
                outer_train = outer_train[
                    frame.iloc[outer_train][endpoint_name].notna().to_numpy()
                ]
                outer_test = outer_test[
                    frame.iloc[outer_test][endpoint_name].notna().to_numpy()
                ]
                if len(outer_train) == 0 or len(outer_test) == 0:
                    raise ValueError(
                        f"endpoint {endpoint_name} has an empty outer analysis fold"
                    )
                outer_seed = (
                    int(seeds["outer_split"])
                    + endpoint_seed_offset[endpoint_name]
                    + comparator_seed_offset[comparator]
                    + nested.outer.fold_id * 101
                )
                x_train, x_test = _feature_pair(
                    frame,
                    comparator,
                    columns,
                    feature_contract,
                    outer_train,
                    outer_test,
                    seed=outer_seed,
                )
                y_train = frame.iloc[outer_train][endpoint_name].to_numpy(dtype=float)
                estimator = _pipeline(x_train, selected_alpha)
                estimator.fit(x_train, y_train)
                prediction = estimator.predict(x_test)
                audit_rows.append(
                    {
                        "endpoint": endpoint_name,
                        "comparator": comparator,
                        "outer_fold": nested.outer.fold_id,
                        "stage": "outer_refit",
                        "inner_fold": None,
                        "alpha": selected_alpha,
                        "fit_participant_ids": _fit_participant_ids(frame, outer_train),
                        "validation_participant_ids": _fit_participant_ids(
                            frame, outer_test
                        ),
                    }
                )
                test_rows = frame.iloc[outer_test]
                for row, y_pred in zip(test_rows.itertuples(index=False), prediction):
                    prediction_rows.append(
                        {
                            "row_id": row.row_id,
                            "participant_id": row.participant_id,
                            "meal_id": row.meal_id,
                            "endpoint": endpoint_name,
                            "outer_fold": nested.outer.fold_id,
                            "comparator": comparator,
                            "y_true": float(getattr(row, endpoint_name)),
                            "y_pred": float(y_pred),
                            "selected_alpha": float(selected_alpha),
                        }
                    )
    predictions = pd.DataFrame.from_records(prediction_rows)
    comparator_order = {value: index for index, value in enumerate(REQUIRED_COMPARATORS)}
    predictions["__comparator_order__"] = predictions["comparator"].map(comparator_order)
    predictions = predictions.sort_values(
        ["endpoint", "outer_fold", "__comparator_order__", "row_id"],
        kind="mergesort",
    ).drop(columns="__comparator_order__").reset_index(drop=True)
    audit = pd.DataFrame.from_records(audit_rows)
    audit["__comparator_order__"] = audit["comparator"].map(comparator_order)
    audit = audit.sort_values(
        [
            "endpoint",
            "outer_fold",
            "__comparator_order__",
            "stage",
            "inner_fold",
            "alpha",
        ],
        kind="mergesort",
        na_position="last",
    ).drop(columns="__comparator_order__").reset_index(drop=True)
    audit["inner_fold"] = pd.Series(
        [None if pd.isna(value) else int(value) for value in audit["inner_fold"]],
        dtype=object,
    )
    return predictions, audit, splits


def _metric_tables(
    predictions: pd.DataFrame,
    config: FrozenValidationConfig,
    endpoints: tuple[dict[str, object], ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = config.payload["seeds"]
    endpoint_by_name = {str(endpoint["name"]): endpoint for endpoint in endpoints}
    absolute_rows = []
    paired_rows = []
    for endpoint_name, endpoint_predictions in predictions.groupby(
        "endpoint", sort=False
    ):
        endpoint_hash = _canonical_hash(endpoint_by_name[endpoint_name])
        for comparator_index, comparator in enumerate(REQUIRED_COMPARATORS):
            comparator_frame = endpoint_predictions.loc[
                endpoint_predictions["comparator"].eq(comparator)
            ].sort_values("row_id")
            y = comparator_frame["y_true"].to_numpy(dtype=float)
            pred = comparator_frame["y_pred"].to_numpy(dtype=float)
            participants = comparator_frame["participant_id"].astype(str).to_numpy()
            metadata = {
                "endpoint": endpoint_name,
                "comparator": comparator,
                "n_participants": int(comparator_frame["participant_id"].nunique()),
                "n_meals": int(comparator_frame["row_id"].nunique()),
                "split_seed": int(seeds["outer_split"]),
                "endpoint_config_sha256": endpoint_hash,
                "validation_config_sha256": config.sha256,
            }
            for metric_index, metric in enumerate(_METRICS):
                interval = participant_bootstrap_ci(
                    y,
                    pred,
                    participants,
                    metric=metric,
                    n_bootstrap=BOOTSTRAP_REPLICATES,
                    seed=int(seeds["bootstrap"])
                    + comparator_index * 1_009
                    + metric_index,
                )
                absolute_rows.append(
                    {
                        **metadata,
                        "metric": metric,
                        "estimate": interval.estimate,
                        "ci_lower": interval.lower,
                        "ci_upper": interval.upper,
                    }
                )

        model_frame = endpoint_predictions.loc[
            endpoint_predictions["comparator"].eq("locked_attribute_gmnps")
        ].sort_values("row_id")
        for reference_index, reference_name in enumerate(
            comparator
            for comparator in REQUIRED_COMPARATORS
            if comparator != "locked_attribute_gmnps"
        ):
            reference_frame = endpoint_predictions.loc[
                endpoint_predictions["comparator"].eq(reference_name)
            ].sort_values("row_id")
            if tuple(model_frame["row_id"]) != tuple(reference_frame["row_id"]):
                raise ValueError("comparators do not share identical person-meal rows")
            y = model_frame["y_true"].to_numpy(dtype=float)
            model_prediction = model_frame["y_pred"].to_numpy(dtype=float)
            reference_prediction = reference_frame["y_pred"].to_numpy(dtype=float)
            participants = model_frame["participant_id"].astype(str).to_numpy()
            metadata = {
                "endpoint": endpoint_name,
                "comparator": "locked_attribute_gmnps",
                "reference": reference_name,
                "n_participants": int(model_frame["participant_id"].nunique()),
                "n_meals": int(model_frame["row_id"].nunique()),
                "split_seed": int(seeds["outer_split"]),
                "endpoint_config_sha256": endpoint_hash,
                "validation_config_sha256": config.sha256,
            }
            for metric_index, metric in enumerate(_METRICS):
                bootstrap_seed = (
                    int(seeds["bootstrap"])
                    + reference_index * 1_009
                    + metric_index
                )
                permutation_seed = (
                    int(seeds["permutation"])
                    + reference_index * 1_009
                    + metric_index
                )
                interval = participant_bootstrap_ci(
                    y,
                    model_prediction,
                    participants,
                    metric=metric,
                    n_bootstrap=BOOTSTRAP_REPLICATES,
                    seed=bootstrap_seed,
                    reference_prediction=reference_prediction,
                )
                permutation = participant_permutation_test(
                    y,
                    model_prediction,
                    reference_prediction,
                    participants,
                    metric=metric,
                    n_permutations=PERMUTATION_REPLICATES,
                    seed=permutation_seed,
                )
                finite_null = np.asarray(permutation.null_distribution, dtype=float)
                finite_null = finite_null[np.isfinite(finite_null)]
                paired_rows.append(
                    {
                        **metadata,
                        "metric": metric,
                        "estimate_delta": interval.estimate,
                        "ci_lower": interval.lower,
                        "ci_upper": interval.upper,
                        "permutation_p_value": permutation.p_value,
                        "permutation_null_mean": (
                            float(np.mean(finite_null))
                            if len(finite_null)
                            else math.nan
                        ),
                    }
                )
    return (
        pd.DataFrame.from_records(absolute_rows),
        pd.DataFrame.from_records(paired_rows),
    )


def run_person_meal_benchmark(
    source_id: str,
    *,
    manifest_path: Path,
    method_lock_paths: object,
    predictors: pd.DataFrame,
    feature_contract: FeatureContract,
) -> BenchmarkResult:
    """Run every locked comparator after the fail-closed production gate."""

    locked = load_locked_outcomes_for_benchmark(
        source_id,
        manifest_path=manifest_path,
        method_lock_paths=method_lock_paths,
    )
    endpoint_specs = _endpoint_specs(locked.verified_lock.config)
    endpoint_names = tuple(str(endpoint["name"]) for endpoint in endpoint_specs)
    validate_feature_contract(
        predictors,
        feature_contract,
        endpoint_names=endpoint_names,
    )
    joined = _prepare_joined_frame(
        predictors,
        locked.outcome.frame,
        endpoint_names,
    )
    predictions, fit_audit, splits = _benchmark_predictions(
        joined,
        locked.verified_lock.config,
        feature_contract,
        endpoint_specs,
    )
    absolute_metrics, paired_metrics = _metric_tables(
        predictions,
        locked.verified_lock.config,
        endpoint_specs,
    )
    return BenchmarkResult(
        predictions=predictions,
        absolute_metrics=absolute_metrics,
        paired_metrics=paired_metrics,
        fit_audit=fit_audit,
        splits=splits,
    )


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "PERMUTATION_REPLICATES",
    "BenchmarkAccessBlocked",
    "BenchmarkResult",
    "BootstrapInterval",
    "FeatureContract",
    "LockedOutcomeInput",
    "PermutationTestResult",
    "REQUIRED_COMPARATORS",
    "VerifiedBenchmarkLock",
    "compute_regression_metrics",
    "load_locked_outcomes_for_benchmark",
    "participant_bootstrap_ci",
    "participant_permutation_test",
    "run_person_meal_benchmark",
    "validate_benchmark_method_lock",
    "validate_feature_contract",
]
