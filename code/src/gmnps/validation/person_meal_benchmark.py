"""Leakage-safe nested validation for locked person-by-meal comparators.

The production entry point reuses the Task 2 method-lock validator, then
independently rereads all four live trust roots before delegating to the Task 2
outcome loader.  This module never discovers endpoints or comparator candidates
from outcomes and never writes benchmark results.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn

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
_FEATURE_BLOCKS = (
    "clinical_demographic_diet",
    "fcs",
    "microbiome",
    "legacy_final_score_offset",
    "locked_attribute_gmnps",
)
_COLUMN_ROLES = frozenset({"predictor", "identifier", "grouping", "mapping_unit"})
_DATA_TYPES = frozenset({"number", "string", "category", "boolean"})
_STRUCTURAL_ROLES = {
    "participant_id": "identifier",
    "meal_id": "identifier",
    "food_id": "mapping_unit",
    "family_id": "grouping",
    "twin_id": "grouping",
    "cohort_id": "grouping",
}
_COMPARATOR_SEMANTICS = {
    "clinical_demographic_diet": (("clinical_demographic_diet",), "identity", "baseline"),
    "fcs_only": (("fcs",), "identity", "baseline"),
    "microbiome_only": (("microbiome",), "identity", "baseline"),
    "fcs_microbiome": (("fcs", "microbiome"), "identity", "baseline"),
    "legacy_final_score_offset": (("legacy_final_score_offset",), "identity", "baseline"),
    "locked_attribute_gmnps": (("locked_attribute_gmnps",), "identity", "primary_model"),
    "random_microbiome": (("microbiome",), "random_null", "null"),
    "shuffled_mapping": (("locked_attribute_gmnps",), "development_derangement", "null"),
}


class BenchmarkAccessBlocked(PermissionError):
    """Raised before benchmark outcome access when trust cannot be proven."""


@dataclass(frozen=True)
class FeatureColumn:
    """One exact column declared by the pre-outcome predictor-only stage."""

    name: str
    data_type: str
    role: str
    nullable: bool
    source_artifact_id: str


@dataclass(frozen=True)
class FeatureContract:
    """Immutable semantic projection of canonical feature-contract bytes."""

    schema_version: str
    construction_version: str
    generated_stage: str
    columns: tuple[FeatureColumn, ...]
    source_artifact_sha256: tuple[tuple[str, str], ...]
    block_artifact_sha256: tuple[tuple[str, str], ...]
    feature_blocks: tuple[tuple[str, tuple[str, ...]], ...]
    allowed_overlaps: tuple[tuple[str, str], ...]
    mapping_unit: str
    comparators: tuple[tuple[str, tuple[str, ...], str, str], ...]
    sha256: str

    @property
    def block_columns(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(dict(self.feature_blocks))

    @property
    def comparator_blocks(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {name: blocks for name, blocks, _, _ in self.comparators}
        )


@dataclass(frozen=True)
class VerifiedBenchmarkLock:
    """Task 3 verification result bound to the live frozen configuration."""

    manifest: Mapping[str, object]
    manifest_sha256: str
    config: FrozenValidationConfig
    feature_contract: FeatureContract


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
    requested_replicates: int
    valid_replicates: int


@dataclass(frozen=True)
class PermutationTestResult:
    estimate_delta: float
    p_value: float
    null_distribution: tuple[float, ...]
    n_participants: int
    requested_replicates: int
    valid_replicates: int


@dataclass(frozen=True)
class BenchmarkResult:
    predictions: pd.DataFrame
    absolute_metrics: pd.DataFrame
    paired_metrics: pd.DataFrame
    fit_audit: pd.DataFrame
    splits: Mapping[str, tuple[NestedGroupSplit, ...]]
    split_audit: pd.DataFrame
    missingness_source: pd.DataFrame


def _read_regular_bytes(path: Path, label: str) -> bytes:
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise BenchmarkAccessBlocked(
                f"benchmark method-lock {label} is missing or is not a regular file"
            )
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as error:
        raise BenchmarkAccessBlocked(
            f"benchmark method-lock {label} is missing or unreadable"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_manifest(path: Path) -> tuple[bytes, dict[str, object]]:
    raw = _read_regular_bytes(path, "manifest")
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkAccessBlocked("benchmark method-lock manifest is invalid") from error
    if not isinstance(payload, dict):
        raise BenchmarkAccessBlocked("benchmark method-lock manifest must be an object")
    return raw, payload


def _json_without_duplicate_keys(raw: bytes, label: str) -> object:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _parse_feature_contract(raw: bytes, digest: str) -> FeatureContract:
    payload = _json_without_duplicate_keys(raw, "feature contract")
    if not isinstance(payload, dict) or raw != _canonical_json_bytes(payload):
        raise ValueError("feature contract must use canonical JSON bytes")
    required = {
        "schema_version",
        "construction_version",
        "generated_stage",
        "columns",
        "source_artifact_sha256",
        "block_artifact_sha256",
        "feature_blocks",
        "allowed_overlaps",
        "mapping_unit",
        "comparators",
    }
    if set(payload) != required:
        raise ValueError("feature contract top-level fields are not exact")
    if payload["schema_version"] != "person-meal-feature-contract-v1":
        raise ValueError("feature contract schema version is unsupported")
    if payload["generated_stage"] != "pre-outcome_predictor_only":
        raise ValueError("feature contract was not generated at the pre-outcome stage")
    if not isinstance(payload["construction_version"], str) or not payload[
        "construction_version"
    ]:
        raise ValueError("feature contract construction version is invalid")

    raw_columns = payload["columns"]
    if not isinstance(raw_columns, list) or not raw_columns:
        raise ValueError("feature contract columns must be a nonempty array")
    columns = []
    seen_columns: set[str] = set()
    for raw_column in raw_columns:
        if not isinstance(raw_column, dict) or set(raw_column) != {
            "name",
            "data_type",
            "role",
            "nullable",
            "source_artifact_id",
        }:
            raise ValueError("feature contract column fields are not exact")
        name = raw_column["name"]
        if not isinstance(name, str) or not name or name in seen_columns:
            raise ValueError("feature contract contains duplicate or invalid column names")
        seen_columns.add(name)
        if raw_column["data_type"] not in _DATA_TYPES:
            raise ValueError(f"feature contract column {name} has invalid data type")
        if raw_column["role"] not in _COLUMN_ROLES:
            raise ValueError(f"feature contract column {name} has forbidden role")
        if not isinstance(raw_column["nullable"], bool):
            raise ValueError(f"feature contract column {name} nullable flag is invalid")
        if not isinstance(raw_column["source_artifact_id"], str) or not raw_column[
            "source_artifact_id"
        ]:
            raise ValueError(f"feature contract column {name} source is invalid")
        columns.append(FeatureColumn(**raw_column))
    role_by_column = {column.name: column.role for column in columns}
    for name, role in _STRUCTURAL_ROLES.items():
        if role_by_column.get(name) != role:
            raise ValueError(f"feature contract structural column {name} has invalid role")

    def digest_map(field: str) -> dict[str, str]:
        value = payload[field]
        if not isinstance(value, dict) or not value:
            raise ValueError(f"feature contract {field} is invalid")
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(item, str)
                or _SHA256_PATTERN.fullmatch(item) is None
            ):
                raise ValueError(f"feature contract {field} contains invalid hash")
        return dict(value)

    source_hashes = digest_map("source_artifact_sha256")
    if set(source_hashes) != {column.source_artifact_id for column in columns}:
        raise ValueError("feature contract source artifact hashes are incomplete or extra")
    block_hashes = digest_map("block_artifact_sha256")

    raw_blocks = payload["feature_blocks"]
    if not isinstance(raw_blocks, dict) or tuple(sorted(raw_blocks)) != tuple(
        sorted(_FEATURE_BLOCKS)
    ):
        raise ValueError("feature contract feature block set is not exact")
    if set(block_hashes) != set(_FEATURE_BLOCKS):
        raise ValueError("feature contract block artifact hashes are incomplete or extra")
    blocks: dict[str, tuple[str, ...]] = {}
    for block in _FEATURE_BLOCKS:
        specification = raw_blocks[block]
        if not isinstance(specification, dict) or set(specification) != {
            "artifact_id",
            "columns",
        }:
            raise ValueError(f"feature contract block {block} fields are not exact")
        if specification["artifact_id"] != block:
            raise ValueError(f"feature contract block {block} artifact ID is invalid")
        names = specification["columns"]
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(f"feature contract block {block} columns are invalid")
        for name in names:
            if name not in role_by_column:
                raise ValueError(f"feature contract block {block} references unknown column")
            if role_by_column[name] != "predictor":
                raise ValueError(
                    f"feature contract block {block} uses nonpredictor role column {name}"
                )
        blocks[block] = tuple(names)

    raw_overlaps = payload["allowed_overlaps"]
    if not isinstance(raw_overlaps, list):
        raise ValueError("feature contract allowed overlaps must be an array")
    allowed_overlaps: set[tuple[str, str]] = set()
    for pair in raw_overlaps:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(value not in blocks for value in pair)
            or pair[0] >= pair[1]
        ):
            raise ValueError("feature contract allowed overlap entry is invalid")
        allowed_overlaps.add((pair[0], pair[1]))
    if len(allowed_overlaps) != len(raw_overlaps):
        raise ValueError("feature contract contains duplicate allowed overlaps")
    observed_overlaps: set[tuple[str, str]] = set()
    for left_index, left in enumerate(_FEATURE_BLOCKS):
        for right in _FEATURE_BLOCKS[left_index + 1 :]:
            if set(blocks[left]).intersection(blocks[right]):
                observed_overlaps.add(tuple(sorted((left, right))))
    if observed_overlaps != allowed_overlaps:
        raise ValueError("feature contract contains illegal or stale feature-block overlap")

    mapping = payload["mapping_unit"]
    if not isinstance(mapping, dict) or set(mapping) != {"column", "role"}:
        raise ValueError("feature contract mapping unit fields are not exact")
    if mapping["role"] != "mapping_unit" or role_by_column.get(mapping["column"]) != "mapping_unit":
        raise ValueError("feature contract mapping unit role is invalid")

    raw_comparators = payload["comparators"]
    if not isinstance(raw_comparators, dict) or set(raw_comparators) != set(
        REQUIRED_COMPARATORS
    ):
        raise ValueError("feature contract comparator set is not exact")
    comparators = []
    for name in REQUIRED_COMPARATORS:
        specification = raw_comparators[name]
        if not isinstance(specification, dict) or set(specification) != {
            "feature_blocks",
            "transform",
            "role",
        }:
            raise ValueError(f"feature contract comparator {name} fields are not exact")
        observed = (
            tuple(specification["feature_blocks"])
            if isinstance(specification["feature_blocks"], list)
            else (),
            specification["transform"],
            specification["role"],
        )
        if observed != _COMPARATOR_SEMANTICS[name]:
            raise ValueError(f"feature contract comparator {name} semantics changed")
        comparators.append((name, *observed))
    return FeatureContract(
        schema_version=str(payload["schema_version"]),
        construction_version=str(payload["construction_version"]),
        generated_stage=str(payload["generated_stage"]),
        columns=tuple(columns),
        source_artifact_sha256=tuple(sorted(source_hashes.items())),
        block_artifact_sha256=tuple(sorted(block_hashes.items())),
        feature_blocks=tuple((block, blocks[block]) for block in _FEATURE_BLOCKS),
        allowed_overlaps=tuple(sorted(allowed_overlaps)),
        mapping_unit=str(mapping["column"]),
        comparators=tuple(comparators),
        sha256=digest,
    )


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

    manifest_bytes, manifest = _read_manifest(Path(manifest_path))
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
    feature_contract_path = _path_from_lock(method_lock_paths, "feature_contract")
    config_bytes = _read_regular_bytes(config_path, "validation config")
    schema_bytes = _read_regular_bytes(schema_path, "schema")
    registry_bytes = _read_regular_bytes(registry_path, "release registry")
    gate_bytes = _read_regular_bytes(gate_path, "gate implementation")
    feature_contract_bytes = _read_regular_bytes(
        feature_contract_path,
        "feature contract",
    )

    live_hashes = {
        "person_meal_validation_config_sha256": sha256(config_bytes).hexdigest(),
        "method_lock_schema_sha256": sha256(schema_bytes).hexdigest(),
        "method_lock_gate_implementation_sha256": sha256(gate_bytes).hexdigest(),
        "feature_contract_sha256": sha256(feature_contract_bytes).hexdigest(),
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
    try:
        feature_contract = _parse_feature_contract(
            feature_contract_bytes,
            live_hashes["feature_contract_sha256"],
        )
    except (TypeError, ValueError) as error:
        raise BenchmarkAccessBlocked(
            "benchmark method-lock feature contract is invalid"
        ) from error
    return VerifiedBenchmarkLock(
        manifest=MappingProxyType(manifest),
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        config=config,
        feature_contract=feature_contract,
    )


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
) -> None:
    """Enforce the exact predictor table and role allowlist in trusted bytes."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("predictors must be a pandas DataFrame")
    if not isinstance(feature_contract, FeatureContract):
        raise TypeError("feature_contract must be a FeatureContract")
    expected = tuple(column.name for column in feature_contract.columns)
    observed = tuple(str(column) for column in frame.columns)
    if len(set(observed)) != len(observed):
        raise ValueError("predictor table contains duplicate column names")
    if set(observed) != set(expected):
        missing = sorted(set(expected).difference(observed))
        unknown = sorted(set(observed).difference(expected))
        raise ValueError(
            f"predictor table violates exact feature contract: missing={missing}, "
            f"unknown={unknown}"
        )
    for column in feature_contract.columns:
        values = frame[column.name]
        if not column.nullable and values.isna().any():
            raise ValueError(f"nonnullable contract column {column.name} contains missing values")
        if column.data_type == "number" and not is_numeric_dtype(values):
            raise ValueError(f"contract column {column.name} is not numeric")
        if column.data_type == "number":
            numeric = values.to_numpy(dtype=float)
            if np.isinf(numeric).any():
                raise ValueError(f"contract column {column.name} contains infinite values")


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
    if not all(np.isfinite(array).all() for array in numeric):
        raise ValueError("metric inputs must contain only finite values")
    return numeric


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
    numeric = _finite_vectors(*arrays)
    y = numeric[0]
    pred = numeric[1]
    reference = numeric[2] if reference_prediction is not None else None
    ids = identifiers.astype(str)
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
        requested_replicates=n_bootstrap,
        valid_replicates=len(finite_replicates),
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
    if identifiers.shape != original_shape:
        raise ValueError("permutation inputs must have the same shape")
    ids = identifiers.astype(str)
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
        requested_replicates=n_permutations,
        valid_replicates=int(np.isfinite(null).sum()),
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


def _predictor_frame_sha256(frame: pd.DataFrame) -> str:
    metadata = _canonical_json_bytes(
        {
            "columns": [str(column) for column in frame.columns],
            "dtypes": [str(dtype) for dtype in frame.dtypes],
            "n_rows": len(frame),
        }
    )
    row_hashes = pd.util.hash_pandas_object(
        frame,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    return sha256(metadata + row_hashes.tobytes()).hexdigest()


def _software_versions() -> str:
    return json.dumps(
        {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "scikit_learn": sklearn.__version__,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(values))
    if not len(finite_positions):
        return adjusted
    order = finite_positions[np.argsort(values[finite_positions], kind="mergesort")]
    running = 0.0
    total = len(order)
    for rank, position in enumerate(order):
        running = max(running, (total - rank) * float(values[position]))
        adjusted[position] = min(1.0, running)
    return adjusted


def _annotate_paired_tests(
    paired: pd.DataFrame,
    config: FrozenValidationConfig,
) -> pd.DataFrame:
    primary = config.payload.get("primary_tests")
    if not isinstance(primary, dict) or primary.get("multiplicity_method") != "holm":
        raise ValueError("frozen primary statistical-test contract is invalid")
    raw_tests = primary.get("tests")
    if not isinstance(raw_tests, list) or len(raw_tests) != 2:
        raise ValueError("frozen primary statistical-test family is invalid")
    primary_keys = {
        (
            test["endpoint"],
            test["analysis_mode"],
            test["model"],
            test["reference"],
            test["metric"],
        )
        for test in raw_tests
    }
    roles = []
    families = []
    methods = []
    null_comparators = set(config.payload["other_tests"]["null_comparators"])
    for row in paired.itertuples(index=False):
        key = (
            row.endpoint,
            row.analysis_mode,
            row.comparator,
            row.reference,
            row.metric,
        )
        if key in primary_keys:
            roles.append("primary")
            families.append(str(primary["correction_family"]))
            methods.append("holm")
        elif row.reference in null_comparators:
            roles.append("exploratory")
            families.append(
                f"exploratory_null::{row.analysis_mode}::{row.endpoint}::{row.metric}"
            )
            methods.append("holm")
        else:
            roles.append("secondary")
            families.append(
                f"secondary::{row.analysis_mode}::{row.endpoint}::{row.metric}"
            )
            methods.append("holm")
    paired = paired.copy()
    paired["test_role"] = roles
    paired["correction_family"] = families
    paired["multiplicity_method"] = methods
    paired["adjusted_p_value"] = np.nan
    for _, positions in paired.groupby("correction_family", sort=False).groups.items():
        index = np.asarray(list(positions), dtype=int)
        paired.loc[index, "adjusted_p_value"] = _holm_adjust(
            paired.loc[index, "permutation_p_value"].to_numpy(dtype=float)
        )
    observed_primary = {
        (
            row.endpoint,
            row.analysis_mode,
            row.comparator,
            row.reference,
            row.metric,
        )
        for row in paired.loc[paired["test_role"].eq("primary")].itertuples(index=False)
    }
    if observed_primary != primary_keys:
        raise ValueError(
            "benchmark output does not contain the complete preregistered primary family"
        )
    return paired


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
    blocks = contract.block_columns
    return {
        comparator: tuple(
            column
            for block in contract.comparator_blocks[comparator]
            for column in blocks[block]
        )
        for comparator in REQUIRED_COMPARATORS
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
        transformers.append(
            (
                "categorical_missing_indicator",
                MissingIndicator(features="all", error_on_new=False),
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
            ("model", Ridge(alpha=float(alpha), solver="lsqr")),
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
    if len(units) < 2:
        raise ValueError("shuffled mapping is not estimable with fewer than two units")
    rng = np.random.default_rng(seed)
    permuted_units = units.copy()
    for position in range(len(permuted_units) - 1, 0, -1):
        swap_position = int(rng.integers(0, position))
        permuted_units[position], permuted_units[swap_position] = (
            permuted_units[swap_position],
            permuted_units[position],
        )
    if np.any(permuted_units == units):
        raise RuntimeError("Sattolo derangement unexpectedly retained a mapping unit")
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
    train = (
        frame.iloc[train_positions]
        .loc[:, list(columns)]
        .copy()
        .replace({None: np.nan})
    )
    evaluation = (
        frame.iloc[evaluation_positions]
        .loc[:, list(columns)]
        .copy()
        .replace({None: np.nan})
    )
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


def _outcome_available(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.isfinite(numeric)


def _missingness_source_table(
    frame: pd.DataFrame,
    feature_contract: FeatureContract,
    endpoints: tuple[dict[str, object], ...],
    splits: Mapping[str, tuple[NestedGroupSplit, ...]],
) -> pd.DataFrame:
    predictor_columns = tuple(
        column.name for column in feature_contract.columns if column.role == "predictor"
    )
    endpoint_names = tuple(
        str(endpoint["name"])
        for endpoint in endpoints
        if str(endpoint["name"]) in frame.columns
    )
    records = []
    for analysis_mode, mode_splits in splits.items():
        for nested in mode_splits:
            split_positions = {
                "train": nested.outer.train_positions,
                "test": nested.outer.test_positions,
                "dropped": nested.outer.dropped_positions,
            }
            for split_name, positions in split_positions.items():
                if not positions:
                    continue
                rows = frame.iloc[list(positions)]
                for cohort_id, cohort_rows in rows.groupby("cohort_id", sort=True):
                    for endpoint_name in endpoint_names:
                        for variable in (*predictor_columns, endpoint_name):
                            values = cohort_rows[variable]
                            if variable == endpoint_name:
                                missing = ~_outcome_available(values)
                                variable_role = "outcome"
                            else:
                                missing = values.isna().to_numpy()
                                variable_role = "predictor"
                            n_rows = len(cohort_rows)
                            records.append(
                                {
                                    "endpoint": endpoint_name,
                                    "cohort": str(cohort_id),
                                    "analysis_mode": analysis_mode,
                                    "outer_fold": nested.outer.fold_id,
                                    "split": split_name,
                                    "variable": variable,
                                    "variable_role": variable_role,
                                    "n_rows": n_rows,
                                    "n_participants": int(
                                        cohort_rows["participant_id"].nunique()
                                    ),
                                    "missing_n": int(np.sum(missing)),
                                    "missing_rate": float(np.mean(missing)),
                                }
                            )
    return pd.DataFrame.from_records(records).sort_values(
        [
            "analysis_mode",
            "endpoint",
            "outer_fold",
            "split",
            "cohort",
            "variable_role",
            "variable",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


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


def _benchmark_predictions_for_mode(
    frame: pd.DataFrame,
    config: FrozenValidationConfig,
    feature_contract: FeatureContract,
    endpoints: tuple[dict[str, object], ...],
    *,
    analysis_mode: str,
    secondary_holdout: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[NestedGroupSplit, ...]]:
    split_config = config.payload["split"]
    seeds = config.payload["seeds"]
    splits = make_nested_group_splits(
        frame,
        outer_folds=int(split_config["outer_folds"]),
        inner_folds=int(split_config["inner_folds"]),
        outer_seed=int(seeds["outer_split"]),
        inner_seed=int(seeds["inner_cv"]),
        secondary_holdout=secondary_holdout,
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
                    train_available = _outcome_available(
                        frame.iloc[inner_train][endpoint_name]
                    )
                    validation_available = _outcome_available(
                        frame.iloc[inner_validation][endpoint_name]
                    )
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
                                "analysis_mode": analysis_mode,
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
                    _outcome_available(frame.iloc[outer_train][endpoint_name])
                ]
                outer_test = outer_test[
                    _outcome_available(frame.iloc[outer_test][endpoint_name])
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
                        "analysis_mode": analysis_mode,
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
                            "analysis_mode": analysis_mode,
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
        ["analysis_mode", "endpoint", "outer_fold", "__comparator_order__", "row_id"],
        kind="mergesort",
    ).drop(columns="__comparator_order__").reset_index(drop=True)
    audit = pd.DataFrame.from_records(audit_rows)
    audit["__comparator_order__"] = audit["comparator"].map(comparator_order)
    audit = audit.sort_values(
        [
            "endpoint",
            "analysis_mode",
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


def _benchmark_predictions(
    frame: pd.DataFrame,
    config: FrozenValidationConfig,
    feature_contract: FeatureContract,
    endpoints: tuple[dict[str, object], ...],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    Mapping[str, tuple[NestedGroupSplit, ...]],
    pd.DataFrame,
]:
    raw_modes = config.payload.get("analysis_modes")
    if not isinstance(raw_modes, list):
        raise ValueError("frozen validation config omits analysis modes")
    prediction_tables = []
    audit_tables = []
    split_plans: dict[str, tuple[NestedGroupSplit, ...]] = {}
    split_audit_rows = []
    expected_modes = (
        "subject_held_out",
        "subject_plus_food_held_out",
        "subject_plus_meal_held_out",
        "cohort_held_out",
    )
    if tuple(mode.get("name") for mode in raw_modes if isinstance(mode, dict)) != expected_modes:
        raise ValueError("frozen analysis mode names or order changed")
    for mode in raw_modes:
        mode_name = str(mode["name"])
        secondary_holdout = mode.get("secondary_unit")
        predictions, audit, splits = _benchmark_predictions_for_mode(
            frame,
            config,
            feature_contract,
            endpoints,
            analysis_mode=mode_name,
            secondary_holdout=(
                None if secondary_holdout is None else str(secondary_holdout)
            ),
        )
        prediction_tables.append(predictions)
        audit_tables.append(audit)
        split_plans[mode_name] = splits
        for nested in splits:
            outer = nested.outer
            dropped_reason = (
                "participant_family_twin_or_secondary_unit_fold_mismatch"
                if outer.dropped_positions
                else None
            )
            split_audit_rows.append(
                {
                    "analysis_mode": mode_name,
                    "analysis_role": str(mode["role"]),
                    "secondary_unit": secondary_holdout,
                    "outer_fold": outer.fold_id,
                    "train_n_rows": len(outer.train_positions),
                    "test_n_rows": len(outer.test_positions),
                    "dropped_n_rows": len(outer.dropped_positions),
                    "dropped_n_participants": int(
                        frame.iloc[list(outer.dropped_positions)]["participant_id"].nunique()
                        if outer.dropped_positions
                        else 0
                    ),
                    "dropped_n_person_meals": len(outer.dropped_positions),
                    "dropped_row_ids": tuple(
                        frame.iloc[list(outer.dropped_positions)]["row_id"].astype(str)
                    ),
                    "dropped_reason": dropped_reason,
                }
            )
    return (
        pd.concat(prediction_tables, ignore_index=True),
        pd.concat(audit_tables, ignore_index=True),
        MappingProxyType(split_plans),
        pd.DataFrame.from_records(split_audit_rows),
    )


def _metric_tables(
    predictions: pd.DataFrame,
    config: FrozenValidationConfig,
    endpoints: tuple[dict[str, object], ...],
    provenance: Mapping[str, object],
    feature_contract: FeatureContract,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = config.payload["seeds"]
    endpoint_by_name = {str(endpoint["name"]): endpoint for endpoint in endpoints}
    absolute_rows = []
    paired_rows = []
    contract_comparators = {
        name: {"blocks": blocks, "transform": transform, "role": role}
        for name, blocks, transform, role in feature_contract.comparators
    }
    block_hashes = dict(feature_contract.block_artifact_sha256)

    def opportunity(comparator: str) -> str:
        specification = contract_comparators[comparator]
        blocks = specification["blocks"]
        return json.dumps(
            {
                "blocks": list(blocks),
                "columns": [
                    column
                    for block in blocks
                    for column in feature_contract.block_columns[block]
                ],
                "block_artifact_sha256": {
                    block: block_hashes[block] for block in blocks
                },
                "transform": specification["transform"],
                "role": specification["role"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    for (analysis_mode, endpoint_name), endpoint_predictions in predictions.groupby(
        ["analysis_mode", "endpoint"], sort=False
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
                **provenance,
                "endpoint": endpoint_name,
                "analysis_mode": analysis_mode,
                "comparator": comparator,
                "comparator_information_opportunity": opportunity(comparator),
                "n_participants": int(comparator_frame["participant_id"].nunique()),
                "n_person_meals": int(comparator_frame["row_id"].nunique()),
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
                        "ci_method": "participant_cluster_percentile_bootstrap_95",
                        "bootstrap_replicates_requested": interval.requested_replicates,
                        "bootstrap_replicates_valid": interval.valid_replicates,
                        "permutation_replicates_requested": 0,
                        "permutation_replicates_valid": 0,
                        "test_role": "descriptive",
                        "correction_family": "none",
                        "multiplicity_method": "none",
                        "adjusted_p_value": math.nan,
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
                **provenance,
                "endpoint": endpoint_name,
                "analysis_mode": analysis_mode,
                "comparator": "locked_attribute_gmnps",
                "reference": reference_name,
                "comparator_information_opportunity": opportunity(
                    "locked_attribute_gmnps"
                ),
                "reference_information_opportunity": opportunity(reference_name),
                "n_participants": int(model_frame["participant_id"].nunique()),
                "n_person_meals": int(model_frame["row_id"].nunique()),
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
                        "ci_method": "paired_participant_cluster_percentile_bootstrap_95",
                        "bootstrap_replicates_requested": interval.requested_replicates,
                        "bootstrap_replicates_valid": interval.valid_replicates,
                        "permutation_replicates_requested": permutation.requested_replicates,
                        "permutation_replicates_valid": permutation.valid_replicates,
                        "permutation_p_value": permutation.p_value,
                        "permutation_null_mean": (
                            float(np.mean(finite_null))
                            if len(finite_null)
                            else math.nan
                        ),
                    }
                )
    absolute = pd.DataFrame.from_records(absolute_rows)
    paired = _annotate_paired_tests(pd.DataFrame.from_records(paired_rows), config)
    return absolute, paired


def run_person_meal_benchmark(
    source_id: str,
    *,
    manifest_path: Path,
    method_lock_paths: object,
    predictors: pd.DataFrame,
) -> BenchmarkResult:
    """Run every locked comparator after the fail-closed production gate."""

    locked = load_locked_outcomes_for_benchmark(
        source_id,
        manifest_path=manifest_path,
        method_lock_paths=method_lock_paths,
    )
    endpoint_specs = _endpoint_specs(locked.verified_lock.config)
    endpoint_names = tuple(str(endpoint["name"]) for endpoint in endpoint_specs)
    feature_contract = locked.verified_lock.feature_contract
    validate_feature_contract(
        predictors,
        feature_contract,
    )
    joined = _prepare_joined_frame(
        predictors,
        locked.outcome.frame,
        endpoint_names,
    )
    predictions, fit_audit, splits, split_audit = _benchmark_predictions(
        joined,
        locked.verified_lock.config,
        feature_contract,
        endpoint_specs,
    )
    missingness_source = _missingness_source_table(
        joined,
        feature_contract,
        endpoint_specs,
        splits,
    )
    provenance = {
        "manifest_sha256": locked.verified_lock.manifest_sha256,
        "outcome_source_id": locked.outcome.source_id,
        "outcome_source_sha256": locked.outcome.sha256,
        "predictor_frame_sha256": _predictor_frame_sha256(predictors),
        "feature_contract_sha256": feature_contract.sha256,
        "predictor_source_artifact_sha256": json.dumps(
            dict(feature_contract.source_artifact_sha256),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "software_versions": _software_versions(),
    }
    absolute_metrics, paired_metrics = _metric_tables(
        predictions,
        locked.verified_lock.config,
        endpoint_specs,
        provenance,
        feature_contract,
    )
    return BenchmarkResult(
        predictions=predictions,
        absolute_metrics=absolute_metrics,
        paired_metrics=paired_metrics,
        fit_audit=fit_audit,
        splits=splits,
        split_audit=split_audit,
        missingness_source=missingness_source,
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
