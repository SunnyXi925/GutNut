"""Fail-closed pre-outcome contract for a future method-lock run instance.

This module hashes and validates only explicitly declared predictor, method,
configuration, and trusted-registry artifacts.  It has no outcome/label path,
performs no outcome loading, and does not write a run-level manifest.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from typing import Mapping, Sequence

from gmnps.scoring.attribute_gmnps import (
    FCS2_FNDDS_REGISTRY_DIGEST_ALGORITHM,
    FCS2_FNDDS_REGISTRY_SCHEMA_VERSION,
    FCS2_FNDDS_REGISTRY_VERSION,
    load_release_registry_snapshot,
)


_CANONICAL_BETA_SCHEMA_VERSION = "gmnps-canonical-beta-v1"
_NORMALIZATION_METHOD_VERSION = "median_mad_iqr_sd_v1"
_LOCKED_BETA_TEMPERATURE = 2.0
_PRODUCTION_ARTIFACT_KEYS = (
    "official_fcs",
    "food_metadata",
    "baseline_attribute_points",
    "food_exposures",
    "effective_attribute_weights",
)
_NORMALIZATION_FIELDS = {
    "nutrient_order",
    "median_values",
    "scale_values",
    "scale_method_values",
    "fit_n",
    "fit_id_sha256",
    "method_version",
    "temperature",
    "state_fingerprint",
}
_ALLOWED_SCALE_METHODS = frozenset(
    {"mad", "iqr", "standard_deviation", "unit"}
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_TRUSTED_METHOD_LOCK_SCHEMA_PATH = (
    _REPOSITORY_ROOT / "docs/methods/method_lock_manifest.schema.json"
)
_TRUSTED_RELEASE_REGISTRY_PATH = (
    _REPOSITORY_ROOT / "code/src/configs/fcs2_fndds_release_registry.json"
)


class MethodLockError(ValueError):
    """Raised whenever the method-lock contract cannot be proven."""


@dataclass(frozen=True)
class MethodLockArtifactPaths:
    """Explicit, outcome-free paths required to materialize a future lock."""

    method_lock_schema: Path
    person_meal_validation_config: Path
    release_registry: Path
    gate_implementation: Path
    development_beta: Path
    normalization_state: Path
    scoring_beta: Path
    official_fcs: Path
    food_metadata: Path
    baseline_attribute_points: Path
    food_exposures: Path
    effective_attribute_weights: Path
    input_manifest: Path
    food_source_linkage: Path

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, (str, Path)):
                raise TypeError(f"{field.name} must be a filesystem path")
            object.__setattr__(self, field.name, Path(value))


@dataclass(frozen=True)
class _CanonicalBeta:
    participant_ids: tuple[str, ...]
    nutrient_order: tuple[str, ...]
    canonical_sha256: str


def canonical_json_sha256(payload: object) -> str:
    """Return SHA-256 over canonical JSON (sorted, compact, finite)."""

    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MethodLockError("payload cannot be represented as canonical JSON") from error
    return sha256(encoded).hexdigest()


def canonical_ids_sha256(identifiers: Sequence[str]) -> str:
    """Match the Phase 1 sorted-newline participant-ID digest contract."""

    if isinstance(identifiers, (str, bytes)):
        raise MethodLockError("participant IDs must be a sequence of strings")
    materialized = tuple(identifiers)
    if not materialized:
        raise MethodLockError("participant IDs must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in materialized):
        raise MethodLockError("participant IDs must be nonempty strings")
    if len(set(materialized)) != len(materialized):
        raise MethodLockError("participant IDs contain duplicate values")
    return sha256("\n".join(sorted(materialized)).encode("utf-8")).hexdigest()


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        if not path.is_file():
            raise MethodLockError(f"{label} is missing or is not a regular file")
        return path.read_bytes()
    except OSError as error:
        raise MethodLockError(f"{label} is missing or unreadable") from error


def _read_json(path: Path, label: str) -> tuple[bytes, object]:
    raw = _read_bytes(path, label)
    try:
        return raw, json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MethodLockError(f"{label} is not valid JSON") from error


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(float(value))
    )


def _load_canonical_beta(path: Path, label: str) -> _CanonicalBeta:
    _, payload = _read_json(path, label)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "participant_ids",
        "nutrient_order",
        "values",
    }:
        raise MethodLockError(f"{label} does not satisfy the canonical beta schema")
    if payload["schema_version"] != _CANONICAL_BETA_SCHEMA_VERSION:
        raise MethodLockError(f"{label} canonical beta schema version is unsupported")
    identifiers = payload["participant_ids"]
    nutrients = payload["nutrient_order"]
    values = payload["values"]
    if not isinstance(identifiers, list):
        raise MethodLockError(f"{label} participant IDs must be a JSON array")
    canonical_ids_sha256(identifiers)
    if (
        not isinstance(nutrients, list)
        or not nutrients
        or any(not isinstance(value, str) or not value for value in nutrients)
        or len(set(nutrients)) != len(nutrients)
    ):
        raise MethodLockError(f"{label} nutrient order must contain unique strings")
    if not isinstance(values, list) or len(values) != len(identifiers):
        raise MethodLockError(f"{label} values do not match participant IDs")
    for row in values:
        if (
            not isinstance(row, list)
            or len(row) != len(nutrients)
            or any(not _is_finite_number(value) for value in row)
        ):
            raise MethodLockError(f"{label} values must be a finite rectangular matrix")
    return _CanonicalBeta(
        participant_ids=tuple(identifiers),
        nutrient_order=tuple(nutrients),
        canonical_sha256=canonical_json_sha256(payload),
    )


def _load_normalization_state(
    path: Path,
    *,
    development_beta: _CanonicalBeta,
) -> dict[str, object]:
    _, payload = _read_json(path, "normalization-state artifact")
    if not isinstance(payload, dict) or set(payload) != _NORMALIZATION_FIELDS:
        raise MethodLockError("normalization-state artifact schema is invalid")
    nutrients = payload["nutrient_order"]
    medians = payload["median_values"]
    scales = payload["scale_values"]
    methods = payload["scale_method_values"]
    if not all(isinstance(value, list) for value in (nutrients, medians, scales, methods)):
        raise MethodLockError("normalization-state vectors must be JSON arrays")
    if tuple(nutrients) != development_beta.nutrient_order:
        raise MethodLockError("normalization-state nutrient order does not match development beta")
    if not (len(nutrients) == len(medians) == len(scales) == len(methods)):
        raise MethodLockError("normalization-state vectors have inconsistent lengths")
    if any(not _is_finite_number(value) for value in medians):
        raise MethodLockError("normalization-state medians must be finite")
    if any(not _is_finite_number(value) or float(value) <= 0 for value in scales):
        raise MethodLockError("normalization-state scales must be finite and positive")
    if any(value not in _ALLOWED_SCALE_METHODS for value in methods):
        raise MethodLockError("normalization-state scale method is unsupported")
    if payload["method_version"] != _NORMALIZATION_METHOD_VERSION:
        raise MethodLockError("normalization-state method version is not locked")
    if payload["temperature"] != _LOCKED_BETA_TEMPERATURE:
        raise MethodLockError("normalization-state temperature is not locked")
    fit_n = payload["fit_n"]
    if isinstance(fit_n, bool) or not isinstance(fit_n, int) or fit_n <= 0:
        raise MethodLockError("normalization-state fit_n must be a positive integer")
    if fit_n != len(development_beta.participant_ids):
        raise MethodLockError("normalization-state fit_n does not match development beta")
    expected_fit_hash = canonical_ids_sha256(development_beta.participant_ids)
    if payload["fit_id_sha256"] != expected_fit_hash:
        raise MethodLockError("normalization-state fit-ID hash does not match development beta")
    state_without_fingerprint = {
        "fit_id_sha256": payload["fit_id_sha256"],
        "fit_n": fit_n,
        "median_values": [float(value) for value in medians],
        "method_version": payload["method_version"],
        "nutrient_order": list(nutrients),
        "scale_method_values": list(methods),
        "scale_values": [float(value) for value in scales],
        "temperature": float(payload["temperature"]),
    }
    if payload["state_fingerprint"] != canonical_json_sha256(state_without_fingerprint):
        raise MethodLockError("normalization-state fingerprint does not match its contents")
    return payload


def _load_schema(path: Path) -> tuple[bytes, dict[str, object]]:
    _require_trusted_path(
        path,
        trusted_path=_TRUSTED_METHOD_LOCK_SCHEMA_PATH,
        label="method-lock schema",
    )
    raw, schema = _read_json(path, "method-lock schema")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise MethodLockError("method-lock schema root is invalid")
    if not isinstance(schema.get("properties"), dict):
        raise MethodLockError("method-lock schema properties are invalid")
    return raw, schema


def _validate_schema_value(value: object, schema: Mapping[str, object], path: str) -> None:
    if "const" in schema and value != schema["const"]:
        raise MethodLockError(f"schema const mismatch at {path}")
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            raise MethodLockError(f"schema type mismatch at {path}: expected object")
        required = schema.get("required", [])
        if not isinstance(required, list):
            raise MethodLockError(f"schema required definition is invalid at {path}")
        missing = [key for key in required if key not in value]
        if missing:
            raise MethodLockError(
                f"schema required property missing at {path}: {', '.join(missing)}"
            )
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise MethodLockError(f"schema properties definition is invalid at {path}")
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                child = properties[key]
                if not isinstance(child, dict):
                    raise MethodLockError(f"schema property definition is invalid at {path}.{key}")
                _validate_schema_value(item, child, f"{path}.{key}")
            elif additional is False:
                raise MethodLockError(f"schema additional property at {path}.{key}")
            elif isinstance(additional, dict):
                _validate_schema_value(item, additional, f"{path}.{key}")
        minimum_properties = schema.get("minProperties")
        if isinstance(minimum_properties, int) and len(value) < minimum_properties:
            raise MethodLockError(f"schema minProperties failure at {path}")
    elif expected_type == "array":
        if not isinstance(value, list):
            raise MethodLockError(f"schema type mismatch at {path}: expected array")
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _validate_schema_value(item, items, f"{path}[{index}]")
    elif expected_type == "string":
        if not isinstance(value, str):
            raise MethodLockError(f"schema type mismatch at {path}: expected string")
        minimum_length = schema.get("minLength")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            raise MethodLockError(f"schema minLength failure at {path}")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.fullmatch(pattern, value) is None:
            raise MethodLockError(f"schema pattern mismatch at {path}")
    elif expected_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise MethodLockError(f"schema type mismatch at {path}: expected integer")
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            raise MethodLockError(f"schema minimum failure at {path}")
    elif expected_type == "number":
        if not _is_finite_number(value):
            raise MethodLockError(f"schema type mismatch at {path}: expected number")
    elif expected_type == "boolean" and not isinstance(value, bool):
        raise MethodLockError(f"schema type mismatch at {path}: expected boolean")


def _validate_against_schema(manifest: object, schema: Mapping[str, object]) -> None:
    _validate_schema_value(manifest, schema, "manifest")


def _schema_const(schema: Mapping[str, object], field: str) -> object:
    properties = schema["properties"]
    if not isinstance(properties, dict):
        raise MethodLockError("method-lock schema properties are invalid")
    definition = properties.get(field)
    if not isinstance(definition, dict) or "const" not in definition:
        raise MethodLockError(f"method-lock schema does not freeze {field}")
    return definition["const"]


def _require_installed_gate(path: Path) -> bytes:
    installed = Path(__file__).resolve()
    try:
        supplied = path.resolve(strict=True)
    except OSError as error:
        raise MethodLockError("gate implementation is missing or unreadable") from error
    if supplied != installed:
        raise MethodLockError("gate implementation path is not the installed method_lock_gate.py")
    return _read_bytes(supplied, "gate implementation")


def _require_trusted_path(
    path: Path,
    *,
    trusted_path: Path,
    label: str,
) -> None:
    try:
        supplied = path.resolve(strict=True)
        trusted = trusted_path.resolve(strict=True)
    except OSError as error:
        raise MethodLockError(f"trusted {label} is missing or unreadable") from error
    if supplied != trusted:
        raise MethodLockError(f"{label} path is not the repository trust root")


def _artifact_sha256(path: Path, label: str) -> str:
    return sha256(_read_bytes(path, label)).hexdigest()


def _matching_approved_entry(
    approved_entries: Sequence[Mapping[str, object]],
    *,
    artifact_hashes: Mapping[str, str],
    linkage_sha256: str,
) -> dict[str, object]:
    matches = [
        entry
        for entry in approved_entries
        if entry.get("artifact_sha256") == dict(artifact_hashes)
        and entry.get("food_source_linkage_sha256") == linkage_sha256
    ]
    if len(matches) != 1:
        raise MethodLockError(
            "trusted release registry must contain exactly one matching approved entry"
        )
    return dict(matches[0])


def _expected_manifest(
    paths: MethodLockArtifactPaths,
    *,
    beta_fit_cohort_id: str,
) -> dict[str, object]:
    if not isinstance(paths, MethodLockArtifactPaths):
        raise TypeError("paths must be MethodLockArtifactPaths")
    if not isinstance(beta_fit_cohort_id, str) or not beta_fit_cohort_id.strip():
        raise MethodLockError("beta_fit_cohort_id must be a nonempty string")

    schema_bytes, schema = _load_schema(paths.method_lock_schema)
    config_bytes = _read_bytes(
        paths.person_meal_validation_config,
        "person-meal validation config",
    )
    _require_trusted_path(
        paths.release_registry,
        trusted_path=_TRUSTED_RELEASE_REGISTRY_PATH,
        label="release registry",
    )
    registry_bytes = _read_bytes(paths.release_registry, "trusted release registry")
    try:
        registry = load_release_registry_snapshot(registry_bytes)
    except (TypeError, ValueError) as error:
        raise MethodLockError("trusted release registry contract is invalid") from error
    if registry.schema_version != FCS2_FNDDS_REGISTRY_SCHEMA_VERSION:
        raise MethodLockError("trusted registry schema version is not constant")
    if registry.registry_version != FCS2_FNDDS_REGISTRY_VERSION:
        raise MethodLockError("trusted registry version is not constant")
    if registry.digest_algorithm != FCS2_FNDDS_REGISTRY_DIGEST_ALGORITHM:
        raise MethodLockError("trusted registry digest algorithm is not sha256")
    if not registry.approved_artifacts:
        raise MethodLockError("trusted release registry has no approved entry")
    gate_bytes = _require_installed_gate(paths.gate_implementation)

    development_beta = _load_canonical_beta(paths.development_beta, "development beta")
    scoring_beta = _load_canonical_beta(paths.scoring_beta, "scoring beta")
    if development_beta.nutrient_order != scoring_beta.nutrient_order:
        raise MethodLockError("development and scoring beta nutrient orders differ")
    overlap = set(development_beta.participant_ids) & set(scoring_beta.participant_ids)
    if overlap:
        raise MethodLockError("development fit and held-out scoring participant IDs overlap")
    normalization = _load_normalization_state(
        paths.normalization_state,
        development_beta=development_beta,
    )

    production_paths = {
        key: getattr(paths, key) for key in _PRODUCTION_ARTIFACT_KEYS
    }
    production_hashes = {
        key: _artifact_sha256(path, f"{key} artifact")
        for key, path in production_paths.items()
    }
    linkage_sha256 = _artifact_sha256(
        paths.food_source_linkage,
        "food-source linkage artifact",
    )
    approved_entry = _matching_approved_entry(
        registry.approved_artifacts,
        artifact_hashes=production_hashes,
        linkage_sha256=linkage_sha256,
    )
    approved_with_digest = {
        **approved_entry,
        "entry_sha256": canonical_json_sha256(approved_entry),
    }

    manifest = {
        "manifest_schema_version": _schema_const(schema, "manifest_schema_version"),
        "method_lock_schema_sha256": sha256(schema_bytes).hexdigest(),
        "method_version": _schema_const(schema, "method_version"),
        "source_hashes": {
            **production_hashes,
            "input_manifest": _artifact_sha256(
                paths.input_manifest,
                "food-bundle input manifest",
            ),
            "food_source_linkage": linkage_sha256,
        },
        "fndds_releases": list(registry.canonical_release_set),
        "fndds_registry_version": registry.registry_version,
        "release_registry_snapshot": {
            "snapshot_sha256": registry.snapshot_sha256,
            "schema_version": registry.schema_version,
            "registry_version": registry.registry_version,
            "digest_algorithm": registry.digest_algorithm,
            "canonical_release_set": list(registry.canonical_release_set),
            "approved_entry": approved_with_digest,
        },
        "beta_fit_cohort": {
            "cohort_id": beta_fit_cohort_id,
            "fit_n": len(development_beta.participant_ids),
        },
        "calibration_fit_ids_sha256": canonical_ids_sha256(
            development_beta.participant_ids
        ),
        "development_beta_sha256": development_beta.canonical_sha256,
        "normalization_state_fingerprint": normalization["state_fingerprint"],
        "scoring_beta_sha256": scoring_beta.canonical_sha256,
        "person_meal_validation_config_sha256": sha256(config_bytes).hexdigest(),
        "mapping_version": _schema_const(schema, "mapping_version"),
        "implementation_source_sha256": _schema_const(
            schema,
            "implementation_source_sha256",
        ),
        "method_lock_gate_implementation_sha256": sha256(gate_bytes).hexdigest(),
        "fixed_parameters": _schema_const(schema, "fixed_parameters"),
        "validation_embargo": _schema_const(schema, "validation_embargo"),
    }
    _validate_against_schema(manifest, schema)
    return manifest


def generate_method_lock_manifest(
    paths: MethodLockArtifactPaths,
    *,
    beta_fit_cohort_id: str,
) -> dict[str, object]:
    """Build but do not write a real manifest from future explicit artifacts."""

    return _expected_manifest(paths, beta_fit_cohort_id=beta_fit_cohort_id)


def validate_method_lock_manifest(
    manifest: Mapping[str, object],
    paths: MethodLockArtifactPaths,
) -> None:
    """Recompute every bound proof and fail closed on any discrepancy."""

    if not isinstance(manifest, Mapping):
        raise MethodLockError("method-lock manifest must be an object")
    _, schema = _load_schema(paths.method_lock_schema)
    materialized = dict(manifest)
    _validate_against_schema(materialized, schema)
    cohort = materialized.get("beta_fit_cohort")
    if not isinstance(cohort, dict) or not isinstance(cohort.get("cohort_id"), str):
        raise MethodLockError("method-lock beta fit cohort is invalid")
    expected = _expected_manifest(
        paths,
        beta_fit_cohort_id=cohort["cohort_id"],
    )
    targeted = (
        ("method_lock_schema_sha256", "method-lock schema hash"),
        ("person_meal_validation_config_sha256", "validation config hash"),
        (
            "method_lock_gate_implementation_sha256",
            "gate implementation hash",
        ),
        ("development_beta_sha256", "development beta hash"),
        ("scoring_beta_sha256", "scoring beta hash"),
        ("normalization_state_fingerprint", "normalization-state fingerprint"),
    )
    for field, label in targeted:
        if materialized.get(field) != expected[field]:
            raise MethodLockError(f"{label} does not match the current artifact")
    if materialized.get("release_registry_snapshot") != expected[
        "release_registry_snapshot"
    ]:
        raise MethodLockError("release-registry snapshot or approved entry does not match")
    if materialized != expected:
        raise MethodLockError("method-lock manifest content does not match bound artifacts")


__all__ = [
    "MethodLockArtifactPaths",
    "MethodLockError",
    "canonical_ids_sha256",
    "canonical_json_sha256",
    "generate_method_lock_manifest",
    "validate_method_lock_manifest",
]
