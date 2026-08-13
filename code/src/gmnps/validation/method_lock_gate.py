"""Fail-closed pre-outcome contract for a future method-lock run instance.

This module hashes and validates only explicitly declared predictor, method,
configuration, and trusted-registry artifacts.  It has no outcome/label path,
performs no outcome loading, and writes a run-level manifest only through the
atomic no-overwrite writer after every predictor-side proof passes.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence

import pandas as pd

from gmnps.scoring.attribute_calibration import fit_beta_normalization
from gmnps.scoring.attribute_gmnps import (
    FCS2_FNDDS_REGISTRY_DIGEST_ALGORITHM,
    FCS2_FNDDS_REGISTRY_SCHEMA_VERSION,
    FCS2_FNDDS_REGISTRY_VERSION,
    load_release_registry_snapshot,
)


_CANONICAL_BETA_SCHEMA_VERSION = "gmnps-canonical-beta-v1"
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
_LOCKED_IMPLEMENTATION_PATHS = frozenset(
    {
        "code/src/gmnps/scoring/fcs2_attribute_rules.py",
        "code/src/gmnps/scoring/fcs2_attribute_mapping.py",
        "code/src/gmnps/scoring/attribute_calibration.py",
        "code/src/gmnps/scoring/attribute_recomposition.py",
        "code/src/gmnps/scoring/attribute_gmnps.py",
        "code/src/configs/attribute_gmnps.yaml",
        "code/src/scripts/run_attribute_gmnps.py",
    }
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
    feature_contract: Path
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
    values: tuple[tuple[float, ...], ...]
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
        values=tuple(tuple(float(value) for value in row) for row in values),
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
    development_frame = pd.DataFrame(
        development_beta.values,
        index=list(development_beta.participant_ids),
        columns=list(development_beta.nutrient_order),
        dtype=float,
    )
    try:
        fitted = fit_beta_normalization(development_frame)
    except (TypeError, ValueError) as error:
        raise MethodLockError(
            "normalization state could not be fitted from canonical development beta"
        ) from error
    expected = {
        "nutrient_order": list(fitted.nutrient_order),
        "median_values": list(fitted.median_values),
        "scale_values": list(fitted.scale_values),
        "scale_method_values": list(fitted.scale_method_values),
        "fit_n": fitted.fit_n,
        "fit_id_sha256": fitted.fit_id_sha256,
        "method_version": fitted.method_version,
        "temperature": fitted.temperature,
        "state_fingerprint": fitted.state_fingerprint,
    }
    labels = {
        "nutrient_order": "nutrient order",
        "median_values": "median values",
        "scale_values": "scale values",
        "scale_method_values": "scale source values",
        "fit_n": "fit_n",
        "fit_id_sha256": "fit-ID hash",
        "method_version": "method version",
        "temperature": "temperature",
        "state_fingerprint": "fingerprint",
    }
    for field in _NORMALIZATION_FIELDS:
        if payload[field] != expected[field]:
            raise MethodLockError(
                f"normalization-state {labels[field]} was not derived from "
                "canonical development beta"
            )
    return expected


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
    all_of = schema.get("allOf", [])
    if not isinstance(all_of, list):
        raise MethodLockError(f"schema allOf definition is invalid at {path}")
    for index, child in enumerate(all_of):
        if not isinstance(child, dict):
            raise MethodLockError(f"schema allOf entry is invalid at {path}[{index}]")
        merged = dict(child)
        if expected_type is not None and "type" not in merged:
            merged["type"] = expected_type
        _validate_schema_value(value, merged, f"{path}.allOf[{index}]")
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


def _verify_locked_implementation_sources(
    schema: Mapping[str, object],
) -> dict[str, str]:
    declared = _schema_const(schema, "implementation_source_sha256")
    if not isinstance(declared, dict):
        raise MethodLockError("implementation hash lock must be an object")
    supplied_paths = set(declared)
    missing = sorted(_LOCKED_IMPLEMENTATION_PATHS - supplied_paths)
    extra = sorted(supplied_paths - _LOCKED_IMPLEMENTATION_PATHS)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise MethodLockError(
            "implementation source path set mismatch: " + ", ".join(details)
        )
    try:
        repository_root = _REPOSITORY_ROOT.resolve(strict=True)
    except OSError as error:
        raise MethodLockError("implementation repository root is missing") from error

    verified: dict[str, str] = {}
    for relative_path, expected_digest in declared.items():
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or not isinstance(expected_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
        ):
            raise MethodLockError("implementation source hash declaration is invalid")
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise MethodLockError("implementation source path escapes the repository root")
        candidate = _REPOSITORY_ROOT / relative
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(repository_root)
        except (OSError, ValueError) as error:
            raise MethodLockError(
                f"implementation source {relative_path} is missing or unreadable"
            ) from error
        actual_digest = sha256(
            _read_bytes(resolved, f"implementation source {relative_path}")
        ).hexdigest()
        if actual_digest != expected_digest:
            raise MethodLockError(
                f"implementation source hash mismatch for {relative_path}"
            )
        verified[relative_path] = actual_digest
    return verified


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


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise MethodLockError(f"{label} must be a lowercase SHA-256 digest")
    return value


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
    expected_release_registry_sha256: str,
) -> dict[str, object]:
    if not isinstance(paths, MethodLockArtifactPaths):
        raise TypeError("paths must be MethodLockArtifactPaths")
    if not isinstance(beta_fit_cohort_id, str) or not beta_fit_cohort_id.strip():
        raise MethodLockError("beta_fit_cohort_id must be a nonempty string")
    expected_registry_sha256 = _require_sha256(
        expected_release_registry_sha256,
        "expected release-registry snapshot",
    )

    schema_bytes, schema = _load_schema(paths.method_lock_schema)
    implementation_source_sha256 = _verify_locked_implementation_sources(schema)
    config_bytes = _read_bytes(
        paths.person_meal_validation_config,
        "person-meal validation config",
    )
    feature_contract_bytes = _read_bytes(
        paths.feature_contract,
        "feature-contract artifact",
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
    if registry.snapshot_sha256 != expected_registry_sha256:
        raise MethodLockError(
            "trusted release-registry snapshot does not match the explicitly "
            "expected pre-label SHA-256"
        )
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
        "feature_contract_sha256": sha256(feature_contract_bytes).hexdigest(),
        "mapping_version": _schema_const(schema, "mapping_version"),
        "implementation_source_sha256": implementation_source_sha256,
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
    expected_release_registry_sha256: str,
) -> dict[str, object]:
    """Build but do not write a real manifest from future explicit artifacts."""

    return _expected_manifest(
        paths,
        beta_fit_cohort_id=beta_fit_cohort_id,
        expected_release_registry_sha256=expected_release_registry_sha256,
    )


def validate_method_lock_manifest(
    manifest: Mapping[str, object],
    paths: MethodLockArtifactPaths,
    *,
    expected_release_registry_sha256: str,
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
        expected_release_registry_sha256=expected_release_registry_sha256,
    )
    targeted = (
        ("method_lock_schema_sha256", "method-lock schema hash"),
        ("person_meal_validation_config_sha256", "validation config hash"),
        ("feature_contract_sha256", "feature-contract hash"),
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


def write_method_lock_manifest(
    paths: MethodLockArtifactPaths,
    *,
    output_path: Path,
    beta_fit_cohort_id: str,
    expected_release_registry_sha256: str,
    expected_person_meal_validation_config_sha256: str,
) -> dict[str, object]:
    """Atomically write a fully generated and revalidated real run manifest.

    Parent directories are created only after all predictor-side proofs pass.
    Existing output is never overwritten, and no output is left on failure.
    """

    if not isinstance(output_path, (str, Path)):
        raise TypeError("output_path must be a filesystem path")
    destination = Path(output_path)
    if destination.exists():
        raise MethodLockError("method-lock manifest output already exists")
    manifest = generate_method_lock_manifest(
        paths,
        beta_fit_cohort_id=beta_fit_cohort_id,
        expected_release_registry_sha256=expected_release_registry_sha256,
    )
    expected_config_sha256 = _require_sha256(
        expected_person_meal_validation_config_sha256,
        "expected person-meal validation config",
    )
    if manifest["person_meal_validation_config_sha256"] != expected_config_sha256:
        raise MethodLockError(
            "person-meal validation config does not match the explicitly frozen SHA-256"
        )
    validate_method_lock_manifest(
        manifest,
        paths,
        expected_release_registry_sha256=expected_release_registry_sha256,
    )

    encoded = (
        json.dumps(
            manifest,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    published = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        try:
            os.link(temporary_path, destination)
            published = True
        except FileExistsError as error:
            raise MethodLockError(
                "method-lock manifest output appeared during write"
            ) from error
        if destination.read_bytes() != encoded:
            raise OSError("published method-lock manifest failed byte validation")
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        temporary_path.unlink()
        temporary_path = None
    except Exception as error:
        if published:
            try:
                destination.unlink(missing_ok=True)
                directory_fd = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        if isinstance(error, MethodLockError):
            raise
        raise MethodLockError(
            "method-lock manifest could not be written atomically"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return manifest


__all__ = [
    "MethodLockArtifactPaths",
    "MethodLockError",
    "canonical_ids_sha256",
    "canonical_json_sha256",
    "generate_method_lock_manifest",
    "validate_method_lock_manifest",
    "write_method_lock_manifest",
]
