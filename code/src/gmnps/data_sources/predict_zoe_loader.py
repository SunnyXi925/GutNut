"""Fail-closed staged loading for PREDICT/ZOE predictor and outcome resources.

Predictor resources are classified and checksum-bound before parsing. Outcome
resources use a separate entry point that rejects synthetic, aggregate, and
unauthorized controlled data before touching their bytes. A real outcome table
can be opened only after the complete method lock and frozen configuration have
been revalidated against the current repository trust roots.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Mapping

import pandas as pd

from gmnps.data_sources.predict_zoe_registry import PREDICT_ZOE_SOURCE_REGISTRY
from gmnps.validation.method_lock_gate import validate_method_lock_manifest


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_PREDICTOR_CONTENT_CLASS = "predictor_only"
_DIRECT_VALIDATION_STATUSES = frozenset({"controlled_real", "real_individual_level"})
_DIRECT_VALIDATION_ACCESS = frozenset({"controlled_granted", "public"})
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_OUTCOME_COLUMN_NAMES = frozenset(
    {
        "outcome",
        "response",
        "label",
        "target",
        "glucose_iAUC_2h",
        "tg_6h_rise",
        "c_peptide_iAUC_2h",
    }
)

_LOCKED_PERSON_MEAL_CONFIG = {
    "schema_version": "person-meal-validation-v1",
    "frozen_on": "2026-08-13",
    "config_role": "pre-outcome-preregistration",
    "validation_embargo": True,
    "endpoints": [
        {
            "name": "glucose_iAUC_2h",
            "role": "primary",
            "unit": "mmol_L_hour",
            "window_hours": [0, 2],
            "summary": "incremental_area_under_curve",
        },
        {
            "name": "tg_6h_rise",
            "role": "primary",
            "unit": "mmol_L",
            "window_hours": [0, 6],
            "summary": "rise_above_baseline",
        },
        {
            "name": "c_peptide_iAUC_2h",
            "role": "secondary",
            "unit": "nmol_L_hour",
            "window_hours": [0, 2],
            "summary": "incremental_area_under_curve",
        },
    ],
    "missingness": {
        "analysis_set": "endpoint_specific_available_case_after_subject_split",
        "outcome_imputation": "forbidden",
        "predictor_imputation": "median_or_mode_with_missing_indicator",
        "predictor_fit_scope": "development_fold_only",
        "report_missingness_by": ["endpoint", "cohort", "split"],
    },
    "split": {
        "primary_unit": "participant",
        "family_twin_grouping": True,
        "outer_folds": 5,
        "inner_folds": 5,
        "secondary_generalization": ["meal_or_food_held_out", "cohort_held_out"],
        "development_scoring_overlap": "forbidden",
    },
    "seeds": {
        "outer_split": 1729,
        "inner_cv": 2718,
        "bootstrap": 31415,
        "permutation": 16180,
    },
}


class PredictorLoadError(ValueError):
    """Raised when immutable predictor/configuration evidence cannot be proven."""


class OutcomeAccessBlocked(PermissionError):
    """Raised before opening outcome bytes when the staged gate is not satisfied."""


@dataclass(frozen=True)
class PredictorTableSpec:
    """Trusted description of one exact predictor-only table."""

    path: Path
    expected_sha256: str
    file_format: str
    source_id: str
    content_class: str
    required_columns: tuple[str, ...]
    allowed_columns: tuple[str, ...]
    unique_key: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        _validate_table_spec_fields(
            expected_sha256=self.expected_sha256,
            file_format=self.file_format,
            source_id=self.source_id,
            required_columns=self.required_columns,
            unique_key=self.unique_key,
        )
        if not isinstance(self.allowed_columns, tuple) or not self.allowed_columns:
            raise PredictorLoadError("allowed_columns must be a nonempty tuple")
        if len(set(self.allowed_columns)) != len(self.allowed_columns):
            raise PredictorLoadError("allowed_columns must be unique")
        if not set(self.required_columns).issubset(self.allowed_columns):
            raise PredictorLoadError("required columns must be included in allowed columns")


@dataclass(frozen=True)
class OutcomeTableSpec:
    """Preclassified metadata for a candidate real individual-level outcome table."""

    path: Path
    expected_sha256: str
    file_format: str
    source_id: str
    real_synthetic_status: str
    allowed_analytical_role: str
    access_status: str
    required_columns: tuple[str, ...]
    unique_key: tuple[str, ...]
    endpoint_units: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        _validate_table_spec_fields(
            expected_sha256=self.expected_sha256,
            file_format=self.file_format,
            source_id=self.source_id,
            required_columns=self.required_columns,
            unique_key=self.unique_key,
        )
        if not isinstance(self.endpoint_units, Mapping) or not self.endpoint_units:
            raise PredictorLoadError("endpoint_units must be a nonempty mapping")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(unit, str)
            or not unit
            for name, unit in self.endpoint_units.items()
        ):
            raise PredictorLoadError("endpoint_units contains an invalid declaration")


@dataclass(frozen=True)
class LoadedTable:
    """A parsed table bound to the exact immutable source bytes."""

    frame: pd.DataFrame
    sha256: str
    source_id: str


@dataclass(frozen=True)
class FrozenValidationConfig:
    """Validated preregistration payload and exact byte digest."""

    payload: dict[str, object]
    sha256: str


def _validate_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise PredictorLoadError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_table_spec_fields(
    *,
    expected_sha256: str,
    file_format: str,
    source_id: str,
    required_columns: tuple[str, ...],
    unique_key: tuple[str, ...],
) -> None:
    _validate_sha256(expected_sha256, "expected table SHA-256")
    if file_format not in {"csv", "parquet", "json_records"}:
        raise PredictorLoadError("file_format must be csv, parquet, or json_records")
    if not isinstance(source_id, str) or not source_id.strip():
        raise PredictorLoadError("source_id must be a nonempty string")
    for label, values in (("required_columns", required_columns), ("unique_key", unique_key)):
        if (
            not isinstance(values, tuple)
            or not values
            or len(set(values)) != len(values)
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise PredictorLoadError(f"{label} must contain unique nonempty strings")
    if not set(unique_key).issubset(required_columns):
        raise PredictorLoadError("unique key columns must be required")


def _read_immutable_bytes(path: Path, label: str) -> bytes:
    try:
        if not path.is_file():
            raise PredictorLoadError(f"{label} is missing or unreadable")
        return path.read_bytes()
    except OSError as error:
        raise PredictorLoadError(f"{label} is missing or unreadable") from error


def _verify_bytes(raw: bytes, expected_sha256: str, label: str) -> str:
    actual = sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise PredictorLoadError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, observed {actual}"
        )
    return actual


def _parse_table(raw: bytes, *, file_format: str, label: str) -> pd.DataFrame:
    try:
        if file_format == "csv":
            frame = pd.read_csv(BytesIO(raw))
        elif file_format == "parquet":
            frame = pd.read_parquet(BytesIO(raw))
        else:
            payload = json.loads(raw)
            if not isinstance(payload, list) or any(
                not isinstance(row, dict) for row in payload
            ):
                raise PredictorLoadError(f"{label} JSON must contain a row-object array")
            frame = pd.DataFrame(payload)
    except PredictorLoadError:
        raise
    except Exception as error:
        raise PredictorLoadError(f"{label} could not be parsed as {file_format}") from error
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise PredictorLoadError(f"{label} must contain at least one row")
    if not frame.columns.is_unique:
        raise PredictorLoadError(f"{label} columns must be unique")
    if any(not isinstance(column, str) or not column for column in frame.columns):
        raise PredictorLoadError(f"{label} columns must be nonempty strings")
    return frame


def _validate_columns_and_keys(
    frame: pd.DataFrame,
    *,
    required_columns: tuple[str, ...],
    unique_key: tuple[str, ...],
    label: str,
) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise PredictorLoadError(f"{label} schema is missing columns: {missing}")
    key_frame = frame.loc[:, list(unique_key)]
    if key_frame.isna().any().any():
        raise PredictorLoadError(f"{label} unique key contains missing values")
    for column in unique_key:
        if key_frame[column].astype(str).str.strip().eq("").any():
            raise PredictorLoadError(f"{label} unique key contains empty values")
    if key_frame.duplicated().any():
        raise PredictorLoadError(f"{label} unique key values must be unique")


def load_predictor_table(spec: PredictorTableSpec) -> LoadedTable:
    """Load one checksum-bound predictor table and reject outcome classifications."""

    if not isinstance(spec, PredictorTableSpec):
        raise TypeError("spec must be PredictorTableSpec")
    if spec.content_class != _PREDICTOR_CONTENT_CLASS:
        raise OutcomeAccessBlocked(
            "only predictor-only resources can enter the predictor reconstruction stage"
        )
    raw = _read_immutable_bytes(spec.path, f"predictor artifact {spec.source_id}")
    digest = _verify_bytes(raw, spec.expected_sha256, "predictor artifact")
    frame = _parse_table(raw, file_format=spec.file_format, label="predictor artifact")
    if tuple(frame.columns) != spec.allowed_columns:
        raise PredictorLoadError(
            "predictor artifact schema/column order does not match the frozen allowlist"
        )
    forbidden = sorted(set(frame.columns) & _OUTCOME_COLUMN_NAMES)
    if forbidden:
        raise OutcomeAccessBlocked(
            f"predictor artifact declares outcome/label columns: {forbidden}"
        )
    _validate_columns_and_keys(
        frame,
        required_columns=spec.required_columns,
        unique_key=spec.unique_key,
        label="predictor artifact",
    )
    return LoadedTable(frame=frame, sha256=digest, source_id=spec.source_id)


def load_frozen_validation_config(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> FrozenValidationConfig:
    """Validate the exact JSON-compatible YAML preregistration bytes."""

    config_path = Path(path)
    raw = _read_immutable_bytes(config_path, "person-meal validation config")
    digest = sha256(raw).hexdigest()
    if expected_sha256 is not None:
        expected = _validate_sha256(expected_sha256, "expected config SHA-256")
        if digest != expected:
            raise PredictorLoadError(
                "person-meal validation config SHA-256 does not match the frozen manifest"
            )
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PredictorLoadError(
            "person-meal validation config must be JSON-compatible YAML"
        ) from error
    if payload != _LOCKED_PERSON_MEAL_CONFIG:
        raise PredictorLoadError(
            "person-meal validation config does not match the frozen preregistration"
        )
    return FrozenValidationConfig(payload=payload, sha256=digest)


def _preflight_outcome_classification(spec: OutcomeTableSpec) -> None:
    if spec.real_synthetic_status not in _DIRECT_VALIDATION_STATUSES:
        raise OutcomeAccessBlocked(
            "synthetic or aggregate resources cannot enter direct outcome validation"
        )
    if spec.allowed_analytical_role != "direct_validation":
        raise OutcomeAccessBlocked(
            "resource is not approved for the direct-validation analytical role"
        )
    if spec.access_status not in _DIRECT_VALIDATION_ACCESS:
        raise OutcomeAccessBlocked(
            "controlled clinical outcome access has not been granted"
        )


def _require_trusted_outcome_source(spec: OutcomeTableSpec) -> None:
    """Bind caller metadata to one repository-trusted direct-validation record."""

    matches = [
        record
        for record in PREDICT_ZOE_SOURCE_REGISTRY
        if record.resource_id == spec.source_id
    ]
    if len(matches) != 1:
        raise OutcomeAccessBlocked(
            "outcome source is absent from the trusted PREDICT/ZOE registry"
        )
    record = matches[0]
    if (
        record.real_synthetic_status != spec.real_synthetic_status
        or record.real_synthetic_status not in _DIRECT_VALIDATION_STATUSES
    ):
        raise OutcomeAccessBlocked(
            "caller outcome classification does not match the trusted registry"
        )
    if (
        record.allowed_analytical_role != "direct_validation"
        or spec.allowed_analytical_role != record.allowed_analytical_role
    ):
        raise OutcomeAccessBlocked(
            "trusted registry has not approved this source for direct validation"
        )
    if (
        record.access_status not in _DIRECT_VALIDATION_ACCESS
        or spec.access_status != record.access_status
    ):
        raise OutcomeAccessBlocked(
            "trusted registry does not record granted access to this outcome source"
        )
    if _SHA256_PATTERN.fullmatch(record.sha256) is None:
        raise OutcomeAccessBlocked(
            "trusted outcome registry record lacks a verified SHA-256 digest"
        )
    if spec.expected_sha256 != record.sha256:
        raise OutcomeAccessBlocked(
            "outcome SHA-256 does not match the trusted registry record"
        )
    registered_path = Path(record.local_path)
    if not registered_path.is_absolute():
        registered_path = _REPOSITORY_ROOT / registered_path
    try:
        if spec.path.resolve(strict=True) != registered_path.resolve(strict=True):
            raise OutcomeAccessBlocked(
                "outcome path does not match the trusted registry record"
            )
    except OSError as error:
        raise OutcomeAccessBlocked(
            "trusted registered outcome path is unavailable"
        ) from error


def _read_manifest(path: Path) -> dict[str, object]:
    raw = _read_immutable_bytes(Path(path), "method-lock manifest")
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OutcomeAccessBlocked("method-lock manifest is invalid") from error
    if not isinstance(payload, dict):
        raise OutcomeAccessBlocked("method-lock manifest must be an object")
    return payload


def _manifest_digest(manifest: Mapping[str, object], field: str) -> str:
    try:
        value = manifest[field]
    except KeyError as error:
        raise OutcomeAccessBlocked(f"method-lock manifest is missing {field}") from error
    try:
        return _validate_sha256(value, f"manifest {field}")
    except PredictorLoadError as error:
        raise OutcomeAccessBlocked(str(error)) from error


def _validate_endpoint_units(
    spec: OutcomeTableSpec,
    config: FrozenValidationConfig,
) -> None:
    endpoints = config.payload["endpoints"]
    assert isinstance(endpoints, list)
    expected_units = {
        endpoint["name"]: endpoint["unit"]
        for endpoint in endpoints
        if isinstance(endpoint, dict)
    }
    declared_endpoint_columns = set(spec.required_columns) - set(spec.unique_key)
    if set(spec.endpoint_units) != declared_endpoint_columns:
        raise PredictorLoadError(
            "outcome endpoint unit declarations must exactly cover required endpoints"
        )
    for endpoint, unit in spec.endpoint_units.items():
        if endpoint not in expected_units:
            raise PredictorLoadError(f"outcome endpoint {endpoint} was not preregistered")
        if unit != expected_units[endpoint]:
            raise PredictorLoadError(
                f"outcome endpoint unit mismatch for {endpoint}: expected "
                f"{expected_units[endpoint]}"
            )


def load_outcome_table_after_gate(
    spec: OutcomeTableSpec,
    *,
    manifest_path: Path,
    method_lock_paths: object,
) -> LoadedTable:
    """Open an eligible real outcome table only after all current proofs pass."""

    if not isinstance(spec, OutcomeTableSpec):
        raise TypeError("spec must be OutcomeTableSpec")
    _preflight_outcome_classification(spec)
    _require_trusted_outcome_source(spec)
    manifest = _read_manifest(Path(manifest_path))
    snapshot = manifest.get("release_registry_snapshot")
    if not isinstance(snapshot, dict):
        raise OutcomeAccessBlocked(
            "method-lock manifest is missing the release-registry snapshot"
        )
    expected_registry_sha256 = _validate_sha256(
        snapshot.get("snapshot_sha256"),
        "manifest release-registry snapshot",
    )
    expected_config_sha256 = _manifest_digest(
        manifest,
        "person_meal_validation_config_sha256",
    )

    try:
        validate_method_lock_manifest(
            manifest,
            method_lock_paths,
            expected_release_registry_sha256=expected_registry_sha256,
        )
    except Exception as error:
        raise OutcomeAccessBlocked(
            "method-lock validation failed before outcome access"
        ) from error
    config_path = getattr(method_lock_paths, "person_meal_validation_config", None)
    if config_path is None:
        raise OutcomeAccessBlocked("method-lock paths omit the frozen validation config")
    config = load_frozen_validation_config(
        Path(config_path),
        expected_sha256=expected_config_sha256,
    )
    _validate_endpoint_units(spec, config)

    raw = _read_immutable_bytes(spec.path, f"eligible real outcome {spec.source_id}")
    digest = _verify_bytes(raw, spec.expected_sha256, "eligible real outcome")
    frame = _parse_table(raw, file_format=spec.file_format, label="eligible real outcome")
    _validate_columns_and_keys(
        frame,
        required_columns=spec.required_columns,
        unique_key=spec.unique_key,
        label="eligible real outcome",
    )
    for endpoint in spec.endpoint_units:
        numeric = pd.to_numeric(frame[endpoint], errors="coerce")
        invalid = frame[endpoint].notna() & numeric.isna()
        if invalid.any():
            raise PredictorLoadError(
                f"eligible real outcome {endpoint} contains nonnumeric values"
            )
    return LoadedTable(frame=frame, sha256=digest, source_id=spec.source_id)


__all__ = [
    "FrozenValidationConfig",
    "LoadedTable",
    "OutcomeAccessBlocked",
    "OutcomeTableSpec",
    "PredictorLoadError",
    "PredictorTableSpec",
    "load_frozen_validation_config",
    "load_outcome_table_after_gate",
    "load_predictor_table",
]
