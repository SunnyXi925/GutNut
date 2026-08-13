"""Repository-trusted, fail-closed PREDICT/ZOE staged data loading.

Callers select only a stable ``source_id``. Paths, digests, content classes,
roles, schemas and endpoint contracts come from committed trust roots. Tabular
predictor headers are checked before a complete table is parsed. Controlled
outcome bytes remain unopened until the real method-lock gate and frozen
source-independent endpoint contract have both been revalidated.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Mapping

import pandas as pd

from gmnps.data_sources.predict_zoe_registry import (
    CONTROLLED_OUTCOME_GRANTS,
    PREDICT_ZOE_SOURCE_REGISTRY,
    PREDICTOR_ARTIFACT_REGISTRY,
    ControlledOutcomeGrant,
    PredictorArtifact,
    PredictZoeSource,
    canonical_predictor_ids_sha256,
    validate_controlled_outcome_grants,
)
from gmnps.validation.method_lock_gate import validate_method_lock_manifest


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_PREDICTOR_ARTIFACTS_BY_ID: Mapping[str, PredictorArtifact] = MappingProxyType(
    {record.source_id: record for record in PREDICTOR_ARTIFACT_REGISTRY}
)
_SOURCE_REGISTRY_BY_ID: Mapping[str, PredictZoeSource] = MappingProxyType(
    {record.resource_id: record for record in PREDICT_ZOE_SOURCE_REGISTRY}
)
_CONTROLLED_GRANTS_BY_ID: Mapping[str, ControlledOutcomeGrant] = MappingProxyType(
    {record.source_id: record for record in CONTROLLED_OUTCOME_GRANTS}
)
_FORBIDDEN_COLUMN_TOKENS = frozenset(
    {
        "outcome",
        "response",
        "label",
        "target",
        "glucose",
        "triglyceride",
        "c_peptide",
        "c-peptide",
    }
)

_LOCKED_PERSON_MEAL_CONFIG = {
    "schema_version": "person-meal-validation-v2",
    "frozen_on": "2026-08-13",
    "config_role": "pre-outcome-preregistration",
    "validation_embargo": True,
    "outcome_contract": {
        "contract_version": "person-meal-outcome-contract-v1",
        "unique_key": ["participant_id", "meal_id"],
        "column_policy": "exact_trusted_grant_projection",
        "primary_endpoint_availability": "all_required",
        "secondary_endpoint_availability": (
            "required_if_trusted_data_dictionary_marks_available_otherwise_omitted"
        ),
        "caller_endpoint_subset": "forbidden",
        "endpoints": [
            {
                "name": "glucose_iAUC_2h",
                "role": "primary",
                "unit": "mmol_L_hour",
                "window_hours": [0, 2],
                "summary": "incremental_area_under_curve",
                "derivation": (
                    "baseline_subtracted_trapezoidal_auc_signed_excursions"
                ),
            },
            {
                "name": "tg_6h_rise",
                "role": "primary",
                "unit": "mmol_L",
                "window_hours": [0, 6],
                "summary": "rise_above_baseline",
                "derivation": "six_hour_value_minus_time_zero_baseline",
            },
            {
                "name": "c_peptide_iAUC_2h",
                "role": "secondary",
                "unit": "nmol_L_hour",
                "window_hours": [0, 2],
                "summary": "incremental_area_under_curve",
                "derivation": (
                    "baseline_subtracted_trapezoidal_auc_signed_excursions"
                ),
            },
        ],
    },
    "missingness": {
        "analysis_set": "endpoint_specific_available_case_after_subject_split",
        "outcome_imputation": "forbidden",
        "predictor_imputation": "median_or_mode_with_missing_indicator",
        "predictor_fit_scope": "development_fold_only",
        "report_missingness_by": [
            "endpoint",
            "cohort",
            "analysis_mode",
            "outer_fold",
            "split",
        ],
    },
    "feature_contract": {
        "schema_version": "person-meal-feature-contract-v1",
        "generated_stage": "pre-outcome_predictor_only",
        "column_policy": "exact_contract_columns_only",
        "source_artifact_hashes": "required",
        "block_artifact_hashes": "required",
        "comparator_block_mapping": "required",
        "mapping_unit_as_predictor": "forbidden",
    },
    "analysis_modes": [
        {
            "name": "subject_held_out",
            "role": "primary",
            "secondary_unit": None,
            "estimand": "generalization_to_unseen_family_twin_connected_components",
            "inference_policy": "family_twin_connected_component_cluster",
        },
        {
            "name": "subject_plus_food_held_out",
            "role": "secondary",
            "secondary_unit": "food_id",
            "estimand": (
                "descriptive_joint_generalization_to_unseen_subject_components_and_foods"
            ),
            "inference_policy": "descriptive_only",
            "descriptive_reason": (
                "multiway_subject_component_and_food_inference_not_implemented"
            ),
        },
        {
            "name": "subject_plus_meal_held_out",
            "role": "secondary",
            "secondary_unit": "meal_id",
            "estimand": (
                "descriptive_joint_generalization_to_unseen_subject_components_and_meals"
            ),
            "inference_policy": "descriptive_only",
            "descriptive_reason": (
                "multiway_subject_component_and_meal_inference_not_implemented"
            ),
        },
        {
            "name": "cohort_held_out",
            "role": "secondary",
            "secondary_unit": "cohort_id",
            "estimand": (
                "descriptive_generalization_to_unseen_whole_linked_cohort_components"
            ),
            "inference_policy": "descriptive_only",
            "descriptive_reason": "cohort_cluster_inference_not_implemented",
        },
    ],
    "split": {
        "primary_unit": "participant",
        "family_twin_grouping": True,
        "outer_folds": 5,
        "inner_folds": 5,
        "cohort_holdout_policy": "whole_cohort_with_linked_cohorts_connected",
        "development_scoring_overlap": "forbidden",
    },
    "primary_tests": {
        "multiplicity_method": "holm",
        "correction_family": (
            "two_primary_endpoints_locked_gmnps_vs_fcs_microbiome_rmse"
        ),
        "tests": [
            {
                "endpoint": "glucose_iAUC_2h",
                "analysis_mode": "subject_held_out",
                "model": "locked_attribute_gmnps",
                "reference": "fcs_microbiome",
                "metric": "rmse",
            },
            {
                "endpoint": "tg_6h_rise",
                "analysis_mode": "subject_held_out",
                "model": "locked_attribute_gmnps",
                "reference": "fcs_microbiome",
                "metric": "rmse",
            },
        ],
    },
    "other_tests": {
        "secondary_correction": "holm_within_analysis_mode_endpoint_metric",
        "null_comparators": ["random_microbiome", "shuffled_mapping"],
        "null_test_role": "exploratory",
    },
    "benchmark_specification": {
        "specification_id": "person-meal-ridge-nested-v1",
        "estimator": {
            "class": "sklearn.linear_model.Ridge",
            "solver": "lsqr",
            "alpha_grid": [0.1, 1.0, 10.0],
            "fit_intercept": True,
            "tol": 0.0001,
            "max_iter": None,
            "copy_x": True,
            "positive": False,
        },
        "preprocessing": {
            "numeric_imputation": "median_with_indicator",
            "numeric_scaling": "standard_mean_and_variance",
            "categorical_imputation": "most_frequent",
            "categorical_missing_indicator": "explicit_all_columns",
            "categorical_encoding": "one_hot_ignore_unknown_dense",
            "remainder": "drop",
        },
        "tuning": {
            "metric": "mae",
            "scope": "development_inner_folds_only",
            "selection_rule": "minimum_mean_inner_mae_then_smallest_alpha",
        },
        "seed_derivation": {
            "identifier": "additive-indexed-v1",
            "split_formula": (
                "base_plus_mode_10000019_plus_outer_fold_for_inner"
            ),
            "fit_formula": (
                "base_plus_mode_10000019_plus_endpoint_1000003_plus_"
                "comparator_10007_plus_outer_101_plus_inner"
            ),
            "resample_formula": (
                "base_plus_mode_10000019_plus_endpoint_1000003_plus_"
                "comparison_10007_plus_metric"
            ),
        },
        "bootstrap": {
            "replicates": 2_000,
            "ci_level": 0.95,
            "method": "cluster_percentile",
            "minimum_valid_fraction": 0.9,
        },
        "permutation": {
            "replicates": 2_000,
            "method": "complete_cluster_label_swap",
            "alternative": "two_sided",
            "plus_one_correction": True,
            "minimum_valid_fraction": 0.9,
        },
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
    """Raised before outcome bytes are opened when any staged proof is missing."""


@dataclass(frozen=True)
class LoadedTable:
    """Parsed bytes bound to one repository-trusted source ID and digest."""

    frame: pd.DataFrame
    sha256: str
    source_id: str


@dataclass(frozen=True)
class FrozenValidationConfig:
    """Validated preregistration payload and exact byte digest."""

    payload: dict[str, object]
    sha256: str


@dataclass(frozen=True)
class _PredictorSnapshot:
    """One immutable read used for digest, schema validation and parsing."""

    raw: bytes
    sha256: str


@dataclass(frozen=True)
class _OutcomeTableContract:
    source_columns: tuple[str, ...]
    canonical_columns: tuple[str, ...]
    unique_key: tuple[str, ...]
    endpoint_source_to_canonical: Mapping[str, str]
    endpoint_names: tuple[str, ...]


def _validate_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise PredictorLoadError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_immutable_bytes(path: Path, label: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise PredictorLoadError(f"{label} is missing, a symlink, or unreadable")
        return path.read_bytes()
    except OSError as error:
        raise PredictorLoadError(f"{label} is missing or unreadable") from error


def _verify_bytes(raw: bytes, expected_sha256: str, label: str) -> str:
    expected = _validate_sha256(expected_sha256, f"expected {label} SHA-256")
    observed = sha256(raw).hexdigest()
    if observed != expected:
        raise PredictorLoadError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _trusted_predictor_path(record: PredictorArtifact) -> Path:
    if record.path_policy != "repository_relative_exact_no_symlink":
        raise PredictorLoadError("trusted predictor path policy is unsupported")
    relative = Path(record.expected_cache_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise PredictorLoadError("trusted predictor path policy rejected an unsafe path")
    root = _REPOSITORY_ROOT.resolve()
    candidate = _REPOSITORY_ROOT / relative
    try:
        if candidate.is_symlink():
            raise PredictorLoadError("trusted predictor path policy forbids symlinks")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        if not resolved.is_file():
            raise PredictorLoadError("trusted predictor artifact is not a regular file")
    except PredictorLoadError:
        raise
    except (OSError, ValueError) as error:
        raise PredictorLoadError("trusted predictor artifact path is unavailable") from error
    return resolved


def _lookup_predictor(source_id: str) -> PredictorArtifact:
    if not isinstance(source_id, str) or not source_id:
        raise TypeError("source_id must be a nonempty string")
    try:
        record = _PREDICTOR_ARTIFACTS_BY_ID[source_id]
    except KeyError as error:
        raise OutcomeAccessBlocked(
            "predictor source is unknown to the repository-trusted registry"
        ) from error
    if (
        record.content_class != "predictor_only"
        or record.allowed_analytical_role != "predictor_reconstruction"
    ):
        raise OutcomeAccessBlocked(
            "repository registry does not classify this source as predictor-only"
        )
    return record


def _validate_predictor_columns(
    columns: tuple[str, ...],
    record: PredictorArtifact,
) -> None:
    if not columns or any(not column for column in columns) or len(set(columns)) != len(columns):
        raise PredictorLoadError("trusted predictor header contains invalid columns")
    forbidden = sorted(
        column
        for column in columns
        if any(token in column.lower() for token in _FORBIDDEN_COLUMN_TOKENS)
    )
    if forbidden:
        raise OutcomeAccessBlocked(
            f"predictor header declares outcome/label columns: {forbidden}"
        )
    if columns != record.allowed_columns:
        raise PredictorLoadError(
            "predictor header does not match the repository-trusted schema/column order"
        )


def _read_predictor_snapshot(
    path: Path,
    record: PredictorArtifact,
) -> _PredictorSnapshot:
    """Read a trusted regular file once through one descriptor and pin its bytes."""

    if not isinstance(record.size_bytes, int) or record.size_bytes < 1:
        raise PredictorLoadError("trusted predictor artifact has an invalid expected size")
    try:
        before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise PredictorLoadError(
                "trusted predictor artifact is not a non-symlink regular file"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise PredictorLoadError("trusted predictor descriptor is not a regular file")
            identity_fields = ("st_dev", "st_ino")
            if any(getattr(before, field) != getattr(opened, field) for field in identity_fields):
                raise PredictorLoadError("trusted predictor path changed before snapshot read")
            if opened.st_size != record.size_bytes:
                raise PredictorLoadError(
                    "predictor artifact size mismatch: "
                    f"expected {record.size_bytes}, observed {opened.st_size}"
                )
            raw = handle.read(record.size_bytes + 1)
            after = os.fstat(handle.fileno())
            stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(opened, field) != getattr(after, field) for field in stable_fields):
                raise PredictorLoadError("trusted predictor artifact changed during snapshot read")
    except PredictorLoadError:
        raise
    except OSError as error:
        raise PredictorLoadError("trusted predictor artifact snapshot is unreadable") from error
    if len(raw) != record.size_bytes:
        raise PredictorLoadError(
            "predictor artifact size mismatch: "
            f"expected {record.size_bytes}, observed {len(raw)}"
        )
    expected = _validate_sha256(record.sha256, "trusted predictor SHA-256")
    observed = sha256(raw).hexdigest()
    if observed != expected:
        raise PredictorLoadError(
            f"predictor artifact SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return _PredictorSnapshot(raw=raw, sha256=observed)


def _decode_json_records(raw: bytes) -> list[dict[str, object]]:
    def reject_duplicate_keys(pairs):
        keys = [key for key, _ in pairs]
        if len(set(keys)) != len(keys):
            raise PredictorLoadError("trusted predictor JSON contains duplicate keys")
        return dict(pairs)

    try:
        payload = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=reject_duplicate_keys)
    except PredictorLoadError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PredictorLoadError("trusted predictor JSON is unreadable") from error
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise PredictorLoadError("trusted predictor JSON must be a list of row objects")
    return payload


def _validate_predictor_snapshot_schema(
    raw: bytes,
    record: PredictorArtifact,
) -> None:
    """Validate the trusted header/schema from the same bytes later parsed."""

    if record.file_format == "rdata":
        expected_schema = (
            "microbiome_feature_id_axis",
            "sample_id_axis",
            "relative_abundance_value",
        )
        if (
            record.schema_kind != "relative_abundance_matrix_feature_by_sample"
            or record.allowed_columns != expected_schema
        ):
            raise PredictorLoadError("trusted RData matrix schema contract is invalid")
        return
    try:
        if record.file_format == "parquet":
            import pyarrow.parquet as pq

            columns = tuple(pq.ParquetFile(BytesIO(raw)).schema_arrow.names)
            _validate_predictor_columns(columns, record)
            return
        if record.file_format == "json":
            records = _decode_json_records(raw)
            if not records:
                raise PredictorLoadError("trusted predictor JSON must contain rows")
            for row in records:
                _validate_predictor_columns(tuple(row), record)
            return
        if record.file_format not in {"csv", "tsv"}:
            raise PredictorLoadError("trusted predictor table format is unsupported")
        first_line = BytesIO(raw).readline(1024 * 1024 + 1)
        if len(first_line) > 1024 * 1024:
            raise PredictorLoadError("trusted predictor header exceeds the safety limit")
        delimiter = "," if record.file_format == "csv" else "\t"
        decoded = first_line.decode("utf-8-sig")
        columns = tuple(next(csv.reader(StringIO(decoded), delimiter=delimiter)))
        _validate_predictor_columns(columns, record)
    except (PredictorLoadError, OutcomeAccessBlocked):
        raise
    except Exception as error:
        raise PredictorLoadError("trusted predictor header is unreadable") from error


def _parse_trusted_table(raw: bytes, record: PredictorArtifact) -> pd.DataFrame:
    try:
        if record.file_format == "csv":
            frame = pd.read_csv(BytesIO(raw))
        elif record.file_format == "tsv":
            frame = pd.read_csv(BytesIO(raw), sep="\t")
        elif record.file_format == "json":
            frame = pd.DataFrame.from_records(_decode_json_records(raw))
        elif record.file_format == "parquet":
            frame = pd.read_parquet(BytesIO(raw))
        else:
            raise PredictorLoadError("RData matrices use the dedicated trusted parser")
    except PredictorLoadError:
        raise
    except Exception as error:
        raise PredictorLoadError("trusted predictor table could not be parsed") from error
    return frame


def _validate_table_frame(frame: pd.DataFrame, record: PredictorArtifact) -> None:
    if frame.empty:
        raise PredictorLoadError("trusted predictor table must contain rows")
    if tuple(frame.columns) != record.allowed_columns:
        raise PredictorLoadError("parsed predictor schema differs from validated header")
    if len(frame) != record.expected_records:
        raise PredictorLoadError(
            f"predictor record count mismatch: expected {record.expected_records}"
        )
    missing = [column for column in record.required_columns if column not in frame.columns]
    if missing:
        raise PredictorLoadError(f"predictor schema is missing required columns: {missing}")
    keys = frame.loc[:, list(record.unique_key)]
    if keys.isna().any().any():
        raise PredictorLoadError("predictor unique key contains missing values")
    if keys.astype(str).apply(lambda column: column.str.strip().eq("")).any().any():
        raise PredictorLoadError("predictor unique key contains empty values")
    if keys.duplicated().any():
        raise PredictorLoadError("predictor unique key values must be unique")
    identifiers = tuple(str(value) for value in frame[record.id_column])
    if canonical_predictor_ids_sha256(identifiers) != record.id_sha256:
        raise PredictorLoadError("predictor ID set does not match the trusted digest")


def _parse_trusted_rdata(raw: bytes, record: PredictorArtifact) -> pd.DataFrame:
    try:
        import rdata

        parsed = rdata.parser.parse_data(raw, extension=".rda")
        objects = rdata.conversion.convert(parsed)
        if not isinstance(objects, dict) or len(objects) != 1:
            raise PredictorLoadError("trusted RData must contain exactly one matrix")
        object_name, matrix = next(iter(objects.items()))
        if any(token in str(object_name).lower() for token in _FORBIDDEN_COLUMN_TOKENS):
            raise OutcomeAccessBlocked("trusted RData declares an outcome/label object")
        if getattr(matrix, "ndim", None) != 2:
            raise PredictorLoadError("trusted RData object is not a two-dimensional matrix")
        if isinstance(matrix, pd.DataFrame):
            feature_ids = tuple(str(value) for value in matrix.index)
            sample_ids = tuple(str(value) for value in matrix.columns)
            matrix_values = matrix.to_numpy()
        else:
            feature_dim, sample_dim = matrix.dims
            feature_ids = tuple(str(value) for value in matrix.coords[feature_dim].values)
            sample_ids = tuple(str(value) for value in matrix.coords[sample_dim].values)
            matrix_values = matrix.values
        if len(sample_ids) != record.expected_records:
            raise PredictorLoadError("trusted RData sample count does not match registry")
        if canonical_predictor_ids_sha256(sample_ids) != record.id_sha256:
            raise PredictorLoadError("trusted RData sample IDs do not match registry")
        if (
            record.feature_id_sha256 is None
            or canonical_predictor_ids_sha256(feature_ids) != record.feature_id_sha256
        ):
            raise PredictorLoadError("trusted RData feature IDs do not match registry")
        frame = pd.DataFrame(matrix_values, index=feature_ids, columns=sample_ids)
    except (PredictorLoadError, OutcomeAccessBlocked):
        raise
    except Exception as error:
        raise PredictorLoadError("trusted RData matrix could not be parsed") from error
    return frame


def load_predictor_table(source_id: str) -> LoadedTable:
    """Load one exact predictor artifact selected only by trusted source ID."""

    record = _lookup_predictor(source_id)
    path = _trusted_predictor_path(record)
    snapshot = _read_predictor_snapshot(path, record)
    _validate_predictor_snapshot_schema(snapshot.raw, record)
    if record.file_format == "rdata":
        frame = _parse_trusted_rdata(snapshot.raw, record)
    else:
        frame = _parse_trusted_table(snapshot.raw, record)
        _validate_table_frame(frame, record)
    return LoadedTable(frame=frame, sha256=snapshot.sha256, source_id=record.source_id)


def load_frozen_validation_config(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> FrozenValidationConfig:
    """Validate exact JSON-compatible YAML preregistration bytes."""

    raw = _read_immutable_bytes(Path(path), "person-meal validation config")
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


def _lookup_controlled_source(
    source_id: str,
) -> tuple[PredictZoeSource, ControlledOutcomeGrant]:
    if not isinstance(source_id, str) or not source_id:
        raise TypeError("source_id must be a nonempty string")
    source = _SOURCE_REGISTRY_BY_ID.get(source_id)
    grant = _CONTROLLED_GRANTS_BY_ID.get(source_id)
    if source is None or grant is None:
        raise OutcomeAccessBlocked(
            "outcome source is absent from the repository-trusted controlled registry"
        )
    try:
        validate_controlled_outcome_grants((grant,), sources=(source,))
    except (TypeError, ValueError) as error:
        raise OutcomeAccessBlocked("controlled outcome grant evidence is invalid") from error
    if (
        source.real_synthetic_status != "controlled_real"
        or source.access_status != "controlled_granted"
        or source.allowed_analytical_role != "direct_validation"
        or grant.access_status != "controlled_granted"
        or grant.allowed_analytical_role != "direct_validation"
    ):
        raise OutcomeAccessBlocked(
            "controlled access/direct-validation grant has not been established"
        )
    return source, grant


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        raw = _read_immutable_bytes(path, "method-lock manifest")
        payload = json.loads(raw)
    except PredictorLoadError as error:
        raise OutcomeAccessBlocked(str(error)) from error
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OutcomeAccessBlocked("method-lock manifest is invalid") from error
    if not isinstance(payload, dict):
        raise OutcomeAccessBlocked("method-lock manifest must be an object")
    return payload


def _manifest_digest(manifest: Mapping[str, object], field: str) -> str:
    try:
        return _validate_sha256(manifest[field], f"manifest {field}")
    except KeyError as error:
        raise OutcomeAccessBlocked(f"method-lock manifest is missing {field}") from error
    except PredictorLoadError as error:
        raise OutcomeAccessBlocked(str(error)) from error


def _derive_outcome_table_contract(
    grant: ControlledOutcomeGrant,
    config: FrozenValidationConfig,
) -> _OutcomeTableContract:
    """Derive all outcome columns from trusted grant plus frozen config only."""

    raw_contract = config.payload.get("outcome_contract")
    if not isinstance(raw_contract, dict):
        raise PredictorLoadError("frozen outcome contract is missing")
    if (
        raw_contract.get("primary_endpoint_availability") != "all_required"
        or raw_contract.get("secondary_endpoint_availability")
        != "required_if_trusted_data_dictionary_marks_available_otherwise_omitted"
        or raw_contract.get("caller_endpoint_subset") != "forbidden"
        or raw_contract.get("column_policy") != "exact_trusted_grant_projection"
    ):
        raise PredictorLoadError("frozen outcome availability/subset contract is invalid")
    unique_key_raw = raw_contract.get("unique_key")
    if unique_key_raw != ["participant_id", "meal_id"]:
        raise PredictorLoadError("frozen participant/meal unique-key contract is invalid")
    unique_key = tuple(unique_key_raw)
    if grant.participant_meal_key_contract != unique_key:
        raise OutcomeAccessBlocked(
            "trusted source grant does not match the frozen participant/meal key contract"
        )
    config_endpoints = raw_contract.get("endpoints")
    if not isinstance(config_endpoints, list) or not config_endpoints:
        raise PredictorLoadError("frozen endpoint contract is missing")
    expected_by_name = {
        endpoint.get("name"): endpoint
        for endpoint in config_endpoints
        if isinstance(endpoint, dict) and isinstance(endpoint.get("name"), str)
    }
    grant_by_name = {endpoint.name: endpoint for endpoint in grant.endpoint_contracts}
    if set(grant_by_name) != set(expected_by_name):
        raise OutcomeAccessBlocked(
            "trusted source grant does not completely cover frozen endpoint contract"
        )
    source_columns = list(unique_key)
    canonical_columns = list(unique_key)
    source_to_canonical: dict[str, str] = {}
    available_endpoint_names = []
    for endpoint_name, expected in expected_by_name.items():
        granted = grant_by_name[endpoint_name]
        semantic_fields = {
            "role": granted.role,
            "unit": granted.unit,
            "window_hours": list(granted.window_hours),
            "summary": granted.summary,
            "derivation": granted.derivation,
        }
        if any(semantic_fields[field] != expected.get(field) for field in semantic_fields):
            raise OutcomeAccessBlocked(
                f"trusted source grant changes frozen endpoint contract for {endpoint_name}"
            )
        if granted.role == "primary" and granted.availability != "available":
            raise OutcomeAccessBlocked(
                f"primary endpoint {endpoint_name} is required by the frozen contract"
            )
        if granted.availability == "available":
            if not isinstance(granted.source_column, str) or not granted.source_column:
                raise OutcomeAccessBlocked("available endpoint lacks dictionary-attested column")
            source_columns.append(granted.source_column)
            canonical_columns.append(endpoint_name)
            source_to_canonical[granted.source_column] = endpoint_name
            available_endpoint_names.append(endpoint_name)
        elif granted.role != "secondary" or granted.availability != "not_available":
            raise OutcomeAccessBlocked("endpoint availability violates the frozen contract")
    if len(set(source_columns)) != len(source_columns):
        raise OutcomeAccessBlocked("trusted outcome source columns are not unique")
    return _OutcomeTableContract(
        source_columns=tuple(source_columns),
        canonical_columns=tuple(canonical_columns),
        unique_key=unique_key,
        endpoint_source_to_canonical=MappingProxyType(source_to_canonical),
        endpoint_names=tuple(available_endpoint_names),
    )


def _outcome_header(path: Path, file_format: str) -> tuple[str, ...]:
    if file_format == "parquet":
        try:
            import pyarrow.parquet as pq

            return tuple(pq.ParquetFile(path).schema_arrow.names)
        except Exception as error:
            raise PredictorLoadError("controlled outcome parquet header is unreadable") from error
    if file_format not in {"csv", "tsv"}:
        raise OutcomeAccessBlocked("controlled outcome file format is not preregistered")
    delimiter = "," if file_format == "csv" else "\t"
    try:
        with path.open("rb") as handle:
            first_line = handle.readline(1024 * 1024 + 1)
        if len(first_line) > 1024 * 1024:
            raise PredictorLoadError("controlled outcome header exceeds safety limit")
        return tuple(
            next(csv.reader(StringIO(first_line.decode("utf-8-sig")), delimiter=delimiter))
        )
    except PredictorLoadError:
        raise
    except Exception as error:
        raise PredictorLoadError("controlled outcome header is unreadable") from error


def _read_outcome_bytes(path: Path, expected_sha256: str) -> tuple[bytes, str]:
    raw = _read_immutable_bytes(path, "eligible controlled outcome")
    digest = _verify_bytes(raw, expected_sha256, "eligible controlled outcome")
    return raw, digest


def _parse_outcome_table(raw: bytes, file_format: str) -> pd.DataFrame:
    try:
        if file_format == "csv":
            frame = pd.read_csv(BytesIO(raw))
        elif file_format == "tsv":
            frame = pd.read_csv(BytesIO(raw), sep="\t")
        elif file_format == "parquet":
            frame = pd.read_parquet(BytesIO(raw))
        else:
            raise PredictorLoadError("controlled outcome file format is unsupported")
    except PredictorLoadError:
        raise
    except Exception as error:
        raise PredictorLoadError("controlled outcome table could not be parsed") from error
    if frame.empty or not frame.columns.is_unique:
        raise PredictorLoadError("controlled outcome table/header is empty or duplicated")
    return frame


def _validate_outcome_frame(
    frame: pd.DataFrame,
    contract: _OutcomeTableContract,
) -> pd.DataFrame:
    if tuple(frame.columns) != contract.source_columns:
        raise OutcomeAccessBlocked(
            "controlled outcome columns do not match the dictionary-attested frozen contract"
        )
    renamed = frame.rename(columns=dict(contract.endpoint_source_to_canonical))
    if tuple(renamed.columns) != contract.canonical_columns:
        raise PredictorLoadError("controlled outcome canonical projection failed")
    keys = renamed.loc[:, list(contract.unique_key)]
    if keys.isna().any().any():
        raise PredictorLoadError("controlled outcome unique key contains missing values")
    if keys.astype(str).apply(lambda column: column.str.strip().eq("")).any().any():
        raise PredictorLoadError("controlled outcome unique key contains empty values")
    if keys.duplicated().any():
        raise PredictorLoadError("controlled outcome participant/meal key must be unique")
    for endpoint in contract.endpoint_names:
        numeric = pd.to_numeric(renamed[endpoint], errors="coerce")
        invalid = renamed[endpoint].notna() & numeric.isna()
        if invalid.any():
            raise PredictorLoadError(
                f"controlled outcome endpoint {endpoint} contains nonnumeric values"
            )
    return renamed


def load_outcome_table_after_gate(
    source_id: str,
    *,
    manifest_path: Path,
    method_lock_paths: object,
) -> LoadedTable:
    """Read controlled outcomes only after grant, method lock and config pass."""

    source, grant = _lookup_controlled_source(source_id)
    manifest = _read_manifest(Path(manifest_path))
    snapshot = manifest.get("release_registry_snapshot")
    if not isinstance(snapshot, dict):
        raise OutcomeAccessBlocked(
            "method-lock manifest is missing the release-registry snapshot"
        )
    try:
        expected_registry_sha256 = _validate_sha256(
            snapshot.get("snapshot_sha256"),
            "manifest release-registry snapshot",
        )
    except PredictorLoadError as error:
        raise OutcomeAccessBlocked(str(error)) from error
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
    contract = _derive_outcome_table_contract(grant, config)

    assert grant.verified_local_path is not None
    assert grant.sha256 is not None
    assert grant.file_format is not None
    outcome_path = Path(grant.verified_local_path)
    header = _outcome_header(outcome_path, grant.file_format)
    if header != contract.source_columns:
        raise OutcomeAccessBlocked(
            "controlled outcome header does not match the frozen trusted contract"
        )
    raw, digest = _read_outcome_bytes(outcome_path, grant.sha256)
    frame = _parse_outcome_table(raw, grant.file_format)
    canonical = _validate_outcome_frame(frame, contract)
    return LoadedTable(frame=canonical, sha256=digest, source_id=source.resource_id)


__all__ = [
    "FrozenValidationConfig",
    "LoadedTable",
    "OutcomeAccessBlocked",
    "PredictorLoadError",
    "load_frozen_validation_config",
    "load_outcome_table_after_gate",
    "load_predictor_table",
]
