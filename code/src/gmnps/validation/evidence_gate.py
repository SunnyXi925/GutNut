"""Task 3 result exporter and path-only direct-response evidence gate."""
from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
from io import BytesIO
import json
from math import isfinite
import os
from pathlib import Path
import re
import stat
from typing import Mapping

import numpy as np
import pandas as pd

from gmnps.data_sources.predict_zoe_registry import get_controlled_outcome_grant
from gmnps.validation.cohort_split import (
    family_twin_component_ids,
    make_nested_group_splits,
)
from gmnps.validation.method_lock_gate import MethodLockArtifactPaths
from gmnps.validation.person_meal_benchmark import (
    REQUIRED_COMPARATORS,
    BenchmarkResult,
    load_locked_outcomes_for_benchmark,
)


COMPUTATIONAL_FEASIBILITY = "computational_feasibility"
DIRECT_EXTERNAL_VALIDITY = "direct_external_validity"
TESTING_ONLY_NO_CLAIM_UPGRADE = "testing_only_no_claim_upgrade"

_PRIMARY_ENDPOINTS = ("glucose_iAUC_2h", "tg_6h_rise")
_DIRECT_ROLE = "direct_validation"
_DIRECT_DATA_CLASS = "real_observed_participant_meal_outcomes"
_PRIMARY_FAMILY = "two_primary_endpoints_locked_gmnps_vs_fcs_microbiome_rmse"
_CI_METHOD = "paired_family_twin_component_cluster_percentile_bootstrap_95"
_MINIMUM_REPLICATES = 2_000
_MINIMUM_VALID_FRACTION = 0.90
_ALPHA = 0.05
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TRUSTED_RESULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/direct_validity_result_registry.json"
)
_LOCK_FIELDS = (
    "person_meal_validation_config_sha256",
    "feature_contract_sha256",
    "predictor_frame_sha256",
    "cohort_split_implementation_sha256",
    "person_meal_benchmark_implementation_sha256",
    "benchmark_specification_sha256",
    "method_lock_gate_implementation_sha256",
)
_TASK3_PROVENANCE_BINDING = {
    "manifest_sha256": "method_lock_manifest_sha256",
    "outcome_source_id": "outcome_source_id",
    "outcome_source_sha256": "outcome_sha256",
    "predictor_frame_sha256": "predictor_frame_sha256",
    "feature_contract_sha256": "feature_contract_sha256",
    "method_lock_gate_implementation_sha256": "method_lock_gate_implementation_sha256",
    "cohort_split_implementation_sha256": "cohort_split_implementation_sha256",
    "person_meal_benchmark_implementation_sha256": "person_meal_benchmark_implementation_sha256",
    "benchmark_specification_sha256": "benchmark_specification_sha256",
    "validation_config_sha256": "person_meal_validation_config_sha256",
}
_PROHIBITED_COMPUTATIONAL_CLAIMS = (
    "transforms_postprandial_response",
    "precision_ready",
    "clinical_validity",
    "external_validity",
    "direct_response_validity",
    "causal_dietary_effect",
)


@dataclass(frozen=True)
class VerifiedRunBinding:
    """Task 3 run identity copied from a live verified lock/outcome input."""

    run_id: str
    method_lock_manifest_sha256: str
    outcome_source_id: str
    outcome_sha256: str
    person_meal_validation_config_sha256: str
    feature_contract_sha256: str
    predictor_frame_sha256: str
    cohort_split_implementation_sha256: str
    person_meal_benchmark_implementation_sha256: str
    benchmark_specification_sha256: str
    method_lock_gate_implementation_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        if not isinstance(self.outcome_source_id, str) or not self.outcome_source_id:
            raise ValueError("outcome_source_id must be a nonempty string")
        for field in fields(self):
            if field.name in {"run_id", "outcome_source_id"}:
                continue
            if not _is_digest(getattr(self, field.name)):
                raise ValueError(f"{field.name} must be a lowercase SHA-256 digest")

    @classmethod
    def from_locked_outcome(cls, run_id: str, locked: object) -> "VerifiedRunBinding":
        verified = getattr(locked, "verified_lock", None)
        outcome = getattr(locked, "outcome", None)
        manifest = getattr(verified, "manifest", None)
        if not isinstance(manifest, Mapping):
            raise TypeError("locked input must contain a verified method-lock manifest")
        values = {
            field: manifest.get(field)
            for field in _LOCK_FIELDS
        }
        return cls(
            run_id=run_id,
            method_lock_manifest_sha256=getattr(verified, "manifest_sha256", None),
            outcome_source_id=getattr(outcome, "source_id", None),
            outcome_sha256=getattr(outcome, "sha256", None),
            **values,
        )

    def payload(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @property
    def run_binding_sha256(self) -> str:
        return _canonical_hash(self.payload())


@dataclass(frozen=True)
class ExportedBenchmarkArtifacts:
    result_run_manifest: Path
    paired_metrics: Path
    predictions: Path
    split_audit: Path
    analysis_status: Path


@dataclass(frozen=True)
class EvidenceGateArtifactPaths:
    """Immutable filesystem inputs required by the production gate."""

    method_lock_manifest: Path
    method_lock_artifacts: MethodLockArtifactPaths
    result_run_manifest: Path
    paired_metrics: Path
    predictions: Path
    split_audit: Path
    analysis_status: Path
    controlled_outcome_source_id: str

    def __post_init__(self) -> None:
        for name in (
            "method_lock_manifest",
            "result_run_manifest",
            "paired_metrics",
            "predictions",
            "split_audit",
            "analysis_status",
        ):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)):
                raise TypeError(f"{name} must be a filesystem path")
            object.__setattr__(self, name, Path(value))
        if not isinstance(self.method_lock_artifacts, MethodLockArtifactPaths):
            raise TypeError("method_lock_artifacts must be MethodLockArtifactPaths")
        if not isinstance(self.controlled_outcome_source_id, str):
            raise TypeError("controlled_outcome_source_id must be a source identifier")


@dataclass(frozen=True)
class EvidenceGateOutcome:
    tier: str
    passed: bool
    blockers: tuple[str, ...]
    endpoint_checks: pd.DataFrame
    allowed_claims: tuple[str, ...]
    prohibited_claims: tuple[str, ...]
    source_state: str
    gate_version: str = "direct-response-evidence-gate-v3"


def _canonical_hash(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _canonical_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")


def _bound_frame(frame: pd.DataFrame, binding: VerifiedRunBinding) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("BenchmarkResult artifacts must be pandas DataFrames")
    if {"run_id", "run_binding_sha256"} & set(frame.columns):
        raise ValueError("BenchmarkResult artifacts already contain reserved run binding columns")
    bound = frame.copy()
    bound["run_id"] = binding.run_id
    bound["run_binding_sha256"] = binding.run_binding_sha256
    return bound


def _task3_provenance_blockers(
    frame: pd.DataFrame,
    artifact_name: str,
    binding: VerifiedRunBinding,
) -> list[str]:
    blockers: list[str] = []
    for column, binding_field in _TASK3_PROVENANCE_BINDING.items():
        if column not in frame:
            blockers.append(f"{artifact_name} omits Task 3 provenance column {column}")
            continue
        expected = getattr(binding, binding_field)
        if frame.empty or not frame[column].astype(str).eq(str(expected)).all():
            blockers.append(
                f"{artifact_name} Task 3 provenance {column} does not match verified run binding"
            )
    return blockers


def export_benchmark_result(
    result: BenchmarkResult,
    binding: VerifiedRunBinding,
    output_directory: str | Path,
) -> ExportedBenchmarkArtifacts:
    """Export the real Task 3 schema and one hash-bound result-run manifest."""

    if not isinstance(result, BenchmarkResult):
        raise TypeError("result must be a person_meal_benchmark.BenchmarkResult")
    if not isinstance(binding, VerifiedRunBinding):
        raise TypeError("binding must be a VerifiedRunBinding")
    required = {
        "paired_metrics": {"comparator", "reference", "adjusted_p_value"},
        "predictions": {"comparator", "row_id", "outer_fold"},
        "analysis_status": {"analysis_status", "endpoint"},
        "split_audit": {"analysis_mode", "outer_fold", "train_n_rows", "test_n_rows"},
    }
    for name, columns in required.items():
        frame = getattr(result, name)
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"BenchmarkResult {name} omits columns: {sorted(missing)}")
        if name != "predictions":
            provenance_blockers = _task3_provenance_blockers(frame, name, binding)
            if provenance_blockers:
                raise ValueError("; ".join(provenance_blockers))
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    artifact_paths = {
        "paired_metrics": output / "paired_metrics.csv",
        "predictions": output / "predictions.csv",
        "split_audit": output / "split_audit.csv",
        "analysis_status": output / "analysis_status.csv",
    }
    artifact_manifest: dict[str, dict[str, object]] = {}
    for name, path in artifact_paths.items():
        payload = _canonical_csv_bytes(_bound_frame(getattr(result, name), binding))
        path.write_bytes(payload)
        artifact_manifest[name] = {
            "file": path.name,
            "sha256": sha256(payload).hexdigest(),
            "format": "canonical_csv_v1",
        }
    manifest = {
        "schema_version": "direct-validity-result-run-v2",
        "evidence_role": _DIRECT_ROLE,
        "data_class": _DIRECT_DATA_CLASS,
        **binding.payload(),
        "run_binding_sha256": binding.run_binding_sha256,
        "artifacts": artifact_manifest,
    }
    manifest_path = output / "result_run_manifest.json"
    manifest_path.write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return ExportedBenchmarkArtifacts(
        result_run_manifest=manifest_path,
        **artifact_paths,
    )


def _read_regular_bytes(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing or unreadable") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be an immutable regular non-symlink file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{label} changed during immutable open")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"{label} changed during live reread")
    return b"".join(chunks)


def _json_object(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _fail(
    blockers: list[str],
    checks: list[dict[str, object]] | None = None,
    *,
    source_state: str = "provided_artifacts_failed_gate",
) -> EvidenceGateOutcome:
    return EvidenceGateOutcome(
        tier=COMPUTATIONAL_FEASIBILITY,
        passed=False,
        blockers=tuple(dict.fromkeys(blockers)),
        endpoint_checks=pd.DataFrame.from_records(checks or []),
        allowed_claims=(
            "computational_feasibility",
            "correctly_specified_synthetic_positive_control",
            "biological_consistency",
            "mechanistic_consistency",
        ),
        prohibited_claims=_PROHIBITED_COMPUTATIONAL_CLAIMS,
        source_state=source_state,
    )


def _current_missing_blockers() -> list[str]:
    source_id = "predict_controlled_clinical_zenodo"
    try:
        access = get_controlled_outcome_grant(source_id).access_status
    except Exception:
        access = "unavailable"
    return [
        f"eligible observed participant-by-meal outcomes are unavailable: {source_id} access_status={access}",
        "run-level method-lock manifest path is missing",
        "result-run manifest path is missing",
        "paired metrics path is missing",
        "predictions path is missing",
        "split audit summary path is missing",
        "analysis status path is missing",
    ]


def _trusted_approval(manifest_bytes: bytes, run_id: object) -> str | None:
    registry = _json_object(
        _read_regular_bytes(_TRUSTED_RESULT_REGISTRY_PATH, "trusted result registry"),
        "trusted result registry",
    )
    if registry.get("schema_version") != "direct-validity-result-registry-v1":
        raise ValueError("trusted result registry schema version is invalid")
    approved = registry.get("approved_runs")
    if not isinstance(approved, list):
        raise ValueError("trusted result registry approved_runs is invalid")
    matches = [
        row
        for row in approved
        if isinstance(row, dict) and row.get("run_id") == run_id
    ]
    if len(matches) != 1:
        return None
    expected = matches[0].get("result_run_manifest_sha256")
    if not _is_digest(expected):
        raise ValueError("trusted registry result-run manifest digest is invalid")
    return str(expected) if expected == sha256(manifest_bytes).hexdigest() else "mismatch"


def _strict_number(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not isfinite(numeric):
        raise ValueError(f"{label} must be a finite number")
    return numeric


def _strict_integer(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be a strict non-boolean integer")
    return int(value)


def _validate_endpoint_rows(rows: pd.DataFrame) -> tuple[list[dict[str, object]], list[str]]:
    required = {
        "endpoint", "analysis_mode", "comparator", "reference", "metric",
        "estimate_delta", "ci_lower", "ci_upper", "ci_method",
        "bootstrap_replicates_requested", "bootstrap_replicates_valid",
        "bootstrap_minimum_valid_fraction", "bootstrap_valid_fraction",
        "permutation_replicates_requested", "permutation_replicates_valid",
        "permutation_minimum_valid_fraction", "permutation_valid_fraction",
        "permutation_p_value", "adjusted_p_value", "test_role",
        "correction_family", "multiplicity_method", "inference_status",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        return [], ["paired metrics omit Task 3 columns: " + ", ".join(missing)]
    primary = rows.loc[
        rows["analysis_mode"].eq("subject_held_out")
        & rows["comparator"].eq("locked_attribute_gmnps")
        & rows["reference"].eq("fcs_microbiome")
        & rows["metric"].eq("rmse")
        & rows["test_role"].eq("primary")
    ].copy()
    if len(primary) != 2 or set(primary["endpoint"]) != set(_PRIMARY_ENDPOINTS):
        return [], [
            "paired metrics do not contain exactly the two frozen subject-held-out primary RMSE rows"
        ]
    parsed: dict[object, dict[str, object]] = {}
    blockers: list[str] = []
    for index, row in primary.iterrows():
        endpoint = str(row["endpoint"])
        try:
            estimate = _strict_number(row["estimate_delta"], f"{endpoint} estimate")
            lower = _strict_number(row["ci_lower"], f"{endpoint} confidence interval lower")
            upper = _strict_number(row["ci_upper"], f"{endpoint} confidence interval upper")
            raw_p = _strict_number(row["permutation_p_value"], f"{endpoint} raw permutation p-value")
            supplied = _strict_number(row["adjusted_p_value"], f"{endpoint} adjusted p-value")
            if not 0.0 <= raw_p <= 1.0:
                raise ValueError(f"{endpoint} raw permutation p-value must be in [0,1]")
            if not 0.0 <= supplied <= 1.0:
                raise ValueError(f"{endpoint} adjusted p-value must be in [0,1]")
            if not lower <= upper < 0.0 or row["ci_method"] != _CI_METHOD:
                raise ValueError(f"{endpoint} confidence interval order/direction/method is invalid")
            requested_boot = _strict_integer(
                row["bootstrap_replicates_requested"],
                f"{endpoint} bootstrap requested count",
            )
            valid_boot = _strict_integer(
                row["bootstrap_replicates_valid"],
                f"{endpoint} bootstrap valid count",
            )
            requested_perm = _strict_integer(
                row["permutation_replicates_requested"],
                f"{endpoint} permutation requested count",
            )
            valid_perm = _strict_integer(
                row["permutation_replicates_valid"],
                f"{endpoint} permutation valid count",
            )
            for label, requested, valid in (
                ("bootstrap", requested_boot, valid_boot),
                ("permutation", requested_perm, valid_perm),
            ):
                if requested < 0 or valid < 0:
                    raise ValueError(f"{endpoint} {label} counts must be nonnegative")
                if valid > requested:
                    raise ValueError(f"{endpoint} {label} valid count cannot exceed requested")
                if requested != _MINIMUM_REPLICATES:
                    raise ValueError(f"{endpoint} {label} requested count must equal 2000")
                if valid / requested < _MINIMUM_VALID_FRACTION:
                    raise ValueError(f"{endpoint} does not meet the valid-resample threshold")
            for label, observed, expected in (
                ("bootstrap minimum", row["bootstrap_minimum_valid_fraction"], 0.9),
                ("permutation minimum", row["permutation_minimum_valid_fraction"], 0.9),
                ("bootstrap valid", row["bootstrap_valid_fraction"], valid_boot / requested_boot),
                ("permutation valid", row["permutation_valid_fraction"], valid_perm / requested_perm),
            ):
                if abs(_strict_number(observed, f"{endpoint} {label} fraction") - expected) > 1e-12:
                    raise ValueError(f"{endpoint} {label} fraction is inconsistent")
            parsed[index] = {
                "endpoint": endpoint,
                "estimate": estimate,
                "lower": lower,
                "upper": upper,
                "raw_p": raw_p,
                "supplied": supplied,
                "counts_passed": True,
                "specification": (
                    row["correction_family"] == _PRIMARY_FAMILY
                    and row["multiplicity_method"] == "holm"
                    and row["inference_status"] == "completed"
                ),
            }
        except ValueError as error:
            blockers.append(str(error))
    if blockers:
        return [], blockers
    ordered = sorted(parsed, key=lambda index: (parsed[index]["raw_p"], str(index)))
    adjusted: dict[object, float] = {}
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, parsed[index]["raw_p"] * (2 - rank)))
        adjusted[index] = running
    checks: list[dict[str, object]] = []
    for index, values in parsed.items():
        endpoint = str(values["endpoint"])
        direction = float(values["estimate"]) < 0.0
        multiplicity = (
            abs(float(values["supplied"]) - adjusted[index]) <= 1e-12
            and adjusted[index] <= _ALPHA
            and bool(values["specification"])
        )
        checks.append(
            {
                "endpoint": endpoint,
                "estimate_delta": values["estimate"],
                "ci_lower": values["lower"],
                "ci_upper": values["upper"],
                "raw_permutation_p_value": values["raw_p"],
                "recomputed_holm_adjusted_p_value": adjusted[index],
                "direction_passed": direction,
                "ci_passed": True,
                "multiplicity_passed": multiplicity,
                "valid_replicates_passed": True,
                "specification_passed": bool(values["specification"]),
                "passed": direction and multiplicity,
            }
        )
        if not direction:
            blockers.append(f"{endpoint} RMSE paired improvement is not in the required direction")
        if not multiplicity:
            blockers.append(f"{endpoint} does not pass the recomputed Holm test")
    return checks, blockers


def _strict_fold(value: object, label: str) -> int:
    return _strict_integer(value, label)


def _split_and_prediction_blockers(
    predictions: pd.DataFrame,
    split_audit: pd.DataFrame,
    predictor_frame: pd.DataFrame,
    outcome_frame: pd.DataFrame,
    config: object,
) -> list[str]:
    blockers: list[str] = []
    payload = getattr(config, "payload", None)
    if not isinstance(payload, Mapping):
        return ["verified validation config payload is unavailable"]
    split_config = payload.get("split")
    seeds = payload.get("seeds")
    if not isinstance(split_config, Mapping) or not isinstance(seeds, Mapping):
        return ["verified validation config split/seeds are invalid"]
    try:
        outer_folds = _strict_integer(split_config.get("outer_folds"), "config outer_folds")
        inner_folds = _strict_integer(split_config.get("inner_folds"), "config inner_folds")
        outer_seed = _strict_integer(seeds.get("outer_split"), "config outer split seed")
        inner_seed = _strict_integer(seeds.get("inner_cv"), "config inner CV seed")
        nested = make_nested_group_splits(
            predictor_frame,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            outer_seed=outer_seed,
            inner_seed=inner_seed,
            secondary_holdout=None,
        )
    except (TypeError, ValueError) as error:
        return [f"production subject-held-out split recomputation failed: {error}"]
    working = predictor_frame.copy().reset_index(drop=True)
    required_predictor = {"participant_id", "meal_id", "family_id", "twin_id"}
    if required_predictor - set(working.columns):
        return ["verified predictor frame omits split/key columns"]
    if working.duplicated(["participant_id", "meal_id"]).any():
        return ["verified predictor opportunities contain duplicate person-meal keys"]
    working.insert(
        0,
        "row_id",
        working["participant_id"].astype(str) + "::" + working["meal_id"].astype(str),
    )
    working["recomputed_component"] = family_twin_component_ids(working).astype(str)
    expected_fold: dict[str, int] = {}
    for fold in nested:
        train = working.iloc[list(fold.outer.train_positions)]
        test = working.iloc[list(fold.outer.test_positions)]
        if train.empty or test.empty:
            blockers.append(f"recomputed subject-held-out fold {fold.outer.fold_id} is empty")
        train_components = set(train["recomputed_component"])
        test_components = set(test["recomputed_component"])
        if train_components & test_components:
            blockers.append(
                f"recomputed subject-held-out fold {fold.outer.fold_id} has development/test component leakage"
            )
        for row_id in test["row_id"].astype(str):
            if row_id in expected_fold:
                blockers.append("predictor opportunity appears in multiple recomputed test folds")
            expected_fold[row_id] = fold.outer.fold_id
    working["recomputed_test_fold"] = working["row_id"].map(expected_fold)
    if working["recomputed_test_fold"].isna().any():
        blockers.append("not every predictor opportunity has one recomputed test fold")
    if working.groupby("participant_id")["recomputed_test_fold"].nunique().gt(1).any():
        blockers.append("participant coverage is inconsistent across recomputed test folds")
    if working.groupby("recomputed_component")["recomputed_test_fold"].nunique().gt(1).any():
        blockers.append("family/twin component coverage is inconsistent across recomputed test folds")
    summary_required = {
        "analysis_mode", "analysis_status", "outer_fold",
        "train_n_rows", "test_n_rows", "dropped_n_rows",
    }
    missing_summary = sorted(summary_required - set(split_audit.columns))
    if missing_summary:
        blockers.append("split audit summary omits Task 3 columns: " + ", ".join(missing_summary))
    else:
        summary = split_audit.loc[split_audit["analysis_mode"].eq("subject_held_out")]
        if len(summary) != outer_folds:
            blockers.append("split audit summary has an unexpected fold count")
        observed_folds: set[int] = set()
        for _, row in summary.iterrows():
            try:
                fold_id = _strict_fold(row["outer_fold"], "split summary outer_fold")
                train_n = _strict_integer(row["train_n_rows"], "split summary train_n_rows")
                test_n = _strict_integer(row["test_n_rows"], "split summary test_n_rows")
                dropped_n = _strict_integer(row["dropped_n_rows"], "split summary dropped_n_rows")
            except ValueError as error:
                blockers.append(str(error))
                continue
            observed_folds.add(fold_id)
            if fold_id not in range(outer_folds):
                blockers.append("split audit summary outer_fold is outside the frozen range")
                continue
            expected = nested[fold_id].outer
            if (
                train_n != len(expected.train_positions)
                or test_n != len(expected.test_positions)
                or dropped_n != len(expected.dropped_positions)
            ):
                blockers.append(
                    f"fold {fold_id} split summary does not match recomputed production split"
                )
            if train_n <= 0 or test_n <= 0 or row["analysis_status"] != "completed":
                blockers.append(f"fold {fold_id} split summary is empty or incomplete")
        if observed_folds != set(range(outer_folds)):
            blockers.append("split audit summary does not contain every frozen outer fold")
    prediction_required = {
        "analysis_mode", "row_id", "participant_id", "meal_id",
        "inference_cluster_id", "endpoint", "outer_fold", "comparator",
    }
    missing_predictions = sorted(prediction_required - set(predictions.columns))
    if missing_predictions:
        return blockers + [
            "predictions omit Task 3 columns: " + ", ".join(missing_predictions)
        ]
    primary = predictions.loc[
        predictions["analysis_mode"].eq("subject_held_out")
        & predictions["endpoint"].isin(_PRIMARY_ENDPOINTS)
    ].copy()
    if primary.empty:
        return blockers + ["predictions contain no primary subject-held-out rows"]
    if primary.duplicated(["endpoint", "comparator", "row_id"]).any():
        blockers.append("predictions contain duplicate endpoint/comparator/row_id rows")
    lookup = working.set_index("row_id")
    for row in primary.itertuples(index=False):
        row_id = str(row.row_id)
        if row_id not in expected_fold:
            blockers.append("prediction row_id is outside the recomputed predictor opportunities")
            continue
        try:
            observed_fold = _strict_fold(row.outer_fold, "prediction outer_fold")
        except ValueError as error:
            blockers.append(str(error))
            continue
        if observed_fold != expected_fold[row_id]:
            blockers.append(f"prediction row {row_id} is not in its recomputed test fold")
        expected_row = lookup.loc[row_id]
        if str(row.participant_id) != str(expected_row["participant_id"]):
            blockers.append(f"prediction row {row_id} participant does not match predictor frame")
        if str(row.meal_id) != str(expected_row["meal_id"]):
            blockers.append(f"prediction row {row_id} meal does not match predictor frame")
        if str(row.inference_cluster_id) != str(expected_row["recomputed_component"]):
            blockers.append(
                f"prediction row {row_id} does not use the recomputed family/twin component"
            )
    if outcome_frame.duplicated(["participant_id", "meal_id"]).any():
        blockers.append("trusted outcomes contain duplicate person-meal keys")
        return blockers
    joined = working.merge(
        outcome_frame,
        on=["participant_id", "meal_id"],
        how="left",
        validate="one_to_one",
    )
    if set(primary["comparator"].astype(str)) != set(REQUIRED_COMPARATORS):
        blockers.append("primary predictions do not contain every locked Task 3 comparator")
    for endpoint in _PRIMARY_ENDPOINTS:
        if endpoint not in joined:
            blockers.append(f"trusted outcomes omit primary endpoint {endpoint}")
            continue
        finite = np.isfinite(pd.to_numeric(joined[endpoint], errors="coerce"))
        eligible = set(joined.loc[finite, "row_id"].astype(str))
        for comparator in REQUIRED_COMPARATORS:
            observed = set(
                primary.loc[
                    primary["endpoint"].eq(endpoint)
                    & primary["comparator"].eq(comparator),
                    "row_id",
                ].astype(str)
            )
            if observed != eligible:
                blockers.append(
                    f"{endpoint}/{comparator} prediction coverage does not match finite-outcome predictor opportunities"
                )
    return blockers


def _status_blockers(status: pd.DataFrame) -> list[str]:
    required = {"analysis_mode", "endpoint", "analysis_status", "endpoint_role"}
    missing = sorted(required - set(status.columns))
    if missing:
        return ["analysis status omits Task 3 columns: " + ", ".join(missing)]
    primary = status.loc[
        status["analysis_mode"].eq("subject_held_out")
        & status["endpoint"].isin(_PRIMARY_ENDPOINTS)
        & status["endpoint_role"].eq("primary")
    ]
    if len(primary) != 2 or set(primary["endpoint"]) != set(_PRIMARY_ENDPOINTS):
        return ["analysis status does not contain exactly two frozen primary endpoint rows"]
    if not primary["analysis_status"].eq("completed").all():
        return ["primary analysis status is not completed"]
    return []


def evaluate_evidence_gate(
    paths: EvidenceGateArtifactPaths | None = None,
) -> EvidenceGateOutcome:
    """Revalidate live Task 3 inputs and evaluate only exported immutable artifacts."""

    if paths is None:
        return _fail(
            _current_missing_blockers(),
            source_state="absent_real_validation_artifacts",
        )
    if not isinstance(paths, EvidenceGateArtifactPaths):
        raise TypeError("production evidence gate accepts only EvidenceGateArtifactPaths")
    blockers: list[str] = []
    checks: list[dict[str, object]] = []
    try:
        manifest_bytes = _read_regular_bytes(paths.result_run_manifest, "result-run manifest")
        manifest = _json_object(manifest_bytes, "result-run manifest")
    except ValueError as error:
        return _fail([str(error)])
    run_id = manifest.get("run_id")
    try:
        approval = _trusted_approval(manifest_bytes, run_id)
        if approval is None:
            blockers.append("result run is not independently approved by the trusted registry")
        elif approval == "mismatch":
            blockers.append("result-run manifest hash does not match the trusted registry")
    except ValueError as error:
        blockers.append(str(error))
    if manifest.get("schema_version") != "direct-validity-result-run-v2":
        blockers.append("result-run manifest is not the Task 3 exporter schema")
    if manifest.get("evidence_role") != _DIRECT_ROLE or manifest.get("data_class") != _DIRECT_DATA_CLASS:
        blockers.append("supporting evidence role/data class cannot upgrade direct external validity")
    if not isinstance(run_id, str) or not run_id:
        blockers.append("result-run manifest run_id is invalid")
    try:
        loaded = load_locked_outcomes_for_benchmark(
            paths.controlled_outcome_source_id,
            manifest_path=paths.method_lock_manifest,
            method_lock_paths=paths.method_lock_artifacts,
        )
    except Exception as error:
        blockers.append(
            f"trusted method-lock/outcome loader contract failed: {type(error).__name__}: {error}"
        )
        loaded = None
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        artifacts = {}
        blockers.append("result-run manifest artifact hash map is missing")
    frames: dict[str, pd.DataFrame] = {}
    for name, path in (
        ("paired_metrics", paths.paired_metrics),
        ("predictions", paths.predictions),
        ("split_audit", paths.split_audit),
        ("analysis_status", paths.analysis_status),
    ):
        try:
            raw = _read_regular_bytes(path, name)
            record = artifacts.get(name)
            expected = record.get("sha256") if isinstance(record, Mapping) else None
            if not _is_digest(expected) or sha256(raw).hexdigest() != expected:
                blockers.append(f"{name} hash does not match result-run manifest")
            frames[name] = pd.read_csv(BytesIO(raw))
        except Exception as error:
            blockers.append(f"{name} live reread failed: {error}")
    if loaded is not None:
        try:
            live_binding = VerifiedRunBinding.from_locked_outcome(str(run_id), loaded)
            lock_path_sha = sha256(
                _read_regular_bytes(paths.method_lock_manifest, "method-lock manifest")
            ).hexdigest()
            if lock_path_sha != live_binding.method_lock_manifest_sha256:
                blockers.append("method-lock manifest live hash does not match trusted loader")
            for key, expected in live_binding.payload().items():
                if manifest.get(key) != expected:
                    blockers.append(f"result-run {key} does not match live trusted inputs")
            if manifest.get("run_binding_sha256") != live_binding.run_binding_sha256:
                blockers.append("result-run binding is not independently reproducible")
            for name, frame in frames.items():
                if "run_id" not in frame or not frame["run_id"].eq(run_id).all():
                    blockers.append(f"{name} rows are not bound to the same run_id")
                if (
                    "run_binding_sha256" not in frame
                    or not frame["run_binding_sha256"].eq(live_binding.run_binding_sha256).all()
                ):
                    blockers.append(f"{name} rows are not bound to the same run/manifest")
                if name != "predictions":
                    blockers.extend(_task3_provenance_blockers(frame, name, live_binding))
            paired = frames.get("paired_metrics")
            if paired is not None:
                checks, endpoint_blockers = _validate_endpoint_rows(paired)
                blockers.extend(endpoint_blockers)
            status = frames.get("analysis_status")
            if status is not None:
                blockers.extend(_status_blockers(status))
            predictions = frames.get("predictions")
            split_audit = frames.get("split_audit")
            if predictions is not None and split_audit is not None:
                blockers.extend(
                    _split_and_prediction_blockers(
                        predictions,
                        split_audit,
                        loaded.verified_lock.predictor_frame,
                        loaded.outcome.frame,
                        loaded.verified_lock.config,
                    )
                )
        except (TypeError, ValueError, KeyError, AttributeError) as error:
            blockers.append(f"live Task 3 result validation failed closed: {error}")
    if blockers:
        return _fail(blockers, checks)
    if len(checks) != 2 or not all(row["passed"] for row in checks):
        return _fail(["frozen primary endpoint checks are incomplete"], checks)
    return EvidenceGateOutcome(
        tier=DIRECT_EXTERNAL_VALIDITY,
        passed=True,
        blockers=(),
        endpoint_checks=pd.DataFrame.from_records(checks),
        allowed_claims=(
            "direct_external_validity_for_locked_primary_endpoints",
            "computational_feasibility",
        ),
        prohibited_claims=("clinical_utility", "causal_dietary_effect"),
        source_state="verified_direct_validation_artifacts",
    )


def evaluate_testing_evidence(
    outcome_manifest: Mapping[str, object] | None,
    paired_primary_results: pd.DataFrame | None,
) -> EvidenceGateOutcome:
    """Inspect in-memory fixtures without any possibility of a claim upgrade."""

    blockers = ["in-memory evidence evaluator is testing-only and cannot upgrade claims"]
    if not isinstance(outcome_manifest, Mapping):
        blockers.append("testing outcome manifest is missing")
    elif outcome_manifest.get("evidence_role") != _DIRECT_ROLE:
        blockers.append("supporting evidence cannot upgrade direct external validity")
    if not isinstance(paired_primary_results, pd.DataFrame) or paired_primary_results.empty:
        blockers.append("testing paired results are missing or empty")
    return EvidenceGateOutcome(
        tier=TESTING_ONLY_NO_CLAIM_UPGRADE,
        passed=False,
        blockers=tuple(blockers),
        endpoint_checks=pd.DataFrame(),
        allowed_claims=("testing_only",),
        prohibited_claims=_PROHIBITED_COMPUTATIONAL_CLAIMS,
        source_state="testing_only_in_memory",
    )


__all__ = [
    "COMPUTATIONAL_FEASIBILITY",
    "DIRECT_EXTERNAL_VALIDITY",
    "TESTING_ONLY_NO_CLAIM_UPGRADE",
    "EvidenceGateArtifactPaths",
    "EvidenceGateOutcome",
    "ExportedBenchmarkArtifacts",
    "VerifiedRunBinding",
    "evaluate_evidence_gate",
    "evaluate_testing_evidence",
    "export_benchmark_result",
]
