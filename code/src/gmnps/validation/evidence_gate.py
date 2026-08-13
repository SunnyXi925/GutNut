"""Path-only, fail-closed gate for direct person-by-meal response validity.

Only independently approved immutable result artifacts can reach the direct
validity tier.  The in-memory helper is explicitly testing-only and cannot
upgrade claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from math import isfinite
import os
from pathlib import Path
import re
import stat
from typing import Mapping

import pandas as pd

from gmnps.data_sources.predict_zoe_registry import get_controlled_outcome_grant
from gmnps.validation.method_lock_gate import MethodLockArtifactPaths
from gmnps.validation.person_meal_benchmark import load_locked_outcomes_for_benchmark


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
_PROHIBITED_COMPUTATIONAL_CLAIMS = (
    "transforms_postprandial_response",
    "precision_ready",
    "clinical_validity",
    "external_validity",
    "direct_response_validity",
    "causal_dietary_effect",
)


@dataclass(frozen=True)
class EvidenceGateArtifactPaths:
    """Immutable filesystem inputs required by the production gate."""

    method_lock_manifest: Path
    method_lock_artifacts: MethodLockArtifactPaths
    result_run_manifest: Path
    paired_results: Path
    split_audit: Path
    analysis_status: Path
    controlled_outcome_source_id: str

    def __post_init__(self) -> None:
        for name in (
            "method_lock_manifest",
            "result_run_manifest",
            "paired_results",
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
    gate_version: str = "direct-response-evidence-gate-v2"


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


def _read_regular_bytes(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing or unreadable") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be an immutable regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{label} changed during immutable open")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
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


def _fail(blockers: list[str], checks: list[dict[str, object]] | None = None):
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
        "paired primary results path is missing",
        "hashed detailed split audit path is missing",
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


def _binding_payload(
    run_id: str,
    lock_sha: str,
    source_id: str,
    outcome_sha: str,
    lock_manifest: Mapping[str, object],
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "method_lock_manifest_sha256": lock_sha,
        "outcome_source_id": source_id,
        "outcome_sha256": outcome_sha,
        **{field: lock_manifest.get(field) for field in _LOCK_FIELDS},
    }


def _validate_endpoint_rows(rows: pd.DataFrame) -> tuple[list[dict[str, object]], list[str]]:
    blockers: list[str] = []
    checks: list[dict[str, object]] = []
    required = {
        "endpoint", "analysis_mode", "model", "reference", "metric",
        "estimate_delta", "ci_lower", "ci_upper", "ci_method",
        "bootstrap_replicates_requested", "bootstrap_replicates_valid",
        "permutation_replicates_requested", "permutation_replicates_valid",
        "permutation_p_value", "holm_adjusted_p_value", "test_role",
        "correction_family", "inference_status",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        return checks, ["paired primary results omit columns: " + ", ".join(missing)]
    primary = rows.loc[
        rows["analysis_mode"].eq("subject_held_out")
        & rows["model"].eq("locked_attribute_gmnps")
        & rows["reference"].eq("fcs_microbiome")
        & rows["metric"].eq("rmse")
        & rows["test_role"].eq("primary")
    ].copy()
    if len(primary) != 2 or set(primary["endpoint"]) != set(_PRIMARY_ENDPOINTS):
        return checks, [
            "paired results do not contain exactly the two frozen subject-held-out primary RMSE rows"
        ]
    raw_p = pd.to_numeric(primary["permutation_p_value"], errors="coerce")
    order = raw_p.sort_values(kind="stable").index
    adjusted: dict[object, float] = {}
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, float(raw_p.loc[index]) * (len(order) - rank))
        running = max(running, value)
        adjusted[index] = running
    for endpoint in _PRIMARY_ENDPOINTS:
        row = primary.loc[primary["endpoint"].eq(endpoint)].iloc[0]
        supplied = pd.to_numeric(pd.Series([row["holm_adjusted_p_value"]]), errors="coerce").iloc[0]
        recomputed = adjusted.get(row.name, float("nan"))
        estimate = float(row["estimate_delta"])
        lower = float(row["ci_lower"])
        upper = float(row["ci_upper"])
        bootstrap_requested = int(row["bootstrap_replicates_requested"])
        bootstrap_valid = int(row["bootstrap_replicates_valid"])
        permutation_requested = int(row["permutation_replicates_requested"])
        permutation_valid = int(row["permutation_replicates_valid"])
        direction = isfinite(estimate) and estimate < 0
        ci = (
            isfinite(lower) and isfinite(upper) and lower <= upper < 0
            and row["ci_method"] == _CI_METHOD
        )
        multiplicity = (
            isfinite(recomputed)
            and recomputed <= _ALPHA
            and isfinite(float(supplied))
            and abs(float(supplied) - recomputed) <= 1e-12
            and row["correction_family"] == _PRIMARY_FAMILY
        )
        valid = (
            bootstrap_requested == _MINIMUM_REPLICATES
            and permutation_requested == _MINIMUM_REPLICATES
            and bootstrap_valid / bootstrap_requested >= _MINIMUM_VALID_FRACTION
            and permutation_valid / permutation_requested >= _MINIMUM_VALID_FRACTION
        )
        complete = row["inference_status"] == "completed"
        checks.append(
            {
                "endpoint": endpoint,
                "estimate_delta": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "raw_permutation_p_value": float(raw_p.loc[row.name]),
                "recomputed_holm_adjusted_p_value": recomputed,
                "direction_passed": direction,
                "ci_passed": ci,
                "multiplicity_passed": multiplicity,
                "valid_replicates_passed": valid,
                "specification_passed": complete,
                "passed": all((direction, ci, multiplicity, valid, complete)),
            }
        )
        if not direction:
            blockers.append(f"{endpoint} RMSE paired improvement is not in the required direction")
        if not ci:
            blockers.append(f"{endpoint} paired component-bootstrap CI/method is invalid")
        if not multiplicity:
            blockers.append(f"{endpoint} does not pass the recomputed Holm test")
        if not valid:
            blockers.append(f"{endpoint} does not meet the frozen valid-resample threshold")
        if not complete:
            blockers.append(f"{endpoint} primary inference is not completed")
    return checks, blockers


def _split_blockers(split: pd.DataFrame, outcome_participants: set[str]) -> list[str]:
    required = {
        "analysis_mode", "outer_fold", "participant_id",
        "family_twin_component_id", "partition",
    }
    missing = sorted(required - set(split.columns))
    if missing:
        return ["detailed split audit omits columns: " + ", ".join(missing)]
    relevant = split.loc[split["analysis_mode"].eq("subject_held_out")]
    if relevant.empty:
        return ["detailed split audit has no subject-held-out rows"]
    blockers: list[str] = []
    for fold, frame in relevant.groupby("outer_fold", dropna=False):
        development = frame.loc[frame["partition"].eq("development")]
        test = frame.loc[frame["partition"].eq("test")]
        participant_overlap = set(development["participant_id"].astype(str)) & set(
            test["participant_id"].astype(str)
        )
        component_overlap = set(
            development["family_twin_component_id"].astype(str)
        ) & set(test["family_twin_component_id"].astype(str))
        if participant_overlap:
            blockers.append(f"subject-held-out fold {fold} participant leakage")
        if component_overlap:
            blockers.append(f"subject-held-out fold {fold} family/twin component leakage")
        audited = set(frame["participant_id"].astype(str))
        if audited != outcome_participants:
            blockers.append(f"subject-held-out fold {fold} participant universe does not match loaded outcomes")
    return blockers


def evaluate_evidence_gate(
    paths: EvidenceGateArtifactPaths | None = None,
) -> EvidenceGateOutcome:
    """Evaluate only immutable production artifacts; in-memory evidence is rejected."""

    if paths is None:
        return _fail(_current_missing_blockers())
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
        blockers.append(f"trusted method-lock/outcome loader contract failed: {type(error).__name__}: {error}")
        loaded = None

    artifact_frames: dict[str, object] = {}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        blockers.append("result-run manifest artifact hash map is missing")
        artifacts = {}
    for name, path, kind in (
        ("paired_results", paths.paired_results, "csv"),
        ("split_audit", paths.split_audit, "csv"),
        ("analysis_status", paths.analysis_status, "json"),
    ):
        try:
            raw = _read_regular_bytes(path, name)
            expected = artifacts.get(name, {}).get("sha256") if isinstance(artifacts.get(name), dict) else None
            if not _is_digest(expected) or sha256(raw).hexdigest() != expected:
                blockers.append(f"{name} hash does not match result-run manifest")
            artifact_frames[name] = (
                pd.read_csv(BytesIO(raw)) if kind == "csv" else _json_object(raw, name)
            )
        except Exception as error:
            blockers.append(f"{name} live reread failed: {error}")

    if loaded is not None:
        lock = loaded.verified_lock
        lock_manifest = lock.manifest
        lock_path_sha = sha256(
            _read_regular_bytes(paths.method_lock_manifest, "method-lock manifest")
        ).hexdigest()
        if lock_path_sha != lock.manifest_sha256:
            blockers.append("method-lock manifest live hash does not match trusted loader")
        if manifest.get("method_lock_manifest_sha256") != lock.manifest_sha256:
            blockers.append("result-run method-lock manifest hash does not match trusted loader")
        if manifest.get("outcome_source_id") != loaded.outcome.source_id or paths.controlled_outcome_source_id != loaded.outcome.source_id:
            blockers.append("outcome source binding does not match trusted loader")
        if manifest.get("outcome_sha256") != loaded.outcome.sha256:
            blockers.append("outcome artifact hash does not match trusted loader")
        for field in _LOCK_FIELDS:
            if manifest.get(field) != lock_manifest.get(field):
                blockers.append(f"result-run {field} does not match live method lock")
        if isinstance(run_id, str):
            expected_binding = _canonical_hash(
                _binding_payload(
                    run_id,
                    lock.manifest_sha256,
                    loaded.outcome.source_id,
                    loaded.outcome.sha256,
                    lock_manifest,
                )
            )
            if manifest.get("run_binding_sha256") != expected_binding:
                blockers.append("result-run binding is not independently reproducible")
            for name, value in artifact_frames.items():
                if isinstance(value, pd.DataFrame):
                    if "run_id" not in value or not value["run_id"].eq(run_id).all():
                        blockers.append(f"{name} rows are not bound to the same run_id")
                    if "run_binding_sha256" not in value or not value["run_binding_sha256"].eq(expected_binding).all():
                        blockers.append(f"{name} rows are not bound to the same run/manifest")
                elif isinstance(value, dict):
                    if value.get("run_id") != run_id or value.get("run_binding_sha256") != expected_binding:
                        blockers.append(f"{name} is not bound to the same run/manifest")
            status = artifact_frames.get("analysis_status")
            if not isinstance(status, dict) or status.get("analysis_status") != "completed":
                blockers.append("analysis status is not completed")
            paired = artifact_frames.get("paired_results")
            if isinstance(paired, pd.DataFrame):
                try:
                    checks, endpoint_blockers = _validate_endpoint_rows(paired)
                    blockers.extend(endpoint_blockers)
                except (TypeError, ValueError, OverflowError) as error:
                    blockers.append(f"paired primary results are numerically invalid: {error}")
            split = artifact_frames.get("split_audit")
            outcome_frame = loaded.outcome.frame
            if isinstance(split, pd.DataFrame) and isinstance(outcome_frame, pd.DataFrame):
                if "participant_id" not in outcome_frame:
                    blockers.append("trusted outcome table omits participant_id")
                else:
                    blockers.extend(
                        _split_blockers(
                            split,
                            set(outcome_frame["participant_id"].astype(str)),
                        )
                    )

    if blockers:
        return _fail(blockers, checks)
    if len(checks) != len(_PRIMARY_ENDPOINTS) or not all(row["passed"] for row in checks):
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
    )


__all__ = [
    "COMPUTATIONAL_FEASIBILITY",
    "DIRECT_EXTERNAL_VALIDITY",
    "TESTING_ONLY_NO_CLAIM_UPGRADE",
    "EvidenceGateArtifactPaths",
    "EvidenceGateOutcome",
    "evaluate_evidence_gate",
    "evaluate_testing_evidence",
]
