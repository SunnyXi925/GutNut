"""Fail-closed evidence gate for direct person-by-meal response validity."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import pandas as pd


COMPUTATIONAL_FEASIBILITY = "computational_feasibility"
DIRECT_EXTERNAL_VALIDITY = "direct_external_validity"

_PRIMARY_ENDPOINTS = ("glucose_iAUC_2h", "tg_6h_rise")
_DIRECT_ROLE = "direct_validation"
_DIRECT_DATA_CLASS = "real_observed_participant_meal_outcomes"
_PRIMARY_FAMILY = (
    "two_primary_endpoints_locked_gmnps_vs_fcs_microbiome_rmse"
)
_MINIMUM_REPLICATES = 2_000
_MINIMUM_VALID_FRACTION = 0.90
_ALPHA = 0.05
_HEX = frozenset("0123456789abcdef")
_SUPPORTING_ROLES = frozenset(
    {
        "synthetic_identifiability_stress_test",
        "biological_consistency",
        "mechanistic_consistency",
        "predictor_reconstruction",
    }
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
class EvidenceGateOutcome:
    """Auditable claim tier and every reason that prevented an upgrade."""

    tier: str
    passed: bool
    blockers: tuple[str, ...]
    endpoint_checks: pd.DataFrame
    allowed_claims: tuple[str, ...]
    prohibited_claims: tuple[str, ...]
    gate_version: str = "direct-response-evidence-gate-v1"


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _subject_ids(value: object, label: str, blockers: list[str]) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        blockers.append(f"{label} are missing or not an identifier sequence")
        return ()
    identifiers = tuple(value)
    if not identifiers or any(
        not isinstance(item, str) or not item.strip() for item in identifiers
    ):
        blockers.append(f"{label} are missing or contain blank identifiers")
        return ()
    if len(set(identifiers)) != len(identifiers):
        blockers.append(f"{label} contain duplicate identifiers")
    return identifiers


def _manifest_blockers(manifest: Mapping[str, object]) -> list[str]:
    blockers: list[str] = []
    role = manifest.get("evidence_role")
    data_class = manifest.get("data_class")
    if role in _SUPPORTING_ROLES or data_class != _DIRECT_DATA_CLASS:
        blockers.append(
            f"supporting evidence role/data class {role!r}/{data_class!r} cannot "
            "upgrade direct external validity"
        )
    elif role != _DIRECT_ROLE:
        blockers.append("evidence_role is not direct_validation")
    if manifest.get("provenance_status") != "eligible_verified":
        blockers.append("outcome provenance is not eligible_verified")
    if manifest.get("observed_participant_meal_outcomes") is not True:
        blockers.append("outcomes are not verified observed participant-by-meal records")
    if manifest.get("microbiome_linkage_verified") is not True:
        blockers.append("participant-to-microbiome linkage is not verified")
    if manifest.get("method_lock_status") != "passed":
        blockers.append("run-level method lock did not pass")
    for field in ("method_lock_manifest_sha256", "outcome_manifest_sha256"):
        if not _is_sha256(manifest.get(field)):
            blockers.append(f"{field} is missing or is not a lowercase SHA-256 digest")
    development = _subject_ids(
        manifest.get("development_subject_ids"),
        "development subject identifiers",
        blockers,
    )
    test = _subject_ids(
        manifest.get("test_subject_ids"), "test subject identifiers", blockers
    )
    overlap = sorted(set(development) & set(test))
    if overlap:
        blockers.append(
            "development and test subject identifiers overlap: " + ", ".join(overlap)
        )
    return blockers


def _finite(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _endpoint_row(endpoint: str, rows: pd.DataFrame) -> tuple[dict[str, object], list[str]]:
    expected = rows.loc[
        rows.get("endpoint", pd.Series(index=rows.index, dtype=object)).eq(endpoint)
        & rows.get("analysis_mode", pd.Series(index=rows.index, dtype=object)).eq(
            "subject_held_out"
        )
        & rows.get("model", pd.Series(index=rows.index, dtype=object)).eq(
            "locked_attribute_gmnps"
        )
        & rows.get("reference", pd.Series(index=rows.index, dtype=object)).eq(
            "fcs_microbiome"
        )
        & rows.get("metric", pd.Series(index=rows.index, dtype=object)).eq("rmse")
    ]
    blockers: list[str] = []
    if len(expected) != 1:
        blockers.append(
            f"{endpoint} does not have exactly one locked primary row "
            "for subject-held-out RMSE versus fcs_microbiome"
        )
        return {
            "endpoint": endpoint,
            "direction_passed": False,
            "ci_passed": False,
            "multiplicity_passed": False,
            "valid_replicates_passed": False,
            "specification_passed": False,
            "passed": False,
        }, blockers

    row = expected.iloc[0]
    estimate = _finite(row.get("estimate_delta"))
    ci_lower = _finite(row.get("ci_lower"))
    ci_upper = _finite(row.get("ci_upper"))
    adjusted = _finite(row.get("adjusted_p_value"))
    bootstrap_requested = _finite(row.get("bootstrap_replicates_requested"))
    bootstrap_valid = _finite(row.get("bootstrap_replicates_valid"))
    permutation_requested = _finite(row.get("permutation_replicates_requested"))
    permutation_valid = _finite(row.get("permutation_replicates_valid"))
    direction = estimate is not None and estimate < 0.0
    ci_passed = (
        ci_lower is not None
        and ci_upper is not None
        and ci_lower <= ci_upper < 0.0
        and isinstance(row.get("ci_method"), str)
        and row.get("ci_method", "").endswith("_95")
    )
    multiplicity = (
        row.get("test_role") == "primary"
        and row.get("multiplicity_method") == "holm"
        and row.get("correction_family") == _PRIMARY_FAMILY
        and adjusted is not None
        and adjusted <= _ALPHA
    )
    valid_replicates = (
        bootstrap_requested == _MINIMUM_REPLICATES
        and permutation_requested == _MINIMUM_REPLICATES
        and bootstrap_valid is not None
        and permutation_valid is not None
        and bootstrap_valid / bootstrap_requested >= _MINIMUM_VALID_FRACTION
        and permutation_valid / permutation_requested >= _MINIMUM_VALID_FRACTION
        and row.get("bootstrap_minimum_valid_fraction") == _MINIMUM_VALID_FRACTION
        and row.get("permutation_minimum_valid_fraction") == _MINIMUM_VALID_FRACTION
    )
    specification = row.get("inference_status") == "completed"
    checks = {
        "endpoint": endpoint,
        "estimate_delta": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "adjusted_p_value": adjusted,
        "direction_passed": direction,
        "ci_passed": ci_passed,
        "multiplicity_passed": multiplicity,
        "valid_replicates_passed": valid_replicates,
        "specification_passed": specification,
        "passed": all(
            (direction, ci_passed, multiplicity, valid_replicates, specification)
        ),
    }
    if not direction:
        blockers.append(f"{endpoint} RMSE paired improvement is not in the required direction")
    if not ci_passed:
        blockers.append(f"{endpoint} 95% paired CI does not exclude zero in the improvement direction")
    if not multiplicity:
        blockers.append(f"{endpoint} does not pass the frozen Holm-adjusted primary test")
    if not valid_replicates:
        blockers.append(f"{endpoint} does not meet the frozen valid-resample threshold")
    if not specification:
        blockers.append(f"{endpoint} primary inference_status is not completed")
    return checks, blockers


def evaluate_evidence_gate(
    outcome_manifest: Mapping[str, object] | None = None,
    paired_primary_results: pd.DataFrame | None = None,
) -> EvidenceGateOutcome:
    """Return direct validity only after every frozen real-data check passes.

    Synthetic, aggregate, knowledge-graph and predictor-only evidence are always
    supporting evidence and cannot satisfy this gate.
    """

    blockers: list[str] = []
    if outcome_manifest is None:
        blockers.append(
            "eligible real observed participant-by-meal outcome manifest is missing"
        )
    elif not isinstance(outcome_manifest, Mapping):
        blockers.append("outcome manifest is not a mapping")
    else:
        blockers.extend(_manifest_blockers(outcome_manifest))

    endpoint_rows: list[dict[str, object]] = []
    if paired_primary_results is None:
        blockers.append("locked subject-held-out primary paired-results table is missing")
    elif not isinstance(paired_primary_results, pd.DataFrame):
        blockers.append("paired primary results are not a pandas DataFrame")
    elif paired_primary_results.empty:
        blockers.append("locked subject-held-out primary paired-results table is empty")
    else:
        for endpoint in _PRIMARY_ENDPOINTS:
            checks, endpoint_blockers = _endpoint_row(endpoint, paired_primary_results)
            endpoint_rows.append(checks)
            blockers.extend(endpoint_blockers)

    endpoint_checks = pd.DataFrame.from_records(endpoint_rows)
    passed = not blockers and len(endpoint_checks) == len(_PRIMARY_ENDPOINTS)
    if passed:
        return EvidenceGateOutcome(
            tier=DIRECT_EXTERNAL_VALIDITY,
            passed=True,
            blockers=(),
            endpoint_checks=endpoint_checks,
            allowed_claims=(
                "direct_external_validity_for_locked_primary_endpoints",
                "computational_feasibility",
            ),
            prohibited_claims=("clinical_utility", "causal_dietary_effect"),
        )
    return EvidenceGateOutcome(
        tier=COMPUTATIONAL_FEASIBILITY,
        passed=False,
        blockers=tuple(blockers),
        endpoint_checks=endpoint_checks,
        allowed_claims=(
            "computational_feasibility",
            "synthetic_identifiability_stress_test",
            "biological_consistency",
            "mechanistic_consistency",
        ),
        prohibited_claims=_PROHIBITED_COMPUTATIONAL_CLAIMS,
    )


__all__ = [
    "COMPUTATIONAL_FEASIBILITY",
    "DIRECT_EXTERNAL_VALIDITY",
    "EvidenceGateOutcome",
    "evaluate_evidence_gate",
]
