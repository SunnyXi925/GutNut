"""Machine-enforced claim policy bound to an evidence-gate decision."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from gmnps.validation.evidence_gate import EvidenceGateOutcome


_FORBIDDEN_PATTERNS = (
    r"transform(s|ed|ing)?\s+postprandial response",
    r"precision[- ]ready",
    r"clinical validity",
    r"external validity",
    r"direct response validity",
    r"causal dietary effect",
)


@dataclass(frozen=True)
class ClaimViolation:
    path: Path
    pattern: str


def canonical_payload_sha256(payload: dict[str, object]) -> str:
    core = {key: value for key, value in payload.items() if key != "payload_sha256"}
    return sha256(
        json.dumps(
            core,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def build_claim_policy_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "claim-policy-v1",
        "tier": "computational_feasibility",
        "allowed_claims": [
            "computational feasibility",
            "correctly specified synthetic positive-control",
            "biological consistency",
            "mechanistic consistency",
        ],
        "forbidden_patterns": list(_FORBIDDEN_PATTERNS),
        "scope": "Phase 3 manuscript and build inputs",
        "phase3_must_consume": True,
    }
    payload["payload_sha256"] = canonical_payload_sha256(payload)
    return payload


def write_claim_restriction_artifacts(
    output_directory: str | Path,
    outcome: "EvidenceGateOutcome",
) -> dict[str, Path]:
    """Deterministically materialize a gate decision and its bound claim policy."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    policy = build_claim_policy_payload()
    policy_bytes = _json_bytes(policy)
    policy_path = output / "claim_policy.json"
    policy_path.write_bytes(policy_bytes)
    decision: dict[str, object] = {
        "schema_version": "evidence-gate-decision-v1",
        "gate_version": outcome.gate_version,
        "tier": outcome.tier,
        "passed": outcome.passed,
        "blockers": list(outcome.blockers),
        "allowed_claims": list(outcome.allowed_claims),
        "forbidden_claims": list(outcome.prohibited_claims),
        "claim_policy_file": policy_path.name,
        "claim_policy_file_sha256": sha256(policy_bytes).hexdigest(),
        "source_state": "absent_real_validation_artifacts",
        "manuscript_checked_or_revised": False,
    }
    decision["payload_sha256"] = canonical_payload_sha256(decision)
    decision_path = output / "current_gate_decision.json"
    decision_path.write_bytes(_json_bytes(decision))
    return {"decision": decision_path, "policy": policy_path}


def _load_verified_json(path: Path, label: str) -> tuple[bytes, dict[str, object]]:
    raw = Path(path).read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    expected = payload.get("payload_sha256")
    if not isinstance(expected, str) or expected != canonical_payload_sha256(payload):
        raise ValueError(f"{label} payload hash is invalid")
    return raw, payload


def check_claim_inputs(
    decision_path: str | Path,
    policy_path: str | Path,
    input_paths: Iterable[str | Path],
) -> tuple[ClaimViolation, ...]:
    """Return forbidden-claim matches, failing first on policy/decision tamper."""

    policy_raw, policy = _load_verified_json(Path(policy_path), "claim policy")
    _, decision = _load_verified_json(Path(decision_path), "gate decision")
    if decision.get("claim_policy_file_sha256") != sha256(policy_raw).hexdigest():
        raise ValueError("claim policy hash does not match gate decision")
    if decision.get("tier") != policy.get("tier"):
        raise ValueError("claim policy tier does not match gate decision")
    patterns = policy.get("forbidden_patterns")
    if not isinstance(patterns, list) or not all(
        isinstance(pattern, str) and pattern for pattern in patterns
    ):
        raise ValueError("claim policy forbidden_patterns is invalid")
    violations: list[ClaimViolation] = []
    for value in input_paths:
        path = Path(value)
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                violations.append(ClaimViolation(path=path, pattern=pattern))
    return tuple(violations)


__all__ = [
    "ClaimViolation",
    "build_claim_policy_payload",
    "canonical_payload_sha256",
    "check_claim_inputs",
    "write_claim_restriction_artifacts",
]
