"""Machine-enforced claims bound to the repository's path-only evidence gate."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from gmnps.validation.evidence_gate import (
        EvidenceGateArtifactPaths,
        EvidenceGateOutcome,
    )


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_TRUSTED_DECISION_RELATIVE = Path(
    "results/phase2/evidence-gate/current_gate_decision.json"
)
_TRUSTED_POLICY_RELATIVE = Path("results/phase2/evidence-gate/claim_policy.json")
_TRUSTED_REGISTRY_RELATIVE = Path("code/src/configs/claim_policy_registry.json")
_TRUSTED_DECISION_PATH = _REPOSITORY_ROOT / _TRUSTED_DECISION_RELATIVE
_TRUSTED_POLICY_PATH = _REPOSITORY_ROOT / _TRUSTED_POLICY_RELATIVE
_TRUSTED_POLICY_REGISTRY_PATH = _REPOSITORY_ROOT / _TRUSTED_REGISTRY_RELATIVE

_FORBIDDEN_PATTERNS = (
    r"transform(s|ed|ing)?\s+postprandial response",
    r"precision[- ]ready",
    r"clinical validity",
    r"clinical utility",
    r"causal dietary effect",
    r"external validity",
    r"direct response validity",
)
_DIRECT_SCOPED_SENTENCE = (
    "The locked primary endpoints glucose_iAUC_2h and tg_6h_rise demonstrate "
    "direct external validity."
)
_NEGATIVE_LIMITATION_PREFIX = re.compile(
    r"(?:does|do|did)\s+not\s+(?:establish|demonstrate|show|support|claim)\s*$"
    r"|(?:cannot|can't)\s+(?:establish|demonstrate|show|support|claim)\s*$"
    r"|(?:is|are)\s+not\s+(?:evidence|proof)\s+of\s*$"
    r"|no\s+(?:evidence|claim)\s+of\s*$",
    flags=re.IGNORECASE,
)
_DIGEST = re.compile(r"[0-9a-f]{64}")


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


def _build_claim_policy_payload(
    outcome: "EvidenceGateOutcome",
    *,
    production_authorized: bool,
) -> dict[str, object]:
    """Build policy bytes; only the path-only gate wrapper may authorize them."""

    direct = outcome.tier == "direct_external_validity" and outcome.passed is True
    payload: dict[str, object] = {
        "schema_version": "claim-policy-v2",
        "tier": outcome.tier,
        "source_state": outcome.source_state,
        "allowed_claims": list(outcome.allowed_claims),
        "forbidden_claims": list(outcome.prohibited_claims),
        "forbidden_patterns": list(_FORBIDDEN_PATTERNS),
        "direct_scope_whitelist": [_DIRECT_SCOPED_SENTENCE] if direct else [],
        "negative_limitation_sentences_allowed": True,
        "scope": "Phase 3 manuscript and build inputs",
        "phase3_must_consume": True,
        "authorization": (
            "production_path_only_evidence_gate"
            if production_authorized
            else "testing_only_unauthorized"
        ),
    }
    payload["payload_sha256"] = canonical_payload_sha256(payload)
    return payload


def _write_claim_artifacts(
    output_directory: Path,
    outcome: "EvidenceGateOutcome",
    *,
    production_authorized: bool,
) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    policy = _build_claim_policy_payload(
        outcome,
        production_authorized=production_authorized,
    )
    policy_bytes = _json_bytes(policy)
    policy_path = output_directory / "claim_policy.json"
    policy_path.write_bytes(policy_bytes)
    decision: dict[str, object] = {
        "schema_version": "evidence-gate-decision-v2",
        "gate_version": outcome.gate_version,
        "tier": outcome.tier,
        "passed": outcome.passed,
        "blockers": list(outcome.blockers),
        "allowed_claims": list(outcome.allowed_claims),
        "forbidden_claims": list(outcome.prohibited_claims),
        "claim_policy_file": policy_path.name,
        "claim_policy_file_sha256": sha256(policy_bytes).hexdigest(),
        "source_state": outcome.source_state,
        "authorization": policy["authorization"],
        "manuscript_checked_or_revised": False,
    }
    decision["payload_sha256"] = canonical_payload_sha256(decision)
    decision_path = output_directory / "current_gate_decision.json"
    decision_path.write_bytes(_json_bytes(decision))
    return {"decision": decision_path, "policy": policy_path}


def _write_testing_only_claim_artifacts(
    output_directory: str | Path,
    outcome: "EvidenceGateOutcome",
) -> dict[str, Path]:
    """Materialize an explicitly unauthorized fixture from an in-memory outcome."""

    return _write_claim_artifacts(
        Path(output_directory),
        outcome,
        production_authorized=False,
    )


def build_claim_policy_from_evidence_gate(
    paths: "EvidenceGateArtifactPaths | None" = None,
) -> dict[str, Path]:
    """Rerun the path-only production gate and write only the fixed policy bundle."""

    from gmnps.validation.evidence_gate import (
        EvidenceGateArtifactPaths,
        evaluate_evidence_gate,
    )

    if paths is not None:
        if not isinstance(paths, EvidenceGateArtifactPaths):
            raise TypeError("claim policy production builder accepts only gate paths")
    outcome = evaluate_evidence_gate(paths)
    return _write_claim_artifacts(
        _TRUSTED_DECISION_PATH.parent,
        outcome,
        production_authorized=True,
    )


def _read_regular_bytes(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing or unreadable") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
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


def _load_verified_json(path: Path, label: str) -> tuple[bytes, dict[str, object]]:
    raw = _read_regular_bytes(path, label)
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


def _validate_registry(
    decision_raw: bytes,
    policy_raw: bytes,
    decision: dict[str, object],
    policy: dict[str, object],
) -> None:
    raw = _read_regular_bytes(_TRUSTED_POLICY_REGISTRY_PATH, "claim policy registry")
    try:
        registry = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("claim policy registry is not valid JSON") from error
    if not isinstance(registry, dict) or set(registry) != {
        "schema_version",
        "approved_bundles",
    }:
        raise ValueError("claim policy registry schema is invalid")
    if registry.get("schema_version") != "claim-policy-registry-v1":
        raise ValueError("claim policy registry version is invalid")
    approved = registry.get("approved_bundles")
    if not isinstance(approved, list) or not approved:
        raise ValueError("claim policy registry has no approved bundle")
    required = {
        "bundle_id",
        "tier",
        "source_state",
        "decision_path",
        "policy_path",
        "decision_sha256",
        "policy_sha256",
    }
    valid_rows: list[dict[str, object]] = []
    for row in approved:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("claim policy registry approved entry is invalid")
        if not isinstance(row["bundle_id"], str) or not row["bundle_id"]:
            raise ValueError("claim policy registry bundle_id is invalid")
        if row["decision_path"] != _TRUSTED_DECISION_RELATIVE.as_posix():
            raise ValueError("claim policy registry decision path is not fixed")
        if row["policy_path"] != _TRUSTED_POLICY_RELATIVE.as_posix():
            raise ValueError("claim policy registry policy path is not fixed")
        if not _DIGEST.fullmatch(str(row["decision_sha256"])) or not _DIGEST.fullmatch(
            str(row["policy_sha256"])
        ):
            raise ValueError("claim policy registry digest is invalid")
        valid_rows.append(row)
    decision_digest = sha256(decision_raw).hexdigest()
    policy_digest = sha256(policy_raw).hexdigest()
    matches = [
        row
        for row in valid_rows
        if row["decision_sha256"] == decision_digest
        and row["policy_sha256"] == policy_digest
        and row["tier"] == decision.get("tier") == policy.get("tier")
        and row["source_state"]
        == decision.get("source_state")
        == policy.get("source_state")
    ]
    if len(matches) != 1:
        raise ValueError("decision/policy pair is not approved by the fixed registry")


def _validate_bound_bundle() -> dict[str, object]:
    policy_raw, policy = _load_verified_json(_TRUSTED_POLICY_PATH, "claim policy")
    decision_raw, decision = _load_verified_json(
        _TRUSTED_DECISION_PATH, "gate decision"
    )
    if decision.get("schema_version") != "evidence-gate-decision-v2":
        raise ValueError("gate decision schema version is invalid")
    if policy.get("schema_version") != "claim-policy-v2":
        raise ValueError("claim policy schema version is invalid")
    if decision.get("authorization") != "production_path_only_evidence_gate" or policy.get(
        "authorization"
    ) != "production_path_only_evidence_gate":
        raise ValueError("claim bundle was not produced by the production path-only gate")
    if decision.get("claim_policy_file") != _TRUSTED_POLICY_PATH.name:
        raise ValueError("claim policy filename does not match fixed path")
    if decision.get("claim_policy_file_sha256") != sha256(policy_raw).hexdigest():
        raise ValueError("claim policy hash does not match gate decision")
    for field in ("tier", "source_state", "allowed_claims", "forbidden_claims"):
        if decision.get(field) != policy.get(field):
            raise ValueError(f"claim policy {field} does not match gate decision")
    patterns = policy.get("forbidden_patterns")
    if patterns != list(_FORBIDDEN_PATTERNS):
        raise ValueError("claim policy forbidden_patterns is invalid or incomplete")
    whitelist = policy.get("direct_scope_whitelist")
    expected_whitelist = (
        [_DIRECT_SCOPED_SENTENCE]
        if decision.get("tier") == "direct_external_validity"
        and decision.get("passed") is True
        else []
    )
    if whitelist != expected_whitelist:
        raise ValueError("claim policy direct scope whitelist is invalid")
    if policy.get("negative_limitation_sentences_allowed") is not True:
        raise ValueError("claim policy negative-limitation rule is invalid")
    _validate_registry(decision_raw, policy_raw, decision, policy)
    return policy


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(
        match.group(0).strip()
        for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text)
        if match.group(0).strip()
    )


def _is_negative_limitation(sentence: str, match: re.Match[str]) -> bool:
    return _NEGATIVE_LIMITATION_PREFIX.search(sentence[: match.start()]) is not None


def _is_direct_scoped_sentence(sentence: str, policy: dict[str, object]) -> bool:
    if policy.get("tier") != "direct_external_validity":
        return False
    normalized = re.sub(r"\s+", " ", sentence).strip().casefold()
    return normalized == _DIRECT_SCOPED_SENTENCE.casefold()


def check_claim_inputs(
    input_paths: Iterable[str | Path],
) -> tuple[ClaimViolation, ...]:
    """Check inputs against the fixed, registry-approved production claim bundle."""

    policy = _validate_bound_bundle()
    patterns = policy["forbidden_patterns"]
    violations: list[ClaimViolation] = []
    for value in input_paths:
        path = Path(value)
        text = path.read_text(encoding="utf-8")
        for sentence in _sentences(text):
            for pattern in patterns:
                match = re.search(pattern, sentence, flags=re.IGNORECASE)
                if match is None:
                    continue
                if _is_negative_limitation(sentence, match):
                    continue
                if _is_direct_scoped_sentence(sentence, policy):
                    continue
                violations.append(ClaimViolation(path=path, pattern=pattern))
                break
    return tuple(violations)


__all__ = [
    "ClaimViolation",
    "build_claim_policy_from_evidence_gate",
    "canonical_payload_sha256",
    "check_claim_inputs",
]
