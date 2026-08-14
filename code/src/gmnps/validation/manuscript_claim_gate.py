"""Strict manuscript-only exceptions around the frozen Phase 2 claim gate."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Iterable

from gmnps.validation.claim_policy import ClaimViolation, check_claim_inputs


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_MATRIX_PATH = (
    _REPOSITORY_ROOT
    / "manuscript/nature_food_submission/claim_evidence_matrix.csv"
)
_DEFAULT_REFERENCE_AUDIT_PATH = (
    _REPOSITORY_ROOT / "manuscript/nature_food_submission/reference_audit.csv"
)
_MATRIX_COLUMNS = {
    "claim_id",
    "section",
    "allowed_wording",
    "evidence_tier",
    "evidence_role",
    "data_class",
    "claim_scope",
    "status",
    "citation_keys",
}
_REFERENCE_AUDIT_COLUMNS = {
    "claim_id",
    "citation_key",
    "support_role",
    "verification_status",
    "stable_identifier",
    "retraction_status",
}
_PUBLISHED_ROUTE = {
    "evidence_tier": "published_context",
    "evidence_role": "literature_context",
    "data_class": "published_literature_context",
    "claim_scope": "published_field_context",
    "status": "supported",
}
_MARKER = re.compile(
    r"(?m)^[ \t]*%[ \t]+CLAIM_ID:[ \t]*(INT-\d{2})[ \t]*$"
)
_CITATION = re.compile(
    r"\\(?:auto|paren|text)?cite[tp]?"
    r"(?:\s*\[[^\[\]]*\]){0,2}\s*\{([^{}]+)\}"
)
_PROJECT_LANGUAGE = re.compile(
    r"\b(?:GMNPS|we|our|this\s+study|present\s+study|current\s+analysis)\b",
    flags=re.IGNORECASE,
)
_PLACEHOLDER_IDENTIFIER = re.compile(
    r"(?:pending|unknown|none|n/?a|not[_ -]?applicable|tbd)",
    flags=re.IGNORECASE,
)
_ROUTE_VIOLATION = "manuscript_published_context_route"


@dataclass(frozen=True)
class _ClaimBlock:
    claim_id: str
    body: str


def _read_csv(
    path: Path,
    *,
    required_columns: set[str],
    label: str,
) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            if not required_columns <= columns:
                missing = ", ".join(sorted(required_columns - columns))
                raise ValueError(f"{label} lacks required columns: {missing}")
            return list(reader)
    except OSError as error:
        raise ValueError(f"{label} is missing or unreadable: {path}") from error


def _matrix_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(
        path,
        required_columns=_MATRIX_COLUMNS,
        label="claim evidence matrix",
    )
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        claim_id = row["claim_id"].strip()
        if not claim_id or claim_id in indexed:
            raise ValueError("claim evidence matrix claim_id values must be unique")
        indexed[claim_id] = row
    return indexed


def _claim_blocks(text: str) -> tuple[_ClaimBlock, ...]:
    markers = tuple(_MARKER.finditer(text))
    blocks: list[_ClaimBlock] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        blocks.append(
            _ClaimBlock(
                claim_id=marker.group(1),
                body=text[marker.end() : end].strip(),
            )
        )
    return tuple(blocks)


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(
        match.group(0).strip()
        for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text)
        if match.group(0).strip()
    )


def _matrix_citation_keys(value: str) -> tuple[str, ...]:
    return tuple(key.strip() for key in value.split("|") if key.strip())


def _claim_text_and_citations(body: str) -> tuple[str, tuple[str, ...]] | None:
    sentences = _sentences(body)
    if len(sentences) != 1 or sentences[0] != body.strip():
        return None
    citations: list[str] = []
    for match in _CITATION.finditer(body):
        citations.extend(key.strip() for key in match.group(1).split(","))
    if not citations or any(not key for key in citations):
        return None
    prose = _CITATION.sub("", body)
    prose = re.sub(r"\s+", " ", prose).strip()
    prose = re.sub(r"\s+([.,;:!?])", r"\1", prose)
    return prose, tuple(citations)


def _has_verified_references(
    claim_id: str,
    citation_keys: tuple[str, ...],
    audit_rows: list[dict[str, str]],
) -> bool:
    for citation_key in citation_keys:
        matches = [
            row
            for row in audit_rows
            if row["claim_id"].strip() == claim_id
            and row["citation_key"].strip() == citation_key
        ]
        if len(matches) != 1:
            return False
        row = matches[0]
        identifier = row["stable_identifier"].strip()
        if (
            row["verification_status"].strip() != "verified"
            or not row["support_role"].strip()
            or not identifier
            or _PLACEHOLDER_IDENTIFIER.fullmatch(identifier)
            or row["retraction_status"].strip() != "clear"
        ):
            return False
    return True


def _is_route_eligible(
    block: _ClaimBlock,
    *,
    row: dict[str, str] | None,
    marker_count: int,
    audit_rows: list[dict[str, str]],
) -> bool:
    if row is None or marker_count != 1 or row["section"].strip() != "Introduction":
        return False
    if any(row[field].strip() != expected for field, expected in _PUBLISHED_ROUTE.items()):
        return False
    parsed = _claim_text_and_citations(block.body)
    if parsed is None:
        return False
    prose, citation_keys = parsed
    expected_keys = _matrix_citation_keys(row["citation_keys"])
    return (
        prose == row["allowed_wording"].strip()
        and citation_keys == expected_keys
        and len(citation_keys) == len(set(citation_keys))
        and _PROJECT_LANGUAGE.search(prose) is None
        and _has_verified_references(
            block.claim_id,
            citation_keys,
            audit_rows,
        )
    )


def _isolated_violations(body: str) -> tuple[ClaimViolation, ...]:
    with TemporaryDirectory(prefix="gmnps-manuscript-claim-") as directory:
        path = Path(directory) / "claim.txt"
        path.write_text(body, encoding="utf-8")
        return check_claim_inputs([path])


def _remove_matching_violation(
    violations: list[ClaimViolation],
    *,
    path: Path,
    pattern: str,
) -> bool:
    for index, violation in enumerate(violations):
        if violation.path == path and violation.pattern == pattern:
            del violations[index]
            return True
    return False


def check_manuscript_claim_inputs(
    input_paths: Iterable[str | Path],
    matrix_path: str | Path | None = None,
    reference_audit_path: str | Path | None = None,
) -> tuple[ClaimViolation, ...]:
    """Apply Phase 2 policy, then audit exact Introduction literature exceptions."""

    matrix = _matrix_rows(Path(matrix_path or _DEFAULT_MATRIX_PATH))
    audit_rows = _read_csv(
        Path(reference_audit_path or _DEFAULT_REFERENCE_AUDIT_PATH),
        required_columns=_REFERENCE_AUDIT_COLUMNS,
        label="reference audit",
    )
    paths = tuple(Path(value) for value in input_paths)
    violations = list(check_claim_inputs(paths))

    for path in paths:
        blocks = _claim_blocks(path.read_text(encoding="utf-8"))
        marker_counts = {
            claim_id: sum(block.claim_id == claim_id for block in blocks)
            for claim_id in {block.claim_id for block in blocks}
        }
        for block in blocks:
            row = matrix.get(block.claim_id)
            published_candidate = row is None or (
                row["section"].strip() == "Introduction"
                and (
                    row["evidence_tier"].strip() == "published_context"
                    or row["claim_scope"].strip() == "published_field_context"
                )
            )
            if not published_candidate:
                continue
            if not _is_route_eligible(
                block,
                row=row,
                marker_count=marker_counts[block.claim_id],
                audit_rows=audit_rows,
            ):
                violations.append(ClaimViolation(path=path, pattern=_ROUTE_VIOLATION))
                continue
            underlying = _isolated_violations(block.body)
            if len(underlying) == 1:
                _remove_matching_violation(
                    violations,
                    path=path,
                    pattern=underlying[0].pattern,
                )

    return tuple(violations)


__all__ = ["check_manuscript_claim_inputs"]
