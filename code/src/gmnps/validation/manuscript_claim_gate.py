"""Strict manuscript-only exceptions around the frozen Phase 2 claim gate."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Iterable
from urllib.parse import urlsplit

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
_MARKER = re.compile(r"[ \t]*%[ \t]+CLAIM_ID:[ \t]*(INT-\d{2})[ \t]*")
_CITATION = re.compile(
    r"\\(?:auto|paren|text)?cite[tp]?"
    r"(?:\s*\[[^\[\]]*\]){0,2}\s*\{([^{}]+)\}"
)
_PROJECT_LANGUAGE = re.compile(
    r"\b(?:GMNPS|we|our|this\s+study|present\s+study|current\s+analysis|"
    r"this\s+(?:work|paper|article|manuscript)|here\s+we|"
    r"the\s+proposed\s+(?:framework|method|approach|system|model|algorithm)|"
    r"the\s+(?:framework|method|approach|system|model|algorithm)\s+"
    r"(?:introduced|presented|reported|developed|proposed)\s+here|"
    r"(?:introduced|presented|reported|developed|proposed)\s+here)\b",
    flags=re.IGNORECASE,
)
_PENDING_IDENTIFIER = re.compile(
    r"(?:pending|unknown|none|n/?a|not[_ -]?applicable|tbd|todo|placeholder)",
    flags=re.IGNORECASE,
)
_DOI_IDENTIFIER = re.compile(r"doi:10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
_PMID_IDENTIFIER = re.compile(r"pmid:\d+")
_PMCID_IDENTIFIER = re.compile(r"pmcid:PMC\d+")
_ROUTE_VIOLATION = "manuscript_published_context_route"


@dataclass(frozen=True)
class _ClaimBlock:
    claim_id: str
    body: str
    line_index: int


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
    blocks: list[_ClaimBlock] = []
    lines = text.splitlines()
    for marker_index, line in enumerate(lines):
        marker = _MARKER.fullmatch(line)
        if marker is None:
            continue
        claim_line_index = marker_index + 1
        blocks.append(
            _ClaimBlock(
                claim_id=marker.group(1),
                body=(
                    lines[claim_line_index].strip()
                    if claim_line_index < len(lines)
                    else ""
                ),
                line_index=claim_line_index,
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


def _is_stable_identifier(value: str) -> bool:
    if not value or _PENDING_IDENTIFIER.search(value):
        return False
    if (
        _DOI_IDENTIFIER.fullmatch(value)
        or _PMID_IDENTIFIER.fullmatch(value)
        or _PMCID_IDENTIFIER.fullmatch(value)
    ):
        return True
    if not value.startswith("url:https://") or re.search(r"\s", value):
        return False
    parsed = urlsplit(value.removeprefix("url:"))
    return (
        parsed.scheme == "https"
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
    )


def _reference_audit_is_exact(
    requested_pairs: tuple[tuple[str, str], ...],
    audit_rows: list[dict[str, str]],
) -> bool:
    if len(requested_pairs) != len(set(requested_pairs)):
        return False
    observed_pairs = tuple(
        (row["claim_id"].strip(), row["citation_key"].strip())
        for row in audit_rows
    )
    if (
        len(observed_pairs) != len(set(observed_pairs))
        or set(observed_pairs) != set(requested_pairs)
    ):
        return False
    return all(
        row["support_role"].strip() == "direct_claim_support"
        and row["verification_status"].strip() == "verified"
        and _is_stable_identifier(row["stable_identifier"].strip())
        and row["retraction_status"].strip() == "clear"
        for row in audit_rows
    )


def _route_citation_keys(
    block: _ClaimBlock,
    *,
    row: dict[str, str] | None,
    marker_count: int,
) -> tuple[str, ...] | None:
    if row is None or marker_count != 1 or row["section"].strip() != "Introduction":
        return None
    if any(row[field].strip() != expected for field, expected in _PUBLISHED_ROUTE.items()):
        return None
    parsed = _claim_text_and_citations(block.body)
    if parsed is None:
        return None
    prose, citation_keys = parsed
    expected_keys = _matrix_citation_keys(row["citation_keys"])
    if not (
        prose == row["allowed_wording"].strip()
        and citation_keys == expected_keys
        and len(citation_keys) == len(set(citation_keys))
        and _PROJECT_LANGUAGE.search(prose) is None
    ):
        return None
    return citation_keys


def _sanitized_policy_violations(
    documents: tuple[tuple[Path, str, set[int]], ...],
) -> tuple[ClaimViolation, ...]:
    with TemporaryDirectory(prefix="gmnps-manuscript-claim-") as directory:
        temporary_paths: list[Path] = []
        original_paths: dict[Path, Path] = {}
        for index, (original_path, text, removed_lines) in enumerate(documents):
            lines = text.splitlines(keepends=True)
            for line_index in removed_lines:
                if line_index >= len(lines):
                    continue
                if lines[line_index].endswith("\r\n"):
                    lines[line_index] = "\r\n"
                elif lines[line_index].endswith("\n"):
                    lines[line_index] = "\n"
                else:
                    lines[line_index] = ""
            temporary_path = Path(directory) / f"{index:04d}-{original_path.name}"
            temporary_path.write_text("".join(lines), encoding="utf-8")
            temporary_paths.append(temporary_path)
            original_paths[temporary_path] = original_path
        return tuple(
            ClaimViolation(path=original_paths[item.path], pattern=item.pattern)
            for item in check_claim_inputs(temporary_paths)
        )


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
    documents: list[tuple[Path, str, set[int]]] = []
    route_failures: list[ClaimViolation] = []
    requested_pairs: list[tuple[str, str]] = []
    structurally_eligible: list[tuple[int, _ClaimBlock]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        blocks = _claim_blocks(text)
        marker_counts = {
            claim_id: sum(block.claim_id == claim_id for block in blocks)
            for claim_id in {block.claim_id for block in blocks}
        }
        document_index = len(documents)
        documents.append((path, text, set()))
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
            citation_keys = _route_citation_keys(
                block,
                row=row,
                marker_count=marker_counts[block.claim_id],
            )
            if citation_keys is None:
                route_failures.append(
                    ClaimViolation(path=path, pattern=_ROUTE_VIOLATION)
                )
                continue
            requested_pairs.extend(
                (block.claim_id, citation_key) for citation_key in citation_keys
            )
            structurally_eligible.append((document_index, block))

    if _reference_audit_is_exact(tuple(requested_pairs), audit_rows):
        for document_index, block in structurally_eligible:
            documents[document_index][2].add(block.line_index)
    else:
        route_failures.extend(
            ClaimViolation(path=documents[index][0], pattern=_ROUTE_VIOLATION)
            for index, _ in structurally_eligible
        )

    return _sanitized_policy_violations(tuple(documents)) + tuple(route_failures)


__all__ = ["check_manuscript_claim_inputs"]
