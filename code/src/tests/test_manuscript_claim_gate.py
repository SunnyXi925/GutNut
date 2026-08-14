from __future__ import annotations

import csv
from difflib import SequenceMatcher
from pathlib import Path
import re

import pytest

from gmnps.validation.claim_policy import check_claim_inputs


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "manuscript/nature_food_submission/claim_evidence_matrix.csv"
OUTLINE = ROOT / "manuscript/nature_food_submission/narrative_outline.md"

REQUIRED_COLUMNS = {
    "claim_id",
    "section",
    "paragraph_id",
    "paragraph_job",
    "canonical_claim",
    "allowed_wording",
    "evidence_tier",
    "evidence_role",
    "evidence_family",
    "data_class",
    "dataset_or_artifact",
    "analysis_unit",
    "n_effective",
    "statistic_or_invariant",
    "figure_or_table",
    "main_or_supplementary",
    "status",
    "prohibited_overstatement",
    "boundary",
    "source_path",
}

LEGAL_COMBINATIONS = {
    ("published_context", "literature_context", "conditional"),
    ("locked_method_definition", "method_definition_or_invariant", "supported"),
    ("computational_feasibility", "audited_data_unavailability", "supported"),
    (
        "computational_feasibility",
        "fail_closed_computational_design",
        "supported",
    ),
    (
        "computational_feasibility",
        "correctly_specified_synthetic_positive_control",
        "supported",
    ),
    ("expert_content_validity", "expert_review_content_validity", "conditional"),
    ("blocked_external_data", "future_external_empirical_test", "blocked"),
}


def _rows() -> list[dict[str, str]]:
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _canonical_tokens(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "only",
        "the",
        "to",
        "with",
    }
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    return {token for token in tokens if token not in stop}


def test_matrix_exists_and_has_complete_schema():
    assert MATRIX.is_file()
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert REQUIRED_COLUMNS <= set(reader.fieldnames or ())
        rows = list(reader)
    assert rows
    for row in rows:
        for field in REQUIRED_COLUMNS - {"allowed_wording"}:
            assert row[field].strip(), f"{row.get('claim_id', '<unknown>')} lacks {field}"


def test_claim_ids_are_unique_and_all_planned_claims_are_covered_once():
    rows = _rows()
    claim_ids = [row["claim_id"] for row in rows]
    assert len(claim_ids) == len(set(claim_ids))
    assert all(re.fullmatch(r"(?:ABS|RES|DIS)-\d{2}", value) for value in claim_ids)
    assert {row["section"] for row in rows} == {"Abstract", "Results", "Discussion"}

    outline = OUTLINE.read_text(encoding="utf-8")
    for claim_id in claim_ids:
        assert len(re.findall(rf"\b{re.escape(claim_id)}\b", outline)) == 1


def test_evidence_tier_role_and_status_combinations_are_legal():
    for row in _rows():
        combination = (row["evidence_tier"], row["evidence_role"], row["status"])
        assert combination in LEGAL_COMBINATIONS, (row["claim_id"], combination)


def test_blocked_claims_have_no_allowed_wording_or_main_figure():
    blocked = [row for row in _rows() if row["status"] == "blocked"]
    assert blocked
    for row in blocked:
        assert row["allowed_wording"] == ""
        assert row["evidence_tier"] == "blocked_external_data"
        assert row["figure_or_table"].startswith("blocked_external_data:")
        assert row["main_or_supplementary"] == "blocked_external_data"


def test_no_current_quantitative_main_figure_is_claimed():
    rows = _rows()
    assert {
        row["figure_or_table"]
        for row in rows
        if row["figure_or_table"].startswith("blocked_external_data:Fig.")
    } >= {
        "blocked_external_data:Fig. 2",
        "blocked_external_data:Fig. 3",
        "blocked_external_data:Fig. 4",
    }
    for row in rows:
        if row["status"] != "blocked":
            assert row["main_or_supplementary"] != "main_quantitative"


def test_synthetic_evidence_is_one_supplementary_positive_control_family():
    synthetic = [row for row in _rows() if row["data_class"] == "synthetic"]
    assert synthetic
    assert {row["evidence_family"] for row in synthetic} == {
        "phase2_correctly_specified_synthetic_positive_control"
    }
    assert {row["evidence_role"] for row in synthetic} == {
        "correctly_specified_synthetic_positive_control"
    }
    assert {row["main_or_supplementary"] for row in synthetic} == {
        "supplementary_method"
    }
    assert all("real" not in row["data_class"].casefold() for row in synthetic)


def test_outline_has_required_paragraph_level_architecture_and_terminology():
    text = OUTLINE.read_text(encoding="utf-8")
    for paragraph_id in (
        *(f"ABS-S{number}" for number in range(1, 7)),
        *(f"INT-P{number}" for number in range(1, 5)),
        *(f"RES-P{number}" for number in range(1, 9)),
        *(f"DIS-P{number}" for number in range(1, 6)),
    ):
        assert paragraph_id in text

    required_terms = (
        "nutrient profiling system",
        "personalized calibration",
        "Gut Microbiome-informed Nutrient Profiling System (GMNPS)",
        "correctly specified synthetic positive-control",
        "Food Compass 2.0",
        "baseline implementation",
        "method definition -> current feasibility -> robustness positive-control -> implications + hard boundary",
        "not empirical Article submission-ready",
    )
    for term in required_terms:
        assert term in text
    assert "precision-ready" not in text.casefold()
    assert "transforms" not in text.casefold()


def test_current_claim_policy_accepts_every_nonblocked_allowed_wording(tmp_path):
    for row in _rows():
        wording = row["allowed_wording"]
        if not wording:
            continue
        candidate = tmp_path / f"{row['claim_id']}.txt"
        candidate.write_text(wording + "\n", encoding="utf-8")
        assert check_claim_inputs([candidate]) == (), row["claim_id"]


@pytest.mark.parametrize(
    "overclaim",
    [
        "GMNPS validates real participant responses.",
        "GMNPS preserves population rankings across 9,234 foods.",
        "GMNPS provides clinical dietary guidance.",
        "GMNPS demonstrates biological superiority of the expert mask.",
        "GMNPS predicts disease in an independent cohort.",
        "GMNPS transforms nutrient profiling for precision nutrition.",
    ],
)
def test_current_claim_policy_rejects_forbidden_or_real_data_overclaims(
    tmp_path, overclaim
):
    candidate = tmp_path / "overclaim.txt"
    candidate.write_text(overclaim, encoding="utf-8")
    assert check_claim_inputs([candidate])


def test_canonical_claims_have_no_semantic_near_duplicates():
    rows = _rows()
    for index, left in enumerate(rows):
        left_tokens = _canonical_tokens(left["canonical_claim"])
        for right in rows[index + 1 :]:
            right_tokens = _canonical_tokens(right["canonical_claim"])
            union = left_tokens | right_tokens
            jaccard = len(left_tokens & right_tokens) / len(union) if union else 1.0
            sequence = SequenceMatcher(
                None,
                " ".join(sorted(left_tokens)),
                " ".join(sorted(right_tokens)),
            ).ratio()
            assert max(jaccard, sequence) < 0.88, (
                left["claim_id"],
                right["claim_id"],
                jaccard,
                sequence,
            )


def test_source_paths_exist_or_name_an_external_blocker():
    for row in _rows():
        for source in row["source_path"].split("|"):
            source = source.strip()
            if source.startswith("EXTERNAL_BLOCKER:"):
                assert row["status"] == "blocked"
                assert len(source.removeprefix("EXTERNAL_BLOCKER:").strip()) >= 12
            else:
                assert (ROOT / source).is_file(), (row["claim_id"], source)
