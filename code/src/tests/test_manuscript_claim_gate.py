from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import re

import pytest

from gmnps.validation.claim_policy import check_claim_inputs
from gmnps.validation.manuscript_claim_gate import check_manuscript_claim_inputs


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "manuscript/nature_food_submission/claim_evidence_matrix.csv"
OUTLINE = ROOT / "manuscript/nature_food_submission/narrative_outline.md"
REFERENCE_AUDIT = ROOT / "manuscript/nature_food_submission/reference_audit.csv"
REFERENCES = ROOT / "manuscript/nature_food_submission/references.bib"
SUBMISSION_DIR = ROOT / "manuscript/nature_food_submission"
ABSTRACT_ROOT = SUBMISSION_DIR / "sn-article.tex"
EXPERT_ITEM_AUDIT = (
    ROOT / "manuscript/nature_food_submission/expert_review_item_audit.csv"
)

REQUIRED_COLUMNS = {
    "claim_id",
    "assertion_key",
    "section",
    "paragraph_id",
    "paragraph_job",
    "canonical_claim",
    "allowed_wording",
    "claim_scope",
    "citation_keys",
    "evidence_tier",
    "evidence_role",
    "evidence_family",
    "data_class",
    "dataset_or_artifact",
    "analysis_unit",
    "inference_unit",
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
    ("published_context", "literature_context", "supported"),
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
    ("expert_content_validity", "expert_review_content_validity", "supported"),
    ("blocked_external_data", "future_external_empirical_test", "blocked"),
}

EXPECTED_CLAIM_MAP = {
    "ABS-01": ("Abstract", "ABS-S1"),
    "ABS-02": ("Abstract", "ABS-S2"),
    "ABS-03": ("Abstract", "ABS-S3"),
    "ABS-04": ("Abstract", "ABS-S4"),
    "ABS-05": ("Abstract", "ABS-S5"),
    "ABS-06": ("Abstract", "ABS-S6"),
    "INT-01": ("Introduction", "INT-P1"),
    "INT-02": ("Introduction", "INT-P1"),
    "INT-03": ("Introduction", "INT-P2"),
    "INT-04": ("Introduction", "INT-P2"),
    "INT-05": ("Introduction", "INT-P3"),
    "INT-06": ("Introduction", "INT-P3"),
    "INT-07": ("Introduction", "INT-P4"),
    "INT-08": ("Introduction", "INT-P4"),
    "RES-01": ("Results", "RES-P1"),
    "RES-02": ("Results", "RES-P1"),
    "RES-03": ("Results", "RES-P2"),
    "RES-04": ("Results", "RES-P3"),
    "RES-05": ("Results", "RES-P3"),
    "RES-06": ("Results", "RES-P4"),
    "RES-07": ("Results", "RES-P5"),
    "RES-08": ("Results", "RES-P5"),
    "RES-09": ("Results", "RES-P6"),
    "RES-10": ("Results", "RES-P7"),
    "RES-11": ("Results", "RES-P8"),
    "RES-12": ("Results", "RES-P8"),
    "RES-13": ("Results", "RES-P8"),
    "RES-14": ("Results", "RES-P8"),
    "DIS-01": ("Discussion", "DIS-P1"),
    "DIS-02": ("Discussion", "DIS-P2"),
    "DIS-03": ("Discussion", "DIS-P3"),
    "DIS-04": ("Discussion", "DIS-P4"),
    "DIS-05": ("Discussion", "DIS-P4"),
    "DIS-06": ("Discussion", "DIS-P5"),
}


def _rows() -> list[dict[str, str]]:
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalized_claim(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _assert_assertion_key_rules(rows: list[dict[str, str]]) -> None:
    seen_sections: set[tuple[str, str]] = set()
    key_families: dict[str, tuple[str, str]] = {}
    text_keys: dict[str, str] = {}
    for row in rows:
        section = (row["assertion_key"], row["section"])
        assert section not in seen_sections, section
        seen_sections.add(section)

        family = (row["claim_scope"], row["evidence_family"])
        assert key_families.setdefault(row["assertion_key"], family) == family

        canonical = _normalized_claim(row["canonical_claim"])
        assert text_keys.setdefault(canonical, row["assertion_key"]) == row[
            "assertion_key"
        ]


def _outline_claim_map(text: str) -> dict[str, tuple[str, str]]:
    section = ""
    paragraph_id = ""
    mapped: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        section_match = re.fullmatch(r"## (Abstract|Introduction|Results|Discussion)", line)
        if section_match:
            section = section_match.group(1)
            continue
        paragraph_match = re.fullmatch(r"### ((?:ABS-S|INT-P|RES-P|DIS-P)\d+):.*", line)
        if paragraph_match:
            paragraph_id = paragraph_match.group(1)
            continue
        if not section or not paragraph_id:
            continue
        for claim_id in re.findall(r"\b(?:ABS|INT|RES|DIS)-\d{2}\b", line):
            assert claim_id not in mapped, claim_id
            mapped[claim_id] = (section, paragraph_id)
    return mapped


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _route_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    wording = "Microbiome-informed framework supports individualized dietary response prediction."
    matrix = tmp_path / "matrix.csv"
    _write_csv(
        matrix,
        [
            {
                "claim_id": "INT-01",
                "section": "Introduction",
                "allowed_wording": wording,
                "evidence_tier": "published_context",
                "evidence_role": "literature_context",
                "evidence_family": "published_test_context",
                "data_class": "published_literature_context",
                "claim_scope": "published_field_context",
                "status": "supported",
                "citation_keys": "Smith2025",
                "source_path": "manuscript/nature_food_submission/references.bib",
            }
        ],
    )
    audit = tmp_path / "reference_audit.csv"
    _write_csv(
        audit,
        [
            {
                "claim_id": "INT-01",
                "citation_key": "Smith2025",
                "support_role": "direct_claim_support",
                "verification_status": "verified",
                "stable_identifier": "doi:10.1038/s41586-025-01234-5",
                "retraction_status": "clear",
            }
        ],
    )
    candidate = tmp_path / "introduction.tex"
    candidate.write_text(
        "% CLAIM_ID: INT-01\n"
        "Microbiome-informed framework supports individualized dietary response "
        "prediction \\cite{Smith2025}.\n",
        encoding="utf-8",
    )
    return candidate, matrix, audit


def _project_route_fixture(
    tmp_path: Path,
    *,
    claim_id: str = "INT-04",
) -> tuple[Path, Path]:
    wording = (
        "GMNPS retains a shared food-level reference and represents personal "
        "information as a separate bounded calibration."
        if claim_id == "INT-04"
        else "GMNPS applies bounded changes to native attributes before domain recomposition."
    )
    matrix = tmp_path / "project-matrix.csv"
    _write_csv(
        matrix,
        [
            {
                "claim_id": claim_id,
                "section": "Introduction",
                "allowed_wording": wording,
                "evidence_tier": "locked_method_definition",
                "evidence_role": "method_definition_or_invariant",
                "evidence_family": "locked_attribute_level_method",
                "data_class": "computational_method",
                "claim_scope": "gmnps_project",
                "status": "supported",
                "citation_keys": "not_applicable",
                "source_path": (
                    "docs/methods/attribute_level_gmnps_spec.md|"
                    "code/src/configs/attribute_gmnps.yaml"
                ),
            }
        ],
    )
    candidate = tmp_path / "project-introduction.tex"
    candidate.write_text(
        f"% CLAIM_ID: {claim_id}\n{wording}\n",
        encoding="utf-8",
    )
    return candidate, matrix


def _visible_citation_keys() -> set[str]:
    cited_keys: set[str] = set()
    for path in sorted(SUBMISSION_DIR.glob("*.tex")):
        for group in re.findall(
            r"\\cite[a-zA-Z*]*\{([^}]+)\}", path.read_text(encoding="utf-8")
        ):
            cited_keys.update(key.strip() for key in group.split(","))
    return cited_keys


def _bibliography_entries(text: str) -> list[tuple[str, str]]:
    return [
        (match.group(1).strip(), match.group(2))
        for match in re.finditer(
            r"(?ms)^@\w+\{\s*([^,\s]+)\s*,(.*?)(?=^@\w+\{|\Z)",
            text,
        )
    ]


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


def test_claim_ids_and_matrix_paragraph_mapping_are_frozen():
    rows = _rows()
    observed = {
        row["claim_id"]: (row["section"], row["paragraph_id"]) for row in rows
    }
    assert len(observed) == len(rows)
    assert observed == EXPECTED_CLAIM_MAP


def test_outline_paragraph_blocks_have_exact_frozen_claim_ids():
    assert _outline_claim_map(OUTLINE.read_text(encoding="utf-8")) == EXPECTED_CLAIM_MAP


def test_assertion_keys_obey_identity_and_evidence_family_rules():
    _assert_assertion_key_rules(_rows())


@pytest.mark.parametrize("broken_rule", ["section", "family", "canonical"])
def test_assertion_key_rules_reject_negative_fixtures(broken_rule):
    rows = [
        {
            "assertion_key": "shared",
            "section": "Abstract",
            "paragraph_id": "ABS-S1",
            "claim_scope": "gmnps_project",
            "evidence_family": "family-a",
            "canonical_claim": "A locked method claim.",
        },
        {
            "assertion_key": "shared",
            "section": "Results",
            "paragraph_id": "RES-P1",
            "claim_scope": "gmnps_project",
            "evidence_family": "family-a",
            "canonical_claim": "A Results restatement.",
        },
    ]
    if broken_rule == "section":
        rows[1]["section"] = "Abstract"
        rows[1]["paragraph_id"] = "ABS-S2"
    elif broken_rule == "family":
        rows[1]["evidence_family"] = "family-b"
    else:
        rows[1]["assertion_key"] = "different"
        rows[1]["canonical_claim"] = " A LOCKED method claim! "
    with pytest.raises(AssertionError):
        _assert_assertion_key_rules(rows)


def test_claim_scopes_and_task3_literature_routes_are_explicit():
    rows = _rows()
    assert {row["claim_scope"] for row in rows} <= {
        "published_field_context",
        "gmnps_project",
        "future_empirical",
    }
    literature = [row for row in rows if row["evidence_tier"] == "published_context"]
    assert literature
    assert all(row["claim_scope"] == "published_field_context" for row in literature)
    assert all(row["status"] == "supported" for row in literature)
    assert all(row["citation_keys"] != "PENDING_TASK3" for row in literature)
    assert {
        row["claim_id"]: row["citation_keys"] for row in literature
    } == {
        "ABS-01": "labonte2018nutrientprofiles",
        "ABS-02": "zeevi2015personalized",
        "INT-01": "labonte2018nutrientprofiles",
        "INT-02": "scarborough2007developing",
        "INT-03": "zeevi2015personalized",
        "INT-06": "zeevi2015personalized",
    }

    project_rows = {row["claim_id"]: row for row in rows if row["claim_id"] in {"INT-04", "INT-05"}}
    assert set(project_rows) == {"INT-04", "INT-05"}
    assert {
        claim_id: row["allowed_wording"] for claim_id, row in project_rows.items()
    } == {
        "INT-04": (
            "GMNPS retains a shared food-level reference and represents personal "
            "information as a separate bounded calibration."
        ),
        "INT-05": (
            "GMNPS applies bounded changes to native attributes before domain "
            "recomposition."
        ),
    }
    for row in project_rows.values():
        assert row["claim_scope"] == "gmnps_project"
        assert row["citation_keys"] == "not_applicable"
        assert row["evidence_tier"] == "locked_method_definition"
        assert row["evidence_role"] == "method_definition_or_invariant"
        assert row["evidence_family"] == "locked_attribute_level_method"
        assert row["data_class"] == "computational_method"
        assert row["status"] == "supported"
        assert "docs/methods/attribute_level_gmnps_spec.md" in row[
            "source_path"
        ].split("|")


def test_task3_reference_audit_has_exact_columns_and_rows():
    with REFERENCE_AUDIT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "claim_id",
            "citation_key",
            "support_role",
            "verification_status",
            "stable_identifier",
            "retraction_status",
        ]
        rows = list(reader)

    assert rows == [
        {
            "claim_id": "INT-01",
            "citation_key": "labonte2018nutrientprofiles",
            "support_role": "direct_claim_support",
            "verification_status": "verified",
            "stable_identifier": "doi:10.1093/advances/nmy045",
            "retraction_status": "clear",
        },
        {
            "claim_id": "INT-02",
            "citation_key": "scarborough2007developing",
            "support_role": "direct_claim_support",
            "verification_status": "verified",
            "stable_identifier": "doi:10.1017/S1368980007223870",
            "retraction_status": "clear",
        },
        {
            "claim_id": "INT-03",
            "citation_key": "zeevi2015personalized",
            "support_role": "direct_claim_support",
            "verification_status": "verified",
            "stable_identifier": "doi:10.1016/j.cell.2015.11.001",
            "retraction_status": "clear",
        },
        {
            "claim_id": "INT-06",
            "citation_key": "zeevi2015personalized",
            "support_role": "direct_claim_support",
            "verification_status": "verified",
            "stable_identifier": "doi:10.1016/j.cell.2015.11.001",
            "retraction_status": "clear",
        },
    ]


def test_task3_abstract_has_six_uncited_sentences_and_locked_context():
    text = ABSTRACT_ROOT.read_text(encoding="utf-8")
    match = re.search(r"\\abstract\{([^{}]+)\}", text)
    assert match is not None
    abstract = match.group(1)
    assert len(re.findall(r"[^.!?]+[.!?]", abstract)) == 6
    assert "\\cite" not in abstract
    assert abstract.startswith(
        "Nutrient-profile models rate the nutritional quality of individual foods "
        "for defined public-health applications. Postprandial glycaemic responses "
        "to the same foods can differ substantially between individuals."
    )
    assert (
        "The Gut Microbiome-informed Nutrient Profiling System (GMNPS) uses "
        "bounded attribute-level calibration with locked inputs."
    ) in abstract
    assert abstract.endswith(
        "The framework provides an auditable basis for future empirical testing, "
        "but GMNPS does not establish external validity."
    )


def test_task3_main_manuscript_routes_pass_exact_claim_gate():
    main_tex = [ABSTRACT_ROOT, SUBMISSION_DIR / "sections.tex"]
    assert check_manuscript_claim_inputs(main_tex) == ()


def test_full_visible_tex_package_passes_exact_claim_gate():
    visible_tex = sorted(SUBMISSION_DIR.glob("*.tex"))
    assert check_manuscript_claim_inputs(visible_tex) == ()


def test_task3_bibliography_records_corrections_and_removes_misrouted_keys():
    text = REFERENCES.read_text(encoding="utf-8")
    for key in (
        "labonte2018nutrientprofiles",
        "scarborough2007developing",
        "asnicar2026gut",
        "gkouskou2020digitaltwins",
    ):
        assert f"@article{{{key}," in text
    assert "vanCalster2019calibration" not in text
    assert "mozaffarian2021foodcompass" not in text
    assert "adams2020digitaltwins" not in text
    assert "asnicar2025gut" not in text
    assert "10.1038/s41591-020-1130-y" in text
    assert "10.1136/bmj.q902" in text
    assert "version of record in Nature 650 (2026)" in text


def test_task3_visible_citations_resolve_and_do_not_misroute_personalization():
    bibliography = REFERENCES.read_text(encoding="utf-8")
    entries = _bibliography_entries(bibliography)
    defined_keys = {key for key, _ in entries}
    assert len(defined_keys) == len(entries)
    assert defined_keys == _visible_citation_keys()
    visible_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(SUBMISSION_DIR.glob("*.tex"))
    )
    assert "vanCalster2019calibration" not in visible_text
    assert "mozaffarian2021foodcompass" not in visible_text


def test_task3_bibliography_has_no_duplicate_fields_or_malformed_dois():
    entries = _bibliography_entries(REFERENCES.read_text(encoding="utf-8"))
    assert entries
    for key, body in entries:
        fields = re.findall(r"(?m)^\s*([A-Za-z][\w-]*)\s*=", body)
        assert len(fields) == len({field.casefold() for field in fields}), key
        dois = re.findall(r"(?im)^\s*doi\s*=\s*\{([^{}]+)\}", body)
        assert len(dois) == 1, key
        assert re.fullmatch(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", dois[0]), key


def test_task3_has_no_pending_reference_placeholders():
    for path in (MATRIX, OUTLINE, ABSTRACT_ROOT, SUBMISSION_DIR / "sections.tex"):
        assert "PENDING_TASK3" not in path.read_text(encoding="utf-8"), path


def test_int08_is_one_current_evidence_tier_assertion():
    row = next(row for row in _rows() if row["claim_id"] == "INT-08")
    assert row["assertion_key"] == "current_evidence_tier"
    assert row["canonical_claim"] == "The current evidence tier is computational feasibility."
    joined = " ".join(row.values()).casefold()
    assert "production" not in joined
    assert "eligible observed participant-by-meal outcomes" not in joined


def test_evidence_tier_role_and_status_combinations_are_legal():
    for row in _rows():
        combination = (row["evidence_tier"], row["evidence_role"], row["status"])
        assert combination in LEGAL_COMBINATIONS, (row["claim_id"], combination)


def test_blocked_claims_have_no_allowed_wording_or_main_figure():
    blocked = [row for row in _rows() if row["status"] == "blocked"]
    assert blocked
    for row in blocked:
        assert row["allowed_wording"] == ""
        assert row["claim_scope"] == "future_empirical"
        assert row["evidence_tier"] == "blocked_external_data"
        assert row["figure_or_table"].startswith("blocked_external_data:")
        assert row["main_or_supplementary"] == "blocked_external_data"


def test_blocked_analyses_have_distinct_rows_and_explicit_units():
    rows = {row["claim_id"]: row for row in _rows()}
    expected = {
        "RES-10": ("food", "food"),
        "RES-11": ("participant-meal observation", "family/twin connected component"),
        "RES-12": ("independent GMrepo person", "independent GMrepo person/component"),
        "RES-13": ("ZOE food/rank-table", "ZOE food/rank-table"),
        "RES-14": ("knowledge path", "knowledge path"),
    }
    for claim_id, (analysis_unit, inference_unit) in expected.items():
        assert rows[claim_id]["analysis_unit"] == analysis_unit
        assert rows[claim_id]["inference_unit"] == inference_unit
    assert all("source-specific" not in row["analysis_unit"] for row in rows.values())
    assert all("source-specific" not in row["inference_unit"] for row in rows.values())


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


def test_expert_claims_respect_the_audited_boundary():
    expert = [row for row in _rows() if row["evidence_tier"] == "expert_content_validity"]
    assert expert
    joined = " ".join(" ".join(row.values()) for row in expert).casefold()
    for prohibited in ("97.2%", "two requests", "six experts", "6 experts"):
        assert prohibited not in joined
    assert all("expert_review_item_audit.csv" in row["source_path"] for row in expert)
    assert all("content review" in row["allowed_wording"].casefold() for row in expert)


def test_expert_item_audit_is_hash_bound_and_has_correct_distributions():
    assert EXPERT_ITEM_AUDIT.is_file()
    with EXPERT_ITEM_AUDIT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 90
    assert set(rows[0]) == {
        "workbook_id",
        "source_sha256",
        "nutrient",
        "assignment_rating",
        "evidence_rating",
        "comment_present",
    }
    assert Counter(row["assignment_rating"] for row in rows) == {
        "reasonable": 86,
        "needs_clarification": 1,
        "unreasonable": 3,
    }
    assert Counter(row["evidence_rating"] for row in rows) == {
        "sufficient": 82,
        "insufficient": 7,
        "needs_clarification": 1,
    }
    expected_sources = {
        "workbook_01": "28b4a92a9b612d47c24d0ac0c6b4631f265c9cf7bb80b757e82820986c2efe3c",
        "workbook_02": "4875bc4e85f8c234d2db697c0dd716198971034ac30822b2b216ed16f2b190a2",
        "workbook_03": "bba1caac1f9bef7e80a41b706c0541c26c5cdd1e4542db62adf3233ed4f3a052",
    }
    assert {row["source_sha256"] for row in rows} == set(expected_sources.values())
    assert Counter(row["source_sha256"] for row in rows) == {
        digest: 30 for digest in expected_sources.values()
    }
    assert {row["workbook_id"] for row in rows} == set(expected_sources)
    assert all(
        row["source_sha256"] == expected_sources[row["workbook_id"]] for row in rows
    )
    assert {row["comment_present"] for row in rows} <= {"true", "false"}
    assert all(len(row["source_sha256"]) == 64 for row in rows)


def test_outline_has_required_architecture_and_terminology():
    text = OUTLINE.read_text(encoding="utf-8")
    required_terms = (
        "nutrient profiling system",
        "personalized calibration",
        "Gut Microbiome-informed Nutrient Profiling System (GMNPS)",
        "correctly specified synthetic positive-control",
        "Food Compass 2.0",
        "baseline implementation",
        "method definition -> current feasibility -> robustness positive-control -> implications + hard boundary",
        "not empirical Article submission-ready",
        "AUTHOR_INPUT_NEEDED",
    )
    for term in required_terms:
        assert term in text
    lowered = text.casefold()
    assert "precision-ready" not in lowered
    assert "transforms" not in lowered
    assert "97.2%" not in text
    assert "six experts" not in lowered
    assert "two requests" not in lowered


def test_current_claim_policy_accepts_every_project_allowed_wording(tmp_path):
    for row in _rows():
        wording = row["allowed_wording"]
        if (
            not wording
            or row["claim_scope"] != "gmnps_project"
            or row["claim_id"] in {"INT-04", "INT-05"}
        ):
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


@pytest.mark.parametrize("claim_id", ["INT-04", "INT-05"])
def test_locked_project_method_route_sanitizes_only_exact_matrix_backed_claims(
    tmp_path, claim_id
):
    candidate, matrix = _project_route_fixture(tmp_path, claim_id=claim_id)
    assert check_claim_inputs([candidate])
    assert check_manuscript_claim_inputs(
        [candidate],
        matrix_path=matrix,
        reference_audit_path=REFERENCE_AUDIT,
    ) == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("section", "Methods"),
        ("claim_scope", "published_field_context"),
        ("evidence_tier", "computational_feasibility"),
        ("evidence_role", "literature_context"),
        ("evidence_family", "locked_attribute_level_method_v2"),
        ("data_class", "audited_metadata"),
        ("status", "conditional"),
        ("citation_keys", "Smith2025"),
        ("allowed_wording", "GMNPS uses a bounded calibration."),
        ("source_path", "README.md"),
        (
            "source_path",
            "docs/methods/attribute_level_gmnps_spec.md|missing-method-source.md",
        ),
        (
            "source_path",
            "docs/methods/attribute_level_gmnps_spec.md|EXTERNAL_BLOCKER: pending source",
        ),
        ("source_path", "../GMNPS/docs/methods/attribute_level_gmnps_spec.md"),
    ],
)
def test_locked_project_method_route_rejects_any_matrix_or_source_mismatch(
    tmp_path, field, value
):
    candidate, matrix = _project_route_fixture(tmp_path)
    with matrix.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    row[field] = value
    _write_csv(matrix, [row])
    assert check_manuscript_claim_inputs(
        [candidate],
        matrix_path=matrix,
        reference_audit_path=REFERENCE_AUDIT,
    )


@pytest.mark.parametrize(
    "candidate_text",
    [
        "GMNPS retains a shared food-level reference and represents personal information as a separate bounded calibration.\n",
        "% CLAIM_ID: INT-04\n% CLAIM_ID: INT-04\nGMNPS retains a shared food-level reference and represents personal information as a separate bounded calibration.\n",
        "% CLAIM_ID: INT-04\nGMNPS retains a shared food-level reference and represents personal information as a separate bounded calibration.\n% CLAIM_ID: INT-04 trailing-text\n",
        "% CLAIM_ID: INT-04\nGMNPS retains a shared food-level reference and represents personal information as a separate bounded calibration. A second sentence.\n",
        "% CLAIM_ID: INT-04\nGMNPS retains a shared food-level reference and represents personal information as a separate bounded calibration \\cite{Smith2025}.\n",
        "% CLAIM_ID: INT-04\nGMNPS retains a shared food-level reference and represents personal\ninformation as a separate bounded calibration.\n",
        "% CLAIM_ID: INT-04\nGMNPS retains a shared food-level reference and represents personal information as a separate bounded calibration that improves outcomes.\n",
    ],
)
def test_locked_project_method_route_rejects_inexact_marker_sentence_or_citation(
    tmp_path, candidate_text
):
    candidate, matrix = _project_route_fixture(tmp_path)
    candidate.write_text(candidate_text, encoding="utf-8")
    assert check_manuscript_claim_inputs(
        [candidate],
        matrix_path=matrix,
        reference_audit_path=REFERENCE_AUDIT,
    )


def test_locked_project_method_route_requires_one_marker_across_all_inputs(tmp_path):
    candidate, matrix = _project_route_fixture(tmp_path)
    duplicate = tmp_path / "duplicate.tex"
    duplicate.write_text(candidate.read_text(encoding="utf-8"), encoding="utf-8")
    assert check_manuscript_claim_inputs(
        [candidate, duplicate],
        matrix_path=matrix,
        reference_audit_path=REFERENCE_AUDIT,
    )


@pytest.mark.parametrize(
    "suffix",
    [
        " It improves outcomes.",
        "\nIt improves outcomes.",
        "\nIt improves\noutcomes.",
        "\nIt improves % formatting note\noutcomes.",
        "\nIt improves\n% formatting note\noutcomes.",
        "\nIt improves\n\\label{claim:x}\noutcomes.",
        "\nIt improves\n\\index{claim assertion}\noutcomes.",
        "\nIt improves\n\\phantomsection\noutcomes.",
        "\nIt impro% formatting note\nves outcomes.",
        "\nIt impro\\label{claim:x}ves outcomes.",
        "\nIt impro\\index{claim assertion}ves outcomes.",
        "\nIt impro\\phantomsection ves outcomes.",
        "\nThis improves prediction performance.",
        "\nThat improves response validity.",
        "\nThese improve outcomes.",
        "\nThose improve outcomes.",
        "\nThey improve outcomes.",
        "\nPerformance improves.",
        "\nPerformance\nimproves.",
        "\nPrediction accuracy improves.",
    ],
)
def test_locked_project_route_cannot_authorize_appended_empirical_assertions(
    tmp_path, suffix
):
    candidate, matrix = _project_route_fixture(tmp_path)
    candidate.write_text(
        candidate.read_text(encoding="utf-8").rstrip("\n") + suffix + "\n",
        encoding="utf-8",
    )
    assert check_manuscript_claim_inputs(
        [candidate],
        matrix_path=matrix,
        reference_audit_path=REFERENCE_AUDIT,
    )


@pytest.mark.parametrize(
    "continuation",
    [
        " It improves outcomes.",
        "\nIt improves outcomes.",
        "\nIt improves\noutcomes.",
        "\nIt improves % formatting note\noutcomes.",
        "\nIt improves\n% formatting note\noutcomes.",
        "\nIt improves\n\\label{claim:x}\noutcomes.",
        "\nIt improves\n\\index{claim assertion}\noutcomes.",
        "\nIt improves\n\\phantomsection\noutcomes.",
        "\nIt impro% formatting note\nves outcomes.",
        "\nIt impro\\label{claim:x}ves outcomes.",
        "\nIt impro\\index{claim assertion}ves outcomes.",
        "\nIt impro\\phantomsection ves outcomes.",
    ],
)
def test_manuscript_wrapper_rejects_assertions_after_exact_identifier(
    tmp_path, continuation
):
    candidate, matrix = _project_route_fixture(tmp_path)
    candidate.write_text(
        r"The frozen specification is \texttt{attribute-gmnps-v1}."
        + continuation
        + "\n",
        encoding="utf-8",
    )
    assert check_manuscript_claim_inputs(
        [candidate],
        matrix_path=matrix,
        reference_audit_path=REFERENCE_AUDIT,
    )


def test_locked_project_route_preserves_benign_following_lines(tmp_path):
    candidate, matrix = _project_route_fixture(tmp_path)
    candidate.write_text(
        candidate.read_text(encoding="utf-8")
        + "This response is stored\nin the manifest.\n"
        + "Performance was not evaluated.\n"
        + "This does not improve % formatting note\noutcomes.\n"
        + "This response is stored\n\\label{manifest:x}\nin the manifest.\n"
        + r"The threshold is 20\%." "\n"
        + "Performance\n\\section*{Methods}\nimproves.\n",
        encoding="utf-8",
    )
    assert check_manuscript_claim_inputs(
        [candidate],
        matrix_path=matrix,
        reference_audit_path=REFERENCE_AUDIT,
    ) == ()


def test_published_context_route_can_bypass_one_phase2_violation(tmp_path):
    candidate, matrix, audit = _route_fixture(tmp_path)
    assert check_claim_inputs([candidate])
    assert check_manuscript_claim_inputs(
        [candidate], matrix_path=matrix, reference_audit_path=audit
    ) == ()


def test_published_route_sanitizes_only_the_authorized_physical_line(tmp_path):
    candidate, matrix, audit = _route_fixture(tmp_path)
    authorized_claim = (
        "Microbiome-informed framework supports individualized dietary response "
        "prediction \\cite{Smith2025}."
    )
    project_claim = "A separate framework supports individualized dietary response prediction."
    authorized_path = tmp_path / "authorized-only.tex"
    authorized_path.write_text(authorized_claim + "\n", encoding="utf-8")
    project_path = tmp_path / "project-only.tex"
    project_path.write_text(project_claim + "\n", encoding="utf-8")
    authorized_violations = check_claim_inputs([authorized_path])
    project_violations = check_claim_inputs([project_path])
    assert len(authorized_violations) == 1
    assert len(project_violations) == 1
    assert authorized_violations[0].pattern
    assert authorized_violations[0].pattern == project_violations[0].pattern

    candidate.write_text(
        project_claim
        + "\n% CLAIM_ID: INT-01\n"
        + authorized_claim
        + "\n",
        encoding="utf-8",
    )
    observed = check_manuscript_claim_inputs(
        [candidate], matrix_path=matrix, reference_audit_path=audit
    )
    assert len(observed) == 1
    assert observed[0].path == candidate
    assert observed[0].pattern == project_violations[0].pattern


def test_last_introduction_marker_does_not_consume_following_results(tmp_path):
    candidate, matrix, audit = _route_fixture(tmp_path)
    results_claim = "Results validate clinical response prediction."
    candidate.write_text(
        "% CLAIM_ID: INT-01\n"
        "Microbiome-informed framework supports individualized dietary response "
        "prediction \\cite{Smith2025}.\n"
        + results_claim
        + "\n",
        encoding="utf-8",
    )
    results_only = tmp_path / "results-only.tex"
    results_only.write_text(results_claim + "\n", encoding="utf-8")
    expected_patterns = [item.pattern for item in check_claim_inputs([results_only])]
    observed = check_manuscript_claim_inputs(
        [candidate], matrix_path=matrix, reference_audit_path=audit
    )
    assert [item.pattern for item in observed] == expected_patterns
    assert all(item.path == candidate for item in observed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evidence_tier", "computational_feasibility"),
        ("evidence_role", "method_definition_or_invariant"),
        ("data_class", "computational_method"),
        ("claim_scope", "gmnps_project"),
        ("status", "conditional"),
    ],
)
def test_published_context_route_requires_exact_matrix_classification(
    tmp_path, field, value
):
    candidate, matrix, audit = _route_fixture(tmp_path)
    with matrix.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    row[field] = value
    _write_csv(matrix, [row])
    assert check_manuscript_claim_inputs(
        [candidate], matrix_path=matrix, reference_audit_path=audit
    )


@pytest.mark.parametrize(
    "candidate_text",
    [
        "Microbiome-informed framework supports individualized dietary response prediction \\cite{Smith2025}.\n",
        "% CLAIM_ID: INT-01\n% CLAIM_ID: INT-01\nMicrobiome-informed framework supports individualized dietary response prediction \\cite{Smith2025}.\n",
        "% CLAIM_ID: INT-01\nMicrobiome-informed framework supports individualized dietary response prediction \\cite{Smith2025}. A second sentence.\n",
        "% CLAIM_ID: INT-01\nMicrobiome-informed framework supports broad dietary response prediction \\cite{Smith2025}.\n",
        "% CLAIM_ID: INT-01\nMicrobiome-informed framework supports individualized dietary response prediction.\n",
        "% CLAIM_ID: INT-01\nMicrobiome-informed framework supports individualized dietary response prediction \\cite{Smith2025,Extra2026}.\n",
        "% CLAIM_ID: INT-01\nOur microbiome-informed framework supports individualized dietary response prediction \\cite{Smith2025}.\n",
    ],
)
def test_published_context_route_rejects_inexact_claims(tmp_path, candidate_text):
    candidate, matrix, audit = _route_fixture(tmp_path)
    candidate.write_text(candidate_text, encoding="utf-8")
    assert check_manuscript_claim_inputs(
        [candidate], matrix_path=matrix, reference_audit_path=audit
    )


@pytest.mark.parametrize(
    "deictic_wording",
    [
        "The framework introduced here supports individualized dietary response prediction.",
        "The framework presented here supports individualized dietary response prediction.",
        "The framework reported here supports individualized dietary response prediction.",
        "The framework developed here supports individualized dietary response prediction.",
        "Here we describe a framework supporting individualized dietary response prediction.",
        "This work describes a framework supporting individualized dietary response prediction.",
        "This paper describes a framework supporting individualized dietary response prediction.",
        "This article describes a framework supporting individualized dietary response prediction.",
        "The framework proposed here supports individualized dietary response prediction.",
        "This manuscript describes a framework supporting individualized dietary response prediction.",
        *[
            f"The proposed {noun} supports individualized dietary response prediction."
            for noun in (
                "framework",
                "method",
                "approach",
                "system",
                "model",
                "algorithm",
            )
        ],
        *[
            f"The {noun} {action} here supports individualized dietary response prediction."
            for noun in (
                "framework",
                "method",
                "approach",
                "system",
                "model",
                "algorithm",
            )
            for action in (
                "proposed",
                "developed",
                "introduced",
                "presented",
                "reported",
            )
        ],
    ],
)
def test_published_context_route_rejects_project_deixis(tmp_path, deictic_wording):
    candidate, matrix, audit = _route_fixture(tmp_path)
    with matrix.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    row["allowed_wording"] = deictic_wording
    _write_csv(matrix, [row])
    candidate.write_text(
        "% CLAIM_ID: INT-01\n"
        + deictic_wording.removesuffix(".")
        + " \\cite{Smith2025}.\n",
        encoding="utf-8",
    )
    assert check_manuscript_claim_inputs(
        [candidate], matrix_path=matrix, reference_audit_path=audit
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("verification_status", "pending"),
        ("stable_identifier", ""),
        ("retraction_status", "unknown"),
        ("citation_key", "Other2025"),
        ("claim_id", "INT-02"),
    ],
)
def test_published_context_route_requires_verified_reference_pairs(
    tmp_path, field, value
):
    candidate, matrix, audit = _route_fixture(tmp_path)
    with audit.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    row[field] = value
    _write_csv(audit, [row])
    assert check_manuscript_claim_inputs(
        [candidate], matrix_path=matrix, reference_audit_path=audit
    )


@pytest.mark.parametrize(
    "identifier",
    [
        "doi:10.1038/s41586-025-01234-5",
        "pmid:12345678",
        "pmcid:PMC1234567",
        "url:https://example.org/articles/stable-record",
    ],
)
def test_published_context_route_accepts_explicit_stable_identifiers(
    tmp_path, identifier
):
    candidate, matrix, audit = _route_fixture(tmp_path)
    with audit.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    row["stable_identifier"] = identifier
    _write_csv(audit, [row])
    assert check_manuscript_claim_inputs(
        [candidate], matrix_path=matrix, reference_audit_path=audit
    ) == ()


@pytest.mark.parametrize(
    "identifier",
    [
        "10.1038/s41586-025-01234-5",
        "doi:pending",
        "doi:10.1038/PENDING_TASK3",
        "pmid:not-a-number",
        "pmcid:1234567",
        "url:http://example.org/article",
        "verified elsewhere",
    ],
)
def test_published_context_route_rejects_non_stable_identifiers(
    tmp_path, identifier
):
    candidate, matrix, audit = _route_fixture(tmp_path)
    with audit.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    row["stable_identifier"] = identifier
    _write_csv(audit, [row])
    assert check_manuscript_claim_inputs(
        [candidate], matrix_path=matrix, reference_audit_path=audit
    )


def test_published_context_route_requires_direct_claim_support(tmp_path):
    candidate, matrix, audit = _route_fixture(tmp_path)
    with audit.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    row["support_role"] = "field_context"
    _write_csv(audit, [row])
    assert check_manuscript_claim_inputs(
        [candidate], matrix_path=matrix, reference_audit_path=audit
    )


@pytest.mark.parametrize("extra_kind", ["duplicate", "unrequested"])
def test_published_context_route_rejects_non_exact_audit_rows(tmp_path, extra_kind):
    candidate, matrix, audit = _route_fixture(tmp_path)
    with audit.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    extra = dict(row)
    if extra_kind == "unrequested":
        extra["citation_key"] = "Unrequested2026"
    _write_csv(audit, [row, extra])
    assert check_manuscript_claim_inputs(
        [candidate], matrix_path=matrix, reference_audit_path=audit
    )


def test_conditional_task1_literature_row_remains_denied(tmp_path):
    candidate, matrix, audit = _route_fixture(tmp_path)
    with matrix.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    row["status"] = "conditional"
    _write_csv(matrix, [row])
    assert check_manuscript_claim_inputs(
        [candidate], matrix_path=matrix, reference_audit_path=audit
    )


def test_unmarked_and_marked_project_claims_remain_default_denied(tmp_path):
    candidate, matrix, audit = _route_fixture(tmp_path)
    project_claim = "GMNPS supports individualized dietary response prediction."
    with matrix.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    row.update(
        {
            "allowed_wording": project_claim,
            "evidence_tier": "locked_method_definition",
            "evidence_role": "method_definition_or_invariant",
            "data_class": "computational_method",
            "claim_scope": "gmnps_project",
        }
    )
    _write_csv(matrix, [row])
    for text in (project_claim, f"% CLAIM_ID: INT-01\n{project_claim}"):
        candidate.write_text(text, encoding="utf-8")
        assert check_manuscript_claim_inputs(
            [candidate], matrix_path=matrix, reference_audit_path=audit
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
