from __future__ import annotations

import csv
from pathlib import Path
import re

import pytest

from gmnps.data_sources.predict_zoe_registry import (
    ACCESS_ROUTES,
    ACCESS_STATUSES,
    ANALYTICAL_ROLES,
    PREDICT_ZOE_SOURCE_REGISTRY,
    PROVENANCE_FIELDS,
    get_predict_zoe_source,
    validate_predict_zoe_registry,
)


ROOT = Path(__file__).resolve().parents[3]
DOCS_DATA = ROOT / "docs/data"
SOURCE_MANIFEST = DOCS_DATA / "predict_zoe_source_manifest.csv"
REAL_SYNTHETIC_AUDIT = DOCS_DATA / "predict1_real_vs_synthetic_audit.csv"
ELIGIBILITY_AUDIT = DOCS_DATA / "eligible_for_main_validation.csv"
AVAILABILITY_AUDIT = DOCS_DATA / "data_availability_audit.md"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_registry_is_complete_unique_and_uses_one_route_and_role_per_resource():
    validate_predict_zoe_registry(PREDICT_ZOE_SOURCE_REGISTRY)
    assert PREDICT_ZOE_SOURCE_REGISTRY
    assert len({row.resource_id for row in PREDICT_ZOE_SOURCE_REGISTRY}) == len(
        PREDICT_ZOE_SOURCE_REGISTRY
    )
    for row in PREDICT_ZOE_SOURCE_REGISTRY:
        payload = row.as_dict()
        assert set(PROVENANCE_FIELDS) <= set(payload)
        assert all(str(payload[field]).strip() for field in PROVENANCE_FIELDS)
        assert row.access_route in ACCESS_ROUTES
        assert row.access_status in ACCESS_STATUSES
        assert row.allowed_analytical_role in ANALYTICAL_ROLES
        assert ";" not in row.access_route
        assert ";" not in row.allowed_analytical_role
        assert SHA256.fullmatch(row.sha256) or row.sha256.startswith("not_computed_")


def test_known_identifiers_versions_and_uploaded_workbook_digest_are_verified():
    expected = {
        "predict1_ena_raw_metagenomes": "PRJEB39223",
        "predict1_experimenthub_profiles": "EH5458",
        "predict2_ena_raw_metagenomes": "PRJEB75460",
        "predict3_us21_ena_raw_metagenomes": "PRJEB75462",
        "predict3_us22a_ena_raw_metagenomes": "PRJEB75463",
        "predict3_uk22a_ena_raw_metagenomes": "PRJEB75464",
    }
    for resource_id, identifier in expected.items():
        assert get_predict_zoe_source(resource_id).source_identifier == identifier

    public_zenodo = get_predict_zoe_source("predict_zoe_public_profiles_zenodo")
    assert public_zenodo.source_identifier == "10.5281/zenodo.15307999"
    assert public_zenodo.version_identifier == "10.5281/zenodo.15308000"

    code_zenodo = get_predict_zoe_source("predict_external_archive_zenodo")
    assert code_zenodo.source_identifier == "10.5281/zenodo.17236261"
    assert code_zenodo.version_identifier == "10.5281/zenodo.17236262"

    controlled = get_predict_zoe_source("predict_controlled_clinical_zenodo")
    assert controlled.source_identifier == "10.5281/zenodo.17236382"
    assert controlled.version_identifier == "10.5281/zenodo.17236383"

    release = get_predict_zoe_source("inverse_var_weight_v1_0_0")
    assert release.version_identifier == "ab1a974bf1abd66175d190b55183d2957224b39f"

    workbook = get_predict_zoe_source("nature_zoe_supplementary_tables")
    assert workbook.sha256 == "3b6034d6212f0676bbc60b6eb0f3713cd5266799cbaf37e035900afe3892e581"


def test_source_manifest_matches_registry_and_requires_explicit_unknowns():
    rows = _read_csv(SOURCE_MANIFEST)
    assert rows
    assert set(PROVENANCE_FIELDS) <= set(rows[0])
    assert {row["resource_id"] for row in rows} == {
        row.resource_id for row in PREDICT_ZOE_SOURCE_REGISTRY
    }
    assert rows == [row.as_dict() for row in PREDICT_ZOE_SOURCE_REGISTRY]
    for row in rows:
        assert all(row[field].strip() for field in PROVENANCE_FIELDS)
        assert row["access_route"] in ACCESS_ROUTES
        assert row["allowed_analytical_role"] in ANALYTICAL_ROLES
        assert row["sample_size"] == "unknown" or row["sample_size"].isdigit()
        assert row["n_value"].strip()
        assert row["n_unit"].strip()
        assert row["participant_n"].strip()
        assert row["access_status"].strip()


def test_ena_counts_are_explicitly_sample_run_counts_not_participant_counts():
    ena_rows = [
        row
        for row in PREDICT_ZOE_SOURCE_REGISTRY
        if row.source_identifier.startswith("PRJEB")
    ]
    assert ena_rows
    for row in ena_rows:
        assert row.n_value == row.sample_size
        assert row.n_unit == "ENA sample/run record"
        assert row.participant_n == "unknown"


def test_experimenthub_evidence_has_no_unreproducible_snapshot_date():
    row = get_predict_zoe_source("predict1_experimenthub_profiles")
    assert "snapshot dated" not in row.evidence_basis
    assert "2026-07-30" not in row.evidence_basis


def test_live_zoe_ranking_and_immutable_supplementary_s5_are_separate_resources():
    supplement = get_predict_zoe_source("nature_supplementary_table_s5_rankings")
    live = get_predict_zoe_source("zoe_live_microbiome_rankings_2024")

    assert "media.springernature.com" in supplement.stable_source
    assert supplement.source_identifier.endswith(":S5")
    assert supplement.access_route == "within_paper_or_supplement"
    assert supplement.local_path.endswith("zoe_health_rank_S5.csv")
    assert SHA256.fullmatch(supplement.sha256)

    assert live.stable_source == "https://zoe.com/our-science/microbiome-ranking"
    assert live.access_route == "reused_public_source"
    assert live.local_path == "not_downloaded"
    assert live.sha256.startswith("not_computed_")
    assert "supplementary" not in live.version_identifier.lower()
    assert live.sha256 != supplement.sha256


def test_synthetic_aggregate_and_controlled_resources_are_never_direct_validation():
    forbidden_statuses = {"synthetic_local", "aggregate_public", "controlled_real"}
    for row in PREDICT_ZOE_SOURCE_REGISTRY:
        if row.real_synthetic_status in forbidden_statuses:
            assert row.allowed_analytical_role != "direct_validation"

    audit = _read_csv(REAL_SYNTHETIC_AUDIT)
    by_endpoint = {row["declared_response_endpoint"]: row for row in audit}
    for endpoint in ("glucose_iAUC_2h", "tg_6h_rise", "c_peptide_iAUC_2h"):
        assert by_endpoint[endpoint]["real_synthetic_status"] == "synthetic_local"
        assert by_endpoint[endpoint]["allowed_analytical_role"] == "synthetic_stress_test"
        assert "PROVENANCE.md" in by_endpoint[endpoint]["evidence_basis"]
        assert by_endpoint[endpoint]["access_route"] == "within_paper_or_supplement"
        assert (
            by_endpoint[endpoint]["access_status"]
            == "local_pending_submission_package"
        )

    local_synthetic = [
        row
        for row in PREDICT_ZOE_SOURCE_REGISTRY
        if row.real_synthetic_status == "synthetic_local"
    ]
    assert local_synthetic
    for row in local_synthetic:
        assert row.access_route == "within_paper_or_supplement"
        assert row.access_status == "local_pending_submission_package"
        assert "pending" in row.source_identifier


def test_every_audit_row_has_exactly_one_known_access_route_and_role():
    for path in (SOURCE_MANIFEST, REAL_SYNTHETIC_AUDIT, ELIGIBILITY_AUDIT):
        for row in _read_csv(path):
            assert row["access_route"] in ACCESS_ROUTES
            assert row["allowed_analytical_role"] in ANALYTICAL_ROLES
            assert row["access_status"] in ACCESS_STATUSES
            assert ";" not in row["access_route"]
            assert ";" not in row["allowed_analytical_role"]
            assert row["access_status"].strip()


def test_no_current_resource_is_marked_eligible_for_main_validation():
    rows = _read_csv(ELIGIBILITY_AUDIT)
    assert rows
    assert {row["eligible_for_main_validation"] for row in rows} == {"false"}
    for row in rows:
        assert row["exclusion_reason"].strip()
        assert row["allowed_analytical_role"] != "direct_validation"


def test_unproven_participant_and_meal_linkages_remain_unknown():
    rows = _read_csv(ELIGIBILITY_AUDIT)
    by_id = {row["resource_id"]: row for row in rows}
    unproven_microbiome_linkage = {
        "predict1_ena_raw_metagenomes",
        "predict1_experimenthub_profiles",
        "predict2_ena_raw_metagenomes",
        "predict3_us21_ena_raw_metagenomes",
        "predict3_us22a_ena_raw_metagenomes",
        "predict3_uk22a_ena_raw_metagenomes",
        "predict_zoe_public_profiles_zenodo",
        "predict_controlled_clinical_zenodo",
        "local_predict1_real_subject_metadata",
        "local_predict1_synthetic_glucose",
        "local_predict1_synthetic_triglyceride",
        "local_predict1_synthetic_c_peptide",
    }
    for resource_id in unproven_microbiome_linkage:
        assert by_id[resource_id]["has_linkable_microbiome"] == "unknown"

    for resource_id in (
        "local_predict1_synthetic_glucose",
        "local_predict1_synthetic_triglyceride",
        "local_predict1_synthetic_c_peptide",
    ):
        assert by_id[resource_id]["has_meal_or_food_key"] == "unknown"


def test_data_availability_audit_records_source_hierarchy_fair_scope_and_access_date():
    text = AVAILABILITY_AUDIT.read_text(encoding="utf-8")
    for required in (
        "https://www.nature.com/articles/s41586-025-09854-7#data-availability",
        "https://www.nature.com/articles/s41586-025-09854-7#code-availability",
        "2026-08-13",
        "FAIR",
        "participant-level outcome",
        "3b6034d6212f0676bbc60b6eb0f3713cd5266799cbaf37e035900afe3892e581",
        "10.5281/zenodo.17236383",
        "local_pending_submission_package",
        "identifier pending",
        "before submission",
        "ENA sample/run record",
        "participant_n",
    ):
        assert required in text

    assert "nature_supplementary_table_s5_rankings" in text
    assert "zoe_live_microbiome_rankings_2024" in text


def test_registry_rejects_duplicate_ids_missing_provenance_and_invalid_roles():
    row = PREDICT_ZOE_SOURCE_REGISTRY[0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_predict_zoe_registry((row, row))
    with pytest.raises(ValueError, match="allowed analytical role"):
        validate_predict_zoe_registry(
            (row.__class__(**{**row.as_dict(), "allowed_analytical_role": "direct_validation"}),)
        )
