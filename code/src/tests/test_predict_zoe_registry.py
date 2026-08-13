from __future__ import annotations

import csv
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import re

import pytest

from gmnps.data_sources.predict_zoe_registry import (
    ACCESS_ROUTES,
    ACCESS_STATUSES,
    ANALYTICAL_ROLES,
    CONTROLLED_OUTCOME_GRANTS,
    PREDICT1_EH5458_OVERLAP_AUDIT,
    PREDICT_ZOE_SOURCE_REGISTRY,
    PREDICTOR_ARTIFACT_REGISTRY,
    PROVENANCE_FIELDS,
    ControlledOutcomeGrant,
    OutcomeEndpointGrant,
    PredictorArtifact,
    canonical_predictor_ids_sha256,
    get_predict_zoe_source,
    predictor_id_overlap_audit,
    validate_controlled_outcome_grants,
    validate_predictor_artifact_registry,
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


def test_current_synthetic_aggregate_and_ungranted_controlled_resources_are_not_direct():
    forbidden_statuses = {"synthetic_local", "aggregate_public"}
    for row in PREDICT_ZOE_SOURCE_REGISTRY:
        if row.real_synthetic_status in forbidden_statuses:
            assert row.allowed_analytical_role != "direct_validation"

    controlled = get_predict_zoe_source("predict_controlled_clinical_zenodo")
    assert controlled.real_synthetic_status == "controlled_real"
    assert controlled.access_status == "controlled_not_granted"
    assert controlled.allowed_analytical_role == "controlled_eligibility_assessment"

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
    with pytest.raises(ValueError, match="allowed analytical role|direct validation"):
        validate_predict_zoe_registry(
            (row.__class__(**{**row.as_dict(), "allowed_analytical_role": "direct_validation"}),)
        )


def test_repository_predictor_artifacts_pin_locally_audited_digests_and_schema():
    validate_predictor_artifact_registry(PREDICTOR_ARTIFACT_REGISTRY)
    by_id = {row.source_id: row for row in PREDICTOR_ARTIFACT_REGISTRY}
    assert set(by_id) == {
        "experimenthub_eh5458_relative_abundance",
        "ena_prjeb39223_sequencing_metadata",
        "ena_prjeb75460_sequencing_metadata",
        "ena_prjeb75462_sequencing_metadata",
        "ena_prjeb75463_sequencing_metadata",
        "ena_prjeb75464_sequencing_metadata",
    }
    expected = {
        "experimenthub_eh5458_relative_abundance": "89c635061b357583351ca33f520a72d0efced4a2963e73182e529228c8395c54",
        "ena_prjeb39223_sequencing_metadata": "d630be36c1abd3a71b6aa295dc56e1557557f965f6c8fdaa673759b227b28e36",
        "ena_prjeb75460_sequencing_metadata": "2ce6d90cd5d380581737a9dc193460b3f5df7f836f443845bba793ecb67de1a9",
        "ena_prjeb75462_sequencing_metadata": "ca6d7d5a1b7fdcd069a309b5bc8dd076a53ea8bc4016ec98b1b755658fd20f08",
        "ena_prjeb75463_sequencing_metadata": "ce028e6072e610ad7575ba7e494347892cd66e33a5eaec92dea7177bcb413f56",
        "ena_prjeb75464_sequencing_metadata": "f81d06da5f9127864967f99bb13a2b7f6eedddc8c56282f88c5b04985cd4bb94",
    }
    assert {source_id: row.sha256 for source_id, row in by_id.items()} == expected
    for row in by_id.values():
        assert row.content_class == "predictor_only"
        assert row.allowed_analytical_role == "predictor_reconstruction"
        assert row.path_policy == "repository_relative_exact_no_symlink"
        assert row.digest_basis == "locally_audited_sha256_pinned_in_repository"
        assert SHA256.fullmatch(row.sha256)
        assert row.allowed_columns
        assert row.schema_kind


def _valid_temporary_grant(tmp_path: Path):
    outcome = tmp_path / "controlled.csv"
    outcome.write_bytes(
        b"participant_id,meal_id,glucose_source,tg_source\nP1,M1,1.0,0.2\n"
    )
    dictionary = tmp_path / "data_dictionary.json"
    dictionary.write_bytes(b'{"version":"test-dictionary-v1"}\n')
    source = replace(
        get_predict_zoe_source("predict_controlled_clinical_zenodo"),
        access_status="controlled_granted",
        allowed_analytical_role="direct_validation",
        local_path=str(outcome),
        sha256=sha256(outcome.read_bytes()).hexdigest(),
        exclusion_reason="none_after_verified_controlled_grant",
        evidence_basis="temporary non-production grant fixture",
    )
    endpoints = (
        OutcomeEndpointGrant(
            name="glucose_iAUC_2h",
            source_column="glucose_source",
            availability="available",
            role="primary",
            unit="mmol_L_hour",
            window_hours=(0, 2),
            summary="incremental_area_under_curve",
            derivation="baseline_subtracted_trapezoidal_auc_signed_excursions",
        ),
        OutcomeEndpointGrant(
            name="tg_6h_rise",
            source_column="tg_source",
            availability="available",
            role="primary",
            unit="mmol_L",
            window_hours=(0, 6),
            summary="rise_above_baseline",
            derivation="six_hour_value_minus_time_zero_baseline",
        ),
        OutcomeEndpointGrant(
            name="c_peptide_iAUC_2h",
            source_column=None,
            availability="not_available",
            role="secondary",
            unit="nmol_L_hour",
            window_hours=(0, 2),
            summary="incremental_area_under_curve",
            derivation="baseline_subtracted_trapezoidal_auc_signed_excursions",
        ),
    )
    grant = ControlledOutcomeGrant(
        source_id=source.resource_id,
        access_status="controlled_granted",
        allowed_analytical_role="direct_validation",
        verified_local_path=str(outcome),
        sha256=source.sha256,
        file_format="csv",
        version_doi="10.5281/zenodo.17236383",
        approval_evidence_identifier="DUA-TEST-APPROVED",
        data_dictionary_path=str(dictionary),
        data_dictionary_sha256=sha256(dictionary.read_bytes()).hexdigest(),
        participant_meal_key_contract=("participant_id", "meal_id"),
        endpoint_contracts=endpoints,
        microbiome_linkage_evidence_identifier="LINKAGE-TEST-VERIFIED",
        microbiome_linkage_key_contract=("participant_id",),
    )
    return source, grant


@pytest.mark.parametrize(
    "field",
    [
        "verified_local_path",
        "sha256",
        "file_format",
        "version_doi",
        "approval_evidence_identifier",
        "data_dictionary_path",
        "data_dictionary_sha256",
        "participant_meal_key_contract",
        "endpoint_contracts",
        "microbiome_linkage_evidence_identifier",
        "microbiome_linkage_key_contract",
    ],
)
def test_controlled_grant_requires_all_evidence_before_direct_validation(tmp_path, field):
    source, grant = _valid_temporary_grant(tmp_path)
    validate_controlled_outcome_grants((grant,), sources=(source,))
    missing = () if field == "endpoint_contracts" else None
    with pytest.raises(ValueError, match="grant|evidence|contract|missing"):
        validate_controlled_outcome_grants(
            (replace(grant, **{field: missing}),),
            sources=(source,),
        )


def test_production_controlled_grant_remains_not_granted_but_contract_is_reachable():
    validate_controlled_outcome_grants(
        CONTROLLED_OUTCOME_GRANTS,
        sources=PREDICT_ZOE_SOURCE_REGISTRY,
    )
    assert len(CONTROLLED_OUTCOME_GRANTS) == 1
    grant = CONTROLLED_OUTCOME_GRANTS[0]
    assert grant.source_id == "predict_controlled_clinical_zenodo"
    assert grant.access_status == "controlled_not_granted"
    assert grant.allowed_analytical_role == "controlled_eligibility_assessment"
    assert grant.verified_local_path is None


def test_predict1_overlap_digest_contract_is_exact_case_sorted_newline_without_trailing_newline():
    assert canonical_predictor_ids_sha256(("b", "A", "a")) == sha256(
        b"A\na\nb"
    ).hexdigest()
    with pytest.raises(ValueError, match="duplicate"):
        canonical_predictor_ids_sha256(("same", "same"))

    audit = predictor_id_overlap_audit(
        ("development-only", "SAMEA2", "SAMEA1"),
        ("SAMEA1", "SAMEA2"),
    )
    assert audit == {
        "development_n": 3,
        "development_ids_sha256": sha256(
            b"SAMEA1\nSAMEA2\ndevelopment-only"
        ).hexdigest(),
        "candidate_n": 2,
        "candidate_ids_sha256": sha256(b"SAMEA1\nSAMEA2").hexdigest(),
        "intersection_n": 2,
        "intersection_ids_sha256": sha256(b"SAMEA1\nSAMEA2").hexdigest(),
    }

    recorded = dict(PREDICT1_EH5458_OVERLAP_AUDIT)
    assert recorded["development_n"] == 15492
    assert recorded["eh5458_n"] == recorded["intersection_n"] == 1098
    assert recorded["eh5458_ids_sha256"] == recorded["intersection_ids_sha256"]
