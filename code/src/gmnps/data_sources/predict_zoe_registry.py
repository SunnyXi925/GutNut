"""Pre-outcome provenance registry for PREDICT and ZOE resources.

The registry classifies source eligibility without loading participant-level
outcome or label values.  It is intentionally conservative: no Task 1 record
is authorized for direct validation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Iterable, Mapping


ACCESS_ROUTES = frozenset(
    {
        "public_repository",
        "controlled_access_repository",
        "within_paper_or_supplement",
        "reused_public_source",
        "third_party_restricted",
        "available_on_justified_request",
        "not_applicable",
    }
)

ANALYTICAL_ROLES = frozenset(
    {
        "predictor_reconstruction",
        "biological_consistency",
        "aggregate_supporting_evidence",
        "synthetic_stress_test",
        "controlled_eligibility_assessment",
        "reproducibility_support",
        "provenance_only",
    }
)

REAL_SYNTHETIC_STATUSES = frozenset(
    {
        "real_public",
        "controlled_real",
        "aggregate_public",
        "synthetic_local",
        "software_public",
        "metadata_public",
    }
)

PROVENANCE_FIELDS = (
    "resource_id",
    "source_name",
    "stable_source",
    "source_identifier",
    "version_identifier",
    "accessed_on",
    "owner",
    "cohort",
    "unit_of_observation",
    "sample_size",
    "data_modality",
    "declared_response_endpoint",
    "access_route",
    "local_path",
    "sha256",
    "real_synthetic_status",
    "allowed_analytical_role",
    "exclusion_reason",
    "evidence_basis",
)


@dataclass(frozen=True)
class PredictZoeSource:
    """One provenance-classified resource with one route and one role."""

    resource_id: str
    source_name: str
    stable_source: str
    source_identifier: str
    version_identifier: str
    accessed_on: str
    owner: str
    cohort: str
    unit_of_observation: str
    sample_size: str
    data_modality: str
    declared_response_endpoint: str
    access_route: str
    local_path: str
    sha256: str
    real_synthetic_status: str
    allowed_analytical_role: str
    exclusion_reason: str
    evidence_basis: str

    def as_dict(self) -> dict[str, str]:
        """Return a CSV-compatible copy of this immutable record."""

        return asdict(self)


_ACCESSED = "2026-08-13"
_NOT_DOWNLOADED = "not_downloaded"
_NO_REMOTE_SHA = "not_computed_remote_not_downloaded"
_NATURE_DATA = (
    "https://www.nature.com/articles/s41586-025-09854-7#data-availability"
)


PREDICT_ZOE_SOURCE_REGISTRY = (
    PredictZoeSource(
        resource_id="nature_zoe_supplementary_tables",
        source_name="Nature Supplementary Tables 1-25",
        stable_source=(
            "https://media.springernature.com/original/springer-static/esm/"
            "art%3A10.1038%2Fs41586-025-09854-7/MediaObjects/"
            "41586_2025_9854_MOESM3_ESM.xlsx"
        ),
        source_identifier="10.1038/s41586-025-09854-7:MOESM3",
        version_identifier="41586_2025_9854_MOESM3_ESM.xlsx",
        accessed_on=_ACCESSED,
        owner="Asnicar et al.; hosted by Springer Nature",
        cohort="ZOE PREDICT and non-PREDICT analyses",
        unit_of_observation="published aggregate table row",
        sample_size="unknown",
        data_modality="aggregate supplementary tables and species rankings",
        declared_response_endpoint=(
            "aggregate clinical-marker prediction, association, and intervention summaries"
        ),
        access_route="within_paper_or_supplement",
        local_path="/Users/fengxi.25/Downloads/41586_2025_9854_MOESM3_ESM.xlsx",
        sha256="3b6034d6212f0676bbc60b6eb0f3713cd5266799cbaf37e035900afe3892e581",
        real_synthetic_status="aggregate_public",
        allowed_analytical_role="aggregate_supporting_evidence",
        exclusion_reason=(
            "Aggregate workbook; it does not expose eligible participant-by-meal outcomes."
        ),
        evidence_basis=(
            "Official Nature supplementary download; uploaded and official bytes were identical."
        ),
    ),
    PredictZoeSource(
        resource_id="predict1_ena_raw_metagenomes",
        source_name="PREDICT 1 raw metagenomes",
        stable_source="https://www.ebi.ac.uk/ena/browser/view/PRJEB39223",
        source_identifier="PRJEB39223",
        version_identifier="ERP122716",
        accessed_on=_ACCESSED,
        owner="University of Trento",
        cohort="PREDICT 1",
        unit_of_observation="ENA metagenomic sample/run",
        sample_size="2196",
        data_modality="shotgun metagenomic sequencing",
        declared_response_endpoint="none in the ENA sequencing resource",
        access_route="reused_public_source",
        local_path=_NOT_DOWNLOADED,
        sha256=_NO_REMOTE_SHA,
        real_synthetic_status="real_public",
        allowed_analytical_role="predictor_reconstruction",
        exclusion_reason=(
            "Sequencing predictors only; no public participant-by-meal metabolic response table."
        ),
        evidence_basis=(
            "ENA study metadata and count API; the Nature article reports 1,098 PREDICT 1 participants."
        ),
    ),
    PredictZoeSource(
        resource_id="predict1_experimenthub_profiles",
        source_name="curatedMetagenomicData AsnicarF_2021 relative abundance",
        stable_source="https://experimenthub.bioconductor.org/fetch/5501",
        source_identifier="EH5458",
        version_identifier="2021-03-31.AsnicarF_2021.relative_abundance",
        accessed_on=_ACCESSED,
        owner="curatedMetagenomicData / ExperimentHub; source provider NCBI",
        cohort="PREDICT 1",
        unit_of_observation="participant microbiome profile",
        sample_size="unknown",
        data_modality="curated species relative-abundance profiles",
        declared_response_endpoint="none in the ExperimentHub profile resource",
        access_route="reused_public_source",
        local_path=(
            "data/project_data/predict_multi/L2_phenotype/predict1_real/"
            "AsnicarF_2021_relative_abundance.rda"
        ),
        sha256="89c635061b357583351ca33f520a72d0efced4a2963e73182e529228c8395c54",
        real_synthetic_status="real_public",
        allowed_analytical_role="predictor_reconstruction",
        exclusion_reason=(
            "Real microbiome predictors, but no observed meal-level response endpoint."
        ),
        evidence_basis=(
            "ExperimentHub metadata snapshot dated 2026-07-30 identifies EH5458 and dispatch path 5501."
        ),
    ),
    PredictZoeSource(
        resource_id="predict2_ena_raw_metagenomes",
        source_name="PREDICT 2 raw metagenomes",
        stable_source="https://www.ebi.ac.uk/ena/browser/view/PRJEB75460",
        source_identifier="PRJEB75460",
        version_identifier="ERP160037",
        accessed_on=_ACCESSED,
        owner="ZOE Limited",
        cohort="PREDICT 2",
        unit_of_observation="ENA metagenomic sample/run",
        sample_size="975",
        data_modality="shotgun metagenomic sequencing",
        declared_response_endpoint="none in the ENA sequencing resource",
        access_route="reused_public_source",
        local_path=_NOT_DOWNLOADED,
        sha256=_NO_REMOTE_SHA,
        real_synthetic_status="real_public",
        allowed_analytical_role="predictor_reconstruction",
        exclusion_reason="Public metagenomes do not include person-by-meal response outcomes.",
        evidence_basis="Nature Data Availability statement and ENA study/count metadata.",
    ),
    PredictZoeSource(
        resource_id="predict3_us21_ena_raw_metagenomes",
        source_name="PREDICT 3 US21 raw metagenomes",
        stable_source="https://www.ebi.ac.uk/ena/browser/view/PRJEB75462",
        source_identifier="PRJEB75462",
        version_identifier="ERP160039",
        accessed_on=_ACCESSED,
        owner="ZOE Limited",
        cohort="PREDICT 3 US21",
        unit_of_observation="ENA metagenomic sample/run",
        sample_size="11797",
        data_modality="shotgun metagenomic sequencing",
        declared_response_endpoint="none in the ENA sequencing resource",
        access_route="reused_public_source",
        local_path=_NOT_DOWNLOADED,
        sha256=_NO_REMOTE_SHA,
        real_synthetic_status="real_public",
        allowed_analytical_role="predictor_reconstruction",
        exclusion_reason="Public metagenomes do not include person-by-meal response outcomes.",
        evidence_basis="Nature Data Availability statement and ENA study/count metadata.",
    ),
    PredictZoeSource(
        resource_id="predict3_us22a_ena_raw_metagenomes",
        source_name="PREDICT 3 US22A raw metagenomes",
        stable_source="https://www.ebi.ac.uk/ena/browser/view/PRJEB75463",
        source_identifier="PRJEB75463",
        version_identifier="ERP160040",
        accessed_on=_ACCESSED,
        owner="ZOE Limited",
        cohort="PREDICT 3 US22A",
        unit_of_observation="ENA metagenomic sample/run",
        sample_size="8469",
        data_modality="shotgun metagenomic sequencing",
        declared_response_endpoint="none in the ENA sequencing resource",
        access_route="reused_public_source",
        local_path=_NOT_DOWNLOADED,
        sha256=_NO_REMOTE_SHA,
        real_synthetic_status="real_public",
        allowed_analytical_role="predictor_reconstruction",
        exclusion_reason="Public metagenomes do not include person-by-meal response outcomes.",
        evidence_basis="Nature Data Availability statement and ENA study/count metadata.",
    ),
    PredictZoeSource(
        resource_id="predict3_uk22a_ena_raw_metagenomes",
        source_name="PREDICT 3 UK22A raw metagenomes",
        stable_source="https://www.ebi.ac.uk/ena/browser/view/PRJEB75464",
        source_identifier="PRJEB75464",
        version_identifier="ERP160041",
        accessed_on=_ACCESSED,
        owner="ZOE Limited",
        cohort="PREDICT 3 UK22A",
        unit_of_observation="ENA metagenomic sample/run",
        sample_size="12353",
        data_modality="shotgun metagenomic sequencing",
        declared_response_endpoint="none in the ENA sequencing resource",
        access_route="reused_public_source",
        local_path=_NOT_DOWNLOADED,
        sha256=_NO_REMOTE_SHA,
        real_synthetic_status="real_public",
        allowed_analytical_role="predictor_reconstruction",
        exclusion_reason="Public metagenomes do not include person-by-meal response outcomes.",
        evidence_basis="Nature Data Availability statement and ENA study/count metadata.",
    ),
    PredictZoeSource(
        resource_id="predict_zoe_public_profiles_zenodo",
        source_name="PREDICT metadata and MetaPhlAn profiles",
        stable_source="https://doi.org/10.5281/zenodo.15307999",
        source_identifier="10.5281/zenodo.15307999",
        version_identifier="10.5281/zenodo.15308000",
        accessed_on=_ACCESSED,
        owner="Asnicar et al.",
        cohort="PREDICT 1, PREDICT 2, and PREDICT 3 cohorts",
        unit_of_observation="participant microbiome profile and public demographic metadata",
        sample_size="unknown",
        data_modality="compressed metadata and MetaPhlAn taxonomic profiles",
        declared_response_endpoint="none in the public Zenodo record",
        access_route="reused_public_source",
        local_path=_NOT_DOWNLOADED,
        sha256=_NO_REMOTE_SHA,
        real_synthetic_status="real_public",
        allowed_analytical_role="predictor_reconstruction",
        exclusion_reason=(
            "Public files expose microbiome and limited demographics, not eligible meal-response outcomes."
        ),
        evidence_basis=(
            "Zenodo API resolves the concept DOI to open version 10.5281/zenodo.15308000."
        ),
    ),
    PredictZoeSource(
        resource_id="zoe_microbiome_rankings",
        source_name="ZOE Microbiome Health and Diet Rankings",
        stable_source="https://zoe.com/our-science/microbiome-ranking",
        source_identifier="ZOE-MB-rankings",
        version_identifier="Nature-2025-supplementary-table-S5",
        accessed_on=_ACCESSED,
        owner="ZOE Ltd. and study authors",
        cohort="cross-cohort PREDICT ranking",
        unit_of_observation="species-level genome bin ranking",
        sample_size="unknown",
        data_modality="microbiome species rank",
        declared_response_endpoint="health-associated and diet-associated species ranks",
        access_route="reused_public_source",
        local_path="data/project_data/verification/zoe2025/zoe_health_rank_S5.csv",
        sha256="51bfe4bb68f9071c7ed1dd07ab8729878ffa0d8907526402db1ff19c35b010e3",
        real_synthetic_status="aggregate_public",
        allowed_analytical_role="biological_consistency",
        exclusion_reason=(
            "Aggregate species ranks cannot validate person-food metabolic responses."
        ),
        evidence_basis=f"{_NATURE_DATA}; official supplementary Table S5.",
    ),
    PredictZoeSource(
        resource_id="predict_controlled_clinical_zenodo",
        source_name="Encrypted ZOE PREDICT clinical-data archive",
        stable_source="https://doi.org/10.5281/zenodo.17236382",
        source_identifier="10.5281/zenodo.17236382",
        version_identifier="10.5281/zenodo.17236383",
        accessed_on=_ACCESSED,
        owner="ZOE Ltd.",
        cohort="ZOE PREDICT studies",
        unit_of_observation="controlled participant clinical record",
        sample_size="unknown",
        data_modality="encrypted participant clinical and host-parameter archive",
        declared_response_endpoint=(
            "ordered host parameters; exact glucose, triglyceride, C-peptide, meal, unit, and linkage fields unconfirmed"
        ),
        access_route="controlled_access_repository",
        local_path=_NOT_DOWNLOADED,
        sha256=_NO_REMOTE_SHA,
        real_synthetic_status="controlled_real",
        allowed_analytical_role="controlled_eligibility_assessment",
        exclusion_reason=(
            "Access has not been granted and endpoint, meal-key, unit, and microbiome-linkage eligibility is unverified."
        ),
        evidence_basis=(
            "Nature requires proposal review and a ZOE data-sharing agreement; Zenodo version 17236383 contains an encrypted archive."
        ),
    ),
    PredictZoeSource(
        resource_id="predict_external_archive_zenodo",
        source_name="External-cohort and code archive cited by the Nature article",
        stable_source="https://doi.org/10.5281/zenodo.17236261",
        source_identifier="10.5281/zenodo.17236261",
        version_identifier="10.5281/zenodo.17236262",
        accessed_on=_ACCESSED,
        owner="Asnicar, Francesco",
        cohort="non-PREDICT public cohorts and associated software",
        unit_of_observation="software archive in the currently visible Zenodo version",
        sample_size="unknown",
        data_modality="inverse_var_weight and MetaPhlAn source archives",
        declared_response_endpoint="not applicable to the visible software files",
        access_route="public_repository",
        local_path=_NOT_DOWNLOADED,
        sha256=_NO_REMOTE_SHA,
        real_synthetic_status="software_public",
        allowed_analytical_role="reproducibility_support",
        exclusion_reason=(
            "Current Zenodo file metadata exposes software archives only, not person-by-meal outcomes."
        ),
        evidence_basis=(
            "Zenodo API version 17236262; this conflicts with the broader external-data wording in Nature Data Availability."
        ),
    ),
    PredictZoeSource(
        resource_id="inverse_var_weight_v1_0_0",
        source_name="SegataLab inverse_var_weight release v1.0.0",
        stable_source=(
            "https://github.com/SegataLab/inverse_var_weight/releases/tag/v1.0.0"
        ),
        source_identifier="SegataLab/inverse_var_weight:v1.0.0",
        version_identifier="ab1a974bf1abd66175d190b55183d2957224b39f",
        accessed_on=_ACCESSED,
        owner="SegataLab",
        cohort="not applicable",
        unit_of_observation="software release",
        sample_size="unknown",
        data_modality="Python source code",
        declared_response_endpoint="not applicable",
        access_route="public_repository",
        local_path=_NOT_DOWNLOADED,
        sha256=_NO_REMOTE_SHA,
        real_synthetic_status="software_public",
        allowed_analytical_role="reproducibility_support",
        exclusion_reason="Software is reproducibility material, not validation data.",
        evidence_basis=(
            "GitHub release page and git ls-remote both resolve v1.0.0 to the fixed commit."
        ),
    ),
    PredictZoeSource(
        resource_id="local_predict1_real_subject_metadata",
        source_name="Local PREDICT 1 real subject metadata derivative",
        stable_source=(
            "data/project_data/predict_multi/L2_phenotype/predict1_real/PROVENANCE.md"
        ),
        source_identifier="local:subjects_real.parquet",
        version_identifier="sha256-bound-local-artifact",
        accessed_on=_ACCESSED,
        owner="GMNPS local project; upstream curatedMetagenomicData",
        cohort="PREDICT 1",
        unit_of_observation="participant/sample metadata record",
        sample_size="unknown",
        data_modality="demographic and sample-linkage metadata",
        declared_response_endpoint="none",
        access_route="reused_public_source",
        local_path=(
            "data/project_data/predict_multi/L2_phenotype/predict1_real/subjects_real.parquet"
        ),
        sha256="ecde68e98fba42cc0630e7a06bf050ca107c7cc580ff6795b415973512d843c3",
        real_synthetic_status="real_public",
        allowed_analytical_role="predictor_reconstruction",
        exclusion_reason="Real participant metadata contain no observed meal-response endpoint.",
        evidence_basis="Local PROVENANCE.md maps these fields to curatedMetagenomicData AsnicarF_2021.",
    ),
    PredictZoeSource(
        resource_id="local_predict1_synthetic_glucose",
        source_name="Local statistically anchored glucose response",
        stable_source=(
            "data/project_data/predict_multi/L2_phenotype/predict1_real/PROVENANCE.md"
        ),
        source_identifier="local:meals_glucose.parquet#glucose_iAUC_2h",
        version_identifier="sha256-bound-local-artifact",
        accessed_on=_ACCESSED,
        owner="GMNPS local project",
        cohort="PREDICT 1-like simulation",
        unit_of_observation="synthetic participant-meal record",
        sample_size="unknown",
        data_modality="statistically anchored synthetic response",
        declared_response_endpoint="glucose_iAUC_2h",
        access_route="not_applicable",
        local_path=(
            "data/project_data/predict_multi/L2_phenotype/predict1_real/meals_glucose.parquet"
        ),
        sha256="7cdbf828d6ba8013e3fe0c9159266763c552a40a0f7e03e3832436fa3e7a924e",
        real_synthetic_status="synthetic_local",
        allowed_analytical_role="synthetic_stress_test",
        exclusion_reason="The endpoint is generated, not an observed PREDICT outcome.",
        evidence_basis=(
            "Local PROVENANCE.md explicitly labels glucose_iAUC_2h statistically anchored synthetic."
        ),
    ),
    PredictZoeSource(
        resource_id="local_predict1_synthetic_triglyceride",
        source_name="Local statistically anchored triglyceride response",
        stable_source=(
            "data/project_data/predict_multi/L2_phenotype/predict1_real/PROVENANCE.md"
        ),
        source_identifier="local:meals_tg_cp.parquet#tg_6h_rise",
        version_identifier="sha256-bound-local-artifact",
        accessed_on=_ACCESSED,
        owner="GMNPS local project",
        cohort="PREDICT 1-like simulation",
        unit_of_observation="synthetic participant-meal record",
        sample_size="unknown",
        data_modality="statistically anchored synthetic response",
        declared_response_endpoint="tg_6h_rise",
        access_route="not_applicable",
        local_path=(
            "data/project_data/predict_multi/L2_phenotype/predict1_real/meals_tg_cp.parquet"
        ),
        sha256="a09216c78e40f964009099828ab20895b1bdccef2413eef4b7d22618cee485da",
        real_synthetic_status="synthetic_local",
        allowed_analytical_role="synthetic_stress_test",
        exclusion_reason="The endpoint is generated, not an observed PREDICT outcome.",
        evidence_basis=(
            "Local PROVENANCE.md explicitly labels tg_6h_rise statistically anchored synthetic."
        ),
    ),
    PredictZoeSource(
        resource_id="local_predict1_synthetic_c_peptide",
        source_name="Local statistically anchored C-peptide response",
        stable_source=(
            "data/project_data/predict_multi/L2_phenotype/predict1_real/PROVENANCE.md"
        ),
        source_identifier="local:meals_tg_cp.parquet#c_peptide_iAUC_2h",
        version_identifier="sha256-bound-local-artifact",
        accessed_on=_ACCESSED,
        owner="GMNPS local project",
        cohort="PREDICT 1-like simulation",
        unit_of_observation="synthetic participant-meal record",
        sample_size="unknown",
        data_modality="statistically anchored synthetic response",
        declared_response_endpoint="c_peptide_iAUC_2h",
        access_route="not_applicable",
        local_path=(
            "data/project_data/predict_multi/L2_phenotype/predict1_real/meals_tg_cp.parquet"
        ),
        sha256="a09216c78e40f964009099828ab20895b1bdccef2413eef4b7d22618cee485da",
        real_synthetic_status="synthetic_local",
        allowed_analytical_role="synthetic_stress_test",
        exclusion_reason="The endpoint is generated, not an observed PREDICT outcome.",
        evidence_basis=(
            "Local PROVENANCE.md explicitly labels c_peptide_iAUC_2h statistically anchored synthetic."
        ),
    ),
)


def validate_predict_zoe_registry(records: Iterable[PredictZoeSource]) -> None:
    """Fail closed on incomplete, duplicated, or analytically unsafe records."""

    materialized = tuple(records)
    if not materialized:
        raise ValueError("PREDICT/ZOE source registry must not be empty")
    identifiers = [record.resource_id for record in materialized]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("PREDICT/ZOE source registry contains duplicate resource IDs")
    for record in materialized:
        if not isinstance(record, PredictZoeSource):
            raise TypeError("registry entries must be PredictZoeSource records")
        payload = record.as_dict()
        missing = [field for field in PROVENANCE_FIELDS if not str(payload[field]).strip()]
        if missing:
            raise ValueError(f"registry record {record.resource_id!r} has missing provenance")
        if record.access_route not in ACCESS_ROUTES:
            raise ValueError(f"registry record {record.resource_id!r} has invalid access route")
        if record.allowed_analytical_role not in ANALYTICAL_ROLES:
            raise ValueError(
                f"registry record {record.resource_id!r} has invalid allowed analytical role"
            )
        if record.real_synthetic_status not in REAL_SYNTHETIC_STATUSES:
            raise ValueError(
                f"registry record {record.resource_id!r} has invalid real/synthetic status"
            )
        if record.sample_size != "unknown" and not record.sample_size.isdigit():
            raise ValueError(f"registry record {record.resource_id!r} has invalid sample size")
        if record.real_synthetic_status in {
            "synthetic_local",
            "aggregate_public",
            "controlled_real",
        } and record.allowed_analytical_role == "direct_validation":
            raise ValueError(
                f"registry record {record.resource_id!r} cannot be direct validation"
            )


validate_predict_zoe_registry(PREDICT_ZOE_SOURCE_REGISTRY)

_REGISTRY_BY_ID: Mapping[str, PredictZoeSource] = MappingProxyType(
    {record.resource_id: record for record in PREDICT_ZOE_SOURCE_REGISTRY}
)


def get_predict_zoe_source(resource_id: str) -> PredictZoeSource:
    """Return one source record by stable internal ID."""

    if not isinstance(resource_id, str) or not resource_id:
        raise ValueError("resource_id must be a nonempty string")
    try:
        return _REGISTRY_BY_ID[resource_id]
    except KeyError as error:
        raise KeyError(f"unknown PREDICT/ZOE source: {resource_id}") from error


__all__ = [
    "ACCESS_ROUTES",
    "ANALYTICAL_ROLES",
    "PREDICT_ZOE_SOURCE_REGISTRY",
    "PROVENANCE_FIELDS",
    "PredictZoeSource",
    "REAL_SYNTHETIC_STATUSES",
    "get_predict_zoe_source",
    "validate_predict_zoe_registry",
]
