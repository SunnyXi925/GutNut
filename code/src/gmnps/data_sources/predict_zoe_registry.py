"""Pre-outcome provenance registry for PREDICT and ZOE resources.

The registry classifies source eligibility without loading participant-level
outcome or label values.  It is intentionally conservative: no Task 1 record
is authorized for direct validation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import re
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

ACCESS_STATUSES = frozenset(
    {
        "public",
        "controlled_granted",
        "controlled_not_granted",
        "local_copy_available",
        "local_pending_submission_package",
    }
)

ANALYTICAL_ROLES = frozenset(
    {
        "predictor_reconstruction",
        "biological_consistency",
        "aggregate_supporting_evidence",
        "synthetic_stress_test",
        "controlled_eligibility_assessment",
        "direct_validation",
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
    "n_value",
    "n_unit",
    "participant_n",
    "data_modality",
    "declared_response_endpoint",
    "access_route",
    "access_status",
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
    n_value: str
    n_unit: str
    participant_n: str
    data_modality: str
    declared_response_endpoint: str
    access_route: str
    access_status: str
    local_path: str
    sha256: str
    real_synthetic_status: str
    allowed_analytical_role: str
    exclusion_reason: str
    evidence_basis: str

    def as_dict(self) -> dict[str, str]:
        """Return a CSV-compatible copy of this immutable record."""

        return asdict(self)


@dataclass(frozen=True)
class PredictorArtifact:
    """Repository-trusted contract for one exact predictor artifact."""

    source_id: str
    resource_id: str
    stable_source: str
    expected_cache_path: str
    path_policy: str
    sha256: str
    digest_basis: str
    size_bytes: int
    content_class: str
    allowed_analytical_role: str
    file_format: str
    schema_kind: str
    required_columns: tuple[str, ...]
    allowed_columns: tuple[str, ...]
    unique_key: tuple[str, ...]
    expected_records: int
    id_column: str
    id_sha256: str
    feature_id_sha256: str | None


@dataclass(frozen=True)
class OutcomeEndpointGrant:
    """Data-dictionary-attested mapping for one preregistered endpoint."""

    name: str
    source_column: str | None
    availability: str
    role: str
    unit: str
    window_hours: tuple[int, int]
    summary: str
    derivation: str


@dataclass(frozen=True)
class ControlledOutcomeGrant:
    """Evidence required to promote one controlled source to direct validation."""

    source_id: str
    access_status: str
    allowed_analytical_role: str
    verified_local_path: str | None
    sha256: str | None
    file_format: str | None
    version_doi: str
    approval_evidence_identifier: str | None
    data_dictionary_path: str | None
    data_dictionary_sha256: str | None
    participant_meal_key_contract: tuple[str, ...] | None
    endpoint_contracts: tuple[OutcomeEndpointGrant, ...]
    microbiome_linkage_evidence_identifier: str | None
    microbiome_linkage_key_contract: tuple[str, ...] | None


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
        n_value="unknown",
        n_unit="published aggregate table row",
        participant_n="unknown",
        data_modality="aggregate supplementary tables and species rankings",
        declared_response_endpoint=(
            "aggregate clinical-marker prediction, association, and intervention summaries"
        ),
        access_route="within_paper_or_supplement",
        access_status="public",
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
        n_value="2196",
        n_unit="ENA sample/run record",
        participant_n="unknown",
        data_modality="shotgun metagenomic sequencing",
        declared_response_endpoint="none in the ENA sequencing resource",
        access_route="reused_public_source",
        access_status="public",
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
        n_value="unknown",
        n_unit="participant microbiome profile",
        participant_n="unknown",
        data_modality="curated species relative-abundance profiles",
        declared_response_endpoint="none in the ExperimentHub profile resource",
        access_route="reused_public_source",
        access_status="public",
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
            "Official ExperimentHub metadata and fetch endpoint verified EH5458 and dispatch path 5501 on the audit date."
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
        n_value="975",
        n_unit="ENA sample/run record",
        participant_n="unknown",
        data_modality="shotgun metagenomic sequencing",
        declared_response_endpoint="none in the ENA sequencing resource",
        access_route="reused_public_source",
        access_status="public",
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
        n_value="11797",
        n_unit="ENA sample/run record",
        participant_n="unknown",
        data_modality="shotgun metagenomic sequencing",
        declared_response_endpoint="none in the ENA sequencing resource",
        access_route="reused_public_source",
        access_status="public",
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
        n_value="8469",
        n_unit="ENA sample/run record",
        participant_n="unknown",
        data_modality="shotgun metagenomic sequencing",
        declared_response_endpoint="none in the ENA sequencing resource",
        access_route="reused_public_source",
        access_status="public",
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
        n_value="12353",
        n_unit="ENA sample/run record",
        participant_n="unknown",
        data_modality="shotgun metagenomic sequencing",
        declared_response_endpoint="none in the ENA sequencing resource",
        access_route="reused_public_source",
        access_status="public",
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
        n_value="unknown",
        n_unit="participant microbiome profile",
        participant_n="unknown",
        data_modality="compressed metadata and MetaPhlAn taxonomic profiles",
        declared_response_endpoint="none in the public Zenodo record",
        access_route="reused_public_source",
        access_status="public",
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
        resource_id="nature_supplementary_table_s5_rankings",
        source_name="Nature Supplementary Table S5 microbiome rankings",
        stable_source=(
            "https://media.springernature.com/original/springer-static/esm/"
            "art%3A10.1038%2Fs41586-025-09854-7/MediaObjects/"
            "41586_2025_9854_MOESM3_ESM.xlsx"
        ),
        source_identifier="10.1038/s41586-025-09854-7:MOESM3:S5",
        version_identifier="41586_2025_9854_MOESM3_ESM.xlsx:Table-S5",
        accessed_on=_ACCESSED,
        owner="Asnicar et al.; hosted by Springer Nature",
        cohort="cross-cohort PREDICT ranking",
        unit_of_observation="published species-level genome bin rank",
        sample_size="unknown",
        n_value="unknown",
        n_unit="published species-level genome bin rank",
        participant_n="unknown",
        data_modality="microbiome species rank",
        declared_response_endpoint="health-associated and diet-associated species ranks",
        access_route="within_paper_or_supplement",
        access_status="public",
        local_path="data/project_data/verification/zoe2025/zoe_health_rank_S5.csv",
        sha256="51bfe4bb68f9071c7ed1dd07ab8729878ffa0d8907526402db1ff19c35b010e3",
        real_synthetic_status="aggregate_public",
        allowed_analytical_role="biological_consistency",
        exclusion_reason=(
            "Aggregate species ranks cannot validate person-food metabolic responses."
        ),
        evidence_basis=f"{_NATURE_DATA}; immutable official Supplementary Table S5.",
    ),
    PredictZoeSource(
        resource_id="zoe_live_microbiome_rankings_2024",
        source_name="Live ZOE Microbiome Ranking 2024 page",
        stable_source="https://zoe.com/our-science/microbiome-ranking",
        source_identifier="ZOE-MB-rankings-live-2024",
        version_identifier="live-page-accessed-2026-08-13",
        accessed_on=_ACCESSED,
        owner="ZOE Ltd.",
        cohort="cross-cohort ranking displayed by ZOE",
        unit_of_observation="displayed species-level genome bin rank",
        sample_size="unknown",
        n_value="unknown",
        n_unit="displayed species-level genome bin rank",
        participant_n="unknown",
        data_modality="mutable microbiome species ranking page",
        declared_response_endpoint="health-associated and diet-associated species ranks",
        access_route="reused_public_source",
        access_status="public",
        local_path=_NOT_DOWNLOADED,
        sha256=_NO_REMOTE_SHA,
        real_synthetic_status="aggregate_public",
        allowed_analytical_role="biological_consistency",
        exclusion_reason=(
            "The mutable aggregate page cannot validate person-food metabolic responses."
        ),
        evidence_basis=(
            "Live ZOE page accessed on the audit date; it is not the immutable Nature Table S5 artifact."
        ),
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
        n_value="unknown",
        n_unit="controlled participant clinical record",
        participant_n="unknown",
        data_modality="encrypted participant clinical and host-parameter archive",
        declared_response_endpoint=(
            "ordered host parameters; exact glucose, triglyceride, C-peptide, meal, unit, and linkage fields unconfirmed"
        ),
        access_route="controlled_access_repository",
        access_status="controlled_not_granted",
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
        n_value="not_applicable",
        n_unit="not_applicable",
        participant_n="not_applicable",
        data_modality="inverse_var_weight and MetaPhlAn source archives",
        declared_response_endpoint="not applicable to the visible software files",
        access_route="public_repository",
        access_status="public",
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
        n_value="not_applicable",
        n_unit="not_applicable",
        participant_n="not_applicable",
        data_modality="Python source code",
        declared_response_endpoint="not applicable",
        access_route="public_repository",
        access_status="public",
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
        n_value="unknown",
        n_unit="participant/sample metadata record",
        participant_n="unknown",
        data_modality="demographic and sample-linkage metadata",
        declared_response_endpoint="none",
        access_route="reused_public_source",
        access_status="local_copy_available",
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
        stable_source="pending:GMNPS-Supplementary-or-Source-Data-package",
        source_identifier="identifier_pending_submission_package:glucose_iAUC_2h",
        version_identifier="sha256-bound-local-artifact-pending-deposition",
        accessed_on=_ACCESSED,
        owner="GMNPS local project",
        cohort="PREDICT 1-like simulation",
        unit_of_observation="synthetic participant-meal record",
        sample_size="unknown",
        n_value="unknown",
        n_unit="synthetic participant-meal record",
        participant_n="unknown",
        data_modality="statistically anchored synthetic response",
        declared_response_endpoint="glucose_iAUC_2h",
        access_route="within_paper_or_supplement",
        access_status="local_pending_submission_package",
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
        stable_source="pending:GMNPS-Supplementary-or-Source-Data-package",
        source_identifier="identifier_pending_submission_package:tg_6h_rise",
        version_identifier="sha256-bound-local-artifact-pending-deposition",
        accessed_on=_ACCESSED,
        owner="GMNPS local project",
        cohort="PREDICT 1-like simulation",
        unit_of_observation="synthetic participant-meal record",
        sample_size="unknown",
        n_value="unknown",
        n_unit="synthetic participant-meal record",
        participant_n="unknown",
        data_modality="statistically anchored synthetic response",
        declared_response_endpoint="tg_6h_rise",
        access_route="within_paper_or_supplement",
        access_status="local_pending_submission_package",
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
        stable_source="pending:GMNPS-Supplementary-or-Source-Data-package",
        source_identifier="identifier_pending_submission_package:c_peptide_iAUC_2h",
        version_identifier="sha256-bound-local-artifact-pending-deposition",
        accessed_on=_ACCESSED,
        owner="GMNPS local project",
        cohort="PREDICT 1-like simulation",
        unit_of_observation="synthetic participant-meal record",
        sample_size="unknown",
        n_value="unknown",
        n_unit="synthetic participant-meal record",
        participant_n="unknown",
        data_modality="statistically anchored synthetic response",
        declared_response_endpoint="c_peptide_iAUC_2h",
        access_route="within_paper_or_supplement",
        access_status="local_pending_submission_package",
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


_SHA256 = re.compile(r"[0-9a-f]{64}")
_PREDICTOR_CACHE_ROOT = (
    "data/project_data/predict_multi/L2_phenotype/"
    "predictor_reconstruction/cache/predictor"
)
_ENA_COLUMNS = (
    "run_accession",
    "study_accession",
    "sample_accession",
    "instrument_platform",
    "library_strategy",
    "library_source",
    "library_layout",
)
_ENA_QUERY_FIELDS = (
    "study_accession",
    "sample_accession",
    "run_accession",
    "instrument_platform",
    "library_strategy",
    "library_source",
    "library_layout",
)


def _ena_api_source(accession: str) -> str:
    fields = ",".join(_ENA_QUERY_FIELDS)
    return (
        "https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&"
        f"query=study_accession%3D%22{accession}%22&fields={fields}&"
        "format=tsv&limit=0"
    )


def _ena_predictor_artifact(
    *,
    source_id: str,
    resource_id: str,
    accession: str,
    digest: str,
    size_bytes: int,
    records: int,
    id_digest: str,
) -> PredictorArtifact:
    return PredictorArtifact(
        source_id=source_id,
        resource_id=resource_id,
        stable_source=_ena_api_source(accession),
        expected_cache_path=(
            f"{_PREDICTOR_CACHE_ROOT}/{accession}_read_run_metadata.tsv"
        ),
        path_policy="repository_relative_exact_no_symlink",
        sha256=digest,
        digest_basis="locally_audited_sha256_pinned_in_repository",
        size_bytes=size_bytes,
        content_class="predictor_only",
        allowed_analytical_role="predictor_reconstruction",
        file_format="tsv",
        schema_kind="delimited_table_exact_header",
        required_columns=_ENA_COLUMNS,
        allowed_columns=_ENA_COLUMNS,
        unique_key=("run_accession",),
        expected_records=records,
        id_column="run_accession",
        id_sha256=id_digest,
        feature_id_sha256=None,
    )


PREDICTOR_ARTIFACT_REGISTRY = (
    PredictorArtifact(
        source_id="experimenthub_eh5458_relative_abundance",
        resource_id="predict1_experimenthub_profiles",
        stable_source="https://experimenthub.bioconductor.org/fetch/5501",
        expected_cache_path=(
            f"{_PREDICTOR_CACHE_ROOT}/"
            "2021-03-31.AsnicarF_2021.relative_abundance.rda"
        ),
        path_policy="repository_relative_exact_no_symlink",
        sha256="89c635061b357583351ca33f520a72d0efced4a2963e73182e529228c8395c54",
        digest_basis="locally_audited_sha256_pinned_in_repository",
        size_bytes=755545,
        content_class="predictor_only",
        allowed_analytical_role="predictor_reconstruction",
        file_format="rdata",
        schema_kind="relative_abundance_matrix_feature_by_sample",
        required_columns=(
            "microbiome_feature_id_axis",
            "sample_id_axis",
            "relative_abundance_value",
        ),
        allowed_columns=(
            "microbiome_feature_id_axis",
            "sample_id_axis",
            "relative_abundance_value",
        ),
        unique_key=("microbiome_feature_id_axis", "sample_id_axis"),
        expected_records=1098,
        id_column="sample_id_axis",
        id_sha256="db9a7fabfa390e5d1b91b32c2b9569aa3d1d5e57f9708297bfa91c1e1d92974c",
        feature_id_sha256=(
            "631e4df06aa1c00eb2c7ad9160cead46f1ad4f5a0b430d543c59d8c51741ebe8"
        ),
    ),
    _ena_predictor_artifact(
        source_id="ena_prjeb39223_sequencing_metadata",
        resource_id="predict1_ena_raw_metagenomes",
        accession="PRJEB39223",
        digest="d630be36c1abd3a71b6aa295dc56e1557557f965f6c8fdaa673759b227b28e36",
        size_bytes=152736,
        records=2196,
        id_digest="bc366b1d637c43146b7da9864410278ee70d70749f0d836e2f2720eb64df48b0",
    ),
    _ena_predictor_artifact(
        source_id="ena_prjeb75460_sequencing_metadata",
        resource_id="predict2_ena_raw_metagenomes",
        accession="PRJEB75460",
        digest="2ce6d90cd5d380581737a9dc193460b3f5df7f836f443845bba793ecb67de1a9",
        size_bytes=68364,
        records=975,
        id_digest="3831b07c2a0f7d4a5231cec530c96ed3f362d1cc5429739565c0af6b1323b39a",
    ),
    _ena_predictor_artifact(
        source_id="ena_prjeb75462_sequencing_metadata",
        resource_id="predict3_us21_ena_raw_metagenomes",
        accession="PRJEB75462",
        digest="ca6d7d5a1b7fdcd069a309b5bc8dd076a53ea8bc4016ec98b1b755658fd20f08",
        size_bytes=825904,
        records=11797,
        id_digest="9c2d70abf8006b21a8f88d4716eaa2b52b34ace808a0b5e57e9c683e553d1b3c",
    ),
    _ena_predictor_artifact(
        source_id="ena_prjeb75463_sequencing_metadata",
        resource_id="predict3_us22a_ena_raw_metagenomes",
        accession="PRJEB75463",
        digest="ce028e6072e610ad7575ba7e494347892cd66e33a5eaec92dea7177bcb413f56",
        size_bytes=592944,
        records=8469,
        id_digest="f192ad9e1a97f00941783fa0a360e091c1322ab4d5ff04316727ba77dc2db519",
    ),
    _ena_predictor_artifact(
        source_id="ena_prjeb75464_sequencing_metadata",
        resource_id="predict3_uk22a_ena_raw_metagenomes",
        accession="PRJEB75464",
        digest="f81d06da5f9127864967f99bb13a2b7f6eedddc8c56282f88c5b04985cd4bb94",
        size_bytes=864824,
        records=12353,
        id_digest="8cd1d25dfe557f50e58b93eaab8871ddc9e3be99ad9105584f8948da9f611529",
    ),
)


CONTROLLED_OUTCOME_GRANTS = (
    ControlledOutcomeGrant(
        source_id="predict_controlled_clinical_zenodo",
        access_status="controlled_not_granted",
        allowed_analytical_role="controlled_eligibility_assessment",
        verified_local_path=None,
        sha256=None,
        file_format=None,
        version_doi="10.5281/zenodo.17236383",
        approval_evidence_identifier=None,
        data_dictionary_path=None,
        data_dictionary_sha256=None,
        participant_meal_key_contract=None,
        endpoint_contracts=(),
        microbiome_linkage_evidence_identifier=None,
        microbiome_linkage_key_contract=None,
    ),
)

PREDICT1_EH5458_OVERLAP_AUDIT: Mapping[str, object] = MappingProxyType(
    {
        "canonicalization": (
            "exact_case_unique_nonempty_unicode_sort_utf8_lf_no_trailing_lf"
        ),
        "development_artifact_sha256": (
            "6552d061524da99e544c55c3d7cfeafd7c2eef999222b755d4c336bc40cec1b1"
        ),
        "development_n": 15492,
        "development_ids_sha256": (
            "e93e35038391cf56b4781e138841a93bd7f84c5d3ceb532d02aaf2f7dade6082"
        ),
        "eh5458_artifact_sha256": (
            "89c635061b357583351ca33f520a72d0efced4a2963e73182e529228c8395c54"
        ),
        "eh5458_n": 1098,
        "eh5458_ids_sha256": (
            "db9a7fabfa390e5d1b91b32c2b9569aa3d1d5e57f9708297bfa91c1e1d92974c"
        ),
        "intersection_n": 1098,
        "intersection_ids_sha256": (
            "db9a7fabfa390e5d1b91b32c2b9569aa3d1d5e57f9708297bfa91c1e1d92974c"
        ),
    }
)


def canonical_predictor_ids_sha256(identifiers: Iterable[str]) -> str:
    """Hash exact-case unique IDs as sorted UTF-8 lines with no trailing newline."""

    if isinstance(identifiers, (str, bytes)):
        raise ValueError("predictor IDs must be an iterable of strings")
    values = tuple(identifiers)
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("predictor IDs must contain nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError("predictor IDs contain duplicate values")
    return sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def predictor_id_overlap_audit(
    development_ids: Iterable[str],
    candidate_ids: Iterable[str],
) -> dict[str, object]:
    """Return deterministic counts and ID digests for a held-out overlap audit."""

    development = tuple(development_ids)
    candidate = tuple(candidate_ids)
    development_digest = canonical_predictor_ids_sha256(development)
    candidate_digest = canonical_predictor_ids_sha256(candidate)
    intersection = tuple(sorted(set(development) & set(candidate)))
    intersection_digest = (
        canonical_predictor_ids_sha256(intersection) if intersection else None
    )
    return {
        "development_n": len(development),
        "development_ids_sha256": development_digest,
        "candidate_n": len(candidate),
        "candidate_ids_sha256": candidate_digest,
        "intersection_n": len(intersection),
        "intersection_ids_sha256": intersection_digest,
    }


def validate_predictor_artifact_registry(
    records: Iterable[PredictorArtifact],
) -> None:
    """Validate the committed predictor trust root without reading artifacts."""

    materialized = tuple(records)
    if not materialized:
        raise ValueError("predictor artifact registry must not be empty")
    source_ids = [record.source_id for record in materialized]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("predictor artifact registry contains duplicate source IDs")
    provenance_ids = {record.resource_id for record in PREDICT_ZOE_SOURCE_REGISTRY}
    for record in materialized:
        if not isinstance(record, PredictorArtifact):
            raise TypeError("predictor artifact entries must be PredictorArtifact records")
        if record.resource_id not in provenance_ids:
            raise ValueError(f"predictor artifact {record.source_id!r} lacks provenance")
        if not record.stable_source.startswith("https://"):
            raise ValueError(f"predictor artifact {record.source_id!r} lacks stable HTTPS")
        cache_path = Path(record.expected_cache_path)
        if cache_path.is_absolute() or ".." in cache_path.parts:
            raise ValueError(f"predictor artifact {record.source_id!r} has unsafe cache path")
        if record.path_policy != "repository_relative_exact_no_symlink":
            raise ValueError(f"predictor artifact {record.source_id!r} has unsafe path policy")
        if _SHA256.fullmatch(record.sha256) is None:
            raise ValueError(f"predictor artifact {record.source_id!r} has invalid SHA-256")
        if record.digest_basis != "locally_audited_sha256_pinned_in_repository":
            raise ValueError(f"predictor artifact {record.source_id!r} has untrusted digest basis")
        if record.size_bytes <= 0 or record.expected_records <= 0:
            raise ValueError(f"predictor artifact {record.source_id!r} has invalid counts")
        if record.content_class != "predictor_only":
            raise ValueError(f"predictor artifact {record.source_id!r} is not predictor-only")
        if record.allowed_analytical_role != "predictor_reconstruction":
            raise ValueError(f"predictor artifact {record.source_id!r} has invalid role")
        if record.file_format not in {"csv", "tsv", "parquet", "rdata"}:
            raise ValueError(f"predictor artifact {record.source_id!r} has invalid format")
        if not record.schema_kind or not record.allowed_columns:
            raise ValueError(f"predictor artifact {record.source_id!r} lacks schema contract")
        if len(set(record.allowed_columns)) != len(record.allowed_columns):
            raise ValueError(f"predictor artifact {record.source_id!r} repeats columns")
        if not set(record.required_columns).issubset(record.allowed_columns):
            raise ValueError(f"predictor artifact {record.source_id!r} has invalid required columns")
        if not set(record.unique_key).issubset(record.required_columns):
            raise ValueError(f"predictor artifact {record.source_id!r} has invalid unique key")
        if record.id_column not in record.allowed_columns:
            raise ValueError(f"predictor artifact {record.source_id!r} lacks ID column")
        if _SHA256.fullmatch(record.id_sha256) is None:
            raise ValueError(f"predictor artifact {record.source_id!r} has invalid ID digest")
        if (
            record.feature_id_sha256 is not None
            and _SHA256.fullmatch(record.feature_id_sha256) is None
        ):
            raise ValueError(f"predictor artifact {record.source_id!r} has invalid feature digest")


def _source_map(sources: Iterable[PredictZoeSource]) -> dict[str, PredictZoeSource]:
    materialized = tuple(sources)
    identifiers = [source.resource_id for source in materialized]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("controlled grant sources contain duplicate resource IDs")
    return {source.resource_id: source for source in materialized}


def validate_controlled_outcome_grants(
    grants: Iterable[ControlledOutcomeGrant],
    *,
    sources: Iterable[PredictZoeSource] = PREDICT_ZOE_SOURCE_REGISTRY,
) -> None:
    """Require complete, file-bound evidence for every direct-validation grant."""

    materialized = tuple(grants)
    if not materialized:
        raise ValueError("controlled outcome grant registry must not be empty")
    source_by_id = _source_map(sources)
    identifiers = [grant.source_id for grant in materialized]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("controlled outcome grants contain duplicate source IDs")
    for grant in materialized:
        if not isinstance(grant, ControlledOutcomeGrant):
            raise TypeError("controlled grant entries must be ControlledOutcomeGrant records")
        source = source_by_id.get(grant.source_id)
        if source is None or source.real_synthetic_status != "controlled_real":
            raise ValueError(f"controlled grant {grant.source_id!r} lacks controlled provenance")
        if grant.version_doi != source.version_identifier:
            raise ValueError(f"controlled grant {grant.source_id!r} has wrong version DOI")
        if grant.access_status == "controlled_not_granted":
            evidence = (
                grant.verified_local_path,
                grant.sha256,
                grant.file_format,
                grant.approval_evidence_identifier,
                grant.data_dictionary_path,
                grant.data_dictionary_sha256,
                grant.participant_meal_key_contract,
                grant.microbiome_linkage_evidence_identifier,
                grant.microbiome_linkage_key_contract,
            )
            if grant.allowed_analytical_role != "controlled_eligibility_assessment":
                raise ValueError("ungranted controlled source cannot have direct-validation role")
            if any(value is not None for value in evidence) or grant.endpoint_contracts:
                raise ValueError("ungranted controlled source must not claim grant evidence")
            if source.access_status != "controlled_not_granted":
                raise ValueError("controlled source/grant access states disagree")
            continue
        if grant.access_status != "controlled_granted":
            raise ValueError(f"controlled grant {grant.source_id!r} has invalid access state")
        if grant.allowed_analytical_role != "direct_validation":
            raise ValueError("granted controlled source requires direct-validation role")
        if source.access_status != grant.access_status or source.allowed_analytical_role != grant.allowed_analytical_role:
            raise ValueError("controlled source/grant role or access state disagrees")
        required_strings = {
            "verified local path": grant.verified_local_path,
            "SHA-256": grant.sha256,
            "file format": grant.file_format,
            "approval evidence": grant.approval_evidence_identifier,
            "data dictionary path": grant.data_dictionary_path,
            "data dictionary SHA-256": grant.data_dictionary_sha256,
            "microbiome linkage evidence": grant.microbiome_linkage_evidence_identifier,
        }
        missing = [name for name, value in required_strings.items() if not isinstance(value, str) or not value.strip()]
        if missing:
            raise ValueError(f"controlled grant is missing evidence: {missing}")
        assert grant.verified_local_path is not None
        assert grant.sha256 is not None
        assert grant.data_dictionary_path is not None
        assert grant.data_dictionary_sha256 is not None
        if _SHA256.fullmatch(grant.sha256) is None or _SHA256.fullmatch(grant.data_dictionary_sha256) is None:
            raise ValueError("controlled grant evidence requires lowercase SHA-256 digests")
        if source.local_path != grant.verified_local_path or source.sha256 != grant.sha256:
            raise ValueError("controlled grant path/SHA does not match source registry")
        outcome_path = Path(grant.verified_local_path)
        if outcome_path.is_symlink() or not outcome_path.is_file():
            raise ValueError("controlled grant verified local path is unavailable or a symlink")
        dictionary_path = Path(grant.data_dictionary_path)
        if dictionary_path.is_symlink() or not dictionary_path.is_file():
            raise ValueError("controlled grant data-dictionary evidence is unavailable")
        if sha256(dictionary_path.read_bytes()).hexdigest() != grant.data_dictionary_sha256:
            raise ValueError("controlled grant data-dictionary hash mismatch")
        if grant.participant_meal_key_contract != ("participant_id", "meal_id"):
            raise ValueError("controlled grant participant/meal key contract is invalid")
        if not grant.endpoint_contracts:
            raise ValueError("controlled grant endpoint contract is missing")
        endpoint_names = [endpoint.name for endpoint in grant.endpoint_contracts]
        if len(set(endpoint_names)) != len(endpoint_names):
            raise ValueError("controlled grant endpoint contract contains duplicates")
        source_columns = []
        for endpoint in grant.endpoint_contracts:
            if not isinstance(endpoint, OutcomeEndpointGrant):
                raise TypeError("endpoint grant entries must be OutcomeEndpointGrant records")
            if endpoint.availability not in {"available", "not_available"}:
                raise ValueError("controlled grant endpoint availability is invalid")
            if endpoint.role not in {"primary", "secondary"}:
                raise ValueError("controlled grant endpoint role is invalid")
            if (
                not endpoint.name
                or not endpoint.unit
                or not endpoint.summary
                or not endpoint.derivation
                or len(endpoint.window_hours) != 2
            ):
                raise ValueError("controlled grant endpoint contract is incomplete")
            if endpoint.availability == "available":
                if not isinstance(endpoint.source_column, str) or not endpoint.source_column:
                    raise ValueError("available endpoint grant lacks a source column")
                source_columns.append(endpoint.source_column)
            elif endpoint.source_column is not None:
                raise ValueError("unavailable endpoint grant must not declare a source column")
        if len(set(source_columns)) != len(source_columns):
            raise ValueError("controlled grant endpoint source columns are not unique")
        if (
            not isinstance(grant.microbiome_linkage_key_contract, tuple)
            or not grant.microbiome_linkage_key_contract
            or any(not isinstance(value, str) or not value for value in grant.microbiome_linkage_key_contract)
        ):
            raise ValueError("controlled grant microbiome linkage key contract is missing")


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
        if record.access_status not in ACCESS_STATUSES:
            raise ValueError(f"registry record {record.resource_id!r} has invalid access status")
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
        for field_name in ("n_value", "participant_n"):
            value = getattr(record, field_name)
            if value not in {"unknown", "not_applicable"} and not value.isdigit():
                raise ValueError(
                    f"registry record {record.resource_id!r} has invalid {field_name}"
                )
        if record.source_identifier.startswith("PRJEB") and not (
            record.n_value == record.sample_size
            and record.n_unit == "ENA sample/run record"
            and record.participant_n == "unknown"
        ):
            raise ValueError(
                f"registry record {record.resource_id!r} misstates ENA count units"
            )
        if record.real_synthetic_status == "synthetic_local" and not (
            record.access_route == "within_paper_or_supplement"
            and record.access_status == "local_pending_submission_package"
            and "identifier_pending" in record.source_identifier
        ):
            raise ValueError(
                f"registry record {record.resource_id!r} has invalid synthetic access plan"
            )
        if record.real_synthetic_status in {
            "synthetic_local",
            "aggregate_public",
        } and record.allowed_analytical_role == "direct_validation":
            raise ValueError(
                f"registry record {record.resource_id!r} cannot be direct validation"
            )
        if record.real_synthetic_status == "controlled_real":
            is_granted_direct = (
                record.access_status == "controlled_granted"
                and record.allowed_analytical_role == "direct_validation"
            )
            is_ungranted_assessment = (
                record.access_status == "controlled_not_granted"
                and record.allowed_analytical_role
                == "controlled_eligibility_assessment"
            )
            if not (is_granted_direct or is_ungranted_assessment):
                raise ValueError(
                    f"registry record {record.resource_id!r} has invalid controlled state"
                )


validate_predict_zoe_registry(PREDICT_ZOE_SOURCE_REGISTRY)
validate_predictor_artifact_registry(PREDICTOR_ARTIFACT_REGISTRY)
validate_controlled_outcome_grants(
    CONTROLLED_OUTCOME_GRANTS,
    sources=PREDICT_ZOE_SOURCE_REGISTRY,
)

_REGISTRY_BY_ID: Mapping[str, PredictZoeSource] = MappingProxyType(
    {record.resource_id: record for record in PREDICT_ZOE_SOURCE_REGISTRY}
)
_PREDICTOR_ARTIFACT_BY_ID: Mapping[str, PredictorArtifact] = MappingProxyType(
    {record.source_id: record for record in PREDICTOR_ARTIFACT_REGISTRY}
)
_CONTROLLED_GRANT_BY_ID: Mapping[str, ControlledOutcomeGrant] = MappingProxyType(
    {record.source_id: record for record in CONTROLLED_OUTCOME_GRANTS}
)


def get_predict_zoe_source(resource_id: str) -> PredictZoeSource:
    """Return one source record by stable internal ID."""

    if not isinstance(resource_id, str) or not resource_id:
        raise ValueError("resource_id must be a nonempty string")
    try:
        return _REGISTRY_BY_ID[resource_id]
    except KeyError as error:
        raise KeyError(f"unknown PREDICT/ZOE source: {resource_id}") from error


def get_predictor_artifact(source_id: str) -> PredictorArtifact:
    """Return one committed predictor artifact contract by source ID."""

    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be a nonempty string")
    try:
        return _PREDICTOR_ARTIFACT_BY_ID[source_id]
    except KeyError as error:
        raise KeyError(f"unknown trusted predictor artifact: {source_id}") from error


def get_controlled_outcome_grant(source_id: str) -> ControlledOutcomeGrant:
    """Return the repository-controlled access grant for one source."""

    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be a nonempty string")
    try:
        return _CONTROLLED_GRANT_BY_ID[source_id]
    except KeyError as error:
        raise KeyError(f"unknown controlled outcome grant: {source_id}") from error


__all__ = [
    "ACCESS_ROUTES",
    "ACCESS_STATUSES",
    "ANALYTICAL_ROLES",
    "CONTROLLED_OUTCOME_GRANTS",
    "PREDICT1_EH5458_OVERLAP_AUDIT",
    "PREDICT_ZOE_SOURCE_REGISTRY",
    "PREDICTOR_ARTIFACT_REGISTRY",
    "PROVENANCE_FIELDS",
    "ControlledOutcomeGrant",
    "OutcomeEndpointGrant",
    "PredictorArtifact",
    "PredictZoeSource",
    "REAL_SYNTHETIC_STATUSES",
    "canonical_predictor_ids_sha256",
    "get_controlled_outcome_grant",
    "get_predictor_artifact",
    "get_predict_zoe_source",
    "predictor_id_overlap_audit",
    "validate_controlled_outcome_grants",
    "validate_predictor_artifact_registry",
    "validate_predict_zoe_registry",
]
