"""FCS2.0-style supplementary-material registry.

The registry is intentionally independent of individual experiments: it records
every required supplementary table even when an upstream experiment has not
produced its source output yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SupplementSection:
    title: str
    purpose: str


_SECTIONS = (
    SupplementSection("Supplementary Methods 1: Data resources and provenance", "Data resources, provenance and reproducibility."),
    SupplementSection("Supplementary Methods 2: beta_i and microbiome health index construction", "Microbiome-health and nutrient-weight construction."),
    SupplementSection("Supplementary Methods 3: Personalized calibration and population consensus validation", "Calibration algorithm and consensus validation."),
    SupplementSection("Supplementary Methods 4: KEGG/GMMAD2 knowledge graph and adjudication", "Knowledge-graph evidence and label adjudication."),
    SupplementSection("Supplementary Methods 5: Dietary-response prediction and ablations", "Dietary-response modelling and ablations."),
    SupplementSection("Supplementary Figures", "Supplementary figures and their source data."),
    SupplementSection("Supplementary Tables", "Supplementary tables and source-output manifest."),
    SupplementSection("Data availability", "Data access, licensing and repository boundaries."),
    SupplementSection("Code availability", "Code, configuration and reproducibility information."),
    SupplementSection("Reporting Summary", "Study reporting checklist and analysis summary."),
    SupplementSection("Source-data index", "Index of source tables and their provenance records."),
)

_TABLE_S2_REQUIRED_COLUMNS = frozenset(
    {
        "resource",
        "source_path",
        "source_exists",
        "bytes",
        "sha256",
        "provenance",
        "version",
        "access_date",
        "licence",
    }
)

_TABLE_SOURCES = {
    "Table S1. Personalized calibration algorithm updates and rationales": "experiment_manifest.json",
    "Table S2. Data resources, versions, access dates, checksums and licences": "data_resource_audit.csv",
    "Table S3. USDA/FNDDS-to-GMNPS 65-nutrient mapping, units and transformations": "tables/usda_fndds_gmnps_nutrient_mapping.csv",
    "Table S4. beta_i nutrient weights, evidence channels and coefficient distributions": "tables/beta_i_nutrient_weights.csv",
    "Table S5. Food-level FCS2.0 and GMNPS scores for 9,234 foods": "section1_population_consensus/food_summary_by_amplification.csv",
    "Table S6. Major food-group and subgroup score distributions": "section1_population_consensus/food_summary_by_amplification.csv",
    "Table S7. Universal and microbiome-conditioned food-group classification": "section1_population_consensus/food_group_consensus.csv",
    "Table S8. Clinical metadata completeness and cohort-selection audit": "section2_clinical_consistency/cra013939_clinical_completeness.csv",
    "Table S9. Official/proxy microbiome health validation metrics": "section2_clinical_consistency/microbiome_health_retention.csv",
    "Table S10. Nutrient-level beta_i clinical and demographic associations": "tables/beta_i_clinical_associations.csv",
    "Table S11. Nutrient-microbe association heatmap source data": "tables/nutrient_microbe_associations.csv",
    "Table S12. KEGG/GMMAD2 KG schema, edge counts and evidence sources": "section3_kg_label_adjudication/kg_direct_nutrient_disease_edges.csv",
    "Table S13. KG path evidence and adjudication metrics": "section3_kg_label_adjudication/kg_path_scores.csv",
    "Table S14. Response-prediction outcome registry and split assignment": "section4_response_prediction/response_outcome_diagnostics.csv",
    "Table S15. Cross-validated model metrics": "section4_response_prediction/model_metrics.csv",
    "Table S16. Dietary-response model comparisons and ablations": "section4_response_prediction/model_comparisons.csv",
    "Table S17. Sensitivity analyses for calibration caps, temperatures and beta compression": "tables/calibration_sensitivity_analyses.csv",
}


def required_supplement_sections() -> list[str]:
    return [section.title for section in _SECTIONS]


def required_supplement_tables() -> list[str]:
    return list(_TABLE_SOURCES)


def build_supplement_manifest(output_dir: Path) -> pd.DataFrame:
    """Build a complete supplement manifest for structured experiment outputs."""
    output_dir = Path(output_dir)
    rows = [
        {
            "kind": "section",
            "name": section.title,
            "purpose": section.purpose,
            "source_path": "",
            "source_exists": True,
            "status": "required",
        }
        for section in _SECTIONS
    ]
    for name, relative_source in _TABLE_SOURCES.items():
        source = output_dir / relative_source
        source_exists = source.is_file()
        status = "available" if source_exists else "missing_source"
        if source_exists and name.startswith("Table S2."):
            try:
                columns = set(pd.read_csv(source, nrows=0).columns)
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                columns = set()
            if not _TABLE_S2_REQUIRED_COLUMNS.issubset(columns):
                status = "invalid_schema"
        rows.append(
            {
                "kind": "table",
                "name": name,
                "purpose": "",
                "source_path": relative_source,
                "source_exists": source_exists,
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=["kind", "name", "purpose", "source_path", "source_exists", "status"])
