import pandas as pd
import pytest

from gmnps.supplement.fcs_style import (
    build_supplement_manifest,
    required_supplement_sections,
    required_supplement_tables,
)
from scripts.run_personalized_calibration_experiments import write_supplement


def test_fcs_style_supplement_sections_are_complete():
    sections = required_supplement_sections()

    assert "Supplementary Methods 1: Data resources and provenance" in sections
    assert "Supplementary Methods 4: KEGG/GMMAD2 knowledge graph and adjudication" in sections
    assert "Reporting Summary" in sections
    assert "Source-data index" in sections


def test_fcs_style_supplement_tables_include_algorithm_and_full_score_tables():
    tables = required_supplement_tables()

    assert "Table S1. Personalized calibration algorithm updates and rationales" in tables
    assert "Table S5. Food-level FCS2.0 and GMNPS scores for 9,234 foods" in tables
    assert "Table S16. Dietary-response model comparisons and ablations" in tables


def test_manifest_marks_absent_table_source_without_omitting_required_table(tmp_path):
    manifest = build_supplement_manifest(tmp_path)

    table = manifest.loc[manifest["name"] == "Table S16. Dietary-response model comparisons and ablations"].iloc[0]
    assert table["status"] == "missing_source"
    assert table["source_path"] == "section4_response_prediction/model_comparisons.csv"


def test_table_s2_requires_provenance_schema_not_just_an_existing_file(tmp_path):
    pd.DataFrame(
        {
            "resource": ["food"],
            "source_path": ["data/food.csv"],
            "source_exists": [True],
            "bytes": [42],
            "sha256": ["abc123"],
            "provenance": ["runner_input_registry"],
        }
    ).to_csv(tmp_path / "data_resource_audit.csv", index=False)

    manifest = build_supplement_manifest(tmp_path)
    table = manifest.loc[manifest["name"].str.startswith("Table S2.")].iloc[0]

    assert bool(table["source_exists"])
    assert table["status"] == "invalid_schema"


@pytest.mark.parametrize("provenance_field", ["version", "access_date", "licence"])
@pytest.mark.parametrize("placeholder", ["not_recorded", "", "nan", "None", None])
def test_table_s2_rejects_placeholder_required_provenance_values(
    tmp_path, provenance_field, placeholder
):
    provenance = {
        "version": "FNDDS 2021-2022",
        "access_date": "2026-08-04",
        "licence": "CC-BY-4.0",
    }
    provenance[provenance_field] = placeholder
    pd.DataFrame(
        {
            "resource": ["food"],
            "source_path": ["data/food.csv"],
            "source_exists": [True],
            "bytes": [42],
            "sha256": ["abc123"],
            "provenance": ["runner_input_registry"],
            **{key: [value] for key, value in provenance.items()},
        }
    ).to_csv(tmp_path / "data_resource_audit.csv", index=False)

    manifest = build_supplement_manifest(tmp_path)
    table = manifest.loc[manifest["name"].str.startswith("Table S2.")].iloc[0]

    assert table["status"] == "invalid_schema"


def test_table_s2_accepts_curated_required_provenance_values(tmp_path):
    pd.DataFrame(
        {
            "resource": ["food"],
            "source_path": ["data/food.csv"],
            "source_exists": [True],
            "bytes": [42],
            "sha256": ["abc123"],
            "provenance": ["curated_resource_registry"],
            "version": ["FNDDS 2021-2022"],
            "access_date": ["2026-08-04"],
            "licence": ["CC-BY-4.0"],
        }
    ).to_csv(tmp_path / "data_resource_audit.csv", index=False)

    manifest = build_supplement_manifest(tmp_path)
    table = manifest.loc[manifest["name"].str.startswith("Table S2.")].iloc[0]

    assert table["status"] == "available"


def test_supplement_renderer_writes_required_outputs_and_available_s7_preview(tmp_path):
    section1 = tmp_path / "section1_population_consensus"
    section1.mkdir()
    pd.DataFrame(
        [{"amplification": 1.0, "food_group": "Vegetables", "n_foods": 42}]
    ).to_csv(section1 / "food_group_consensus.csv", index=False)

    write_supplement(tmp_path, {"sections": {}})

    supplement_dir = tmp_path / "supplement"
    assert (supplement_dir / "supplementary_material.tex").is_file()
    assert (supplement_dir / "supplement_manifest.csv").is_file()
    manifest = pd.read_csv(supplement_dir / "supplement_manifest.csv")
    s7 = manifest.loc[
        manifest["name"] == "Table S7. Universal and microbiome-conditioned food-group classification"
    ].iloc[0]
    assert s7["source_path"] == "section1_population_consensus/food_group_consensus.csv"
    assert bool(s7["source_exists"])
    assert s7["status"] == "available"
    latex = (supplement_dir / "supplementary_material.tex").read_text(encoding="utf-8")
    assert r"\n" not in latex
    assert "Vegetables" in latex


def test_supplement_renderer_includes_table_s2_provenance_headers_when_available(tmp_path):
    pd.DataFrame(
        {
            "resource": ["food"],
            "source_path": ["data/food.csv"],
            "source_exists": [True],
            "bytes": [42],
            "sha256": ["abc123"],
            "provenance": ["curated_resource_registry"],
            "version": ["FNDDS 2021-2022"],
            "access_date": ["2026-08-04"],
            "licence": ["CC-BY-4.0"],
        }
    ).to_csv(tmp_path / "data_resource_audit.csv", index=False)

    write_supplement(tmp_path, {"sections": {}})

    latex = (tmp_path / "supplement" / "supplementary_material.tex").read_text(encoding="utf-8")
    for header in ["version", "access\\_date", "licence", "sha256", "source\\_path"]:
        assert header in latex
