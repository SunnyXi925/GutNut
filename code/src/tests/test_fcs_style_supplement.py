from gmnps.supplement.fcs_style import (
    build_supplement_manifest,
    required_supplement_sections,
    required_supplement_tables,
)


def test_fcs_style_supplement_sections_are_complete():
    sections = required_supplement_sections()

    assert "Supplementary Methods 1: Data resources and provenance" in sections
    assert "Supplementary Methods 4: KEGG/GMMAD2 knowledge graph and adjudication" in sections
    assert "Reporting Summary" in sections


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
