from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
from pathlib import Path
import re
import shutil
import struct

import pandas as pd
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
SUBMISSION = ROOT / "manuscript/nature_food_submission"
FIGURES = SUBMISSION / "figures"
SOURCE_DATA = SUBMISSION / "source_data"
SCRIPT = ROOT / "code/src/scripts/make_main_figures.py"
MATRIX = SUBMISSION / "claim_evidence_matrix.csv"
MANIFEST = FIGURES / "figure_manifest.csv"
ALLOWLIST = FIGURES / "submission_asset_allowlist.txt"
FROZEN = ROOT / "results/phase2/source-data/correctly_specified_synthetic_positive_control"

FROZEN_HASHES = {
    "synthetic_attribute_twin_config.json": "fd5a21d815406225f3fe5a6222cd39dafed7b98014dbc25190b61399fdb97c35",
    "synthetic_attribute_twin_manifest.json": "df0684efa66377bdbdaec77182cdcbadc03b35cd4c5ec0403febf3c7d1096263",
    "synthetic_attribute_twin_replicate_metrics.csv": "b0a0e379e8654ef1c370d614abe932a2a7cd0cedb4ad13ccf52ef6db0a7e70e6",
    "synthetic_attribute_twin_success_checks.csv": "091bea04c4baed04fa4f4898d8c980cac081e3bc7dfb58435b5799fdfac32719",
    "synthetic_attribute_twin_summary.csv": "b78ab05011a7c5f9bb150b22d50d72ff84797721c6a5c1d3bc8c37a3edbf9bdd",
}
COMPARATORS = [
    "fcs_baseline",
    "locked_attribute_gmnps",
    "random_microbiome",
    "sattolo_deranged_microbiome",
]
LABELS = [
    "FCS baseline",
    "Locked attribute GMNPS",
    "Random microbiome assignment",
    "Sattolo-deranged assignment",
]
SEEDS = [1701, 1702, 1703, 1704, 1705]
FIGURE_MANIFEST_COLUMNS = [
    "figure_id",
    "panel_id",
    "display_order",
    "title",
    "status",
    "display_class",
    "claim_ids",
    "evidence_tier",
    "evidence_role",
    "data_class",
    "analysis_unit",
    "inference_unit",
    "n_effective",
    "interval_definition",
    "source_artifacts",
    "source_sha256",
    "generator_script",
    "output_paths",
    "unlock_artifacts",
    "caption_boundary",
    "exclusion_reason",
]
SOURCE_MANIFEST_COLUMNS = [
    "file_path",
    "figure_id",
    "panel_id",
    "status",
    "claim_ids",
    "evidence_tier",
    "evidence_role",
    "data_class",
    "source_artifact_path",
    "source_artifact_sha256",
    "source_payload_sha256",
    "generated_file_sha256",
    "row_count",
    "column_schema_sha256",
    "analysis_unit",
    "inference_unit",
    "n_effective",
    "interval_type",
    "interval_level",
    "replicates_requested",
    "replicates_valid",
    "seed_set",
    "generator_script",
    "generator_script_sha256",
    "generated_utc",
    "exclusion_reason",
]
PANEL_COMMON_COLUMNS = [
    "comparator_order",
    "comparator_key",
    "comparator_label",
    "seed",
]
PANEL_TRAILING_COLUMNS = [
    "metric_status",
    "mean_across_seeds",
    "replicate_interval_lower",
    "replicate_interval_upper",
    "replicates_requested",
    "replicates_valid",
    "n_individuals_per_seed",
    "n_foods_per_seed",
    "n_pairs_per_seed",
    "evidence_role",
    "data_class",
    "seed_set",
    "source_file_sha256",
    "source_payload_sha256",
    "bootstrap_replicates_requested",
    "bootstrap_replicates_valid",
    "per_seed_bootstrap_interval_lower",
    "per_seed_bootstrap_interval_upper",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_builder():
    spec = importlib.util.spec_from_file_location("task4_make_main_figures", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png_phys(path: Path) -> tuple[int, int, int]:
    raw = path.read_bytes()
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", raw[16:24])
    offset = 8
    pixels_per_metre = 0
    while offset < len(raw):
        length = struct.unpack(">I", raw[offset : offset + 4])[0]
        kind = raw[offset + 4 : offset + 8]
        payload = raw[offset + 8 : offset + 8 + length]
        if kind == b"pHYs":
            x_ppm, y_ppm, unit = struct.unpack(">IIB", payload)
            assert x_ppm == y_ppm and unit == 1
            pixels_per_metre = x_ppm
            break
        offset += 12 + length
    return width, height, pixels_per_metre


def test_frozen_phase2_hashes_are_exact() -> None:
    assert {name: _sha256(FROZEN / name) for name in FROZEN_HASHES} == FROZEN_HASHES


def test_figure_manifest_contract_and_matrix_authorization() -> None:
    rows = _rows(MANIFEST)
    assert list(rows[0]) == FIGURE_MANIFEST_COLUMNS
    allowed_statuses = {
        "authorized_conceptual",
        "authorized_supplementary_synthetic",
        "blocked_external_data",
        "excluded_stale",
    }
    assert {row["status"] for row in rows} <= allowed_statuses
    assert [(r["figure_id"], r["panel_id"]) for r in rows if r["status"].startswith("authorized")] == [
        ("Fig1", "a"),
        ("Fig1", "b"),
        ("Fig1", "c"),
        ("Fig1", "d"),
        ("FigS1", "a"),
        ("FigS1", "b"),
    ]

    matrix = {row["claim_id"]: row for row in _rows(MATRIX)}
    for row in rows:
        if not row["status"].startswith("authorized"):
            continue
        for claim_id in row["claim_ids"].split("|"):
            assert matrix[claim_id]["status"] == "supported"
            assert row["display_class"] == matrix[claim_id]["main_or_supplementary"]
            expected = "Fig. 1" if row["figure_id"] == "Fig1" else "Supplementary Fig. S1"
            assert expected in matrix[claim_id]["figure_or_table"]


def test_exact_blocked_rows_are_fail_closed() -> None:
    blocked = {r["figure_id"]: r for r in _rows(MANIFEST) if r["status"] == "blocked_external_data"}
    assert list(blocked) == ["Fig2", "Fig3", "Fig4"]
    assert blocked["Fig2"]["title"] == "Population safety and food-group heterogeneity"
    assert blocked["Fig3"]["title"] == "Direct participant-by-meal response validity and ablations"
    assert blocked["Fig4"]["title"] == "Generalization robustness and evidence boundaries"
    assert blocked["Fig2"]["claim_ids"] == "RES-10|DIS-06"
    assert blocked["Fig3"]["claim_ids"] == "RES-11|DIS-06"
    assert blocked["Fig4"]["claim_ids"] == "RES-12|RES-13|RES-14|DIS-06"
    for row in blocked.values():
        assert row["panel_id"] == "ALL"
        assert row["source_sha256"] == ""
        assert row["generator_script"] == ""
        assert row["output_paths"] == ""
        assert row["unlock_artifacts"]
        assert row["exclusion_reason"]


def test_allowlist_contains_only_authorized_task4_assets() -> None:
    entries = [line for line in ALLOWLIST.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    assert len(entries) == len(set(entries))
    assert "figures/figure_manifest.csv" in entries
    assert "source_data/source_data_manifest.csv" in entries
    assert "source_data/figS1a_programmed_mapping_rmse.csv" in entries
    assert "source_data/figS1b_assignment_control_spearman.csv" in entries
    text = "\n".join(entries).lower()
    assert not re.search(r"figure[_-]?[234]|fig[234]|historical|framework|validation", text)
    for row in _rows(MANIFEST):
        if row["status"] in {"excluded_stale", "blocked_external_data"}:
            for path in filter(None, row["source_artifacts"].split("|")):
                assert path not in entries


def test_builder_has_no_rng_or_simulator_path_and_figure1_is_pure() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("synthetic_twin" in name or "simulator" in name for name in imports)
    source = SCRIPT.read_text(encoding="utf-8")
    assert "default_rng" not in source
    assert "np.random" not in source
    figure1 = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "render_figure_1"
    )
    figure1_source = ast.get_source_segment(source, figure1) or ""
    assert "read_csv" not in figure1_source
    assert "read_json" not in figure1_source
    assert "open(" not in figure1_source
    assert "CONCEPTUAL METHOD" in figure1_source
    assert "baseline + 12" not in figure1_source
    assert "score centring" not in figure1_source.casefold()


@pytest.mark.parametrize(
    ("filename", "metric", "summary_metric", "bootstrap_valid", "bootstrap_lower", "bootstrap_upper"),
    [
        (
            "figS1a_programmed_mapping_rmse.csv",
            "residual_rmse",
            "residual_rmse",
            "bootstrap_replicates_valid_rmse",
            "residual_rmse_ci_lower",
            "residual_rmse_ci_upper",
        ),
        (
            "figS1b_assignment_control_spearman.csv",
            "residual_spearman",
            "residual_spearman",
            "bootstrap_replicates_valid_spearman",
            "residual_spearman_ci_lower",
            "residual_spearman_ci_upper",
        ),
    ],
)
def test_panel_csvs_are_exact_filters(
    filename: str,
    metric: str,
    summary_metric: str,
    bootstrap_valid: str,
    bootstrap_lower: str,
    bootstrap_upper: str,
) -> None:
    panel = pd.read_csv(SOURCE_DATA / filename)
    assert panel.columns.tolist() == PANEL_COMMON_COLUMNS + [metric] + PANEL_TRAILING_COLUMNS
    assert panel.shape == (20, len(PANEL_COMMON_COLUMNS) + 1 + len(PANEL_TRAILING_COLUMNS))
    assert panel["comparator_key"].drop_duplicates().tolist() == COMPARATORS
    assert panel["comparator_label"].drop_duplicates().tolist() == LABELS
    assert panel.groupby("comparator_key", sort=False)["seed"].apply(list).tolist() == [SEEDS] * 4
    assert set(panel["replicates_requested"]) == {5}
    assert set(panel["replicates_valid"]) == ({0, 5} if metric == "residual_spearman" else {5})
    assert set(panel["n_individuals_per_seed"]) == {24}
    assert set(panel["n_foods_per_seed"]) == {12}
    assert set(panel["n_pairs_per_seed"]) == {288}
    assert set(panel["data_class"]) == {"synthetic"}
    assert set(panel["evidence_role"]) == {"correctly_specified_synthetic_positive_control"}

    replicate = pd.read_csv(FROZEN / "synthetic_attribute_twin_replicate_metrics.csv")
    expected = replicate[replicate["comparator"].isin(COMPARATORS)].copy()
    expected["comparator"] = pd.Categorical(expected["comparator"], COMPARATORS, ordered=True)
    expected = expected.sort_values(["comparator", "seed"])
    estimated = panel["metric_status"].eq("estimated")
    expected_estimated = expected.loc[estimated.to_numpy()]
    assert panel.loc[estimated, metric].tolist() == expected_estimated[metric].tolist()
    assert panel.loc[estimated, "bootstrap_replicates_valid"].tolist() == expected_estimated[bootstrap_valid].tolist()
    assert panel.loc[estimated, "per_seed_bootstrap_interval_lower"].tolist() == expected_estimated[bootstrap_lower].tolist()
    assert panel.loc[estimated, "per_seed_bootstrap_interval_upper"].tolist() == expected_estimated[bootstrap_upper].tolist()

    if metric == "residual_spearman":
        baseline = panel[panel["comparator_key"] == "fcs_baseline"]
        assert set(baseline["metric_status"]) == {"not_estimable_constant_residual"}
        assert baseline[metric].isna().all()
        assert baseline["mean_across_seeds"].isna().all()
        assert baseline["replicate_interval_lower"].isna().all()
        assert baseline["replicate_interval_upper"].isna().all()
        assert set(baseline["bootstrap_replicates_valid"]) == {0}

    summary = pd.read_csv(FROZEN / "synthetic_attribute_twin_summary.csv")
    summary = summary[(summary["comparator"].isin(COMPARATORS)) & (summary["metric"] == summary_metric)]
    summary = summary.set_index("comparator").loc[COMPARATORS]
    for key, (_, row) in zip(COMPARATORS, summary.iterrows()):
        selected = panel[panel["comparator_key"] == key]
        if metric == "residual_spearman" and key == "fcs_baseline":
            continue
        assert set(selected["mean_across_seeds"]) == {row["estimate"]}
        assert set(selected["replicate_interval_lower"]) == {row["ci_lower"]}
        assert set(selected["replicate_interval_upper"]) == {row["ci_upper"]}


def test_source_manifest_hashes_and_semantics() -> None:
    manifest = SOURCE_DATA / "source_data_manifest.csv"
    rows = _rows(manifest)
    assert list(rows[0]) == SOURCE_MANIFEST_COLUMNS
    assert [r["file_path"] for r in rows[:2]] == [
        "source_data/figS1a_programmed_mapping_rmse.csv",
        "source_data/figS1b_assignment_control_spearman.csv",
    ]
    assert [r["file_path"] for r in rows[2:]] == [
        f"source_data/frozen_synthetic_positive_control/{name}"
        for name in FROZEN_HASHES
    ]
    for row in rows:
        path = SUBMISSION / row["file_path"]
        assert row["generated_file_sha256"] == _sha256(path)
        assert row["generator_script_sha256"] == _sha256(SCRIPT)
        for key in (
            "source_artifact_sha256",
            "source_payload_sha256",
            "generated_file_sha256",
            "column_schema_sha256",
            "generator_script_sha256",
        ):
            assert re.fullmatch(r"[0-9a-f]{64}", row[key])
        assert row["data_class"] == "synthetic"
    for row in rows[:2]:
        assert row["inference_unit"] == "independent simulation seed"
        assert row["n_effective"] == "5"
        assert row["interval_type"] == "across-seed percentile range"
        assert row["replicates_requested"] == "5"
    for name, row in zip(FROZEN_HASHES, rows[2:]):
        assert row["generated_file_sha256"] == FROZEN_HASHES[name]
        assert row["source_artifact_sha256"] == FROZEN_HASHES[name]


def test_export_dimensions_fonts_vectors_and_text() -> None:
    expected = {
        "figure_1_attribute_calibration": ((183, 126), (4323, 2976)),
        "supplementary_figure_S1_synthetic_positive_control": ((183, 72), (4323, 1701)),
    }
    for stem, ((width_mm, height_mm), pixels) in expected.items():
        svg = (FIGURES / f"{stem}.svg").read_text(encoding="utf-8")
        assert f'width="{width_mm}mm"' in svg
        assert f'height="{height_mm}mm"' in svg
        assert "<text" in svg and "<image" not in svg
        assert "font-size: 7px" in svg or "font-size: 7.5px" in svg
        pdf = (FIGURES / f"{stem}.pdf").read_bytes()
        assert b"/Subtype /Image" not in pdf
        assert b"/FontFile2" in pdf and b"/ToUnicode" in pdf
        width, height, ppm = _png_phys(FIGURES / f"{stem}.png")
        assert (width, height) == pixels
        assert ppm == pytest.approx(23622, abs=1)
        with Image.open(FIGURES / f"{stem}.png") as image:
            assert image.mode in {"RGB", "RGBA"}
            assert image.info["dpi"][0] == pytest.approx(600, abs=0.1)
            assert image.info["icc_profile"]
    fig1_svg = (FIGURES / "figure_1_attribute_calibration.svg").read_text(encoding="utf-8")
    assert "CONCEPTUAL METHOD" in fig1_svg
    s1_svg = (FIGURES / "supplementary_figure_S1_synthetic_positive_control.svg").read_text(encoding="utf-8")
    assert "CORRECTLY SPECIFIED SYNTHETIC POSITIVE-CONTROL" in s1_svg
    assert "Method-only; not biological, clinical or external validation" in s1_svg
    assert "confidence interval" not in s1_svg.casefold()
    assert "not estimable (constant residual)" in s1_svg


def test_outputs_use_accessible_redundant_encoding() -> None:
    builder = _load_builder()
    assert builder.PALETTE == {
        "locked": "#0072B2",
        "mapping": "#009E73",
        "baseline": "#767676",
        "random": "#E69F00",
        "sattolo": "#CC79A7",
        "text": "#222222",
        "hairline": "#D9D9D9",
        "method_fill": "#DCEAF7",
    }
    encodings = {
        (builder.COMPARATOR_STYLES[key]["marker"], builder.COMPARATOR_STYLES[key]["filled"])
        for key in COMPARATORS
    }
    assert len(encodings) == 4


def test_two_clean_builds_are_deterministic(tmp_path: Path) -> None:
    builder = _load_builder()
    roots = [tmp_path / "one", tmp_path / "two"]
    for destination in roots:
        shutil.copytree(SUBMISSION, destination)
        builder.build_submission_assets(ROOT, destination, MANIFEST, ALLOWLIST)
    relative = [
        Path("source_data/figS1a_programmed_mapping_rmse.csv"),
        Path("source_data/figS1b_assignment_control_spearman.csv"),
        *[
            Path("source_data/frozen_synthetic_positive_control") / name
            for name in FROZEN_HASHES
        ],
        Path("figures/figure_1_attribute_calibration.svg"),
        Path("figures/figure_1_attribute_calibration.pdf"),
        Path("figures/figure_1_attribute_calibration.png"),
        Path("figures/supplementary_figure_S1_synthetic_positive_control.svg"),
        Path("figures/supplementary_figure_S1_synthetic_positive_control.pdf"),
        Path("figures/supplementary_figure_S1_synthetic_positive_control.png"),
        Path("source_data/source_data_manifest.csv"),
    ]
    for path in relative:
        assert (roots[0] / path).read_bytes() == (roots[1] / path).read_bytes(), path


def test_tex_captions_preserve_evidence_boundaries() -> None:
    main = (SUBMISSION / "sections.tex").read_text(encoding="utf-8")
    supplement = (SUBMISSION / "supplementary_results.tex").read_text(encoding="utf-8")
    assert "figure_1_attribute_calibration.pdf" in main
    assert "Every panel is conceptual; none contains observed or synthetic data" in " ".join(main.split())
    assert "Supplementary Fig. S1" in " ".join(main.split())
    assert "supplementary_figure_S1_synthetic_positive_control.pdf" in supplement
    assert "2.5th--97.5th percentile range across five seeds" in supplement
    assert "not estimable because its predicted residual is constant" in " ".join(supplement.split())
    assert "Biological misspecification was not assessed" in " ".join(supplement.split())
    assert "External validity was not established" in " ".join(supplement.split())
    assert "confidence interval" not in supplement.casefold()
