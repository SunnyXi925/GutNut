#!/usr/bin/env python3
"""Build the evidence-bound Phase 3 Task 4 figure package."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import pandas as pd
from PIL import Image, ImageCms


REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMISSION_RELATIVE = Path("manuscript/nature_food_submission")
FROZEN_RELATIVE = Path(
    "results/phase2/source-data/correctly_specified_synthetic_positive_control"
)
SCRIPT_RELATIVE = Path("code/src/scripts/make_main_figures.py")
BUILD_TIMESTAMP = "2026-08-14T00:00:00Z"
BUILD_DATETIME = datetime(2026, 8, 14, tzinfo=timezone.utc)
SEED_SET = [1701, 1702, 1703, 1704, 1705]
SEED_SET_TEXT = "[1701,1702,1703,1704,1705]"
EVIDENCE_ROLE = "correctly_specified_synthetic_positive_control"
DATA_CLASS = "synthetic"

FROZEN_HASHES = {
    "synthetic_attribute_twin_config.json": (
        "fd5a21d815406225f3fe5a6222cd39dafed7b98014dbc25190b61399fdb97c35"
    ),
    "synthetic_attribute_twin_manifest.json": (
        "df0684efa66377bdbdaec77182cdcbadc03b35cd4c5ec0403febf3c7d1096263"
    ),
    "synthetic_attribute_twin_replicate_metrics.csv": (
        "b0a0e379e8654ef1c370d614abe932a2a7cd0cedb4ad13ccf52ef6db0a7e70e6"
    ),
    "synthetic_attribute_twin_success_checks.csv": (
        "091bea04c4baed04fa4f4898d8c980cac081e3bc7dfb58435b5799fdfac32719"
    ),
    "synthetic_attribute_twin_summary.csv": (
        "b78ab05011a7c5f9bb150b22d50d72ff84797721c6a5c1d3bc8c37a3edbf9bdd"
    ),
}
COMPARATORS = [
    "fcs_baseline",
    "locked_attribute_gmnps",
    "random_microbiome",
    "sattolo_deranged_microbiome",
]
COMPARATOR_LABELS = {
    "fcs_baseline": "FCS baseline",
    "locked_attribute_gmnps": "Locked attribute GMNPS",
    "random_microbiome": "Random microbiome assignment",
    "sattolo_deranged_microbiome": "Sattolo-deranged assignment",
}
PALETTE = {
    "locked": "#0072B2",
    "mapping": "#009E73",
    "baseline": "#767676",
    "random": "#E69F00",
    "sattolo": "#CC79A7",
    "text": "#222222",
    "hairline": "#D9D9D9",
    "method_fill": "#DCEAF7",
}
COMPARATOR_STYLES = {
    "fcs_baseline": {"color": PALETTE["baseline"], "marker": "o", "filled": False},
    "locked_attribute_gmnps": {"color": PALETTE["locked"], "marker": "o", "filled": True},
    "random_microbiome": {"color": PALETTE["random"], "marker": "^", "filled": True},
    "sattolo_deranged_microbiome": {"color": PALETTE["sattolo"], "marker": "D", "filled": True},
}
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
FIGURE_OUTPUT_MANIFEST_COLUMNS = [
    "file_path",
    "figure_id",
    "status",
    "sha256",
    "bytes",
    "width_mm",
    "height_mm",
    "dpi",
    "color_space",
    "vector_text",
    "raster_images",
    "generator_script",
    "generator_script_sha256",
]
PANEL_TRAILING_COLUMNS = [
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
EXPECTED_OUTPUTS = {
    "Fig1": [
        "figures/figure_1_attribute_calibration.svg",
        "figures/figure_1_attribute_calibration.pdf",
        "figures/figure_1_attribute_calibration.png",
    ],
    "FigS1": [
        "figures/supplementary_figure_S1_synthetic_positive_control.svg",
        "figures/supplementary_figure_S1_synthetic_positive_control.pdf",
        "figures/supplementary_figure_S1_synthetic_positive_control.png",
    ],
}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(items: list[tuple[str, str]]) -> str:
    payload = "\n".join(f"{key}:{value}" for key, value in sorted(items)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return reader.fieldnames, list(reader)


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 0.8,
            "svg.fonttype": "none",
            "svg.hashsalt": "gmnps-phase3-task4",
            "pdf.fonttype": 42,
            "pdf.compression": 0,
            "savefig.transparent": False,
        }
    )


def _rewrite_svg_dimensions(path: Path, width_mm: int, height_mm: int) -> None:
    text = path.read_text(encoding="utf-8")
    text, width_count = __import__("re").subn(
        r'width="[^"]+"', f'width="{width_mm}mm"', text, count=1
    )
    text, height_count = __import__("re").subn(
        r'height="[^"]+"', f'height="{height_mm}mm"', text, count=1
    )
    if width_count != 1 or height_count != 1:
        raise ValueError(f"Unable to normalize SVG dimensions: {path}")
    text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    width_mm: int,
    height_mm: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    svg = output_dir / f"{stem}.svg"
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    figure.savefig(
        svg,
        format="svg",
        metadata={"Creator": "GMNPS Phase 3 Task 4", "Date": "2026-08-14"},
        facecolor="white",
    )
    _rewrite_svg_dimensions(svg, width_mm, height_mm)
    figure.savefig(
        pdf,
        format="pdf",
        metadata={
            "Creator": "GMNPS Phase 3 Task 4",
            "Producer": f"Matplotlib {mpl.__version__}",
            "CreationDate": BUILD_DATETIME,
            "ModDate": BUILD_DATETIME,
        },
        facecolor="white",
    )
    original_size = figure.get_size_inches().copy()
    pixel_width = round(width_mm / 25.4 * 600)
    pixel_height = round(height_mm / 25.4 * 600)
    figure.set_size_inches(pixel_width / 600, pixel_height / 600, forward=False)
    figure.savefig(png, format="png", dpi=600, facecolor="white")
    figure.set_size_inches(original_size, forward=False)
    srgb_profile = bytearray(
        ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    )
    srgb_profile[24:36] = struct.pack(">6H", 2026, 8, 14, 0, 0, 0)
    with Image.open(png) as image:
        image.convert("RGB").save(
            png,
            format="PNG",
            dpi=(600, 600),
            compress_level=9,
            icc_profile=bytes(srgb_profile),
        )
    plt.close(figure)


def _panel_letter(axis: plt.Axes, letter: str) -> None:
    axis.text(
        0,
        1.01,
        letter,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=PALETTE["text"],
    )


def render_figure_1(output_dir: Path) -> None:
    """Render the conceptual locked method without reading any data file."""
    set_publication_style()
    width_mm, height_mm = 183, 126
    figure = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white")
    figure.text(
        0.5,
        0.965,
        "CONCEPTUAL METHOD — NO OBSERVED OR SYNTHETIC DATA",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=PALETTE["text"],
    )
    figure.add_artist(
        Line2D([0.03, 0.97], [0.94, 0.94], transform=figure.transFigure, color=PALETTE["hairline"], lw=0.7)
    )

    panel_a = figure.add_axes([0.035, 0.555, 0.365, 0.34])
    panel_a.set_axis_off()
    _panel_letter(panel_a, "a")
    panel_a.text(0.02, 0.91, "Person i", ha="left", va="top", fontsize=7.5, fontweight="bold")
    panel_a.text(0.02, 0.82, "normalized nutrient\ncalibration", ha="left", va="top", fontsize=6.8)
    panel_a.text(0.35, 0.91, "Food j / attribute a", ha="left", va="top", fontsize=7.5, fontweight="bold")
    panel_a.text(0.35, 0.82, "normalized substrate exposure", ha="left", va="top", fontsize=6.8)
    panel_a.text(0.78, 0.91, "Attribute response", ha="left", va="top", fontsize=7.5, fontweight="bold")
    for y, person in zip([0.63, 0.55, 0.47], [r"$z_{i1}$", r"$\cdots$", r"$z_{in}$"]):
        panel_a.text(0.10, y, person, fontsize=8.5, ha="center", va="center", color=PALETTE["mapping"])
        panel_a.plot([0.16, 0.39], [y, y], color=PALETTE["hairline"], lw=0.6)
    panel_a.text(0.45, 0.55, r"$e_{jna}$", fontsize=9, ha="center", va="center", color=PALETTE["text"])
    panel_a.plot([0.51, 0.70], [0.55, 0.55], color=PALETTE["hairline"], lw=0.6)
    panel_a.text(0.61, 0.65, r"reviewed allocation $w_{na}$", fontsize=6.8, ha="center", va="bottom")
    panel_a.text(0.84, 0.63, r"$r_{ij1}$", fontsize=8.5, ha="center", va="center", color=PALETTE["locked"])
    panel_a.text(0.84, 0.55, r"$\cdots$", fontsize=8.5, ha="center", va="center", color=PALETTE["locked"])
    panel_a.text(0.84, 0.47, r"$r_{ija}$", fontsize=8.5, ha="center", va="center", color=PALETTE["locked"])
    panel_a.text(0.5, 0.35, r"$r_{ija}=\sum_n z_{in}\,e_{jna}\,w_{na}$", fontsize=9, ha="center", va="center")
    notes = [
        "development-frozen normalization",
        "per-100-kcal unless the published rule specifies another basis",
        "allocation weights sum to one within nutrient",
        "calibration feature; no causal effect inferred",
    ]
    for y, note in zip([0.23, 0.165, 0.10, 0.035], notes):
        panel_a.text(0.03, y, note, fontsize=6.8, ha="left", va="center", color=PALETTE["text"])
    panel_a.set_xlim(0, 1)
    panel_a.set_ylim(0, 1)

    panel_b = figure.add_axes([0.435, 0.555, 0.53, 0.34])
    panel_b.set_axis_off()
    _panel_letter(panel_b, "b")
    panel_b.text(0.03, 0.91, "published native point range", fontsize=7.5, fontweight="bold", ha="left")
    y_ruler = 0.57
    panel_b.plot([0.08, 0.93], [y_ruler, y_ruler], color=PALETTE["text"], lw=1.0)
    panel_b.plot([0.08, 0.08], [0.50, 0.64], color=PALETTE["text"], lw=1.0)
    panel_b.plot([0.93, 0.93], [0.50, 0.64], color=PALETTE["text"], lw=1.0)
    panel_b.text(0.08, 0.47, r"$l_a$", fontsize=8, ha="center", va="top")
    panel_b.text(0.93, 0.47, r"$h_a$", fontsize=8, ha="center", va="top")
    panel_b.text(0.03, y_ruler, r"$[l_a,h_a]$", fontsize=8, ha="right", va="center")
    baseline_x, calibrated_x = 0.38, 0.64
    panel_b.scatter([baseline_x], [y_ruler], s=36, facecolor="white", edgecolor=PALETTE["baseline"], linewidth=1.2, marker="o", zorder=4)
    panel_b.scatter([calibrated_x], [y_ruler], s=35, facecolor="white", edgecolor=PALETTE["mapping"], linewidth=1.2, marker="s", zorder=4)
    panel_b.add_patch(FancyArrowPatch((baseline_x + 0.02, y_ruler), (calibrated_x - 0.02, y_ruler), arrowstyle="-|>", mutation_scale=8, lw=1.2, color=PALETTE["locked"]))
    panel_b.text(baseline_x - 0.015, 0.43, r"baseline native point $p_{ja}$", fontsize=7, ha="right", va="top", color=PALETTE["baseline"])
    panel_b.text(calibrated_x + 0.015, 0.43, r"calibrated native point $p_{ija}$", fontsize=7, ha="left", va="top", color=PALETTE["mapping"])
    bracket_y = 0.76
    panel_b.plot([baseline_x, calibrated_x], [bracket_y, bracket_y], color=PALETTE["locked"], lw=1.0)
    panel_b.plot([baseline_x, baseline_x], [bracket_y - 0.025, bracket_y + 0.025], color=PALETTE["locked"], lw=1.0)
    panel_b.plot([calibrated_x, calibrated_x], [bracket_y - 0.025, bracket_y + 0.025], color=PALETTE["locked"], lw=1.0)
    panel_b.text(0.51, 0.80, r"maximum programmed movement = $0.20(h_a-l_a)$", fontsize=7.5, ha="center", va="bottom", color=PALETTE["locked"])
    panel_b.text(0.05, 0.28, r"$\mathrm{raw\_delta}_{ija}=0.20(h_a-l_a)\tanh(r_{ija}/2)$", fontsize=8.5, ha="left", va="center")
    panel_b.text(0.05, 0.15, r"$p_{ija}=\mathrm{clip}[p_{ja}+\mathrm{raw\_delta}_{ija},l_a,h_a]$", fontsize=8.5, ha="left", va="center")
    panel_b.text(0.95, 0.28, "primary locked mode: 20%", fontsize=7, ha="right", va="center", fontweight="bold")
    panel_b.text(0.95, 0.10, "Conceptual; not to scale", fontsize=7, ha="right", va="center", color=PALETTE["baseline"])
    panel_b.set_xlim(0, 1)
    panel_b.set_ylim(0, 1)

    panel_c = figure.add_axes([0.035, 0.065, 0.625, 0.41])
    panel_c.set_axis_off()
    _panel_letter(panel_c, "c")
    panel_c.text(0.03, 0.92, r"personalized native points $p_{ija}$", fontsize=7.5, fontweight="bold", ha="left")
    panel_c.text(0.51, 0.92, "published effective weights\nand applicability", fontsize=7, ha="center", va="top")
    paths = [
        "Nutrient ratios",
        "Vitamins — dynamic top 5",
        "Minerals — dynamic top 5",
        "Specific lipids — dynamic top 3;\ncontribution x0.5",
        "Fiber and protein",
        "Phytochemicals — contribution x0.5",
    ]
    ys = [0.78, 0.665, 0.55, 0.435, 0.32, 0.205]
    for index, (label, y) in enumerate(zip(paths, ys)):
        panel_c.text(0.03, y, label, fontsize=7, ha="left", va="center")
        for x in [0.38, 0.405, 0.43]:
            panel_c.scatter([x], [y], s=9, facecolor="white", edgecolor=PALETTE["mapping"], linewidth=0.7)
        panel_c.plot([0.45, 0.64], [y, y], color=PALETTE["mapping"], lw=1.0 if index in {1, 2, 3} else 0.8)
        panel_c.plot([0.47, 0.47], [y - 0.025, y + 0.025], color=PALETTE["hairline"], lw=0.6)
        panel_c.plot([0.62, 0.62], [y - 0.025, y + 0.025], color=PALETTE["hairline"], lw=0.6)
        panel_c.text(0.67, y, r"$D_{ijd}$", fontsize=8, ha="center", va="center", color=PALETTE["locked"])
        panel_c.plot([0.71, 0.79], [y, 0.50], color=PALETTE["hairline"], lw=0.6)
    panel_c.text(0.80, 0.50, r"$U_{ij}=Q_j+\sum_d D_{ijd}$", fontsize=9, ha="left", va="center", color=PALETTE["text"])
    panel_c.text(0.03, 0.115, r"food-specific fixed residual $Q_j$", fontsize=7.2, ha="left", va="center", color=PALETTE["baseline"])
    panel_c.plot([0.31, 0.75], [0.115, 0.115], color=PALETTE["baseline"], lw=0.8)
    panel_c.plot([0.75, 0.75], [0.115, 0.47], color=PALETTE["baseline"], lw=0.8)
    panel_c.plot([0.75, 0.79], [0.47, 0.50], color=PALETTE["baseline"], lw=0.8)
    panel_c.text(0.03, 0.005, "membership recalculated after point movement; immutable tie order", fontsize=6.8, ha="left", va="bottom")
    panel_c.text(0.98, 0.06, r"domains outside the selected set remain in $Q_j$", fontsize=6.8, ha="right", va="center")
    panel_c.set_xlim(0, 1)
    panel_c.set_ylim(0, 1)

    panel_d = figure.add_axes([0.695, 0.065, 0.27, 0.41])
    panel_d.set_axis_off()
    _panel_letter(panel_d, "d")
    panel_d.text(0.04, 0.91, "symbolic shared scale", fontsize=7.5, fontweight="bold", ha="left")
    scale_y = 0.74
    panel_d.plot([0.10, 0.90], [scale_y, scale_y], color=PALETTE["text"], lw=1.0)
    panel_d.plot([0.10, 0.10], [scale_y - 0.04, scale_y + 0.04], color=PALETTE["text"], lw=0.8)
    panel_d.plot([0.90, 0.90], [scale_y - 0.04, scale_y + 0.04], color=PALETTE["text"], lw=0.8)
    panel_d.text(0.10, 0.67, "1", fontsize=7, ha="center")
    panel_d.text(0.90, 0.67, "100", fontsize=7, ha="center")
    anchor_x = 0.51
    panel_d.plot([anchor_x, anchor_x], [scale_y - 0.07, scale_y + 0.07], color=PALETTE["locked"], lw=1.4)
    panel_d.text(anchor_x, 0.84, r"Food Compass 2.0 anchor $F_j$", fontsize=7, ha="center", va="bottom", color=PALETTE["locked"])
    panel_d.plot([0.37, 0.65], [0.60, 0.60], color=PALETTE["random"], lw=1.1)
    panel_d.plot([0.37, 0.37], [0.575, 0.625], color=PALETTE["random"], lw=1.0)
    panel_d.plot([0.65, 0.65], [0.575, 0.625], color=PALETTE["random"], lw=1.0)
    panel_d.text(0.51, 0.55, "allowed final deviation: -12 to +12 points", fontsize=7, ha="center", va="top", color=PALETTE["random"])
    panel_d.text(0.04, 0.44, r"$F_{\mathrm{native},ij}=\mathrm{unscaled\_to\_FCS}(U_{ij})$", fontsize=8, ha="left", va="center")
    panel_d.text(0.04, 0.34, r"$S_{ij}=\mathrm{clip}_{[1,100]}\{F_j+$", fontsize=8, ha="left", va="center")
    panel_d.text(0.17, 0.27, r"$\mathrm{clip}_{[-12,12]}(F_{\mathrm{native},ij}-F_j)\}$", fontsize=8, ha="left", va="center")
    panel_d.text(0.04, 0.19, "score_centering = none", fontsize=7, ha="left", va="center", fontweight="bold")
    panel_d.text(0.04, 0.14, "shared scale retained by construction", fontsize=7, ha="left", va="center")
    panel_d.text(0.04, 0.09, "computational bound, not observed", fontsize=7, ha="left", va="center")
    panel_d.text(0.04, 0.045, "population preservation", fontsize=7, ha="left", va="center")
    panel_d.text(0.96, 0.005, "Conceptual; not to scale", fontsize=7, ha="right", va="center", color=PALETTE["baseline"])
    panel_d.set_xlim(0, 1)
    panel_d.set_ylim(0, 1)

    save_figure(figure, output_dir, "figure_1_attribute_calibration", width_mm, height_mm)


def verify_frozen_bundle(repo_root: Path) -> dict[str, Any]:
    frozen_dir = repo_root / FROZEN_RELATIVE
    actual = {name: sha256_path(frozen_dir / name) for name in FROZEN_HASHES}
    if actual != FROZEN_HASHES:
        mismatches = {
            name: {"expected": FROZEN_HASHES[name], "actual": actual.get(name)}
            for name in FROZEN_HASHES
            if actual.get(name) != FROZEN_HASHES[name]
        }
        raise ValueError(f"Frozen Phase 2 input hash mismatch: {mismatches}")

    config = json.loads((frozen_dir / "synthetic_attribute_twin_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((frozen_dir / "synthetic_attribute_twin_manifest.json").read_text(encoding="utf-8"))
    replicate = pd.read_csv(
        frozen_dir / "synthetic_attribute_twin_replicate_metrics.csv", dtype=str
    )
    success = pd.read_csv(
        frozen_dir / "synthetic_attribute_twin_success_checks.csv", dtype=str
    )
    summary = pd.read_csv(
        frozen_dir / "synthetic_attribute_twin_summary.csv", dtype=str
    )
    if config.get("seeds") != SEED_SET or manifest.get("seeds") != SEED_SET:
        raise ValueError("Frozen seed set differs from the authorized five seeds")
    if manifest.get("config", {}).get("n_individuals") != 24 or manifest.get("config", {}).get("n_foods") != 12:
        raise ValueError("Frozen nested simulation dimensions differ from 24 x 12")
    if manifest.get("evidence_role") != EVIDENCE_ROLE or manifest.get("data_class") != DATA_CLASS:
        raise ValueError("Frozen evidence role or data class is not authorized")
    if manifest.get("clinical_or_external_validation") is not False:
        raise ValueError("Frozen manifest must deny clinical and external validation")
    if set(replicate["seed"].astype(int).unique()) != set(SEED_SET):
        raise ValueError("Replicate table contains an unauthorized seed set")
    if set(success["seed_set"].unique()) != {"[1701,1702,1703,1704,1705]"}:
        raise ValueError("Success-check table contains an unauthorized seed set")
    if not success["passed"].eq("True").all():
        raise ValueError("Frozen positive-control success checks are not all true")

    payload_items = [("manifest", str(manifest["payload_sha256"])), ("config", str(config["payload_sha256"]))]
    for filename, metadata in sorted(manifest["files"].items()):
        payload_items.append((filename, str(metadata["payload_sha256"])))
    return {
        "frozen_dir": frozen_dir,
        "config": config,
        "manifest": manifest,
        "replicate": replicate,
        "success": success,
        "summary": summary,
        "source_artifact_sha256": canonical_digest(list(actual.items())),
        "source_payload_sha256": canonical_digest(payload_items),
    }


def _panel_table(bundle: dict[str, Any], metric: str) -> pd.DataFrame:
    replicate: pd.DataFrame = bundle["replicate"]
    summary: pd.DataFrame = bundle["summary"]
    if metric == "residual_rmse":
        bootstrap_valid = "bootstrap_replicates_valid_rmse"
        bootstrap_lower = "residual_rmse_ci_lower"
        bootstrap_upper = "residual_rmse_ci_upper"
    elif metric == "residual_spearman":
        bootstrap_valid = "bootstrap_replicates_valid_spearman"
        bootstrap_lower = "residual_spearman_ci_lower"
        bootstrap_upper = "residual_spearman_ci_upper"
    else:
        raise ValueError(f"Unauthorized S1 metric: {metric}")

    frames = []
    summary_rows = summary[(summary["comparator"].isin(COMPARATORS)) & (summary["metric"] == metric)]
    if len(summary_rows) != len(COMPARATORS):
        raise ValueError(f"Frozen summary selection for {metric} is incomplete")
    summary_by_key = summary_rows.set_index("comparator")
    for order, comparator in enumerate(COMPARATORS, start=1):
        selected = replicate[
            (replicate["comparator"] == comparator)
            & (replicate["seed"].astype(int).isin(SEED_SET))
        ].copy()
        selected["seed_integer"] = selected["seed"].astype(int)
        selected = selected.sort_values("seed_integer")
        if selected["seed_integer"].tolist() != SEED_SET:
            raise ValueError(f"Frozen replicate selection is incomplete for {comparator}")
        summary_row = summary_by_key.loc[comparator]
        frame = pd.DataFrame(
            {
                "comparator_order": order,
                "comparator_key": comparator,
                "comparator_label": COMPARATOR_LABELS[comparator],
                "seed": selected["seed_integer"].to_numpy(),
                metric: selected[metric].to_numpy(),
                "mean_across_seeds": summary_row["estimate"],
                "replicate_interval_lower": summary_row["ci_lower"],
                "replicate_interval_upper": summary_row["ci_upper"],
                "replicates_requested": int(summary_row["replicates_requested"]),
                "replicates_valid": int(summary_row["replicates_valid"]),
                "n_individuals_per_seed": selected["n_individuals"].to_numpy(),
                "n_foods_per_seed": selected["n_foods"].to_numpy(),
                "n_pairs_per_seed": selected["n_individual_food_pairs"].to_numpy(),
                "evidence_role": EVIDENCE_ROLE,
                "data_class": DATA_CLASS,
                "seed_set": SEED_SET_TEXT,
                "source_file_sha256": bundle["source_artifact_sha256"],
                "source_payload_sha256": bundle["source_payload_sha256"],
                "bootstrap_replicates_requested": selected["bootstrap_replicates_requested"].to_numpy(),
                "bootstrap_replicates_valid": selected[bootstrap_valid].to_numpy(),
                "per_seed_bootstrap_interval_lower": selected[bootstrap_lower].to_numpy(),
                "per_seed_bootstrap_interval_upper": selected[bootstrap_upper].to_numpy(),
            }
        )
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    expected_columns = [
        "comparator_order",
        "comparator_key",
        "comparator_label",
        "seed",
        metric,
        *PANEL_TRAILING_COLUMNS,
    ]
    return result.loc[:, expected_columns]


def export_panel_tables(repo_root: Path, source_data_dir: Path) -> dict[str, Path]:
    bundle = verify_frozen_bundle(repo_root)
    source_data_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "a": source_data_dir / "figS1a_programmed_mapping_rmse.csv",
        "b": source_data_dir / "figS1b_assignment_control_spearman.csv",
    }
    _panel_table(bundle, "residual_rmse").to_csv(
        outputs["a"], index=False, lineterminator="\n"
    )
    _panel_table(bundle, "residual_spearman").to_csv(
        outputs["b"], index=False, lineterminator="\n"
    )
    return outputs


def _plot_s1_panel(
    axis: plt.Axes,
    table: pd.DataFrame,
    metric: str,
    panel_letter: str,
    panel_title: str,
    xlabel: str,
    x_limits: tuple[float, float],
    zero_line: bool = False,
) -> None:
    y_positions = [3, 2, 1, 0]
    seed_offsets = [-0.12, -0.06, 0.0, 0.06, 0.12]
    if zero_line:
        axis.axvline(0, color=PALETTE["baseline"], lw=0.7, zorder=0)
    for comparator, y in zip(COMPARATORS, y_positions):
        selected = table[table["comparator_key"] == comparator].sort_values("seed")
        style = COMPARATOR_STYLES[comparator]
        x_values = selected[metric].to_numpy()
        facecolor = style["color"] if style["filled"] else "white"
        axis.scatter(
            x_values,
            [y + offset for offset in seed_offsets],
            s=18,
            marker=style["marker"],
            facecolor=facecolor,
            edgecolor=style["color"],
            linewidth=0.7,
            alpha=0.72,
            zorder=3,
        )
        mean = float(selected["mean_across_seeds"].iloc[0])
        lower = float(selected["replicate_interval_lower"].iloc[0])
        upper = float(selected["replicate_interval_upper"].iloc[0])
        axis.plot([lower, upper], [y, y], color=style["color"], lw=2.2, solid_capstyle="butt", zorder=2)
        axis.scatter(
            [mean],
            [y],
            s=52,
            marker=style["marker"],
            facecolor=facecolor,
            edgecolor=style["color"],
            linewidth=1.2,
            zorder=4,
        )
        span = x_limits[1] - x_limits[0]
        if mean < x_limits[0] + 0.25 * span:
            label_x = upper + 0.025 * span
            horizontal_alignment = "left"
        elif mean > x_limits[0] + 0.75 * span:
            label_x = lower - 0.025 * span
            horizontal_alignment = "right"
        else:
            label_x = mean
            horizontal_alignment = "center"
        axis.text(
            label_x,
            y + 0.17,
            f"{mean:.3f} [{lower:.3f}, {upper:.3f}]",
            fontsize=7,
            color=PALETTE["text"],
            ha=horizontal_alignment,
            va="center",
        )
    labels = [
        "FCS baseline",
        "Locked attribute\nGMNPS",
        "Random microbiome\nassignment",
        "Sattolo-deranged\nassignment",
    ]
    axis.set_yticks(y_positions, labels)
    axis.set_ylim(-0.45, 3.45)
    axis.set_xlim(*x_limits)
    axis.set_xlabel(xlabel, labelpad=4)
    axis.set_title(f"{panel_letter}  {panel_title}", loc="left", fontsize=8, fontweight="bold", pad=8)
    axis.grid(axis="x", color=PALETTE["hairline"], lw=0.5)
    axis.tick_params(axis="y", length=0, pad=4)
    axis.tick_params(axis="x", width=0.6, length=3)
    for side in ["top", "right", "left"]:
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(PALETTE["text"])
    axis.spines["bottom"].set_linewidth(0.6)


def render_supplementary_figure_s1(output_dir: Path, panel_tables: dict[str, Path]) -> None:
    set_publication_style()
    rmse = pd.read_csv(panel_tables["a"])
    spearman = pd.read_csv(panel_tables["b"])
    width_mm, height_mm = 183, 72
    figure = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white")
    figure.text(
        0.5,
        0.95,
        "CORRECTLY SPECIFIED SYNTHETIC POSITIVE-CONTROL",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=PALETTE["text"],
    )
    axis_a = figure.add_axes([0.155, 0.35, 0.31, 0.48])
    axis_b = figure.add_axes([0.66, 0.35, 0.31, 0.48])
    _plot_s1_panel(
        axis_a,
        rmse,
        "residual_rmse",
        "a",
        "Programmed residual error",
        "Residual RMSE against programmed\nsynthetic response (score units)",
        (0.0, 5.45),
    )
    _plot_s1_panel(
        axis_b,
        spearman,
        "residual_spearman",
        "b",
        "Programmed residual rank recovery",
        "Spearman correlation with programmed\nsynthetic residual",
        (-0.42, 1.36),
        zero_line=True,
    )
    figure.text(
        0.5,
        0.115,
        "Small points: seeds 1701–1705; large point: mean; line: 95% empirical replicate interval",
        ha="center",
        va="center",
        fontsize=7,
        color=PALETTE["text"],
    )
    figure.text(
        0.5,
        0.04,
        "Method-only; not biological, clinical or external validation",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=PALETTE["text"],
    )
    save_figure(
        figure,
        output_dir,
        "supplementary_figure_S1_synthetic_positive_control",
        width_mm,
        height_mm,
    )


def validate_authorization(
    repo_root: Path,
    figure_manifest_path: Path,
    allowlist_path: Path,
) -> list[dict[str, str]]:
    fields, rows = read_csv_rows(figure_manifest_path)
    if fields != FIGURE_MANIFEST_COLUMNS:
        raise ValueError("Figure manifest schema differs from the Task 4 contract")
    legal_status = {
        "authorized_conceptual",
        "authorized_supplementary_synthetic",
        "blocked_external_data",
        "excluded_stale",
    }
    if any(row["status"] not in legal_status for row in rows):
        raise ValueError("Figure manifest contains an illegal status")

    matrix_fields, matrix_rows = read_csv_rows(
        repo_root / "manuscript/nature_food_submission/claim_evidence_matrix.csv"
    )
    if "claim_id" not in matrix_fields:
        raise ValueError("Claim/evidence matrix has no claim_id field")
    matrix = {row["claim_id"]: row for row in matrix_rows}
    for row in rows:
        if row["status"].startswith("authorized"):
            for claim_id in row["claim_ids"].split("|"):
                if claim_id not in matrix or matrix[claim_id]["status"] != "supported":
                    raise ValueError(f"Rendered panel maps to unsupported claim {claim_id}")
                if matrix[claim_id]["main_or_supplementary"] != row["display_class"]:
                    raise ValueError(f"Display class mismatch for {claim_id}")
        if row["status"] == "blocked_external_data":
            if row["source_sha256"] or row["generator_script"] or row["output_paths"]:
                raise ValueError(f"Blocked display {row['figure_id']} has a render path")
            if not row["unlock_artifacts"] or not row["exclusion_reason"]:
                raise ValueError(f"Blocked display {row['figure_id']} lacks an unlock boundary")

    allowlist = [
        line.strip()
        for line in allowlist_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(allowlist) != len(set(allowlist)):
        raise ValueError("Submission allowlist contains duplicate entries")
    allowed = set(allowlist)
    for row in rows:
        if row["status"] in {"blocked_external_data", "excluded_stale"}:
            excluded = set(filter(None, row["source_artifacts"].split("|")))
            excluded.update(filter(None, row["output_paths"].split("|")))
            overlap = excluded & allowed
            if overlap:
                raise ValueError(f"Excluded assets entered package allowlist: {sorted(overlap)}")
    for outputs in EXPECTED_OUTPUTS.values():
        if not set(outputs) <= allowed:
            raise ValueError("Authorized output is missing from the package allowlist")
    return rows


def write_source_data_manifest(
    repo_root: Path,
    submission_dir: Path,
    panel_tables: dict[str, Path],
) -> Path:
    bundle = verify_frozen_bundle(repo_root)
    script = repo_root / SCRIPT_RELATIVE
    script_hash = sha256_path(script)
    source_paths = "|".join(str(FROZEN_RELATIVE / name) for name in FROZEN_HASHES)
    definitions = [
        (
            "a",
            "source_data/figS1a_programmed_mapping_rmse.csv",
            "ABS-05|RES-06|DIS-02",
        ),
        (
            "b",
            "source_data/figS1b_assignment_control_spearman.csv",
            "ABS-05|RES-07|RES-08|DIS-02",
        ),
    ]
    rows = []
    for panel, relative_path, claim_ids in definitions:
        table = panel_tables[panel]
        columns, data_rows = read_csv_rows(table)
        rows.append(
            {
                "file_path": relative_path,
                "figure_id": "FigS1",
                "panel_id": panel,
                "status": "authorized_supplementary_synthetic",
                "claim_ids": claim_ids,
                "evidence_tier": "computational_feasibility",
                "evidence_role": EVIDENCE_ROLE,
                "data_class": DATA_CLASS,
                "source_artifact_path": source_paths,
                "source_artifact_sha256": bundle["source_artifact_sha256"],
                "source_payload_sha256": bundle["source_payload_sha256"],
                "generated_file_sha256": sha256_path(table),
                "row_count": str(len(data_rows)),
                "column_schema_sha256": hashlib.sha256(("\n".join(columns) + "\n").encode("utf-8")).hexdigest(),
                "analysis_unit": "independent simulation seed",
                "inference_unit": "independent simulation seed",
                "n_effective": "5",
                "interval_type": "empirical replicate interval",
                "interval_level": "95% (2.5th-97.5th percentiles across seeds)",
                "replicates_requested": "5",
                "replicates_valid": "5",
                "seed_set": SEED_SET_TEXT,
                "generator_script": str(SCRIPT_RELATIVE),
                "generator_script_sha256": script_hash,
                "generated_utc": BUILD_TIMESTAMP,
                "exclusion_reason": "",
            }
        )
    output = submission_dir / "source_data/source_data_manifest.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_figure_output_manifest(repo_root: Path, submission_dir: Path) -> Path:
    script_hash = sha256_path(repo_root / SCRIPT_RELATIVE)
    definitions = [
        ("Fig1", "authorized_conceptual", "figure_1_attribute_calibration", 183, 126),
        (
            "FigS1",
            "authorized_supplementary_synthetic",
            "supplementary_figure_S1_synthetic_positive_control",
            183,
            72,
        ),
    ]
    rows = []
    for figure_id, status, stem, width_mm, height_mm in definitions:
        for extension in ["svg", "pdf", "png"]:
            relative = Path("figures") / f"{stem}.{extension}"
            path = submission_dir / relative
            rows.append(
                {
                    "file_path": str(relative),
                    "figure_id": figure_id,
                    "status": status,
                    "sha256": sha256_path(path),
                    "bytes": str(path.stat().st_size),
                    "width_mm": str(width_mm),
                    "height_mm": str(height_mm),
                    "dpi": "600" if extension == "png" else "not_applicable",
                    "color_space": "sRGB" if extension == "png" else "vector",
                    "vector_text": "true" if extension in {"svg", "pdf"} else "not_applicable",
                    "raster_images": "0" if extension in {"svg", "pdf"} else "1",
                    "generator_script": str(SCRIPT_RELATIVE),
                    "generator_script_sha256": script_hash,
                }
            )
    output = submission_dir / "figures/figure_output_manifest.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIGURE_OUTPUT_MANIFEST_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return output


def build_submission_assets(
    repo_root: Path,
    submission_dir: Path,
    figure_manifest_path: Path,
    allowlist_path: Path,
) -> None:
    repo_root = repo_root.resolve()
    submission_dir = submission_dir.resolve()
    validate_authorization(repo_root, figure_manifest_path.resolve(), allowlist_path.resolve())
    figures_dir = submission_dir / "figures"
    source_data_dir = submission_dir / "source_data"
    panel_tables = export_panel_tables(repo_root, source_data_dir)
    render_figure_1(figures_dir)
    render_supplementary_figure_s1(figures_dir, panel_tables)
    write_source_data_manifest(repo_root, submission_dir, panel_tables)
    write_figure_output_manifest(repo_root, submission_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the evidence-bound conceptual Fig. 1 and synthetic method-only Fig. S1"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--submission-dir", type=Path)
    parser.add_argument("--figure-manifest", type=Path)
    parser.add_argument("--allowlist", type=Path)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    repo_root = arguments.repo_root.resolve()
    submission_dir = (
        arguments.submission_dir.resolve()
        if arguments.submission_dir
        else repo_root / SUBMISSION_RELATIVE
    )
    figure_manifest = (
        arguments.figure_manifest.resolve()
        if arguments.figure_manifest
        else repo_root / SUBMISSION_RELATIVE / "figures/figure_manifest.csv"
    )
    allowlist = (
        arguments.allowlist.resolve()
        if arguments.allowlist
        else repo_root / SUBMISSION_RELATIVE / "figures/submission_asset_allowlist.txt"
    )
    build_submission_assets(repo_root, submission_dir, figure_manifest, allowlist)
    return 0


if __name__ == "__main__":
    sys.exit(main())
