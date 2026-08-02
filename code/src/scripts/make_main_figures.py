#!/usr/bin/env python3
"""Create manuscript main-figure drafts from GMNPS source data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


PALETTE = {
    "blue": "#0F4D92",
    "blue2": "#3775BA",
    "green": "#8BCF8B",
    "green_light": "#DDF3DE",
    "red": "#B64342",
    "red_light": "#F6CFCB",
    "neutral": "#CFCECE",
    "dark": "#272727",
    "teal": "#42949E",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 12,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.4,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_framework(out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.set_axis_off()
    boxes = [
        (0.05, 0.62, "Static NPS prior\nNPS_j", PALETTE["neutral"]),
        (0.40, 0.62, "Microbiome-calibrated\ndeviation D_ij", PALETTE["green_light"]),
        (0.74, 0.62, "Anchored GMNPS\nclip(NPS_j + D_ij)", PALETTE["blue2"]),
    ]
    for x, y, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.22, 0.18, facecolor=color, edgecolor="black", linewidth=1.2))
        ax.text(x + 0.11, y + 0.09, text, ha="center", va="center", fontsize=11)
    ax.annotate("", xy=(0.39, 0.71), xytext=(0.28, 0.71), arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.annotate("", xy=(0.73, 0.71), xytext=(0.63, 0.71), arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.set_title("a  Anchored personalized calibration", loc="left", fontweight="bold")

    ax = axes[0, 1]
    z = np.linspace(-4, 4, 200)
    d = 12 * np.tanh(z / 2)
    ax.plot(z, d, color=PALETTE["blue"], lw=2.5)
    ax.axhline(12, color=PALETTE["red"], lw=1, ls="--")
    ax.axhline(-12, color=PALETTE["red"], lw=1, ls="--")
    ax.axhline(0, color=PALETTE["dark"], lw=0.8)
    ax.set_xlabel("Robust z-score of raw response")
    ax.set_ylabel("Deviation D_ij")
    ax.set_title("b  Bounded deviation transform", loc="left", fontweight="bold")

    ax = axes[1, 0]
    channels = ["MAC/SCFA\nplant matrix", "Lipid/bile acid\nTMAO", "Residual"]
    values = [1.0, 1.0, 0.55]
    colors = [PALETTE["green"], PALETTE["red_light"], PALETTE["neutral"]]
    ax.bar(channels, values, color=colors, edgecolor="black")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Reported attribution status")
    ax.set_title("c  Mechanistic channel reporting", loc="left", fontweight="bold")
    ax.text(2, 0.62, "used for allocation\nnot primary report", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 1]
    rng = np.random.default_rng(5)
    fcs = 68
    deviations = np.sort(12 * np.tanh(rng.normal(0, 1.2, 80) / 2))
    scores = np.clip(fcs + deviations, 1, 100)
    ax.scatter(scores, np.zeros_like(scores), color=PALETTE["blue2"], alpha=0.55, s=28)
    ax.axvline(fcs, color=PALETTE["dark"], lw=2, label="NPS prior")
    ax.set_yticks([])
    ax.set_xlabel("Personalized score for one food")
    ax.set_title("d  Illustrative single-food score distribution", loc="left", fontweight="bold")
    ax.legend(loc="upper left")
    fig.suptitle("Figure 1. Personalized calibration transforms static nutrient profiling into an anchored precision-nutrition score.", y=1.02, fontsize=14)
    fig.tight_layout(pad=1.6)
    save_figure(fig, out_dir, "figure_1_framework")


def figure_preservation(out_dir: Path, fcs_audit: Path, scores: Path, preservation: Path) -> None:
    audit = pd.read_json(fcs_audit, typ="series")
    individual = pd.read_csv(scores)
    metrics = pd.read_json(preservation, typ="series")
    mean_scores = individual.groupby(["food_id", "FCS2"], as_index=False)["GMNPS_score"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    vals = [audit["row_count"], audit["unique_foodcode_count"], audit["duplicate_foodcodes"].__len__(), audit["missing_nutri_score_count"]]
    labels = ["Rows", "Unique\nFoodcode", "Duplicate\ncodes", "Missing\nlabels"]
    ax.bar(labels, vals, color=[PALETTE["blue2"], PALETTE["green"], PALETTE["red_light"], PALETTE["neutral"]], edgecolor="black")
    ax.set_title("a  FCS2 Table S5 extraction audit", loc="left", fontweight="bold")
    ax.set_ylabel("Count")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    ax.scatter(mean_scores["FCS2"], mean_scores["GMNPS_score"], s=20, alpha=0.65, color=PALETTE["blue2"], edgecolor="none")
    ax.plot([1, 100], [1, 100], color=PALETTE["dark"], lw=1.2, ls="--")
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Baseline NPS prior")
    ax.set_ylabel("Population-mean GMNPS")
    ax.set_title("b  Population ranking is preserved", loc="left", fontweight="bold")
    ax.text(5, 92, f"Spearman = {metrics['spearman_fcs2_gmnps_mean']:.6f}", fontsize=10)

    ax = axes[2]
    bins = [0, 30, 70, 100]
    labels = ["Minimize", "Moderate", "Encourage"]
    base_cat = pd.cut(mean_scores["FCS2"], bins=bins, labels=labels, include_lowest=True)
    gmnps_cat = pd.cut(mean_scores["GMNPS_score"], bins=bins, labels=labels, include_lowest=True)
    counts = pd.crosstab(base_cat, gmnps_cat).reindex(index=labels, columns=labels, fill_value=0)
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    matrix = counts.div(row_sums, axis=0).fillna(0).to_numpy(dtype=float)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), labels, rotation=35, ha="right")
    ax.set_yticks(range(3), labels)
    for row in range(3):
        for col in range(3):
            ax.text(col, row, str(int(counts.iloc[row, col])), ha="center", va="center", fontsize=10)
    ax.set_xlabel("GMNPS population category")
    ax.set_ylabel("Baseline NPS category")
    ax.set_title("c  No population-mean category shifts", loc="left", fontweight="bold")
    ax.text(0.5, -0.55, f"Shift fraction = {metrics['category_shift_fraction']:.3f}", transform=ax.transAxes, ha="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Figure 2. Anchored calibration preserves population-level nutrient-profiling consensus.", y=1.03, fontsize=14)
    fig.tight_layout(pad=1.4)
    save_figure(fig, out_dir, "figure_2_preservation")


def figure_heterogeneity(out_dir: Path, scores: Path, heterogeneity: Path, summary: Path) -> None:
    individual = pd.read_csv(scores)
    hetero = pd.read_csv(heterogeneity)
    food_summary = pd.read_csv(summary)
    pivot = individual.pivot(index="individual_id", columns="food_id", values="GMNPS_delta")
    col_order = individual[["food_id", "food_group"]].drop_duplicates().sort_values(["food_group", "food_id"])["food_id"]
    pivot = pivot[col_order]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    im = ax.imshow(pivot.to_numpy()[:80, :80], aspect="auto", cmap="RdBu_r", vmin=-12, vmax=12)
    ax.set_xlabel("Foods ordered by group")
    ax.set_ylabel("Individuals")
    ax.set_title("a  Representative 80 x 80 deviation subset", loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, label="GMNPS delta")

    ax = axes[0, 1]
    x = np.arange(len(hetero))
    ax.bar(x - 0.18, hetero["mac_variance"], width=0.36, label="MAC", color=PALETTE["green"], edgecolor="black")
    ax.bar(x + 0.18, hetero["lipid_variance"], width=0.36, label="LIPID", color=PALETTE["red_light"], edgecolor="black")
    ax.set_xticks(x, hetero["food_group"])
    ax.set_ylabel("Variance")
    ax.set_title("b  Channel variance by food group", loc="left", fontweight="bold")
    ax.legend()

    ax = axes[1, 0]
    color_map = {"plant": PALETTE["green"], "animal": PALETTE["red"], "mixed": PALETTE["teal"]}
    for group, frame in food_summary.groupby("food_group"):
        ax.scatter(frame["MAC_variance"], frame["LIPID_variance"], label=group, alpha=0.75, s=34, color=color_map.get(group, PALETTE["neutral"]))
    ax.set_xlabel("MAC variance")
    ax.set_ylabel("LIPID variance")
    ax.set_title("c  Food-level mechanistic heterogeneity", loc="left", fontweight="bold")
    ax.legend()

    ax = axes[1, 1]
    examples = food_summary.sort_values("GMNPS_sd", ascending=False).head(8)
    y = np.arange(len(examples))
    ax.hlines(y, examples["GMNPS_p05"], examples["GMNPS_p95"], color=PALETTE["blue"], lw=3)
    ax.scatter(examples["FCS2"], y, color=PALETTE["dark"], zorder=3, label="NPS prior")
    ax.scatter(examples["GMNPS_mean"], y, color=PALETTE["blue2"], zorder=3, label="GMNPS mean")
    ax.set_yticks(y, examples["food_name"].str.replace("Synthetic ", "", regex=False).str.slice(0, 28))
    ax.set_xlabel("Score")
    ax.set_title("d  Eight most heterogeneous synthetic foods", loc="left", fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("Figure 3. GMNPS reveals bounded and interpretable heterogeneity across foods and individuals.", y=1.02, fontsize=14)
    fig.tight_layout(pad=1.4)
    save_figure(fig, out_dir, "figure_3_heterogeneity")


def figure_benchmark(out_dir: Path, benchmark: Path) -> None:
    bench = pd.read_csv(benchmark)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    order = ["FCS2 only", "unanchored microbiome score", "anchored GMNPS", "random microbiome", "shuffled microbiome"]
    b = bench.set_index("model").loc[order].reset_index()
    colors = [PALETTE["neutral"], PALETTE["red_light"], PALETTE["blue"], PALETTE["neutral"], PALETTE["neutral"]]

    ax = axes[0]
    ax.barh(b["model"], b["individual_response_spearman"], color=colors, edgecolor="black")
    ax.set_xlabel("Spearman")
    ax.set_title("a  Simulated individual response recovery", loc="left", fontweight="bold")
    ax.set_xlim(0, 1)

    ax = axes[1]
    vals = b["personalized_residual_spearman"].copy()
    ax.barh(b["model"], vals.fillna(0), color=colors, edgecolor="black")
    for idx, row in b.iterrows():
        if pd.isna(row["personalized_residual_spearman"]):
            ax.text(0.02, idx, "NA\nconstant residual", va="center", ha="left", fontsize=8)
    ax.set_xlabel("Spearman")
    ax.set_title("b  Personalized residual recovery", loc="left", fontweight="bold")
    ax.set_xlim(0, max(0.45, vals.max(skipna=True) + 0.05))

    ax = axes[2]
    ax.scatter(bench["nps_preservation_spearman"], bench["personalized_residual_spearman"], s=80, color=PALETTE["neutral"], edgecolor="black")
    anchor = bench[bench["model"] == "anchored GMNPS"].iloc[0]
    ax.scatter(anchor["nps_preservation_spearman"], anchor["personalized_residual_spearman"], s=130, color=PALETTE["blue"], edgecolor="black", label="anchored GMNPS")
    unanchored = bench[bench["model"] == "unanchored microbiome score"].iloc[0]
    ax.scatter(unanchored["nps_preservation_spearman"], unanchored["personalized_residual_spearman"], s=100, color=PALETTE["red_light"], edgecolor="black", label="unanchored")
    ax.set_xlabel("NPS preservation")
    ax.set_ylabel("Residual recovery")
    ax.set_title("c  Preservation-personalization trade-off", loc="left", fontweight="bold")
    ax.set_xlim(0.90, 1.003)
    ax.set_ylim(0, 0.45)
    ax.annotate(
        "FCS2 only:\nresidual NA",
        xy=(1.0, 0.02),
        xytext=(0.965, 0.08),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": PALETTE["dark"]},
        fontsize=9,
    )
    ax.legend(fontsize=9)

    fig.suptitle("Figure 4. Synthetic digital-gut-twin benchmark supports the anchored calibration trade-off.", y=1.03, fontsize=14)
    fig.tight_layout(pad=1.4)
    save_figure(fig, out_dir, "figure_4_benchmark")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create GMNPS main figure drafts")
    parser.add_argument("--synthetic-dir", default="outputs/nature_food_article_synthetic")
    parser.add_argument("--fcs-audit", default="outputs/fcs2/table_s5_audit.json")
    parser.add_argument("--output-dir", default="outputs/manuscript_figures")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_style()
    synthetic = Path(args.synthetic_dir)
    out = Path(args.output_dir)
    figure_framework(out)
    figure_preservation(
        out,
        Path(args.fcs_audit),
        synthetic / "tables" / "individual_food_scores.csv",
        synthetic / "tables" / "nps_preservation_metrics.json",
    )
    figure_heterogeneity(
        out,
        synthetic / "tables" / "individual_food_scores.csv",
        synthetic / "figure_source_data" / "food_group_heterogeneity.csv",
        synthetic / "tables" / "food_summary.csv",
    )
    figure_benchmark(out, synthetic / "figure_source_data" / "synthetic_benchmark.csv")


if __name__ == "__main__":
    main()
