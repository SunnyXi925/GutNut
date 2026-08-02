#!/usr/bin/env python3
"""Create manuscript-ready figure drafts from GMNPS v4 outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "ink": "#222222",
    "muted": "#6A6A6A",
    "line": "#D6D6D6",
    "blue": "#2E6FAD",
    "blue_light": "#CFE1F2",
    "green": "#3F8F6B",
    "green_light": "#CDE8D8",
    "red": "#B44E4B",
    "red_light": "#F1CECB",
    "gold": "#C9972B",
    "gold_light": "#F2E2B8",
    "violet": "#7B6AB0",
}


GROUP_ORDER = [
    "0000_Beverages",
    "1000_Grains",
    "2000_Fruit",
    "3000_Vegetables",
    "4000_LegNut",
    "5000_MPE",
    "5800_Seafood",
    "6000_Dairy",
    "7000_Fats Oils",
    "8000_Mixed",
    "8600_SauceCondiment",
    "9000_SavorySweet",
]


GROUP_LABELS = {
    "0000_Beverages": "Beverages",
    "1000_Grains": "Grains",
    "2000_Fruit": "Fruit",
    "3000_Vegetables": "Vegetables",
    "4000_LegNut": "Legumes/nuts",
    "5000_MPE": "Meat/poultry/eggs",
    "5800_Seafood": "Seafood",
    "6000_Dairy": "Dairy",
    "7000_Fats Oils": "Fats/oils",
    "8000_Mixed": "Mixed dishes",
    "8600_SauceCondiment": "Sauces/condiments",
    "9000_SavorySweet": "Savory/sweet",
}


def ordered_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Return all known groups, preserving observed extras after the canonical order."""

    observed = list(frame["food_group"].dropna().unique())
    order = [group for group in GROUP_ORDER if group in observed]
    order.extend(sorted(group for group in observed if group not in order))
    return frame.set_index("food_group").reindex(order).dropna(subset=["n_foods"])


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top")


def short_food_name(value: str, limit: int = 42) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "."


def figure_1_framework(out_dir: Path) -> None:
    fig = plt.figure(figsize=(7.2, 4.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], width_ratios=[1.05, 1.05, 1.1])

    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    boxes = [
        (0.02, 0.54, 0.24, 0.27, "Population NPS prior", "food-level ranking\n1-100 scale", PALETTE["blue_light"]),
        (0.38, 0.54, 0.24, 0.27, "Personalized deviation", "microbiome-informed\nbounded response", PALETTE["green_light"]),
        (0.74, 0.54, 0.24, 0.27, "GMNPS score", "prior plus centered\nindividual calibration", PALETTE["gold_light"]),
    ]
    for x, y, w, h, title, body, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=PALETTE["ink"], linewidth=0.9))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontweight="bold")
        ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", color=PALETTE["muted"])
    ax.annotate("", xy=(0.36, 0.675), xytext=(0.27, 0.675), arrowprops={"arrowstyle": "->", "lw": 1.3})
    ax.annotate("", xy=(0.72, 0.675), xytext=(0.63, 0.675), arrowprops={"arrowstyle": "->", "lw": 1.3})
    ax.text(0.5, 0.18, r"$\mathrm{GMNPS}_{ij}=\mathrm{clip}_{1,100}(\mathrm{NPS}_{j}+D_{ij})$",
            ha="center", va="center", fontsize=10)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[1, 0])
    z = np.linspace(-4, 4, 300)
    d = 12 * np.tanh(z / 2)
    ax.plot(z, d, color=PALETTE["blue"], lw=1.8)
    ax.axhline(12, color=PALETTE["red"], lw=0.8, ls=":")
    ax.axhline(-12, color=PALETTE["red"], lw=0.8, ls=":")
    ax.axhline(0, color=PALETTE["line"], lw=0.8)
    ax.set_xlabel("Robust z-score")
    ax.set_ylabel("Deviation")
    ax.set_title("Bounded transform")
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    channels = ["MAC", "LIPID", "OTHER"]
    values = [12, 14, 4]
    colors = [PALETTE["green"], PALETTE["red_light"], "#E5E5E5"]
    ax.bar(channels, values, color=colors, edgecolor=PALETTE["ink"], linewidth=0.6)
    ax.set_ylabel("Nutrients")
    ax.set_title("Expert-revised masks")
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 2])
    rng = np.random.default_rng(12)
    baseline = 72
    values = np.clip(baseline + 12 * np.tanh(rng.normal(0, 1.1, 150) / 2), 1, 100)
    y = rng.normal(0, 0.04, size=len(values))
    ax.scatter(values, y, s=8, alpha=0.42, color=PALETTE["blue"])
    ax.axvline(baseline, color=PALETTE["ink"], lw=1.2)
    ax.set_yticks([])
    ax.set_xlim(50, 95)
    ax.set_xlabel("Individual scores for one food")
    ax.set_title("Heterogeneity around the prior")
    panel_label(ax, "d")

    fig.suptitle("Personalized calibration extends nutrient profiling without replacing the population prior",
                 y=0.99, fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, out_dir, "figure_1_framework")


def figure_2_preservation(v4_dir: Path, out_dir: Path) -> None:
    summary = pd.read_csv(v4_dir / "tables" / "S_food_summary_v4.csv")
    group = pd.read_csv(v4_dir / "validation" / "food_group_summary.csv")
    transition = pd.read_csv(v4_dir / "validation" / "category_transition.csv", index_col=0)
    metrics = read_json(v4_dir / "validation" / "nps_preservation_metrics.json")
    v3v4 = read_json(v4_dir / "validation" / "v3_vs_v4_summary.json")

    fig = plt.figure(figsize=(7.2, 6.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(summary["FCS2"], summary["GMNPS_mean"], s=5, alpha=0.28, color=PALETTE["blue"], linewidths=0)
    ax.plot([1, 100], [1, 100], color=PALETTE["ink"], lw=0.8, ls="--")
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Baseline NPS prior")
    ax.set_ylabel("Population-mean GMNPS")
    ax.text(
        5,
        92,
        f"n = {int(metrics['n_foods']):,}\nSpearman = {metrics['spearman_fcs2_gmnps_mean']:.6f}",
        fontsize=7,
        va="top",
    )
    ax.set_title("Food-level ranking is preserved")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    matrix = transition.reindex(index=["minimize", "moderate", "encourage"],
                                columns=["minimize", "moderate", "encourage"], fill_value=0)
    row_norm = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    im = ax.imshow(row_norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    labels = ["Minimize", "Moderate", "Encourage"]
    ax.set_xticks(range(3), labels, rotation=35, ha="right")
    ax.set_yticks(range(3), labels)
    for r in range(3):
        for c in range(3):
            ax.text(c, r, f"{int(matrix.iloc[r, c]):,}", ha="center", va="center", fontsize=7)
    ax.set_xlabel("GMNPS category")
    ax.set_ylabel("Baseline category")
    ax.text(0.5, -0.33, f"Category shifts: {metrics['category_shift_count']} ({metrics['category_shift_fraction']:.3%})",
            transform=ax.transAxes, ha="center", va="top")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    ax.set_title("Population category changes are rare")
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    group = ordered_groups(group)
    x = np.arange(len(group))
    ax.plot(x, group["FCS2_mean"], marker="o", ms=3.5, lw=1.2, color=PALETTE["ink"], label="Baseline")
    ax.plot(x, group["GMNPS_mean"], marker="o", ms=3.5, lw=1.2, color=PALETTE["blue"], label="GMNPS mean")
    ax.set_xticks(x, [GROUP_LABELS.get(g, g) for g in group.index], rotation=45, ha="right")
    ax.set_ylabel("Mean score")
    ax.set_title("Food-group ordering remains aligned")
    ax.legend(loc="upper left")
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    labels = ["v3 old mask", "v4 revised"]
    rho = [v3v4["v3_spearman_fcs2_total_mean"], v3v4["v4_spearman_fcs2_gmnps_mean"]]
    variance = [v3v4["v3_total_var_mean"], v3v4["v4_delta_var_mean"]]
    x = np.arange(2)
    ax.bar(x - 0.18, rho, width=0.34, color=PALETTE["blue"], edgecolor=PALETTE["ink"], linewidth=0.6, label="NPS rho")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, variance, width=0.34, color=PALETTE["gold_light"], edgecolor=PALETTE["ink"], linewidth=0.6, label="Mean variance")
    ax.set_ylim(0, 1.05)
    ax2.set_ylim(0, max(variance) * 1.25)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Spearman")
    ax2.set_ylabel("Mean variance")
    ax.set_title("Revised mask improves anchoring")
    panel_label(ax, "d")
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, loc="upper center", fontsize=6)

    fig.suptitle("Anchored calibration preserves population-level NPS consensus", y=0.99, fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_figure(fig, out_dir, "figure_2_preservation")


def figure_3_heterogeneity(v4_dir: Path, out_dir: Path) -> None:
    summary = pd.read_csv(v4_dir / "tables" / "S_food_summary_v4.csv")
    group = pd.read_csv(v4_dir / "validation" / "food_group_summary.csv")
    group = ordered_groups(group)

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(group))
    ax.bar(x, group["delta_var_mean"], color=PALETTE["blue_light"], edgecolor=PALETTE["ink"], linewidth=0.5)
    ax.set_xticks(x, [GROUP_LABELS.get(g, g) for g in group.index], rotation=45, ha="right")
    ax.set_ylabel("Mean variance")
    ax.set_title("Bounded individual variation by food group")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x - 0.18, group["MAC_variance_mean"], width=0.36, color=PALETTE["green"], edgecolor=PALETTE["ink"], linewidth=0.5, label="MAC")
    ax.bar(x + 0.18, group["LIPID_variance_mean"], width=0.36, color=PALETTE["red_light"], edgecolor=PALETTE["ink"], linewidth=0.5, label="LIPID")
    ax.set_xticks(x, [GROUP_LABELS.get(g, g) for g in group.index], rotation=45, ha="right")
    ax.set_ylabel("Mean channel variance")
    ax.set_title("Channel attribution differs across groups")
    ax.legend(loc="upper left")
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    colors = np.where(summary["dominant_channel"].eq("MAC"), PALETTE["green"], PALETTE["red_light"])
    ax.scatter(summary["MAC_variance"], summary["LIPID_variance"], s=6, alpha=0.35, color=colors, linewidths=0)
    lim = max(summary["MAC_variance"].quantile(0.995), summary["LIPID_variance"].quantile(0.995))
    ax.plot([0, lim], [0, lim], color=PALETTE["ink"], lw=0.7, ls=":")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("MAC variance")
    ax.set_ylabel("LIPID variance")
    ax.set_title("Food-level channel dominance")
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    top = summary.sort_values("GMNPS_delta_var", ascending=False).head(10).iloc[::-1]
    y = np.arange(len(top))
    ax.hlines(y, top["GMNPS_p05"], top["GMNPS_p95"], color=PALETTE["blue"], lw=2.0)
    ax.scatter(top["FCS2"], y, color=PALETTE["ink"], s=12, label="Baseline")
    ax.scatter(top["GMNPS_mean"], y, color=PALETTE["blue"], s=12, label="GMNPS mean")
    ax.set_yticks(y, [short_food_name(v, 34) for v in top["food_name"]])
    ax.set_xlabel("Score")
    ax.set_title("High-heterogeneity foods remain anchored")
    ax.legend(loc="lower right", fontsize=6)
    panel_label(ax, "d")

    fig.suptitle("GMNPS expresses bounded and channel-attributed personalized heterogeneity", y=0.99,
                 fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_figure(fig, out_dir, "figure_3_heterogeneity")


def figure_4_validation(sim_dir: Path, external_dir: Path, out_dir: Path) -> None:
    bench = pd.read_csv(sim_dir / "figure_source_data" / "synthetic_benchmark.csv")
    design = pd.read_csv(sim_dir / "figure_source_data" / "design_ablation.csv")
    caps = pd.read_csv(sim_dir / "figure_source_data" / "delta_cap_sensitivity.csv")
    auroc = pd.read_csv(external_dir / "gmrepo_v2_gmnps_auroc.csv")

    fig = plt.figure(figsize=(7.2, 6.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    disease_order = ["Healthy_vs_T2D", "Healthy_vs_CAD", "Healthy_vs_CRC", "Healthy_vs_CD", "Healthy_vs_UC", "Healthy_vs_IBD"]
    forest = auroc.set_index("comparison").reindex(disease_order).dropna(subset=["auroc"]).iloc[::-1]
    y = np.arange(len(forest))
    colors = [PALETTE["blue"] if c in {"Healthy_vs_T2D", "Healthy_vs_CAD"} else "#BBBBBB" for c in forest.index]
    ax.hlines(y, forest["auroc_ci_lo"], forest["auroc_ci_hi"], color=colors, lw=1.8)
    ax.scatter(forest["auroc"], y, color=colors, s=20, zorder=3, edgecolor=PALETTE["ink"], linewidth=0.4)
    ax.axvline(0.5, color=PALETTE["ink"], lw=0.8, ls=":")
    ax.set_yticks(y, [v.replace("Healthy_vs_", "") for v in forest.index])
    ax.set_xlabel("AUROC versus healthy controls")
    ax.set_title("GMrepo cardiometabolic plausibility")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    order = ["FCS2 only", "unanchored microbiome score", "anchored GMNPS", "random microbiome", "shuffled microbiome"]
    b = bench.set_index("model").reindex(order).dropna(how="all").reset_index()
    y = np.arange(len(b))
    colors = [PALETTE["gold_light"], PALETTE["red_light"], PALETTE["blue"], "#C9C9C9", "#C9C9C9"]
    ax.barh(y, b["individual_response_spearman"], color=colors, edgecolor=PALETTE["ink"], linewidth=0.5)
    ax.set_yticks(y, [m.replace(" microbiome", "\nmicrobiome").replace(" score", "\nscore") for m in b["model"]])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Spearman")
    ax.set_title("Known-response recovery")
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(bench["nps_preservation_spearman"], bench["personalized_residual_spearman"],
               s=28, color="#BFBFBF", edgecolor=PALETTE["ink"], linewidth=0.5)
    for model, color in [("anchored GMNPS", PALETTE["blue"]), ("unanchored microbiome score", PALETTE["red"])]:
        row = bench.loc[bench["model"].eq(model)].iloc[0]
        ax.scatter(row["nps_preservation_spearman"], row["personalized_residual_spearman"],
                   s=54, color=color, edgecolor=PALETTE["ink"], linewidth=0.6, label=model)
    ax.annotate("FCS2-only residual\nnot estimable", xy=(1.0, 0.02), xytext=(0.965, 0.12),
                arrowprops={"arrowstyle": "->", "lw": 0.8, "color": PALETTE["ink"]}, fontsize=6)
    ax.set_xlim(0.925, 1.004)
    ax.set_ylim(0, 0.43)
    ax.set_xlabel("NPS preservation")
    ax.set_ylabel("Personalized residual recovery")
    ax.set_title("Preservation-personalization trade-off")
    ax.legend(loc="upper left", fontsize=6)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    d_order = ["FCS2 only", "anchored GMNPS", "no centering", "unanchored microbiome score", "random microbiome", "shuffled microbiome"]
    d = design.set_index("model").reindex(d_order).reset_index()
    x = np.arange(len(d))
    residual = d["personalized_residual_spearman"].fillna(0)
    preservation = d["nps_preservation_spearman"]
    ax.bar(x - 0.17, residual, width=0.34, color=PALETTE["blue"], edgecolor=PALETTE["ink"], linewidth=0.5, label="Residual")
    ax.bar(x + 0.17, preservation, width=0.34, color=PALETTE["green_light"], edgecolor=PALETTE["ink"], linewidth=0.5, label="Preservation")
    ax.text(-0.17, 0.04, "NA", ha="center", va="bottom", rotation=90, fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, [str(v).replace(" microbiome", "\nmicrobiome").replace(" score", "\nscore") for v in d["model"]],
                  rotation=35, ha="right")
    ax.set_ylabel("Spearman")
    ax.set_title("Ablation and cap sensitivity")
    ax.legend(loc="upper left", fontsize=6)
    inset = ax.inset_axes([0.54, 0.48, 0.42, 0.40])
    inset.plot(caps["delta_cap"], caps["personalized_residual_spearman"], marker="o", ms=3,
               color=PALETTE["blue"], lw=1.0)
    inset.plot(caps["delta_cap"], caps["nps_preservation_spearman"], marker="o", ms=3,
               color=PALETTE["green"], lw=1.0)
    inset.set_xlabel("Cap", fontsize=6)
    inset.set_ylabel("Spearman", fontsize=6)
    inset.set_ylim(0.35, 1.02)
    inset.tick_params(labelsize=6)
    panel_label(ax, "d")

    fig.suptitle("External plausibility and digital-gut-twin benchmarks support bounded calibration",
                 y=0.99, fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_figure(fig, out_dir, "figure_4_validation")


def export_figure_source_summary(v4_dir: Path, sim_dir: Path, external_dir: Path, out_dir: Path) -> None:
    source_dir = out_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    for src, dest in [
        (v4_dir / "tables" / "S_food_summary_v4.csv", source_dir / "figure2_3_food_summary_v4.csv"),
        (v4_dir / "validation" / "food_group_summary.csv", source_dir / "figure2_3_food_group_summary_v4.csv"),
        (v4_dir / "validation" / "category_transition.csv", source_dir / "figure2_category_transition_v4.csv"),
        (sim_dir / "figure_source_data" / "synthetic_benchmark.csv", source_dir / "figure4_synthetic_benchmark.csv"),
        (sim_dir / "figure_source_data" / "design_ablation.csv", source_dir / "figure4_design_ablation.csv"),
        (sim_dir / "figure_source_data" / "delta_cap_sensitivity.csv", source_dir / "figure4_delta_cap_sensitivity.csv"),
        (external_dir / "gmrepo_v2_gmnps_auroc.csv", source_dir / "figure4_gmrepo_auroc.csv"),
    ]:
        pd.read_csv(src).to_csv(dest, index=False)
    (source_dir / "figure2_nps_preservation_metrics.json").write_text(
        (v4_dir / "validation" / "nps_preservation_metrics.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (source_dir / "figure2_v3_vs_v4_summary.json").write_text(
        (v4_dir / "validation" / "v3_vs_v4_summary.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create GMNPS v4 main figures")
    parser.add_argument("--v4-dir", default="outputs/gmnps_v4")
    parser.add_argument("--simulation-dir", default="outputs/gmnps_v4_simulation")
    parser.add_argument("--external-dir", default="outputs/external_validation")
    parser.add_argument("--output-dir", default="outputs/main_figures_v4")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_style()
    v4_dir = Path(args.v4_dir)
    sim_dir = Path(args.simulation_dir)
    external_dir = Path(args.external_dir)
    out_dir = Path(args.output_dir)
    figure_1_framework(out_dir)
    figure_2_preservation(v4_dir, out_dir)
    figure_3_heterogeneity(v4_dir, out_dir)
    figure_4_validation(sim_dir, external_dir, out_dir)
    export_figure_source_summary(v4_dir, sim_dir, external_dir, out_dir)


if __name__ == "__main__":
    main()
