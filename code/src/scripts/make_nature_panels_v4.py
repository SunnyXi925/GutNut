#!/usr/bin/env python3
"""Panel-first Nature-style figure builder for the GMNPS manuscript.

Each panel is rendered as an independent file before figure assembly. The final
figure pages are composed from the panel previews and receive panel labels only
at the assembly step.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PALETTE = {
    "ink": "#222222",
    "muted": "#686868",
    "hair": "#D8D8D8",
    "pale": "#F7F7F4",
    "blue": "#0F4D92",
    "blue2": "#3775BA",
    "blue_pale": "#DDEAF6",
    "green": "#3F8F6B",
    "green_pale": "#DDF3DE",
    "red": "#B64342",
    "red_pale": "#F6CFCB",
    "gold": "#C9972B",
    "gold_pale": "#F3E4B8",
    "grey": "#C9C9C9",
    "grey2": "#767676",
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
    "5000_MPE": "Meat/eggs",
    "5800_Seafood": "Seafood",
    "6000_Dairy": "Dairy",
    "7000_Fats Oils": "Fats/oils",
    "8000_Mixed": "Mixed",
    "8600_SauceCondiment": "Sauces",
    "9000_SavorySweet": "Savory/sweet",
}


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
            "axes.linewidth": 0.72,
            "axes.edgecolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_sources(source_dir: Path) -> dict[str, pd.DataFrame | dict[str, object]]:
    def csv(name: str) -> pd.DataFrame:
        return pd.read_csv(source_dir / name)

    def js(name: str) -> dict[str, object]:
        return json.loads((source_dir / name).read_text(encoding="utf-8"))

    return {
        "food": csv("figure2_3_food_summary_v4.csv"),
        "group": csv("figure2_3_food_group_summary_v4.csv"),
        "transition": csv("figure2_category_transition_v4.csv"),
        "metrics": js("figure2_nps_preservation_metrics.json"),
        "boot": csv("figure2_bootstrap_preservation.csv"),
        "mask_counts": csv("figure1_mask_counts.csv"),
        "examples": csv("figure1_individual_food_examples.csv"),
        "heat": csv("figure3_individual_food_delta_heatmap.csv"),
        "channel_fraction": csv("figure3_group_channel_fraction.csv"),
        "gmrepo": csv("figure4_gmrepo_fdr.csv"),
        "benchmark": csv("figure4_synthetic_benchmark.csv"),
        "synthetic_repeated": csv("figure4_synthetic_repeated_seeds.csv"),
        "ablation_repeated": csv("figure4_ablation_repeated_seeds.csv"),
        "cap_repeated": csv("figure4_cap_repeated_seeds.csv"),
    }


def save_panel(fig: plt.Figure, panel_dir: Path, name: str) -> None:
    panel_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(panel_dir / f"{name}.svg", bbox_inches="tight")
    fig.savefig(panel_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(panel_dir / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def short(text: object, n: int = 30) -> str:
    value = str(text)
    return value if len(value) <= n else value[: n - 1] + "."


def ordered_groups(group: pd.DataFrame) -> pd.DataFrame:
    observed = list(group["food_group"].dropna().unique())
    order = [g for g in GROUP_ORDER if g in observed] + [g for g in sorted(observed) if g not in GROUP_ORDER]
    return group.set_index("food_group").reindex(order).dropna(subset=["n_foods"])


def panel_1a_concept(panel_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.set_axis_off()
    x = np.linspace(0.06, 0.94, 220)
    prior = 0.20 + 0.58 / (1 + np.exp(-8.2 * (x - 0.50)))
    cap = 0.12 + 0.015 * np.sin(4 * np.pi * x)
    upper = prior + cap
    lower = prior - cap
    ax.fill_between(x, lower, upper, color=PALETTE["blue_pale"], alpha=0.90, transform=ax.transAxes, zorder=1)
    ax.plot(x, prior, color=PALETTE["blue"], lw=3.0, transform=ax.transAxes, zorder=3)

    rng = np.random.default_rng(19)
    xs = np.linspace(0.13, 0.87, 42) + rng.normal(0, 0.008, 42)
    offsets = rng.normal(0, 0.055, 42)
    offsets[::7] *= 1.9
    for xi, off in zip(xs, offsets):
        y0 = np.interp(xi, x, prior)
        yi = np.clip(y0 + off, np.interp(xi, x, lower) + 0.006, np.interp(xi, x, upper) - 0.006)
        ax.plot([xi, xi], [y0, yi], color=PALETTE["green"], lw=0.65, alpha=0.48, transform=ax.transAxes, zorder=2)
        ax.scatter([xi], [yi], s=13, color=PALETTE["green"], alpha=0.82, transform=ax.transAxes, zorder=4)

    # Abstract microbiome motifs: restrained, repeated geometry rather than cartoon bacteria.
    for cx, cy, scale in [(0.18, 0.80, 1.0), (0.25, 0.74, 0.72), (0.78, 0.22, 0.95), (0.86, 0.28, 0.68)]:
        theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ax.scatter(cx + 0.030 * scale * np.cos(theta), cy + 0.020 * scale * np.sin(theta),
                   s=8 * scale, color=PALETTE["green_pale"], edgecolor=PALETTE["green"],
                   linewidth=0.45, transform=ax.transAxes, zorder=2)
        ax.scatter(cx, cy, s=18 * scale, color=PALETTE["green"], transform=ax.transAxes, zorder=3)

    ax.text(0.08, 0.91, "retained population prior", color=PALETTE["blue"], weight="bold", transform=ax.transAxes)
    ax.text(0.55, 0.86, "bounded personalized\ncalibration", color=PALETTE["green"], weight="bold", transform=ax.transAxes)
    ax.text(0.08, 0.08, "same food-quality scale; person-specific deviations remain visible", color=PALETTE["muted"], transform=ax.transAxes)
    ax.annotate("", xy=(0.72, 0.62), xytext=(0.60, 0.76), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-", color=PALETTE["green"], lw=0.9))
    ax.annotate("", xy=(0.46, 0.49), xytext=(0.25, 0.49), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", color=PALETTE["ink"], lw=0.9))
    save_panel(fig, panel_dir, "fig1a_concept")


def panel_1b_transform(panel_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.8, 1.55))
    z = np.linspace(-4, 4, 400)
    y = 12 * np.tanh(z / 2)
    ax.fill_between(z, -12, 12, color=PALETTE["pale"], zorder=0)
    ax.plot(z, y, color=PALETTE["blue"], lw=2.1)
    ax.axhline(12, color=PALETTE["red"], lw=0.8, ls=":")
    ax.axhline(-12, color=PALETTE["red"], lw=0.8, ls=":")
    ax.axhline(0, color=PALETTE["hair"], lw=0.8)
    ax.set_xlabel("Individual response z-score")
    ax.set_ylabel("Deviation")
    ax.set_ylim(-13, 13)
    save_panel(fig, panel_dir, "fig1b_transform")


def panel_1c_masks(data: dict[str, object], panel_dir: Path) -> None:
    mask = data["mask_counts"].copy()
    label_map = {
        "microbiota-accessible and plant matrix": "Plant-matrix\nchannel",
        "lipid and bile-acid related": "Lipid-related\nchannel",
        "not in primary channel": "Excluded from\nprimary channel",
    }
    mask["label"] = mask["channel"].map(label_map).fillna(mask["channel"])
    mask = mask.sort_values("n_nutrients", ascending=True)
    fig, ax = plt.subplots(figsize=(2.7, 1.75))
    y = np.arange(len(mask))
    colors = [PALETTE["grey"] if "Excluded" in lab else PALETTE["green"] if "Plant" in lab else PALETTE["red_pale"] for lab in mask["label"]]
    ax.barh(y, mask["n_nutrients"], color=colors, edgecolor=PALETTE["ink"], linewidth=0.6)
    ax.set_yticks(y, mask["label"])
    ax.set_xlabel("Primary-channel nutrients")
    ax.set_xlim(0, 16)
    for yy, value in zip(y, mask["n_nutrients"]):
        ax.text(value + 0.45, yy, str(int(value)), va="center", fontsize=7)
    save_panel(fig, panel_dir, "fig1c_masks")


def panel_1d_envelopes(data: dict[str, object], panel_dir: Path) -> None:
    ex = data["examples"].copy()
    fig, ax = plt.subplots(figsize=(2.85, 1.85))
    y = np.arange(len(ex))[::-1]
    colors = np.where(ex["dominant_channel"].eq("MAC"), PALETTE["green"], PALETTE["red_pale"])
    ax.hlines(y, ex["score_p05"], ex["score_p95"], color=colors, lw=3.0)
    ax.scatter(ex["FCS2"], y, color=PALETTE["ink"], s=17, zorder=3, label="Prior")
    ax.scatter(ex["score_p50"], y, color="white", edgecolor=PALETTE["ink"], s=20, zorder=4, label="Median")
    ax.set_yticks(y, [short(v, 24) for v in ex["food_name"]])
    ax.set_xlim(0, 101)
    ax.set_xlabel("Score")
    ax.legend(loc="lower right", fontsize=6)
    save_panel(fig, panel_dir, "fig1d_envelopes")


def panel_2a_density(data: dict[str, object], panel_dir: Path) -> None:
    food = data["food"]
    metrics = data["metrics"]
    fig, ax = plt.subplots(figsize=(3.6, 2.45))
    hb = ax.hexbin(food["FCS2"], food["GMNPS_mean"], gridsize=42, extent=(0, 100, 0, 100), mincnt=1, cmap="Blues", linewidths=0)
    ax.plot([0, 100], [0, 100], color=PALETTE["ink"], lw=0.75, ls="--")
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Baseline prior")
    ax.set_ylabel("Population-mean GMNPS")
    ax.text(
        0.05,
        0.93,
        f"n = {int(metrics['n_foods']):,}\nSpearman = {metrics['spearman_fcs2_gmnps_mean']:.6f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", edgecolor=PALETTE["hair"], pad=3),
    )
    fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.03, label="Foods")
    save_panel(fig, panel_dir, "fig2a_density")


def panel_2b_transitions(data: dict[str, object], panel_dir: Path) -> None:
    transition = data["transition"].set_index("FCS2").reindex(["minimize", "moderate", "encourage"])
    cats = ["minimize", "moderate", "encourage"]
    fig, ax = plt.subplots(figsize=(2.45, 2.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 2.5)
    ax.set_axis_off()
    y0 = dict(zip(cats, [2, 1, 0]))
    for cat in cats:
        ax.text(0.05, y0[cat], cat.title(), ha="left", va="center", fontsize=7, color=PALETTE["muted"])
        ax.text(0.95, y0[cat], cat.title(), ha="right", va="center", fontsize=7, color=PALETTE["muted"])
    max_count = transition.to_numpy().max()
    for left in cats:
        for right in cats:
            count = int(transition.loc[left, right])
            if count == 0:
                continue
            color = PALETTE["blue"] if left == right else PALETTE["gold"]
            alpha = 0.24 if left == right else 0.78
            lw = 0.4 + 7.5 * count / max_count
            ax.plot([0.25, 0.75], [y0[left], y0[right]], color=color, alpha=alpha, lw=lw, solid_capstyle="round")
            if left != right:
                ax.text(0.50, (y0[left] + y0[right]) / 2 + 0.12, str(count), ha="center", fontsize=7, color=PALETTE["gold"])
    ax.text(0.05, 2.36, "Baseline", ha="left", weight="bold")
    ax.text(0.95, 2.36, "Calibrated mean", ha="right", weight="bold")
    save_panel(fig, panel_dir, "fig2b_transitions")


def panel_2c_groups(data: dict[str, object], panel_dir: Path) -> None:
    group = ordered_groups(data["group"])
    group = group.sort_values("FCS2_mean")
    y = np.arange(len(group))
    fig, ax = plt.subplots(figsize=(3.1, 2.65))
    ax.hlines(y, group["FCS2_mean"], group["GMNPS_mean"], color=PALETTE["hair"], lw=1.4)
    ax.scatter(group["FCS2_mean"], y, color=PALETTE["ink"], s=14, label="Baseline")
    ax.scatter(group["GMNPS_mean"], y, color=PALETTE["blue"], s=18, label="GMNPS mean")
    ax.set_yticks(y, [GROUP_LABELS.get(idx, idx) for idx in group.index])
    ax.set_xlabel("Mean score")
    ax.set_xlim(0, 90)
    ax.legend(loc="lower right", fontsize=6)
    save_panel(fig, panel_dir, "fig2c_groups")


def panel_2d_robust(data: dict[str, object], panel_dir: Path) -> None:
    boot = data["boot"]
    metrics = data["metrics"]
    vals = pd.DataFrame(
        {
            "label": ["criterion", "bootstrap 5th", "observed"],
            "rho": [0.90, boot["spearman_fcs2_gmnps_mean"].quantile(0.05), metrics["spearman_fcs2_gmnps_mean"]],
        }
    )
    fig, ax = plt.subplots(figsize=(2.1, 1.55))
    y = np.arange(len(vals))
    ax.hlines(y, 0.90, vals["rho"], color=[PALETTE["red_pale"], PALETTE["blue_pale"], PALETTE["blue"]], lw=4)
    ax.scatter(vals["rho"], y, color=[PALETTE["red"], PALETTE["blue2"], PALETTE["blue"]], edgecolor=PALETTE["ink"], linewidth=0.5, zorder=3, s=28)
    ax.set_yticks(y, vals["label"])
    ax.set_xlim(0.885, 1.004)
    ax.set_xlabel("Spearman")
    save_panel(fig, panel_dir, "fig2d_robustness")


def panel_3a_landscape(data: dict[str, object], panel_dir: Path) -> None:
    food = data["food"].copy()
    fig, ax = plt.subplots(figsize=(4.05, 2.7))
    mac = food["MAC_variance"].clip(upper=food["MAC_variance"].quantile(0.995))
    lipid = food["LIPID_variance"].clip(upper=food["LIPID_variance"].quantile(0.995))
    colors = np.where(food["dominant_channel"].eq("MAC"), PALETTE["green"], PALETTE["red_pale"])
    ax.scatter(mac, lipid, s=5, alpha=0.32, color=colors, linewidths=0)
    lim = float(max(mac.quantile(0.995), lipid.quantile(0.995)))
    ax.plot([0, lim], [0, lim], color=PALETTE["ink"], lw=0.75, ls=":")
    ax.text(lim * 0.12, lim * 0.86, "lipid-related\ndominance", color=PALETTE["red"], fontsize=7)
    ax.text(lim * 0.66, lim * 0.10, "plant-matrix\ndominance", color=PALETTE["green"], fontsize=7)
    ax.set_xlabel("Plant-matrix channel variance")
    ax.set_ylabel("Lipid-related channel variance")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    save_panel(fig, panel_dir, "fig3a_landscape")


def panel_3b_heatmap(data: dict[str, object], panel_dir: Path) -> None:
    heat = data["heat"]
    matrix = heat.pivot(index="individual_order", columns="food_order", values="GMNPS_delta").sort_index()
    fig, ax = plt.subplots(figsize=(2.6, 2.45))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-12, vmax=12, interpolation="nearest")
    foods = (
        heat[["food_order", "food_group"]]
        .drop_duplicates()
        .sort_values("food_order")["food_group"]
        .map(lambda v: GROUP_LABELS.get(v, v).split("/")[0])
        .tolist()
    )
    ax.set_xticks(range(len(foods)), foods, rotation=70, ha="right", fontsize=5)
    ax.set_yticks([])
    ax.set_xlabel("Representative foods")
    ax.set_ylabel("48 high-variance profiles")
    fig.colorbar(im, ax=ax, fraction=0.050, pad=0.03, label="Deviation")
    save_panel(fig, panel_dir, "fig3b_heatmap")


def panel_3c_channel_stack(data: dict[str, object], panel_dir: Path) -> None:
    group = ordered_groups(data["channel_fraction"])
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    x = np.arange(len(group))
    ax.bar(x, group["mac_fraction"], color=PALETTE["green"], edgecolor=PALETTE["ink"], linewidth=0.35, label="Plant-matrix")
    ax.bar(x, group["lipid_fraction"], bottom=group["mac_fraction"], color=PALETTE["red_pale"], edgecolor=PALETTE["ink"], linewidth=0.35, label="Lipid-related")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of channel variance")
    ax.set_xticks(x, [GROUP_LABELS.get(idx, idx) for idx in group.index], rotation=45, ha="right")
    ax.legend(loc="upper left", ncols=2, fontsize=6)
    save_panel(fig, panel_dir, "fig3c_channel_stack")


def panel_3d_examples(data: dict[str, object], panel_dir: Path) -> None:
    food = data["food"].sort_values("GMNPS_delta_var", ascending=False).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    y = np.arange(len(food))
    colors = np.where(food["dominant_channel"].eq("MAC"), PALETTE["green"], PALETTE["red_pale"])
    ax.hlines(y, food["GMNPS_p05"], food["GMNPS_p95"], color=colors, lw=2.6)
    ax.scatter(food["FCS2"], y, color=PALETTE["ink"], s=13, label="Prior")
    ax.scatter(food["GMNPS_mean"], y, color="white", edgecolor=PALETTE["ink"], s=17, label="Mean")
    ax.set_yticks(y, [short(v, 31) for v in food["food_name"]])
    ax.set_xlabel("Score")
    ax.set_xlim(0, 101)
    ax.legend(loc="lower right", fontsize=6)
    save_panel(fig, panel_dir, "fig3d_examples")


def panel_4a_pareto(data: dict[str, object], panel_dir: Path) -> None:
    bench = data["benchmark"].copy()
    fig, ax = plt.subplots(figsize=(4.55, 3.10))
    plot = bench.loc[bench["personalized_residual_defined"].fillna(False)].copy()
    colors = {
        "anchored GMNPS": PALETTE["blue"],
        "unanchored microbiome score": PALETTE["red"],
        "random microbiome": PALETTE["grey"],
        "shuffled microbiome": PALETTE["grey2"],
        "original mask": PALETTE["green_pale"],
        "expert-revised mask": PALETTE["blue_pale"],
    }
    ax.axvspan(0.90, 1.005, ymin=0.48, ymax=1.0, color=PALETTE["blue_pale"], alpha=0.45, zorder=0)
    ax.text(0.904, 0.426, "desired region:\npreserved prior +\nindividual residual", fontsize=7, color=PALETTE["blue"], va="top")
    for _, row in plot.iterrows():
        size = 92 if row["model"] == "anchored GMNPS" else 62
        ax.scatter(row["nps_preservation_spearman"], row["personalized_residual_spearman"], s=size, color=colors.get(row["model"], PALETTE["grey"]), edgecolor=PALETTE["ink"], linewidth=0.55, zorder=3)
    label_offsets = {
        "anchored GMNPS": (0.002, 0.012),
        "unanchored microbiome score": (0.002, 0.010),
        "random microbiome": (-0.022, 0.055),
        "shuffled microbiome": (-0.022, -0.028),
    }
    for model in ["anchored GMNPS", "unanchored microbiome score", "random microbiome", "shuffled microbiome"]:
        row = plot.loc[plot["model"].eq(model)].iloc[0]
        dx, dy = label_offsets[model]
        ax.text(
            row["nps_preservation_spearman"] + dx,
            row["personalized_residual_spearman"] + dy,
            model.replace("unanchored microbiome score", "unanchored\nmicrobiome score").replace("anchored GMNPS", "anchored GMNPS").replace("random microbiome", "random\nmicrobiome").replace("shuffled microbiome", "shuffled\nmicrobiome"),
            fontsize=6,
        )
    ax.axvline(0.90, color=PALETTE["hair"], lw=0.8, ls=":")
    ax.set_xlabel("Population-prior preservation")
    ax.set_ylabel("Personalized residual recovery")
    ax.set_xlim(0.90, 1.005)
    ax.set_ylim(-0.02, 0.45)
    save_panel(fig, panel_dir, "fig4a_pareto")


def panel_4b_repeated(data: dict[str, object], panel_dir: Path) -> None:
    rep = data["synthetic_repeated"]
    order = ["FCS2 only", "unanchored microbiome score", "anchored GMNPS", "random microbiome", "shuffled microbiome"]
    summary = (
        rep.loc[rep["model"].isin(order)]
        .groupby("model", as_index=False)
        .agg(mean=("individual_response_spearman", "mean"), sd=("individual_response_spearman", "std"))
        .set_index("model")
        .reindex(order)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(2.85, 2.2))
    y = np.arange(len(summary))
    colors = [PALETTE["gold_pale"], PALETTE["red_pale"], PALETTE["blue"], PALETTE["grey"], PALETTE["grey"]]
    ax.barh(y, summary["mean"], xerr=summary["sd"], color=colors, edgecolor=PALETTE["ink"], linewidth=0.5)
    ax.set_yticks(y, [m.replace("FCS2 only", "baseline only").replace(" microbiome", "\nmicrobiome").replace(" score", "\nscore") for m in summary["model"]])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Full response recovery")
    save_panel(fig, panel_dir, "fig4b_repeated")


def panel_4c_ablation(data: dict[str, object], panel_dir: Path) -> None:
    abl = data["ablation_repeated"].copy()
    order = ["FCS2 only", "anchored GMNPS", "no centering", "unanchored microbiome score", "random microbiome", "shuffled microbiome"]
    summary = (
        abl.groupby("model", as_index=False)
        .agg(residual=("personalized_residual_spearman", "mean"), preservation=("nps_preservation_spearman", "mean"))
        .set_index("model")
        .reindex(order)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(3.55, 2.45))
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary["residual"].fillna(0), width=0.35, color=PALETTE["blue"], edgecolor=PALETTE["ink"], linewidth=0.45, label="Residual")
    ax.bar(x + 0.18, summary["preservation"], width=0.35, color=PALETTE["green_pale"], edgecolor=PALETTE["ink"], linewidth=0.45, label="Prior")
    ax.text(-0.25, 0.045, "not\ndefined", rotation=0, fontsize=6, ha="center")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Spearman")
    ax.set_xticks(x, [m.replace("FCS2 only", "baseline").replace(" microbiome", "\nmicrobiome").replace(" score", "") for m in summary["model"]], rotation=45, ha="right", fontsize=6)
    ax.legend(loc="upper left", fontsize=6)
    save_panel(fig, panel_dir, "fig4c_ablation")


def panel_4d_gmrepo(data: dict[str, object], panel_dir: Path) -> None:
    gm = data["gmrepo"].copy()
    order = ["Healthy_vs_T2D", "Healthy_vs_CAD", "Healthy_vs_CRC", "Healthy_vs_CD", "Healthy_vs_UC", "Healthy_vs_IBD"]
    gm = gm.set_index("comparison").reindex(order).dropna(subset=["auroc"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(2.55, 2.25))
    y = np.arange(len(gm))
    colors = [PALETTE["blue"] if idx in {"Healthy_vs_T2D", "Healthy_vs_CAD"} else PALETTE["grey"] for idx in gm.index]
    ax.hlines(y, gm["auroc_ci_lo"], gm["auroc_ci_hi"], color=colors, lw=2)
    ax.scatter(gm["auroc"], y, color=colors, edgecolor=PALETTE["ink"], linewidth=0.45, s=24, zorder=3)
    ax.axvline(0.5, color=PALETTE["ink"], lw=0.8, ls=":")
    ax.set_yticks(y, [idx.replace("Healthy_vs_", "") for idx in gm.index])
    ax.set_xlabel("AUROC")
    ax.set_xlim(0.40, 0.77)
    save_panel(fig, panel_dir, "fig4d_gmrepo")


def panel_4e_caps(data: dict[str, object], panel_dir: Path) -> None:
    caps = data["cap_repeated"].copy()
    summary = caps.groupby("delta_cap", as_index=False).agg(
        residual=("personalized_residual_spearman", "mean"),
        residual_sd=("personalized_residual_spearman", "std"),
        preservation=("nps_preservation_spearman", "mean"),
        preservation_sd=("nps_preservation_spearman", "std"),
    )
    fig, ax = plt.subplots(figsize=(2.25, 1.75))
    ax.errorbar(summary["delta_cap"], summary["residual"], yerr=summary["residual_sd"], marker="o", color=PALETTE["blue"], lw=1.3, ms=4, label="Residual")
    ax.errorbar(summary["delta_cap"], summary["preservation"], yerr=summary["preservation_sd"], marker="o", color=PALETTE["green"], lw=1.3, ms=4, label="Prior")
    ax.set_ylim(0.32, 1.02)
    ax.set_xlabel("Deviation cap")
    ax.set_ylabel("Spearman")
    ax.legend(loc="center right", fontsize=6)
    save_panel(fig, panel_dir, "fig4e_caps")


FIGURE_LAYOUTS = {
    "figure_1_framework": {
        "size": (3600, 2200),
        "panels": [
            ("a", "fig1a_concept", (90, 70, 2280, 1280)),
            ("b", "fig1b_transform", (2550, 140, 850, 720)),
            ("c", "fig1c_masks", (2500, 1030, 900, 640)),
            ("d", "fig1d_envelopes", (210, 1510, 2180, 560)),
        ],
    },
    "figure_2_preservation": {
        "size": (3600, 2350),
        "panels": [
            ("a", "fig2a_density", (90, 80, 2020, 1180)),
            ("b", "fig2b_transitions", (2320, 100, 1100, 1080)),
            ("c", "fig2c_groups", (110, 1420, 2040, 850)),
            ("d", "fig2d_robustness", (2440, 1500, 920, 680)),
        ],
    },
    "figure_3_heterogeneity": {
        "size": (3600, 2460),
        "panels": [
            ("a", "fig3a_landscape", (90, 80, 2100, 1260)),
            ("b", "fig3b_heatmap", (2380, 80, 1080, 1260)),
            ("c", "fig3c_channel_stack", (120, 1510, 1850, 790)),
            ("d", "fig3d_examples", (2150, 1460, 1250, 870)),
        ],
    },
    "figure_4_validation": {
        "size": (3600, 2460),
        "panels": [
            ("a", "fig4a_pareto", (90, 80, 2060, 1350)),
            ("b", "fig4b_repeated", (2340, 130, 1110, 1040)),
            ("c", "fig4c_ablation", (90, 1580, 1460, 760)),
            ("d", "fig4d_gmrepo", (1760, 1570, 900, 760)),
            ("e", "fig4e_caps", (2840, 1660, 670, 620)),
        ],
    },
}


def paste_panel(canvas: Image.Image, panel_path: Path, box: tuple[int, int, int, int]) -> None:
    img = Image.open(panel_path).convert("RGBA")
    x, y, w, h = box
    img.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas.alpha_composite(img, (x, y))


def draw_label(draw: ImageDraw.ImageDraw, label: str, xy: tuple[int, int]) -> None:
    try:
        font = ImageFont.truetype("Arial Bold.ttf", 72)
    except OSError:
        font = ImageFont.load_default()
    draw.text(xy, label, fill=PALETTE["ink"], font=font)


def assemble_figures(panel_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for figure, spec in FIGURE_LAYOUTS.items():
        canvas = Image.new("RGBA", spec["size"], "white")
        draw = ImageDraw.Draw(canvas)
        for label, stem, box in spec["panels"]:
            paste_panel(canvas, panel_dir / f"{stem}.png", box)
            draw_label(draw, label, (box[0] - 78, box[1] - 18))
        rgb = canvas.convert("RGB")
        rgb.save(out_dir / f"{figure}.png", dpi=(600, 600))
        rgb.save(out_dir / f"{figure}.pdf", resolution=600)


AI_PROMPTS = {
    "Fig. 1a": """Create a Nature journal style scientific schematic for a paper on precision nutrition. The panel should be a wide conceptual landscape, not a flowchart. Show a smooth blue population food-quality prior ridge across foods, with a pale blue bounded ribbon around the ridge. Add many small muted-green individual points and short vertical deviation strokes within the ribbon to represent microbiome-informed personalized calibration. Include subtle abstract microbiome motifs as small geometric circular clusters near the ribbon, but do not draw cartoon bacteria, organs or food icons. White background, restrained blue-green palette, one quiet gold accent allowed, no rectangular process boxes, no logos, no numerical data and no fake axes. The visual message is: personalization occurs as bounded calibration around a retained population prior.""",
    "Fig. 1 alternative graphical abstract": """Create a high-end Nature Food graphical abstract showing nutrient profiling as a central food-quality landscape, with a stable population ridge and person-specific microbiome calibration contours around it. Use clean vector-like geometry, soft blue for the prior, muted green for microbiome/person-specific deviations, and a single gold accent for the bounded calibration layer. Avoid arrows-in-boxes, avoid marketing style, avoid photorealistic food images, avoid invented data labels. The output should look like a scientific concept panel that can sit above quantitative panels.""",
    "Fig. 4 digital gut twin schematic": """Create a restrained Nature-style methods schematic for a digital gut twin benchmark. Show three linked abstract layers: synthetic microbiome capacity, real-like food nutrient substrate space and known host-response surface. Use continuous surfaces, small points and alignment guides, not cartoon organs. Indicate that anchored calibration is evaluated against known ground truth, with negative controls kept in grey. White background, blue for population prior, green for personalized residual signal, one muted red for unanchored comparator. No fabricated numerical values, no logos and no clinical claims.""",
}


def write_ai_prompts(out_dir: Path) -> None:
    lines = ["# AI-ready schematic prompts for GMNPS figures\n\n"]
    lines.append("These prompts are for conceptual schematic panels only. They must not be used to create or alter quantitative data panels.\n\n")
    for name, prompt in AI_PROMPTS.items():
        lines.append(f"## {name}\n\n")
        lines.append(textwrap.fill(prompt, width=100, break_on_hyphens=False))
        lines.append("\n\n")
    (out_dir / "figure_ai_prompts.md").write_text("".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render GMNPS Nature figures panel by panel")
    parser.add_argument("--source-dir", default="manuscript/nature_food_submission/source_data")
    parser.add_argument("--panel-dir", default="outputs/nature_panels_v4")
    parser.add_argument("--output-dir", default="outputs/nature_figures_v4")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_style()
    source_dir = Path(args.source_dir)
    panel_dir = Path(args.panel_dir)
    out_dir = Path(args.output_dir)
    data = load_sources(source_dir)
    panel_1a_concept(panel_dir)
    panel_1b_transform(panel_dir)
    panel_1c_masks(data, panel_dir)
    panel_1d_envelopes(data, panel_dir)
    panel_2a_density(data, panel_dir)
    panel_2b_transitions(data, panel_dir)
    panel_2c_groups(data, panel_dir)
    panel_2d_robust(data, panel_dir)
    panel_3a_landscape(data, panel_dir)
    panel_3b_heatmap(data, panel_dir)
    panel_3c_channel_stack(data, panel_dir)
    panel_3d_examples(data, panel_dir)
    panel_4a_pareto(data, panel_dir)
    panel_4b_repeated(data, panel_dir)
    panel_4c_ablation(data, panel_dir)
    panel_4d_gmrepo(data, panel_dir)
    panel_4e_caps(data, panel_dir)
    assemble_figures(panel_dir, out_dir)
    write_ai_prompts(out_dir)


if __name__ == "__main__":
    main()
