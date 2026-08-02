#!/usr/bin/env python3
"""Build Nature-style source data tables from reproduced GMNPS outputs.

The script derives compact, figure-ready tables from existing analysis outputs.
It does not simulate missing empirical data. Synthetic repeated-seed outputs are
generated only for the digital-gut-twin benchmark where ground truth is defined
by the simulator.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gmnps.scoring.masks import (  # noqa: E402
    EXPERT_REVISED_LIPID,
    EXPERT_REVISED_MAC,
    PRIMARY_EXCLUDED_FROM_CHANNELS,
)
from gmnps.validation.synthetic_twin import (  # noqa: E402
    run_delta_cap_sensitivity,
    run_design_ablation,
    run_synthetic_benchmark,
    simulate_synthetic_twin,
)


def write_json(path: Path, payload: dict[str, object]) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    path.write_text(json.dumps(clean(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    values = pvalues.astype(float).to_numpy()
    order = np.argsort(values)
    ranked = values[order]
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        running = min(running, ranked[idx] * n / (idx + 1))
        adjusted[idx] = running
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0, 1)
    return pd.Series(out, index=pvalues.index)


def copy_core_sources(input_root: Path, out_dir: Path) -> None:
    mapping = {
        input_root / "outputs/gmnps_v4/tables/S_food_summary_v4.csv": "figure2_3_food_summary_v4.csv",
        input_root / "outputs/gmnps_v4/validation/food_group_summary.csv": "figure2_3_food_group_summary_v4.csv",
        input_root / "outputs/gmnps_v4/validation/category_transition.csv": "figure2_category_transition_v4.csv",
        input_root / "outputs/gmnps_v4/validation/nps_preservation_metrics.json": "figure2_nps_preservation_metrics.json",
        input_root / "outputs/gmnps_v4/validation/v3_vs_v4_summary.json": "figure2_v3_vs_v4_summary.json",
        input_root / "outputs/gmnps_v4_simulation/figure_source_data/synthetic_benchmark.csv": "figure4_synthetic_benchmark.csv",
        input_root / "outputs/gmnps_v4_simulation/figure_source_data/design_ablation.csv": "figure4_design_ablation.csv",
        input_root / "outputs/gmnps_v4_simulation/figure_source_data/delta_cap_sensitivity.csv": "figure4_delta_cap_sensitivity.csv",
        input_root / "outputs/external_validation/gmrepo_v2_gmnps_auroc.csv": "figure4_gmrepo_auroc.csv",
    }
    for src, dest in mapping.items():
        if src.suffix == ".json":
            write_json(out_dir / dest, json.loads(src.read_text(encoding="utf-8")))
        else:
            pd.read_csv(src).to_csv(out_dir / dest, index=False)


def build_mask_matrix(out_dir: Path) -> None:
    rows = []
    for nutrient in sorted(EXPERT_REVISED_MAC):
        rows.append({"nutrient": nutrient, "channel": "microbiota-accessible and plant matrix", "status": "primary"})
    for nutrient in sorted(EXPERT_REVISED_LIPID):
        rows.append({"nutrient": nutrient, "channel": "lipid and bile-acid related", "status": "primary"})
    for nutrient in sorted(PRIMARY_EXCLUDED_FROM_CHANNELS):
        rows.append({"nutrient": nutrient, "channel": "not in primary channel", "status": "expert-revised exclusion"})
    pd.DataFrame(rows).to_csv(out_dir / "figure1_mask_matrix.csv", index=False)

    counts = (
        pd.DataFrame(rows)
        .groupby(["channel", "status"], as_index=False)
        .size()
        .rename(columns={"size": "n_nutrients"})
    )
    counts.to_csv(out_dir / "figure1_mask_counts.csv", index=False)


def matrix_columns(matrix_dir: Path, prefix: str, food_ids: list[str]) -> pd.DataFrame:
    selected = []
    target = set(map(str, food_ids))
    for path in sorted(matrix_dir.glob(f"{prefix}_individual_x_food_v4_chunk_*.parquet")):
        columns = pq.read_schema(path).names
        keep = [col for col in columns if str(col) in target]
        if keep:
            selected.append(pd.read_parquet(path, columns=keep))
    if not selected:
        raise ValueError(f"No requested food ids found for {prefix}: {food_ids[:5]}")
    return pd.concat(selected, axis=1)


def build_individual_examples(input_root: Path, out_dir: Path) -> None:
    summary = pd.read_csv(input_root / "outputs/gmnps_v4/tables/S_food_summary_v4.csv")
    group_order = [
        "3000_Vegetables",
        "2000_Fruit",
        "4000_LegNut",
        "1000_Grains",
        "6000_Dairy",
        "5000_MPE",
        "5800_Seafood",
        "7000_Fats Oils",
        "8000_Mixed",
        "9000_SavorySweet",
        "0000_Beverages",
        "8600_SauceCondiment",
    ]
    ranked = summary.sort_values("GMNPS_delta_var", ascending=False)
    top_rows = []
    for group in group_order:
        candidate = ranked.loc[ranked["food_group"].eq(group)]
        if not candidate.empty:
            top_rows.append(candidate.iloc[0])
    top = pd.DataFrame(top_rows).head(12).copy()
    food_ids = top["food_id"].astype(str).tolist()
    matrix_dir = input_root / "outputs/gmnps_v4/matrices"
    deltas = matrix_columns(matrix_dir, "S_delta", food_ids)
    scores = matrix_columns(matrix_dir, "S_score", food_ids)

    metadata = top.set_index(top["food_id"].astype(str))
    dist_rows = []
    for food_id in food_ids[:4]:
        delta = deltas[food_id]
        score = scores[food_id]
        qs = delta.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        dist_rows.append(
            {
                "food_id": food_id,
                "food_name": metadata.loc[food_id, "food_name"],
                "food_group": metadata.loc[food_id, "food_group"],
                "dominant_channel": metadata.loc[food_id, "dominant_channel"],
                "FCS2": metadata.loc[food_id, "FCS2"],
                "GMNPS_mean": metadata.loc[food_id, "GMNPS_mean"],
                "score_p05": score.quantile(0.05),
                "score_p50": score.quantile(0.50),
                "score_p95": score.quantile(0.95),
                "delta_p05": qs.loc[0.05],
                "delta_p25": qs.loc[0.25],
                "delta_p50": qs.loc[0.50],
                "delta_p75": qs.loc[0.75],
                "delta_p95": qs.loc[0.95],
                "n_individuals": len(delta),
            }
        )
    pd.DataFrame(dist_rows).to_csv(out_dir / "figure1_individual_food_examples.csv", index=False)

    spread = deltas.std(axis=1).sort_values(ascending=False)
    individuals = spread.head(48).index.tolist()
    heat = deltas.loc[individuals, food_ids].copy()
    long = heat.reset_index(names="individual_id").melt(
        id_vars="individual_id", var_name="food_id", value_name="GMNPS_delta"
    )
    long["individual_order"] = long["individual_id"].map({v: i for i, v in enumerate(individuals)})
    long["food_order"] = long["food_id"].map({v: i for i, v in enumerate(food_ids)})
    long = long.merge(
        metadata[["food_name", "food_group", "FCS2", "GMNPS_mean", "dominant_channel"]],
        left_on="food_id",
        right_index=True,
        how="left",
    )
    long.to_csv(out_dir / "figure3_individual_food_delta_heatmap.csv", index=False)


def build_uncertainty_tables(input_root: Path, out_dir: Path) -> None:
    summary = pd.read_csv(input_root / "outputs/gmnps_v4/tables/S_food_summary_v4.csv")
    rng = np.random.default_rng(20260802)
    rows = []
    for seed in range(500):
        sample = summary.sample(n=len(summary), replace=True, random_state=int(rng.integers(0, 2**31 - 1)))
        rows.append(
            {
                "bootstrap": seed,
                "spearman_fcs2_gmnps_mean": sample["FCS2"].rank().corr(sample["GMNPS_mean"].rank()),
                "mean_delta_variance": sample["GMNPS_delta_var"].mean(),
                "mean_abs_population_shift": (sample["GMNPS_mean"] - sample["FCS2"]).abs().mean(),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "figure2_bootstrap_preservation.csv", index=False)

    group = pd.read_csv(input_root / "outputs/gmnps_v4/validation/food_group_summary.csv")
    group["mac_fraction"] = group["MAC_variance_mean"] / (group["MAC_variance_mean"] + group["LIPID_variance_mean"])
    group["lipid_fraction"] = 1 - group["mac_fraction"]
    group.to_csv(out_dir / "figure3_group_channel_fraction.csv", index=False)

    gmrepo = pd.read_csv(input_root / "outputs/external_validation/gmrepo_v2_gmnps_auroc.csv")
    gmrepo["mw_fdr_bh"] = bh_fdr(gmrepo["mw_p"])
    gmrepo.to_csv(out_dir / "figure4_gmrepo_fdr.csv", index=False)


def build_repeated_synthetic(out_dir: Path, seeds: int) -> None:
    bench_rows = []
    ablation_rows = []
    cap_rows = []
    for seed in range(seeds):
        bundle = simulate_synthetic_twin(n_individuals=160, n_foods=100, seed=42 + seed)
        bench = run_synthetic_benchmark(bundle=bundle, seed=42 + seed)
        bench["seed"] = 42 + seed
        bench_rows.append(bench)
        ablation = run_design_ablation(bundle=bundle, seed=42 + seed)
        ablation["seed"] = 42 + seed
        ablation_rows.append(ablation)
        caps = run_delta_cap_sensitivity(bundle=bundle, seed=42 + seed)
        caps["seed"] = 42 + seed
        cap_rows.append(caps)
    pd.concat(bench_rows, ignore_index=True).to_csv(out_dir / "figure4_synthetic_repeated_seeds.csv", index=False)
    pd.concat(ablation_rows, ignore_index=True).to_csv(out_dir / "figure4_ablation_repeated_seeds.csv", index=False)
    pd.concat(cap_rows, ignore_index=True).to_csv(out_dir / "figure4_cap_repeated_seeds.csv", index=False)


def build_manifest(out_dir: Path) -> None:
    rows = []
    for path in sorted(out_dir.iterdir()):
        if path.suffix.lower() in {".csv", ".json"}:
            rows.append({"file": path.name, "bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(out_dir / "source_data_manifest.csv", index=False)
    (out_dir / "README.md").write_text(
        "# GMNPS Nature source data v2\n\n"
        "Tables in this directory are generated from reproduced GMNPS scoring, "
        "external validation and digital-gut-twin benchmark outputs. Participant-"
        "level matrices are reduced to compact, figure-ready extracts.\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build GMNPS Nature figure source data v2")
    parser.add_argument("--input-root", default=".")
    parser.add_argument("--output-dir", default="outputs/nature_source_data_v2")
    parser.add_argument("--synthetic-seeds", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_core_sources(input_root, out_dir)
    build_mask_matrix(out_dir)
    build_individual_examples(input_root, out_dir)
    build_uncertainty_tables(input_root, out_dir)
    build_repeated_synthetic(out_dir, args.synthetic_seeds)
    build_manifest(out_dir)


if __name__ == "__main__":
    main()
