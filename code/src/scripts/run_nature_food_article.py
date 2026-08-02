#!/usr/bin/env python3
"""Run article-facing GMNPS scoring and validation assets.

Examples:
    python scripts/run_nature_food_article.py score \
      --weights weights.csv \
      --nutrients food_nutrients.csv \
      --food-metadata food_metadata.csv \
      --output-dir outputs/nature_food_article

    python scripts/run_nature_food_article.py synthetic \
      --output-dir outputs/nature_food_article/synthetic
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gmnps.manuscript import export_article_bundle
from gmnps.scoring import AnchoredScoringConfig, audit_primary_mask, score_individual_foods
from gmnps.validation import heterogeneity_metrics, nps_preservation_metrics
from gmnps.validation.synthetic_twin import (
    run_delta_cap_sensitivity,
    run_design_ablation,
    run_synthetic_benchmark,
    simulate_synthetic_twin,
)


def _read_indexed_csv(path: str | Path, index_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if index_col is None:
        index_col = df.columns[0]
    if index_col not in df.columns:
        raise ValueError(f"{path} does not contain index column {index_col!r}")
    return df.set_index(index_col)


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    Path(path).write_text(
        json.dumps(clean(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def run_score(args: argparse.Namespace) -> None:
    weights = _read_indexed_csv(args.weights, args.individual_id_col)
    nutrients = _read_indexed_csv(args.nutrients, args.food_id_col)
    food_metadata = _read_indexed_csv(args.food_metadata, args.food_id_col)

    cfg = AnchoredScoringConfig(delta_cap=args.delta_cap, mask_version=args.mask_version)
    individual_food, food_summary, manifest = score_individual_foods(
        weights,
        nutrients,
        food_metadata,
        cfg,
    )
    preservation = nps_preservation_metrics(individual_food, min_rho=args.min_preservation_rho)
    heterogeneity = heterogeneity_metrics(individual_food)
    manifest["primary_mask_audit"] = audit_primary_mask(nutrients.columns)
    manifest["input_files"] = {
        "weights": str(args.weights),
        "nutrients": str(args.nutrients),
        "food_metadata": str(args.food_metadata),
    }

    paths = export_article_bundle(
        args.output_dir,
        individual_food,
        food_summary,
        manifest,
        preservation_metrics=preservation,
        heterogeneity=heterogeneity,
    )
    _write_json(Path(args.output_dir) / "run_score_report.json", {"paths": paths, "preservation": preservation})


def run_synthetic(args: argparse.Namespace) -> None:
    bundle = simulate_synthetic_twin(
        n_individuals=args.n_individuals,
        n_foods=args.n_foods,
        seed=args.seed,
    )
    individual_food, food_summary, manifest = score_individual_foods(
        bundle.weights,
        bundle.nutrients,
        bundle.food_metadata,
        AnchoredScoringConfig(delta_cap=args.delta_cap),
    )
    preservation = nps_preservation_metrics(individual_food, min_rho=args.min_preservation_rho)
    heterogeneity = heterogeneity_metrics(individual_food)
    benchmark = run_synthetic_benchmark(bundle=bundle, seed=args.seed)
    delta_cap_sensitivity = run_delta_cap_sensitivity(bundle=bundle, seed=args.seed)
    design_ablation = run_design_ablation(bundle=bundle, seed=args.seed)
    manifest["synthetic_twin"] = {
        "n_individuals": args.n_individuals,
        "n_foods": args.n_foods,
        "seed": args.seed,
        "ground_truth": "universal_food_quality + microbiome_conditioned_MAC_LIPID_effect + noise",
    }

    paths = export_article_bundle(
        args.output_dir,
        individual_food,
        food_summary,
        manifest,
        preservation_metrics=preservation,
        heterogeneity=heterogeneity,
        benchmark=benchmark,
    )
    figure_dir = Path(args.output_dir) / "figure_source_data"
    sensitivity_path = figure_dir / "delta_cap_sensitivity.csv"
    ablation_path = figure_dir / "design_ablation.csv"
    delta_cap_sensitivity.to_csv(sensitivity_path, index=False)
    design_ablation.to_csv(ablation_path, index=False)
    paths["delta_cap_sensitivity_csv"] = str(sensitivity_path)
    paths["design_ablation_csv"] = str(ablation_path)
    _write_json(
        Path(args.output_dir) / "run_synthetic_report.json",
        {
            "paths": paths,
            "preservation": preservation,
            "benchmark": benchmark.to_dict(orient="records"),
            "delta_cap_sensitivity": delta_cap_sensitivity.to_dict(orient="records"),
            "design_ablation": design_ablation.to_dict(orient="records"),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GMNPS Nature Food article runner")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Score individual-food pairs from CSV inputs")
    score.add_argument("--weights", required=True, help="CSV: individual_id plus nutrient beta columns")
    score.add_argument("--nutrients", required=True, help="CSV: food_id plus standardized nutrient columns")
    score.add_argument("--food-metadata", required=True, help="CSV: food_id, food_name, food_group, FCS2")
    score.add_argument("--output-dir", required=True)
    score.add_argument("--individual-id-col", default=None)
    score.add_argument("--food-id-col", default=None)
    score.add_argument("--delta-cap", type=float, default=12.0)
    score.add_argument("--mask-version", default="expert_revised_dual_channel")
    score.add_argument("--min-preservation-rho", type=float, default=0.90)
    score.set_defaults(func=run_score)

    synthetic = sub.add_parser("synthetic", help="Run digital-gut-twin validation benchmark")
    synthetic.add_argument("--output-dir", required=True)
    synthetic.add_argument("--n-individuals", type=int, default=120)
    synthetic.add_argument("--n-foods", type=int, default=80)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--delta-cap", type=float, default=12.0)
    synthetic.add_argument("--min-preservation-rho", type=float, default=0.90)
    synthetic.set_defaults(func=run_synthetic)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
