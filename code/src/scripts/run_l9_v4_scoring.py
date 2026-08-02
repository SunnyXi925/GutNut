#!/usr/bin/env python3
"""Run full L9 Food Compass GMNPS scoring with expert-revised v4 masks.

This runner avoids materialising a 15492 x 9234 long table. It computes
individual-by-food score matrices in food chunks, writes chunked parquet
matrices and exports manuscript-facing food summaries and validation source
data.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gmnps.scoring.masks import (  # noqa: E402
    EXPERT_REVISED_V4_MASK_VERSION,
    ORIGINAL_MASK_VERSION,
    PRIMARY_EXCLUDED_FROM_CHANNELS,
    audit_primary_mask,
    build_channel_vectors,
)
from gmnps.validation.preservation import fcs_category  # noqa: E402


def write_json(path: Path, payload: dict[str, object]) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, tuple):
            return [clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights = pd.read_parquet(args.weights)
    nutrients = pd.read_parquet(args.nutrients)
    fcs = pd.read_csv(args.fcs_scores)

    if weights.index.name is None:
        weights.index.name = "sample_id"
    nutrients.index = nutrients.index.astype(str)
    fcs["foodcode"] = fcs["foodcode"].astype(str)
    fcs = fcs.drop_duplicates("foodcode", keep="first").set_index("foodcode")

    shared_foods = [code for code in fcs.index if code in nutrients.index]
    if args.max_foods:
        shared_foods = shared_foods[: args.max_foods]
    if args.max_individuals:
        weights = weights.iloc[: args.max_individuals].copy()
    if not shared_foods:
        raise ValueError("No overlapping food codes between FCS scores and nutrient matrix.")

    food_meta = fcs.loc[shared_foods, ["description", "food_group", "FCS_2_0", "FCS_1_0", "NOVA", "HSR", "Nutri_Score"]].copy()
    food_meta = food_meta.rename(columns={"description": "food_name", "FCS_2_0": "FCS2"})
    nutrients = nutrients.loc[shared_foods]
    return weights, nutrients, food_meta


def preprocess_nutrients(
    nutrients: pd.DataFrame,
    winsor_lo: float = 0.01,
    winsor_hi: float = 0.99,
    robust_clip_iqr: float = 4.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply the L9-style nutrient preprocessing recorded in the v3 manifest."""

    x = nutrients.astype(float).copy()
    nonnegative = (x.min(axis=0) >= 0).fillna(False)
    q99 = x.quantile(0.99)
    heavy_tailed = [c for c in x.columns if bool(nonnegative.get(c, False)) and float(q99.get(c, 0.0)) > 10.0]
    if heavy_tailed:
        x.loc[:, heavy_tailed] = np.log1p(x.loc[:, heavy_tailed])

    lo = x.quantile(winsor_lo)
    hi = x.quantile(winsor_hi)
    x = x.clip(lower=lo, upper=hi, axis=1)

    med = x.median(axis=0)
    q75 = x.quantile(0.75)
    q25 = x.quantile(0.25)
    iqr_scale = (q75 - q25) / 1.349
    std = x.std(axis=0, ddof=0)
    scale = iqr_scale.where(iqr_scale.abs() > 1e-12, std)
    scale = scale.where(scale.abs() > 1e-12, 1.0)
    z = (x - med) / scale
    z = z.clip(lower=-robust_clip_iqr, upper=robust_clip_iqr, axis=1)
    manifest = {
        "log1p_on_heavy_tailed_cols": True,
        "heavy_tailed_rule": "column min >= 0 and p99 > 10",
        "heavy_tailed_cols": heavy_tailed,
        "winsor_lo": winsor_lo,
        "winsor_hi": winsor_hi,
        "robust_scaler_clip_iqr": robust_clip_iqr,
    }
    return z.astype("float32"), manifest


def robust_delta(raw: np.ndarray, cap: float, z_scale: float) -> np.ndarray:
    med = np.nanmedian(raw, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(raw - med), axis=0, keepdims=True) * 1.4826
    std = np.nanstd(raw, axis=0, keepdims=True)
    scale = np.where(mad > 1e-9, mad, std)
    scale = np.where(scale > 1e-9, scale, 1.0)
    z = (raw - med) / scale
    delta = cap * np.tanh(z / z_scale)
    delta = delta - np.nanmean(delta, axis=0, keepdims=True)
    delta = np.clip(delta, -cap, cap)
    delta = delta - np.nanmean(delta, axis=0, keepdims=True)
    return np.clip(delta, -cap, cap).astype("float32")


def matrix_to_parquet(
    matrix: np.ndarray,
    index: pd.Index,
    columns: list[str],
    path: Path,
) -> None:
    frame = pd.DataFrame(matrix, index=index, columns=columns)
    frame.index.name = index.name or "sample_id"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def summarize_chunk(
    food_codes: list[str],
    food_meta: pd.DataFrame,
    score: np.ndarray,
    delta: np.ndarray,
    mac_delta: np.ndarray,
    lipid_delta: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "food_id": food_codes,
            "food_name": food_meta.loc[food_codes, "food_name"].to_numpy(),
            "food_group": food_meta.loc[food_codes, "food_group"].to_numpy(),
            "FCS2": food_meta.loc[food_codes, "FCS2"].astype(float).to_numpy(),
            "GMNPS_mean": np.nanmean(score, axis=0),
            "GMNPS_sd": np.nanstd(score, axis=0, ddof=1),
            "GMNPS_p05": np.nanpercentile(score, 5, axis=0),
            "GMNPS_p95": np.nanpercentile(score, 95, axis=0),
            "GMNPS_delta_mean": np.nanmean(delta, axis=0),
            "GMNPS_delta_var": np.nanvar(delta, axis=0, ddof=1),
            "MAC_variance": np.nanvar(mac_delta, axis=0, ddof=1),
            "LIPID_variance": np.nanvar(lipid_delta, axis=0, ddof=1),
        }
    ).assign(
        dominant_channel=lambda d: np.where(d["MAC_variance"] >= d["LIPID_variance"], "MAC", "LIPID")
    )


def preservation_outputs(food_summary: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    rho = float(food_summary["FCS2"].rank().corr(food_summary["GMNPS_mean"].rank()))
    try:
        tau = float(food_summary["FCS2"].rank().corr(food_summary["GMNPS_mean"].rank(), method="kendall"))
    except ModuleNotFoundError:
        tau = None
    base_cat = food_summary["FCS2"].map(fcs_category)
    gmnps_cat = food_summary["GMNPS_mean"].map(fcs_category)
    shifts = base_cat != gmnps_cat
    metrics = {
        "n_foods": int(len(food_summary)),
        "spearman_fcs2_gmnps_mean": rho,
        "kendall_fcs2_gmnps_mean": tau,
        "passes_min_rho_0_90": bool(rho >= 0.90),
        "category_shift_count": int(shifts.sum()),
        "category_shift_fraction": float(shifts.mean()),
    }
    write_json(out_dir / "validation" / "nps_preservation_metrics.json", metrics)

    transition = pd.crosstab(base_cat, gmnps_cat).reindex(
        index=["minimize", "moderate", "encourage"],
        columns=["minimize", "moderate", "encourage"],
        fill_value=0,
    )
    transition.to_csv(out_dir / "validation" / "category_transition.csv")

    group = (
        food_summary.groupby("food_group")
        .agg(
            n_foods=("food_id", "size"),
            FCS2_mean=("FCS2", "mean"),
            GMNPS_mean=("GMNPS_mean", "mean"),
            GMNPS_sd_mean=("GMNPS_sd", "mean"),
            delta_var_mean=("GMNPS_delta_var", "mean"),
            MAC_variance_mean=("MAC_variance", "mean"),
            LIPID_variance_mean=("LIPID_variance", "mean"),
        )
        .reset_index()
    )
    group["dominant_channel"] = np.where(group["MAC_variance_mean"] >= group["LIPID_variance_mean"], "MAC", "LIPID")
    group.to_csv(out_dir / "validation" / "food_group_summary.csv", index=False)
    group.rename(
        columns={
            "n_foods": "n",
            "delta_var_mean": "total_var_mean",
            "MAC_variance_mean": "mac_var_mean",
            "LIPID_variance_mean": "lipid_var_mean",
        }
    )[["food_group", "n", "total_var_mean", "mac_var_mean", "lipid_var_mean"]].to_csv(
        out_dir / "figure_source_data" / "food_group_heterogeneity.csv",
        index=False,
    )
    return metrics


def compare_v3_v4(v3_summary_path: Path, v4_summary: pd.DataFrame, out_dir: Path) -> None:
    if not v3_summary_path.exists():
        return
    v3 = pd.read_csv(v3_summary_path)
    v3["food_id"] = v3["foodcode"].astype(str)
    merged = v3.merge(v4_summary, on="food_id", suffixes=("_v3", "_v4"))
    if merged.empty:
        return
    comparison = {
        "n_overlap_foods": int(len(merged)),
        "v3_spearman_fcs2_total_mean": float(merged["FCS_2_0"].rank().corr(merged["total_mean"].rank())),
        "v4_spearman_fcs2_gmnps_mean": float(merged["FCS2"].rank().corr(merged["GMNPS_mean"].rank())),
        "v3_total_var_mean": float(merged["total_var"].mean()),
        "v4_delta_var_mean": float(merged["GMNPS_delta_var"].mean()),
        "v3_mac_var_mean": float(merged["mac_var"].mean()),
        "v4_mac_var_mean": float(merged["MAC_variance"].mean()),
        "v3_lipid_var_mean": float(merged["lipid_var"].mean()),
        "v4_lipid_var_mean": float(merged["LIPID_variance"].mean()),
    }
    write_json(out_dir / "validation" / "v3_vs_v4_summary.json", comparison)
    merged[
        [
            "food_id",
            "food_name",
            "food_group_v4",
            "FCS2",
            "total_mean",
            "GMNPS_mean",
            "total_var",
            "GMNPS_delta_var",
            "mac_var",
            "MAC_variance",
            "lipid_var",
            "LIPID_variance",
        ]
    ].to_csv(out_dir / "validation" / "v3_vs_v4_food_comparison.csv", index=False)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    for subdir in ["matrices", "tables", "validation", "figure_source_data", "manifests"]:
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    weights, nutrients_raw, food_meta = read_inputs(args)
    shared_nutrients = [c for c in weights.columns if c in nutrients_raw.columns]
    if not shared_nutrients:
        raise ValueError("No shared nutrient columns between W_personalized and nutrient matrix.")

    nutrients, preprocessing = preprocess_nutrients(
        nutrients_raw[shared_nutrients],
        winsor_lo=args.winsor_lo,
        winsor_hi=args.winsor_hi,
        robust_clip_iqr=args.robust_clip_iqr,
    )
    weights = weights[shared_nutrients].astype("float32")
    masks = build_channel_vectors(shared_nutrients, args.mask_version)
    mac_mask = masks["mac"].astype("float32")
    lipid_mask = masks["lipid"].astype("float32")

    w = weights.to_numpy(dtype="float32", copy=True)
    w_mac = w * mac_mask[None, :]
    w_lipid = w * lipid_mask[None, :]
    w_other = w * masks["other"].astype("float32")[None, :]

    summaries = []
    chunks: list[dict[str, str | int]] = []
    food_ids = list(nutrients.index.astype(str))
    sample_index = weights.index.copy()
    sample_index.name = weights.index.name or "sample_id"

    for chunk_id, start in enumerate(range(0, len(food_ids), args.chunk_size)):
        chunk_foods = food_ids[start : start + args.chunk_size]
        x = nutrients.loc[chunk_foods, shared_nutrients].to_numpy(dtype="float32", copy=True)
        raw_total = w @ x.T
        raw_mac = w_mac @ x.T
        raw_lipid = w_lipid @ x.T
        raw_other = w_other @ x.T
        delta = robust_delta(raw_total, args.delta_cap, args.z_scale)
        fcs = food_meta.loc[chunk_foods, "FCS2"].astype("float32").to_numpy()
        score = np.clip(delta + fcs[None, :], args.score_min, args.score_max).astype("float32")

        denom = np.abs(raw_mac) + np.abs(raw_lipid) + np.abs(raw_other)
        mac_share = np.divide(np.abs(raw_mac), denom, out=np.zeros_like(raw_mac), where=denom > 1e-12)
        lipid_share = np.divide(np.abs(raw_lipid), denom, out=np.zeros_like(raw_lipid), where=denom > 1e-12)
        mac_delta = (np.sign(raw_mac) * np.abs(delta) * mac_share).astype("float32")
        lipid_delta = (np.sign(raw_lipid) * np.abs(delta) * lipid_share).astype("float32")

        stem = f"chunk_{chunk_id:04d}_{start:05d}_{start + len(chunk_foods) - 1:05d}"
        chunk_paths = {
            "chunk_id": chunk_id,
            "start": start,
            "n_foods": len(chunk_foods),
            "score": str(out_dir / "matrices" / f"S_score_individual_x_food_v4_{stem}.parquet"),
            "delta": str(out_dir / "matrices" / f"S_delta_individual_x_food_v4_{stem}.parquet"),
            "mac": str(out_dir / "matrices" / f"S_mac_individual_x_food_v4_{stem}.parquet"),
            "lipid": str(out_dir / "matrices" / f"S_lipid_individual_x_food_v4_{stem}.parquet"),
        }
        matrix_to_parquet(score, sample_index, chunk_foods, Path(chunk_paths["score"]))
        matrix_to_parquet(delta, sample_index, chunk_foods, Path(chunk_paths["delta"]))
        matrix_to_parquet(mac_delta, sample_index, chunk_foods, Path(chunk_paths["mac"]))
        matrix_to_parquet(lipid_delta, sample_index, chunk_foods, Path(chunk_paths["lipid"]))
        chunks.append(chunk_paths)
        summaries.append(summarize_chunk(chunk_foods, food_meta, score, delta, mac_delta, lipid_delta))
        print(f"wrote {stem} ({len(chunk_foods)} foods)", flush=True)

    food_summary = pd.concat(summaries, ignore_index=True)
    food_summary.to_csv(out_dir / "tables" / "S_food_summary_v4.csv", index=False)
    food_summary.to_parquet(out_dir / "tables" / "S_food_summary_v4.parquet", index=False)

    metrics = preservation_outputs(food_summary, out_dir)
    compare_v3_v4(Path(args.v3_summary), food_summary, out_dir)

    food_summary.sort_values("GMNPS_delta_var", ascending=False).to_csv(
        out_dir / "validation" / "food_variance_ranked_total.csv",
        index=False,
    )
    food_summary.sort_values("MAC_variance", ascending=False).to_csv(
        out_dir / "validation" / "food_variance_ranked_mac.csv",
        index=False,
    )
    food_summary.sort_values("LIPID_variance", ascending=False).to_csv(
        out_dir / "validation" / "food_variance_ranked_lipid.csv",
        index=False,
    )

    manifest = {
        "version": "v4",
        "description": "Expert-revised GMNPS matrix scoring with Food Compass 2.0 prior",
        "mask_version": args.mask_version,
        "n_individuals": int(weights.shape[0]),
        "n_foods": int(len(food_ids)),
        "n_nutrients_total": int(len(shared_nutrients)),
        "n_nutrients_mac": int(np.sum(masks["mac"])),
        "n_nutrients_lipid": int(np.sum(masks["lipid"])),
        "mac_nutrients": masks["mac_nutrients"],
        "lipid_nutrients": masks["lipid_nutrients"],
        "excluded_from_primary_channels": sorted(PRIMARY_EXCLUDED_FROM_CHANNELS),
        "mask_audit": audit_primary_mask(shared_nutrients),
        "preprocessing": preprocessing,
        "score_range": [args.score_min, args.score_max],
        "delta_cap": args.delta_cap,
        "z_scale": args.z_scale,
        "chunk_size": args.chunk_size,
        "chunks": chunks,
        "input_files": {
            "weights": args.weights,
            "nutrients": args.nutrients,
            "fcs_scores": args.fcs_scores,
            "v3_summary": args.v3_summary,
        },
        "preservation": metrics,
        "sensitivity_note": "L9 v3 used the original 15+15 mask and should be treated as sensitivity, not the primary analysis.",
    }
    write_json(out_dir / "manifests" / "scoring_manifest_v4.json", manifest)
    write_json(out_dir / "run_l9_v4_report.json", {"manifest": manifest})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run L9 expert-revised v4 GMNPS matrix scoring")
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--weights", default=str(root / "data/project_data/predict_multi/L7_nutrient_bridge/W_personalized.parquet"))
    parser.add_argument("--nutrients", default=str(root / "data/project_data/predict_multi/L6_layer_c_fndds/processed/N_food_nutrient_imputed.parquet"))
    parser.add_argument("--fcs-scores", default=str(root / "data/project_data/predict_multi/L9_food_compass/food_compass_scores.csv"))
    parser.add_argument("--v3-summary", default=str(root / "data/project_data/predict_multi/L9_food_compass/S_food_summary_v3.csv"))
    parser.add_argument("--output-dir", default=str(root / "outputs/gmnps_v4"))
    parser.add_argument("--mask-version", default=EXPERT_REVISED_V4_MASK_VERSION)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-foods", type=int, default=None)
    parser.add_argument("--max-individuals", type=int, default=None)
    parser.add_argument("--delta-cap", type=float, default=12.0)
    parser.add_argument("--z-scale", type=float, default=2.0)
    parser.add_argument("--score-min", type=float, default=1.0)
    parser.add_argument("--score-max", type=float, default=100.0)
    parser.add_argument("--winsor-lo", type=float, default=0.01)
    parser.add_argument("--winsor-hi", type=float, default=0.99)
    parser.add_argument("--robust-clip-iqr", type=float, default=4.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
