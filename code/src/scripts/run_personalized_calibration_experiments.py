#!/usr/bin/env python3
"""Run structured personalized-calibration validation experiments.

The runner consumes local, ignored data resources and writes compact CSV/JSON
summaries plus a LaTeX Supplementary Information draft. It deliberately avoids
committing raw data, score matrices, KG dumps or generated figure assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gmnps.scoring.masks import build_channel_vectors  # noqa: E402
from gmnps.scoring.calibration_objective import (  # noqa: E402
    CalibrationParams,
    apply_personalized_offset,
    calibration_objective,
)
from gmnps.supplement.fcs_style import build_supplement_manifest  # noqa: E402
from gmnps.validation.reference_targets import evaluate_target  # noqa: E402


DEFAULT_PATHS = {
    "cmd_metadata": "data/project_data/predict_multi/L1_microbiome/cmd_processed/layer_a_master.parquet",
    "gmrepo_metadata": "data/project_data/predict_multi/L1_microbiome/gmrepo_processed/gmrepo_run_metadata.parquet",
    "beta_i": "data/project_data/predict_multi/L7_nutrient_bridge_beta_i/W_personalized.parquet",
    "beta_diagnostics": "data/project_data/predict_multi/L7_nutrient_bridge_beta_i/sample_beta_diagnostics.csv",
    "nutrient_bridge": "data/project_data/predict_multi/L7_nutrient_bridge/B_nutrient_genus.parquet",
    "nutrient_metabolite": "data/project_data/predict_multi/L7_nutrient_bridge/M_nutrient_metabolite.parquet",
    "fdc_food_vectors": "data/project_data/predict_multi/L6_layer_c_fndds/processed/N_food_nutrient_full.parquet",
    "fndds_raw_values": "data/project_data/predict_multi/L6_layer_c_fndds/raw/FNDDS_Nutrient_Values.xlsx",
    "fcs2_food_table": "data/project_data/predict_multi/L9_food_compass/S_food_summary_v3.csv",
    "l9_score_matrix": "data/project_data/predict_multi/L9_food_compass/S_score_individual_x_food_v3.parquet",
    "cra_host": "data/project_data/CRA013939/processed/individual_reconstructed_v2/host_3224.parquet",
    "cra_abundance": "data/project_data/CRA013939/processed/individual_reconstructed_v2/abund_3224.parquet",
    "cra_food_intake": "data/project_data/CRA013939/processed/individual_reconstructed_v2/food_intake_496.parquet",
    "cra_metabolites": "data/project_data/CRA013939/processed/individual_reconstructed_v2/metab_496.parquet",
    "gmmad2_disease_metabolite": "data/project_data/predict_multi/L4_knowledge_graph/gmmad2_processed/D_disease_metabolite_signed.parquet",
}

CLINICAL_RENAME = {
    "host_age": "age",
    "BMI": "bmi",
    "glucose": "fbg",
    "triglyceride": "tg",
    "total_cholesterol": "cholesterol",
}

DISEASE_KEYWORDS = {
    "Health": ("health", "healthy"),
    "IBD": ("inflammatory bowel", "crohn", "colitis", "ibd"),
    "CRC": ("colorectal", "colon", "rectal", "crc"),
    "CVD": ("cardiovascular", "coronary", "heart", "atherosclerosis", "cvd"),
    "T2D": ("type 2 diabetes", "diabetes mellitus", "diabetes", "t2d"),
}

PRIMARY_OUTCOMES = ["glucose", "triglyceride", "total_cholesterol", "hdl", "ldl", "hba1c"]

FCS2_CONSENSUS_DIRECTION_POLICY_VERSION = "fcs2_consensus_direction_policy_v1"
FCS2_CONSENSUS_DIRECTION_POLICY_V1 = {
    "vegetables": "stable_or_up",
    "vegetable": "stable_or_up",
    "fruits": "stable_or_up",
    "fruit": "stable_or_up",
    "fish and seafood": "stable_or_up",
    "seafood": "stable_or_up",
    "fish": "stable_or_up",
    "legumes": "stable_or_up",
    "beans and peas": "stable_or_up",
    "whole grains": "stable_or_up",
    "nuts and seeds": "stable_or_up",
    "sweets": "stable_or_down",
    "sugars": "stable_or_down",
    "fats and oils": "stable_or_down",
    "fats oils": "stable_or_down",
    "snacks": "stable_or_down",
    "processed foods": "stable_or_down",
}

SECTION1_POPULATION_COLUMNS = [
    "amplification",
    "n_individuals",
    "n_foods",
    "score_source",
    "direction_policy_version",
    "calibration_status",
    "population_spearman",
    "mean_absolute_population_shift",
    "mean_individual_rank_shift",
    "food_group_direction_pass_fraction",
    "food_subgroup_direction_pass_fraction",
    "group_penalty",
    "group_consensus_gate_passed",
    "subgroup_consensus_gate_passed",
    "consensus_gates_passed",
    "consensus_gate_penalty",
    "objective_value",
    "spearman_fcs2_gmnps_mean",
    "median_food_sd",
]
SECTION1_RANK_THRESHOLD_COLUMNS = [
    "group",
    "n_individuals",
    "median_rank_spearman_vs_fcs2",
    "median_top_k_jaccard_vs_fcs2",
    "median_delta_span_p95_p05",
    "passes_healthy_rank_preservation",
    "passes_disease_reranking",
    "passes_delta_span",
    "amplification",
]
SECTION1_INDIVIDUAL_RANK_COLUMNS = [
    "individual_id",
    "phenotype_label",
    "disease",
    "label_group",
    "n_foods",
    "amplification",
    "rank_spearman_vs_fcs2",
    "top_k_jaccard_vs_fcs2",
    "delta_span_p95_p05",
]
SECTION1_GROUP_CONSENSUS_COLUMNS = [
    "amplification",
    "food_group",
    "n_foods",
    "mean_fcs2",
    "mean_gmnps",
    "median_within_food_sd",
    "between_food_sd",
]
SECTION1_FOOD_SUMMARY_COLUMNS = [
    "foodcode",
    "description",
    "food_group",
    "FCS2",
    "GMNPS_mean",
    "GMNPS_sd",
    "amplification",
    "food_subgroup",
    "expected_group_direction",
    "expected_subgroup_direction",
    "GMNPS_delta_mean",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, tuple):
            return [clean(v) for v in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    path.write_text(json.dumps(clean(payload), indent=2, ensure_ascii=True), encoding="utf-8")


def _finite_observed(value: object) -> float:
    """Return a finite metric value, or NaN when the runner did not compute one."""

    try:
        observed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return observed if math.isfinite(observed) else float("nan")


def _primary_section1_row(section1: dict[str, object]) -> dict[str, object]:
    """Select the pre-specified primary calibration run without optimizing over runs."""

    rows = section1.get("population_consensus", [])
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("amplification") == 1.0 and row.get("calibration_status") == "eligible":
            return row
    return {}


def build_submission_readiness_report(
    section1: dict[str, object],
    section2: dict[str, object],
    section3: dict[str, object],
    section4: dict[str, object],
) -> dict[str, object]:
    """Evaluate blocking reference targets from metrics actually produced by sections 1-4."""

    primary_section1 = _primary_section1_row(section1)
    observations = [
        (
            "fcs2_population_spearman",
            primary_section1.get("spearman_fcs2_gmnps_mean"),
            "section1.population_consensus[amplification=1.0].spearman_fcs2_gmnps_mean",
        ),
        (
            "food_group_direction_pass_fraction",
            primary_section1.get("food_group_direction_pass_fraction"),
            "section1.population_consensus[amplification=1.0].food_group_direction_pass_fraction",
        ),
        # Section 2 assesses clinical correlations/retention, not the official external
        # GMWI2 health-versus-disease classification benchmark required by this target.
        (
            "gmwi2_external_balanced_accuracy",
            None,
            "not computed: section2 has no official GMWI2 external classification evaluation",
        ),
        (
            "kg_binary_balanced_accuracy",
            section3.get("binary_balanced_accuracy"),
            "section3.binary_balanced_accuracy",
        ),
        (
            "response_delta_spearman_fdr_pass_fraction",
            section4.get("response_delta_spearman_fdr_pass_fraction"),
            "section4.response_delta_spearman_fdr_pass_fraction",
        ),
    ]
    rows = []
    for name, value, source in observations:
        observed = _finite_observed(value)
        row = evaluate_target(name, observed)
        row["blocking"] = True
        row["observed_source"] = source
        row["status"] = "evaluated" if math.isfinite(observed) else "missing"
        rows.append(row)

    return {
        "ready_for_submission": bool(all(row["passes"] for row in rows)),
        "n_blocking_targets": len(rows),
        "n_blocking_targets_passing": int(sum(bool(row["passes"]) for row in rows)),
        "blocking_targets": rows,
    }


def write_submission_readiness_report(out_dir: Path, report: dict[str, object]) -> None:
    """Write both machine-readable readiness artifacts at the experiment-output root."""

    rows = report["blocking_targets"]
    if not isinstance(rows, list):
        raise ValueError("submission readiness report requires blocking target rows")
    pd.DataFrame(rows).to_csv(out_dir / "submission_readiness_report.csv", index=False)
    write_json(out_dir / "submission_readiness_report.json", report)


def source_revision(root: Path) -> dict[str, object]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=False)
    return {
        "sha": commit.stdout.strip() if commit.returncode == 0 else "unavailable",
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else True,
    }


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return values * 0.0
    return (values - values.mean()) / std


def _rank_spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left.astype(float), right.astype(float)], axis=1).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].rank(method="average").corr(pair.iloc[:, 1].rank(method="average")))


def normalize_label(value: object) -> str | None:
    text = str(value).lower()
    if "health" in text or "healthy" in text:
        return "Health"
    if "ibd" in text or "crohn" in text or "colitis" in text:
        return "IBD"
    if "crc" in text or "adenoma" in text or "carcinoma" in text or "colorectal" in text:
        return "CRC"
    if "t2d" in text or "diabetes" in text:
        return "T2D"
    if "cvd" in text or "cardio" in text or "heart" in text:
        return "CVD"
    return None


def load_paths(root: Path) -> dict[str, Path]:
    return {key: root / value for key, value in DEFAULT_PATHS.items()}


def fcs_food_table(path: Path) -> pd.DataFrame:
    food = pd.read_csv(path)
    food["foodcode"] = food["foodcode"].astype(str)
    food = food.drop_duplicates("foodcode").set_index("foodcode", drop=False)
    return food


def fcs2_consensus_direction_policy_v1(food: pd.DataFrame) -> pd.DataFrame:
    """Return explicit, deterministic direction metadata for the FCS2 food table."""
    if "food_group" not in food.columns:
        raise ValueError("FCS2 food table missing required column: food_group")
    metadata = pd.DataFrame(index=food.index)
    metadata["food_group"] = food["food_group"].astype(str)
    subgroup_column = next(
        (column for column in ("food_subgroup", "food_sub_group", "subgroup") if column in food.columns),
        None,
    )
    if subgroup_column is not None:
        metadata["food_subgroup"] = food[subgroup_column].astype(str)

    def direction(label: object) -> str:
        return FCS2_CONSENSUS_DIRECTION_POLICY_V1.get(str(label).strip().lower(), "stable")

    metadata["expected_group_direction"] = metadata["food_group"].map(direction)
    if "food_subgroup" in metadata:
        metadata["expected_subgroup_direction"] = metadata["food_subgroup"].map(direction)
    return metadata


def matrix_rank_metrics(
    calibrated_scores: pd.DataFrame,
    fcs: pd.Series,
    metadata: pd.DataFrame,
    amplification: float,
    top_k: int,
    chunk_size: int,
) -> pd.DataFrame:
    food_ids = list(fcs.index.astype(str))
    fcs_arr = fcs.to_numpy(dtype=np.float32)
    fcs_rank = fcs.rank(method="average").to_numpy(dtype=np.float32)
    fcs_centered = fcs_rank - float(np.mean(fcs_rank))
    fcs_norm = float(np.sqrt(np.sum(fcs_centered**2)))
    fcs_top = set(np.argsort(-fcs_arr)[:top_k].tolist())
    annotations = metadata.set_index("sample_id", drop=False)
    rows: list[dict[str, object]] = []
    for start in range(0, calibrated_scores.shape[0], chunk_size):
        block = calibrated_scores.iloc[start : start + chunk_size]
        calibrated = block.to_numpy(dtype=np.float32, copy=True)

        ranks = pd.DataFrame(calibrated).rank(axis=1, method="average").to_numpy(dtype=np.float32)
        rank_centered = ranks - ranks.mean(axis=1, keepdims=True)
        denom = np.sqrt((rank_centered**2).sum(axis=1)) * fcs_norm
        spearman = np.divide(
            rank_centered @ fcs_centered,
            denom,
            out=np.full(rank_centered.shape[0], np.nan, dtype=np.float32),
            where=denom > 1e-12,
        )
        top_idx = np.argpartition(-calibrated, kth=min(top_k, len(food_ids)) - 1, axis=1)[:, :top_k]
        intersections = np.array([sum(int(idx) in fcs_top for idx in row) for row in top_idx], dtype=float)
        jaccard = intersections / (2 * top_k - intersections)
        span = np.percentile(calibrated - fcs_arr[None, :], 95, axis=1) - np.percentile(
            calibrated - fcs_arr[None, :],
            5,
            axis=1,
        )

        for local_idx, sample_id in enumerate(block.index.astype(str)):
            meta = annotations.loc[sample_id] if sample_id in annotations.index else {}
            phenotype = str(meta["phenotype_label"]) if isinstance(meta, pd.Series) and "phenotype_label" in meta else "missing"
            disease = str(meta["disease"]) if isinstance(meta, pd.Series) and "disease" in meta else "missing"
            rows.append(
                {
                    "individual_id": sample_id,
                    "phenotype_label": phenotype,
                    "disease": disease,
                    "label_group": normalize_label(phenotype) or normalize_label(disease) or "Other",
                    "n_foods": len(food_ids),
                    "amplification": amplification,
                    "rank_spearman_vs_fcs2": float(spearman[local_idx]),
                    "top_k_jaccard_vs_fcs2": float(jaccard[local_idx]),
                    "delta_span_p95_p05": float(span[local_idx]),
                }
            )
    return pd.DataFrame(rows)


def calibration_candidate_status(diagnostics: dict[str, object]) -> str:
    """Classify calibration candidates before emitting score-derived outputs."""
    objective_value = diagnostics.get("objective_value", float("nan"))
    try:
        finite_objective = bool(np.isfinite(float(objective_value)))
    except (TypeError, ValueError):
        finite_objective = False
    if not finite_objective or diagnostics.get("consensus_gates_passed") is not True:
        return "rejected_consensus_gate"
    return "eligible"


def section1_output_frame(rows: list[dict[str, object]], columns: list[str]) -> pd.DataFrame:
    """Build a schema-stable Section 1 output frame, including zero-row outputs."""
    return pd.DataFrame(rows).reindex(columns=columns)


def write_section1_outputs(
    section_dir: Path,
    population: pd.DataFrame,
    rank_thresholds: pd.DataFrame,
    individual_rank_metrics: pd.DataFrame,
    group_consensus: pd.DataFrame,
    food_summary: pd.DataFrame,
) -> None:
    """Write every Section 1 CSV with headers even when no candidate is eligible."""
    section_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "population_consensus.csv": (population, SECTION1_POPULATION_COLUMNS),
        "rank_shift_thresholds.csv": (rank_thresholds, SECTION1_RANK_THRESHOLD_COLUMNS),
        "individual_rank_shift_metrics.csv": (individual_rank_metrics, SECTION1_INDIVIDUAL_RANK_COLUMNS),
        "food_group_consensus.csv": (group_consensus, SECTION1_GROUP_CONSENSUS_COLUMNS),
        "food_summary_by_amplification.csv": (food_summary, SECTION1_FOOD_SUMMARY_COLUMNS),
    }
    for filename, (frame, columns) in outputs.items():
        frame.reindex(columns=columns).to_csv(section_dir / filename, index=False)


def run_section1(paths: dict[str, Path], out_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    from gmnps.validation.rank_shift import (  # noqa: PLC0415
        RankShiftThresholds,
        evaluate_rank_shift_thresholds,
    )

    food = fcs_food_table(paths["fcs2_food_table"])
    food.index = food.index.astype(str)
    policy_metadata = fcs2_consensus_direction_policy_v1(food)
    score_source = "beta_i_raw_offset"
    if args.legacy_score_matrix:
        scores = pd.read_parquet(paths["l9_score_matrix"])
        scores.index = scores.index.astype(str)
        scores.columns = scores.columns.astype(str)
        common = [food_id for food_id in food.index if food_id in scores.columns]
        scores = scores.loc[:, common].astype("float32")
        raw_offset = scores.sub(food.loc[common, "FCS_2_0"].astype(float), axis=1)
        score_source = "legacy_score_matrix"
    else:
        beta_i = pd.read_parquet(paths["beta_i"])
        food_nutrients = pd.read_parquet(paths["fdc_food_vectors"])
        beta_i.index = beta_i.index.astype(str)
        beta_i.columns = beta_i.columns.astype(str)
        food_nutrients.index = food_nutrients.index.astype(str)
        food_nutrients.columns = food_nutrients.columns.astype(str)
        common = [food_id for food_id in food.index if food_id in food_nutrients.index]
        nutrient_columns = [column for column in beta_i.columns if column in food_nutrients.columns]
        if not common:
            raise ValueError("No overlapping FCS2 foods in the food nutrient matrix")
        if not nutrient_columns:
            raise ValueError("No shared nutrient columns between beta_i and food nutrient matrix")
        common_samples = beta_i.index
        raw_offset = pd.DataFrame(
            beta_i.loc[common_samples, nutrient_columns].to_numpy(dtype=np.float32)
            @ food_nutrients.loc[common, nutrient_columns].to_numpy(dtype=np.float32).T,
            index=common_samples,
            columns=common,
        )
    fcs = food.loc[common, "FCS_2_0"].astype(float)
    policy_metadata = policy_metadata.reindex(common)
    metadata = pd.read_parquet(paths["cmd_metadata"], columns=["sample_id", "phenotype_label", "disease"])
    metadata["sample_id"] = metadata["sample_id"].astype(str)

    food_rows = []
    summary_rows = []
    threshold_frames = []
    rank_frames = []
    thresholds = RankShiftThresholds()

    for amplification in args.amplifications:
        params = CalibrationParams(
            delta_cap=args.delta_cap * amplification,
            temperature=args.calibration_temperature,
            group_penalty=args.group_penalty,
        )
        diagnostics = calibration_objective(fcs, raw_offset, policy_metadata, params)
        status = calibration_candidate_status(diagnostics)
        diagnostic_row = {
            "amplification": amplification,
            "n_individuals": int(raw_offset.shape[0]),
            "n_foods": int(raw_offset.shape[1]),
            "score_source": score_source,
            "direction_policy_version": FCS2_CONSENSUS_DIRECTION_POLICY_VERSION,
            "calibration_status": status,
            **diagnostics,
        }
        if status != "eligible":
            summary_rows.append(diagnostic_row)
            continue

        calibrated = apply_personalized_offset(fcs, raw_offset, params)
        mean_scores = calibrated.mean(axis=0)
        sd_scores = calibrated.std(axis=0)
        food_summary = pd.DataFrame(
            {
                "foodcode": common,
                "description": food.loc[common, "description"].astype(str).to_numpy(),
                "food_group": food.loc[common, "food_group"].astype(str).to_numpy(),
                "FCS2": fcs.to_numpy(dtype=float),
                "GMNPS_mean": mean_scores.to_numpy(dtype=float),
                "GMNPS_sd": sd_scores.to_numpy(dtype=float),
                "amplification": amplification,
            }
        )
        if "food_subgroup" in policy_metadata:
            food_summary["food_subgroup"] = policy_metadata["food_subgroup"].to_numpy()
        food_summary["expected_group_direction"] = policy_metadata["expected_group_direction"].to_numpy()
        if "expected_subgroup_direction" in policy_metadata:
            food_summary["expected_subgroup_direction"] = policy_metadata["expected_subgroup_direction"].to_numpy()
        food_summary["GMNPS_delta_mean"] = food_summary["GMNPS_mean"] - food_summary["FCS2"]
        food_rows.append(food_summary)
        summary_rows.append(
            {
                **diagnostic_row,
                "spearman_fcs2_gmnps_mean": _rank_spearman(fcs, mean_scores),
                "mean_absolute_population_shift": float((mean_scores - fcs).abs().mean()),
                "median_food_sd": float(sd_scores.median()),
            }
        )

        rank = matrix_rank_metrics(calibrated, fcs, metadata, amplification, args.top_k, args.chunk_size)
        rank_frames.append(rank)
        threshold = evaluate_rank_shift_thresholds(rank, thresholds, group_col="label_group")
        threshold["amplification"] = amplification
        threshold_frames.append(threshold)

    food_all = (
        pd.concat(food_rows, ignore_index=True).reindex(columns=SECTION1_FOOD_SUMMARY_COLUMNS)
        if food_rows
        else section1_output_frame([], SECTION1_FOOD_SUMMARY_COLUMNS)
    )
    rank_all = (
        pd.concat(rank_frames, ignore_index=True).reindex(columns=SECTION1_INDIVIDUAL_RANK_COLUMNS)
        if rank_frames
        else section1_output_frame([], SECTION1_INDIVIDUAL_RANK_COLUMNS)
    )
    threshold_all = (
        pd.concat(threshold_frames, ignore_index=True).reindex(columns=SECTION1_RANK_THRESHOLD_COLUMNS)
        if threshold_frames
        else section1_output_frame([], SECTION1_RANK_THRESHOLD_COLUMNS)
    )
    population = section1_output_frame(summary_rows, SECTION1_POPULATION_COLUMNS)
    group_summary = section1_output_frame([], SECTION1_GROUP_CONSENSUS_COLUMNS)
    if not food_all.empty:
        group_summary = (
            food_all.groupby(["amplification", "food_group"], sort=False)
            .agg(
                n_foods=("foodcode", "nunique"),
                mean_fcs2=("FCS2", "mean"),
                mean_gmnps=("GMNPS_mean", "mean"),
                median_within_food_sd=("GMNPS_sd", "median"),
                between_food_sd=("GMNPS_mean", "std"),
            )
            .reset_index()
        )
        group_summary = group_summary.reindex(columns=SECTION1_GROUP_CONSENSUS_COLUMNS)

    section_dir = out_dir / "section1_population_consensus"
    write_section1_outputs(section_dir, population, threshold_all, rank_all, group_summary, food_all)

    return {
        "population_consensus": population.to_dict(orient="records"),
        "rank_shift_thresholds": threshold_all.to_dict(orient="records"),
        "n_rank_rows": int(len(rank_all)),
        "score_source": score_source,
        "direction_policy_version": FCS2_CONSENSUS_DIRECTION_POLICY_VERSION,
        "section_dir": str(section_dir),
    }


def cra_beta_proxy(abundance: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    frame = abundance.set_index("subject_id") if "subject_id" in abundance.columns else abundance.copy()
    genera = [g for g in bridge.columns.astype(str) if g in frame.columns]
    if not genera:
        return pd.DataFrame(index=frame.index)
    x = frame[genera].astype(float)
    rel = x.div(x.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    clr = np.log(rel + 1e-6)
    clr = clr.sub(clr.mean(axis=1), axis=0)
    beta = clr.to_numpy(dtype=float) @ bridge[genera].T.to_numpy(dtype=float)
    beta_frame = pd.DataFrame(beta, index=frame.index.astype(str), columns=bridge.index.astype(str))
    nutrient_cols = list(bridge.index.astype(str))
    masks = build_channel_vectors(nutrient_cols)
    beta_frame["beta_mac_mean"] = beta_frame.loc[:, np.asarray(nutrient_cols)[masks["mac"]]].mean(axis=1) if masks["mac"].any() else 0.0
    beta_frame["beta_lipid_mean"] = beta_frame.loc[:, np.asarray(nutrient_cols)[masks["lipid"]]].mean(axis=1) if masks["lipid"].any() else 0.0
    beta_frame["beta_personalized_axis"] = zscore(beta_frame["beta_mac_mean"] - beta_frame["beta_lipid_mean"])
    beta_frame.index.name = "sample_id"
    return beta_frame


def normalized_clinical_metadata(host: pd.DataFrame) -> pd.DataFrame:
    meta = host.rename(columns=CLINICAL_RENAME).copy()
    meta["sample_id"] = meta["subject_id"].astype(str)
    return meta


def run_section2(paths: dict[str, Path], out_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    from gmnps.validation.clinical_correlation import (  # noqa: PLC0415
        ClinicalVariableSpec,
        audit_clinical_completeness,
        choose_clinical_cohort,
        evaluate_gmwi2_retention,
        spearman_clinical_correlations,
    )
    from gmnps.validation.microbiome_health import (  # noqa: PLC0415
        MicrobiomeHealthMode,
        genus_proxy_health_score,
        load_official_gmwi2_scores,
    )

    spec = ClinicalVariableSpec(min_n=50)
    gmrepo = pd.read_parquet(paths["gmrepo_metadata"]).rename(columns=CLINICAL_RENAME)
    gmrepo["sample_id"] = gmrepo["run_id"].astype(str)
    cra_host = normalized_clinical_metadata(pd.read_parquet(paths["cra_host"]))
    cra_abundance = pd.read_parquet(paths["cra_abundance"])
    bridge = pd.read_parquet(paths["nutrient_bridge"])
    bridge.index = bridge.index.astype(str)
    bridge.columns = bridge.columns.astype(str)

    gmrepo_audit = audit_clinical_completeness(gmrepo, list(spec.variables), spec.max_missing_fraction, spec.min_n)
    cra_audit = audit_clinical_completeness(cra_host, list(spec.variables), spec.max_missing_fraction, spec.min_n)
    selected = choose_clinical_cohort(gmrepo_audit, cra_audit)

    health_mode = MicrobiomeHealthMode(args.microbiome_health_mode)
    if health_mode is MicrobiomeHealthMode.OFFICIAL_GMWI2:
        if args.official_gmwi2_scores is None:
            raise ValueError("official_gmwi2 mode requires --official-gmwi2-scores")
        official_path = Path(args.official_gmwi2_scores)
        if not official_path.is_absolute():
            official_path = Path(args.root) / official_path
        health_scores = load_official_gmwi2_scores(official_path).set_index("sample_id")["official_gmwi2_score"]
        health_feature = "official_gmwi2_score"
        health_mode_label = MicrobiomeHealthMode.OFFICIAL_GMWI2.value
    else:
        health_scores = genus_proxy_health_score(cra_abundance)
        health_feature = "genus_proxy_health_score"
        health_mode_label = health_scores.attrs["mode"]

    beta_proxy = cra_beta_proxy(cra_abundance, bridge)
    features = pd.DataFrame({"sample_id": health_scores.index.astype(str), health_feature: health_scores.to_numpy(dtype=float)})
    features["microbiome_health_mode"] = health_mode_label
    if not beta_proxy.empty:
        features = features.merge(
            beta_proxy[["beta_personalized_axis", "beta_mac_mean", "beta_lipid_mean"]].reset_index(),
            on="sample_id",
            how="left",
        )
        features["personalized_health_axis"] = zscore(features[health_feature]) + 0.15 * zscore(
            features["beta_personalized_axis"].fillna(0.0)
        )
    else:
        features["personalized_health_axis"] = features[health_feature]

    correlations = spearman_clinical_correlations(
        features,
        cra_host,
        list(spec.variables),
        [health_feature, "personalized_health_axis", "beta_personalized_axis"],
    )
    retention = evaluate_gmwi2_retention(
        correlations.loc[correlations["feature"].eq(health_feature)],
        correlations.loc[correlations["feature"].eq("personalized_health_axis")],
        spec.min_retained_fraction,
        spec.max_abs_loss,
    )

    section_dir = out_dir / "section2_clinical_consistency"
    section_dir.mkdir(parents=True, exist_ok=True)
    gmrepo_audit.to_csv(section_dir / "gmrepo_clinical_completeness.csv", index=False)
    cra_audit.to_csv(section_dir / "cra013939_clinical_completeness.csv", index=False)
    correlations.to_csv(section_dir / "clinical_correlations.csv", index=False)
    retention.to_csv(section_dir / "microbiome_health_retention.csv", index=False)
    features.to_csv(section_dir / "cra013939_health_beta_features.csv", index=False)

    return {
        "selected_clinical_cohort": selected,
        "gmrepo_variables_passing": int(gmrepo_audit["passes"].sum()),
        "cra013939_variables_passing": int(cra_audit["passes"].sum()),
        "microbiome_health_mode": health_mode_label,
        "retention_pass_fraction": float(retention["passes_retention"].mean()) if len(retention) else float("nan"),
        "section_dir": str(section_dir),
    }


def disease_rows(matrix: pd.DataFrame, label: str) -> pd.DataFrame:
    names = matrix.index.astype(str)
    mask = np.zeros(len(names), dtype=bool)
    for keyword in DISEASE_KEYWORDS[label]:
        mask |= pd.Series(names).str.lower().str.contains(keyword, regex=False).to_numpy()
    if label == "Health":
        mask |= pd.Series(names).str.lower().str.contains("D006262".lower(), regex=False).to_numpy()
    return matrix.loc[mask]


def build_direct_kg_edges(nutrient_metabolite: pd.DataFrame, disease_metabolite: pd.DataFrame) -> pd.DataFrame:
    nutrient_metabolite.index = nutrient_metabolite.index.astype(str)
    nutrient_metabolite.columns = nutrient_metabolite.columns.astype(str)
    disease_metabolite.index = disease_metabolite.index.astype(str)
    disease_metabolite.columns = disease_metabolite.columns.astype(str)
    common = [c for c in nutrient_metabolite.columns if c in disease_metabolite.columns]
    rows = []
    for label in DISEASE_KEYWORDS:
        d_rows = disease_rows(disease_metabolite[common], label)
        if d_rows.empty:
            continue
        disease_vector = d_rows.astype(float).mean(axis=0)
        assoc = nutrient_metabolite[common].astype(float).fillna(0.0).to_numpy() @ disease_vector.fillna(0.0).to_numpy()
        assoc = pd.Series(assoc, index=nutrient_metabolite.index)
        scale = assoc.abs().quantile(0.95)
        scale = float(scale) if np.isfinite(scale) and scale > 1e-12 else 1.0
        for nutrient, value in assoc.items():
            weight = float(np.clip(value / scale, -1.0, 1.0))
            if abs(weight) < 0.05:
                continue
            rows.append(
                {
                    "source": nutrient,
                    "target": label,
                    "source_type": "nutrient",
                    "target_type": "disease",
                    "relation": "nutrient_metabolite_disease_projection",
                    "weight": weight,
                    "evidence_source": "KEGG;GMMAD2",
                }
            )
    return pd.DataFrame(rows)


def run_section3(paths: dict[str, Path], out_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    from gmnps.knowledge_graph import (  # noqa: PLC0415
        adjudicate_labels_rule_ai,
        build_adjudication_prompt_table,
        label_metrics,
        score_nutrient_disease_paths,
    )

    beta = pd.read_parquet(paths["beta_i"])
    beta.index = beta.index.astype(str)
    beta.columns = beta.columns.astype(str)
    metadata = pd.read_parquet(paths["cmd_metadata"], columns=["sample_id", "phenotype_label", "disease"])
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata["true_label"] = [
        normalize_label(pheno) or normalize_label(dis) for pheno, dis in zip(metadata["phenotype_label"], metadata["disease"])
    ]
    truth = metadata.loc[metadata["sample_id"].isin(beta.index) & metadata["true_label"].notna(), ["sample_id", "true_label"]]
    truth = truth.loc[truth["true_label"].isin(DISEASE_KEYWORDS.keys())].drop_duplicates("sample_id")
    if len(truth) > args.kg_max_samples:
        truth = truth.sample(n=args.kg_max_samples, random_state=20260804)
    beta_subset = beta.loc[truth["sample_id"].astype(str)]

    nutrient_metabolite = pd.read_parquet(paths["nutrient_metabolite"])
    disease_metabolite = pd.read_parquet(paths["gmmad2_disease_metabolite"])
    edges = build_direct_kg_edges(nutrient_metabolite, disease_metabolite)
    path_scores, paths_all = score_nutrient_disease_paths(beta_subset, edges, top_n=10)
    prompt = build_adjudication_prompt_table(
        path_scores,
        leakage_columns=["true_label", "phenotype_label", "diagnosis", "cohort_label", "study_id"],
    )
    predictions = adjudicate_labels_rule_ai(prompt, list(DISEASE_KEYWORDS.keys()))
    metrics = label_metrics(predictions, truth)
    binary_predictions, binary_metrics = kg_binary_adjudication(
        path_scores,
        truth,
        paths["beta_diagnostics"],
        n_folds=5,
    )

    section_dir = out_dir / "section3_kg_label_adjudication"
    section_dir.mkdir(parents=True, exist_ok=True)
    edges.to_csv(section_dir / "kg_direct_nutrient_disease_edges.csv", index=False)
    path_scores.to_csv(section_dir / "kg_path_scores.csv", index=False)
    paths_all.sort_values("path_score", key=lambda s: s.abs(), ascending=False).head(5000).to_csv(
        section_dir / "kg_top_paths_detail.csv",
        index=False,
    )
    predictions.to_csv(section_dir / "kg_label_predictions.csv", index=False)
    truth.to_csv(section_dir / "kg_label_truth.csv", index=False)
    binary_predictions.to_csv(section_dir / "kg_binary_label_predictions.csv", index=False)
    write_json(section_dir / "kg_label_metrics.json", metrics)
    write_json(section_dir / "kg_binary_label_metrics.json", binary_metrics)

    return {
        "n_edges": int(len(edges)),
        "n_samples": int(metrics["n_samples"]),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "majority_baseline_accuracy": metrics["majority_baseline_accuracy"],
        "binary_accuracy": binary_metrics["accuracy"],
        "binary_balanced_accuracy": binary_metrics["balanced_accuracy"],
        "section_dir": str(section_dir),
    }


def _balanced_accuracy_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for label in [0, 1]:
        mask = y_true == label
        if mask.any():
            scores.append(float((y_pred[mask] == label).mean()))
    return float(np.mean(scores)) if scores else float("nan")


def _best_threshold(train_score: np.ndarray, train_y: np.ndarray) -> float:
    candidates = np.unique(np.quantile(train_score, np.linspace(0.05, 0.95, 37)))
    best_threshold = float(np.median(train_score))
    best_score = -1.0
    for threshold in candidates:
        pred = (train_score >= threshold).astype(int)
        score = _balanced_accuracy_from_arrays(train_y, pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def kg_binary_adjudication(
    path_scores: pd.DataFrame,
    truth: pd.DataFrame,
    beta_diagnostics_path: Path,
    n_folds: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Cross-validated KG adjudication of Health versus disease status."""
    from gmnps.validation.response_prediction import group_folds  # noqa: PLC0415

    pivot = path_scores.pivot_table(
        index="sample_id",
        columns="disease",
        values="evidence_score",
        aggfunc="sum",
        fill_value=0.0,
    )
    diagnostics = pd.read_csv(beta_diagnostics_path)
    diagnostics["sample_id"] = diagnostics["sample_id"].astype(str)
    diag = diagnostics.set_index("sample_id")
    frame = truth.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame = frame.loc[frame["sample_id"].isin(pivot.index)].copy()
    frame["is_disease"] = frame["true_label"].ne("Health").astype(int)
    disease_cols = [c for c in pivot.columns if c != "Health"]
    frame = frame.set_index("sample_id")
    frame["kg_disease_score"] = pivot.reindex(frame.index)[disease_cols].max(axis=1) if disease_cols else 0.0
    frame["kg_health_score"] = pivot.reindex(frame.index)["Health"] if "Health" in pivot.columns else 0.0
    if "baseline_health_index" in diag.columns:
        health = pd.to_numeric(diag.reindex(frame.index)["baseline_health_index"], errors="coerce")
        frame["health_index_component"] = -zscore(health.fillna(health.median()))
    else:
        frame["health_index_component"] = 0.0
    frame["decision_score"] = zscore(frame["kg_disease_score"] - frame["kg_health_score"]) + frame["health_index_component"]

    y = frame["is_disease"].to_numpy(dtype=int)
    score = frame["decision_score"].to_numpy(dtype=float)
    folds = group_folds(pd.Series(frame.index, index=frame.index), min(n_folds, len(frame)), 20260804)
    pred = np.zeros(len(frame), dtype=int)
    thresholds = []
    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(len(frame)), test_idx)
        threshold = _best_threshold(score[train_idx], y[train_idx])
        thresholds.append(threshold)
        pred[test_idx] = (score[test_idx] >= threshold).astype(int)
    out = frame.reset_index()[["sample_id", "true_label", "kg_disease_score", "kg_health_score", "decision_score"]]
    out["true_binary_label"] = np.where(y == 1, "Disease", "Health")
    out["predicted_binary_label"] = np.where(pred == 1, "Disease", "Health")
    accuracy = float((pred == y).mean())
    majority = float(max(np.mean(y == 0), np.mean(y == 1)))
    metrics = {
        "n_samples": int(len(frame)),
        "accuracy": accuracy,
        "balanced_accuracy": _balanced_accuracy_from_arrays(y, pred),
        "majority_baseline_accuracy": majority,
        "mean_cv_threshold": float(np.mean(thresholds)),
    }
    return out, metrics


def pca_features(frame: pd.DataFrame, n_components: int, prefix: str) -> pd.DataFrame:
    x = frame.astype(float).copy()
    x = x.fillna(x.median(axis=0))
    x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0).replace(0, 1.0)
    u, s, _ = np.linalg.svd(x.to_numpy(dtype=float), full_matrices=False)
    n = min(n_components, u.shape[1])
    return pd.DataFrame(u[:, :n] * s[:n], index=frame.index, columns=[f"{prefix}{i+1}" for i in range(n)])


def design_matrix(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    x = data.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(axis=0))
    x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0).replace(0, 1.0)
    x.insert(0, "intercept", 1.0)
    return x


def run_cv_models(data: pd.DataFrame, outcome: str, feature_sets: dict[str, list[str]], folds: int, alpha: float) -> pd.DataFrame:
    from gmnps.validation.response_prediction import (  # noqa: PLC0415
        fit_ridge_predict,
        group_folds,
    )

    subset = data.dropna(subset=[outcome, "subject_id"]).copy()
    fold_indices = group_folds(subset["subject_id"], min(folds, subset["subject_id"].nunique()), 20260804)
    rows = []
    for fold_id, test_idx in enumerate(fold_indices):
        test_mask = np.zeros(len(subset), dtype=bool)
        test_mask[test_idx] = True
        train = subset.loc[~test_mask]
        test = subset.loc[test_mask]
        for model, columns in feature_sets.items():
            x_train = design_matrix(train, columns)
            x_test = design_matrix(test, columns)
            x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)
            pred = fit_ridge_predict(
                x_train.to_numpy(dtype=float),
                train[outcome].to_numpy(dtype=float),
                x_test.to_numpy(dtype=float),
                alpha,
            )
            for row_id, true, hat in zip(test.index, test[outcome].to_numpy(dtype=float), pred):
                rows.append(
                    {
                        "row_id": f"{outcome}:{row_id}",
                        "outcome": outcome,
                        "fold": fold_id,
                        "model": model,
                        "subject_id": test.at[row_id, "subject_id"],
                        "y_true": float(true),
                        "y_pred": float(hat),
                    }
                )
    return pd.DataFrame(rows)


def eligible_metabolite_outcomes(metabolites: pd.DataFrame, n_outcomes: int) -> list[str]:
    """Choose deterministic metabolite candidates without inspecting GMNPS features."""

    meta = metabolites.set_index("subject_id", drop=False)
    numeric = meta.drop(columns=["subject_id"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    candidates = []
    for column in numeric.columns:
        values = numeric[column].dropna()
        if len(values) < 100 or values.nunique() < 5:
            continue
        candidates.append(str(column))
    return sorted(candidates)[:n_outcomes]


def read_response_outcome_list(path: Path) -> list[str]:
    """Read one pre-registered response outcome per non-comment line."""

    outcomes = []
    for line in path.read_text().splitlines():
        outcome = line.split("#", maxsplit=1)[0].strip()
        if outcome:
            outcomes.append(outcome)
    return list(dict.fromkeys(outcomes))


def bootstrap_ci_p_value(estimate: float, ci_low: float, ci_high: float) -> float:
    """Approximate a two-sided p-value from a 95% bootstrap confidence interval."""

    if not all(np.isfinite([estimate, ci_low, ci_high])):
        return math.nan
    standard_error = (ci_high - ci_low) / (2 * 1.96)
    if standard_error <= 0:
        return 0.0 if estimate != 0 else 1.0
    return float(math.erfc(abs(estimate / standard_error) / math.sqrt(2.0)))


def run_section4(paths: dict[str, Path], out_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    from gmnps.validation.response_benchmark import (  # noqa: PLC0415
        fdr_bh,
        split_outcome_discovery_validation,
        summarize_added_value,
    )
    from gmnps.validation.response_prediction import (  # noqa: PLC0415
        paired_bootstrap_delta,
        regression_metrics,
        required_ablation_models,
    )

    host = normalized_clinical_metadata(pd.read_parquet(paths["cra_host"]))
    food = pd.read_parquet(paths["cra_food_intake"])
    abundance = pd.read_parquet(paths["cra_abundance"])
    metabolites = pd.read_parquet(paths["cra_metabolites"]) if "cra_metabolites" in paths else None
    bridge = pd.read_parquet(paths["nutrient_bridge"])
    bridge.index = bridge.index.astype(str)
    bridge.columns = bridge.columns.astype(str)

    host = host.set_index("subject_id", drop=False)
    food = food.set_index("subject_id", drop=False)
    abundance = abundance.set_index("subject_id", drop=False)
    food_features = food.drop(columns=["subject_id"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    micro_features = abundance.drop(columns=["subject_id"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    food_pc = pca_features(food_features, 6, "food_pc_")
    micro_pc = pca_features(micro_features, 8, "micro_pc_")
    beta_proxy = cra_beta_proxy(abundance.reset_index(drop=True), bridge)
    beta_proxy.index = beta_proxy.index.astype(str)

    data = host.join(food_pc, how="inner").join(micro_pc, how="inner").join(
        beta_proxy[["beta_personalized_axis", "beta_mac_mean", "beta_lipid_mean"]],
        how="left",
    )
    data["sex_female"] = pd.to_numeric(data.get("sex_female", 0.0), errors="coerce").fillna(0.0)
    rng = np.random.default_rng(20260804)
    data["population_beta"] = float(data["beta_personalized_axis"].mean())
    data["shuffled_beta"] = rng.permutation(data["beta_personalized_axis"].fillna(0.0).to_numpy())
    data["random_beta"] = rng.normal(0, 1, size=len(data))
    data["high_compression_beta"] = 0.2 * data["beta_personalized_axis"].fillna(0.0)
    data["kg_evidence_axis"] = zscore(data["beta_personalized_axis"].fillna(0.0)) + 0.1 * zscore(data["micro_pc_1"])
    selected_metabolites: list[str] = []
    metabolite_split: dict[str, str] = {}
    metabolite_source = "not_available"
    requested_outcomes = (
        read_response_outcome_list(args.response_outcome_list)
        if args.response_outcome_list is not None
        else None
    )
    if metabolites is not None:
        metab = metabolites.set_index("subject_id", drop=False)
        if requested_outcomes is not None:
            selected_metabolites = [outcome for outcome in requested_outcomes if outcome in metab.columns]
            metabolite_split = {outcome: "validation" for outcome in selected_metabolites}
            metabolite_source = "prespecified_list"
        else:
            candidates = eligible_metabolite_outcomes(metab, args.metabolite_outcomes)
            split = split_outcome_discovery_validation(candidates, args.response_outcome_split_seed)
            selected_metabolites = [*split["discovery"], *split["validation"]]
            metabolite_split = {
                outcome: split_name
                for split_name, outcomes in split.items()
                for outcome in outcomes
            }
            metabolite_source = "deterministic_discovery_validation_split"
        if selected_metabolites:
            data = data.join(metab[selected_metabolites], how="left")

    base_cols = [c for c in ["age", "bmi", "sex_female"] if c in data.columns]
    food_cols = list(food_pc.columns)
    micro_cols = list(micro_pc.columns)
    feature_sets = {
        "FCS2_only": base_cols,
        "food_nutrients": base_cols + food_cols,
        "microbiome_only": base_cols + micro_cols,
        "food_plus_microbiome": base_cols + food_cols + micro_cols,
        "GMNPS_full": base_cols + food_cols + micro_cols + ["beta_personalized_axis", "beta_mac_mean", "beta_lipid_mean", "kg_evidence_axis"],
        "population_beta": base_cols + food_cols + micro_cols + ["population_beta"],
        "shuffled_beta": base_cols + food_cols + micro_cols + ["shuffled_beta"],
        "random_beta": base_cols + food_cols + micro_cols + ["random_beta"],
        "no_personalized_offset": base_cols + food_cols + micro_cols + ["kg_evidence_axis"],
        "no_KG": base_cols + food_cols + micro_cols + ["beta_personalized_axis", "beta_mac_mean", "beta_lipid_mean"],
        "high_compression": base_cols + food_cols + micro_cols + ["high_compression_beta"],
    }
    missing = set(required_ablation_models()).difference(feature_sets)
    if missing:
        raise RuntimeError(f"missing required ablation models: {sorted(missing)}")

    if requested_outcomes is not None:
        outcome_sources = {
            outcome: ("prespecified_list", "validation")
            for outcome in requested_outcomes
            if outcome in data.columns
        }
        if not outcome_sources:
            raise ValueError("--response-outcome-list contains no outcomes available in the response data")
    else:
        outcome_sources = {
            outcome: ("prespecified_primary", "validation")
            for outcome in PRIMARY_OUTCOMES
            if outcome in data.columns
        }
        outcome_sources.update(
            {
                outcome: (metabolite_source, metabolite_split[outcome])
                for outcome in selected_metabolites
                if outcome in data.columns
            }
        )
    predictions = []
    for outcome in outcome_sources:
        predictions.append(run_cv_models(data, outcome, feature_sets, args.response_folds, args.ridge_alpha))
    pred = pd.concat(predictions, ignore_index=True)
    metrics_rows = []
    for (outcome, model), frame in pred.groupby(["outcome", "model"], sort=False):
        metrics_rows.append(
            {
                "outcome": outcome,
                "outcome_source": outcome_sources[outcome][0],
                "split": outcome_sources[outcome][1],
                "model": model,
                "n": int(len(frame)),
                **regression_metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()),
            }
        )
    metrics = pd.DataFrame(metrics_rows)
    comparison_rows = []
    for outcome in sorted(pred["outcome"].unique()):
        p = pred.loc[pred["outcome"].eq(outcome)].copy()
        for model, baseline in [
            ("GMNPS_full", "food_plus_microbiome"),
            ("GMNPS_full", "population_beta"),
            ("GMNPS_full", "shuffled_beta"),
            ("GMNPS_full", "random_beta"),
            ("GMNPS_full", "no_personalized_offset"),
            ("GMNPS_full", "no_KG"),
            ("GMNPS_full", "high_compression"),
        ]:
            for metric in ["rmse", "r2", "spearman"]:
                est, lo, hi = paired_bootstrap_delta(p, model, baseline, metric, "subject_id", args.bootstrap, 20260804)
                comparison_rows.append(
                    {
                        "outcome": outcome,
                        "outcome_source": outcome_sources[outcome][0],
                        "split": outcome_sources[outcome][1],
                        "model": model,
                        "baseline": baseline,
                        "metric": f"delta_{metric}",
                        "estimate_model_minus_baseline": est,
                        "ci_low": lo,
                        "ci_high": hi,
                        "p_value": bootstrap_ci_p_value(est, lo, hi),
                    }
                )
    comparisons = pd.DataFrame(comparison_rows)
    added_value = (
        comparisons.loc[
            comparisons["baseline"].eq("food_plus_microbiome")
            & comparisons["metric"].eq("delta_spearman")
            & comparisons["split"].eq("validation")
        ]
        .sort_values("estimate_model_minus_baseline", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    added_value["fdr_q_value"] = fdr_bh(added_value["p_value"])
    added_value["improved_over_food_plus_microbiome"] = (
        added_value["estimate_model_minus_baseline"].gt(0)
        & added_value["fdr_q_value"].le(0.05)
    )
    added_value_summary = summarize_added_value(added_value)

    section_dir = out_dir / "section4_response_prediction"
    section_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(section_dir / "fold_predictions.parquet", index=False)
    metrics.to_csv(section_dir / "model_metrics.csv", index=False)
    comparisons.to_csv(section_dir / "model_comparisons.csv", index=False)
    added_value.to_csv(section_dir / "gmnps_added_value_summary.csv", index=False)
    added_value_summary.to_csv(section_dir / "gmnps_added_value_overall_summary.csv", index=False)
    data.reset_index(drop=True).loc[:, ["subject_id", *base_cols, "beta_personalized_axis", "kg_evidence_axis"]].to_csv(
        section_dir / "response_feature_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "outcome": outcome,
                "outcome_source": source,
                "split": split_name,
                "contributes_to_final_added_value_summary": split_name == "validation",
            }
            for outcome, (source, split_name) in outcome_sources.items()
        ]
    ).to_csv(section_dir / "response_outcome_diagnostics.csv", index=False)

    full = metrics.loc[metrics["model"].eq("GMNPS_full")]
    metabolite_added = added_value.loc[added_value["outcome"].isin(selected_metabolites)]
    response_fdr_pass_fraction = (
        float(added_value["improved_over_food_plus_microbiome"].mean()) if len(added_value) else float("nan")
    )
    return {
        "n_subjects": int(data["subject_id"].nunique()),
        "n_outcomes": int(metrics["outcome"].nunique()),
        "n_metabolite_outcomes": int(len(selected_metabolites)),
        "n_metabolite_outcomes_with_positive_spearman_gain": int(
            metabolite_added["improved_over_food_plus_microbiome"].sum()
        ),
        "metabolite_outcome_source": metabolite_source,
        "final_added_value_summary_split": "validation_only",
        "best_full_spearman": float(full["spearman"].max()),
        "median_full_r2": float(full["r2"].median()),
        "response_delta_spearman_fdr_pass_fraction": response_fdr_pass_fraction,
        "section_dir": str(section_dir),
    }


def data_resource_audit(paths: dict[str, Path], root: Path) -> pd.DataFrame:
    rows = []
    for key, path in paths.items():
        rows.append(
            {
                "resource": key,
                "path": str(path.relative_to(root)) if path.exists() else str(path),
                "exists": bool(path.exists()),
                "bytes": int(path.stat().st_size) if path.exists() and path.is_file() else 0,
                "sha256": sha256_file(path) if path.exists() and path.is_file() and path.stat().st_size < 200_000_000 else "not_recorded_large_or_missing",
            }
        )
    return pd.DataFrame(rows)


def latex_escape(value: object) -> str:
    text = str(value)
    for old, new in [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
    ]:
        text = text.replace(old, new)
    return text


def simple_latex_table(frame: pd.DataFrame, columns: list[str], caption: str, label: str, max_rows: int = 12) -> str:
    columns = [column for column in columns if column in frame.columns]
    if not columns:
        return rf"\noindent {latex_escape(caption)} No rows or compatible columns were available."
    view = frame.loc[:, columns].head(max_rows).copy()
    header = " & ".join(latex_escape(c) for c in columns) + r" \\"
    body = []
    for _, row in view.iterrows():
        body.append(" & ".join(latex_escape(f"{row[c]:.4g}") if isinstance(row[c], float) else latex_escape(row[c]) for c in columns) + r" \\")
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{label}}}",
            r"\begin{tabular}{" + "l" * len(columns) + r"}",
            r"\toprule",
            header,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def read_csv_with_schema(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a result table while tolerating legacy zero-byte eligible outputs."""
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame()
    return frame.reindex(columns=[*columns, *[column for column in frame.columns if column not in columns]])


def read_supplement_source(path: Path) -> pd.DataFrame:
    """Read a manifest source into a compact, renderable table preview."""
    if path.suffix.lower() == ".csv":
        return read_csv_with_schema(path, [])
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return pd.DataFrame(
                {"field": list(payload), "value": [json.dumps(value, ensure_ascii=True) for value in payload.values()]}
            )
        if isinstance(payload, list):
            return pd.json_normalize(payload)
    return pd.DataFrame()


def write_supplement(out_dir: Path, manifest: dict[str, object]) -> None:
    supplement_dir = out_dir / "supplement"
    supplement_dir.mkdir(parents=True, exist_ok=True)
    supplement_manifest = build_supplement_manifest(out_dir)
    supplement_manifest.to_csv(supplement_dir / "supplement_manifest.csv", index=False)
    population_consensus_path = out_dir / "section1_population_consensus" / "population_consensus.csv"
    population_statuses = []
    if population_consensus_path.exists():
        population_consensus = read_csv_with_schema(population_consensus_path, SECTION1_POPULATION_COLUMNS)
        population_statuses = sorted(
            str(value)
            for value in population_consensus.get("calibration_status", pd.Series(dtype=object)).dropna().unique()
        )
    calibration_status_text = ", ".join(latex_escape(status) for status in population_statuses) or "missing source"

    section_text = [
        f"\\section*{{{latex_escape(name)}}}\n{latex_escape(purpose)}"
        for name, purpose in supplement_manifest.loc[
            supplement_manifest["kind"] == "section", ["name", "purpose"]
        ].itertuples(index=False, name=None)
    ]
    table_text = []
    for row in supplement_manifest.loc[supplement_manifest["kind"] == "table"].itertuples(index=False):
        source = row.source_path if row.source_path else "No source path registered"
        blocks = [
            rf"\subsection*{{{latex_escape(row.name)}}}",
            rf"Status: \texttt{{{latex_escape(row.status)}}}. Source: \texttt{{{latex_escape(source)}}}.",
        ]
        if row.source_exists:
            source_path = out_dir / row.source_path
            preview = read_supplement_source(source_path)
            if preview.empty and not list(preview.columns):
                blocks.append("The source was available but contained no renderable rows.")
            else:
                blocks.append(
                    simple_latex_table(
                        preview,
                        list(preview.columns[:6]),
                        f"Preview of {row.name}",
                        f"supplement-preview-{len(table_text) + 1}",
                        max_rows=12,
                    )
                )
        table_text.append("\n".join(blocks))
    text = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{hyperref}",
        r"\title{Supplementary Information: Personalized Calibration of Nutrient Profiling by Gut Microbiome Signals}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        "This Supplementary Information summarizes computational experiments generated by the reproducible GMNPS personalized-calibration pipeline. The analyses use local copies of USDA FoodData Central/FNDDS food composition tables, GMrepo/cMD microbiome metadata, CRA013939 host and microbiome tables, and KEGG/GMMAD2-derived evidence matrices. Large primary data files and generated matrices are intentionally stored outside Git; compact result tables and manifests are written under the local output directory.",
        *section_text,
        f"Calibration candidate diagnostics, including {calibration_status_text} outcomes, are registered in the population-consensus source outputs.",
        *table_text,
        r"\section*{Run manifest}",
        r"\begin{verbatim}",
        json.dumps(manifest, indent=2, ensure_ascii=True)[:5000],
        r"\end{verbatim}",
        r"\end{document}",
        "",
    ]
    rendered = "\n\n".join(text)
    (supplement_dir / "supplementary_material.tex").write_text(rendered, encoding="utf-8")
    # Preserve the historical location for callers that have not adopted the manifest directory.
    (out_dir / "supplementary_material.tex").write_text(rendered, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.microbiome_health_mode == "official_gmwi2" and args.official_gmwi2_scores is None:
        raise ValueError("official_gmwi2 mode requires --official-gmwi2-scores")
    root = Path(args.root).resolve()
    paths = load_paths(root)
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)

    audit = data_resource_audit(paths, root)
    audit.to_csv(out_dir / "data_resource_audit.csv", index=False)

    section1 = run_section1(paths, out_dir, args)
    section2 = run_section2(paths, out_dir, args)
    section3 = run_section3(paths, out_dir, args)
    section4 = run_section4(paths, out_dir, args)
    readiness = build_submission_readiness_report(section1, section2, section3, section4)
    write_submission_readiness_report(out_dir, readiness)

    manifest = {
        "analysis": "personalized_calibration_experiments",
        "source_revision": source_revision(root),
        "python_version": sys.version.split()[0],
        "output_dir": str(out_dir.relative_to(root)),
        "sections": {
            "population_consensus": section1,
            "clinical_consistency": section2,
            "kg_label_adjudication": section3,
            "response_prediction": section4,
        },
        "submission_readiness": readiness,
        "interpretation_boundary": "Results are computational validation analyses from available local resources; beta_i and GMNPS offsets are microbiome-derived calibration signals, not causal treatment effects.",
    }
    write_json(out_dir / "experiment_manifest.json", manifest)
    write_supplement(out_dir, manifest)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Run GMNPS personalized-calibration experiments")
    parser.add_argument("--root", default=str(root))
    parser.add_argument("--output-dir", default="outputs/personalized_calibration_experiments")
    parser.add_argument("--amplifications", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--delta-cap", type=float, default=20.0)
    parser.add_argument("--calibration-temperature", type=float, default=1.0)
    parser.add_argument("--group-penalty", type=float, default=0.0)
    parser.add_argument("--legacy-score-matrix", action="store_true")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--kg-max-samples", type=int, default=5000)
    parser.add_argument("--response-folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--metabolite-outcomes", type=int, default=12)
    parser.add_argument(
        "--response-outcome-list",
        type=Path,
        help="Path to a pre-registered, one-outcome-per-line response outcome list.",
    )
    parser.add_argument("--response-outcome-split-seed", type=int, default=20260805)
    parser.add_argument("--official-gmwi2-scores", type=Path)
    parser.add_argument(
        "--microbiome-health-mode",
        choices=["official_gmwi2", "genus_proxy"],
        default="genus_proxy",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
