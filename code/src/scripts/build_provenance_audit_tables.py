#!/usr/bin/env python3
"""Build provenance audit tables for the GMNPS Nature Food evidence chain."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gmnps.scoring.masks import PRIMARY_EXCLUDED_FROM_CHANNELS, build_channel_vectors  # noqa: E402


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_row(root: Path, dataset_id: str, role: str, path: Path, source: str, version: str, notes: str) -> dict[str, object]:
    exists = path.exists()
    return {
        "dataset_id": dataset_id,
        "role": role,
        "local_path": str(path.relative_to(root)) if exists and path.is_relative_to(root) else str(path),
        "source": source,
        "version_or_release": version,
        "access_status": "local_file_present" if exists else "missing",
        "bytes": path.stat().st_size if exists else "",
        "sha256": sha256_file(path) if exists and path.is_file() else "",
        "notes": notes,
    }


def build_input_data_provenance(root: Path, out_dir: Path) -> pd.DataFrame:
    l6 = root / "data/project_data/predict_multi/L6_layer_c_fndds/processed"
    l7 = root / "data/project_data/predict_multi/L7_nutrient_bridge"
    l9 = root / "data/project_data/predict_multi/L9_food_compass"
    rows = [
        file_row(
            root,
            "FNDDS_2021_2023_raw",
            "food_composition_raw",
            root / "data/project_data/predict_multi/L6_layer_c_fndds/raw/FNDDS_Nutrient_Values.xlsx",
            "USDA FSRG/FoodData Central download dataset",
            "FNDDS 2021-2023",
            "Primary current food-composition release used by processed L6 layer.",
        ),
        file_row(
            root,
            "FNDDS_harmonised_full",
            "food_composition_processed",
            l6 / "N_food_nutrient_full.parquet",
            "USDA FSRG FNDDS releases",
            "2015-2016 to 2021-2023 harmonised",
            "Harmonised food-nutrient matrix before FCS2 imputation.",
        ),
        file_row(
            root,
            "FNDDS_imputed_for_FCS2",
            "food_composition_processed",
            l6 / "N_food_nutrient_imputed.parquet",
            "USDA FSRG FNDDS releases plus code-prefix median imputation",
            "2015-2016 to 2021-2023 harmonised/imputed",
            "Primary nutrient matrix used by v4 scoring.",
        ),
        file_row(
            root,
            "Food_Compass_2_0_Table_S5",
            "baseline_nps_prior",
            l9 / "food_compass_scores.csv",
            "Food Compass 2.0 Supplementary Table S5 extraction",
            "2024 article supplement",
            "Selected population NPS prior and food universe.",
        ),
        file_row(
            root,
            "CMD_microbiome_profiles",
            "microbiome_source",
            root / "data/project_data/predict_multi/L1_microbiome/cmd_processed/layer_a_master.parquet",
            "curatedMetagenomicData-derived local pool",
            "local processed 2026-04/05",
            "Source pool for microbiome-derived calibration profiles.",
        ),
        file_row(
            root,
            "GMMAD2_bridge",
            "microbe_metabolite_knowledge_graph",
            root / "data/project_data/predict_multi/L4_knowledge_graph/gmmad2_processed/B_genus_metabolite.parquet",
            "GMMAD v2 processed locally",
            "local processed",
            "Knowledge graph used to bridge nutrients, metabolites and genera.",
        ),
        file_row(
            root,
            "W_personalized",
            "microbiome_derived_calibration_weights",
            l7 / "W_personalized.parquet",
            "regularized nutrient-genus bridge reconstruction",
            "L7 local output",
            "15492 x 65 calibration feature matrix, not matched food-response cohort.",
        ),
        file_row(
            root,
            "W_personalized_beta_i",
            "microbiome_derived_nutrient_response_weights",
            root / "data/project_data/predict_multi/L7_nutrient_bridge_beta_i/W_personalized.parquet",
            "GMWI2-style health-index finite-difference beta_i",
            "local reproducible beta_i output",
            "beta_i,k = finite-difference change in fitted gut microbiome health index after nutrient-linked perturbation.",
        ),
    ]
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "input_data_provenance.csv", index=False)
    return result


def build_food_alignment_audit(root: Path, out_dir: Path) -> pd.DataFrame:
    fcs = pd.read_csv(root / "data/project_data/predict_multi/L9_food_compass/food_compass_scores.csv")
    nutrient = pd.read_parquet(root / "data/project_data/predict_multi/L6_layer_c_fndds/processed/N_food_nutrient_imputed.parquet")
    food_meta = pd.read_parquet(root / "data/project_data/predict_multi/L6_layer_c_fndds/processed/food_meta_imputed.parquet")
    summary = pd.read_csv(root / "outputs/gmnps_v4/tables/S_food_summary_v4.csv")
    fcs["foodcode"] = fcs["foodcode"].astype(str)
    nutrient_codes = set(nutrient.index.astype(str))
    included_codes = set(summary["food_id"].astype(str))
    duplicated = set(fcs.loc[fcs["foodcode"].duplicated(keep=False), "foodcode"])
    meta = food_meta.copy()
    meta["Food code"] = meta["Food code"].astype(str)
    meta = meta.drop_duplicates("Food code", keep="first").set_index("Food code")
    rows = []
    for _, row in fcs.iterrows():
        code = row["foodcode"]
        meta_row = meta.loc[code] if code in meta.index else None
        rows.append(
            {
                "foodcode": code,
                "description": row["description"],
                "food_group": row["food_group"],
                "FCS2": row["FCS_2_0"],
                "FCS_source_page": row.get("source_page", ""),
                "duplicate_foodcode_flag": code in duplicated,
                "nutrient_vector_available": code in nutrient_codes,
                "included_primary_run": code in included_codes,
                "exclusion_reason": "" if code in included_codes else "missing_nutrient_vector_or_duplicate_filter",
                "fndds_release": "" if meta_row is None else meta_row.get("fndds_release", ""),
                "imputation_level": "" if meta_row is None else meta_row.get("imputation_level", ""),
                "WWEIA_category_number": "" if meta_row is None else meta_row.get("WWEIA Category number", ""),
                "WWEIA_category_description": "" if meta_row is None else meta_row.get("WWEIA Category description", ""),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "food_alignment_audit.csv", index=False)
    return result


def build_beta_weight_audit(root: Path, out_dir: Path) -> pd.DataFrame:
    weights = pd.read_parquet(root / "data/project_data/predict_multi/L7_nutrient_bridge/W_personalized.parquet")
    r2 = pd.read_csv(root / "data/project_data/predict_multi/L7_nutrient_bridge/W_reconstruction_r2.csv")
    sparsity = pd.read_csv(root / "data/project_data/predict_multi/L7_nutrient_bridge/W_sparsity.csv")
    diagnostics = read_json(root / "data/project_data/predict_multi/L7_nutrient_bridge/W_solver_diagnostics.json")
    r2 = r2.rename(columns={r2.columns[1]: "reconstruction_r2"})
    sparsity = sparsity.rename(columns={sparsity.columns[1]: "sparsity"})
    nnz = (weights.abs() > 1e-12).sum(axis=1).rename("n_nonzero_beta").reset_index()
    nnz = nnz.rename(columns={weights.index.name or "index": "sample_id"})
    audit = r2.merge(sparsity, on="sample_id", how="outer").merge(nnz, on="sample_id", how="outer")
    audit["n_nutrients"] = weights.shape[1]
    audit["alpha"] = diagnostics["config"]["alpha"]
    audit["l1_ratio"] = diagnostics["config"]["l1_ratio"]
    audit["max_iter"] = diagnostics["config"]["max_iter"]
    audit["n_genera_used"] = diagnostics["config"]["n_genera"]
    audit["interpretation_boundary"] = "regularized calibration features, not validated nutrient-response effects"
    audit.to_csv(out_dir / "beta_weight_audit.csv", index=False)
    return audit


def build_nutrient_processing_mask_audit(root: Path, out_dir: Path) -> pd.DataFrame:
    scoring_manifest = read_json(root / "outputs/gmnps_v4/manifests/scoring_manifest_v4.json")
    weights = pd.read_parquet(root / "data/project_data/predict_multi/L7_nutrient_bridge/W_personalized.parquet")
    nutrients = pd.read_parquet(root / "data/project_data/predict_multi/L6_layer_c_fndds/processed/N_food_nutrient_imputed.parquet")
    nutrient_meta = pd.read_parquet(root / "data/project_data/predict_multi/L6_layer_c_fndds/processed/nutrient_meta.parquet")
    coverage = read_json(root / "data/project_data/predict_multi/L7_nutrient_bridge/bridge_coverage.json")
    provenance = pd.read_csv(root / "data/project_data/predict_multi/L7_nutrient_bridge/bridge_provenance.csv")
    mapped_nutrients = set(provenance["nutrient"].dropna().astype(str))
    masks = build_channel_vectors([c for c in weights.columns if c in nutrients.columns])
    channel = {}
    for nutrient, is_mac, is_lipid in zip(masks["columns"], masks["mac"], masks["lipid"]):
        channel[nutrient] = "MAC" if is_mac else "LIPID" if is_lipid else "OTHER"
    meta = nutrient_meta.set_index("nutrient")
    heavy = set(scoring_manifest["preprocessing"].get("heavy_tailed_cols", []))
    rows = []
    for nutrient in weights.columns:
        bridge_rows = provenance.loc[provenance["nutrient"].astype(str).eq(nutrient)]
        rows.append(
            {
                "nutrient": nutrient,
                "unit": meta.at[nutrient, "unit"] if nutrient in meta.index else "",
                "nutrient_group": meta.at[nutrient, "group"] if nutrient in meta.index else "",
                "in_weight_matrix": nutrient in weights.columns,
                "in_nutrient_matrix": nutrient in nutrients.columns,
                "log1p_applied": nutrient in heavy,
                "winsor_lo": scoring_manifest["preprocessing"]["winsor_lo"],
                "winsor_hi": scoring_manifest["preprocessing"]["winsor_hi"],
                "robust_clip_iqr": scoring_manifest["preprocessing"]["robust_scaler_clip_iqr"],
                "channel": channel.get(nutrient, "NOT_SHARED"),
                "excluded_reason": "expert_revised_exclusion" if nutrient in PRIMARY_EXCLUDED_FROM_CHANNELS else "",
                "mapped_to_microbiome_bridge": nutrient in mapped_nutrients,
                "bridge_tiers": ";".join(map(str, sorted(bridge_rows["tier"].dropna().unique()))) if not bridge_rows.empty else "",
                "n_bridge_metabolites": int(bridge_rows["metabolite_id"].nunique()) if not bridge_rows.empty else 0,
                "coverage_boundary": coverage["mapping_strategy"] if nutrient in mapped_nutrients else "unmapped or indirectly represented",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "nutrient_processing_mask_audit.csv", index=False)
    return result


def build_source_data_manifest(root: Path, out_dir: Path) -> None:
    git_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    files = []
    for base in [
        root / "outputs/posthoc_validation",
        root / "manuscript/nature_food_submission/source_data",
        root / "outputs/nature_source_data_v2",
    ]:
        if not base.exists():
            continue
        for path in sorted(base.glob("*")):
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md"}:
                try:
                    rel = path.relative_to(root)
                except ValueError:
                    rel = path
                files.append(
                    {
                        "source_data_file": str(rel),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "git_commit": git_commit,
                        "python_version": sys.version.split()[0],
                    }
                )
    pd.DataFrame(files).to_csv(out_dir / "source_data_reproducibility_manifest.csv", index=False)


def build_summary_report(root: Path, out_dir: Path, food_audit: pd.DataFrame, beta_audit: pd.DataFrame) -> None:
    l6_manifest = read_json(root / "data/project_data/predict_multi/L6_layer_c_fndds/processed/layer_c_manifest.json")
    full_manifest = read_json(root / "data/project_data/predict_multi/L6_layer_c_fndds/processed/layer_c_full_manifest.json")
    impute_manifest = read_json(root / "data/project_data/predict_multi/L6_layer_c_fndds/processed/imputation_manifest.json")
    beta_i_manifest_path = root / "data/project_data/predict_multi/L7_nutrient_bridge_beta_i/beta_i_manifest.json"
    beta_i_manifest = read_json(beta_i_manifest_path) if beta_i_manifest_path.is_file() else {}
    report = {
        "food_composition_primary_source": "USDA FSRG/FoodData Central FNDDS 2021-2023 raw files are present locally with SHA256 checksums.",
        "current_primary_scoring_matrix": "N_food_nutrient_imputed.parquet",
        "fndds_2021_2023_n_foods": l6_manifest["n_foods"],
        "harmonised_releases": full_manifest["releases"],
        "harmonised_exact_fcs2_coverage_pct": full_manifest["fcs_audit"]["coverage_pct"],
        "imputed_fcs2_coverage_pct": impute_manifest["fcs_audit_after_imputation"]["coverage_pct"],
        "fcs2_rows": int(len(food_audit)),
        "included_primary_run": int(food_audit["included_primary_run"].sum()),
        "missing_after_imputation": impute_manifest["fcs_audit_after_imputation"]["still_missing"],
        "historical_bridge_reconstruction_profiles": int(len(beta_audit)),
        "historical_bridge_reconstruction_r2_median": float(beta_audit["reconstruction_r2"].median()),
        "historical_bridge_reconstruction_sparsity_median": float(beta_audit["sparsity"].median()),
        "beta_i_method": beta_i_manifest.get("method"),
        "beta_i_formula": beta_i_manifest.get("formula"),
        "beta_i_profiles": beta_i_manifest.get("n_samples"),
        "beta_i_nutrients": beta_i_manifest.get("n_nutrients"),
        "beta_i_health_training_summary": beta_i_manifest.get("health_model_training_summary"),
        "beta_i_health_label_diagnostics": beta_i_manifest.get("health_label_diagnostics"),
        "beta_i_interpretation_boundary": beta_i_manifest.get("interpretation_boundary"),
        "beta_i_manifest_path": str(beta_i_manifest_path.relative_to(root)),
        "boundary": "Food composition is real USDA FNDDS-derived data; imputed nutrient vectors and beta calibration weights require explicit Methods boundaries.",
    }
    Path(out_dir / "provenance_audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build GMNPS provenance audit tables")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="outputs/provenance_audit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_input_data_provenance(root, out_dir)
    food_audit = build_food_alignment_audit(root, out_dir)
    beta_audit = build_beta_weight_audit(root, out_dir)
    build_nutrient_processing_mask_audit(root, out_dir)
    build_source_data_manifest(root, out_dir)
    build_summary_report(root, out_dir, food_audit, beta_audit)


if __name__ == "__main__":
    main()
