#!/usr/bin/env python3
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

from gmnps.beta_i.beta_estimator import BetaEstimatorConfig, compute_beta_matrix
from gmnps.beta_i.health_index import HealthIndexConfig, derive_binary_health_labels, fit_health_index, save_health_index
from gmnps.beta_i.nutrient_perturbation import (
    NutrientPerturbationConfig,
    build_nutrient_perturbations,
    summarize_perturbations,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    m_clr_path = root / "data/project_data/predict_multi/L5_gmnps_pipeline_v2/M_clr.parquet"
    metadata_path = root / "data/project_data/predict_multi/L1_microbiome/cmd_processed/layer_a_master.parquet"
    bridge_path = root / "data/project_data/predict_multi/L7_nutrient_bridge/B_nutrient_genus.parquet"
    nutrient_index_path = root / "data/project_data/predict_multi/L8_inference_bundle_v2/nutrient_index.json"

    clr = pd.read_parquet(m_clr_path)
    clr.index = clr.index.astype(str)
    metadata = pd.read_parquet(metadata_path, columns=["sample_id", "phenotype_label", "disease"])
    labels = derive_binary_health_labels(metadata)
    health_config = HealthIndexConfig(c_value=args.c_value, max_iter=args.max_iter, random_state=args.random_state)
    model = fit_health_index(clr, labels, health_config)
    save_health_index(model, out_dir / "health_index.joblib")

    nutrient_order = json.loads(nutrient_index_path.read_text(encoding="utf-8"))
    bridge = pd.read_parquet(bridge_path).reindex(index=nutrient_order).fillna(0.0)
    perturb = build_nutrient_perturbations(bridge, model, NutrientPerturbationConfig())
    perturb = perturb.reindex(index=nutrient_order).fillna(0.0)
    beta, diagnostics = compute_beta_matrix(
        clr,
        perturb,
        model,
        BetaEstimatorConfig(dose=args.dose, clip_abs_beta=args.clip_abs_beta, batch_size=args.batch_size),
    )
    beta = beta.reindex(columns=nutrient_order)
    beta.to_parquet(out_dir / "W_personalized.parquet")
    perturb.to_parquet(out_dir / "nutrient_perturbations.parquet")
    diagnostics.to_csv(out_dir / "sample_beta_diagnostics.csv", index=False)
    summarize_perturbations(perturb).to_csv(out_dir / "nutrient_perturbation_summary.csv", index=False)

    git_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    files = [
        out_dir / "W_personalized.parquet",
        out_dir / "health_index.joblib",
        out_dir / "nutrient_perturbations.parquet",
        out_dir / "sample_beta_diagnostics.csv",
        out_dir / "nutrient_perturbation_summary.csv",
    ]
    manifest = {
        "method": "GMWI2-style health-index finite-difference beta_i",
        "formula": "beta_i,k = (H(M_i + dose * P_k) - H(M_i)) / dose",
        "n_samples": int(beta.shape[0]),
        "n_nutrients": int(beta.shape[1]),
        "n_genera": int(clr.shape[1]),
        "health_model_training_summary": model.training_summary,
        "dose": float(args.dose),
        "clip_abs_beta": float(args.clip_abs_beta),
        "input_files": {
            "M_clr": str(m_clr_path.relative_to(root)),
            "metadata": str(metadata_path.relative_to(root)),
            "B_nutrient_genus": str(bridge_path.relative_to(root)),
            "nutrient_index": str(nutrient_index_path.relative_to(root)),
        },
        "output_sha256": {path.name: sha256_file(path) for path in files},
        "git_commit": git_commit,
        "python_version": sys.version.split()[0],
        "interpretation_boundary": "beta_i estimates health-index finite-difference response to nutrient-linked microbiome perturbations; it is not a causal nutrient effect.",
    }
    write_json(out_dir / "beta_i_manifest.json", manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build reproducible GMNPS beta_i nutrient response weights")
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--root", default=str(root))
    parser.add_argument("--output-dir", default=str(root / "data/project_data/predict_multi/L7_nutrient_bridge_beta_i"))
    parser.add_argument("--c-value", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=20260803)
    parser.add_argument("--dose", type=float, default=0.1)
    parser.add_argument("--clip-abs-beta", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
