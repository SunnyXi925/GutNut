#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import asdict
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


HEALTH_LABEL_CONTRADICTION_POLICY = (
    "Exclude CLR-overlapping sample IDs with contradictory healthy and nonhealthy "
    "metadata evidence before deriving binary health labels or fitting the health index; "
    "do not adjudicate them to either class. Excluded IDs are omitted only from health-index "
    "training labels and remain in the beta_i scoring output."
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "pyarrow", "scikit-learn", "joblib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def source_revision(root: Path) -> dict[str, str | bool]:
    """Return the complete Git revision and source-tree state at build start."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if commit.returncode != 0 or not commit.stdout.strip():
        raise RuntimeError(f"unable to determine source revision for beta_i build at {root}")

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0:
        raise RuntimeError(f"unable to determine source-tree state for beta_i build at {root}")

    return {"sha": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}


def prepare_health_labels(
    metadata: pd.DataFrame,
    clr_index: pd.Index,
) -> tuple[pd.Series, dict[str, object], pd.DataFrame]:
    """Audit CLR-overlapping metadata and exclude contradictory sample IDs."""
    required = {"sample_id", "phenotype_label", "disease"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")

    frame = metadata.loc[:, ["sample_id", "phenotype_label", "disease"]].copy()
    sample_ids = frame["sample_id"].where(frame["sample_id"].notna(), "").astype(str).str.strip()
    clr_sample_ids = pd.Index(clr_index.astype(str))
    overlapping = frame.loc[sample_ids.isin(clr_sample_ids)].copy()
    overlapping["sample_id"] = sample_ids.loc[overlapping.index]

    exclusions: list[dict[str, str]] = []
    for sample_id, group in overlapping.groupby("sample_id", sort=False):
        try:
            derive_binary_health_labels(group)
        except ValueError as error:
            if "contradictory health metadata" not in str(error) and "conflicting health labels" not in str(error):
                raise
            reason = (
                "contradictory_health_metadata"
                if "contradictory health metadata" in str(error)
                else "conflicting_duplicate_health_labels"
            )
            exclusions.append({"sample_id": str(sample_id), "reason": reason})

    exclusion_audit = pd.DataFrame(exclusions, columns=["sample_id", "reason"])
    excluded_sample_ids = exclusion_audit["sample_id"]

    labels = derive_binary_health_labels(
        overlapping.loc[~overlapping["sample_id"].isin(excluded_sample_ids)]
    )
    diagnostics: dict[str, object] = {
        "health_label_contradiction_policy": HEALTH_LABEL_CONTRADICTION_POLICY,
        "n_clr_overlapping_metadata_rows": int(len(overlapping)),
        "n_excluded_contradictory_sample_ids": int(len(excluded_sample_ids)),
        "n_derived_labeled_samples": int(len(labels)),
        "n_healthy": int(labels.sum()),
        "n_nonhealthy": int((1 - labels).sum()),
        "exclusion_scope": (
            "Excluded sample IDs are omitted from health-index training labels but remain "
            "in W_personalized.parquet beta_i scoring output."
        ),
    }
    return labels, diagnostics, exclusion_audit


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    out_dir = Path(args.output_dir).resolve()
    revision = source_revision(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    m_clr_path = root / "data/project_data/predict_multi/L5_gmnps_pipeline_v2/M_clr.parquet"
    metadata_path = root / "data/project_data/predict_multi/L1_microbiome/cmd_processed/layer_a_master.parquet"
    bridge_path = root / "data/project_data/predict_multi/L7_nutrient_bridge/B_nutrient_genus.parquet"
    nutrient_index_path = root / "data/project_data/predict_multi/L8_inference_bundle_v2/nutrient_index.json"

    clr = pd.read_parquet(m_clr_path)
    clr.index = clr.index.astype(str)
    metadata = pd.read_parquet(metadata_path, columns=["sample_id", "phenotype_label", "disease"])
    labels, label_diagnostics, exclusion_audit = prepare_health_labels(metadata, clr.index)
    health_config = HealthIndexConfig(
        c_value=args.c_value,
        max_iter=args.max_iter,
        random_state=args.random_state,
        training_backend=args.health_backend,
    )
    model = fit_health_index(clr, labels, health_config)
    health_model_path = out_dir / "health_index.joblib"
    save_health_index(model, health_model_path, serialization_backend=args.serialization_backend)
    health_model_manifest_path = Path(f"{health_model_path}.manifest.json")

    nutrient_order = json.loads(nutrient_index_path.read_text(encoding="utf-8"))
    bridge = pd.read_parquet(bridge_path).reindex(index=nutrient_order).fillna(0.0)
    perturbation_config = NutrientPerturbationConfig()
    perturb = build_nutrient_perturbations(bridge, model, perturbation_config)
    perturb = perturb.reindex(index=nutrient_order).fillna(0.0)
    beta_config = BetaEstimatorConfig(
        dose=args.dose,
        clip_abs_beta=args.clip_abs_beta,
        batch_size=args.batch_size,
    )
    beta, diagnostics = compute_beta_matrix(
        clr,
        perturb,
        model,
        beta_config,
    )
    beta = beta.reindex(columns=nutrient_order)
    beta.to_parquet(out_dir / "W_personalized.parquet")
    perturb.to_parquet(out_dir / "nutrient_perturbations.parquet")
    diagnostics.to_csv(out_dir / "sample_beta_diagnostics.csv", index=False)
    summarize_perturbations(perturb, perturbation_config).to_csv(
        out_dir / "nutrient_perturbation_summary.csv", index=False
    )
    exclusion_audit_path = out_dir / "excluded_health_label_samples.csv"
    exclusion_audit.to_csv(exclusion_audit_path, index=False)

    files = [
        out_dir / "W_personalized.parquet",
        health_model_path,
        health_model_manifest_path,
        out_dir / "nutrient_perturbations.parquet",
        out_dir / "sample_beta_diagnostics.csv",
        out_dir / "nutrient_perturbation_summary.csv",
        exclusion_audit_path,
    ]
    manifest = {
        "method": "GMWI2-style health-index finite-difference beta_i",
        "formula": "beta_i,k = (H(M_i + dose * P_k) - H(M_i)) / dose",
        "n_samples": int(beta.shape[0]),
        "n_nutrients": int(beta.shape[1]),
        "n_genera": int(clr.shape[1]),
        "health_model_training_summary": model.training_summary,
        "health_label_diagnostics": label_diagnostics,
        "health_index_config": asdict(health_config),
        "nutrient_perturbation_config": asdict(perturbation_config),
        "beta_estimator_config": asdict(beta_config),
        "health_model_manifest": {
            "path": health_model_manifest_path.name,
            "sha256": sha256_file(health_model_manifest_path),
        },
        "health_training_backend": model.training_summary["training_backend"],
        "health_training_solver": model.training_summary["training_solver"],
        "health_model_serialization_backend": args.serialization_backend,
        "excluded_health_label_samples": {
            "path": exclusion_audit_path.name,
            "sha256": sha256_file(exclusion_audit_path),
            "scope": (
                "Omitted from health-index training labels only; retained in beta_i scoring output."
            ),
        },
        "input_files": {
            "M_clr": str(m_clr_path.relative_to(root)),
            "metadata": str(metadata_path.relative_to(root)),
            "B_nutrient_genus": str(bridge_path.relative_to(root)),
            "nutrient_index": str(nutrient_index_path.relative_to(root)),
        },
        "input_sha256": {
            "M_clr": sha256_file(m_clr_path),
            "metadata": sha256_file(metadata_path),
            "B_nutrient_genus": sha256_file(bridge_path),
            "nutrient_index": sha256_file(nutrient_index_path),
        },
        "output_sha256": {path.name: sha256_file(path) for path in files},
        "source_revision": revision,
        "git_commit": revision["sha"],
        "git_dirty": revision["dirty"],
        "python_version": sys.version.split()[0],
        "package_versions": package_versions(),
        "interpretation_boundary": "beta_i is an evidence-oriented calibration direction measuring health-index finite-difference response to nutrient-linked microbiome perturbations; it is not a causal nutrient effect.",
    }
    write_json(out_dir / "beta_i_manifest.json", manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build reproducible GMNPS beta_i nutrient response weights")
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--root", default=str(root))
    parser.add_argument("--output-dir", default=str(root / "data/project_data/predict_multi/L7_nutrient_bridge_beta_i"))
    parser.add_argument("--c-value", type=float, default=0.005)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=20260803)
    parser.add_argument("--health-backend", choices=("numpy", "sklearn"), default="numpy")
    parser.add_argument("--serialization-backend", choices=("pickle", "joblib"), default="pickle")
    parser.add_argument("--dose", type=float, default=0.1)
    parser.add_argument("--clip-abs-beta", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
