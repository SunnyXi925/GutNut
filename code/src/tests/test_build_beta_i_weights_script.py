import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

from scripts.build_beta_i_weights import build_parser as build_beta_i_parser
from scripts.build_beta_i_weights import run, source_revision


class Args:
    def __init__(self, root: Path, output_dir: Path):
        self.root = str(root)
        self.output_dir = str(output_dir)
        self.c_value = 10.0
        self.max_iter = 500
        self.random_state = 11
        self.health_backend = "numpy"
        self.serialization_backend = "pickle"
        self.dose = 0.1
        self.clip_abs_beta = 12.0
        self.batch_size = 2
        self.beta_response_scale = "probability"
        self.beta_difference = "forward"
        self.perturbation_l2_norm = 1.0
        self.coefficient_power = 0.0


def test_build_beta_i_weights_script_writes_expected_bundle(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    l5 = root / "data/project_data/predict_multi/L5_gmnps_pipeline_v2"
    l1 = root / "data/project_data/predict_multi/L1_microbiome/cmd_processed"
    l7 = root / "data/project_data/predict_multi/L7_nutrient_bridge"
    l8 = root / "data/project_data/predict_multi/L8_inference_bundle_v2"
    for path in [l5, l1, l7, l8]:
        path.mkdir(parents=True)
    pd.DataFrame(
        {
            "Akkermansia": [2.0, 1.8, 1.5, 1.6, -1.0, -1.2, -1.4],
            "Faecalibacterium": [1.5, 1.2, 1.0, 1.1, -1.1, -0.9, -1.3],
            "Escherichia": [-1.2, -1.0, -0.8, -0.9, 1.4, 1.6, 1.2],
        },
        index=["s1", "s2", "s3", "s7", "s4", "s5", "s6"],
    ).to_parquet(l5 / "M_clr.parquet")
    pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2", "s3", "s7", "s4", "s5", "s6", "not_in_clr"],
            "phenotype_label": ["Health", "IBD", "Health", "Health", "Health", "IBD", "CRC", "T2D", "Health"],
            "disease": ["healthy", "IBD", "healthy", "healthy", "healthy", "IBD", "CRC", "T2D", "healthy"],
        }
    ).to_parquet(l1 / "layer_a_master.parquet")
    pd.DataFrame(
        {
            "Akkermansia": {"Fiber, total dietary (g)": 1.0, "Total Fat (g)": -0.2},
            "Faecalibacterium": {"Fiber, total dietary (g)": 0.6, "Total Fat (g)": -0.1},
            "Escherichia": {"Fiber, total dietary (g)": -0.4, "Total Fat (g)": 0.8},
        }
    ).to_parquet(l7 / "B_nutrient_genus.parquet")
    (l8 / "nutrient_index.json").write_text(json.dumps(["Fiber, total dietary (g)", "Total Fat (g)"]))

    out_dir = tmp_path / "out"
    revision = {"sha": "a" * 40, "dirty": False}
    monkeypatch.setattr("scripts.build_beta_i_weights.source_revision", lambda _: revision)
    run(Args(root, out_dir))

    weights = pd.read_parquet(out_dir / "W_personalized.parquet")
    assert weights.shape == (7, 2)
    assert list(weights.columns) == ["Fiber, total dietary (g)", "Total Fat (g)"]
    assert (out_dir / "health_index.joblib").exists()
    assert (out_dir / "nutrient_perturbations.parquet").exists()
    assert (out_dir / "sample_beta_diagnostics.csv").exists()
    assert (out_dir / "nutrient_perturbation_summary.csv").exists()
    exclusion_audit = pd.read_csv(out_dir / "excluded_health_label_samples.csv")
    assert exclusion_audit.to_dict("records") == [
        {"sample_id": "s1", "reason": "conflicting_duplicate_health_labels"}
    ]
    manifest = json.loads((out_dir / "beta_i_manifest.json").read_text())
    assert manifest["method"] == "GMWI2-style health-index finite-difference beta_i"
    assert manifest["n_samples"] == 7
    assert manifest["n_nutrients"] == 2
    assert manifest["health_label_diagnostics"] == {
        "health_label_contradiction_policy": (
            "Exclude CLR-overlapping sample IDs with contradictory healthy and nonhealthy "
            "metadata evidence before deriving binary health labels or fitting the health index; "
            "do not adjudicate them to either class. Excluded IDs are omitted only from health-index "
            "training labels and remain in the beta_i scoring output."
        ),
        "n_clr_overlapping_metadata_rows": 8,
        "n_excluded_contradictory_sample_ids": 1,
        "n_derived_labeled_samples": 6,
        "n_healthy": 3,
        "n_nonhealthy": 3,
        "exclusion_scope": (
            "Excluded sample IDs are omitted from health-index training labels but remain "
            "in W_personalized.parquet beta_i scoring output."
        ),
    }
    assert manifest["health_index_config"] == {
        "c_value": 10.0,
        "max_iter": 500,
        "random_state": 11,
        "min_abs_coefficient": 1e-12,
        "training_backend": "numpy",
    }
    assert manifest["nutrient_perturbation_config"] == {
        "mac_channel_weight": 1.0,
        "lipid_channel_weight": 1.0,
        "other_channel_weight": 0.25,
        "mac_evidence_direction": 1.0,
        "lipid_evidence_direction": -1.0,
        "other_evidence_direction": 1.0,
        "min_abs_bridge": 0.0,
        "l2_norm": 1.0,
        "coefficient_power": 0.0,
    }
    assert manifest["beta_estimator_config"] == {
        "dose": 0.1,
        "clip_abs_beta": 12.0,
        "batch_size": 2,
        "response_scale": "probability",
        "difference": "forward",
    }
    assert manifest["health_model_manifest"] == {
        "path": "health_index.joblib.manifest.json",
        "sha256": manifest["output_sha256"]["health_index.joblib.manifest.json"],
    }
    assert manifest["health_training_backend"] == "numpy"
    assert manifest["health_training_solver"] == "proximal_gradient"
    assert manifest["health_model_serialization_backend"] == "pickle"
    assert manifest["source_revision"] == revision
    assert manifest["git_commit"] == revision["sha"]
    assert manifest["git_dirty"] is False
    assert manifest["excluded_health_label_samples"]["sha256"] == manifest["output_sha256"][
        "excluded_health_label_samples.csv"
    ]
    assert set(manifest["input_sha256"]) == {
        "M_clr",
        "metadata",
        "B_nutrient_genus",
        "nutrient_index",
    }
    for key, relative_path in manifest["input_files"].items():
        assert hashlib.sha256((root / relative_path).read_bytes()).hexdigest() == manifest["input_sha256"][key]
    assert set(manifest["package_versions"]) == {
        "python",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "joblib",
    }
    assert set(manifest["output_sha256"]) == {
        "W_personalized.parquet",
        "health_index.joblib",
        "health_index.joblib.manifest.json",
        "nutrient_perturbations.parquet",
        "sample_beta_diagnostics.csv",
        "nutrient_perturbation_summary.csv",
        "excluded_health_label_samples.csv",
    }
    for filename, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((out_dir / filename).read_bytes()).hexdigest() == digest
    assert "not a causal nutrient effect" in manifest["interpretation_boundary"]


def test_source_revision_reports_full_sha_and_dirty_state(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.email", "tests@example.com")
    git("config", "user.name", "Beta I Tests")
    tracked_file = root / "tracked.txt"
    tracked_file.write_text("clean\n")
    git("add", "tracked.txt")
    git("commit", "-m", "test revision metadata")

    clean = source_revision(root)
    assert re.fullmatch(r"[0-9a-f]{40}", clean["sha"])
    assert clean["dirty"] is False

    tracked_file.write_text("dirty\n")
    dirty = source_revision(root)
    assert dirty["sha"] == clean["sha"]
    assert dirty["dirty"] is True


def test_production_build_defaults_select_sparse_beta_i_bundle():
    beta_args = build_beta_i_parser().parse_args([])

    assert beta_args.c_value == 0.05
    assert beta_args.health_backend == "numpy"
    assert beta_args.serialization_backend == "pickle"
    assert beta_args.dose == 0.25
    assert beta_args.clip_abs_beta == 25.0
    assert beta_args.beta_response_scale == "logit"
    assert beta_args.beta_difference == "central"
    assert beta_args.perturbation_l2_norm == 2.0
    assert beta_args.coefficient_power == 0.5


def test_build_beta_i_weights_accepts_dual_channel_gmwi2_arguments(tmp_path):
    args = build_beta_i_parser().parse_args(
        [
            "--health-model",
            "dual_channel_gmwi2",
            "--dual-channel-feature-weights",
            str(tmp_path / "weights.csv"),
            "--official-gmwi2-scores",
            str(tmp_path / "official.csv"),
            "--dual-channel-compression-temperature",
            "0.35",
        ]
    )

    assert args.health_model == "dual_channel_gmwi2"
    assert args.dual_channel_feature_weights == tmp_path / "weights.csv"
    assert args.official_gmwi2_scores == tmp_path / "official.csv"
    assert args.dual_channel_compression_temperature == 0.35
