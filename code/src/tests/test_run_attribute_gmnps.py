import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from test_attribute_gmnps import (
    baseline_points,
    development_beta,
    food_exposures,
    make_bundle,
    score_beta,
)

from gmnps.scoring.attribute_gmnps import fit_attribute_gmnps, score_attribute_gmnps


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/scripts/run_attribute_gmnps.py"
CONFIG = ROOT / "src/configs/attribute_gmnps.yaml"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_inputs(tmp_path, *, release="FNDDS 2001-2018"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "development_beta": tmp_path / "development_beta.csv",
        "score_beta": tmp_path / "score_beta.csv",
        "food_metadata": tmp_path / "food_metadata.csv",
        "baseline_attribute_points": tmp_path / "baseline_attribute_points.csv",
        "food_exposures": tmp_path / "food_exposures.csv",
    }
    development_beta().rename_axis("individual_id").to_csv(paths["development_beta"])
    beta = score_beta(("person_b", "person_a"))
    beta.loc["person_b", "Vitamin C (mg)"] = 4.0
    beta.rename_axis("individual_id").to_csv(paths["score_beta"])
    bundle = make_bundle()
    bundle.food_metadata.rename_axis("food_id").to_csv(paths["food_metadata"])
    baseline_points().to_csv(paths["baseline_attribute_points"])
    exposures = food_exposures()
    exposures.loc[:, "Vitamin C (mg)"] = 22.5
    exposures.to_csv(paths["food_exposures"])

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = {
        "files": {name: {"sha256": sha256(path)} for name, path in paths.items()},
        "fndds_releases": [release],
        "production_label": "production" if release == "FNDDS 2001-2018" else "non-production",
        "exposure_basis": "per_100_kcal",
        "nutrient_units": {
            column: "source_unit_per_100_kcal" for column in exposures.columns
        },
        "model_config": {
            "method": "attribute_recomposition",
            "mapping_version": config["mapping_version"],
            "attribute_point_mode": config["attribute_point_mode"],
            "final_cap_mode": config["final_cap_mode"],
            "recomputed_domains": config["recomputed_domains"],
            "scoring_version": config["scoring_version"],
        },
    }
    manifest_path = tmp_path / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    paths["input_manifest"] = manifest_path
    return paths


def command(paths, output_dir, *extra):
    return [
        sys.executable,
        str(SCRIPT),
        "--development-beta",
        str(paths["development_beta"]),
        "--score-beta",
        str(paths["score_beta"]),
        "--food-metadata",
        str(paths["food_metadata"]),
        "--baseline-attribute-points",
        str(paths["baseline_attribute_points"]),
        "--food-exposures",
        str(paths["food_exposures"]),
        "--input-manifest",
        str(paths["input_manifest"]),
        "--config",
        str(CONFIG),
        "--output-dir",
        str(output_dir),
        *extra,
    ]


def test_yaml_contains_locked_defaults_without_arbitrary_primary_parameters():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["primary_method"] == "attribute_recomposition"
    assert config["baseline"] == "FoodCompass2.0"
    assert config["attribute_point_fraction_cap"] == 0.20
    assert config["attribute_point_fraction_sensitivity"] == [0.10, 0.30]
    assert config["final_delta_cap"] == 12
    assert config["final_delta_cap_sensitivity"] == [8, 15]
    assert config["beta_temperature"] == 2
    assert config["carbohydrate_policy"] == "sensitivity_proxy_only"
    assert config["clinical_experiment"] is False


def test_chunked_runner_matches_unchunked_api_and_completes_atomically(tmp_path):
    paths = write_inputs(tmp_path)
    output = tmp_path / "output"
    completed = subprocess.run(
        command(paths, output, "--individual-chunk-size", "1", "--food-chunk-size", "1"),
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output / "_SUCCESS.json").is_file()
    assert not list(output.glob("*.tmp"))

    actual = pd.read_csv(output / "individual_food.csv").sort_values(
        ["individual_id", "food_id"]
    ).reset_index(drop=True)
    bundle = make_bundle()
    exposures = bundle.food_exposures.copy()
    exposures.loc[:, "Vitamin C (mg)"] = 22.5
    exposures.attrs["basis"] = "per_100_kcal"
    bundle = replace(bundle, food_exposures=exposures, fingerprint="")
    beta = score_beta(("person_b", "person_a"))
    beta.loc["person_b", "Vitamin C (mg)"] = 4.0
    expected = score_attribute_gmnps(
        fit_attribute_gmnps(development_beta()), beta, bundle
    ).individual_food.sort_values(["individual_id", "food_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-12, rtol=0.0)


def test_cli_is_byte_deterministic_and_orders_outputs(tmp_path):
    paths = write_inputs(tmp_path)
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        completed = subprocess.run(command(paths, output), text=True, capture_output=True)
        assert completed.returncode == 0, completed.stderr

    names = sorted(path.name for path in outputs[0].iterdir())
    assert names == sorted(path.name for path in outputs[1].iterdir())
    for name in names:
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
    rows = pd.read_csv(outputs[0] / "individual_food.csv")
    assert rows[["individual_id", "food_id"]].values.tolist() == sorted(
        rows[["individual_id", "food_id"]].values.tolist()
    )


def test_manifest_hash_tamper_and_release_mismatch_are_rejected(tmp_path):
    paths = write_inputs(tmp_path)
    paths["score_beta"].write_text(paths["score_beta"].read_text() + "\n", encoding="utf-8")
    tampered = subprocess.run(command(paths, tmp_path / "tampered"), text=True, capture_output=True)
    assert tampered.returncode != 0
    assert "SHA-256" in tampered.stderr
    assert not (tmp_path / "tampered" / "_SUCCESS.json").exists()

    release_paths = write_inputs(tmp_path / "release", release="FNDDS 2021-2023")
    rejected = subprocess.run(command(release_paths, tmp_path / "rejected"), text=True, capture_output=True)
    assert rejected.returncode != 0
    assert "development-smoke-test" in rejected.stderr


def test_2021_2023_requires_smoke_flag_and_is_labelled_nonproduction(tmp_path):
    paths = write_inputs(tmp_path, release="FNDDS 2021-2023")
    output = tmp_path / "smoke"
    completed = subprocess.run(
        command(paths, output, "--development-smoke-test"), text=True, capture_output=True
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["production_label"] == "non-production"
    assert manifest["development_smoke_test"] is True


def test_legacy_cli_requires_explicit_sensitivity_role(tmp_path):
    paths = write_inputs(tmp_path)
    rejected = subprocess.run(
        command(paths, tmp_path / "legacy_bad", "--method", "legacy_final_score_offset"),
        text=True,
        capture_output=True,
    )
    assert rejected.returncode != 0
    assert "method-role sensitivity" in rejected.stderr

    output = tmp_path / "legacy"
    accepted = subprocess.run(
        command(
            paths,
            output,
            "--method",
            "legacy_final_score_offset",
            "--method-role",
            "sensitivity",
            "--development-smoke-test",
        ),
        text=True,
        capture_output=True,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert set(pd.read_csv(output / "individual_food.csv")["method_role"]) == {"sensitivity"}
