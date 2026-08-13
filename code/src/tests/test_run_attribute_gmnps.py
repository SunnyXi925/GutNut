import hashlib
import importlib.util
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
from gmnps.scoring.fcs2_attribute_rules import NOT_CALCULATED


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/scripts/run_attribute_gmnps.py"
CONFIG = ROOT / "src/configs/attribute_gmnps.yaml"
REGISTRY = ROOT / "src/configs/fcs2_fndds_release_registry.json"
NOT_CALCULATED_TOKEN = "__GMNPS_NOT_CALCULATED_V1__"
NOT_CALCULATED_SCHEMA_VERSION = "gmnps-not-calculated-csv-v1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_inputs(tmp_path, *, release="FNDDS 2021-2023", not_calculated_mask=()):
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
    points = baseline_points()
    mask_rows = []
    for food_id, attribute in not_calculated_mask:
        points[attribute] = points[attribute].astype(object)
        points.loc[food_id, attribute] = NOT_CALCULATED_TOKEN
        mask_rows.append({"food_id": food_id, "attribute": attribute})
    points.to_csv(paths["baseline_attribute_points"])
    exposures = food_exposures()
    exposures.loc[:, "Vitamin C (mg)"] = 22.5
    exposures.to_csv(paths["food_exposures"])

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    releases = [release] if isinstance(release, str) else list(release)
    production = releases == registry["canonical_release_set"]
    manifest = {
        "files": {name: {"sha256": sha256(path)} for name, path in paths.items()},
        "fndds_releases": releases,
        "production_label": "production" if production else "non-production",
        "registry_version": registry["registry_version"],
        "food_source_linkage_sha256": "6" * 64,
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
        "not_calculated_serialization": {
            "schema_version": NOT_CALCULATED_SCHEMA_VERSION,
            "token": NOT_CALCULATED_TOKEN,
            "baseline_attribute_points_mask": sorted(
                mask_rows, key=lambda row: (row["food_id"], row["attribute"])
            ),
        },
    }
    manifest_path = tmp_path / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    paths["input_manifest"] = manifest_path
    return paths


def command(paths, output_dir, *extra, smoke=True):
    result = [
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
    ]
    if smoke:
        result.append("--development-smoke-test")
    result.extend(extra)
    return result


def load_runner_module():
    spec = importlib.util.spec_from_file_location("task5_runner_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    assert not list(output.glob(".*.tmp"))

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
        fit_attribute_gmnps(
            development_beta(),
            __import__("gmnps.scoring.attribute_gmnps", fromlist=["AttributeGMNPSConfig"])
            .AttributeGMNPSConfig(development_smoke_test=True),
        ),
        beta,
        bundle,
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

    release_paths = write_inputs(tmp_path / "release", release="FNDDS 2001-2099")
    rejected = subprocess.run(
        command(release_paths, tmp_path / "rejected", smoke=False),
        text=True,
        capture_output=True,
    )
    assert rejected.returncode != 0
    assert "FNDDS 2001-2018" in rejected.stderr


def test_2021_2023_requires_smoke_flag_and_is_labelled_nonproduction(tmp_path):
    paths = write_inputs(tmp_path, release="FNDDS 2021-2023")
    output = tmp_path / "smoke"
    completed = subprocess.run(
        command(paths, output), text=True, capture_output=True
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
        ),
        text=True,
        capture_output=True,
    )
    assert accepted.returncode == 0, accepted.stderr
    legacy = pd.read_csv(output / "individual_food.csv")
    assert set(legacy["method_role"]) == {"sensitivity"}
    assert "top_positive_drivers" not in legacy
    assert "top_negative_drivers" not in legacy
    assert set(legacy["driver_basis"]) == {"raw_nutrient_contribution"}


def test_empty_trusted_registry_rejects_self_authored_production_inputs(tmp_path):
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    paths = write_inputs(tmp_path, release=registry["canonical_release_set"])
    output = tmp_path / "production"

    rejected = subprocess.run(
        command(paths, output, smoke=False), text=True, capture_output=True
    )
    assert rejected.returncode != 0
    assert "no approved production bundle" in rejected.stderr
    assert not (output / "_SUCCESS.json").exists()

    manifest = json.loads(paths["input_manifest"].read_text(encoding="utf-8"))
    manifest["nutrient_units"] = {
        column: "wrong_unit" for column in manifest["nutrient_units"]
    }
    paths["input_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    wrong_units = subprocess.run(
        command(paths, tmp_path / "wrong_units", smoke=False),
        text=True,
        capture_output=True,
    )
    assert wrong_units.returncode != 0
    assert "no approved production bundle" in wrong_units.stderr


def test_all_tables_and_aggregate_manifest_are_chunk_size_invariant(tmp_path):
    paths = write_inputs(tmp_path)
    outputs = [tmp_path / "one", tmp_path / "full"]
    commands = [
        command(paths, outputs[0], "--individual-chunk-size", "1", "--food-chunk-size", "1"),
        command(paths, outputs[1], "--individual-chunk-size", "99", "--food-chunk-size", "99"),
    ]
    for invocation in commands:
        completed = subprocess.run(invocation, text=True, capture_output=True)
        assert completed.returncode == 0, completed.stderr

    for name in (
        "individual_food.csv",
        "food_summary.csv",
        "attribute_attribution.csv",
        "domain_attribution.csv",
        "run_manifest.json",
        "_SUCCESS.json",
    ):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes(), name
    manifest = json.loads((outputs[0] / "run_manifest.json").read_text(encoding="utf-8"))
    fingerprints = manifest["aggregate_fingerprints"]
    assert fingerprints["scope"] == "fully_sorted_final_outputs_and_ordered_provenance_v1"
    assert set(fingerprints["tables"]) == {
        "individual_food",
        "food_summary",
        "attribute_attribution",
        "domain_attribution",
    }
    assert len(fingerprints["final_fingerprint"]) == 64
    attribute = pd.read_csv(outputs[0] / "attribute_attribution.csv")
    domain = pd.read_csv(outputs[0] / "domain_attribution.csv")
    assert "calibration_fingerprint" not in attribute
    assert "recomposition_fingerprint" not in domain
    assert "aggregate_attribute_attribution_fingerprint" in attribute
    assert "aggregate_domain_attribution_fingerprint" in domain
    assert attribute["aggregate_attribute_attribution_fingerprint"].str.fullmatch(
        r"[0-9a-f]{64}"
    ).all()
    assert domain["aggregate_domain_attribution_fingerprint"].str.fullmatch(
        r"[0-9a-f]{64}"
    ).all()


def test_inputs_are_parsed_from_one_immutable_byte_snapshot(tmp_path, monkeypatch):
    paths = write_inputs(tmp_path)
    module = load_runner_module()
    parser_args = command(paths, tmp_path / "snapshot")[2:]
    args = module.build_parser().parse_args(parser_args)
    counts = {}
    original = Path.read_bytes

    def counted_read_bytes(path):
        resolved = path.resolve()
        counts[resolved] = counts.get(resolved, 0) + 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    module.run(args)
    expected_once = [*paths.values(), CONFIG]
    assert {path.resolve(): counts.get(path.resolve(), 0) for path in expected_once} == {
        path.resolve(): 1 for path in expected_once
    }


def test_failed_rerun_removes_existing_success_marker(tmp_path):
    paths = write_inputs(tmp_path)
    output = tmp_path / "rerun"
    first = subprocess.run(command(paths, output), text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    assert (output / "_SUCCESS.json").exists()

    paths["score_beta"].write_text(
        paths["score_beta"].read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    failed = subprocess.run(command(paths, output), text=True, capture_output=True)
    assert failed.returncode != 0
    assert not (output / "_SUCCESS.json").exists()


def test_not_calculated_round_trip_and_chunk_equivalence(tmp_path):
    mask = (("food_1", "fiber_to_carbohydrate_ratio"),)
    paths = write_inputs(tmp_path, not_calculated_mask=mask)
    outputs = [tmp_path / "notcalc_one", tmp_path / "notcalc_full"]
    for output, sizes in zip(outputs, (("1", "1"), ("99", "99"))):
        completed = subprocess.run(
            command(
                paths,
                output,
                "--individual-chunk-size",
                sizes[0],
                "--food-chunk-size",
                sizes[1],
            ),
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stderr

    for name in (
        "individual_food.csv",
        "food_summary.csv",
        "attribute_attribution.csv",
        "domain_attribution.csv",
        "run_manifest.json",
    ):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes(), name
    attribution = pd.read_csv(outputs[0] / "attribute_attribution.csv", keep_default_na=False)
    gated = attribution.query(
        "food_id == 'food_1' and attribute == 'fiber_to_carbohydrate_ratio'"
    )
    assert set(gated["baseline_points"]) == {NOT_CALCULATED_TOKEN}
    manifest = json.loads((outputs[0] / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["not_calculated_serialization"]["baseline_attribute_points_mask"] == [
        {"attribute": "fiber_to_carbohydrate_ratio", "food_id": "food_1"}
    ]


def test_not_calculated_token_is_rejected_outside_ratio_columns(tmp_path):
    paths = write_inputs(
        tmp_path,
        not_calculated_mask=(("food_1", "vitamin_c"),),
    )
    rejected = subprocess.run(command(paths, tmp_path / "bad_token"), text=True, capture_output=True)
    assert rejected.returncode != 0
    assert "NOT_CALCULATED token is allowed only for ratio attributes" in rejected.stderr
