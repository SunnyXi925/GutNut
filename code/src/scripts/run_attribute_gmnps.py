#!/usr/bin/env python3
"""Validate inputs and run attribute-level GMNPS in deterministic chunks."""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gmnps.scoring.attribute_gmnps import (  # noqa: E402
    ATTRIBUTE_GMNPS_SCORING_VERSION,
    LEGACY_METHOD,
    PRIMARY_METHOD,
    AttributeGMNPSConfig,
    AttributeGMNPSResult,
    FoodAttributeBundle,
    fit_attribute_gmnps,
    score_attribute_gmnps,
    summarize_attribute_gmnps_foods,
)
from gmnps.scoring.fcs2_attribute_mapping import (  # noqa: E402
    PRIMARY_MAPPING_VERSION,
    reconstruction_status_table,
)


_INPUT_NAMES = (
    "development_beta",
    "score_beta",
    "food_metadata",
    "baseline_attribute_points",
    "food_exposures",
)
_LOCKED_CONFIG = {
    "primary_method": PRIMARY_METHOD,
    "baseline": "FoodCompass2.0",
    "mapping_version": PRIMARY_MAPPING_VERSION,
    "attribute_point_mode": "primary",
    "attribute_point_fraction_cap": 0.20,
    "attribute_point_fraction_sensitivity": [0.10, 0.30],
    "final_cap_mode": "primary",
    "final_delta_cap": 12,
    "final_delta_cap_sensitivity": [8, 15],
    "beta_temperature": 2,
    "carbohydrate_policy": "sensitivity_proxy_only",
    "clinical_experiment": False,
    "recomputed_domains": [
        "nutrient_ratios",
        "vitamins",
        "minerals",
        "specific_lipids",
        "fiber_and_protein",
        "phytochemicals",
    ],
    "scoring_version": ATTRIBUTE_GMNPS_SCORING_VERSION,
}


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be a readable JSON object: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def load_locked_config(path: Path) -> dict[str, object]:
    """Load JSON-compatible YAML and reject any drift from locked defaults."""

    supplied = _read_json(path, "attribute GMNPS YAML")
    if supplied != _LOCKED_CONFIG:
        missing = sorted(set(_LOCKED_CONFIG) - set(supplied))
        extra = sorted(set(supplied) - set(_LOCKED_CONFIG))
        changed = sorted(
            key
            for key in set(supplied) & set(_LOCKED_CONFIG)
            if supplied[key] != _LOCKED_CONFIG[key]
        )
        raise ValueError(
            "attribute GMNPS config does not match locked defaults; "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return supplied


def _manifest_hash(entry: object, name: str) -> str:
    if isinstance(entry, str):
        value = entry
    elif isinstance(entry, dict):
        value = entry.get("sha256")
    else:
        value = None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"input manifest files.{name} must contain a SHA-256 digest")
    return value


def validate_input_manifest(
    manifest: dict[str, object],
    paths: dict[str, Path],
    locked_config: dict[str, object],
) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("input manifest must contain a files object with SHA-256 digests")
    for name in _INPUT_NAMES:
        if name not in files:
            raise ValueError(f"input manifest is missing files.{name}")
        expected = _manifest_hash(files[name], name)
        actual = _sha256(paths[name])
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {name}: expected {expected}, observed {actual}"
            )
        entry = files[name]
        if isinstance(entry, dict) and "path" in entry:
            declared = Path(str(entry["path"])).expanduser().resolve()
            if declared != paths[name].resolve():
                raise ValueError(f"input manifest path mismatch for {name}")

    expected_model = {
        "method": locked_config["primary_method"],
        "mapping_version": locked_config["mapping_version"],
        "attribute_point_mode": locked_config["attribute_point_mode"],
        "final_cap_mode": locked_config["final_cap_mode"],
        "recomputed_domains": locked_config["recomputed_domains"],
        "scoring_version": locked_config["scoring_version"],
    }
    if manifest.get("model_config") != expected_model:
        raise ValueError("input manifest model_config does not match the locked scoring config")
    if manifest.get("exposure_basis") != "per_100_kcal":
        raise ValueError("input manifest exposure_basis must be per_100_kcal")
    releases = manifest.get("fndds_releases")
    if not isinstance(releases, list) or not releases:
        raise ValueError("input manifest must declare fndds_releases")
    if not isinstance(manifest.get("nutrient_units"), dict):
        raise ValueError("input manifest must declare nutrient_units")
    if manifest.get("production_label") not in {"production", "non-production"}:
        raise ValueError("input manifest must declare production_label")


def _read_indexed_csv(path: Path, index_name: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype={index_name: str})
    except Exception as error:
        raise ValueError(f"unable to read {path} as CSV") from error
    if index_name not in frame.columns:
        raise ValueError(f"{path} must contain the index column {index_name}")
    if frame[index_name].isna().any() or frame[index_name].duplicated().any():
        raise ValueError(f"{path} {index_name} values must be unique and nonmissing")
    frame = frame.set_index(index_name)
    frame.index = frame.index.astype(str)
    return frame.sort_index(kind="mergesort")


def load_inputs(
    paths: dict[str, Path],
    manifest: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, FoodAttributeBundle]:
    development = _read_indexed_csv(paths["development_beta"], "individual_id")
    scoring = _read_indexed_csv(paths["score_beta"], "individual_id")
    metadata = _read_indexed_csv(paths["food_metadata"], "food_id")
    baseline = _read_indexed_csv(paths["baseline_attribute_points"], "food_id")
    exposures = _read_indexed_csv(paths["food_exposures"], "food_id")
    exposures.attrs["basis"] = "per_100_kcal"
    if "FCS2" not in metadata:
        raise ValueError("food_metadata must contain official FCS2")
    official = metadata["FCS2"].copy()
    official.name = "FCS2"
    source_hashes = {
        "official_fcs": _sha256(paths["food_metadata"]),
        "food_metadata": _sha256(paths["food_metadata"]),
        "baseline_attribute_points": _sha256(paths["baseline_attribute_points"]),
        "food_exposures": _sha256(paths["food_exposures"]),
        "input_manifest": _sha256(paths["input_manifest"]),
        "development_beta": _sha256(paths["development_beta"]),
        "score_beta": _sha256(paths["score_beta"]),
    }
    bundle = FoodAttributeBundle(
        official_fcs=official,
        baseline_points=baseline,
        food_exposures=exposures,
        food_metadata=metadata,
        fndds_releases=tuple(manifest["fndds_releases"]),
        source_hashes=source_hashes,
        nutrient_units=dict(manifest["nutrient_units"]),
        exposure_basis=str(manifest["exposure_basis"]),
        reconstruction_status=reconstruction_status_table(),
        production_label=str(manifest["production_label"]),
    )
    return development, scoring, bundle


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _subset_bundle(bundle: FoodAttributeBundle, food_ids: list[str]) -> FoodAttributeBundle:
    exposures = bundle.food_exposures.loc[food_ids].copy()
    exposures.attrs["basis"] = bundle.exposure_basis
    return replace(
        bundle,
        official_fcs=bundle.official_fcs.loc[food_ids].copy(),
        baseline_points=bundle.baseline_points.loc[food_ids].copy(),
        food_exposures=exposures,
        food_metadata=bundle.food_metadata.loc[food_ids].copy(),
        fingerprint="",
    )


def run_attribute_gmnps_chunked(
    model,
    score_beta: pd.DataFrame,
    bundle: FoodAttributeBundle,
    *,
    individual_chunk_size: int,
    food_chunk_size: int,
) -> AttributeGMNPSResult:
    """Score Cartesian chunks and combine tables without numerical aggregation drift."""

    if isinstance(individual_chunk_size, bool) or individual_chunk_size <= 0:
        raise ValueError("individual_chunk_size must be a positive integer")
    if isinstance(food_chunk_size, bool) or food_chunk_size <= 0:
        raise ValueError("food_chunk_size must be a positive integer")
    individuals = sorted(score_beta.index.tolist())
    foods = sorted(bundle.official_fcs.index.tolist())
    results = []
    chunk_provenance = []
    for individual_ids in _chunks(individuals, individual_chunk_size):
        beta_chunk = score_beta.loc[individual_ids].copy()
        for food_ids in _chunks(foods, food_chunk_size):
            result = score_attribute_gmnps(
                model,
                beta_chunk,
                _subset_bundle(bundle, food_ids),
            )
            results.append(result)
            chunk_provenance.append(
                {
                    "individual_ids": individual_ids,
                    "food_ids": food_ids,
                    "bundle_fingerprint": result.run_manifest["bundle_fingerprint"],
                    "final_fingerprint": result.run_manifest.get("final_fingerprint"),
                    "counterfactual_provenance": result.run_manifest.get(
                        "counterfactual_provenance", {}
                    ),
                }
            )

    individual = pd.concat([result.individual_food for result in results], ignore_index=True)
    individual = individual.sort_values(
        ["individual_id", "food_id"], kind="mergesort"
    ).reset_index(drop=True)
    attribute = pd.concat(
        [result.attribute_attribution for result in results], ignore_index=True
    ).sort_values(["individual_id", "food_id", "attribute"], kind="mergesort").reset_index(drop=True)
    domain = pd.concat(
        [result.domain_attribution for result in results], ignore_index=True
    ).sort_values(["individual_id", "food_id", "domain"], kind="mergesort").reset_index(drop=True)
    manifest = dict(results[0].run_manifest)
    manifest.update(
        {
            "bundle_fingerprint": bundle.fingerprint,
            "source_hashes": dict(sorted(bundle.source_hashes.items())),
            "chunking": {
                "individual_chunk_size": individual_chunk_size,
                "food_chunk_size": food_chunk_size,
                "chunk_count": len(results),
            },
            "chunk_provenance": chunk_provenance,
            "counterfactual_provenance": {
                "scope": "all chunks",
                "chunks": [row["counterfactual_provenance"] for row in chunk_provenance],
            },
        }
    )
    return AttributeGMNPSResult(
        individual_food=individual,
        food_summary=summarize_attribute_gmnps_foods(individual),
        attribute_attribution=attribute,
        domain_attribution=domain,
        run_manifest=manifest,
    )


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_outputs(output_dir: Path, result: AttributeGMNPSResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "individual_food.csv": result.individual_food,
        "food_summary.csv": result.food_summary,
        "attribute_attribution.csv": result.attribute_attribution,
        "domain_attribution.csv": result.domain_attribution,
    }
    for name, frame in outputs.items():
        _atomic_csv(output_dir / name, frame)
    _atomic_json(output_dir / "run_manifest.json", result.run_manifest)
    completed_files = sorted([*outputs, "run_manifest.json"])
    marker = {
        "status": "complete",
        "files": {
            name: _sha256(output_dir / name)
            for name in completed_files
        },
    }
    _atomic_json(output_dir / "_SUCCESS.json", marker)


def run(args: argparse.Namespace) -> AttributeGMNPSResult:
    locked = load_locked_config(args.config)
    paths = {
        "development_beta": args.development_beta,
        "score_beta": args.score_beta,
        "food_metadata": args.food_metadata,
        "baseline_attribute_points": args.baseline_attribute_points,
        "food_exposures": args.food_exposures,
        "input_manifest": args.input_manifest,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(
            "required production attribute bundle input is missing; provide verified "
            "FNDDS 2001-2018 files: " + ", ".join(missing)
        )
    manifest = _read_json(args.input_manifest, "input manifest")
    validate_input_manifest(manifest, paths, locked)
    if args.method == LEGACY_METHOD and args.method_role != "sensitivity":
        raise ValueError(
            "legacy_final_score_offset requires explicit --method-role sensitivity"
        )
    config = AttributeGMNPSConfig(
        method=args.method,
        mapping_version=str(locked["mapping_version"]),
        attribute_point_mode=str(locked["attribute_point_mode"]),
        final_cap_mode=str(locked["final_cap_mode"]),
        recomputed_domains=tuple(locked["recomputed_domains"]),
        scoring_version=str(locked["scoring_version"]),
        method_role=args.method_role,
        development_smoke_test=args.development_smoke_test,
    )
    development, scoring, bundle = load_inputs(paths, manifest)
    if bundle.production_label == "non-production" and not args.development_smoke_test:
        raise ValueError(
            "FNDDS 2021-2023 is non-production; rerun only as an explicit "
            "--development-smoke-test"
        )
    model = fit_attribute_gmnps(development, config)
    individual_chunk_size = args.individual_chunk_size or len(scoring)
    food_chunk_size = args.food_chunk_size or len(bundle.official_fcs)
    result = run_attribute_gmnps_chunked(
        model,
        scoring,
        bundle,
        individual_chunk_size=individual_chunk_size,
        food_chunk_size=food_chunk_size,
    )
    result.run_manifest["input_manifest_sha256"] = _sha256(args.input_manifest)
    write_outputs(args.output_dir, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked attribute-level GMNPS pipeline. Production requires an "
            "official FCS2-aligned FNDDS 2001-2018 attribute bundle."
        )
    )
    parser.add_argument("--development-beta", type=Path, required=True)
    parser.add_argument("--score-beta", type=Path, required=True)
    parser.add_argument("--food-metadata", type=Path, required=True)
    parser.add_argument("--baseline-attribute-points", type=Path, required=True)
    parser.add_argument("--food-exposures", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=SRC_ROOT / "configs/attribute_gmnps.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=[PRIMARY_METHOD, LEGACY_METHOD],
        default=PRIMARY_METHOD,
    )
    parser.add_argument(
        "--method-role", choices=["primary", "sensitivity"], default="primary"
    )
    parser.add_argument("--development-smoke-test", action="store_true")
    parser.add_argument("--individual-chunk-size", type=int)
    parser.add_argument("--food-chunk-size", type=int)
    return parser


def main() -> None:
    try:
        run(build_parser().parse_args())
    except (TypeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
