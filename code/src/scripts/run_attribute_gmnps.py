#!/usr/bin/env python3
"""Validate inputs and run attribute-level GMNPS in deterministic chunks."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable

import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gmnps.scoring.attribute_gmnps import (  # noqa: E402
    ATTRIBUTE_GMNPS_SCORING_VERSION,
    FCS2_FNDDS_REGISTRY_VERSION,
    LEGACY_METHOD,
    NOT_CALCULATED_SCHEMA_VERSION,
    NOT_CALCULATED_TOKEN,
    PRIMARY_METHOD,
    AttributeGMNPSConfig,
    AttributeGMNPSResult,
    FoodAttributeBundle,
    _base_manifest,
    _effect_nutrients_by_channel,
    _fingerprint,
    _frame_payload,
    fit_attribute_gmnps,
    score_attribute_gmnps,
    summarize_attribute_gmnps_foods,
)
from gmnps.scoring.fcs2_attribute_mapping import (  # noqa: E402
    PRIMARY_MAPPING_VERSION,
    reconstruction_status_table,
)
from gmnps.scoring.fcs2_attribute_rules import FCS2_RULES, NOT_CALCULATED  # noqa: E402


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


@dataclass(frozen=True)
class InputBlob:
    """One immutable caller input read and hashed from exactly the same bytes."""

    name: str
    path: Path
    data: bytes
    sha256: str


def read_input_blobs(paths: dict[str, Path]) -> dict[str, InputBlob]:
    blobs = {}
    for name, path in paths.items():
        try:
            data = path.read_bytes()
        except OSError as error:
            raise ValueError(
                "required attribute bundle input is missing or unreadable: " f"{path}"
            ) from error
        blobs[name] = InputBlob(
            name=name,
            path=path,
            data=data,
            sha256=sha256(data).hexdigest(),
        )
    return blobs


def _read_json_blob(blob: InputBlob, label: str) -> dict[str, object]:
    try:
        payload = json.loads(blob.data)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"{label} must be a readable JSON object: {blob.path}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def load_locked_config(blob: InputBlob) -> dict[str, object]:
    """Load JSON-compatible YAML and reject any drift from locked defaults."""

    supplied = _read_json_blob(blob, "attribute GMNPS YAML")
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
    blobs: dict[str, InputBlob],
    locked_config: dict[str, object],
) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("input manifest must contain a files object with SHA-256 digests")
    for name in _INPUT_NAMES:
        if name not in files:
            raise ValueError(f"input manifest is missing files.{name}")
        expected = _manifest_hash(files[name], name)
        actual = blobs[name].sha256
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {name}: expected {expected}, observed {actual}"
            )
        entry = files[name]
        if isinstance(entry, dict) and "path" in entry:
            declared = Path(str(entry["path"])).expanduser().resolve()
            if declared != blobs[name].path.resolve():
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
    if manifest.get("registry_version") != FCS2_FNDDS_REGISTRY_VERSION:
        raise ValueError("input manifest registry_version does not match the trusted registry")
    linkage = manifest.get("food_source_linkage_sha256")
    if (
        not isinstance(linkage, str)
        or len(linkage) != 64
        or set(linkage) - set("0123456789abcdef")
    ):
        raise ValueError("input manifest must declare food_source_linkage_sha256")
    serialization = manifest.get("not_calculated_serialization")
    if not isinstance(serialization, dict):
        raise ValueError("input manifest must declare not_calculated_serialization")
    if serialization.get("schema_version") != NOT_CALCULATED_SCHEMA_VERSION:
        raise ValueError("NOT_CALCULATED serialization schema version is invalid")
    if serialization.get("token") != NOT_CALCULATED_TOKEN:
        raise ValueError("NOT_CALCULATED serialization token is invalid")
    if not isinstance(serialization.get("baseline_attribute_points_mask"), list):
        raise ValueError("NOT_CALCULATED serialization mask must be a list")


def _read_indexed_csv(blob: InputBlob, index_name: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(BytesIO(blob.data), dtype={index_name: str})
    except Exception as error:
        raise ValueError(f"unable to read {blob.path} as CSV") from error
    if index_name not in frame.columns:
        raise ValueError(f"{blob.path} must contain the index column {index_name}")
    if frame[index_name].isna().any() or frame[index_name].duplicated().any():
        raise ValueError(
            f"{blob.path} {index_name} values must be unique and nonmissing"
        )
    frame = frame.set_index(index_name)
    frame.index = frame.index.astype(str)
    return frame.sort_index(kind="mergesort")


def _restore_not_calculated(
    baseline: pd.DataFrame,
    serialization: dict[str, object],
) -> pd.DataFrame:
    mask_rows = serialization["baseline_attribute_points_mask"]
    canonical_mask = []
    seen = set()
    for row in mask_rows:
        if not isinstance(row, dict) or set(row) != {"food_id", "attribute"}:
            raise ValueError("NOT_CALCULATED mask rows must name food_id and attribute")
        key = (row["food_id"], row["attribute"])
        if key in seen:
            raise ValueError("NOT_CALCULATED mask contains duplicate rows")
        seen.add(key)
        canonical_mask.append(key)
    if canonical_mask != sorted(canonical_mask):
        raise ValueError("NOT_CALCULATED mask must use canonical sorted order")

    token_positions = []
    for food_id, row in baseline.iterrows():
        for attribute, value in row.items():
            if value == NOT_CALCULATED_TOKEN:
                token_positions.append((food_id, attribute))
    token_positions.sort()
    if token_positions != canonical_mask:
        raise ValueError(
            "NOT_CALCULATED token positions must exactly match the explicit manifest mask"
        )

    restored = baseline.copy()
    for food_id, attribute in token_positions:
        if attribute not in FCS2_RULES or FCS2_RULES[attribute].kind != "log_ratio":
            raise ValueError(
                "NOT_CALCULATED token is allowed only for ratio attributes"
            )
        restored[attribute] = restored[attribute].astype(object)
        restored.loc[food_id, attribute] = NOT_CALCULATED
    for attribute in restored.columns:
        for food_id, value in restored[attribute].items():
            if value is NOT_CALCULATED:
                continue
            try:
                restored.loc[food_id, attribute] = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"baseline_attribute_points[{food_id}, {attribute}] must be numeric "
                    "or the declared NOT_CALCULATED token"
                ) from error
    return restored


def load_inputs(
    blobs: dict[str, InputBlob],
    manifest: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, FoodAttributeBundle]:
    development = _read_indexed_csv(blobs["development_beta"], "individual_id")
    scoring = _read_indexed_csv(blobs["score_beta"], "individual_id")
    metadata = _read_indexed_csv(blobs["food_metadata"], "food_id")
    baseline = _read_indexed_csv(blobs["baseline_attribute_points"], "food_id")
    baseline = _restore_not_calculated(
        baseline, manifest["not_calculated_serialization"]
    )
    exposures = _read_indexed_csv(blobs["food_exposures"], "food_id")
    for label, frame in (
        ("development_beta", development),
        ("score_beta", scoring),
        ("food_metadata", metadata),
        ("food_exposures", exposures),
    ):
        if frame.astype(str).eq(NOT_CALCULATED_TOKEN).any().any():
            raise ValueError(f"NOT_CALCULATED token is prohibited in {label}")
    exposures.attrs["basis"] = "per_100_kcal"
    if "FCS2" not in metadata:
        raise ValueError("food_metadata must contain official FCS2")
    official = metadata["FCS2"].copy()
    official.name = "FCS2"
    source_hashes = {
        "official_fcs": blobs["food_metadata"].sha256,
        "food_metadata": blobs["food_metadata"].sha256,
        "baseline_attribute_points": blobs["baseline_attribute_points"].sha256,
        "food_exposures": blobs["food_exposures"].sha256,
        "input_manifest": blobs["input_manifest"].sha256,
        "development_beta": blobs["development_beta"].sha256,
        "score_beta": blobs["score_beta"].sha256,
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
        registry_version=str(manifest["registry_version"]),
        food_source_linkage_sha256=str(manifest["food_source_linkage_sha256"]),
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


def _aggregate_manifest(
    model,
    bundle: FoodAttributeBundle,
    individual: pd.DataFrame,
    food_summary: pd.DataFrame,
    attribute: pd.DataFrame,
    domain: pd.DataFrame,
) -> dict[str, object]:
    tables = {
        "individual_food": individual,
        "food_summary": food_summary,
        "attribute_attribution": attribute,
        "domain_attribution": domain,
    }
    table_fingerprints = {
        name: _fingerprint(_frame_payload(frame)) for name, frame in tables.items()
    }
    ordered_provenance = []
    for row in individual.loc[:, ["individual_id", "food_id"]].itertuples(index=False):
        individual_id, food_id = row
        pair_individual = individual[
            (individual["individual_id"] == individual_id)
            & (individual["food_id"] == food_id)
        ].reset_index(drop=True)
        pair_attribute = attribute[
            (attribute["individual_id"] == individual_id)
            & (attribute["food_id"] == food_id)
        ].reset_index(drop=True)
        pair_domain = domain[
            (domain["individual_id"] == individual_id)
            & (domain["food_id"] == food_id)
        ].reset_index(drop=True)
        ordered_provenance.append(
            {
                "individual_id": individual_id,
                "food_id": food_id,
                "scoring_unit_fingerprint": _fingerprint(
                    {
                        "individual_food": _frame_payload(pair_individual),
                        "attribute_attribution": _frame_payload(pair_attribute),
                        "domain_attribution": _frame_payload(pair_domain),
                        "model_fingerprint": model.fingerprint,
                        "bundle_fingerprint": bundle.fingerprint,
                    }
                ),
            }
        )
    provenance_fingerprint = _fingerprint(ordered_provenance)
    scope = "fully_sorted_final_outputs_and_ordered_provenance_v1"
    final_fingerprint = _fingerprint(
        {
            "scope": scope,
            "tables": table_fingerprints,
            "ordered_chunk_provenance_sha256": provenance_fingerprint,
            "config_fingerprint": model.config_fingerprint,
            "model_fingerprint": model.fingerprint,
            "bundle_fingerprint": bundle.fingerprint,
        }
    )
    effect_channels = _effect_nutrients_by_channel(model.config.mapping_version)
    manifest = _base_manifest(model, bundle)
    manifest.update(
        {
            "fingerprint_scope": scope,
            "ordered_chunk_provenance": ordered_provenance,
            "aggregate_fingerprints": {
                "scope": scope,
                "tables": table_fingerprints,
                "ordered_chunk_provenance_sha256": provenance_fingerprint,
                "config_fingerprint": model.config_fingerprint,
                "model_fingerprint": model.fingerprint,
                "bundle_fingerprint": bundle.fingerprint,
                "final_fingerprint": final_fingerprint,
            },
            "final_fingerprint": final_fingerprint,
        }
    )
    if model.config.method == PRIMARY_METHOD:
        manifest["counterfactual_provenance"] = {
            "scope": "aggregate_output_columns",
            "MAC_delta": {
                "definition": "score_with_LIPID_frozen_minus_FCS2",
                "frozen_channel": "LIPID",
                "frozen_effect_nutrients": list(effect_channels["LIPID"]),
                "raw_beta_replacement": "development_median",
            },
            "LIPID_delta": {
                "definition": "score_with_MAC_frozen_minus_FCS2",
                "frozen_channel": "MAC",
                "frozen_effect_nutrients": list(effect_channels["MAC"]),
                "raw_beta_replacement": "development_median",
            },
            "channel_interaction_delta": {
                "definition": "GMNPS_delta_minus_MAC_delta_minus_LIPID_delta"
            },
        }
    else:
        manifest["legacy_channel_attribution"] = {
            "scope": "sensitivity_comparator_only",
            "basis": "legacy_raw_channel_magnitude_allocation",
        }
    return manifest


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
    if model.config.method == LEGACY_METHOD:
        results.append(score_attribute_gmnps(model, score_beta, bundle))
    else:
        for individual_ids in _chunks(individuals, individual_chunk_size):
            beta_chunk = score_beta.loc[individual_ids].copy()
            for food_ids in _chunks(foods, food_chunk_size):
                results.append(
                    score_attribute_gmnps(
                        model,
                        beta_chunk,
                        _subset_bundle(bundle, food_ids),
                    )
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
    if "calibration_fingerprint" in attribute:
        aggregate_attribute_attribution_fingerprint = _fingerprint(
            {
                "scope": "aggregate_attribute_attribution",
                "model_fingerprint": model.fingerprint,
                "bundle_fingerprint": bundle.fingerprint,
                "table": _frame_payload(
                    attribute.drop(columns=["calibration_fingerprint"])
                ),
            }
        )
        attribute = attribute.rename(
            columns={
                "calibration_fingerprint": (
                    "aggregate_attribute_attribution_fingerprint"
                )
            }
        )
        attribute["aggregate_attribute_attribution_fingerprint"] = (
            aggregate_attribute_attribution_fingerprint
        )
    if "recomposition_fingerprint" in domain:
        aggregate_domain_attribution_fingerprint = _fingerprint(
            {
                "scope": "aggregate_domain_attribution",
                "model_fingerprint": model.fingerprint,
                "bundle_fingerprint": bundle.fingerprint,
                "table": _frame_payload(
                    domain.drop(columns=["recomposition_fingerprint"])
                ),
            }
        )
        domain = domain.rename(
            columns={
                "recomposition_fingerprint": (
                    "aggregate_domain_attribution_fingerprint"
                )
            }
        )
        domain["aggregate_domain_attribution_fingerprint"] = (
            aggregate_domain_attribution_fingerprint
        )
    food_summary = summarize_attribute_gmnps_foods(individual)
    manifest = _aggregate_manifest(
        model,
        bundle,
        individual,
        food_summary,
        attribute,
        domain,
    )
    return AttributeGMNPSResult(
        individual_food=individual,
        food_summary=food_summary,
        attribute_attribution=attribute,
        domain_attribution=domain,
        run_manifest=manifest,
    )


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    stream = StringIO()
    frame.to_csv(
        stream,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    return stream.getvalue().encode("utf-8")


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _atomic_replace_bytes(path: Path, data: bytes) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _invalidate_success_marker(output_dir: Path) -> None:
    marker = output_dir / "_SUCCESS.json"
    if marker.exists():
        marker.unlink()


def write_outputs(output_dir: Path, result: AttributeGMNPSResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _invalidate_success_marker(output_dir)
    artifacts = {
        "individual_food.csv": _csv_bytes(result.individual_food),
        "food_summary.csv": _csv_bytes(result.food_summary),
        "attribute_attribution.csv": _csv_bytes(result.attribute_attribution),
        "domain_attribution.csv": _csv_bytes(result.domain_attribution),
        "run_manifest.json": _json_bytes(result.run_manifest),
    }
    expected_hashes = {
        name: sha256(data).hexdigest() for name, data in sorted(artifacts.items())
    }
    try:
        for name, data in artifacts.items():
            _atomic_replace_bytes(output_dir / name, data)
        observed_hashes = {
            name: sha256((output_dir / name).read_bytes()).hexdigest()
            for name in sorted(artifacts)
        }
        if observed_hashes != expected_hashes:
            raise OSError("final artifact hash verification failed")
        marker = {
            "status": "complete",
            "files": observed_hashes,
        }
        _atomic_replace_bytes(output_dir / "_SUCCESS.json", _json_bytes(marker))
    except Exception:
        _invalidate_success_marker(output_dir)
        raise


def run(args: argparse.Namespace) -> AttributeGMNPSResult:
    _invalidate_success_marker(args.output_dir)
    paths = {
        "development_beta": args.development_beta,
        "score_beta": args.score_beta,
        "food_metadata": args.food_metadata,
        "baseline_attribute_points": args.baseline_attribute_points,
        "food_exposures": args.food_exposures,
        "input_manifest": args.input_manifest,
        "config": args.config,
    }
    blobs = read_input_blobs(paths)
    locked = load_locked_config(blobs["config"])
    manifest = _read_json_blob(blobs["input_manifest"], "input manifest")
    validate_input_manifest(manifest, blobs, locked)
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
    development, scoring, bundle = load_inputs(blobs, manifest)
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
    result.run_manifest["input_manifest_sha256"] = blobs["input_manifest"].sha256
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
