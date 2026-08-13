#!/usr/bin/env python3
"""Fail-closed runner for the GMNPS population-safety design audit."""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gmnps.validation.attribute_population_safety import (  # noqa: E402
    PopulationArtifactPaths,
    PopulationSafetyAudit,
    PopulationSafetyConfig,
    audit_population_safety,
    load_method_lock_artifact_locator,
    load_verified_population_input,
    validate_population_provenance,
)


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or invalid: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _production_paths(args: argparse.Namespace) -> PopulationArtifactPaths:
    required = {
        "phase1_run_dir": args.phase1_run_dir,
        "upstream_input_manifest": args.upstream_input_manifest,
        "method_lock_manifest": args.method_lock_manifest,
        "method_lock_artifacts": args.method_lock_artifacts,
        "subgroup_enrichment": args.subgroup_enrichment,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError(
            "production requires live Phase 1 artifact paths; missing=" + ",".join(missing)
        )
    run_dir = Path(args.phase1_run_dir)
    return PopulationArtifactPaths(
        task4_input_table=args.individual_food,
        task4_provenance=args.provenance_manifest,
        phase1_individual_food=run_dir / "individual_food.csv",
        phase1_run_manifest=run_dir / "run_manifest.json",
        phase1_success_marker=run_dir / "_SUCCESS.json",
        upstream_input_manifest=args.upstream_input_manifest,
        method_lock_manifest=args.method_lock_manifest,
        method_lock_artifact_locator=args.method_lock_artifacts,
        subgroup_enrichment=args.subgroup_enrichment,
        method_lock_artifacts=load_method_lock_artifact_locator(
            args.method_lock_artifacts
        ),
    )


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode()


def _write_source_data(output_dir: Path, audit: PopulationSafetyAudit) -> None:
    if output_dir.exists():
        raise ValueError("output directory exists; refusing to overwrite")
    if not audit.verified_input_hashes:
        raise ValueError("source-data writing requires verified production input hashes")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent))
    )
    try:
        artifacts = {
            "population_panel_audit.json": _json_bytes(audit.panel_audit),
            "population_global_summary.json": _json_bytes(audit.global_summary),
            "population_group_convergence.csv": _csv_bytes(audit.group_convergence),
            "population_subgroup_convergence.csv": _csv_bytes(audit.subgroup_convergence),
            "population_between_group_discrimination.csv": _csv_bytes(
                audit.between_group_discrimination
            ),
            "population_transition_matrix.csv": _csv_bytes(audit.transition_matrix),
            "population_reversal_audit.csv": _csv_bytes(audit.reversal_audit),
            "population_bootstrap_summary.csv": _csv_bytes(audit.bootstrap_summary),
        }
        for name, data in artifacts.items():
            (temporary / name).write_bytes(data)
        output_manifest = {
            "status": "complete",
            "evidence_role": audit.evidence_role,
            "input_and_upstream_sha256": dict(
                sorted(audit.verified_input_hashes.items())
            ),
            "output_sha256": {
                name: sha256(data).hexdigest() for name, data in sorted(artifacts.items())
            },
        }
        (temporary / "source_data_manifest.json").write_bytes(
            _json_bytes(output_manifest)
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run(args: argparse.Namespace) -> PopulationSafetyAudit:
    manifest = _read_json(args.provenance_manifest, "provenance manifest")
    config = PopulationSafetyConfig(
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        minimum_valid_replicates=args.minimum_valid_replicates,
        minimum_clusters=args.minimum_clusters,
    )
    if manifest.get("production_label") == "production":
        verified = load_verified_population_input(_production_paths(args), config)
        frame = verified.frame
        manifest = dict(verified.provenance)
        token = verified.verification
    else:
        validate_population_provenance(manifest, allow_test_data=args.allow_test_inputs)
        if not args.allow_test_inputs:
            raise ValueError("non-production input requires --allow-test-inputs")
        try:
            frame = pd.read_csv(BytesIO(args.individual_food.read_bytes()))
        except Exception as error:
            raise ValueError("individual-food table is unreadable") from error
        token = None
    audit = audit_population_safety(
        frame,
        manifest,
        config=config,
        allow_test_data=args.allow_test_inputs,
        verified_artifacts=token,
    )
    if not args.write_source_data:
        print(
            json.dumps(
                {
                    "status": "validated_dry_run",
                    "evidence_role": audit.evidence_role,
                    "panel_status": audit.panel_audit["panel_status"],
                    "output_written": False,
                },
                sort_keys=True,
            )
        )
        return audit
    if manifest.get("production_label") != "production":
        raise ValueError("testing/synthetic inputs cannot be written to results")
    _write_source_data(args.output_dir, audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify live Phase 1 artifacts and audit population safety."
    )
    parser.add_argument("--individual-food", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase1-run-dir", type=Path)
    parser.add_argument("--upstream-input-manifest", type=Path)
    parser.add_argument("--method-lock-manifest", type=Path)
    parser.add_argument("--method-lock-artifacts", type=Path)
    parser.add_argument("--subgroup-enrichment", type=Path)
    parser.add_argument("--write-source-data", action="store_true")
    parser.add_argument("--allow-test-inputs", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument("--minimum-valid-replicates", type=int, default=800)
    parser.add_argument("--minimum-clusters", type=int, default=20)
    return parser


def main() -> None:
    try:
        run(build_parser().parse_args())
    except (TypeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
