#!/usr/bin/env python3
"""Run a fail-closed population-safety audit of locked GMNPS scores."""
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
    PopulationSafetyAudit,
    PopulationSafetyConfig,
    audit_population_safety,
    validate_population_provenance,
)


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"provenance manifest is missing or invalid: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("provenance manifest must contain a JSON object")
    return payload


def _load_bound_table(path: Path, manifest: dict[str, object]) -> pd.DataFrame:
    expected = manifest.get("individual_food_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("provenance individual_food_sha256 must be a SHA-256 digest")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"individual-food table is missing or unreadable: {path}") from error
    observed = sha256(data).hexdigest()
    if observed != expected:
        raise ValueError(
            f"individual-food SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    try:
        return pd.read_csv(BytesIO(data))
    except Exception as error:
        raise ValueError("individual-food table is not a readable CSV") from error


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _write_source_data(output_dir: Path, audit: PopulationSafetyAudit) -> None:
    if output_dir.exists():
        raise ValueError("output directory already exists; refusing to overwrite")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent))
    )
    try:
        artifacts = {
            "population_global_summary.json": _json_bytes(audit.global_summary),
            "population_group_convergence.csv": _csv_bytes(audit.group_convergence),
            "population_subgroup_convergence.csv": _csv_bytes(
                audit.subgroup_convergence
            ),
            "population_between_group_discrimination.csv": _csv_bytes(
                audit.between_group_discrimination
            ),
            "population_transition_matrix.csv": _csv_bytes(audit.transition_matrix),
            "population_reversal_audit.csv": _csv_bytes(audit.reversal_audit),
            "population_bootstrap_summary.csv": _csv_bytes(audit.bootstrap_summary),
        }
        for name, data in artifacts.items():
            (temporary / name).write_bytes(data)
        manifest = {
            "status": "complete",
            "evidence_role": audit.evidence_role,
            "files": {
                name: sha256(data).hexdigest() for name, data in sorted(artifacts.items())
            },
        }
        (temporary / "source_data_manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run(args: argparse.Namespace) -> PopulationSafetyAudit:
    # Provenance is intentionally validated before opening the score table.
    manifest = _read_manifest(args.provenance_manifest)
    validate_population_provenance(
        manifest, allow_test_data=args.allow_test_inputs
    )
    table = _load_bound_table(args.individual_food, manifest)
    audit = audit_population_safety(
        table,
        manifest,
        config=PopulationSafetyConfig(
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            minimum_valid_replicates=args.minimum_valid_replicates,
        ),
        allow_test_data=args.allow_test_inputs,
    )
    if not args.write_source_data:
        print(
            json.dumps(
                {
                    "status": "validated_dry_run",
                    "evidence_role": audit.evidence_role,
                    "n_foods": audit.global_summary["n_foods"],
                    "output_written": False,
                },
                sort_keys=True,
            )
        )
        return audit
    if manifest.get("synthetic") is True or manifest.get("testing_only") is True:
        raise ValueError("testing/synthetic inputs cannot be written to results")
    _write_source_data(args.output_dir, audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a locked production 9,234-food attribute-level GMNPS table. "
            "The default is validation-only dry run; writing requires "
            "--write-source-data."
        )
    )
    parser.add_argument("--individual-food", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-source-data", action="store_true")
    parser.add_argument(
        "--allow-test-inputs",
        action="store_true",
        help="Permit non-production fixtures for dry-run testing only.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument("--minimum-valid-replicates", type=int, default=800)
    return parser


def main() -> None:
    try:
        run(build_parser().parse_args())
    except (TypeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
