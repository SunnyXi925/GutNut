#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gmnps.data_sources.gmwi2_official import (  # noqa: E402
    GMWI2_MAIN_HEAD,
    GMWI2_REFERENCE_DOI,
    GMWI2_REPO_URL,
    load_official_gmwi2_sdata5,
)


def run(args: argparse.Namespace) -> None:
    scores = load_official_gmwi2_sdata5(args.sdata5)
    if args.dual_channel_scores is not None:
        import pandas as pd

        dual = pd.read_csv(args.dual_channel_scores)
        required = {"sample_id", "gmwi2_dual_channel_score"}
        missing = required.difference(dual.columns)
        if missing:
            raise ValueError(f"--dual-channel-scores missing columns: {sorted(missing)}")
        dual["sample_id"] = dual["sample_id"].astype(str)
        scores = scores.merge(
            dual.loc[:, ["sample_id", "gmwi2_dual_channel_score"]].drop_duplicates("sample_id"),
            on="sample_id",
            how="left",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False)
    manifest = {
        "method": "official_gmwi2_supplement_score_export",
        "input_sdata5": str(args.sdata5),
        "output": str(args.output),
        "n_scores": int(len(scores)),
        "columns": list(scores.columns),
        "gmwi2_repo_url": GMWI2_REPO_URL,
        "gmwi2_repo_main_head": GMWI2_MAIN_HEAD,
        "gmwi2_reference_doi": GMWI2_REFERENCE_DOI,
        "interpretation_boundary": (
            "This exports official GMWI2 supplementary scores as an immutable baseline. "
            "GMNPS dual-channel scores, when merged, are evaluated as an added layer and "
            "are not a reimplementation of GMWI2."
        ),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Export official GMWI2 supplementary scores")
    parser.add_argument(
        "--sdata5",
        type=Path,
        default=Path("/Users/fengxi.25/Desktop/GMWI2/41467_2024_51651_MOESM8_ESM.xlsx"),
    )
    parser.add_argument("--dual-channel-scores", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/project_data/predict_multi/L1_microbiome/gmwi2_official/official_gmwi2_scores.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "data/project_data/predict_multi/L1_microbiome/gmwi2_official/official_gmwi2_scores_manifest.json",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
