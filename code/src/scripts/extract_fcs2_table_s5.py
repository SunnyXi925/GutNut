#!/usr/bin/env python3
"""Extract Food Compass 2.0 Table S5 into CSV plus an audit JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gmnps.data_sources.fcs2_pdf import extract_fcs2_table_s5, write_extraction_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Food Compass 2.0 Table S5")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Cached pdftotext output from FCS2.0_data.pdf")
    source.add_argument("--pdf", help="Food Compass 2.0 supplementary PDF")
    parser.add_argument("--pdftotext-bin", help="Poppler pdftotext binary, required with --pdf")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless row count and numeric core checks pass.",
    )
    parser.add_argument(
        "--strict-join-ready",
        action="store_true",
        help="Exit non-zero unless extraction is ready for exact keyed joins.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.pdf and not args.pdftotext_bin:
        raise SystemExit("--pdftotext-bin is required when --pdf is used")

    df, audit = extract_fcs2_table_s5(
        pdf_path=args.pdf,
        text_path=args.text,
        pdftotext_bin=args.pdftotext_bin,
    )
    write_extraction_outputs(df, audit, args.output_csv, args.audit_json)
    print(json.dumps(audit.to_dict(), indent=2))

    if args.strict and not audit.core_checks_ok:
        raise SystemExit(1)
    if args.strict_join_ready and not audit.join_ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
