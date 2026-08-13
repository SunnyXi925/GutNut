#!/usr/bin/env python3
"""Fail a Phase 3 manuscript/build input that violates the frozen claim policy."""
from __future__ import annotations

import argparse
from pathlib import Path

from gmnps.validation.claim_policy import check_claim_inputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    arguments = parser.parse_args()
    violations = check_claim_inputs(arguments.inputs)
    for violation in violations:
        print(f"FORBIDDEN_CLAIM {violation.path}: {violation.pattern}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
