"""Extract Food Compass 2.0 Table S5 from the supplementary PDF text.

The PDF table is used as a baseline-prior source for GMNPS. This parser is
deliberately conservative: it preserves all extracted rows and records duplicate
food codes or missing comparison labels in an audit object instead of silently
resolving them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable

import pandas as pd


EXPECTED_TABLE_S5_ROWS = 9273
FOODCODE_LINE_RE = re.compile(r"^\s*\d{8}\b")

TABLE_ROW_RE = re.compile(
    r"^\s*(?P<Foodcode>\d{8})\s+"
    r"(?P<Description>.*?)\s+"
    r"(?P<Food_group>\d{4,5}_[A-Za-z]+(?: [A-Za-z]+)*)\s+"
    r"(?P<FCS_2_0>\d+(?:\.\d+)?)\s+"
    r"(?P<FCS_1_0>\d+(?:\.\d+)?)\s+"
    r"(?P<Difference>-?\d+(?:\.\d+)?)\s+"
    r"(?P<NOVA>\d+(?:\.\d+)?)\s+"
    r"(?P<HSR>\d+(?:\.\d+)?)"
    r"(?:\s+(?P<Nutri_Score>[A-E]))?\s*$"
)


@dataclass(frozen=True)
class FCS2Audit:
    """Quality-control summary for an extracted Food Compass 2.0 table."""

    row_count: int
    expected_row_count: int
    row_count_matches_expected: bool
    unique_foodcode_count: int
    duplicate_foodcodes: list[str]
    missing_nutri_score_count: int
    score_range_ok: bool
    fcs_difference_ok: bool
    unresolved_line_count: int
    unresolved_line_examples: list[str]
    malformed_foodcode_line_count: int
    malformed_foodcode_line_examples: list[str]

    @property
    def core_checks_ok(self) -> bool:
        """Return true when source shape and numeric checks pass."""

        return self.row_count_matches_expected and self.score_range_ok and self.fcs_difference_ok

    @property
    def join_ready(self) -> bool:
        """Return true when the table is safe for exact keyed joins."""

        return (
            self.core_checks_ok
            and not self.duplicate_foodcodes
            and self.missing_nutri_score_count == 0
            and self.unresolved_line_count == 0
            and self.malformed_foodcode_line_count == 0
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _clean_text_fragment(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _is_page_or_note_line(line: str) -> bool:
    stripped = line.strip()
    return (
        not stripped
        or stripped == "a"
        or bool(re.search(r"Page \d+ of 286", line))
        or stripped.startswith("Foodcode ")
    )


def _iter_table_lines(text: str) -> Iterable[str]:
    in_table = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if "Foodcode" in line and "Description" in line and "FCS 2.0" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("HSR Health") or "non-integer NOVA" in line:
            break
        if _is_page_or_note_line(line):
            continue
        yield line


def parse_fcs2_table_s5_text(text: str) -> tuple[pd.DataFrame, FCS2Audit]:
    """Parse Table S5 text extracted from the Food Compass 2.0 supplement.

    The returned data frame keeps all rows in table order. `Foodcode` is stored
    as a string because leading-zero preservation matters for keyed joins.
    """

    rows: list[dict[str, object]] = []
    unresolved: list[str] = []
    malformed_foodcode_lines: list[str] = []
    pending_prefix: list[str] = []
    current: dict[str, object] | None = None

    def finish_current() -> None:
        nonlocal current
        if current is None:
            return
        parts = current.pop("_description_parts")
        current["Description"] = _clean_text_fragment(" ".join(p for p in parts if p))
        rows.append(current)
        current = None

    for line in _iter_table_lines(text):
        match = TABLE_ROW_RE.match(line)
        if match:
            finish_current()
            row = match.groupdict()
            description = row.pop("Description")
            row["_description_parts"] = [*pending_prefix, description]
            pending_prefix = []
            current = row
            continue

        fragment = _clean_text_fragment(line)
        if FOODCODE_LINE_RE.match(line):
            malformed_foodcode_lines.append(line)
        elif current is not None and not current.get("Nutri_Score") and re.fullmatch(r"[A-E]", fragment):
            current["Nutri_Score"] = fragment
        elif current is not None:
            current["_description_parts"].append(fragment)
        else:
            pending_prefix.append(fragment)

    finish_current()
    df = pd.DataFrame(rows)
    if df.empty:
        audit = FCS2Audit(
            row_count=0,
            expected_row_count=EXPECTED_TABLE_S5_ROWS,
            row_count_matches_expected=False,
            unique_foodcode_count=0,
            duplicate_foodcodes=[],
            missing_nutri_score_count=0,
            score_range_ok=False,
            fcs_difference_ok=False,
            unresolved_line_count=len(unresolved),
            unresolved_line_examples=unresolved[:10],
            malformed_foodcode_line_count=len(malformed_foodcode_lines),
            malformed_foodcode_line_examples=malformed_foodcode_lines[:10],
        )
        return df, audit

    numeric_columns = ["FCS_2_0", "FCS_1_0", "Difference", "NOVA", "HSR"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["Foodcode"] = df["Foodcode"].astype(str)
    df["Nutri_Score"] = df["Nutri_Score"].fillna("")
    df = df[
        [
            "Foodcode",
            "Description",
            "Food_group",
            "FCS_2_0",
            "FCS_1_0",
            "Difference",
            "NOVA",
            "HSR",
            "Nutri_Score",
        ]
    ]

    duplicates = sorted(df.loc[df["Foodcode"].duplicated(keep=False), "Foodcode"].unique().tolist())
    diff_ok = ((df["FCS_2_0"] - df["FCS_1_0"] - df["Difference"]).abs() < 1e-6).all()
    score_range_ok = (
        df["FCS_2_0"].between(1, 100).all()
        and df["FCS_1_0"].between(1, 100).all()
        and df["NOVA"].between(1, 4).all()
        and df["HSR"].between(0, 5).all()
    )
    audit = FCS2Audit(
        row_count=int(len(df)),
        expected_row_count=EXPECTED_TABLE_S5_ROWS,
        row_count_matches_expected=len(df) == EXPECTED_TABLE_S5_ROWS,
        unique_foodcode_count=int(df["Foodcode"].nunique()),
        duplicate_foodcodes=duplicates,
        missing_nutri_score_count=int((df["Nutri_Score"] == "").sum()),
        score_range_ok=bool(score_range_ok),
        fcs_difference_ok=bool(diff_ok),
        unresolved_line_count=len(unresolved),
        unresolved_line_examples=unresolved[:10],
        malformed_foodcode_line_count=len(malformed_foodcode_lines),
        malformed_foodcode_line_examples=malformed_foodcode_lines[:10],
    )
    return df, audit


def extract_text_with_pdftotext(pdf_path: str | Path, pdftotext_bin: str | Path) -> str:
    """Extract PDF text with a Poppler `pdftotext` binary."""

    result = subprocess.run(
        [str(pdftotext_bin), "-layout", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def extract_fcs2_table_s5(
    pdf_path: str | Path | None = None,
    text_path: str | Path | None = None,
    pdftotext_bin: str | Path | None = None,
) -> tuple[pd.DataFrame, FCS2Audit]:
    """Extract Table S5 from a cached text file or directly from the PDF."""

    if text_path is not None:
        text = Path(text_path).read_text(encoding="utf-8", errors="replace")
    elif pdf_path is not None and pdftotext_bin is not None:
        text = extract_text_with_pdftotext(pdf_path, pdftotext_bin)
    else:
        raise ValueError("Provide either text_path or both pdf_path and pdftotext_bin.")
    return parse_fcs2_table_s5_text(text)


def write_extraction_outputs(
    df: pd.DataFrame,
    audit: FCS2Audit,
    output_csv: str | Path,
    audit_json: str | Path,
) -> None:
    """Write extracted rows and JSON audit metadata."""

    output_csv = Path(output_csv)
    audit_json = Path(audit_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    audit_json.write_text(json.dumps(audit.to_dict(), indent=2), encoding="utf-8")


def filter_primary_fcs2_foods(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Exclude rows that are not join-ready for the primary food table.

    Rows with duplicated Foodcode values or missing Nutri-Score labels are kept
    out of the primary join table. They remain available in the full extracted
    table for provenance review and sensitivity analyses.
    """

    data = df.copy()
    duplicate_mask = data["Foodcode"].duplicated(keep=False)
    missing_label_mask = data["Nutri_Score"].fillna("").eq("")
    exclude_mask = duplicate_mask | missing_label_mask
    reasons = []
    for is_dup, is_missing in zip(duplicate_mask, missing_label_mask, strict=True):
        row_reasons = []
        if is_dup:
            row_reasons.append("duplicate_foodcode")
        if is_missing:
            row_reasons.append("missing_nutri_score")
        reasons.append(";".join(row_reasons))
    excluded = data.loc[exclude_mask].copy()
    excluded["exclusion_reason"] = [r for r, keep in zip(reasons, exclude_mask, strict=True) if keep]
    primary = data.loc[~exclude_mask].copy()
    summary = {
        "input_rows": int(len(data)),
        "primary_rows": int(len(primary)),
        "excluded_rows": int(len(excluded)),
        "excluded_duplicate_foodcode_rows": int(duplicate_mask.sum()),
        "excluded_missing_nutri_score_rows": int(missing_label_mask.sum()),
    }
    return primary, excluded, summary
