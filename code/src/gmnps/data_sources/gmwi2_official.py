"""Official GMWI2 supplementary-data readers.

These helpers read the Nature Communications GMWI2 supplementary workbooks
without depending on optional Excel engines. The project uses the resulting
official scores as an external baseline, then layers GMNPS dual-channel
features on top; it does not rebrand GMNPS as a GMWI2 reimplementation.
"""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pandas as pd


GMWI2_REPO_URL = "https://github.com/danielchang2002/GMWI2"
GMWI2_MAIN_HEAD = "9850ec74d9ba88c2589cf3e25696b14a846acda2"
GMWI2_REFERENCE_DOI = "10.1038/s41467-024-51651-9"

_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def _column_number(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    number = 0
    for letter in letters:
        number = number * 26 + ord(letter.upper()) - ord("A") + 1
    return number


def _shared_strings(zip_file: ZipFile) -> list[str]:
    try:
        xml = zip_file.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(xml)
    values = []
    for item in root.findall("main:si", _NS):
        texts = [node.text or "" for node in item.findall(".//main:t", _NS)]
        values.append("".join(texts))
    return values


def _first_sheet_path(zip_file: ZipFile) -> str:
    workbook = ET.fromstring(zip_file.read("xl/workbook.xml"))
    rels = ET.fromstring(zip_file.read("xl/_rels/workbook.xml.rels"))
    first_sheet = workbook.find("main:sheets/main:sheet", _NS)
    if first_sheet is None:
        raise ValueError("workbook contains no sheets")
    rel_id = first_sheet.attrib[f"{{{_NS['rel']}}}id"]
    for rel in rels:
        if rel.attrib.get("Id") == rel_id:
            target = rel.attrib["Target"]
            return "xl/" + target.lstrip("/")
    raise ValueError(f"workbook missing relationship for first sheet: {rel_id}")


def read_xlsx_first_sheet(path: Path, header_row: int = 2) -> pd.DataFrame:
    """Read the first worksheet from a simple XLSX file.

    Parameters
    ----------
    path:
        XLSX workbook path.
    header_row:
        One-based row number containing column names. The GMWI2 supplementary
        workbooks use row 1 as a descriptive title and row 2 as the real header.
    """

    path = Path(path)
    if header_row < 1:
        raise ValueError("header_row must be one-based and positive")
    with ZipFile(path) as zip_file:
        shared = _shared_strings(zip_file)
        sheet = ET.fromstring(zip_file.read(_first_sheet_path(zip_file)))
        rows: dict[int, dict[int, object]] = {}
        for row in sheet.findall(".//main:row", _NS):
            row_number = int(row.attrib["r"])
            values: dict[int, object] = {}
            for cell in row.findall("main:c", _NS):
                ref = cell.attrib.get("r", "")
                col = _column_number(ref)
                cell_type = cell.attrib.get("t")
                value_node = cell.find("main:v", _NS)
                inline_node = cell.find("main:is/main:t", _NS)
                if cell_type == "s" and value_node is not None:
                    value: object = shared[int(value_node.text or 0)]
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = inline_node.text or ""
                elif value_node is None:
                    value = None
                else:
                    raw = value_node.text or ""
                    try:
                        value = float(raw)
                    except ValueError:
                        value = raw
                values[col] = value
            rows[row_number] = values

    header_values = rows.get(header_row)
    if not header_values:
        raise ValueError(f"header row {header_row} is empty")
    max_col = max(max(values) for values in rows.values() if values)
    columns = [header_values.get(col) or f"Unnamed: {col - 1}" for col in range(1, max_col + 1)]
    records = []
    for row_number in sorted(row for row in rows if row > header_row):
        values = rows[row_number]
        record = [values.get(col) for col in range(1, max_col + 1)]
        if any(value not in (None, "") for value in record):
            records.append(record)
    return pd.DataFrame(records, columns=columns)


def load_official_gmwi2_sdata5(path: Path) -> pd.DataFrame:
    """Load Supplementary Data 5 into the project score-file schema."""

    frame = read_xlsx_first_sheet(path, header_row=2)
    required = {"Sample accession", "Subject health status", "GMWI2"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"GMWI2 Supplementary Data 5 missing columns: {sorted(missing)}")
    result = frame.loc[:, ["Sample accession", "Subject health status", "GMWI2"]].rename(
        columns={
            "Sample accession": "sample_id",
            "Subject health status": "label",
            "GMWI2": "official_gmwi2_score",
        }
    )
    result["sample_id"] = result["sample_id"].astype(str).str.strip()
    result["label"] = result["label"].astype(str).str.strip()
    result["official_gmwi2_score"] = pd.to_numeric(result["official_gmwi2_score"], errors="coerce")
    result = result.dropna(subset=["official_gmwi2_score"])
    result = result.loc[~result["sample_id"].isin({"", "nan", "None"})].drop_duplicates("sample_id")
    result.attrs["source"] = "GMWI2 Supplementary Data 5"
    result.attrs["repo_url"] = GMWI2_REPO_URL
    result.attrs["repo_head"] = GMWI2_MAIN_HEAD
    result.attrs["doi"] = GMWI2_REFERENCE_DOI
    return result.reset_index(drop=True)
