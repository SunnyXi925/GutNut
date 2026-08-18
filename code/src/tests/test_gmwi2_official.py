from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from gmnps.data_sources.gmwi2_official import (
    GMWI2_MAIN_HEAD,
    load_official_gmwi2_sdata5,
    read_xlsx_first_sheet,
)


def _write_minimal_xlsx(path: Path) -> None:
    shared_strings = [
        "Supplementary Data 5 title",
        "Sample accession",
        "Subject health status",
        "GMWI2",
        "SAMN1",
        "Healthy",
        "SAMN2",
        "Non-healthy",
    ]
    shared_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="8" uniqueCount="8">'
        + "".join(f"<si><t>{value}</t></si>" for value in shared_strings)
        + "</sst>"
    )
    sheet_xml = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c></row>
    <row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2" t="s"><v>2</v></c><c r="C2" t="s"><v>3</v></c></row>
    <row r="3"><c r="A3"/><c r="B3"/><c r="C3"/></row>
    <row r="4"><c r="A4" t="s"><v>4</v></c><c r="B4" t="s"><v>5</v></c><c r="C4"><v>1.25</v></c></row>
    <row r="5"><c r="A5" t="s"><v>6</v></c><c r="B5" t="s"><v>7</v></c><c r="C5"><v>-0.75</v></c></row>
  </sheetData>
</worksheet>
"""
    with ZipFile(path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr(
            "xl/workbook.xml",
            """<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="SData5" sheetId="1" r:id="rId1"/></sheets></workbook>""",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/></Relationships>""",
        )
        zf.writestr("xl/sharedStrings.xml", shared_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def test_read_xlsx_first_sheet_reads_header_row_without_openpyxl(tmp_path):
    path = tmp_path / "gmwi2.xlsx"
    _write_minimal_xlsx(path)

    frame = read_xlsx_first_sheet(path, header_row=2)

    assert frame.columns.tolist() == ["Sample accession", "Subject health status", "GMWI2"]
    assert frame["Sample accession"].tolist() == ["SAMN1", "SAMN2"]


def test_load_official_gmwi2_sdata5_exports_runner_schema(tmp_path):
    path = tmp_path / "gmwi2.xlsx"
    _write_minimal_xlsx(path)

    scores = load_official_gmwi2_sdata5(path)

    assert scores.columns.tolist() == ["sample_id", "label", "official_gmwi2_score"]
    assert scores["sample_id"].tolist() == ["SAMN1", "SAMN2"]
    assert scores["official_gmwi2_score"].tolist() == [1.25, -0.75]
    assert scores.attrs["repo_head"] == GMWI2_MAIN_HEAD
