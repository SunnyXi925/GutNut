# Food Compass 2.0 Table S5 Extraction

The user-provided file `FCS2.0_data.pdf` contains the Food Compass 2.0
supplementary information. Table S5 provides item-level scores for 9,273 foods
and is the current source for `FCS2_j`, the anchored baseline prior used by
GMNPS.

## Reproducible Command

From the repository root:

```bash
python3 code/src/scripts/extract_fcs2_table_s5.py \
  --text .codex_work/fcs2/FCS2.0_data.txt \
  --output-csv outputs/fcs2/table_s5.csv \
  --audit-json outputs/fcs2/table_s5_audit.json \
  --strict
```

If cached text is unavailable, use Poppler:

```bash
python3 code/src/scripts/extract_fcs2_table_s5.py \
  --pdf FCS2.0_data.pdf \
  --pdftotext-bin /path/to/pdftotext \
  --output-csv outputs/fcs2/table_s5.csv \
  --audit-json outputs/fcs2/table_s5_audit.json \
  --strict
```

## Output Columns

- `Foodcode`
- `Description`
- `Food_group`
- `FCS_2_0`
- `FCS_1_0`
- `Difference`
- `NOVA`
- `HSR`
- `Nutri_Score`

## Audit Policy

The parser preserves all extracted rows and reports quality checks. The primary
required checks are row count, score range and score difference consistency.
Duplicate food codes and missing comparison labels are reported because the PDF
source contains ambiguous table rows that must be resolved before keyed joins.

Use `--strict` to require the core extraction checks. Use
`--strict-join-ready` when a downstream step requires a unique exact `Foodcode`
join with no missing comparison labels and no malformed table rows.

For manuscript-grade analyses, prefer an official spreadsheet or machine-readable
table from the authors if available. Until that source is obtained, PDF-derived
Food Compass 2.0 values should be described as audited extraction from the
supplementary PDF.
