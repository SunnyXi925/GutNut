# Zotero Integration Notes

Zotero Desktop was readable from Codex, but the current MCP connection is in
local-only mode. Write operations and bibliography rendering require Zotero web
API credentials.

## Current Read Status

- Collection `GMNPS`: `74X7A6YY`
- Collection `GMNPS/Data`: `3NNHK5KM`
- Confirmed local items include:
  - Food Compass 2.0 is an improved nutrient profiling system to characterize
    healthfulness of foods and beverages.
  - Food Compass is a nutrient profiling system using expanded characteristics
    for assessing healthfulness of foods.
  - Gut micro-organisms associated with health, nutrition and dietary
    interventions.
  - Human postprandial responses to food and potential for precision nutrition.
  - Microbiome connections with host metabolism and habitual diet from 1,098
    deeply phenotyped individuals.

## Blocked Automation

The following operations were blocked by local-only mode:

- Adding missing DOI metadata to the GMNPS collection.
- Exporting a Zotero-rendered BibTeX file or Nature-style bibliography.

To enable automated Zotero writes and bibliography export, configure:

- `ZOTERO_API_KEY`
- `ZOTERO_LIBRARY_ID`

## DOI Checklist For Manual Zotero Completion

- `10.1038/s43016-024-01053-3`
- `10.1038/s43016-021-00381-y`
- `10.1016/j.cell.2015.11.001`
- `10.1038/s41591-020-0934-0`
- `10.1038/s41591-020-01183-8`
- `10.1038/s41586-025-09854-7`
- `10.1038/s41591-024-02951-6`
- `10.1038/s41564-024-01870-z`
- `10.1038/nmeth.4468`
- `10.1128/msystems.00031-18`
- `10.1093/advances/nmaa089`
- `10.1128/msystems.00606-19`
