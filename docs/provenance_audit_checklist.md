# Provenance Audit Checklist

Use this checklist before a dataset supports a manuscript claim.

## Source Authenticity

- Record source URL, DOI, accession or official download page.
- Record download date and data version.
- Preserve original filenames outside Git in `data/`, `datasets/` or
  `outputs/`.
- Record whether redistribution is allowed.

## Table Integrity

- Check row counts against the source documentation.
- Check required identifiers and duplicate keys.
- Check numeric ranges and units.
- Record missingness for all primary GMNPS mask nutrients.
- Record any manual corrections or adjudicated joins.

## Join Integrity

- Prefer exact `Foodcode` or documented USDA identifiers.
- Avoid fuzzy food-name joins in primary analyses.
- If fuzzy joins are unavoidable for sensitivity analyses, record match score,
  manual review status and unmatched records.

## Leakage And Validation

- Separate score construction from validation outcomes.
- Do not select masks or model parameters after inspecting validation outcomes.
- Use shuffled microbiome, random microbiome and shuffled-mask controls.
- For cross-cohort analyses, separate cohorts before feature selection when
  feasible.
- Record study, batch and phenotype fields used for adjustment or stratification.

## Allowed Claim Level

- **Baseline prior:** score source defines population-level NPS consensus.
- **Retrospective plausibility:** observational alignment with known
  diet-microbiome-host gradients.
- **Computational feasibility:** known-ground-truth synthetic stress test.
- **Direct validation:** only if individual-level diet, microbiome and metabolic
  response data are available with approved access and leakage-safe evaluation.
