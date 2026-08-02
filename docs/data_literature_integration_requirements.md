# Real Data and Literature Integration Requirements

## Purpose

This document defines the next stage for GMNPS: move from framework and
synthetic validation toward real data, real literature and manuscript writing.
The title narrative remains:

> Personalized calibration transforms nutrient profiling systems for precision nutrition

The central constraint is that the selected static NPS baseline is an anchored
prior. Food Compass 2.0 may be used because it has broad and up-to-date NPS
attribute coverage, but the manuscript should not become a Food Compass 2.0
paper.

## Evidence Tiers

### Tier 1: Baseline NPS Prior

Goal: define the population-level static nutrition prior.

Candidate source:

- Food Compass 2.0 score tables or reproducible scoring inputs.

Required checks:

- Confirm food identifiers and food names.
- Confirm score range and interpretation categories.
- Confirm whether full Food Compass 2.0 scores are available directly or must
  be reconstructed from nutrient and food attribute tables.
- Document all joins between food composition data and baseline NPS scores.

Manual help likely needed:

- Provide any licensed or unpublished Food Compass 2.0 score tables already
  used by the team.
- Confirm whether we are allowed to redistribute derived score tables in the
  GitHub repository, or whether they must remain private.

### Tier 2: Food Nutrient Vectors

Goal: create standardized food nutrient vectors aligned to the GMNPS masks.

Candidate sources:

- FNDDS or USDA food composition tables.
- Existing project food-nutrient files if provenance can be audited.

Required checks:

- Standardize nutrient names to match `code/src/gmnps/scoring/masks.py`.
- Record units and serving basis.
- Identify missing values for retinol, choline, B12, fatty acids and
  carotenoids.
- Decide whether carbohydrate appears only in sensitivity analyses.

Manual help likely needed:

- Provide the exact food composition table used in earlier GMNPS experiments,
  if it differs from public FNDDS or USDA sources.
- Confirm serving basis: per 100 g, per serving, or per FNDDS reported amount.

### Tier 3: Microbiome-Derived Calibration

Goal: estimate or validate individual nutrient-response weights for bounded
personalized calibration.

Candidate sources:

- PREDICT-style cohorts with diet, gut microbiome and metabolic response data.
- curatedMetagenomicData for standardized metagenomic profiles and health-state
  metadata.
- ZOE microbiome health or diet rankings as external plausibility anchors.
- Existing GMNPS inference bundles in the local project if provenance passes
  audit.

Required checks:

- Verify sample-level provenance and data-use constraints.
- Separate training, validation and external plausibility checks.
- Prevent study-level leakage.
- Avoid clinical efficacy claims unless a true prospective intervention exists.
- Compare anchored GMNPS against FCS2-only, unanchored microbiome score,
  shuffled microbiome and random microbiome controls.

Manual help likely needed:

- Provide login, download approval, or files for restricted PREDICT or ZOE
  tables if they are not public.
- Confirm whether existing local `datasets/` files can be used in manuscript
  claims.
- Identify which cohort metadata fields are reliable enough for publication.

### Tier 4: Literature Base

Goal: support each manuscript claim with real literature.

Literature groups:

- NPS methods and Food Compass 2.0 baseline rationale.
- Precision nutrition and interindividual metabolic response heterogeneity.
- Gut microbiome links to diet, postprandial response and host metabolic traits.
- Computational validation using retrospective cohorts and simulation.
- Limits of microbiome prediction and causal claims.

Required checks:

- Prefer primary peer-reviewed sources over news or product pages.
- Use product or institute pages only for access details, score availability,
  or implementation context.
- Keep claims modest when evidence is observational or retrospective.
- Track each claim to a citation before drafting manuscript Results and
  Discussion.

Manual help likely needed:

- Open Zotero desktop or provide a BibTeX/RIS export for the GMNPS library.
- Identify must-cite papers from the team.
- Confirm whether any papers are excluded because they conflict with the
  intended claim or are considered weak evidence.

## Proposed Repository Outputs

### Data Registry

Create:

- `docs/data_registry.md`

Fields:

- dataset name
- source
- access status
- license or data-use constraint
- sample or food count
- key variables
- planned GMNPS role
- provenance risk
- manuscript claim level

### Literature Matrix

Create:

- `docs/literature_matrix.md`

Fields:

- claim
- citation
- evidence type
- study population or food scope
- supports which manuscript section
- limitations
- whether citation is primary or contextual

### Provenance Audit

Create:

- `docs/provenance_audit_checklist.md`

Checks:

- source authenticity
- data version
- license
- join keys
- missingness
- leakage risk
- reproducibility path
- allowed manuscript claim

## Immediate Next Tasks

1. Build a data registry with public candidate sources and local files.
2. Audit available local `datasets/` and `data/` folders without committing raw
   data.
3. Connect Zotero or ingest exported bibliography.
4. Create the first literature matrix for the Introduction and Discussion.
5. Implement a retrospective validation runner only after source provenance is
   confirmed.

## Current Blockers

- Zotero MCP connection is currently unavailable from this Codex session.
- Restricted cohort data may require manual download or permission.
- Existing local project data must pass provenance and leakage audit before
  supporting manuscript claims.
