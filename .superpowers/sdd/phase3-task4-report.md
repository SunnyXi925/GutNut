# Phase 3 Task 4 Report

## Scope

Task 4 rebuilt the active submission figure package under the locked
`computational_feasibility` evidence tier. The package contains exactly one
conceptual main figure and one supplementary correctly specified synthetic
positive-control. Planned empirical Figs. 2-4 remain blocked.

## Delivered displays

- Main Fig. 1 is a 183 x 126 mm four-panel vector schematic of native
  attribute-level calibration. It reads no data, invokes no random generator,
  shows no observations and labels the display as conceptual.
- Supplementary Fig. S1 is a 183 x 72 mm two-panel comparison of residual RMSE
  and residual Spearman correlation. It reads only five frozen Phase 2 files
  after exact SHA-256 verification and does not rerun the simulator.
- S1 uses five independent simulation seeds as the effective repeat unit.
  The 24 individuals x 12 foods within each seed are nested descriptive pairs,
  not independent replicates. Displayed intervals are empirical 2.5th-97.5th
  percentiles across seeds and are not labelled confidence intervals.

## Integrity controls

- The builder is manifest-driven and refuses unauthorized source files,
  outputs or blocked figure rows.
- `figure_manifest.csv` records Fig. 2-4 as `blocked_external_data` with explicit
  unlock artifacts and interpretation boundaries.
- Historical figures and source-data files remain for provenance but are
  `excluded_stale` and absent from the submission allowlist.
- The two panel CSVs, source-data manifest and figure-output manifest record
  content, schema, script and output hashes.
- SVG text remains text, PDFs contain vector content and embedded fonts, and
  PNG previews are RGB/sRGB at 600 dpi.

## Manuscript integration

- Fig. 1 follows the Results method definition.
- Supplementary Fig. S1 follows Supplementary Table S2 and is cited in the
  Discussion as an implementation-fidelity positive-control.
- Captions deny observed population preservation, biological validity,
  clinical validity and external validity.

## Visual review

- Both PNG previews were inspected at original resolution.
- Fig. 1 was revised after the first preview exposed overlapping labels in
  panels a-c. The final layout uses wider input and ruler panels, separated
  point labels and an orthogonal fixed-residual path.
- S1 uses redundant colour and marker encoding, a common comparator order and
  direct numeric labels. No significance marks or inferential pairwise tests
  are shown.

## Verification

- Focused Task 4 contract tests: 12 passed.
- Nature figure static preflight: 12 passed, 2 non-blocking warnings (no TIFF;
  final width is specified dynamically rather than as a static source token).
- `git diff --check`: passed.
- Full repository suite: 978 passed with one existing NumPy/SciPy compatibility
  warning.

## Evidence boundary

This package demonstrates only a locked computational method and implementation
fidelity under correct synthetic specification. Production food scoring,
population preservation, participant-by-meal response validity, biological
consistency and external generalization remain unavailable until the
registry-bound artifacts listed in `figure_manifest.csv` exist and pass their
evidence gates.
