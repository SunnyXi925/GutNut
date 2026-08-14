# Phase 3 Task 2 Report: Main-text Reconstruction

Date: 2026-08-14

## Status

**DONE_WITH_CONCERNS.** The review fixes provide a standalone Supplementary
root, the requested reproducibility detail, the bound Supplementary Data mapping
artifact and the required three-table structure. Focused and full tests pass.
No TeX engine is installed locally, so compilation is replaced by documented
static root, input, label and environment checks. Task 3 citation verification,
required author inputs and the existing NumPy/SciPy environment warning remain.

## Scope and dirty-worktree protection

The review-fix pass owns the following manuscript-package files:

- `manuscript/nature_food_submission/sn-article.tex`
- `manuscript/nature_food_submission/sections.tex`
- `manuscript/nature_food_submission/supplementary_methods.tex`
- `manuscript/nature_food_submission/supplementary_results.tex`
- `manuscript/nature_food_submission/supplementary_information.tex`
- `manuscript/nature_food_submission/supplementary_data_1_attribute_mapping_table.csv`

The repository was dirty before editing, including dirty author manuscript and
reference files plus unrelated code, data, figures, documents, and local
artifacts. None of those unrelated paths is owned by this task. The current
`sn-article.tex`, `sections.tex`, and `references.bib` were copied byte-for-byte
before editing to the ignored directory
`.superpowers/sdd/phase3-task2-backup/`. The backup and this report are ignored
by `.gitignore`; the backup remains unstaged, while this report is explicitly
staged for the review-fix record.

### Pre-edit backup integrity

| File | SHA-256 | Source-word count |
| --- | --- | ---: |
| `sn-article.tex` | `46317ab8539cfa90733192bf9b1c32cc30ae42c2337ab942cac74478ee3f090e` | 369 |
| `sections.tex` | `9a75dde32961b8c4adc1dbb7d8464ead60bbd0ddca5b778b7a9ce68cbb80d47c` | 5,897 |
| `references.bib` | `608f64544035bf737a0db73f2707486c39a9d4eaf85d3e7ddb1c5e472740821e` | 2,728 |

The source and backup SHA-256 values matched for all three files immediately
after copying.

### Pre-edit section word counts

These counts use a documented source-text approximation because `texcount`,
`detex`, and LaTeX build tools are not installed. An `awk` audit removed TeX
commands, citations, labels, comments, math-control characters, and complete
table/figure/equation/align environments before counting alphanumeric tokens.

| Section | Words |
| --- | ---: |
| Abstract | 167 |
| Introduction | 597 |
| Results | 996 |
| Discussion | 1,116 |
| Methods | 2,537 |
| Data availability | 129 |
| Code availability | 39 |

## Nature-writing routing and argument lock

Detected route: `task=manuscript`, `paper_type=research`,
`section=title+abstract+introduction+results+method+discussion`,
`language=zh-to-en`, `journal=nature`.

One-sentence argument:

> In nutrient profiling, this work defines bounded microbiome-informed
> calibration at the native-attribute level before domain recomposition,
> supported by locked computational invariants and one correctly specified
> synthetic positive-control family, while unavailable real outcomes and a
> missing production run limit the present evidence to computational
> feasibility and preclude external-validity claims.

Evidence chain:

1. Nutrient profiling supplies a shared population-level reference.
2. Personalized calibration should remain separate from that reference.
3. GMNPS acts on native attributes before Food Compass domain recomposition.
4. A fail-closed audit fixes the current tier at computational feasibility.
5. One correctly specified synthetic positive-control checks programmed-mapping
   and assignment behavior only.
6. Eligible observed participant-by-meal outcomes and production evidence are
   unavailable; external validity is not established.

## Locked terminology ledger

| Canonical term | First-use definition or rule | Source variants rejected or bounded |
| --- | --- | --- |
| nutrient profiling | Population-level assessment of food composition; avoid NPS unless repeated use requires it. | Do not imply a clinical recommendation system. |
| Gut Microbiome-informed Nutrient Profiling System (GMNPS) | Exact expansion at first mention. | Do not use unexplained GMNPS in title or first mention. |
| personalized calibration | Central operation applied at the native-attribute level. | Not replacement scoring or free re-ranking. |
| bounded attribute-level calibration | Primary computational object before native-domain recomposition. | Reject “anchored final-score offset” as the primary method. |
| Food Compass 2.0 | Broad nutrient profiling baseline implementation used here. | Not the paper's subject and not revalidated by this project. |
| native attribute point | Food Compass attribute point before domain aggregation. | Distinct from the final 1–100 score. |
| native-domain recomposition | Dynamic reaggregation of selected native domains after bounded attribute movement. | Not post-hoc addition to the final score. |
| microbiome-derived calibration input | Raw participant-by-nutrient beta input transformed with development-only normalization. | Not a causal nutrient effect or dietary response. |
| 20% native-attribute cap | Primary fraction of each published attribute-point range. | 10% and 30% are named settings only, not observed sensitivity results. |
| ±12-point final cap | Primary bound on deviation from the Food Compass 2.0 anchor. | ±8 and ±15 are named settings only, not observed sensitivity results. |
| zero-response identity | GMNPS equals Food Compass 2.0 when normalized beta response is exactly zero. | Designed and tested invariant, not population preservation evidence. |
| correctly specified synthetic positive-control | Complete qualifier required at every use. | Never “validation cohort”, “external validation”, or “digital twin evidence”. |
| computational feasibility | Current evidence tier from the fixed fail-closed gate. | No biological consistency, clinical validity, or external validity upgrade. |
| available expert feedback | Qualitative content review informing carbohydrate, zinc, copper, and vitamin A RAE revisions. | No exact expert denominator, rounds, percentages, consensus, E4 verdicts, readiness, or mask superiority. |
| digital gut twins | Bounded future direction only. | Not a description of the completed synthetic positive-control. |

## Paragraph and section job map

- Abstract: exactly six sentences: field context; design gap; attribute-level
  approach; audited data limitation/current tier; synthetic positive-control;
  implication plus external-validity boundary.
- Introduction: four paragraphs covering `INT-01/02`, `INT-03/04`,
  `INT-05/06`, and `INT-07/08`, respectively. Each literature claim is one
  physical line preceded by its exact `% CLAIM_ID: INT-xx` marker.
- Results: method establishment; fail-closed tier; unavailable real outcomes
  and production execution; qualitative expert-informed revisions. Synthetic
  controls appear only in Supplementary Results.
- Discussion: method-level contribution; narrow positive-control
  interpretation; expert-evidence boundary; empirical-Article and external-
  validity boundary; frozen future empirical path including bounded multimodal,
  longitudinal, and digital-gut-twin directions.
- Main Methods: scoring object; data/provenance boundary; attribute/channel
  rules; beta interface; recomposition/caps; positive-control; evidence gate.
- Supplementary Methods: full algorithms, units, missingness/provenance,
  software, split/inference plans, and blocked-analysis definitions.
- Supplementary Results: the single positive-control family and an evidence-
  availability boundary table; no blocked result placeholders.

## Evidence and authorship boundaries

- Current tier: `computational_feasibility`.
- Figures 2–4 remain blocked and are not referenced or represented by
  placeholders.
- No current production 9,234-food analysis, participant-by-meal benchmark,
  GMrepo analysis, ZOE rank analysis, or knowledge-path result is available.
- Literature claims and citations remain conditional on Phase 3 Task 3.
- Anonymous item-level C3 and E4 records: `AUTHOR_INPUT_NEEDED`.
- Author names/order, affiliations, correspondence, funding, acknowledgements,
  competing interests, ethics wording, and final contribution statements:
  `AUTHOR_INPUT_NEEDED`. No submission metadata is inferred from the dirty draft.

## Implemented reconstruction

### Main manuscript

- Replaced the unsupported title with the provisional defensible title from the
  locked outline.
- Rebuilt the Abstract as exactly six sentences in the required order. It
  contains no historical population, disease, knowledge-graph, response or
  expert-review statistic.
- Rebuilt the Introduction as four paragraphs carrying `INT-01` through
  `INT-08`. Each of the six literature-context claims occupies one physical
  line immediately after its exact claim marker. All cited keys already exist
  in the committed parent version of `references.bib`; the bibliography itself
  was not edited by this task.
- Reordered Results to method establishment, fail-closed evidence tier,
  unavailable observed/production inputs, and qualitative expert-informed
  revisions. No quantitative figure or blocked result slot appears.
- Rebuilt Discussion around the method-level contribution, the narrow
  positive-control interpretation, the expert-evidence boundary, the explicit
  empirical-Article/external-validity limit, and a frozen future evidence path.
  Multimodal, longitudinal, clinical-risk and digital-gut-twin directions are
  labelled as future work.
- Replaced the historical centred final-score-offset Methods with the native
  attribute-calibration and recomposition object. Main Methods now identify the
  data boundary, units, beta interface, 20% native-attribute cap, dynamic domain
  recomposition, +/-12 final cap, no score centring, zero-response identity,
  positive-control design and fail-closed evidence gate.
- Removed all old figure inclusions and references. No result-like figure
  placeholder was introduced.
- Removed unverified rendered author, affiliation, funding, acknowledgement,
  contribution, ethics, competing-interest and correspondence text. Required
  metadata remain explicit source comments only.

### Supplementary Information baseline

- Before the review-fix pass, `supplementary_methods.tex` recorded attribute
  rules, exposure
  units, ratio gates, effective weights, unavailable/fixed-baseline logic,
  development-only normalization, response construction, point movement,
  fixed-residual recomposition, cap settings, zero identity, content-review
  boundary, positive-control design, future split/inference contracts and
  software versions.
- `supplementary_results.tex` contained the one completed quantitative evidence
  family only. The displayed RMSE and Spearman summaries were checked directly
  against the frozen synthetic summary CSV. Random and Sattolo controls remain
  within that same evidence family.
- The earlier package did not yet establish a standalone Supplementary root,
  versioned S numbering, a bound Supplementary Data mapping artifact, or the
  required three-table layout. These are review-fix requirements rather than
  completed baseline deliverables.

## Post-edit section word counts

The same source-text approximation used for the pre-edit audit gives:

| Section | Words |
| --- | ---: |
| Abstract | 82 |
| Introduction | 187 |
| Results | 369 |
| Discussion | 336 |
| Methods | 556 |
| Data availability | 31 |
| Code availability | 18 |
| Supplementary Methods | 1,552 |
| Supplementary Results | 273 |

## Verification record

- Focused manuscript and evidence-policy tests:
  `190 passed, 1 warning in 15.12 s`.
- Abstract/claim/section audit: exactly six Abstract sentences in required
  order; `INT-01` through `INT-08` occur once and in order; literature claims
  are immediate single physical lines; main sections are ordered; no figure
  insertion is present.
- Parent-reference-key audit: every cited key occurs in the committed parent
  version of `references.bib`; no missing key was found.
- Prohibited-claim phrase audit: no historical population correlation,
  category-transition count, GMrepo AUROC, response-gain, knowledge-graph,
  exact expert-outcome, precision-ready, transformation or mask-superiority
  phrase remains.
- Frozen source-data audit: all eight displayed positive-control estimates and
  interval bounds match the frozen CSV after four-decimal rounding.
- Method audit: primary 20% attribute cap, +/-12 final cap,
  `score_centering: none`, zero-response identity, `NOT_CALCULATED` missingness,
  future subject-level holdout contract and recorded software environment are
  present; no final-score offset is described as primary.
- LaTeX static syntax audit: braces and `begin`/`end` environment stacks are
  balanced in all four owned TeX files. No TeX engine, `latexmk`, `tectonic`,
  `chktex`, `texcount` or `detex` executable is installed, so PDF compilation
  was not available in this task.
- `references.bib` integrity: working and backup SHA-256 both remain
  `608f64544035bf737a0db73f2707486c39a9d4eaf85d3e7ddb1c5e472740821e`.
- Repository-wide and owned-file `git diff --check`: passed.

## Concerns

1. The raw claim-policy script reports six lexical hits across visible main and
   Supplementary text: the required `GMNPS` name, the mandated abstract
   boundary `clinical utility`, the mandated discussion boundary `dietary
   recommendations`, and Supplementary specification uses of `GMNPS`, `method`
   and `analysis`. Manual static review confirmed that none is an unauthorized
   historical population, GMrepo, ZOE, knowledge-path, clinical,
   dietary-guidance, exact C3, exact six-expert or E4-result claim; the named
   evidence resources appear only in explicit unavailable or no-inference
   boundaries.
2. Literature citations remain conditional on Phase 3 Task 3. The cited keys are
   parent-committed keys only, but direct-source verification, stable identifiers,
   retraction checks and `reference_audit.csv` do not yet exist. The strict
   manuscript published-context exception therefore cannot pass until Task 3.
3. Author/submission metadata and anonymous item-level C3/E4 records remain
   `AUTHOR_INPUT_NEEDED`.
4. The repository environment still emits the pre-existing warning that NumPy
   2.4.6 is outside SciPy 1.13.1's declared `<2.3.0` range.

## Review-fix record

Completed corrections:

- Removed direct Supplementary inputs from `sections.tex` and added the
  separately compilable `supplementary_information.tex` root. The root resets
  and prefixes section, equation, figure and table counters with `S`.
- Reconciled the tables as Supplementary Table S1 (locked computational
  invariants and settings), S2 (synthetic positive-control summaries) and S3
  (combined evidence and expert-record boundary). The Discussion now names
  Supplementary Table S2 explicitly rather than using a cross-document
  reference.
- Copied `docs/methods/attribute_mapping_table.csv` byte-for-byte as
  `supplementary_data_1_attribute_mapping_table.csv` and cited it as
  Supplementary Data 1. Its source and submission copies share SHA-256
  `f89ae6703e07a20dd206b1832f10091f130159ba38f785b96f1bb8e05cc41e0a`.
- Added the six exact recomputed domain names, exact added-sugar cut points and
  point sequence, exact NOVA pairs, exact FNDDS 2001-2002 through 2017-2018
  sequence, and the full fail-closed production provenance contract.
- Preserved the six-sentence Abstract with the required final implication and
  boundary, and replaced the clinical/dietary-advice wording with the required
  statement that no analysis evaluated clinical utility or dietary
  recommendations.

Verification:

- Focused Task 2 method, evidence and manuscript tests:
  `uv run pytest -q code/src/tests/test_fcs2_attribute_rules.py ...
  code/src/tests/test_manuscript_claim_gate.py` -- `437 passed, 1 warning`.
- Full suite: `uv run pytest -q` -- `828 passed, 1 warning`.
- Static checks: mapping copy byte equality; six Abstract sentences; no direct
  Supplementary inputs in `sections.tex`; expected inputs only in the
  Supplementary root; exactly three Supplementary table environments; unique
  table labels; no `\\ref` cross-document reference; balanced `begin`/`end`
  counts; and `git diff --check` all passed.
- No `latexmk`, `pdflatex`, `xelatex`, `tectonic`, `chktex`, `texcount` or
  `detex` executable is installed, so neither root could be compiled locally.

## Commit

Baseline commit: `269c23f31b48f5642afbb2820fff7d2a89acc775`
(`docs: reconstruct GMNPS manuscript around locked evidence`). The review-fix
commit records the owned manuscript package, report and progress-ledger update.

The baseline commit contains exactly the four original task-owned manuscript
files. The pre-existing dirty `references.bib` remains byte-identical to its
ignored pre-edit backup and is outside this review-fix scope. The backup and
report remain ignored unless explicitly staged for this task.
