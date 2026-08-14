# Final Nature Food Review

Date: 2026-08-14

## Review scope

This review covers the current main manuscript, Supplementary Information,
reference audit, locked attribute-level method, authorized figure package and
machine-readable Source Data. The authorized evidence tier is
`computational_feasibility`. Historical drafts and excluded figures are not
treated as submission evidence.

## Shared assessment

GMNPS now has a coherent computational object: microbiome-derived nutrient
inputs alter selected native nutrient-profile attributes before Food Compass
domain recomposition, while the food-level anchor, attribute bounds, exact
zero-response identity and final score bound remain explicit. The manuscript
correctly treats the synthetic experiment as an implementation positive-control
and does not present it as biological, clinical or external validation.

The current package is not an empirical Nature Food Article. It contains no
authorized production food run and no observed participant-by-meal response
validation. This is an evidence limitation, not a prose defect.

## Reviewer 1: technical soundness

### Strengths

- Calibration occurs at native attribute level rather than as an offset added
  to a completed Food Compass score.
- The microbiome-to-calibration interface now defines the CLR genus matrix,
  development-only sparse health index, nutrient-to-genus direction, central
  finite difference, normalization and leakage boundary.
- The Food Compass baseline residual Spearman coefficient is correctly reported
  as not estimable because the residual prediction is constant.
- The five frozen synthetic inputs are copied into the allowlisted Source Data
  package without altering the frozen artifacts.

### Blocking concern R1-M1: construct validity

The finite-difference calibration input records change in a fitted microbiome
health index along an author-specified nutrient-linked direction. It has not
been shown to measure an individual's response to a nutrient, food or meal.

Resolution requires a locked participant-by-meal test in independent held-out
people, with prespecified metabolic endpoints and comparisons against the food
baseline, microbiome-only, additive food-plus-microbiome, host/diet and combined
multimodal baselines.

### Blocking concern R1-M2: production food implementation

The approved Food Compass/FNDDS artifact chain and production 9,234-food run
are absent. Consequently, baseline reconstruction error, domain behaviour,
clipping, rank preservation, category transitions and food-group heterogeneity
cannot be reported as observed findings.

### Major concern R1-M3: author-specified constants

The attribute fraction, response temperature, final score cap, bridge norm and
allocation weights are prespecified regularization choices rather than
biologically estimated thresholds. They require development-only selection or
justification and locked sensitivity analyses before external testing.

## Reviewer 2: originality and importance

### Strengths

- The work addresses a real methodological tension between a shared public
  nutrition reference and heterogeneous metabolic responses.
- Attribute-level calibration is more interpretable and NPS-compatible than an
  unexplained final-score offset.
- Clinical and causal claims have been removed.

### Blocking concern R2-M1: no direct person-food validity

The correctly specified synthetic positive-control establishes numerical and
assignment fidelity only. It cannot establish precision-nutrition value,
incremental prediction or generalization.

### Major concern R2-M2: distinction from prior work

The current literature base and comparisons do not yet establish the method's
incremental contribution relative to personalized response models, microbiome
health scores, recommender systems and unanchored personalization. These
approaches must be compared on identical splits and endpoints.

### Major concern R2-M3: pivotal role of the microbiome

No observed-data ablation isolates the incremental value of microbiome-derived
calibration beyond demographic, clinical and dietary information. This is
required before describing the microbiome as a pivotal empirical layer.

## Reviewer 3: narrative and submission readiness

### Strengths

- The manuscript separates designed invariants from empirical findings.
- Main and supplementary claims are consistent with the evidence gate.
- Figure S1 labels itself as synthetic and method-only, gives the effective
  replicate unit and uses percentile ranges rather than confidence intervals.

### Blocking concern R3-M1: no main empirical evidence chain

The Results currently lead with method definition and unavailable evidence.
An empirical Article needs the progression `method definition -> production
food behaviour -> direct response validity -> robustness and generalization`.
Audit machinery should then move to Methods or Supplementary Information.

### Major concern R3-M2: figure package

Figure 1 is conceptual and Figure S1 is synthetic. Data-rich main figures for
cohort flow, population food behaviour, direct-response comparisons and
generalization remain blocked.

### Major concern R3-M3: administrative completion

Authors, order, affiliations, correspondence, ORCIDs, funding, contributions,
competing interests, ethics wording, public archive identifiers and licences
are not supplied. Exact C3/E4 claims also require anonymous item-level records.

## Cross-review decision

Repository-local method, claim-gate, figure and compilation issues identified
during Phase 3 are closed. The manuscript is internally defensible as a
computational framework package, but it is not submission-ready as an empirical
Nature Food Article.

The three conditions that change this decision are:

1. an approved production Food Compass/FNDDS run;
2. direct held-out participant-by-meal response validation with strong
   baselines, uncertainty and leakage controls; and
3. complete author, ethics, archive and expert-record metadata.

Until those conditions are met, population preservation, external validity,
biological mechanism, precision-nutrition utility and dietary guidance remain
unsupported claims.
