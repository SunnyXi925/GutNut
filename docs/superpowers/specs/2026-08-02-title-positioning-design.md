# Title Positioning Design

## Decision

The article title is:

> Personalized calibration transforms nutrient profiling systems for precision nutrition

This title positions the paper as a nutrient profiling systems (NPS) methods
article. The main claim is that NPS can evolve from static, population-average
food scoring toward precision nutrition by adding a bounded personalized
calibration layer while preserving population-level nutrition consensus.

## Narrative Strategy

The primary storyline is NPS methodological transformation:

1. Conventional NPS provide population-level dietary guidance.
2. Static food scores cannot express interindividual metabolic heterogeneity.
3. Precision nutrition requires a way to adapt NPS without abandoning public
   health consensus.
4. A Food Compass 2.0-anchored personalized calibration framework provides this
   bridge.
5. GMNPS uses gut microbiome-informed nutrient responses as a proof-of-concept
   calibration layer.

The meaning is elevated to the precision nutrition level, but the manuscript
should remain explicit that it does not claim clinical intervention efficacy.

## Claim Hierarchy

Primary claim:

> Personalized calibration can transform NPS into precision-ready scoring
> systems while preserving population nutrition consensus.

Secondary claim:

> A bounded personalized deviation added to a static NPS prior can reveal
> biologically interpretable individual heterogeneity.

Mechanistic proof-of-concept claim:

> Gut microbiome-informed nutrient responses can instantiate this calibration
> layer in GMNPS.

Validation claim:

> GMNPS demonstrates computational feasibility and retrospective or simulated
> biological plausibility, not causal clinical benefit.

## Terminology

Use `personalized calibration` in the title, abstract, introduction and
discussion because it is clear to nutrition readers and has a constructive
meaning.

Use `anchored personalized deviation` in methods and figure captions when
describing the formula:

```text
GMNPS_ij = clip_1-100(FCS2_j + D_ij)
```

Avoid making `personalized deviation` the title phrase because it is more
mathematical and can sound like departure from nutrition guidance rather than
calibration of it.

## Gut Microbiome Positioning

Gut microbiome should not be the sole subject of the title. It should appear
early in the abstract and introduction as the biological proof-of-concept for
personalized calibration.

Preferred wording:

> Using gut microbiome-informed nutrient responses as a proof-of-concept,
> GMNPS demonstrates that nutrient profiling can be extended toward precision
> nutrition without abandoning universal dietary guidance.

This avoids overclaiming that gut microbiome is the only possible calibration
layer while preserving its pivotal role in the GMNPS framework.

## Abstract Opening Draft

Nutrient profiling systems rank foods for population-level dietary guidance,
yet static scores cannot represent the interindividual metabolic heterogeneity
required for precision nutrition. Here we propose an anchored personalized
calibration framework that preserves population nutrition consensus while
introducing bounded individual deviations. Using gut microbiome-informed
nutrient responses as a proof-of-concept, GMNPS demonstrates that nutrient
profiling can be extended toward precision nutrition without abandoning
universal dietary guidance.

## Manuscript Alignment

The title implies the following changes should be made in a later implementation
step:

- Update manuscript and config title fields to the approved title.
- Reframe the introduction around NPS transformation before introducing GMNPS.
- Define Food Compass 2.0 as the universal NPS prior.
- Define GMNPS as an anchored personalized calibration framework.
- Present gut microbiome as the pivotal proof-of-concept calibration layer.
- Ensure figures show both NPS preservation and personalized heterogeneity.
- Keep clinical efficacy claims out of the main claim language.

## Non-Goals

This title does not claim:

- GMNPS replaces Food Compass 2.0.
- Gut microbiome is the only valid personalization layer.
- The study proves clinical intervention efficacy.
- Personalized scores should override public-health dietary guidance.

## Success Criteria

The title and narrative are successful if a reader understands that:

- The article is within the NPS field.
- The core contribution is a generalizable personalized calibration framework.
- Precision nutrition is the motivating need.
- GMNPS and gut microbiome provide the biological and computational
  demonstration.
- The evidence supports feasibility and plausibility rather than clinical
  causality.
