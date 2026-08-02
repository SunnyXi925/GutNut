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

## Results Narrative Architecture

The Results section should not read as a collection of validation exercises. It
should demonstrate the title claim step by step:

1. **Static NPS establishes the universal prior.** Start by showing Food Compass
   2.0 or the selected baseline NPS as the population-level nutrition consensus
   that GMNPS preserves rather than replaces.
2. **Personalized calibration is mathematically bounded and interpretable.**
   Introduce the anchored score architecture, the personalized deviation term
   and the expert-reviewed mechanistic channels.
3. **Population consensus is preserved.** Show that population-mean GMNPS
   remains highly concordant with the baseline NPS and does not produce
   implausible food-group reversals.
4. **Individual heterogeneity is revealed.** Show that the personalized
   deviation creates non-random individual-food variation absent from static
   NPS, with channel-specific patterns across food groups.
5. **Gut microbiome provides a proof-of-concept calibration layer.** Present
   microbiome-informed MAC and lipid/TMAO axes as biologically interpretable
   sources of personalized calibration.
6. **Digital-gut-twin and retrospective validation support feasibility.**
   Benchmark anchored GMNPS against FCS2-only, unanchored microbiome scoring,
   random microbiome and shuffled-mask controls to show the intended trade-off:
   high NPS preservation plus improved personalized signal.

This order makes every result answer the same question: how can NPS become
precision-ready without losing its public-health foundation?

## Discussion Narrative Architecture

The Discussion should return to the broader NPS transformation rather than
ending as a microbiome modelling paper. Recommended order:

1. **Main finding.** Personalized calibration transforms NPS by adding bounded
   individual adaptation on top of a universal nutrition prior.
2. **Conceptual implication.** The central advance is not a replacement for
   Food Compass 2.0, but a general architecture for precision-ready NPS.
3. **Biological implication.** Gut microbiome is a pivotal proof-of-concept
   because it links diet, microbial metabolism and host metabolic heterogeneity.
4. **Methodological implication.** Anchoring prevents personalized models from
   drifting into unstructured recommendation systems; bounded deviations
   preserve public-health interpretability.
5. **Evidence boundary.** Retrospective and synthetic validation support
   plausibility and feasibility, but not clinical intervention efficacy.
6. **Future direction.** Future NPS can incorporate additional calibration
   layers such as genetics, metabolomics, continuous glucose response or
   lifestyle context, provided they remain anchored, bounded and interpretable.

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
