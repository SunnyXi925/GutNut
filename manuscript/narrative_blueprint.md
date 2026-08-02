# GMNPS Manuscript Narrative Blueprint

## Title

Personalized calibration transforms nutrient profiling systems for precision nutrition

## Central Claim

Nutrient profiling systems can move from static population scores to precision-ready scoring by adding bounded personalized calibration on top of a universal nutrition prior. GMNPS tests this idea by using gut microbiome-informed nutrient responses as a proof-of-concept calibration layer.

## Abstract Opening

Nutrient profiling systems rank foods for population-level dietary guidance, yet static scores cannot represent interindividual metabolic heterogeneity required for precision nutrition. We propose an anchored personalized calibration framework that preserves population nutrition consensus while introducing bounded individual deviations. Using gut microbiome-informed nutrient responses as a proof of concept, GMNPS shows that nutrient profiling can extend toward precision nutrition without abandoning universal dietary guidance.

## Results Architecture

### 1. Static NPS establishes the universal prior

Open Results by defining the selected static NPS baseline as the population-level nutrition consensus. Food Compass 2.0 is the current baseline implementation because of its broad and up-to-date NPS attribute coverage, but the purpose is to show what GMNPS preserves before introducing personalization.

### 2. Personalized calibration is bounded and interpretable

Present the score architecture, the personalized deviation term and the expert-reviewed MAC and lipid channels. Explain that calibration adjusts the baseline score within a bounded range rather than replacing the baseline score.

### 3. Population consensus is preserved

Show that population-mean GMNPS remains highly concordant with the baseline NPS and does not create implausible food-group reversals. This result protects the paper from being read as an unanchored recommendation system.

### 4. Individual heterogeneity is revealed

Show that static NPS has no individual variance while GMNPS creates non-random individual-food variation. Organize the evidence by food group and by mechanistic channel so heterogeneity remains interpretable.

### 5. Gut microbiome provides a proof-of-concept calibration layer

Present microbiome-informed nutrient responses as the biological layer that gives the deviation term meaning. Keep the claim focused on feasibility and interpretability rather than clinical efficacy.

### 6. Digital-gut-twin and retrospective validation support feasibility

Benchmark anchored GMNPS against FCS2-only, unanchored microbiome scoring, random microbiome and shuffled controls. The target pattern is high NPS preservation plus improved personalized signal.

## Discussion Architecture

### Discussion returns to NPS transformation

The Discussion should start from the title claim: personalized calibration transforms NPS by adding bounded individual adaptation to a universal nutrition prior.

### Conceptual implication

The central advance is not a replacement for the baseline NPS. It is a general architecture for precision-ready NPS.

### Biological implication

Gut microbiome is a pivotal proof-of-concept because it links diet, microbial metabolism and host metabolic heterogeneity.

### Methodological implication

Anchoring keeps personalized models from drifting into unstructured recommendation systems. Bounded deviations preserve public-health interpretability.

### Evidence boundary

Retrospective and synthetic validation support feasibility and plausibility, not clinical intervention efficacy.

### Future direction

Future NPS can incorporate genetics, metabolomics, continuous glucose response or lifestyle context if each layer remains anchored, bounded and interpretable.

## Writing Rules

- Use `personalized calibration` in the title, abstract, introduction and discussion.
- Use `anchored personalized deviation` when explaining the formula.
- Avoid excessive quotation marks.
- Avoid em dashes.
- Avoid zombie nouns when a direct verb works.
- Prefer concrete nouns, active verbs and restrained causal language.
- Do not claim clinical intervention efficacy.
- Do not frame GMNPS as replacing Food Compass 2.0.
