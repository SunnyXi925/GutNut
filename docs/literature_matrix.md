# GMNPS Literature Matrix

Working title: **Personalized calibration transforms nutrient profiling systems
for precision nutrition**.

This matrix tracks which sources support each claim. It should be updated with
full bibliographic metadata before manuscript submission.

## Baseline NPS Prior

| Claim | Citation | Evidence type | Manuscript use | Limitation |
| --- | --- | --- | --- | --- |
| A population-level NPS can provide a stable universal nutrition prior for GMNPS. | Food Compass 2.0 is an improved nutrient profiling system to characterize healthfulness of foods and beverages. DOI: `10.1038/s43016-024-01053-3`; PMID: `39379671`. | Nutrient profiling method and item-level supplementary table | Methods; Results baseline preservation | GMNPS should cite it as the selected baseline prior, not as the article's main contribution. |
| Food Compass-style scoring integrates many food attributes into a 1-100 food healthfulness score. | Food Compass is a nutrient profiling system using expanded characteristics for assessing healthfulness of foods. DOI: `10.1038/s43016-021-00381-y`. | Nutrient profiling method | Introduction; Methods rationale | Original Food Compass is background because the primary baseline is Food Compass 2.0. |
| Public food composition resources can provide reproducible nutrient vectors aligned to GMNPS masks. | USDA FoodData Central and FNDDS documentation. | Public data resource | Methods; Data availability | Version, units and join keys must be recorded. |

## Precision Nutrition And Microbiome Calibration

| Claim | Citation | Evidence type | Manuscript use | Limitation |
| --- | --- | --- | --- | --- |
| Individuals can show different postprandial responses to the same foods, motivating personalized calibration of static NPS. | Personalized Nutrition by Prediction of Glycemic Responses. DOI: `10.1016/j.cell.2015.11.001`; PMID: `26590418`. | Human cohort with personalized glycaemic response modelling | Introduction; Discussion | Supports component-level individual response heterogeneity, not GMNPS clinical efficacy. |
| Diet, gut microbiome and host metabolic traits jointly explain interindividual nutritional responses. | Human postprandial responses to food and potential for precision nutrition. DOI: `10.1038/s41591-020-0934-0`; PMID: `32528151`. | Deeply phenotyped cohort | Introduction; retrospective validation rationale | Individual-level data usually require controlled access. |
| Gut microbiome profiles associate with habitual diet and fasting/postprandial cardiometabolic markers. | Microbiome connections with host metabolism and habitual diet from 1,098 deeply phenotyped individuals. DOI: `10.1038/s41591-020-01183-8`; PMID: `33432175`. | Human observational cohort | Introduction; retrospective plausibility | Observational associations should not be phrased as causal mechanisms. |
| Diet-microbiota-host interactions provide a biological basis for microbiome-informed precision nutrition. | Diet-microbiota interactions and personalized nutrition. DOI: `10.1038/s41579-019-0256-8`. | Review | Introduction; Discussion | Contextual source only; not a validation dataset. |

## Retrospective Plausibility Anchors

| Claim | Citation | Evidence type | Manuscript use | Limitation |
| --- | --- | --- | --- | --- |
| Harmonized public metagenomic cohorts can stress-test transportability and confounding. | Accessible, curated metagenomic data through ExperimentHub. DOI: `10.1038/nmeth.4468`. | Public metagenomic data package | Validation; limitations | Not a diet-response cohort. |
| Large public microbiome cohorts can support diet-microbiome plausibility checks. | American Gut: an Open Platform for Citizen Science Microbiome Research. DOI: `10.1128/mSystems.00031-18`; PMID: `29795809`. | Citizen-science microbiome cohort | Validation; limitations | Self-selection and questionnaire bias limit causal claims. |
| Species-level microbiome health rankings can serve as external plausibility anchors. | Gut micro-organisms associated with health, nutrition and dietary interventions. DOI: `10.1038/s41586-025-09854-7`. | Large microbiome association study | External plausibility; Discussion | Rankings should be described as associations, not causal taxa classes. |

## Synthetic And Digital-Twin Rationale

| Claim | Citation | Evidence type | Manuscript use | Limitation |
| --- | --- | --- | --- | --- |
| Digital-twin concepts justify simulation as a controlled feasibility test in precision nutrition. | The Virtual Digital Twins Concept in Precision Nutrition. DOI: `10.1093/advances/nmaa089`; PMID: `32770212`. | Conceptual review | Methods; Validation rationale | Simulation does not replace human validation. |
| Individualized microbiome metabolic models support mechanism-aware in silico dietary perturbation. | MICOM: Metagenome-Scale Modeling To Infer Metabolic Interactions in the Gut Microbiota. DOI: `10.1128/mSystems.00606-19`; PMID: `31964767`. | Computational microbiome modelling method | Methods; Discussion | Model outputs are hypothesis-generating unless externally validated. |

## Writing Guardrails

- Use **personalized calibration** or **bounded personalized deviation** rather
  than unqualified personalized score replacement.
- Use **anchored baseline prior** for Food Compass 2.0.
- Use **retrospective plausibility** for observational cohort analyses.
- Use **computational feasibility** for synthetic digital-gut-twin results.
- Avoid claims of clinical efficacy, causal dietary intervention effects or
  universal good and bad taxa.
