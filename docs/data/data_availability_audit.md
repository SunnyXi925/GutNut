# PREDICT/ZOE Data Availability and FAIR Audit

**Audit date:** 2026-08-13

**Target article:** Nature, *Gut micro-organisms associated with health, nutrition and dietary interventions*, DOI `10.1038/s41586-025-09854-7`

**Task boundary:** provenance and header-level inventory only

## Scope and non-inspection statement

This audit followed the source hierarchy required for a Nature submission: the
official article and its Data/Code Availability statements were checked first;
repository-specific metadata from ENA, ExperimentHub, Zenodo and GitHub were
then used for cross-verification; FAIR and DataCite criteria were applied last.

No participant-level outcome or label value was opened, parsed, summarized or
used to make an eligibility decision. The uploaded workbook was inspected only
for its SHA-256, sheet names, worksheet dimensions and header-level schema. The
local PREDICT classification relies on the existing `PROVENANCE.md` record and
file-level SHA-256 values; outcome-bearing Parquet/CSV contents were not opened.

The complete machine-readable inventories are:

- `predict_zoe_source_manifest.csv`: source, owner, cohort, unit, size,
  modality, endpoint declaration, one access route, checksum status, one
  analytical role and explicit exclusion reason.
- `predict1_real_vs_synthetic_audit.csv`: field-level local evidence boundary.
- `eligible_for_main_validation.csv`: direct-validation eligibility gate.

At this audit date, no inventoried resource is eligible for main
participant-by-meal validation. Public resources provide microbiome predictors
or aggregate evidence; local response fields are synthetic; the potentially
relevant clinical archive remains controlled and its required endpoint/linkage
schema is unverified.

## Primary statements checked

- [Official Data Availability](https://www.nature.com/articles/s41586-025-09854-7#data-availability)
  identifies GRCh37, public PREDICT metagenomes and limited metadata, ENA
  accessions `PRJEB75460`, `PRJEB75462`, `PRJEB75463` and `PRJEB75464`, public
  profiles at Zenodo concept DOI `10.5281/zenodo.15307999`, public ZOE species
  rankings, a proposal-based ZOE route for individual clinical data, controlled
  Zenodo concept DOI `10.5281/zenodo.17236382`, and the external-cohort archive
  DOI `10.5281/zenodo.17236261`.
- [Official Code Availability](https://www.nature.com/articles/s41586-025-09854-7#code-availability)
  identifies `SegataLab/inverse_var_weight`, MetaPhlAn and Zenodo concept DOI
  `10.5281/zenodo.17236261`.
- The controlled-data procedure names `data.papers@joinzoe.com`, review by a
  sub-panel of the ZOE Scientific Advisory Board within four working weeks,
  ethics/privacy/data-protection criteria and a data-sharing agreement. These
  conditions were recorded as one `controlled_access_repository` route, not as
  current direct-validation authorization.

## Cross-verification status

| Resource | Repository evidence checked on 2026-08-13 | Status and bounded interpretation |
| --- | --- | --- |
| Nature article and supplement | Official HTML anchors and Springer Nature media download | Accessible. Uploaded and official XLSX bytes were identical; SHA-256 `3b6034d6212f0676bbc60b6eb0f3713cd5266799cbaf37e035900afe3892e581`. |
| `PRJEB39223` | ENA study and count APIs | Accessible. Title identifies PREDICT 1; ENA returned 2,196 sample/run records. This is not interpreted as 2,196 participants because the article reports 1,098 PREDICT 1 participants. |
| `EH5458` | Official ExperimentHub metadata snapshot and dispatch metadata | Accessible. Current metadata names `2021-03-31.AsnicarF_2021.relative_abundance`; its dispatch path is official fetch ID `5501`. Sample size is not declared in the record and remains unknown here. |
| `PRJEB75460` | ENA study and count APIs | Accessible. PREDICT 2; 975 ENA sample/run records. |
| `PRJEB75462` | ENA study and count APIs | Accessible. PREDICT 3 US21; 11,797 ENA sample/run records. |
| `PRJEB75463` | ENA study and count APIs | Accessible. PREDICT 3 US22A; 8,469 ENA sample/run records. |
| `PRJEB75464` | ENA study and count APIs | Accessible. PREDICT 3 UK22A; 12,353 ENA sample/run records. |
| `10.5281/zenodo.15307999` | Zenodo API | Accessible. Concept DOI resolves to version DOI `10.5281/zenodo.15308000`; the record exposes compressed metadata and MetaPhlAn profiles under CC BY 4.0. No meal-response endpoint is declared. |
| `10.5281/zenodo.17236382` | Nature statement and Zenodo API | Metadata accessible. Concept DOI resolves to `10.5281/zenodo.17236383`, which exposes an approximately 2.5-GB encrypted archive plus MD5 files. Although Zenodo metadata says `open` and CC BY 4.0, the payload is encrypted and Nature requires ZOE approval; analytical access is therefore controlled. |
| `10.5281/zenodo.17236261` | Zenodo API | Accessible. Concept DOI resolves to software record `10.5281/zenodo.17236262`. The visible files are `inverse_var_weight-master.zip` and `MetaPhlAn-4.beta.1.zip`; no external-cohort data file is visible in current metadata. |
| `SegataLab/inverse_var_weight` v1.0.0 | GitHub release HTML and Git smart-protocol tag metadata | Release page accessible; tag resolves to `ab1a974bf1abd66175d190b55183d2957224b39f`. The unauthenticated GitHub REST request returned HTTP 403, so release HTML and `git ls-remote` were used instead. No licence file was confirmed from the fixed tag. |
| ZOE ranking page | Public ZOE page and Nature Supplementary Table S5 | Accessible. The page labels the resource “ZOE Microbiome Ranking 2024” and applies a non-commercial academic-use condition; the fixed article version is Supplementary Table S5. |

ENA counts above are repository sample/run counts, not inferred participant
counts. The four PREDICT 2/3 ENA counts total 33,594, whereas the article reports
33,596 participants across those cohorts. The two quantities have different
declared units and cannot be reconciled without an authorized participant-to-
sample mapping; no correction or imputation was made.

## Supplementary workbook structural inventory

The official workbook contains one summary sheet and Tables S1-S25. Dimensions
and headers below were obtained without reading analytical cell values.

| Sheet | Dimension | Header-level content |
| --- | ---: | --- |
| SUMMARY | A1:B28 | Table identifier and description |
| S1 | A1:G25 | Cohort and main cohort-characteristic labels |
| S2 | A1:G146 | Cohort, marker, classification AUC summaries, Spearman summary and marker type |
| S3 | A1:I45 | Cohort, marker, AUC/Spearman summaries and consistency statistics |
| S4 | A1:F1907 | Cohort, marker pair, category pair and Spearman correlation |
| S5 | A1:Q665 | SGB and cohort/geography/global ZOE Health and Diet ranks |
| S6 | A1:X105 | SGB rank consistency, dispersion and deviation summaries |
| S7 | A1:M103 | SGB rank, abundance and taxonomy labels |
| S8 | A1:G68 | SGB Health/Diet rank difference and taxonomy labels |
| S9 | A1:D31 | Public BMI meta-analysis cohort and BMI-group labels |
| S10 | A1:K226 | Rank set, statistic, study/country and BMI-group comparisons |
| S11 | A1:Q622 | Country-pair comparison and Health/Diet rank statistics |
| S12 | A1:G60 | BMI meta-analysis effect, uncertainty, p value, confidence interval and group sizes |
| S13 | A1:G60 | BMI cumulative-abundance meta-analysis schema |
| S14 | A1:G60 | Unfavourable Health-rank count meta-analysis schema |
| S15 | A1:G60 | Unfavourable Health-rank abundance meta-analysis schema |
| S16 | A1:G60 | Unfavourable Diet-rank count meta-analysis schema |
| S17 | A1:G60 | Favourable Diet-rank abundance meta-analysis schema |
| S18 | A1:G60 | Unfavourable Diet-rank count meta-analysis schema |
| S19 | A1:G60 | Unfavourable Diet-rank abundance meta-analysis schema |
| S20 | A1:D34 | Public case-control cohort, disease and group-size labels |
| S21 | A1:Q71 | Case-control rank-count effect and uncertainty schema |
| S22 | A1:Q72 | Case-control cumulative-abundance effect and uncertainty schema |
| S23 | A1:Q72 | Normalized Health/Diet score effect and uncertainty schema |
| S24 | A1:Q37 | Richness/Shannon case-control effect and uncertainty schema |
| S25 | A1:Q2679 | Cohort, arm, SGB, aggregate baseline/endpoint abundance, prevalence, multiplicity-adjusted statistics and ranks |

The workbook contains aggregate summaries and species-level outputs. Its sheet
structure does not establish an eligible `participant × meal/food × observed
postprandial response` table. It is therefore restricted to
`aggregate_supporting_evidence`.

## Real-versus-synthetic boundary

The local evidence path is
`data/project_data/predict_multi/L2_phenotype/predict1_real/PROVENANCE.md`
(SHA-256 `44ecf5346533c3a178aa38da60d17b875b3c70a55dced1de7a7428bb8ed686a8`).
It attributes participant/sample identifiers, country, sex, age, BMI, twin and
family fields to curatedMetagenomicData AsnicarF_2021. It separately labels
`visceral_fat_kg`, local diet scores, `glucose_iAUC_2h`, `tg_6h_rise` and
`c_peptide_iAUC_2h` as statistically anchored synthetic fields. This audit did
not independently infer those classifications from data values.

Accordingly, the three local metabolic endpoints are allowed only for a
`synthetic_stress_test`. They cannot be described as observed PREDICT outcomes,
retrospective external validation, construct validity or clinical validity.

## FAIR audit

| Principle | Finding | Required action |
| --- | --- | --- |
| Findable | ENA accessions, ExperimentHub accession, article DOI and Zenodo concept/version DOIs are persistent. Local synthetic artifacts have only local paths and checksums. | Deposit any synthetic benchmark intended for publication with a versioned DOI, README and explicit synthetic-data label. |
| Accessible | Public sequencing/profiles and the supplement resolve. The clinical archive is discoverable but encrypted and requires ZOE review. | Make the controlled status explicit in Zenodo metadata; retain public metadata even when payload access is restricted. |
| Interoperable | ENA sequencing records and compressed TSV profiles use community-oriented formats. The XLSX is human-readable but aggregates heterogeneous analyses. | Supply data dictionaries, units, identifier relationships and participant-sample-meal linkage definitions for any authorized validation extract. |
| Reusable | The public profile record declares CC BY 4.0; ZOE rankings state a non-commercial academic-use condition; the controlled archive requires an agreement. The fixed GitHub release licence was not confirmed. | Clarify software licence, controlled-data rights, allowed reuse and version-specific citations. Do not apply an open licence to participant data without authority. |

## Blocking fields and Data Availability draft risks

1. **No eligible public outcome table.** Public ENA, ExperimentHub and Zenodo
   profile resources contain predictors or limited metadata, not confirmed
   participant-by-meal glucose 2-h iAUC, triglyceride 6-h rise or C-peptide 2-h
   iAUC records.
2. **Controlled archive eligibility is unresolved.** Access is not granted;
   endpoint names, units, time windows, meal/food identifiers and microbiome
   linkage keys remain unknown. Controlled status alone must not be relabelled
   as direct validation.
3. **Zenodo access metadata are internally ambiguous.** Version `17236383` is
   marked open/CC BY while its main file is encrypted and governed by a ZOE
   application route. A Nature-ready statement should describe the route and
   restrictions, not call the underlying clinical data openly available.
4. **External-data archive mismatch.** Nature says non-PREDICT external cohort
   data are available at concept DOI `17236261`, but the current version
   metadata exposes only software archives. The missing data location or file
   manifest must be resolved before relying on this sentence.
5. **Version ambiguity.** Nature cites concept DOIs. Reproducible analysis
   should additionally cite version DOIs `15308000`, `17236262` and `17236383`
   and record access date and file checksums.
6. **Ranking version drift.** The live ZOE page labels a 2024 ranking and can be
   updated; Supplementary Table S5 is the immutable version for this article.
7. **Code licensing.** GitHub v1.0.0 is fixed to commit
   `ab1a974bf1abd66175d190b55183d2957224b39f`, but a software licence was not
   confirmed from that release. Zenodo's CC BY field is not a substitute for an
   explicit software licence.

Until these fields are resolved, a GMNPS Data Availability draft must state
that public PREDICT microbiome resources support predictor reconstruction and
biological consistency only; direct person-meal response validation is blocked
pending controlled access and schema verification. It must not imply that the
local synthetic endpoints are participant observations.
