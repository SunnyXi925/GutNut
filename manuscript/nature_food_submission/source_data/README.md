# GMNPS Phase 3 Task 4 Source Data

Build date: 2026-08-14. Build-base repository revision:
`928f002810878fd3620bd660f85c01d25328bc18`. Evidence tier:
`computational_feasibility`.

Fig. 1 is an entirely conceptual display of the locked method and has no source
data table. Supplementary Fig. S1 is correctly specified synthetic,
method-only evidence. It is not biological, construct, clinical or external
validation. No observed participant, meal, microbiome or food result is
included in this Source Data package.

## Delivered Displays

| Display | Delivered files | Claim IDs | Evidence boundary |
| --- | --- | --- | --- |
| Fig. 1a-d | `figures/figure_1_attribute_calibration.{svg,pdf,png}` | ABS-03, INT-07, RES-01, DIS-01 | Conceptual locked method only; no observed or synthetic result. |
| Supplementary Fig. S1a | `figS1a_programmed_mapping_rmse.csv`; `figures/supplementary_figure_S1_synthetic_positive_control.{svg,pdf,png}` | ABS-05, RES-06, DIS-02 | Correctly specified synthetic implementation-fidelity positive-control. |
| Supplementary Fig. S1b | `figS1b_assignment_control_spearman.csv`; same assembled S1 files | ABS-05, RES-07, RES-08, DIS-02 | Synthetic assignment controls in the same evidence family. |

Fig. 2, Fig. 3 and Fig. 4 are `blocked_external_data`. Fig. 2 requires an
approved production food bundle and population-safety run. Fig. 3 requires
eligible observed participant-by-meal outcomes and an approved direct-validity
run. Fig. 4 requires registry-bound GMrepo, ZOE and knowledge-path runs. None
has a generator path, output path or delivered source table. Exact unlock
artifacts and interpretation boundaries are recorded in
`figures/figure_manifest.csv`.

## Frozen Provenance

Supplementary Fig. S1 is filtered from the five frozen Phase 2 files below.
The builder verifies every SHA-256 value before parsing any table and never
reruns the simulator.

| Frozen source file | SHA-256 |
| --- | --- |
| `synthetic_attribute_twin_config.json` | `fd5a21d815406225f3fe5a6222cd39dafed7b98014dbc25190b61399fdb97c35` |
| `synthetic_attribute_twin_manifest.json` | `df0684efa66377bdbdaec77182cdcbadc03b35cd4c5ec0403febf3c7d1096263` |
| `synthetic_attribute_twin_replicate_metrics.csv` | `b0a0e379e8654ef1c370d614abe932a2a7cd0cedb4ad13ccf52ef6db0a7e70e6` |
| `synthetic_attribute_twin_success_checks.csv` | `091bea04c4baed04fa4f4898d8c980cac081e3bc7dfb58435b5799fdfac32719` |
| `synthetic_attribute_twin_summary.csv` | `b78ab05011a7c5f9bb150b22d50d72ff84797721c6a5c1d3bc8c37a3edbf9bdd` |

The frozen files are under
`results/phase2/source-data/correctly_specified_synthetic_positive_control/`.
`source_data_manifest.csv` records a canonical digest of all five file hashes,
a canonical digest of their frozen payload hashes, each generated table hash,
the ordered-column schema hash and the generator-script hash.

## Analysis And Interval Definition

The analysis and inference unit is the independent simulation seed. Every
displayed summary uses five valid seeds of five requested: 1701, 1702, 1703,
1704 and 1705. Each seed contains 24 synthetic individuals and 12 synthetic
foods, giving 288 nested synthetic person-food pairs per seed. The 1,440 pairs
across seeds are descriptive nested pairs, not independent inferential
replicates.

The large point is the across-seed mean. The horizontal line is the 95%
empirical replicate interval, defined as the empirical 2.5th and 97.5th
percentiles of the five seed-level estimates. It is not a confidence interval
or a clinical uncertainty interval. The panel tables also retain the frozen
200-person-bootstrap interval for each comparator and seed. Frozen upstream
column names ending in `_ci_lower` or `_ci_upper` refer only to those per-seed
participant-bootstrap intervals; they are not the displayed across-seed
interval.

## Comparator Order

Comparators appear in this fixed order in both panels:

1. `fcs_baseline`: Food Compass baseline.
2. `locked_attribute_gmnps`: locked attribute GMNPS.
3. `random_microbiome`: random microbiome assignment control.
4. `sattolo_deranged_microbiome`: deterministic Sattolo-deranged assignment control.

The controls are part of the same correctly specified synthetic evidence
family; they are not independent cohorts.

## Column Dictionary

Both panel CSVs contain one row per comparator and seed (20 rows). Empty values
are not permitted.

| Column | Definition and units |
| --- | --- |
| `comparator_order`, `comparator_key`, `comparator_label` | Fixed display order, machine key and display label. |
| `seed` | Independent simulation-seed identifier. |
| `residual_rmse` | Seed-level residual root-mean-square error in score units; S1a only. |
| `residual_spearman` | Seed-level Spearman rank correlation, unitless; S1b only. |
| `mean_across_seeds` | Frozen across-seed arithmetic mean for the panel metric. |
| `replicate_interval_lower`, `replicate_interval_upper` | Frozen empirical 2.5th and 97.5th percentiles across five seeds. |
| `replicates_requested`, `replicates_valid` | Requested and valid independent seed counts; both equal 5. |
| `n_individuals_per_seed`, `n_foods_per_seed`, `n_pairs_per_seed` | Nested descriptive dimensions: 24, 12 and 288. |
| `evidence_role`, `data_class` | `correctly_specified_synthetic_positive_control` and `synthetic`. |
| `seed_set` | Canonical five-seed set. |
| `source_file_sha256`, `source_payload_sha256` | Canonical digests over the verified frozen file and payload hash records. |
| `bootstrap_replicates_requested`, `bootstrap_replicates_valid` | Per-seed participant-bootstrap counts; 200 requested and valid. |
| `per_seed_bootstrap_interval_lower`, `per_seed_bootstrap_interval_upper` | Frozen per-seed participant-bootstrap interval for the panel metric; not plotted. |

## Deterministic Build

From the repository root, run:

```text
PYTHONPATH=code/src .venv/bin/python code/src/scripts/make_main_figures.py
```

The build uses CPython 3.11.15, NumPy 2.4.6, pandas 3.0.5, Matplotlib 3.11.1
and Pillow 12.3.0 in the verified environment. Panel CSVs are UTF-8 with Unix
newlines and deterministic float formatting. SVG text remains editable text;
PDF output is vector with embedded fonts; PNG output is RGB at 600 dpi. Fixed
metadata and a fixed SVG hash salt make clean repeated builds byte-stable.

## Package And Exclusions

`figures/submission_asset_allowlist.txt` is the package boundary. It contains
only the two authorized figures, their manifests and the two S1 source tables.
`figures/figure_output_manifest.csv` records output hashes and technical
properties. Historical figure and source-data files remain in the repository
for provenance but are marked `excluded_stale` in `figure_manifest.csv` and
cannot enter the allowlist. Historical figure builders and
`figure_ai_prompts.md` do not supply submission assets.

The frozen synthetic bundle and generated tables are repository-local research
artifacts. No controlled human data or third-party restricted microdata are
redistributed here. A deposited Source Data identifier has not yet been
assigned; the repository hash manifests are the current integrity identifiers.
