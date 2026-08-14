# Nature Food Submission Readiness Checklist

Date: 2026-08-14

Status key: `PASS`, `BLOCKED_EXTERNAL_DATA`, `AUTHOR_INPUT`, `NOT_APPLICABLE`.

## Method and implementation

| Item | Status | Evidence or action |
| --- | --- | --- |
| Native attribute-level calibration defined | PASS | Locked specification and main/Supplementary Methods |
| Final-score offset excluded as primary method | PASS | Calibration precedes domain recomposition |
| Microbiome-to-`b_in` interface documented | PASS | CLR, health index, bridge direction, central difference and development boundary are stated |
| Expert feedback limited to channel content | PASS | Numerical allocations are identified as author-specified |
| Zero-response identity and score bounds tested | PASS | Repository tests and locked specification |
| Sensitivity settings evaluated | BLOCKED_EXTERNAL_DATA | Lock development-only selection, then evaluate 10/20/30% attribute fractions and 8/12/15-point caps |

## Empirical evidence

| Item | Status | Evidence or action |
| --- | --- | --- |
| Approved Food Compass/FNDDS production artifact | BLOCKED_EXTERNAL_DATA | Complete provenance and release-registry approval |
| Production 9,234-food scoring run | BLOCKED_EXTERNAL_DATA | Required for population behaviour and food-group analyses |
| Baseline reconstruction audit | BLOCKED_EXTERNAL_DATA | Compare reconstructed attributes, domains and final scores with official values |
| Observed participant-meal-microbiome linkage | BLOCKED_EXTERNAL_DATA | Required for direct response validity |
| Prespecified primary response endpoints | BLOCKED_EXTERNAL_DATA | Lock before accessing the final test data |
| Subject-held-out validation | BLOCKED_EXTERNAL_DATA | Report effect sizes, intervals, calibration and permutation nulls |
| Food- or cohort-held-out generalization | BLOCKED_EXTERNAL_DATA | Secondary after valid subject-held-out analysis |
| Strong baseline comparison | BLOCKED_EXTERNAL_DATA | Food baseline, microbiome-only, additive, host/diet and multimodal models |
| Microbiome and attribute-level ablations | BLOCKED_EXTERNAL_DATA | Isolate incremental value of each design component |
| GMrepo/ZOE/knowledge-path analyses | BLOCKED_EXTERNAL_DATA | Supporting biological consistency only, not direct validation |

## Figures and Source Data

| Item | Status | Evidence or action |
| --- | --- | --- |
| Conceptual Figure 1 visibly labelled | PASS | No observed or synthetic data are implied |
| Synthetic Figure S1 labelled method-only | PASS | Caption denies biological, clinical and external validity |
| Constant-residual Spearman handled correctly | PASS | Not estimable; no point, interval or valid replicate count |
| Frozen synthetic inputs included | PASS | Five byte-identical files are allowlisted and hash-bound |
| Panel source tables and manifests | PASS | Machine-readable files and SHA-256 records included |
| Empirical Figures 2-4 | BLOCKED_EXTERNAL_DATA | Remain blocked in `figure_manifest.csv` |

## Manuscript and references

| Item | Status | Evidence or action |
| --- | --- | --- |
| Main and supplementary LaTeX compile | PASS | Main: 8 pages; supplement: 8 pages |
| Undefined citations or references | PASS | None in final compile pass |
| Overfull boxes | PASS | None in final compile pass |
| Claim gate | PASS | Exact default-deny manuscript gate passes |
| Reference metadata audit | PASS | Current 12 cited references are verified for their present claim roles |
| Prior-work taxonomy sufficient for empirical Article | BLOCKED_EXTERNAL_DATA | Expand with same-task comparisons when empirical benchmarks exist |
| Results structured as empirical findings | BLOCKED_EXTERNAL_DATA | Requires production and response results |

## Reproducibility

| Item | Status | Evidence or action |
| --- | --- | --- |
| Focused manuscript/figure suite | PASS | 201 tests passed |
| Full repository suite | PASS | 978 tests passed |
| Dependency warning removed | AUTHOR_INPUT | Re-freeze a supported NumPy/SciPy environment before final archive |
| Source package allowlist | PASS | Only authorized Figure 1, Figure S1, manifests, panel tables and frozen inputs |
| Immutable public code archive | AUTHOR_INPUT | Add URL, release tag, DOI/version and licence |
| Immutable public Source Data archive | AUTHOR_INPUT | Add repository or Supplementary Source Data identifier |

## Governance and declarations

| Item | Status | Evidence or action |
| --- | --- | --- |
| Author names and order | AUTHOR_INPUT | Not supplied |
| Affiliations and correspondence | AUTHOR_INPUT | Not supplied |
| ORCIDs | AUTHOR_INPUT | Not supplied |
| Funding and acknowledgements | AUTHOR_INPUT | Not supplied |
| Author contributions | AUTHOR_INPUT | Not supplied |
| Competing interests | AUTHOR_INPUT | Not supplied |
| Ethics and secondary-data wording | AUTHOR_INPUT | Confirm for every included public or controlled cohort |
| Exact six-expert denominator and expertise claims | AUTHOR_INPUT | Provide anonymous item-level records and denominator audit |
| C3/E4 consensus and translation claims | AUTHOR_INPUT | Provide anonymous item-level C3/E4 records before numerical claims |

## Current release decision

`BLOCKED_EXTERNAL_DATA`

The package may be shared as a computational-feasibility draft. It must not be
submitted as a completed empirical Nature Food Article until the empirical and
administrative blockers above are closed.
