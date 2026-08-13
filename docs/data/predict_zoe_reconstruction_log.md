# PREDICT/ZOE Predictor-First Reconstruction Log

## Status

Audit date: 2026-08-13

Baseline: `7787d59f3eb5634c73e20b324b1d4277fadc31dd`

Overall status: **IMPLEMENTATION_COMPLETE_EXECUTION_BLOCKED.** Stage 2A and the
source-independent configuration contract are implemented; data-dependent
Stages 2B, 2C and 2D remain blocked.

No `results/phase2/method_lock_manifest.json` was generated. The trusted FNDDS
release registry remains unchanged and contains no approved production bundle.
No controlled, synthetic-response, or Nature/ZOE aggregate-outcome table was
used to construct or evaluate a score.

Final review hardening closes the predictor loader TOCTOU boundary. After the
repository registry and exact no-symlink path policy pass, each predictor is
read once through a single regular-file descriptor into immutable bytes.
SHA-256 verification, header/schema checks and parsing all consume that same
snapshot; CSV, TSV, JSON, Parquet and RData never reopen the source path after
hashing. Twenty adversarial tests replace the original path with a malicious
regular file or symlink after snapshot hashing and after schema validation;
all return only the original snapshot. Focused, related and full verification
passed with 95, 216 and 526 tests, respectively.

## Data boundary

The implemented acquisition allowlist is restricted to microbiome profiles,
sequencing accession metadata, Zenodo record metadata, and historical USDA
food-composition releases. Raw FASTQ files are excluded. The committed
predictor artifact registry—not the mutable local acquisition manifest—is the
trust root for stable source, exact cache path, locally audited SHA-256,
content class, analytical role and schema. Loader callers can select only a
`source_id`; they cannot supply a path, digest, content class or column
allowlist. The reviewed registry implementation SHA-256 is
`d7509d6d3349fb4ae144ae771b1ffdb8e88775faa2c510ddd9d678d045ae7837`.
The staged outcome loader requires a repository-trusted
direct-validation grant before it will read a method-lock manifest. The current
production registry contains no such grant.

Two discovery commands crossed the stricter literal label-bearing boundary,
although neither opened a postprandial outcome table: a broad source search
printed rows from the public curatedMetagenomicData sample metadata, and the
Phase 1 beta provenance manifest exposed aggregate health-label diagnostics.
No displayed label was retained, analysed, or used for reconstruction. After
this incident, searches were restricted to schemas, identifiers and explicitly
allowlisted predictor files. This is recorded as a process concern rather than
reported as clean boundary compliance.

## Stage 2A: official predictor-only resources

Artifacts were cached under the ignored directory
`data/project_data/predict_multi/L2_phenotype/predictor_reconstruction/cache`.
The acquisition manifest is a generated local audit artifact and is not a
repository source file or trust root. Because the official predictor endpoints
did not supply SHA-256 checksums, the six successful downloads are explicitly
classified as locally audited digests and their exact values are now pinned in
the committed predictor artifact registry. A future unpinned download is
reported as `downloaded_locally_audited_unpinned` and remains ineligible for
loading until its digest is reviewed and committed.

| Resource | Stable official source | Verified bytes | SHA-256 | Predictor unit |
| --- | --- | ---: | --- | --- |
| ExperimentHub `EH5458` (`fetch/5501`) | `https://experimenthub.bioconductor.org/fetch/5501` | 755,545 | `89c635061b357583351ca33f520a72d0efced4a2963e73182e529228c8395c54` | 1,098 unique microbiome profile IDs; 1,019 taxonomic rows |
| ENA `PRJEB39223` metadata | ENA Portal API, accession query | 152,736 | `d630be36c1abd3a71b6aa295dc56e1557557f965f6c8fdaa673759b227b28e36` | 2,196 unique run/sample records: 1,098 WGS and 1,098 amplicon |
| ENA `PRJEB75460` metadata | ENA Portal API, accession query | 68,364 | `2ce6d90cd5d380581737a9dc193460b3f5df7f836f443845bba793ecb67de1a9` | 975 unique WGS run/sample records |
| ENA `PRJEB75462` metadata | ENA Portal API, accession query | 825,904 | `ca6d7d5a1b7fdcd069a309b5bc8dd076a53ea8bc4016ec98b1b755658fd20f08` | 11,797 unique WGS run/sample records |
| ENA `PRJEB75463` metadata | ENA Portal API, accession query | 592,944 | `ce028e6072e610ad7575ba7e494347892cd66e33a5eaec92dea7177bcb413f56` | 8,469 unique WGS run/sample records |
| ENA `PRJEB75464` metadata | ENA Portal API, accession query | 864,824 | `f81d06da5f9127864967f99bb13a2b7f6eedddc8c56282f88c5b04985cd4bb94` | 12,353 unique WGS run/sample records |

The Zenodo API endpoint
`https://zenodo.org/api/records/15308000` was attempted on 2026-08-13 and
failed before HTTP negotiation with `[Errno 61] Connection refused`. No Zenodo
file was downloaded, and no alternate host was treated as equivalent.

The ExperimentHub RDA was parsed only to validate predictor structure and IDs,
using `rdata` 1.1.0 in the local ignored environment. Its 1,098 profile IDs all
overlap the 15,492 IDs in the existing Phase 1 beta output. Therefore it cannot
serve as an independent held-out scoring cohort. The public PREDICT-2/3 profile
bundle would be the preferred small predictor source, but it was inaccessible
through the blocked Zenodo endpoint. ENA raw WGS reconstruction was not started
because it would require large FASTQ downloads and a separately frozen
profiling pipeline.

### Reproducible PREDICT-1 overlap audit

Only identifier axes were read. The Phase 1 development set was taken from the
`sample_id` index of
`data/project_data/predict_multi/L7_nutrient_bridge_beta_i/W_personalized.parquet`
(artifact SHA-256
`6552d061524da99e544c55c3d7cfeafd7c2eef999222b755d4c336bc40cec1b1`).
The EH5458 set was taken from the sample axis (`dim_1`) of the pinned RDA
(artifact SHA-256
`89c635061b357583351ca33f520a72d0efced4a2963e73182e529228c8395c54`).

Canonicalization preserves exact case and the complete identifier string,
rejects empty or duplicate IDs, sorts by Python Unicode code-point order,
encodes UTF-8, joins records with `\n`, and adds no trailing newline.

| ID set | n | Canonical SHA-256 |
| --- | ---: | --- |
| Phase 1 development | 15,492 | `e93e35038391cf56b4781e138841a93bd7f84c5d3ceb532d02aaf2f7dade6082` |
| EH5458 profiles | 1,098 | `db9a7fabfa390e5d1b91b32c2b9569aa3d1d5e57f9708297bfa91c1e1d92974c` |
| Exact intersection | 1,098 | `db9a7fabfa390e5d1b91b32c2b9569aa3d1d5e57f9708297bfa91c1e1d92974c` |

Thus the public-only EH5458 scoring set has `n = 0`. The digest helper and
registry constants are covered by reproducibility tests; no response or label
column was read for this audit.

## Stage 2B: historical FNDDS acquisition and food-bundle gate

The official index was
`https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fndds-download-databases/`.
The official Food Compass 2.0 supplementary PDF at
`https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs43016-024-01053-3/MediaObjects/43016_2024_1053_MOESM1_ESM.pdf`
was byte-identical to the local source: 3,295,889 bytes, SHA-256
`26af914baa2662fe65e17b607c5a629baa01a043016d21cdab7bd3c03fc3eab4`.
No old extracted food table or old scoring output was used as a production
substitute.

Verified USDA release artifacts:

| Cycle / artifact | SHA-256 |
| --- | --- |
| 2001–2002 `FNDDS1_ASCII.EXE` | `314da60aef851fa9ba8319a43d7f3347b5eb0421934449ed284f222c4d10907b` |
| 2003–2004 `FNDDS2_ASCII.EXE` | `3d1354db6729e6fb2e6954083fed3d45b97a4c8750a815cba9109a6a1395ffd1` |
| 2005–2006 `FNDDS3_ASCII.EXE` | `0c21ddea68f8ed423320ff7f9f6fbe8c1047b4507dbe0e4158e65edd4d2dcc79` |
| 2007–2008 `FNDDS4_ASCII.EXE` | `9566dbf3f08ec2a1f2ae1eefb3aeff641d6d0ae48467194646cffb94752bd844` |
| 2009–2010 `FNDDS5_ASCII.EXE` | `bea1e45b23ab09bff6de63dd4538ef09e7a77f3a12f6ae8f95ed07e782b0f240` |
| 2011–2012 `FNDDS_2011-2012_ASCII.EXE` | `04966072cca1a4b8c183492d2f2666f88a62127342eccc2e80aa1d02ebbbab86` |
| 2013–2014 `FNDDS_2013-2014_ACCESS.EXE` | `61dd33769ecdd45b3db43058424ae2622f6da263c42728e0850d34bbbf89c035` |
| 2015–2016 Foods and Beverages | `a3289f72a032d9453e4f5dcb51e922ff476c07c71b79d32db378070f95af55d9` |
| 2015–2016 Portions and Weights | `6cf6ab20d7b66ffa3ebf283963255e10b0e95c3a5b8081e7d1f71ea9c9daf5f9` |
| 2015–2016 FNDDS Ingredients | `ee64dd919aba271c605e997a5fba7bfd42a89eb994c3a1e0caa907487ab28981` |
| 2015–2016 FNDDS Nutrient Values | `348792470ca422e3c739a37d7d1bd9866beaea3fb143e4143bf51e34a1f2c59b` |
| 2017–2018 Foods and Beverages | `c1c5c25ad0f58c1823990e09ca09d2c690aa70beb8f6d51d7e2fac8a010c3250` |
| 2017–2018 Portions and Weights | `03b809944cdcddcb753d1f774248b54c903e72ff94a567aef8a6f442d8c066fd` |
| 2017–2018 FNDDS Ingredients | `e1a55b8a2d431f4bb52ce6e41bb9730337fa55d0cf25773fe8a6fb35e6d2b91a` |

The first script version did not persist per-request start timestamps or HTTP
status before transfer completion. Two GET rounds were attempted during
2026-08-13 17:33–17:52 CST. The following official URLs accepted a response but
did not complete transfer; retries were stopped on user instruction. A separate
HEAD request for the 2017–2018 FNDDS Nutrient Values URL returned HTTP 200 at
2026-08-13 17:29:21 CST. For the other two URLs, no HTTP error was raised before
the stalled body transfer, but the exact status was not persisted and is
therefore reported as unknown rather than inferred.

- Missing 2015–2016 Ingredient Nutrient Values:
  `https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2015-2016%20FNDDS%20At%20A%20Glance%20-%20Ingredient%20Nutrient%20Values.xlsx`
  (HTTP status not persisted; partial body transfer; two attempts).
- Missing 2017–2018 Ingredient Nutrient Values:
  `https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2017-2018%20FNDDS%20At%20A%20Glance%20-%20Ingredient%20Nutrient%20Values.xlsx`
  (HTTP status not persisted; transfer did not complete; two attempts).
- Missing 2017–2018 FNDDS Nutrient Values:
  `https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2017-2018%20FNDDS%20At%20A%20Glance%20-%20FNDDS%20Nutrient%20Values.xlsx`
  (HEAD HTTP 200; GET transfer did not complete; two attempts).

Seven incomplete `.part` files created by these attempts were deleted; they
contained no accepted artifact and are not recoverable.

The production food bundle remains blocked for three independent reasons:

1. the 2015–2016 and 2017–2018 official release sets are incomplete;
2. the official 9,273-row Food Compass table has not been freshly reconstructed
   and audited to the required 9,234 aligned-food production universe; and
3. cycle-complete FNDDS nutrients alone do not provide all required FPED,
   flavonoid, additive, processing, baseline-attribute-point and linkage fields.

Consequently, no `official_fcs`, `food_metadata`,
`baseline_attribute_points`, `food_exposures`, or
`effective_attribute_weights` production bytes were minted through the Phase 1
same-immutable-bytes loader. No entry was added to
`fcs2_fndds_release_registry.json`.

## Frozen analysis configuration

`code/src/configs/person_meal_validation.yaml` is JSON-compatible YAML with
review-hardened pre-outcome SHA-256
`a8208ebe9c254d163825f9ff8e57f8005aaaed2cd528f680db7a30e1f6662ed7`.
It defines the source-independent `participant_id + meal_id` unique key;
requires both primary endpoints (`glucose_iAUC_2h` and `tg_6h_rise`); applies a
predeclared availability policy to secondary `c_peptide_iAUC_2h`; and freezes
each endpoint's role, unit, time window, summary and derivation. Caller-selected
endpoint subsets are forbidden. It also freezes missingness,
participant/family/twin-aware splitting, the four fixed subject, subject-plus-
food, subject-plus-meal and whole-cohort analysis modes, nested cross-validation,
two primary endpoint tests with Holm correction, and seeds. It also defines the
required schema and generation stage for a future canonical predictor-only
frame and feature contract, the complete Ridge/preprocessing/tuning
specification, analysis-mode estimands and inference policies, deterministic
seed derivation, and minimum valid bootstrap/permutation fractions. The current
real predictor frame and feature contract do not exist.
These are preregistration declarations only; no response value or outcome
summary was read to define them.

The synchronized method-lock schema SHA-256 is
`4726aed7568fa9b8ca2ce46228239887e6e9a3026c9270629b0812ae4d3d78a6`,
and the reviewed gate implementation SHA-256 is
`b2d1cc720bb3f4dd4079e97ddc872b974b36d2c5dc0703ec0d0671fc54f6085f`.
The reviewed cohort-split and benchmark implementation SHA-256 values are
`510d5a584a9fa27067baa60758438ab6dfb7a1f2d3c2afb8284a627520d1df25`
and
`1d64d23937a1c8ee360b3b921242d901e4307b26413ab7349123f4ba7479a782`,
respectively. The schema now requires predictor-frame/feature-contract hashes,
both Task 3 implementation hashes and the canonical benchmark specification
hash. The existing Phase 1 `implementation_source_sha256` constants were not
changed. No placeholder predictor frame, feature contract or run manifest was
created.

## Stages 2C and 2D

Stage 2C is blocked because there is no disjoint held-out scoring beta, no
complete approved production food bundle, no canonical predictor frame or
feature contract, and no approved registry entry. The
empty trusted registry snapshot has SHA-256
`ece380ff9f913d540f286011b97f50110afd9321f4b2a0045018510dae668ba3`.
The manifest writer now requires explicit expected hashes for both this
registry snapshot and the frozen validation config, revalidates all bound
inputs, writes atomically without overwriting an existing manifest, and leaves
no output on failure. After hard-link publication, byte validation, directory
`fsync` and temporary-file cleanup are all inside the rollback boundary; any
failure triggers best-effort destination deletion before the error is raised.
The gate hashes shown above supersede the earlier pre-review implementation
digest because the pre-outcome feature-contract binding is now mandatory.

Stage 2D is blocked because controlled access to Zenodo version DOI
`10.5281/zenodo.17236383` has not been granted and the trusted registry records
the source only for controlled eligibility assessment. The production grant
record remains `controlled_not_granted`. Promotion to `controlled_granted` and
`direct_validation` is structurally reachable only when one trusted record
simultaneously binds a verified local path and SHA-256, version DOI, DUA or
approval evidence identifier, data-dictionary path and hash, participant/meal
key contract, complete endpoint schema/units/windows/summary/derivation
contract, and microbiome-linkage evidence. Missing any element is rejected.
Only after that contract and the complete method-lock instance pass does the
loader inspect the controlled header and bytes.

Required manual action is to request access to the controlled Zenodo record and
obtain: the participant-by-meal files; the endpoint data dictionary with units,
sampling times and derivation rules; participant-to-microbiome linkage keys;
meal/food identifiers and composition provenance; stable file checksums; and
the applicable data-use approval. Endpoint availability must be confirmed
without changing the preregistered endpoint definitions.
