# Phase 2 Task 2 Report: Predictor-First Reconstruction and Outcome Gate

**Status:** `IMPLEMENTATION_COMPLETE_EXECUTION_BLOCKED` — the reviewed
implementation and tests are complete; real Stages 2B–2D remain fail-closed
because required inputs are unavailable or incomplete.

**Baseline HEAD:** `7787d59f3eb5634c73e20b324b1d4277fadc31dd`  
**Initial Task commit:** `94971f2582eb6e98689b2cc3f9223b0f9bbac3ce`  
**Review-fix commit:** `bcd5496f46043b3aa8c54deb5f4e4ade09cda56d`  
**Final TOCTOU-fix commit:** `70d33ee5d17578b6b16ef63369373142e2382aef`  
**Audit date:** 2026-08-13

## Scope and dirty-worktree handling

The initial commit contained seven Task 2-authorized files. The review fix
modifies only the ten authorized Task 2/contract-sync files:

- `code/src/scripts/fetch_official_predict_zoe.py`
- `code/src/gmnps/data_sources/predict_zoe_loader.py`
- `code/src/configs/person_meal_validation.yaml`
- `docs/data/predict_zoe_reconstruction_log.md`
- `code/src/tests/test_predict_zoe_loader.py`
- `code/src/tests/test_method_lock_gate.py`
- `code/src/gmnps/validation/method_lock_gate.py`
- `code/src/gmnps/data_sources/predict_zoe_registry.py`
- `code/src/tests/test_predict_zoe_registry.py`
- `docs/data/data_availability_audit.md`

No `__init__.py`, trusted release registry, generated result, unrelated dirty
file, or other-worker file was staged, reverted or committed. Generated caches
remain under ignored `data/`; no downloaded data were committed.

The final TOCTOU fix commit contains only the authorized loader, loader tests
and reconstruction log. The ignored external Task 2 report was updated in
place and is not part of the repository commit.

## TDD and verification

RED was established before implementation:

- `test_predict_zoe_loader.py` failed collection because the loader did not
  exist.
- Task 2 gate tests failed because no explicit
  `expected_release_registry_sha256` contract or atomic manifest writer
  existed.
- A later tamper RED demonstrated that caller-supplied metadata could
  reclassify a trusted synthetic source unless the loader consulted the Task 1
  registry.
- The review-fix RED failed collection on the absent trusted artifact/grant
  contracts; the new atomic rollback test also failed against the pre-fix
  post-link cleanup behavior.
- The final TOCTOU RED failed because predictor loading had no immutable
  snapshot boundary: header inspection, SHA-256 calculation and parsing each
  reopened the trusted path. The replacement matrix covers CSV, TSV, JSON,
  Parquet and RData at both post-hash and post-schema phases, using malicious
  regular-file and symlink replacements (20 adversarial cases).

Final verification:

- Focused Task 2 tests: `95 passed in 0.95s`.
- Related provenance/method-lock/scoring tests: `216 passed in 2.57s`.
- Full suite: `526 passed in 23.57s`.
- Python byte compilation of the final loader and loader tests: passed.
- `git diff --check`: passed.

## Review remediation

### Final Major — single-snapshot predictor TOCTOU closure

Every trusted predictor load now retains the pre-read registry lookup and
exact path-policy validation, then opens one non-symlink regular-file
descriptor and reads one immutable byte snapshot. Descriptor identity and
size are checked before reading; descriptor identity, size, mtime and ctime
must remain stable through the read. The repository-pinned SHA-256, header or
schema validation, and final parse all consume that same byte string. CSV,
TSV, JSON and Parquet use in-memory buffers; RData uses
`rdata.parser.parse_data(raw, extension=".rda")` and never reopens the source
path or calls `parse_file`. Adversarial replacements after snapshot hashing or
after schema validation therefore return only the original snapshot; none of
the 20 format/phase/replacement combinations can return replacement content.

### M1 — repository-rooted predictor trust

The public loader now accepts only `source_id`. A committed artifact registry
provides canonical stable source, exact repository-relative cache path,
locally audited SHA-256, size, content class, analytical role, schema, allowed
columns, unique key, record count and ID-set digest. Tabular headers are checked
before complete parsing. Unknown sources, label columns, symlink/path swaps,
size/SHA mismatches and ID/schema drift fail before a table is returned. The
mutable acquisition manifest is never a digest authority. A first download
without an upstream checksum is reported as locally audited but unpinned until
its digest is reviewed and committed. Reviewed registry implementation
SHA-256: `d7509d6d3349fb4ae144ae771b1ffdb8e88775faa2c510ddd9d678d045ae7837`.

### M2 — reachable but evidence-complete controlled grant

The registry permits `controlled_granted` plus `direct_validation`, while the
production record remains `controlled_not_granted`. A grant is legal only when
it binds a verified local path/SHA, version DOI, DUA/approval identifier,
data-dictionary path/SHA, participant/meal key contract, complete endpoint
contract and microbiome-linkage evidence. Missing any field fails closed. A
temporary trusted source, dictionary and outcome fixture passes the complete
real method-lock generator/validator and loader path without mocking the gate;
this is contract reachability testing, not empirical validation.

### M3 — source-independent frozen outcome contract

The frozen config now defines the `participant_id + meal_id` unique key,
requires both primary endpoints, defines the secondary availability policy and
freezes role, unit, window, summary and derivation for every endpoint. Callers
cannot supply columns, keys, units or endpoint subsets. The loader derives the
source projection from frozen config plus trusted grant. Row-ID substitution,
missing primary endpoints and any semantic change fail closed. The production
header remains unknown and blocked rather than guessed.

### M4 — atomic publication rollback

After `os.link`, destination byte validation, directory `fsync` and temporary
cleanup are inside the rollback boundary. Any exception triggers best-effort
destination deletion before propagation. Fault injection confirms that a
temporary-file unlink failure after publication leaves no destination. Existing
destinations remain protected by no-overwrite hard-link semantics.

## Stage 2A — predictor-only acquisition

### Outputs

The ignored local acquisition manifest records six verified predictor
resources and one blocked metadata source. Raw FASTQ files were not downloaded.

| Source | SHA-256 | Bytes | Verified predictor unit |
| --- | --- | ---: | --- |
| ExperimentHub `EH5458` / dispatch `5501` | `89c635061b357583351ca33f520a72d0efced4a2963e73182e529228c8395c54` | 755,545 | 1,098 unique profiles; 1,019 taxonomic rows |
| ENA `PRJEB39223` metadata | `d630be36c1abd3a71b6aa295dc56e1557557f965f6c8fdaa673759b227b28e36` | 152,736 | 2,196 unique run/sample records: 1,098 WGS + 1,098 amplicon |
| ENA `PRJEB75460` metadata | `2ce6d90cd5d380581737a9dc193460b3f5df7f836f443845bba793ecb67de1a9` | 68,364 | 975 unique WGS run/sample records |
| ENA `PRJEB75462` metadata | `ca6d7d5a1b7fdcd069a309b5bc8dd076a53ea8bc4016ec98b1b755658fd20f08` | 825,904 | 11,797 unique WGS run/sample records |
| ENA `PRJEB75463` metadata | `ce028e6072e610ad7575ba7e494347892cd66e33a5eaec92dea7177bcb413f56` | 592,944 | 8,469 unique WGS run/sample records |
| ENA `PRJEB75464` metadata | `f81d06da5f9127864967f99bb13a2b7f6eedddc8c56282f88c5b04985cd4bb94` | 864,824 | 12,353 unique WGS run/sample records |

Stable sources were the official ExperimentHub fetch endpoint and ENA Portal
API accession queries. Zenodo record metadata at
`https://zenodo.org/api/records/15308000` was blocked before HTTP negotiation:
`[Errno 61] Connection refused`. No mirror was substituted.

All six successful predictor artifacts were reloaded through the new committed
trust root: EH5458 produced a `1,019 × 1,098` abundance matrix, and the five ENA
tables produced `2,196`, `975`, `11,797`, `8,469` and `12,353` seven-column
records with their pinned digests. No outcome or label table was used.

### Boundary concern

No postprandial response table, controlled file, local synthetic response table
or Nature/ZOE aggregate-outcome workbook was opened. However, two discovery
commands crossed the user's stricter literal label boundary: a broad source
search printed rows from curatedMetagenomicData sample metadata, and the Phase
1 beta provenance manifest exposed aggregate health-label diagnostics. These
values were not retained or analysed. Searches were subsequently restricted to
schemas, IDs and allowlisted predictor files. This prevents a clean
no-concern status and is documented in the committed reconstruction log.

## Stage 2B — scoring beta, food bundle and frozen configuration

### Held-out scoring beta: blocked

The official ExperimentHub PREDICT-1 profile contains 1,098 unique sample IDs.
All 1,098 occur in the existing 15,492-ID Phase 1 beta output; the public-only
ID count is zero. It therefore cannot satisfy development/scoring participant
disjointness. PREDICT-2/3 profiles could provide disjoint predictors, but the
Zenodo profile resource was inaccessible. ENA metadata alone is insufficient,
and large FASTQ reconstruction was intentionally not started.

The exact-case, unique, Unicode-sorted UTF-8/LF/no-trailing-LF ID contract
produced: Phase 1 development `n = 15,492`, SHA-256
`e93e35038391cf56b4781e138841a93bd7f84c5d3ceb532d02aaf2f7dade6082`;
EH5458 `n = 1,098`, SHA-256
`db9a7fabfa390e5d1b91b32c2b9569aa3d1d5e57f9708297bfa91c1e1d92974c`;
intersection `n = 1,098` with the same digest. Source artifact hashes and the
reproducible helper are recorded in the committed reconstruction log/registry.

No canonical held-out scoring-beta artifact, canonical development-beta
artifact or normalization-state artifact was minted for Task 2.

### Frozen configuration: completed

`person_meal_validation.yaml` freezes the source-independent key, complete
endpoint semantics and availability policy, missingness,
participant/family/twin-aware splits, nested folds and deterministic seeds.
Exact SHA-256:

`12bbe284674cac0cf96344112a5e47a279d87e83fe5727e2841635ec0dc70f6b`

The endpoint definitions are preregistration declarations only; no outcome was
used to choose them.

### FNDDS acquisition: partial and blocked for production

Official index:
`https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fndds-download-databases/`

The official Food Compass 2.0 supplement was verified byte-for-byte against the
Springer Nature URL: 3,295,889 bytes, SHA-256
`26af914baa2662fe65e17b607c5a629baa01a043016d21cdab7bd3c03fc3eab4`.

Verified historical release artifacts:

| Artifact | SHA-256 |
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

Downloads were attempted in two GET rounds during 2026-08-13 17:33–17:52
CST. Repeated transfers were stopped on user instruction. Missing artifacts:

1. 2015–2016 Ingredient Nutrient Values — official URL recorded in the
   reconstruction log; HTTP status was not persisted, partial body transfer,
   two attempts.
2. 2017–2018 Ingredient Nutrient Values — official URL recorded; HTTP status
   was not persisted, incomplete body transfer, two attempts.
3. 2017–2018 FNDDS Nutrient Values — official URL recorded; HEAD returned HTTP
   200 at 17:29:21 CST, incomplete GET transfer, two attempts.

Seven incomplete `.part` files created by this task were deleted. They were not
accepted artifacts.

Production food-bundle construction is blocked because the two recent release
sets are incomplete; the 9,273 official foods were not freshly reconstructed
and audited into the required 9,234 aligned-food universe; and the required
FPED, flavonoid, additive, processing, baseline-point and food-linkage fields
are not all supplied by the downloaded FNDDS files. Therefore:

- no production `official_fcs`, `food_metadata`,
  `baseline_attribute_points`, `food_exposures`, or
  `effective_attribute_weights` bytes were created;
- the Phase 1 same-immutable-bytes loader was not falsely invoked on an
  incomplete bundle; and
- no approved registry entry was added.

## Stage 2C — real method-lock manifest

**Blocked; no manifest generated.**

The current trusted release registry remains empty. Its exact SHA-256 is
`ece380ff9f913d540f286011b97f50110afd9321f4b2a0045018510dae668ba3`.
The external method-lock schema SHA-256 is
`6ca18393de8fb99b60aa970feebb9fd7b3323c0dd51c1156376eb38569f5a61b`.
The reviewed gate implementation SHA-256 is
`e73c4601bb102215b20e0e9435a56615714bb0389ee1275f4edc89c3065c62a3`.

The gate now requires an explicit expected registry snapshot SHA-256 for both
generation and validation. The manifest writer additionally requires the
explicit frozen config SHA-256, revalidates before writing, uses an atomic
no-overwrite operation, and rolls back the destination after any post-link
validation, `fsync` or cleanup failure.

`results/phase2/method_lock_manifest.json` does not exist and should not be
committed because no real instance was eligible to generate.

## Stage 2D — controlled outcome access

**Blocked before outcome read.** Controlled Zenodo version DOI
`10.5281/zenodo.17236383` remains `controlled_not_granted` and is not approved
for `direct_validation` in the trusted registry.

The loader independently enforces this order:

1. source-ID lookup and complete trusted grant validation;
2. Task 1 trusted-registry source/role/access/path/SHA/evidence match;
3. current method-lock revalidation, including config/schema/registry/gate
   hashes;
4. frozen key/endpoint/availability contract derivation;
5. controlled header validation; and only then
6. outcome-byte digest, parse, numeric and unique-key validation.

Synthetic, aggregate, caller-reclassified and unauthorized controlled sources
are rejected before manifest or outcome bytes are opened.

## Manual access and data required

The user must request access to Zenodo `10.5281/zenodo.17236383` and obtain:

- participant-by-meal endpoint files;
- data dictionary and derivation rules for glucose 2-h iAUC, triglyceride 6-h
  rise and, if present, C-peptide 2-h iAUC;
- endpoint units and sampling timestamps;
- participant-to-microbiome linkage keys;
- meal/food identifiers and composition provenance;
- stable file checksums and version identifiers; and
- applicable data-use approval/DUA.

After access, the trusted registry must be updated from verified evidence—not
caller declarations—to record granted access, a verified local path and SHA,
and direct-validation eligibility. Endpoint definitions must not be revised
after outcomes are inspected.

## Integrated scientific assessment

- **Reviewer:** direct validity is not established; manuscript claims must
  remain at computational feasibility/biological consistency.
- **Researcher:** PREDICT-1 overlap invalidates held-out scoring; PREDICT-2/3 or
  another genuinely disjoint profile cohort is required.
- **Engineer:** fail-closed staged access, hash rebinding, tamper refusal and
  no-placeholder manifest behavior are implemented and tested.
- **Writer:** no result should describe synthetic, aggregate or inaccessible
  controlled data as observed participant-by-meal validation.

**Superseded claim note (2026-08-14):** The “biological consistency” portion of
the Reviewer statement above is superseded. No Task 4 production evidence has
been executed and bound to the current registry, so a positive biological
consistency claim is not allowed by the current computational policy.

## Frozen-configuration hash supersession note (2026-08-14)

The `12bbe284674cac0cf96344112a5e47a279d87e83fe5727e2841635ec0dc70f6b`
value recorded in the historical “Frozen configuration” section is retained as a
historical hash. The current byte-level SHA-256 of
`code/src/configs/person_meal_validation.yaml` is
`a8208ebe9c254d163825f9ff8e57f8005aaaed2cd528f680db7a30e1f6662ed7`,
which supersedes the historical value for all current Task 3/5 bindings.
