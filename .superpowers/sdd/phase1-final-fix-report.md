# Phase 1 Final Unified Fix Report

Date: 2026-08-13
Baseline reviewed: `e76e12a0cc10b3e903c4843bca7f24ca5fc88519`
Final re-review starting HEAD: `c9de4290366fa1f2711b154f50771eaf5b8d8f`

## Outcome

All five major and three minor final-review findings were implemented without
changing `attribute-gmnps-v1`, introducing outcome-driven parameters, or
altering the primary attribute-level calibration design. No main-controller
design decision is required. The empty production registry remains
intentionally fail-closed; Phase 2 must populate it only with verified artifact
digests.

The subsequent final re-review's remaining one major and two minor findings are
also closed. Production trust no longer depends on an importable token or a
caller-constructible attestation; attribution now exposes effective weights,
selection/activity and active denominators; and registry schema/version/digest
identifiers are fixed constants.

## RED-to-GREEN record

The new final-review test module was written before implementation. The first
RED run was:

```text
PYTHONPATH=code/src .venv/bin/python -m pytest \
  code/src/tests/test_phase1_final_fixes.py -q

ERROR collecting code/src/tests/test_phase1_final_fixes.py
ImportError: cannot import name 'load_food_attribute_bundle_from_bytes'
1 error in 0.70s
```

After implementation, the complete Task 1-6 plus final-fix suite passed:

```text
222 passed in 19.75s
```

The complete repository test suite passed:

```text
426 passed in 22.30s
```

The focused zero-effect and dairy oracle run passed:

```text
4 passed, 62 deselected in 1.94s
```

For the final re-review delta, the new tests were again written first. The RED
run was:

```text
.venv/bin/pytest -q code/src/tests/test_phase1_final_fixes.py \
  code/src/tests/test_attribute_method_lock.py::test_lock_manifest_schema_requires_complete_frozen_provenance \
  code/src/tests/test_run_attribute_gmnps.py::test_nonzero_dairy_attribution_is_public_and_chunk_invariant

ImportError: cannot import name 'FCS2_FNDDS_REGISTRY_DIGEST_ALGORITHM'
1 error in 0.55s
```

The same focused scope passed after implementation (`10 passed in 2.98s`). The
current GREEN state is `227 passed in 22.16s` for Task 1-6 plus final-review
oracles, `431 passed in 24.76s` for the complete test suite, and `8 passed in
3.91s` for the explicit zero-effect/dairy oracle set.

No main-controller design decision is blocked. Phase 2 population of verified
registry entries remains an expected operational prerequisite, not an open
method-design question.

## Final re-review closure

### Major M1 — live repository trust root at every production boundary

- Added explicit `expected_release_registry_sha256` to production model config;
  absence or mismatch fails closed.
- Every production bundle validation, model fit/validation and score call now
  independently rereads immutable bytes from the configured repository trusted
  registry, validates constant schema/version/`sha256` identifiers, recomputes
  the snapshot digest and compares it with attestation, manifest and model
  expectations.
- Bundle validation relocates exactly one approved live-registry entry and
  rechecks source-byte digests, canonical parsed-content fingerprints,
  food/source linkage, nutrient units and exposure basis.
- Regression tests cover the reviewer capability bypass (imported private token
  plus public constructor and self-authored attestation), alternate registry
  bytes under an empty trust root, registry mutation after bundle gating, and
  registry mutation after model fit.
- The specified threat model is explicit: the pre-label method-lock-bound
  repository snapshot is the trust root. Phase 1 does not claim resistance to
  complete repository/host compromise or provide an external signature.

### Minor m1 — auditable attribution state

- Attribute output now includes `effective_attribute_weight`, `calculated`,
  `active`, baseline/personalized selection flags and baseline/personalized
  active-domain denominators.
- Domain output now includes both denominators and calculated/baseline-selected/
  active attribute counts.
- CLI serialization preserves these columns. A non-zero dairy chunk oracle
  verifies weight 0.5, denominator 2.5 and byte-identical chunked outputs.

### Minor m2 — fixed registry identifiers

- The registry parser requires top-level
  `fcs2-fndds-release-registry-schema-v1`,
  `fcs2-fndds-release-registry-v1` and `digest_algorithm: sha256`, and validates
  the approved-entry digest schema consistently.
- The method-lock schema uses constants for registry version, snapshot schema,
  snapshot version and digest algorithm; these fields are no longer arbitrary
  strings.

## Finding resolution

### Major 1 — dairy effective weight

- Added explicit per-food `effective_attribute_weights` to the food bundle,
  source artifacts, content fingerprints, Task 3 calibration state, Task 4
  decomposition/recomposition, manifests, API and CLI.
- Dairy `unsaturated_to_saturated_fat_ratio` is 0.5; non-dairy is 1.0.
- Added an independent end-to-end oracle proving exact zero-effect identity and
  the non-zero dairy denominator of 2.5 versus the non-dairy denominator of 3.0.
- The fixed residual is not used to conceal the weighting difference.

### Major 2 — ratio gate binding

- Promoted the Task 1 gate evaluator to a shared implementation and reused it
  in Task 2 exposure response and food-bundle validation.
- Bundle validation now enforces gate failure iff the value is the canonical
  `NOT_CALCULATED` sentinel and gate passage iff the baseline point is finite.
- Added API and CLI tests for gate-fail numeric points, gate-pass sentinels,
  exposure tampering and manifest-mask override attempts.
- Added a denominator oracle showing the dairy half-weight changes the domain
  denominator without scaling another ratio's point value.

### Major 3 — registry and method-lock separation

- Removed the release-registry file from fixed implementation source hashes.
- Added run-level registry snapshot SHA-256, canonical release set and complete
  approved-entry provenance.
- Production registry bytes must come from the configured trusted registry
  path; alternate self-authored registry bytes are rejected.
- The committed empty registry still rejects every production run. Phase 2 may
  add verified entries without changing the method version.

### Major 4 — production API provenance

- Added a same-immutable-bytes loader that hashes and parses the same byte
  snapshots, fingerprints canonical parsed content, and mints an attestation
  only after registry matching.
- The attestation retains source bytes and binds source digests, parsed-content
  fingerprints, input-manifest hash, registry-snapshot hash and the approved
  entry.
- Direct DataFrame construction remains available only for development and
  non-production. Production requires the verified loader.
- The CLI now reuses this loader. Tests cover valid bytes, DataFrame tampering,
  byte mismatch/TOCTOU, alternate registry bytes and direct-constructor bypass.

### Major 5 — production lock boundary

- Added `run_attribute_gmnps.py` to the fixed method implementation lock.
- The method-lock schema now requires run-instance
  `method_lock_schema_sha256`, release-registry snapshot provenance and
  `method_lock_gate_implementation_sha256`.
- The schema hash is externally computed and is not a const inside its own
  schema, avoiding self-hash recursion.
- The Phase 2 plan requires both the outcome loader and Task 3 entry to
  independently recompute the config, schema, registry and gate implementation
  hashes before outcomes can be opened.

### Minor findings

- Task 4 now calls the unique Task 1 top-k selector; membership-audit and domain
  score consistency are tested against that selector.
- The specification no longer identifies an obsolete commit as the frozen
  state; method version plus implementation source digests define the lock.
- Manifests and fixed parameters now distinguish `score_centering: none` from
  `beta_centering: development_median`.

## Verification commands

- Final re-review focused tests: 10 passed.
- Task 1-6 plus final oracles: 227 passed.
- Full suite: 431 passed.
- Zero-effect/dairy focused run: 8 passed.
- Python compilation: passed.
- JSON schema parse: passed.
- `git diff --check`: passed.

## Scientific and engineering self-review

- Primary analysis remains attribute-level native-domain recomposition.
- No per-food or population score centering was introduced.
- No response endpoint, disease label, target drift or target rank shift enters
  fitting or parameter selection.
- Carbohydrate remains sensitivity-only; legacy remains a comparator.
- Zero-effect identity remains tested at absolute tolerance `1e-8`.
- No experimental result, effect size, p-value, sample size or citation was
  invented.

## Decision status

No design blocker requires main-controller intervention. The only remaining
production prerequisite is the already-planned Phase 2 verification and
population of real release-registry entries; this is an operational gate, not
an unresolved Phase 1 design choice.
