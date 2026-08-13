# Phase 1 Final Unified Fix Report

Date: 2026-08-13
Baseline reviewed: `e76e12a0cc10b3e903c4843bca7f24ca5fc88519`

## Outcome

All five major and three minor final-review findings were implemented without
changing `attribute-gmnps-v1`, introducing outcome-driven parameters, or
altering the primary attribute-level calibration design. No main-controller
design decision is required. The empty production registry remains
intentionally fail-closed; Phase 2 must populate it only with verified artifact
digests.

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

- Task 1-6 plus final oracles: 222 passed.
- Full suite: 426 passed.
- Zero-effect/dairy focused run: 4 passed.
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
