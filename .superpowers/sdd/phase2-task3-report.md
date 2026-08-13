# Phase 2 Task 3 Report: Locked Cohort Split and Baseline Benchmark

**Status:** `implementation_complete_execution_blocked`

**Baseline HEAD:** `70d33ee5d17578b6b16ef63369373142e2382aef`

**Initial Task 3 commit:** `e18db08d69672b95d0208a0328b9743b597466c0`

**Review-fix commit:** `5a93d29be313cdb2f6207f99ed3fe25ad1b91157`
**Audit date:** 2026-08-13

## Scope and outcome boundary

The five Major and three Minor Task 3 review findings are implemented within
the expanded authorized file set. Unrelated dirty worktree files were neither
edited nor reverted.

No real, synthetic-on-disk or aggregate outcome file was opened. No benchmark
result artifact was generated. Test responses are explicitly testing-only,
constructed in memory and used solely to verify software behavior. They are not
empirical validation evidence.

## Review closure

### M1: fixed analysis modes and grouped nested splits

The frozen pre-outcome configuration now registers exactly four modes:
`subject_held_out` (primary), `subject_plus_food_held_out` (secondary),
`subject_plus_meal_held_out` (secondary) and `cohort_held_out` (secondary).
The benchmark generates independent split plans, predictions and metrics for
every mode.

Participant, family and twin links are closed transitively and cannot cross an
outer or inner boundary. Food/meal modes additionally prevent their secondary
unit from crossing either boundary and report crossed rows plus the exclusion
reason. Cohort mode assigns whole cohort components; cohorts linked by a
family/twin component remain together and no cohort rows are partially held
out.

### M2 and M3: canonical feature contract and role allowlist

The benchmark no longer accepts a caller-created feature-contract object. A
future predictor-only stage must generate canonical JSON bytes declaring the
complete exact columns, data types, roles, construction version, source and
block artifact SHA-256 values, allowed overlaps, mapping unit and the fixed
eight-comparator feature-block mapping. `feature_contract_sha256` is bound by
the method-lock schema, generated manifest and real Task 2 validator.

Task 3 reads the feature-contract path once through a no-symlink regular-file
descriptor, hashes the same bytes, compares the digest with the verified
manifest and parses only canonical JSON. Duplicate keys/columns, unknown table
columns, aliases, stale or illegal overlaps, incomplete artifact hashes,
post-outcome roles and any identifier/grouping/mapping column in a model block
fail closed. The mapping unit is available only to the development-fitted
shuffled null and is never a predictor. Metric rows carry the exact comparator
and reference information-opportunity mapping for audit.

### M4: missingness contract

Numeric and categorical preprocessing is fitted inside development folds.
Categorical variables have a separate explicit missing-indicator branch in
addition to development-fitted imputation and one-hot encoding. Outcomes are
split first, retained only as endpoint-specific finite available cases and
never imputed.

The result object includes a missingness source table by endpoint, cohort,
analysis mode, outer fold and split (`train`, `test` or `dropped`), with variable
role, row and participant counts, missing count and missing rate for each
predictor and outcome.

### M5: provenance and preregistered statistics

Every statistical row includes manifest SHA-256, outcome source ID and SHA-256,
predictor-frame SHA-256, feature-contract SHA-256, source-artifact hashes,
analysis mode, software versions, CI method, requested and valid bootstrap and
permutation counts, `n_participants`, `n_person_meals`, endpoint/config hashes
and split seed.

The two primary tests are fixed before outcome access: locked attribute-level
GMNPS versus FCS plus microbiome for RMSE in `subject_held_out`, separately for
`glucose_iAUC_2h` and `tg_6h_rise`, with Holm correction across that two-test
family. Remaining paired tests are labelled secondary or exploratory with an
explicit Holm family and adjusted p-value. Null comparators are never used for
candidate selection.

### Minor findings

- Shuffled mapping uses a deterministic Sattolo derangement. Fewer than two
  development mapping units is explicitly not estimable.
- Metric APIs reject nonfinite inputs instead of silently deleting rows;
  comparators are checked on identical finite person-meal opportunities.
- Tests cover transitive family/twin closure, secondary-unit inner disjointness,
  categorical indicators, alias/overlap/unknown-column rejection and the real
  Task 2 validator through the Task 3 entry.

## Frozen pre-outcome hashes after first review (superseded below)

- `person_meal_validation.yaml`:
  `291946d087998dfac7e43c7edd719a8be8a86f8744700b9bac4718bde3964018`
- `method_lock_manifest.schema.json`:
  `b9894ccf1ac57c2b37603c05566c320d02f6b966d113691ebb105bfd66b59ee5`
- `method_lock_gate.py`:
  `91a0c2721c002a7395f2f1165996e63541b88efc13f98f6dca67c29fadc7fec2`

The Phase 1 `implementation_source_sha256` constants were not changed. No real
feature contract, placeholder manifest or result file was created.

## TDD and verification

RED was observed for each review layer before implementation, including the
missing feature-contract path/hash, trusted-path API, whole-cohort behavior,
four-mode output, strict finite metric contract and updated provenance fields.

Final verification with the repository `.venv` and `PYTHONPATH=code/src`:

- Focused Task 2/3 gate, loader, split and benchmark tests: `99 passed in 37.20s`.
- Related method-lock and locked-scoring tests: `308 passed in 50.89s`.
- Full repository suite: `556 passed, 69 warnings in 54.48s`.
- Python byte compilation and `git diff --check`: passed.

The 69 full-suite warnings arise from pre-existing numerical stress paths in
`anchored.py`, `synthetic_twin.py` and `gmwi2_dual_channel.py`; no warning points
to a Task 3 file. The existing environment does not contain ruff or black, so no
new formatter/linter dependency was installed.

## Execution block and scientific interpretation

Execution remains blocked because no real canonical predictor feature contract
or run-level `results/phase2/method_lock_manifest.json` exists, required held-out
scoring and production food artifacts remain unavailable, and controlled
participant-by-meal outcomes have not been granted. These missing prerequisites
must be generated or approved through the pre-outcome workflow; they cannot be
replaced with placeholders.

Accordingly, this task reports no MAE, RMSE, correlation, calibration estimate,
confidence interval, permutation p-value, model ranking or evidence of GMNPS
superiority. No claim of direct external validity, clinical utility or causal
effect is supported by this implementation-only work.

## Second review hardening

**Status remains:** `implementation_complete_execution_blocked`
**Second review-fix commit:** this report is included in the dedicated fix
commit reported at hand-off.

### Major 1: predictor-value trust

The future predictor-only stage must freeze canonical predictor-frame JSON
bytes as well as the feature contract. The feature contract binds the complete
frame SHA-256 and value-bearing canonical block digests. Both the real Task 2
gate and independent Task 3 entry read immutable regular-file descriptors,
reject symlinks/path replacement, recompute the full-frame and block digests
from the same bytes, and fail before outcome access on any mismatch. The public
benchmark entry accepts no caller DataFrame.

Testing-only fixtures cover valid immutable bytes, value tampering with
unchanged column names, declared digest mismatch, symlink replacement and a
path swap during an open descriptor read.

### Major 2: complete implementation and estimator provenance

The manifest schema now binds live SHA-256 values for `cohort_split.py` and
`person_meal_benchmark.py`, plus the canonical benchmark specification and its
SHA-256. The frozen specification includes the Ridge solver and explicit
estimator parameters, alpha grid, preprocessing, tuning metric/scope/tie-break,
split/fit/resampling seed derivation, bootstrap settings, permutation settings
and minimum valid fractions. Validators recompute the implementation hashes
from installed bytes before outcome access.

Each statistical row records both Task 3 implementation hashes, benchmark
specification ID/hash, seed-derivation ID, actual mode-specific outer/inner
split seeds, outer-fit seed set, actual bootstrap/permutation seeds when run,
software versions and all previously required data/config provenance. Fit and
split audit tables also expose their actual seeds.

### Major 3: estimands and valid inference units

The pre-outcome configuration registers an estimand and inference policy for
every mode. Primary `subject_held_out` inference resamples the transitive
participant/family/twin connected component, exposed by the split module and
carried into predictions. `subject_plus_food_held_out`,
`subject_plus_meal_held_out` and `cohort_held_out` are explicitly descriptive
because valid multiway/cohort cluster inference is not implemented; these modes
produce point estimates but no CI, permutation p-value or adjusted p-value.

### Major 4: structured not-estimable behavior

All four modes are attempted. The result contains an endpoint-by-mode status
table with exact `completed`, `descriptive` or `not_estimable` states, reasons,
estimands, inference policies, seeds and lock provenance. A secondary split,
endpoint or comparator failure returns structured `not_estimable` status while
preserving primary results. A non-estimable primary endpoint is raised only
after all four modes have been attempted. The preregistered primary family also
fails closed if its bootstrap or permutation valid fraction is below the frozen
minimum.

### Minor findings

- Number, string, category and boolean contract types are all validated.
  `food_id` must be the unique mapping-unit role and can never enter a predictor
  block.
- Missingness output now distinguishes `split_all_rows` from the endpoint-
  specific `finite_outcome_analysis_set`; testing-only fixtures contain a
  nonfinite response solely to verify the denominator change and no-imputation
  behavior.
- Bootstrap/permutation rows carry requested, valid, minimum and observed valid
  fractions. Inferential quantities are cleared and marked `not_estimable`
  below threshold. Secondary descriptive modes request no resampling.

## Second review TDD and verification

RED evidence was observed before implementation:

- The original provenance test failed because statistical rows lacked Task 3
  implementation/specification hashes and actual seed fields: `1 failed in
  9.13s`.
- The new connected-component inference-unit test failed at collection because
  `family_twin_component_ids` did not exist: `2 errors in 0.47s`.
- During related verification, two plan-contract tests detected removed
  pre-label DAG/hash phrases. Only the authorized plan text was corrected; the
  two focused regressions then passed.

Final commands used the existing repository `.venv` with
`PYTHONPATH=code/src`:

- Focused Task 3/gate/loader suite (`test_cohort_split.py`,
  `test_person_meal_benchmark.py`, `test_method_lock_gate.py`,
  `test_predict_zoe_loader.py`): `121 passed in 68.72s`.
- Related tracked method-lock/scoring suite: `379 passed, 63 warnings in
  82.47s`.
- Full repository suite: `578 passed, 69 warnings in 82.79s`.
- Python byte compilation and `git diff --check`: passed.

The warnings are existing numerical stress-test warnings in `anchored.py`,
`synthetic_twin.py` and `gmwi2_dual_channel.py`; none points to a Task 3 file.
The existing `.venv` has no ruff executable, so no dependency was installed.

## Current frozen pre-outcome hashes

- `person_meal_validation.yaml`:
  `a8208ebe9c254d163825f9ff8e57f8005aaaed2cd528f680db7a30e1f6662ed7`
- `method_lock_manifest.schema.json`:
  `4726aed7568fa9b8ca2ce46228239887e6e9a3026c9270629b0812ae4d3d78a6`
- `method_lock_gate.py`:
  `e47f1825e30a67008fc0bf5d73884426174996d0769583eb93bf7e2aced3931e`
- `cohort_split.py`:
  `510d5a584a9fa27067baa60758438ab6dfb7a1f2d3c2afb8284a627520d1df25`
- `person_meal_benchmark.py`:
  `294ae374c3507e7591c9c4b73930b844b429dc07d80e4577d3543b70d7726202`

The Phase 1 `implementation_source_sha256` constants remain byte-for-byte
unchanged. No real predictor frame, feature contract, manifest, benchmark
result or placeholder was created. Execution remains blocked by those missing
pre-outcome artifacts, unavailable approved scoring/food inputs and absent
controlled outcome authorization. Therefore this second review also reports no
empirical metric, p-value, ranking or superiority claim.

## Final predictor-universe split correction

**Status remains:** `implementation_complete_execution_blocked`

The final Major finding is closed in implementation. The benchmark now treats
the frozen, validated predictor frame as the complete opportunity universe.
Participant/family/twin connected components and all outer/inner split plans
are constructed from every predictor person-meal key before outcome
availability is considered. Outcomes are then left-joined onto this frozen
universe. An outcome-only key outside the predictor contract fails closed, but
a predictor key with no outcome row remains in every applicable split plan.

Endpoint-specific finite available-case filtering still occurs only after the
split has been fixed. Consequently, missing outcome rows cannot remove an
outcomeless participant that bridges two family/twin links or change component
membership. Such rows do not enter model fitting, tuning, prediction or metric
calculation for the affected endpoint.

Missingness output now explicitly reports `__outcome_row__` availability in
addition to endpoint finite-value missingness. Endpoint-by-mode status records
the frozen predictor opportunity count, present and missing outcome-row counts,
and finite and missing/nonfinite endpoint opportunity counts.

### Final RED/GREEN evidence

- RED: after deleting one complete testing-only outcome row, the primary split
  universe contained 239 rather than all 240 frozen predictor opportunities:
  `1 failed in 8.87s`.
- Focused GREEN for the two new tests: `2 passed in 16.45s`. The first proves a
  predictor key with no outcome row remains in the split universe, is excluded
  only from endpoint analysis and is reported as missing. The second proves an
  outcomeless relative still bridges transitive family/twin components and
  keeps both observed endpoints in the same outer fold.
- Final focused Task 3/gate/loader suite: `123 passed in 80.94s`.
- Final related tracked method-lock/scoring suite: `381 passed, 63 warnings in
  94.95s`.
- Final full repository suite: `580 passed, 69 warnings in 98.07s`.

The final `person_meal_benchmark.py` SHA-256 is
`896470854c1109ac6d5e4100cdcd2d15df442603d6ad03edda1cd4da91a52b83`,
superseding the preceding implementation hash in this report. No real,
on-disk synthetic or aggregate outcome file was opened, and no benchmark result
or placeholder artifact was generated. Execution remains blocked for the same
pre-outcome artifact and controlled-access reasons documented above.
