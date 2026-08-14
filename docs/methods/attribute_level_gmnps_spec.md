# Attribute-level GMNPS method lock

## Status and scope

This document freezes Phase 1 method version `attribute-gmnps-v1`. The primary
method is `attribute_recomposition`, the primary mapping is
`expert_reviewed_attribute_mapping_v1`, and the recomposition implementation is
`native_domain_fixed_residual_v1`. The frozen state is defined by this method
version and the implementation source digests in the reproducibility audit,
not by a historical repository commit. It does not authorize fitting, mapping
revision, or parameter selection on validation outcomes.

Food Compass 2.0 (FCS2) contains 54 conceptual attributes represented by 56 operational rows
because fruits and non-starchy vegetables each have separate
dried and non-dried rows. Of those rows, 54 are active in the reported USDA
analysis. The exact source values and all mapping registries are exported in
`docs/methods/attribute_mapping_table.csv`.

## Frozen versions and parameters

| Item | Primary value | Named sensitivity values |
| --- | --- | --- |
| Scoring version | `attribute-gmnps-v1` | none |
| Primary method | `attribute_recomposition` | `legacy_final_score_offset` (restricted comparator) |
| Mapping | `expert_reviewed_attribute_mapping_v1` | `fiber_all_ratio`, `fiber_all_absolute`, `potassium_all_ratio`, `potassium_all_absolute`, `carbohydrate_proxy` |
| Beta normalization | `median_mad_iqr_sd_v1` | none |
| Beta normalization temperature (`LOCKED_BETA_TEMPERATURE`) | 2.0 | none |
| Attribute response temperature (`LOCKED_ATTRIBUTE_RESPONSE_TEMPERATURE`) | 2.0 | none |
| Attribute point fraction | 0.20 (`primary`) | 0.10 (`low`), 0.30 (`high`) |
| Final FCS2-relative cap | 12 points (`primary`) | 8 (`low`), 15 (`high`) |
| Mask | `expert_revised_v4_dual_channel` | none in this method lock |
| Nitrite threshold rule | `footnote_50` | `table_25` |
| FCS unscaled bounds | -12.1, 35.0 | none |
| FCS score bounds | 1.0, 100.0 | none |

The primary recomputed domains are `nutrient_ratios`, `vitamins`, `minerals`,
`specific_lipids`, `fiber_and_protein`, and `phytochemicals`. Other domains are
held in the food-specific fixed residual.

## Published FCS2 attribute rules

All composition exposures are per 100 kcal unless a rule explicitly uses a
percentage, ratio, binary presence indicator, or NOVA class. For an ordinary
linear rule with exposure `x`, low/high targets `(t0, t1)` and low/high points
`(p0, p1)`, the implementation is:

```text
p(x) = clip(p0 + ((x - t0) / (t1 - t0)) * (p1 - p0), min(p0,p1), max(p0,p1)).
```

Negative linear exposures are invalid. Ratio rules apply the same interpolation
to `log(x)` and require `x > 0`. They are `NOT_CALCULATED`, and are removed from
the domain denominator, when their published exposure gates fail:

- unsaturated:saturated fat requires at least 10% energy from fat;
- fiber:carbohydrate requires at least 10% energy from carbohydrate;
- potassium:sodium requires at least 10 mg each of potassium and sodium per
  100 kcal.

For dairy, the effective weight of the unsaturated:saturated fat ratio is 0.5.
Added sugar uses the discrete cut points 2.5, 5, 10, 15, 20, 30, 40, 50, and
60 percent energy, producing 0 at zero exposure and -1 through -10 thereafter.
The primary nitrite rule `footnote_50` interpolates from -10 at 50% calories
from processed meat to 0 at zero. The contradictory Table S10 value is retained
as the `table_25` sensitivity rule and is never silently substituted.

Binary additive attributes score their low point when present and high point
when absent. NOVA classes interpolate through `(1,10)`, `(2,7.5)`, `(3,5)`, and
`(4,-10)`. Other fermented products force fermentation to its high point. The
published attribute weights are retained, including the 0.5 weights for
fermentation, frying, cholesterol, medium-chain fatty acids,
alpha-linolenic acid, and total protein.

For a domain with calculated attribute points `p_a` and effective weights
`v_a`, aggregation is frozen as follows:

```text
weighted_mean(A) = sum(a in A, p_a * v_a) / sum(a in A, v_a)
```

- `food_ingredients` is the weighted sum, not a mean.
- `vitamins` and `minerals` use the five attributes with largest absolute
  points; `specific_lipids` uses the three with largest absolute points.
- Top-k ties are resolved by immutable registry order.
- Other domains use the weighted mean of calculated active attributes.
- Final `specific_lipids` and `phytochemicals` domain contributions are each
  multiplied by 0.5.

Effective attribute weights are explicit per-food state. For non-dairy foods,
the three calculated nutrient-ratio weights are 1.0 and their denominator is
3.0. For dairy foods, only `unsaturated_to_saturated_fat_ratio` has effective
weight 0.5; when all three ratios are calculated, the denominator is therefore
2.5. These weights are carried through the food bundle, Task 3 calibration
fingerprint, baseline decomposition, personalized recomposition, API and CLI.
They cannot be absorbed into the fixed residual because that would preserve
zero-effect identity while producing an incorrect non-zero ratio delta.

Primary attribute attribution exposes `effective_attribute_weight`,
`calculated`, `active`, `baseline_selected`, `personalized_selected`, and the
baseline and personalized active-domain denominators. Domain attribution
exposes the same denominators together with calculated and selected attribute
counts. These fields are retained by CLI serialization and are invariant to
individual and food chunking, making the dairy weight 0.5 and denominator 2.5
directly auditable in non-zero runs.

Bundle validation recomputes every ratio gate from the same per-100-kcal
exposure row used for calibration. Gate failure is valid only with the
canonical `NOT_CALCULATED` sentinel, and gate passage is valid only with a
finite baseline point. A serialization token or manifest mask records the
sentinel but cannot determine biological applicability.

The unscaled-to-FCS transform clips `U` to `[-12.1,35.0]` and maps it linearly
to `[1,100]`:

```text
FCS(U) = 1 + (clip(U,-12.1,35.0) + 12.1) * 99 / 47.1.
```

The inverse used for an official FCS2 anchor is:

```text
fcs_to_unscaled(F) = -12.1 + (F - 1) * 47.1 / 99.
```

## Unavailable and fixed baseline attributes

`iodine` and `trans_fat_percent_calories` are inactive because Table S9 reports
that they were unavailable in FNDDS, FPED, and the flavonoid database. They are
not scored, imputed, or treated as zero. The mapping CSV preserves their source
values and unavailability reason.

An active attribute can still require data not present in a particular input
bundle. When local 18:3 is not verified as ALA, `alpha_linolenic_acid` is a
fixed baseline attribute within a recomputed domain. It retains its supplied
baseline point and still competes in the dynamically selected
`specific_lipids` top three after other lipid points move. When the separate
flavonoid source is absent, `total_flavonoids` is likewise a fixed baseline
attribute within a recomputed domain: its fixed point remains in the
`phytochemicals` weighted mean while mapped `total_carotenoids` may move. Neither
attribute is assigned to `Q_j`.

Food-ingredient attributes require FPED equivalents; additives require
ingredient/additive records; processing attributes require processing or recipe
records. Total sugar cannot stand in for added sugar, and rounded NOVA cannot
stand in for the original energy-weighted mixed-dish value. Missing required
data are therefore unavailable or fixed-baseline inputs, never zero-filled
substitutes.

## Beta normalization and attribute response

Let `b_in` be raw beta for individual `i` and nutrient `n`, fitted only on the
development cohort. For each nutrient, define the development median `m_n` and
select the first positive scale in this order:

```text
T_beta = LOCKED_BETA_TEMPERATURE = 2.0
s_n = 1.4826 * median(|b_dn - m_n|)
      else (q75_n - q25_n) / 1.349
      else population standard deviation (ddof=0)
      else 1.0.
z_in = tanh(((b_in - m_n) / s_n) / T_beta).
```

The nutrient order, medians, scales, selected scale methods, fit count, fit-ID
SHA-256, method version, and beta normalization temperature are immutable
normalization state. The fit-ID hash is SHA-256 of sorted development IDs joined
by a newline.

For food `j`, author-specified mapping row `n -> a`, allocation `w_na`, and normalized
per-100-kcal exposure `e_jna`, the uncapped attribute response is:

```text
r_ija = sum(n mapped to a, z_in * e_jna * w_na).
```

Effect allocations for one nutrient sum exactly to 1. A nutrient mapped to
multiple native attributes is divided by allocation weights, never assigned
full weight to every representation. Zero component totals produce zero effect
and an explicit `zero_total_component` diagnostic.

Direct exposures divide by the largest nonzero absolute published target.
Composite components each divide by the target after a verified zero-total
check. Folate-food and retinol component fractions algebraically preserve their
component exposure while requiring the DFE or RAE total. Ratio-side policies are:

- fiber contributes `fiber / 9.5` when the carbohydrate gate passes;
- potassium contributes `potassium / 1175` when both ratio sides pass;
- saturated fat contributes the negative saturated fraction of saturated plus
  mono- and polyunsaturated fat when the fat gate passes;
- the `carbohydrate_proxy` sensitivity contributes negative carbohydrate
  energy fraction clipped to `[0,1]` in magnitude.

The complete primary roles (`effect`, `applicability_only`, `explanatory_only`,
`sensitivity_proxy_only`, and `excluded`) and component policies are in the
mapping CSV.

## Native point calibration

For attribute `a`, let `[l_a,h_a]` be its published point range and let
`c=0.20` in the primary mode. The point movement and calibrated point are:

```text
T_response = LOCKED_ATTRIBUTE_RESPONSE_TEMPERATURE = 2.0
lambda_a = c * (h_a - l_a)
raw_delta_ija = lambda_a * tanh(r_ija / T_response)
p_ija = clip(p_ja + raw_delta_ija, l_a, h_a).
```

`T_response` controls only the bounded conversion from attribute response to
native point movement. It is an independent runtime constant from `T_beta`,
which controls only raw-beta normalization. Both are currently 2.0, but equality
of their numeric values does not merge their scientific meaning or versioning.

The low/high sensitivity values for `c` are 0.10 and 0.30. A canonical
`NOT_CALCULATED` baseline remains `NOT_CALCULATED`. Attributes outside reviewed
mapping targets have zero response and zero point movement.

## Fixed-residual native recomposition

For food `j`, invert the supplied official score and aggregate the selected
baseline attribute points:

```text
U0_j = fcs_to_unscaled(FCS2_j)
L0_j = sum(d in recomputed domains, D0_jd)
Q_j = U0_j - L0_j
```

`Q_j` is fixed for that food and contains the official-anchor contribution not
represented by the selected recomputed domains, including whole domains held
outside that set and any anchor discrepancy. It does not contain fixed baseline
attributes inside a recomputed domain: those points are already included in
`L0_j` and remain present when that domain is recomputed. If the official S5
score is a rounded integer, `U0_j` is the score-implied latent anchor; it is not
a claim about an unpublished raw FCS2 domain sum.

After calibration, every selected domain is reaggregated from personalized
native attribute points. Top-five/top-three membership is recalculated after
point movement. Let `D_ijd` be those contributions:

```text
U_ij = Q_j + sum(d in recomputed domains, D_ijd)
F_native_ij = unscaled_to_fcs(U_ij)
delta_uncapped_ij = F_native_ij - FCS2_j
delta_ij = clip(delta_uncapped_ij, -12, 12)
GMNPS_ij = clip(FCS2_j + delta_ij, 1, 100).
```

There is no population centering and no candidate selection. At zero effect,
`z_in=0`, hence every `r_ija=0`, every calibrated point equals baseline, every
recomputed domain equals baseline, and `U_ij=U0_j`. Therefore the exact locked
identity is:

```text
GMNPS_ij = FCS2_j
```

The implementation tests this identity to absolute tolerance `1e-8`.

The manifest states the two distinct transformations explicitly as
`score_centering: none` and `beta_centering: development_median`. The latter
describes frozen nutrient-wise beta normalization and counterfactual
replacement; it must not be misreported as uncentered beta.

MAC and LIPID output deltas are frozen counterfactuals. MAC delta scores with
LIPID effect nutrients replaced by their development medians; LIPID delta does
the converse. The channel interaction is total delta minus those two deltas.
Drivers are ranked by native attribute point delta, not raw nutrient products.

## Development-only fit boundary and validation embargo

The development-only fit boundary permits only `fit_beta_normalization` to
estimate medians and fallback scales from the declared beta fit cohort. Mapping
roles, attribute rules, allocation weights, temperatures, point fractions,
domain set, and final caps are fixed before validation. Held-out and validation
beta are transformed with the frozen state and exact nutrient order; they are
never appended to or used to refit that state.

The method-lock manifest records the beta cohort ID, fit count, calibration fit
IDs hash, `development_beta_sha256`, and `normalization_state_fingerprint` as
the frozen fit state. Each scoring run separately records
`scoring_beta_sha256` and `person_meal_validation_config_sha256` as run-instance
provenance. The latter is computed from the canonical bytes of the frozen
`person_meal_validation.yaml`; the schema defines its SHA-256 shape but does not
invent a value. Its `validation_embargo` flag must be `true`: validation data,
labels, responses, and derived summaries cannot be used to choose mappings,
parameters, sensitivity modes, or revisions to this method lock. Validation is
reserved for a later phase after this lock is signed.

## FNDDS release and source boundary

Production composition must use the exact FCS2-aligned canonical sequence
`FNDDS 2001-2002` through `FNDDS 2017-2018`, as declared by registry
`fcs2-fndds-release-registry-v1`. The release registry is run-level data
provenance, not an implementation source whose current empty bytes are frozen
as the method. Each run binds the real registry snapshot SHA-256, canonical
release set and one complete approved entry. The entry SHA-256 is computed from
canonical JSON of that complete entry with sorted keys and compact separators.
A registry snapshot is accepted only when its top-level schema version is
`fcs2-fndds-release-registry-schema-v1`, its registry version is
`fcs2-fndds-release-registry-v1`, and its digest algorithm is `sha256`; these
values are constants in the method-lock schema rather than caller-defined
strings.

A production bundle must match
that entry for official FCS, metadata, baseline points, exposures, per-food
effective attribute weights, food linkage, nutrient units, and exposure basis.
The committed approved-artifact list is currently empty, so no production
bundle is currently authorized. Phase 2 can add verified digests without
changing or disguising the method version.

Production construction uses the verified same-immutable-bytes loader. It
hashes each source byte snapshot, parses those exact bytes, fingerprints the
canonical parsed tables, and matches the trusted registry before minting an
attestation. The attestation retains immutable source bytes and binds both byte
digests and parsed-content fingerprints. Direct DataFrame construction is
restricted to development/non-production; caller-supplied digest strings cannot
authorize production. A production configuration must explicitly carry
`expected_release_registry_sha256` from the pre-label/method-lock gate. Every
production bundle validation, model fit/validation, and score call independently
reads immutable bytes from the configured repository trusted-registry path,
validates the fixed registry schema/version/digest algorithm, recomputes the
snapshot SHA-256, and compares it with the bundle attestation, input manifest,
and model expectation. Bundle validation then relocates exactly one approved
entry in that live snapshot and rechecks source-byte digests, canonical parsed
content fingerprints, food/source linkage, nutrient units, and exposure basis.
An imported private loader token, a caller-constructed attestation, or alternate
self-consistent registry bytes therefore cannot authorize production. The CLI
uses the same path, and registry modification after a gate or model fit is
detected by the next validation boundary.

The current trust root is the repository trusted-registry snapshot bound by the
pre-label method lock. This threat model rejects untrusted callers and
post-gate file changes while the repository and host trust root remain intact;
it does not claim resistance to complete repository or host compromise. An
external signature is outside the Phase 1 threat model.

`FNDDS 2021-2023` is recognized only as a non-production development smoke
input and requires `development_smoke_test=true`. It cannot support a production
result or replace release-aligned reconstruction.

## Sensitivity variants

Sensitivity analyses must be named and labelled `sensitivity`:

- `fiber_all_ratio` allocates all fiber effect to the ratio; `fiber_all_absolute`
  allocates it all to total fiber.
- `potassium_all_ratio` allocates all potassium effect to the ratio;
  `potassium_all_absolute` allocates it all to potassium.
- `carbohydrate_proxy` activates carbohydrate only as the reviewed ratio-side
  proxy; primary carbohydrate remains `sensitivity_proxy_only` with zero
  allocation.
- Attribute fraction modes `low` and `high` use 0.10 and 0.30.
- Final cap modes `low` and `high` use 8 and 15 FCS points.
- `table_25` is the nitrite threshold sensitivity; `footnote_50` is primary.

No sensitivity mode can be relabelled primary by caller choice.

## Migration from the legacy offset

`legacy_final_score_offset` computes a raw nutrient response, centers and scales
it, and then adds an offset to final FCS2. It is a sensitivity comparator only
because the perturbation occurs after published attribute scoring and domain
aggregation. It cannot express native attribute bounds, ratio applicability,
dynamic top-k membership, fixed-domain invariance, or attribute-point drivers,
and its centering makes one individual's score depend on the comparison
population.

The legacy route is rejected for production and is available only with an
explicit sensitivity role in development smoke mode. Existing legacy exports
remain for migration comparisons; primary scientific claims and attribution
must use `attribute_recomposition`.

## Reproducibility audit

SHA-256 values below bind this document to the approved implementation read for
Task 6. Run data hashes remain run-specific and are required by
`method_lock_manifest.schema.json`.

| Approved source | SHA-256 |
| --- | --- |
| `code/src/gmnps/scoring/fcs2_attribute_rules.py` | `8758d245d2e4d7d23a71829893ddfcc7e4b12b120bf8cc2e1c20d4e1c6efba14` |
| `code/src/gmnps/scoring/fcs2_attribute_mapping.py` | `d885be53736d020c6797b8b8f6d470c185ca0b1e13bf7b07752b32ec0de200d1` |
| `code/src/gmnps/scoring/attribute_calibration.py` | `2dcc803e46f07f2a1c7ca11e31bced942df84bd66496afabd61ec8b86e8585c6` |
| `code/src/gmnps/scoring/attribute_recomposition.py` | `806306099b1d398fce1777c3b5714e1465fcc5949c4cf9a8f6fc2d1b1bf3119e` |
| `code/src/gmnps/scoring/attribute_gmnps.py` | `b1ac1465f69f968232d1e9b07854a96a29675dbd6516c60a359df7868d83a728` |
| `code/src/configs/attribute_gmnps.yaml` | `37a46a0cb0d24da374698dd4f900946efb8f581f89d843c27e35ce54fb39d2f2` |
| `code/src/scripts/run_attribute_gmnps.py` | `3771af05eb598d59658e9edd3d9838a1260c9397df4b4af659c2d2404ed601ad` |

Every run must additionally record method and mapping versions, all input and
linkage SHA-256 values, FNDDS releases, the real registry snapshot SHA-256,
canonical release set and approved entry, beta fit cohort,
fit-ID hash, development beta hash, normalization-state fingerprint, scoring
beta hash, fixed parameters, and the validation embargo flag.

## Phase 2 handoff gate

Task 6 defines the contract but does not invent an instance without real run
inputs. Phase 2 Task 1 implements the provenance inventory and the manifest
generator/validator contract; it does not require a real instance before scoring
inputs exist. Phase 2 Task 2 first reconstructs predictor-only resources, then
freezes held-out scoring beta, the food bundle, and endpoint/analysis config.
It then generates and validates `method_lock_manifest.json` from those actual
inputs and the frozen normalization state. Before any validation endpoint or
label is read, the validator must recompute every applicable hash and
fingerprint, including `person_meal_validation_config_sha256` from the config
file's canonical bytes, `method_lock_schema_sha256` from the external schema
bytes, the release-registry snapshot hash, and the SHA-256 of the implemented
`method_lock_gate.py`. The schema hash is a run-instance field computed by the
external gate and is not embedded as a const in its own schema. The validator
checks the instance against `method_lock_manifest.schema.json` and confirms the
embargo. Any missing field or mismatch must fail closed and stop the validation
run without loading outcomes. Both the outcome loader and the Task 3 entry
independently recompute the schema, registry, gate implementation and config
hashes and compare them with the already verified manifest, so post-gate
modification is rejected at both boundaries.
