# GutNut: Microbiome-informed nutrient profiling for precision nutrition

GutNut provides a microbiome-informed calibration of an established population
nutrient profile, together with supervised models of individual dietary
responses. This repository contains the optimized **core method implementation**:
nutrient-direction projection, MAC/LIPID attribute calibration, five-domain
scoring, empirical-reference calibration, and family-cross-fitted response
learning.

The food-score layer and the measured-response layer have separate outputs.
Food scores retain the Food Compass 2.0 (FCS2) reference scale. Response learners
estimate outcomes in the supplied response units and expose the contribution
of the microbial panel relative to a training-population reference.

## Repository scope

Included: in-memory fitting and prediction code, fixed algorithm settings,
input schemas, documentation, and deterministic synthetic software tests.

Excluded: datasets, cohort loaders, data cleaning and integration pipelines,
downloads, fitted model files, experiment outputs, manuscripts, and paper-figure
generation. CLR transformation, imputation and training-only scaling remain
inside the learners because they are part of the fitted algorithms. All
examples generate small arrays in memory; they contain no study observations.

## Install and run

Python 3.10 or newer is required. The distribution is named `gutnut`; the Python
import namespace remains `gmnps`.

```bash
git clone https://github.com/SunnyXi925/GutNut.git
cd GutNut
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/synthetic_workflow.py
```

A tested Python 3.11 dependency set is recorded in `requirements-tested.txt`.
To use that set, install with
`python -m pip install -c requirements-tested.txt -e ".[dev]"`.

The example demonstrates scoring, development-reference centering, model
training, and predictions for excluded families. Its invented signals and
responses illustrate the API; they are not estimates of study performance.

## Method

### 1. Nutrient-specific microbial signals

`NutrientCapacityProjector` receives a nutrient-by-genus direction matrix and
development microbial abundance profiles. Direction coefficients encode the
caller-supplied signed, weighted evidence and native attribute orientation.

The projector closes each abundance profile to unit sum and computes
`CLR = log(p + 1e-6) - mean(log(p + 1e-6))`. Each nutrient direction is centered
and L2-normalized over genera shared by the direction matrix and the fitted
composition; other aligned genera receive zero projection weight. A zero-norm
direction returns zero signal. The projected signal is standardized using the
development median and `1.4826 × MAD`, with population SD and then 1 as fallbacks
at tolerance `1e-12`.

This step learns reference normalization from development profiles. It does not
use disease labels or a health-classifier derivative. The fitted genus order,
directions, centers and scales are reused unchanged for prediction.

### 2. MAC/LIPID channels and native attribute scoring

The two channels organize evidence mappings and interpretation:

| Channel | Role |
| --- | --- |
| MAC | Fiber-centered mappings and separately supported carotenoid, vitamin and mineral food-matrix mappings |
| LIPID | Lipid-related mappings and separately supported choline, retinol and vitamin B12 mappings |

They are not two separately trained response networks. The mapping registry
contains 26 nutrient inputs, including 20 with attribute-effect roles; the
remaining inputs have applicability or explanatory roles. Only eligible paths
into the five recalculated domains enter the scoring design.

For standardized nutrient signals `z`, normalized food exposures `e`, and
allocation weights `w`, the attribute response and updated points are

```text
r[i,j,a] = sum_n z[i,n] * e[j,n,a] * w[n,a]
p_new[i,j,a] = clip(p[j,a] + 0.20 * (high[a] - low[a]) * tanh(r[i,j,a] / 2),
                    low[a], high[a])
```

Inputs use per-100-kcal exposures and the registry's attribute normalization
targets. Ratio and composite attributes use their component-specific rules.
The scorer recalculates nutrient ratios, vitamins, minerals, fiber and protein,
and phytochemicals. Vitamin/mineral top-five membership is selected once from
absolute baseline points with stable registry tie breaking. The phytochemical
domain has weight 0.5; total flavonoids stay fixed and mapped carotenoids may vary.

The official-score residual retains unchanged domains and components:

```text
U(F[j]) = -12.1 + 47.1 * (F[j] - 1) / 99
Q[j] = U(F[j]) - sum_d D_baseline[j,d]
U_person[i,j] = Q[j] + sum_d D_person[i,j,d]
GutNut[i,j] = 1 + 99 * (clip(U_person[i,j], -12.1, 35.0) + 12.1) / 47.1
```

The default `UncappedConsensusScorer` retains the native 1–100 bounds and applies
**no final ±12 deviation cap**. At zero nutrient signal, it returns the official
baseline score. The `tanh` attribute response, native attribute bounds and
native score bounds remain explicit, separately switchable stages in
`NativeStageEngine` for compression and derivative comparisons.

MAC and LIPID score contributions can be obtained by rerunning the scorer with
the other channel's signals set to zero. Total change minus the two single-channel
changes is the non-additive scoring remainder, not a biological interaction.

### 3. Optional development-reference score calibration

`UncappedConsensusScorer.fit_reference` fits a food-specific reference using
score changes **before native-total clipping**. The fitted reference is bound
to the food order, baseline, native arrays, mapping design and stage settings.

| Output map | Definition and properties |
| --- | --- |
| Default native map | Native 1–100 bounds; exact zero-signal baseline identity; no imposed foodwise mean constraint |
| `additive` | `F + delta_pre - mean_development(delta_pre)`; exact development foodwise mean; preserves pre-bound variance; unbounded output |
| `bounded_logit` | `1 + 99 * sigmoid(b_food + (delta_pre - mean_development) / tau)`; intercept fitted to the development food mean; 1–100 output |

For the logit map, foods with baseline 1 or 100 remain constant at that endpoint.
Reference parameters are frozen during prediction. The two reference maps are
implemented comparison candidates, not automatically selected replacements for
the default scorer. Mean preservation is a calibration property; measured-response
performance is assessed separately.

### 4. Supervised dietary-response learning

Machine learning enters the measured-response layer through a fitted host
predictor and a learned microbial-by-recipe residual correction:

```text
y_hat[i,r] = g(host[i], meal[i,r])
             + alpha * response_scale * (a[r] + M_standardized[i] @ C[:,r])
```

The supplied recipe schema has seven heads (`Meal2`–`Meal8`). The microbial schema
contains 12 species and six pathway coordinates, closed and CLR-transformed
separately, then standardized across unique training participants. Exact feature
names and host/meal input names are exported from `gmnps.response.schema`.
The pathway names are schema coordinates: their source-feature mapping must
remain identical between training and prediction.

`HomeFastingGlucoseModel` implements a recipe-mean baseline, Ridge, and histogram
gradient boosting, with matched `host_only` and `CLR_micro` inputs. The fixed
candidate settings are available as `NULL` and `GRID`. The home response target
used by these modules is signed glucose iAUC over 0–120 minutes divided by 120;
multiplication by 120 converts predictions back to iAUC units when that target
definition is used.

`JointLongitudinalCorrection` learns recipe intercepts and microbial coefficients
from **host residuals generated by family cross-fitting within its own training
set**. Each participant has equal total loss weight. The objective combines
squared residual error, a nuclear-norm penalty on microbial coefficients, and
an L2 penalty of `0.001 / 2` on all coefficients, including intercepts. The
nuclear coefficient is `0.1 × max(operator_norm(response_gradient_at_zero), 1e-8)`.
Zero-initialized accelerated proximal gradients use singular-value thresholding,
restarts every 100 iterations, a 20,000-iteration maximum, and explicit objective-gap
and optimality-residual convergence checks.

Implemented procedures are:

| Procedure | Training supervision |
| --- | --- |
| `host_calibration` | Recipe intercept correction without microbial coefficients |
| `response_only` | Response residuals with microbial-by-recipe coefficients |
| Day-14 auxiliary | Seven response heads and 18 later microbial-coordinate heads sharing a nuclear penalty |
| Same-meal auxiliary | Seven response heads and seven CGM-baseline heads sharing a nuclear penalty |
| `joint_shuffle` | Matched auxiliary procedure with deterministic family-preserving target-panel shuffling |

Auxiliary loss weight is 0.1. Auxiliary measurements are training targets and
are not read during deployment. These variants are available for controlled
comparison; the repository does not designate an auxiliary variant as a proven
performance winner.

`fit_nonnegative_scalar` estimates one stacking coefficient from inner-family-OOF
host and correction predictions, using equal-recipe weights:

```text
alpha = max(0, sum(w * correction * (target - host)) / sum(w * correction**2))
w[event] = 1 / (7 * number_of_training_events_for_its_recipe)
```

An exactly zero denominator yields zero. There is no upper bound on `alpha` and
no `tanh`, ±12 constraint or food-score bound on response predictions.

`fit_stacked_response` composes the learners in memory. It uses three family
folds for stacking; each fold's correction obtains its residual targets from
three additional family crossfits inside that fold's training population. It
then refits the host and correction on the full supplied training partition
and freezes `alpha`. The default host is the fixed 120-iteration, seven-leaf
boosting specification. Callers may supply another fixed host specification;
this helper does not select hyperparameters or perform the outer evaluation.

### 5. Empirical microbial reference

At fixed host and meal context, the microbial reference averages predictions
over complete observed panels from the model's training participants:

```text
reference(h,x) = mean_k f(h,x,M_training[k])
microbial_deviation = f(h,x,M_person) - reference(h,x)
```

For the centered linear correction, this is available analytically from
`JointStackingModel.predict_components`: `reference = host + recipe_calibration`
and `prediction = reference + micro`. The mean microbial deviation is zero over
the training panel at each fixed recipe. For nonlinear models with a single-frame
prediction API, `EmpiricalReferenceResponse` evaluates complete panels explicitly.
A single average or representative microbial panel is not this empirical mean.

## Input and API contracts

| API | Required inputs / outputs |
| --- | --- |
| `NutrientCapacityProjector.fit` | Development person-by-genus nonnegative abundances; nutrient-by-genus evidence directions supplied to the constructor |
| `NativeFoodScorer.from_frames` | Food-indexed metadata with `fcs2`; baseline points and effective weights in `ATTRIBUTES` order; exposures with `attrs['basis'] = 'per_100_kcal'` |
| `UncappedConsensusScorer.predict` | Person-by-nutrient finite standardized signals in exact `NUTRIENTS` order; returns person-by-food scores |
| `fit_stacked_response` | Current-training `people` and `meals` frames; returns a fitted `JointStackingModel` |
| `JointStackingModel.predict_components` | Query people and meal context; returns `host`, `recipe_calibration`, `micro`, `reference`, `prediction` |

All food frames must share unique, identically ordered indices. Uncalculated
native points use `NaN` with zero effective weight; each recalculated domain must
have a valid denominator. Exposures include the named side inputs required by
ratio/composite mappings. The direction matrix and native food arrays are supplied
by the caller; neither is bundled as study data.

For response fitting, `people` has a unique participant index, `family_group`,
and the prescribed host/microbial columns. `meals` supplies `participant_id`,
`family_group`, `meal_id`, unique `meal_event_id`, `target`, and the prescribed
meal/context columns. Each participant has at most one event per recipe; all
seven recipes must occur in each training context. Relatives remain together.
Prediction does not require response or auxiliary labels.

```python
from gmnps.response.training import fit_stacked_response

# train_people/train_meals are already aligned in-memory training inputs.
model = fit_stacked_response(train_people, train_meals, group="response_only")
parts = model.predict_components(query_people, query_meals)
prediction = parts["prediction"]
reference = parts["reference"]
microbial_deviation = parts["micro"]
```

For auxiliary comparisons, use `group="joint_real"` or `"joint_shuffle"` with
`auxiliary="day14", followup=training_followup`, or
`auxiliary="same_meal", baseline=training_baseline_series`. The baseline Series
must be indexed in exact current-training `meal_event_id` order. Lower-level
learners also accept externally generated host OOF predictions when the caller
maintains the same nested family-exclusion contract.

## Code layout

```text
code/src/gmnps/
  beta_i/nutrient_capacity.py       nutrient-direction projection
  scoring/fcs2_attribute_rules.py  native rules and aggregation
  scoring/fcs2_attribute_mapping.py MAC/LIPID mappings and exposure allocation
  scoring/native_scoring.py        aligned five-domain scoring arrays
  scoring/consensus_calibration.py stage diagnostics and reference maps
  scoring/uncapped_consensus_scorer.py default scoring interface
  scoring/empirical_reference_response.py empirical nonlinear response reference
  response/schema.py              ordered inputs and fixed candidate settings
  response/microbiome.py           CLR, family folds and control mappings
  response/host.py                 host and directly augmented learners
  response/joint_longitudinal_response.py convex response/day-14 correction
  response/same_meal_auxiliary.py  same-meal auxiliary correction
  response/stacking.py             nonnegative stacking and decomposition
  response/training.py             in-memory nested fitting procedure
code/src/tests/                    synthetic numerical and integration tests
examples/synthetic_workflow.py     runnable in-memory demonstration
```

## Changes from the earlier repository

The primary signal is now the development-normalized nutrient projection rather
than a scalar health-index finite difference. The default scorer fixes baseline
top-five membership, includes the weighted phytochemical domain, and removes
the final deviation cap. Empirical-reference score maps and supervised response
corrections are explicit separate interfaces. Core algorithms have been detached
from cohort loaders and experiment/plotting runners; the earlier data-source,
configuration and manuscript workflow directories are not part of this release.

Software tests cover scoring identities and bounds, uncapped behavior, reference
means, analytic derivatives, solver objectives and convergence, family exclusion,
auxiliary controls, stacking, serialization and batch-invariant inference.
Study-level efficacy requires the corresponding measured outcomes and evaluation
protocol; passing the synthetic tests does not establish that efficacy.
