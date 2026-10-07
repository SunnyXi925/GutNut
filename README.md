# Code for “Gut microbiome-informed calibration of nutrient profiling reconciles population consensus with precision nutrition”

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/SunnyXi925/GutNut.git
cd GutNut
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## Food scoring

Supply person-by-genus abundance DataFrames and a nutrient-by-genus `directions`
DataFrame. Align `metadata`, `points`, `weights` and `exposures` by food ID.
`metadata` must contain `fcs2`; point and weight columns follow `ATTRIBUTES`
in `scoring/native_scoring.py`. Nutrient exposures use the names defined in
`scoring/fcs2_attribute_mapping.py` and are expressed per 100 kcal.

```python
from gutnut.beta_i.nutrient_capacity import NutrientCapacityProjector
from gutnut.scoring.native_scoring import NativeFoodScorer, NUTRIENTS
from gutnut.scoring.uncapped_consensus_scorer import UncappedConsensusScorer

projector = NutrientCapacityProjector(directions).fit(development_microbiome)
exposures.attrs["basis"] = "per_100_kcal"
native = NativeFoodScorer.from_frames(metadata, points, weights, exposures)
scorer = UncappedConsensusScorer(native)
signals = projector.transform(query_microbiome).loc[:, NUTRIENTS]
scores = scorer.predict(signals)
```

## Response fitting and prediction

Supply participant-indexed `train_people` and event-level `train_meals` containing
the response `target`. Required host, microbial and meal columns are defined in
`response/schema.py`. Include `family_group`, unique participant IDs and unique
`meal_event_id` values; keep related participants in the same training partition.

```python
from gutnut.response.training import fit_stacked_response

model = fit_stacked_response(train_people, train_meals)
predictions = model.predict(query_people, query_meals)
components = model.predict_components(query_people, query_meals)
```
