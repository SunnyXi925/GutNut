"""Executable API demonstration using generated arrays only, with no file I/O.

The invented directions, food points and responses are software examples. They
are not the study evidence registry, food composition data, or validation data.
"""
import numpy as np
import pandas as pd

from gmnps.beta_i.nutrient_capacity import NutrientCapacityProjector
from gmnps.scoring.fcs2_attribute_mapping import PRIMARY_ATTRIBUTE_MAPPINGS
from gmnps.scoring.fcs2_attribute_rules import FCS2_RULES
from gmnps.scoring.native_scoring import ATTRIBUTES, NUTRIENTS, NativeFoodScorer
from gmnps.scoring.uncapped_consensus_scorer import UncappedConsensusScorer
from gmnps.response.schema import MICRO_COLUMNS, NUMERIC, PERSON_CAT, FOOD, CONTEXT, MEALS, GRID
from gmnps.response.training import fit_stacked_response


def scoring_inputs():
    rng = np.random.default_rng(20)
    genera = [f'genus_{k}' for k in range(6)]
    development = pd.DataFrame(rng.lognormal(size=(40, 6)), columns=genera,
                               index=[f'development_{k}' for k in range(40)])
    query = pd.DataFrame(rng.lognormal(size=(3, 6)), columns=genera, index=['query_a', 'query_b', 'query_c'])
    directions = pd.DataFrame(rng.normal(size=(len(NUTRIENTS), 6)), index=NUTRIENTS, columns=genera)
    food_ids = pd.Index(['food_a', 'food_b', 'food_c'])
    metadata = pd.DataFrame({'fcs2': [25., 50., 80.]}, index=food_ids)
    midpoint = [(FCS2_RULES[a].low_points + FCS2_RULES[a].high_points) / 2 for a in ATTRIBUTES]
    points = pd.DataFrame(np.tile(midpoint, (3, 1)), index=food_ids, columns=ATTRIBUTES)
    weights = pd.DataFrame(np.tile([FCS2_RULES[a].weight for a in ATTRIBUTES], (3, 1)),
                           index=food_ids, columns=ATTRIBUTES)
    columns = list(dict.fromkeys([m.nutrient for m in PRIMARY_ATTRIBUTE_MAPPINGS] + [
        'Sodium (mg)', 'Fatty acids, total monounsaturated (g)',
        'Fatty acids, total polyunsaturated (g)', 'Folate, DFE (mcg_DFE)', 'Vitamin A, RAE (mcg_RAE)']))
    exposures = pd.DataFrame(0., index=food_ids, columns=columns)
    for column, value in {'Carbohydrate (g)': 10., 'Sodium (mg)': 100., 'Total Fat (g)': 3.,
                          'Fatty acids, total saturated (g)': 1., 'Fatty acids, total monounsaturated (g)': 1.,
                          'Fatty acids, total polyunsaturated (g)': 1., 'Fiber, total dietary (g)': 2.,
                          'Vitamin C (mg)': 10., 'Potassium (mg)': 150.}.items():
        exposures[column] = value
    exposures.attrs['basis'] = 'per_100_kcal'
    return development, query, directions, metadata, points, weights, exposures


def demo_scoring():
    development, query, directions, metadata, points, weights, exposures = scoring_inputs()
    projector = NutrientCapacityProjector(directions).fit(development)
    native = NativeFoodScorer.from_frames(metadata, points, weights, exposures)
    scorer = UncappedConsensusScorer(native)
    signals = projector.transform(query).loc[:, NUTRIENTS]
    scores = scorer.predict(signals)
    reference_scorer = UncappedConsensusScorer.fit_reference(
        native, projector.transform(development).loc[:, NUTRIENTS], method='additive')
    assert np.allclose(reference_scorer.predict(projector.transform(development)).mean(), metadata.fcs2)
    return scores


def response_inputs(n_families=18):
    rng = np.random.default_rng(7519)
    n = 2 * n_families
    people = pd.DataFrame(index=pd.Index([f'person_{k}' for k in range(n)], name='participant_id'))
    people['family_group'] = np.repeat([f'family_{k}' for k in range(n_families)], 2)
    people[MICRO_COLUMNS] = rng.lognormal(size=(n, len(MICRO_COLUMNS)))
    people[NUMERIC] = rng.normal(size=(n, len(NUMERIC)))
    for column in PERSON_CAT:
        people[column] = rng.choice(['a', 'b'], size=n)
    meals = pd.DataFrame([(p, people.loc[p, 'family_group'], r, f'{p}_{r}')
                          for p in people.index for r in MEALS],
                         columns=['participant_id', 'family_group', 'meal_id', 'meal_event_id'])
    meals[FOOD + CONTEXT] = rng.uniform(.1, 1., size=(len(meals), len(FOOD + CONTEXT)))
    recipe = meals.meal_id.map(dict(zip(MEALS, np.linspace(-.6, .7, len(MEALS))))).to_numpy()
    micro = np.log(people.loc[meals.participant_id, MICRO_COLUMNS[1]].to_numpy())
    meals['target'] = .4 + recipe * micro + .1 * meals.energy_kcal + rng.normal(0., .05, len(meals))
    return people, meals


def demo_response():
    people, meals = response_inputs()
    # Split complete families before calling the training procedure.
    heldout = people.family_group.isin(['family_16', 'family_17'])
    training, query = people.loc[~heldout], people.loc[heldout]
    train_meals = meals.loc[meals.participant_id.isin(training.index)]
    query_meals = meals.loc[meals.participant_id.isin(query.index)].drop(columns='target')
    # A fixed Ridge host keeps the demonstration small; the API default is GRID[1] HGB.
    model = fit_stacked_response(training, train_meals, host_spec=GRID[0], context_label='synthetic_demo')
    components = model.predict_components(query, query_meals)
    np.testing.assert_allclose(components.prediction, components.reference + components.micro)
    return model, components


if __name__ == '__main__':
    print('Illustrative scores (invented inputs):')
    print(demo_scoring().round(4).to_string())
    model, components = demo_response()
    print('\nSynthetic response model alpha:', model.alpha)
    print(components.head().round(4).to_string(index=False))
