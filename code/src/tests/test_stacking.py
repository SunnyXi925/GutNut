"""Synthetic scalar calibration and deployment tests."""

import joblib
import numpy as np
import pandas as pd
import pytest

import gmnps.response.stacking as module


MEALS = module.MEALS


class SyntheticHost:
    def __init__(self, people, meals):
        self.fit_ids = people.index.tolist()
        self.fit_events = meals.meal_event_id.tolist()
        self.fit_families = sorted(people.family_group.unique())

    def predict(self, people, meals):
        return people.loc[meals.participant_id, 'host_signal'].to_numpy()+meals.meal_id.map(dict(zip(MEALS, np.arange(7)*.1))).to_numpy()


class SyntheticCorrection:
    def __init__(self, people, meals, group='joint_real'):
        self.fit_ids = people.index.tolist()
        self.fit_events = meals.meal_event_id.tolist()
        self.fit_families = sorted(people.family_group.unique())
        self.group, self.response_scale = group, 7.
        self.center = float(people[module.MICRO_COLUMNS[0]].mean())

    def components(self, people, meals):
        recipe = meals.meal_id.map(dict(zip(MEALS, range(7)))).to_numpy()
        cal = self.response_scale*.03*(recipe-3.)
        micro = self.response_scale*(people.loc[meals.participant_id, module.MICRO_COLUMNS[0]].to_numpy()-self.center)*(recipe-2.)/5.
        if self.group == 'host_calibration':
            micro = np.zeros(len(meals))
        return {'recipe_calibration': cal, 'micro': micro, 'correction': cal+micro}

    def correction(self, people, meals):
        return self.components(people, meals)['correction']

    def empirical_reference_correction(self, people, meals):
        return self.components(people, meals)['recipe_calibration']


def fixture():
    rng = np.random.default_rng(33)
    people = pd.DataFrame({'family_group': np.repeat([f'f{k}' for k in range(15)], 2),
        'host_signal': rng.normal(size=30), 'micro_signal': rng.normal(size=30)},
        index=pd.Index([f'p{k:02d}' for k in range(30)], name='participant_id'))
    for column in module.MICRO_COLUMNS:
        people[column] = rng.uniform(.1, 1., len(people))
    rows = [{'participant_id': p, 'family_group': people.loc[p, 'family_group'],
             'meal_id': r, 'meal_event_id': p+'_'+r} for p in people.index for r in MEALS]
    meals = pd.DataFrame(rows).iloc[:-1].reset_index(drop=True)
    host, correction = SyntheticHost(people, meals), SyntheticCorrection(people, meals)
    meals['target'] = host.predict(people, meals)+.4*correction.correction(people, meals)+rng.normal(0, .01, len(meals))
    return people, meals


def test_equal_recipe_nnls_matches_independent_recipe_sufficient_statistics():
    recipes = np.asarray([MEALS[0]]*11+list(MEALS[1:]))
    c = np.linspace(.2, 2.8, len(recipes))
    g = np.linspace(-1., 1., len(recipes))
    y = g+c*np.where(recipes == MEALS[0], .1, 2.5)
    result = module.fit_nonnegative_scalar(y, g, c, recipes)
    numerator = sum(np.mean(c[recipes == r]*(y-g)[recipes == r]) for r in MEALS)/7
    denominator = sum(np.mean(c[recipes == r]**2) for r in MEALS)/7
    assert result['alpha'] == pytest.approx(numerator/denominator)
    assert result['alpha'] != pytest.approx(np.dot(c, y-g)/np.dot(c, c))
    explicit = np.mean([np.mean((y[recipes == r]-g[recipes == r]-result['alpha']*c[recipes == r])**2) for r in MEALS])
    assert result['inner_mse']['nonnegative'] == pytest.approx(explicit)
    assert result['kkt_violation'] <= result['kkt_tolerance']


@pytest.mark.parametrize('scale', [2.75, -1.5, 0.])
def test_unbounded_positive_and_nonnegative_boundary(scale):
    c = np.linspace(-.8, .9, 7)
    out = module.fit_nonnegative_scalar(scale*c, np.zeros(7), c, MEALS)
    assert out['free_alpha'] == pytest.approx(scale)
    assert out['alpha'] == pytest.approx(max(scale, 0.))
    if scale < 0.:
        assert out['gradient'] > 0.
    assert out['inner_mse']['nonnegative'] <= out['inner_mse']['zero']+1e-12
    assert out['inner_mse']['nonnegative'] <= out['inner_mse']['unit']+1e-12


def test_exact_zero_denominator_and_no_tiny_energy_threshold():
    zero = module.fit_nonnegative_scalar(np.arange(7), np.zeros(7), np.zeros(7), MEALS)
    assert zero['alpha'] == zero['denominator'] == 0.
    c = np.full(7, 1e-12)
    tiny = module.fit_nonnegative_scalar(5*c, np.zeros(7), c, MEALS)
    assert tiny['denominator'] > 0.
    assert tiny['alpha'] == pytest.approx(5.)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_input_stops(bad):
    y = np.zeros(7)
    y[2] = bad
    with pytest.raises(ValueError, match='Nonfinite'):
        module.fit_nonnegative_scalar(y, np.zeros(7), np.ones(7), MEALS)


def test_overflow_and_missing_recipe_stop():
    with pytest.raises((ValueError, FloatingPointError)):
        module.fit_nonnegative_scalar(np.full(7, 1e308), np.zeros(7), np.full(7, 1e308), MEALS)
    with pytest.raises(ValueError, match='seven'):
        module.fit_nonnegative_scalar(np.ones(6), np.zeros(6), np.ones(6), MEALS[:-1])



def test_wrapper_never_scales_twice_and_preserves_reference(tmp_path):
    people, meals = fixture()
    host, correction = SyntheticHost(people, meals), SyntheticCorrection(people, meals)
    model = module.JointStackingModel(host, correction, 2.75)
    expected = host.predict(people, meals)+2.75*correction.correction(people, meals)
    np.testing.assert_array_equal(model.predict(people, meals), expected)
    assert np.max(abs(expected-host.predict(people, meals))) > .1
    comp = model.predict_components(people, meals)
    np.testing.assert_allclose(comp.prediction, comp.reference+comp.micro, atol=1e-12, rtol=0)
    path = tmp_path/'model.joblib'
    joblib.dump(model, path)
    loaded = joblib.load(path)
    poisoned = meals.copy()
    poisoned['target'] = -1e12
    poison_people = people.copy()
    poison_people['day14_target'] = 1e12
    np.testing.assert_array_equal(loaded.predict(poison_people, poisoned), expected)
    for person in people.index[:3]:
        keep = meals.participant_id.eq(person)
        np.testing.assert_array_equal(loaded.predict(people.loc[[person]], meals.loc[keep]), expected[keep])
    zero = module.JointStackingModel(host, correction, 0.)
    np.testing.assert_array_equal(zero.predict(people, meals), host.predict(people, meals))
    for recipe in MEALS:
        cross = pd.DataFrame({'participant_id': people.index, 'meal_id': recipe})
        assert abs(np.mean(model.alpha*correction.components(people, cross)['micro'])) < 1e-12
