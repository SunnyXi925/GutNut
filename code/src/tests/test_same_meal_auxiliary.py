"""Synthetic contracts specific to masked same-meal auxiliary supervision."""
import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from gmnps.response.joint_longitudinal_response import (
    JointLongitudinalCorrection, SPECIES, PATHWAYS,
)
from gmnps.response.microbiome import microbial_clr
from gmnps.response.same_meal_auxiliary import (
    AUXILIARY_WEIGHT, ETA, MEALS, MICRO_COLUMNS, N_FEATURES, N_RECIPES,
    SCALE_FLOOR, SEED, SameMealBaselineCorrection, _family_panel_mapping,
)


def synthetic(n_families=16):
    rng = np.random.default_rng(6172)
    n = 2 * n_families
    people = pd.DataFrame(index=pd.Index([f'p{i:03}' for i in range(n)], name='participant_id'))
    people['family_group'] = np.repeat([f'f{i:03}' for i in range(n_families)], 2)
    for block in (SPECIES, PATHWAYS):
        people[block] = rng.lognormal(0, .6, (n, len(block)))
    x = microbial_clr(people)
    signal = (x[:, 1] - x[:, 2])
    signal /= signal.std()
    rows, host, baseline = [], [], []
    for p, pid in enumerate(people.index):
        for r, recipe in enumerate(MEALS):
            # Matching multi-family masks, plus a singleton mask stratum.
            if (p // 2 < 4 and p % 2 == 0 and r == 2) or (p == n - 1 and r == 4):
                continue
            offset = .3 + .01 * p + .02 * r
            target = offset + .03 * r + .3 * signal[p] * (.4 + .1 * r)
            rows.append((pid, people.loc[pid, 'family_group'], recipe, f'{pid}-{recipe}', target))
            host.append(offset)
            baseline.append(4.8 + .05 * r + (.3 + .05 * r) * signal[p])
    meals = pd.DataFrame(rows, columns=['participant_id', 'family_group', 'meal_id', 'meal_event_id', 'target'])
    return people, meals, np.array(host), pd.Series(baseline, index=pd.Index(meals.meal_event_id))


@pytest.fixture(scope='module')
def fitted():
    p, m, h, b = synthetic()
    models = {g: SameMealBaselineCorrection(g).fit(p, m, h, b, 'synthetic_same_meal')
              for g in ('joint_real', 'joint_shuffle')}
    return p, m, h, b, models


def event_arrays(people, meals):
    positions = people.index.get_indexer(meals.participant_id)
    recipes = pd.Index(MEALS).get_indexer(meals.meal_id)
    weights = 1. / (len(people) * np.bincount(positions)[positions])
    return positions, recipes, weights


def test_masked_sufficient_statistics_and_directional_gradient(fitted):
    p, m, h, b, models = fitted
    model = models['joint_shuffle']
    positions, recipes, weights = event_arrays(p, m)
    x = model.scaler.transform(microbial_clr(p))
    response = (m.target.to_numpy() - h) / model.response_scale
    mapped = b.index.get_indexer(model.audit['mapped_event_ids'])
    auxiliary = ((b.to_numpy() - model.auxiliary_scaler.mean_[recipes]) /
                 model.auxiliary_scaler.scale_[recipes])[mapped]
    z = np.column_stack([x[positions], np.ones(len(m))])
    rng = np.random.default_rng(441)
    k = rng.normal(0, .07, (N_FEATURES + 1, 2 * N_RECIPES))

    def explicit(coeff):
        prediction = np.sum(z * coeff[:, recipes].T, axis=1)
        aux_prediction = np.sum(z * coeff[:, N_RECIPES + recipes].T, axis=1)
        return (np.sum(weights * (response - prediction) ** 2) +
                AUXILIARY_WEIGHT * np.sum(weights * (auxiliary - aux_prediction) ** 2) +
                ETA * np.sum(coeff ** 2) / 2)

    value, gradient = model.problem_.smooth_value_gradient(k)
    assert value == pytest.approx(explicit(k), abs=2e-15)
    for _ in range(6):
        direction = rng.normal(size=k.shape)
        direction /= np.linalg.norm(direction)
        eps = 1e-6
        fd = (explicit(k + eps * direction) - explicit(k - eps * direction)) / (2 * eps)
        assert np.sum(gradient * direction) == pytest.approx(fd, abs=4e-10)
    diagnostics = model.objective_diagnostics()
    coeff = np.vstack([np.column_stack([model.C, model.D]), np.r_[model.a, model.b]])
    assert diagnostics['objective'] == pytest.approx(model.problem_.value(coeff), abs=2e-15)
    assert model.problem_.auxiliary_constant == pytest.approx(.1, abs=2e-15)
    assert model.problem_.hessian.shape == (14, 19, 19)


def test_weighted_recipe_scaling_and_response_penalty_unchanged(fitted):
    p, m, h, b, models = fitted
    _, recipes, weights = event_arrays(p, m)
    response_only = JointLongitudinalCorrection('response_only').fit(p, m, h, None, 'synthetic_same_meal')
    for model in models.values():
        assert model.response_scale == response_only.response_scale
        assert model.mu == response_only.mu
        np.testing.assert_array_equal(model.scaler.mean_, response_only.scaler.mean_)
        np.testing.assert_array_equal(model.problem_.hessian[:7], response_only.problem_.hessian)
        np.testing.assert_array_equal(model.problem_.linear[:, :7], response_only.problem_.linear)
        for r in range(N_RECIPES):
            selected = recipes == r
            w, values = weights[selected], b.to_numpy()[selected]
            mean = np.sum(w * values) / w.sum()
            variance = np.sum(w * (values - mean) ** 2) / w.sum()
            assert model.auxiliary_scaler.mean_[r] == pytest.approx(mean, abs=3e-15)
            assert model.auxiliary_scaler.var_[r] == pytest.approx(variance, abs=3e-15)
        assert model.audit['optimization']['converged']
        assert model.audit['excluded_events'] == 0
        assert not model.audit['missing_meals_imputed']
        json.dumps(model.audit, allow_nan=False)
    np.testing.assert_array_equal(models['joint_real'].auxiliary_scaler.scale_,
                                  models['joint_shuffle'].auxiliary_scaler.scale_)


def test_family_panel_shuffle_preserves_masks_labels_weights_and_is_outcome_free(fitted):
    p, m, _, b, models = fitted
    model = models['joint_shuffle']
    donors, records = _family_panel_mapping(p, m[['participant_id', 'meal_id']], 'shuffle', SEED, 'synthetic_same_meal')
    assert donors == model.donor_ids and records == model.audit['mapping_strata']
    reversed_donors, _ = _family_panel_mapping(p.iloc[::-1], m.iloc[::-1], 'shuffle', SEED, 'synthetic_same_meal')
    assert dict(zip(p.index, donors)) == dict(zip(p.index[::-1], reversed_donors))
    positions, recipes, weights = event_arrays(p, m)
    mapped = b.index.get_indexer(model.audit['mapped_event_ids'])
    np.testing.assert_array_equal(weights, weights[mapped])
    np.testing.assert_array_equal(recipes, recipes[mapped])
    assert sorted(mapped) == list(range(len(m)))
    for r in range(N_RECIPES):
        keep = recipes == r
        np.testing.assert_array_equal(np.sort(b.to_numpy()[keep]), np.sort(b.to_numpy()[mapped][keep]))
        assert np.sum(weights[keep] * b.to_numpy()[keep]) == pytest.approx(
            np.sum(weights[keep] * b.to_numpy()[mapped][keep]), abs=3e-16)
    for family in p.family_group.unique():
        member_positions = np.flatnonzero(p.family_group.eq(family))
        assert p.loc[np.asarray(donors)[member_positions], 'family_group'].nunique() == 1
    singleton = [r for r in records if r['unmoved_singleton_stratum']]
    assert singleton and all(r['families'] == r['source_families'] for r in singleton)
    assert 0 < model.audit['mapping_moved_person_fraction'] < 1
    assert model.audit['mapping_moved_weight_mass'] == pytest.approx(model.audit['mapping_moved_person_fraction'])
    assert models['joint_real'].audit['mapping_moved_person_fraction'] == 0


def test_only_exact_current_training_event_labels_are_accepted():
    p, m, h, b = synthetic(8)
    train = p.loc[~p.family_group.eq('f000')]
    keep = m.participant_id.isin(train.index).to_numpy()
    tm = m.loc[keep]
    baseline = b.loc[pd.Index(tm.meal_event_id)]
    model = SameMealBaselineCorrection('joint_real').fit(train, tm, h[keep], baseline, 'local_T')
    assert set(model.fit_ids) == set(train.index)
    assert set(model.audit['mapped_event_ids']) == set(tm.meal_event_id)
    for invalid in (b, baseline.iloc[::-1], baseline.to_numpy(), baseline.reset_index(drop=True)):
        with pytest.raises(ValueError, match='exact current-T meal_event_id'):
            SameMealBaselineCorrection('joint_real').fit(train, tm, h[keep], invalid, 'local_T')
    poisoned = baseline.copy()
    poisoned.iloc[0] = np.nan
    with pytest.raises(ValueError, match='Finite baseline'):
        SameMealBaselineCorrection('joint_real').fit(train, tm, h[keep], poisoned, 'local_T')
    # Extraneous columns are never label sources, including held-out placeholders.
    changed = tm.assign(cgm_baseline_mmol_l=np.inf, day14_outcome=np.nan)
    other = SameMealBaselineCorrection('joint_real').fit(train, changed, h[keep], baseline, 'local_T')
    assert model.audit == other.audit
    for key in model.coeff:
        np.testing.assert_array_equal(model.coeff[key], other.coeff[key])


def test_shared_structure_recovers_synthetic_response_and_auxiliary(fitted):
    p, m, h, b, models = fitted
    real, shuffled = models['joint_real'], models['joint_shuffle']
    positions, recipes, weights = event_arrays(p, m)
    response_prediction = real.correction(p, m)
    aux_prediction = real.predict_auxiliary(p)[positions, recipes]
    assert np.corrcoef(response_prediction, m.target.to_numpy() - h)[0, 1] > .98
    assert np.corrcoef(aux_prediction, b.to_numpy())[0, 1] > .98
    assert np.linalg.norm(real.C - shuffled.C) > 1e-5
    real_error = np.sum(weights * (b.to_numpy() - aux_prediction) ** 2)
    shuffled_error = np.sum(weights * (b.to_numpy() - shuffled.predict_auxiliary(p)[positions, recipes]) ** 2)
    assert real_error < shuffled_error
    assert real.D.shape == (18, 7) and real.b.shape == (7,)


@pytest.mark.parametrize('group', ['joint_real', 'joint_shuffle'])
def test_label_free_batch_permutation_and_reload_stability(fitted, group, tmp_path):
    p, m, _, _, models = fitted
    model = models[group]
    query = p[MICRO_COLUMNS].copy()
    query['cgm_baseline_mmol_l'] = np.inf
    meals = m[['participant_id', 'meal_id']].copy()
    meals['target'] = np.nan
    meals['cgm_baseline_mmol_l'] = np.inf
    expected = model.components(query, meals)
    auxiliary = model.predict_auxiliary(query)
    audit = copy.deepcopy(model.audit)
    for i, pid in enumerate(query.index):
        keep = meals.participant_id.eq(pid).to_numpy()
        np.testing.assert_array_equal(model.correction(query.loc[[pid]], meals.loc[keep]), expected['correction'][keep])
        np.testing.assert_array_equal(model.predict_auxiliary(query.loc[[pid]]), auxiliary[[i]])
    np.testing.assert_array_equal(model.correction(query.iloc[::-1], meals.iloc[::-1]), expected['correction'][::-1])
    np.testing.assert_array_equal(model.predict_auxiliary(query.iloc[::-1]), auxiliary[::-1])
    np.testing.assert_array_equal(model.empirical_reference_correction(query, meals), expected['recipe_calibration'])
    for recipe in MEALS:
        grid = pd.DataFrame({'participant_id': query.index, 'meal_id': recipe})
        assert abs(model.components(query, grid)['micro'].mean()) < 1e-15
    path = tmp_path / 'same_meal.joblib'
    joblib.dump(model, path)
    loaded = joblib.load(path)
    np.testing.assert_array_equal(loaded.predict_auxiliary(query), auxiliary)
    np.testing.assert_array_equal(loaded.correction(query, meals), expected['correction'])
    assert model.audit == audit


def test_zero_targets_and_no_microbial_variation_do_not_create_signal():
    p, m, h, b = synthetic(8)
    zero = pd.Series(0., index=b.index)
    for group in ('joint_real', 'joint_shuffle'):
        model = SameMealBaselineCorrection(group).fit(p, m, m.target.to_numpy(), zero, 'zero')
        assert model.response_scale == SCALE_FLOOR
        np.testing.assert_array_equal(model.correction(p, m), np.zeros(len(m)))
        np.testing.assert_array_equal(model.predict_auxiliary(p), np.zeros((len(p), 7)))
        assert model.objective_diagnostics()['objective'] == 0
        assert np.all(model.auxiliary_scaler.scale_ == SCALE_FLOOR)
        # Auxiliary signal alone cannot make a nonzero response correction.
        aux_only = SameMealBaselineCorrection(group).fit(p, m, m.target.to_numpy(), b, 'aux_only')
        np.testing.assert_allclose(aux_only.correction(p, m), 0, atol=1e-20, rtol=0)
    constant = p.copy()
    constant[MICRO_COLUMNS] = 1.
    no_signal = SameMealBaselineCorrection('joint_real').fit(constant, m, h, b, 'no_signal')
    np.testing.assert_array_equal(no_signal.C, np.zeros((18, 7)))
    np.testing.assert_array_equal(no_signal.D, np.zeros((18, 7)))


def test_training_row_permutation_is_numerically_stable(fitted):
    p, m, h, b, models = fitted
    for group, first in models.items():
        second = SameMealBaselineCorrection(group).fit(p.iloc[::-1], m.iloc[::-1], h[::-1], b.iloc[::-1], 'synthetic_same_meal')
        np.testing.assert_allclose(second.correction(p, m), first.correction(p, m), atol=2e-13, rtol=0)
        np.testing.assert_allclose(second.predict_auxiliary(p), first.predict_auxiliary(p), atol=2e-13, rtol=0)
        assert first.response_scale == pytest.approx(second.response_scale, abs=1e-16)


def test_rejects_old_controls_and_repeated_fit(fitted):
    p, m, h, b, models = fitted
    for group in ('host_calibration', 'response_only', 'unknown'):
        with pytest.raises(ValueError, match='group must'):
            SameMealBaselineCorrection(group)
    with pytest.raises(ValueError, match='fresh SameMeal'):
        models['joint_real'].fit(p, m, h, b, 'again')
