"""Synthetic objective, leakage, convergence and deployment contracts."""
import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from gmnps.response import joint_longitudinal_response as joint
from gmnps.response.joint_longitudinal_response import (
    AUXILIARY_WEIGHT, ETA, GAP_TOLERANCE, GROUPS, MEALS, MICRO_COLUMNS,
    MU_RATIO, N_FEATURES, N_RECIPES, PATHWAYS, SCALE_FLOOR, SPECIES,
    JointLongitudinalCorrection, OptimizationConvergenceError, QuadraticObjective,
)
from gmnps.response.microbiome import microbial_clr


def synthetic(n_families=16):
    rng = np.random.default_rng(7519)
    n = 2 * n_families
    people = pd.DataFrame(index=pd.Index([f'p{i:03}' for i in range(n)], name='participant_id'))
    people['family_group'] = np.repeat([f'f{i:03}' for i in range(n_families)], 2)
    followup = pd.DataFrame(index=people.index)
    for block in (SPECIES, PATHWAYS):
        baseline = rng.lognormal(0., .7, (n, len(block)))
        future = np.exp(.65 * np.log(baseline) + rng.normal(0., .15, baseline.shape))
        people[block] = baseline
        followup[block] = future
    people.loc[people.index[::9], SPECIES[0]] = 0.
    clr = microbial_clr(people)
    signal = (clr[:, 1] - clr[:, 2])
    signal /= signal.std()
    recipe_effect = np.linspace(-.6, .7, N_RECIPES)
    rows, offsets = [], []
    for person, pid in enumerate(people.index):
        for recipe, meal in enumerate(MEALS):
            # Deliberate missing event tests equal-person weighting.
            if person == 0 and recipe == 2:
                continue
            host = .4 + .03 * recipe + .01 * person
            value = host + .05 * (recipe - 3) + .25 * signal[person] * recipe_effect[recipe]
            value += rng.normal(0., .01)
            rows.append((pid, people.loc[pid, 'family_group'], meal, f'{pid}-{meal}', value))
            offsets.append(host)
    meals = pd.DataFrame(rows, columns=['participant_id', 'family_group', 'meal_id', 'meal_event_id', 'target'])
    return people, meals, np.asarray(offsets), followup


@pytest.fixture(scope='module')
def fitted():
    people, meals, host, followup = synthetic()
    paired = followup.loc[people.family_group.map(lambda f: int(f[1:]) % 2 == 0)]
    models = {group: JointLongitudinalCorrection(group).fit(people, meals, host, paired, 'synthetic')
              for group in GROUPS}
    return people, meals, host, paired, models


def assert_same(first, second):
    for name in ('C', 'D', 'a', 'b'):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.audit == second.audit


def test_fixed_scaling_shared_mu_and_all_current_training_responses(fitted):
    people, meals, host, paired, models = fitted
    counts = meals.participant_id.value_counts()
    weights = 1. / (len(people) * meals.participant_id.map(counts).to_numpy())
    expected_scale = np.sqrt(np.sum(weights * (meals.target.to_numpy() - host) ** 2))
    expected_target = microbial_clr(paired.loc[people.index[people.index.isin(paired.index)]])
    for model in models.values():
        assert model.response_scale == expected_scale
        assert model.fit_ids == people.index.tolist()
        assert model.fit_events == meals.meal_event_id.tolist()
        assert model.audit['equal_person_weight_min'] == pytest.approx(1. / len(people))
        assert model.audit['equal_person_weight_max'] == pytest.approx(1. / len(people))
        assert model.audit['l2_includes_intercepts']
        assert model.audit['eta'] == ETA
        assert model.audit['gap_bound'] <= model.optimization['gap_threshold']
        assert model.optimization['converged']
        json.dumps(model.audit, allow_nan=False)
    raw, real, shuffled = [models[g] for g in GROUPS[1:]]
    assert raw.mu == real.mu == shuffled.mu
    assert real.auxiliary_weight == shuffled.auxiliary_weight == AUXILIARY_WEIGHT
    assert raw.auxiliary_weight == 0.
    for model in (real, shuffled):
        np.testing.assert_allclose(model.auxiliary_scaler.mean_, expected_target.mean(0), atol=0, rtol=0)
        np.testing.assert_allclose(model.auxiliary_scaler.var_, expected_target.var(0), atol=2e-16, rtol=0)
        np.testing.assert_array_equal(model.scaler.mean_, raw.scaler.mean_)
        assert model.audit['paired_people'] == len(paired)
    np.testing.assert_array_equal(real.auxiliary_scaler.scale_, shuffled.auxiliary_scaler.scale_)
    assert np.linalg.norm(real.C - raw.C) > 1e-5  # auxiliary task actually couples to response coefficients
    assert np.linalg.norm(real.C - shuffled.C) > 1e-5


def test_explicit_masked_objective_and_directional_gradient(fitted):
    people, meals, host, paired, models = fitted
    model = models['joint_shuffle']
    rng = np.random.default_rng(815)
    coefficients = rng.normal(0., .08, (19, 25))
    x = model.scaler.transform(microbial_clr(people))
    positions = people.index.get_indexer(meals.participant_id)
    recipes = pd.Index(MEALS).get_indexer(meals.meal_id)
    counts = np.bincount(positions, minlength=len(people))
    weights = 1. / (len(people) * counts[positions])
    response = (meals.target.to_numpy() - host) / model.response_scale
    pair_positions = people.index.get_indexer(model.pair_ids)
    auxiliary = model.auxiliary_scaler.transform(microbial_clr(paired.loc[model.donor_ids]))

    def explicit_smooth(k):
        c, d, a, b = k[:18, :7], k[:18, 7:], k[18, :7], k[18, 7:]
        response_prediction = a[recipes] + np.sum(x[positions] * c[:, recipes].T, axis=1)
        response_loss = np.sum(weights * (response - response_prediction) ** 2)
        auxiliary_loss = np.mean((auxiliary - b - x[pair_positions] @ d) ** 2)
        return response_loss + AUXILIARY_WEIGHT * auxiliary_loss + ETA * np.sum(k ** 2) / 2.

    value, gradient = model.problem_.smooth_value_gradient(coefficients)
    assert value == pytest.approx(explicit_smooth(coefficients), abs=2e-15)
    for _ in range(5):
        direction = rng.normal(size=coefficients.shape)
        direction /= np.linalg.norm(direction)
        eps = 1e-6
        numerical = (explicit_smooth(coefficients + eps * direction)
                     - explicit_smooth(coefficients - eps * direction)) / (2 * eps)
        assert np.sum(gradient * direction) == pytest.approx(numerical, abs=3e-10)
    nuclear = np.linalg.svd(coefficients[:-1], compute_uv=False).sum()
    assert model.problem_.value(coefficients) == pytest.approx(value + model.mu * nuclear, abs=1e-15)
    assert np.linalg.eigvalsh(model.problem_.hessian).min() >= ETA - 1e-14
    assert model.audit['lipschitz'] == np.linalg.eigvalsh(model.problem_.hessian).max()
    explicit_gradient = np.zeros((18, 7))
    for recipe in range(7):
        keep = recipes == recipe
        explicit_gradient[:, recipe] = -2 * np.sum(
            (weights[keep] * response[keep])[:, None] * x[positions[keep]], axis=0)
    expected_mu = MU_RATIO * max(np.linalg.svd(explicit_gradient, compute_uv=False)[0], SCALE_FLOOR)
    assert model.mu == pytest.approx(expected_mu, abs=2e-17)
    np.testing.assert_allclose(model.objective_diagnostics()['objective'], model.audit['objective'], atol=0, rtol=0)


def test_nuclear_optimizer_has_known_exact_solution_and_valid_gap():
    # Isotropic quadratic has a closed-form singular-value threshold solution.
    hessian = np.repeat((2. * np.eye(19))[None, :, :], 7, axis=0)
    linear = np.zeros((19, 7))
    linear[:3, :3] = np.diag([-.6, -.3, -.01])
    linear[-1] = np.linspace(-.14, .14, 7)
    problem = QuadraticObjective(hessian, linear, 1., 0., 0., .05)
    expected = -linear / 2.
    left, singular, right = np.linalg.svd(expected[:-1], full_matrices=False)
    expected[:-1] = (left * np.maximum(singular - problem.mu / 2., 0.)) @ right
    actual, diagnostics = joint._solve(problem)
    np.testing.assert_allclose(actual, expected, atol=1e-16, rtol=0)
    assert np.linalg.matrix_rank(actual[:-1]) == 2
    gap = problem.value(actual) - problem.value(expected)
    assert -1e-14 <= gap <= diagnostics['gap_bound'] + 1e-14
    assert diagnostics['gap_bound'] <= diagnostics['gap_threshold']
    assert diagnostics['proximal_optimality_residual'] < 1e-14
    assert diagnostics['iterations'] == 1


@pytest.mark.parametrize('group', ['joint_real', 'joint_shuffle'])
def test_only_current_training_intersection_can_supply_followup(group):
    people, meals, host, followup = synthetic(12)
    train = people.loc[~people.family_group.isin(['f000', 'f001', 'f002'])]
    mask = meals.participant_id.isin(train.index).to_numpy()
    tm = meals.loc[mask].reset_index(drop=True)
    poisoned = followup.copy()
    poisoned.loc[~poisoned.index.isin(train.index), MICRO_COLUMNS] = np.nan
    original = JointLongitudinalCorrection(group).fit(train, tm, host[mask], followup, 'local_T')
    checked = JointLongitudinalCorrection(group).fit(train, tm, host[mask], poisoned, 'local_T')
    assert_same(original, checked)
    assert set(checked.pair_ids) == set(train.index)
    assert set(checked.donor_ids) <= set(train.index)
    assert not set(checked.audit['pair_families']) & {'f000', 'f001', 'f002'}


def test_controls_do_not_read_disallowed_labels_or_microbes(fitted):
    people, meals, host, paired, models = fitted
    host_only = JointLongitudinalCorrection('host_calibration').fit(
        people[['family_group']], meals, host, object(), 'synthetic')
    assert_same(host_only, models['host_calibration'])
    raw = JointLongitudinalCorrection('response_only').fit(people, meals, host, object(), 'synthetic')
    assert_same(raw, models['response_only'])
    assert raw.audit['pair_ids'] == []
    assert not raw.audit['auxiliary_values_read']
    assert not np.any(raw.D) and not np.any(raw.b)
    with pytest.raises(ValueError, match='Only joint groups'):
        raw.predict_auxiliary(people)


def test_family_block_shuffle_is_closed_deterministic_and_retains_singleton_strata():
    people, meals, host, followup = synthetic(10)
    people.loc[people.index[-4:], 'family_group'] = ['unique3', 'unique3', 'unique3', 'singleton']
    meals['family_group'] = meals.participant_id.map(people.family_group)
    followup = followup.drop(index=[people.index[1], people.index[3]])
    first = JointLongitudinalCorrection('joint_shuffle').fit(people, meals, host, followup, 'shuffle')
    second = JointLongitudinalCorrection('joint_shuffle').fit(people, meals, host, followup, 'shuffle')
    assert_same(first, second)
    assert set(first.pair_ids) == set(first.donor_ids)
    recipients = people.loc[first.pair_ids, 'family_group'].to_numpy()
    donors = people.loc[first.donor_ids, 'family_group'].to_numpy()
    for family in set(recipients):
        assert len(set(donors[recipients == family])) == 1
    saw_singleton = False
    for block in first.audit['mapping_strata']:
        if len(block['families']) > 1:
            assert block['nonzero_cycle_shift'] > 0
            assert all(a != b for a, b in zip(block['families'], block['source_families']))
        else:
            saw_singleton = True
            assert block['unmoved_singleton_stratum']
            assert block['families'] == block['source_families']
    assert saw_singleton


def test_joint_objective_uses_unpaired_response_labels_and_own_training_targets(fitted):
    people, meals, host, paired, models = fitted
    changed_meals = meals.copy()
    unpaired = ~changed_meals.participant_id.isin(paired.index)
    changed_meals.loc[unpaired, 'target'] += np.linspace(-.1, .1, unpaired.sum())
    changed = JointLongitudinalCorrection('joint_real').fit(people, changed_meals, host, paired, 'synthetic')
    original = models['joint_real']
    assert np.linalg.norm(changed.C - original.C) > 1e-5
    assert np.linalg.norm(changed.D - original.D) > 1e-5
    np.testing.assert_array_equal(changed.auxiliary_scaler.mean_, original.auxiliary_scaler.mean_)
    changed_aux = paired.copy()
    changed_aux.loc[changed_aux.index[0], SPECIES[1]] *= 20.
    updated = JointLongitudinalCorrection('joint_real').fit(people, meals, host, changed_aux, 'synthetic')
    assert np.linalg.norm(updated.C - original.C) > 1e-5
    # This is ordinary full-T supervised learning, not an auxiliary OOF descriptor.
    assert np.max(abs(updated.predict_auxiliary(people) - original.predict_auxiliary(people))) > 1e-5


@pytest.mark.parametrize('group', GROUPS)
def test_deployment_is_label_free_stable_and_reloadable(fitted, group, tmp_path):
    people, meals, host, paired, models = fitted
    model = models[group]
    expected = model.components(people, meals)
    audit = copy.deepcopy(model.audit)
    for person in people.index:
        mask = meals.participant_id.eq(person).to_numpy()
        actual = model.components(people.loc[[person]], meals.loc[mask])
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key][mask])
    for start in range(0, len(meals), 13):
        part = meals.iloc[start:start + 13]
        query = people.loc[people.index.isin(part.participant_id)]
        np.testing.assert_array_equal(model.correction(query, part), expected['correction'][start:start + 13])
    np.testing.assert_array_equal(model.correction(people.iloc[::-1], meals.iloc[::-1]), expected['correction'][::-1])
    query = people[MICRO_COLUMNS].copy()
    rename = {pid: f'new-{pid}' for pid in people.index}
    query.index = query.index.map(rename)
    query['target'] = np.nan
    query['day14_outcome'] = np.inf
    qm = meals[['participant_id', 'meal_id']].copy()
    qm['participant_id'] = qm.participant_id.map(rename)
    np.testing.assert_array_equal(model.correction(query, qm), expected['correction'])
    np.testing.assert_array_equal(model.empirical_reference_correction(query, qm), expected['recipe_calibration'])
    np.testing.assert_array_equal(expected['correction'], expected['recipe_calibration'] + expected['micro'])
    path = tmp_path / (group + '.joblib')
    joblib.dump(model, path)
    loaded = joblib.load(path)
    np.testing.assert_array_equal(loaded.correction(query, qm), expected['correction'])
    if group.startswith('joint_'):
        auxiliary = model.predict_auxiliary(people)
        for position, pid in enumerate(people.index):
            np.testing.assert_array_equal(model.predict_auxiliary(people.loc[[pid]]), auxiliary[[position]])
        np.testing.assert_array_equal(model.predict_auxiliary(query), auxiliary)
        np.testing.assert_array_equal(loaded.predict_auxiliary(query), auxiliary)
    assert model.audit == audit


def test_component_amplitude_and_person_mean_profile_decompositions(fitted):
    people, meals, host, paired, models = fitted
    model = models['joint_real']
    components = model.components(people, meals)
    # Fixed-recipe mean is over all training people, not an observed joint context.
    for meal in MEALS:
        grid = pd.DataFrame({'participant_id': people.index, 'meal_id': meal})
        assert abs(model.components(people, grid)['micro'].mean()) < 1e-15
    frame = pd.DataFrame({'person': meals.participant_id, 'micro': components['micro'],
                          'base': host + components['recipe_calibration'], 'y': meals.target})
    frame['full'] = frame.base + frame.micro
    center = lambda s: s - s.groupby(frame.person).transform('mean')
    average = lambda s: s.groupby(frame.person).mean().mean()
    residual = center(frame.y - frame.base)
    adjustment = center(frame.micro)
    gain = average(center(frame.y - frame.base) ** 2) - average(center(frame.y - frame.full) ** 2)
    assert gain == pytest.approx(2 * average(residual * adjustment) - average(adjustment ** 2), abs=1e-16)
    error = frame.full - frame.y
    assert average(error ** 2) == pytest.approx(
        np.mean(error.groupby(frame.person).mean() ** 2) + average(center(error) ** 2), abs=1e-16)
    assert np.sqrt(average(adjustment ** 2)) > 0.


def test_invalid_inputs_and_incomplete_optimization_are_rejected(monkeypatch):
    people, meals, host, followup = synthetic(8)
    with pytest.raises(ValueError, match='group must'):
        JointLongitudinalCorrection('unknown')
    with pytest.raises(ValueError, match='must be fitted'):
        JointLongitudinalCorrection('response_only').correction(people, meals)
    with pytest.raises(ValueError, match='one-dimensional host_oof'):
        JointLongitudinalCorrection('response_only').fit(people, meals, host[:, None], followup, 'bad')
    with pytest.raises(ValueError, match='exact meal-row or event-ID order'):
        JointLongitudinalCorrection('response_only').fit(
            people, meals, pd.Series(host, index=meals.index[::-1]), followup, 'bad')
    mismatch = meals.copy()
    mismatch.loc[0, 'family_group'] = 'another_family'
    with pytest.raises(ValueError, match='families must match'):
        JointLongitudinalCorrection('response_only').fit(people, mismatch, host, followup, 'bad')
    with pytest.raises(ValueError, match='paired day14'):
        JointLongitudinalCorrection('joint_real').fit(people, meals, host, followup.iloc[:0], 'bad')
    invalid = followup.copy()
    invalid.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match='Finite nonnegative'):
        JointLongitudinalCorrection('joint_real').fit(people, meals, host, invalid, 'bad')
    monkeypatch.setattr(joint, 'MAX_STEPS', 1)
    unfinished = JointLongitudinalCorrection('joint_real')
    with pytest.raises(OptimizationConvergenceError):
        unfinished.fit(people, meals, host, followup, 'insufficient')
    assert unfinished.audit['status'] == 'optimization_incomplete'
    assert not unfinished.audit['optimization']['converged']
    with pytest.raises(ValueError, match='must be fitted'):
        unfinished.correction(people, meals)
    with pytest.raises(ValueError, match='fresh Joint'):
        unfinished.fit(people, meals, host, followup, 'retry')


def test_zero_residual_floor_is_finite_and_does_not_create_signal():
    people, meals, host, followup = synthetic(8)
    perfect = meals.target.to_numpy().copy()
    for group in ('host_calibration', 'response_only'):
        model = JointLongitudinalCorrection(group).fit(people, meals, perfect, None, 'zero')
        assert model.response_scale == SCALE_FLOOR
        np.testing.assert_array_equal(model.correction(people, meals), np.zeros(len(meals)))
        assert model.audit['objective'] == 0.
        assert model.audit['gap_bound'] == 0.
        assert model.mu == (MU_RATIO * SCALE_FLOOR if group == 'response_only' else 0.)
        json.dumps(model.audit, allow_nan=False)
