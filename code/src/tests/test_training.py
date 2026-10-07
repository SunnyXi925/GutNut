"""Integration tests for in-memory training and family exclusion."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmnps.response.host import HomeFastingGlucoseModel
from gmnps.response.training import fit_stacked_response
from gmnps.response.schema import GRID

path = Path(__file__).resolve().parents[3] / 'examples/synthetic_workflow.py'
spec = importlib.util.spec_from_file_location('synthetic_workflow', path)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_public_scoring_demo_accepts_frames_and_preserves_bounds():
    score = example.demo_scoring()
    assert score.shape == (3, 3)
    assert ((score >= 1) & (score <= 100)).all().all()


def test_nested_training_excludes_families_from_host_residuals(monkeypatch):
    people, meals = example.response_inputs()
    calls = []
    original_fit = HomeFastingGlucoseModel.fit
    original_predict = HomeFastingGlucoseModel.predict

    def fit(self, people, meals, target):
        calls.append(set(people.family_group))
        return original_fit(self, people, meals, target)

    def predict(self, people, meals):
        # During cross-fitting each call uses an excluded family panel.
        assert not set(people.family_group).intersection(self.fit_families)
        return original_predict(self, people, meals)

    monkeypatch.setattr(HomeFastingGlucoseModel, 'fit', fit)
    monkeypatch.setattr(HomeFastingGlucoseModel, 'predict', predict)
    model = fit_stacked_response(people, meals, host_spec=GRID[0], context_label='test_training')
    assert len(calls) == 13
    assert len(model.training_audit['folds']) == 3
    assert model.fit_ids == list(people.index)
    for split in model.training_audit['folds']:
        assert not set(split['training_families']) & set(split['validation_families'])
    monkeypatch.setattr(HomeFastingGlucoseModel, 'predict', original_predict)
    components = model.predict_components(people, meals.drop(columns='target'))
    np.testing.assert_allclose(components.prediction, components.reference + components.micro)
    for recipe in example.MEALS:
        cross = meals.loc[meals.meal_id.eq(recipe)].drop(columns='target')
        assert abs(model.predict_components(people, cross).micro.mean()) < 1e-10


def test_invalid_auxiliary_combination_rejected():
    people, meals = example.response_inputs()
    with pytest.raises(ValueError, match='disagree'):
        fit_stacked_response(people, meals, group='joint_real')


@pytest.mark.parametrize('auxiliary', ['day14', 'same_meal'])
def test_auxiliary_training_requires_no_deployment_labels(auxiliary):
    people, meals = example.response_inputs()
    kwargs = {'followup': people[example.MICRO_COLUMNS].copy()} if auxiliary == 'day14' else {
        'baseline': pd.Series(5. + .1 * meals.target.to_numpy(), index=pd.Index(meals.meal_event_id))}
    model = fit_stacked_response(people, meals, group='joint_real', auxiliary=auxiliary,
                                 host_spec=GRID[0], context_label='auxiliary_test', **kwargs)
    prediction = model.predict(people, meals.drop(columns='target'))
    assert prediction.shape == (len(meals),)
    assert np.isfinite(prediction).all()
