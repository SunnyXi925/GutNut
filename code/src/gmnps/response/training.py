"""In-memory family cross-fitting for a fixed host/correction specification.

This composes the released learners without changing their fitting objectives.
All arguments describe the current training partition. Evaluation participants
are supplied only to the returned model's prediction methods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .host import HomeFastingGlucoseModel
from .joint_longitudinal_response import JointLongitudinalCorrection, _event_positions
from .microbiome import family_folds
from .same_meal_auxiliary import SameMealBaselineCorrection
from .schema import GRID
from .stacking import JointStackingModel, fit_nonnegative_scalar


def _parts(people, meals):
    assignment = family_folds(people)
    if set(assignment) != {0, 1, 2}:
        raise ValueError('Each training context requires three nonempty family folds')
    for fold in range(3):
        training = people.iloc[np.flatnonzero(assignment != fold)]
        validation = people.iloc[np.flatnonzero(assignment == fold)]
        train_mask = meals.participant_id.isin(training.index).to_numpy()
        valid_mask = meals.participant_id.isin(validation.index).to_numpy()
        yield training, meals.loc[train_mask], validation, meals.loc[valid_mask], valid_mask


def _fit_host(people, meals, spec):
    return HomeFastingGlucoseModel('host_only', spec).fit(people, meals, meals.target.to_numpy(float))


def crossfit_host(people, meals, *, host_spec=None):
    """Return host predictions from three fits excluding each recipient family."""
    _event_positions(people, meals, training=True)
    spec = dict(GRID[1] if host_spec is None else host_spec)
    prediction = np.full(len(meals), np.nan)
    for tp, tm, vp, vm, mask in _parts(people, meals):
        prediction[mask] = _fit_host(tp, tm, spec).predict(vp, vm)
    if not np.isfinite(prediction).all():
        raise ValueError('Incomplete host cross-fitting')
    return pd.Series(prediction, index=pd.Index(meals.meal_event_id), name='host_oof')


def fit_stacked_response(people, meals, *, group='response_only', auxiliary=None,
                         host_spec=None, followup=None, baseline=None, context_label='training'):
    """Fit a fixed response procedure, including inner-family-OOF stacking.

    ``auxiliary=None`` supports response_only and host_calibration controls.
    ``auxiliary='day14'`` or ``'same_meal'`` supports joint_real/joint_shuffle.
    Day-14 panels use the ordered microbial schema. Same-meal baseline labels
    are a Series indexed in exact current-training meal_event_id order.
    ``host_spec`` is fixed before this call; no candidate selection occurs here.
    """
    _event_positions(people, meals, training=True)
    if not isinstance(context_label, str) or not context_label:
        raise ValueError('Nonempty context_label required')
    if auxiliary not in (None, 'day14', 'same_meal'):
        raise ValueError('Unknown auxiliary mode')
    allowed = ('response_only', 'host_calibration') if auxiliary is None else ('joint_real', 'joint_shuffle')
    if group not in allowed:
        raise ValueError('Correction group and auxiliary mode disagree')
    if auxiliary == 'day14' and followup is None:
        raise ValueError('Day-14 training panels required')
    if auxiliary == 'same_meal' and (not isinstance(baseline, pd.Series) or
            not baseline.index.equals(pd.Index(meals.meal_event_id))):
        raise ValueError('Same-meal labels must match current-training event order')
    spec = dict(GRID[1] if host_spec is None else host_spec)

    def fit_correction(p, m, host_oof, label):
        if auxiliary == 'same_meal':
            return SameMealBaselineCorrection(group).fit(p, m, host_oof,
                baseline.loc[pd.Index(m.meal_event_id)], label)
        future = followup if auxiliary == 'day14' else None
        return JointLongitudinalCorrection(group).fit(p, m, host_oof, future, label)

    host_oof = np.full(len(meals), np.nan)
    correction_oof = np.full(len(meals), np.nan)
    split_audit = []
    for fold, (tp, tm, vp, vm, mask) in enumerate(_parts(people, meals)):
        host = _fit_host(tp, tm, spec)
        correction = fit_correction(tp, tm, crossfit_host(tp, tm, host_spec=spec),
                                    f'{context_label}/inner{fold}')
        host_oof[mask] = host.predict(vp, vm)
        correction_oof[mask] = correction.correction(vp, vm)
        split_audit.append({'training_ids': list(tp.index), 'validation_ids': list(vp.index),
                            'training_families': sorted(tp.family_group.unique()),
                            'validation_families': sorted(vp.family_group.unique())})
    scalar = fit_nonnegative_scalar(meals.target.to_numpy(float), host_oof, correction_oof, meals.meal_id)
    final_host = _fit_host(people, meals, spec)
    final_correction = fit_correction(people, meals,
        pd.Series(host_oof, index=pd.Index(meals.meal_event_id)), f'{context_label}/full')
    model = JointStackingModel(final_host, final_correction, scalar['alpha'])
    model.training_audit = {'context': context_label, 'host_spec': spec, 'auxiliary': auxiliary,
                            'group': group, 'folds': split_audit, 'stacking': scalar}
    return model
