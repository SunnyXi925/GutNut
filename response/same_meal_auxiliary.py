"""Masked same-meal meal-start CGM-baseline supervision of a frozen host.

Only current-training events may be supplied. ``meals.target`` is already the
signed response divided by 120; the host OOF provenance remains caller-owned.
The CGM baseline window is [-15, 0] minutes, including time zero. This is a
custom signed-response plus baseline objective, not a log absolute-AUC replica.
Deployment reads baseline microbial abundances only, never CGM baseline labels.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .joint_longitudinal_response import (
    AUXILIARY_WEIGHT, ETA, MEALS, MICRO_COLUMNS, MU_RATIO, N_FEATURES,
    N_RECIPES, SCALE_FLOOR, SEED, FrozenColumnScaler, JointLongitudinalCorrection,
    OptimizationConvergenceError, QuadraticObjective, _array_hash,
    _event_positions, _host_array, _solve,
)
from .microbiome import _family_labels, _labels, microbial_clr

GROUPS = ('joint_real', 'joint_shuffle')


class RecipeBaselineScaler(FrozenColumnScaler):
    """Recipe moments under the current-training person-meal measure."""

    @classmethod
    def from_events(cls, values, recipes, weights):
        mass = np.bincount(recipes, weights=weights, minlength=N_RECIPES)
        counts = np.bincount(recipes, minlength=N_RECIPES)
        if np.any(mass <= 0):
            raise ValueError('Each baseline recipe requires observed training events')
        mean = np.bincount(recipes, weights=weights * values, minlength=N_RECIPES) / mass
        variance = np.bincount(recipes, weights=weights * (values - mean[recipes]) ** 2,
                               minlength=N_RECIPES) / mass
        result = cls(mean, np.maximum(np.sqrt(variance), SCALE_FLOOR), variance, len(values))
        result.recipe_mass_ = mass
        result.recipe_counts_ = counts
        return result

    def specification(self):
        return {'mean': self.mean_.tolist(), 'scale': self.scale_.tolist(),
                'variance': self.var_.tolist(), 'n_events': int(self.n_samples_seen_),
                'recipe_event_counts': self.recipe_counts_.tolist(),
                'recipe_weight_mass': self.recipe_mass_.tolist(), 'recipes': list(MEALS),
                'weight_rule': 'current-T 1/(n_people * observed_meals_per_person), normalized within recipe',
                'sd_floor': SCALE_FLOOR, 'ddof': 0}


def _family_panel_mapping(people, meals, mode, seed, context_label):
    """One outcome-free family permutation within ordered observation masks."""
    if mode not in ('real', 'shuffle'):
        raise ValueError('Mapping mode must be real or shuffle')
    families = _family_labels(people)
    positions, recipes = _event_positions(people, meals)
    masks = np.zeros((len(people), N_RECIPES), dtype=bool)
    masks[positions, recipes] = True
    by_family = {}
    for pid in people.index:
        by_family.setdefault(families.loc[pid], []).append(pid)
    strata = {}
    for family, ids in by_family.items():
        ids.sort(key=str)
        key = (len(ids), tuple(tuple(int(v) for v in masks[people.index.get_loc(pid)]) for pid in ids))
        strata.setdefault(key, []).append(family)
    donors = list(people.index)
    records = []
    for (size, ordered_masks), members in sorted(strata.items()):
        members.sort()
        payload = json.dumps([int(seed), context_label, 'same_meal_full', size, ordered_masks],
                             ensure_ascii=False, separators=(',', ':'))
        rng_seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big')
        order, shift = list(members), 0
        if mode == 'shuffle' and len(order) > 1:
            rng = np.random.default_rng(rng_seed)
            order = [order[i] for i in rng.permutation(len(order))]
            shift = int(rng.integers(1, len(order)))
        sources = order[shift:] + order[:shift]
        for recipient, source in zip(order, sources):
            for pid, donor in zip(by_family[recipient], by_family[source]):
                donors[people.index.get_loc(pid)] = donor
        records.append({'members_per_family': size, 'ordered_recipe_masks': [list(m) for m in ordered_masks],
                        'families': order, 'source_families': sources, 'rng_seed': rng_seed,
                        'nonzero_cycle_shift': shift, 'unmoved_singleton_stratum': len(order) == 1})
    if len(set(donors)) != len(people) or set(donors) != set(people.index):
        raise RuntimeError('Family-panel mapping must be a closed bijection')
    donor_positions = people.index.get_indexer(donors)
    if not np.array_equal(masks, masks[donor_positions]):
        raise RuntimeError('Family-panel mapping changed an observation mask')
    return donors, records


def _build_objective(x, positions, recipes, weights, targets, auxiliary):
    """Fourteen masked heads with the response-only nuclear penalty scale."""
    dimensions = N_FEATURES + 1
    hessian = np.repeat((ETA * np.eye(dimensions))[None], 2 * N_RECIPES, axis=0)
    linear = np.zeros((dimensions, 2 * N_RECIPES))
    design = np.column_stack([x[positions], np.ones(len(positions))])
    for recipe in range(N_RECIPES):
        chosen = recipes == recipe
        z, w = design[chosen], weights[chosen]
        gram = 2. * z.T @ (w[:, None] * z)
        hessian[recipe] += gram
        hessian[N_RECIPES + recipe] += AUXILIARY_WEIGHT * gram
        linear[:, recipe] = -2. * z.T @ (w * targets[chosen])
        linear[:, N_RECIPES + recipe] = -2. * AUXILIARY_WEIGHT * z.T @ (w * auxiliary[chosen])
    hessian = (hessian + hessian.swapaxes(1, 2)) / 2.
    mu = MU_RATIO * max(float(np.linalg.svd(linear[:N_FEATURES, :N_RECIPES], compute_uv=False)[0]), SCALE_FLOOR)
    return QuadraticObjective(hessian, linear, float(np.sum(weights * targets ** 2)),
                              float(AUXILIARY_WEIGHT * np.sum(weights * auxiliary ** 2)),
                              AUXILIARY_WEIGHT, mu)


class SameMealBaselineCorrection(JointLongitudinalCorrection):
    """Seven response and seven CGM-baseline heads sharing a nuclear penalty.

    ``baseline`` must be a finite Series in exact meal_event_id order, containing
    only current-T observed events. Missing meals create no auxiliary loss terms.
    ``predict_auxiliary`` returns n-by-seven CGM baseline predictions in mmol/L.
    The inherited baseline-only inference methods never access these labels.
    """

    def __init__(self, group):
        if group not in GROUPS:
            raise ValueError(f'group must be one of {GROUPS}')
        super().__init__(group)

    def fit(self, people, meals, host_oof, baseline, context_label):
        if self._fit_started:
            raise ValueError('Use a fresh SameMealBaselineCorrection for every fit')
        if not isinstance(context_label, str) or not context_label:
            raise ValueError('Nonempty context_label required')
        self._fit_started = True
        families = _family_labels(people)
        positions, recipes = _event_positions(people, meals, training=True)
        if not isinstance(baseline, pd.Series) or not baseline.index.equals(pd.Index(meals.meal_event_id)):
            raise ValueError('baseline must be a Series in exact current-T meal_event_id order')
        values = baseline.to_numpy(dtype=np.float64)
        observed = meals.target.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.isfinite(observed).all():
            raise ValueError('Finite baseline and response labels required for every observed current-T event')
        host = _host_array(host_oof, meals)
        counts = np.bincount(positions, minlength=len(people))
        weights = 1. / (len(people) * counts[positions])
        residual = observed - host
        self.response_scale = max(float(np.sqrt(np.sum(weights * residual ** 2))), SCALE_FLOOR)
        self.scale = self.response_scale
        targets = residual / self.response_scale
        raw = microbial_clr(people)
        self.scaler = FrozenColumnScaler.from_array(raw)
        x = self.scaler.transform(raw)
        self.auxiliary_scaler = RecipeBaselineScaler.from_events(values, recipes, weights)
        donors, strata = _family_panel_mapping(people, meals, 'real' if self.group == 'joint_real' else 'shuffle',
                                              SEED, context_label)
        donor_positions = people.index.get_indexer(donors)
        event_lookup = {(int(p), int(r)): i for i, (p, r) in enumerate(zip(positions, recipes))}
        mapped = np.asarray([event_lookup[(int(donor_positions[p]), int(r))] for p, r in zip(positions, recipes)])
        if not np.array_equal(weights, weights[mapped]) or len(np.unique(mapped)) != len(meals):
            raise RuntimeError('Auxiliary mapping must preserve every recipe target weight')
        auxiliary = ((values - self.auxiliary_scaler.mean_[recipes]) / self.auxiliary_scaler.scale_[recipes])[mapped]
        self.context_label = context_label
        self.fit_ids, self.fit_events = _labels(people.index), _labels(meals.meal_event_id)
        self.fit_families = sorted(families.unique().tolist())
        self.pair_ids, self.donor_ids = self.fit_ids.copy(), _labels(donors)
        self.auxiliary_weight = AUXILIARY_WEIGHT
        self.problem_ = _build_objective(x, positions, recipes, weights, targets, auxiliary)
        self.mu = self.problem_.mu
        moved = donor_positions != np.arange(len(people))
        moved_families = families.to_numpy() != families.to_numpy()[donor_positions]
        self.audit = {
            'group': self.group, 'context_label': context_label, 'status': 'preparing',
            'fit_ids': self.fit_ids, 'fit_events': self.fit_events, 'fit_families': self.fit_families,
            'response_people': len(people), 'response_events': len(meals), 'features': list(MICRO_COLUMNS),
            'recipes': list(MEALS), 'auxiliary_heads': list(MEALS), 'response_scale': self.response_scale,
            'response_scale_rule': 'equal-person current-T host-OOF residual RMS, floor 1e-8',
            'host_oof_provenance': 'caller must verify family crossfits within this exact T',
            'host_oof_sha256': _array_hash(host), 'response_target_sha256': _array_hash(observed),
            'response_weights_sha256': _array_hash(weights), 'baseline_target_sha256': _array_hash(values),
            'baseline_clr_sha256': _array_hash(raw), 'mapped_standardized_auxiliary_sha256': _array_hash(auxiliary),
            'mapped_event_ids': _labels(meals.meal_event_id.iloc[mapped]),
            'equal_person_weight_min': float(np.bincount(positions, weights=weights).min()),
            'equal_person_weight_max': float(np.bincount(positions, weights=weights).max()),
            'lambda': AUXILIARY_WEIGHT, 'eta': ETA, 'l2_includes_intercepts': True,
            'mu': self.mu, 'mu_ratio': MU_RATIO,
            'mu_rule': '0.1*max(operator_norm(response_gradient_at_zero),1e-8)',
            'pair_ids': self.pair_ids, 'donor_ids': self.donor_ids, 'pair_families': self.fit_families,
            'paired_people': len(people), 'mapping_strata': strata, 'mapping_seed': SEED,
            'mapping_fit_label': 'same_meal_full', 'auxiliary_values_read': True,
            'auxiliary_target': 'cgm_baseline_mmol_l; current-T observed same-meal events only',
            'auxiliary_loss_rule': '0.1 * sum(person_meal_weight * standardized_baseline_error_squared)',
            'mapping_moved_person_fraction': float(moved.mean()),
            'mapping_moved_event_fraction': float(moved[positions].mean()),
            'mapping_moved_weight_mass': float(np.sum(weights * moved[positions])),
            'mapping_moved_family_fraction': float(families[moved_families].nunique() / families.nunique()),
            'missing_meals_imputed': False, 'excluded_events': 0,
        }
        try:
            coefficients, self.optimization = _solve(self.problem_)
        except OptimizationConvergenceError as error:
            self.audit.update({'status': 'optimization_incomplete', 'optimization': error.diagnostics})
            raise
        self.C, self.D = coefficients[:N_FEATURES, :N_RECIPES].copy(), coefficients[:N_FEATURES, N_RECIPES:].copy()
        self.a, self.b = coefficients[-1, :N_RECIPES].copy(), coefficients[-1, N_RECIPES:].copy()
        self.coeff = {'C': self.C, 'D': self.D, 'a': self.a, 'b': self.b}
        self.scales = {'response': self.response_scale, 'baseline': self.scaler.specification(),
                       'auxiliary': self.auxiliary_scaler.specification()}
        self.loss_diagnostics = self.problem_.diagnostics(coefficients)
        singular = np.linalg.svd(np.column_stack([self.C, self.D]), compute_uv=False)
        threshold = max(N_FEATURES, 2 * N_RECIPES) * np.finfo(float).eps * singular[0]
        self.audit.update({
            'status': 'converged', 'scales': self.scales, 'baseline_scaler': self.scales['baseline'],
            'auxiliary_scaler': self.scales['auxiliary'], 'optimization': self.optimization,
            'losses': self.loss_diagnostics, 'objective': self.loss_diagnostics['objective'],
            'gap_bound': self.optimization['gap_bound'], 'iterations': self.optimization['iterations'],
            'lipschitz': self.optimization['lipschitz'], 'singular_values': singular.tolist(),
            'response_singular_values': np.linalg.svd(self.C, compute_uv=False).tolist(),
            'numerical_rank': int(np.sum(singular > threshold)), 'rank_threshold': float(threshold),
            'coefficients': {k: v.tolist() for k, v in self.coeff.items()},
            'response_gradient_operator_norm': float(np.linalg.svd(self.problem_.linear[:N_FEATURES, :N_RECIPES], compute_uv=False)[0]),
            'inference': 'baseline microbial abundances only; no CGM-baseline label access; stable column accumulation',
            'reference': 'zero standardized microbial vector; current-T empirical CLR mean',
        })
        self._fitted = True
        return self
