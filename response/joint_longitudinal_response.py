"""Fixed-budget convex response/day14 correction to an externally fitted host.

This module does not fit a host model, load cohort files, split observations, or
write artifacts. ``host_oof`` must come from family crossfits *inside* the exact
training set passed to ``fit``; the caller owns that provenance check. Every
free coefficient, including intercepts, receives the fixed L2 penalty.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from .microbiome import (
    MICRO_COLUMNS, PATHWAYS, SEED, SPECIES, _family_labels, _labels, _mapping,
    _validate_index, microbial_clr,
)
from .schema import MEALS


GROUPS = ('host_calibration', 'response_only', 'joint_real', 'joint_shuffle')
N_FEATURES = len(MICRO_COLUMNS)
N_RECIPES = len(MEALS)
ETA = 0.001
AUXILIARY_WEIGHT = 0.1
MU_RATIO = 0.1
SCALE_FLOOR = 1e-8
GAP_TOLERANCE = 1e-8
MAX_STEPS = 20_000
RESTART_EVERY = 100
PROXIMAL_RESIDUAL_TOLERANCE = 1e-10


def _array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


@dataclass
class FrozenColumnScaler:
    """Population moments with the protocol's explicit SD floor, not clipping."""

    mean_: np.ndarray
    scale_: np.ndarray
    var_: np.ndarray
    n_samples_seen_: int

    @classmethod
    def from_array(cls, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError('Nonempty finite matrix required for training moments')
        mean = values.mean(axis=0)
        variance = ((values - mean) ** 2).mean(axis=0)
        return cls(mean, np.maximum(np.sqrt(variance), SCALE_FLOOR), variance, len(values))

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mean_) or not np.isfinite(values).all():
            raise ValueError('Finite matrix with the fitted columns required')
        return np.ascontiguousarray((values - self.mean_) / self.scale_)

    def inverse_transform(self, values):
        return np.asarray(values, dtype=np.float64) * self.scale_ + self.mean_

    def specification(self):
        return {'mean': self.mean_.tolist(), 'scale': self.scale_.tolist(),
                'variance': self.var_.tolist(), 'n_people': int(self.n_samples_seen_),
                'sd_floor': SCALE_FLOOR, 'ddof': 0}


@dataclass
class QuadraticObjective:
    """Sufficient statistics for a strictly convex smooth loss plus trace norm.

    Each column of coefficients holds one head: microbial slopes first and its
    intercept last. Hessian blocks already include ETA on every diagonal.
    """

    hessian: np.ndarray
    linear: np.ndarray
    response_constant: float
    auxiliary_constant: float
    auxiliary_weight: float
    mu: float
    eta: float = ETA

    def smooth_value_gradient(self, coefficients):
        applied = np.einsum('kij,jk->ik', self.hessian, coefficients, optimize=False)
        value = (0.5 * np.sum(coefficients * applied) + np.sum(self.linear * coefficients)
                 + self.response_constant + self.auxiliary_constant)
        return float(value), applied + self.linear

    def value(self, coefficients):
        smooth, _ = self.smooth_value_gradient(coefficients)
        nuclear = float(np.linalg.svd(coefficients[:-1], compute_uv=False).sum())
        return smooth + self.mu * nuclear

    def diagnostics(self, coefficients):
        applied = np.einsum('kij,jk->ik', self.hessian, coefficients, optimize=False)
        per_head = np.sum(0.5 * coefficients * applied + self.linear * coefficients, axis=0)
        l2_per_head = self.eta * np.sum(coefficients ** 2, axis=0) / 2
        response = float(np.sum((per_head - l2_per_head)[:N_RECIPES]) + self.response_constant)
        auxiliary_weighted = float(np.sum((per_head - l2_per_head)[N_RECIPES:]) + self.auxiliary_constant)
        singular = np.linalg.svd(coefficients[:-1], compute_uv=False)
        nuclear_penalty = float(self.mu * singular.sum())
        l2_penalty = float(l2_per_head.sum())
        return {'response_loss': response,
                'auxiliary_loss': auxiliary_weighted / self.auxiliary_weight if self.auxiliary_weight else 0.,
                'weighted_auxiliary_loss': auxiliary_weighted,
                'l2_penalty': l2_penalty, 'nuclear_penalty': nuclear_penalty,
                'objective': response + auxiliary_weighted + l2_penalty + nuclear_penalty,
                'singular_values': singular.tolist()}

    def lipschitz(self):
        return float(np.linalg.eigvalsh(self.hessian).max())


class OptimizationConvergenceError(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__('Joint correction did not meet the fixed optimization gap certificate: '
                         f"{diagnostics['gap_bound']:.6g} after {diagnostics['iterations']} steps")


def _solve(problem):
    """One zero-start, fixed-restart FISTA trajectory; no validation stopping."""
    lipschitz = problem.lipschitz()
    if not np.isfinite(lipschitz) or lipschitz <= 0:
        raise ValueError('Finite positive training Hessian Lipschitz constant required')
    step = 1. / lipschitz
    current = np.zeros_like(problem.linear)
    extrapolated = current.copy()
    momentum = 1.
    initial = problem.value(current)
    trace = [{'iteration': 0, 'objective': initial}]
    diagnostics = {}
    for iteration in range(1, MAX_STEPS + 1):
        _, gradient_u = problem.smooth_value_gradient(extrapolated)
        proposal = extrapolated - step * gradient_u
        left, singular, right = np.linalg.svd(proposal[:-1], full_matrices=False)
        threshold = step * problem.mu
        proposal[:-1] = (left * np.maximum(singular - threshold, 0.)) @ right
        smooth, gradient_v = problem.smooth_value_gradient(proposal)
        value = smooth + problem.mu * float(np.maximum(singular - threshold, 0.).sum())

        # The exact proximal step supplies a subgradient h of the full objective.
        h = (extrapolated - proposal) / step + gradient_v - gradient_u
        gap_bound = float(np.sum(h * h) / (2 * problem.eta))
        proximal_subgradient = np.zeros_like(proposal)
        proximal_subgradient[:-1] = (left * np.minimum(singular / step, problem.mu)) @ right
        proximal_residual = ((extrapolated - proposal) / step - gradient_u
                             - proximal_subgradient)
        proximal_norm = float(np.linalg.norm(proximal_residual))
        tolerance = GAP_TOLERANCE * max(1., abs(float(value)))
        diagnostics = {
            'optimizer': 'fixed_restart_fista', 'initialization': 'zero',
            'iterations': iteration, 'maximum_steps': MAX_STEPS,
            'restart_every': RESTART_EVERY, 'lipschitz': lipschitz, 'step_size': step,
            'initial_objective': initial, 'objective': float(value),
            'gap_bound': gap_bound, 'gap_threshold': tolerance,
            'subgradient_norm': float(np.linalg.norm(h)),
            'proximal_optimality_residual': proximal_norm,
            'proximal_residual_threshold': PROXIMAL_RESIDUAL_TOLERANCE,
            'converged': bool(gap_bound <= tolerance and proximal_norm <= PROXIMAL_RESIDUAL_TOLERANCE),
        }
        if not np.isfinite([value, gap_bound, proximal_norm]).all():
            raise ValueError('Nonfinite convex optimization diagnostics')
        if iteration % RESTART_EVERY == 0 or diagnostics['converged'] or iteration == MAX_STEPS:
            trace.append({'iteration': iteration, 'objective': float(value),
                          'gap_bound': gap_bound, 'proximal_optimality_residual': proximal_norm})
        if diagnostics['converged']:
            diagnostics['trace'] = trace
            return proposal, diagnostics
        if iteration % RESTART_EVERY == 0:
            next_momentum = 1.
            extrapolated = proposal.copy()
        else:
            next_momentum = (1. + np.sqrt(1. + 4. * momentum * momentum)) / 2.
            extrapolated = proposal + ((momentum - 1.) / next_momentum) * (proposal - current)
        momentum = next_momentum
        current = proposal
    diagnostics['trace'] = trace
    raise OptimizationConvergenceError(diagnostics)


def _event_positions(people, meals, training=False):
    _validate_index(people, 'people')
    if not isinstance(meals, pd.DataFrame) or not meals.columns.is_unique:
        raise ValueError('meals must be a DataFrame with unique columns')
    required = {'participant_id', 'meal_id'}
    if training:
        required |= {'meal_event_id', 'family_group', 'target'}
    if not required.issubset(meals.columns):
        raise ValueError(f'Missing meal columns: {sorted(required - set(meals.columns))}')
    if meals[list(required)].isna().any().any():
        raise ValueError('Nonmissing event identities, families, recipes and target required')
    positions = people.index.get_indexer(meals.participant_id)
    if (positions < 0).any():
        raise ValueError('Every meal must match an allowed person')
    recipes = pd.Index(MEALS).get_indexer(meals.meal_id)
    if (recipes < 0).any():
        raise ValueError('Only the fixed seven Meal2..Meal8 recipes are supported')
    if training:
        if not len(meals) or set(meals.participant_id) != set(people.index):
            raise ValueError('Every training person must have observed response meals')
        if not meals.meal_event_id.is_unique or meals.duplicated(['participant_id', 'meal_id']).any():
            raise ValueError('Unique events and one meal per training person/recipe required')
        if set(meals.meal_id) != set(MEALS):
            raise ValueError('All seven recipes must occur in each training context')
        expected = people.family_group.astype(str).to_numpy()[positions]
        if not np.array_equal(expected, meals.family_group.astype(str).to_numpy()):
            raise ValueError('Meal families must match current-training person families')
    return positions, recipes


def _host_array(host_oof, meals):
    if isinstance(host_oof, pd.Series):
        event_index = pd.Index(meals.meal_event_id)
        if not host_oof.index.equals(meals.index) and not host_oof.index.equals(event_index):
            raise ValueError('host_oof Series must have exact meal-row or event-ID order')
    values = np.asarray(host_oof, dtype=np.float64)
    if values.shape != (len(meals),) or not np.isfinite(values).all():
        raise ValueError('Finite row-aligned one-dimensional host_oof required')
    return values


def _build_objective(x, positions, recipes, weights, targets, pair_positions, auxiliary, auxiliary_weight):
    outputs = N_RECIPES + (N_FEATURES if auxiliary_weight else 0)
    dimensions = N_FEATURES + 1
    hessian = np.repeat((ETA * np.eye(dimensions))[None, :, :], outputs, axis=0)
    linear = np.zeros((dimensions, outputs), dtype=np.float64)
    augmented = np.column_stack([x[positions], np.ones(len(positions))])
    for recipe in range(N_RECIPES):
        chosen = recipes == recipe
        design, w = augmented[chosen], weights[chosen]
        hessian[recipe] += 2. * design.T @ (w[:, None] * design)
        linear[:, recipe] = -2. * design.T @ (w * targets[chosen])
    auxiliary_constant = 0.
    if auxiliary_weight:
        design = np.column_stack([x[pair_positions], np.ones(len(pair_positions))])
        weight = auxiliary_weight / (N_FEATURES * len(pair_positions))
        hessian[N_RECIPES:] += 2. * weight * (design.T @ design)[None, :, :]
        linear[:, N_RECIPES:] = -2. * weight * design.T @ auxiliary
        auxiliary_constant = float(weight * np.sum(auxiliary ** 2))
    hessian = (hessian + hessian.swapaxes(1, 2)) / 2.
    response_gradient = linear[:N_FEATURES, :N_RECIPES]
    mu = MU_RATIO * max(float(np.linalg.svd(response_gradient, compute_uv=False)[0]), SCALE_FLOOR)
    return QuadraticObjective(hessian, linear, float(np.sum(weights * targets ** 2)),
                              auxiliary_constant, auxiliary_weight, mu)


class JointLongitudinalCorrection:
    """A full-T correction with a frozen, caller-supplied host OOF offset.

    The host-only branch does not read microbial columns or followup values.
    The response-only branch never reads followup values. Joint branches read
    only the followup rows in ``people.index``. Deployment never reads labels.
    ``predict_auxiliary`` returns predicted CLR-coordinate units (not a closed
    abundance composition), in MICRO_COLUMNS order.
    """

    def __init__(self, group):
        if group not in GROUPS:
            raise ValueError(f'group must be one of {GROUPS}')
        self.group = group
        self._fit_started = False
        self._fitted = False

    def fit(self, people, meals, host_oof, followup, context_label):
        if self._fit_started:
            raise ValueError('Use a fresh JointLongitudinalCorrection for every fit')
        if not isinstance(context_label, str) or not context_label:
            raise ValueError('Nonempty context_label required')
        self._fit_started = True
        families = _family_labels(people)
        positions, recipes = _event_positions(people, meals, training=True)
        host = _host_array(host_oof, meals)
        observed = meals.target.to_numpy(dtype=np.float64)
        if not np.isfinite(observed).all():
            raise ValueError('Finite current-training response labels required')
        counts = np.bincount(positions, minlength=len(people))
        weights = 1. / (len(people) * counts[positions])
        residual = observed - host
        self.response_scale = max(float(np.sqrt(np.sum(weights * residual ** 2))), SCALE_FLOOR)
        self.scale = self.response_scale
        targets = residual / self.response_scale
        self.context_label = context_label
        self.fit_ids = _labels(people.index)
        self.fit_events = _labels(meals.meal_event_id)
        self.fit_families = sorted(families.unique().tolist())
        self.auxiliary_weight = AUXILIARY_WEIGHT if self.group.startswith('joint_') else 0.
        self.scaler = None
        self.auxiliary_scaler = None
        self.pair_ids = []
        self.donor_ids = []
        self.C = np.zeros((N_FEATURES, N_RECIPES), dtype=np.float64)
        self.D = np.zeros((N_FEATURES, N_FEATURES), dtype=np.float64)
        self.a = np.zeros(N_RECIPES, dtype=np.float64)
        self.b = np.zeros(N_FEATURES, dtype=np.float64)
        self.audit = {
            'group': self.group, 'context_label': context_label, 'status': 'preparing',
            'fit_ids': self.fit_ids, 'fit_events': self.fit_events, 'fit_families': self.fit_families,
            'response_people': len(people), 'response_events': len(meals),
            'features': list(MICRO_COLUMNS) if self.group != 'host_calibration' else [],
            'recipes': list(MEALS), 'response_scale': self.response_scale,
            'response_scale_rule': 'equal-person current-T host-OOF residual RMS, floor 1e-8',
            'host_oof_provenance': 'caller must verify family crossfits within this exact T',
            'host_oof_sha256': _array_hash(host), 'response_target_sha256': _array_hash(observed),
            'response_weights_sha256': _array_hash(weights),
            'equal_person_weight_min': float(np.bincount(positions, weights=weights).min()),
            'equal_person_weight_max': float(np.bincount(positions, weights=weights).max()),
            'eta': ETA, 'l2_includes_intercepts': True, 'lambda': self.auxiliary_weight,
            'mu_ratio': MU_RATIO, 'mu_rule': '0.1*max(operator_norm(response_gradient_at_zero),1e-8)',
            'pair_ids': [], 'donor_ids': [], 'pair_families': [], 'mapping_strata': [],
            'auxiliary_values_read': bool(self.auxiliary_weight),
        }
        if self.group == 'host_calibration':
            mass = np.bincount(recipes, weights=weights, minlength=N_RECIPES)
            first = np.bincount(recipes, weights=weights * targets, minlength=N_RECIPES)
            self.a = 2. * first / (2. * mass + ETA)
            gradient = (2. * mass + ETA) * self.a - 2. * first
            response_loss = float(np.sum(weights * (targets - self.a[recipes]) ** 2))
            penalty = float(ETA * np.sum(self.a ** 2) / 2.)
            objective = response_loss + penalty
            gap = float(np.sum(gradient ** 2) / (2 * ETA))
            self.mu = 0.
            self.problem_ = None
            self.optimization = {
                'optimizer': 'closed_form_recipe_ridge', 'iterations': 0,
                'maximum_steps': 0, 'lipschitz': float((2 * mass + ETA).max()),
                'initial_objective': float(np.sum(weights * targets ** 2)),
                'objective': objective, 'gap_bound': gap,
                'gap_threshold': GAP_TOLERANCE * max(1., abs(objective)),
                'subgradient_norm': float(np.linalg.norm(gradient)),
                'proximal_optimality_residual': 0., 'converged': True,
            }
            self.loss_diagnostics = {
                'response_loss': response_loss, 'auxiliary_loss': 0., 'weighted_auxiliary_loss': 0.,
                'l2_penalty': penalty, 'nuclear_penalty': 0., 'objective': objective,
                'singular_values': [0.] * N_FEATURES,
            }
        else:
            raw = microbial_clr(people)
            self.scaler = FrozenColumnScaler.from_array(raw)
            x = self.scaler.transform(raw)
            pair_positions = np.empty(0, dtype=np.int64)
            auxiliary = np.empty((0, N_FEATURES), dtype=np.float64)
            self.audit['baseline_clr_sha256'] = _array_hash(raw)
            self.audit['baseline_scaler'] = self.scaler.specification()
            if self.auxiliary_weight:
                _validate_index(followup, 'followup')
                pair_index = people.index[people.index.isin(followup.index)]
                if not len(pair_index):
                    raise ValueError('Joint objective requires current-training paired day14 rows')
                # No value from any other followup row is accessed.
                real_target = microbial_clr(followup.loc[pair_index])
                self.auxiliary_scaler = FrozenColumnScaler.from_array(real_target)
                mode = 'real' if self.group == 'joint_real' else 'shuffle'
                donors, strata = _mapping(pair_index, families, mode, SEED, context_label, 'joint_full')
                donor_positions = pair_index.get_indexer(donors)
                auxiliary = self.auxiliary_scaler.transform(real_target)[donor_positions]
                pair_positions = people.index.get_indexer(pair_index)
                self.pair_ids = _labels(pair_index)
                self.donor_ids = _labels(donors)
                self.audit.update({
                    'pair_ids': self.pair_ids, 'donor_ids': self.donor_ids,
                    'pair_families': sorted(families.loc[pair_index].unique().tolist()),
                    'paired_people': len(pair_index), 'mapping_strata': strata,
                    'mapping_seed': SEED, 'mapping_fit_label': 'joint_full',
                    'auxiliary_scaler': self.auxiliary_scaler.specification(),
                    'real_auxiliary_clr_sha256': _array_hash(real_target),
                    'mapped_standardized_auxiliary_sha256': _array_hash(auxiliary),
                })
            self.problem_ = _build_objective(x, positions, recipes, weights, targets,
                                             pair_positions, auxiliary, self.auxiliary_weight)
            self.mu = self.problem_.mu
            try:
                coefficients, self.optimization = _solve(self.problem_)
            except OptimizationConvergenceError as error:
                self.audit.update({'status': 'optimization_incomplete', 'mu': self.mu,
                                   'optimization': error.diagnostics})
                raise
            self.C = coefficients[:N_FEATURES, :N_RECIPES].copy()
            self.a = coefficients[-1, :N_RECIPES].copy()
            if self.auxiliary_weight:
                self.D = coefficients[:N_FEATURES, N_RECIPES:].copy()
                self.b = coefficients[-1, N_RECIPES:].copy()
            self.loss_diagnostics = self.problem_.diagnostics(coefficients)
            self.audit['response_gradient_operator_norm'] = float(
                np.linalg.svd(self.problem_.linear[:N_FEATURES, :N_RECIPES], compute_uv=False)[0])
            self.audit['training_standardized_micro_mean_maxabs'] = float(abs(x.mean(axis=0)).max())
            self.audit['fixed_recipe_training_micro_mean_maxabs'] = float(
                abs(self.response_scale * self._stable_product(x, self.C).mean(axis=0)).max())
        self.coeff = {'C': self.C, 'D': self.D, 'a': self.a, 'b': self.b}
        self.scales = {'response': self.response_scale,
                       'baseline': None if self.scaler is None else self.scaler.specification(),
                       'auxiliary': None if self.auxiliary_scaler is None else self.auxiliary_scaler.specification()}
        singular = np.linalg.svd(np.column_stack([self.C, self.D]), compute_uv=False)
        rank_threshold = max(18, 25) * np.finfo(np.float64).eps * (float(singular[0]) if len(singular) else 0.)
        self.audit.update({
            'status': 'converged', 'mu': self.mu, 'scales': self.scales,
            'optimization': self.optimization, 'losses': self.loss_diagnostics,
            'gap_bound': self.optimization['gap_bound'], 'iterations': self.optimization['iterations'],
            'objective': self.loss_diagnostics['objective'], 'lipschitz': self.optimization['lipschitz'],
            'singular_values': singular.tolist(),
            'response_singular_values': np.linalg.svd(self.C, compute_uv=False).tolist(),
            'numerical_rank': int(np.sum(singular > rank_threshold)), 'rank_threshold': rank_threshold,
            'coefficients': {name: value.tolist() for name, value in self.coeff.items()},
            'inference': 'stable column accumulation; baseline-only, no ID-dependent prediction path',
            'reference': 'zero standardized microbial vector; current-T empirical CLR mean',
        })
        self._fitted = True
        return self

    def _require_fitted(self):
        if not self._fitted:
            raise ValueError('JointLongitudinalCorrection must be fitted before prediction')

    @staticmethod
    def _stable_product(values, coefficients):
        result = np.zeros((len(values), coefficients.shape[1]), dtype=np.float64)
        for column in range(N_FEATURES):
            result += values[:, column, None] * coefficients[column][None, :]
        return result

    def components(self, people, meals):
        self._require_fitted()
        positions, recipes = _event_positions(people, meals)
        calibration = self.response_scale * self.a[recipes]
        micro = np.zeros(len(meals), dtype=np.float64)
        if self.group != 'host_calibration':
            x = self.scaler.transform(microbial_clr(people))
            panel = self._stable_product(x, self.C)
            micro = self.response_scale * panel[positions, recipes]
        correction = calibration + micro
        if not np.isfinite(correction).all():
            raise ValueError('Nonfinite joint correction')
        return {'recipe_calibration': calibration, 'micro': micro, 'correction': correction}

    def correction(self, people, meals):
        return self.components(people, meals)['correction']

    def empirical_reference_correction(self, people, meals):
        self._require_fitted()
        _, recipes = _event_positions(people, meals)
        return self.response_scale * self.a[recipes]

    def predict_auxiliary(self, people):
        self._require_fitted()
        if not self.auxiliary_weight:
            raise ValueError('Only joint groups have a trained auxiliary decoder')
        x = self.scaler.transform(microbial_clr(people))
        standardized = self._stable_product(x, self.D) + self.b[None, :]
        result = self.auxiliary_scaler.inverse_transform(standardized)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite auxiliary prediction')
        return result

    def objective_diagnostics(self):
        self._require_fitted()
        if self.problem_ is None:
            return dict(self.loss_diagnostics)
        coefficients = np.vstack([np.column_stack([self.C, self.D]) if self.auxiliary_weight else self.C,
                                  np.r_[self.a, self.b] if self.auxiliary_weight else self.a])
        return self.problem_.diagnostics(coefficients)
