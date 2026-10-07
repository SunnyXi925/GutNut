"""Development-only empirical consensus calibration and native-stage diagnostics.

These are structural score transformations, not estimators of metabolic benefit.
The original native scorer is unchanged. All vitamin/mineral top-five members
remain fixed from baseline for every stage and candidate in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from .native_scoring import ATTRIBUTES, DOMAINS, DOMAIN_WEIGHTS
from .fcs2_attribute_rules import FCS2_RULES


@dataclass(frozen=True)
class NativeStageSpec:
    fraction: float = .2
    temperature: float = 2.
    response: str = 'tanh'
    attribute_clip: bool = True
    native_clip: bool = True
    cap: float | None = None

    def __post_init__(self):
        if not np.isfinite(self.fraction) or not 0 <= self.fraction <= 1:
            raise ValueError('fraction must be finite in [0,1]')
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        if self.response not in ('linear', 'tanh'):
            raise ValueError('response must be linear or tanh')
        if self.cap is not None and (not np.isfinite(self.cap) or self.cap < 0):
            raise ValueError('cap must be nonnegative or None')


class NativeStageEngine:
    """Fixed-membership scorer with independently switchable compression stages."""

    def __init__(self, scorer):
        self.scorer = scorer
        self.low = np.array([min(FCS2_RULES[a].low_points, FCS2_RULES[a].high_points) for a in ATTRIBUTES])
        self.high = np.array([max(FCS2_RULES[a].low_points, FCS2_RULES[a].high_points) for a in ATTRIBUTES])
        coefficients = np.zeros_like(scorer.weights)
        for domain in DOMAINS:
            ix = np.array([i for i, a in enumerate(ATTRIBUTES) if FCS2_RULES[a].domain == domain])
            p, w = scorer.points[:, ix], scorer.weights[:, ix]
            selected = np.isfinite(p)
            if domain in ('vitamins', 'minerals'):
                order = np.argsort(-np.where(np.isfinite(p), abs(p), -np.inf), axis=1, kind='stable')[:, :5]
                mask = np.zeros(p.shape, dtype=bool)
                np.put_along_axis(mask, order, True, axis=1)
                selected &= mask
            kept = np.where(selected, w, 0.)
            denominator = kept.sum(axis=1, keepdims=True)
            if (denominator <= 0).any():
                raise ValueError('incomplete fixed domain')
            coefficients[:, ix] = DOMAIN_WEIGHTS[domain] * kept / denominator
        self.coefficients = coefficients * (99./47.1)
        self.food_ids = tuple(scorer.food_ids)

    def evaluate(self, signals, spec=NativeStageSpec(), *, food_slice=None, direction=None):
        z = np.asarray(signals, dtype=float)
        if z.ndim != 2 or z.shape[1] != self.scorer.design.shape[1] or not np.isfinite(z).all():
            raise ValueError('finite people by ordered nutrient matrix required')
        sl = slice(None) if food_slice is None else food_slice
        design, point, coefficient = self.scorer.design[sl], self.scorer.points[sl], self.coefficients[sl]
        response = np.einsum('in,fna->ifa', z, design, optimize=True)
        scaled = response / spec.temperature
        if spec.response == 'tanh':
            transformed = np.tanh(scaled)
            attenuation = 1.-transformed**2
        else:
            transformed, attenuation = scaled, np.ones_like(scaled)
        raw = point[None] + spec.fraction * (self.high-self.low) * transformed
        personalized = np.clip(raw, self.low, self.high) if spec.attribute_clip else raw
        point_change = np.where(np.isfinite(point)[None], personalized-point[None], 0.)
        delta = np.einsum('ifa,fa->if', point_change, coefficient, optimize=True)
        baseline = self.scorer.baseline[sl]
        unbounded = baseline[None] + delta
        if spec.native_clip:
            delta = np.clip(unbounded, 1., 100.)-baseline[None]
        precap = delta.copy()
        if spec.cap is not None:
            delta = np.clip(delta, -spec.cap, spec.cap)
        contributing = (np.any(design != 0, axis=1) & (coefficient != 0))[None]
        count = int(len(z) * contributing.sum())
        clipped = np.isfinite(raw) & (abs(raw-personalized) > 1e-10)
        diag = {
            'contributing_attribute_cells': count,
            'tanh_derivative_below_0_1_n': int(np.count_nonzero(contributing & (attenuation < .1))),
            'tanh_abs_above_0_99_n': int(np.count_nonzero(contributing & (abs(transformed) > .99))) if spec.response=='tanh' else 0,
            'tanh_derivative_ratio_sum': float(np.where(contributing, attenuation, 0.).sum()),
            'contributing_attribute_clip_n': int(np.count_nonzero(contributing & clipped)),
            'all_attribute_clip_n': int(np.count_nonzero(clipped)),
            'native_clip_n': int(np.count_nonzero(abs(unbounded-np.clip(unbounded,1,100))>1e-10)) if spec.native_clip else 0,
            'cap_clip_n': int(np.count_nonzero(abs(precap-delta)>1e-10)),
        }
        derivative = None
        if direction is not None:
            direction = np.asarray(direction, dtype=float)
            if direction.shape != (z.shape[1],) or not np.isfinite(direction).all():
                raise ValueError('direction must match nutrient order')
            dr = np.einsum('n,fna->fa', direction, design, optimize=True)
            da = spec.fraction * (self.high-self.low)/spec.temperature * attenuation * dr[None]
            if spec.attribute_clip:
                da = np.where((raw > self.low) & (raw < self.high), da, 0.)
            derivative = np.einsum('ifa,fa->if', da, coefficient, optimize=True)
            if spec.native_clip:
                derivative *= (unbounded > 1.) & (unbounded < 100.)
            if spec.cap is not None:
                derivative *= (precap > -spec.cap) & (precap < spec.cap)
        return delta, derivative, diag


class DevelopmentConsensusCalibrator:
    """Fit a frozen foodwise development reference; never recenter test people.

    additive: baseline + delta - development_mean_delta (not range bounded).
    bounded_logit: 1 + 99*sigmoid(offset + (delta-dev_mean)/temperature),
    with offset fitted so each development food mean equals its baseline.
    Exact endpoints 1 or 100 are necessarily constant under the bounded map.
    Neither empirical consensus rule promises zero-signal baseline identity.
    """

    def __init__(self, method='additive', temperature=25.):
        if method not in ('additive', 'bounded_logit'):
            raise ValueError('unknown calibration method')
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        self.method, self.temperature = method, float(temperature)

    def fit(self, development_delta, baseline, food_ids, *, development_ids):
        x, b = np.asarray(development_delta, float), np.asarray(baseline, float)
        ids = tuple(food_ids)
        if x.ndim != 2 or b.shape != (x.shape[1],) or len(ids) != len(b) or len(set(ids)) != len(ids):
            raise ValueError('unique ordered foods must match delta and baseline')
        if not np.isfinite(x).all() or not np.isfinite(b).all() or ((b<1)|(b>100)).any():
            raise ValueError('invalid calibration values')
        if len(development_ids) != len(x) or len(set(development_ids)) != len(x):
            raise ValueError('unique development reference IDs required')
        self.food_ids, self.baseline = ids, b.copy()
        self.development_ids = tuple(development_ids)
        self.reference_mean = x.mean(axis=0)
        self.endpoint = (b == 1.) | (b == 100.)
        self.offset = np.zeros(len(b))
        if self.method == 'bounded_logit':
            for j in np.flatnonzero(~self.endpoint):
                values = np.ascontiguousarray((x[:,j]-self.reference_mean[j])/self.temperature)
                target = (b[j]-1)/99.
                fun = lambda offset: float(expit(offset+values).mean()-target)
                self.offset[j] = brentq(fun, -float(values.max())-50, -float(values.min())+50, xtol=1e-13)
        self.development_reference_mean_max_error = float(abs(self.transform(x, ids).mean(axis=0)-b).max())
        if self.development_reference_mean_max_error > 1e-9:
            raise ArithmeticError('development consensus identity failed')
        return self

    def _check(self, delta, food_ids):
        if tuple(food_ids) != self.food_ids:
            raise ValueError('food IDs/order differ from fitted development reference')
        x = np.asarray(delta, float)
        if x.ndim != 2 or x.shape[1] != len(self.food_ids) or not np.isfinite(x).all():
            raise ValueError('finite people by fitted food matrix required')
        return x

    def transform(self, delta, food_ids):
        x = self._check(delta, food_ids)
        centered = x-self.reference_mean[None]
        if self.method == 'additive':
            return self.baseline[None]+centered
        score = 1.+99.*expit(self.offset[None]+centered/self.temperature)
        score[:,self.endpoint] = self.baseline[self.endpoint]
        return score

    def derivative(self, delta, food_ids):
        x = self._check(delta, food_ids)
        if self.method == 'additive':
            return np.ones_like(x)
        q = expit(self.offset[None]+(x-self.reference_mean[None])/self.temperature)
        derivative = 99./self.temperature*q*(1.-q)
        derivative[:,self.endpoint] = 0.
        return derivative
