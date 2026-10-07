"""Nonnegative inner-OOF stacking and decomposition of a frozen response model."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .schema import MEALS, MICRO_COLUMNS

def recipe_weights(meal_ids):
    """Weights whose squared-error average is exactly the registered A squared."""
    ids = pd.Series(np.asarray(meal_ids, dtype=str))
    if set(ids) != set(MEALS) or ids.empty:
        raise ValueError('All seven registered recipes are required')
    counts = ids.value_counts()
    return 1. / (len(MEALS) * ids.map(counts).to_numpy(float))



def fit_nonnegative_scalar(target, baseline, correction, meal_ids):
    """Exact one-dimensional weighted NNLS; no upper bound or small-energy floor."""
    y, g, c = [np.asarray(a, dtype=np.float64) for a in (target, baseline, correction)]
    if y.ndim != 1 or g.shape != y.shape or c.shape != y.shape or len(meal_ids) != len(y):
        raise ValueError('Scalar-calibration arrays must be aligned one-dimensional vectors')
    if not all(np.isfinite(a).all() for a in (y, g, c)):
        raise ValueError('Nonfinite calibration input')
    w = recipe_weights(meal_ids)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        residual = y-g
        numerator = float(np.sum(w*c*residual))
        denominator = float(np.sum(w*c*c))
        if not np.isfinite([numerator, denominator]).all() or denominator < 0.:
            raise ValueError('Nonfinite or negative NNLS sufficient statistic')
        free_alpha = numerator/denominator if denominator != 0. else 0.
        alpha = max(0., free_alpha)
        if not np.isfinite([free_alpha, alpha]).all():
            raise ValueError('Nonfinite stacking coefficient')
        losses = {name: float(np.sum(w*(residual-a*c)**2)) for name, a in
                  [('zero', 0.), ('unit', 1.), ('free', free_alpha), ('nonnegative', alpha)]}
        gradient = 2.*(denominator*alpha-numerator)
        free_gradient = 2.*(denominator*free_alpha-numerator)
        violation = abs(gradient) if alpha > 0. else max(0., -gradient)
        complementarity = abs(alpha*gradient)
        tolerance = 1e-10*max(1., abs(numerator), abs(denominator*alpha))
        if not np.isfinite(list(losses.values())+[gradient, free_gradient, complementarity]).all():
            raise ValueError('Nonfinite scalar calibration diagnostics')
        if violation > tolerance:
            raise ArithmeticError('Scalar NNLS KKT condition failed')
        if losses['nonnegative'] > min(losses['zero'], losses['unit']) + 1e-10*max(1., *losses.values()):
            raise ArithmeticError('NNLS loss exceeds a feasible zero/unit coefficient')
    return {'alpha': alpha, 'free_alpha': free_alpha, 'numerator': numerator,
            'denominator': denominator, 'exact_zero_denominator': denominator == 0.,
            'correction_energy': denominator, 'correction_RMS': float(np.sqrt(denominator)),
            'inner_mse': losses, 'inner_rmse': {k: float(np.sqrt(v)) for k, v in losses.items()},
            'gradient': gradient, 'free_gradient': free_gradient, 'kkt_violation': violation,
            'complementarity': complementarity, 'kkt_tolerance': tolerance,
            'weights': '1/(7 * current-training recipe event count)',
            'scope': 'Scalar fitted on pooled inner-family-OOF predictions; calibration loss is not an independent test'}



class JointStackingModel:
    """Use the full frozen correction, regardless of the old binary selector."""
    def __init__(self, host, correction, alpha):
        if not np.isfinite(alpha) or alpha < 0.:
            raise ValueError('A finite nonnegative alpha is required')
        self.host, self.correction_model, self.alpha = host, correction, float(alpha)
        self.group, self.enabled = correction.group, self.alpha > 0.
        self.fit_ids = list(host.fit_ids)
        self.fit_events = list(host.fit_events)
        self.fit_families = list(host.fit_families)

    def predict_components(self, people, meals):
        host = np.asarray(self.host.predict(people, meals), dtype=np.float64)
        raw = self.correction_model.components(people, meals)
        # The frozen correction already includes response_scale in outcome units.
        with np.errstate(over='raise', invalid='raise'):
            total = self.alpha*np.asarray(raw['correction'], dtype=np.float64)
            calibration = self.alpha*np.asarray(raw['recipe_calibration'], dtype=np.float64)
            micro = self.alpha*np.asarray(raw['micro'], dtype=np.float64)
            result = pd.DataFrame({'host': host, 'recipe_calibration': calibration,
                'micro': micro, 'reference': host+calibration, 'prediction': host+total}, index=meals.index)
        if result.shape != (len(meals), 5) or not np.isfinite(result.to_numpy()).all():
            raise ValueError('Nonfinite or misaligned stacked prediction')
        return result

    def predict(self, people, meals):
        return self.predict_components(people, meals).prediction.to_numpy()

    def predict_reference(self, people, meals):
        return self.predict_components(people, meals).reference.to_numpy()
