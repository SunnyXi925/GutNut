"""Development-fitted nutrient-direction projections of microbial composition.

These features are association-based candidate representations, not measured
metabolic capacities, causal responses, or clinical benefit estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class NutrientCapacityProjector(BaseEstimator, TransformerMixin):
    """Project all-input CLR onto centered nutrient directions over covered genera.

    B rows are nutrients and columns are genera. Centering is performed over
    genera shared by B and the fitted composition, then directions are embedded
    with zero weights for other genera. No health labels or classifier enter.
    """
    def __init__(self, directions, pseudocount=1e-6, zero_tolerance=1e-12):
        self.directions = directions
        self.pseudocount = pseudocount
        self.zero_tolerance = zero_tolerance

    @staticmethod
    def _check_frame(x):
        if not isinstance(x, pd.DataFrame) or x.columns.has_duplicates or x.empty:
            raise ValueError('Expected nonempty DataFrame with unique genus columns')
        a = x.to_numpy(dtype=float)
        if not np.isfinite(a).all() or (a < 0).any() or (a.sum(axis=1) <= 0).any():
            raise ValueError('Expected finite nonnegative nonempty compositions')
        return a

    def _raw_array(self, a):
        logs = np.log(a/a.sum(axis=1, keepdims=True)+self.pseudocount)
        clr = logs-logs.mean(axis=1,keepdims=True)
        result = clr@self.projection_.T
        result[:, ~self.identifiable_] = 0.0
        return result

    def fit(self, x, y=None):
        if y is not None:
            raise ValueError('Health labels are forbidden for capacity projection fitting')
        a = self._check_frame(x)
        b = self.directions
        if not isinstance(b,pd.DataFrame) or b.empty or b.columns.has_duplicates or b.index.has_duplicates:
            raise ValueError('Directions must be nutrient x genus DataFrame with unique labels')
        if not np.isfinite(b.to_numpy(dtype=float)).all():
            raise ValueError('Directions contain nonfinite coefficients')
        if self.pseudocount <= 0 or self.zero_tolerance <= 0:
            raise ValueError('Expected positive pseudocount and zero tolerance')
        self.feature_names_in_ = np.asarray(x.columns, dtype=object)
        self.n_features_in_ = len(x.columns)
        self.nutrients_ = list(b.index)
        self.fit_sample_ids_ = list(x.index)
        self.fit_n_samples_ = len(x)
        covered = [g for g in b.columns if g in x.columns]
        if not covered:
            raise ValueError('No shared genus between directions and composition')
        matrix = b.loc[:,covered].to_numpy(dtype=float)
        centered = matrix-matrix.mean(axis=1,keepdims=True)
        norms = np.linalg.norm(centered,axis=1)
        self.identifiable_ = norms > self.zero_tolerance
        centered[self.identifiable_] /= norms[self.identifiable_,None]
        centered[~self.identifiable_] = 0
        self.projection_ = np.zeros((len(b),len(x.columns)))
        self.projection_[:,x.columns.get_indexer(covered)] = centered
        raw = self._raw_array(a)
        self.center_ = np.median(raw,axis=0)
        mad = 1.4826*np.median(np.abs(raw-self.center_),axis=0)
        std = np.std(raw,axis=0)
        self.scale_ = np.where(mad>self.zero_tolerance,mad,np.where(std>self.zero_tolerance,std,1.0))
        scale_rule = np.where(mad>self.zero_tolerance,'1.4826*MAD',np.where(std>self.zero_tolerance,'standard_deviation_fallback','constant_scale_one'))
        total_mass = np.abs(b.to_numpy(dtype=float)).sum(axis=1)
        covered_mass = np.abs(matrix).sum(axis=1)
        self.diagnostics_ = pd.DataFrame({
            'nutrient':self.nutrients_, 'identifiable_direction':self.identifiable_,
            'status':np.where(self.identifiable_,'direction_identifiable','not_identifiable'),
            'source_direction_all_zero':np.max(np.abs(b.to_numpy(dtype=float)),axis=1)<=self.zero_tolerance,
            'covered_centered_direction_l2_before_normalization':norms,
            'covered_absolute_B_mass_fraction':np.divide(covered_mass,total_mass,out=np.zeros_like(total_mass),where=total_mass>0),
            'development_median':self.center_, 'development_scale':self.scale_, 'scale_rule':scale_rule,
            'development_raw_std':std,
        }).set_index('nutrient')
        self.coverage_ = {'input_genera':len(x.columns),'B_genera':len(b.columns),'covered_B_genera':len(covered),
                          'B_genus_coverage_fraction':len(covered)/len(b.columns),
                          'missing_B_genera':sorted(set(b.columns)-set(x.columns)),
                          'direction_centering_universe':'shared B and input genera; all other input weights zero',
                          'fit_scope':'development only; no labels',
                          'scope':'candidate compositional representation, not measured metabolic capacity'}
        return self

    def transform_raw(self,x):
        check_is_fitted(self, 'projection_')
        a = self._check_frame(x)
        if list(x.columns) != list(self.feature_names_in_):
            raise ValueError('Genus names/order changed; explicitly align to feature_names_in_')
        return pd.DataFrame(self._raw_array(a),index=x.index,columns=self.nutrients_)

    def transform(self,x):
        raw = self.transform_raw(x)
        result = (raw-self.center_)/self.scale_
        result.loc[:,~self.identifiable_] = 0.0
        return result

    def get_feature_names_out(self,input_features=None):
        check_is_fitted(self,'projection_')
        return np.asarray(self.nutrients_,dtype=object)


def signal_rank_report(signal):
    """SVD rank and correlation summary on variable columns; no outcome labels."""
    if not isinstance(signal,pd.DataFrame) or not np.isfinite(signal.to_numpy()).all():
        raise ValueError('Expected finite signal DataFrame')
    variable = signal.std(ddof=0)>1e-12
    x = signal.loc[:,variable].to_numpy(dtype=float, copy=True)
    x -= x.mean(axis=0)
    if x.shape[1]:
        x /= x.std(axis=0)
        values = np.linalg.svd(x,compute_uv=False)
        mass = values**2
        mass /= mass.sum()
        entropy_rank = float(np.exp(-(mass[mass>0]*np.log(mass[mass>0])).sum()))
        rank = int(np.linalg.matrix_rank(x))
        corr = np.corrcoef(x,rowvar=False) if x.shape[1]>1 else np.ones((1,1))
        off = np.abs(corr[np.triu_indices(len(corr),1)])
    else:
        values,off=[],[]
        rank,entropy_rank=0,0.0
    return {'nutrients':list(signal.columns),'n_columns':signal.shape[1],
            'variable_columns':list(signal.columns[variable]),'n_variable':int(variable.sum()),
            'zero_or_constant_columns':list(signal.columns[~variable]),
            'matrix_rank':rank,'entropy_effective_rank':entropy_rank,
            'singular_values_standardized':list(map(float,values)),
            'median_absolute_pairwise_correlation':float(np.median(off)) if len(off) else None,
            'max_absolute_pairwise_correlation':float(np.max(off)) if len(off) else None}
