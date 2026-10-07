"""Separate a fitted response into an empirical-reference mean and deviation.

The reference consists of complete observed microbial panels from the current
model's training participants. No outcome is used, no model is fitted, and no
bound or amplitude compression is imposed. The decomposition applies at a
fixed host/meal context; it does not assert that marginal panel replacement is
a biological intervention or that its mean equals the observed joint mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class EmpiricalReferenceResponse:
    def __init__(self, model, training_reference, microbial_columns, batch_rows=65536):
        self.model = model
        self.microbial_columns = tuple(microbial_columns)
        if not self.microbial_columns or len(set(self.microbial_columns)) != len(self.microbial_columns):
            raise ValueError('Distinct microbial column names required')
        if not training_reference.index.is_unique or not len(training_reference):
            raise ValueError('Nonempty reference with one panel per training participant required')
        if set(training_reference.index) != set(model.fit_ids):
            raise ValueError('Reference must contain exactly the fitted training participants')
        self.reference_ids = tuple(training_reference.index)
        self.reference = training_reference.loc[:, self.microbial_columns].to_numpy(float).copy()
        if not np.isfinite(self.reference).all():
            raise ValueError('Finite observed microbial reference required')
        if batch_rows < len(self.reference):
            raise ValueError('Batch must accommodate at least one full reference panel set')
        self.batch_rows = int(batch_rows)

    def reference_predictions(self, fixed_context):
        """Evaluate all training panels at one fixed context, retaining panel order."""
        if len(fixed_context) != 1:
            raise ValueError('Exactly one fixed host/meal context required')
        frame = fixed_context.iloc[np.zeros(len(self.reference), dtype=int)].copy()
        frame.loc[:, self.microbial_columns] = self.reference
        prediction = np.asarray(self.model.predict(frame), dtype=float)
        if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
            raise ValueError('Finite scalar response per reference panel required')
        return prediction

    def reference_mean(self, frame):
        """Compute E_reference[f(h, M)] for each supplied fixed context h."""
        n_ref = len(self.reference)
        n_query = max(1, self.batch_rows // n_ref)
        result = np.empty(len(frame), dtype=float)
        for begin in range(0, len(frame), n_query):
            stop = min(begin + n_query, len(frame))
            query = frame.iloc[begin:stop]
            crossed = query.iloc[np.repeat(np.arange(len(query)), n_ref)].copy()
            crossed.loc[:, self.microbial_columns] = np.tile(self.reference, (len(query), 1))
            prediction = np.asarray(self.model.predict(crossed), dtype=float)
            if prediction.shape != (len(crossed),) or not np.isfinite(prediction).all():
                raise ValueError('Finite scalar response per crossed panel required')
            result[begin:stop] = prediction.reshape(len(query), n_ref).mean(axis=1)
        return result

    def components(self, frame):
        prediction = np.asarray(self.model.predict(frame), dtype=float)
        if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
            raise ValueError('Finite scalar response per input required')
        reference = self.reference_mean(frame)
        return pd.DataFrame({'prediction': prediction,
                             'reference_mean_response': reference,
                             'microbial_deviation': prediction - reference}, index=frame.index)

    def specification(self):
        return {'version': 'empirical_reference_response_v1', 'reference_people': len(self.reference),
                'reference': 'equal-weight complete observed training microbial panels',
                'centering': 'fixed host and meal/context; marginal empirical microbial reference',
                'output_cap': None, 'amplitude_compression': None, 'new_fitting': False,
                'scope': 'response units of underlying fitted model; no food-score or causal claim'}
