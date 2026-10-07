"""Versioned fixed-membership native scorer without a final deviation cap.

The optional empirical reference maps are research alternatives. None of these
maps estimates metabolic validity; prediction of measured responses is evaluated
in the separate family-grouped response models.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import numpy as np
import pandas as pd

from .consensus_calibration import NativeStageEngine, NativeStageSpec, DevelopmentConsensusCalibrator
from .native_scoring import NUTRIENTS


class UncappedConsensusScorer:
    version = 'fixed_five_domain_no_final_deviation_cap_v4'

    @staticmethod
    def native_signature(native_scorer):
        digest = hashlib.sha256(json.dumps(list(map(str, native_scorer.food_ids))).encode())
        for name in ('baseline', 'points', 'weights', 'design'):
            a = np.ascontiguousarray(getattr(native_scorer, name), dtype=np.float64)
            digest.update(name.encode())
            digest.update(str(a.shape).encode())
            digest.update(a.tobytes())
        return digest.hexdigest()

    def __init__(self, native_scorer, *, calibrator=None, fraction=.2, temperature=2.):
        self.engine = NativeStageEngine(native_scorer)
        self.calibrator = calibrator
        self.spec = NativeStageSpec(fraction=fraction, temperature=temperature,
                                   attribute_clip=True, native_clip=calibrator is None, cap=None)
        if calibrator is not None:
            if tuple(calibrator.food_ids) != self.engine.food_ids:
                raise ValueError('Reference map and native food order differ')
            if not np.array_equal(calibrator.baseline, native_scorer.baseline):
                raise ValueError('Reference map and native baselines differ')
            if getattr(calibrator, 'native_stage_spec', None) != asdict(self.spec):
                raise ValueError('Reference map was not fitted to this native stage specification')
            if getattr(calibrator, 'native_signature', None) != self.native_signature(native_scorer):
                raise ValueError('Reference map was not fitted to this native points/weights/design')
        self.nutrient_order = tuple(NUTRIENTS)
        self.food_ids = self.engine.food_ids

    @classmethod
    def fit_reference(cls, native_scorer, development_signals, *, method='additive',
                      output_temperature=25., fraction=.2, temperature=2., chunk_foods=24):
        """Fit the empirical map from the same stage, native arrays and signals.

        The signal preprocessing contract remains the caller's responsibility;
        the exact finite development DataFrame is recorded by IDs and hash.
        """
        uncalibrated = cls(native_scorer, fraction=fraction, temperature=temperature)
        uncalibrated.spec = NativeStageSpec(fraction=fraction, temperature=temperature,
                                           native_clip=False, cap=None)
        raw_scores = uncalibrated.predict(development_signals, chunk_foods=chunk_foods)
        delta = raw_scores.to_numpy()-native_scorer.baseline[None]
        reference = DevelopmentConsensusCalibrator(method, output_temperature).fit(
            delta, native_scorer.baseline, native_scorer.food_ids,
            development_ids=development_signals.index.tolist())
        reference.native_stage_spec = asdict(uncalibrated.spec)
        reference.native_signature = cls.native_signature(native_scorer)
        reference.development_signal_sha256 = hashlib.sha256(
            pd.util.hash_pandas_object(development_signals, index=True).values.tobytes()).hexdigest()
        return cls(native_scorer, calibrator=reference, fraction=fraction, temperature=temperature)

    def predict(self, signals: pd.DataFrame, *, chunk_foods=24) -> pd.DataFrame:
        if not isinstance(signals, pd.DataFrame) or tuple(signals.columns) != self.nutrient_order:
            raise ValueError('DataFrame with the exact declared nutrient-column order required')
        if not signals.index.is_unique:
            raise ValueError('Unique profile IDs required')
        if not isinstance(chunk_foods, int) or chunk_foods < 1:
            raise ValueError('Positive integer food chunk size required')
        z = signals.to_numpy(float)
        if not np.isfinite(z).all():
            raise ValueError('Finite normalized nutrient signals required')
        delta = np.empty((len(z), len(self.food_ids)))
        for first in range(0, len(self.food_ids), chunk_foods):
            sl = slice(first, first+chunk_foods)
            delta[:, sl], _, _ = self.engine.evaluate(z, self.spec, food_slice=sl)
        if self.calibrator is None:
            score = self.engine.scorer.baseline[None] + delta
        else:
            score = self.calibrator.transform(delta, self.food_ids)
        return pd.DataFrame(score, index=signals.index.copy(), columns=self.food_ids)

    def specification(self):
        return {
            'version': self.version, 'native_stage': asdict(self.spec),
            'top_five_policy': 'baseline_fixed',
            'nutrient_order': list(self.nutrient_order), 'food_count': len(self.food_ids),
            'output_map': 'native_1_100' if self.calibrator is None else self.calibrator.method,
            'output_temperature': None if self.calibrator is None else self.calibrator.temperature,
            'exact_zero_signal_baseline': self.calibrator is None,
            'exact_development_food_mean': self.calibrator is not None,
            'range_1_100': self.calibrator is None or self.calibrator.method == 'bounded_logit',
            'learned_from_metabolic_outcomes': False,
            'reference_native_signature_verified': self.calibrator is not None,
            'normalized_signal_contract': 'Exact ordered NUTRIENTS; same preprocessing as development',
        }
