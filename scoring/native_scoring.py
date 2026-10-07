"""Five-domain scoring arrays constructed from caller-supplied, aligned frames."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .fcs2_attribute_mapping import PRIMARY_ATTRIBUTE_MAPPINGS, build_food_specific_response
from .fcs2_attribute_rules import FCS2_RULES, NOT_CALCULATED, aggregate_domains

DOMAINS = ('nutrient_ratios', 'vitamins', 'minerals', 'fiber_and_protein', 'phytochemicals')
ATTRIBUTES = tuple(a for a, r in FCS2_RULES.items() if r.active and r.domain in DOMAINS)
DOMAIN_WEIGHTS = {d: (0.5 if d == 'phytochemicals' else 1.0) for d in DOMAINS}
NUTRIENTS = tuple(dict.fromkeys(m.nutrient for m in PRIMARY_ATTRIBUTE_MAPPINGS if m.role == 'effect'))

def _validate_index(frames: list[pd.DataFrame]) -> pd.Index:
    ids = frames[0].index
    if not ids.is_unique or not all(ids.equals(f.index) for f in frames[1:]):
        raise ValueError("food inputs must have identical unique indices")
    return ids



def aggregate_extended(points: np.ndarray, weights: np.ndarray, reference_points: np.ndarray | None = None) -> np.ndarray:
    """Return the sum of weighted complete domains, including the 0.5 domain."""
    if points.shape[-1] != len(ATTRIBUTES) or weights.shape != points.shape[-2:]:
        raise ValueError("attribute or food dimensions do not agree")
    if reference_points is not None and reference_points.shape != points.shape[-2:]:
        raise ValueError("reference points must have food by attribute shape")
    result = np.zeros(points.shape[:-1], dtype=float)
    for domain in DOMAINS:
        ix = [i for i, a in enumerate(ATTRIBUTES) if FCS2_RULES[a].domain == domain]
        p = points[..., ix]
        w = np.broadcast_to(weights[:, ix], p.shape)
        if domain in {"vitamins", "minerals"}:
            ranking = p if reference_points is None else reference_points[:, ix]
            order = np.argsort(-np.where(np.isfinite(ranking), np.abs(ranking), -np.inf), axis=-1, kind="stable")[..., :5]
            order = np.broadcast_to(order, p.shape[:-1] + (5,))
            p, w = np.take_along_axis(p, order, axis=-1), np.take_along_axis(w, order, axis=-1)
        valid = np.isfinite(p)
        denominator = np.where(valid, w, 0.0).sum(axis=-1)
        if (denominator <= 0).any():
            raise ValueError("a complete domain has no active denominator")
        result += DOMAIN_WEIGHTS[domain] * (np.where(valid, p, 0.0) * w).sum(axis=-1) / denominator
    return result



@dataclass
class NativeFoodScorer:
    food_ids: tuple
    baseline: np.ndarray
    points: np.ndarray
    weights: np.ndarray
    design: np.ndarray
    baseline_domain_total: np.ndarray

    @classmethod
    def from_frames(cls, metadata, points, weights, exposures):
        if exposures.attrs.get("basis") != "per_100_kcal":
            raise ValueError("food exposure metadata basis must be per_100_kcal")
        ids = _validate_index([metadata, points, weights, exposures])
        p, w = points.loc[:, ATTRIBUTES].to_numpy(float), weights.loc[:, ATTRIBUTES].to_numpy(float)
        low = np.array([min(FCS2_RULES[a].low_points, FCS2_RULES[a].high_points) for a in ATTRIBUTES])
        high = np.array([max(FCS2_RULES[a].low_points, FCS2_RULES[a].high_points) for a in ATTRIBUTES])
        if not np.isfinite(w).all() or (w < 0).any() or ((p < low-1e-10) | (p > high+1e-10)).any():
            raise ValueError("invalid native points or effective weights")
        if ((~np.isfinite(p)) & (w != 0)).any():
            raise ValueError("uncalculated attributes must have zero effective weights")
        if not np.isfinite(p[:, [ATTRIBUTES.index(a) for a in ("total_flavonoids", "total_carotenoids")]]).all():
            raise ValueError("phytochemical domain must contain both observed attributes")
        design = np.zeros((len(ids), len(NUTRIENTS), len(ATTRIBUTES)))
        beta = pd.Series(1.0, index=NUTRIENTS)
        for i, (_, exposure) in enumerate(exposures.iterrows()):
            exposure = exposure.copy()
            exposure.attrs["basis"] = "per_100_kcal"
            diagnostics = build_food_specific_response(beta, exposure).diagnostics
            for row in diagnostics.itertuples(index=False):
                if row.attribute in ATTRIBUTES:
                    design[i, NUTRIENTS.index(row.nutrient), ATTRIBUTES.index(row.attribute)] += row.normalized_exposure * row.allocation_weight
        if np.any(design[:, :, ATTRIBUTES.index("total_flavonoids")] != 0):
            raise AssertionError("flavonoid signal must remain fixed")
        baseline = metadata.fcs2.to_numpy(float)
        if not np.isfinite(baseline).all() or ((baseline < 1) | (baseline > 100)).any():
            raise ValueError("invalid official score")
        total = aggregate_extended(p, w)
        for i in range(len(ids)):
            attrs = {a: (float(v) if np.isfinite(v) else NOT_CALCULATED) for a, v in zip(ATTRIBUTES, p[i])}
            canonical = sum(aggregate_domains(attrs, effective_attribute_weights=dict(zip(ATTRIBUTES, w[i]))).values())
            if not np.isclose(canonical, total[i], atol=1e-10, rtol=0):
                raise AssertionError("canonical five-domain recomposition mismatch")
        return cls(tuple(ids), baseline, p, w, design, total)
