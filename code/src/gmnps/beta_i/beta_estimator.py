from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gmnps.beta_i.health_index import HealthIndexModel, score_health_index


@dataclass(frozen=True)
class BetaEstimatorConfig:
    dose: float = 0.1
    clip_abs_beta: float = 12.0
    batch_size: int = 32


def compute_beta_matrix(
    clr: pd.DataFrame,
    perturbations: pd.DataFrame,
    model: HealthIndexModel,
    config: BetaEstimatorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.dose <= 0:
        raise ValueError("dose must be positive")
    aligned_clr = clr.reindex(columns=model.genus_names).astype(float)
    aligned_clr = aligned_clr.fillna(model.mean_)
    aligned_perturb = perturbations.reindex(columns=model.genus_names).fillna(0.0).astype(float)
    baseline = score_health_index(model, aligned_clr)
    beta_blocks = []
    nutrient_names = list(aligned_perturb.index.astype(str))
    for start in range(0, len(nutrient_names), config.batch_size):
        batch = nutrient_names[start : start + config.batch_size]
        cols = []
        for nutrient in batch:
            shifted = aligned_clr + config.dose * aligned_perturb.loc[nutrient]
            shifted_score = score_health_index(model, shifted)
            cols.append(((shifted_score - baseline) / config.dose).rename(nutrient))
        beta_blocks.append(pd.concat(cols, axis=1))
    beta = pd.concat(beta_blocks, axis=1) if beta_blocks else pd.DataFrame(index=aligned_clr.index)
    beta = beta.clip(lower=-config.clip_abs_beta, upper=config.clip_abs_beta).astype("float32")
    diagnostics = pd.DataFrame(
        {
            "sample_id": aligned_clr.index.astype(str),
            "baseline_health_index": baseline.to_numpy(dtype=float),
            "beta_mean": beta.mean(axis=1).to_numpy(dtype=float),
            "beta_sd": beta.std(axis=1, ddof=0).to_numpy(dtype=float),
            "beta_abs_max": beta.abs().max(axis=1).to_numpy(dtype=float),
        }
    )
    beta.index = beta.index.astype(str)
    beta.columns = beta.columns.astype(str)
    return beta, diagnostics
