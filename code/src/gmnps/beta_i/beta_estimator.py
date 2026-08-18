from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gmnps.beta_i.health_index import HealthIndexModel, score_health_index, score_health_index_logit


@dataclass(frozen=True)
class BetaEstimatorConfig:
    dose: float = 0.25
    clip_abs_beta: float = 25.0
    batch_size: int = 32
    response_scale: str = "logit"
    difference: str = "central"


def compute_beta_matrix(
    clr: pd.DataFrame,
    perturbations: pd.DataFrame,
    model: HealthIndexModel,
    config: BetaEstimatorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.dose <= 0:
        raise ValueError("dose must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.response_scale not in {"probability", "logit"}:
        raise ValueError("response_scale must be 'probability' or 'logit'")
    if config.difference not in {"forward", "central"}:
        raise ValueError("difference must be 'forward' or 'central'")
    aligned_clr = clr.reindex(columns=model.genus_names).astype(float)
    aligned_clr = aligned_clr.fillna(model.mean_)
    aligned_perturb = perturbations.reindex(columns=model.genus_names).fillna(0.0).astype(float)
    scorer = score_health_index_logit if config.response_scale == "logit" else score_health_index
    baseline = scorer(model, aligned_clr)
    beta_blocks = []
    nutrient_names = list(aligned_perturb.index)
    for start in range(0, len(nutrient_names), config.batch_size):
        batch = nutrient_names[start : start + config.batch_size]
        cols = []
        for nutrient in batch:
            direction = aligned_perturb.loc[nutrient]
            if config.difference == "central":
                shifted_plus = aligned_clr + config.dose * direction
                shifted_minus = aligned_clr - config.dose * direction
                response = (scorer(model, shifted_plus) - scorer(model, shifted_minus)) / (2.0 * config.dose)
            else:
                shifted = aligned_clr + config.dose * direction
                response = (scorer(model, shifted) - baseline) / config.dose
            cols.append(response.rename(nutrient))
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
            "beta_response_scale": config.response_scale,
            "beta_difference": config.difference,
            "beta_delta_span_p95_p05": beta.quantile(0.95, axis=1).to_numpy(dtype=float)
            - beta.quantile(0.05, axis=1).to_numpy(dtype=float),
        }
    )
    beta.index = beta.index.astype(str)
    beta.columns = beta.columns.astype(str)
    return beta, diagnostics
