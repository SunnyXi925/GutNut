from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gmnps.scoring.masks import build_channel_vectors


@dataclass(frozen=True)
class DualChannelBetaConfig:
    dose: float = 0.25
    clip_abs_beta: float = 35.0
    response_scale: str = "logit"
    compression_temperature: float = 0.35


def _standardize_frame(frame: pd.DataFrame, mean: pd.Series, scale: pd.Series) -> pd.DataFrame:
    safe_scale = scale.replace(0.0, np.nan)
    return ((frame.astype(float) - mean) / safe_scale).fillna(0.0)


def _sigmoid(values: pd.Series) -> pd.Series:
    clipped = values.clip(lower=-40.0, upper=40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _mask_vector(
    channel_masks: dict[str, np.ndarray | list[str] | str],
    perturbations: pd.DataFrame,
    name: str,
) -> np.ndarray:
    mask = channel_masks.get(name)
    if mask is None or isinstance(mask, str):
        mask = build_channel_vectors(perturbations.index)[name]
    mask_values = np.asarray(mask)
    if mask_values.dtype.kind in {"O", "S", "U"} and not all(isinstance(v, (bool, np.bool_)) for v in mask_values):
        names = {str(v) for v in mask_values}
        return np.asarray([str(nutrient) in names for nutrient in perturbations.index], dtype=bool)
    mask_array = mask_values.astype(bool)
    if len(mask_array) != len(perturbations.index):
        mask_array = np.asarray(build_channel_vectors(perturbations.index)[name], dtype=bool)
    return mask_array


def _channel_weights(aligned_perturb: pd.DataFrame, mask: np.ndarray) -> pd.Series:
    if not mask.any():
        return pd.Series(0.0, index=aligned_perturb.columns)
    return aligned_perturb.loc[mask].mean(axis=0)


def _raw_dual_features(
    clr: pd.DataFrame,
    perturbations: pd.DataFrame,
    channel_masks: dict[str, np.ndarray | list[str] | str],
) -> pd.DataFrame:
    aligned_perturb = perturbations.reindex(columns=clr.columns).fillna(0.0).astype(float)
    mac = _mask_vector(channel_masks, aligned_perturb, "mac")
    lipid = _mask_vector(channel_masks, aligned_perturb, "lipid")
    mac_weights = _channel_weights(aligned_perturb, mac)
    lipid_weights = _channel_weights(aligned_perturb, lipid)
    mac_capacity = clr.mul(mac_weights, axis=1).sum(axis=1)
    lipid_capacity = clr.mul(lipid_weights, axis=1).sum(axis=1)
    return pd.DataFrame(
        {
            "official_gmwi2_z": mac_capacity - lipid_capacity,
            "mac_capacity_z": mac_capacity,
            "lipid_risk_capacity_z": lipid_capacity,
        },
        index=clr.index,
    )


def _dual_features(
    clr: pd.DataFrame,
    perturbations: pd.DataFrame,
    channel_masks: dict[str, np.ndarray | list[str] | str],
    feature_mean: pd.Series,
    feature_scale: pd.Series,
) -> pd.DataFrame:
    features = _standardize_frame(_raw_dual_features(clr, perturbations, channel_masks), feature_mean, feature_scale)
    features["mac_x_gmwi2"] = features["mac_capacity_z"] * features["official_gmwi2_z"]
    features["lipid_x_gmwi2"] = features["lipid_risk_capacity_z"] * features["official_gmwi2_z"]
    return features


def _score(
    features: pd.DataFrame,
    feature_weights: pd.Series,
    config: DualChannelBetaConfig,
) -> pd.Series:
    aligned_weights = feature_weights.reindex(features.columns).fillna(0.0).astype(float)
    raw = features.mul(aligned_weights, axis=1).sum(axis=1) / config.compression_temperature
    raw = raw.rename("dual_channel_health_logit")
    if config.response_scale == "logit":
        return raw
    return _sigmoid(raw).rename("dual_channel_health_probability")


def compute_dual_channel_beta(
    clr: pd.DataFrame,
    perturbations: pd.DataFrame,
    feature_weights: pd.Series,
    channel_masks: dict[str, np.ndarray | list[str] | str],
    config: DualChannelBetaConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.dose <= 0:
        raise ValueError("dose must be positive")
    if config.clip_abs_beta <= 0:
        raise ValueError("clip_abs_beta must be positive")
    if config.compression_temperature <= 0:
        raise ValueError("compression_temperature must be positive")
    if config.response_scale not in {"logit", "probability"}:
        raise ValueError("response_scale must be 'logit' or 'probability'")

    aligned_clr = clr.copy()
    aligned_clr.index = aligned_clr.index.astype(str)
    aligned_clr.columns = aligned_clr.columns.astype(str)
    aligned_clr = aligned_clr.astype(float).fillna(0.0)
    aligned_perturb = perturbations.copy()
    aligned_perturb.index = aligned_perturb.index.astype(str)
    aligned_perturb.columns = aligned_perturb.columns.astype(str)
    aligned_perturb = aligned_perturb.reindex(columns=aligned_clr.columns).fillna(0.0).astype(float)
    baseline_raw_features = _raw_dual_features(aligned_clr, aligned_perturb, channel_masks)
    feature_mean = baseline_raw_features.mean(axis=0)
    feature_scale = baseline_raw_features.std(axis=0, ddof=0).replace(0.0, 1.0)

    beta_columns: list[pd.Series] = []
    for nutrient in aligned_perturb.index:
        direction = aligned_perturb.loc[nutrient]
        plus_features = _dual_features(
            aligned_clr + config.dose * direction,
            aligned_perturb,
            channel_masks,
            feature_mean,
            feature_scale,
        )
        minus_features = _dual_features(
            aligned_clr - config.dose * direction,
            aligned_perturb,
            channel_masks,
            feature_mean,
            feature_scale,
        )
        response = (_score(plus_features, feature_weights, config) - _score(minus_features, feature_weights, config)) / (
            2.0 * config.dose
        )
        beta_columns.append(response.rename(nutrient))

    if beta_columns:
        beta = pd.concat(beta_columns, axis=1)
        beta = beta.clip(lower=-config.clip_abs_beta, upper=config.clip_abs_beta).astype("float32")
    else:
        beta = pd.DataFrame(index=aligned_clr.index)

    diagnostics = pd.DataFrame(
        {
            "sample_id": beta.index.astype(str),
            "beta_model": "gmwi2_dual_channel",
            "beta_mean": beta.mean(axis=1).to_numpy(dtype=float),
            "beta_sd": beta.std(axis=1, ddof=0).to_numpy(dtype=float),
            "beta_abs_max": beta.abs().max(axis=1).to_numpy(dtype=float),
            "beta_delta_span_p95_p05": beta.quantile(0.95, axis=1).to_numpy(dtype=float)
            - beta.quantile(0.05, axis=1).to_numpy(dtype=float),
            "beta_compression_temperature": float(config.compression_temperature),
        }
    )
    beta.index = beta.index.astype(str)
    beta.columns = beta.columns.astype(str)
    return beta, diagnostics
