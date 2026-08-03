from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gmnps.beta_i.health_index import HealthIndexModel
from gmnps.scoring.masks import build_channel_vectors


@dataclass(frozen=True)
class NutrientPerturbationConfig:
    mac_channel_weight: float = 1.0
    lipid_channel_weight: float = 1.0
    other_channel_weight: float = 0.25
    mac_evidence_direction: float = 1.0
    lipid_evidence_direction: float = -1.0
    other_evidence_direction: float = 1.0
    min_abs_bridge: float = 0.0
    l2_norm: float = 1.0


def _channel_weights(nutrients: list[str], config: NutrientPerturbationConfig) -> pd.Series:
    masks = build_channel_vectors(nutrients)
    values = np.where(
        masks["mac"],
        config.mac_channel_weight,
        np.where(masks["lipid"], config.lipid_channel_weight, config.other_channel_weight),
    )
    return pd.Series(values.astype(float), index=nutrients, name="channel_weight")


def _channel_labels(nutrients: list[str]) -> pd.Series:
    masks = build_channel_vectors(nutrients)
    labels = np.where(masks["mac"], "MAC", np.where(masks["lipid"], "LIPID", "OTHER"))
    return pd.Series(labels, index=nutrients, name="channel")


def _evidence_directions(nutrients: list[str], config: NutrientPerturbationConfig) -> pd.Series:
    masks = build_channel_vectors(nutrients)
    values = np.where(
        masks["mac"],
        config.mac_evidence_direction,
        np.where(masks["lipid"], config.lipid_evidence_direction, config.other_evidence_direction),
    )
    return pd.Series(values.astype(float), index=nutrients, name="evidence_direction")


def build_nutrient_perturbations(
    nutrient_genus: pd.DataFrame,
    model: HealthIndexModel,
    config: NutrientPerturbationConfig,
) -> pd.DataFrame:
    if config.l2_norm < 0 or min(
        config.mac_channel_weight,
        config.lipid_channel_weight,
        config.other_channel_weight,
    ) < 0:
        raise ValueError("l2_norm and channel weights must be nonnegative")
    bridge = nutrient_genus.reindex(columns=model.genus_names).fillna(0.0).astype(float)
    bridge.index = bridge.index.astype(str)
    if config.min_abs_bridge > 0:
        bridge = bridge.where(bridge.abs() >= config.min_abs_bridge, 0.0)
    health_direction = np.sign(model.coefficients.reindex(model.genus_names).fillna(0.0))
    oriented = bridge.mul(health_direction, axis=1)
    nutrients = list(oriented.index)
    evidence_directions = _evidence_directions(nutrients, config)
    oriented = oriented.mul(evidence_directions, axis=0)
    centered = oriented.sub(oriented.mean(axis=1), axis=0)
    norms = np.sqrt((centered**2).sum(axis=1)).replace(0.0, np.nan)
    normalized = centered.div(norms, axis=0).fillna(0.0)
    weights = _channel_weights(nutrients, config)
    scaled = normalized.mul(weights, axis=0) * config.l2_norm
    scaled.index = scaled.index.astype(str)
    scaled.columns = scaled.columns.astype(str)
    return scaled.astype("float32")


def summarize_perturbations(
    perturbations: pd.DataFrame,
    config: NutrientPerturbationConfig | None = None,
) -> pd.DataFrame:
    config = config or NutrientPerturbationConfig()
    nutrients = list(perturbations.index.astype(str))
    labels = _channel_labels(nutrients)
    weights = _channel_weights(nutrients, config)
    evidence_directions = _evidence_directions(nutrients, config)
    return pd.DataFrame(
        {
            "nutrient": nutrients,
            "channel": labels.reindex(nutrients).to_numpy(),
            "channel_weight": weights.reindex(nutrients).to_numpy(),
            "evidence_direction": evidence_directions.reindex(nutrients).to_numpy(),
            "n_nonzero_genera": (perturbations.abs() > 0).sum(axis=1).astype(int).to_numpy(),
            "l1_norm": perturbations.abs().sum(axis=1).astype(float).to_numpy(),
            "l2_norm": np.sqrt((perturbations.astype(float) ** 2).sum(axis=1)).to_numpy(),
        }
    )
