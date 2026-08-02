"""Nutrient masks for the Food Compass 2.0-anchored GMNPS article model.

The primary mask is the expert-revised dual-channel definition. It keeps
GMNPS inside nutrient profiling by using Food Compass 2.0 as the universal
baseline and restricts microbiome information to interpretable deviations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


SCORING_VERSION = "anchored-gmnps-v1"
EXPERT_REVISED_V4_MASK_VERSION = "expert_revised_v4_dual_channel"
PRIMARY_MASK_VERSION = EXPERT_REVISED_V4_MASK_VERSION
LEGACY_EXPERT_REVISED_MASK_VERSION = "expert_revised_dual_channel"
ORIGINAL_MASK_VERSION = "original_15x15_dual_channel"


EXPERT_REVISED_MAC = frozenset(
    {
        "Fiber, total dietary (g)",
        "Carotene, alpha (mcg)",
        "Carotene, beta (mcg)",
        "Cryptoxanthin, beta (mcg)",
        "Lycopene (mcg)",
        "Lutein + zeaxanthin (mcg)",
        "Vitamin C (mg)",
        "Folate, food (mcg)",
        "Magnesium (mg)",
        "Potassium (mg)",
        "Vitamin K (phylloquinone) (mcg)",
        "Vitamin E (alpha-tocopherol) (mg)",
    }
)


EXPERT_REVISED_LIPID = frozenset(
    {
        "Total Fat (g)",
        "Fatty acids, total saturated (g)",
        "Cholesterol (mg)",
        "Retinol (mcg)",
        "4:0 (g)",
        "6:0 (g)",
        "8:0 (g)",
        "10:0 (g)",
        "12:0 (g)",
        "14:0 (g)",
        "16:0 (g)",
        "18:0 (g)",
        "Choline, total (mg)",
        "Vitamin B-12 (mcg)",
    }
)


ORIGINAL_MAC = frozenset(
    set(EXPERT_REVISED_MAC)
    | {
        "Carbohydrate (g)",
        "Copper (mg)",
        "Zinc (mg)",
    }
)


ORIGINAL_LIPID = frozenset(
    set(EXPERT_REVISED_LIPID)
    | {
        "Vitamin A, RAE (mcg_RAE)",
    }
)


PRIMARY_EXCLUDED_FROM_CHANNELS = frozenset(
    {
        "Carbohydrate (g)",
        "Copper (mg)",
        "Zinc (mg)",
        "Vitamin A, RAE (mcg_RAE)",
    }
)


MASK_POLICY = {
    "baseline_nps": "FoodCompass2.0",
    "primary_mask": PRIMARY_MASK_VERSION,
    "carbohydrate_policy": "sensitivity_proxy_only",
    "zinc_copper_policy": "remove_from_microbiome_channels",
    "vitamin_a_rae_policy": "remove_from_lipid_channel",
    "retinol_policy": "retain_lipid_marker",
    "clinical_experiment": False,
}


@dataclass(frozen=True)
class ChannelMaskDefinition:
    """Named sets of nutrients used for channel decomposition."""

    version: str
    mac: frozenset[str]
    lipid: frozenset[str]

    @property
    def excluded(self) -> frozenset[str]:
        return PRIMARY_EXCLUDED_FROM_CHANNELS if self.version == PRIMARY_MASK_VERSION else frozenset()


def get_mask_definition(version: str = PRIMARY_MASK_VERSION) -> ChannelMaskDefinition:
    """Return the requested channel definition."""

    if version in {PRIMARY_MASK_VERSION, LEGACY_EXPERT_REVISED_MASK_VERSION}:
        return ChannelMaskDefinition(version, EXPERT_REVISED_MAC, EXPERT_REVISED_LIPID)
    if version == ORIGINAL_MASK_VERSION:
        return ChannelMaskDefinition(version, ORIGINAL_MAC, ORIGINAL_LIPID)
    raise ValueError(f"Unknown mask version: {version}")


def build_channel_vectors(
    nutrient_columns: Iterable[str],
    version: str = PRIMARY_MASK_VERSION,
) -> dict[str, np.ndarray | list[str] | str]:
    """Create boolean MAC, lipid and OTHER masks aligned to nutrient columns."""

    columns = list(nutrient_columns)
    definition = get_mask_definition(version)
    mac = np.array([c in definition.mac for c in columns], dtype=bool)
    lipid = np.array([c in definition.lipid for c in columns], dtype=bool)
    if np.any(mac & lipid):
        overlap = [c for c, m, l in zip(columns, mac, lipid) if m and l]
        raise ValueError(f"MAC/LIPID masks overlap: {overlap}")
    other = ~(mac | lipid)
    return {
        "version": definition.version,
        "columns": columns,
        "mac": mac,
        "lipid": lipid,
        "other": other,
        "mac_nutrients": [c for c, m in zip(columns, mac) if m],
        "lipid_nutrients": [c for c, m in zip(columns, lipid) if m],
        "other_nutrients": [c for c, m in zip(columns, other) if m],
    }


def audit_primary_mask(nutrient_columns: Iterable[str]) -> dict[str, list[str]]:
    """Audit expert-revised mask boundaries against available nutrient columns."""

    columns = set(nutrient_columns)
    definition = get_mask_definition(PRIMARY_MASK_VERSION)
    return {
        "missing_mac": sorted(definition.mac - columns),
        "missing_lipid": sorted(definition.lipid - columns),
        "excluded_present_as_columns": sorted(PRIMARY_EXCLUDED_FROM_CHANNELS & columns),
        "excluded_in_mac": sorted(definition.mac & PRIMARY_EXCLUDED_FROM_CHANNELS),
        "excluded_in_lipid": sorted(definition.lipid & PRIMARY_EXCLUDED_FROM_CHANNELS),
        "retinol_present": ["Retinol (mcg)"] if "Retinol (mcg)" in columns else [],
    }
