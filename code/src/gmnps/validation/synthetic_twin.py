"""Synthetic digital-gut-twin benchmark for anchored GMNPS."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gmnps.scoring import AnchoredScoringConfig, score_individual_foods
from gmnps.scoring.masks import (
    EXPERT_REVISED_LIPID,
    EXPERT_REVISED_MAC,
    ORIGINAL_MASK_VERSION,
    PRIMARY_MASK_VERSION,
)


@dataclass(frozen=True)
class SyntheticTwinBundle:
    """Synthetic data with known individual response ground truth."""

    weights: pd.DataFrame
    nutrients: pd.DataFrame
    food_metadata: pd.DataFrame
    true_response: pd.DataFrame
    capacities: pd.DataFrame


def simulate_synthetic_twin(
    n_individuals: int = 120,
    n_foods: int = 80,
    seed: int = 42,
) -> SyntheticTwinBundle:
    """Create a controlled diet-microbiome-host response world."""

    rng = np.random.default_rng(seed)
    mac = list(EXPERT_REVISED_MAC)
    lipid = list(EXPERT_REVISED_LIPID)
    other = [f"Other nutrient {i}" for i in range(8)]
    columns = mac + lipid + other

    food_ids = [f"food_{i:03d}" for i in range(n_foods)]
    individual_ids = [f"person_{i:03d}" for i in range(n_individuals)]
    food_type = rng.choice(["plant", "animal", "mixed"], size=n_foods, p=[0.45, 0.35, 0.20])

    nutrients = pd.DataFrame(0.0, index=food_ids, columns=columns)
    for j, kind in enumerate(food_type):
        mac_base = 1.8 if kind == "plant" else 0.3 if kind == "animal" else 0.9
        lipid_base = 0.3 if kind == "plant" else 1.7 if kind == "animal" else 0.9
        nutrients.iloc[j, : len(mac)] = rng.gamma(shape=2.0, scale=mac_base / 2.0, size=len(mac))
        nutrients.iloc[j, len(mac) : len(mac) + len(lipid)] = rng.gamma(
            shape=2.0, scale=lipid_base / 2.0, size=len(lipid)
        )
        nutrients.iloc[j, -len(other) :] = rng.gamma(shape=2.0, scale=0.5, size=len(other))

    mac_load = nutrients[mac].mean(axis=1)
    lipid_load = nutrients[lipid].mean(axis=1)
    mac_z = (mac_load - mac_load.mean()) / (mac_load.std(ddof=0) or 1.0)
    lipid_z = (lipid_load - lipid_load.mean()) / (lipid_load.std(ddof=0) or 1.0)
    universal = 58 + 18 * mac_z - 16 * lipid_z
    fcs2 = np.clip(universal.to_numpy(dtype=float), 1, 100)
    food_metadata = pd.DataFrame(
        {
            "food_name": [f"Synthetic {kind} food {i}" for i, kind in enumerate(food_type)],
            "food_group": food_type,
            "FCS2": fcs2,
        },
        index=food_ids,
    )

    capacity_mac = rng.normal(0.0, 1.0, size=n_individuals)
    capacity_lipid = rng.normal(0.0, 1.0, size=n_individuals)
    capacities = pd.DataFrame(
        {"capacity_MAC": capacity_mac, "capacity_LIPID": capacity_lipid},
        index=individual_ids,
    )

    weights = pd.DataFrame(0.0, index=individual_ids, columns=columns)
    weights.loc[:, mac] = capacity_mac[:, None] * rng.uniform(0.7, 1.3, size=len(mac))
    weights.loc[:, lipid] = -capacity_lipid[:, None] * rng.uniform(0.7, 1.3, size=len(lipid))
    weights.loc[:, other] = rng.normal(0, 0.05, size=(n_individuals, len(other)))

    personalized = (
        9.0 * capacity_mac[:, None] * mac_z.to_numpy()[None, :]
        - 9.0 * capacity_lipid[:, None] * lipid_z.to_numpy()[None, :]
    )
    noise = rng.normal(0, 2.0, size=(n_individuals, n_foods))
    true_response = pd.DataFrame(
        np.clip(fcs2[None, :] + personalized + noise, 1, 100),
        index=individual_ids,
        columns=food_ids,
    )
    return SyntheticTwinBundle(weights, nutrients, food_metadata, true_response, capacities)


def _flat_spearman(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_rank = pd.Series(pred.ravel()).rank()
    truth_rank = pd.Series(truth.ravel()).rank()
    if pred_rank.nunique() <= 1 or truth_rank.nunique() <= 1:
        return 0.0
    corr = pred_rank.corr(truth_rank)
    return 0.0 if pd.isna(corr) else float(corr)


def _mean_preservation(pred: np.ndarray, fcs2: np.ndarray) -> float:
    return float(pd.Series(pred.mean(axis=0)).rank().corr(pd.Series(fcs2).rank()))


def _scale_to_1_100(raw: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(raw, [1, 99])
    if hi <= lo:
        return np.full_like(raw, 50.0, dtype=float)
    return np.clip(1 + 99 * (raw - lo) / (hi - lo), 1, 100)


def run_synthetic_benchmark(
    bundle: SyntheticTwinBundle | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Benchmark anchored GMNPS against static, unanchored and null alternatives."""

    bundle = bundle or simulate_synthetic_twin(seed=seed)
    truth = bundle.true_response.to_numpy(dtype=float)
    fcs2 = bundle.food_metadata["FCS2"].to_numpy(dtype=float)

    anchored, _, _ = score_individual_foods(
        bundle.weights,
        bundle.nutrients,
        bundle.food_metadata,
        AnchoredScoringConfig(mask_version=PRIMARY_MASK_VERSION),
    )
    anchored_matrix = anchored.pivot(index="individual_id", columns="food_id", values="GMNPS_score")
    anchored_matrix = anchored_matrix.loc[bundle.weights.index, bundle.nutrients.index].to_numpy()

    original, _, _ = score_individual_foods(
        bundle.weights,
        bundle.nutrients,
        bundle.food_metadata,
        AnchoredScoringConfig(mask_version=ORIGINAL_MASK_VERSION),
    )
    original_matrix = original.pivot(index="individual_id", columns="food_id", values="GMNPS_score")
    original_matrix = original_matrix.loc[bundle.weights.index, bundle.nutrients.index].to_numpy()

    fcs_only = np.broadcast_to(fcs2[None, :], truth.shape)
    raw_unanchored = bundle.weights.to_numpy() @ bundle.nutrients.to_numpy().T
    unanchored = _scale_to_1_100(raw_unanchored)

    rng = np.random.default_rng(seed + 1)
    shuffled_weights = bundle.weights.sample(frac=1.0, random_state=seed + 1).copy()
    shuffled_weights.index = bundle.weights.index
    shuffled, _, _ = score_individual_foods(
        shuffled_weights,
        bundle.nutrients,
        bundle.food_metadata,
        AnchoredScoringConfig(mask_version=PRIMARY_MASK_VERSION),
    )
    shuffled_matrix = shuffled.pivot(index="individual_id", columns="food_id", values="GMNPS_score")
    shuffled_matrix = shuffled_matrix.loc[bundle.weights.index, bundle.nutrients.index].to_numpy()

    random_weights = pd.DataFrame(
        rng.normal(0, 1, size=bundle.weights.shape),
        index=bundle.weights.index,
        columns=bundle.weights.columns,
    )
    random_scored, _, _ = score_individual_foods(
        random_weights,
        bundle.nutrients,
        bundle.food_metadata,
        AnchoredScoringConfig(mask_version=PRIMARY_MASK_VERSION),
    )
    random_matrix = random_scored.pivot(index="individual_id", columns="food_id", values="GMNPS_score")
    random_matrix = random_matrix.loc[bundle.weights.index, bundle.nutrients.index].to_numpy()

    rows = []
    for name, pred in [
        ("FCS2 only", fcs_only),
        ("unanchored microbiome score", unanchored),
        ("anchored GMNPS", anchored_matrix),
        ("random microbiome", random_matrix),
        ("shuffled microbiome", shuffled_matrix),
        ("original mask", original_matrix),
        ("expert-revised mask", anchored_matrix),
    ]:
        rows.append(
            {
                "model": name,
                "individual_response_spearman": _flat_spearman(pred, truth),
                "personalized_residual_spearman": _flat_spearman(
                    pred - fcs2[None, :],
                    truth - fcs2[None, :],
                ),
                "nps_preservation_spearman": _mean_preservation(pred, fcs2),
            }
        )
    return pd.DataFrame(rows)
