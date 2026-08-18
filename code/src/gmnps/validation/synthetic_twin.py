"""Synthetic digital-gut-twin benchmark for anchored GMNPS."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gmnps.scoring import AnchoredScoringConfig, score_individual_foods
from gmnps.scoring.anchored import raw_channel_responses, robust_zscore_by_food
from gmnps.scoring.masks import (
    EXPERT_REVISED_LIPID,
    EXPERT_REVISED_MAC,
    ORIGINAL_MASK_VERSION,
    PRIMARY_EXCLUDED_FROM_CHANNELS,
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


def simulate_mask_sensitive_twin(
    n_individuals: int = 160,
    n_foods: int = 100,
    seed: int = 42,
) -> SyntheticTwinBundle:
    """Create a benchmark where expert exclusions are identifiable.

    The standard synthetic twin uses only expert-revised MAC/LIPID nutrients, so
    the original and expert-revised masks are intentionally equivalent. This
    variant adds broad proxy nutrients excluded from the primary channels. They
    are correlated with the food matrix and individual weights but do not enter
    the true personalized response. A good primary mask should therefore avoid
    attributing response recovery to these proxy variables.
    """

    rng = np.random.default_rng(seed)
    mac = list(EXPERT_REVISED_MAC)
    lipid = list(EXPERT_REVISED_LIPID)
    excluded = sorted(PRIMARY_EXCLUDED_FROM_CHANNELS)
    other = [f"Other nutrient {i}" for i in range(8)]
    columns = mac + lipid + excluded + other

    food_ids = [f"food_{i:03d}" for i in range(n_foods)]
    individual_ids = [f"person_{i:03d}" for i in range(n_individuals)]
    food_type = rng.choice(["plant", "animal", "mixed"], size=n_foods, p=[0.45, 0.35, 0.20])

    nutrients = pd.DataFrame(0.0, index=food_ids, columns=columns)
    for j, kind in enumerate(food_type):
        mac_base = 1.8 if kind == "plant" else 0.3 if kind == "animal" else 0.9
        lipid_base = 0.3 if kind == "plant" else 1.7 if kind == "animal" else 0.9
        nutrients.iloc[j, : len(mac)] = rng.gamma(shape=2.0, scale=mac_base / 2.0, size=len(mac))
        start = len(mac)
        nutrients.iloc[j, start : start + len(lipid)] = rng.gamma(
            shape=2.0, scale=lipid_base / 2.0, size=len(lipid)
        )

    mac_load = nutrients[mac].mean(axis=1)
    lipid_load = nutrients[lipid].mean(axis=1)
    mac_z = (mac_load - mac_load.mean()) / (mac_load.std(ddof=0) or 1.0)
    lipid_z = (lipid_load - lipid_load.mean()) / (lipid_load.std(ddof=0) or 1.0)

    # Broad proxies are correlated with the true substrate axes but are not
    # included in the ground-truth personalized response.
    proxy_values = {
        "Carbohydrate (g)": 0.75 * mac_z + rng.normal(0, 0.75, size=n_foods),
        "Zinc (mg)": 0.35 * mac_z + rng.normal(0, 0.85, size=n_foods),
        "Copper (mg)": 0.30 * mac_z + rng.normal(0, 0.85, size=n_foods),
        "Vitamin A, RAE (mcg_RAE)": 0.55 * lipid_z + rng.normal(0, 0.75, size=n_foods),
    }
    for nutrient, values in proxy_values.items():
        if nutrient in nutrients.columns:
            shifted = values - values.min() + 0.05
            nutrients[nutrient] = shifted.astype(float)
    for nutrient in other:
        nutrients[nutrient] = rng.gamma(shape=2.0, scale=0.5, size=n_foods)

    universal = 58 + 18 * mac_z - 16 * lipid_z
    fcs2 = np.clip(universal.to_numpy(dtype=float), 1, 100)
    food_metadata = pd.DataFrame(
        {
            "food_name": [f"Mask-sensitive {kind} food {i}" for i, kind in enumerate(food_type)],
            "food_group": food_type,
            "FCS2": fcs2,
        },
        index=food_ids,
    )

    capacity_mac = rng.normal(0.0, 1.0, size=n_individuals)
    capacity_lipid = rng.normal(0.0, 1.0, size=n_individuals)
    spurious_proxy = rng.normal(0.0, 1.0, size=n_individuals)
    capacities = pd.DataFrame(
        {
            "capacity_MAC": capacity_mac,
            "capacity_LIPID": capacity_lipid,
            "spurious_proxy_capacity": spurious_proxy,
        },
        index=individual_ids,
    )

    weights = pd.DataFrame(0.0, index=individual_ids, columns=columns)
    weights.loc[:, mac] = capacity_mac[:, None] * rng.uniform(0.7, 1.3, size=len(mac))
    weights.loc[:, lipid] = -capacity_lipid[:, None] * rng.uniform(0.7, 1.3, size=len(lipid))
    for nutrient in excluded:
        if nutrient in weights.columns:
            weights[nutrient] = spurious_proxy * rng.uniform(0.8, 1.4)
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


def _flat_spearman(pred: np.ndarray, truth: np.ndarray, undefined: float = 0.0) -> float:
    pred_rank = pd.Series(pred.ravel()).rank()
    truth_rank = pd.Series(truth.ravel()).rank()
    if pred_rank.nunique() <= 1 or truth_rank.nunique() <= 1:
        return undefined
    corr = pred_rank.corr(truth_rank)
    return undefined if pd.isna(corr) else float(corr)


def _mean_preservation(pred: np.ndarray, fcs2: np.ndarray) -> float:
    return float(pd.Series(pred.mean(axis=0)).rank().corr(pd.Series(fcs2).rank()))


def _scale_to_1_100(raw: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(raw, [1, 99])
    if hi <= lo:
        return np.full_like(raw, 50.0, dtype=float)
    return np.clip(1 + 99 * (raw - lo) / (hi - lo), 1, 100)


def _score_to_matrix(scored: pd.DataFrame, bundle: SyntheticTwinBundle) -> np.ndarray:
    matrix = scored.pivot(index="individual_id", columns="food_id", values="GMNPS_score")
    return matrix.loc[bundle.weights.index, bundle.nutrients.index].to_numpy()


def _uncentered_gmnps_matrix(
    bundle: SyntheticTwinBundle,
    delta_cap: float = 12.0,
    z_scale: float = 2.0,
) -> np.ndarray:
    responses = raw_channel_responses(bundle.weights, bundle.nutrients, PRIMARY_MASK_VERSION)
    z = robust_zscore_by_food(responses["total"]).to_numpy(dtype=float)
    delta = np.clip(delta_cap * np.tanh(z / z_scale), -delta_cap, delta_cap)
    fcs2 = bundle.food_metadata.loc[bundle.nutrients.index, "FCS2"].to_numpy(dtype=float)
    return np.clip(delta + fcs2[None, :], 1, 100)


def _benchmark_row(name: str, pred: np.ndarray, truth: np.ndarray, fcs2: np.ndarray) -> dict[str, object]:
    residual_pred = pred - fcs2[None, :]
    residual_truth = truth - fcs2[None, :]
    residual_defined = (
        pd.Series(residual_pred.ravel()).rank().nunique() > 1
        and pd.Series(residual_truth.ravel()).rank().nunique() > 1
    )
    return {
        "model": name,
        "individual_response_spearman": _flat_spearman(pred, truth),
        "personalized_residual_spearman": _flat_spearman(
            residual_pred,
            residual_truth,
            undefined=np.nan,
        ),
        "personalized_residual_defined": bool(residual_defined),
        "nps_preservation_spearman": _mean_preservation(pred, fcs2),
    }


def _excluded_false_driver_rate(
    bundle: SyntheticTwinBundle,
    mask_version: str,
) -> float:
    """Estimate the raw contribution share from expert-excluded nutrients."""

    shared = [c for c in bundle.weights.columns if c in bundle.nutrients.columns]
    excluded = [c for c in shared if c in PRIMARY_EXCLUDED_FROM_CHANNELS]
    if not excluded:
        return 0.0
    all_contrib = np.abs(bundle.weights[shared].to_numpy()[:, None, :] * bundle.nutrients[shared].to_numpy()[None, :, :])
    excl_contrib = np.abs(
        bundle.weights[excluded].to_numpy()[:, None, :] * bundle.nutrients[excluded].to_numpy()[None, :, :]
    )
    denom = float(all_contrib.sum())
    if denom <= 0:
        return 0.0
    if mask_version == PRIMARY_MASK_VERSION:
        return 0.0
    return float(excl_contrib.sum() / denom)


def _delta_mse(pred: np.ndarray, truth: np.ndarray, fcs2: np.ndarray) -> float:
    residual_pred = pred - fcs2[None, :]
    residual_truth = truth - fcs2[None, :]
    return float(np.mean((residual_pred - residual_truth) ** 2))


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
        rows.append(_benchmark_row(name, pred, truth, fcs2))
    return pd.DataFrame(rows)


def run_mask_sensitive_benchmark(
    bundle: SyntheticTwinBundle | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Benchmark primary mask against original mask in a mask-sensitive world."""

    bundle = bundle or simulate_mask_sensitive_twin(seed=seed)
    truth = bundle.true_response.to_numpy(dtype=float)
    fcs2 = bundle.food_metadata["FCS2"].to_numpy(dtype=float)

    rows = []
    for name, mask_version in [
        ("expert-revised mask", PRIMARY_MASK_VERSION),
        ("original mask", ORIGINAL_MASK_VERSION),
    ]:
        scored, _, _ = score_individual_foods(
            bundle.weights,
            bundle.nutrients,
            bundle.food_metadata,
            AnchoredScoringConfig(mask_version=mask_version),
        )
        pred = _score_to_matrix(scored, bundle)
        row = _benchmark_row(name, pred, truth, fcs2)
        row["mask_version"] = mask_version
        row["delta_mse"] = _delta_mse(pred, truth, fcs2)
        row["excluded_nutrient_false_driver_rate"] = _excluded_false_driver_rate(bundle, mask_version)
        rows.append(row)
    return pd.DataFrame(rows)


def run_delta_cap_sensitivity(
    bundle: SyntheticTwinBundle | None = None,
    caps: tuple[float, ...] = (8.0, 12.0, 15.0),
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate anchored GMNPS across bounded-deviation caps."""

    bundle = bundle or simulate_synthetic_twin(seed=seed)
    truth = bundle.true_response.to_numpy(dtype=float)
    fcs2 = bundle.food_metadata["FCS2"].to_numpy(dtype=float)
    rows = []
    for cap in caps:
        scored, _, _ = score_individual_foods(
            bundle.weights,
            bundle.nutrients,
            bundle.food_metadata,
            AnchoredScoringConfig(delta_cap=cap, mask_version=PRIMARY_MASK_VERSION),
        )
        row = _benchmark_row(f"anchored GMNPS cap {cap:g}", _score_to_matrix(scored, bundle), truth, fcs2)
        row["delta_cap"] = cap
        rows.append(row)
    return pd.DataFrame(rows)


def run_design_ablation(
    bundle: SyntheticTwinBundle | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Ablate anchoring, centering and microbiome specificity."""

    bundle = bundle or simulate_synthetic_twin(seed=seed)
    truth = bundle.true_response.to_numpy(dtype=float)
    fcs2 = bundle.food_metadata["FCS2"].to_numpy(dtype=float)
    fcs_only = np.broadcast_to(fcs2[None, :], truth.shape)

    centered, _, _ = score_individual_foods(
        bundle.weights,
        bundle.nutrients,
        bundle.food_metadata,
        AnchoredScoringConfig(mask_version=PRIMARY_MASK_VERSION),
    )
    centered_matrix = _score_to_matrix(centered, bundle)
    uncentered_matrix = _uncentered_gmnps_matrix(bundle)
    unanchored = _scale_to_1_100(bundle.weights.to_numpy() @ bundle.nutrients.to_numpy().T)

    rng = np.random.default_rng(seed + 1)
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
    shuffled_weights = bundle.weights.sample(frac=1.0, random_state=seed + 1).copy()
    shuffled_weights.index = bundle.weights.index
    shuffled_scored, _, _ = score_individual_foods(
        shuffled_weights,
        bundle.nutrients,
        bundle.food_metadata,
        AnchoredScoringConfig(mask_version=PRIMARY_MASK_VERSION),
    )

    variants = [
        ("FCS2 only", fcs_only, "static prior"),
        ("anchored GMNPS", centered_matrix, "full model"),
        ("no centering", uncentered_matrix, "remove food-level centering"),
        ("unanchored microbiome score", unanchored, "remove NPS prior"),
        ("random microbiome", _score_to_matrix(random_scored, bundle), "negative control"),
        ("shuffled microbiome", _score_to_matrix(shuffled_scored, bundle), "negative control"),
    ]
    rows = []
    for name, pred, ablation in variants:
        row = _benchmark_row(name, pred, truth, fcs2)
        row["ablation"] = ablation
        rows.append(row)
    return pd.DataFrame(rows)
