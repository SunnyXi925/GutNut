"""Correctly specified synthetic positive-control for locked attribute GMNPS.

The data-generating truth is constructed before model scoring as the sum of a
universal food-quality component, programmed microbiome-conditioned attribute
effects and a separate raw Gaussian noise stream. This positive-control does not
provide clinical or external validity evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from math import isfinite
from pathlib import Path
import platform
from typing import Mapping

import numpy as np
import pandas as pd

# Concrete-module imports deliberately avoid the dirty package re-export layer.
from gmnps.scoring.attribute_calibration import (
    ATTRIBUTE_POINT_FRACTION_MODES,
    attribute_response,
    calibrate_attribute_points,
)
from gmnps.scoring.attribute_gmnps import (
    FCS2_FNDDS_REGISTRY_VERSION,
    PRIMARY_RECOMPUTED_DOMAINS,
    AttributeGMNPSConfig,
    FoodAttributeBundle,
    fit_attribute_gmnps,
)
from gmnps.scoring.attribute_recomposition import (
    FINAL_DEVIATION_CAP_MODES,
    compose_personalized_fcs,
    decompose_official_baseline,
    recompute_personalized_domains,
)
from gmnps.scoring.fcs2_attribute_mapping import (
    PRIMARY_ATTRIBUTE_MAPPINGS,
    PRIMARY_MAPPING_VERSION,
    reconstruction_status_table,
)
from gmnps.scoring.fcs2_attribute_rules import FCS2_RULES, aggregate_domains
from gmnps.scoring.masks import (
    ORIGINAL_MASK_VERSION,
    PRIMARY_EXCLUDED_FROM_CHANNELS,
    PRIMARY_MASK_VERSION,
    build_channel_vectors,
)


EVIDENCE_ROLE = "correctly_specified_synthetic_positive_control"
DATA_CLASS = "synthetic"
SIMULATOR_VERSION = "attribute-synthetic-twin-v2"
_MAD_NORMAL_CONSISTENCY = 1.4826
_FCS_PER_UNSCALED_POINT = 99.0 / 47.1
_EFFECT_ROWS = tuple(row for row in PRIMARY_ATTRIBUTE_MAPPINGS if row.role == "effect")
_EFFECT_NUTRIENTS = tuple(dict.fromkeys(row.nutrient for row in _EFFECT_ROWS))
_TARGET_ATTRIBUTES = tuple(dict.fromkeys(row.attribute for row in _EFFECT_ROWS))
_ACTIVE_ATTRIBUTES = tuple(name for name, rule in FCS2_RULES.items() if rule.active)
_PROXY_NUTRIENTS = tuple(sorted(PRIMARY_EXCLUDED_FROM_CHANNELS))
_RNG_NAMES = (
    "food_quality",
    "attribute_exposures",
    "capacities",
    "excluded_proxies",
    "noise",
    "random_microbiome",
    "sattolo_derangement",
    "bootstrap",
)
_COMPONENT_TOTALS = {
    "folate_dfe_b9": "Folate, DFE (mcg_DFE)",
    "vitamin_a_rae": "Vitamin A, RAE (mcg_RAE)",
}
_RATIO_SIDE_INPUTS = {
    "fiber_to_carbohydrate_ratio": ("Carbohydrate (g)",),
    "potassium_to_sodium_ratio": ("Sodium (mg)",),
    "unsaturated_to_saturated_fat_ratio": (
        "Total Fat (g)",
        "Fatty acids, total monounsaturated (g)",
        "Fatty acids, total polyunsaturated (g)",
    ),
}
_CODE_FIXED_CHECKS = {
    "programmed_mapping_recovery_vs_fcs": (
        "locked programmed-mapping residual RMSE strictly below FCS baseline"
    ),
    "random_assignment_loses_recovery": (
        "locked residual RMSE strictly below random microbiome assignment"
    ),
    "sattolo_assignment_loses_recovery": (
        "locked residual RMSE strictly below Sattolo-deranged microbiome assignment"
    ),
    "locked_preserves_universal_rank": "mean universal-rank Spearman >= 0.90",
    "expert_mask_excludes_programmed_proxies": (
        "expert-mask excluded-proxy contribution is zero and original-mask contribution is positive"
    ),
}
_PUBLISHABLE_INTERPRETATION = (
    "Locked implementation recovered the programmed mapping in a correctly specified "
    "synthetic positive-control and lost recovery after random or Sattolo-deranged "
    "assignment."
)


@dataclass(frozen=True)
class AttributeSyntheticTwinConfig:
    n_individuals: int = 48
    n_foods: int = 24
    seed: int = 1701
    noise_sd: float = 0.35
    effect_scale: float = 1.0
    proxy_beta_scale: float = 3.5

    def __post_init__(self) -> None:
        if not isinstance(self.n_individuals, int) or self.n_individuals < 4:
            raise ValueError("n_individuals must be an integer of at least four")
        if not isinstance(self.n_foods, int) or self.n_foods < 4:
            raise ValueError("n_foods must be an integer of at least four")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("noise_sd", "effect_scale", "proxy_beta_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")


@dataclass(frozen=True)
class AttributeSyntheticExperimentConfig:
    seeds: tuple[int, ...] = (1701, 1702, 1703, 1704, 1705)
    n_individuals: int = 48
    n_foods: int = 24
    noise_sd: float = 0.35
    effect_scale: float = 1.0
    proxy_beta_scale: float = 3.5
    bootstrap_replicates: int = 200

    def __post_init__(self) -> None:
        if not isinstance(self.seeds, tuple) or len(self.seeds) < 3:
            raise ValueError("seeds must contain at least three code-fixed replicates")
        if len(set(self.seeds)) != len(self.seeds) or any(
            not isinstance(seed, int) or seed < 0 for seed in self.seeds
        ):
            raise ValueError("seeds must be unique nonnegative integers")
        AttributeSyntheticTwinConfig(
            n_individuals=self.n_individuals,
            n_foods=self.n_foods,
            seed=self.seeds[0],
            noise_sd=self.noise_sd,
            effect_scale=self.effect_scale,
            proxy_beta_scale=self.proxy_beta_scale,
        )
        if not isinstance(self.bootstrap_replicates, int) or self.bootstrap_replicates < 20:
            raise ValueError("bootstrap_replicates must be an integer of at least 20")


@dataclass(frozen=True)
class AttributeSyntheticTwinBundle:
    config: AttributeSyntheticTwinConfig
    development_beta: pd.DataFrame
    score_beta: pd.DataFrame
    normalized_exposures: pd.DataFrame
    food_bundle: FoodAttributeBundle
    capacities: pd.DataFrame
    pair_design: pd.DataFrame
    universal_component: pd.Series
    attribute_effects: pd.DataFrame
    noiseless_truth: pd.DataFrame
    raw_gaussian_noise: pd.DataFrame
    effective_noise: pd.DataFrame
    clipping_indicator: pd.DataFrame
    observed_response: pd.DataFrame
    rng_streams: Mapping[str, int]
    truth_definition_sha256: str


@dataclass(frozen=True)
class AttributeSyntheticBenchmark:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    attribute_cap_diagnostics: pd.DataFrame
    final_cap_diagnostics: pd.DataFrame
    seed: int


@dataclass(frozen=True)
class _LockedSyntheticScore:
    scores: pd.Series
    deltas: pd.Series
    attribute_diagnostics: pd.DataFrame


@dataclass(frozen=True)
class AttributeSyntheticExperiment:
    replicate_metrics: pd.DataFrame
    summary: pd.DataFrame
    success_checks: pd.DataFrame
    manifest: dict[str, object]
    passed: bool


def _annotated_csv_bytes(
    frame: pd.DataFrame,
    *,
    seeds: tuple[int, ...],
) -> tuple[bytes, str]:
    core = frame.copy()
    core_bytes = core.to_csv(index=False, lineterminator="\n").encode("utf-8")
    payload_sha256 = sha256(core_bytes).hexdigest()
    annotated = core.copy()
    annotated["evidence_role"] = EVIDENCE_ROLE
    annotated["data_class"] = DATA_CLASS
    annotated["seed_set"] = json.dumps(list(seeds), separators=(",", ":"))
    annotated["payload_sha256"] = payload_sha256
    return annotated.to_csv(index=False, lineterminator="\n").encode("utf-8"), payload_sha256


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _source_and_environment_manifest() -> tuple[dict[str, str], dict[str, object]]:
    scoring_root = Path(__file__).resolve().parents[1] / "scoring"
    source_paths = {
        "simulator": Path(__file__).resolve(),
        "attribute_calibration": scoring_root / "attribute_calibration.py",
        "attribute_gmnps": scoring_root / "attribute_gmnps.py",
        "attribute_recomposition": scoring_root / "attribute_recomposition.py",
        "fcs2_attribute_mapping": scoring_root / "fcs2_attribute_mapping.py",
        "fcs2_attribute_rules": scoring_root / "fcs2_attribute_rules.py",
        "masks": scoring_root / "masks.py",
    }
    source_hashes = {
        name: sha256(path.read_bytes()).hexdigest() for name, path in source_paths.items()
    }
    environment = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "packages": {
            package: _package_version(package)
            for package in ("numpy", "pandas", "scipy", "scikit-learn")
        },
    }
    return source_hashes, environment


def _frame_hash(frame: pd.DataFrame | pd.Series) -> str:
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    payload = {
        "columns": [str(column) for column in frame.columns],
        "index": [
            list(item) if isinstance(item, tuple) else str(item) for item in frame.index
        ],
        "values": frame.astype(object).where(pd.notna(frame), None).values.tolist(),
    }
    return _canonical_hash(payload)


def _spawn_streams(seed: int) -> tuple[dict[str, int], dict[str, np.random.Generator]]:
    children = np.random.SeedSequence(seed).spawn(len(_RNG_NAMES))
    stream_seeds = {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(_RNG_NAMES, children)
    }
    return stream_seeds, {
        name: np.random.default_rng(stream_seed)
        for name, stream_seed in stream_seeds.items()
    }


def _normalization_target(attribute: str) -> float:
    rule = FCS2_RULES[attribute]
    values = [
        abs(float(value))
        for value in (rule.low_target, rule.high_target)
        if value is not None and float(value) != 0.0
    ]
    if not values:
        raise ValueError(f"attribute {attribute} has no nonzero normalization target")
    return max(values)


def _exposure_columns() -> tuple[str, ...]:
    columns = list(_EFFECT_NUTRIENTS)
    for total in _COMPONENT_TOTALS.values():
        if total not in columns:
            columns.append(total)
    for side_inputs in _RATIO_SIDE_INPUTS.values():
        for column in side_inputs:
            if column not in columns:
                columns.append(column)
    for proxy in _PROXY_NUTRIENTS:
        if proxy not in columns:
            columns.append(proxy)
    return tuple(columns)


def _generate_exposures(
    food_ids: pd.Index,
    quality: np.ndarray,
    exposure_rng: np.random.Generator,
    proxy_rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = _exposure_columns()
    exposures = pd.DataFrame(0.0, index=food_ids, columns=columns)
    normalized = pd.DataFrame(0.0, index=food_ids, columns=columns)
    nutrient_channel = {row.nutrient: row.channel for row in _EFFECT_ROWS}
    nutrient_attributes: dict[str, list[str]] = {}
    for row in _EFFECT_ROWS:
        nutrient_attributes.setdefault(row.nutrient, []).append(row.attribute)
    for nutrient in _EFFECT_NUTRIENTS:
        attributes = nutrient_attributes[nutrient]
        target = max(_normalization_target(attribute) for attribute in attributes)
        direction = 0.22 if nutrient_channel[nutrient] == "MAC" else -0.12
        intensity = np.exp(
            direction * quality + exposure_rng.normal(0.0, 0.28, len(food_ids))
        )
        intensity = np.clip(intensity, 0.15, 1.75)
        exposures[nutrient] = target * intensity
        normalized[nutrient] = intensity

    proxy_base = np.exp(0.35 * quality + proxy_rng.normal(0.0, 0.35, len(food_ids)))
    for index, proxy in enumerate(_PROXY_NUTRIENTS):
        proxy_values = np.clip(proxy_base * (1.0 + 0.15 * index), 0.05, None)
        if proxy not in _EFFECT_NUTRIENTS:
            exposures[proxy] = proxy_values
        normalized[proxy] = proxy_values
    # Enforce ratio gates and component-total contracts after proxy generation.
    exposures["Carbohydrate (g)"] = np.maximum(exposures["Carbohydrate (g)"], 8.0)
    exposures["Sodium (mg)"] = np.maximum(
        exposures.get("Sodium (mg)", pd.Series(0.0, index=food_ids)), 80.0
    )
    exposures["Potassium (mg)"] = np.maximum(exposures["Potassium (mg)"], 120.0)
    saturated = np.maximum(exposures["Fatty acids, total saturated (g)"], 0.8)
    exposures["Fatty acids, total saturated (g)"] = saturated
    exposures["Fatty acids, total monounsaturated (g)"] = 2.2 + np.exp(
        exposure_rng.normal(0.0, 0.25, len(food_ids))
    )
    exposures["Fatty acids, total polyunsaturated (g)"] = 1.4 + np.exp(
        exposure_rng.normal(0.0, 0.25, len(food_ids))
    )
    component_fat = (
        saturated
        + exposures["Fatty acids, total monounsaturated (g)"]
        + exposures["Fatty acids, total polyunsaturated (g)"]
    )
    exposures["Total Fat (g)"] = np.maximum(component_fat + 1.0, 3.0)
    exposures["Folate, DFE (mcg_DFE)"] = exposures["Folate, food (mcg)"] * 1.25
    exposures["Vitamin A, RAE (mcg_RAE)"] = exposures["Retinol (mcg)"] * 1.35
    for column in exposures.columns:
        if column not in normalized or normalized[column].eq(0.0).all():
            scale = max(float(exposures[column].median()), 1e-12)
            normalized[column] = exposures[column] / scale
    exposures.attrs["basis"] = "per_100_kcal"
    normalized.attrs["basis"] = "fixed_physical_target_no_population_centering"
    return exposures.astype(float), normalized.astype(float)


def _baseline_points(food_ids: pd.Index) -> pd.DataFrame:
    row = {
        attribute: (float(rule.low_points) + float(rule.high_points)) / 2.0
        for attribute, rule in FCS2_RULES.items()
        if rule.active
    }
    return pd.DataFrame(
        [row.copy() for _ in food_ids], index=food_ids, columns=_ACTIVE_ATTRIBUTES
    )


def _build_food_bundle(
    official_fcs: pd.Series,
    exposures: pd.DataFrame,
) -> FoodAttributeBundle:
    food_ids = official_fcs.index
    baseline = _baseline_points(food_ids)
    weights = pd.DataFrame(
        {
            attribute: [float(FCS2_RULES[attribute].weight)] * len(food_ids)
            for attribute in _ACTIVE_ATTRIBUTES
        },
        index=food_ids,
    )
    metadata = pd.DataFrame(
        {
            "food_name": [f"Synthetic food {index:03d}" for index in range(len(food_ids))],
            "food_group": ["synthetic_panel"] * len(food_ids),
            "is_dairy": [False] * len(food_ids),
            "FCS2": official_fcs,
        },
        index=food_ids,
    )
    registry_path = Path(__file__).resolve().parents[2] / "configs/fcs2_fndds_release_registry.json"
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    source_hashes = {
        "official_fcs": _frame_hash(official_fcs),
        "food_metadata": _frame_hash(metadata),
        "baseline_attribute_points": _frame_hash(baseline),
        "food_exposures": _frame_hash(exposures),
        "effective_attribute_weights": _frame_hash(weights),
        "input_manifest": _canonical_hash(
            {"data_class": DATA_CLASS, "evidence_role": EVIDENCE_ROLE}
        ),
    }
    return FoodAttributeBundle(
        official_fcs=official_fcs,
        baseline_points=baseline,
        food_exposures=exposures,
        effective_attribute_weights=weights,
        food_metadata=metadata,
        fndds_releases=("FNDDS 2021-2023",),
        source_hashes=source_hashes,
        nutrient_units={column: "synthetic_unit_per_100_kcal" for column in exposures},
        exposure_basis="per_100_kcal",
        reconstruction_status=reconstruction_status_table(),
        production_label="non-production",
        registry_version=FCS2_FNDDS_REGISTRY_VERSION,
        release_registry_snapshot_sha256=sha256(registry_bytes).hexdigest(),
        release_registry_canonical_release_set=tuple(registry["canonical_release_set"]),
        food_source_linkage_sha256=_canonical_hash(list(food_ids)),
    )


def _mapping_normalized_exposure(row: object, exposure: pd.Series) -> float:
    attribute = row.attribute
    nutrient = row.nutrient
    if attribute == "fiber_to_carbohydrate_ratio":
        return float(exposure[nutrient]) / _normalization_target("total_fiber")
    if attribute == "potassium_to_sodium_ratio":
        return float(exposure[nutrient]) / _normalization_target("potassium")
    if attribute == "unsaturated_to_saturated_fat_ratio":
        total = sum(
            float(exposure[column])
            for column in (
                "Fatty acids, total saturated (g)",
                "Fatty acids, total monounsaturated (g)",
                "Fatty acids, total polyunsaturated (g)",
            )
        )
        return -float(exposure[nutrient]) / total
    return float(exposure[nutrient]) / _normalization_target(attribute)


def simulate_attribute_synthetic_twin(
    config: AttributeSyntheticTwinConfig | None = None,
) -> AttributeSyntheticTwinBundle:
    """Generate programmed attribute truth without calling a GMNPS scorer."""

    resolved = config or AttributeSyntheticTwinConfig()
    streams, rng = _spawn_streams(resolved.seed)
    individual_ids = pd.Index(
        [f"person_{index:03d}" for index in range(resolved.n_individuals)],
        name="individual_id",
    )
    food_ids = pd.Index(
        [f"food_{index:03d}" for index in range(resolved.n_foods)], name="food_id"
    )
    quality = rng["food_quality"].normal(0.0, 1.0, resolved.n_foods)
    official_values = 50.0 + 22.0 * np.tanh(quality / 1.5)
    official_fcs = pd.Series(official_values, index=food_ids, name="FCS2")
    exposures, normalized = _generate_exposures(
        food_ids,
        quality,
        rng["attribute_exposures"],
        rng["excluded_proxies"],
    )

    capacity_mac = rng["capacities"].normal(0.0, 1.0, resolved.n_individuals)
    capacity_lipid = rng["capacities"].normal(0.0, 1.0, resolved.n_individuals)
    proxy_capacity = rng["capacities"].normal(0.0, 1.0, resolved.n_individuals)
    capacities = pd.DataFrame(
        {
            "capacity_MAC": capacity_mac,
            "capacity_LIPID": capacity_lipid,
            "excluded_proxy_capacity": proxy_capacity,
        },
        index=individual_ids,
    )
    score_columns = tuple(dict.fromkeys((*_EFFECT_NUTRIENTS, *_PROXY_NUTRIENTS)))
    score_beta = pd.DataFrame(0.0, index=individual_ids, columns=score_columns)
    nutrient_channel = {row.nutrient: row.channel for row in _EFFECT_ROWS}
    for nutrient in _EFFECT_NUTRIENTS:
        source = capacity_mac if nutrient_channel[nutrient] == "MAC" else capacity_lipid
        score_beta[nutrient] = resolved.effect_scale * source
    for nutrient in _PROXY_NUTRIENTS:
        if nutrient in score_beta:
            score_beta[nutrient] = (
                resolved.effect_scale * resolved.proxy_beta_scale * proxy_capacity
            )

    dev_axis = np.array([-2, -1, 0, 1, 2], dtype=float) / _MAD_NORMAL_CONSISTENCY
    development_beta = pd.DataFrame(
        np.repeat(dev_axis[:, None], len(score_columns), axis=1),
        index=pd.Index([f"dev_{index}" for index in range(len(dev_axis))], name="individual_id"),
        columns=score_columns,
    )
    pair_index = pd.MultiIndex.from_product(
        [individual_ids, food_ids], names=["individual_id", "food_id"]
    )
    universal = pd.Series(
        np.tile(official_fcs.to_numpy(dtype=float), resolved.n_individuals),
        index=pair_index,
        name="response",
    )
    pair_design = pd.DataFrame(
        {
            "capacity_MAC": np.repeat(capacity_mac, resolved.n_foods),
            "capacity_LIPID": np.repeat(capacity_lipid, resolved.n_foods),
            "excluded_proxy_capacity": np.repeat(proxy_capacity, resolved.n_foods),
        },
        index=pair_index,
    )

    baseline_template = {
        attribute: (float(FCS2_RULES[attribute].low_points) + float(FCS2_RULES[attribute].high_points))
        / 2.0
        for attribute in _ACTIVE_ATTRIBUTES
    }
    effective_weights = {
        attribute: float(FCS2_RULES[attribute].weight)
        for attribute in _ACTIVE_ATTRIBUTES
    }
    baseline_domain_sum = sum(
        aggregate_domains(
            baseline_template, effective_attribute_weights=effective_weights
        ).values()
    )
    effect_rows: list[dict[str, object]] = []
    for individual_position, individual_id in enumerate(individual_ids):
        channel_latent = {
            "MAC": float(
                np.tanh(resolved.effect_scale * capacity_mac[individual_position] / 2.0)
            ),
            "LIPID": float(
                np.tanh(resolved.effect_scale * capacity_lipid[individual_position] / 2.0)
            ),
        }
        for food_id in food_ids:
            exposure = exposures.loc[food_id]
            response_by_attribute = {attribute: 0.0 for attribute in _TARGET_ATTRIBUTES}
            for mapping in _EFFECT_ROWS:
                response_by_attribute[mapping.attribute] += (
                    channel_latent[mapping.channel]
                    * _mapping_normalized_exposure(mapping, exposure)
                    * float(mapping.allocation_weight)
                )
            personalized_points = dict(baseline_template)
            attribute_state: dict[str, tuple[float, float]] = {}
            for attribute, response in response_by_attribute.items():
                rule = FCS2_RULES[attribute]
                point_range = abs(float(rule.high_points) - float(rule.low_points))
                point_delta = (
                    ATTRIBUTE_POINT_FRACTION_MODES["primary"]
                    * point_range
                    * float(np.tanh(response / 2.0))
                )
                personalized_points[attribute] = float(
                    np.clip(
                        baseline_template[attribute] + point_delta,
                        min(float(rule.low_points), float(rule.high_points)),
                        max(float(rule.low_points), float(rule.high_points)),
                    )
                )
                attribute_state[attribute] = (response, point_delta)
            personalized_domain_sum = sum(
                aggregate_domains(
                    personalized_points,
                    effective_attribute_weights=effective_weights,
                ).values()
            )
            uncapped_total_effect = _FCS_PER_UNSCALED_POINT * (
                personalized_domain_sum - baseline_domain_sum
            )
            total_effect = float(np.clip(uncapped_total_effect, -12.0, 12.0))
            total_effect = float(
                np.clip(
                    float(official_fcs.loc[food_id]) + total_effect, 1.0, 100.0
                )
                - float(official_fcs.loc[food_id])
            )
            marginal_effects: dict[str, float] = {}
            for attribute in _TARGET_ATTRIBUTES:
                counterfactual = dict(personalized_points)
                counterfactual[attribute] = baseline_template[attribute]
                counterfactual_sum = sum(
                    aggregate_domains(
                        counterfactual,
                        effective_attribute_weights=effective_weights,
                    ).values()
                )
                marginal_effects[attribute] = _FCS_PER_UNSCALED_POINT * (
                    personalized_domain_sum - counterfactual_sum
                )
            interaction_residual = total_effect - sum(marginal_effects.values())
            allocation_denominator = sum(
                abs(attribute_state[attribute][1])
                for attribute in _TARGET_ATTRIBUTES
            )
            for attribute in _TARGET_ATTRIBUTES:
                response, point_delta = attribute_state[attribute]
                allocation = (
                    abs(point_delta) / allocation_denominator
                    if allocation_denominator > 1e-12
                    else 1.0 / len(_TARGET_ATTRIBUTES)
                )
                score_effect = marginal_effects[attribute] + interaction_residual * allocation
                effect_rows.append(
                    {
                        "individual_id": individual_id,
                        "food_id": food_id,
                        "attribute": attribute,
                        "channel": next(
                            row.channel
                            for row in _EFFECT_ROWS
                            if row.attribute == attribute
                        ),
                        "latent_attribute_response": response,
                        "true_attribute_point_delta": point_delta,
                        "true_effect": score_effect,
                    }
                )
    attribute_effects = pd.DataFrame.from_records(effect_rows).set_index(
        ["individual_id", "food_id", "attribute"]
    )
    total_effect = attribute_effects.groupby(level=["individual_id", "food_id"])[
        "true_effect"
    ].sum()
    noiseless_values = universal.add(total_effect).clip(1.0, 100.0)
    raw_noise = rng["noise"].normal(0.0, resolved.noise_sd, len(pair_index))
    preclip_values = noiseless_values + raw_noise
    observed_values = preclip_values.clip(1.0, 100.0)
    effective_noise = observed_values - noiseless_values
    noiseless_truth = noiseless_values.rename("response").to_frame()
    raw_gaussian_noise = pd.Series(
        raw_noise, index=pair_index, name="raw_gaussian_noise"
    ).to_frame()
    effective_noise_frame = effective_noise.rename("effective_noise").to_frame()
    clipping_indicator = preclip_values.ne(observed_values).rename("was_clipped").to_frame()
    observed_response = observed_values.rename("response").to_frame()
    truth_definition_sha256 = _canonical_hash(
        {
            "attribute_fraction": ATTRIBUTE_POINT_FRACTION_MODES["primary"],
            "attribute_mapping": PRIMARY_MAPPING_VERSION,
            "attribute_targets": list(_TARGET_ATTRIBUTES),
            "equation": (
                "FCS2 + sum_a[leave-one-attribute-out published-domain marginal + "
                "allocated top-k interaction residual], with 0.20 attribute-range and "
                "plus/minus 12 final caps, + separate_raw_gaussian_noise, then "
                "response-bound clipping"
            ),
            "simulator_version": SIMULATOR_VERSION,
        }
    )
    return AttributeSyntheticTwinBundle(
        config=resolved,
        development_beta=development_beta,
        score_beta=score_beta,
        normalized_exposures=normalized,
        food_bundle=_build_food_bundle(official_fcs, exposures),
        capacities=capacities,
        pair_design=pair_design,
        universal_component=universal,
        attribute_effects=attribute_effects,
        noiseless_truth=noiseless_truth,
        raw_gaussian_noise=raw_gaussian_noise,
        effective_noise=effective_noise_frame,
        clipping_indicator=clipping_indicator,
        observed_response=observed_response,
        rng_streams=streams,
        truth_definition_sha256=truth_definition_sha256,
    )


def sattolo_derangement(n: int, *, seed: int) -> np.ndarray:
    """Return one deterministic Sattolo cycle, hence no fixed points."""

    if not isinstance(n, int) or n < 2:
        raise ValueError("Sattolo derangement requires at least two items")
    values = np.arange(n)
    rng = np.random.default_rng(seed)
    for position in range(n - 1, 0, -1):
        swap = int(rng.integers(0, position))
        values[position], values[swap] = values[swap], values[position]
    if np.any(values == np.arange(n)):  # pragma: no cover - Sattolo invariant
        raise RuntimeError("Sattolo implementation produced a fixed point")
    return values


def _locked_score(
    bundle: AttributeSyntheticTwinBundle,
    beta: pd.DataFrame,
    *,
    attribute_mode: str = "primary",
    final_mode: str = "primary",
) -> _LockedSyntheticScore:
    config = AttributeGMNPSConfig(
        attribute_point_mode=attribute_mode,
        final_cap_mode=final_mode,
        development_smoke_test=True,
    )
    model = fit_attribute_gmnps(bundle.development_beta, config)
    responses = attribute_response(
        model.normalization_state,
        beta,
        bundle.food_bundle.food_exposures,
        mapping_version=config.mapping_version,
    )
    calibration = calibrate_attribute_points(
        bundle.food_bundle.baseline_points,
        responses,
        mode=config.attribute_point_mode,
        effective_attribute_weights=bundle.food_bundle.effective_attribute_weights,
    )
    selected_attributes = tuple(
        attribute
        for attribute in _ACTIVE_ATTRIBUTES
        if FCS2_RULES[attribute].domain in PRIMARY_RECOMPUTED_DOMAINS
    )
    baseline = decompose_official_baseline(
        bundle.food_bundle.official_fcs,
        bundle.food_bundle.baseline_points.loc[:, selected_attributes],
        recomputed_domains=PRIMARY_RECOMPUTED_DOMAINS,
        effective_attribute_weights=bundle.food_bundle.effective_attribute_weights.loc[
            :, selected_attributes
        ],
    )
    recomputed = recompute_personalized_domains(baseline, calibration)
    final = compose_personalized_fcs(baseline, recomputed, cap_mode=config.final_cap_mode)
    diagnostics = calibration.diagnostics.reset_index().rename(
        columns={"point_delta": "attribute_point_delta"}
    )
    return _LockedSyntheticScore(
        scores=final.scores,
        deltas=final.deltas,
        attribute_diagnostics=diagnostics,
    )


def _matrix_from_locked(
    result: _LockedSyntheticScore, bundle: AttributeSyntheticTwinBundle
) -> np.ndarray:
    matrix = result.scores.unstack("food_id")
    return matrix.loc[
        bundle.score_beta.index, bundle.food_bundle.official_fcs.index
    ].to_numpy(dtype=float)


def _legacy_form_matrix(
    bundle: AttributeSyntheticTwinBundle,
    beta: pd.DataFrame,
    mask_version: str,
) -> tuple[np.ndarray, float]:
    columns = [
        column
        for column in beta.columns
        if column in bundle.normalized_exposures.columns
    ]
    masks = build_channel_vectors(columns, mask_version)
    selected = np.asarray(masks["mac"], dtype=bool) | np.asarray(
        masks["lipid"], dtype=bool
    )
    selected_columns = np.asarray(columns, dtype=object)[selected].tolist()
    raw = (
        beta[selected_columns].to_numpy(dtype=float)
        @ bundle.normalized_exposures[selected_columns].to_numpy(dtype=float).T
    ) / max(np.sqrt(len(selected_columns)), 1.0)
    delta = 6.0 * np.tanh(raw / 2.0)
    fcs = bundle.food_bundle.official_fcs.to_numpy(dtype=float)
    score = np.clip(fcs[None, :] + delta, 1.0, 100.0)
    proxy_columns = [column for column in selected_columns if column in _PROXY_NUTRIENTS]
    if proxy_columns:
        proxy = (
            beta[proxy_columns].to_numpy(dtype=float)
            @ bundle.normalized_exposures[proxy_columns].to_numpy(dtype=float).T
        ) / max(np.sqrt(len(selected_columns)), 1.0)
        fraction = float(np.abs(proxy).sum() / max(np.abs(raw).sum(), 1e-12))
    else:
        fraction = 0.0
    return score, fraction


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left.ravel()).rank(method="average")
    right_rank = pd.Series(right.ravel()).rank(method="average")
    if left_rank.nunique() < 2 or right_rank.nunique() < 2:
        return 0.0
    value = left_rank.corr(right_rank)
    return 0.0 if pd.isna(value) else float(value)


def _metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    fcs: np.ndarray,
) -> dict[str, float]:
    pred_residual = prediction - fcs[None, :]
    truth_residual = truth - fcs[None, :]
    per_individual_rmse = np.sqrt(np.mean((pred_residual - truth_residual) ** 2, axis=1))
    individual_spearman = np.array(
        [_spearman(pred_residual[index], truth_residual[index]) for index in range(len(prediction))]
    )
    x = pred_residual.ravel()
    y = truth_residual.ravel()
    if np.std(x) > 1e-12:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = 0.0, float(np.mean(y))
    return {
        "residual_rmse": float(np.sqrt(np.mean((pred_residual - truth_residual) ** 2))),
        "residual_spearman": _spearman(pred_residual, truth_residual),
        "mean_individual_residual_rmse": float(per_individual_rmse.mean()),
        "mean_individual_residual_spearman": float(individual_spearman.mean()),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "universal_rank_preservation_spearman": _spearman(
            prediction.mean(axis=0), fcs
        ),
    }


def _bootstrap_metric_intervals(
    prediction: np.ndarray,
    truth: np.ndarray,
    fcs: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    rmse: list[float] = []
    spearman: list[float] = []
    for _ in range(replicates):
        sampled = rng.integers(0, prediction.shape[0], prediction.shape[0])
        values = _metrics(prediction[sampled], truth[sampled], fcs)
        if isfinite(values["residual_rmse"]):
            rmse.append(values["residual_rmse"])
        if isfinite(values["residual_spearman"]):
            spearman.append(values["residual_spearman"])
    return {
        "bootstrap_replicates_requested": replicates,
        "bootstrap_replicates_valid_rmse": len(rmse),
        "bootstrap_replicates_valid_spearman": len(spearman),
        "residual_rmse_ci_lower": float(np.quantile(rmse, 0.025)),
        "residual_rmse_ci_upper": float(np.quantile(rmse, 0.975)),
        "residual_spearman_ci_lower": float(np.quantile(spearman, 0.025)),
        "residual_spearman_ci_upper": float(np.quantile(spearman, 0.975)),
    }


def benchmark_attribute_synthetic_twin(
    bundle: AttributeSyntheticTwinBundle,
    *,
    bootstrap_replicates: int = 200,
    include_sensitivities: bool = False,
) -> AttributeSyntheticBenchmark:
    """Benchmark identical person-food opportunities without data-based centering."""

    if not isinstance(bundle, AttributeSyntheticTwinBundle):
        raise TypeError("bundle must be an AttributeSyntheticTwinBundle")
    if not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 20:
        raise ValueError("bootstrap_replicates must be an integer of at least 20")
    if not isinstance(include_sensitivities, bool):
        raise TypeError("include_sensitivities must be boolean")
    primary = _locked_score(bundle, bundle.score_beta)
    fcs = bundle.food_bundle.official_fcs.to_numpy(dtype=float)
    shape = (bundle.config.n_individuals, bundle.config.n_foods)
    truth = bundle.observed_response["response"].to_numpy(dtype=float).reshape(shape)
    fcs_matrix = np.broadcast_to(fcs[None, :], shape).copy()

    random_rng = np.random.default_rng(bundle.rng_streams["random_microbiome"])
    random_beta = pd.DataFrame(
        random_rng.normal(0.0, 1.0, bundle.score_beta.shape),
        index=bundle.score_beta.index,
        columns=bundle.score_beta.columns,
    )
    derangement = sattolo_derangement(
        bundle.config.n_individuals, seed=bundle.rng_streams["sattolo_derangement"]
    )
    deranged_beta = bundle.score_beta.iloc[derangement].copy()
    deranged_beta.index = bundle.score_beta.index
    random_locked = _locked_score(bundle, random_beta)
    deranged_locked = _locked_score(bundle, deranged_beta)
    expert_legacy, expert_proxy_fraction = _legacy_form_matrix(
        bundle, bundle.score_beta, PRIMARY_MASK_VERSION
    )
    original_legacy, original_proxy_fraction = _legacy_form_matrix(
        bundle, bundle.score_beta, ORIGINAL_MASK_VERSION
    )
    expert_raw_columns = [
        column
        for column in bundle.score_beta.columns
        if column in build_channel_vectors(bundle.score_beta.columns, PRIMARY_MASK_VERSION)[
            "mac_nutrients"
        ]
        or column
        in build_channel_vectors(bundle.score_beta.columns, PRIMARY_MASK_VERSION)[
            "lipid_nutrients"
        ]
    ]
    expert_raw = (
        bundle.score_beta[expert_raw_columns].to_numpy(dtype=float)
        @ bundle.normalized_exposures[expert_raw_columns].to_numpy(dtype=float).T
    ) / max(np.sqrt(len(expert_raw_columns)), 1.0)
    unanchored = np.clip(50.0 + 12.0 * np.tanh(expert_raw / 2.0), 1.0, 100.0)
    prediction_matrices = {
        "fcs_baseline": fcs_matrix,
        "locked_attribute_gmnps": _matrix_from_locked(primary, bundle),
        "legacy_final_score_offset": expert_legacy,
        "unanchored_microbiome_score": unanchored,
        "random_microbiome": _matrix_from_locked(random_locked, bundle),
        "sattolo_deranged_microbiome": _matrix_from_locked(deranged_locked, bundle),
        "original_mask_legacy_offset": original_legacy,
        "expert_revised_mask_legacy_offset": expert_legacy,
    }
    proxy_fractions = {
        "original_mask_legacy_offset": original_proxy_fraction,
        "expert_revised_mask_legacy_offset": expert_proxy_fraction,
    }
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    pair_index = bundle.observed_response.index
    for comparator, matrix in prediction_matrices.items():
        frame = pair_index.to_frame(index=False)
        frame["comparator"] = comparator
        frame["prediction"] = matrix.ravel()
        frame["FCS2"] = fcs_matrix.ravel()
        frame["observed_synthetic_response"] = truth.ravel()
        prediction_rows.append(frame)
        metric_rows.append(
            {
                "comparator": comparator,
                **_metrics(matrix, truth, fcs),
                **_bootstrap_metric_intervals(
                    matrix,
                    truth,
                    fcs,
                    replicates=bootstrap_replicates,
                    seed=(
                        bundle.rng_streams["bootstrap"]
                        + list(prediction_matrices).index(comparator) * 10_007
                    ),
                ),
                "n_individuals": bundle.config.n_individuals,
                "n_foods": bundle.config.n_foods,
                "n_individual_food_pairs": len(pair_index),
                "seed": bundle.config.seed,
                "uses_population_or_per_food_centering": False,
                "excluded_proxy_contribution_fraction": proxy_fractions.get(
                    comparator, 0.0
                ),
                "evidence_role": EVIDENCE_ROLE,
                "data_class": DATA_CLASS,
            }
        )

    attribute_diagnostics: list[pd.DataFrame] = []
    final_diagnostics: list[pd.DataFrame] = []
    scored_cache: dict[tuple[str, str], object] = {("primary", "primary"): primary}
    attribute_modes = (
        tuple(ATTRIBUTE_POINT_FRACTION_MODES) if include_sensitivities else ("primary",)
    )
    final_modes = (
        tuple(FINAL_DEVIATION_CAP_MODES) if include_sensitivities else ("primary",)
    )
    for mode in attribute_modes:
        key = (mode, "primary")
        scored = scored_cache.get(key) or _locked_score(
            bundle, bundle.score_beta, attribute_mode=mode
        )
        scored_cache[key] = scored
        attribution = scored.attribute_diagnostics.copy()
        attribution["attribute_point_mode"] = mode
        attribute_diagnostics.append(attribution)
    for mode in final_modes:
        key = ("primary", mode)
        scored = scored_cache.get(key) or _locked_score(
            bundle, bundle.score_beta, final_mode=mode
        )
        scored_cache[key] = scored
        diagnostic = scored.deltas.rename("GMNPS_delta").reset_index()
        diagnostic["final_cap_mode"] = mode
        final_diagnostics.append(diagnostic)
    return AttributeSyntheticBenchmark(
        predictions=pd.concat(prediction_rows, ignore_index=True),
        metrics=pd.DataFrame.from_records(metric_rows),
        attribute_cap_diagnostics=pd.concat(attribute_diagnostics, ignore_index=True),
        final_cap_diagnostics=pd.concat(final_diagnostics, ignore_index=True),
        seed=bundle.config.seed,
    )


def run_attribute_synthetic_experiment(
    config: AttributeSyntheticExperimentConfig | None = None,
) -> AttributeSyntheticExperiment:
    """Run code-fixed independent seeds and summarize across true replicates."""

    resolved = config or AttributeSyntheticExperimentConfig()
    replicate_frames = []
    truth_hashes: list[str] = []
    rng_manifests: dict[str, Mapping[str, int]] = {}
    for seed in resolved.seeds:
        twin_config = AttributeSyntheticTwinConfig(
            n_individuals=resolved.n_individuals,
            n_foods=resolved.n_foods,
            seed=seed,
            noise_sd=resolved.noise_sd,
            effect_scale=resolved.effect_scale,
            proxy_beta_scale=resolved.proxy_beta_scale,
        )
        bundle = simulate_attribute_synthetic_twin(twin_config)
        benchmark = benchmark_attribute_synthetic_twin(
            bundle, bootstrap_replicates=resolved.bootstrap_replicates
        )
        replicate_frames.append(benchmark.metrics)
        truth_hashes.append(bundle.truth_definition_sha256)
        rng_manifests[str(seed)] = bundle.rng_streams
    replicate_metrics = pd.concat(replicate_frames, ignore_index=True)
    summary_rows: list[dict[str, object]] = []
    metric_names = (
        "residual_rmse",
        "residual_spearman",
        "mean_individual_residual_rmse",
        "mean_individual_residual_spearman",
        "calibration_slope",
        "calibration_intercept",
        "universal_rank_preservation_spearman",
        "excluded_proxy_contribution_fraction",
    )
    for comparator, frame in replicate_metrics.groupby("comparator", sort=False):
        for metric in metric_names:
            values = pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy(float)
            summary_rows.append(
                {
                    "comparator": comparator,
                    "metric": metric,
                    "estimate": float(values.mean()),
                    "ci_lower": float(np.quantile(values, 0.025)),
                    "ci_upper": float(np.quantile(values, 0.975)),
                    "replicates_requested": len(resolved.seeds),
                    "replicates_valid": len(values),
                    "seeds": json.dumps(list(resolved.seeds), separators=(",", ":")),
                    "evidence_role": EVIDENCE_ROLE,
                    "data_class": DATA_CLASS,
                }
            )
    summary = pd.DataFrame.from_records(summary_rows)
    means = replicate_metrics.groupby("comparator", sort=False).mean(numeric_only=True)
    checks = [
        (
            "programmed_mapping_recovery_vs_fcs",
            means.loc["locked_attribute_gmnps", "residual_rmse"]
            < means.loc["fcs_baseline", "residual_rmse"],
        ),
        (
            "random_assignment_loses_recovery",
            means.loc["locked_attribute_gmnps", "residual_rmse"]
            < means.loc["random_microbiome", "residual_rmse"],
        ),
        (
            "sattolo_assignment_loses_recovery",
            means.loc["locked_attribute_gmnps", "residual_rmse"]
            < means.loc["sattolo_deranged_microbiome", "residual_rmse"],
        ),
        (
            "locked_preserves_universal_rank",
            means.loc[
                "locked_attribute_gmnps", "universal_rank_preservation_spearman"
            ]
            >= 0.90,
        ),
        (
            "expert_mask_excludes_programmed_proxies",
            means.loc[
                "expert_revised_mask_legacy_offset",
                "excluded_proxy_contribution_fraction",
            ]
            == 0.0
            and means.loc[
                "original_mask_legacy_offset",
                "excluded_proxy_contribution_fraction",
            ]
            > 0.0,
        ),
    ]
    success_checks = pd.DataFrame(
        [
            {
                "criterion_id": identifier,
                "criterion": _CODE_FIXED_CHECKS[identifier],
                "passed": bool(passed),
            }
            for identifier, passed in checks
        ]
    )
    source_hashes, environment = _source_and_environment_manifest()
    manifest = {
        "schema_version": "attribute-synthetic-experiment-manifest-v2",
        "evidence_role": EVIDENCE_ROLE,
        "data_class": DATA_CLASS,
        "simulator_version": SIMULATOR_VERSION,
        "seeds": list(resolved.seeds),
        "rng_streams": rng_manifests,
        "replicates": len(resolved.seeds),
        "config": asdict(resolved),
        "config_sha256": _canonical_hash(asdict(resolved)),
        "truth_definition_sha256": sorted(set(truth_hashes))[0],
        "code_fixed_checks": _CODE_FIXED_CHECKS,
        "checks_code_fixed_in_same_release": True,
        "independently_preregistered": False,
        "formal_run_cap_modes": {
            "attribute_point_fraction": "primary_20_percent",
            "final_deviation": "primary_plus_or_minus_12",
        },
        "cap_sensitivities": (
            "10/30 percent attribute and +/-8/15 final caps are unit boundary tests only; "
            "they are not part of the frozen formal run"
        ),
        "source_sha256": source_hashes,
        "environment": environment,
        "clinical_or_external_validation": False,
        "interpretation": _PUBLISHABLE_INTERPRETATION,
        "interpretation_limit": (
            "Correctly specified synthetic positive-control only; not clinical, external, "
            "construct, or biological mask validation."
        ),
    }
    return AttributeSyntheticExperiment(
        replicate_metrics=replicate_metrics,
        summary=summary,
        success_checks=success_checks,
        manifest=manifest,
        passed=bool(success_checks["passed"].all()),
    )


def freeze_attribute_synthetic_experiment(
    experiment: AttributeSyntheticExperiment,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Write deterministic, explicitly synthetic source-data and hash manifest."""

    if not isinstance(experiment, AttributeSyntheticExperiment):
        raise TypeError("experiment must be an AttributeSyntheticExperiment")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(seed) for seed in experiment.manifest["seeds"])
    tables = {
        "replicate_metrics": experiment.replicate_metrics,
        "summary": experiment.summary,
        "success_checks": experiment.success_checks,
    }
    paths: dict[str, Path] = {}
    file_manifest: dict[str, dict[str, str]] = {}
    for name, frame in tables.items():
        payload, payload_sha256 = _annotated_csv_bytes(frame, seeds=seeds)
        path = output / f"synthetic_attribute_twin_{name}.csv"
        path.write_bytes(payload)
        paths[name] = path
        file_manifest[path.name] = {
            "file_sha256": sha256(payload).hexdigest(),
            "payload_sha256": payload_sha256,
        }

    config_payload = {
        "evidence_role": EVIDENCE_ROLE,
        "data_class": DATA_CLASS,
        "seeds": list(seeds),
        "config": experiment.manifest["config"],
        "config_sha256": experiment.manifest["config_sha256"],
    }
    config_payload["payload_sha256"] = _canonical_hash(config_payload)
    config_bytes = (
        json.dumps(config_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    config_path = output / "synthetic_attribute_twin_config.json"
    config_path.write_bytes(config_bytes)
    paths["config"] = config_path
    file_manifest[config_path.name] = {
        "file_sha256": sha256(config_bytes).hexdigest(),
        "payload_sha256": str(config_payload["payload_sha256"]),
    }

    manifest_payload = dict(experiment.manifest)
    manifest_payload.update(
        {
            "passed": experiment.passed,
            "files": file_manifest,
            "gate_implication": "none_direct_external_validity_remains_fail_closed",
        }
    )
    manifest_payload["payload_sha256"] = _canonical_hash(manifest_payload)
    manifest_bytes = (
        json.dumps(manifest_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    manifest_path = output / "synthetic_attribute_twin_manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    paths["manifest"] = manifest_path
    return paths


__all__ = [
    "AttributeSyntheticBenchmark",
    "AttributeSyntheticExperiment",
    "AttributeSyntheticExperimentConfig",
    "AttributeSyntheticTwinBundle",
    "AttributeSyntheticTwinConfig",
    "benchmark_attribute_synthetic_twin",
    "freeze_attribute_synthetic_experiment",
    "run_attribute_synthetic_experiment",
    "sattolo_derangement",
    "simulate_attribute_synthetic_twin",
]
