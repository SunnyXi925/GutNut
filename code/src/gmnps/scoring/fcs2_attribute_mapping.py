"""Expert-reviewed nutrient mappings for attribute-level GMNPS calibration."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from math import isfinite
from types import MappingProxyType
from typing import Iterable, NamedTuple

import pandas as pd

from gmnps.scoring.fcs2_attribute_rules import (
    FCS2_RULES,
    ratio_gate_passes_from_exposures,
)


PRIMARY_MAPPING_VERSION = "expert_reviewed_attribute_mapping_v1"
CARBOHYDRATE_PROXY_MAPPING_VERSION = "carbohydrate_proxy"

_EFFECT_ROLE = "effect"
_ZERO_ALLOCATION_ROLES = frozenset(
    {"applicability_only", "explanatory_only", "sensitivity_proxy_only", "excluded"}
)


@dataclass(frozen=True)
class AttributeCalibrationMapping:
    """One immutable nutrient-to-Food-Compass calibration assignment."""

    nutrient: str
    attribute: str
    channel: str
    allocation_weight: float
    role: str
    component_policy: str
    version: str


def _mapping(
    nutrient: str,
    attribute: str,
    channel: str,
    allocation_weight: float = 1.0,
    *,
    role: str = _EFFECT_ROLE,
    component_policy: str = "direct",
    version: str = PRIMARY_MAPPING_VERSION,
) -> AttributeCalibrationMapping:
    return AttributeCalibrationMapping(
        nutrient=nutrient,
        attribute=attribute,
        channel=channel,
        allocation_weight=allocation_weight,
        role=role,
        component_policy=component_policy,
        version=version,
    )


PRIMARY_ATTRIBUTE_MAPPINGS = (
    _mapping(
        "Fiber, total dietary (g)",
        "fiber_to_carbohydrate_ratio",
        "MAC",
        0.5,
        component_policy="numerator_component",
    ),
    _mapping("Fiber, total dietary (g)", "total_fiber", "MAC", 0.5),
    _mapping("Carotene, alpha (mcg)", "total_carotenoids", "MAC", component_policy="composite_component"),
    _mapping("Carotene, beta (mcg)", "total_carotenoids", "MAC", component_policy="composite_component"),
    _mapping("Cryptoxanthin, beta (mcg)", "total_carotenoids", "MAC", component_policy="composite_component"),
    _mapping("Lycopene (mcg)", "total_carotenoids", "MAC", component_policy="composite_component"),
    _mapping("Lutein + zeaxanthin (mcg)", "total_carotenoids", "MAC", component_policy="composite_component"),
    _mapping("Vitamin C (mg)", "vitamin_c", "MAC"),
    _mapping("Folate, food (mcg)", "folate_dfe_b9", "MAC", component_policy="component_fraction"),
    _mapping("Magnesium (mg)", "magnesium", "MAC"),
    _mapping(
        "Potassium (mg)",
        "potassium_to_sodium_ratio",
        "MAC",
        0.5,
        component_policy="numerator_component",
    ),
    _mapping("Potassium (mg)", "potassium", "MAC", 0.5),
    _mapping("Vitamin K (phylloquinone) (mcg)", "vitamin_k_phylloquinone", "MAC"),
    _mapping("Vitamin E (alpha-tocopherol) (mg)", "vitamin_e_alpha_tocopherol", "MAC"),
    # The expert mask calibrates only the saturated denominator; MUFA/PUFA exposure stays fixed.
    _mapping(
        "Fatty acids, total saturated (g)",
        "unsaturated_to_saturated_fat_ratio",
        "LIPID",
        component_policy="denominator_component",
    ),
    _mapping("Cholesterol (mg)", "cholesterol", "LIPID"),
    _mapping("Retinol (mcg)", "vitamin_a_rae", "LIPID", component_policy="component_fraction"),
    _mapping("8:0 (g)", "medium_chain_fatty_acids", "LIPID", component_policy="composite_component"),
    _mapping("10:0 (g)", "medium_chain_fatty_acids", "LIPID", component_policy="composite_component"),
    _mapping("12:0 (g)", "medium_chain_fatty_acids", "LIPID", component_policy="composite_component"),
    _mapping("Choline, total (mg)", "choline_total", "LIPID"),
    _mapping("Vitamin B-12 (mcg)", "cobalamin_b12", "LIPID"),
    _mapping(
        "Total Fat (g)",
        "unsaturated_to_saturated_fat_ratio",
        "LIPID",
        0.0,
        role="applicability_only",
        component_policy="gate_input",
    ),
    *(
        _mapping(
            nutrient,
            "unsaturated_to_saturated_fat_ratio",
            "LIPID",
            0.0,
            role="explanatory_only",
            component_policy="denominator_subcomponent",
        )
        for nutrient in ("4:0 (g)", "6:0 (g)", "14:0 (g)", "16:0 (g)", "18:0 (g)")
    ),
    _mapping(
        "Carbohydrate (g)",
        "fiber_to_carbohydrate_ratio",
        "EXCLUDED",
        0.0,
        role="sensitivity_proxy_only",
        component_policy="denominator_component",
    ),
    _mapping("Zinc (mg)", "zinc", "EXCLUDED", 0.0, role="excluded", component_policy="no_calibration"),
    _mapping("Copper (mg)", "copper", "EXCLUDED", 0.0, role="excluded", component_policy="no_calibration"),
    _mapping(
        "Vitamin A, RAE (mcg_RAE)",
        "vitamin_a_rae",
        "EXCLUDED",
        0.0,
        role="excluded",
        component_policy="component_total_only",
    ),
)


def _sensitivity_variant(
    version: str,
    reallocated_nutrient: str,
    selected_attribute: str,
) -> tuple[AttributeCalibrationMapping, ...]:
    rows: list[AttributeCalibrationMapping] = []
    for row in PRIMARY_ATTRIBUTE_MAPPINGS:
        if row.nutrient == reallocated_nutrient and row.role == _EFFECT_ROLE:
            if row.attribute == selected_attribute:
                rows.append(replace(row, allocation_weight=1.0, version=version))
            continue
        rows.append(replace(row, version=version))
    return tuple(rows)


def _carbohydrate_proxy_variant() -> tuple[AttributeCalibrationMapping, ...]:
    rows: list[AttributeCalibrationMapping] = []
    for row in PRIMARY_ATTRIBUTE_MAPPINGS:
        if row.nutrient == "Carbohydrate (g)":
            rows.append(
                replace(
                    row,
                    allocation_weight=1.0,
                    channel="MAC",
                    role=_EFFECT_ROLE,
                    version=CARBOHYDRATE_PROXY_MAPPING_VERSION,
                )
            )
        else:
            rows.append(replace(row, version=CARBOHYDRATE_PROXY_MAPPING_VERSION))
    return tuple(rows)


SENSITIVITY_ATTRIBUTE_MAPPINGS = MappingProxyType(
    {
        "fiber_all_ratio": _sensitivity_variant(
            "fiber_all_ratio", "Fiber, total dietary (g)", "fiber_to_carbohydrate_ratio"
        ),
        "fiber_all_absolute": _sensitivity_variant(
            "fiber_all_absolute", "Fiber, total dietary (g)", "total_fiber"
        ),
        "potassium_all_ratio": _sensitivity_variant(
            "potassium_all_ratio", "Potassium (mg)", "potassium_to_sodium_ratio"
        ),
        "potassium_all_absolute": _sensitivity_variant(
            "potassium_all_absolute", "Potassium (mg)", "potassium"
        ),
        CARBOHYDRATE_PROXY_MAPPING_VERSION: _carbohydrate_proxy_variant(),
    }
)

_PROTECTED_PRIMARY_ROWS = MappingProxyType(
    {
        row.nutrient: row
        for row in PRIMARY_ATTRIBUTE_MAPPINGS
        if row.nutrient
        in {
            "Total Fat (g)",
            "4:0 (g)",
            "6:0 (g)",
            "14:0 (g)",
            "16:0 (g)",
            "18:0 (g)",
            "Carbohydrate (g)",
            "Zinc (mg)",
            "Copper (mg)",
            "Vitamin A, RAE (mcg_RAE)",
        }
    }
)


class AllocationMatrix(NamedTuple):
    """Labeled weights and role diagnostics for one mapping version."""

    weights: pd.DataFrame
    role_diagnostics: pd.DataFrame


class FoodSpecificResponse(NamedTuple):
    """Uncapped attribute responses and per-mapping exposure diagnostics."""

    response: pd.Series
    diagnostics: pd.DataFrame


def validate_attribute_mappings(mappings: Iterable[AttributeCalibrationMapping]) -> None:
    """Fail closed when a mapping registry violates allocation invariants."""

    rows = tuple(mappings)
    if not rows:
        raise ValueError("attribute mappings must not be empty")

    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.nutrient, row.attribute)
        if key in seen:
            raise ValueError(f"duplicate attribute mapping: {key}")
        seen.add(key)

    versions = {row.version for row in rows}
    if len(versions) != 1:
        raise ValueError("attribute mappings must contain exactly one registry version")
    version = next(iter(versions))
    builtin_carbohydrate_proxy = (
        version == CARBOHYDRATE_PROXY_MAPPING_VERSION
        and rows == SENSITIVITY_ATTRIBUTE_MAPPINGS[CARBOHYDRATE_PROXY_MAPPING_VERSION]
    )

    effect_totals: dict[str, float] = defaultdict(float)
    for row in rows:
        if row.nutrient == "OTHER" or row.attribute == "OTHER" or row.channel == "OTHER":
            raise ValueError("OTHER catch-all mappings are prohibited")
        if row.attribute not in FCS2_RULES:
            raise ValueError(f"unknown Food Compass attribute: {row.attribute}")
        if not FCS2_RULES[row.attribute].active:
            raise ValueError(f"inactive Food Compass attribute: {row.attribute}")
        try:
            allocation_weight = float(row.allocation_weight)
        except (TypeError, ValueError) as error:
            raise ValueError("allocation weights must be finite numbers") from error
        if isinstance(row.allocation_weight, bool) or not isfinite(allocation_weight):
            raise ValueError("allocation weights must be finite numbers")
        if allocation_weight < 0.0:
            raise ValueError("allocation weights cannot be negative")
        if row.role == _EFFECT_ROLE:
            if allocation_weight <= 0.0:
                raise ValueError("effect mappings require a positive allocation weight")
            effect_totals[row.nutrient] += allocation_weight
        elif row.role in _ZERO_ALLOCATION_ROLES:
            if allocation_weight != 0.0:
                raise ValueError(f"{row.role} mappings must have zero allocation")
        else:
            raise ValueError(f"unknown mapping role: {row.role}")
        if row.nutrient in _PROTECTED_PRIMARY_ROWS:
            if row.nutrient == "Carbohydrate (g)" and builtin_carbohydrate_proxy:
                expected = next(
                    candidate
                    for candidate in SENSITIVITY_ATTRIBUTE_MAPPINGS[CARBOHYDRATE_PROXY_MAPPING_VERSION]
                    if candidate.nutrient == row.nutrient
                )
            else:
                expected = replace(_PROTECTED_PRIMARY_ROWS[row.nutrient], version=version)
            if row != expected:
                raise ValueError(f"protected nutrient role cannot be changed: {row.nutrient}")

    invalid = {nutrient: total for nutrient, total in effect_totals.items() if total != 1.0}
    if invalid:
        raise ValueError(f"effect allocations must sum exactly to 1 per nutrient: {invalid}")


validate_attribute_mappings(PRIMARY_ATTRIBUTE_MAPPINGS)
for _sensitivity_mappings in SENSITIVITY_ATTRIBUTE_MAPPINGS.values():
    validate_attribute_mappings(_sensitivity_mappings)


def _mappings_for_version(version: str) -> tuple[AttributeCalibrationMapping, ...]:
    if version == PRIMARY_MAPPING_VERSION:
        return PRIMARY_ATTRIBUTE_MAPPINGS
    try:
        return SENSITIVITY_ATTRIBUTE_MAPPINGS[version]
    except KeyError as error:
        choices = (PRIMARY_MAPPING_VERSION, *SENSITIVITY_ATTRIBUTE_MAPPINGS)
        raise ValueError(f"unknown attribute mapping version: {version}; expected one of {choices}") from error


def build_allocation_matrix(
    nutrient_columns: Iterable[str],
    version: str = PRIMARY_MAPPING_VERSION,
) -> AllocationMatrix:
    """Build aligned nutrient-by-active-attribute weights and role diagnostics."""

    columns = list(nutrient_columns)
    if len(columns) != len(set(columns)):
        raise ValueError("nutrient_columns contains duplicate labels")
    mappings = _mappings_for_version(version)
    required = {row.nutrient for row in mappings if row.role == _EFFECT_ROLE}
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"missing required mapping nutrients: {missing}")

    active_attributes = [name for name, rule in FCS2_RULES.items() if rule.active]
    weights = pd.DataFrame(0.0, index=columns, columns=active_attributes)
    rows_by_nutrient: dict[str, list[AttributeCalibrationMapping]] = defaultdict(list)
    for row in mappings:
        rows_by_nutrient[row.nutrient].append(row)
        if row.role == _EFFECT_ROLE:
            weights.loc[row.nutrient, row.attribute] = row.allocation_weight

    diagnostic_rows: list[dict[str, object]] = []
    for nutrient in columns:
        nutrient_rows = rows_by_nutrient.get(nutrient, [])
        roles = sorted({row.role for row in nutrient_rows})
        policies = sorted({row.component_policy for row in nutrient_rows})
        diagnostic_rows.append(
            {
                "nutrient": nutrient,
                "role": "|".join(roles) if roles else "unmapped",
                "component_policy": "|".join(policies) if policies else "unmapped",
                "allocation_total": float(weights.loc[nutrient].sum()),
                "mapped": bool(nutrient_rows),
                "version": version,
            }
        )
    diagnostics = pd.DataFrame(diagnostic_rows).set_index("nutrient")
    return AllocationMatrix(weights=weights, role_diagnostics=diagnostics)


_COMPONENT_TOTAL_COLUMNS = {
    "folate_dfe_b9": "Folate, DFE (mcg_DFE)",
    "vitamin_a_rae": "Vitamin A, RAE (mcg_RAE)",
}
_RATIO_SIDE_INPUTS = {
    "fiber_to_carbohydrate_ratio": (
        "Fiber, total dietary (g)",
        "Carbohydrate (g)",
    ),
    "potassium_to_sodium_ratio": (
        "Potassium (mg)",
        "Sodium (mg)",
    ),
    "unsaturated_to_saturated_fat_ratio": (
        "Fatty acids, total saturated (g)",
        "Fatty acids, total monounsaturated (g)",
        "Fatty acids, total polyunsaturated (g)",
        "Total Fat (g)",
    ),
}


def _as_labeled_series(values: pd.Series, label: str) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{label} must be a pandas Series with nutrient labels")
    if not values.index.is_unique:
        raise ValueError(f"{label} contains duplicate nutrient labels")
    return values


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite nonnegative number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite nonnegative number") from error
    if not isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return numeric


def _finite_beta(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite beta value")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite beta value") from error
    if not isfinite(numeric):
        raise ValueError(f"{label} must be a finite beta value")
    return numeric


def _response_mapping_rows(
    mapping: str | Iterable[AttributeCalibrationMapping] | AllocationMatrix,
) -> tuple[tuple[AttributeCalibrationMapping, ...], AllocationMatrix | None]:
    if isinstance(mapping, str):
        return _mappings_for_version(mapping), None
    if isinstance(mapping, AllocationMatrix):
        validated = _validate_allocation_matrix(mapping)
        version = str(validated.role_diagnostics["version"].iloc[0])
        return _mappings_for_version(version), validated
    rows = tuple(mapping)
    validate_attribute_mappings(rows)
    return rows, None


def _normalization_target(attribute: str) -> float:
    rule = FCS2_RULES[attribute]
    targets = [abs(float(value)) for value in (rule.low_target, rule.high_target) if value is not None and value != 0]
    if not targets:
        raise ValueError(f"attribute {attribute} has no nonzero exposure normalization target")
    return max(targets)


def _validate_allocation_matrix(allocation: AllocationMatrix) -> AllocationMatrix:
    if not isinstance(allocation.weights, pd.DataFrame) or not isinstance(
        allocation.role_diagnostics, pd.DataFrame
    ):
        raise ValueError("allocation matrix weights and diagnostics must be pandas DataFrames")
    if not allocation.weights.index.is_unique or not allocation.weights.columns.is_unique:
        raise ValueError("allocation matrix labels must be unique")
    if "version" not in allocation.role_diagnostics.columns or allocation.role_diagnostics.empty:
        raise ValueError("allocation matrix diagnostics must contain a version")
    versions = set(allocation.role_diagnostics["version"])
    if len(versions) != 1:
        raise ValueError("allocation matrix diagnostics must identify exactly one version")
    version = next(iter(versions))
    if not isinstance(version, str):
        raise ValueError("allocation matrix version must be a string")
    try:
        numeric_weights = allocation.weights.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("allocation matrix weights must be finite numbers") from error
    if not all(isfinite(float(value)) for value in numeric_weights.flat):
        raise ValueError("allocation matrix weights must be finite numbers")

    expected = build_allocation_matrix(allocation.weights.index.tolist(), version)
    try:
        pd.testing.assert_frame_equal(allocation.weights, expected.weights, check_exact=True)
        pd.testing.assert_frame_equal(
            allocation.role_diagnostics,
            expected.role_diagnostics,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError("allocation matrix does not match its named registry version") from error
    return expected


def _clip_unit_interval(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _ratio_component_exposure(
    row: AttributeCalibrationMapping,
    exposure: pd.Series,
) -> tuple[float, bool, str]:
    if row.attribute == "fiber_to_carbohydrate_ratio":
        fiber = _finite_nonnegative(exposure["Fiber, total dietary (g)"], "Fiber, total dietary (g)")
        carbohydrate = _finite_nonnegative(exposure["Carbohydrate (g)"], "Carbohydrate (g)")
        if not ratio_gate_passes_from_exposures(row.attribute, exposure):
            return 0.0, False, "applicability_gate_not_met"
        if row.nutrient == "Fiber, total dietary (g)":
            return fiber / _normalization_target("total_fiber"), False, "ok"
        if row.nutrient == "Carbohydrate (g)":
            return -_clip_unit_interval(carbohydrate * 4.0 / 100.0), False, "ok"
    elif row.attribute == "potassium_to_sodium_ratio":
        potassium = _finite_nonnegative(exposure["Potassium (mg)"], "Potassium (mg)")
        sodium = _finite_nonnegative(exposure["Sodium (mg)"], "Sodium (mg)")
        if not ratio_gate_passes_from_exposures(row.attribute, exposure):
            return 0.0, False, "applicability_gate_not_met"
        if row.nutrient == "Potassium (mg)":
            return potassium / _normalization_target("potassium"), False, "ok"
    elif row.attribute == "unsaturated_to_saturated_fat_ratio":
        total_fat = _finite_nonnegative(exposure["Total Fat (g)"], "Total Fat (g)")
        if not ratio_gate_passes_from_exposures(row.attribute, exposure):
            return 0.0, False, "applicability_gate_not_met"
        saturated = _finite_nonnegative(
            exposure["Fatty acids, total saturated (g)"],
            "Fatty acids, total saturated (g)",
        )
        monounsaturated = _finite_nonnegative(
            exposure["Fatty acids, total monounsaturated (g)"],
            "Fatty acids, total monounsaturated (g)",
        )
        polyunsaturated = _finite_nonnegative(
            exposure["Fatty acids, total polyunsaturated (g)"],
            "Fatty acids, total polyunsaturated (g)",
        )
        component_total = saturated + monounsaturated + polyunsaturated
        if component_total == 0.0:
            return 0.0, True, "zero_total_component"
        if row.nutrient == "Fatty acids, total saturated (g)":
            return -_clip_unit_interval(saturated / component_total), False, "ok"
    raise ValueError(f"unsupported ratio component mapping: {row.nutrient} -> {row.attribute}")


def build_food_specific_response(
    beta: pd.Series,
    food_exposure: pd.Series,
    mapping: str | Iterable[AttributeCalibrationMapping] | AllocationMatrix = PRIMARY_MAPPING_VERSION,
    *,
    allocation: AllocationMatrix | None = None,
) -> FoodSpecificResponse:
    """Combine beta and per-100-kcal food dose into uncapped attribute responses."""

    beta = _as_labeled_series(beta, "beta")
    food_exposure = _as_labeled_series(food_exposure, "food_exposure")
    if food_exposure.attrs.get("basis") != "per_100_kcal":
        raise ValueError("food_exposure must declare attrs['basis'] = 'per_100_kcal'")
    if allocation is not None:
        if mapping != PRIMARY_MAPPING_VERSION:
            raise ValueError("pass an allocation either positionally or by keyword, not both")
        mapping = allocation
    rows, supplied_allocation = _response_mapping_rows(mapping)
    effect_rows = [row for row in rows if row.role == _EFFECT_ROLE]

    required_beta = {row.nutrient for row in effect_rows}
    missing_beta = sorted(required_beta - set(beta.index))
    if missing_beta:
        raise ValueError(f"missing required beta nutrients: {missing_beta}")

    required_exposure = set(required_beta)
    for row in effect_rows:
        if row.component_policy == "component_fraction":
            required_exposure.add(_COMPONENT_TOTAL_COLUMNS[row.attribute])
        if row.attribute in _RATIO_SIDE_INPUTS:
            required_exposure.update(_RATIO_SIDE_INPUTS[row.attribute])
    missing_exposure = sorted(required_exposure - set(food_exposure.index))
    if missing_exposure:
        raise ValueError(f"missing required food exposure nutrients: {missing_exposure}")

    for nutrient in required_beta:
        _finite_beta(beta[nutrient], nutrient)
    for nutrient in required_exposure:
        _finite_nonnegative(food_exposure[nutrient], nutrient)

    active_attributes = [name for name, rule in FCS2_RULES.items() if rule.active]
    response = pd.Series(0.0, index=active_attributes, name=food_exposure.name, dtype=float)
    composite_totals = {
        attribute: sum(
            _finite_nonnegative(food_exposure[row.nutrient], row.nutrient)
            for row in effect_rows
            if row.attribute == attribute and row.component_policy == "composite_component"
        )
        for attribute in {row.attribute for row in effect_rows if row.component_policy == "composite_component"}
    }
    diagnostic_rows: list[dict[str, object]] = []
    for row in effect_rows:
        nutrient_exposure = _finite_nonnegative(food_exposure[row.nutrient], row.nutrient)
        zero_total = False
        status = "ok"
        if row.component_policy == "component_fraction":
            total_column = _COMPONENT_TOTAL_COLUMNS[row.attribute]
            component_total = _finite_nonnegative(food_exposure[total_column], total_column)
            if component_total == 0.0:
                if nutrient_exposure != 0.0:
                    raise ValueError(
                        f"{row.nutrient} is nonzero while required component total {total_column} is zero"
                    )
                normalized_exposure = 0.0
                zero_total = True
                status = "zero_total_component"
            else:
                if nutrient_exposure > component_total:
                    raise ValueError(f"{row.nutrient} cannot exceed component total {total_column}")
                component_fraction = nutrient_exposure / component_total
                normalized_exposure = component_total / _normalization_target(row.attribute) * component_fraction
        elif row.component_policy == "composite_component":
            component_total = composite_totals[row.attribute]
            if component_total == 0.0:
                normalized_exposure = 0.0
                zero_total = True
                status = "zero_total_component"
            else:
                normalized_exposure = nutrient_exposure / _normalization_target(row.attribute)
        elif row.attribute in _RATIO_SIDE_INPUTS:
            normalized_exposure, zero_total, status = _ratio_component_exposure(row, food_exposure)
        else:
            normalized_exposure = nutrient_exposure / _normalization_target(row.attribute)

        allocation_weight = row.allocation_weight
        if supplied_allocation is not None:
            try:
                allocation_weight = float(supplied_allocation.weights.loc[row.nutrient, row.attribute])
            except KeyError as error:
                raise ValueError("allocation matrix is not aligned to its mapping registry") from error
        contribution = _finite_beta(beta[row.nutrient], row.nutrient) * normalized_exposure * allocation_weight
        response[row.attribute] += contribution
        diagnostic_rows.append(
            {
                "nutrient": row.nutrient,
                "attribute": row.attribute,
                "channel": row.channel,
                "component_policy": row.component_policy,
                "allocation_weight": allocation_weight,
                "normalized_exposure": normalized_exposure,
                "contribution": contribution,
                "zero_total_component": zero_total,
                "status": status,
                "exposure_basis": "per_100_kcal",
            }
        )

    return FoodSpecificResponse(response=response, diagnostics=pd.DataFrame(diagnostic_rows))


_FIXED_BASELINE_REASONS = {
    "alpha_linolenic_acid": (
        "18:3 is not verified ALA in the local source, so alpha-linolenic acid is a fixed "
        "baseline attribute within a recomputed domain."
    ),
    "total_flavonoids": (
        "Total flavonoids are absent from N_food_nutrient_full and require the separate source "
        "database, so they are a fixed baseline attribute within a recomputed domain."
    ),
}
_AUXILIARY_REASONS = {
    "added_sugar_percent_calories": (
        "An added-sugar source is required; total sugars cannot substitute for added sugar."
    ),
    "nova_processing_level": (
        "The original energy-weighted mixed-dish value is required; rounded NOVA cannot substitute for it."
    ),
}


def reconstruction_status_table() -> pd.DataFrame:
    """Classify reconstruction inputs for every active FCS operational rule."""

    rows: list[dict[str, str]] = []
    auxiliary_domains = {"food_ingredients", "additives", "processing"}
    for attribute, rule in FCS2_RULES.items():
        if not rule.active:
            continue
        if attribute in _FIXED_BASELINE_REASONS:
            status = "fixed_baseline_in_recomputed_domain"
            source = "official Food Compass baseline attribute points"
            reason = _FIXED_BASELINE_REASONS[attribute]
        elif rule.domain in auxiliary_domains:
            status = "auxiliary_ingredient_or_processing_required"
            if rule.domain == "food_ingredients":
                source = "FPED ingredient equivalents"
            elif rule.domain == "additives":
                source = "ingredient and additive records"
            else:
                source = "processing and recipe records"
            reason = _AUXILIARY_REASONS.get(
                attribute,
                "This operational rule requires its named auxiliary ingredient or processing input.",
            )
        else:
            status = "nutrient_derived"
            source = "verified FNDDS/FoodData Central nutrient fields"
            reason = "The required nutrient field or verified nutrient components are available."
        rows.append(
            {
                "attribute": attribute,
                "domain": rule.domain,
                "reconstruction_status": status,
                "required_source": source,
                "reason": reason,
                "missing_input_policy": "unavailable_or_fixed_baseline",
            }
        )
    return pd.DataFrame(rows)
