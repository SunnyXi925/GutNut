"""Published Food Compass 2.0 attribute scoring rules.

The constants in this module are transcribed from Supplementary Tables S9-S10
of Mozaffarian et al., Food Compass 2.0 (2024). All supplied exposures are per
100 kcal unless the rule name states that it accepts a percentage or a ratio.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from types import MappingProxyType
from typing import Mapping


UNSCALED_MIN = -12.1
UNSCALED_MAX = 35.0
FCS_MIN = 1.0
FCS_MAX = 100.0


class PublishedRuleAmbiguityError(ValueError):
    """Raised when Table S10 has contradictory or incomplete scoring details."""


class UnavailableAttributeError(ValueError):
    """Raised when a Table S10 candidate was unavailable in the reported analysis."""


class AttributeScore(float):
    """Raw attribute points with the effective published aggregation weight."""

    def __new__(cls, points: float, weight: float) -> AttributeScore:
        instance = super().__new__(cls, points)
        instance.weight = weight
        return instance


@dataclass(frozen=True)
class DomainAttributeSelection:
    """Canonical Task 1 membership and denominator for one complete domain."""

    domain: str
    selected_attributes: tuple[str, ...]
    active_weight_denominator: float


@dataclass(frozen=True)
class NotCalculated:
    """Explicit result for a ratio below its Table S10 exposure gate."""

    reason: str = "minimum exposure gate not met"


NOT_CALCULATED = NotCalculated()


@dataclass(frozen=True)
class AttributeRule:
    """One Table S10 candidate attribute and its reported scoring metadata."""

    name: str
    domain: str
    kind: str
    low_target: float | None
    high_target: float | None
    low_points: float
    high_points: float
    unit: str
    conceptual_name: str
    weight: float = 1.0
    active: bool = True
    unavailability_reason: str | None = None
    ambiguity: str | None = None
    primary_version: str | None = None
    scoring_versions: tuple[tuple[str, float], ...] = ()


def _rule(
    name: str,
    domain: str,
    kind: str,
    low_target: float | None,
    high_target: float | None,
    low_points: float,
    high_points: float,
    unit: str,
    *,
    conceptual_name: str | None = None,
    weight: float = 1.0,
    active: bool = True,
    unavailability_reason: str | None = None,
    ambiguity: str | None = None,
    primary_version: str | None = None,
    scoring_versions: tuple[tuple[str, float], ...] = (),
) -> AttributeRule:
    return AttributeRule(
        name=name,
        domain=domain,
        kind=kind,
        low_target=low_target,
        high_target=high_target,
        low_points=low_points,
        high_points=high_points,
        unit=unit,
        conceptual_name=conceptual_name or {
            "fruits_dried": "fruits",
            "vegetables_non_starchy_dried": "vegetables_non_starchy",
        }.get(name, name),
        weight=weight,
        active=active,
        unavailability_reason=unavailability_reason,
        ambiguity=ambiguity,
        primary_version=primary_version,
        scoring_versions=scoring_versions,
    )


_UNAVAILABLE_USDA = (
    "Excluded from the reported FCS2 USDA analysis because it was unavailable "
    "in FNDDS, FPED, and the flavonoid database (Table S9 footnote)."
)
_NITRITES_CONFLICT = (
    "Table S10 lists 25% calories from processed meats as the low target, but "
    "the Table S10 additive footnote specifies 50%; no threshold is selected."
)
NITRITE_RULE_PRIMARY_VERSION = "footnote_50"
NITRITE_RULE_TABLE_SENSITIVITY_VERSION = "table_25"


_RULES = (
    # Nutrient ratios
    _rule("unsaturated_to_saturated_fat_ratio", "nutrient_ratios", "log_ratio", -0.66, 1.77, -10, 10, "ratio"),
    _rule("fiber_to_carbohydrate_ratio", "nutrient_ratios", "log_ratio", -7.02, -0.78, -10, 10, "ratio"),
    _rule("potassium_to_sodium_ratio", "nutrient_ratios", "log_ratio", -2.02, 3.30, -10, 10, "ratio"),
    # Vitamins
    _rule("vitamin_a_rae", "vitamins", "linear", 0, 225, 0, 10, "ug RAE"),
    _rule("thiamin_b1", "vitamins", "linear", 0, 0.3, 0, 10, "mg"),
    _rule("riboflavin_b2", "vitamins", "linear", 0, 0.325, 0, 10, "mg"),
    _rule("niacin_b3", "vitamins", "linear", 0, 4, 0, 10, "mg"),
    _rule("vitamin_b6", "vitamins", "linear", 0, 0.325, 0, 10, "mg"),
    _rule("folate_dfe_b9", "vitamins", "linear", 0, 100, 0, 10, "ug DFE"),
    _rule("cobalamin_b12", "vitamins", "linear", 0, 0.6, 0, 10, "ug"),
    _rule("vitamin_c", "vitamins", "linear", 0, 22.5, 0, 10, "mg"),
    _rule("vitamin_d_d2_plus_d3", "vitamins", "linear", 0, 3.75, 0, 10, "ug"),
    _rule("vitamin_e_alpha_tocopherol", "vitamins", "linear", 0, 3.75, 0, 10, "mg"),
    _rule("vitamin_k_phylloquinone", "vitamins", "linear", 0, 30, 0, 10, "ug"),
    _rule("choline_total", "vitamins", "linear", 0, 137.5, 0, 10, "mg"),
    # Minerals
    _rule("calcium", "minerals", "linear", 0, 250, 0, 10, "mg"),
    _rule("phosphorus", "minerals", "linear", 0, 175, 0, 10, "mg"),
    _rule("magnesium", "minerals", "linear", 0, 105, 0, 10, "mg"),
    _rule("iron", "minerals", "linear", 0, 4.5, 0, 10, "mg"),
    _rule("zinc", "minerals", "linear", 0, 2.75, 0, 10, "mg"),
    _rule("copper", "minerals", "linear", 0, 0.225, 0, 10, "mg"),
    _rule("selenium", "minerals", "linear", 0, 13.75, 0, 10, "ug"),
    _rule("sodium", "minerals", "linear", 575, 0, -10, 0, "mg"),
    _rule("potassium", "minerals", "linear", 0, 1175, 0, 10, "mg"),
    _rule(
        "iodine", "minerals", "linear", 0, 37.5, 0, 10, "ug",
        active=False, unavailability_reason=_UNAVAILABLE_USDA,
    ),
    # Food ingredients
    _rule("fruits", "food_ingredients", "linear", 0, 1.75, 0, 10, "cups"),
    _rule("fruits_dried", "food_ingredients", "linear", 0, 0.75, 0, 10, "cups"),
    _rule("vegetables_non_starchy", "food_ingredients", "linear", 0, 4.77, 0, 10, "cups"),
    _rule("vegetables_non_starchy_dried", "food_ingredients", "linear", 0, 4.18, 0, 10, "cups"),
    _rule("beans_and_legumes", "food_ingredients", "linear", 0, 0.50, 0, 10, "cups"),
    _rule("whole_grains", "food_ingredients", "linear", 0, 1.12, 0, 10, "oz"),
    _rule("nuts_and_seeds", "food_ingredients", "linear", 0, 1.35, 0, 10, "oz"),
    _rule("seafood", "food_ingredients", "linear", 0, 3.86, 0, 10, "oz"),
    _rule("yogurt", "food_ingredients", "linear", 0, 0.81, 0, 10, "cups"),
    _rule("plant_oils", "food_ingredients", "linear", 0, 11.31, 0, 10, "g"),
    _rule("refined_carbohydrates", "food_ingredients", "linear", 1.36, 0, -10, 0, "oz"),
    _rule("red_or_processed_meat", "food_ingredients", "linear", 2.69, 0, -10, 0, "oz"),
    # Additives
    _rule("added_sugar_percent_calories", "additives", "added_sugar", 60, 0, -10, 0, "% calories"),
    _rule(
        "nitrites_percent_calories_from_processed_meat", "additives", "nitrites", 25, 0, -10, 0,
        "% calories", ambiguity=_NITRITES_CONFLICT,
        primary_version=NITRITE_RULE_PRIMARY_VERSION,
        scoring_versions=(
            (NITRITE_RULE_PRIMARY_VERSION, 50.0),
            (NITRITE_RULE_TABLE_SENSITIVITY_VERSION, 25.0),
        ),
    ),
    _rule("artificial_sweeteners_flavors_or_colors", "additives", "binary", None, None, -1, 0, "present"),
    _rule("partially_hydrogenated_oils", "additives", "binary", None, None, -1, 0, "present"),
    _rule("interesterified_or_hydrogenated_oils", "additives", "binary", None, None, -1, 0, "present"),
    _rule("high_fructose_corn_syrup", "additives", "binary", None, None, -1, 0, "present"),
    _rule("monosodium_glutamate", "additives", "binary", None, None, -1, 0, "present"),
    # Processing
    _rule("nova_processing_level", "processing", "nova", 4, 1, -10, 10, "NOVA class"),
    _rule("fermentation_percent_calories", "processing", "linear", 0, 50, 0, 10, "% calories", weight=0.5),
    _rule("frying", "processing", "binary", None, None, -10, 0, "present", weight=0.5),
    # Specific lipids
    _rule("cholesterol", "specific_lipids", "linear", 75, 0, -10, 0, "mg", weight=0.5),
    _rule("medium_chain_fatty_acids", "specific_lipids", "linear", 0, 0.32, 0, 10, "g", weight=0.5),
    _rule("alpha_linolenic_acid", "specific_lipids", "linear", 0, 0.4, 0, 10, "g", weight=0.5),
    _rule("epa_plus_dha", "specific_lipids", "linear", 0, 62.5, 0, 10, "mg"),
    _rule(
        "trans_fat_percent_calories", "specific_lipids", "linear", 30, 0, -10, 0, "% calories",
        active=False, unavailability_reason=_UNAVAILABLE_USDA,
    ),
    # Fiber and protein
    _rule("total_fiber", "fiber_and_protein", "linear", 0, 9.5, 0, 10, "g"),
    _rule("total_protein", "fiber_and_protein", "linear", 0, 14, 0, 10, "g", weight=0.5),
    # Phytochemicals
    _rule("total_flavonoids", "phytochemicals", "linear", 0, 23.53, 0, 10, "mg"),
    _rule("total_carotenoids", "phytochemicals", "linear", 0, 8746.81, 0, 10, "mcg"),
)

FCS2_RULES: Mapping[str, AttributeRule] = MappingProxyType({rule.name: rule for rule in _RULES})
FCS2_OPERATIONAL_RULE_COUNT = len(FCS2_RULES)
FCS2_CONCEPTUAL_ATTRIBUTE_COUNT = len({rule.conceptual_name for rule in FCS2_RULES.values()})

_TOP_K = MappingProxyType({"vitamins": 5, "minerals": 5, "specific_lipids": 3})
_DOMAIN_WEIGHTS = MappingProxyType({"specific_lipids": 0.5, "phytochemicals": 0.5})
_RULE_ORDER = MappingProxyType({rule.name: index for index, rule in enumerate(FCS2_RULES.values())})
_active_rules_by_domain: dict[str, list[AttributeRule]] = {}
for _registry_rule in FCS2_RULES.values():
    if _registry_rule.active:
        _active_rules_by_domain.setdefault(_registry_rule.domain, []).append(_registry_rule)
_ACTIVE_RULES_BY_DOMAIN: Mapping[str, tuple[AttributeRule, ...]] = MappingProxyType(
    {domain: tuple(rules) for domain, rules in _active_rules_by_domain.items()}
)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not isfinite(numeric):
        raise ValueError(f"{label} must be a finite number")
    return numeric


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, min(low, high)), max(low, high))


def _resolve_rule(rule: str | AttributeRule) -> AttributeRule:
    if isinstance(rule, AttributeRule):
        return rule
    try:
        return FCS2_RULES[rule]
    except KeyError as error:
        raise ValueError(f"unknown Food Compass 2.0 attribute: {rule}") from error


def to_per_100_kcal(value: float, energy_kcal: float) -> float:
    """Convert an absolute food exposure to the Table S10 per-100-kcal basis."""

    numeric_value = _finite_number(value, "value")
    calories = _finite_number(energy_kcal, "energy_kcal")
    if calories <= 0:
        raise ValueError("energy_kcal must be greater than zero")
    return numeric_value * 100.0 / calories


def ratio_gate_passes(rule: str | AttributeRule, context: Mapping[str, object]) -> bool:
    """Evaluate the published ratio applicability gate from canonical context."""

    rule = _resolve_rule(rule)
    if rule.kind != "log_ratio":
        raise ValueError(f"{rule.name} is not a ratio attribute")
    if rule.name == "unsaturated_to_saturated_fat_ratio":
        if "fat_energy_percent" in context:
            return _finite_number(context["fat_energy_percent"], "fat_energy_percent") >= 10.0
        if "total_fat_g" in context:
            return _finite_number(context["total_fat_g"], "total_fat_g") * 9.0 >= 10.0
        raise ValueError("unsaturated_to_saturated_fat_ratio requires fat_energy_percent or total_fat_g")
    if rule.name == "fiber_to_carbohydrate_ratio":
        if "carbohydrate_energy_percent" in context:
            return _finite_number(context["carbohydrate_energy_percent"], "carbohydrate_energy_percent") >= 10.0
        if "carbohydrate_g" in context:
            return _finite_number(context["carbohydrate_g"], "carbohydrate_g") * 4.0 >= 10.0
        raise ValueError("fiber_to_carbohydrate_ratio requires carbohydrate_energy_percent or carbohydrate_g")
    if rule.name == "potassium_to_sodium_ratio":
        required = ("potassium_mg", "sodium_mg")
        missing = [name for name in required if name not in context]
        if missing:
            raise ValueError("potassium_to_sodium_ratio requires potassium_mg and sodium_mg")
        return all(_finite_number(context[name], name) >= 10.0 for name in required)
    raise ValueError(f"missing ratio gate for {rule.name}")


def ratio_gate_passes_from_exposures(
    rule: str | AttributeRule,
    food_exposure: Mapping[str, object],
) -> bool:
    """Evaluate a ratio gate directly from the per-100-kcal food exposure row.

    This is the single adapter used by Task 2 response construction and by the
    FoodAttributeBundle biological validation boundary.
    """

    selected = _resolve_rule(rule)
    if selected.name == "unsaturated_to_saturated_fat_ratio":
        required = ("Total Fat (g)",)
    elif selected.name == "fiber_to_carbohydrate_ratio":
        required = ("Carbohydrate (g)",)
    elif selected.name == "potassium_to_sodium_ratio":
        required = ("Potassium (mg)", "Sodium (mg)")
    else:
        raise ValueError(f"{selected.name} is not a supported ratio attribute")
    missing = [name for name in required if name not in food_exposure]
    if missing:
        raise ValueError(
            f"{selected.name} ratio gate requires food exposures: {missing}"
        )
    if selected.name == "unsaturated_to_saturated_fat_ratio":
        context = {"total_fat_g": food_exposure[required[0]]}
    elif selected.name == "fiber_to_carbohydrate_ratio":
        context = {"carbohydrate_g": food_exposure[required[0]]}
    else:
        context = {
            "potassium_mg": food_exposure[required[0]],
            "sodium_mg": food_exposure[required[1]],
        }
    return ratio_gate_passes(selected, context)


def _linear_score(rule: AttributeRule, value: float) -> float:
    assert rule.low_target is not None and rule.high_target is not None
    if value < 0:
        raise ValueError(f"{rule.name} cannot be negative")
    fraction = (value - rule.low_target) / (rule.high_target - rule.low_target)
    points = rule.low_points + fraction * (rule.high_points - rule.low_points)
    return _clip(points, rule.low_points, rule.high_points)


def score_attribute(
    rule: str | AttributeRule,
    value: float | bool,
    *,
    context: Mapping[str, object] | None = None,
    nitrite_rule_version: str = NITRITE_RULE_PRIMARY_VERSION,
) -> AttributeScore | NotCalculated:
    """Score one active Table S10 attribute on its published point scale.

    Ratio values are raw ratios, not their logs. Ratio gates are evaluated from
    per-100-kcal values in ``context``. A gated ratio returns
    ``NOT_CALCULATED`` and is excluded from its domain denominator.
    """

    selected = _resolve_rule(rule)
    if not selected.active:
        raise UnavailableAttributeError(
            f"{selected.name} was unavailable in the reported FCS2 analysis: "
            f"{selected.unavailability_reason}"
        )
    context = context or {}
    if selected.kind == "linear":
        points = _linear_score(selected, _finite_number(value, selected.name))
        if selected.name == "fermentation_percent_calories" and context.get("is_other_fermented_product"):
            points = selected.high_points
    elif selected.kind == "log_ratio":
        if not ratio_gate_passes(selected, context):
            return NOT_CALCULATED
        ratio = _finite_number(value, selected.name)
        if ratio <= 0:
            raise ValueError(f"{selected.name} must be greater than zero")
        log_ratio = log(ratio)
        assert selected.low_target is not None and selected.high_target is not None
        fraction = (log_ratio - selected.low_target) / (selected.high_target - selected.low_target)
        points = _clip(
            selected.low_points + fraction * (selected.high_points - selected.low_points),
            selected.low_points,
            selected.high_points,
        )
        if selected.name == "unsaturated_to_saturated_fat_ratio" and context.get("is_dairy"):
            return AttributeScore(points, 0.5)
    elif selected.kind == "added_sugar":
        percent = _finite_number(value, selected.name)
        if not 0.0 <= percent <= 100.0:
            raise ValueError("added_sugar_percent_calories must be between 0 and 100")
        if percent == 0:
            points = 0.0
        else:
            thresholds = (2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0)
            points = -float(next((index + 1 for index, threshold in enumerate(thresholds) if percent < threshold), 10))
    elif selected.kind == "nitrites":
        percent = _finite_number(value, selected.name)
        if not 0.0 <= percent <= 100.0:
            raise ValueError("nitrites_percent_calories_from_processed_meat must be between 0 and 100")
        targets = dict(selected.scoring_versions)
        try:
            low_target = targets[nitrite_rule_version]
        except KeyError as error:
            raise ValueError(
                "nitrite_rule_version must be one of: " + ", ".join(targets)
            ) from error
        fraction = (percent - low_target) / -low_target
        points = _clip(
            selected.low_points + fraction * (selected.high_points - selected.low_points),
            selected.low_points,
            selected.high_points,
        )
    elif selected.kind == "binary":
        if not isinstance(value, bool):
            raise ValueError(f"{selected.name} must be a boolean presence indicator")
        points = selected.low_points if value else selected.high_points
    elif selected.kind == "nova":
        nova = _finite_number(value, selected.name)
        if not 1.0 <= nova <= 4.0:
            raise ValueError("nova_processing_level must be between 1 and 4")
        anchors = ((1.0, 10.0), (2.0, 7.5), (3.0, 5.0), (4.0, -10.0))
        for (low_class, low_score), (high_class, high_score) in zip(anchors, anchors[1:], strict=True):
            if low_class <= nova <= high_class:
                fraction = (nova - low_class) / (high_class - low_class)
                points = low_score + fraction * (high_score - low_score)
                break
        else:  # pragma: no cover - range is validated above
            raise ValueError("unable to score NOVA processing level")
    else:  # pragma: no cover - registry construction controls kinds
        raise ValueError(f"unsupported Food Compass rule kind: {selected.kind}")
    return AttributeScore(points, selected.weight)


def _aggregate_value(
    rule: AttributeRule,
    value: float | NotCalculated,
    effective_weight: object | None = None,
) -> tuple[float, float] | None:
    if value is NOT_CALCULATED:
        if rule.kind != "log_ratio":
            raise ValueError(f"{rule.name} can only be NOT_CALCULATED for a ratio exposure gate")
        return None
    if isinstance(value, NotCalculated):
        raise ValueError("NOT_CALCULATED must use the module sentinel")
    points = _finite_number(value, rule.name)
    if effective_weight is None:
        weight = value.weight if isinstance(value, AttributeScore) else rule.weight
    else:
        weight = _finite_number(effective_weight, f"{rule.name} effective weight")
    if not isfinite(weight) or weight <= 0:
        raise ValueError(f"{rule.name} weight must be finite and greater than zero")
    return points, weight


def _weighted_mean(scores: list[tuple[AttributeRule, float, float]]) -> float:
    denominator = sum(weight for _, _, weight in scores)
    if denominator <= 0:  # pragma: no cover - weights are validated at construction/use
        raise ValueError("domain has no active attribute weight")
    return sum(points * weight for _, points, weight in scores) / denominator


def _group_domain_scores(
    attribute_scores: Mapping[str, float | NotCalculated],
    effective_attribute_weights: Mapping[str, object] | None,
) -> dict[str, dict[str, tuple[float, float] | None]]:
    if effective_attribute_weights is not None:
        if set(effective_attribute_weights) != set(attribute_scores):
            raise ValueError(
                "effective_attribute_weights must exactly cover supplied attributes"
            )
    grouped: dict[str, dict[str, tuple[float, float] | None]] = {}
    for name, score in attribute_scores.items():
        rule = _resolve_rule(name)
        if not rule.active:
            raise UnavailableAttributeError(
                f"{name} is unavailable in the reported FCS2 analysis"
            )
        effective_weight = (
            None
            if effective_attribute_weights is None
            else effective_attribute_weights[name]
        )
        grouped.setdefault(rule.domain, {})[name] = _aggregate_value(
            rule, score, effective_weight
        )
    return grouped


def select_domain_attributes(
    attribute_scores: Mapping[str, float | NotCalculated],
    domain: str,
    effective_attribute_weights: Mapping[str, object] | None = None,
) -> DomainAttributeSelection:
    """Return the unique canonical Task 1 selector used by all recomposition."""

    if domain not in _ACTIVE_RULES_BY_DOMAIN:
        raise ValueError(f"unknown Food Compass domain: {domain}")
    grouped = _group_domain_scores(attribute_scores, effective_attribute_weights)
    if set(grouped) != {domain}:
        raise ValueError("attribute_scores must contain exactly one requested domain")
    supplied = grouped[domain]
    expected = _ACTIVE_RULES_BY_DOMAIN[domain]
    missing = [rule.name for rule in expected if rule.name not in supplied]
    if missing:
        raise ValueError(
            f"incomplete {domain} domain; missing active rules: {', '.join(missing)}"
        )
    scores = [
        (rule, *value)
        for rule in expected
        if (value := supplied[rule.name]) is not None
    ]
    if not scores:
        raise ValueError(f"{domain} domain has no calculated active rules")
    selected = (
        sorted(
            scores,
            key=lambda item: (-abs(item[1]), _RULE_ORDER[item[0].name]),
        )[: _TOP_K[domain]]
        if domain in _TOP_K
        else scores
    )
    return DomainAttributeSelection(
        domain=domain,
        selected_attributes=tuple(item[0].name for item in selected),
        active_weight_denominator=float(sum(item[2] for item in selected)),
    )


def aggregate_domains(
    attribute_scores: Mapping[str, float | NotCalculated],
    *,
    effective_attribute_weights: Mapping[str, object] | None = None,
) -> dict[str, float]:
    """Aggregate final attribute points into Table S9 domain contributions.

    The returned values include the two published half-weighted domain factors,
    so their sum is the local contribution to an unscaled Food Compass score.
    Each requested domain must provide every active operational rule. Top-k ties
    are resolved by immutable FCS2 registry order, never caller mapping order.
    """

    grouped = _group_domain_scores(attribute_scores, effective_attribute_weights)

    domains: dict[str, float] = {}
    for domain, supplied in grouped.items():
        expected = _ACTIVE_RULES_BY_DOMAIN[domain]
        missing = [rule.name for rule in expected if rule.name not in supplied]
        if missing:
            raise ValueError(f"incomplete {domain} domain; missing active rules: {', '.join(missing)}")
        scores = [
            (rule, *value)
            for rule in expected
            if (value := supplied[rule.name]) is not None
        ]
        if not scores:
            raise ValueError(f"{domain} domain has no calculated active rules")
        domain_weights = (
            None
            if effective_attribute_weights is None
            else {
                rule.name: effective_attribute_weights[rule.name]
                for rule in expected
            }
        )
        selection = select_domain_attributes(
            {rule.name: attribute_scores[rule.name] for rule in expected},
            domain,
            domain_weights,
        )
        if domain == "food_ingredients":
            domain_score = sum(points * weight for _, points, weight in scores)
        elif domain in _TOP_K:
            selected_names = set(selection.selected_attributes)
            selected = [item for item in scores if item[0].name in selected_names]
            domain_score = _weighted_mean(selected)
        else:
            domain_score = _weighted_mean(scores)
        domains[domain] = domain_score * _DOMAIN_WEIGHTS.get(domain, 1.0)
    return domains


def unscaled_to_fcs(unscaled_score: float) -> float:
    """Apply the Table S10 5th/95th-percentile truncation and FCS scaling."""

    clipped = _clip(_finite_number(unscaled_score, "unscaled_score"), UNSCALED_MIN, UNSCALED_MAX)
    if clipped == UNSCALED_MIN:
        return FCS_MIN
    if clipped == UNSCALED_MAX:
        return FCS_MAX
    return FCS_MIN + (clipped - UNSCALED_MIN) * (FCS_MAX - FCS_MIN) / (UNSCALED_MAX - UNSCALED_MIN)


def fcs_to_unscaled(fcs: float) -> float:
    """Invert the published [1, 100] Food Compass scaling exactly."""

    score = _finite_number(fcs, "fcs")
    if not FCS_MIN <= score <= FCS_MAX:
        raise ValueError("fcs must be between 1 and 100")
    return UNSCALED_MIN + (score - FCS_MIN) * (UNSCALED_MAX - UNSCALED_MIN) / (FCS_MAX - FCS_MIN)
