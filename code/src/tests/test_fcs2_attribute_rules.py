import math

import pytest

from gmnps.scoring.fcs2_attribute_rules import (
    FCS2_CONCEPTUAL_ATTRIBUTE_COUNT,
    FCS2_OPERATIONAL_RULE_COUNT,
    FCS2_RULES,
    NITRITE_RULE_PRIMARY_VERSION,
    NITRITE_RULE_TABLE_SENSITIVITY_VERSION,
    aggregate_domains,
    fcs_to_unscaled,
    score_attribute,
    to_per_100_kcal,
    unscaled_to_fcs,
)


def test_registry_contains_all_published_candidates_and_54_active_rules():
    assert len(FCS2_RULES) == FCS2_OPERATIONAL_RULE_COUNT == 56
    assert len({rule.conceptual_name for rule in FCS2_RULES.values()}) == FCS2_CONCEPTUAL_ATTRIBUTE_COUNT == 54
    assert sum(rule.active for rule in FCS2_RULES.values()) == 54
    assert FCS2_RULES["fruits"].conceptual_name == FCS2_RULES["fruits_dried"].conceptual_name
    assert (
        FCS2_RULES["vegetables_non_starchy"].conceptual_name
        == FCS2_RULES["vegetables_non_starchy_dried"].conceptual_name
    )
    assert FCS2_RULES["iodine"].active is False
    assert FCS2_RULES["trans_fat_percent_calories"].active is False
    assert FCS2_RULES["iodine"].unavailability_reason
    assert FCS2_RULES["trans_fat_percent_calories"].unavailability_reason


def test_table_s10_per_100_kcal_conversion_and_linear_targets_clip():
    assert to_per_100_kcal(45.0, 180.0) == pytest.approx(25.0)
    assert score_attribute("vitamin_c", 0.0) == 0.0
    assert score_attribute("vitamin_c", 22.5) == 10.0
    assert score_attribute("vitamin_c", 45.0) == 10.0
    assert score_attribute("sodium", 0.0) == 0.0
    assert score_attribute("sodium", 575.0) == -10.0
    assert score_attribute("sodium", 700.0) == -10.0
    with pytest.raises(ValueError, match="energy_kcal"):
        to_per_100_kcal(1.0, 0.0)


@pytest.mark.parametrize(
    ("percent_calories", "expected"),
    [
        (0.0, 0.0),
        (0.001, -1.0),
        (2.499, -1.0),
        (2.5, -2.0),
        (4.999, -2.0),
        (5.0, -3.0),
        (59.999, -9.0),
        (60.0, -10.0),
        (100.0, -10.0),
    ],
)
def test_table_s10_added_sugar_bins(percent_calories, expected):
    assert score_attribute("added_sugar_percent_calories", percent_calories) == expected


def test_table_s10_nova_ordering_and_fractional_interpolation():
    assert score_attribute("nova_processing_level", 1.0) == 10.0
    assert score_attribute("nova_processing_level", 2.0) == 7.5
    assert score_attribute("nova_processing_level", 3.0) == 5.0
    assert score_attribute("nova_processing_level", 3.5) == -2.5
    assert score_attribute("nova_processing_level", 4.0) == -10.0


def test_table_s10_log_ratios_apply_published_targets_and_minimum_exposure_gates():
    assert score_attribute(
        "unsaturated_to_saturated_fat_ratio",
        math.exp(-0.66),
        context={"fat_energy_percent": 10.0},
    ) == pytest.approx(-10.0)
    assert score_attribute(
        "unsaturated_to_saturated_fat_ratio",
        math.exp(1.77),
        context={"fat_energy_percent": 10.0},
    ) == pytest.approx(10.0)
    assert score_attribute(
        "unsaturated_to_saturated_fat_ratio",
        math.exp(1.77),
        context={"fat_energy_percent": 9.999},
    ) == 0.0
    assert score_attribute(
        "fiber_to_carbohydrate_ratio",
        math.exp(-0.78),
        context={"carbohydrate_energy_percent": 10.0},
    ) == pytest.approx(10.0)
    assert score_attribute(
        "potassium_to_sodium_ratio",
        math.exp(3.30),
        context={"potassium_mg": 10.0, "sodium_mg": 10.0},
    ) == pytest.approx(10.0)
    assert score_attribute(
        "potassium_to_sodium_ratio",
        math.exp(3.30),
        context={"potassium_mg": 9.999, "sodium_mg": 10.0},
    ) == 0.0


def test_table_s10_dairy_and_emerging_attribute_half_weights_apply_to_points():
    ratio = math.exp(1.77)
    assert score_attribute(
        "unsaturated_to_saturated_fat_ratio",
        ratio,
        context={"fat_energy_percent": 10.0},
    ) == pytest.approx(10.0)
    assert score_attribute(
        "unsaturated_to_saturated_fat_ratio",
        ratio,
        context={"fat_energy_percent": 10.0, "is_dairy": True},
    ) == pytest.approx(5.0)
    assert score_attribute("total_protein", 14.0) == pytest.approx(5.0)
    assert score_attribute("cholesterol", 75.0) == pytest.approx(-5.0)
    assert score_attribute("fermentation_percent_calories", 50.0) == pytest.approx(5.0)
    assert score_attribute("frying", True) == pytest.approx(-5.0)


def test_table_s10_fermentation_keyword_and_binary_additives():
    assert score_attribute("fermentation_percent_calories", 0.0) == 0.0
    assert score_attribute(
        "fermentation_percent_calories", 0.0, context={"is_other_fermented_product": True}
    ) == 5.0
    assert score_attribute("artificial_sweeteners_flavors_or_colors", True) == -1.0
    assert score_attribute("artificial_sweeteners_flavors_or_colors", False) == 0.0


def test_nitrites_exposes_the_footnote_primary_and_table_sensitivity_versions():
    rule = FCS2_RULES["nitrites_percent_calories_from_processed_meat"]
    assert rule.active is True
    assert rule.ambiguity
    assert rule.primary_version == NITRITE_RULE_PRIMARY_VERSION == "footnote_50"
    assert score_attribute(rule, 25.0) == pytest.approx(-5.0)
    assert score_attribute(
        rule,
        25.0,
        nitrite_rule_version=NITRITE_RULE_TABLE_SENSITIVITY_VERSION,
    ) == pytest.approx(-10.0)
    with pytest.raises(ValueError, match="nitrite_rule_version"):
        score_attribute(rule, 25.0, nitrite_rule_version="unpublished")


def test_aggregation_uses_signed_absolute_top_k_and_published_domain_weights():
    vitamin_scores = {
        name: score
        for name, score in zip(
            [rule.name for rule in FCS2_RULES.values() if rule.domain == "vitamins"],
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0],
            strict=True,
        )
    }
    mineral_scores = {
        name: score
        for name, score in zip(
            [
                rule.name
                for rule in FCS2_RULES.values()
                if rule.domain == "minerals" and rule.active
            ],
            [10.0, 9.0, 8.0, 7.0, -10.0, 6.0, 5.0, 4.0, 3.0],
            strict=True,
        )
    }
    lipid_scores = {
        "cholesterol": -8.0,
        "medium_chain_fatty_acids": 7.0,
        "alpha_linolenic_acid": 6.0,
        "epa_plus_dha": 5.0,
    }

    domains = aggregate_domains(vitamin_scores | mineral_scores | lipid_scores)

    assert domains["vitamins"] == pytest.approx(8.0)
    assert domains["minerals"] == pytest.approx(4.8)
    assert domains["specific_lipids"] == pytest.approx(5.0 / 6.0)


def test_aggregation_applies_processing_fiber_and_phytochemical_weights():
    domains = aggregate_domains(
        {
            "nova_processing_level": 10.0,
            "fermentation_percent_calories": 5.0,
            "frying": -5.0,
            "total_fiber": 10.0,
            "total_protein": 5.0,
            "total_flavonoids": 10.0,
            "total_carotenoids": 10.0,
        }
    )

    assert domains["processing"] == pytest.approx(10.0 / 3.0)
    assert domains["fiber_and_protein"] == pytest.approx(7.5)
    assert domains["phytochemicals"] == pytest.approx(5.0)


def test_table_s10_unscaled_truncation_and_exact_inverse_scaling():
    assert unscaled_to_fcs(-50.0) == 1.0
    assert unscaled_to_fcs(-12.1) == 1.0
    assert unscaled_to_fcs(35.0) == 100.0
    assert unscaled_to_fcs(50.0) == 100.0
    assert fcs_to_unscaled(1.0) == pytest.approx(-12.1)
    assert fcs_to_unscaled(100.0) == pytest.approx(35.0)
    assert fcs_to_unscaled(unscaled_to_fcs(7.25)) == pytest.approx(7.25)
