from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from gmnps.scoring.fcs2_attribute_mapping import (
    PRIMARY_ATTRIBUTE_MAPPINGS,
    PRIMARY_MAPPING_VERSION,
    SENSITIVITY_ATTRIBUTE_MAPPINGS,
    AttributeCalibrationMapping,
    build_allocation_matrix,
    build_food_specific_response,
    reconstruction_status_table,
    validate_attribute_mappings,
)
from gmnps.scoring.fcs2_attribute_rules import FCS2_RULES


EXCLUDED_NUTRIENTS = {
    "Carbohydrate (g)",
    "Zinc (mg)",
    "Copper (mg)",
    "Vitamin A, RAE (mcg_RAE)",
}

CANONICAL_COLUMNS = [
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
    "Carbohydrate (g)",
    "Zinc (mg)",
    "Copper (mg)",
    "Vitamin A, RAE (mcg_RAE)",
]


def test_mapping_record_is_immutable():
    row = PRIMARY_ATTRIBUTE_MAPPINGS[0]

    with pytest.raises(FrozenInstanceError):
        row.allocation_weight = 0.0


def test_primary_effect_mappings_target_active_food_compass_attributes():
    effect_rows = [row for row in PRIMARY_ATTRIBUTE_MAPPINGS if row.role == "effect"]

    assert effect_rows
    assert all(row.attribute in FCS2_RULES for row in effect_rows)
    assert all(FCS2_RULES[row.attribute].active for row in effect_rows)
    assert all(row.channel in {"MAC", "LIPID"} for row in effect_rows)
    assert not any(row.nutrient == "OTHER" or row.attribute == "OTHER" for row in PRIMARY_ATTRIBUTE_MAPPINGS)


def test_primary_effect_allocations_sum_to_one_per_nutrient():
    totals: dict[str, float] = {}
    for row in PRIMARY_ATTRIBUTE_MAPPINGS:
        if row.role == "effect":
            totals[row.nutrient] = totals.get(row.nutrient, 0.0) + row.allocation_weight

    assert totals
    assert set(totals.values()) == {1.0}
    validate_attribute_mappings(PRIMARY_ATTRIBUTE_MAPPINGS)


def test_carbohydrate_proxy_registry_has_its_effective_mac_channel():
    rows = SENSITIVITY_ATTRIBUTE_MAPPINGS["carbohydrate_proxy"]
    carbohydrate = [row for row in rows if row.nutrient == "Carbohydrate (g)"]

    assert len(carbohydrate) == 1
    assert carbohydrate[0].role == "effect"
    assert carbohydrate[0].channel == "MAC"


def test_primary_exclusions_and_role_only_variables_have_zero_allocation():
    rows_by_nutrient: dict[str, list[AttributeCalibrationMapping]] = {}
    for row in PRIMARY_ATTRIBUTE_MAPPINGS:
        rows_by_nutrient.setdefault(row.nutrient, []).append(row)

    assert EXCLUDED_NUTRIENTS <= rows_by_nutrient.keys()
    for nutrient in EXCLUDED_NUTRIENTS:
        assert all(row.role != "effect" and row.allocation_weight == 0.0 for row in rows_by_nutrient[nutrient])

    assert {row.role for row in rows_by_nutrient["Total Fat (g)"]} == {"applicability_only"}
    for nutrient in {"4:0 (g)", "6:0 (g)", "14:0 (g)", "16:0 (g)", "18:0 (g)"}:
        assert {row.role for row in rows_by_nutrient[nutrient]} == {"explanatory_only"}


def test_primary_mapping_has_required_component_policies_and_exact_splits():
    keyed = {(row.nutrient, row.attribute): row for row in PRIMARY_ATTRIBUTE_MAPPINGS}

    assert keyed[("Folate, food (mcg)", "folate_dfe_b9")].component_policy == "component_fraction"
    assert keyed[("Retinol (mcg)", "vitamin_a_rae")].component_policy == "component_fraction"
    saturated_ratio = keyed[("Fatty acids, total saturated (g)", "unsaturated_to_saturated_fat_ratio")]
    assert saturated_ratio.component_policy == "denominator_component"
    assert keyed[("Fiber, total dietary (g)", "fiber_to_carbohydrate_ratio")].allocation_weight == 0.5
    assert keyed[("Fiber, total dietary (g)", "total_fiber")].allocation_weight == 0.5
    assert keyed[("Potassium (mg)", "potassium_to_sodium_ratio")].allocation_weight == 0.5
    assert keyed[("Potassium (mg)", "potassium")].allocation_weight == 0.5
    assert all(row.version == PRIMARY_MAPPING_VERSION for row in PRIMARY_ATTRIBUTE_MAPPINGS)


def test_unsaturated_exposure_is_fixed_and_never_receives_an_expert_mask_beta():
    all_registries = (PRIMARY_ATTRIBUTE_MAPPINGS, *SENSITIVITY_ATTRIBUTE_MAPPINGS.values())
    unsaturated_columns = {
        "Fatty acids, total monounsaturated (g)",
        "Fatty acids, total polyunsaturated (g)",
    }

    for rows in all_registries:
        assert not any(row.nutrient in unsaturated_columns and row.role == "effect" for row in rows)
    primary_ratio_effects = {
        row.nutrient
        for row in PRIMARY_ATTRIBUTE_MAPPINGS
        if row.attribute == "unsaturated_to_saturated_fat_ratio" and row.role == "effect"
    }
    assert primary_ratio_effects == {"Fatty acids, total saturated (g)"}


def test_mapping_validation_rejects_duplicates():
    row = PRIMARY_ATTRIBUTE_MAPPINGS[0]

    with pytest.raises(ValueError, match="duplicate"):
        validate_attribute_mappings((row, row))


def test_mapping_validation_rejects_duplicates_independent_of_version():
    row = PRIMARY_ATTRIBUTE_MAPPINGS[2]

    with pytest.raises(ValueError, match="duplicate"):
        validate_attribute_mappings((row, replace(row, version="another_version")))


def test_mapping_validation_requires_one_registry_version():
    first = PRIMARY_ATTRIBUTE_MAPPINGS[2]
    second = replace(PRIMARY_ATTRIBUTE_MAPPINGS[3], version="another_version")

    with pytest.raises(ValueError, match="one registry version"):
        validate_attribute_mappings((first, second))


def test_mapping_validation_requires_exact_allocation_conservation():
    row = replace(PRIMARY_ATTRIBUTE_MAPPINGS[2], allocation_weight=1.0 + 5e-13)

    with pytest.raises(ValueError, match="sum exactly to 1"):
        validate_attribute_mappings((row,))


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (replace(PRIMARY_ATTRIBUTE_MAPPINGS[0], attribute="not_an_attribute"), "unknown Food Compass"),
        (replace(PRIMARY_ATTRIBUTE_MAPPINGS[0], attribute="iodine"), "inactive Food Compass"),
        (replace(PRIMARY_ATTRIBUTE_MAPPINGS[0], nutrient="OTHER"), "OTHER catch-all"),
        (replace(PRIMARY_ATTRIBUTE_MAPPINGS[-1], allocation_weight=0.1), "excluded mappings must have zero"),
    ],
)
def test_mapping_validation_rejects_invalid_targets_and_role_allocations(row, message):
    with pytest.raises(ValueError, match=message):
        validate_attribute_mappings((row,))


@pytest.mark.parametrize(
    "nutrient",
    [
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
    ],
)
def test_mapping_validation_rejects_protected_nutrient_promotion(nutrient):
    row = next(row for row in PRIMARY_ATTRIBUTE_MAPPINGS if row.nutrient == nutrient)
    promoted = replace(row, role="effect", allocation_weight=1.0)

    with pytest.raises(ValueError, match="protected nutrient"):
        validate_attribute_mappings((promoted,))


@pytest.mark.parametrize(
    ("version", "nutrient", "expected_attribute"),
    [
        ("fiber_all_ratio", "Fiber, total dietary (g)", "fiber_to_carbohydrate_ratio"),
        ("fiber_all_absolute", "Fiber, total dietary (g)", "total_fiber"),
        ("potassium_all_ratio", "Potassium (mg)", "potassium_to_sodium_ratio"),
        ("potassium_all_absolute", "Potassium (mg)", "potassium"),
    ],
)
def test_sensitivity_variants_reallocate_to_one_target(version, nutrient, expected_attribute):
    rows = SENSITIVITY_ATTRIBUTE_MAPPINGS[version]
    selected = [row for row in rows if row.nutrient == nutrient and row.role == "effect"]

    assert [(row.attribute, row.allocation_weight) for row in selected] == [(expected_attribute, 1.0)]
    validate_attribute_mappings(rows)


def test_allocation_variants_do_not_activate_carbohydrate_proxy():
    primary = [row for row in PRIMARY_ATTRIBUTE_MAPPINGS if row.nutrient == "Carbohydrate (g)"]
    assert [(row.role, row.allocation_weight) for row in primary] == [("sensitivity_proxy_only", 0.0)]

    allocation_versions = {
        "fiber_all_ratio",
        "fiber_all_absolute",
        "potassium_all_ratio",
        "potassium_all_absolute",
    }
    for version in allocation_versions:
        rows = SENSITIVITY_ATTRIBUTE_MAPPINGS[version]
        carbohydrate = [row for row in rows if row.nutrient == "Carbohydrate (g)"]
        assert [(row.role, row.allocation_weight) for row in carbohydrate] == [
            ("sensitivity_proxy_only", 0.0)
        ], version


def test_carbohydrate_proxy_is_a_separate_sensitivity_registry():
    assert set(SENSITIVITY_ATTRIBUTE_MAPPINGS) == {
        "fiber_all_ratio",
        "fiber_all_absolute",
        "potassium_all_ratio",
        "potassium_all_absolute",
        "carbohydrate_proxy",
    }
    rows = SENSITIVITY_ATTRIBUTE_MAPPINGS["carbohydrate_proxy"]
    carbohydrate = [row for row in rows if row.nutrient == "Carbohydrate (g)"]

    assert [(row.attribute, row.role, row.component_policy, row.allocation_weight) for row in carbohydrate] == [
        ("fiber_to_carbohydrate_ratio", "effect", "denominator_component", 1.0)
    ]
    validate_attribute_mappings(rows)


def test_allocation_matrix_preserves_labels_and_reports_zero_allocation_roles():
    allocation = build_allocation_matrix(CANONICAL_COLUMNS, PRIMARY_MAPPING_VERSION)

    assert allocation.weights.index.tolist() == CANONICAL_COLUMNS
    assert allocation.weights.columns.tolist() == [name for name, rule in FCS2_RULES.items() if rule.active]
    assert allocation.weights.loc["Fiber, total dietary (g)", "fiber_to_carbohydrate_ratio"] == 0.5
    assert allocation.weights.loc["Fiber, total dietary (g)", "total_fiber"] == 0.5
    assert allocation.weights.loc["Total Fat (g)"].sum() == 0.0
    assert allocation.role_diagnostics.loc["Total Fat (g)", "role"] == "applicability_only"
    assert allocation.role_diagnostics.loc["4:0 (g)", "role"] == "explanatory_only"
    assert allocation.role_diagnostics.loc["Zinc (mg)", "role"] == "excluded"
    assert allocation.role_diagnostics.loc["Carbohydrate (g)", "role"] == "sensitivity_proxy_only"


def test_allocation_matrix_fails_when_a_required_effect_nutrient_is_missing():
    incomplete = [column for column in CANONICAL_COLUMNS if column != "Vitamin C (mg)"]

    with pytest.raises(ValueError, match="missing required mapping nutrients.*Vitamin C"):
        build_allocation_matrix(incomplete, PRIMARY_MAPPING_VERSION)


def _complete_response_inputs():
    beta = pd.Series(0.0, index=CANONICAL_COLUMNS, name="person_1")
    exposure = pd.Series(0.0, index=CANONICAL_COLUMNS, name="food_1")
    exposure["Carbohydrate (g)"] = 10.0
    exposure["Sodium (mg)"] = 100.0
    exposure["Total Fat (g)"] = 10.0
    exposure["Fatty acids, total saturated (g)"] = 1.0
    exposure["Fatty acids, total monounsaturated (g)"] = 4.0
    exposure["Fatty acids, total polyunsaturated (g)"] = 2.0
    exposure["Folate, DFE (mcg_DFE)"] = 0.0
    exposure.attrs["basis"] = "per_100_kcal"
    return beta, exposure


def test_food_specific_response_combines_direct_composite_and_component_fraction_effects():
    beta, exposure = _complete_response_inputs()
    beta["Vitamin C (mg)"] = 2.0
    exposure["Vitamin C (mg)"] = 11.25
    beta["Carotene, alpha (mcg)"] = 1.0
    beta["Carotene, beta (mcg)"] = 2.0
    exposure["Carotene, alpha (mcg)"] = 1000.0
    exposure["Carotene, beta (mcg)"] = 500.0
    beta["Folate, food (mcg)"] = 3.0
    exposure["Folate, food (mcg)"] = 25.0
    exposure["Folate, DFE (mcg_DFE)"] = 50.0
    beta["Retinol (mcg)"] = 4.0
    exposure["Retinol (mcg)"] = 45.0
    exposure["Vitamin A, RAE (mcg_RAE)"] = 90.0
    beta["8:0 (g)"] = 1.0
    beta["10:0 (g)"] = 2.0
    exposure["8:0 (g)"] = 0.08
    exposure["10:0 (g)"] = 0.04

    result = build_food_specific_response(beta, exposure, PRIMARY_MAPPING_VERSION)

    assert result.response.index.tolist() == [name for name, rule in FCS2_RULES.items() if rule.active]
    assert result.response["vitamin_c"] == pytest.approx(1.0)
    assert result.response["total_carotenoids"] == pytest.approx(2000.0 / 8746.81)
    assert result.response["folate_dfe_b9"] == pytest.approx(3.0 * 0.25)
    assert result.response["vitamin_a_rae"] == pytest.approx(4.0 * 0.20)
    assert result.response["medium_chain_fatty_acids"] == pytest.approx((0.08 + 2.0 * 0.04) / 0.32)


def test_food_specific_response_reports_zero_component_totals_explicitly():
    beta, exposure = _complete_response_inputs()

    result = build_food_specific_response(beta, exposure)
    zero_rows = result.diagnostics[result.diagnostics["zero_total_component"]]

    assert {"total_carotenoids", "medium_chain_fatty_acids", "folate_dfe_b9", "vitamin_a_rae"} <= set(
        zero_rows["attribute"]
    )
    assert (zero_rows["status"] == "zero_total_component").all()
    assert result.response.loc[zero_rows["attribute"].unique()].eq(0.0).all()


def test_food_specific_response_excluded_and_role_only_betas_never_add_effects():
    beta, exposure = _complete_response_inputs()
    beta["Vitamin C (mg)"] = 1.0
    exposure["Vitamin C (mg)"] = 22.5
    baseline = build_food_specific_response(beta, exposure).response

    for nutrient in EXCLUDED_NUTRIENTS | {
        "Total Fat (g)",
        "4:0 (g)",
        "6:0 (g)",
        "14:0 (g)",
        "16:0 (g)",
        "18:0 (g)",
    }:
        beta[nutrient] = 1000.0

    changed = build_food_specific_response(beta, exposure).response
    pd.testing.assert_series_equal(changed, baseline)


def test_food_specific_response_denominator_effect_has_opposite_ratio_direction():
    beta, exposure = _complete_response_inputs()
    beta["Fatty acids, total saturated (g)"] = 1.0

    result = build_food_specific_response(beta, exposure)

    assert result.response["unsaturated_to_saturated_fat_ratio"] < 0.0


def test_ratio_effects_use_component_specific_normalized_exposures():
    beta, exposure = _complete_response_inputs()
    beta["Fiber, total dietary (g)"] = 1.0
    exposure["Fiber, total dietary (g)"] = 9.5
    beta["Potassium (mg)"] = 1.0
    exposure["Potassium (mg)"] = 1175.0
    beta["Fatty acids, total saturated (g)"] = 1.0
    exposure["Fatty acids, total saturated (g)"] = 2.0
    exposure["Fatty acids, total monounsaturated (g)"] = 4.0
    exposure["Fatty acids, total polyunsaturated (g)"] = 2.0

    result = build_food_specific_response(beta, exposure)

    assert result.response["fiber_to_carbohydrate_ratio"] == pytest.approx(0.5)
    assert result.response["potassium_to_sodium_ratio"] == pytest.approx(0.5)
    assert result.response["unsaturated_to_saturated_fat_ratio"] == pytest.approx(-0.25)


def test_zero_unsaturated_numerator_does_not_erase_saturated_denominator_effect():
    beta, exposure = _complete_response_inputs()
    beta["Fatty acids, total saturated (g)"] = 1.0
    exposure["Fatty acids, total saturated (g)"] = 2.0
    exposure["Fatty acids, total monounsaturated (g)"] = 0.0
    exposure["Fatty acids, total polyunsaturated (g)"] = 0.0

    result = build_food_specific_response(beta, exposure)

    assert result.response["unsaturated_to_saturated_fat_ratio"] == pytest.approx(-1.0)


def test_ratio_component_effects_respect_published_applicability_gates():
    beta, exposure = _complete_response_inputs()
    beta["Fatty acids, total saturated (g)"] = 1.0
    exposure["Total Fat (g)"] = 1.0

    fat_gated = build_food_specific_response(beta, exposure)
    assert fat_gated.response["unsaturated_to_saturated_fat_ratio"] == 0.0

    beta["Carbohydrate (g)"] = 1.0
    exposure["Carbohydrate (g)"] = 2.0
    carbohydrate_gated = build_food_specific_response(beta, exposure, "carbohydrate_proxy")
    assert carbohydrate_gated.response["fiber_to_carbohydrate_ratio"] == 0.0


def test_carbohydrate_proxy_uses_signed_clipped_energy_fraction_independent_of_fiber():
    beta, exposure = _complete_response_inputs()
    beta["Carbohydrate (g)"] = 1.0
    exposure["Carbohydrate (g)"] = 12.5
    exposure["Fiber, total dietary (g)"] = 0.0

    result = build_food_specific_response(beta, exposure, "carbohydrate_proxy")

    assert result.response["fiber_to_carbohydrate_ratio"] == pytest.approx(-0.5)

    exposure["Carbohydrate (g)"] = 50.0
    clipped = build_food_specific_response(beta, exposure, "carbohydrate_proxy")
    assert clipped.response["fiber_to_carbohydrate_ratio"] == pytest.approx(-1.0)


def test_ratio_effects_fail_closed_when_side_inputs_are_missing():
    beta, exposure = _complete_response_inputs()

    with pytest.raises(ValueError, match="missing required food exposure.*monounsaturated"):
        build_food_specific_response(beta, exposure.drop(index="Fatty acids, total monounsaturated (g)"))
    with pytest.raises(ValueError, match="missing required food exposure.*Sodium"):
        build_food_specific_response(beta, exposure.drop(index="Sodium (mg)"))


def test_food_specific_response_rejects_tampered_allocation_matrices():
    beta, exposure = _complete_response_inputs()

    def set_effect_weights(allocation):
        allocation.weights.loc[:, "vitamin_c"] = 2.0

    def set_nonfinite_weights(allocation):
        allocation.weights.loc[:, "zinc"] = float("nan")

    def set_protected_role_weight(allocation):
        allocation.weights.loc[
            "Total Fat (g)",
            "unsaturated_to_saturated_fat_ratio",
        ] = 0.1

    def add_attribute_label(allocation):
        allocation.weights.loc[:, "unexpected_attribute"] = 0.0

    def change_version(allocation):
        allocation.role_diagnostics.loc[:, "version"] = "fiber_all_ratio"

    mutations = (
        set_effect_weights,
        set_nonfinite_weights,
        set_protected_role_weight,
        add_attribute_label,
        change_version,
    )
    for mutate in mutations:
        allocation = build_allocation_matrix(CANONICAL_COLUMNS)
        mutate(allocation)
        with pytest.raises(ValueError, match="allocation matrix"):
            build_food_specific_response(beta, exposure, allocation)


def test_food_specific_response_accepts_an_untampered_allocation_matrix():
    beta, exposure = _complete_response_inputs()
    allocation = build_allocation_matrix(CANONICAL_COLUMNS)

    by_version = build_food_specific_response(beta, exposure)
    by_allocation = build_food_specific_response(beta, exposure, allocation)

    pd.testing.assert_series_equal(by_allocation.response, by_version.response)


def test_food_specific_response_requires_declared_units_and_all_required_labels():
    beta, exposure = _complete_response_inputs()
    exposure.attrs.clear()
    with pytest.raises(ValueError, match="per_100_kcal"):
        build_food_specific_response(beta, exposure)

    beta, exposure = _complete_response_inputs()
    with pytest.raises(ValueError, match="missing required food exposure.*Folate, DFE"):
        build_food_specific_response(beta, exposure.drop(index="Folate, DFE (mcg_DFE)"))

    with pytest.raises(ValueError, match="missing required beta.*Vitamin C"):
        build_food_specific_response(beta.drop(index="Vitamin C (mg)"), exposure)


def test_reconstruction_status_covers_every_active_operational_rule_once():
    status = reconstruction_status_table()
    active = {name for name, rule in FCS2_RULES.items() if rule.active}

    assert status["attribute"].is_unique
    assert set(status["attribute"]) == active
    assert status.shape[0] == 54
    assert status["missing_input_policy"].eq("unavailable_or_fixed_baseline").all()


def test_reconstruction_status_counts_and_scientific_non_substitution_labels():
    status = reconstruction_status_table().set_index("attribute")

    assert status["reconstruction_status"].value_counts().to_dict() == {
        "nutrient_derived": 30,
        "auxiliary_ingredient_or_processing_required": 22,
        "fixed_baseline_in_recomputed_domain": 2,
    }
    assert "total sugars cannot substitute" in status.loc["added_sugar_percent_calories", "reason"]
    assert "rounded NOVA cannot substitute" in status.loc["nova_processing_level", "reason"]
    assert "18:3 is not verified ALA" in status.loc["alpha_linolenic_acid", "reason"]
    assert (
        status.loc["total_flavonoids", "reconstruction_status"]
        == "fixed_baseline_in_recomputed_domain"
    )
