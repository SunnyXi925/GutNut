from dataclasses import FrozenInstanceError, fields, replace
import inspect

import numpy as np
import pandas as pd
import pytest

from gmnps.scoring.attribute_calibration import (
    LOCKED_ATTRIBUTE_RESPONSE_TEMPERATURE,
    AttributeCalibrationResult,
    attribute_response,
    calibrate_attribute_points,
    fit_beta_normalization,
    validate_calibration_result,
)
from gmnps.scoring.attribute_recomposition import (
    FINAL_DEVIATION_CAP_MODES,
    BaselineDecomposition,
    compose_personalized_fcs,
    decompose_official_baseline,
    recompute_personalized_domains,
)
from gmnps.scoring.fcs2_attribute_mapping import PRIMARY_ATTRIBUTE_MAPPINGS
from gmnps.scoring.fcs2_attribute_rules import (
    FCS2_RULES,
    NOT_CALCULATED,
    aggregate_domains,
    fcs_to_unscaled,
    select_domain_attributes,
)


ACTIVE_ATTRIBUTES = tuple(name for name, rule in FCS2_RULES.items() if rule.active)
ALL_BETA_NUTRIENTS = tuple(
    dict.fromkeys(row.nutrient for row in PRIMARY_ATTRIBUTE_MAPPINGS)
)
TARGET_ATTRIBUTES = tuple(
    dict.fromkeys(
        row.attribute for row in PRIMARY_ATTRIBUTE_MAPPINGS if row.role == "effect"
    )
)
TARGET_DOMAINS = tuple(
    dict.fromkeys(FCS2_RULES[attribute].domain for attribute in TARGET_ATTRIBUTES)
)


def _domain_attributes(domain):
    return tuple(
        name
        for name, rule in FCS2_RULES.items()
        if rule.active and rule.domain == domain
    )


def _baseline(food_ids=("food_1",), *, not_calculated=()):
    values = {
        attribute: (
            FCS2_RULES[attribute].low_points + FCS2_RULES[attribute].high_points
        )
        / 2.0
        for attribute in ACTIVE_ATTRIBUTES
    }
    baseline = pd.DataFrame(
        [values.copy() for _ in food_ids],
        index=pd.Index(food_ids, name="food_id"),
        columns=ACTIVE_ATTRIBUTES,
    )
    for attribute in not_calculated:
        baseline[attribute] = pd.Series(
            [NOT_CALCULATED] * len(baseline), index=baseline.index, dtype=object
        )
    return baseline


def _development_beta():
    values = np.repeat(
        np.asarray([[-1.0], [0.0], [1.0]]), len(ALL_BETA_NUTRIENTS), axis=1
    )
    return pd.DataFrame(
        values,
        index=["dev_1", "dev_2", "dev_3"],
        columns=ALL_BETA_NUTRIENTS,
    )


def _exposures(food_ids=("food_1",), *, multiplier=0.0):
    columns = tuple(
        dict.fromkeys(
            ALL_BETA_NUTRIENTS
            + (
                "Sodium (mg)",
                "Fatty acids, total monounsaturated (g)",
                "Fatty acids, total polyunsaturated (g)",
                "Folate, DFE (mcg_DFE)",
            )
        )
    )
    frame = pd.DataFrame(multiplier, index=food_ids, columns=columns, dtype=float)
    frame.index.name = "food_id"
    frame.attrs["basis"] = "per_100_kcal"
    return frame


def _calibration(
    baseline,
    *,
    transformed=None,
    exposure_updates=None,
    exposure_multiplier=0.0,
    mode="primary",
):
    state = fit_beta_normalization(_development_beta())
    transformed = transformed or {"person_1": {}}
    raw = pd.DataFrame(0.0, index=transformed, columns=state.nutrient_order)
    for individual_id, values in transformed.items():
        for nutrient, transformed_value in values.items():
            raw.loc[individual_id, nutrient] = (
                state.medians[nutrient]
                + state.scales[nutrient]
                * 2.0
                * np.arctanh(float(transformed_value))
            )
    exposures = _exposures(tuple(baseline.index), multiplier=exposure_multiplier)
    for nutrient, value in (exposure_updates or {}).items():
        exposures.loc[:, nutrient] = value
    responses = attribute_response(state, raw, exposures)
    return calibrate_attribute_points(baseline, responses, mode=mode)


def _decomposition(official, baseline, domains):
    columns = tuple(
        attribute
        for attribute in ACTIVE_ATTRIBUTES
        if FCS2_RULES[attribute].domain in domains
    )
    return decompose_official_baseline(
        official,
        baseline.loc[:, columns],
        recomputed_domains=domains,
    )


def _forge_calibration(result, **changes):
    forged = object.__new__(AttributeCalibrationResult)
    for field in fields(AttributeCalibrationResult):
        object.__setattr__(forged, field.name, getattr(result, field.name))
    for name, value in changes.items():
        object.__setattr__(forged, name, value)
    return forged


@pytest.mark.parametrize("official_values", [[1.0, 47.0, 100.0], [5.0, 55.0, 95.0]])
def test_zero_response_recovers_every_official_score_exactly(official_values):
    foods = tuple(f"food_{number}" for number in range(len(official_values)))
    baseline = _baseline(foods)
    official = pd.Series(official_values, index=pd.Index(foods, name="food_id"), name="FCS2")
    decomposition = _decomposition(official, baseline, TARGET_DOMAINS)
    recomputed = recompute_personalized_domains(
        decomposition, _calibration(baseline)
    )

    result = compose_personalized_fcs(decomposition, recomputed, cap_mode="primary")

    expected = pd.Series(
        official_values,
        index=pd.MultiIndex.from_product(
            [["person_1"], foods], names=["individual_id", "food_id"]
        ),
        name="personalized_fcs",
    )
    pd.testing.assert_series_equal(result.scores, expected, atol=1e-8, rtol=0.0)
    assert np.max(np.abs(result.deltas.to_numpy())) <= 1e-8
    assert decomposition.anchor_interpretation == "latent_anchor_from_supplied_official_fcs"
    assert set(decomposition.diagnostics["official_anchor_kind"]) == {
        "published_score_implied_latent_unscaled_anchor"
    }


def test_top_five_and_top_three_membership_are_reselected_after_calibration():
    baseline = _baseline()
    vitamins = _domain_attributes("vitamins")
    vitamin_values = [10.0, 9.0, 8.0, 7.0, 6.0] + [0.0] * (len(vitamins) - 5)
    baseline.loc["food_1", vitamins] = vitamin_values
    baseline.loc["food_1", "vitamin_c"] = 5.5
    baseline.loc["food_1", ["cholesterol", "medium_chain_fatty_acids", "alpha_linolenic_acid", "epa_plus_dha"]] = [
        -5.8,
        5.7,
        5.6,
        5.5,
    ]
    calibration = _calibration(
        baseline,
        transformed={
            "person_1": {"Vitamin C (mg)": 0.99, "Cholesterol (mg)": 0.99}
        },
        exposure_updates={"Vitamin C (mg)": 22.5, "Cholesterol (mg)": 75.0},
    )
    decomposition = _decomposition(
        pd.Series([50.0], index=baseline.index, name="FCS2"),
        baseline,
        ("vitamins", "specific_lipids"),
    )

    recomputed = recompute_personalized_domains(decomposition, calibration)
    audit = recomputed.membership_audit

    vitamin = audit.xs(("person_1", "food_1", "vitamin_c"))
    cholesterol = audit.xs(("person_1", "food_1", "cholesterol"))
    assert not vitamin["baseline_selected"]
    assert vitamin["personalized_selected"]
    assert cholesterol["baseline_selected"]
    assert not cholesterol["personalized_selected"]


def test_membership_audit_and_domain_score_match_the_shared_task1_selector():
    baseline = _baseline()
    vitamins = _domain_attributes("vitamins")
    baseline.loc["food_1", vitamins] = np.arange(len(vitamins), dtype=float) % 11
    decomposition = _decomposition(
        pd.Series([50.0], index=baseline.index, name="FCS2"),
        baseline,
        ("vitamins",),
    )
    recomputed = recompute_personalized_domains(
        decomposition, _calibration(baseline)
    )
    values = {
        attribute: recomputed.attribute_audit.loc[
            ("person_1", "food_1", attribute), "personalized_points"
        ]
        for attribute in vitamins
    }
    weights = {
        attribute: decomposition.effective_attribute_weights.loc[
            "food_1", attribute
        ]
        for attribute in vitamins
    }
    selection = select_domain_attributes(values, "vitamins", weights)
    audit = recomputed.membership_audit.xs(("person_1", "food_1"))

    assert set(audit.index[audit["personalized_selected"]]) == set(
        selection.selected_attributes
    )
    assert audit["active_weight_denominator"].nunique() == 1
    assert audit["active_weight_denominator"].iloc[0] == pytest.approx(
        selection.active_weight_denominator
    )
    expected = aggregate_domains(
        values, effective_attribute_weights=weights
    )["vitamins"]
    assert recomputed.domain_contributions.loc[
        ("person_1", "food_1"), "vitamins"
    ] == pytest.approx(expected)


def test_fixed_residual_is_food_specific_but_identical_across_individuals():
    baseline = _baseline(("food_1", "food_2"))
    official = pd.Series([30.0, 70.0], index=baseline.index, name="FCS2")
    decomposition = _decomposition(official, baseline, ("vitamins",))
    calibration = _calibration(
        baseline,
        transformed={"person_a": {}, "person_b": {"Vitamin C (mg)": 0.8}},
        exposure_updates={"Vitamin C (mg)": 22.5},
    )

    recomputed = recompute_personalized_domains(decomposition, calibration)
    result = compose_personalized_fcs(decomposition, recomputed)

    for food_id in baseline.index:
        rows = result.diagnostics.xs(food_id, level="food_id")
        assert rows["fixed_residual"].nunique() == 1
        assert rows["fixed_residual"].iloc[0] == decomposition.fixed_residual.loc[food_id]


def test_residualized_baseline_points_never_enter_selected_domain_delta():
    baseline_a = _baseline()
    baseline_b = baseline_a.copy()
    baseline_b.loc["food_1", "nova_processing_level"] = -10.0
    decomposition = _decomposition(
        pd.Series([50.0], index=baseline_a.index, name="FCS2"),
        baseline_a,
        ("vitamins",),
    )
    calibration_a = _calibration(
        baseline_a,
        transformed={"person_1": {"Vitamin C (mg)": 0.8}},
        exposure_updates={"Vitamin C (mg)": 22.5},
    )
    calibration_b = _calibration(
        baseline_b,
        transformed={"person_1": {"Vitamin C (mg)": 0.8}},
        exposure_updates={"Vitamin C (mg)": 22.5},
    )

    first = recompute_personalized_domains(decomposition, calibration_a)
    second = recompute_personalized_domains(decomposition, calibration_b)

    pd.testing.assert_frame_equal(first.domain_contributions, second.domain_contributions)
    pd.testing.assert_frame_equal(first.attribute_audit, second.attribute_audit)


def test_calibrating_an_attribute_in_a_residualized_domain_fails_closed():
    baseline = _baseline()
    decomposition = _decomposition(
        pd.Series([50.0], index=baseline.index, name="FCS2"),
        baseline,
        ("vitamins",),
    )
    calibration = _calibration(
        baseline,
        transformed={"person_1": {"Cholesterol (mg)": 0.8}},
        exposure_updates={"Cholesterol (mg)": 75.0},
    )

    with pytest.raises(ValueError, match="residualized domain.*specific_lipids"):
        recompute_personalized_domains(decomposition, calibration)


def test_dynamic_not_calculated_ratio_is_excluded_from_active_weight_denominator():
    baseline = _baseline(not_calculated=("potassium_to_sodium_ratio",))
    baseline.loc["food_1", "unsaturated_to_saturated_fat_ratio"] = 10.0
    baseline.loc["food_1", "fiber_to_carbohydrate_ratio"] = 0.0
    decomposition = _decomposition(
        pd.Series([60.0], index=baseline.index, name="FCS2"),
        baseline,
        ("nutrient_ratios",),
    )

    recomputed = recompute_personalized_domains(
        decomposition, _calibration(baseline)
    )
    audit = recomputed.membership_audit.xs(
        ("person_1", "food_1", "potassium_to_sodium_ratio")
    )

    assert decomposition.baseline_domain_contributions.loc["food_1", "nutrient_ratios"] == 5.0
    assert recomputed.domain_contributions.loc[("person_1", "food_1"), "nutrient_ratios"] == 5.0
    assert not audit["calculated"]
    assert audit["active_weight_denominator"] == 2.0


@pytest.mark.parametrize("cap_mode, cap", [("low", 8.0), ("primary", 12.0), ("high", 15.0)])
def test_only_named_final_deviation_caps_are_applied(cap_mode, cap):
    baseline = _baseline()
    for attribute in TARGET_ATTRIBUTES:
        rule = FCS2_RULES[attribute]
        baseline.loc["food_1", attribute] = (
            min(rule.low_points, rule.high_points)
            + max(rule.low_points, rule.high_points)
        ) / 2.0
    official = pd.Series([40.0], index=baseline.index, name="FCS2")
    decomposition = _decomposition(official, baseline, TARGET_DOMAINS)
    calibration = _calibration(
        baseline,
        transformed={
            "person_1": {nutrient: 0.99 for nutrient in ALL_BETA_NUTRIENTS}
        },
        exposure_multiplier=1000.0,
        mode="high",
    )
    recomputed = recompute_personalized_domains(decomposition, calibration)

    result = compose_personalized_fcs(decomposition, recomputed, cap_mode=cap_mode)

    assert FINAL_DEVIATION_CAP_MODES[cap_mode] == cap
    assert result.deltas.iloc[0] == pytest.approx(cap)
    assert result.diagnostics.iloc[0]["deviation_cap_applied"]


@pytest.mark.parametrize("bad_mode", [8, 12.0, 15, "medium", None])
def test_arbitrary_numeric_or_unnamed_final_caps_are_rejected(bad_mode):
    baseline = _baseline()
    decomposition = _decomposition(
        pd.Series([50.0], index=baseline.index, name="FCS2"), baseline, ("vitamins",)
    )
    recomputed = recompute_personalized_domains(decomposition, _calibration(baseline))

    with pytest.raises((TypeError, ValueError), match="cap_mode"):
        compose_personalized_fcs(decomposition, recomputed, cap_mode=bad_mode)
    assert "cap" not in inspect.signature(compose_personalized_fcs).parameters


def test_native_unscaled_truncation_occurs_before_final_score_cap():
    baseline = _baseline()
    official = pd.Series([99.0], index=baseline.index, name="FCS2")
    decomposition = _decomposition(official, baseline, ("vitamins",))
    calibration = _calibration(
        baseline,
        transformed={"person_1": {"Vitamin C (mg)": 0.99}},
        exposure_updates={"Vitamin C (mg)": 1000.0},
        mode="high",
    )
    recomputed = recompute_personalized_domains(decomposition, calibration)

    result = compose_personalized_fcs(decomposition, recomputed, cap_mode="low")

    assert result.unscaled_raw.iloc[0] > 35.0
    assert result.diagnostics.iloc[0]["native_truncated"]
    assert result.diagnostics.iloc[0]["native_fcs"] == 100.0
    assert result.scores.iloc[0] == 100.0
    assert result.deltas.iloc[0] == 1.0


def test_nonzero_mean_response_remains_nonzero_without_hidden_centering():
    baseline = _baseline()
    decomposition = _decomposition(
        pd.Series([50.0], index=baseline.index, name="FCS2"), baseline, ("vitamins",)
    )
    calibration = _calibration(
        baseline,
        transformed={
            "person_a": {"Vitamin C (mg)": 0.5},
            "person_b": {"Vitamin C (mg)": 0.9},
        },
        exposure_updates={"Vitamin C (mg)": 22.5},
    )
    recomputed = recompute_personalized_domains(decomposition, calibration)

    result = compose_personalized_fcs(decomposition, recomputed)

    assert (result.deltas > 0.0).all()
    assert result.deltas.mean() > 0.0


@pytest.mark.parametrize(
    "official",
    [
        pd.Series([np.nan], index=pd.Index(["food_1"], name="food_id")),
        pd.Series([0.0], index=pd.Index(["food_1"], name="food_id")),
        pd.Series([101.0], index=pd.Index(["food_1"], name="food_id")),
        pd.Series([50.0], index=pd.Index([1], name="food_id")),
        pd.Series([50.0, 60.0], index=pd.Index(["food_1", "food_1"], name="food_id")),
    ],
)
def test_decomposition_rejects_invalid_official_scores_and_labels(official):
    baseline = _baseline(tuple(official.index))

    with pytest.raises(ValueError, match="official|food ID|unique|between"):
        _decomposition(official, baseline, ("vitamins",))


def test_decomposition_requires_exact_complete_selected_domain_labels():
    baseline = _baseline()
    official = pd.Series([50.0], index=baseline.index, name="FCS2")
    vitamin_points = baseline.loc[:, _domain_attributes("vitamins")]

    with pytest.raises(ValueError, match="exact active rule labels.*vitamins"):
        decompose_official_baseline(
            official,
            vitamin_points.drop(columns="vitamin_c"),
            recomputed_domains=("vitamins",),
        )
    with pytest.raises(ValueError, match="residualized domain"):
        decompose_official_baseline(
            official,
            pd.concat([vitamin_points, baseline[["cholesterol"]]], axis=1),
            recomputed_domains=("vitamins",),
        )


def test_decomposition_is_immutable_copy_safe_and_fingerprint_bound():
    baseline = _baseline()
    decomposition = _decomposition(
        pd.Series([50.0], index=baseline.index, name="FCS2"), baseline, ("vitamins",)
    )

    with pytest.raises(FrozenInstanceError):
        decomposition.fingerprint = "0" * 64
    exposed = decomposition.baseline_points
    exposed.loc["food_1", "vitamin_c"] = 999.0
    assert decomposition.baseline_points.loc["food_1", "vitamin_c"] != 999.0
    with pytest.raises(ValueError, match="fingerprint"):
        replace(decomposition, fingerprint="0" * 64)


def test_recomposition_rejects_arbitrary_frames_and_calibration_tampering():
    baseline = _baseline()
    decomposition = _decomposition(
        pd.Series([50.0], index=baseline.index, name="FCS2"), baseline, ("vitamins",)
    )
    calibration = _calibration(baseline)

    with pytest.raises(TypeError, match="AttributeCalibrationResult"):
        recompute_personalized_domains(decomposition, calibration.points)

    changed = calibration.points.copy()
    changed.loc[("person_1", "food_1"), "vitamin_c"] += 1.0
    tampered = _forge_calibration(calibration, points=changed)
    with pytest.raises(ValueError, match="calibration.*inconsistent|tamper"):
        recompute_personalized_domains(decomposition, tampered)


def test_nonpersonalized_attributes_in_selected_domains_must_stay_at_baseline():
    baseline = _baseline()
    decomposition = _decomposition(
        pd.Series([50.0], index=baseline.index, name="FCS2"), baseline, ("minerals",)
    )
    calibration = _calibration(baseline)
    points = calibration.points.copy()
    deltas = calibration.deltas.copy()
    diagnostics = calibration.diagnostics.copy()
    key = ("person_1", "food_1", "zinc")
    response = 1.0
    raw_delta = 2.0 * np.tanh(response / LOCKED_ATTRIBUTE_RESPONSE_TEMPERATURE)
    points.loc[("person_1", "food_1"), "zinc"] += raw_delta
    deltas.loc[("person_1", "food_1"), "zinc"] = raw_delta
    diagnostics.loc[key, "attribute_response"] = response
    diagnostics.loc[key, "raw_delta"] = raw_delta
    diagnostics.loc[key, "point_delta"] = raw_delta
    forged = _forge_calibration(
        calibration, points=points, deltas=deltas, diagnostics=diagnostics
    )

    with pytest.raises(ValueError, match="outside reviewed target.*zinc"):
        recompute_personalized_domains(decomposition, forged)


def test_recomposition_inherits_exact_task3_provenance_without_reminting():
    baseline = _baseline()
    decomposition = _decomposition(
        pd.Series([50.0], index=baseline.index, name="FCS2"), baseline, ("vitamins",)
    )
    calibration = _calibration(baseline)

    recomputed = recompute_personalized_domains(decomposition, calibration)

    assert validate_calibration_result(calibration) is calibration
    assert recomputed.calibration_fingerprint == calibration.calibration_fingerprint
    assert recomputed.normalization_fingerprint == calibration.normalization_fingerprint
    assert recomputed.mapping_version == calibration.mapping_version
    assert recomputed.reviewed_targets == calibration.reviewed_targets
    assert recomputed.response_fingerprint == calibration.response_fingerprint
    assert recomputed.calibration_mode == calibration.calibration_mode
    assert recomputed.calibration_fraction == calibration.calibration_fraction


@pytest.mark.parametrize(
    "changes",
    [
        {"mapping_version": "fiber_all_ratio"},
        {"reviewed_targets": ("zinc",)},
        {"response_fingerprint": "0" * 64},
        {"calibration_mode": "low"},
        {"calibration_fraction": 0.10},
    ],
)
def test_recomposition_rejects_forged_task3_provenance(changes):
    baseline = _baseline()
    decomposition = _decomposition(
        pd.Series([50.0], index=baseline.index, name="FCS2"), baseline, ("vitamins",)
    )
    calibration = _forge_calibration(_calibration(baseline), **changes)

    with pytest.raises(ValueError, match="fingerprint|mapping|reviewed target|mode|fraction"):
        recompute_personalized_domains(decomposition, calibration)


def test_recomposition_rejects_tampered_not_calculated_audit():
    baseline = _baseline(not_calculated=("potassium_to_sodium_ratio",))
    decomposition = _decomposition(
        pd.Series([50.0], index=baseline.index, name="FCS2"),
        baseline,
        ("nutrient_ratios",),
    )
    calibration = _calibration(baseline)
    diagnostics = calibration.diagnostics.copy()
    diagnostics.loc[
        ("person_1", "food_1", "potassium_to_sodium_ratio"), "clipped"
    ] = True

    with pytest.raises(ValueError, match="NOT_CALCULATED|calibration fingerprint"):
        recompute_personalized_domains(
            decomposition,
            _forge_calibration(calibration, diagnostics=diagnostics),
        )


def test_composition_rejects_cross_decomposition_provenance():
    baseline = _baseline()
    first = _decomposition(
        pd.Series([40.0], index=baseline.index, name="FCS2"), baseline, ("vitamins",)
    )
    second = _decomposition(
        pd.Series([60.0], index=baseline.index, name="FCS2"), baseline, ("vitamins",)
    )
    recomputed = recompute_personalized_domains(first, _calibration(baseline))

    with pytest.raises(ValueError, match="provenance|fingerprint"):
        compose_personalized_fcs(second, recomputed)


def test_decomposition_uses_task1_aggregation_and_published_weights():
    baseline = _baseline()
    selected = ("vitamins", "specific_lipids", "phytochemicals")
    decomposition = _decomposition(
        pd.Series([50.0], index=baseline.index, name="FCS2"), baseline, selected
    )
    expected = aggregate_domains(
        {
            attribute: baseline.loc["food_1", attribute]
            for attribute in decomposition.attribute_labels
        }
    )

    assert decomposition.baseline_domain_contributions.loc["food_1"].to_dict() == pytest.approx(expected)
    assert decomposition.fixed_residual.loc["food_1"] == pytest.approx(
        fcs_to_unscaled(50.0) - sum(expected.values())
    )
