from dataclasses import FrozenInstanceError, fields, replace
from hashlib import sha256
import inspect

import numpy as np
import pandas as pd
import pytest

from gmnps.scoring import attribute_calibration as calibration_module
from gmnps.scoring.attribute_calibration import (
    ATTRIBUTE_POINT_FRACTION_MODES,
    BETA_NORMALIZATION_METHOD_VERSION,
    AttributeCalibrationResult,
    BetaNormalizationState,
    attribute_response,
    calibrate_attribute_points,
    fit_beta_normalization,
    transform_beta,
    validate_calibration_result,
)
from gmnps.scoring.fcs2_attribute_mapping import (
    PRIMARY_ATTRIBUTE_MAPPINGS,
    PRIMARY_MAPPING_VERSION,
)
from gmnps.scoring.fcs2_attribute_rules import FCS2_RULES, NOT_CALCULATED


ALL_BETA_NUTRIENTS = list(dict.fromkeys(row.nutrient for row in PRIMARY_ATTRIBUTE_MAPPINGS))
ACTIVE_ATTRIBUTES = [name for name, rule in FCS2_RULES.items() if rule.active]


def _development_beta(columns=ALL_BETA_NUTRIENTS):
    values = np.repeat(np.array([[-1.0], [0.0], [1.0]]), len(columns), axis=1)
    return pd.DataFrame(values, index=["dev_1", "dev_2", "dev_3"], columns=list(columns))


def _complete_exposures(food_ids=("food_1",)):
    columns = list(
        dict.fromkeys(
            ALL_BETA_NUTRIENTS
            + [
                "Sodium (mg)",
                "Fatty acids, total monounsaturated (g)",
                "Fatty acids, total polyunsaturated (g)",
                "Folate, DFE (mcg_DFE)",
            ]
        )
    )
    exposures = pd.DataFrame(0.0, index=list(food_ids), columns=columns)
    exposures.attrs["basis"] = "per_100_kcal"
    return exposures


def _raw_for_transformed_vitamin_c(state, values):
    raw = pd.DataFrame(0.0, index=[f"person_{i}" for i in range(len(values))], columns=state.nutrient_order)
    for row, transformed_value in enumerate(values):
        raw.iloc[row, raw.columns.get_loc("Vitamin C (mg)")] = (
            state.medians["Vitamin C (mg)"]
            + state.scales["Vitamin C (mg)"]
            * calibration_module.LOCKED_BETA_TEMPERATURE
            * np.arctanh(transformed_value)
        )
    return raw


def _response_result(transformed_vitamin_c=(0.8,), food_ids=("food_1",)):
    state = fit_beta_normalization(_development_beta())
    raw = _raw_for_transformed_vitamin_c(state, transformed_vitamin_c)
    exposures = _complete_exposures(food_ids)
    exposures.loc[:, "Vitamin C (mg)"] = 22.5
    return state, attribute_response(state, raw, exposures)


def _baseline(food_ids=("food_1",), *, not_calculated=()):
    values = {
        attribute: (FCS2_RULES[attribute].low_points + FCS2_RULES[attribute].high_points) / 2.0
        for attribute in ACTIVE_ATTRIBUTES
    }
    baseline = pd.DataFrame(
        [values.copy() for _ in food_ids],
        index=pd.Index(list(food_ids), name="food_id"),
    )
    for attribute in not_calculated:
        baseline[attribute] = pd.Series(
            [NOT_CALCULATED] * len(baseline), index=baseline.index, dtype=object
        )
    return baseline


def _forge_state(state, **changes):
    forged = object.__new__(BetaNormalizationState)
    for field in fields(BetaNormalizationState):
        object.__setattr__(forged, field.name, getattr(state, field.name))
    for name, value in changes.items():
        object.__setattr__(forged, name, value)
    return forged


def _forge_calibration(result, **changes):
    forged = object.__new__(AttributeCalibrationResult)
    for field in fields(AttributeCalibrationResult):
        object.__setattr__(forged, field.name, getattr(result, field.name))
    for name, value in changes.items():
        object.__setattr__(forged, name, value)
    return forged


def test_fit_records_an_immutable_locked_development_state():
    development = pd.DataFrame(
        {"nutrient_b": [2.0, 4.0, 8.0], "nutrient_a": [10.0, 10.0, 10.0]},
        index=pd.Index(["dev_2", "dev_1", "dev_3"], name="individual_id"),
    )

    state = fit_beta_normalization(development)

    assert state.nutrient_order == ("nutrient_b", "nutrient_a")
    assert state.medians == {"nutrient_b": 4.0, "nutrient_a": 10.0}
    assert state.fit_n == 3
    assert state.fit_id_sha256 == sha256(b"dev_1\ndev_2\ndev_3").hexdigest()
    assert state.method_version == BETA_NORMALIZATION_METHOD_VERSION
    assert state.temperature == calibration_module.LOCKED_BETA_TEMPERATURE
    assert len(state.state_fingerprint) == 64
    with pytest.raises(FrozenInstanceError):
        state.fit_n = 4
    with pytest.raises(TypeError):
        state.medians["nutrient_a"] = 0.0


def test_beta_and_attribute_response_temperatures_are_distinct_runtime_constants():
    assert calibration_module.LOCKED_BETA_TEMPERATURE == 2.0
    assert hasattr(calibration_module, "LOCKED_ATTRIBUTE_RESPONSE_TEMPERATURE")
    assert calibration_module.LOCKED_ATTRIBUTE_RESPONSE_TEMPERATURE == 2.0

    validation_source = inspect.getsource(calibration_module.validate_calibration_result)
    calibration_source = inspect.getsource(calibration_module.calibrate_attribute_points)
    assert "np.tanh(response / LOCKED_ATTRIBUTE_RESPONSE_TEMPERATURE)" in validation_source
    assert (
        "np.tanh(numeric_response / LOCKED_ATTRIBUTE_RESPONSE_TEMPERATURE)"
        in calibration_source
    )
    assert "np.tanh(response / 2.0)" not in validation_source
    assert "np.tanh(numeric_response / 2.0)" not in calibration_source


def test_fit_has_no_public_temperature_override():
    assert "temperature" not in inspect.signature(fit_beta_normalization).parameters

    with pytest.raises(TypeError, match="temperature"):
        fit_beta_normalization(_development_beta(["a"]), temperature=3.0)


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([1, "1"], dtype=object),
        pd.Index(["dev_1", ""]),
        pd.Index(["dev_1", "   "]),
    ],
)
def test_fit_requires_nonempty_string_participant_ids_without_string_collisions(index):
    development = pd.DataFrame({"a": [0.0, 1.0]}, index=index)

    with pytest.raises(ValueError, match="nonempty strings"):
        fit_beta_normalization(development)


def test_fit_uses_mad_then_iqr_then_standard_deviation_then_unit_scale():
    development = pd.DataFrame(
        {
            "mad": [0.0, 1.0, 2.0, 3.0, 4.0],
            "iqr": [0.0, 0.0, 0.0, 10.0, 10.0],
            "standard_deviation": [0.0, 0.0, 0.0, 0.0, 10.0],
            "unit": [7.0, 7.0, 7.0, 7.0, 7.0],
        },
        index=[f"d{number}" for number in range(5)],
    )

    state = fit_beta_normalization(development)

    assert state.scales["mad"] == pytest.approx(1.4826)
    assert state.scales["iqr"] == pytest.approx(10.0 / 1.349)
    assert state.scales["standard_deviation"] == pytest.approx(4.0)
    assert state.scales["unit"] == 1.0
    assert state.scale_methods == {
        "mad": "mad",
        "iqr": "iqr",
        "standard_deviation": "standard_deviation",
        "unit": "unit",
    }


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_fit_rejects_nonfinite_development_values(bad_value):
    development = pd.DataFrame({"a": [0.0, bad_value]}, index=["d1", "d2"])

    with pytest.raises(ValueError, match="finite"):
        fit_beta_normalization(development)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"method_version": "forged"}, "method version"),
        ({"temperature": 3.0}, "temperature"),
        ({"median_values": (0.0,)}, "equal lengths"),
        ({"scale_values": (np.inf,) * len(ALL_BETA_NUTRIENTS)}, "scales"),
        ({"scale_method_values": ("invented",) * len(ALL_BETA_NUTRIENTS)}, "scale methods"),
        ({"state_fingerprint": "0" * 64}, "fingerprint"),
    ],
)
def test_transform_revalidates_and_rejects_forged_state(changes, message):
    state = fit_beta_normalization(_development_beta())
    forged = _forge_state(state, **changes)
    beta = pd.DataFrame(0.0, index=["held_out"], columns=state.nutrient_order)

    with pytest.raises(ValueError, match=message):
        transform_beta(forged, beta)


def test_transform_uses_only_frozen_state_and_aligns_held_out_columns():
    development = pd.DataFrame(
        {"a": [0.0, 1.0, 2.0], "b": [2.0, 4.0, 6.0]},
        index=["d1", "d2", "d3"],
    )
    state = fit_beta_normalization(development)
    held_out = pd.DataFrame({"b": [8.0], "a": [1.0]}, index=["held_out"])

    transformed = transform_beta(state, held_out)

    expected_b = np.tanh(
        ((8.0 - state.medians["b"]) / state.scales["b"])
        / calibration_module.LOCKED_BETA_TEMPERATURE
    )
    assert transformed.columns.tolist() == ["a", "b"]
    assert transformed.loc["held_out", "a"] == 0.0
    assert transformed.loc["held_out", "b"] == pytest.approx(expected_b)


@pytest.mark.parametrize(
    ("held_out", "message"),
    [
        (pd.DataFrame({"a": [0.0]}, index=["h1"]), "missing.*b"),
        (pd.DataFrame({"a": [0.0], "b": [0.0], "c": [0.0]}, index=["h1"]), "extra.*c"),
        (pd.DataFrame({"a": [np.nan], "b": [0.0]}, index=["h1"]), "finite"),
        (pd.DataFrame({"a": [0.0], "b": [np.inf]}, index=["h1"]), "finite"),
    ],
)
def test_transform_rejects_missing_extra_and_nonfinite_inputs(held_out, message):
    state = fit_beta_normalization(
        pd.DataFrame({"a": [0.0, 1.0], "b": [2.0, 3.0]}, index=["d1", "d2"])
    )

    with pytest.raises(ValueError, match=message):
        transform_beta(state, held_out)


def test_raw_beta_scored_alone_or_in_a_batch_is_identical_through_response_api():
    state = fit_beta_normalization(_development_beta())
    target = _raw_for_transformed_vitamin_c(state, [0.6])
    target.index = ["target"]
    batch = _raw_for_transformed_vitamin_c(state, [-0.9, 0.6, 0.9])
    batch.index = ["low", "target", "high"]
    exposures = _complete_exposures()
    exposures.loc["food_1", "Vitamin C (mg)"] = 22.5

    alone = attribute_response(state, target, exposures)
    together = attribute_response(state, batch, exposures)

    assert alone.responses.loc[("target", "food_1"), "vitamin_c"] == together.responses.loc[
        ("target", "food_1"), "vitamin_c"
    ]


def test_attribute_response_transforms_raw_beta_and_carries_locked_provenance():
    state = fit_beta_normalization(_development_beta())
    raw = _raw_for_transformed_vitamin_c(state, [0.8])
    exposures = _complete_exposures(("full_dose", "half_dose"))
    exposures.loc["full_dose", "Vitamin C (mg)"] = 22.5
    exposures.loc["half_dose", "Vitamin C (mg)"] = 11.25

    result = attribute_response(state, raw, exposures)

    assert result.normalization_fingerprint == state.state_fingerprint
    assert result.normalization_state == state
    assert result.mapping_version == PRIMARY_MAPPING_VERSION
    assert result.responses.columns.tolist() == ACTIVE_ATTRIBUTES
    assert "vitamin_c" in result.reviewed_targets
    assert "zinc" not in result.reviewed_targets
    assert "copper" not in result.reviewed_targets
    assert len(result.response_fingerprint) == 64
    assert result.responses.loc[("person_0", "full_dose"), "vitamin_c"] == pytest.approx(0.8)
    assert result.responses.loc[("person_0", "half_dose"), "vitamin_c"] == pytest.approx(0.4)


def test_role_only_and_excluded_raw_beta_cannot_inject_responses():
    state = fit_beta_normalization(_development_beta())
    baseline_raw = pd.DataFrame(0.0, index=["person_1"], columns=state.nutrient_order)
    injected_raw = baseline_raw.copy()
    excluded = {
        "Carbohydrate (g)",
        "Zinc (mg)",
        "Copper (mg)",
        "Vitamin A, RAE (mcg_RAE)",
        "Total Fat (g)",
        "4:0 (g)",
        "6:0 (g)",
        "14:0 (g)",
        "16:0 (g)",
        "18:0 (g)",
    }
    injected_raw.loc[:, list(excluded)] = 1000.0
    exposures = _complete_exposures()

    baseline = attribute_response(state, baseline_raw, exposures)
    injected = attribute_response(state, injected_raw, exposures)

    pd.testing.assert_frame_equal(injected.responses, baseline.responses)


def test_attribute_response_does_not_center_or_read_endpoint_columns():
    state = fit_beta_normalization(_development_beta())
    raw = _raw_for_transformed_vitamin_c(state, [0.4, 0.8])
    exposures = _complete_exposures()
    exposures.loc["food_1", "Vitamin C (mg)"] = 22.5
    exposures["response_endpoint"] = -999999.0

    first = attribute_response(state, raw, exposures)
    exposures["response_endpoint"] = 999999.0
    second = attribute_response(state, raw, exposures)

    pd.testing.assert_frame_equal(first.responses, second.responses)
    assert first.response_fingerprint == second.response_fingerprint
    assert first.responses["vitamin_c"].tolist() == pytest.approx([0.4, 0.8])
    assert first.responses["vitamin_c"].mean() == pytest.approx(0.6)


def test_calibration_zero_response_exactly_recovers_baseline_and_not_calculated():
    state = fit_beta_normalization(_development_beta())
    raw = pd.DataFrame(0.0, index=["person_1"], columns=state.nutrient_order)
    responses = attribute_response(state, raw, _complete_exposures())
    baseline = _baseline(not_calculated=("potassium_to_sodium_ratio",))

    result = calibrate_attribute_points(baseline, responses)

    pair = ("person_1", "food_1")
    for attribute in set(ACTIVE_ATTRIBUTES) - {"potassium_to_sodium_ratio"}:
        assert result.points.loc[pair, attribute] == baseline.loc["food_1", attribute]
        assert result.deltas.loc[pair, attribute] == 0.0
    assert result.points.loc[pair, "potassium_to_sodium_ratio"] is NOT_CALCULATED
    assert result.deltas.loc[pair, "potassium_to_sodium_ratio"] is NOT_CALCULATED

    diagnostic = result.diagnostics.loc[
        ("person_1", "food_1", "potassium_to_sodium_ratio")
    ]
    numeric_fields = [
        "baseline_points",
        "attribute_response",
        "raw_delta",
        "point_delta",
    ]
    assert diagnostic[numeric_fields].isna().all()
    assert bool(diagnostic["clipped"]) is False
    assert bool(diagnostic["not_calculated"]) is True
    assert diagnostic["mode"] == "primary"


@pytest.mark.parametrize(
    ("mode", "fraction_cap"),
    [("low", 0.10), ("primary", 0.20), ("high", 0.30)],
)
def test_calibration_uses_only_named_attribute_range_fraction_modes(mode, fraction_cap):
    _, responses = _response_result((0.8,))
    baseline = _baseline()

    result = calibrate_attribute_points(baseline, responses, mode=mode)

    expected_lambda = fraction_cap * 10.0
    expected_delta = expected_lambda * np.tanh(
        0.8 / calibration_module.LOCKED_ATTRIBUTE_RESPONSE_TEMPERATURE
    )
    assert ATTRIBUTE_POINT_FRACTION_MODES[mode] == fraction_cap
    assert result.deltas.loc[("person_0", "food_1"), "vitamin_c"] == pytest.approx(expected_delta)
    vitamin_c = result.diagnostics.xs("vitamin_c", level="attribute")
    assert set(vitamin_c["lambda_points"]) == {expected_lambda}
    assert set(vitamin_c["mode"]) == {mode}


@pytest.mark.parametrize("mode", ["medium", 0.15, 0.20, None])
def test_calibration_rejects_unnamed_or_unapproved_fraction_modes(mode):
    _, responses = _response_result()

    with pytest.raises((TypeError, ValueError), match="mode"):
        calibrate_attribute_points(_baseline(), responses, mode=mode)


def test_calibration_clips_at_published_bounds_and_preserves_native_direction():
    _, positive = _response_result((0.9,))
    _, negative = _response_result((-0.9,))
    baseline = _baseline()
    baseline.loc["food_1", "vitamin_c"] = 9.5
    baseline.loc["food_1", "cholesterol"] = -9.5

    positive_result = calibrate_attribute_points(baseline, positive)
    negative_result = calibrate_attribute_points(baseline, negative)

    pair = ("person_0", "food_1")
    assert positive_result.points.loc[pair, "vitamin_c"] == 10.0
    assert negative_result.points.loc[pair, "vitamin_c"] < 9.5
    # Cholesterol has zero response in this fixture and therefore remains exact.
    assert positive_result.points.loc[pair, "cholesterol"] == -9.5


def test_positive_response_makes_a_harmful_attribute_less_negative():
    state = fit_beta_normalization(_development_beta())
    raw = pd.DataFrame(0.0, index=["person_1"], columns=state.nutrient_order)
    raw.loc["person_1", "Cholesterol (mg)"] = (
        state.scales["Cholesterol (mg)"]
        * calibration_module.LOCKED_BETA_TEMPERATURE
        * np.arctanh(0.8)
    )
    exposures = _complete_exposures()
    exposures.loc["food_1", "Cholesterol (mg)"] = 75.0
    responses = attribute_response(state, raw, exposures)
    baseline = _baseline()
    baseline.loc["food_1", "cholesterol"] = -5.0

    result = calibrate_attribute_points(baseline, responses)

    assert result.points.loc[("person_1", "food_1"), "cholesterol"] > -5.0


def test_calibration_preserves_nonzero_population_mean_without_hidden_centering():
    _, responses = _response_result((0.5, 0.9))
    result = calibrate_attribute_points(_baseline(), responses)

    assert (result.deltas["vitamin_c"] > 0.0).all()
    assert result.deltas["vitamin_c"].mean() > 0.0


@pytest.mark.parametrize("attribute", ["zinc", "copper"])
def test_calibration_rejects_nonzero_response_outside_reviewed_targets(attribute):
    _, valid = _response_result()
    changed = valid.responses.copy()
    changed.loc[:, attribute] = 0.5
    injected = replace(valid, responses=changed)

    with pytest.raises(ValueError, match="outside reviewed target set"):
        calibrate_attribute_points(_baseline(), injected)


def test_calibration_rejects_manual_vitamin_a_rae_response_injection_by_fingerprint():
    _, valid = _response_result()
    assert "vitamin_a_rae" in valid.reviewed_targets  # Retinol is the reviewed component source.
    changed = valid.responses.copy()
    changed.loc[:, "vitamin_a_rae"] = 0.5
    injected = replace(valid, responses=changed)

    with pytest.raises(ValueError, match="response fingerprint"):
        calibrate_attribute_points(_baseline(), injected)


def test_calibration_rejects_role_only_or_extra_response_columns():
    _, valid = _response_result()
    changed = valid.responses.copy()
    changed["Total Fat (g)"] = 0.5
    injected = replace(valid, responses=changed)

    with pytest.raises(ValueError, match="exact active Food Compass attributes"):
        calibrate_attribute_points(_baseline(), injected)


def test_calibration_rejects_missing_response_columns():
    _, valid = _response_result()
    changed = valid.responses.drop(columns="vitamin_c")
    incomplete = replace(valid, responses=changed)

    with pytest.raises(ValueError, match="exact active Food Compass attributes"):
        calibrate_attribute_points(_baseline(), incomplete)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: replace(result, normalization_fingerprint="0" * 64),
        lambda result: replace(result, mapping_version="fiber_all_ratio"),
        lambda result: replace(result, reviewed_targets=result.reviewed_targets + ("zinc",)),
        lambda result: replace(result, response_fingerprint="0" * 64),
    ],
)
def test_calibration_rejects_forged_response_provenance(mutation):
    _, valid = _response_result()

    with pytest.raises(ValueError, match="fingerprint|mapping|reviewed target"):
        calibrate_attribute_points(_baseline(), mutation(valid))


def test_calibration_requires_exact_food_and_attribute_labels():
    _, valid = _response_result(food_ids=("food_1", "food_2"))
    baseline = _baseline(("food_1", "food_2"))

    with pytest.raises(ValueError, match="exact response attributes"):
        calibrate_attribute_points(baseline.drop(columns="vitamin_c"), valid)
    with pytest.raises(ValueError, match="exact response food IDs"):
        calibrate_attribute_points(baseline.drop(index="food_2"), valid)


def test_calibration_result_carries_frozen_end_to_end_provenance():
    _, responses = _response_result()

    result = calibrate_attribute_points(_baseline(), responses, mode="high")

    assert result.normalization_fingerprint == responses.normalization_fingerprint
    assert result.mapping_version == responses.mapping_version
    assert result.reviewed_targets == responses.reviewed_targets
    assert result.response_fingerprint == responses.response_fingerprint
    assert result.calibration_mode == "high"
    assert result.calibration_fraction == ATTRIBUTE_POINT_FRACTION_MODES["high"]
    assert len(result.calibration_fingerprint) == 64
    assert validate_calibration_result(result) is result
    with pytest.raises(FrozenInstanceError):
        result.mapping_version = "forged"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mapping_version": "fiber_all_ratio"}, "fingerprint|mapping"),
        ({"reviewed_targets": ("zinc",)}, "reviewed target"),
        ({"response_fingerprint": "0" * 64}, "response fingerprint|calibration fingerprint"),
        ({"calibration_mode": "low"}, "mode|calibration fingerprint"),
        ({"calibration_fraction": 0.10}, "fraction|calibration fingerprint"),
        ({"calibration_fingerprint": "0" * 64}, "calibration fingerprint"),
    ],
)
def test_validate_calibration_result_rejects_forged_provenance(changes, message):
    _, responses = _response_result()
    result = calibrate_attribute_points(_baseline(), responses)

    with pytest.raises(ValueError, match=message):
        validate_calibration_result(_forge_calibration(result, **changes))


def test_validate_rejects_synchronized_nonreviewed_target_forgery():
    _, responses = _response_result()
    result = calibrate_attribute_points(_baseline(), responses)
    points = result.points.copy()
    deltas = result.deltas.copy()
    diagnostics = result.diagnostics.copy()
    pair = ("person_0", "food_1")
    key = (*pair, "zinc")
    response = 1.0
    raw_delta = 2.0 * np.tanh(
        response / calibration_module.LOCKED_ATTRIBUTE_RESPONSE_TEMPERATURE
    )
    points.loc[pair, "zinc"] += raw_delta
    deltas.loc[pair, "zinc"] = raw_delta
    diagnostics.loc[key, "attribute_response"] = response
    diagnostics.loc[key, "raw_delta"] = raw_delta
    diagnostics.loc[key, "point_delta"] = raw_delta

    forged = _forge_calibration(
        result,
        points=points,
        deltas=deltas,
        diagnostics=diagnostics,
    )

    with pytest.raises(ValueError, match="outside reviewed target|calibration fingerprint"):
        validate_calibration_result(forged)


def test_calibration_fingerprint_binds_matrix_labels_and_complete_diagnostics():
    _, responses = _response_result()
    result = calibrate_attribute_points(_baseline(), responses)

    points = result.points.rename(columns={"vitamin_c": "forged_vitamin_c"})
    with pytest.raises(ValueError, match="attribute labels|calibration fingerprint"):
        validate_calibration_result(_forge_calibration(result, points=points))

    diagnostics = result.diagnostics.copy()
    diagnostics.loc[("person_0", "food_1", "vitamin_c"), "lower_bound"] = 1.0
    with pytest.raises(ValueError, match="bounds|calibration fingerprint"):
        validate_calibration_result(_forge_calibration(result, diagnostics=diagnostics))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("baseline_points", 0.0),
        ("attribute_response", 0.0),
        ("raw_delta", 0.0),
        ("point_delta", 0.0),
        ("clipped", True),
        ("not_calculated", False),
        ("mode", "low"),
        ("fraction_cap", 0.10),
    ],
)
def test_validate_rejects_tampered_not_calculated_diagnostics(column, value):
    state = fit_beta_normalization(_development_beta())
    raw = pd.DataFrame(0.0, index=["person_1"], columns=state.nutrient_order)
    responses = attribute_response(state, raw, _complete_exposures())
    result = calibrate_attribute_points(
        _baseline(not_calculated=("potassium_to_sodium_ratio",)), responses
    )
    diagnostics = result.diagnostics.copy()
    diagnostics.loc[
        ("person_1", "food_1", "potassium_to_sodium_ratio"), column
    ] = value

    with pytest.raises(ValueError, match="NOT_CALCULATED|mode|fraction|calibration fingerprint"):
        validate_calibration_result(
            _forge_calibration(result, diagnostics=diagnostics)
        )


def test_public_functions_have_no_outcome_or_final_score_inputs():
    forbidden = {"outcome", "disease", "endpoint", "final_score", "desired_drift"}

    for function in (
        fit_beta_normalization,
        transform_beta,
        attribute_response,
        calibrate_attribute_points,
    ):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)
