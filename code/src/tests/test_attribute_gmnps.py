from dataclasses import FrozenInstanceError, fields, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmnps.scoring.attribute_calibration import transform_beta
from gmnps.scoring.attribute_gmnps import (
    ATTRIBUTE_GMNPS_SCORING_VERSION,
    FCS2_FNDDS_REGISTRY_VERSION,
    NOT_CALCULATED_TOKEN,
    AttributeGMNPSConfig,
    FoodAttributeBundle,
    _effect_nutrients_by_channel,
    fit_attribute_gmnps,
    score_attribute_gmnps,
)
from gmnps.scoring.fcs2_attribute_mapping import (
    PRIMARY_ATTRIBUTE_MAPPINGS,
    PRIMARY_MAPPING_VERSION,
    SENSITIVITY_ATTRIBUTE_MAPPINGS,
    reconstruction_status_table,
)
from gmnps.scoring.fcs2_attribute_rules import FCS2_RULES, NOT_CALCULATED


ACTIVE_ATTRIBUTES = tuple(name for name, rule in FCS2_RULES.items() if rule.active)
BETA_NUTRIENTS = tuple(dict.fromkeys(row.nutrient for row in PRIMARY_ATTRIBUTE_MAPPINGS))
EXPOSURE_NUTRIENTS = tuple(
    dict.fromkeys(
        BETA_NUTRIENTS
        + (
            "Sodium (mg)",
            "Fatty acids, total monounsaturated (g)",
            "Fatty acids, total polyunsaturated (g)",
            "Folate, DFE (mcg_DFE)",
        )
    )
)


def development_beta():
    values = np.repeat(np.array([[-1.0], [0.0], [1.0]]), len(BETA_NUTRIENTS), axis=1)
    return pd.DataFrame(values, index=["dev_1", "dev_2", "dev_3"], columns=BETA_NUTRIENTS)


def baseline_points(food_ids=("food_1", "food_2")):
    row = {
        attribute: (rule.low_points + rule.high_points) / 2.0
        for attribute, rule in FCS2_RULES.items()
        if rule.active
    }
    return pd.DataFrame(
        [row.copy() for _ in food_ids],
        index=pd.Index(food_ids, name="food_id"),
        columns=ACTIVE_ATTRIBUTES,
    )


def food_exposures(food_ids=("food_1", "food_2")):
    frame = pd.DataFrame(0.0, index=pd.Index(food_ids, name="food_id"), columns=EXPOSURE_NUTRIENTS)
    frame.loc[:, "Carbohydrate (g)"] = 10.0
    frame.loc[:, "Sodium (mg)"] = 100.0
    frame.loc[:, "Potassium (mg)"] = 100.0
    frame.loc[:, "Total Fat (g)"] = 10.0
    frame.loc[:, "Fatty acids, total saturated (g)"] = 1.0
    frame.loc[:, "Fatty acids, total monounsaturated (g)"] = 4.0
    frame.loc[:, "Fatty acids, total polyunsaturated (g)"] = 2.0
    frame.attrs["basis"] = "per_100_kcal"
    return frame


def make_bundle(
    *,
    releases=("FNDDS 2021-2023",),
    production_label="non-production",
    source_hashes=None,
    nutrient_units=None,
    registry_version=FCS2_FNDDS_REGISTRY_VERSION,
):
    foods = ("food_1", "food_2")
    official = pd.Series([40.0, 70.0], index=pd.Index(foods, name="food_id"), name="FCS2")
    metadata = pd.DataFrame(
        {
            "food_name": ["Food one", "Food two"],
            "food_group": ["Group A", "Group B"],
            "is_dairy": [False, False],
            "FCS2": official,
        },
        index=official.index,
    )
    exposures = food_exposures(foods)
    source_hashes = source_hashes or {
        "official_fcs": "1" * 64,
        "food_metadata": "2" * 64,
        "baseline_attribute_points": "3" * 64,
        "food_exposures": "4" * 64,
        "effective_attribute_weights": "7" * 64,
        "input_manifest": "5" * 64,
    }
    effective_weights = pd.DataFrame(
        {
            attribute: [float(FCS2_RULES[attribute].weight)] * len(foods)
            for attribute in ACTIVE_ATTRIBUTES
        },
        index=official.index,
    )
    registry_bytes = (
        Path(__file__).resolve().parents[1]
        / "configs/fcs2_fndds_release_registry.json"
    ).read_bytes()
    registry = json.loads(registry_bytes)
    return FoodAttributeBundle(
        official_fcs=official,
        baseline_points=baseline_points(foods),
        food_exposures=exposures,
        effective_attribute_weights=effective_weights,
        food_metadata=metadata,
        fndds_releases=releases,
        source_hashes=source_hashes,
        nutrient_units=nutrient_units
        or {column: "source_unit_per_100_kcal" for column in exposures.columns},
        exposure_basis="per_100_kcal",
        reconstruction_status=reconstruction_status_table(),
        production_label=production_label,
        registry_version=registry_version,
        release_registry_snapshot_sha256=sha256(registry_bytes).hexdigest(),
        release_registry_canonical_release_set=tuple(
            registry["canonical_release_set"]
        ),
        food_source_linkage_sha256="6" * 64,
    )


def score_beta(index=("person_1", "person_2"), value=0.0):
    return pd.DataFrame(value, index=index, columns=BETA_NUTRIENTS, dtype=float)


def smoke_config(**changes):
    return AttributeGMNPSConfig(development_smoke_test=True, **changes)


def fitted_smoke_model(config=None):
    return fit_attribute_gmnps(development_beta(), config or smoke_config())


def test_config_is_frozen_and_exposes_only_named_locked_modes():
    config = AttributeGMNPSConfig()

    assert config.method == "attribute_recomposition"
    assert config.mapping_version == PRIMARY_MAPPING_VERSION
    assert config.attribute_point_mode == "primary"
    assert config.final_cap_mode == "primary"
    assert config.scoring_version == ATTRIBUTE_GMNPS_SCORING_VERSION
    assert {field.name for field in fields(config)}.isdisjoint(
        {"delta_cap", "beta_temperature", "attribute_point_fraction", "calibration_strength"}
    )
    with pytest.raises(FrozenInstanceError):
        config.final_cap_mode = "low"


def test_named_sensitivity_modes_are_labelled_sensitivity_automatically():
    config = AttributeGMNPSConfig(attribute_point_mode="low", method_role=None)

    assert config.method_role == "sensitivity"


def test_primary_api_is_exported_without_removing_legacy_exports():
    import gmnps.scoring as scoring

    assert scoring.AttributeGMNPSConfig is AttributeGMNPSConfig
    assert scoring.FoodAttributeBundle is FoodAttributeBundle
    assert scoring.fit_attribute_gmnps is fit_attribute_gmnps
    assert scoring.score_attribute_gmnps is score_attribute_gmnps
    assert scoring.AnchoredScoringConfig.__name__ == "AnchoredScoringConfig"


def test_fit_binds_config_and_development_only_normalization_fingerprint():
    development = development_beta()
    config = smoke_config()
    model = fit_attribute_gmnps(development, config)

    assert model.config == config
    assert model.normalization_state.fit_n == len(development)
    assert model.normalization_fingerprint == model.normalization_state.state_fingerprint
    assert len(model.fingerprint) == 64
    held_out = score_beta(("held_out",), value=100.0)
    transformed = transform_beta(model.normalization_state, held_out)
    assert transformed.index.tolist() == ["held_out"]


def test_zero_identity_bounds_schemas_and_no_centering():
    model = fitted_smoke_model()
    bundle = make_bundle()
    zero = score_beta()
    result = score_attribute_gmnps(model, zero, bundle)

    required = {
        "individual_id",
        "food_id",
        "food_name",
        "food_group",
        "FCS2",
        "GMNPS_delta",
        "GMNPS_score",
        "MAC_delta",
        "LIPID_delta",
        "channel_interaction_delta",
        "top_positive_drivers",
        "top_negative_drivers",
        "mask_version",
        "scoring_version",
        "method_role",
    }
    assert required <= set(result.individual_food)
    assert set(result.individual_food["driver_basis"]) == {"attribute_point_delta"}
    assert np.allclose(result.individual_food["GMNPS_delta"], 0.0, atol=1e-8)
    assert np.allclose(result.individual_food["GMNPS_score"], result.individual_food["FCS2"], atol=1e-8)
    assert result.individual_food["GMNPS_score"].between(1.0, 100.0).all()
    assert {
        "GMNPS_mean",
        "GMNPS_sd",
        "GMNPS_p05",
        "GMNPS_p95",
        "MAC_variance",
        "LIPID_variance",
        "channel_interaction_variance",
        "dominant_channel",
        "source_cohort_n",
    } <= set(result.food_summary)
    assert {"attribute", "attribute_point_delta", "channel"} <= set(result.attribute_attribution)
    assert {"domain", "domain_delta"} <= set(result.domain_attribution)

    positive = score_beta(("positive_a", "positive_b"))
    positive.loc[:, "Vitamin C (mg)"] = 10.0
    exposed = replace(bundle, food_exposures=bundle.food_exposures.copy(), fingerprint="")
    exposed.food_exposures.loc[:, "Vitamin C (mg)"] = 22.5
    object.__setattr__(exposed, "fingerprint", exposed.compute_fingerprint())
    shifted = score_attribute_gmnps(model, positive, exposed).individual_food
    assert not np.isclose(shifted.groupby("food_id")["GMNPS_delta"].mean(), 0.0).all()


def test_channel_values_are_exact_locked_counterfactuals_and_preserve_identity():
    model = fitted_smoke_model()
    bundle = make_bundle()
    exposures = bundle.food_exposures.copy()
    exposures.loc[:, "Vitamin C (mg)"] = 22.5
    exposures.loc[:, "Cholesterol (mg)"] = 75.0
    exposures.attrs["basis"] = "per_100_kcal"
    bundle = replace(bundle, food_exposures=exposures, fingerprint="")
    beta = score_beta(("person_1",))
    beta.loc[:, "Vitamin C (mg)"] = 4.0
    beta.loc[:, "Cholesterol (mg)"] = 4.0

    result = score_attribute_gmnps(model, beta, bundle)
    rows = result.individual_food
    np.testing.assert_allclose(
        rows["GMNPS_delta"],
        rows["MAC_delta"] + rows["LIPID_delta"] + rows["channel_interaction_delta"],
        atol=1e-12,
    )
    provenance = result.run_manifest["counterfactual_provenance"]
    assert provenance["MAC_delta"]["frozen_channel"] == "LIPID"
    assert provenance["LIPID_delta"]["frozen_channel"] == "MAC"
    assert provenance["MAC_delta"]["raw_beta_replacement"] == "development_median"
    assert len(provenance["MAC_delta"]["final_fingerprint"]) == 64


def test_drivers_are_signed_attribute_point_deltas():
    model = fitted_smoke_model()
    bundle = make_bundle()
    exposures = bundle.food_exposures.copy()
    exposures.loc[:, "Vitamin C (mg)"] = 22.5
    exposures.attrs["basis"] = "per_100_kcal"
    bundle = replace(bundle, food_exposures=exposures, fingerprint="")
    beta = score_beta(("person_1",))
    beta.loc[:, "Vitamin C (mg)"] = 4.0

    result = score_attribute_gmnps(model, beta, bundle)
    driver = result.individual_food.iloc[0]["top_positive_drivers"]
    positive_attributes = set(
        result.attribute_attribution.loc[
            result.attribute_attribution["attribute_point_delta"] > 0, "attribute"
        ]
    )
    assert driver
    assert driver.split("=", 1)[0] in positive_attributes


def test_bundle_tampering_and_nonfinite_or_missing_labels_fail_closed():
    model = fitted_smoke_model()
    bundle = make_bundle()
    bundle.food_exposures.iloc[0, 0] = 999.0
    with pytest.raises(ValueError, match="fingerprint"):
        score_attribute_gmnps(model, score_beta(), bundle)

    bad = food_exposures()
    bad.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        replace(make_bundle(), food_exposures=bad, fingerprint="")


def test_production_rejects_2021_2023_and_smoke_mode_is_explicitly_nonproduction():
    with pytest.raises(ValueError, match="FNDDS 2001-2018"):
        make_bundle(releases=("FNDDS 2021-2023",), production_label="production")

    smoke_bundle = make_bundle(
        releases=("FNDDS 2021-2023",), production_label="non-production"
    )
    with pytest.raises(ValueError, match="expected_release_registry_sha256"):
        fit_attribute_gmnps(development_beta())

    smoke_model = fit_attribute_gmnps(
        development_beta(), smoke_config()
    )
    assert set(score_attribute_gmnps(smoke_model, score_beta(), smoke_bundle).individual_food["method_role"]) == {"primary"}


def test_legacy_is_explicit_sensitivity_only_and_rejected_in_production():
    with pytest.raises(ValueError, match="production"):
        AttributeGMNPSConfig(
            method="legacy_final_score_offset",
            method_role="sensitivity",
        )
    with pytest.raises(ValueError, match="sensitivity"):
        AttributeGMNPSConfig(
            method="legacy_final_score_offset",
            method_role="primary",
            development_smoke_test=True,
        )

    config = AttributeGMNPSConfig(
        method="legacy_final_score_offset",
        method_role="sensitivity",
        development_smoke_test=True,
    )
    model = fit_attribute_gmnps(development_beta(), config)
    bundle = make_bundle()
    result = score_attribute_gmnps(model, score_beta(), bundle)
    assert set(result.individual_food["method_role"]) == {"sensitivity"}
    assert result.run_manifest["method"] == "legacy_final_score_offset"


def test_trusted_registry_is_committed_empty_and_all_production_is_rejected():
    registry_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "configs/fcs2_fndds_release_registry.json"
    )
    registry = __import__("json").loads(registry_path.read_text(encoding="utf-8"))
    assert registry["registry_version"] == FCS2_FNDDS_REGISTRY_VERSION
    assert registry["approved_artifacts"] == []
    canonical = tuple(registry["canonical_release_set"])

    with pytest.raises(ValueError, match="no approved production bundle"):
        make_bundle(releases=canonical, production_label="production")
    with pytest.raises(ValueError, match="FNDDS 2001-2018"):
        make_bundle(releases=("FNDDS 2001-2099",), production_label="production")
    with pytest.raises(ValueError, match="no approved production bundle"):
        make_bundle(
            releases=canonical,
            production_label="production",
            source_hashes={
                "official_fcs": "a" * 64,
                "food_metadata": "b" * 64,
                "baseline_attribute_points": "c" * 64,
                "food_exposures": "d" * 64,
                "effective_attribute_weights": "f" * 64,
                "input_manifest": "e" * 64,
            },
        )
    wrong_units = {
        column: "wrong_unit" for column in food_exposures().columns
    }
    with pytest.raises(ValueError, match="no approved production bundle"):
        make_bundle(
            releases=canonical,
            production_label="production",
            nutrient_units=wrong_units,
        )


@pytest.mark.parametrize(
    "mapping_version",
    [PRIMARY_MAPPING_VERSION, *SENSITIVITY_ATTRIBUTE_MAPPINGS.keys()],
)
def test_every_named_mapping_derives_exact_mac_and_lipid_effect_nutrients(mapping_version):
    channels = _effect_nutrients_by_channel(mapping_version)

    assert set(channels) == {"MAC", "LIPID"}
    assert channels["MAC"]
    assert channels["LIPID"]
    assert set(channels["MAC"]).isdisjoint(channels["LIPID"])
    if mapping_version == "carbohydrate_proxy":
        assert "Carbohydrate (g)" in channels["MAC"]


@pytest.mark.parametrize(
    ("mapping_version", "nutrient", "exposure_column", "expected_channel"),
    [
        (PRIMARY_MAPPING_VERSION, "Vitamin C (mg)", "Vitamin C (mg)", "MAC"),
        ("fiber_all_ratio", "Vitamin C (mg)", "Vitamin C (mg)", "MAC"),
        ("fiber_all_absolute", "Cholesterol (mg)", "Cholesterol (mg)", "LIPID"),
        ("potassium_all_ratio", "Vitamin C (mg)", "Vitamin C (mg)", "MAC"),
        ("potassium_all_absolute", "Cholesterol (mg)", "Cholesterol (mg)", "LIPID"),
        ("carbohydrate_proxy", "Carbohydrate (g)", "Carbohydrate (g)", "MAC"),
    ],
)
def test_single_channel_effects_freeze_the_reviewed_opposite_channel(
    mapping_version, nutrient, exposure_column, expected_channel
):
    config = smoke_config(
        mapping_version=mapping_version,
        method_role=None,
    )
    model = fitted_smoke_model(config)
    bundle = make_bundle()
    exposures = bundle.food_exposures.copy()
    exposures.loc[:, exposure_column] = 75.0 if "Cholesterol" in exposure_column else 22.5
    if nutrient == "Carbohydrate (g)":
        exposures.loc[:, "Fiber, total dietary (g)"] = 2.0
        exposures.loc[:, exposure_column] = 10.0
    exposures.attrs["basis"] = "per_100_kcal"
    bundle = replace(bundle, food_exposures=exposures, fingerprint="")
    beta = score_beta(("person_1",))
    beta.loc[:, nutrient] = 4.0

    rows = score_attribute_gmnps(model, beta, bundle).individual_food
    opposite = "LIPID_delta" if expected_channel == "MAC" else "MAC_delta"
    np.testing.assert_allclose(rows[opposite], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        rows["GMNPS_delta"],
        rows["MAC_delta"] + rows["LIPID_delta"] + rows["channel_interaction_delta"],
        atol=1e-12,
    )


def test_channel_identity_survives_final_cap_and_native_truncation_boundaries():
    config = smoke_config(final_cap_mode="low", method_role=None)
    model = fitted_smoke_model(config)
    bundle = make_bundle()
    official = pd.Series(
        [1.0, 100.0], index=bundle.official_fcs.index, name="FCS2"
    )
    metadata = bundle.food_metadata.copy()
    metadata["FCS2"] = official
    beta = score_beta(("very_low", "very_high"))
    beta.loc["very_low"] = -1e6
    beta.loc["very_high"] = 1e6
    exposures = bundle.food_exposures.copy()
    exposures.loc[:, :] = 1e6
    exposures.attrs["basis"] = "per_100_kcal"
    bundle = replace(
        bundle,
        official_fcs=official,
        food_metadata=metadata,
        food_exposures=exposures,
        fingerprint="",
    )

    rows = score_attribute_gmnps(model, beta, bundle).individual_food
    assert rows["GMNPS_score"].between(1.0, 100.0).all()
    assert set(rows["GMNPS_delta"]) == {-8.0, 0.0, 8.0}
    assert rows.query("individual_id == 'very_low' and food_id == 'food_1'")[
        "GMNPS_score"
    ].item() == 1.0
    assert rows.query("individual_id == 'very_high' and food_id == 'food_2'")[
        "GMNPS_score"
    ].item() == 100.0
    np.testing.assert_allclose(
        rows["GMNPS_delta"],
        rows["MAC_delta"] + rows["LIPID_delta"] + rows["channel_interaction_delta"],
        atol=1e-12,
    )


def test_not_calculated_token_is_stable_and_only_ratio_sentinel_is_accepted():
    assert NOT_CALCULATED_TOKEN == "__GMNPS_NOT_CALCULATED_V1__"
    points = baseline_points()
    points["fiber_to_carbohydrate_ratio"] = points[
        "fiber_to_carbohydrate_ratio"
    ].astype(object)
    points.loc["food_1", "fiber_to_carbohydrate_ratio"] = NOT_CALCULATED
    bundle = make_bundle()
    exposures = bundle.food_exposures.copy()
    exposures.loc["food_1", "Carbohydrate (g)"] = 0.0
    exposures.attrs["basis"] = "per_100_kcal"
    bundle = replace(
        bundle,
        baseline_points=points,
        food_exposures=exposures,
        fingerprint="",
    )
    result = score_attribute_gmnps(fitted_smoke_model(), score_beta(), bundle)
    row = result.attribute_attribution.query(
        "food_id == 'food_1' and attribute == 'fiber_to_carbohydrate_ratio'"
    ).iloc[0]
    assert row["baseline_points"] == NOT_CALCULATED_TOKEN
    assert row["personalized_points"] == NOT_CALCULATED_TOKEN
    assert row["attribute_point_delta"] == NOT_CALCULATED_TOKEN
    assert bool(row["not_calculated"])


def test_legacy_driver_schema_is_semantically_distinct_from_primary():
    config = AttributeGMNPSConfig(
        method="legacy_final_score_offset",
        method_role="sensitivity",
        development_smoke_test=True,
    )
    result = score_attribute_gmnps(
        fit_attribute_gmnps(development_beta(), config), score_beta(), make_bundle()
    )
    columns = set(result.individual_food)
    assert "top_positive_drivers" not in columns
    assert "top_negative_drivers" not in columns
    assert {
        "legacy_top_positive_nutrient_drivers",
        "legacy_top_negative_nutrient_drivers",
        "driver_basis",
    } <= columns
    assert set(result.individual_food["driver_basis"]) == {"raw_nutrient_contribution"}
