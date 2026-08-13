from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pandas as pd
import pytest

from gmnps.scoring.attribute_calibration import transform_beta
from gmnps.scoring.attribute_gmnps import (
    ATTRIBUTE_GMNPS_SCORING_VERSION,
    AttributeGMNPSConfig,
    FoodAttributeBundle,
    fit_attribute_gmnps,
    score_attribute_gmnps,
)
from gmnps.scoring.fcs2_attribute_mapping import (
    PRIMARY_ATTRIBUTE_MAPPINGS,
    PRIMARY_MAPPING_VERSION,
    reconstruction_status_table,
)
from gmnps.scoring.fcs2_attribute_rules import FCS2_RULES


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
    frame.loc[:, "Total Fat (g)"] = 10.0
    frame.loc[:, "Fatty acids, total saturated (g)"] = 1.0
    frame.loc[:, "Fatty acids, total monounsaturated (g)"] = 4.0
    frame.loc[:, "Fatty acids, total polyunsaturated (g)"] = 2.0
    frame.attrs["basis"] = "per_100_kcal"
    return frame


def make_bundle(*, releases=("FNDDS 2001-2018",), production_label="production"):
    foods = ("food_1", "food_2")
    official = pd.Series([40.0, 70.0], index=pd.Index(foods, name="food_id"), name="FCS2")
    metadata = pd.DataFrame(
        {
            "food_name": ["Food one", "Food two"],
            "food_group": ["Group A", "Group B"],
            "FCS2": official,
        },
        index=official.index,
    )
    exposures = food_exposures(foods)
    return FoodAttributeBundle(
        official_fcs=official,
        baseline_points=baseline_points(foods),
        food_exposures=exposures,
        food_metadata=metadata,
        fndds_releases=releases,
        source_hashes={
            "official_fcs": "1" * 64,
            "food_metadata": "2" * 64,
            "baseline_attribute_points": "3" * 64,
            "food_exposures": "4" * 64,
            "input_manifest": "5" * 64,
        },
        nutrient_units={column: "source_unit_per_100_kcal" for column in exposures.columns},
        exposure_basis="per_100_kcal",
        reconstruction_status=reconstruction_status_table(),
        production_label=production_label,
    )


def score_beta(index=("person_1", "person_2"), value=0.0):
    return pd.DataFrame(value, index=index, columns=BETA_NUTRIENTS, dtype=float)


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
    model = fit_attribute_gmnps(development)

    assert model.config == AttributeGMNPSConfig()
    assert model.normalization_state.fit_n == len(development)
    assert model.normalization_fingerprint == model.normalization_state.state_fingerprint
    assert len(model.fingerprint) == 64
    held_out = score_beta(("held_out",), value=100.0)
    transformed = transform_beta(model.normalization_state, held_out)
    assert transformed.index.tolist() == ["held_out"]


def test_zero_identity_bounds_schemas_and_no_centering():
    model = fit_attribute_gmnps(development_beta())
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
    model = fit_attribute_gmnps(development_beta())
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
    model = fit_attribute_gmnps(development_beta())
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
    model = fit_attribute_gmnps(development_beta())
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
    production_model = fit_attribute_gmnps(development_beta())
    with pytest.raises(ValueError, match="development_smoke_test"):
        score_attribute_gmnps(production_model, score_beta(), smoke_bundle)

    smoke_model = fit_attribute_gmnps(
        development_beta(), AttributeGMNPSConfig(development_smoke_test=True)
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
    bundle = make_bundle(production_label="production")
    result = score_attribute_gmnps(model, score_beta(), bundle)
    assert set(result.individual_food["method_role"]) == {"sensitivity"}
    assert result.run_manifest["method"] == "legacy_final_score_offset"
