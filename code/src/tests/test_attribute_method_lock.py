import csv
from dataclasses import asdict
import json
from pathlib import Path

from gmnps.scoring.fcs2_attribute_mapping import (
    PRIMARY_ATTRIBUTE_MAPPINGS,
    SENSITIVITY_ATTRIBUTE_MAPPINGS,
)
from gmnps.scoring.fcs2_attribute_rules import FCS2_RULES


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / "docs/methods/method_lock_manifest.schema.json"
MAPPING_PATH = ROOT / "docs/methods/attribute_mapping_table.csv"
SPEC_PATH = ROOT / "docs/methods/attribute_level_gmnps_spec.md"

CANONICAL_FNDDS_RELEASES = [
    "FNDDS 2001-2002",
    "FNDDS 2003-2004",
    "FNDDS 2005-2006",
    "FNDDS 2007-2008",
    "FNDDS 2009-2010",
    "FNDDS 2011-2012",
    "FNDDS 2013-2014",
    "FNDDS 2015-2016",
    "FNDDS 2017-2018",
]

LOCKED_FIXED_PARAMETERS = {
    "baseline": "FoodCompass2.0",
    "primary_method": "attribute_recomposition",
    "beta_normalization_method": "median_mad_iqr_sd_v1",
    "beta_mad_normal_consistency": 1.4826,
    "beta_iqr_normal_consistency": 1.349,
    "beta_temperature": 2.0,
    "attribute_response_temperature": 2.0,
    "attribute_point_mode": "primary",
    "attribute_point_fraction_cap": 0.20,
    "attribute_point_fraction_sensitivity": [0.10, 0.30],
    "final_cap_mode": "primary",
    "final_delta_cap": 12.0,
    "final_delta_cap_sensitivity": [8.0, 15.0],
    "carbohydrate_policy": "sensitivity_proxy_only",
    "clinical_experiment": False,
    "recomputed_domains": [
        "nutrient_ratios",
        "vitamins",
        "minerals",
        "specific_lipids",
        "fiber_and_protein",
        "phytochemicals",
    ],
    "recomposition_method": "native_domain_fixed_residual_v1",
    "mask_version": "expert_revised_v4_dual_channel",
    "nitrite_primary_rule": "footnote_50",
    "nitrite_sensitivity_rule": "table_25",
    "fcs_unscaled_min": -12.1,
    "fcs_unscaled_max": 35.0,
    "fcs_min": 1.0,
    "fcs_max": 100.0,
    "top_k_vitamins": 5,
    "top_k_minerals": 5,
    "top_k_specific_lipids": 3,
    "specific_lipids_domain_weight": 0.5,
    "phytochemicals_domain_weight": 0.5,
}


def test_lock_manifest_schema_requires_complete_frozen_provenance():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "manifest_schema_version",
        "method_version",
        "source_hashes",
        "fndds_releases",
        "fndds_registry_version",
        "beta_fit_cohort",
        "calibration_fit_ids_sha256",
        "mapping_version",
        "fixed_parameters",
        "validation_embargo",
    }

    properties = schema["properties"]
    assert properties["manifest_schema_version"]["const"] == "attribute-gmnps-method-lock-v1"
    assert properties["method_version"]["const"] == "attribute-gmnps-v1"
    assert properties["mapping_version"]["const"] == "expert_reviewed_attribute_mapping_v1"
    assert properties["fndds_registry_version"]["const"] == "fcs2-fndds-release-registry-v1"
    assert properties["fndds_releases"]["const"] == CANONICAL_FNDDS_RELEASES
    assert properties["validation_embargo"]["const"] is True
    assert properties["fixed_parameters"]["const"] == LOCKED_FIXED_PARAMETERS

    hashes = properties["source_hashes"]
    assert hashes["additionalProperties"] is False
    assert set(hashes["required"]) == {
        "official_fcs",
        "food_metadata",
        "baseline_attribute_points",
        "food_exposures",
        "input_manifest",
        "food_source_linkage",
        "fcs2_fndds_release_registry",
    }
    for field in hashes["required"]:
        assert hashes["properties"][field]["pattern"] == "^[0-9a-f]{64}$"

    cohort = properties["beta_fit_cohort"]
    assert cohort["additionalProperties"] is False
    assert set(cohort["required"]) == {"cohort_id", "fit_n"}
    assert properties["calibration_fit_ids_sha256"]["pattern"] == "^[0-9a-f]{64}$"


def test_schema_has_no_outcome_dependent_acceptance_criteria():
    schema_text = SCHEMA_PATH.read_text(encoding="utf-8").lower()
    forbidden = {
        "target_rank_shift",
        "target_mean_shift",
        "response_advantage",
        "minimum_auc",
        "minimum_spearman",
        "acceptance_threshold",
    }
    assert not any(term in schema_text for term in forbidden)


def _mapping_payload(rows, attribute):
    return [asdict(row) for row in rows if row.attribute == attribute]


def _optional_float(value):
    return None if value == "" else float(value)


def test_mapping_table_is_a_complete_runtime_registry_export():
    with MAPPING_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "attribute",
        "conceptual_name",
        "domain",
        "kind",
        "low_target",
        "high_target",
        "low_points",
        "high_points",
        "unit",
        "weight",
        "active",
        "unavailability_reason",
        "ambiguity",
        "primary_rule_version",
        "scoring_versions_json",
        "primary_mappings_json",
        "sensitivity_mappings_json",
    ]
    assert len(rows) == len(FCS2_RULES) == 56
    assert [row["attribute"] for row in rows] == list(FCS2_RULES)

    for row in rows:
        rule = FCS2_RULES[row["attribute"]]
        assert row["conceptual_name"] == rule.conceptual_name
        assert row["domain"] == rule.domain
        assert row["kind"] == rule.kind
        assert _optional_float(row["low_target"]) == rule.low_target
        assert _optional_float(row["high_target"]) == rule.high_target
        assert float(row["low_points"]) == rule.low_points
        assert float(row["high_points"]) == rule.high_points
        assert row["unit"] == rule.unit
        assert float(row["weight"]) == rule.weight
        assert row["active"] == ("true" if rule.active else "false")
        assert row["unavailability_reason"] == (rule.unavailability_reason or "")
        assert row["ambiguity"] == (rule.ambiguity or "")
        assert row["primary_rule_version"] == (rule.primary_version or "")
        assert json.loads(row["scoring_versions_json"]) == [
            {"version": version, "low_target": target}
            for version, target in rule.scoring_versions
        ]
        assert json.loads(row["primary_mappings_json"]) == _mapping_payload(
            PRIMARY_ATTRIBUTE_MAPPINGS, rule.name
        )
        expected_sensitivity = {
            version: _mapping_payload(mappings, rule.name)
            for version, mappings in SENSITIVITY_ATTRIBUTE_MAPPINGS.items()
            if _mapping_payload(mappings, rule.name)
        }
        assert json.loads(row["sensitivity_mappings_json"]) == expected_sensitivity

    inactive = {row["attribute"] for row in rows if row["active"] == "false"}
    assert inactive == {"iodine", "trans_fat_percent_calories"}


def test_method_spec_covers_locked_science_and_migration_boundary():
    text = SPEC_PATH.read_text(encoding="utf-8")
    required_phrases = {
        "54 conceptual attributes",
        "56 operational rows",
        "U0_j = fcs_to_unscaled(FCS2_j)",
        "Q_j = U0_j - L0_j",
        "GMNPS_ij = FCS2_j",
        "development-only fit boundary",
        "validation embargo",
        "iodine",
        "trans_fat_percent_calories",
        "fiber_all_ratio",
        "fiber_all_absolute",
        "potassium_all_ratio",
        "potassium_all_absolute",
        "carbohydrate_proxy",
        "footnote_50",
        "table_25",
        "legacy_final_score_offset",
        "sensitivity comparator only",
        "FNDDS 2021-2023",
        "non-production",
    }
    assert all(phrase in text for phrase in required_phrases)

    for rule_name in FCS2_RULES:
        assert f"`{rule_name}`" in text or rule_name in MAPPING_PATH.read_text(encoding="utf-8")

    approved_sources = {
        "code/src/gmnps/scoring/fcs2_attribute_rules.py": "60dcc7deb872d404700303aa01deec0dea344d6069e0fed14ae2bc4ceb3bdae0",
        "code/src/gmnps/scoring/fcs2_attribute_mapping.py": "a31c412f4ccdff196801b8ca72575864d17d18ef5d92674d9f8ffca9c3391a61",
        "code/src/gmnps/scoring/attribute_calibration.py": "daf129b39f900e7693da09d16ae422fb82e9e4d27b5d065059a5e20af00b27a1",
        "code/src/gmnps/scoring/attribute_recomposition.py": "f0db2e8febdd48ef2421a1286f84b2758d87e6d75250dbd6383ef8e3701d6ea1",
        "code/src/gmnps/scoring/attribute_gmnps.py": "47797b1a5304757111a6a1a755194d05dc0bf57d9ca21e7b3c2b890ca546a1f5",
        "code/src/configs/attribute_gmnps.yaml": "f450453811653e074ee4143cf6d9a21b9e68a5512924490fe9c64bc5f402e390",
        "code/src/configs/fcs2_fndds_release_registry.json": "621834bb75d91574bc8b408c21cd05ffb7adf15bd64241633c1cf3fff6481d13",
    }
    for source, digest in approved_sources.items():
        assert source in text
        assert digest in text

    forbidden = {
        "target rank shift",
        "target mean shift",
        "response advantage",
        "minimum auc",
        "minimum spearman",
        "acceptance threshold",
    }
    assert not any(term in text.lower() for term in forbidden)
