import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re

from gmnps.scoring import attribute_calibration, attribute_recomposition
from gmnps.scoring.attribute_gmnps import (
    ATTRIBUTE_GMNPS_SCORING_VERSION,
    FCS2_FNDDS_REGISTRY_DIGEST_ALGORITHM,
    FCS2_FNDDS_REGISTRY_SCHEMA_VERSION,
    FCS2_FNDDS_REGISTRY_VERSION,
    _mapping_rows,
)
from gmnps.scoring.fcs2_attribute_mapping import (
    PRIMARY_ATTRIBUTE_MAPPINGS,
    PRIMARY_MAPPING_VERSION,
    SENSITIVITY_ATTRIBUTE_MAPPINGS,
)
from gmnps.scoring import fcs2_attribute_rules
from gmnps.scoring.fcs2_attribute_rules import (
    FCS2_RULES,
    NITRITE_RULE_PRIMARY_VERSION,
    NITRITE_RULE_TABLE_SENSITIVITY_VERSION,
)
from gmnps.scoring.masks import PRIMARY_MASK_VERSION


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / "docs/methods/method_lock_manifest.schema.json"
MAPPING_PATH = ROOT / "docs/methods/attribute_mapping_table.csv"
SPEC_PATH = ROOT / "docs/methods/attribute_level_gmnps_spec.md"
PHASE_2_PLAN_PATH = (
    ROOT / "docs/superpowers/plans/2026-08-13-predict-zoe-data-validation.md"
)
CONFIG_PATH = ROOT / "code/src/configs/attribute_gmnps.yaml"
RELEASE_REGISTRY_PATH = ROOT / "code/src/configs/fcs2_fndds_release_registry.json"

APPROVED_IMPLEMENTATION_SOURCES = (
    "code/src/gmnps/scoring/fcs2_attribute_rules.py",
    "code/src/gmnps/scoring/fcs2_attribute_mapping.py",
    "code/src/gmnps/scoring/attribute_calibration.py",
    "code/src/gmnps/scoring/attribute_recomposition.py",
    "code/src/gmnps/scoring/attribute_gmnps.py",
    "code/src/configs/attribute_gmnps.yaml",
    "code/src/scripts/run_attribute_gmnps.py",
)


def _runtime_fixed_parameters():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    point_modes = attribute_calibration.ATTRIBUTE_POINT_FRACTION_MODES
    cap_modes = attribute_recomposition.FINAL_DEVIATION_CAP_MODES
    top_k = fcs2_attribute_rules._TOP_K
    domain_weights = fcs2_attribute_rules._DOMAIN_WEIGHTS
    return {
        "baseline": config["baseline"],
        "primary_method": config["primary_method"],
        "beta_normalization_method": attribute_calibration.BETA_NORMALIZATION_METHOD_VERSION,
        "beta_mad_normal_consistency": attribute_calibration._MAD_NORMAL_CONSISTENCY,
        "beta_iqr_normal_consistency": attribute_calibration._IQR_NORMAL_CONSISTENCY,
        "beta_temperature": attribute_calibration.LOCKED_BETA_TEMPERATURE,
        "score_centering": config["score_centering"],
        "beta_centering": config["beta_centering"],
        "attribute_response_temperature": getattr(
            attribute_calibration, "LOCKED_ATTRIBUTE_RESPONSE_TEMPERATURE", None
        ),
        "attribute_point_mode": config["attribute_point_mode"],
        "attribute_point_fraction_cap": point_modes["primary"],
        "attribute_point_fraction_sensitivity": [point_modes["low"], point_modes["high"]],
        "final_cap_mode": config["final_cap_mode"],
        "final_delta_cap": cap_modes["primary"],
        "final_delta_cap_sensitivity": [cap_modes["low"], cap_modes["high"]],
        "carbohydrate_policy": config["carbohydrate_policy"],
        "clinical_experiment": config["clinical_experiment"],
        "recomputed_domains": config["recomputed_domains"],
        "recomposition_method": attribute_recomposition.RECOMPOSITION_METHOD_VERSION,
        "mask_version": PRIMARY_MASK_VERSION,
        "nitrite_primary_rule": NITRITE_RULE_PRIMARY_VERSION,
        "nitrite_sensitivity_rule": NITRITE_RULE_TABLE_SENSITIVITY_VERSION,
        "fcs_unscaled_min": fcs2_attribute_rules.UNSCALED_MIN,
        "fcs_unscaled_max": fcs2_attribute_rules.UNSCALED_MAX,
        "fcs_min": fcs2_attribute_rules.FCS_MIN,
        "fcs_max": fcs2_attribute_rules.FCS_MAX,
        "top_k_vitamins": top_k["vitamins"],
        "top_k_minerals": top_k["minerals"],
        "top_k_specific_lipids": top_k["specific_lipids"],
        "specific_lipids_domain_weight": domain_weights["specific_lipids"],
        "phytochemicals_domain_weight": domain_weights["phytochemicals"],
    }


def test_lock_manifest_schema_requires_complete_frozen_provenance():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    release_registry = json.loads(RELEASE_REGISTRY_PATH.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "manifest_schema_version",
        "method_lock_schema_sha256",
        "method_version",
        "source_hashes",
        "fndds_releases",
        "fndds_registry_version",
        "release_registry_snapshot",
        "beta_fit_cohort",
        "calibration_fit_ids_sha256",
        "development_beta_sha256",
        "normalization_state_fingerprint",
        "scoring_beta_sha256",
        "person_meal_validation_config_sha256",
        "mapping_version",
        "implementation_source_sha256",
        "method_lock_gate_implementation_sha256",
        "fixed_parameters",
        "validation_embargo",
    }

    properties = schema["properties"]
    assert properties["manifest_schema_version"]["const"] == "attribute-gmnps-method-lock-v1"
    assert properties["method_version"]["const"] == ATTRIBUTE_GMNPS_SCORING_VERSION
    assert properties["mapping_version"]["const"] == PRIMARY_MAPPING_VERSION
    assert properties["fndds_registry_version"]["const"] == FCS2_FNDDS_REGISTRY_VERSION
    assert properties["fndds_releases"]["const"] == release_registry["canonical_release_set"]
    assert properties["validation_embargo"]["const"] is True
    assert properties["fixed_parameters"]["const"] == _runtime_fixed_parameters()

    hashes = properties["source_hashes"]
    assert hashes["additionalProperties"] is False
    assert set(hashes["required"]) == {
        "official_fcs",
        "food_metadata",
        "baseline_attribute_points",
        "food_exposures",
        "effective_attribute_weights",
        "input_manifest",
        "food_source_linkage",
    }
    for field in hashes["required"]:
        assert hashes["properties"][field]["pattern"] == "^[0-9a-f]{64}$"

    cohort = properties["beta_fit_cohort"]
    assert cohort["additionalProperties"] is False
    assert set(cohort["required"]) == {"cohort_id", "fit_n"}
    assert properties["calibration_fit_ids_sha256"]["pattern"] == "^[0-9a-f]{64}$"
    for field in (
        "development_beta_sha256",
        "normalization_state_fingerprint",
        "scoring_beta_sha256",
    ):
        assert properties[field]["pattern"] == "^[0-9a-f]{64}$"
    assert "frozen fit state" in properties["development_beta_sha256"]["description"].lower()
    assert "frozen fit state" in properties["normalization_state_fingerprint"]["description"].lower()
    assert "run instance" in properties["scoring_beta_sha256"]["description"].lower()
    config_hash = properties["person_meal_validation_config_sha256"]
    assert config_hash["pattern"] == "^[0-9a-f]{64}$"
    assert "run instance" in config_hash["description"].lower()
    assert "person_meal_validation.yaml" in config_hash["description"]
    schema_hash = properties["method_lock_schema_sha256"]
    assert schema_hash["pattern"] == "^[0-9a-f]{64}$"
    assert "const" not in schema_hash
    gate_hash = properties["method_lock_gate_implementation_sha256"]
    assert gate_hash["pattern"] == "^[0-9a-f]{64}$"
    registry_snapshot = properties["release_registry_snapshot"]
    assert registry_snapshot["additionalProperties"] is False
    assert set(registry_snapshot["required"]) == {
        "snapshot_sha256",
        "schema_version",
        "registry_version",
        "digest_algorithm",
        "canonical_release_set",
        "approved_entry",
    }
    assert registry_snapshot["properties"]["schema_version"]["const"] == (
        FCS2_FNDDS_REGISTRY_SCHEMA_VERSION
    )
    assert registry_snapshot["properties"]["registry_version"]["const"] == (
        FCS2_FNDDS_REGISTRY_VERSION
    )
    assert registry_snapshot["properties"]["digest_algorithm"]["const"] == (
        FCS2_FNDDS_REGISTRY_DIGEST_ALGORITHM
    )
    assert registry_snapshot["properties"]["canonical_release_set"]["const"] == (
        release_registry["canonical_release_set"]
    )
    approved_entry = registry_snapshot["properties"]["approved_entry"]
    assert set(approved_entry["required"]) == {
        "bundle_id",
        "entry_sha256",
        "fndds_releases",
        "artifact_sha256",
        "food_source_linkage_sha256",
        "nutrient_units",
        "exposure_basis",
    }


def test_schema_has_no_outcome_dependent_acceptance_criteria():
    locked_text = "\n".join(
        path.read_text(encoding="utf-8").lower() for path in (SCHEMA_PATH, SPEC_PATH)
    )
    forbidden_patterns = (
        r"\baccuracy\b",
        r"\bauroc\b",
        r"\bauc\b",
        r"rank[\s_-]*shift",
        r"mean[\s_-]*shift",
        r"response[\s_-]*advantage",
        r"select[\s_-]*best",
        r"minimum[\s_-]*performance",
        r"minimum[\s_-]*(auc|auroc|accuracy|spearman)",
        r"acceptance[\s_-]*threshold",
    )
    assert not any(re.search(pattern, locked_text) for pattern in forbidden_patterns)


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
            version: _mapping_payload(_mapping_rows(version), rule.name)
            for version in SENSITIVITY_ATTRIBUTE_MAPPINGS
            if _mapping_payload(_mapping_rows(version), rule.name)
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
        "fixed baseline attribute within a recomputed domain",
        "Phase 2 Task 2",
        "before any validation endpoint or label is read",
        "fail closed",
        "LOCKED_BETA_TEMPERATURE",
        "LOCKED_ATTRIBUTE_RESPONSE_TEMPERATURE",
        "independent runtime constant",
        "person_meal_validation_config_sha256",
        "method_lock_schema_sha256",
        "release-registry snapshot",
        "method_lock_gate.py",
        "same-immutable-bytes loader",
        "score_centering: none",
        "beta_centering: development_median",
    }
    normalized_text = " ".join(text.lower().split())
    assert all(
        " ".join(phrase.lower().split()) in normalized_text
        for phrase in required_phrases
    )

    for rule_name in FCS2_RULES:
        assert f"`{rule_name}`" in text or rule_name in MAPPING_PATH.read_text(encoding="utf-8")

    assert "remains in the fixed residual" not in text


def test_recorded_implementation_hashes_match_real_source_bytes():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema_hashes = schema["properties"]["implementation_source_sha256"]["const"]
    spec_hashes = dict(
        re.findall(
            r"\| `([^`]+)` \| `([0-9a-f]{64})` \|",
            SPEC_PATH.read_text(encoding="utf-8"),
        )
    )
    actual_hashes = {
        source: sha256((ROOT / source).read_bytes()).hexdigest()
        for source in APPROVED_IMPLEMENTATION_SOURCES
    }

    assert schema_hashes == actual_hashes
    assert {source: spec_hashes[source] for source in actual_hashes} == actual_hashes


def test_release_registry_is_run_provenance_not_a_fixed_implementation_hash():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    implementation = schema["properties"]["implementation_source_sha256"]["const"]

    assert "code/src/configs/fcs2_fndds_release_registry.json" not in implementation
    assert "code/src/scripts/run_attribute_gmnps.py" in implementation
    assert "release_registry_snapshot" in schema["required"]
    assert schema["properties"]["method_version"]["const"] == (
        ATTRIBUTE_GMNPS_SCORING_VERSION
    )


def test_runtime_carbohydrate_proxy_is_single_mac_definition():
    static_rows = SENSITIVITY_ATTRIBUTE_MAPPINGS["carbohydrate_proxy"]
    runtime_rows = _mapping_rows("carbohydrate_proxy")
    assert runtime_rows is static_rows
    carbohydrate = [row for row in runtime_rows if row.nutrient == "Carbohydrate (g)"]
    assert len(carbohydrate) == 1
    assert carbohydrate[0].role == "effect"
    assert carbohydrate[0].channel == "MAC"


def _phase_2_task(plan, number, next_number=None):
    section = plan.split(f"### Task {number}:", 1)[1]
    if next_number is not None:
        section = section.split(f"### Task {next_number}:", 1)[0]
    return section.lower()


def test_phase_2_plan_has_an_acyclic_pre_label_method_lock_dag():
    plan = PHASE_2_PLAN_PATH.read_text(encoding="utf-8")
    task_1 = _phase_2_task(plan, 1, 2)
    task_2 = _phase_2_task(plan, 2, 3)
    task_3 = _phase_2_task(plan, 3, 4)

    task_1_contract = {
        "provenance inventory",
        "generator/validator contract",
        "does not generate or require a run-level instance",
    }
    assert all(phrase in task_1 for phrase in task_1_contract)

    stages = (
        "stage 2a: predictor-only acquisition and reconstruction",
        "stage 2b: freeze scoring inputs and analysis config",
        "stage 2c: generate and validate the real method-lock instance",
        "stage 2d: unlock outcome loader after gate success",
    )
    assert all(stage in task_2 for stage in stages)
    assert [task_2.index(stage) for stage in stages] == sorted(
        task_2.index(stage) for stage in stages
    )

    task_2_required = {
        "method_lock_manifest.json",
        "development_beta_sha256",
        "normalization_state_fingerprint",
        "scoring_beta_sha256",
        "held-out scoring beta",
        "food bundle",
        "person_meal_validation.yaml",
        "must not read outcome or label tables",
        "only after the gate succeeds",
        "fail closed",
        "synthetic and aggregate supplementary labels cannot bypass",
        "record the direct-validation path as blocked",
        "must not fabricate values",
    }
    assert all(phrase in task_2 for phrase in task_2_required)
    assert task_2.index("must not read outcome or label tables") < task_2.index(stages[2])
    assert task_2.index("person_meal_validation.yaml") < task_2.index(stages[2])
    assert "create: `code/src/configs/person_meal_validation.yaml`" not in task_3
    assert "consume the frozen `person_meal_validation.yaml`" in task_3


def test_phase_2_plan_revalidates_config_hash_and_rejects_tampering_at_both_entries():
    plan = PHASE_2_PLAN_PATH.read_text(encoding="utf-8")
    task_2 = _phase_2_task(plan, 2, 3)
    task_3 = _phase_2_task(plan, 3, 4)
    stages = (
        "stage 2b: freeze scoring inputs and analysis config",
        "stage 2c: generate and validate the real method-lock instance",
        "stage 2d: unlock outcome loader after gate success",
    )
    stage_2b = task_2.split(stages[0], 1)[1].split(stages[1], 1)[0]
    stage_2c = task_2.split(stages[1], 1)[1].split(stages[2], 1)[0]
    stage_2d = task_2.split(stages[2], 1)[1]
    field = "person_meal_validation_config_sha256"

    assert field in stage_2b and "canonical bytes" in stage_2b
    assert field in stage_2c and "write" in stage_2c and "manifest" in stage_2c
    assert field in stage_2d and "recompute" in stage_2d
    assert "compare" in stage_2d and "verified manifest" in stage_2d
    assert field in task_3 and "recompute" in task_3
    assert "compare" in task_3 and "verified manifest" in task_3
    for run_hash in (
        "method_lock_schema_sha256",
        "release-registry snapshot",
        "method_lock_gate.py",
    ):
        assert run_hash in stage_2c
        assert run_hash in stage_2d
        assert run_hash in task_3

    tamper_contract = {
        "missing config",
        "config file modification",
        "hash mismatch",
        "fail closed",
        "must not load outcomes",
    }
    assert all(phrase in task_2 for phrase in tamper_contract)
    assert all(phrase in task_3 for phrase in tamper_contract)
