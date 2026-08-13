import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from gmnps.validation.attribute_population_safety import (
    PopulationSafetyConfig,
    audit_population_safety,
    validate_population_provenance,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/scripts/run_attribute_validation.py"


def _scores() -> pd.DataFrame:
    foods = [
        ("leaf", "Leafy greens", "Vegetables", "Dark-green vegetables", 90.0),
        ("bean", "Beans", "Vegetables", "Legumes", 75.0),
        ("bread", "Whole-grain bread", "Grains", "Whole grains", 65.0),
        ("rice", "White rice", "Grains", "Refined grains", 45.0),
        ("soda", "Soda", "Beverages", "Sugar-sweetened beverages", 20.0),
        ("water", "Water", "Beverages", "Water", 100.0),
    ]
    rows = []
    shifts = {
        "p1": [0, 2, 3, -2, 1, -1],
        "p2": [1, 0, 2, -1, 0, -2],
        "p3": [-2, 1, -3, 2, -1, 0],
        # Explicit extreme reversal used to exercise the safety audit.
        "p4": [-70, 0, 1, 0, 65, -1],
    }
    components = {"p1": "c1", "p2": "c1", "p3": "c2", "p4": "c3"}
    for individual_id, deltas in shifts.items():
        for food, delta in zip(foods, deltas):
            food_id, food_name, group, subgroup, fcs2 = food
            rows.append(
                {
                    "individual_id": individual_id,
                    "component_id": components[individual_id],
                    "food_id": food_id,
                    "food_name": food_name,
                    "food_group": group,
                    "food_subgroup": subgroup,
                    "FCS2": fcs2,
                    "GMNPS_delta": float(delta),
                    "GMNPS_score": fcs2 + float(delta),
                    "method_role": "primary",
                }
            )
    return pd.DataFrame(rows)


def _manifest(*, production: bool = False) -> dict[str, object]:
    testing = not production
    return {
        "schema_version": "gmnps-attribute-validation-input-v1",
        "analysis_kind": "population_safety",
        "evidence_role": "population_safety_design_audit",
        "data_class": (
            "synthetic_test_fixture" if testing else "locked_attribute_level_gmnps"
        ),
        "method": "attribute_recomposition",
        "method_role": "primary",
        "score_centering": "none",
        "production_label": "non-production" if testing else "production",
        "synthetic": testing,
        "testing_only": testing,
        "scoring_frozen_before_labels": True,
        "n_foods": 6 if testing else 9234,
        "independent_unit": "component_id",
    }


def _config() -> PopulationSafetyConfig:
    return PopulationSafetyConfig(
        bootstrap_replicates=80,
        bootstrap_seed=31415,
        minimum_valid_replicates=40,
    )


def test_group_and_subgroup_distribution_convergence_and_discrimination():
    audit = audit_population_safety(
        _scores(), _manifest(), config=_config(), allow_test_data=True
    )

    assert audit.evidence_role == "population_safety_design_audit"
    assert audit.global_summary["estimand"] == "food-level population mean"
    assert audit.global_summary["score_centering"] == "none"
    assert audit.global_summary["n_foods"] == 6
    assert audit.global_summary["n_individuals"] == 4

    group = audit.group_convergence.set_index("stratum")
    assert set(group.index) == {"Vegetables", "Grains", "Beverages"}
    assert {
        "n_foods",
        "n_individuals",
        "fcs2_median",
        "gmnps_population_median",
        "median_shift",
        "mean_absolute_food_shift",
        "fcs2_iqr",
        "gmnps_population_iqr",
        "distribution_l1_quantile_distance",
        "within_stratum_spearman",
        "evidence_role",
    }.issubset(group.columns)

    subgroup = audit.subgroup_convergence
    assert subgroup["stratum"].nunique() == 6
    assert (subgroup["stratum_level"] == "food_subgroup").all()

    discrimination = audit.between_group_discrimination
    assert len(discrimination) == 3
    assert {
        "stratum_a",
        "stratum_b",
        "fcs2_mean_difference",
        "gmnps_population_mean_difference",
        "direction_preserved",
        "absolute_difference_ratio",
    }.issubset(discrimination.columns)
    assert discrimination["evidence_role"].eq(
        "population_safety_design_audit"
    ).all()


def test_transition_matrix_direction_magnitude_and_implausible_reversal_audit():
    audit = audit_population_safety(
        _scores(), _manifest(), config=_config(), allow_test_data=True
    )

    transitions = audit.transition_matrix
    assert len(transitions) == 9
    assert transitions["n_individual_foods"].sum() == len(_scores())
    assert {
        "fcs2_category",
        "gmnps_category",
        "direction",
        "mean_delta",
        "median_absolute_delta",
        "origin_fraction",
    }.issubset(transitions.columns)

    reversals = audit.reversal_audit
    assert set(reversals["reversal_type"]) == {
        "encourage_to_minimize",
        "minimize_to_encourage",
    }
    assert set(reversals["food_id"]) == {"leaf", "soda"}
    assert reversals["audit_status"].eq(
        "flagged_for_clinical_nutritional_review"
    ).all()
    assert audit.global_summary["implausible_reversal_count"] == 2


def test_cluster_bootstrap_reports_units_ci_seed_valid_repeats_and_exclusions():
    frame = _scores()
    frame.loc[(frame["individual_id"] == "p3") & (frame["food_id"] == "rice"), "GMNPS_score"] = float("nan")
    audit = audit_population_safety(
        frame, _manifest(), config=_config(), allow_test_data=True
    )
    bootstrap = audit.bootstrap_summary.set_index("metric").loc[
        "spearman_fcs2_vs_population_gmnps"
    ]

    assert bootstrap["cluster_column"] == "component_id"
    assert bootstrap["n_clusters"] == 3
    assert bootstrap["n_individuals"] == 4
    assert bootstrap["replicates_requested"] == 80
    assert 40 <= bootstrap["replicates_valid"] <= 80
    assert bootstrap["ci_method"] == "cluster_percentile"
    assert bootstrap["bootstrap_seed"] == 31415
    assert bootstrap["ci_lower"] <= bootstrap["estimate"] <= bootstrap["ci_upper"]
    assert bootstrap["missing_n_rows"] == 1
    assert bootstrap["excluded_n_rows"] == 1

    repeated = audit_population_safety(
        frame, _manifest(), config=_config(), allow_test_data=True
    ).bootstrap_summary
    pd.testing.assert_frame_equal(audit.bootstrap_summary, repeated)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("method", "legacy_final_score_offset", "attribute_recomposition"),
        ("method_role", "sensitivity", "method_role"),
        ("score_centering", "population_mean", "score_centering"),
        ("evidence_role", "external_validation", "evidence_role"),
        ("scoring_frozen_before_labels", False, "frozen"),
    ],
)
def test_population_provenance_fails_closed(field, value, message):
    manifest = _manifest()
    manifest[field] = value
    with pytest.raises(ValueError, match=message):
        validate_population_provenance(manifest, allow_test_data=True)


def test_population_input_rejects_label_leakage_and_inconsistent_uncentered_delta():
    leaked = _scores().assign(glucose_iAUC_2h=1.0)
    with pytest.raises(ValueError, match="outcome/label leakage"):
        audit_population_safety(
            leaked, _manifest(), config=_config(), allow_test_data=True
        )

    centered = _scores()
    centered.loc[0, "GMNPS_delta"] += 1.0
    with pytest.raises(ValueError, match="GMNPS_delta must equal GMNPS_score - FCS2"):
        audit_population_safety(
            centered, _manifest(), config=_config(), allow_test_data=True
        )


def test_cli_defaults_fail_closed_and_invalid_provenance_is_checked_before_table(tmp_path):
    no_args = subprocess.run(
        [sys.executable, str(SCRIPT)], text=True, capture_output=True
    )
    assert no_args.returncode != 0
    assert not list(tmp_path.iterdir())

    table = tmp_path / "not_a_csv.csv"
    table.write_text("not,a,valid\n\"unterminated", encoding="utf-8")
    manifest = _manifest(production=True)
    manifest["evidence_role"] = "external_validation"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "results"
    invalid = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--individual-food",
            str(table),
            "--provenance-manifest",
            str(manifest_path),
            "--output-dir",
            str(output),
            "--write-source-data",
        ],
        text=True,
        capture_output=True,
    )
    assert invalid.returncode != 0
    assert "evidence_role" in invalid.stderr
    assert not output.exists()


def test_cli_dry_run_writes_nothing_and_synthetic_values_can_never_be_results(tmp_path):
    frame = _scores()
    table = tmp_path / "individual_food.csv"
    frame.to_csv(table, index=False)
    manifest = _manifest()
    manifest["individual_food_sha256"] = hashlib.sha256(table.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "results"

    dry = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--individual-food",
            str(table),
            "--provenance-manifest",
            str(manifest_path),
            "--output-dir",
            str(output),
            "--allow-test-inputs",
        ],
        text=True,
        capture_output=True,
    )
    assert dry.returncode == 0, dry.stderr
    assert "validated_dry_run" in dry.stdout
    assert not output.exists()

    write = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--individual-food",
            str(table),
            "--provenance-manifest",
            str(manifest_path),
            "--output-dir",
            str(output),
            "--allow-test-inputs",
            "--write-source-data",
        ],
        text=True,
        capture_output=True,
    )
    assert write.returncode != 0
    assert "testing/synthetic inputs cannot be written" in write.stderr
    assert not output.exists()
