import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmnps.validation.attribute_population_safety import (
    PopulationSafetyConfig,
    audit_population_safety,
    task4_binding_hashes,
    validate_population_provenance,
    verify_phase1_run_artifacts,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/scripts/run_attribute_validation.py"


def _scores(n_people: int = 4) -> pd.DataFrame:
    foods = [
        ("leaf", "Vegetables", "Leafy", 90.0),
        ("bean", "Vegetables", "Legumes", 75.0),
        ("bread", "Grains", "Whole grains", 65.0),
        ("rice", "Grains", "Refined grains", 45.0),
        ("soda", "Beverages", "Sweet drinks", 20.0),
        ("water", "Beverages", "Water", 100.0),
    ]
    shifts = [
        [0, 2, 3, -2, 1, -1],
        [1, 0, 2, -1, 0, -2],
        [-2, 1, -3, 2, -1, 0],
        [-70, 0, 1, 0, 65, -1],
    ]
    rows = []
    for person_index in range(n_people):
        person = f"p{person_index + 1}"
        for (food, group, subgroup, fcs2), delta in zip(
            foods, shifts[person_index % len(shifts)]
        ):
            rows.append(
                {
                    "individual_id": person,
                    "component_id": f"c{person_index + 1}",
                    "food_id": food,
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
    return {
        "schema_version": "gmnps-attribute-validation-input-v2",
        "analysis_kind": "population_safety",
        "evidence_role": "population_safety_design_audit",
        "data_class": (
            "locked_attribute_level_gmnps" if production else "synthetic_test_fixture"
        ),
        "method": "attribute_recomposition",
        "method_role": "primary",
        "score_centering": "none",
        "production_label": "production" if production else "non-production",
        "synthetic": not production,
        "testing_only": not production,
        "scoring_frozen_before_labels": True,
        "n_foods": 9234 if production else 6,
        "independent_unit": "component_id",
        "bundle_id": "registered-production-bundle" if production else "fixture",
    }


def _config(**changes) -> PopulationSafetyConfig:
    values = {
        "bootstrap_replicates": 60,
        "bootstrap_seed": 31415,
        "minimum_valid_replicates": 30,
        "minimum_clusters": 3,
    }
    values.update(changes)
    return PopulationSafetyConfig(**values)


def _audit(frame=None, config=None):
    return audit_population_safety(
        _scores() if frame is None else frame,
        _manifest(),
        config=_config() if config is None else config,
        allow_test_data=True,
    )


def test_complete_panel_group_subgroup_transition_and_hand_calculated_counts():
    audit = _audit()
    assert audit.panel_audit == {
        "panel_status": "complete_rectangular",
        "n_individuals": 4,
        "n_independent_units": 4,
        "n_foods": 6,
        "expected_rows": 24,
        "observed_rows": 24,
        "min_effective_units_per_food": 4,
        "max_effective_units_per_food": 4,
    }
    assert audit.global_summary["n_foods"] == 6
    assert audit.transition_matrix["n_individual_foods"].sum() == 24
    assert audit.global_summary["implausible_reversal_count"] == 2
    assert set(audit.reversal_audit["food_id"]) == {"leaf", "soda"}

    subgroup = audit.subgroup_convergence
    assert {"food_group", "food_subgroup"}.issubset(subgroup.columns)
    assert len(subgroup) == 6
    assert subgroup["n_effective_units_per_food_min"].eq(4).all()
    assert audit.group_convergence["n_effective_units_per_food_min"].eq(4).all()


@pytest.mark.parametrize("mutation", ["missing_row", "duplicate", "nan", "bad_delta"])
def test_population_panel_and_values_fail_closed(mutation):
    frame = _scores()
    if mutation == "missing_row":
        frame = frame.iloc[1:].copy()
        message = "identical complete food panel"
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        message = "one row per individual_id-food_id"
    elif mutation == "nan":
        frame.loc[0, "GMNPS_score"] = np.nan
        message = "finite"
    else:
        frame.loc[0, "GMNPS_delta"] += 1
        message = "GMNPS_delta must equal"
    with pytest.raises(ValueError, match=message):
        _audit(frame)


def test_subgroup_has_exactly_one_parent_group():
    frame = _scores()
    frame.loc[frame["food_id"].eq("bread"), "food_subgroup"] = "Leafy"
    with pytest.raises(ValueError, match="unique parent food_group"):
        _audit(frame)


@pytest.mark.parametrize(
    ("baseline", "personalized", "expected"),
    [
        ((80.0, 20.0), (70.0, 30.0), "preserved"),
        ((80.0, 20.0), (30.0, 70.0), "reversed"),
        ((80.0, 20.0), (50.0, 50.0), "collapsed"),
        ((50.0, 50.0), (60.0, 40.0), "not_estimable"),
    ],
)
def test_between_group_discrimination_status(baseline, personalized, expected):
    rows = []
    for person in ("p1", "p2", "p3"):
        for index, group in enumerate(("A", "B")):
            fcs2 = baseline[index]
            score = personalized[index]
            rows.append(
                {
                    "individual_id": person,
                    "component_id": person,
                    "food_id": f"f{index}",
                    "food_group": group,
                    "food_subgroup": f"{group}-sub",
                    "FCS2": fcs2,
                    "GMNPS_score": score,
                    "GMNPS_delta": score - fcs2,
                }
            )
    manifest = _manifest()
    manifest["n_foods"] = 2
    audit = audit_population_safety(
        pd.DataFrame(rows), manifest, config=_config(), allow_test_data=True
    )
    row = audit.between_group_discrimination.iloc[0]
    assert row["discrimination_status"] == expected
    assert bool(row["direction_preserved"]) is (expected == "preserved")


def test_minimum_clusters_is_frozen_separately_from_valid_replicates():
    audit = _audit(_scores(n_people=2), _config(minimum_clusters=3))
    row = audit.bootstrap_summary.iloc[0]
    assert row["analysis_status"] == "not_estimable"
    assert row["reason"] == "fewer_than_minimum_independent_clusters"
    assert row["n_clusters"] == 2
    assert row["replicates_valid"] == 0
    assert pd.isna(row["ci_lower"])
    assert pd.isna(row["ci_upper"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bootstrap_replicates", 60.5),
        ("minimum_valid_replicates", 30.5),
        ("minimum_clusters", 3.5),
        ("bootstrap_seed", 2.5),
    ],
)
def test_replicates_thresholds_and_seed_are_strict_integers(field, value):
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


def test_production_requires_explicit_false_and_live_verification_token():
    manifest = _manifest(production=True)
    manifest.pop("synthetic")
    with pytest.raises(ValueError, match="synthetic must be explicitly false"):
        validate_population_provenance(manifest)

    manifest = _manifest(production=True)
    with pytest.raises(ValueError, match="live Phase 1 artifact verification"):
        audit_population_safety(pd.DataFrame(), manifest)


def _write_phase1_binding(tmp_path: Path):
    table = tmp_path / "individual_food.csv"
    table.write_bytes(b"individual_id,food_id\np1,f1\n")
    upstream = tmp_path / "input_manifest.json"
    upstream.write_bytes(b'{"production_label":"production"}\n')
    run_payload = {
        "method": "attribute_recomposition",
        "method_role": "primary",
        "score_centering": "none",
        "production_label": "production",
        "development_smoke_test": False,
        "input_manifest_sha256": hashlib.sha256(upstream.read_bytes()).hexdigest(),
        "source_hashes": {
            "input_manifest": hashlib.sha256(upstream.read_bytes()).hexdigest()
        },
    }
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(json.dumps(run_payload, sort_keys=True), encoding="utf-8")
    success = tmp_path / "_SUCCESS.json"
    success.write_text(
        json.dumps(
            {
                "status": "complete",
                "files": {
                    "individual_food.csv": hashlib.sha256(table.read_bytes()).hexdigest(),
                    "run_manifest.json": hashlib.sha256(run_manifest.read_bytes()).hexdigest(),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return table, run_manifest, success, upstream


def test_phase1_success_marker_is_live_hash_binding_and_rejects_tamper(tmp_path):
    paths = _write_phase1_binding(tmp_path)
    verified = verify_phase1_run_artifacts(*paths)
    assert verified["phase1_individual_food_sha256"] == hashlib.sha256(
        paths[0].read_bytes()
    ).hexdigest()
    paths[0].write_bytes(paths[0].read_bytes() + b"p2,f1\n")
    with pytest.raises(ValueError, match="_SUCCESS.*individual_food"):
        verify_phase1_run_artifacts(*paths)


def test_phase1_live_reader_rejects_symlink(tmp_path):
    table, run_manifest, success, upstream = _write_phase1_binding(tmp_path)
    link = tmp_path / "table_link.csv"
    link.symlink_to(table)
    with pytest.raises(ValueError, match="missing or unreadable|regular file"):
        verify_phase1_run_artifacts(link, run_manifest, success, upstream)


def test_task4_binding_hashes_include_implementation_and_threshold_config():
    first = task4_binding_hashes(_config())
    second = task4_binding_hashes(_config(minimum_clusters=4))
    assert {
        "attribute_population_safety_implementation_sha256",
        "biological_consistency_implementation_sha256",
        "run_attribute_validation_implementation_sha256",
        "population_safety_config_sha256",
    } == set(first)
    assert first["population_safety_config_sha256"] != second[
        "population_safety_config_sha256"
    ]


def test_cli_failure_never_creates_output(tmp_path):
    table = tmp_path / "individual_food.csv"
    _scores().to_csv(table, index=False)
    manifest = _manifest(production=True)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "results"
    completed = subprocess.run(
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
    assert completed.returncode != 0
    assert "live Phase 1" in completed.stderr or "artifact" in completed.stderr
    assert not output.exists()
