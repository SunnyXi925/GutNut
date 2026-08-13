"""Testing-only synthetic checks for the locked person-meal benchmark.

All response values are generated in memory for software verification only.
They are not observed, controlled, aggregate, or manuscript benchmark results.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

import gmnps.validation.person_meal_benchmark as benchmark_module
from gmnps.data_sources.predict_zoe_loader import LoadedTable
from gmnps.scoring.attribute_gmnps import load_release_registry_snapshot
from gmnps.validation.person_meal_benchmark import (
    BenchmarkAccessBlocked,
    FeatureContract,
    REQUIRED_COMPARATORS,
    compute_regression_metrics,
    load_locked_outcomes_for_benchmark,
    participant_bootstrap_ci,
    participant_permutation_test,
    run_person_meal_benchmark,
    validate_benchmark_method_lock,
    validate_feature_contract,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "code/src/configs/person_meal_validation.yaml"
REGISTRY_PATH = ROOT / "code/src/configs/fcs2_fndds_release_registry.json"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _testing_only_lock_fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "person_meal_validation.yaml"
    config.write_bytes(CONFIG_PATH.read_bytes())
    schema = tmp_path / "method_lock_manifest.schema.json"
    schema.write_bytes(b'{"testing_only":"schema bytes"}\n')
    registry = tmp_path / "fcs2_fndds_release_registry.json"
    registry.write_bytes(REGISTRY_PATH.read_bytes())
    gate = tmp_path / "method_lock_gate.py"
    gate.write_bytes(b'"""testing-only gate byte fixture."""\n')

    parsed_registry = load_release_registry_snapshot(registry.read_bytes())
    manifest = {
        "method_lock_schema_sha256": _sha(schema),
        "person_meal_validation_config_sha256": _sha(config),
        "method_lock_gate_implementation_sha256": _sha(gate),
        "release_registry_snapshot": {
            "snapshot_sha256": _sha(registry),
            "schema_version": parsed_registry.schema_version,
            "registry_version": parsed_registry.registry_version,
            "digest_algorithm": parsed_registry.digest_algorithm,
            "canonical_release_set": list(parsed_registry.canonical_release_set),
        },
    }
    manifest_path = tmp_path / "method_lock_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    paths = SimpleNamespace(
        person_meal_validation_config=config,
        method_lock_schema=schema,
        release_registry=registry,
        gate_implementation=gate,
    )
    return paths, manifest_path, manifest


def _testing_only_tables() -> tuple[pd.DataFrame, pd.DataFrame, FeatureContract]:
    predictors: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    for participant_number in range(20):
        participant_id = f"testing-p{participant_number:02d}"
        family_number = participant_number // 2
        age = 25.0 + participant_number
        diet_score = float((participant_number * 3) % 11)
        microbiome_1 = float(np.sin(participant_number / 3))
        microbiome_2 = float(np.cos(participant_number / 4))
        for meal_number in range(3):
            meal_id = f"testing-meal-{meal_number}"
            food_id = f"testing-food-{meal_number}"
            fcs_score = 35.0 + 8.0 * meal_number
            legacy_offset = fcs_score + microbiome_1
            gmnps_attribute_1 = fcs_score + 2.0 * microbiome_1
            gmnps_attribute_2 = diet_score - microbiome_2 * (meal_number + 1)
            predictors.append(
                {
                    "participant_id": participant_id,
                    "meal_id": meal_id,
                    "food_id": food_id,
                    "family_id": f"testing-family-{family_number:02d}",
                    "twin_id": (
                        f"testing-twin-{family_number:02d}"
                        if participant_number < 4
                        else None
                    ),
                    "cohort_id": f"testing-cohort-{family_number % 3}",
                    "age": age,
                    "diet_score": diet_score,
                    "fcs_score": fcs_score,
                    "microbiome_1": microbiome_1,
                    "microbiome_2": microbiome_2,
                    "legacy_offset": legacy_offset,
                    "gmnps_attribute_1": gmnps_attribute_1,
                    "gmnps_attribute_2": gmnps_attribute_2,
                }
            )
            outcomes.append(
                {
                    "participant_id": participant_id,
                    "meal_id": meal_id,
                    "glucose_iAUC_2h": (
                        0.03 * age + 0.08 * fcs_score + microbiome_1
                    ),
                    "tg_6h_rise": (
                        0.02 * diet_score + 0.01 * fcs_score - 0.2 * microbiome_2
                    ),
                }
            )
    feature_contract = FeatureContract(
        clinical_demographic_diet=("age", "diet_score"),
        fcs=("fcs_score",),
        microbiome=("microbiome_1", "microbiome_2"),
        legacy_final_score_offset=("legacy_offset",),
        locked_attribute_gmnps=("gmnps_attribute_1", "gmnps_attribute_2"),
        mapping_unit="food_id",
    )
    return pd.DataFrame(predictors), pd.DataFrame(outcomes), feature_contract


@pytest.mark.parametrize("tamper", ["config", "schema", "registry", "gate"])
def test_independent_task3_hash_revalidation_fails_closed_before_outcome_loader(
    tmp_path, monkeypatch, tamper
):
    paths, manifest_path, _ = _testing_only_lock_fixture(tmp_path)
    monkeypatch.setattr(
        benchmark_module,
        "validate_method_lock_manifest",
        lambda *args, **kwargs: None,
    )
    outcome_reads: list[str] = []
    monkeypatch.setattr(
        benchmark_module,
        "load_outcome_table_after_gate",
        lambda *args, **kwargs: outcome_reads.append("outcome-read"),
    )
    target = {
        "config": paths.person_meal_validation_config,
        "schema": paths.method_lock_schema,
        "registry": paths.release_registry,
        "gate": paths.gate_implementation,
    }[tamper]
    target.write_bytes(target.read_bytes() + b"tamper")

    with pytest.raises(BenchmarkAccessBlocked, match="lock|hash|registry|config|gate"):
        load_locked_outcomes_for_benchmark(
            "testing-only-controlled-source",
            manifest_path=manifest_path,
            method_lock_paths=paths,
        )
    assert outcome_reads == []


def test_missing_manifest_and_real_task2_validator_failure_take_precedence(
    tmp_path, monkeypatch
):
    paths, manifest_path, _ = _testing_only_lock_fixture(tmp_path)
    outcome_reads: list[str] = []
    monkeypatch.setattr(
        benchmark_module,
        "load_outcome_table_after_gate",
        lambda *args, **kwargs: outcome_reads.append("outcome-read"),
    )
    with pytest.raises(BenchmarkAccessBlocked, match="manifest"):
        load_locked_outcomes_for_benchmark(
            "testing-only-controlled-source",
            manifest_path=tmp_path / "missing-manifest.json",
            method_lock_paths=paths,
        )

    validator_calls: list[str] = []

    def fail_real_validator(*args, **kwargs):
        validator_calls.append("task2-validator")
        raise ValueError("testing-only validator failure")

    monkeypatch.setattr(
        benchmark_module,
        "validate_method_lock_manifest",
        fail_real_validator,
    )
    with pytest.raises(BenchmarkAccessBlocked, match="Task 2|method-lock|validation"):
        load_locked_outcomes_for_benchmark(
            "testing-only-controlled-source",
            manifest_path=manifest_path,
            method_lock_paths=paths,
        )
    assert validator_calls == ["task2-validator"]
    assert outcome_reads == []


def test_registry_constants_are_compared_with_live_registry_bytes(tmp_path, monkeypatch):
    paths, manifest_path, manifest = _testing_only_lock_fixture(tmp_path)
    payload = json.loads(paths.release_registry.read_text(encoding="utf-8"))
    payload["registry_version"] = "testing-only-tampered-version"
    paths.release_registry.write_text(json.dumps(payload), encoding="utf-8")
    manifest["release_registry_snapshot"]["snapshot_sha256"] = _sha(
        paths.release_registry
    )
    manifest["release_registry_snapshot"]["registry_version"] = payload[
        "registry_version"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        benchmark_module,
        "validate_method_lock_manifest",
        lambda *args, **kwargs: None,
    )
    outcome_reads: list[str] = []
    monkeypatch.setattr(
        benchmark_module,
        "load_outcome_table_after_gate",
        lambda *args, **kwargs: outcome_reads.append("outcome-read"),
    )

    with pytest.raises(BenchmarkAccessBlocked, match="registry|constant|version"):
        load_locked_outcomes_for_benchmark(
            "testing-only-controlled-source",
            manifest_path=manifest_path,
            method_lock_paths=paths,
        )
    assert outcome_reads == []


def _run_testing_only_benchmark(tmp_path, monkeypatch):
    predictors, outcomes, feature_contract = _testing_only_tables()
    paths, manifest_path, _ = _testing_only_lock_fixture(tmp_path)
    validation_calls: list[str] = []

    def record_validation(*args, **kwargs):
        validation_calls.append("task2-validator")

    monkeypatch.setattr(
        benchmark_module,
        "validate_method_lock_manifest",
        record_validation,
    )
    monkeypatch.setattr(
        benchmark_module,
        "load_outcome_table_after_gate",
        lambda *args, **kwargs: LoadedTable(
            frame=outcomes.copy(),
            sha256="1" * 64,
            source_id="testing-only-controlled-source",
        ),
    )
    monkeypatch.setattr(benchmark_module, "BOOTSTRAP_REPLICATES", 12)
    monkeypatch.setattr(benchmark_module, "PERMUTATION_REPLICATES", 15)
    result = run_person_meal_benchmark(
        "testing-only-controlled-source",
        manifest_path=manifest_path,
        method_lock_paths=paths,
        predictors=predictors,
        feature_contract=feature_contract,
    )
    assert validation_calls == ["task2-validator"]
    return result


def test_all_locked_comparators_use_identical_rows_and_splits(tmp_path, monkeypatch):
    result = _run_testing_only_benchmark(tmp_path, monkeypatch)
    predictions = result.predictions

    assert set(predictions["comparator"]) == set(REQUIRED_COMPARATORS)
    assert set(predictions["endpoint"]) == {"glucose_iAUC_2h", "tg_6h_rise"}
    for (_, _), frame in predictions.groupby(["endpoint", "outer_fold"]):
        opportunities = {
            comparator: tuple(sorted(group["row_id"].astype(str)))
            for comparator, group in frame.groupby("comparator")
        }
        assert len(set(opportunities.values())) == 1
        split_labels = {
            comparator: tuple(sorted(group["participant_id"].astype(str).unique()))
            for comparator, group in frame.groupby("comparator")
        }
        assert len(set(split_labels.values())) == 1

    required_provenance = {
        "n_participants",
        "n_meals",
        "split_seed",
        "endpoint_config_sha256",
        "validation_config_sha256",
    }
    assert required_provenance.issubset(result.absolute_metrics.columns)
    assert required_provenance.issubset(result.paired_metrics.columns)
    assert set(result.paired_metrics["reference"]) == set(REQUIRED_COMPARATORS) - {
        "locked_attribute_gmnps"
    }


def test_nested_fit_audit_never_uses_outer_test_participants(tmp_path, monkeypatch):
    result = _run_testing_only_benchmark(tmp_path, monkeypatch)
    predictors, _, _ = _testing_only_tables()
    for event in result.fit_audit.itertuples(index=False):
        outer = result.splits[event.outer_fold].outer
        outer_test_participants = set(
            predictors.iloc[list(outer.test_positions)]["participant_id"].astype(str)
        )
        assert set(event.fit_participant_ids).isdisjoint(outer_test_participants)
        assert set(event.fit_participant_ids).isdisjoint(
            set(event.validation_participant_ids)
        )
        if event.stage == "inner_tuning":
            assert event.inner_fold is not None
            assert event.validation_participant_ids
        else:
            assert event.stage == "outer_refit"
            assert event.inner_fold is None


def test_benchmark_is_deterministic_including_null_comparators(tmp_path, monkeypatch):
    first = _run_testing_only_benchmark(tmp_path / "first", monkeypatch)
    second = _run_testing_only_benchmark(tmp_path / "second", monkeypatch)

    assert_frame_equal(first.predictions, second.predictions)
    assert_frame_equal(first.absolute_metrics, second.absolute_metrics)
    assert_frame_equal(first.paired_metrics, second.paired_metrics)
    assert_frame_equal(first.fit_audit, second.fit_audit)


@pytest.mark.parametrize(
    "forbidden_feature",
    ["participant_id", "glucose_iAUC_2h", "post_split_mean", "response_proxy"],
)
def test_response_ids_and_post_split_statistics_cannot_be_predictors(
    forbidden_feature: str,
):
    predictors, _, feature_contract = _testing_only_tables()
    if forbidden_feature not in predictors:
        predictors[forbidden_feature] = 0.0
    leaked = FeatureContract(
        clinical_demographic_diet=(forbidden_feature,),
        fcs=feature_contract.fcs,
        microbiome=feature_contract.microbiome,
        legacy_final_score_offset=feature_contract.legacy_final_score_offset,
        locked_attribute_gmnps=feature_contract.locked_attribute_gmnps,
        mapping_unit=feature_contract.mapping_unit,
    )
    with pytest.raises(ValueError, match="leak|forbidden|endpoint|identifier"):
        validate_feature_contract(
            predictors,
            leaked,
            endpoint_names=(
                "glucose_iAUC_2h",
                "tg_6h_rise",
                "c_peptide_iAUC_2h",
            ),
        )


def test_metric_edge_cases_and_participant_bootstrap_are_well_defined():
    perfect = compute_regression_metrics(
        np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])
    )
    assert perfect == {
        "mae": pytest.approx(0.0),
        "rmse": pytest.approx(0.0),
        "spearman": pytest.approx(1.0),
        "calibration_slope": pytest.approx(1.0),
        "calibration_intercept": pytest.approx(0.0),
    }

    constant = compute_regression_metrics(
        np.array([1.0, 2.0, 3.0, np.nan]),
        np.array([2.0, 2.0, 2.0, 5.0]),
    )
    assert np.isnan(constant["spearman"])
    assert np.isnan(constant["calibration_slope"])
    assert constant["calibration_intercept"] == pytest.approx(2.0)

    interval = participant_bootstrap_ci(
        np.array([1.0, 1.2, 2.0, 2.2]),
        np.array([1.1, 1.1, 2.1, 2.1]),
        np.array(["p1", "p1", "p2", "p2"]),
        metric="mae",
        n_bootstrap=40,
        seed=31415,
    )
    assert interval.lower <= interval.estimate <= interval.upper
    assert interval.n_participants == 2

    with pytest.raises(ValueError, match="shape"):
        compute_regression_metrics(np.array([1.0]), np.array([1.0, 2.0]))


def test_permutation_null_swaps_complete_participant_clusters():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    model = np.array([0.1, 1.1, 1.9, 3.1])
    reference = np.array([2.0, 2.0, 2.0, 2.0])
    participants = np.array(["p1", "p1", "p2", "p2"])
    seed = 16180
    n_permutations = 9

    result = participant_permutation_test(
        y,
        model,
        reference,
        participants,
        metric="mae",
        n_permutations=n_permutations,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    expected = []
    for _ in range(n_permutations):
        swap_by_participant = dict(
            zip(["p1", "p2"], rng.integers(0, 2, size=2).astype(bool))
        )
        permuted_model = model.copy()
        permuted_reference = reference.copy()
        for participant_id, swap in swap_by_participant.items():
            if swap:
                mask = participants == participant_id
                permuted_model[mask] = reference[mask]
                permuted_reference[mask] = model[mask]
        expected.append(
            compute_regression_metrics(y, permuted_model)["mae"]
            - compute_regression_metrics(y, permuted_reference)["mae"]
        )
    assert result.null_distribution == pytest.approx(tuple(expected))
    assert result.n_participants == 2
    assert 0.0 < result.p_value <= 1.0
