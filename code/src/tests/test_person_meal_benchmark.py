"""Testing-only synthetic checks for the locked person-meal benchmark.

All response values are generated in memory for software verification only.
They are not observed, controlled, aggregate, or manuscript benchmark results.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import inspect
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
from gmnps.validation.method_lock_gate import (
    MethodLockError,
    validate_predictor_artifact_binding,
)
from gmnps.validation.person_meal_benchmark import (
    BenchmarkAccessBlocked,
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
COHORT_SPLIT_PATH = ROOT / "code/src/gmnps/validation/cohort_split.py"
BENCHMARK_PATH = ROOT / "code/src/gmnps/validation/person_meal_benchmark.py"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _testing_only_predictor_frame_bytes(predictors: pd.DataFrame) -> bytes:
    rows = []
    for raw_row in predictors.itertuples(index=False, name=None):
        row = []
        for value in raw_row:
            if value is None or pd.isna(value):
                row.append(None)
            elif isinstance(value, np.generic):
                row.append(value.item())
            else:
                row.append(value)
        rows.append(row)
    return _canonical_bytes(
        {
            "schema_version": "person-meal-predictor-frame-v1",
            "construction_version": "testing-only-construction-v1",
            "generated_stage": "pre-outcome_predictor_only",
            "columns": list(predictors.columns),
            "rows": rows,
        }
    )


def _testing_only_block_sha256(
    predictor_payload: dict[str, object],
    predictor_sha256: str,
    block: str,
    columns: list[str],
) -> str:
    positions = [predictor_payload["columns"].index(column) for column in columns]
    return sha256(
        _canonical_bytes(
            {
                "schema_version": "person-meal-predictor-block-v1",
                "construction_version": predictor_payload["construction_version"],
                "predictor_frame_sha256": predictor_sha256,
                "block": block,
                "columns": columns,
                "rows": [
                    [row[position] for position in positions]
                    for row in predictor_payload["rows"]
                ],
            }
        )
    ).hexdigest()


def _testing_only_lock_fixture(
    tmp_path: Path,
    predictors: pd.DataFrame | None = None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    if predictors is None:
        predictors, _ = _testing_only_tables()
    config = tmp_path / "person_meal_validation.yaml"
    config.write_bytes(CONFIG_PATH.read_bytes())
    schema = tmp_path / "method_lock_manifest.schema.json"
    schema.write_bytes(b'{"testing_only":"schema bytes"}\n')
    registry = tmp_path / "fcs2_fndds_release_registry.json"
    registry.write_bytes(REGISTRY_PATH.read_bytes())
    gate = tmp_path / "method_lock_gate.py"
    gate.write_bytes(b'"""testing-only gate byte fixture."""\n')
    cohort_split = tmp_path / "cohort_split.py"
    cohort_split.write_bytes(COHORT_SPLIT_PATH.read_bytes())
    benchmark = tmp_path / "person_meal_benchmark.py"
    benchmark.write_bytes(BENCHMARK_PATH.read_bytes())
    predictor_frame = tmp_path / "predictor_frame.json"
    predictor_frame.write_bytes(_testing_only_predictor_frame_bytes(predictors))
    feature_contract = tmp_path / "feature_contract.json"
    feature_contract.write_bytes(
        _testing_only_feature_contract_bytes(predictors, predictor_frame.read_bytes())
    )

    parsed_registry = load_release_registry_snapshot(registry.read_bytes())
    benchmark_specification = json.loads(config.read_bytes())["benchmark_specification"]
    manifest = {
        "method_lock_schema_sha256": _sha(schema),
        "person_meal_validation_config_sha256": _sha(config),
        "method_lock_gate_implementation_sha256": _sha(gate),
        "cohort_split_implementation_sha256": _sha(cohort_split),
        "person_meal_benchmark_implementation_sha256": _sha(benchmark),
        "feature_contract_sha256": _sha(feature_contract),
        "predictor_frame_sha256": _sha(predictor_frame),
        "benchmark_specification": benchmark_specification,
        "benchmark_specification_sha256": sha256(
            _canonical_bytes(benchmark_specification)[:-1]
        ).hexdigest(),
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
        cohort_split_implementation=cohort_split,
        person_meal_benchmark_implementation=benchmark,
        feature_contract=feature_contract,
        predictor_frame=predictor_frame,
    )
    return paths, manifest_path, manifest


def _testing_only_feature_contract_bytes(
    predictors: pd.DataFrame,
    predictor_frame_bytes: bytes,
) -> bytes:
    identifier_roles = {
        "participant_id": "identifier",
        "meal_id": "identifier",
        "food_id": "mapping_unit",
        "family_id": "grouping",
        "twin_id": "grouping",
        "cohort_id": "grouping",
    }
    feature_blocks = {
        "clinical_demographic_diet": [
            "age",
            "diet_score",
            "diet_category",
            "diet_flag",
        ],
        "fcs": ["fcs_score"],
        "microbiome": ["microbiome_1", "microbiome_2"],
        "legacy_final_score_offset": ["legacy_offset"],
        "locked_attribute_gmnps": ["gmnps_attribute_1", "gmnps_attribute_2"],
    }
    columns = []
    for name, role in identifier_roles.items():
        columns.append(
            {
                "name": name,
                "data_type": "string",
                "role": role,
                "nullable": name == "twin_id",
                "source_artifact_id": "testing-only-predictor-table",
            }
        )
    for block, names in feature_blocks.items():
        for name in names:
            columns.append(
                {
                    "name": name,
                    "data_type": (
                        "category"
                        if name == "diet_category"
                        else "boolean"
                        if name == "diet_flag"
                        else "number"
                    ),
                    "role": "predictor",
                    "nullable": name == "diet_category",
                    "source_artifact_id": f"testing-only-{block}",
                }
            )
    comparators = {
        "clinical_demographic_diet": {
            "feature_blocks": ["clinical_demographic_diet"],
            "transform": "identity",
            "role": "baseline",
        },
        "fcs_only": {"feature_blocks": ["fcs"], "transform": "identity", "role": "baseline"},
        "microbiome_only": {
            "feature_blocks": ["microbiome"],
            "transform": "identity",
            "role": "baseline",
        },
        "fcs_microbiome": {
            "feature_blocks": ["fcs", "microbiome"],
            "transform": "identity",
            "role": "baseline",
        },
        "legacy_final_score_offset": {
            "feature_blocks": ["legacy_final_score_offset"],
            "transform": "identity",
            "role": "baseline",
        },
        "locked_attribute_gmnps": {
            "feature_blocks": ["locked_attribute_gmnps"],
            "transform": "identity",
            "role": "primary_model",
        },
        "random_microbiome": {
            "feature_blocks": ["microbiome"],
            "transform": "random_null",
            "role": "null",
        },
        "shuffled_mapping": {
            "feature_blocks": ["locked_attribute_gmnps"],
            "transform": "development_derangement",
            "role": "null",
        },
    }
    source_ids = {column["source_artifact_id"] for column in columns}
    predictor_payload = json.loads(predictor_frame_bytes)
    predictor_sha256 = sha256(predictor_frame_bytes).hexdigest()
    payload = {
        "schema_version": "person-meal-feature-contract-v1",
        "construction_version": "testing-only-construction-v1",
        "generated_stage": "pre-outcome_predictor_only",
        "columns": columns,
        "source_artifact_sha256": {
            source_id: sha256(source_id.encode()).hexdigest()
            for source_id in sorted(source_ids)
        },
        "predictor_frame_artifact": {
            "artifact_id": "testing-only-canonical-predictor-frame",
            "schema_version": "person-meal-predictor-frame-v1",
            "sha256": predictor_sha256,
        },
        "block_artifact_sha256": {
            block: _testing_only_block_sha256(
                predictor_payload,
                predictor_sha256,
                block,
                names,
            )
            for block, names in feature_blocks.items()
        },
        "feature_blocks": {
            block: {"artifact_id": block, "columns": names}
            for block, names in feature_blocks.items()
        },
        "allowed_overlaps": [],
        "mapping_unit": {"column": "food_id", "role": "mapping_unit"},
        "comparators": comparators,
    }
    assert tuple(predictors.columns) == tuple(
        column["name"] for column in columns
    )
    return _canonical_bytes(payload)


def _rename_bound_predictor_column(
    paths,
    manifest_path: Path,
    manifest: dict[str, object],
    *,
    old_name: str,
    new_name: str,
) -> None:
    predictor_payload = json.loads(paths.predictor_frame.read_bytes())
    position = predictor_payload["columns"].index(old_name)
    predictor_payload["columns"][position] = new_name
    paths.predictor_frame.write_bytes(_canonical_bytes(predictor_payload))
    predictor_sha256 = _sha(paths.predictor_frame)

    contract = json.loads(paths.feature_contract.read_bytes())
    for column in contract["columns"]:
        if column["name"] == old_name:
            column["name"] = new_name
    for specification in contract["feature_blocks"].values():
        specification["columns"] = [
            new_name if column == old_name else column
            for column in specification["columns"]
        ]
    contract["predictor_frame_artifact"]["sha256"] = predictor_sha256
    contract["block_artifact_sha256"] = {
        block: _testing_only_block_sha256(
            predictor_payload,
            predictor_sha256,
            block,
            specification["columns"],
        )
        for block, specification in contract["feature_blocks"].items()
    }
    paths.feature_contract.write_bytes(_canonical_bytes(contract))
    manifest["predictor_frame_sha256"] = predictor_sha256
    manifest["feature_contract_sha256"] = _sha(paths.feature_contract)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _testing_only_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictors: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    for participant_number in range(20):
        participant_id = f"testing-p{participant_number:02d}"
        family_number = participant_number // 2
        age = 25.0 + participant_number
        diet_score = float((participant_number * 3) % 11)
        microbiome_1 = float(np.sin(participant_number / 3))
        microbiome_2 = float(np.cos(participant_number / 4))
        for meal_number in range(12):
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
                    "twin_id": f"testing-twin-{family_number:02d}",
                    "cohort_id": f"testing-cohort-{family_number:02d}",
                    "age": age,
                    "diet_score": diet_score,
                    "diet_category": None if meal_number == 0 else f"testing-diet-{meal_number}",
                    "diet_flag": meal_number % 2 == 0,
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
    outcome_frame = pd.DataFrame(outcomes)
    outcome_frame.loc[outcome_frame.index[0], "glucose_iAUC_2h"] = np.nan
    return pd.DataFrame(predictors), outcome_frame


@pytest.mark.parametrize(
    "tamper",
    [
        "config",
        "schema",
        "registry",
        "gate",
        "cohort_split",
        "benchmark",
        "feature_contract",
        "predictor_frame",
    ],
)
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
        "cohort_split": paths.cohort_split_implementation,
        "benchmark": paths.person_meal_benchmark_implementation,
        "feature_contract": paths.feature_contract,
        "predictor_frame": paths.predictor_frame,
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


def _run_testing_only_benchmark(
    tmp_path,
    monkeypatch,
    *,
    predictors: pd.DataFrame | None = None,
    outcomes: pd.DataFrame | None = None,
):
    default_predictors, default_outcomes = _testing_only_tables()
    predictors = default_predictors if predictors is None else predictors
    outcomes = default_outcomes if outcomes is None else outcomes
    paths, manifest_path, _ = _testing_only_lock_fixture(tmp_path, predictors)
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
    monkeypatch.setattr(benchmark_module, "_ALPHA_GRID", (1.0,))
    result = run_person_meal_benchmark(
        "testing-only-controlled-source",
        manifest_path=manifest_path,
        method_lock_paths=paths,
    )
    assert validation_calls == ["task2-validator"]
    return result


def test_all_locked_comparators_use_identical_rows_and_splits(tmp_path, monkeypatch):
    result = _run_testing_only_benchmark(tmp_path, monkeypatch)
    predictions = result.predictions

    assert set(predictions["comparator"]) == set(REQUIRED_COMPARATORS)
    assert set(predictions["endpoint"]) == {"glucose_iAUC_2h", "tg_6h_rise"}
    assert set(predictions["analysis_mode"]) == {
        "subject_held_out",
        "subject_plus_food_held_out",
        "subject_plus_meal_held_out",
        "cohort_held_out",
    }
    for (_, _, _), frame in predictions.groupby(
        ["analysis_mode", "endpoint", "outer_fold"]
    ):
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
        "manifest_sha256",
        "outcome_source_id",
        "outcome_source_sha256",
        "predictor_frame_sha256",
        "feature_contract_sha256",
        "cohort_split_implementation_sha256",
        "person_meal_benchmark_implementation_sha256",
        "benchmark_specification_id",
        "benchmark_specification_sha256",
        "seed_derivation_id",
        "outer_split_seed",
        "inner_cv_seed",
        "bootstrap_seed",
        "permutation_seed",
        "software_versions",
        "n_participants",
        "n_person_meals",
        "split_seed",
        "endpoint_config_sha256",
        "validation_config_sha256",
        "ci_method",
        "bootstrap_replicates_requested",
        "bootstrap_replicates_valid",
        "permutation_replicates_requested",
        "permutation_replicates_valid",
    }
    assert required_provenance.issubset(result.absolute_metrics.columns)
    assert required_provenance.issubset(result.paired_metrics.columns)
    assert {"fit_seed", "outer_split_seed", "inner_cv_seed"}.issubset(
        result.fit_audit.columns
    )
    assert {"outer_fit_seed", "inference_cluster_id"}.issubset(predictions.columns)
    assert set(result.paired_metrics["reference"]) == set(REQUIRED_COMPARATORS) - {
        "locked_attribute_gmnps"
    }
    assert set(result.split_audit["analysis_mode"]) == set(
        predictions["analysis_mode"]
    )
    cohort_audit = result.split_audit.loc[
        result.split_audit["analysis_mode"].eq("cohort_held_out")
    ]
    assert cohort_audit["dropped_n_rows"].eq(0).all()
    crossed = result.split_audit.loc[
        result.split_audit["analysis_mode"].isin(
            ["subject_plus_food_held_out", "subject_plus_meal_held_out"]
        )
    ]
    assert crossed["dropped_n_rows"].gt(0).all()
    assert crossed["dropped_reason"].notna().all()
    missingness = result.missingness_source
    assert {
        "endpoint",
        "cohort",
        "analysis_mode",
        "outer_fold",
        "split",
        "variable",
        "variable_role",
        "n_rows",
        "n_participants",
        "missing_n",
        "missing_rate",
    }.issubset(missingness.columns)
    categorical = missingness.loc[missingness["variable"].eq("diet_category")]
    assert categorical["missing_n"].gt(0).any()
    assert set(missingness["denominator_scope"]) == {
        "split_all_rows",
        "finite_outcome_analysis_set",
    }
    glucose_missingness = missingness.loc[
        missingness["endpoint"].eq("glucose_iAUC_2h")
    ]
    denominator_sizes = glucose_missingness.groupby(
        ["analysis_mode", "outer_fold", "split", "cohort", "variable"]
    )["n_rows"].agg(list)
    assert any(len(set(values)) > 1 for values in denominator_sizes)
    assert glucose_missingness.loc[
        glucose_missingness["denominator_scope"].eq(
            "finite_outcome_analysis_set"
        )
        & glucose_missingness["variable_role"].eq("outcome"),
        "missing_n",
    ].eq(0).all()

    status = result.analysis_status
    assert set(status["analysis_status"]) == {
        "completed",
        "descriptive",
        "not_estimable",
    }
    completed = status.loc[
        status["endpoint"].isin({"glucose_iAUC_2h", "tg_6h_rise"})
    ]
    assert completed.loc[
        completed["analysis_mode"].eq("subject_held_out"), "analysis_status"
    ].eq("completed").all()
    assert completed.loc[
        ~completed["analysis_mode"].eq("subject_held_out"), "analysis_status"
    ].eq("descriptive").all()

    secondary_absolute = result.absolute_metrics.loc[
        ~result.absolute_metrics["analysis_mode"].eq("subject_held_out")
    ]
    secondary_paired = result.paired_metrics.loc[
        ~result.paired_metrics["analysis_mode"].eq("subject_held_out")
    ]
    assert secondary_absolute["inference_status"].eq("descriptive").all()
    assert secondary_absolute[["ci_lower", "ci_upper"]].isna().all().all()
    assert secondary_absolute["bootstrap_replicates_requested"].eq(0).all()
    assert secondary_paired["inference_status"].eq("descriptive").all()
    assert secondary_paired[
        ["ci_lower", "ci_upper", "permutation_p_value", "adjusted_p_value"]
    ].isna().all().all()
    assert secondary_paired["permutation_replicates_requested"].eq(0).all()

    primary = result.paired_metrics.loc[
        result.paired_metrics["test_role"].eq("primary")
    ]
    assert len(primary) == 2
    assert set(primary["endpoint"]) == {"glucose_iAUC_2h", "tg_6h_rise"}
    assert primary["analysis_mode"].eq("subject_held_out").all()
    assert primary["reference"].eq("fcs_microbiome").all()
    assert primary["metric"].eq("rmse").all()
    assert primary["multiplicity_method"].eq("holm").all()
    assert primary["adjusted_p_value"].ge(primary["permutation_p_value"]).all()
    assert set(result.paired_metrics["test_role"]) == {
        "primary",
        "secondary",
        "exploratory",
    }
    assert {
        tuple(json.loads(value)["blocks"])
        for value in primary["reference_information_opportunity"]
    } == {("fcs", "microbiome")}
    assert {
        tuple(json.loads(value)["blocks"])
        for value in primary["comparator_information_opportunity"]
    } == {("locked_attribute_gmnps",)}


def test_nested_fit_audit_never_uses_outer_test_participants(tmp_path, monkeypatch):
    result = _run_testing_only_benchmark(tmp_path, monkeypatch)
    predictors, _ = _testing_only_tables()
    for event in result.fit_audit.itertuples(index=False):
        outer = result.splits[event.analysis_mode][event.outer_fold].outer
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
    assert_frame_equal(first.split_audit, second.split_audit)
    assert_frame_equal(first.missingness_source, second.missingness_source)
    assert_frame_equal(first.analysis_status, second.analysis_status)


def test_primary_inference_clusters_family_twin_components(tmp_path, monkeypatch):
    result = _run_testing_only_benchmark(tmp_path, monkeypatch)
    primary_predictions = result.predictions.loc[
        result.predictions["analysis_mode"].eq("subject_held_out")
    ]
    by_participant = primary_predictions.groupby("participant_id")[
        "inference_cluster_id"
    ].first()
    for participant_number in range(0, 20, 2):
        left = f"testing-p{participant_number:02d}"
        right = f"testing-p{participant_number + 1:02d}"
        assert by_participant[left] == by_participant[right]
    assert result.absolute_metrics.loc[
        result.absolute_metrics["analysis_mode"].eq("subject_held_out"),
        "inference_unit",
    ].eq("family_twin_connected_component").all()


def test_secondary_mode_split_failure_is_structured_and_preserves_primary(
    tmp_path, monkeypatch
):
    real_splitter = benchmark_module.make_nested_group_splits

    def fail_food_mode(*args, **kwargs):
        if kwargs.get("secondary_holdout") == "food_id":
            raise ValueError("testing-only insufficient food groups")
        return real_splitter(*args, **kwargs)

    monkeypatch.setattr(benchmark_module, "make_nested_group_splits", fail_food_mode)
    result = _run_testing_only_benchmark(tmp_path, monkeypatch)

    food_status = result.analysis_status.loc[
        result.analysis_status["analysis_mode"].eq("subject_plus_food_held_out")
    ]
    assert food_status["analysis_status"].eq("not_estimable").all()
    assert food_status["reason"].str.contains("insufficient food groups").all()
    assert not result.predictions["analysis_mode"].eq(
        "subject_plus_food_held_out"
    ).any()
    assert result.predictions["analysis_mode"].eq("subject_held_out").any()


def test_primary_failure_is_reported_only_after_all_four_modes_are_attempted(
    tmp_path, monkeypatch
):
    real_splitter = benchmark_module.make_nested_group_splits
    attempted: list[str | None] = []

    def fail_primary_mode(*args, **kwargs):
        secondary = kwargs.get("secondary_holdout")
        attempted.append(secondary)
        if secondary is None:
            raise ValueError("testing-only primary not estimable")
        return real_splitter(*args, **kwargs)

    monkeypatch.setattr(benchmark_module, "make_nested_group_splits", fail_primary_mode)
    with pytest.raises(ValueError, match="primary analysis is not estimable"):
        _run_testing_only_benchmark(tmp_path, monkeypatch)
    assert attempted == [None, "food_id", "meal_id", "cohort_id"]


def test_predictor_opportunity_missing_an_entire_outcome_row_stays_in_split_universe(
    tmp_path, monkeypatch
):
    predictors, outcomes = _testing_only_tables()
    participant_id = "testing-p05"
    meal_id = "testing-meal-7"
    missing_key = outcomes["participant_id"].eq(participant_id) & outcomes[
        "meal_id"
    ].eq(meal_id)
    outcomes = outcomes.loc[~missing_key].reset_index(drop=True)

    result = _run_testing_only_benchmark(
        tmp_path,
        monkeypatch,
        predictors=predictors,
        outcomes=outcomes,
    )

    target_position = predictors.index[
        predictors["participant_id"].eq(participant_id)
        & predictors["meal_id"].eq(meal_id)
    ].item()
    primary_test_universe = {
        position
        for nested in result.splits["subject_held_out"]
        for position in nested.outer.test_positions
    }
    assert primary_test_universe == set(range(len(predictors)))
    assert target_position in primary_test_universe
    assert not result.predictions["row_id"].eq(
        f"{participant_id}::{meal_id}"
    ).any()

    tg_status = result.analysis_status.loc[
        result.analysis_status["endpoint"].eq("tg_6h_rise")
    ]
    assert tg_status["n_predictor_opportunities"].eq(len(predictors)).all()
    assert tg_status["n_missing_outcome_opportunities"].eq(1).all()
    missingness = result.missingness_source.loc[
        result.missingness_source["endpoint"].eq("tg_6h_rise")
        & result.missingness_source["variable"].eq("tg_6h_rise")
    ]
    assert missingness.loc[
        missingness["denominator_scope"].eq("split_all_rows"), "missing_n"
    ].gt(0).any()
    assert missingness.loc[
        missingness["denominator_scope"].eq("finite_outcome_analysis_set"),
        "missing_n",
    ].eq(0).all()


def test_outcomeless_relative_bridge_still_connects_split_components(
    tmp_path, monkeypatch
):
    predictors, outcomes = _testing_only_tables()
    bridge_participant = "testing-p01"
    predictors.loc[
        predictors["participant_id"].eq(bridge_participant), "twin_id"
    ] = "testing-outcomeless-bridge"
    predictors.loc[
        predictors["participant_id"].eq("testing-p02"), "twin_id"
    ] = "testing-outcomeless-bridge"
    outcomes = outcomes.loc[
        ~outcomes["participant_id"].eq(bridge_participant)
    ].reset_index(drop=True)

    result = _run_testing_only_benchmark(
        tmp_path,
        monkeypatch,
        predictors=predictors,
        outcomes=outcomes,
    )

    participant_fold: dict[str, int] = {}
    for nested in result.splits["subject_held_out"]:
        held_out = predictors.iloc[list(nested.outer.test_positions)]
        for participant in held_out["participant_id"].astype(str).unique():
            participant_fold[participant] = nested.outer.fold_id
    connected = {
        "testing-p00",
        bridge_participant,
        "testing-p02",
        "testing-p03",
    }
    assert len({participant_fold[participant] for participant in connected}) == 1

    observed = result.predictions.loc[
        result.predictions["analysis_mode"].eq("subject_held_out")
        & result.predictions["endpoint"].eq("tg_6h_rise")
        & result.predictions["comparator"].eq("fcs_only")
        & result.predictions["participant_id"].isin(
            {"testing-p00", "testing-p02"}
        )
    ]
    assert observed.groupby("participant_id")["outer_fold"].first().nunique() == 1
    assert observed.groupby("participant_id")[
        "inference_cluster_id"
    ].first().nunique() == 1
    bridge_status = result.analysis_status.loc[
        result.analysis_status["endpoint"].eq("tg_6h_rise")
    ]
    assert bridge_status["n_missing_outcome_opportunities"].eq(12).all()


def test_benchmark_api_requires_trusted_path_contract_not_caller_object():
    parameters = inspect.signature(run_person_meal_benchmark).parameters
    assert "feature_contract" not in parameters
    assert "predictors" not in parameters


def test_task3_loads_exact_bound_predictor_values_from_canonical_bytes(
    tmp_path, monkeypatch
):
    predictors, _ = _testing_only_tables()
    paths, manifest_path, _ = _testing_only_lock_fixture(tmp_path, predictors)
    monkeypatch.setattr(
        benchmark_module, "validate_method_lock_manifest", lambda *args, **kwargs: None
    )

    verified = validate_benchmark_method_lock(
        manifest_path=manifest_path,
        method_lock_paths=paths,
    )

    assert_frame_equal(verified.predictor_frame, predictors)
    assert verified.predictor_frame_sha256 == _sha(paths.predictor_frame)


def test_predictor_value_tamper_and_declared_digest_mismatch_fail_before_outcomes(
    tmp_path, monkeypatch
):
    predictors, _ = _testing_only_tables()
    paths, manifest_path, manifest = _testing_only_lock_fixture(tmp_path, predictors)
    monkeypatch.setattr(
        benchmark_module, "validate_method_lock_manifest", lambda *args, **kwargs: None
    )
    outcome_reads: list[str] = []
    monkeypatch.setattr(
        benchmark_module,
        "load_outcome_table_after_gate",
        lambda *args, **kwargs: outcome_reads.append("outcome-read"),
    )
    payload = json.loads(paths.predictor_frame.read_bytes())
    age_position = payload["columns"].index("age")
    payload["rows"][0][age_position] += 1.0
    paths.predictor_frame.write_bytes(_canonical_bytes(payload))

    with pytest.raises(BenchmarkAccessBlocked, match="predictor|digest|hash|artifact"):
        load_locked_outcomes_for_benchmark(
            "testing-only-controlled-source",
            manifest_path=manifest_path,
            method_lock_paths=paths,
        )
    assert outcome_reads == []

    paths.predictor_frame.write_bytes(_testing_only_predictor_frame_bytes(predictors))
    manifest["predictor_frame_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BenchmarkAccessBlocked, match="predictor|digest|hash|artifact"):
        load_locked_outcomes_for_benchmark(
            "testing-only-controlled-source",
            manifest_path=manifest_path,
            method_lock_paths=paths,
        )
    assert outcome_reads == []


def test_predictor_symlink_and_path_swap_fail_closed(tmp_path, monkeypatch):
    predictors, _ = _testing_only_tables()
    paths, manifest_path, _ = _testing_only_lock_fixture(tmp_path, predictors)
    monkeypatch.setattr(
        benchmark_module, "validate_method_lock_manifest", lambda *args, **kwargs: None
    )
    original = paths.predictor_frame
    target = tmp_path / "predictor-target.json"
    target.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(target)
    with pytest.raises(BenchmarkAccessBlocked, match="predictor|symlink|regular|unreadable"):
        validate_benchmark_method_lock(
            manifest_path=manifest_path,
            method_lock_paths=paths,
        )

    original.unlink()
    original.write_bytes(target.read_bytes())
    replacement = tmp_path / "predictor-replacement.json"
    replacement.write_bytes(target.read_bytes())
    real_open = benchmark_module.os.open

    def swap_after_open(path, flags):
        descriptor = real_open(path, flags)
        if Path(path) == original:
            replacement.replace(original)
        return descriptor

    monkeypatch.setattr(benchmark_module.os, "open", swap_after_open)
    with pytest.raises(BenchmarkAccessBlocked, match="changed|swap|predictor|immutable"):
        validate_benchmark_method_lock(
            manifest_path=manifest_path,
            method_lock_paths=paths,
        )


def _assert_reserved_collision_fails_before_outcome_access(
    tmp_path,
    monkeypatch,
    *,
    old_name: str,
    reserved_name: str,
) -> None:
    paths, manifest_path, manifest = _testing_only_lock_fixture(tmp_path)
    _rename_bound_predictor_column(
        paths,
        manifest_path,
        manifest,
        old_name=old_name,
        new_name=reserved_name,
    )

    with pytest.raises(MethodLockError, match="reserved|internal|column"):
        validate_predictor_artifact_binding(
            paths.feature_contract.read_bytes(),
            paths.predictor_frame.read_bytes(),
        )

    outcome_reads: list[str] = []
    monkeypatch.setattr(
        benchmark_module,
        "validate_method_lock_manifest",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        benchmark_module,
        "load_outcome_table_after_gate",
        lambda *args, **kwargs: outcome_reads.append("outcome-read"),
    )
    with pytest.raises(BenchmarkAccessBlocked, match="reserved|contract|binding"):
        load_locked_outcomes_for_benchmark(
            "testing-only-controlled-source",
            manifest_path=manifest_path,
            method_lock_paths=paths,
        )
    assert outcome_reads == []


def test_join_internal_column_collision_fails_before_outcome_access(
    tmp_path, monkeypatch
):
    _assert_reserved_collision_fails_before_outcome_access(
        tmp_path,
        monkeypatch,
        old_name="age",
        reserved_name="__outcome_row_present__",
    )


def test_inference_internal_column_collision_fails_before_outcome_access(
    tmp_path, monkeypatch
):
    _assert_reserved_collision_fails_before_outcome_access(
        tmp_path,
        monkeypatch,
        old_name="microbiome_1",
        reserved_name="inference_cluster_id",
    )


def test_categorical_preprocessing_has_explicit_missing_indicator():
    frame = pd.DataFrame(
        {"numeric": [1.0, 2.0, 3.0], "category": ["a", np.nan, "b"]}
    )
    pipeline = benchmark_module._pipeline(frame, 1.0)
    pipeline.fit(frame, np.array([0.0, 1.0, 2.0]))
    preprocess = pipeline.named_steps["preprocess"]
    transformed = preprocess.transform(frame)
    indicator_slice = preprocess.output_indices_["categorical_missing_indicator"]
    assert transformed[:, indicator_slice].ravel().tolist() == [0.0, 1.0, 0.0]


def test_shuffled_mapping_is_deranged_and_too_few_units_are_not_estimable():
    frame = pd.DataFrame(
        {
            "food_id": ["u1", "u2", "u3"],
            "feature": [1.0, 2.0, 3.0],
        }
    )
    shuffled, _ = benchmark_module._shuffled_mapping(
        frame,
        np.array([0, 1, 2]),
        np.array([0, 1, 2]),
        ("feature",),
        "food_id",
        seed=16180,
    )
    assert shuffled["feature"].to_list() != frame["feature"].to_list()
    assert all(
        observed != original
        for observed, original in zip(shuffled["feature"], frame["feature"])
    )
    with pytest.raises(ValueError, match="not estimable|fewer than two"):
        benchmark_module._shuffled_mapping(
            frame.iloc[[0]],
            np.array([0]),
            np.array([0]),
            ("feature",),
            "food_id",
            seed=16180,
        )


def test_exact_contract_rejects_unknown_predictor_columns(tmp_path, monkeypatch):
    predictors, _ = _testing_only_tables()
    predictors["caller_unknown"] = 0.0
    paths, manifest_path, _ = _testing_only_lock_fixture(tmp_path)
    monkeypatch.setattr(
        benchmark_module, "validate_method_lock_manifest", lambda *args, **kwargs: None
    )
    lock = validate_benchmark_method_lock(
        manifest_path=manifest_path,
        method_lock_paths=paths,
    )
    with pytest.raises(ValueError, match="unknown|exact|contract"):
        validate_feature_contract(predictors, lock.feature_contract)


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("age", "not-a-number", "numeric|number"),
        ("participant_id", 123, "string"),
        ("diet_category", 123, "category"),
        ("diet_flag", "true", "boolean"),
    ],
)
def test_contract_validates_every_declared_data_type(
    tmp_path, monkeypatch, column, bad_value, message
):
    predictors, _ = _testing_only_tables()
    paths, manifest_path, _ = _testing_only_lock_fixture(tmp_path, predictors)
    monkeypatch.setattr(
        benchmark_module, "validate_method_lock_manifest", lambda *args, **kwargs: None
    )
    lock = validate_benchmark_method_lock(
        manifest_path=manifest_path,
        method_lock_paths=paths,
    )
    predictors[column] = predictors[column].astype(object)
    predictors.loc[predictors.index[0], column] = bad_value
    with pytest.raises(ValueError, match=message):
        validate_feature_contract(predictors, lock.feature_contract)


@pytest.mark.parametrize("mutation", ["wrong_mapping_column", "second_mapping_role"])
def test_contract_requires_food_id_as_the_single_mapping_unit(
    tmp_path, monkeypatch, mutation
):
    predictors, _ = _testing_only_tables()
    paths, manifest_path, manifest = _testing_only_lock_fixture(tmp_path, predictors)
    payload = json.loads(paths.feature_contract.read_bytes())
    if mutation == "wrong_mapping_column":
        payload["mapping_unit"]["column"] = "meal_id"
    else:
        next(
            column for column in payload["columns"] if column["name"] == "meal_id"
        )["role"] = "mapping_unit"
    paths.feature_contract.write_bytes(_canonical_bytes(payload))
    manifest["feature_contract_sha256"] = _sha(paths.feature_contract)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        benchmark_module, "validate_method_lock_manifest", lambda *args, **kwargs: None
    )
    with pytest.raises(
        (BenchmarkAccessBlocked, ValueError), match="mapping|food_id|feature contract"
    ):
        validate_benchmark_method_lock(
            manifest_path=manifest_path,
            method_lock_paths=paths,
        )


@pytest.mark.parametrize("mutation", ["identifier_in_block", "duplicate_alias", "illegal_overlap"])
def test_contract_rejects_nonpredictor_roles_aliases_and_overlap(
    tmp_path, monkeypatch, mutation
):
    predictors, _ = _testing_only_tables()
    paths, manifest_path, manifest = _testing_only_lock_fixture(tmp_path)
    payload = json.loads(paths.feature_contract.read_bytes())
    if mutation == "identifier_in_block":
        payload["feature_blocks"]["fcs"]["columns"] = ["participant_id"]
    elif mutation == "duplicate_alias":
        payload["columns"].append(dict(payload["columns"][0]))
    else:
        payload["feature_blocks"]["fcs"]["columns"].append("microbiome_1")
    paths.feature_contract.write_bytes(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    manifest["feature_contract_sha256"] = _sha(paths.feature_contract)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        benchmark_module, "validate_method_lock_manifest", lambda *args, **kwargs: None
    )
    with pytest.raises(
        (ValueError, BenchmarkAccessBlocked),
        match="role|duplicate|overlap|contract",
    ):
        lock = validate_benchmark_method_lock(
            manifest_path=manifest_path,
            method_lock_paths=paths,
        )
        validate_feature_contract(predictors, lock.feature_contract)


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
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 2.0, 2.0]),
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
    with pytest.raises(ValueError, match="finite"):
        compute_regression_metrics(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


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


@pytest.mark.parametrize("resampler", ["bootstrap", "permutation"])
def test_low_valid_resampling_fraction_fails_closed_for_preregistered_primary(
    tmp_path, monkeypatch, resampler
):
    if resampler == "bootstrap":
        real_resampler = benchmark_module.participant_bootstrap_ci

        def low_valid(*args, **kwargs):
            return replace(real_resampler(*args, **kwargs), valid_replicates=0)

        monkeypatch.setattr(benchmark_module, "participant_bootstrap_ci", low_valid)
    else:
        real_resampler = benchmark_module.participant_permutation_test

        def low_valid(*args, **kwargs):
            return replace(real_resampler(*args, **kwargs), valid_replicates=0)

        monkeypatch.setattr(
            benchmark_module, "participant_permutation_test", low_valid
        )
    with pytest.raises(ValueError, match="primary.*not estimable|valid.*fraction"):
        _run_testing_only_benchmark(tmp_path, monkeypatch)
