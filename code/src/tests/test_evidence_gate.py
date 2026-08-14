from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gmnps.data_sources.predict_zoe_loader import LoadedTable
from gmnps.validation import claim_policy as policy_module
from gmnps.validation import evidence_gate as gate_module
from gmnps.validation.claim_policy import (
    build_claim_policy_from_evidence_gate,
    check_claim_inputs,
)
from gmnps.validation.cohort_split import (
    family_twin_component_ids,
    make_nested_group_splits,
)
from gmnps.validation.evidence_gate import (
    COMPUTATIONAL_FEASIBILITY,
    DIRECT_EXTERNAL_VALIDITY,
    TESTING_ONLY_NO_CLAIM_UPGRADE,
    EvidenceGateArtifactPaths,
    EvidenceGateOutcome,
    VerifiedRunBinding,
    evaluate_evidence_gate,
    evaluate_testing_evidence,
    export_benchmark_result,
)
from gmnps.validation.method_lock_gate import MethodLockArtifactPaths
from gmnps.validation.person_meal_benchmark import (
    BootstrapInterval,
    PermutationTestResult,
    REQUIRED_COMPARATORS,
    BenchmarkResult,
)
import gmnps.validation.person_meal_benchmark as benchmark_module
from tests.test_person_meal_benchmark import (
    _testing_only_lock_fixture as _task3_lock_fixture,
    _testing_only_tables as _task3_tables,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "code/src/configs/person_meal_validation.yaml"
SOURCE_ID = "predict_controlled_clinical_zenodo"
TEST_SOURCE_ID = "testing-only-controlled-source"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def real_task3_roundtrip(tmp_path_factory):
    """Run the actual Task 3 producer once, then export and gate its result."""

    base = tmp_path_factory.mktemp("real-task3-producer")
    monkeypatch = pytest.MonkeyPatch()
    predictors, _ = _task3_tables()
    outcomes = predictors[["participant_id", "meal_id"]].copy()
    outcomes["glucose_iAUC_2h"] = (
        0.15 * predictors["gmnps_attribute_1"].to_numpy()
        + 1.2 * predictors["gmnps_attribute_2"].to_numpy()
    )
    outcomes["tg_6h_rise"] = (
        -0.05 * predictors["gmnps_attribute_1"].to_numpy()
        + 0.8 * predictors["gmnps_attribute_2"].to_numpy()
    )
    outcomes.loc[0, "glucose_iAUC_2h"] = np.nan
    lock_paths, manifest_path, _ = _task3_lock_fixture(base / "lock", predictors)
    monkeypatch.setattr(
        benchmark_module,
        "validate_method_lock_manifest",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        benchmark_module,
        "load_outcome_table_after_gate",
        lambda *args, **kwargs: LoadedTable(
            frame=outcomes.copy(),
            sha256="1" * 64,
            source_id=TEST_SOURCE_ID,
        ),
    )
    monkeypatch.setattr(benchmark_module, "_ALPHA_GRID", (1.0,))

    def fast_bootstrap(
        y,
        prediction,
        clusters,
        *,
        metric,
        n_bootstrap,
        seed,
        reference_prediction=None,
    ):
        if reference_prediction is None:
            estimate = benchmark_module.compute_regression_metrics(y, prediction)[metric]
            lower, upper = estimate - 0.1, estimate + 0.1
        else:
            estimate = benchmark_module._metric_delta(
                y, prediction, reference_prediction, metric
            )
            lower, upper = (
                (estimate - 0.2, min(-1e-6, estimate / 2.0))
                if estimate < 0
                else (estimate - 0.1, estimate + 0.1)
            )
        return BootstrapInterval(
            estimate=estimate,
            lower=lower,
            upper=upper,
            n_clusters=len(set(clusters)),
            requested_replicates=2_000,
            valid_replicates=1_900,
        )

    def fast_permutation(
        y,
        model_prediction,
        reference_prediction,
        clusters,
        *,
        metric,
        n_permutations,
        seed,
    ):
        estimate = benchmark_module._metric_delta(
            y, model_prediction, reference_prediction, metric
        )
        return PermutationTestResult(
            estimate_delta=estimate,
            p_value=0.01,
            null_distribution=(0.0,),
            n_clusters=len(set(clusters)),
            requested_replicates=2_000,
            valid_replicates=1_900,
        )

    monkeypatch.setattr(benchmark_module, "participant_bootstrap_ci", fast_bootstrap)
    monkeypatch.setattr(
        benchmark_module, "participant_permutation_test", fast_permutation
    )
    monkeypatch.setattr(gate_module, "participant_bootstrap_ci", fast_bootstrap)
    monkeypatch.setattr(gate_module, "participant_permutation_test", fast_permutation)
    original_loader = benchmark_module.load_locked_outcomes_for_benchmark
    captured: dict[str, object] = {}

    def recording_loader(*args, **kwargs):
        locked = original_loader(*args, **kwargs)
        captured["locked"] = locked
        return locked

    monkeypatch.setattr(
        benchmark_module, "load_locked_outcomes_for_benchmark", recording_loader
    )
    result = benchmark_module.run_person_meal_benchmark(
        TEST_SOURCE_ID,
        manifest_path=manifest_path,
        method_lock_paths=lock_paths,
    )
    locked = captured["locked"]
    binding = VerifiedRunBinding.from_locked_outcome("task3-producer-run", locked)
    exported = export_benchmark_result(result, binding, base / "export")
    registry_path = base / "trusted-result-registry.json"
    _write_json(
        registry_path,
        {
            "schema_version": "direct-validity-result-registry-v1",
            "approved_runs": [
                {
                    "run_id": binding.run_id,
                    "result_run_manifest_sha256": _sha(
                        exported.result_run_manifest
                    ),
                }
            ],
        },
    )
    monkeypatch.setattr(gate_module, "_TRUSTED_RESULT_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(
        gate_module,
        "load_locked_outcomes_for_benchmark",
        lambda *args, **kwargs: locked,
    )
    dummy = base / "unused-lock-artifact"
    dummy.write_bytes(b"unused\n")
    formal_lock_paths = MethodLockArtifactPaths(
        **{
            field.name: getattr(lock_paths, field.name, dummy)
            for field in fields(MethodLockArtifactPaths)
        }
    )
    paths = EvidenceGateArtifactPaths(
        method_lock_manifest=manifest_path,
        method_lock_artifacts=formal_lock_paths,
        result_run_manifest=exported.result_run_manifest,
        paired_metrics=exported.paired_metrics,
        predictions=exported.predictions,
        split_audit=exported.split_audit,
        analysis_status=exported.analysis_status,
        controlled_outcome_source_id=TEST_SOURCE_ID,
    )
    try:
        yield {
            "result": result,
            "exported": exported,
            "paths": paths,
            "locked": locked,
        }
    finally:
        monkeypatch.undo()


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictors: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    for person in range(10):
        participant = f"p{person:02d}"
        for meal in range(2):
            meal_id = f"m{meal}"
            predictors.append(
                {
                    "participant_id": participant,
                    "meal_id": meal_id,
                    "food_id": f"f{meal}",
                    "family_id": None,
                    "twin_id": None,
                    "cohort_id": "c0",
                    "feature": float(person + meal),
                }
            )
            # p09 deliberately has predictor opportunities but no outcome rows.
            if person < 9:
                outcomes.append(
                    {
                        "participant_id": participant,
                        "meal_id": meal_id,
                        "glucose_iAUC_2h": float(person + meal + 1),
                        "tg_6h_rise": float(person - meal + 2),
                    }
                )
    outcome = pd.DataFrame(outcomes)
    outcome.loc[0, "glucose_iAUC_2h"] = np.nan
    return pd.DataFrame(predictors), outcome


def _benchmark_result(
    predictors: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: dict[str, object],
    binding: VerifiedRunBinding,
) -> BenchmarkResult:
    split = config["split"]
    seeds = config["seeds"]
    nested = make_nested_group_splits(
        predictors,
        outer_folds=int(split["outer_folds"]),
        inner_folds=int(split["inner_folds"]),
        outer_seed=int(seeds["outer_split"]),
        inner_seed=int(seeds["inner_cv"]),
        secondary_holdout=None,
    )
    working = predictors.copy()
    working.insert(
        0,
        "row_id",
        working["participant_id"].astype(str)
        + "::"
        + working["meal_id"].astype(str),
    )
    working["inference_cluster_id"] = family_twin_component_ids(working)
    joined = working.merge(
        outcomes,
        on=["participant_id", "meal_id"],
        how="left",
        validate="one_to_one",
    )
    prediction_rows: list[dict[str, object]] = []
    for endpoint in ("glucose_iAUC_2h", "tg_6h_rise"):
        for fold in nested:
            test = joined.iloc[list(fold.outer.test_positions)]
            test = test.loc[np.isfinite(pd.to_numeric(test[endpoint], errors="coerce"))]
            for comparator in REQUIRED_COMPARATORS:
                for row in test.itertuples(index=False):
                    prediction_rows.append(
                        {
                            "analysis_mode": "subject_held_out",
                            "row_id": row.row_id,
                            "participant_id": row.participant_id,
                            "meal_id": row.meal_id,
                            "inference_cluster_id": row.inference_cluster_id,
                            "endpoint": endpoint,
                            "outer_fold": fold.outer.fold_id,
                            "comparator": comparator,
                            "y_true": float(getattr(row, endpoint)),
                            "y_pred": float(
                                getattr(row, endpoint)
                                + (
                                    0.1
                                    if comparator == "locked_attribute_gmnps"
                                    else 1.0
                                    if comparator == "fcs_microbiome"
                                    else 0.5
                                )
                            ),
                            "selected_alpha": 1.0,
                            "outer_fit_seed": 100 + fold.outer.fold_id,
                            "outer_split_seed": int(seeds["outer_split"]),
                            "inner_cv_seed": int(seeds["inner_cv"]),
                        }
                    )
    provenance = {
        "manifest_sha256": binding.method_lock_manifest_sha256,
        "outcome_source_id": binding.outcome_source_id,
        "outcome_source_sha256": binding.outcome_sha256,
        "predictor_frame_sha256": binding.predictor_frame_sha256,
        "feature_contract_sha256": binding.feature_contract_sha256,
        "method_lock_gate_implementation_sha256": binding.method_lock_gate_implementation_sha256,
        "cohort_split_implementation_sha256": binding.cohort_split_implementation_sha256,
        "person_meal_benchmark_implementation_sha256": binding.person_meal_benchmark_implementation_sha256,
        "benchmark_specification_sha256": binding.benchmark_specification_sha256,
        "validation_config_sha256": binding.person_meal_validation_config_sha256,
    }
    predictions = pd.DataFrame(prediction_rows)
    paired_records = []
    for endpoint_index, endpoint in enumerate(("glucose_iAUC_2h", "tg_6h_rise")):
        model = predictions.loc[
            predictions["endpoint"].eq(endpoint)
            & predictions["comparator"].eq("locked_attribute_gmnps")
        ].sort_values("row_id")
        reference = predictions.loc[
            predictions["endpoint"].eq(endpoint)
            & predictions["comparator"].eq("fcs_microbiome")
        ].sort_values("row_id")
        y = model["y_true"].to_numpy(dtype=float)
        model_prediction = model["y_pred"].to_numpy(dtype=float)
        reference_prediction = reference["y_pred"].to_numpy(dtype=float)
        clusters = model["inference_cluster_id"].astype(str).to_numpy()
        bootstrap_seed = int(seeds["bootstrap"]) + endpoint_index * 1_000_003 + 110_078
        permutation_seed = int(seeds["permutation"]) + endpoint_index * 1_000_003 + 110_078
        interval = gate_module.participant_bootstrap_ci(
            y,
            model_prediction,
            clusters,
            metric="rmse",
            n_bootstrap=2_000,
            seed=bootstrap_seed,
            reference_prediction=reference_prediction,
        )
        permutation = gate_module.participant_permutation_test(
            y,
            model_prediction,
            reference_prediction,
            clusters,
            metric="rmse",
            n_permutations=2_000,
            seed=permutation_seed,
        )
        paired_records.append(
            {
                **provenance,
                "endpoint": endpoint,
                "analysis_mode": "subject_held_out",
                "analysis_status": "completed",
                "comparator": "locked_attribute_gmnps",
                "reference": "fcs_microbiome",
                "metric": "rmse",
                "estimate_delta": interval.estimate,
                "ci_lower": interval.lower,
                "ci_upper": interval.upper,
                "ci_method": "paired_family_twin_component_cluster_percentile_bootstrap_95",
                "bootstrap_seed": bootstrap_seed,
                "permutation_seed": permutation_seed,
                "bootstrap_replicates_requested": interval.requested_replicates,
                "bootstrap_replicates_valid": interval.valid_replicates,
                "bootstrap_minimum_valid_fraction": 0.9,
                "bootstrap_valid_fraction": interval.valid_replicates / 2_000,
                "permutation_replicates_requested": permutation.requested_replicates,
                "permutation_replicates_valid": permutation.valid_replicates,
                "permutation_minimum_valid_fraction": 0.9,
                "permutation_valid_fraction": permutation.valid_replicates / 2_000,
                "permutation_p_value": permutation.p_value,
                "adjusted_p_value": np.nan,
                "test_role": "primary",
                "correction_family": "two_primary_endpoints_locked_gmnps_vs_fcs_microbiome_rmse",
                "multiplicity_method": "holm",
                "inference_status": "completed",
            }
        )
    paired = pd.DataFrame(paired_records)
    order = np.argsort(paired["permutation_p_value"].to_numpy(), kind="stable")
    running = 0.0
    for rank, position in enumerate(order):
        running = max(
            running,
            min(1.0, float(paired.loc[position, "permutation_p_value"]) * (2 - rank)),
        )
        paired.loc[position, "adjusted_p_value"] = running
    split_audit = pd.DataFrame(
        [
            {
                **provenance,
                "analysis_mode": "subject_held_out",
                "analysis_role": "primary",
                "secondary_unit": None,
                "analysis_status": "completed",
                "reason": None,
                "outer_fold": fold.outer.fold_id,
                "train_n_rows": len(fold.outer.train_positions),
                "test_n_rows": len(fold.outer.test_positions),
                "dropped_n_rows": len(fold.outer.dropped_positions),
                "dropped_n_participants": 0,
                "dropped_n_person_meals": 0,
                "dropped_row_ids": (),
                "dropped_reason": None,
                "outer_split_seed": int(seeds["outer_split"]),
                "inner_cv_seed": int(seeds["inner_cv"]),
            }
            for fold in nested
        ]
    )
    status = pd.DataFrame(
        [
            {
                **provenance,
                "analysis_mode": "subject_held_out",
                "endpoint": endpoint,
                "endpoint_role": "primary",
                "analysis_status": "completed",
                "reason": None,
                "n_prediction_rows": int(
                    pd.DataFrame(prediction_rows)["endpoint"].eq(endpoint).sum()
                ),
                "analysis_role": "primary",
                "outer_split_seed": int(seeds["outer_split"]),
                "inner_cv_seed": int(seeds["inner_cv"]),
            }
            for endpoint in ("glucose_iAUC_2h", "tg_6h_rise")
        ]
    )
    return BenchmarkResult(
        predictions=predictions,
        absolute_metrics=pd.DataFrame(),
        paired_metrics=paired,
        fit_audit=pd.DataFrame(),
        splits={"subject_held_out": nested},
        split_audit=split_audit,
        missingness_source=pd.DataFrame(),
        analysis_status=status,
    )


def _production_fixture(tmp_path: Path, monkeypatch, *, actual_resampling: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_payload = json.loads(CONFIG_PATH.read_text())
    predictors, outcomes = _tables()
    method_lock_path = tmp_path / "method-lock.json"
    method_lock_path.write_bytes(b"trusted method lock fixture\n")
    lock_sha = _sha(method_lock_path)
    outcome_sha = "d" * 64
    lock_manifest = {
        "person_meal_validation_config_sha256": "1" * 64,
        "feature_contract_sha256": "2" * 64,
        "predictor_frame_sha256": "3" * 64,
        "cohort_split_implementation_sha256": "4" * 64,
        "person_meal_benchmark_implementation_sha256": "5" * 64,
        "benchmark_specification_sha256": "6" * 64,
        "method_lock_gate_implementation_sha256": "7" * 64,
    }
    loaded = SimpleNamespace(
        outcome=SimpleNamespace(frame=outcomes, sha256=outcome_sha, source_id=SOURCE_ID),
        verified_lock=SimpleNamespace(
            manifest=lock_manifest,
            manifest_sha256=lock_sha,
            predictor_frame=predictors,
            config=SimpleNamespace(
                payload=config_payload,
                sha256=lock_manifest["person_meal_validation_config_sha256"],
            ),
        ),
    )
    monkeypatch.setattr(
        gate_module,
        "load_locked_outcomes_for_benchmark",
        lambda *args, **kwargs: loaded,
    )
    if not actual_resampling:
        def fast_bootstrap(
            y,
            prediction,
            clusters,
            *,
            metric,
            n_bootstrap,
            seed,
            reference_prediction=None,
        ):
            model_rmse = float(np.sqrt(np.mean((y - prediction) ** 2)))
            reference_rmse = float(
                np.sqrt(np.mean((y - reference_prediction) ** 2))
            )
            estimate = model_rmse - reference_rmse
            return BootstrapInterval(
                estimate=estimate,
                lower=estimate - 0.05,
                upper=estimate + 0.05,
                n_clusters=len(set(clusters)),
                requested_replicates=n_bootstrap,
                valid_replicates=n_bootstrap,
            )

        def fast_permutation(
            y,
            model_prediction,
            reference_prediction,
            clusters,
            *,
            metric,
            n_permutations,
            seed,
        ):
            model_rmse = float(np.sqrt(np.mean((y - model_prediction) ** 2)))
            reference_rmse = float(
                np.sqrt(np.mean((y - reference_prediction) ** 2))
            )
            estimate = model_rmse - reference_rmse
            return PermutationTestResult(
                estimate_delta=estimate,
                p_value=0.001,
                null_distribution=(0.0,),
                n_clusters=len(set(clusters)),
                requested_replicates=n_permutations,
                valid_replicates=n_permutations,
            )

        monkeypatch.setattr(gate_module, "participant_bootstrap_ci", fast_bootstrap)
        monkeypatch.setattr(gate_module, "participant_permutation_test", fast_permutation)
    binding = VerifiedRunBinding.from_locked_outcome("direct-run-001", loaded)
    result = _benchmark_result(predictors, outcomes, config_payload, binding)
    exported = export_benchmark_result(result, binding, tmp_path / "export")
    registry_path = tmp_path / "trusted-registry.json"
    _write_json(
        registry_path,
        {
            "schema_version": "direct-validity-result-registry-v1",
            "approved_runs": [
                {
                    "run_id": binding.run_id,
                    "result_run_manifest_sha256": _sha(exported.result_run_manifest),
                }
            ],
        },
    )
    monkeypatch.setattr(gate_module, "_TRUSTED_RESULT_REGISTRY_PATH", registry_path)
    dummy = tmp_path / "dummy"
    dummy.write_bytes(b"x")
    lock_paths = MethodLockArtifactPaths(
        **{field.name: dummy for field in fields(MethodLockArtifactPaths)}
    )
    paths = EvidenceGateArtifactPaths(
        method_lock_manifest=method_lock_path,
        method_lock_artifacts=lock_paths,
        result_run_manifest=exported.result_run_manifest,
        paired_metrics=exported.paired_metrics,
        predictions=exported.predictions,
        split_audit=exported.split_audit,
        analysis_status=exported.analysis_status,
        controlled_outcome_source_id=SOURCE_ID,
    )
    return paths, exported, loaded, registry_path


def _rehash(paths: EvidenceGateArtifactPaths, registry_path: Path) -> None:
    manifest = json.loads(paths.result_run_manifest.read_text())
    for name, path in (
        ("paired_metrics", paths.paired_metrics),
        ("predictions", paths.predictions),
        ("split_audit", paths.split_audit),
        ("analysis_status", paths.analysis_status),
    ):
        manifest["artifacts"][name]["sha256"] = _sha(path)
    _write_json(paths.result_run_manifest, manifest)
    registry = json.loads(registry_path.read_text())
    registry["approved_runs"][0]["result_run_manifest_sha256"] = _sha(
        paths.result_run_manifest
    )
    _write_json(registry_path, registry)


def _make_primary_predictions_identical(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    reference = result.loc[
        result["comparator"].eq("fcs_microbiome")
    ].set_index(["endpoint", "row_id"])["y_pred"]
    model = result["comparator"].eq("locked_attribute_gmnps")
    keys = pd.MultiIndex.from_frame(result.loc[model, ["endpoint", "row_id"]])
    result.loc[model, "y_pred"] = reference.reindex(keys).to_numpy()
    return result


def _corrupt_primary_y_true(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    index = result.index[result["comparator"].eq("locked_attribute_gmnps")][0]
    result.loc[index, "y_true"] = float(result.loc[index, "y_true"]) + 9.0
    return result


def _make_primary_y_pred_nonfinite(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    index = result.index[result["comparator"].eq("locked_attribute_gmnps")][0]
    result.loc[index, "y_pred"] = np.inf
    return result


def test_actual_task3_producer_exporter_and_path_gate_roundtrip(real_task3_roundtrip):
    result = real_task3_roundtrip["result"]
    paths = real_task3_roundtrip["paths"]
    exported = real_task3_roundtrip["exported"]
    assert isinstance(result, BenchmarkResult)
    assert isinstance(result.analysis_status, pd.DataFrame)
    assert set(result.splits) == {
        "subject_held_out",
        "subject_plus_food_held_out",
        "subject_plus_meal_held_out",
        "cohort_held_out",
    }
    assert len(result.splits["subject_held_out"]) == 5
    assert {
        "analysis_mode",
        "row_id",
        "participant_id",
        "meal_id",
        "inference_cluster_id",
        "endpoint",
        "outer_fold",
        "comparator",
        "y_true",
        "y_pred",
        "selected_alpha",
        "outer_fit_seed",
    }.issubset(result.predictions.columns)
    paired = pd.read_csv(exported.paired_metrics)
    status = pd.read_csv(exported.analysis_status)
    split = pd.read_csv(exported.split_audit)
    manifest = json.loads(exported.result_run_manifest.read_text())
    assert "comparator" in paired and "adjusted_p_value" in paired
    assert "model" not in paired and "holm_adjusted_p_value" not in paired
    primary_status = status.loc[
        status["analysis_mode"].eq("subject_held_out")
        & status["endpoint"].isin(["glucose_iAUC_2h", "tg_6h_rise"])
    ]
    assert set(primary_status["analysis_status"]) == {"completed"}
    primary_split = split.loc[split["analysis_mode"].eq("subject_held_out")]
    assert set(primary_split["outer_fold"]) == set(range(5))
    assert "predictions" in manifest["artifacts"]
    outcome = evaluate_evidence_gate(paths)
    assert outcome.tier == DIRECT_EXTERNAL_VALIDITY, outcome.blockers
    assert outcome.passed is True


def test_small_gate_fixture_runs_actual_task3_2000_repeat_algorithms(
    tmp_path, monkeypatch
):
    paths, _, _, _ = _production_fixture(
        tmp_path, monkeypatch, actual_resampling=True
    )
    outcome = evaluate_evidence_gate(paths)
    assert outcome.passed is True
    assert len(outcome.endpoint_checks) == 2
    assert outcome.endpoint_checks["valid_replicates_passed"].all()


def test_production_api_is_path_only_and_current_state_fails_closed():
    with pytest.raises(TypeError):
        evaluate_evidence_gate({})
    outcome = evaluate_evidence_gate()
    assert outcome.tier == COMPUTATIONAL_FEASIBILITY
    assert "controlled_not_granted" in " ".join(outcome.blockers)


@pytest.mark.parametrize(
    "column,value,fragment",
    [
        ("permutation_p_value", np.nan, "raw permutation p-value"),
        ("permutation_p_value", -0.01, "raw permutation p-value"),
        ("permutation_p_value", 1.01, "raw permutation p-value"),
        ("ci_lower", np.nan, "confidence interval"),
        ("ci_lower", 0.0, "confidence interval"),
        ("bootstrap_replicates_requested", True, "strict non-boolean integer"),
        ("bootstrap_replicates_requested", 2000.5, "strict non-boolean integer"),
        ("bootstrap_replicates_valid", -1, "nonnegative"),
        ("bootstrap_replicates_valid", 2001, "cannot exceed requested"),
        ("permutation_replicates_valid", 1799, "valid-resample threshold"),
    ],
)
def test_strict_primary_numeric_domains_fail_closed(
    tmp_path, monkeypatch, column, value, fragment
):
    paths, _, _, registry = _production_fixture(tmp_path, monkeypatch)
    paired = pd.read_csv(paths.paired_metrics)
    if isinstance(value, bool) or (
        isinstance(value, float) and not float(value).is_integer()
    ):
        paired[column] = paired[column].astype(object)
    paired.loc[0, column] = value
    paired.to_csv(paths.paired_metrics, index=False)
    _rehash(paths, registry)
    outcome = evaluate_evidence_gate(paths)
    assert any(fragment in blocker for blocker in outcome.blockers)


def test_holm_is_recomputed_from_task3_raw_p_and_compared_to_adjusted(
    tmp_path, monkeypatch
):
    paths, _, _, registry = _production_fixture(tmp_path, monkeypatch)
    paired = pd.read_csv(paths.paired_metrics)
    paired.loc[1, "adjusted_p_value"] = 0.001
    paired.to_csv(paths.paired_metrics, index=False)
    _rehash(paths, registry)
    outcome = evaluate_evidence_gate(paths)
    assert any("recomputed Holm" in blocker for blocker in outcome.blockers)


@pytest.mark.parametrize(
    "artifact,mutator,fragment",
    [
        (
            "predictions",
            _make_primary_predictions_identical,
            "recomputed RMSE delta",
        ),
        (
            "predictions",
            _corrupt_primary_y_true,
            "trusted outcome",
        ),
        (
            "predictions",
            _make_primary_y_pred_nonfinite,
            "finite y_pred",
        ),
        (
            "paired_metrics",
            lambda frame: frame.assign(
                bootstrap_seed=frame["bootstrap_seed"].where(
                    frame.index != 0, frame.loc[0, "bootstrap_seed"] + 1
                )
            ),
            "bootstrap seed",
        ),
    ],
)
def test_primary_statistics_are_recomputed_from_trusted_outcomes_and_predictions(
    tmp_path, monkeypatch, artifact, mutator, fragment
):
    paths, _, _, registry = _production_fixture(tmp_path, monkeypatch)
    path = getattr(paths, artifact)
    frame = mutator(pd.read_csv(path))
    frame.to_csv(path, index=False)
    _rehash(paths, registry)
    outcome = evaluate_evidence_gate(paths)
    assert outcome.passed is False
    assert any(fragment in blocker for blocker in outcome.blockers)


@pytest.mark.parametrize(
    "artifact,mutator,fragment",
    [
        (
            "paired_metrics",
            lambda frame: frame.assign(
                manifest_sha256=frame["manifest_sha256"].where(
                    frame.index != 0, "0" * 64
                )
            ),
            "Task 3 provenance manifest_sha256",
        ),
        (
            "predictions",
            lambda frame: frame.assign(
                outer_fold=frame["outer_fold"].where(frame.index != 0, 4)
            ),
            "recomputed test fold",
        ),
        (
            "predictions",
            lambda frame: frame.assign(
                inference_cluster_id=frame["inference_cluster_id"].where(
                    frame.index != 0, "forged-component"
                )
            ),
            "recomputed family/twin component",
        ),
        (
            "split_audit",
            lambda frame: frame.assign(
                test_n_rows=frame["test_n_rows"].where(frame.index != 0, 999)
            ),
            "split summary does not match recomputed",
        ),
    ],
)
def test_predictions_and_split_summary_are_checked_against_recomputed_production_splits(
    tmp_path, monkeypatch, artifact, mutator, fragment
):
    paths, _, _, registry = _production_fixture(tmp_path, monkeypatch)
    path = getattr(paths, artifact)
    frame = mutator(pd.read_csv(path))
    frame.to_csv(path, index=False)
    _rehash(paths, registry)
    outcome = evaluate_evidence_gate(paths)
    assert any(fragment in blocker for blocker in outcome.blockers)


def test_missing_outcomes_and_predictor_only_participants_are_allowed(tmp_path, monkeypatch):
    paths, _, loaded, _ = _production_fixture(tmp_path, monkeypatch)
    assert "p09" in set(loaded.verified_lock.predictor_frame["participant_id"])
    assert "p09" not in set(loaded.outcome.frame["participant_id"])
    assert loaded.outcome.frame["glucose_iAUC_2h"].isna().any()
    assert evaluate_evidence_gate(paths).passed is True


def test_trusted_predictor_relationships_not_uploaded_component_labels_define_splits(
    tmp_path, monkeypatch
):
    paths, _, loaded, _ = _production_fixture(tmp_path, monkeypatch)
    predictors = loaded.verified_lock.predictor_frame
    predictors.loc[predictors["participant_id"].isin(["p00", "p09"]), "family_id"] = (
        "trusted-shared-family"
    )
    outcome = evaluate_evidence_gate(paths)
    assert outcome.passed is False
    assert any(
        "recomputed test fold" in blocker
        or "prediction coverage" in blocker
        or "split summary does not match recomputed" in blocker
        for blocker in outcome.blockers
    )


def test_prediction_tamper_and_supporting_relabel_cannot_upgrade(tmp_path, monkeypatch):
    paths, _, _, _ = _production_fixture(tmp_path, monkeypatch)
    paths.predictions.write_bytes(paths.predictions.read_bytes() + b"\n")
    assert any(
        "predictions hash" in blocker
        for blocker in evaluate_evidence_gate(paths).blockers
    )
    paths, _, _, registry = _production_fixture(tmp_path / "role", monkeypatch)
    manifest = json.loads(paths.result_run_manifest.read_text())
    manifest["evidence_role"] = "correctly_specified_synthetic_positive_control"
    _write_json(paths.result_run_manifest, manifest)
    registry_payload = json.loads(registry.read_text())
    registry_payload["approved_runs"][0]["result_run_manifest_sha256"] = _sha(
        paths.result_run_manifest
    )
    _write_json(registry, registry_payload)
    outcome = evaluate_evidence_gate(paths)
    assert any("supporting evidence" in blocker for blocker in outcome.blockers)


def test_testing_only_in_memory_evaluator_never_upgrades():
    outcome = evaluate_testing_evidence(
        {"evidence_role": "direct_validation"}, pd.DataFrame({"x": [1]})
    )
    assert outcome.tier == TESTING_ONLY_NO_CLAIM_UPGRADE
    assert outcome.passed is False


def _direct_outcome() -> EvidenceGateOutcome:
    return EvidenceGateOutcome(
        tier=DIRECT_EXTERNAL_VALIDITY,
        passed=True,
        blockers=(),
        endpoint_checks=pd.DataFrame(),
        allowed_claims=("direct_external_validity_for_locked_primary_endpoints",),
        prohibited_claims=("clinical_utility", "causal_dietary_effect"),
        source_state="verified_direct_validation_artifacts",
    )


def _activate_claim_bundle(
    tmp_path: Path,
    monkeypatch,
    outcome: EvidenceGateOutcome,
    *,
    production_authorized: bool = True,
    empty_patterns: bool = False,
) -> dict[str, Path]:
    root = tmp_path / "repository"
    output = root / "results/phase2/evidence-gate"
    generated = policy_module._write_claim_artifacts(
        output,
        outcome,
        production_authorized=production_authorized,
    )
    if empty_patterns:
        policy = json.loads(generated["policy"].read_text())
        policy["forbidden_patterns"] = []
        policy["payload_sha256"] = policy_module.canonical_payload_sha256(policy)
        generated["policy"].write_bytes(policy_module._json_bytes(policy))
        decision = json.loads(generated["decision"].read_text())
        decision["claim_policy_file_sha256"] = _sha(generated["policy"])
        decision["payload_sha256"] = policy_module.canonical_payload_sha256(decision)
        generated["decision"].write_bytes(policy_module._json_bytes(decision))
    registry = root / "code/src/configs/claim_policy_registry.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        registry,
        {
            "schema_version": "claim-policy-registry-v1",
            "approved_bundles": [
                {
                    "bundle_id": "testing-only-approved-pair",
                    "tier": outcome.tier,
                    "source_state": outcome.source_state,
                    "decision_path": (
                        "results/phase2/evidence-gate/current_gate_decision.json"
                    ),
                    "policy_path": "results/phase2/evidence-gate/claim_policy.json",
                    "decision_sha256": _sha(generated["decision"]),
                    "policy_sha256": _sha(generated["policy"]),
                }
            ],
        },
    )
    monkeypatch.setattr(policy_module, "_REPOSITORY_ROOT", root)
    monkeypatch.setattr(
        policy_module, "_TRUSTED_DECISION_PATH", generated["decision"]
    )
    monkeypatch.setattr(policy_module, "_TRUSTED_POLICY_PATH", generated["policy"])
    monkeypatch.setattr(policy_module, "_TRUSTED_POLICY_REGISTRY_PATH", registry)
    return generated


def test_production_policy_builder_reruns_path_only_gate_and_rejects_outcome_authority(
    tmp_path, monkeypatch
):
    output = tmp_path / "results/phase2/evidence-gate"
    monkeypatch.setattr(
        policy_module, "_TRUSTED_DECISION_PATH", output / "current_gate_decision.json"
    )
    monkeypatch.setattr(
        policy_module, "_TRUSTED_POLICY_PATH", output / "claim_policy.json"
    )
    monkeypatch.setattr(
        policy_module,
        "_TRUSTED_POLICY_REGISTRY_PATH",
        tmp_path / "code/src/configs/claim_policy_registry.json",
    )
    generated = build_claim_policy_from_evidence_gate()
    decision = json.loads(generated["decision"].read_text())
    policy = json.loads(generated["policy"].read_text())
    assert decision["tier"] == COMPUTATIONAL_FEASIBILITY
    assert decision["source_state"] == "absent_real_validation_artifacts"
    assert decision["authorization"] == "production_path_only_evidence_gate"
    assert policy["direct_scope_whitelist"] == []
    with pytest.raises(TypeError):
        build_claim_policy_from_evidence_gate(_direct_outcome())


def test_self_signed_direct_bundle_cannot_authorize_production_claims(
    tmp_path, monkeypatch
):
    _activate_claim_bundle(
        tmp_path,
        monkeypatch,
        _direct_outcome(),
        production_authorized=False,
    )
    claim = tmp_path / "claim.txt"
    claim.write_text(
        "For the locked primary endpoints glucose_iAUC_2h and tg_6h_rise, "
        "locked_attribute_gmnps had lower subject-held-out RMSE than fcs_microbiome."
    )
    with pytest.raises(ValueError, match="production path-only gate"):
        check_claim_inputs([claim])


def test_registry_schema_empty_entries_and_empty_patterns_fail_closed(
    tmp_path, monkeypatch
):
    _activate_claim_bundle(
        tmp_path / "patterns",
        monkeypatch,
        evaluate_evidence_gate(),
        empty_patterns=True,
    )
    claim = tmp_path / "claim.txt"
    claim.write_text("Computational feasibility only.")
    with pytest.raises(ValueError, match="forbidden_patterns"):
        check_claim_inputs([claim])

    generated = _activate_claim_bundle(
        tmp_path / "registry", monkeypatch, evaluate_evidence_gate()
    )
    registry = policy_module._TRUSTED_POLICY_REGISTRY_PATH
    _write_json(
        registry,
        {"schema_version": "claim-policy-registry-v1", "approved_bundles": []},
    )
    assert generated["decision"].exists()
    with pytest.raises(ValueError, match="no approved bundle"):
        check_claim_inputs([claim])


def test_direct_policy_rejects_generic_external_validity_and_allows_exact_scope(
    tmp_path, monkeypatch
):
    _activate_claim_bundle(tmp_path, monkeypatch, _direct_outcome())
    generic = tmp_path / "generic.txt"
    scoped = tmp_path / "scoped.txt"
    clinical = tmp_path / "clinical.txt"
    generic.write_text("The model demonstrates external validity.")
    scoped.write_text(
        "For the locked primary endpoints glucose_iAUC_2h and tg_6h_rise, "
        "locked_attribute_gmnps had lower subject-held-out RMSE than fcs_microbiome."
    )
    clinical.write_text("The locked primary endpoints demonstrate clinical validity.")
    assert check_claim_inputs([generic])
    assert check_claim_inputs([scoped]) == ()
    assert check_claim_inputs([clinical])


@pytest.mark.parametrize(
    "sentence",
    [
        "The model was validated in an independent cohort.",
        "These findings generalize across populations.",
        "The system improved prediction of meal responses.",
        "The score captures metabolic response.",
        "The model provides dietary guidance and recommendations.",
        "The results establish biological consistency.",
        "This experiment provides mechanistic validation.",
        "The score is clinically actionable.",
        "The association is causal.",
        "The platform is precision-ready.",
        "External validation was not performed, and external validity is established.",
    ],
)
def test_sentence_level_final_review_claim_examples_are_rejected(
    tmp_path, monkeypatch, sentence
):
    _activate_claim_bundle(tmp_path, monkeypatch, evaluate_evidence_gate())
    claim = tmp_path / "claim.txt"
    claim.write_text(sentence)
    assert check_claim_inputs([claim])


@pytest.mark.parametrize(
    "sentence",
    [
        "GMNPS accurately predicts individual postprandial glycaemic excursions.",
        (
            "Personalized GMNPS scores outperform Food Compass for glycaemic "
            "and triglyceride outcomes."
        ),
        "The GMNPS model predicts individual postprandial glycaemic responses.",
        "The personalization framework improves glycaemic response prediction.",
        (
            "The microbiome-informed score is associated with better metabolic "
            "outcomes."
        ),
        (
            "The GMNPS model hash supports personalized nutrition and artifact "
            "verification."
        ),
    ],
)
def test_unlisted_subject_assertion_claims_fail_closed(
    tmp_path, monkeypatch, sentence
):
    _activate_claim_bundle(tmp_path, monkeypatch, evaluate_evidence_gate())
    claim = tmp_path / "claim.txt"
    claim.write_text(sentence)
    assert check_claim_inputs([claim])


@pytest.mark.parametrize(
    "sentence",
    [
        "GMNPS yields lower RMSE than Food Compass.",
        "GMNPS achieves superior glycaemic forecasting.",
        "GMNPS stratifies individuals by glycaemic excursion.",
        "The GMNPS method is transparent.",
    ],
)
def test_subject_sentences_default_deny_without_assertion_vocabulary(
    tmp_path, monkeypatch, sentence
):
    _activate_claim_bundle(tmp_path, monkeypatch, evaluate_evidence_gate())
    claim = tmp_path / "claim.txt"
    claim.write_text(sentence)
    assert check_claim_inputs([claim])


@pytest.mark.parametrize(
    "sentence",
    [
        (
            "Although external validation was not performed, external validity "
            "is established."
        ),
        (
            "The GMNPS model does not establish external validity but accurately "
            "predicts glycaemic response."
        ),
        (
            "No evidence of external validity was found; the personalized score "
            "outperforms Food Compass."
        ),
        "The GMNPS model does not fail and accurately predicts glycaemic response.",
        "External validity was not assessed and is established by the model.",
    ],
)
def test_negative_cues_do_not_exempt_later_positive_claims(
    tmp_path, monkeypatch, sentence
):
    _activate_claim_bundle(tmp_path, monkeypatch, evaluate_evidence_gate())
    claim = tmp_path / "claim.txt"
    claim.write_text(sentence)
    assert check_claim_inputs([claim])


@pytest.mark.parametrize(
    "sentence",
    [
        "The GMNPS model does not fail to outperform Food Compass.",
        "The GMNPS model does not fail to predict glycaemic response.",
        "The GMNPS model cannot fail to outperform Food Compass.",
        "The GMNPS model cannot fail to predict glycaemic response.",
        "The GMNPS model does not only outperform Food Compass.",
        "The GMNPS model not only predicts but also outperforms Food Compass.",
        "The GMNPS model does not appear to outperform Food Compass.",
        "The GMNPS model does not seem to predict glycaemic response.",
    ],
)
def test_inverted_or_indirect_negation_cannot_bypass_claim_gate(
    tmp_path, monkeypatch, sentence
):
    _activate_claim_bundle(tmp_path, monkeypatch, evaluate_evidence_gate())
    claim = tmp_path / "claim.txt"
    claim.write_text(sentence)
    assert check_claim_inputs([claim])


@pytest.mark.parametrize(
    "sentence",
    [
        "GMNPS does not establish external validity.",
        "No evidence supports clinical validity.",
        "No evidence of clinical validity is available.",
        "External validation was not performed.",
        "No analysis evaluated clinical utility or dietary recommendations.",
    ],
)
def test_strict_negative_limitation_templates_remain_allowed(
    tmp_path, monkeypatch, sentence
):
    _activate_claim_bundle(tmp_path, monkeypatch, evaluate_evidence_gate())
    claim = tmp_path / "claim.txt"
    claim.write_text(sentence)
    assert check_claim_inputs([claim]) == ()


def test_computational_tier_accepts_only_exact_safe_methods_and_limitations(
    tmp_path, monkeypatch
):
    _activate_claim_bundle(tmp_path, monkeypatch, evaluate_evidence_gate())
    safe = tmp_path / "safe.txt"
    safe.write_text(
        "The locked implementation recovered the programmed mapping in a correctly "
        "specified synthetic positive-control.\n"
        "Eligible observed participant-by-meal outcomes are unavailable in the audited data.\n"
        "The evidence gate implements a computational, fail-closed design.\n"
        "This analysis does not establish external validity.\n"
        "The GMNPS model source hash supports reproducible artifact verification."
    )
    assert check_claim_inputs([safe]) == ()


@pytest.mark.parametrize(
    "sentence",
    [
        "The GMNPS model is implemented as a deterministic scoring pipeline.",
        "GMNPS computes attribute-level scores from locked inputs.",
        "The GMNPS implementation loads the fixed registry.",
        "The GMNPS implementation verifies artifact hashes.",
        "The evidence gate hash-binds the claim policy to the gate decision.",
        "The GMNPS method uses bounded attribute calibration.",
        "The GMNPS model source hash supports reproducible artifact verification.",
        r"The frozen specification is \texttt{attribute-gmnps-v1}.",
    ],
)
def test_enumerated_methods_infrastructure_templates_are_allowed(
    tmp_path, monkeypatch, sentence
):
    _activate_claim_bundle(tmp_path, monkeypatch, evaluate_evidence_gate())
    methods = tmp_path / "methods.txt"
    methods.write_text(sentence)
    assert check_claim_inputs([methods]) == ()


@pytest.mark.parametrize(
    "sentence",
    [
        "GMNPS computes glycaemic outcomes.",
        "The GMNPS model is implemented as a validation pipeline.",
        "The GMNPS implementation verifies superior performance.",
        "The GMNPS method uses bounded attribute calibration for dietary guidance.",
    ],
)
def test_methods_templates_cannot_carry_scientific_claim_semantics(
    tmp_path, monkeypatch, sentence
):
    _activate_claim_bundle(tmp_path, monkeypatch, evaluate_evidence_gate())
    methods = tmp_path / "methods.txt"
    methods.write_text(sentence)
    assert check_claim_inputs([methods])


@pytest.mark.parametrize(
    "sentence",
    [
        r"The frozen specification is \texttt{attribute-gmnps-v2}.",
        r"The frozen specification is \texttt{attribute-gmnps-v1-beta}.",
        r"The frozen specification uses \texttt{attribute-gmnps-v1}.",
        r"The frozen specification is \texttt{attribute-gmnps-v1} and validates outcomes.",
        "No analysis evaluated clinical utility or dietary recommendations and GMNPS improves outcomes.",
        "The framework provides an auditable basis for future empirical testing, but GMNPS establishes external validity.",
    ],
)
def test_exact_methods_and_limitation_allowances_reject_near_matches_and_extensions(
    tmp_path, monkeypatch, sentence
):
    _activate_claim_bundle(tmp_path, monkeypatch, evaluate_evidence_gate())
    candidate = tmp_path / "candidate.txt"
    candidate.write_text(sentence)
    assert check_claim_inputs([candidate])


@pytest.mark.parametrize(
    "text",
    [
        r"The frozen specification is \texttt{attribute-gmnps-v1}. It improves outcomes.",
        "The frozen specification is \\texttt{attribute-gmnps-v1}.\nIt improves outcomes.",
        "The frozen specification is \\texttt{attribute-gmnps-v1}.\nIt improves\noutcomes.",
        "The frozen specification is \\texttt{attribute-gmnps-v1}.\nIt improves % formatting note\noutcomes.",
        "The frozen specification is \\texttt{attribute-gmnps-v1}.\nIt improves\n\\label{claim:x}\noutcomes.",
        "The frozen specification is \\texttt{attribute-gmnps-v1}.\nIt impro\\label\n{claim:x}ves outcomes.",
        "The frozen specification is \\texttt{attribute-gmnps-v1}.\nIt impro\\index\n{claim assertion}ves outcomes.",
        "The frozen specification is \\texttt{attribute-gmnps-v1}.\nIt impro\\phantomsection\nves outcomes.",
        "It improves outcomes.",
        "It improves\noutcomes.",
        "It improves % formatting note\noutcomes.",
        "It improves\n% formatting note\noutcomes.",
        "It improves\n\\label{claim:x}\noutcomes.",
        "It improves\n\\index{claim assertion}\noutcomes.",
        "It improves\n\\phantomsection\noutcomes.",
        "It impro% formatting note\nves outcomes.",
        "It impro\\label{claim:x}ves outcomes.",
        "It impro\\index{claim assertion}ves outcomes.",
        "It impro\\phantomsection ves outcomes.",
        "It impro\\label\n{claim:x}ves outcomes.",
        "It impro\\index\n{claim assertion}ves outcomes.",
        "It impro\\phantomsection\nves outcomes.",
        "The frozen specification is \\texttt{attribute-gmnps-v1}.\n\\label\n\n{It improves outcomes.}",
        "The frozen specification is \\texttt{attribute-gmnps-v1}.\n\\index\n\n{It improves outcomes.}",
        r"The threshold is 20\% and it improves" "\noutcomes.",
        "This improves prediction performance.",
        "That improves response accuracy.",
        "These improve outcomes.",
        "Those improve outcomes.",
        "They improve outcomes.",
        "Performance improves.",
        "Performance\nimproves.",
        "Prediction accuracy improves.",
        "Improves RMSE performance.",
        "Predicts AUROC performance.",
    ],
)
def test_allowed_policy_sentences_cannot_authorize_appended_empirical_assertions(
    tmp_path, monkeypatch, text
):
    _activate_claim_bundle(tmp_path, monkeypatch, evaluate_evidence_gate())
    candidate = tmp_path / "appended-assertion.txt"
    candidate.write_text(text)
    assert check_claim_inputs([candidate])


def test_anaphoric_rule_preserves_negative_and_non_empirical_prose(
    tmp_path, monkeypatch
):
    _activate_claim_bundle(tmp_path, monkeypatch, evaluate_evidence_gate())
    candidate = tmp_path / "benign.txt"
    candidate.write_text(
        r"The frozen specification is \texttt{attribute-gmnps-v1}."
        "\nThis response is stored\nin the manifest."
        "\nPerformance was not evaluated."
        "\nThis does not improve\noutcomes."
        "\nThey record the configuration digest."
        "\n% This improves\noutcomes."
        "\n\\section*{Methods}"
        "\nThis response is recorded\nin the manifest."
    )
    assert check_claim_inputs([candidate]) == ()


@pytest.mark.parametrize(
    "text",
    [
        "This does not improve % formatting note\noutcomes.",
        "This response is stored\n% formatting note\nin the manifest.",
        "This response is stored\n\\label{manifest:x}\nin the manifest.",
        "This response is stored\n\\index{manifest}\nin the manifest.",
        "This response is stored\n\\phantomsection\nin the manifest.",
        r"The threshold is 20\%" "\nand is recorded.",
        "Performance\n\nimproves.",
        "Performance\n\\section*{Methods}\nimproves.",
        "Performance\n\\[\nx = 1\n\\]\nimproves.",
    ],
)
def test_tex_semantic_view_preserves_benign_controls(tmp_path, monkeypatch, text):
    _activate_claim_bundle(tmp_path, monkeypatch, evaluate_evidence_gate())
    candidate = tmp_path / "benign-tex.txt"
    candidate.write_text(text)
    assert check_claim_inputs([candidate]) == ()


def test_tex_semantic_view_preserves_visible_command_arguments(tmp_path, monkeypatch):
    _activate_claim_bundle(tmp_path, monkeypatch, evaluate_evidence_gate())
    candidate = tmp_path / "visible-command.txt"
    candidate.write_text(r"The \href{urn:example}{model} is deterministic.")
    assert check_claim_inputs([candidate])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "This response is stored \\label\n{manifest:{nested}}\nin the manifest.",
            "This response is stored in the manifest.",
        ),
        (
            "This response is stored \\index\n{manifest {nested}}\nin the manifest.",
            "This response is stored in the manifest.",
        ),
        (
            "This response is stored \\phantomsection\nin the manifest.",
            "This response is stored in the manifest.",
        ),
        (r"The threshold is 20\% and is recorded.", r"The threshold is 20\% and is recorded."),
    ],
)
def test_tex_semantic_view_parses_whole_text_zero_width_commands(text, expected):
    assert policy_module._build_tex_semantic_view(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "It impro\\label\n{claim:xves outcomes.",
        "It impro\\index\nclaim assertion}ves outcomes.",
        "\\label\n\n{It improves outcomes.}",
        "\\index\n\n{It improves outcomes.}",
    ],
)
def test_tex_semantic_view_fails_closed_for_malformed_zero_width_commands(
    tmp_path, monkeypatch, text
):
    _activate_claim_bundle(tmp_path, monkeypatch, evaluate_evidence_gate())
    candidate = tmp_path / "malformed-tex.txt"
    candidate.write_text(text)
    violations = check_claim_inputs([candidate])
    assert [item.pattern for item in violations] == [
        "malformed_zero_width_tex_command"
    ]


def test_exact_wrapped_non_empirical_allowances_reject_empirical_extensions(
    tmp_path, monkeypatch
):
    _activate_claim_bundle(tmp_path, monkeypatch, evaluate_evidence_gate())
    candidate = tmp_path / "exact-non-empirical.txt"
    candidate.write_text(
        "The locked implementation, evidence-gate code and audit scripts are\n"
        "maintained in the project repository."
    )
    assert check_claim_inputs([candidate]) == ()

    candidate.write_text(
        "The locked implementation, evidence-gate code and audit scripts are\n"
        "maintained in the project repository and improves outcomes."
    )
    assert check_claim_inputs([candidate])


def test_negative_limitation_sentence_is_allowed_by_current_policy(tmp_path):
    negative = tmp_path / "negative.txt"
    positive = tmp_path / "positive.txt"
    negative.write_text(
        "This correctly specified control does not establish external validity."
    )
    positive.write_text("The model establishes external validity.")
    assert check_claim_inputs([negative]) == ()
    assert check_claim_inputs([positive])


def test_current_decision_policy_registry_and_cli_are_fixed(tmp_path, monkeypatch):
    decision = ROOT / "results/phase2/evidence-gate/current_gate_decision.json"
    policy = ROOT / "results/phase2/evidence-gate/claim_policy.json"
    registry = ROOT / "code/src/configs/claim_policy_registry.json"
    output = tmp_path / "generated"
    monkeypatch.setattr(
        policy_module, "_TRUSTED_DECISION_PATH", output / "current_gate_decision.json"
    )
    monkeypatch.setattr(
        policy_module, "_TRUSTED_POLICY_PATH", output / "claim_policy.json"
    )
    monkeypatch.setattr(
        policy_module,
        "_TRUSTED_POLICY_REGISTRY_PATH",
        output / "claim_policy_registry.json",
    )
    generated = build_claim_policy_from_evidence_gate()
    assert generated["decision"].read_bytes() == decision.read_bytes()
    assert generated["policy"].read_bytes() == policy.read_bytes()
    assert generated["registry"].read_bytes() == registry.read_bytes()
    monkeypatch.undo()
    forbidden = tmp_path / "forbidden.txt"
    forbidden.write_text("The system has clinical validity.")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "code/src/scripts/check_claim_policy.py"),
            str(forbidden),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    bypass = subprocess.run(
        [
            sys.executable,
            str(ROOT / "code/src/scripts/check_claim_policy.py"),
            "--decision",
            str(decision),
            "--policy",
            str(policy),
            str(forbidden),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bypass.returncode == 2


def test_public_checker_has_no_arbitrary_decision_or_policy_path_api(tmp_path):
    text = tmp_path / "input.txt"
    text.write_text("Computational feasibility only.")
    with pytest.raises(TypeError):
        check_claim_inputs(
            ROOT / "results/phase2/evidence-gate/current_gate_decision.json",
            ROOT / "results/phase2/evidence-gate/claim_policy.json",
            [text],
        )
