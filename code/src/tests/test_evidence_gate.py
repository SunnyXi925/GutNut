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

from gmnps.validation import evidence_gate as gate_module
from gmnps.validation.claim_policy import (
    build_claim_policy_payload,
    check_claim_inputs,
    write_claim_restriction_artifacts,
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
    REQUIRED_COMPARATORS,
    BenchmarkResult,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "code/src/configs/person_meal_validation.yaml"
SOURCE_ID = "predict_controlled_clinical_zenodo"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


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
                            "y_pred": float(getattr(row, endpoint) - 0.1),
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
    paired = pd.DataFrame(
        [
            {
                **provenance,
                "endpoint": endpoint,
                "analysis_mode": "subject_held_out",
                "analysis_status": "completed",
                "comparator": "locked_attribute_gmnps",
                "reference": "fcs_microbiome",
                "metric": "rmse",
                "estimate_delta": -0.20,
                "ci_lower": -0.30,
                "ci_upper": -0.10,
                "ci_method": "paired_family_twin_component_cluster_percentile_bootstrap_95",
                "bootstrap_replicates_requested": 2000,
                "bootstrap_replicates_valid": 1900,
                "bootstrap_minimum_valid_fraction": 0.9,
                "bootstrap_valid_fraction": 0.95,
                "permutation_replicates_requested": 2000,
                "permutation_replicates_valid": 1900,
                "permutation_minimum_valid_fraction": 0.9,
                "permutation_valid_fraction": 0.95,
                "permutation_p_value": raw_p,
                "adjusted_p_value": adjusted,
                "test_role": "primary",
                "correction_family": "two_primary_endpoints_locked_gmnps_vs_fcs_microbiome_rmse",
                "multiplicity_method": "holm",
                "inference_status": "completed",
            }
            for endpoint, raw_p, adjusted in (
                ("glucose_iAUC_2h", 0.01, 0.02),
                ("tg_6h_rise", 0.03, 0.03),
            )
        ]
    )
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
        predictions=pd.DataFrame(prediction_rows),
        absolute_metrics=pd.DataFrame(),
        paired_metrics=paired,
        fit_audit=pd.DataFrame(),
        splits={"subject_held_out": nested},
        split_audit=split_audit,
        missingness_source=pd.DataFrame(),
        analysis_status=status,
    )


def _production_fixture(tmp_path: Path, monkeypatch):
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


def test_exporter_roundtrips_actual_benchmark_result_schema(tmp_path, monkeypatch):
    paths, exported, _, _ = _production_fixture(tmp_path, monkeypatch)
    paired = pd.read_csv(exported.paired_metrics)
    status = pd.read_csv(exported.analysis_status)
    split = pd.read_csv(exported.split_audit)
    manifest = json.loads(exported.result_run_manifest.read_text())
    assert "comparator" in paired and "adjusted_p_value" in paired
    assert "model" not in paired and "holm_adjusted_p_value" not in paired
    assert set(status["analysis_status"]) == {"completed"}
    assert set(split["outer_fold"]) == set(range(5))
    assert "predictions" in manifest["artifacts"]
    outcome = evaluate_evidence_gate(paths)
    assert outcome.tier == DIRECT_EXTERNAL_VALIDITY
    assert outcome.passed is True


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


def test_claim_policy_is_dynamic_conservative_and_allows_negative_limitations(tmp_path):
    computational = evaluate_evidence_gate()
    computational_policy = build_claim_policy_payload(computational)
    assert computational_policy["tier"] == COMPUTATIONAL_FEASIBILITY
    assert computational_policy["source_state"] == "absent_real_validation_artifacts"
    direct = EvidenceGateOutcome(
        tier=DIRECT_EXTERNAL_VALIDITY,
        passed=True,
        blockers=(),
        endpoint_checks=pd.DataFrame(),
        allowed_claims=("direct_external_validity_for_locked_primary_endpoints",),
        prohibited_claims=("clinical_utility", "causal_dietary_effect"),
        source_state="verified_direct_validation_artifacts",
    )
    direct_policy = build_claim_policy_payload(direct)
    assert direct_policy["tier"] == DIRECT_EXTERNAL_VALIDITY
    assert "external validity" not in direct_policy["forbidden_patterns"]
    assert "clinical validity" in direct_policy["forbidden_patterns"]
    assert "causal dietary effect" in direct_policy["forbidden_patterns"]

    generated = write_claim_restriction_artifacts(tmp_path / "gate", computational)
    negative = tmp_path / "negative.txt"
    positive = tmp_path / "positive.txt"
    negative.write_text("This correctly specified control does not establish external validity.")
    positive.write_text("The model establishes external validity.")
    assert check_claim_inputs(generated["decision"], generated["policy"], [negative]) == ()
    assert check_claim_inputs(generated["decision"], generated["policy"], [positive])


def test_current_decision_artifacts_are_deterministic_and_cli_enforced(tmp_path):
    decision = ROOT / "results/phase2/evidence-gate/current_gate_decision.json"
    policy = ROOT / "results/phase2/evidence-gate/claim_policy.json"
    generated = write_claim_restriction_artifacts(
        tmp_path / "generated", evaluate_evidence_gate()
    )
    assert generated["decision"].read_bytes() == decision.read_bytes()
    assert generated["policy"].read_bytes() == policy.read_bytes()
    forbidden = tmp_path / "forbidden.txt"
    forbidden.write_text("The system has clinical validity.")
    completed = subprocess.run(
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
    assert completed.returncode == 1
