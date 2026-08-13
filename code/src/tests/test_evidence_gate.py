from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from gmnps.validation import evidence_gate as gate_module
from gmnps.validation.claim_policy import (
    check_claim_inputs,
    write_claim_restriction_artifacts,
)
from gmnps.validation.evidence_gate import (
    COMPUTATIONAL_FEASIBILITY,
    DIRECT_EXTERNAL_VALIDITY,
    TESTING_ONLY_NO_CLAIM_UPGRADE,
    EvidenceGateArtifactPaths,
    evaluate_evidence_gate,
    evaluate_testing_evidence,
)
from gmnps.validation.method_lock_gate import MethodLockArtifactPaths


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _production_fixture(tmp_path: Path, monkeypatch, mutation: str | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    run_id = "direct-run-001"
    outcome_sha = "d" * 64
    method_lock_path = tmp_path / "method-lock.json"
    method_lock_path.write_bytes(b"trusted fixture bytes\n")
    lock_sha = _sha(method_lock_path)
    lock_manifest = {
        "person_meal_validation_config_sha256": "1" * 64,
        "feature_contract_sha256": "2" * 64,
        "predictor_frame_sha256": "3" * 64,
        "cohort_split_implementation_sha256": "4" * 64,
        "person_meal_benchmark_implementation_sha256": "5" * 64,
        "benchmark_specification_sha256": "6" * 64,
        "method_lock_gate_implementation_sha256": "7" * 64,
    }
    binding = _canonical_hash(
        {
            "run_id": run_id,
            "method_lock_manifest_sha256": lock_sha,
            "outcome_source_id": "predict_controlled_clinical_zenodo",
            "outcome_sha256": outcome_sha,
            **lock_manifest,
        }
    )
    paired = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "run_binding_sha256": binding,
                "endpoint": endpoint,
                "analysis_mode": "subject_held_out",
                "model": "locked_attribute_gmnps",
                "reference": "fcs_microbiome",
                "metric": "rmse",
                "estimate_delta": -0.20,
                "ci_lower": -0.30,
                "ci_upper": -0.10,
                "ci_method": "paired_family_twin_component_cluster_percentile_bootstrap_95",
                "bootstrap_replicates_requested": 2000,
                "bootstrap_replicates_valid": 1900,
                "permutation_replicates_requested": 2000,
                "permutation_replicates_valid": 1900,
                "permutation_p_value": raw_p,
                "holm_adjusted_p_value": adjusted,
                "test_role": "primary",
                "correction_family": "two_primary_endpoints_locked_gmnps_vs_fcs_microbiome_rmse",
                "inference_status": "completed",
            }
            for endpoint, raw_p, adjusted in (
                ("glucose_iAUC_2h", 0.01, 0.02),
                ("tg_6h_rise", 0.03, 0.03),
            )
        ]
    )
    split = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "run_binding_sha256": binding,
                "analysis_mode": "subject_held_out",
                "outer_fold": 0,
                "participant_id": participant,
                "family_twin_component_id": component,
                "partition": partition,
            }
            for participant, component, partition in (
                ("p1", "c1", "development"),
                ("p2", "c2", "development"),
                ("p3", "c3", "test"),
                ("p4", "c4", "test"),
            )
        ]
    )
    status = {
        "run_id": run_id,
        "run_binding_sha256": binding,
        "analysis_status": "completed",
    }
    evidence_role = "direct_validation"
    if mutation == "mismatched_run":
        paired.loc[0, "run_id"] = "other-run"
    elif mutation == "forged_holm":
        paired.loc[paired["endpoint"].eq("tg_6h_rise"), "permutation_p_value"] = 0.20
    elif mutation == "participant_leakage":
        split.loc[len(split)] = {
            **split.iloc[0].to_dict(),
            "partition": "test",
        }
    elif mutation == "component_leakage":
        split.loc[split["participant_id"].eq("p3"), "family_twin_component_id"] = "c1"
    elif mutation == "supporting_relabel":
        evidence_role = "correctly_specified_synthetic_positive_control"
    elif mutation == "wrong_direction":
        paired.loc[paired["endpoint"].eq("glucose_iAUC_2h"), "estimate_delta"] = 0.10
    elif mutation == "wrong_ci_method":
        paired.loc[paired["endpoint"].eq("glucose_iAUC_2h"), "ci_method"] = "percentile_95"
    elif mutation == "low_valid_fraction":
        paired.loc[
            paired["endpoint"].eq("tg_6h_rise"),
            "bootstrap_replicates_valid",
        ] = 1799
    elif mutation == "missing_primary":
        paired = paired.loc[paired["endpoint"].ne("tg_6h_rise")].copy()

    paired_path = tmp_path / "paired.csv"
    split_path = tmp_path / "split.csv"
    status_path = tmp_path / "status.json"
    paired.to_csv(paired_path, index=False)
    split.to_csv(split_path, index=False)
    _write_json(status_path, status)

    manifest = {
        "schema_version": "direct-validity-result-run-v1",
        "run_id": run_id,
        "run_binding_sha256": binding,
        "evidence_role": evidence_role,
        "data_class": "real_observed_participant_meal_outcomes",
        "outcome_source_id": "predict_controlled_clinical_zenodo",
        "outcome_sha256": outcome_sha,
        "method_lock_manifest_sha256": lock_sha,
        **lock_manifest,
        "artifacts": {
            "paired_results": {"sha256": _sha(paired_path)},
            "split_audit": {"sha256": _sha(split_path)},
            "analysis_status": {"sha256": _sha(status_path)},
        },
    }
    if mutation == "fake_digest":
        manifest["artifacts"]["paired_results"]["sha256"] = "0" * 64
    manifest_path = tmp_path / "result-manifest.json"
    _write_json(manifest_path, manifest)

    registry = {
        "schema_version": "direct-validity-result-registry-v1",
        "approved_runs": [
            {
                "run_id": run_id,
                "result_run_manifest_sha256": _sha(manifest_path),
            }
        ],
    }
    registry_path = tmp_path / "trusted-registry.json"
    _write_json(registry_path, registry)
    monkeypatch.setattr(gate_module, "_TRUSTED_RESULT_REGISTRY_PATH", registry_path)

    dummy = tmp_path / "dummy"
    dummy.write_bytes(b"x")
    lock_paths = MethodLockArtifactPaths(
        **{field.name: dummy for field in fields(MethodLockArtifactPaths)}
    )
    paths = EvidenceGateArtifactPaths(
        method_lock_manifest=method_lock_path,
        method_lock_artifacts=lock_paths,
        result_run_manifest=manifest_path,
        paired_results=paired_path,
        split_audit=split_path,
        analysis_status=status_path,
        controlled_outcome_source_id="predict_controlled_clinical_zenodo",
    )
    loaded = SimpleNamespace(
        outcome=SimpleNamespace(
            frame=pd.DataFrame(
                {"participant_id": ["p1", "p2", "p3", "p4"]}
            ),
            sha256=outcome_sha,
            source_id="predict_controlled_clinical_zenodo",
        ),
        verified_lock=SimpleNamespace(
            manifest=lock_manifest,
            manifest_sha256=lock_sha,
        ),
    )
    monkeypatch.setattr(
        gate_module,
        "load_locked_outcomes_for_benchmark",
        lambda *args, **kwargs: loaded,
    )
    return paths, manifest_path, registry_path


def test_production_api_is_path_only_and_absent_artifacts_fail_closed():
    with pytest.raises(TypeError):
        evaluate_evidence_gate({})
    with pytest.raises(TypeError):
        evaluate_evidence_gate(pd.DataFrame())
    outcome = evaluate_evidence_gate()
    assert outcome.tier == COMPUTATIONAL_FEASIBILITY
    assert outcome.passed is False
    assert "controlled_not_granted" in " ".join(outcome.blockers)
    assert "result-run manifest path is missing" in outcome.blockers


def test_fixed_empty_trusted_registry_and_self_signed_manifest_cannot_upgrade(
    tmp_path, monkeypatch
):
    paths, manifest_path, _ = _production_fixture(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text())
    manifest["self_signed_approval"] = {
        "run_id": manifest["run_id"],
        "result_run_manifest_sha256": _sha(manifest_path),
    }
    _write_json(manifest_path, manifest)
    empty_registry = tmp_path / "fixed-empty-registry.json"
    _write_json(
        empty_registry,
        {
            "schema_version": "direct-validity-result-registry-v1",
            "approved_runs": [],
        },
    )
    monkeypatch.setattr(gate_module, "_TRUSTED_RESULT_REGISTRY_PATH", empty_registry)
    outcome = evaluate_evidence_gate(paths)
    assert outcome.tier == COMPUTATIONAL_FEASIBILITY
    assert any("not independently approved" in item for item in outcome.blockers)


@pytest.mark.parametrize(
    "mutation, blocker_fragment",
    [
        ("fake_digest", "paired_results hash"),
        ("supporting_relabel", "supporting evidence"),
        ("mismatched_run", "same run_id"),
        ("forged_holm", "recomputed Holm"),
        ("participant_leakage", "participant leakage"),
        ("component_leakage", "family/twin component leakage"),
        ("wrong_direction", "required direction"),
        ("wrong_ci_method", "CI/method"),
        ("low_valid_fraction", "valid-resample threshold"),
        ("missing_primary", "exactly the two frozen"),
    ],
)
def test_production_gate_rejects_forgery_and_leakage(
    tmp_path, monkeypatch, mutation, blocker_fragment
):
    paths, _, _ = _production_fixture(tmp_path, monkeypatch, mutation)
    outcome = evaluate_evidence_gate(paths)
    assert outcome.tier == COMPUTATIONAL_FEASIBILITY
    assert any(blocker_fragment in item for item in outcome.blockers)


def test_artifact_tamper_and_registry_manifest_mismatch_fail_closed(tmp_path, monkeypatch):
    paths, manifest_path, _ = _production_fixture(tmp_path, monkeypatch)
    paths.paired_results.write_bytes(paths.paired_results.read_bytes() + b"\n")
    tampered = evaluate_evidence_gate(paths)
    assert any("paired_results hash" in item for item in tampered.blockers)

    paths, manifest_path, _ = _production_fixture(tmp_path / "second", monkeypatch)
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    mismatch = evaluate_evidence_gate(paths)
    assert any("trusted registry" in item for item in mismatch.blockers)


def test_only_full_path_validated_run_can_reach_direct_external_validity(
    tmp_path, monkeypatch
):
    paths, _, _ = _production_fixture(tmp_path, monkeypatch)
    outcome = evaluate_evidence_gate(paths)
    assert outcome.tier == DIRECT_EXTERNAL_VALIDITY
    assert outcome.passed is True
    assert outcome.endpoint_checks["passed"].all()


def test_testing_only_in_memory_evaluator_and_supporting_evidence_never_upgrade():
    outcome = evaluate_testing_evidence(
        {"evidence_role": "direct_validation"},
        pd.DataFrame({"endpoint": ["glucose_iAUC_2h"]}),
    )
    assert outcome.tier == TESTING_ONLY_NO_CLAIM_UPGRADE
    assert outcome.tier != DIRECT_EXTERNAL_VALIDITY
    supporting = evaluate_testing_evidence(
        {"evidence_role": "correctly_specified_synthetic_positive_control"},
        pd.DataFrame(),
    )
    assert supporting.passed is False


def test_claim_policy_checker_rejects_forbidden_claims_and_tamper(tmp_path):
    decision = tmp_path / "decision.json"
    policy = tmp_path / "policy.json"
    allowed = tmp_path / "allowed.txt"
    forbidden = tmp_path / "forbidden.txt"
    policy_payload = {
        "schema_version": "claim-policy-v1",
        "tier": COMPUTATIONAL_FEASIBILITY,
        "allowed_claims": ["computational feasibility"],
        "forbidden_patterns": ["external validity", "precision[- ]ready"],
    }
    policy_payload["payload_sha256"] = _canonical_hash(policy_payload)
    _write_json(policy, policy_payload)
    decision_payload = {
        "schema_version": "evidence-gate-decision-v1",
        "tier": COMPUTATIONAL_FEASIBILITY,
        "passed": False,
        "claim_policy_file_sha256": _sha(policy),
    }
    decision_payload["payload_sha256"] = _canonical_hash(decision_payload)
    _write_json(decision, decision_payload)
    allowed.write_text("This work demonstrates computational feasibility.")
    forbidden.write_text("The system has external validity and is precision-ready.")
    assert check_claim_inputs(decision, policy, [allowed]) == ()
    violations = check_claim_inputs(decision, policy, [forbidden])
    assert {item.pattern for item in violations} == {
        "external validity",
        "precision[- ]ready",
    }
    policy.write_bytes(policy.read_bytes() + b" ")
    with pytest.raises(ValueError, match="claim policy hash"):
        check_claim_inputs(decision, policy, [allowed])


def test_current_decision_artifacts_and_cli_enforce_phase3_inputs(tmp_path):
    root = Path(__file__).resolve().parents[3]
    decision = root / "results/phase2/evidence-gate/current_gate_decision.json"
    policy = root / "results/phase2/evidence-gate/claim_policy.json"
    script = root / "code/src/scripts/check_claim_policy.py"
    allowed = tmp_path / "allowed.txt"
    forbidden = tmp_path / "forbidden.txt"
    allowed.write_text("Correctly specified synthetic positive-control only.")
    forbidden.write_text("This establishes clinical validity.")
    generated = write_claim_restriction_artifacts(
        tmp_path / "generated",
        evaluate_evidence_gate(),
    )
    assert generated["decision"].read_bytes() == decision.read_bytes()
    assert generated["policy"].read_bytes() == policy.read_bytes()
    assert check_claim_inputs(decision, policy, [allowed]) == ()
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
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
    assert "FORBIDDEN_CLAIM" in completed.stdout

    decision_payload = json.loads(decision.read_text())
    decision_payload["tier"] = DIRECT_EXTERNAL_VALIDITY
    tampered = tmp_path / "tampered-decision.json"
    _write_json(tampered, decision_payload)
    with pytest.raises(ValueError, match="gate decision payload hash"):
        check_claim_inputs(tampered, policy, [allowed])
