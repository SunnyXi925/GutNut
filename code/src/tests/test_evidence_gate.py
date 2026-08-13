from __future__ import annotations

import pandas as pd

from gmnps.validation.evidence_gate import (
    COMPUTATIONAL_FEASIBILITY,
    DIRECT_EXTERNAL_VALIDITY,
    evaluate_evidence_gate,
)


def _eligible_manifest(**changes: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "evidence_role": "direct_validation",
        "data_class": "real_observed_participant_meal_outcomes",
        "provenance_status": "eligible_verified",
        "observed_participant_meal_outcomes": True,
        "microbiome_linkage_verified": True,
        "method_lock_status": "passed",
        "method_lock_manifest_sha256": "a" * 64,
        "outcome_manifest_sha256": "b" * 64,
        "development_subject_ids": ["d1", "d2"],
        "test_subject_ids": ["t1", "t2"],
    }
    manifest.update(changes)
    return manifest


def _passing_primary_results(**changes: object) -> pd.DataFrame:
    rows = []
    for endpoint in ("glucose_iAUC_2h", "tg_6h_rise"):
        row: dict[str, object] = {
            "endpoint": endpoint,
            "analysis_mode": "subject_held_out",
            "model": "locked_attribute_gmnps",
            "reference": "fcs_microbiome",
            "metric": "rmse",
            "estimate_delta": -0.2,
            "ci_lower": -0.3,
            "ci_upper": -0.1,
            "ci_method": "paired_family_twin_component_cluster_percentile_bootstrap_95",
            "inference_status": "completed",
            "bootstrap_replicates_requested": 2000,
            "bootstrap_replicates_valid": 1900,
            "bootstrap_minimum_valid_fraction": 0.9,
            "permutation_replicates_requested": 2000,
            "permutation_replicates_valid": 1900,
            "permutation_minimum_valid_fraction": 0.9,
            "test_role": "primary",
            "multiplicity_method": "holm",
            "correction_family": "two_primary_endpoints_locked_gmnps_vs_fcs_microbiome_rmse",
            "adjusted_p_value": 0.03,
        }
        row.update(changes)
        rows.append(row)
    return pd.DataFrame(rows)


def test_current_missing_real_manifest_and_results_fail_closed_with_exact_blockers():
    outcome = evaluate_evidence_gate()
    assert outcome.tier == COMPUTATIONAL_FEASIBILITY
    assert outcome.passed is False
    assert outcome.blockers == (
        "eligible real observed participant-by-meal outcome manifest is missing",
        "locked subject-held-out primary paired-results table is missing",
    )
    assert "clinical_validity" in outcome.prohibited_claims
    assert "external_validity" in outcome.prohibited_claims
    assert "precision_ready" in outcome.prohibited_claims
    assert "transforms_postprandial_response" in outcome.prohibited_claims


def test_direct_external_validity_requires_all_provenance_independence_and_performance_checks():
    outcome = evaluate_evidence_gate(_eligible_manifest(), _passing_primary_results())
    assert outcome.passed is True
    assert outcome.tier == DIRECT_EXTERNAL_VALIDITY
    assert outcome.blockers == ()
    assert set(outcome.endpoint_checks["endpoint"]) == {
        "glucose_iAUC_2h",
        "tg_6h_rise",
    }
    assert outcome.endpoint_checks["passed"].all()


def test_gate_rejects_ineligible_provenance_and_subject_overlap():
    outcome = evaluate_evidence_gate(
        _eligible_manifest(
            provenance_status="unverified",
            development_subject_ids=["shared", "d2"],
            test_subject_ids=["shared", "t2"],
        ),
        _passing_primary_results(),
    )
    assert outcome.tier == COMPUTATIONAL_FEASIBILITY
    assert "outcome provenance is not eligible_verified" in outcome.blockers
    assert "development and test subject identifiers overlap: shared" in outcome.blockers


def test_gate_rejects_wrong_direction_ci_multiplicity_and_valid_replicate_fraction():
    rows = _passing_primary_results()
    rows.loc[rows["endpoint"].eq("glucose_iAUC_2h"), "estimate_delta"] = 0.1
    rows.loc[rows["endpoint"].eq("glucose_iAUC_2h"), "ci_upper"] = 0.2
    rows.loc[rows["endpoint"].eq("tg_6h_rise"), "multiplicity_method"] = "none"
    rows.loc[rows["endpoint"].eq("tg_6h_rise"), "adjusted_p_value"] = 0.2
    rows.loc[rows["endpoint"].eq("tg_6h_rise"), "bootstrap_replicates_valid"] = 1799
    outcome = evaluate_evidence_gate(_eligible_manifest(), rows)
    assert outcome.tier == COMPUTATIONAL_FEASIBILITY
    checks = outcome.endpoint_checks.set_index("endpoint")
    assert bool(checks.loc["glucose_iAUC_2h", "direction_passed"]) is False
    assert bool(checks.loc["glucose_iAUC_2h", "ci_passed"]) is False
    assert bool(checks.loc["tg_6h_rise", "multiplicity_passed"]) is False
    assert bool(checks.loc["tg_6h_rise", "valid_replicates_passed"]) is False


def test_supporting_or_synthetic_evidence_can_never_upgrade_the_gate():
    for role, data_class in (
        ("synthetic_identifiability_stress_test", "synthetic"),
        ("biological_consistency", "aggregate_public"),
        ("mechanistic_consistency", "knowledge_graph"),
        ("predictor_reconstruction", "real_public_microbiome"),
    ):
        outcome = evaluate_evidence_gate(
            _eligible_manifest(evidence_role=role, data_class=data_class),
            _passing_primary_results(),
        )
        assert outcome.tier == COMPUTATIONAL_FEASIBILITY
        assert outcome.passed is False
        assert any("cannot upgrade" in blocker for blocker in outcome.blockers)


def test_duplicate_or_missing_primary_endpoint_rows_fail_closed():
    duplicate = pd.concat(
        [_passing_primary_results(), _passing_primary_results().iloc[[0]]],
        ignore_index=True,
    )
    duplicated = evaluate_evidence_gate(_eligible_manifest(), duplicate)
    assert duplicated.tier == COMPUTATIONAL_FEASIBILITY
    assert any("exactly one locked primary row" in item for item in duplicated.blockers)

    missing = evaluate_evidence_gate(
        _eligible_manifest(),
        _passing_primary_results().query("endpoint == 'glucose_iAUC_2h'"),
    )
    assert missing.tier == COMPUTATIONAL_FEASIBILITY
    assert any("tg_6h_rise" in item for item in missing.blockers)
