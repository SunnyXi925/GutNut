import pandas as pd
import pytest

from gmnps.validation.biological_consistency import (
    BiologicalConsistencyConfig,
    disease_cohort_consistency,
    knowledge_path_consistency,
    validate_evidence_provenance,
    zoe_aggregate_rank_consistency,
)


def _gmrepo_rows() -> pd.DataFrame:
    metadata = [
        ("h_a", "component_1", "cohort_a", "Health", "curated_a"),
        ("d_a", "component_2", "cohort_a", "T2D", "curated_a"),
        ("h_b", "component_3", "cohort_b", "Health", "curated_b"),
        ("d_b", "component_4", "cohort_b", "T2D", "curated_b"),
    ]
    values = {"h_a": 1.5, "d_a": -2.0, "h_b": 0.8, "d_b": -1.0}
    rows = []
    for individual, component, cohort, label, provenance in metadata:
        for food_index in range(4):
            rows.append(
                {
                    "individual_id": individual,
                    "component_id": component,
                    "food_id": f"food_{food_index}",
                    "cohort_id": cohort,
                    "disease_label": label,
                    "label_source": "GMrepo",
                    "label_provenance": provenance,
                    "GMNPS_delta": values[individual] + food_index * 0.1,
                }
            )
    return pd.DataFrame(rows)


def _manifest(source_kind: str, role: str) -> dict[str, object]:
    return {
        "schema_version": "gmnps-supporting-evidence-v1",
        "source_kind": source_kind,
        "evidence_role": role,
        "data_class": "synthetic_test_fixture",
        "production_label": "non-production",
        "synthetic": True,
        "testing_only": True,
        "source_verified": True,
        "scoring_frozen_before_labels": True,
        "label_access_authorized": True,
        "outcome_columns": [],
        "reference_label": "Health",
    }


def _config() -> BiologicalConsistencyConfig:
    return BiologicalConsistencyConfig(
        bootstrap_replicates=80,
        shuffle_replicates=80,
        seed=2718,
        minimum_valid_replicates=40,
    )


def test_gmrepo_disease_analysis_is_cohort_stratified_and_counts_people_not_rows():
    result = disease_cohort_consistency(
        _gmrepo_rows(),
        _manifest("gmrepo_disease_labels", "biological_consistency"),
        config=_config(),
        allow_test_data=True,
    )

    estimates = result.estimates.sort_values(["cohort_id", "disease_label"])
    assert estimates["cohort_id"].tolist() == ["cohort_a", "cohort_b"]
    assert estimates["disease_label"].eq("T2D").all()
    assert estimates["reference_label"].eq("Health").all()
    assert estimates["n_individuals"].eq(2).all()
    assert estimates["n_rows"].eq(8).all()
    assert estimates["n_independent_units"].eq(2).all()
    assert estimates["label_source"].eq("GMrepo").all()
    assert estimates["label_provenance"].tolist() == [
        "curated_a",
        "curated_b",
    ]
    assert estimates["evidence_role"].eq("biological_consistency").all()
    assert "external_validation" not in estimates.to_csv(index=False)


def test_gmrepo_cluster_bootstrap_and_shuffle_controls_are_reproducible():
    result = disease_cohort_consistency(
        _gmrepo_rows(),
        _manifest("gmrepo_disease_labels", "biological_consistency"),
        config=_config(),
        allow_test_data=True,
    )

    inference = result.inference
    assert inference["cluster_column"].eq("component_id").all()
    assert inference["ci_method"].eq("cluster_percentile").all()
    assert inference["bootstrap_seed"].eq(2718).all()
    assert inference["bootstrap_replicates_requested"].eq(80).all()
    assert inference["bootstrap_replicates_valid"].between(40, 80).all()
    assert (inference["ci_lower"] <= inference["estimate"]).all()
    assert (inference["estimate"] <= inference["ci_upper"]).all()

    controls = result.shuffle_controls
    assert set(controls["control"]) == {"shuffled_label", "shuffled_microbiome"}
    assert controls["shuffle_replicates_requested"].eq(80).all()
    assert controls["shuffle_replicates_valid"].eq(80).all()
    assert controls["seed"].notna().all()
    assert controls["evidence_role"].eq("biological_consistency").all()

    repeated = disease_cohort_consistency(
        _gmrepo_rows(),
        _manifest("gmrepo_disease_labels", "biological_consistency"),
        config=_config(),
        allow_test_data=True,
    )
    pd.testing.assert_frame_equal(result.inference, repeated.inference)
    pd.testing.assert_frame_equal(result.shuffle_controls, repeated.shuffle_controls)


def test_disease_labels_retain_person_level_cohort_and_label_provenance():
    conflicting = pd.concat(
        [
            _gmrepo_rows(),
            _gmrepo_rows().iloc[[0]].assign(
                cohort_id="cohort_other", label_provenance="unknown"
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="inconsistent cohort_id"):
        disease_cohort_consistency(
            conflicting,
            _manifest("gmrepo_disease_labels", "biological_consistency"),
            config=_config(),
            allow_test_data=True,
        )

    missing_person = _gmrepo_rows().drop(columns="individual_id")
    with pytest.raises(ValueError, match="individual_id"):
        disease_cohort_consistency(
            missing_person,
            _manifest("gmrepo_disease_labels", "biological_consistency"),
            config=_config(),
            allow_test_data=True,
        )


@pytest.mark.parametrize(
    ("source_kind", "allowed_role"),
    [
        ("gmrepo_disease_labels", "biological_consistency"),
        ("zoe_aggregate_ranks", "biological_consistency"),
        ("knowledge_paths", "mechanistic_consistency"),
    ],
)
def test_evidence_role_guard_accepts_only_predefined_supporting_roles(
    source_kind, allowed_role
):
    validate_evidence_provenance(
        _manifest(source_kind, allowed_role), allow_test_data=True
    )
    for forbidden in ("external_validation", "construct_validity", "clinical_validity"):
        manifest = _manifest(source_kind, forbidden)
        with pytest.raises(ValueError, match="evidence_role"):
            validate_evidence_provenance(manifest, allow_test_data=True)


def test_label_bearing_input_fails_closed_on_leakage_or_prelock_access():
    manifest = _manifest("gmrepo_disease_labels", "biological_consistency")
    manifest["scoring_frozen_before_labels"] = False
    with pytest.raises(ValueError, match="frozen"):
        disease_cohort_consistency(
            _gmrepo_rows(), manifest, config=_config(), allow_test_data=True
        )

    leaked = _gmrepo_rows().assign(glucose_iAUC_2h=1.0)
    with pytest.raises(ValueError, match="outcome/label leakage"):
        disease_cohort_consistency(
            leaked,
            _manifest("gmrepo_disease_labels", "biological_consistency"),
            config=_config(),
            allow_test_data=True,
        )


def test_zoe_aggregate_rank_and_knowledge_path_outputs_fix_evidence_roles():
    zoe = pd.DataFrame(
        {
            "food_id": ["a", "b", "c", "d"],
            "gmnps_rank": [1, 2, 4, 3],
            "zoe_rank": [1, 3, 4, 2],
            "rank_source": ["published_aggregate"] * 4,
        }
    )
    rank = zoe_aggregate_rank_consistency(
        zoe,
        _manifest("zoe_aggregate_ranks", "biological_consistency"),
        allow_test_data=True,
    )
    assert rank["n_foods"] == 4
    assert -1 <= rank["spearman_rank_correlation"] <= 1
    assert rank["evidence_role"] == "biological_consistency"
    assert rank["analysis_boundary"] == "aggregate_ranks_not_person_food_responses"

    paths = pd.DataFrame(
        {
            "path_id": ["p1", "p2", "p3"],
            "source_node": ["taxon_a", "taxon_b", "taxon_c"],
            "target_node": ["fiber", "lipid", "sodium"],
            "direction": ["positive", "negative", "positive"],
            "is_direction_consistent": [True, False, True],
            "path_provenance": ["paper_a", "paper_b", "paper_c"],
        }
    )
    path = knowledge_path_consistency(
        paths,
        _manifest("knowledge_paths", "mechanistic_consistency"),
        allow_test_data=True,
    )
    assert path["n_paths"] == 3
    assert path["n_direction_consistent"] == 2
    assert path["evidence_role"] == "mechanistic_consistency"
    assert path["analysis_boundary"] == "mechanistic_paths_only"
