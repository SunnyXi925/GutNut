import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from gmnps.validation.biological_consistency import (
    BiologicalConsistencyConfig,
    canonical_frame_sha256,
    disease_cohort_consistency,
    evaluate_microbiome_shuffle_rerun,
    knowledge_path_consistency,
    validate_evidence_provenance,
    verify_supporting_evidence_file,
    zoe_aggregate_rank_consistency,
)


def _gmrepo_rows() -> pd.DataFrame:
    rows = []
    values = {
        "cohort_a": {"h1": 0.0, "h2": 2.0, "d1": 4.0, "d2": 6.0},
        "cohort_b": {"h3": 10.0, "h4": 12.0, "d3": 14.0, "d4": 16.0},
    }
    for cohort, people in values.items():
        for person, base in people.items():
            label = "Health" if person.startswith("h") else "T2D"
            for food_index in range(3):
                rows.append(
                    {
                        "individual_id": person,
                        "component_id": f"component_{person}",
                        "food_id": f"food_{food_index}",
                        "cohort_id": cohort,
                        "disease_label": label,
                        "label_source": "GMrepo",
                        "label_provenance": f"{cohort}_curation",
                        "GMNPS_delta": base + food_index * 100.0,
                    }
                )
    return pd.DataFrame(rows)


def _manifest(source_kind: str, role: str) -> dict[str, object]:
    return {
        "schema_version": "gmnps-supporting-evidence-v2",
        "source_kind": source_kind,
        "source_id": f"testing-{source_kind}",
        "accession": f"TEST:{source_kind}",
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
        "table_sha256": "0" * 64,
    }


def _config(**changes) -> BiologicalConsistencyConfig:
    values = {
        "bootstrap_replicates": 60,
        "shuffle_replicates": 60,
        "seed": 2718,
        "minimum_valid_replicates": 30,
        "minimum_arm_units": 2,
    }
    values.update(changes)
    return BiologicalConsistencyConfig(**values)


def _result(frame=None):
    return disease_cohort_consistency(
        _gmrepo_rows() if frame is None else frame,
        _manifest("gmrepo_disease_labels", "biological_consistency"),
        config=_config(),
        allow_test_data=True,
    )


def test_food_standardized_cohort_estimate_is_hand_calculated_and_counts_arms():
    result = _result()
    estimates = result.estimates.set_index("cohort_id")
    expected = 4.0 / np.sqrt(5.0)
    assert estimates.loc["cohort_a", "estimate"] == pytest.approx(expected)
    assert estimates.loc["cohort_b", "estimate"] == pytest.approx(expected)
    assert estimates["summary_method"].eq(
        "within_cohort_food_zscore_then_person_mean"
    ).all()
    assert estimates["n_reference_units"].eq(2).all()
    assert estimates["n_disease_units"].eq(2).all()
    assert result.inference["analysis_status"].eq("estimable").all()
    assert result.inference["bootstrap_replicates_valid"].eq(60).all()
    assert set(result.label_permutation["control"]) == {"shuffled_label"}
    assert result.microbiome_shuffle.iloc[0]["analysis_status"] == "not_estimable"
    assert result.microbiome_shuffle.iloc[0]["reason"] == "verified_rerun_not_provided"


@pytest.mark.parametrize("mutation", ["duplicate", "different_panel"])
def test_gmrepo_requires_unique_person_food_and_identical_predeclared_panel(mutation):
    frame = _gmrepo_rows()
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        message = "unique person-food"
    else:
        frame = frame.drop(frame.index[(frame["individual_id"] == "d1") & (frame["food_id"] == "food_2")])
        message = "identical predeclared food panel"
    with pytest.raises(ValueError, match=message):
        _result(frame)


def test_sparse_or_all_missing_arm_is_not_estimable_without_ci():
    sparse = _gmrepo_rows().query("individual_id != 'd2'").copy()
    result = _result(sparse)
    row = result.inference.query("cohort_id == 'cohort_a'").iloc[0]
    assert row["analysis_status"] == "not_estimable"
    assert row["reason"] == "fewer_than_minimum_units_in_one_or_both_arms"
    assert row["n_reference_units"] == 2
    assert row["n_disease_units"] == 1
    assert row["bootstrap_replicates_valid"] == 0
    assert pd.isna(row["ci_lower"])
    assert pd.isna(row["ci_upper"])

    missing = _gmrepo_rows()
    missing.loc[
        (missing["cohort_id"] == "cohort_a")
        & (missing["disease_label"] == "T2D"),
        "GMNPS_delta",
    ] = np.nan
    result = _result(missing)
    row = result.inference.query("cohort_id == 'cohort_a'").iloc[0]
    assert row["analysis_status"] == "not_estimable"
    assert row["n_disease_units"] == 0
    assert pd.isna(row["estimate"])


def test_supporting_production_is_not_self_authorized_and_requires_explicit_false(tmp_path):
    manifest = _manifest("zoe_aggregate_ranks", "biological_consistency")
    manifest.update(
        {
            "data_class": "published_aggregate_ranks",
            "production_label": "production",
            "synthetic": False,
            "testing_only": False,
        }
    )
    with pytest.raises(ValueError, match="repository-trusted source verification"):
        validate_evidence_provenance(manifest)

    manifest.pop("synthetic")
    with pytest.raises(ValueError, match="synthetic must be explicitly false"):
        validate_evidence_provenance(manifest)

    table = tmp_path / "zoe.csv"
    table.write_text("food_id,gmnps_rank,zoe_rank\na,1,1\n", encoding="utf-8")
    manifest["synthetic"] = False
    manifest["table_sha256"] = hashlib.sha256(table.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="trusted supporting-evidence registry"):
        verify_supporting_evidence_file(table, manifest)


def test_zoe_reports_exclusions_and_not_estimable_for_nan_constant_or_too_few():
    manifest = _manifest("zoe_aggregate_ranks", "biological_consistency")
    frame = pd.DataFrame(
        {
            "food_id": ["a", "b", "c", "d"],
            "gmnps_rank": [1.0, 2.0, np.nan, 4.0],
            "zoe_rank": [1.0, 1.0, 3.0, 1.0],
            "rank_source": ["aggregate"] * 4,
        }
    )
    result = zoe_aggregate_rank_consistency(frame, manifest, allow_test_data=True)
    assert result["n_total"] == 4
    assert result["n_valid"] == 3
    assert result["n_excluded"] == 1
    assert result["analysis_status"] == "not_estimable"
    assert result["reason"] == "constant_rank_vector"
    assert result["spearman_rank_correlation"] is None

    two = frame.iloc[:2].assign(zoe_rank=[1.0, 2.0])
    result = zoe_aggregate_rank_consistency(two, manifest, allow_test_data=True)
    assert result["analysis_status"] == "not_estimable"
    assert result["reason"] == "fewer_than_three_valid_rank_pairs"


def test_true_microbiome_shuffle_requires_deranged_mapping_and_locked_rerun_contract():
    original = _gmrepo_rows().query("cohort_id == 'cohort_a'").copy()
    rerun = original.copy()
    units = sorted(original["component_id"].unique())
    shifted = units[1:] + units[:1]
    mapping = pd.DataFrame(
        {"independent_unit_id": units, "shuffled_profile_unit_id": shifted}
    )
    source_by_person = dict(
        original[["individual_id", "component_id"]].drop_duplicates().to_numpy()
    )
    mapped = dict(zip(units, shifted))
    rerun["profile_source_unit_id"] = rerun["individual_id"].map(
        lambda person: mapped[source_by_person[person]]
    )
    rerun["GMNPS_delta"] = rerun["GMNPS_delta"] + 0.25
    manifest = _manifest("microbiome_shuffle_rerun", "biological_consistency")
    manifest.update(
        {
            "original_table_sha256": canonical_frame_sha256(original),
            "rerun_table_sha256": canonical_frame_sha256(rerun),
            "mapping_sha256": canonical_frame_sha256(mapping),
            "original_scoring_manifest_sha256": "1" * 64,
            "rerun_scoring_manifest_sha256": "2" * 64,
            "locked_scoring_contract_sha256": "3" * 64,
        }
    )
    result = evaluate_microbiome_shuffle_rerun(
        original, rerun, mapping, manifest, allow_test_data=True
    )
    assert result["analysis_status"] == "contract_verified_test_only"
    assert result["mapping_status"] == "bijective_derangement"
    assert result["n_independent_units"] == 4
    assert result["n_foods"] == 3
    assert result["mean_absolute_score_change"] == pytest.approx(0.25)

    bad = mapping.copy()
    first = bad.loc[0, "shuffled_profile_unit_id"]
    bad.loc[0, "shuffled_profile_unit_id"] = bad.loc[0, "independent_unit_id"]
    bad.loc[len(bad) - 1, "shuffled_profile_unit_id"] = first
    manifest["mapping_sha256"] = canonical_frame_sha256(bad)
    with pytest.raises(ValueError, match="derangement"):
        evaluate_microbiome_shuffle_rerun(
            original, rerun, bad, manifest, allow_test_data=True
        )


def test_knowledge_paths_require_nonempty_nodes_and_provenance_and_are_summary_only():
    manifest = _manifest("knowledge_paths", "mechanistic_consistency")
    paths = pd.DataFrame(
        {
            "path_id": ["p1", "p2"],
            "source_node": ["taxon_a", "taxon_b"],
            "target_node": ["fiber", "lipid"],
            "direction": ["positive", "negative"],
            "is_direction_consistent": [True, False],
            "path_provenance": ["paper_a", "paper_b"],
        }
    )
    result = knowledge_path_consistency(paths, manifest, allow_test_data=True)
    assert result["analysis_boundary"] == "adjudicated_path_summary_only"
    assert result["n_paths"] == 2
    bad = paths.copy()
    bad.loc[0, "source_node"] = ""
    with pytest.raises(ValueError, match="source_node.*nonempty"):
        knowledge_path_consistency(bad, manifest, allow_test_data=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bootstrap_replicates", 60.5),
        ("shuffle_replicates", 60.5),
        ("minimum_valid_replicates", 30.5),
        ("minimum_arm_units", 2.5),
        ("seed", 1.5),
    ],
)
def test_biological_config_integer_fields_are_strict(field, value):
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})
