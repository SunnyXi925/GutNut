import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmnps.validation.biological_consistency import (
    BiologicalConsistencyConfig,
    MicrobiomeShuffleArtifactPaths,
    canonical_frame_sha256,
    disease_cohort_consistency,
    disease_cohort_consistency_from_paths,
    evaluate_microbiome_shuffle_rerun,
    evaluate_microbiome_shuffle_rerun_from_paths,
    knowledge_path_consistency,
    knowledge_path_consistency_from_paths,
    toy_profile_attribute_rescore,
    validate_evidence_provenance,
    verify_supporting_evidence_file,
    zoe_aggregate_rank_consistency,
    zoe_aggregate_rank_consistency_from_paths,
)
from gmnps.validation import biological_consistency as biological_module
from gmnps.scoring.attribute_gmnps import fit_attribute_gmnps, score_attribute_gmnps
from test_attribute_gmnps import development_beta, make_bundle, score_beta, smoke_config
from test_method_lock_gate import _fixture as method_lock_fixture
from test_method_lock_gate import _generate as generate_method_lock
from test_method_lock_gate import _normalization_payload


ROOT = Path(__file__).resolve().parents[2]


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
    assert result.microbiome_shuffle.iloc[0]["reason"] == (
        "production_rerun_artifact_paths_not_provided"
    )


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


def test_complete_case_set_is_fixed_before_standardization_and_permutation():
    frame = _gmrepo_rows()
    extra = frame.query("individual_id == 'h1'").copy()
    extra["individual_id"] = "h_extra"
    extra["component_id"] = "component_h_extra"
    extra.loc[extra["food_id"] == "food_2", "GMNPS_delta"] = np.nan
    augmented = pd.concat([frame, extra], ignore_index=True)
    result = _result(augmented)
    reference = _result(frame)
    observed = result.estimates.set_index("cohort_id").loc["cohort_a"]
    expected = reference.estimates.set_index("cohort_id").loc["cohort_a"]
    assert observed["estimate"] == pytest.approx(expected["estimate"])
    assert observed["n_excluded_units"] == 1
    permutation = result.label_permutation.set_index("cohort_id").loc["cohort_a"]
    assert permutation["n_analysis_units"] == 4


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
    with pytest.raises(ValueError, match="testing/in-memory-only"):
        validate_evidence_provenance(manifest)

    manifest.pop("synthetic")
    with pytest.raises(ValueError, match="synthetic must be explicitly false"):
        validate_evidence_provenance(manifest)

    table = tmp_path / "zoe.csv"
    table.write_text("food_id,gmnps_rank,zoe_rank\na,1,1\n", encoding="utf-8")
    manifest["synthetic"] = False
    manifest["table_sha256"] = hashlib.sha256(table.read_bytes()).hexdigest()
    provenance = tmp_path / "zoe.json"
    provenance.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="trusted supporting-evidence registry"):
        verify_supporting_evidence_file(table, provenance)


def test_supporting_production_public_api_is_path_only_and_rechecks_table(
    tmp_path, monkeypatch
):
    table = tmp_path / "zoe.csv"
    pd.DataFrame(
        {
            "food_id": ["a", "b", "c"],
            "gmnps_rank": [1, 2, 3],
            "zoe_rank": [1, 3, 2],
            "rank_source": ["published"] * 3,
        }
    ).to_csv(table, index=False)
    manifest = _manifest("zoe_aggregate_ranks", "biological_consistency")
    manifest.update(
        {
            "data_class": "published_aggregate_ranks",
            "production_label": "production",
            "synthetic": False,
            "testing_only": False,
            "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        }
    )
    provenance = tmp_path / "zoe.json"
    provenance.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    provenance_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    registry = tmp_path / "supporting_registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "gmnps-supporting-evidence-trust-v1",
                "sources": [
                    {
                        "source_kind": manifest["source_kind"],
                        "source_id": manifest["source_id"],
                        "accession": manifest["accession"],
                        "table_sha256": manifest["table_sha256"],
                        "provenance_sha256": provenance_hash,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    monkeypatch.setattr(
        biological_module, "_TRUSTED_SUPPORTING_REGISTRY_PATH", registry
    )
    result = zoe_aggregate_rank_consistency_from_paths(table, provenance)
    assert result["analysis_status"] == "estimable"
    table.write_text(table.read_text() + "d,4,4,published\n")
    with pytest.raises(ValueError, match="content hash mismatch"):
        zoe_aggregate_rank_consistency_from_paths(table, provenance)
    assert set(inspect.signature(zoe_aggregate_rank_consistency_from_paths).parameters) == {
        "table_path",
        "provenance_path",
    }
    assert "verified_source" not in inspect.signature(
        zoe_aggregate_rank_consistency
    ).parameters


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
    profiles = pd.DataFrame(
        {
            "independent_unit_id": ["u1", "u2", "u3", "u4"],
            "mac_profile": [0.2, 0.5, -0.4, 0.8],
            "lipid_profile": [0.1, -0.3, 0.6, 0.4],
        }
    )
    foods = pd.DataFrame(
        {
            "food_id": ["f1", "f2", "f3"],
            "FCS2": [40.0, 60.0, 80.0],
            "mac_attribute": [2.0, -1.0, 0.5],
            "lipid_attribute": [1.0, 2.0, -1.0],
        }
    )
    original = toy_profile_attribute_rescore(profiles, foods)
    units = sorted(profiles["independent_unit_id"])
    shifted = units[1:] + units[:1]
    sidecar = pd.DataFrame(
        {
            "target_individual_id": units,
            "source_individual_id": shifted,
            "target_component_id": units,
            "source_component_id": shifted,
        }
    )
    rerun = toy_profile_attribute_rescore(profiles, foods, mapping=sidecar)
    assert "component_id" not in original
    assert "profile_source_unit_id" not in rerun
    manifest = _manifest("microbiome_shuffle_rerun", "biological_consistency")
    manifest.update(
        {
            "original_table_sha256": canonical_frame_sha256(original),
            "rerun_table_sha256": canonical_frame_sha256(rerun),
            "mapping_sidecar_sha256": canonical_frame_sha256(sidecar),
            "original_scoring_manifest_sha256": "1" * 64,
            "rerun_scoring_manifest_sha256": "2" * 64,
            "locked_scoring_contract_sha256": "3" * 64,
        }
    )
    result = evaluate_microbiome_shuffle_rerun(
        original, rerun, sidecar, manifest, allow_test_data=True
    )
    assert result["analysis_status"] == "contract_verified_test_only"
    assert result["mapping_status"] == "bijective_derangement"
    assert result["n_independent_units"] == 4
    assert result["n_foods"] == 3
    assert result["mean_absolute_score_change"] > 0

    bad = sidecar.copy()
    first = bad.loc[0, "source_individual_id"]
    bad.loc[0, "source_individual_id"] = bad.loc[0, "target_individual_id"]
    bad.loc[0, "source_component_id"] = bad.loc[0, "target_component_id"]
    bad.loc[len(bad) - 1, "source_individual_id"] = first
    bad.loc[len(bad) - 1, "source_component_id"] = first
    manifest["mapping_sidecar_sha256"] = canonical_frame_sha256(bad)
    with pytest.raises(ValueError, match="derangement"):
        evaluate_microbiome_shuffle_rerun(
            original, rerun, bad, manifest, allow_test_data=True
        )
    extra = pd.concat(
        [
            sidecar,
            pd.DataFrame(
                {
                    "target_individual_id": ["extra"],
                    "source_individual_id": ["extra2"],
                    "target_component_id": ["extra"],
                    "source_component_id": ["extra2"],
                }
            ),
        ],
        ignore_index=True,
    )
    manifest["mapping_sidecar_sha256"] = canonical_frame_sha256(extra)
    with pytest.raises(ValueError, match="individual sets must exactly equal"):
        evaluate_microbiome_shuffle_rerun(
            original, rerun, extra, manifest, allow_test_data=True
        )


@pytest.mark.parametrize("leaked_artifact", ["original", "rerun", "sidecar"])
def test_microbiome_shuffle_rejects_outcome_columns_in_every_artifact(leaked_artifact):
    profiles = pd.DataFrame(
        {
            "independent_unit_id": ["u1", "u2"],
            "mac_profile": [0.2, -0.3],
            "lipid_profile": [0.1, 0.4],
        }
    )
    foods = pd.DataFrame(
        {
            "food_id": ["f1", "f2"],
            "FCS2": [40.0, 70.0],
            "mac_attribute": [1.0, 2.0],
            "lipid_attribute": [2.0, -1.0],
        }
    )
    sidecar = pd.DataFrame(
        {
            "target_individual_id": ["u1", "u2"],
            "source_individual_id": ["u2", "u1"],
            "target_component_id": ["c1", "c2"],
            "source_component_id": ["c2", "c1"],
        }
    )
    original = toy_profile_attribute_rescore(profiles, foods)
    rerun = toy_profile_attribute_rescore(profiles, foods, mapping=sidecar)
    selected = {"original": original, "rerun": rerun, "sidecar": sidecar}[
        leaked_artifact
    ]
    selected["outcome"] = 1
    manifest = _manifest("microbiome_shuffle_rerun", "biological_consistency")
    manifest.update(
        {
            "original_table_sha256": canonical_frame_sha256(original),
            "rerun_table_sha256": canonical_frame_sha256(rerun),
            "mapping_sidecar_sha256": canonical_frame_sha256(sidecar),
            "original_scoring_manifest_sha256": "1" * 64,
            "rerun_scoring_manifest_sha256": "2" * 64,
            "locked_scoring_contract_sha256": "3" * 64,
        }
    )
    with pytest.raises(ValueError, match="outcome/label leakage"):
        evaluate_microbiome_shuffle_rerun(
            original, rerun, sidecar, manifest, allow_test_data=True
        )


def test_mapping_sidecar_rejects_within_component_person_shuffle():
    profiles = pd.DataFrame(
        {
            "independent_unit_id": ["u1", "u2", "u3", "u4"],
            "mac_profile": [0.1, 0.2, 0.3, 0.4],
            "lipid_profile": [0.4, 0.3, 0.2, 0.1],
        }
    )
    foods = pd.DataFrame(
        {
            "food_id": ["f1"],
            "FCS2": [50.0],
            "mac_attribute": [1.0],
            "lipid_attribute": [1.0],
        }
    )
    sidecar = pd.DataFrame(
        {
            "target_individual_id": ["u1", "u2", "u3", "u4"],
            "source_individual_id": ["u2", "u1", "u4", "u3"],
            "target_component_id": ["c1", "c1", "c2", "c2"],
            "source_component_id": ["c1", "c1", "c2", "c2"],
        }
    )
    with pytest.raises(ValueError, match="component-level derangement"):
        toy_profile_attribute_rescore(profiles, foods, mapping=sidecar)


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


def test_blank_cluster_and_outcome_columns_fail_closed():
    frame = _gmrepo_rows()
    frame.loc[0, "component_id"] = " "
    with pytest.raises(ValueError, match="component_id must be nonempty"):
        _result(frame)
    zoe = pd.DataFrame(
        {
            "food_id": ["a", "b", "c"],
            "gmnps_rank": [1, 2, 3],
            "zoe_rank": [1, 2, 3],
            "rank_source": ["aggregate"] * 3,
            "outcome": [0, 0, 0],
        }
    )
    with pytest.raises(ValueError, match="outcome/label leakage"):
        zoe_aggregate_rank_consistency(
            zoe,
            _manifest("zoe_aggregate_ranks", "biological_consistency"),
            allow_test_data=True,
        )
    paths = pd.DataFrame(
        {
            "path_id": ["p1"],
            "source_node": ["taxon"],
            "target_node": ["fiber"],
            "direction": ["positive"],
            "is_direction_consistent": [True],
            "path_provenance": ["paper"],
            "outcome": [1],
        }
    )
    with pytest.raises(ValueError, match="outcome/label leakage"):
        knowledge_path_consistency(
            paths,
            _manifest("knowledge_paths", "mechanistic_consistency"),
            allow_test_data=True,
        )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _success_marker(path: Path, table: Path, manifest: Path) -> None:
    _write_json(
        path,
        {
            "status": "complete",
            "files": {
                "individual_food.csv": hashlib.sha256(table.read_bytes()).hexdigest(),
                "run_manifest.json": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            },
        },
    )


def _production_shuffle_chain(tmp_path, monkeypatch):
    people = ["person_1", "person_2", "person_3", "person_4"]
    dev = development_beta()
    original_score_beta = score_beta(people)
    for index, person in enumerate(people):
        original_score_beta.loc[person, :] = (-0.75, -0.25, 0.5, 1.0)[index]
    sidecar = pd.DataFrame(
        {
            "target_individual_id": people,
            "source_individual_id": ["person_3", "person_4", "person_1", "person_2"],
            "target_component_id": ["component_a", "component_a", "component_b", "component_b"],
            "source_component_id": ["component_b", "component_b", "component_a", "component_a"],
        }
    )
    source_by_target = dict(
        zip(sidecar["target_individual_id"], sidecar["source_individual_id"])
    )
    rerun_score_beta = original_score_beta.loc[
        [source_by_target[person] for person in people]
    ].copy()
    rerun_score_beta.index = people
    model = fit_attribute_gmnps(dev, smoke_config())
    bundle = make_bundle()
    original_scored = score_attribute_gmnps(model, original_score_beta, bundle)
    rerun_scored = score_attribute_gmnps(model, rerun_score_beta, bundle)
    assert "component_id" not in original_scored.individual_food
    assert "profile_source_unit_id" not in rerun_scored.individual_food

    lock_paths, _ = method_lock_fixture(tmp_path / "lock", monkeypatch=monkeypatch)
    beta_payload = lambda frame: {
        "schema_version": "gmnps-canonical-beta-v1",
        "participant_ids": list(map(str, frame.index)),
        "nutrient_order": list(map(str, frame.columns)),
        "values": frame.to_numpy(float).tolist(),
    }
    _write_json(lock_paths.development_beta, beta_payload(dev))
    _write_json(lock_paths.scoring_beta, beta_payload(original_score_beta))
    _write_json(
        lock_paths.normalization_state,
        _normalization_payload(
            list(map(str, dev.index)),
            list(map(str, dev.columns)),
            dev.to_numpy(float).tolist(),
        ),
    )
    bundle.official_fcs.rename("FCS2").to_csv(lock_paths.official_fcs)
    bundle.food_metadata.to_csv(lock_paths.food_metadata)
    bundle.baseline_points.to_csv(lock_paths.baseline_attribute_points)
    bundle.food_exposures.to_csv(lock_paths.food_exposures)
    bundle.effective_attribute_weights.to_csv(lock_paths.effective_attribute_weights)
    release_registry = json.loads(lock_paths.release_registry.read_text())
    release_registry["approved_artifacts"][0]["artifact_sha256"] = {
        name: hashlib.sha256(getattr(lock_paths, name).read_bytes()).hexdigest()
        for name in (
            "official_fcs",
            "food_metadata",
            "baseline_attribute_points",
            "food_exposures",
            "effective_attribute_weights",
        )
    }
    _write_json(lock_paths.release_registry, release_registry)

    sidecar_path = tmp_path / "mapping_sidecar.csv"
    sidecar.to_csv(sidecar_path, index=False)
    sidecar_hash = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    rerun_beta = tmp_path / "rerun_beta.json"
    _write_json(rerun_beta, beta_payload(rerun_score_beta))
    phase1_artifacts = {
        "development_beta": lock_paths.development_beta,
        "score_beta": lock_paths.scoring_beta,
        "food_metadata": lock_paths.food_metadata,
        "baseline_attribute_points": lock_paths.baseline_attribute_points,
        "food_exposures": lock_paths.food_exposures,
        "effective_attribute_weights": lock_paths.effective_attribute_weights,
    }
    original_input = {
        "production_label": "production",
        "files": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in phase1_artifacts.items()
        },
        "model_contract": "attribute-recomposition-test-chain",
    }
    _write_json(lock_paths.input_manifest, original_input)
    rerun_input_payload = json.loads(json.dumps(original_input))
    rerun_input_payload["files"]["score_beta"] = hashlib.sha256(
        rerun_beta.read_bytes()
    ).hexdigest()
    rerun_input_payload["microbiome_shuffle_sidecar_sha256"] = sidecar_hash
    rerun_input = tmp_path / "rerun_input.json"
    _write_json(rerun_input, rerun_input_payload)

    method_lock = generate_method_lock(lock_paths)
    method_lock_path = tmp_path / "method_lock.json"
    _write_json(method_lock_path, method_lock)
    locator = tmp_path / "method_lock_locator.json"
    _write_json(
        locator,
        {name: str(getattr(lock_paths, name)) for name in lock_paths.__dataclass_fields__},
    )

    original_table = tmp_path / "original" / "individual_food.csv"
    rerun_table = tmp_path / "rerun" / "individual_food.csv"
    original_table.parent.mkdir()
    rerun_table.parent.mkdir()
    original_scored.individual_food.to_csv(original_table, index=False)
    rerun_scored.individual_food.to_csv(rerun_table, index=False)

    contracts = []
    for scored in (original_scored, rerun_scored):
        run_contract = dict(scored.run_manifest)
        run_contract.update(
            {
                "production_label": "production",
                "development_smoke_test": False,
            }
        )
        contracts.append(run_contract)
    contract = contracts[0]
    original_run = tmp_path / "original" / "run_manifest.json"
    rerun_run = tmp_path / "rerun" / "run_manifest.json"
    for path, upstream, run_contract in (
        (original_run, lock_paths.input_manifest, contracts[0]),
        (rerun_run, rerun_input, contracts[1]),
    ):
        upstream_hash = hashlib.sha256(upstream.read_bytes()).hexdigest()
        _write_json(
            path,
            {
                **run_contract,
                "input_manifest_sha256": upstream_hash,
                "source_hashes": {
                    **dict(run_contract["source_hashes"]),
                    "input_manifest": upstream_hash,
                },
            },
        )
    original_success = tmp_path / "original" / "_SUCCESS.json"
    rerun_success = tmp_path / "rerun" / "_SUCCESS.json"
    _success_marker(original_success, original_table, original_run)
    _success_marker(rerun_success, rerun_table, rerun_run)

    provenance = _manifest("microbiome_shuffle_rerun", "biological_consistency")
    provenance.update(
        {
            "data_class": "locked_microbiome_shuffle_rerun",
            "production_label": "production",
            "synthetic": False,
            "testing_only": False,
            "table_sha256": hashlib.sha256(rerun_table.read_bytes()).hexdigest(),
        }
    )
    observed = {
        "original_table_sha256": hashlib.sha256(original_table.read_bytes()).hexdigest(),
        "rerun_table_sha256": hashlib.sha256(rerun_table.read_bytes()).hexdigest(),
        "mapping_sidecar_sha256": sidecar_hash,
        "original_scoring_manifest_sha256": hashlib.sha256(original_run.read_bytes()).hexdigest(),
        "rerun_scoring_manifest_sha256": hashlib.sha256(rerun_run.read_bytes()).hexdigest(),
        "locked_scoring_contract_sha256": biological_module._canonical_mapping_sha256(
            biological_module._scoring_contract(contract)
        ),
        "original_input_manifest_sha256": hashlib.sha256(lock_paths.input_manifest.read_bytes()).hexdigest(),
        "rerun_input_manifest_sha256": hashlib.sha256(rerun_input.read_bytes()).hexdigest(),
        "original_scoring_beta_sha256": hashlib.sha256(lock_paths.scoring_beta.read_bytes()).hexdigest(),
        "rerun_scoring_beta_sha256": hashlib.sha256(rerun_beta.read_bytes()).hexdigest(),
        "method_lock_manifest_sha256": hashlib.sha256(method_lock_path.read_bytes()).hexdigest(),
        "method_lock_artifact_locator_sha256": hashlib.sha256(locator.read_bytes()).hexdigest(),
        "phase1_release_registry_sha256": hashlib.sha256(lock_paths.release_registry.read_bytes()).hexdigest(),
        "scoring_implementation_sha256": hashlib.sha256(
            (ROOT / "src/scripts/run_attribute_gmnps.py").read_bytes()
        ).hexdigest(),
        "original_success_marker_sha256": hashlib.sha256(original_success.read_bytes()).hexdigest(),
        "rerun_success_marker_sha256": hashlib.sha256(rerun_success.read_bytes()).hexdigest(),
    }
    provenance.update(observed)
    provenance_path = tmp_path / "shuffle_provenance.json"
    _write_json(provenance_path, provenance)
    provenance_hash = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    registry = tmp_path / "supporting_registry.json"
    _write_json(
        registry,
        {
            "schema_version": "gmnps-supporting-evidence-trust-v1",
            "sources": [
                {
                    "source_kind": provenance["source_kind"],
                    "source_id": provenance["source_id"],
                    "accession": provenance["accession"],
                    "table_sha256": provenance["table_sha256"],
                    "provenance_sha256": provenance_hash,
                    **observed,
                }
            ],
        },
    )
    monkeypatch.setattr(biological_module, "_TRUSTED_SUPPORTING_REGISTRY_PATH", registry)
    paths = MicrobiomeShuffleArtifactPaths(
        provenance=provenance_path,
        original_score_table=original_table,
        original_run_manifest=original_run,
        original_success_marker=original_success,
        original_input_manifest=lock_paths.input_manifest,
        rerun_score_table=rerun_table,
        rerun_run_manifest=rerun_run,
        rerun_success_marker=rerun_success,
        rerun_input_manifest=rerun_input,
        mapping_sidecar=sidecar_path,
        original_scoring_beta=lock_paths.scoring_beta,
        rerun_scoring_beta=rerun_beta,
        method_lock_manifest=method_lock_path,
        method_lock_artifact_locator=locator,
        method_lock_artifacts=lock_paths,
    )
    return paths, registry


def test_production_microbiome_shuffle_verifies_two_live_phase1_chains(
    tmp_path, monkeypatch
):
    paths, _ = _production_shuffle_chain(tmp_path, monkeypatch)
    result = evaluate_microbiome_shuffle_rerun_from_paths(paths)
    assert result["analysis_status"] == "verified"
    assert result["mapping_status"] == "bijective_derangement"
    assert result["n_independent_units"] == 2
    assert result["verified_input_hashes"]["rerun_scoring_beta_sha256"] == hashlib.sha256(
        paths.rerun_scoring_beta.read_bytes()
    ).hexdigest()


def test_production_microbiome_shuffle_rejects_self_signed_rerun_tamper(
    tmp_path, monkeypatch
):
    paths, _ = _production_shuffle_chain(tmp_path, monkeypatch)
    payload = json.loads(paths.rerun_scoring_beta.read_text())
    payload["values"][0][0] += 1
    _write_json(paths.rerun_scoring_beta, payload)
    with pytest.raises(ValueError, match="rerun input manifest does not bind"):
        evaluate_microbiome_shuffle_rerun_from_paths(paths)


def test_production_microbiome_shuffle_rejects_hash_coherent_wrong_beta_mapping(
    tmp_path, monkeypatch
):
    paths, registry = _production_shuffle_chain(tmp_path, monkeypatch)
    rerun_beta = json.loads(paths.rerun_scoring_beta.read_text())
    rerun_beta["values"][0] = list(rerun_beta["values"][1])
    _write_json(paths.rerun_scoring_beta, rerun_beta)

    rerun_input = json.loads(paths.rerun_input_manifest.read_text())
    rerun_input["files"]["score_beta"] = hashlib.sha256(
        paths.rerun_scoring_beta.read_bytes()
    ).hexdigest()
    _write_json(paths.rerun_input_manifest, rerun_input)
    rerun_run = json.loads(paths.rerun_run_manifest.read_text())
    input_hash = hashlib.sha256(paths.rerun_input_manifest.read_bytes()).hexdigest()
    rerun_run["input_manifest_sha256"] = input_hash
    rerun_run["source_hashes"]["input_manifest"] = input_hash
    _write_json(paths.rerun_run_manifest, rerun_run)
    _success_marker(
        paths.rerun_success_marker,
        paths.rerun_score_table,
        paths.rerun_run_manifest,
    )

    provenance = json.loads(paths.provenance.read_text())
    updates = {
        "rerun_scoring_beta_sha256": hashlib.sha256(
            paths.rerun_scoring_beta.read_bytes()
        ).hexdigest(),
        "rerun_input_manifest_sha256": hashlib.sha256(
            paths.rerun_input_manifest.read_bytes()
        ).hexdigest(),
        "rerun_scoring_manifest_sha256": hashlib.sha256(
            paths.rerun_run_manifest.read_bytes()
        ).hexdigest(),
        "rerun_success_marker_sha256": hashlib.sha256(
            paths.rerun_success_marker.read_bytes()
        ).hexdigest(),
    }
    provenance.update(updates)
    _write_json(paths.provenance, provenance)
    registry_payload = json.loads(registry.read_text())
    registry_payload["sources"][0].update(updates)
    registry_payload["sources"][0]["provenance_sha256"] = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json(registry, registry_payload)

    with pytest.raises(ValueError, match="not exactly the declared source individual"):
        evaluate_microbiome_shuffle_rerun_from_paths(paths)


def test_production_dataframe_authority_cannot_be_forged_or_reused():
    manifest = _manifest("zoe_aggregate_ranks", "biological_consistency")
    manifest.update(
        {
            "data_class": "published_aggregate_ranks",
            "production_label": "production",
            "synthetic": False,
            "testing_only": False,
        }
    )
    frame = pd.DataFrame(
        {
            "food_id": ["a", "b", "c"],
            "gmnps_rank": [1, 2, 3],
            "zoe_rank": [1, 2, 3],
            "rank_source": ["published"] * 3,
        }
    )
    with pytest.raises(ValueError, match="testing/in-memory-only"):
        zoe_aggregate_rank_consistency(frame, manifest)
    assert not hasattr(biological_module, "VerifiedSupportingEvidence")
    assert not hasattr(biological_module, "VerifiedMicrobiomeShuffleRerun")
    assert "verified_source" not in inspect.signature(
        zoe_aggregate_rank_consistency
    ).parameters


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
