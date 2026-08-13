from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import inspect
import json
from pathlib import Path
from types import MappingProxyType

import pandas as pd
import pytest

from gmnps.data_sources import predict_zoe_loader as loader_module
from gmnps.data_sources.predict_zoe_loader import (
    OutcomeAccessBlocked,
    PredictorLoadError,
    load_frozen_validation_config,
    load_outcome_table_after_gate,
    load_predictor_table,
)
from gmnps.data_sources.predict_zoe_registry import (
    ControlledOutcomeGrant,
    OutcomeEndpointGrant,
    PredictorArtifact,
    get_predict_zoe_source,
)
from gmnps.scoring.attribute_calibration import fit_beta_normalization
from gmnps.validation import method_lock_gate as gate_module
from gmnps.validation.method_lock_gate import (
    MethodLockArtifactPaths,
    canonical_ids_sha256,
    generate_method_lock_manifest,
)
from gmnps.validation.person_meal_benchmark import validate_benchmark_method_lock
from scripts import fetch_official_predict_zoe as fetcher


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "code/src/configs/person_meal_validation.yaml"
SCHEMA_PATH = ROOT / "docs/methods/method_lock_manifest.schema.json"
GATE_PATH = ROOT / "code/src/gmnps/validation/method_lock_gate.py"
COHORT_SPLIT_PATH = ROOT / "code/src/gmnps/validation/cohort_split.py"
BENCHMARK_PATH = ROOT / "code/src/gmnps/validation/person_meal_benchmark.py"
CANONICAL_RELEASES = [f"FNDDS {start}-{start + 1}" for start in range(2001, 2018, 2)]
PRODUCTION_ARTIFACT_KEYS = (
    "official_fcs",
    "food_metadata",
    "baseline_attribute_points",
    "food_exposures",
    "effective_attribute_weights",
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _testing_only_predictor_frame_bytes() -> bytes:
    return _canonical_bytes(
        {
            "schema_version": "person-meal-predictor-frame-v1",
            "construction_version": "testing-only-construction-v1",
            "generated_stage": "pre-outcome_predictor_only",
            "columns": [
                "participant_id",
                "meal_id",
                "food_id",
                "family_id",
                "twin_id",
                "cohort_id",
                "clinical_feature",
                "fcs_score",
                "microbiome_feature",
                "legacy_score",
                "gmnps_score",
            ],
            "rows": [
                ["p1", "m1", "f1", "fam1", None, "c1", 1.0, 2.0, 3.0, 4.0, 5.0],
                ["p2", "m2", "f2", "fam2", None, "c2", 1.5, 2.5, 3.5, 4.5, 5.5],
            ],
        }
    )


def _testing_only_feature_contract_bytes(predictor_frame_bytes: bytes) -> bytes:
    structural = {
        "participant_id": "identifier",
        "meal_id": "identifier",
        "food_id": "mapping_unit",
        "family_id": "grouping",
        "twin_id": "grouping",
        "cohort_id": "grouping",
    }
    blocks = {
        "clinical_demographic_diet": ["clinical_feature"],
        "fcs": ["fcs_score"],
        "microbiome": ["microbiome_feature"],
        "legacy_final_score_offset": ["legacy_score"],
        "locked_attribute_gmnps": ["gmnps_score"],
    }
    source_id = "testing-only-predictor-source"
    predictor_payload = json.loads(predictor_frame_bytes)
    predictor_sha256 = sha256(predictor_frame_bytes).hexdigest()
    columns = [
        {
            "name": name,
            "data_type": "string",
            "role": role,
            "nullable": name == "twin_id",
            "source_artifact_id": source_id,
        }
        for name, role in structural.items()
    ] + [
        {
            "name": name,
            "data_type": "number",
            "role": "predictor",
            "nullable": True,
            "source_artifact_id": source_id,
        }
        for names in blocks.values()
        for name in names
    ]
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
    payload = {
        "schema_version": "person-meal-feature-contract-v1",
        "construction_version": "testing-only-construction-v1",
        "generated_stage": "pre-outcome_predictor_only",
        "columns": columns,
        "source_artifact_sha256": {source_id: sha256(source_id.encode()).hexdigest()},
        "predictor_frame_artifact": {
            "artifact_id": "testing-only-canonical-predictor-frame",
            "schema_version": "person-meal-predictor-frame-v1",
            "sha256": predictor_sha256,
        },
        "block_artifact_sha256": {
            block: sha256(
                _canonical_bytes(
                    {
                        "schema_version": "person-meal-predictor-block-v1",
                        "construction_version": predictor_payload[
                            "construction_version"
                        ],
                        "predictor_frame_sha256": predictor_sha256,
                        "block": block,
                        "columns": names,
                        "rows": [
                            [
                                row[predictor_payload["columns"].index(column)]
                                for column in names
                            ]
                            for row in predictor_payload["rows"]
                        ],
                    }
                )
            ).hexdigest()
            for block, names in blocks.items()
        },
        "feature_blocks": {
            block: {"artifact_id": block, "columns": names}
            for block, names in blocks.items()
        },
        "allowed_overlaps": [],
        "mapping_unit": {"column": "food_id", "role": "mapping_unit"},
        "comparators": comparators,
    }
    return _canonical_bytes(payload)


def _predictor_record(
    path: Path,
    *,
    file_format: str = "csv",
) -> PredictorArtifact:
    if file_format == "rdata":
        required_columns = (
            "microbiome_feature_id_axis",
            "sample_id_axis",
            "relative_abundance_value",
        )
        schema_kind = "relative_abundance_matrix_feature_by_sample"
        unique_key = ("microbiome_feature_id_axis", "sample_id_axis")
        id_column = "sample_id_axis"
        feature_id_sha256 = sha256(b"0\n1").hexdigest()
    else:
        required_columns = ("participant_id", "meal_id", "fiber_g")
        schema_kind = "delimited_table_exact_header"
        unique_key = ("participant_id", "meal_id")
        id_column = "participant_id"
        feature_id_sha256 = None
    return PredictorArtifact(
        source_id="official-test-predictors",
        resource_id="predict1_ena_raw_metagenomes",
        stable_source="https://example.test/official-predictors.csv",
        expected_cache_path=path.name,
        path_policy="repository_relative_exact_no_symlink",
        sha256=_sha(path),
        digest_basis="locally_audited_sha256_pinned_in_repository",
        size_bytes=path.stat().st_size,
        content_class="predictor_only",
        allowed_analytical_role="predictor_reconstruction",
        file_format=file_format,
        schema_kind=schema_kind,
        required_columns=required_columns,
        allowed_columns=required_columns,
        unique_key=unique_key,
        expected_records=2,
        id_column=id_column,
        id_sha256=sha256(b"p1\np2").hexdigest(),
        feature_id_sha256=feature_id_sha256,
    )


def _install_predictor_registry(monkeypatch, root: Path, record: PredictorArtifact) -> None:
    monkeypatch.setattr(loader_module, "_REPOSITORY_ROOT", root)
    monkeypatch.setattr(
        loader_module,
        "_PREDICTOR_ARTIFACTS_BY_ID",
        MappingProxyType({record.source_id: record}),
    )


def _write_predictor_fixture(
    path: Path,
    *,
    file_format: str,
    participant_ids: tuple[str, str],
) -> None:
    rows = [
        {"participant_id": participant_ids[0], "meal_id": "m1", "fiber_g": 4.5},
        {"participant_id": participant_ids[1], "meal_id": "m1", "fiber_g": 7.0},
    ]
    if file_format == "csv":
        pd.DataFrame(rows).to_csv(path, index=False)
    elif file_format == "tsv":
        pd.DataFrame(rows).to_csv(path, index=False, sep="\t")
    elif file_format == "json":
        path.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    elif file_format == "parquet":
        pd.DataFrame(rows).to_parquet(path, index=False)
    elif file_format == "rdata":
        import rdata

        matrix = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            columns=list(participant_ids),
        )
        rdata.write_rda(path, {"matrix": matrix})
    else:
        raise AssertionError(f"unsupported fixture format: {file_format}")


def _assert_original_predictor_snapshot(
    frame: pd.DataFrame,
    *,
    file_format: str,
) -> None:
    if file_format == "rdata":
        assert tuple(frame.columns) == ("p1", "p2")
        assert frame.to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]
    else:
        assert frame["participant_id"].tolist() == ["p1", "p2"]


def _endpoint_grants() -> tuple[OutcomeEndpointGrant, ...]:
    return (
        OutcomeEndpointGrant(
            name="glucose_iAUC_2h",
            source_column="glucose_source",
            availability="available",
            role="primary",
            unit="mmol_L_hour",
            window_hours=(0, 2),
            summary="incremental_area_under_curve",
            derivation="baseline_subtracted_trapezoidal_auc_signed_excursions",
        ),
        OutcomeEndpointGrant(
            name="tg_6h_rise",
            source_column="tg_source",
            availability="available",
            role="primary",
            unit="mmol_L",
            window_hours=(0, 6),
            summary="rise_above_baseline",
            derivation="six_hour_value_minus_time_zero_baseline",
        ),
        OutcomeEndpointGrant(
            name="c_peptide_iAUC_2h",
            source_column=None,
            availability="not_available",
            role="secondary",
            unit="nmol_L_hour",
            window_hours=(0, 2),
            summary="incremental_area_under_curve",
            derivation="baseline_subtracted_trapezoidal_auc_signed_excursions",
        ),
    )


def _controlled_source_and_grant(tmp_path: Path):
    outcome = tmp_path / "controlled_outcome.csv"
    _write_csv(
        outcome,
        [
            {
                "participant_id": "p1",
                "meal_id": "m1",
                "glucose_source": 1.0,
                "tg_source": 0.2,
            }
        ],
    )
    dictionary = tmp_path / "data_dictionary.json"
    dictionary.write_bytes(b'{"schema":"temporary-controlled-test-v1"}\n')
    source = replace(
        get_predict_zoe_source("predict_controlled_clinical_zenodo"),
        access_status="controlled_granted",
        allowed_analytical_role="direct_validation",
        local_path=str(outcome),
        sha256=_sha(outcome),
        exclusion_reason="none_after_verified_controlled_grant",
        evidence_basis="temporary trusted registry fixture",
    )
    grant = ControlledOutcomeGrant(
        source_id=source.resource_id,
        access_status="controlled_granted",
        allowed_analytical_role="direct_validation",
        verified_local_path=str(outcome),
        sha256=source.sha256,
        file_format="csv",
        version_doi="10.5281/zenodo.17236383",
        approval_evidence_identifier="DUA-TEST-APPROVED",
        data_dictionary_path=str(dictionary),
        data_dictionary_sha256=_sha(dictionary),
        participant_meal_key_contract=("participant_id", "meal_id"),
        endpoint_contracts=_endpoint_grants(),
        microbiome_linkage_evidence_identifier="LINKAGE-TEST-VERIFIED",
        microbiome_linkage_key_contract=("participant_id",),
    )
    return source, grant, outcome


def _normalization_payload(
    fit_ids: list[str], nutrients: list[str], values: list[list[float]]
) -> dict[str, object]:
    state = fit_beta_normalization(pd.DataFrame(values, index=fit_ids, columns=nutrients))
    return {
        "nutrient_order": list(state.nutrient_order),
        "median_values": list(state.median_values),
        "scale_values": list(state.scale_values),
        "scale_method_values": list(state.scale_method_values),
        "fit_n": state.fit_n,
        "fit_id_sha256": state.fit_id_sha256,
        "method_version": state.method_version,
        "temperature": state.temperature,
        "state_fingerprint": state.state_fingerprint,
    }


def _real_method_lock_fixture(tmp_path: Path, monkeypatch):
    lock_root = tmp_path / "method_lock"
    lock_root.mkdir()
    production = {}
    for key in PRODUCTION_ARTIFACT_KEYS:
        path = lock_root / f"{key}.csv"
        path.write_bytes(f"canonical-{key}\n".encode())
        production[key] = path
    linkage = lock_root / "food_source_linkage.json"
    linkage.write_bytes(b'{"linkage":"verified"}\n')
    input_manifest = lock_root / "input_manifest.json"
    input_manifest.write_bytes(b'{"production_label":"production"}\n')

    fit_ids = ["development-2", "development-1"]
    scoring_ids = ["held-out-1", "held-out-2"]
    nutrients = ["fiber", "potassium"]
    development = lock_root / "development_beta.json"
    scoring = lock_root / "scoring_beta.json"
    normalization = lock_root / "normalization_state.json"
    fit_values = [[0.0, 0.0], [1.0, 1.0]]
    _write_json(
        development,
        {
            "schema_version": "gmnps-canonical-beta-v1",
            "participant_ids": fit_ids,
            "nutrient_order": nutrients,
            "values": fit_values,
        },
    )
    _write_json(
        scoring,
        {
            "schema_version": "gmnps-canonical-beta-v1",
            "participant_ids": scoring_ids,
            "nutrient_order": nutrients,
            "values": [[0.5, 0.5], [1.5, 1.5]],
        },
    )
    _write_json(normalization, _normalization_payload(fit_ids, nutrients, fit_values))

    entry = {
        "bundle_id": "verified-test-bundle-v1",
        "fndds_releases": CANONICAL_RELEASES,
        "artifact_sha256": {key: _sha(path) for key, path in production.items()},
        "food_source_linkage_sha256": _sha(linkage),
        "nutrient_units": {"fiber": "g_per_100_kcal", "potassium": "mg_per_100_kcal"},
        "exposure_basis": "per_100_kcal",
    }
    registry_payload = {
        "schema_version": "fcs2-fndds-release-registry-schema-v1",
        "registry_version": "fcs2-fndds-release-registry-v1",
        "digest_algorithm": "sha256",
        "canonical_release_set": CANONICAL_RELEASES,
        "approved_artifact_entry_schema": {
            "required_fields": [
                "bundle_id",
                "fndds_releases",
                "artifact_sha256",
                "food_source_linkage_sha256",
                "nutrient_units",
                "exposure_basis",
            ],
            "artifact_sha256_keys": list(PRODUCTION_ARTIFACT_KEYS),
            "digest_algorithm": "sha256",
        },
        "approved_artifacts": [entry],
    }
    release_registry = lock_root / "release_registry.json"
    _write_json(release_registry, registry_payload)
    schema = lock_root / "method_lock_manifest.schema.json"
    schema.write_bytes(SCHEMA_PATH.read_bytes())
    config = lock_root / "person_meal_validation.yaml"
    config.write_bytes(CONFIG_PATH.read_bytes())
    predictor_frame = lock_root / "predictor_frame.json"
    predictor_frame.write_bytes(_testing_only_predictor_frame_bytes())
    feature_contract = lock_root / "feature_contract.json"
    feature_contract.write_bytes(
        _testing_only_feature_contract_bytes(predictor_frame.read_bytes())
    )

    implementation_hashes = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))[
        "properties"
    ]["implementation_source_sha256"]["const"]
    implementation_root = lock_root / "repository_root"
    for relative in implementation_hashes:
        destination = implementation_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    monkeypatch.setattr(gate_module, "_REPOSITORY_ROOT", implementation_root)
    monkeypatch.setattr(gate_module, "_TRUSTED_METHOD_LOCK_SCHEMA_PATH", schema)
    monkeypatch.setattr(gate_module, "_TRUSTED_RELEASE_REGISTRY_PATH", release_registry)

    paths = MethodLockArtifactPaths(
        method_lock_schema=schema,
        person_meal_validation_config=config,
        feature_contract=feature_contract,
        predictor_frame=predictor_frame,
        release_registry=release_registry,
        gate_implementation=GATE_PATH,
        cohort_split_implementation=COHORT_SPLIT_PATH,
        person_meal_benchmark_implementation=BENCHMARK_PATH,
        development_beta=development,
        normalization_state=normalization,
        scoring_beta=scoring,
        official_fcs=production["official_fcs"],
        food_metadata=production["food_metadata"],
        baseline_attribute_points=production["baseline_attribute_points"],
        food_exposures=production["food_exposures"],
        effective_attribute_weights=production["effective_attribute_weights"],
        input_manifest=input_manifest,
        food_source_linkage=linkage,
    )
    registry_sha = _sha(release_registry)
    manifest = generate_method_lock_manifest(
        paths,
        beta_fit_cohort_id="development-v1",
        expected_release_registry_sha256=registry_sha,
    )
    manifest_path = lock_root / "method_lock_manifest.json"
    _write_json(manifest_path, manifest)
    return paths, manifest_path


def test_predictor_api_accepts_only_repository_source_id():
    parameters = inspect.signature(load_predictor_table).parameters
    assert tuple(parameters) == ("source_id",)
    assert "path" not in parameters
    assert "expected_sha256" not in parameters
    assert "content_class" not in parameters
    assert "allowed_columns" not in parameters


def test_predictor_loader_uses_trusted_path_digest_schema_and_keys(tmp_path, monkeypatch):
    path = tmp_path / "predictors.csv"
    _write_csv(
        path,
        [
            {"participant_id": "p1", "meal_id": "m1", "fiber_g": 4.5},
            {"participant_id": "p2", "meal_id": "m1", "fiber_g": 7.0},
        ],
    )
    record = _predictor_record(path)
    _install_predictor_registry(monkeypatch, tmp_path, record)

    loaded = load_predictor_table(record.source_id)
    assert loaded.sha256 == record.sha256
    assert loaded.source_id == record.source_id
    assert loaded.frame.shape == (2, 3)

    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(PredictorLoadError, match="SHA-256|size"):
        load_predictor_table(record.source_id)


def test_unknown_source_path_swap_and_symlink_are_refused_before_parse(tmp_path, monkeypatch):
    path = tmp_path / "predictors.csv"
    _write_csv(
        path,
        [
            {"participant_id": "p1", "meal_id": "m1", "fiber_g": 4.5},
            {"participant_id": "p2", "meal_id": "m1", "fiber_g": 7.0},
        ],
    )
    record = _predictor_record(path)
    _install_predictor_registry(monkeypatch, tmp_path, record)
    with pytest.raises(OutcomeAccessBlocked, match="trusted|unknown|registry"):
        load_predictor_table("caller-disguised-source")

    target = tmp_path / "swapped.csv"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(PredictorLoadError, match="symlink|path policy"):
        load_predictor_table(record.source_id)


def test_unknown_label_column_is_rejected_from_header_before_table_parse(
    tmp_path, monkeypatch
):
    path = tmp_path / "predictors.csv"
    path.write_bytes(b"participant_id,meal_id,label\np1,m1,1\np2,m1,0\n")
    trusted = replace(
        _predictor_record(path),
        sha256=_sha(path),
        size_bytes=path.stat().st_size,
    )
    _install_predictor_registry(monkeypatch, tmp_path, trusted)
    parsed = []
    monkeypatch.setattr(
        loader_module,
        "_parse_trusted_table",
        lambda *args, **kwargs: parsed.append(True),
    )

    with pytest.raises(OutcomeAccessBlocked, match="schema|column|label"):
        load_predictor_table(trusted.source_id)
    assert parsed == []


@pytest.mark.parametrize("file_format", ["csv", "tsv", "json", "parquet", "rdata"])
@pytest.mark.parametrize(
    "replacement_phase",
    ["after_snapshot_hash", "after_header_schema"],
)
@pytest.mark.parametrize("replacement_kind", ["regular_file", "symlink"])
def test_predictor_load_uses_one_snapshot_across_hash_schema_and_parse(
    tmp_path,
    monkeypatch,
    file_format,
    replacement_phase,
    replacement_kind,
):
    suffix = {"rdata": "rda"}.get(file_format, file_format)
    path = tmp_path / f"predictors.{suffix}"
    attacker = tmp_path / f"attacker.{suffix}"
    _write_predictor_fixture(
        path,
        file_format=file_format,
        participant_ids=("p1", "p2"),
    )
    _write_predictor_fixture(
        attacker,
        file_format=file_format,
        participant_ids=("attacker-1", "attacker-2"),
    )
    record = _predictor_record(path, file_format=file_format)
    _install_predictor_registry(monkeypatch, tmp_path, record)

    hook_name = {
        "after_snapshot_hash": "_read_predictor_snapshot",
        "after_header_schema": "_validate_predictor_snapshot_schema",
    }[replacement_phase]
    original_hook = getattr(loader_module, hook_name)

    def replace_path_after_phase(*args, **kwargs):
        result = original_hook(*args, **kwargs)
        path.unlink()
        if replacement_kind == "regular_file":
            attacker.replace(path)
        else:
            path.symlink_to(attacker)
        return result

    monkeypatch.setattr(loader_module, hook_name, replace_path_after_phase)

    loaded = load_predictor_table(record.source_id)
    assert loaded.sha256 == record.sha256
    _assert_original_predictor_snapshot(loaded.frame, file_format=file_format)


def test_fetcher_uses_pinned_repository_digest_not_mutable_acquisition_manifest():
    by_id = {artifact.artifact_id: artifact for artifact in fetcher.PREDICTOR_ARTIFACTS}
    ena = by_id["ena_prjeb39223_sequencing_metadata"]
    attacker_manifest = {
        ena.artifact_id: {
            "stable_url": ena.stable_url,
            "sha256": "0" * 64,
        }
    }
    assert ena.expected_sha256 == (
        "d630be36c1abd3a71b6aa295dc56e1557557f965f6c8fdaa673759b227b28e36"
    )
    assert ena.digest_basis == "locally_audited_sha256_pinned_in_repository"
    assert fetcher._known_digest(ena, attacker_manifest) == ena.expected_sha256
    assert tuple(inspect.signature(fetcher.fetch_artifact).parameters)[0] == "source_id"
    with pytest.raises(ValueError, match="unknown official artifact"):
        fetcher.fetch_artifact(
            "caller-disguised-source",
            cache_dir=Path("unused"),
            previous=attacker_manifest,
            timeout_seconds=1,
        )


def test_frozen_config_contains_source_independent_complete_outcome_contract():
    frozen = load_frozen_validation_config(CONFIG_PATH)
    payload = frozen.payload
    contract = payload["outcome_contract"]
    assert frozen.sha256 == _sha(CONFIG_PATH)
    assert contract["unique_key"] == ["participant_id", "meal_id"]
    assert contract["primary_endpoint_availability"] == "all_required"
    assert contract["secondary_endpoint_availability"] == (
        "required_if_trusted_data_dictionary_marks_available_otherwise_omitted"
    )
    assert contract["caller_endpoint_subset"] == "forbidden"
    endpoints = contract["endpoints"]
    assert [(row["name"], row["role"]) for row in endpoints] == [
        ("glucose_iAUC_2h", "primary"),
        ("tg_6h_rise", "primary"),
        ("c_peptide_iAUC_2h", "secondary"),
    ]
    assert all(row["derivation"] for row in endpoints)
    assert payload["missingness"]["outcome_imputation"] == "forbidden"
    assert payload["split"]["development_scoring_overlap"] == "forbidden"
    assert [mode["name"] for mode in payload["analysis_modes"]] == [
        "subject_held_out",
        "subject_plus_food_held_out",
        "subject_plus_meal_held_out",
        "cohort_held_out",
    ]
    modes = {mode["name"]: mode for mode in payload["analysis_modes"]}
    assert modes["subject_held_out"]["estimand"] == (
        "generalization_to_unseen_family_twin_connected_components"
    )
    assert modes["subject_held_out"]["inference_policy"] == (
        "family_twin_connected_component_cluster"
    )
    for name in (
        "subject_plus_food_held_out",
        "subject_plus_meal_held_out",
        "cohort_held_out",
    ):
        assert modes[name]["inference_policy"] == "descriptive_only"
        assert modes[name]["estimand"]
    assert payload["feature_contract"]["generated_stage"] == (
        "pre-outcome_predictor_only"
    )
    assert payload["primary_tests"]["multiplicity_method"] == "holm"
    assert len(payload["primary_tests"]["tests"]) == 2
    specification = payload["benchmark_specification"]
    assert specification["specification_id"] == "person-meal-ridge-nested-v1"
    assert specification["estimator"] == {
        "class": "sklearn.linear_model.Ridge",
        "solver": "lsqr",
        "alpha_grid": [0.1, 1.0, 10.0],
        "fit_intercept": True,
        "tol": 0.0001,
        "max_iter": None,
        "copy_x": True,
        "positive": False,
    }
    assert specification["preprocessing"]["categorical_missing_indicator"] == (
        "explicit_all_columns"
    )
    assert specification["tuning"] == {
        "metric": "mae",
        "scope": "development_inner_folds_only",
        "selection_rule": "minimum_mean_inner_mae_then_smallest_alpha",
    }
    assert specification["bootstrap"]["replicates"] == 2000
    assert specification["bootstrap"]["minimum_valid_fraction"] == 0.9
    assert specification["permutation"]["replicates"] == 2000
    assert specification["permutation"]["minimum_valid_fraction"] == 0.9
    assert specification["seed_derivation"]["identifier"] == (
        "additive-indexed-v1"
    )
    assert specification["seed_derivation"]["split_formula"] == (
        "base_plus_mode_10000019_plus_outer_fold_for_inner"
    )


def test_frozen_config_missing_modified_or_wrong_hash_fails_closed(tmp_path):
    with pytest.raises(PredictorLoadError, match="missing|unreadable"):
        load_frozen_validation_config(tmp_path / "missing.yaml")
    changed = tmp_path / "changed.yaml"
    changed.write_bytes(CONFIG_PATH.read_bytes() + b"\n")
    with pytest.raises(PredictorLoadError, match="config.*SHA-256"):
        load_frozen_validation_config(changed, expected_sha256=_sha(CONFIG_PATH))


def test_production_controlled_source_is_blocked_before_manifest_or_outcome_open(
    tmp_path, monkeypatch
):
    events = []
    monkeypatch.setattr(
        loader_module,
        "_read_outcome_bytes",
        lambda *args, **kwargs: events.append("outcome-read"),
    )
    with pytest.raises(OutcomeAccessBlocked, match="grant|access|direct"):
        load_outcome_table_after_gate(
            "predict_controlled_clinical_zenodo",
            manifest_path=tmp_path / "missing.json",
            method_lock_paths=object(),
        )
    assert events == []


def test_outcome_api_does_not_accept_caller_schema_key_units_or_endpoint_subset():
    parameters = inspect.signature(load_outcome_table_after_gate).parameters
    assert tuple(parameters) == ("source_id", "manifest_path", "method_lock_paths")
    for forbidden in (
        "required_columns",
        "unique_key",
        "endpoint_units",
        "endpoint_subset",
        "path",
        "expected_sha256",
    ):
        assert forbidden not in parameters


@pytest.mark.parametrize(
    "tamper",
    [
        "row_id",
        "missing_primary",
        "role",
        "unit",
        "window_hours",
        "summary",
        "derivation",
    ],
)
def test_frozen_contract_rejects_key_primary_or_endpoint_semantic_tampering(
    tmp_path, tamper
):
    _, grant, _ = _controlled_source_and_grant(tmp_path)
    config = load_frozen_validation_config(CONFIG_PATH)
    endpoints = list(grant.endpoint_contracts)
    if tamper == "row_id":
        grant = replace(grant, participant_meal_key_contract=("row_id",))
    elif tamper == "missing_primary":
        grant = replace(grant, endpoint_contracts=tuple(endpoints[:1] + endpoints[2:]))
    else:
        field_values = {
            "role": "secondary",
            "unit": "mg_dL_hour",
            "window_hours": (0, 3),
            "summary": "caller_selected_summary",
            "derivation": "outcome_dependent_derivation",
        }
        endpoints[0] = replace(endpoints[0], **{tamper: field_values[tamper]})
        grant = replace(grant, endpoint_contracts=tuple(endpoints))
    with pytest.raises((PredictorLoadError, OutcomeAccessBlocked), match="contract|endpoint|key"):
        loader_module._derive_outcome_table_contract(grant, config)


def test_task3_entry_reuses_real_task2_validator_end_to_end(tmp_path, monkeypatch):
    paths, manifest_path = _real_method_lock_fixture(tmp_path, monkeypatch)

    verified = validate_benchmark_method_lock(
        manifest_path=manifest_path,
        method_lock_paths=paths,
    )

    assert verified.feature_contract.generated_stage == "pre-outcome_predictor_only"
    assert verified.feature_contract.sha256 == _sha(paths.feature_contract)
    assert verified.manifest["feature_contract_sha256"] == _sha(
        paths.feature_contract
    )


def test_temporary_verified_controlled_grant_reaches_real_gate_and_reads_after_success(
    tmp_path, monkeypatch
):
    source, grant, outcome = _controlled_source_and_grant(tmp_path)
    paths, manifest_path = _real_method_lock_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        loader_module,
        "_SOURCE_REGISTRY_BY_ID",
        MappingProxyType({source.resource_id: source}),
    )
    monkeypatch.setattr(
        loader_module,
        "_CONTROLLED_GRANTS_BY_ID",
        MappingProxyType({grant.source_id: grant}),
    )
    events = []
    real_reader = loader_module._read_outcome_bytes

    def record_read(path, expected_sha256):
        assert path == outcome
        events.append("outcome-read")
        return real_reader(path, expected_sha256)

    monkeypatch.setattr(loader_module, "_read_outcome_bytes", record_read)
    loaded = load_outcome_table_after_gate(
        source.resource_id,
        manifest_path=manifest_path,
        method_lock_paths=paths,
    )

    assert events == ["outcome-read"]
    assert loaded.sha256 == source.sha256
    assert tuple(loaded.frame.columns) == (
        "participant_id",
        "meal_id",
        "glucose_iAUC_2h",
        "tg_6h_rise",
    )
    assert loaded.frame.shape == (1, 4)


def test_controlled_outcome_digest_tampering_is_refused(
    tmp_path, monkeypatch
):
    source, grant, outcome = _controlled_source_and_grant(tmp_path)
    paths, manifest_path = _real_method_lock_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        loader_module,
        "_SOURCE_REGISTRY_BY_ID",
        MappingProxyType({source.resource_id: source}),
    )
    monkeypatch.setattr(
        loader_module,
        "_CONTROLLED_GRANTS_BY_ID",
        MappingProxyType({grant.source_id: grant}),
    )
    outcome.write_bytes(outcome.read_bytes() + b"p1,m1,2.0,0.4\n")
    with pytest.raises(PredictorLoadError, match="SHA-256|unique"):
        load_outcome_table_after_gate(
            source.resource_id,
            manifest_path=manifest_path,
            method_lock_paths=paths,
        )


def test_controlled_outcome_duplicate_person_meal_key_is_refused_after_digest_passes(
    tmp_path, monkeypatch
):
    source, grant, outcome = _controlled_source_and_grant(tmp_path)
    outcome.write_bytes(outcome.read_bytes() + b"p1,m1,2.0,0.4\n")
    source = replace(source, sha256=_sha(outcome))
    grant = replace(grant, sha256=source.sha256)
    paths, manifest_path = _real_method_lock_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        loader_module,
        "_SOURCE_REGISTRY_BY_ID",
        MappingProxyType({source.resource_id: source}),
    )
    monkeypatch.setattr(
        loader_module,
        "_CONTROLLED_GRANTS_BY_ID",
        MappingProxyType({grant.source_id: grant}),
    )
    with pytest.raises(PredictorLoadError, match="participant/meal key.*unique"):
        load_outcome_table_after_gate(
            source.resource_id,
            manifest_path=manifest_path,
            method_lock_paths=paths,
        )


@pytest.mark.parametrize("tamper", ["config", "data_dictionary"])
def test_config_or_grant_evidence_tamper_fails_before_outcome_bytes(
    tmp_path, monkeypatch, tamper
):
    source, grant, _ = _controlled_source_and_grant(tmp_path)
    paths, manifest_path = _real_method_lock_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        loader_module,
        "_SOURCE_REGISTRY_BY_ID",
        MappingProxyType({source.resource_id: source}),
    )
    monkeypatch.setattr(
        loader_module,
        "_CONTROLLED_GRANTS_BY_ID",
        MappingProxyType({grant.source_id: grant}),
    )
    if tamper == "config":
        paths.person_meal_validation_config.write_bytes(
            paths.person_meal_validation_config.read_bytes() + b"\n"
        )
    else:
        assert grant.data_dictionary_path is not None
        dictionary = Path(grant.data_dictionary_path)
        dictionary.write_bytes(dictionary.read_bytes() + b"tamper\n")
    outcome_reads = []
    monkeypatch.setattr(
        loader_module,
        "_read_outcome_bytes",
        lambda *args, **kwargs: outcome_reads.append(True),
    )

    with pytest.raises((PredictorLoadError, OutcomeAccessBlocked)):
        load_outcome_table_after_gate(
            source.resource_id,
            manifest_path=manifest_path,
            method_lock_paths=paths,
        )
    assert outcome_reads == []
