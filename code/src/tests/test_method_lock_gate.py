from __future__ import annotations

from dataclasses import fields, replace
from hashlib import sha256
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from gmnps.scoring.attribute_calibration import fit_beta_normalization
from gmnps.validation.method_lock_gate import (
    MethodLockArtifactPaths,
    MethodLockError,
    canonical_ids_sha256,
    canonical_json_sha256,
    generate_method_lock_manifest,
    validate_method_lock_manifest,
)
from gmnps.validation import method_lock_gate as gate_module


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / "docs/methods/method_lock_manifest.schema.json"
GATE_PATH = ROOT / "code/src/gmnps/validation/method_lock_gate.py"
CANONICAL_RELEASES = [f"FNDDS {start}-{start + 1}" for start in range(2001, 2018, 2)]
ARTIFACT_KEYS = (
    "official_fcs",
    "food_metadata",
    "baseline_attribute_points",
    "food_exposures",
    "effective_attribute_weights",
)
IMPLEMENTATION_HASHES = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))[
    "properties"
]["implementation_source_sha256"]["const"]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _registry_sha256(paths: MethodLockArtifactPaths) -> str:
    return sha256(paths.release_registry.read_bytes()).hexdigest()


def _generate(paths: MethodLockArtifactPaths, *, cohort_id: str = "development-v1"):
    return generate_method_lock_manifest(
        paths,
        beta_fit_cohort_id=cohort_id,
        expected_release_registry_sha256=_registry_sha256(paths),
    )


def _validate(manifest, paths: MethodLockArtifactPaths) -> None:
    validate_method_lock_manifest(
        manifest,
        paths,
        expected_release_registry_sha256=_registry_sha256(paths),
    )


def _normalization_payload(
    fit_ids: list[str],
    nutrient_order: list[str],
    values: list[list[float]],
) -> dict[str, object]:
    state = fit_beta_normalization(
        pd.DataFrame(values, index=fit_ids, columns=nutrient_order)
    )
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


def _self_consistent_but_underived_state(
    fit_ids: list[str], nutrient_order: list[str]
) -> dict[str, object]:
    payload = {
        "nutrient_order": nutrient_order,
        "median_values": [0.0 for _ in nutrient_order],
        "scale_values": [1.0 for _ in nutrient_order],
        "scale_method_values": ["unit" for _ in nutrient_order],
        "fit_n": len(fit_ids),
        "fit_id_sha256": canonical_ids_sha256(fit_ids),
        "method_version": "median_mad_iqr_sd_v1",
        "temperature": 2.0,
    }
    payload["state_fingerprint"] = canonical_json_sha256(payload)
    return payload


def _fixture(
    tmp_path: Path,
    *,
    monkeypatch,
    overlap: bool = False,
    approved: bool = True,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_paths = {}
    for key in ARTIFACT_KEYS:
        path = tmp_path / f"{key}.csv"
        path.write_bytes(f"canonical-{key}\n".encode())
        source_paths[key] = path
    input_manifest = tmp_path / "input_manifest.json"
    input_manifest.write_bytes(b'{"production_label":"production"}\n')
    linkage = tmp_path / "food_source_linkage.json"
    linkage.write_bytes(b'{"linkage":"verified"}\n')

    fit_ids = ["development-2", "development-1"]
    scoring_ids = ["held-out-1", "development-1" if overlap else "held-out-2"]
    nutrients = ["fiber", "potassium"]
    development_beta = tmp_path / "development_beta.json"
    scoring_beta = tmp_path / "scoring_beta.json"
    _write_json(
        development_beta,
        {
            "schema_version": "gmnps-canonical-beta-v1",
            "participant_ids": fit_ids,
            "nutrient_order": nutrients,
            "values": [[0.0, 0.0], [1.0, 1.0]],
        },
    )
    _write_json(
        scoring_beta,
        {
            "schema_version": "gmnps-canonical-beta-v1",
            "participant_ids": scoring_ids,
            "nutrient_order": nutrients,
            "values": [[0.5, 0.5], [1.5, 1.5]],
        },
    )
    normalization_state = tmp_path / "normalization_state.json"
    _write_json(
        normalization_state,
        _normalization_payload(
            fit_ids,
            nutrients,
            [[0.0, 0.0], [1.0, 1.0]],
        ),
    )

    entry = {
        "bundle_id": "verified-test-bundle-v1",
        "fndds_releases": CANONICAL_RELEASES,
        "artifact_sha256": {
            key: sha256(path.read_bytes()).hexdigest()
            for key, path in source_paths.items()
        },
        "food_source_linkage_sha256": sha256(linkage.read_bytes()).hexdigest(),
        "nutrient_units": {"fiber": "g_per_100_kcal", "potassium": "mg_per_100_kcal"},
        "exposure_basis": "per_100_kcal",
    }
    registry = {
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
            "artifact_sha256_keys": list(ARTIFACT_KEYS),
            "digest_algorithm": "sha256",
        },
        "approved_artifacts": [entry] if approved else [],
    }
    registry_path = tmp_path / "release_registry.json"
    _write_json(registry_path, registry)
    config = tmp_path / "person_meal_validation.yaml"
    config.write_text("schema_version: person-meal-validation-v1\n", encoding="utf-8")
    feature_contract = tmp_path / "feature_contract.json"
    feature_contract.write_bytes(
        b'{"generated_stage":"pre-outcome_predictor_only","schema_version":"testing-only"}\n'
    )
    schema = tmp_path / "method_lock_manifest.schema.json"
    schema.write_bytes(SCHEMA_PATH.read_bytes())
    paths = MethodLockArtifactPaths(
        method_lock_schema=schema,
        person_meal_validation_config=config,
        feature_contract=feature_contract,
        release_registry=registry_path,
        gate_implementation=GATE_PATH,
        development_beta=development_beta,
        normalization_state=normalization_state,
        scoring_beta=scoring_beta,
        official_fcs=source_paths["official_fcs"],
        food_metadata=source_paths["food_metadata"],
        baseline_attribute_points=source_paths["baseline_attribute_points"],
        food_exposures=source_paths["food_exposures"],
        effective_attribute_weights=source_paths["effective_attribute_weights"],
        input_manifest=input_manifest,
        food_source_linkage=linkage,
    )
    monkeypatch.setattr(gate_module, "_TRUSTED_METHOD_LOCK_SCHEMA_PATH", schema)
    monkeypatch.setattr(gate_module, "_TRUSTED_RELEASE_REGISTRY_PATH", registry_path)
    implementation_root = tmp_path / "repository_root"
    for relative_path in IMPLEMENTATION_HASHES:
        source = ROOT / relative_path
        destination = implementation_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    monkeypatch.setattr(gate_module, "_REPOSITORY_ROOT", implementation_root)
    return paths, entry


def test_artifact_contract_is_explicit_and_has_no_outcome_or_label_path():
    names = {field.name for field in fields(MethodLockArtifactPaths)}
    assert names == {
        "method_lock_schema",
        "person_meal_validation_config",
        "feature_contract",
        "release_registry",
        "gate_implementation",
        "development_beta",
        "normalization_state",
        "scoring_beta",
        "official_fcs",
        "food_metadata",
        "baseline_attribute_points",
        "food_exposures",
        "effective_attribute_weights",
        "input_manifest",
        "food_source_linkage",
    }
    assert not any("outcome" in name or "label" in name for name in names)


def test_canonical_hashes_are_order_invariant_and_fit_ids_match_phase1_contract():
    assert canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256(
        {"a": 1, "b": 2}
    )
    assert canonical_ids_sha256(["b", "a"]) == sha256(b"a\nb").hexdigest()
    with pytest.raises(MethodLockError, match="duplicate"):
        canonical_ids_sha256(["same", "same"])


def test_generator_binds_external_schema_config_registry_gate_and_approved_entry(
    tmp_path,
    monkeypatch,
):
    paths, entry = _fixture(tmp_path, monkeypatch=monkeypatch)
    manifest = _generate(paths)
    assert manifest["method_lock_schema_sha256"] == sha256(
        paths.method_lock_schema.read_bytes()
    ).hexdigest()
    assert manifest["person_meal_validation_config_sha256"] == sha256(
        paths.person_meal_validation_config.read_bytes()
    ).hexdigest()
    assert manifest["feature_contract_sha256"] == sha256(
        paths.feature_contract.read_bytes()
    ).hexdigest()
    assert manifest["method_lock_gate_implementation_sha256"] == sha256(
        GATE_PATH.read_bytes()
    ).hexdigest()
    snapshot = manifest["release_registry_snapshot"]
    assert snapshot["snapshot_sha256"] == sha256(paths.release_registry.read_bytes()).hexdigest()
    assert snapshot["schema_version"] == "fcs2-fndds-release-registry-schema-v1"
    assert snapshot["registry_version"] == "fcs2-fndds-release-registry-v1"
    assert snapshot["digest_algorithm"] == "sha256"
    assert snapshot["canonical_release_set"] == CANONICAL_RELEASES
    assert snapshot["approved_entry"] == {
        **entry,
        "entry_sha256": canonical_json_sha256(entry),
    }
    assert manifest["validation_embargo"] is True
    _validate(manifest, paths)


def test_generator_rejects_self_consistent_state_not_fitted_from_development_beta(
    tmp_path,
    monkeypatch,
):
    paths, _ = _fixture(tmp_path, monkeypatch=monkeypatch)
    _write_json(
        paths.normalization_state,
        _self_consistent_but_underived_state(
            ["development-2", "development-1"],
            ["fiber", "potassium"],
        ),
    )

    with pytest.raises(MethodLockError, match="normalization.*(median|scale|derived)"):
        _generate(paths)


def test_normalization_fit_receives_development_beta_only(tmp_path, monkeypatch):
    paths, _ = _fixture(tmp_path, monkeypatch=monkeypatch)
    observed_indexes = []
    real_fit = fit_beta_normalization

    def recording_fit(frame):
        observed_indexes.append(frame.index.tolist())
        return real_fit(frame)

    monkeypatch.setattr(gate_module, "fit_beta_normalization", recording_fit)
    _generate(paths)

    assert observed_indexes == [["development-2", "development-1"]]
    assert not any(identifier.startswith("held-out") for identifier in observed_indexes[0])


def test_schema_implementation_lock_includes_runner():
    assert "code/src/scripts/run_attribute_gmnps.py" in IMPLEMENTATION_HASHES


@pytest.mark.parametrize("relative_path", tuple(IMPLEMENTATION_HASHES))
def test_generator_rehashes_every_locked_implementation_file(
    tmp_path,
    monkeypatch,
    relative_path,
):
    paths, _ = _fixture(tmp_path, monkeypatch=monkeypatch)
    implementation_path = gate_module._REPOSITORY_ROOT / relative_path
    implementation_path.write_bytes(implementation_path.read_bytes() + b"tamper\n")

    with pytest.raises(MethodLockError, match="implementation.*(hash|mismatch)"):
        _generate(paths)


def test_generator_rejects_missing_locked_implementation_file(tmp_path, monkeypatch):
    paths, _ = _fixture(tmp_path, monkeypatch=monkeypatch)
    relative_path = next(iter(IMPLEMENTATION_HASHES))
    (gate_module._REPOSITORY_ROOT / relative_path).unlink()

    with pytest.raises(MethodLockError, match="implementation.*(missing|unreadable)"):
        _generate(paths)


@pytest.mark.parametrize("mutation", ["missing_schema_entry", "extra_schema_entry"])
def test_generator_rejects_missing_or_extra_schema_implementation_entry(
    tmp_path,
    monkeypatch,
    mutation,
):
    paths, _ = _fixture(tmp_path, monkeypatch=monkeypatch)
    schema = json.loads(paths.method_lock_schema.read_text(encoding="utf-8"))
    locked = schema["properties"]["implementation_source_sha256"]["const"]
    if mutation == "missing_schema_entry":
        locked.pop(next(iter(locked)))
    else:
        locked["code/src/scripts/unexpected_runner.py"] = "0" * 64
    _write_json(paths.method_lock_schema, schema)

    with pytest.raises(MethodLockError, match="implementation.*(missing|extra|set)"):
        _generate(paths)


def test_generator_fails_closed_on_overlap_or_missing_approved_entry(
    tmp_path,
    monkeypatch,
):
    overlap_paths, _ = _fixture(
        tmp_path / "overlap",
        monkeypatch=monkeypatch,
        overlap=True,
    )
    with pytest.raises(MethodLockError, match="overlap"):
        _generate(overlap_paths)

    empty_paths, _ = _fixture(
        tmp_path / "empty",
        monkeypatch=monkeypatch,
        approved=False,
    )
    with pytest.raises(MethodLockError, match="approved"):
        _generate(empty_paths)


@pytest.mark.parametrize(
    "field",
    [
        "method_lock_schema",
        "person_meal_validation_config",
        "feature_contract",
        "release_registry",
        "scoring_beta",
        "food_exposures",
    ],
)
def test_validator_fails_closed_after_any_bound_artifact_is_modified(
    tmp_path,
    monkeypatch,
    field,
):
    paths, _ = _fixture(tmp_path, monkeypatch=monkeypatch)
    manifest = _generate(paths)
    getattr(paths, field).write_bytes(getattr(paths, field).read_bytes() + b"tamper\n")
    with pytest.raises(MethodLockError):
        _validate(manifest, paths)


def test_validator_rejects_manifest_schema_failure_and_gate_hash_tampering(
    tmp_path,
    monkeypatch,
):
    paths, _ = _fixture(tmp_path, monkeypatch=monkeypatch)
    manifest = _generate(paths)

    missing = dict(manifest)
    missing.pop("validation_embargo")
    with pytest.raises(MethodLockError, match="schema|required"):
        _validate(missing, paths)

    missing_contract = dict(manifest)
    missing_contract.pop("feature_contract_sha256")
    with pytest.raises(MethodLockError, match="schema|required|feature"):
        _validate(missing_contract, paths)

    changed = dict(manifest)
    changed["method_lock_gate_implementation_sha256"] = "0" * 64
    with pytest.raises(MethodLockError, match="gate|implementation"):
        _validate(changed, paths)


def test_feature_contract_is_required_and_tamper_evident(tmp_path, monkeypatch):
    paths, _ = _fixture(tmp_path, monkeypatch=monkeypatch)
    manifest = _generate(paths)

    paths.feature_contract.unlink()
    with pytest.raises(MethodLockError, match="feature.contract.*(missing|unreadable)"):
        _generate(paths)

    paths.feature_contract.write_bytes(b'{"testing_only":"changed"}\n')
    with pytest.raises(MethodLockError, match="feature.contract.*hash|bound artifacts"):
        _validate(manifest, paths)


def test_generator_does_not_create_a_run_level_manifest_or_open_outcomes(
    tmp_path,
    monkeypatch,
):
    paths, _ = _fixture(tmp_path, monkeypatch=monkeypatch)
    outcome = tmp_path / "participant_outcomes.csv"
    outcome.write_bytes(b"participant_id,outcome\nforbidden,99\n")
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path):
        if path.resolve() == outcome.resolve():
            raise AssertionError("method-lock gate attempted to open an outcome artifact")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    generated = _generate(paths)
    assert generated["manifest_schema_version"] == "attribute-gmnps-method-lock-v1"
    assert not (tmp_path / "method_lock_manifest.json").exists()


def test_alternate_self_consistent_schema_or_registry_path_is_rejected(
    tmp_path,
    monkeypatch,
):
    paths, _ = _fixture(tmp_path, monkeypatch=monkeypatch)
    alternate_registry = tmp_path / "alternate_registry.json"
    alternate_registry.write_bytes(paths.release_registry.read_bytes())
    with pytest.raises(MethodLockError, match="trust root"):
        generate_method_lock_manifest(
            replace(paths, release_registry=alternate_registry),
            beta_fit_cohort_id="development-v1",
            expected_release_registry_sha256=_registry_sha256(paths),
        )

    alternate_schema = tmp_path / "alternate_schema.json"
    alternate_schema.write_bytes(paths.method_lock_schema.read_bytes())
    with pytest.raises(MethodLockError, match="trust root"):
        generate_method_lock_manifest(
            replace(paths, method_lock_schema=alternate_schema),
            beta_fit_cohort_id="development-v1",
            expected_release_registry_sha256=_registry_sha256(paths),
        )


def test_task2_generator_and_validator_require_explicit_registry_snapshot_hash(
    tmp_path,
    monkeypatch,
):
    paths, _ = _fixture(tmp_path, monkeypatch=monkeypatch)
    expected_registry_sha256 = sha256(paths.release_registry.read_bytes()).hexdigest()

    generate_parameter = inspect.signature(generate_method_lock_manifest).parameters[
        "expected_release_registry_sha256"
    ]
    validate_parameter = inspect.signature(validate_method_lock_manifest).parameters[
        "expected_release_registry_sha256"
    ]
    assert generate_parameter.default is inspect.Parameter.empty
    assert validate_parameter.default is inspect.Parameter.empty

    manifest = generate_method_lock_manifest(
        paths,
        beta_fit_cohort_id="development-v1",
        expected_release_registry_sha256=expected_registry_sha256,
    )
    validate_method_lock_manifest(
        manifest,
        paths,
        expected_release_registry_sha256=expected_registry_sha256,
    )

    with pytest.raises(MethodLockError, match="expected.*registry|snapshot"):
        generate_method_lock_manifest(
            paths,
            beta_fit_cohort_id="development-v1",
            expected_release_registry_sha256="0" * 64,
        )


def test_task2_manifest_writer_is_atomic_and_leaves_no_output_on_gate_failure(
    tmp_path,
    monkeypatch,
):
    paths, _ = _fixture(tmp_path / "inputs", monkeypatch=monkeypatch)
    output = tmp_path / "results/phase2/method_lock_manifest.json"
    expected_registry_sha256 = sha256(paths.release_registry.read_bytes()).hexdigest()

    written = gate_module.write_method_lock_manifest(
        paths,
        output_path=output,
        beta_fit_cohort_id="development-v1",
        expected_release_registry_sha256=expected_registry_sha256,
        expected_person_meal_validation_config_sha256=sha256(
            paths.person_meal_validation_config.read_bytes()
        ).hexdigest(),
    )
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == written

    output.unlink()
    paths.person_meal_validation_config.write_bytes(
        paths.person_meal_validation_config.read_bytes() + b"tamper\n"
    )
    with pytest.raises(MethodLockError):
        gate_module.write_method_lock_manifest(
            paths,
            output_path=output,
            beta_fit_cohort_id="development-v1",
            expected_release_registry_sha256=expected_registry_sha256,
            expected_person_meal_validation_config_sha256="0" * 64,
        )
    assert not output.exists()


def test_manifest_writer_rolls_back_destination_if_temp_cleanup_fails_after_link(
    tmp_path,
    monkeypatch,
):
    paths, _ = _fixture(tmp_path / "inputs", monkeypatch=monkeypatch)
    output = tmp_path / "results/phase2/method_lock_manifest.json"
    expected_registry_sha256 = sha256(paths.release_registry.read_bytes()).hexdigest()
    original_unlink = Path.unlink
    faulted = False

    def fail_first_temp_cleanup(path, *args, **kwargs):
        nonlocal faulted
        if not faulted and path.suffix == ".tmp" and output.exists():
            faulted = True
            raise OSError("injected temporary cleanup failure after link")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_first_temp_cleanup)
    with pytest.raises(MethodLockError, match="atomically"):
        gate_module.write_method_lock_manifest(
            paths,
            output_path=output,
            beta_fit_cohort_id="development-v1",
            expected_release_registry_sha256=expected_registry_sha256,
            expected_person_meal_validation_config_sha256=sha256(
                paths.person_meal_validation_config.read_bytes()
            ).hexdigest(),
        )

    assert faulted is True
    assert not output.exists()
