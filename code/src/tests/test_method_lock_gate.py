from __future__ import annotations

from dataclasses import fields, replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _normalization_payload(fit_ids: list[str], nutrient_order: list[str]) -> dict[str, object]:
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
    _write_json(normalization_state, _normalization_payload(fit_ids, nutrients))

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
    schema = tmp_path / "method_lock_manifest.schema.json"
    schema.write_bytes(SCHEMA_PATH.read_bytes())
    paths = MethodLockArtifactPaths(
        method_lock_schema=schema,
        person_meal_validation_config=config,
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
    return paths, entry


def test_artifact_contract_is_explicit_and_has_no_outcome_or_label_path():
    names = {field.name for field in fields(MethodLockArtifactPaths)}
    assert names == {
        "method_lock_schema",
        "person_meal_validation_config",
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
    manifest = generate_method_lock_manifest(paths, beta_fit_cohort_id="development-v1")
    assert manifest["method_lock_schema_sha256"] == sha256(
        paths.method_lock_schema.read_bytes()
    ).hexdigest()
    assert manifest["person_meal_validation_config_sha256"] == sha256(
        paths.person_meal_validation_config.read_bytes()
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
    validate_method_lock_manifest(manifest, paths)


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
        generate_method_lock_manifest(overlap_paths, beta_fit_cohort_id="development-v1")

    empty_paths, _ = _fixture(
        tmp_path / "empty",
        monkeypatch=monkeypatch,
        approved=False,
    )
    with pytest.raises(MethodLockError, match="approved"):
        generate_method_lock_manifest(empty_paths, beta_fit_cohort_id="development-v1")


@pytest.mark.parametrize(
    "field",
    [
        "method_lock_schema",
        "person_meal_validation_config",
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
    manifest = generate_method_lock_manifest(paths, beta_fit_cohort_id="development-v1")
    getattr(paths, field).write_bytes(getattr(paths, field).read_bytes() + b"tamper\n")
    with pytest.raises(MethodLockError):
        validate_method_lock_manifest(manifest, paths)


def test_validator_rejects_manifest_schema_failure_and_gate_hash_tampering(
    tmp_path,
    monkeypatch,
):
    paths, _ = _fixture(tmp_path, monkeypatch=monkeypatch)
    manifest = generate_method_lock_manifest(paths, beta_fit_cohort_id="development-v1")

    missing = dict(manifest)
    missing.pop("validation_embargo")
    with pytest.raises(MethodLockError, match="schema|required"):
        validate_method_lock_manifest(missing, paths)

    changed = dict(manifest)
    changed["method_lock_gate_implementation_sha256"] = "0" * 64
    with pytest.raises(MethodLockError, match="gate|implementation"):
        validate_method_lock_manifest(changed, paths)


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
    generated = generate_method_lock_manifest(paths, beta_fit_cohort_id="development-v1")
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
        )

    alternate_schema = tmp_path / "alternate_schema.json"
    alternate_schema.write_bytes(paths.method_lock_schema.read_bytes())
    with pytest.raises(MethodLockError, match="trust root"):
        generate_method_lock_manifest(
            replace(paths, method_lock_schema=alternate_schema),
            beta_fit_cohort_id="development-v1",
        )
