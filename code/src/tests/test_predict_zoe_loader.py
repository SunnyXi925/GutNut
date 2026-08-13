from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest

from gmnps.data_sources import predict_zoe_loader as loader_module
from gmnps.data_sources.predict_zoe_loader import (
    OutcomeAccessBlocked,
    OutcomeTableSpec,
    PredictorLoadError,
    PredictorTableSpec,
    load_frozen_validation_config,
    load_outcome_table_after_gate,
    load_predictor_table,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "code/src/configs/person_meal_validation.yaml"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _predictor_spec(path: Path, **changes) -> PredictorTableSpec:
    values = {
        "path": path,
        "expected_sha256": _sha(path),
        "file_format": "csv",
        "source_id": "official-test-predictors",
        "content_class": "predictor_only",
        "required_columns": ("participant_id", "meal_id", "fiber_g"),
        "allowed_columns": ("participant_id", "meal_id", "fiber_g"),
        "unique_key": ("participant_id", "meal_id"),
    }
    values.update(changes)
    return PredictorTableSpec(**values)


def _outcome_spec(path: Path, **changes) -> OutcomeTableSpec:
    values = {
        "path": path,
        "expected_sha256": _sha(path),
        "file_format": "csv",
        "source_id": "controlled-real-test",
        "real_synthetic_status": "controlled_real",
        "allowed_analytical_role": "direct_validation",
        "access_status": "controlled_granted",
        "required_columns": (
            "participant_id",
            "meal_id",
            "glucose_iAUC_2h",
            "tg_6h_rise",
        ),
        "unique_key": ("participant_id", "meal_id"),
        "endpoint_units": {
            "glucose_iAUC_2h": "mmol_L_hour",
            "tg_6h_rise": "mmol_L",
        },
    }
    values.update(changes)
    return OutcomeTableSpec(**values)


def _manifest(path: Path, *, config_sha256: str) -> Path:
    manifest_path = path / "method_lock_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "person_meal_validation_config_sha256": config_sha256,
                "release_registry_snapshot": {"snapshot_sha256": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_predictor_loader_hashes_exact_bytes_and_validates_schema_and_keys(tmp_path):
    path = tmp_path / "predictors.csv"
    _write_csv(
        path,
        [
            {"participant_id": "p1", "meal_id": "m1", "fiber_g": 4.5},
            {"participant_id": "p2", "meal_id": "m1", "fiber_g": 7.0},
        ],
    )
    loaded = load_predictor_table(_predictor_spec(path))

    assert loaded.sha256 == _sha(path)
    assert loaded.source_id == "official-test-predictors"
    assert loaded.frame.loc[0, "participant_id"] == "p1"
    assert loaded.frame.shape == (2, 3)

    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(PredictorLoadError, match="SHA-256"):
        load_predictor_table(_predictor_spec(path, expected_sha256=loaded.sha256))


@pytest.mark.parametrize(
    "rows,match",
    [
        (
            [
                {"participant_id": "p1", "meal_id": "m1", "fiber_g": 1.0},
                {"participant_id": "p1", "meal_id": "m1", "fiber_g": 2.0},
            ],
            "unique",
        ),
        (
            [{"participant_id": "p1", "meal_id": "m1", "label": 1}],
            "schema|column",
        ),
    ],
)
def test_predictor_loader_rejects_duplicate_keys_or_schema_drift(tmp_path, rows, match):
    path = tmp_path / "predictors.csv"
    _write_csv(path, rows)
    with pytest.raises(PredictorLoadError, match=match):
        load_predictor_table(_predictor_spec(path))


def test_non_predictor_classification_is_refused_before_file_open(tmp_path, monkeypatch):
    path = tmp_path / "forbidden.csv"
    path.write_bytes(b"participant_id,meal_id,outcome\np1,m1,99\n")
    spec = _predictor_spec(
        path,
        content_class="outcome_bearing",
        required_columns=("participant_id",),
        allowed_columns=("participant_id",),
        unique_key=("participant_id",),
    )
    opened = []

    def forbidden_read(_path, _label):
        opened.append(_path)
        raise AssertionError("outcome-bearing bytes were opened")

    monkeypatch.setattr(loader_module, "_read_immutable_bytes", forbidden_read)
    with pytest.raises(OutcomeAccessBlocked, match="predictor-only"):
        load_predictor_table(spec)
    assert opened == []


def test_frozen_config_preregisters_endpoints_units_windows_missingness_and_splits():
    frozen = load_frozen_validation_config(CONFIG_PATH)
    payload = frozen.payload

    assert frozen.sha256 == sha256(CONFIG_PATH.read_bytes()).hexdigest()
    assert payload["validation_embargo"] is True
    endpoints = payload["endpoints"]
    assert [(row["name"], row["role"]) for row in endpoints] == [
        ("glucose_iAUC_2h", "primary"),
        ("tg_6h_rise", "primary"),
        ("c_peptide_iAUC_2h", "secondary"),
    ]
    assert [(row["unit"], row["window_hours"]) for row in endpoints] == [
        ("mmol_L_hour", [0, 2]),
        ("mmol_L", [0, 6]),
        ("nmol_L_hour", [0, 2]),
    ]
    assert payload["missingness"]["outcome_imputation"] == "forbidden"
    assert payload["missingness"]["predictor_fit_scope"] == "development_fold_only"
    assert payload["split"]["primary_unit"] == "participant"
    assert payload["split"]["family_twin_grouping"] is True
    assert payload["seeds"] == {
        "outer_split": 1729,
        "inner_cv": 2718,
        "bootstrap": 31415,
        "permutation": 16180,
    }


def test_frozen_config_missing_modified_or_wrong_hash_fails_closed(tmp_path):
    missing = tmp_path / "missing.yaml"
    with pytest.raises(PredictorLoadError, match="missing|unreadable"):
        load_frozen_validation_config(missing)

    expected = _sha(CONFIG_PATH)
    changed = tmp_path / "changed.yaml"
    changed.write_bytes(CONFIG_PATH.read_bytes() + b"\n")
    with pytest.raises(PredictorLoadError, match="config.*SHA-256"):
        load_frozen_validation_config(changed, expected_sha256=expected)


@pytest.mark.parametrize(
    "status,role",
    [
        ("synthetic_local", "synthetic_stress_test"),
        ("aggregate_public", "aggregate_supporting_evidence"),
        ("controlled_real", "controlled_eligibility_assessment"),
    ],
)
def test_synthetic_aggregate_or_unapproved_controlled_data_cannot_reach_gate_or_open(
    tmp_path,
    monkeypatch,
    status,
    role,
):
    outcome = tmp_path / "forbidden_outcome.csv"
    _write_csv(
        outcome,
        [
            {
                "participant_id": "p1",
                "meal_id": "m1",
                "glucose_iAUC_2h": 1.0,
                "tg_6h_rise": 0.2,
            }
        ],
    )
    spec = _outcome_spec(
        outcome,
        real_synthetic_status=status,
        allowed_analytical_role=role,
        access_status=(
            "controlled_not_granted" if status == "controlled_real" else "public"
        ),
    )
    events = []
    monkeypatch.setattr(
        loader_module,
        "validate_method_lock_manifest",
        lambda *args, **kwargs: events.append("gate"),
    )
    monkeypatch.setattr(
        loader_module,
        "_read_immutable_bytes",
        lambda *args, **kwargs: events.append("read"),
    )

    with pytest.raises(OutcomeAccessBlocked):
        load_outcome_table_after_gate(
            spec,
            manifest_path=tmp_path / "missing-manifest.json",
            method_lock_paths=object(),
        )
    assert events == []


def test_caller_cannot_reclassify_trusted_synthetic_source_as_controlled_real(
    tmp_path,
    monkeypatch,
):
    disguised = tmp_path / "disguised.csv"
    disguised.write_bytes(b"participant_id,meal_id,glucose_iAUC_2h,tg_6h_rise\np1,m1,1,2\n")
    spec = _outcome_spec(
        disguised,
        source_id="local_predict1_synthetic_glucose",
        real_synthetic_status="controlled_real",
        allowed_analytical_role="direct_validation",
        access_status="controlled_granted",
    )
    events = []
    monkeypatch.setattr(
        loader_module,
        "_read_immutable_bytes",
        lambda *args, **kwargs: events.append("read"),
    )
    monkeypatch.setattr(
        loader_module,
        "validate_method_lock_manifest",
        lambda *args, **kwargs: events.append("gate"),
    )

    with pytest.raises(OutcomeAccessBlocked, match="trusted|registry|approved"):
        load_outcome_table_after_gate(
            spec,
            manifest_path=tmp_path / "missing-manifest.json",
            method_lock_paths=object(),
        )
    assert events == []


def test_real_outcome_bytes_are_read_only_after_gate_and_config_revalidation(
    tmp_path,
    monkeypatch,
):
    outcome = tmp_path / "controlled_outcome.csv"
    _write_csv(
        outcome,
        [
            {
                "participant_id": "p1",
                "meal_id": "m1",
                "glucose_iAUC_2h": 1.0,
                "tg_6h_rise": 0.2,
            }
        ],
    )
    config = tmp_path / "person_meal_validation.yaml"
    config.write_bytes(CONFIG_PATH.read_bytes())
    paths = type("Paths", (), {"person_meal_validation_config": config})()
    manifest_path = _manifest(tmp_path, config_sha256=_sha(config))
    events = []
    real_read = loader_module._read_immutable_bytes

    def record_gate(manifest, supplied_paths, *, expected_release_registry_sha256):
        assert supplied_paths is paths
        assert expected_release_registry_sha256 == "a" * 64
        events.append("gate")

    def record_read(path, label):
        if path == outcome:
            events.append("outcome-read")
        return real_read(path, label)

    monkeypatch.setattr(loader_module, "validate_method_lock_manifest", record_gate)
    monkeypatch.setattr(loader_module, "_require_trusted_outcome_source", lambda spec: None)
    monkeypatch.setattr(loader_module, "_read_immutable_bytes", record_read)
    loaded = load_outcome_table_after_gate(
        _outcome_spec(outcome),
        manifest_path=manifest_path,
        method_lock_paths=paths,
    )

    assert events == ["gate", "outcome-read"]
    assert loaded.frame.shape == (1, 4)


@pytest.mark.parametrize("tamper", ["config", "manifest_hash", "missing_config"])
def test_config_tampering_fails_before_outcome_open(tmp_path, monkeypatch, tamper):
    outcome = tmp_path / "controlled_outcome.csv"
    _write_csv(
        outcome,
        [
            {
                "participant_id": "p1",
                "meal_id": "m1",
                "glucose_iAUC_2h": 1.0,
                "tg_6h_rise": 0.2,
            }
        ],
    )
    config = tmp_path / "person_meal_validation.yaml"
    config.write_bytes(CONFIG_PATH.read_bytes())
    locked_hash = _sha(config)
    manifest_path = _manifest(
        tmp_path,
        config_sha256=("0" * 64 if tamper == "manifest_hash" else locked_hash),
    )
    if tamper == "config":
        config.write_bytes(config.read_bytes() + b"\n")
    elif tamper == "missing_config":
        config.unlink()
    paths = type("Paths", (), {"person_meal_validation_config": config})()
    outcome_reads = []
    real_read = loader_module._read_immutable_bytes

    monkeypatch.setattr(
        loader_module,
        "validate_method_lock_manifest",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(loader_module, "_require_trusted_outcome_source", lambda spec: None)

    def guarded_read(path, label):
        if path == outcome:
            outcome_reads.append(path)
            raise AssertionError("outcome opened after config tamper")
        return real_read(path, label)

    monkeypatch.setattr(loader_module, "_read_immutable_bytes", guarded_read)
    with pytest.raises((PredictorLoadError, OutcomeAccessBlocked)):
        load_outcome_table_after_gate(
            _outcome_spec(outcome),
            manifest_path=manifest_path,
            method_lock_paths=paths,
        )
    assert outcome_reads == []


def test_outcome_schema_units_and_person_meal_uniqueness_are_enforced_after_gate(
    tmp_path,
    monkeypatch,
):
    outcome = tmp_path / "controlled_outcome.csv"
    _write_csv(
        outcome,
        [
            {
                "participant_id": "p1",
                "meal_id": "m1",
                "glucose_iAUC_2h": 1.0,
                "tg_6h_rise": 0.2,
            },
            {
                "participant_id": "p1",
                "meal_id": "m1",
                "glucose_iAUC_2h": 1.1,
                "tg_6h_rise": 0.3,
            },
        ],
    )
    config = tmp_path / "person_meal_validation.yaml"
    config.write_bytes(CONFIG_PATH.read_bytes())
    paths = type("Paths", (), {"person_meal_validation_config": config})()
    manifest_path = _manifest(tmp_path, config_sha256=_sha(config))
    monkeypatch.setattr(
        loader_module,
        "validate_method_lock_manifest",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(loader_module, "_require_trusted_outcome_source", lambda spec: None)

    with pytest.raises(PredictorLoadError, match="unique"):
        load_outcome_table_after_gate(
            _outcome_spec(outcome),
            manifest_path=manifest_path,
            method_lock_paths=paths,
        )

    wrong_units = replace(
        _outcome_spec(outcome),
        endpoint_units={"glucose_iAUC_2h": "mg_dL_hour", "tg_6h_rise": "mmol_L"},
    )
    with pytest.raises(PredictorLoadError, match="unit"):
        load_outcome_table_after_gate(
            wrong_units,
            manifest_path=manifest_path,
            method_lock_paths=paths,
        )
