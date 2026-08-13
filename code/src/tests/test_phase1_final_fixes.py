from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pytest

from gmnps.scoring.attribute_gmnps import (
    FCS2_FNDDS_REGISTRY_DIGEST_ALGORITHM,
    FCS2_FNDDS_REGISTRY_SCHEMA_VERSION,
    FCS2_FNDDS_REGISTRY_VERSION,
    AttributeGMNPSConfig,
    FoodAttributeBundle,
    ProductionBundleAttestation,
    fit_attribute_gmnps,
    load_food_attribute_bundle_from_bytes,
    load_release_registry_snapshot,
    score_attribute_gmnps,
)
from gmnps.scoring.fcs2_attribute_rules import (
    FCS2_RULES,
    NOT_CALCULATED,
    aggregate_domains,
    ratio_gate_passes_from_exposures,
    select_domain_attributes,
)
from test_attribute_gmnps import (
    ACTIVE_ATTRIBUTES,
    baseline_points,
    development_beta,
    food_exposures,
    make_bundle,
    score_beta,
    smoke_config,
)


def _effective_weights(food_ids=("food_1", "food_2"), *, dairy=(True, False)):
    weights = pd.DataFrame(
        {
            attribute: [float(FCS2_RULES[attribute].weight)] * len(food_ids)
            for attribute in ACTIVE_ATTRIBUTES
        },
        index=pd.Index(food_ids, name="food_id"),
    )
    for food_id, is_dairy in zip(food_ids, dairy, strict=True):
        weights.loc[food_id, "unsaturated_to_saturated_fat_ratio"] = (
            0.5 if is_dairy else 1.0
        )
    return weights


def test_dairy_effective_weight_survives_zero_and_nonzero_end_to_end_oracle():
    bundle = make_bundle()
    metadata = bundle.food_metadata.copy()
    metadata["is_dairy"] = [True, False]
    points = bundle.baseline_points.copy()
    points.loc[:, [
        "unsaturated_to_saturated_fat_ratio",
        "fiber_to_carbohydrate_ratio",
        "potassium_to_sodium_ratio",
    ]] = 0.0
    weights = _effective_weights()
    bundle = replace(
        bundle,
        baseline_points=points,
        food_metadata=metadata,
        effective_attribute_weights=weights,
        fingerprint="",
    )
    model = fit_attribute_gmnps(development_beta(), smoke_config())

    zero = score_attribute_gmnps(model, score_beta(("zero",)), bundle).individual_food
    np.testing.assert_allclose(zero["GMNPS_score"], zero["FCS2"], atol=1e-8)

    beta = score_beta(("nonzero",))
    beta.loc[:, "Fatty acids, total saturated (g)"] = 4.0
    result = score_attribute_gmnps(model, beta, bundle)
    ratio_rows = result.attribute_attribution.query(
        "attribute == 'unsaturated_to_saturated_fat_ratio'"
    ).set_index("food_id")
    assert ratio_rows.loc["food_1", "effective_attribute_weight"] == 0.5
    assert ratio_rows.loc["food_1", "active_domain_denominator"] == 2.5
    assert bool(ratio_rows.loc["food_1", "calculated"])
    assert bool(ratio_rows.loc["food_1", "active"])
    assert bool(ratio_rows.loc["food_1", "baseline_selected"])
    assert bool(ratio_rows.loc["food_1", "personalized_selected"])
    point_delta = ratio_rows["attribute_point_delta"]
    scale = 99.0 / 47.1
    expected_dairy = 0.5 * point_delta["food_1"] / 2.5 * scale
    expected_non_dairy = point_delta["food_2"] / 3.0 * scale
    actual = result.individual_food.set_index("food_id")["GMNPS_delta"]
    assert actual["food_1"] == pytest.approx(expected_dairy)
    assert actual["food_2"] == pytest.approx(expected_non_dairy)
    assert result.run_manifest["effective_attribute_weights_sha256"] == bundle.effective_weights_fingerprint


def test_ratio_gate_is_recomputed_from_exposures_and_binds_canonical_sentinel():
    bundle = make_bundle()
    assert ratio_gate_passes_from_exposures(
        "unsaturated_to_saturated_fat_ratio", bundle.food_exposures.loc["food_1"]
    )

    failed_exposures = bundle.food_exposures.copy()
    failed_exposures.loc["food_1", "Total Fat (g)"] = 0.0
    failed_exposures.attrs["basis"] = "per_100_kcal"
    with pytest.raises(ValueError, match="gate fail.*NOT_CALCULATED"):
        replace(bundle, food_exposures=failed_exposures, fingerprint="")

    pass_with_sentinel = bundle.baseline_points.copy()
    pass_with_sentinel["fiber_to_carbohydrate_ratio"] = pass_with_sentinel[
        "fiber_to_carbohydrate_ratio"
    ].astype(object)
    pass_with_sentinel.loc["food_1", "fiber_to_carbohydrate_ratio"] = NOT_CALCULATED
    with pytest.raises(ValueError, match="gate pass.*finite point"):
        replace(bundle, baseline_points=pass_with_sentinel, fingerprint="")


def test_dairy_weight_changes_only_denominator_not_other_ratio_point_delta():
    scores = {
        "unsaturated_to_saturated_fat_ratio": 0.0,
        "fiber_to_carbohydrate_ratio": 0.0,
        "potassium_to_sodium_ratio": 6.0,
    }
    weights = {
        "unsaturated_to_saturated_fat_ratio": 0.5,
        "fiber_to_carbohydrate_ratio": 1.0,
        "potassium_to_sodium_ratio": 1.0,
    }
    selected = select_domain_attributes(scores, "nutrient_ratios", weights)
    assert selected.active_weight_denominator == 2.5
    assert aggregate_domains(scores, effective_attribute_weights=weights)[
        "nutrient_ratios"
    ] == pytest.approx(6.0 / 2.5)


def _production_bytes():
    foods = ("food_1", "food_2")
    metadata = pd.DataFrame(
        {
            "food_id": foods,
            "food_name": ["Food one", "Food two"],
            "food_group": ["Dairy", "Other"],
            "is_dairy": [True, False],
            "FCS2": [40.0, 70.0],
        }
    )
    points = baseline_points(foods).reset_index()
    exposures = food_exposures(foods).reset_index()
    weights = _effective_weights(foods).reset_index()
    source_bytes = {
        "food_metadata": metadata.to_csv(index=False, lineterminator="\n").encode(),
        "baseline_attribute_points": points.to_csv(index=False, lineterminator="\n").encode(),
        "food_exposures": exposures.to_csv(index=False, lineterminator="\n").encode(),
        "effective_attribute_weights": weights.to_csv(index=False, lineterminator="\n").encode(),
    }
    artifact_hashes = {
        "official_fcs": sha256(source_bytes["food_metadata"]).hexdigest(),
        **{name: sha256(data).hexdigest() for name, data in source_bytes.items()},
    }
    canonical = [
        f"FNDDS {start}-{start + 1}"
        for start in range(2001, 2018, 2)
    ]
    units = {
        column: "source_unit_per_100_kcal"
        for column in food_exposures(foods).columns
    }
    approved_entry = {
        "bundle_id": "verified-fixture-v1",
        "fndds_releases": canonical,
        "artifact_sha256": artifact_hashes,
        "food_source_linkage_sha256": "6" * 64,
        "nutrient_units": units,
        "exposure_basis": "per_100_kcal",
    }
    registry = {
        "schema_version": FCS2_FNDDS_REGISTRY_SCHEMA_VERSION,
        "registry_version": FCS2_FNDDS_REGISTRY_VERSION,
        "digest_algorithm": FCS2_FNDDS_REGISTRY_DIGEST_ALGORITHM,
        "canonical_release_set": canonical,
        "approved_artifact_entry_schema": {
            "required_fields": list(approved_entry),
            "artifact_sha256_keys": list(artifact_hashes),
            "digest_algorithm": "sha256",
        },
        "approved_artifacts": [approved_entry],
    }
    registry_bytes = (json.dumps(registry, sort_keys=True) + "\n").encode()
    manifest = {
        "files": {
            name: {"sha256": sha256(data).hexdigest()}
            for name, data in source_bytes.items()
        },
        "fndds_releases": canonical,
        "production_label": "production",
        "registry_version": FCS2_FNDDS_REGISTRY_VERSION,
        "release_registry_snapshot_sha256": sha256(registry_bytes).hexdigest(),
        "food_source_linkage_sha256": "6" * 64,
        "exposure_basis": "per_100_kcal",
        "nutrient_units": units,
        "not_calculated_serialization": {
            "schema_version": "gmnps-not-calculated-csv-v1",
            "token": "__GMNPS_NOT_CALCULATED_V1__",
            "baseline_attribute_points_mask": [],
        },
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    return source_bytes, manifest_bytes, registry_bytes


def _empty_registry_bytes(registry_bytes):
    registry = json.loads(registry_bytes)
    registry["approved_artifacts"] = []
    return (json.dumps(registry, sort_keys=True) + "\n").encode()


def test_registry_parser_const_validates_top_level_schema_and_digest_algorithm():
    _, _, registry_bytes = _production_bytes()
    snapshot = load_release_registry_snapshot(registry_bytes)
    assert snapshot.schema_version == FCS2_FNDDS_REGISTRY_SCHEMA_VERSION
    assert snapshot.digest_algorithm == FCS2_FNDDS_REGISTRY_DIGEST_ALGORITHM

    for field, value in (
        ("schema_version", "caller-defined-schema"),
        ("digest_algorithm", "sha512"),
    ):
        registry = json.loads(registry_bytes)
        registry[field] = value
        with pytest.raises(ValueError, match=field.replace("_", " ")):
            load_release_registry_snapshot(json.dumps(registry).encode())


def test_production_loader_attests_same_immutable_bytes_and_rejects_bypasses(
    tmp_path, monkeypatch
):
    from gmnps.scoring import attribute_gmnps

    source_bytes, manifest_bytes, registry_bytes = _production_bytes()
    trusted_registry = tmp_path / "trusted_release_registry.json"
    trusted_registry.write_bytes(registry_bytes)
    monkeypatch.setattr(
        attribute_gmnps, "_TRUSTED_REGISTRY_PATH", trusted_registry
    )
    bundle = load_food_attribute_bundle_from_bytes(
        source_bytes,
        input_manifest_bytes=manifest_bytes,
    )
    assert bundle.production_attestation is not None
    assert bundle.production_attestation.approved_bundle_id == "verified-fixture-v1"
    bundle.validate()

    with pytest.raises(ValueError, match="trusted registry path"):
        load_food_attribute_bundle_from_bytes(
            source_bytes,
            input_manifest_bytes=manifest_bytes,
            release_registry_bytes=registry_bytes + b"\n",
        )

    tampered = bundle.food_exposures.copy()
    tampered.iloc[0, 0] += 1.0
    tampered.attrs["basis"] = "per_100_kcal"
    with pytest.raises(ValueError, match="attestation|parsed content"):
        replace(bundle, food_exposures=tampered, fingerprint="")

    changed_source = dict(source_bytes)
    changed_source["food_exposures"] += b"\n"
    with pytest.raises(ValueError, match="SHA-256 mismatch|same immutable bytes"):
        load_food_attribute_bundle_from_bytes(
            changed_source,
            input_manifest_bytes=manifest_bytes,
        )

    with pytest.raises(ValueError, match="verified byte loader"):
        FoodAttributeBundle(
            official_fcs=bundle.official_fcs,
            baseline_points=bundle.baseline_points,
            food_exposures=bundle.food_exposures,
            effective_attribute_weights=bundle.effective_attribute_weights,
            food_metadata=bundle.food_metadata,
            fndds_releases=bundle.fndds_releases,
            source_hashes=bundle.source_hashes,
            nutrient_units=bundle.nutrient_units,
            exposure_basis=bundle.exposure_basis,
            reconstruction_status=bundle.reconstruction_status,
            production_label="production",
            registry_version=bundle.registry_version,
            release_registry_snapshot_sha256=bundle.release_registry_snapshot_sha256,
            release_registry_canonical_release_set=bundle.release_registry_canonical_release_set,
            food_source_linkage_sha256=bundle.food_source_linkage_sha256,
        )


def test_imported_private_token_and_self_authored_attestation_cannot_bypass_empty_registry(
    tmp_path, monkeypatch
):
    from gmnps.scoring import attribute_gmnps

    source_bytes, manifest_bytes, alternate_registry_bytes = _production_bytes()
    trusted_registry = tmp_path / "trusted_release_registry.json"
    trusted_registry.write_bytes(alternate_registry_bytes)
    monkeypatch.setattr(attribute_gmnps, "_TRUSTED_REGISTRY_PATH", trusted_registry)
    legitimate = load_food_attribute_bundle_from_bytes(
        source_bytes,
        input_manifest_bytes=manifest_bytes,
    )
    original = legitimate.production_attestation
    assert original is not None
    self_authored = ProductionBundleAttestation(
        source_bytes=original.source_bytes,
        input_manifest_bytes=original.input_manifest_bytes,
        release_registry_bytes=alternate_registry_bytes,
        source_byte_sha256=original.source_byte_sha256,
        parsed_content_fingerprints=original.parsed_content_fingerprints,
        input_manifest_sha256=original.input_manifest_sha256,
        release_registry_snapshot_sha256=original.release_registry_snapshot_sha256,
        approved_bundle_id=original.approved_bundle_id,
        approved_entry_json=original.approved_entry_json,
        fingerprint=original.fingerprint,
    )

    trusted_registry.write_bytes(_empty_registry_bytes(alternate_registry_bytes))
    with pytest.raises(ValueError, match="trusted release registry|snapshot|no approved"):
        FoodAttributeBundle(
            official_fcs=legitimate.official_fcs,
            baseline_points=legitimate.baseline_points,
            food_exposures=legitimate.food_exposures,
            effective_attribute_weights=legitimate.effective_attribute_weights,
            food_metadata=legitimate.food_metadata,
            fndds_releases=legitimate.fndds_releases,
            source_hashes=legitimate.source_hashes,
            nutrient_units=legitimate.nutrient_units,
            exposure_basis=legitimate.exposure_basis,
            reconstruction_status=legitimate.reconstruction_status,
            production_label=legitimate.production_label,
            registry_version=legitimate.registry_version,
            release_registry_snapshot_sha256=legitimate.release_registry_snapshot_sha256,
            release_registry_canonical_release_set=(
                legitimate.release_registry_canonical_release_set
            ),
            food_source_linkage_sha256=legitimate.food_source_linkage_sha256,
            production_attestation=self_authored,
            _factory_token=attribute_gmnps._VERIFIED_BYTE_LOADER_TOKEN,
        )


def test_production_bundle_validation_rereads_trusted_registry_after_gate(
    tmp_path, monkeypatch
):
    from gmnps.scoring import attribute_gmnps

    source_bytes, manifest_bytes, registry_bytes = _production_bytes()
    trusted_registry = tmp_path / "trusted_release_registry.json"
    trusted_registry.write_bytes(registry_bytes)
    monkeypatch.setattr(attribute_gmnps, "_TRUSTED_REGISTRY_PATH", trusted_registry)
    bundle = load_food_attribute_bundle_from_bytes(
        source_bytes,
        input_manifest_bytes=manifest_bytes,
    )

    trusted_registry.write_bytes(_empty_registry_bytes(registry_bytes))
    with pytest.raises(ValueError, match="trusted release registry|snapshot|no approved"):
        bundle.validate()


def test_production_model_score_rechecks_registry_bound_during_fit(
    tmp_path, monkeypatch
):
    from gmnps.scoring import attribute_gmnps

    source_bytes, manifest_bytes, registry_bytes = _production_bytes()
    trusted_registry = tmp_path / "trusted_release_registry.json"
    trusted_registry.write_bytes(registry_bytes)
    monkeypatch.setattr(attribute_gmnps, "_TRUSTED_REGISTRY_PATH", trusted_registry)
    bundle = load_food_attribute_bundle_from_bytes(
        source_bytes,
        input_manifest_bytes=manifest_bytes,
    )
    expected_snapshot = sha256(registry_bytes).hexdigest()
    model = fit_attribute_gmnps(
        development_beta(),
        AttributeGMNPSConfig(
            expected_release_registry_sha256=expected_snapshot,
        ),
    )

    trusted_registry.write_bytes(_empty_registry_bytes(registry_bytes))
    with pytest.raises(ValueError, match="trusted release registry|snapshot"):
        score_attribute_gmnps(model, score_beta(("held_out",)), bundle)
