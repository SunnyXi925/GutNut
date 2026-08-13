"""Public, provenance-bound API for attribute-level GMNPS scoring."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from io import BytesIO
import json
from math import isfinite
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from gmnps.scoring.anchored import AnchoredScoringConfig, score_individual_foods
from gmnps.scoring.attribute_calibration import (
    ATTRIBUTE_POINT_FRACTION_MODES,
    BetaNormalizationState,
    attribute_response,
    calibrate_attribute_points,
    fit_beta_normalization,
)
from gmnps.scoring.attribute_recomposition import (
    FINAL_DEVIATION_CAP_MODES,
    compose_personalized_fcs,
    decompose_official_baseline,
    recompute_personalized_domains,
)
from gmnps.scoring.fcs2_attribute_mapping import (
    PRIMARY_ATTRIBUTE_MAPPINGS,
    PRIMARY_MAPPING_VERSION,
    SENSITIVITY_ATTRIBUTE_MAPPINGS,
    reconstruction_status_table,
)
from gmnps.scoring.fcs2_attribute_rules import (
    FCS2_RULES,
    NOT_CALCULATED,
    NotCalculated,
    ratio_gate_passes_from_exposures,
)
from gmnps.scoring.masks import PRIMARY_MASK_VERSION


ATTRIBUTE_GMNPS_SCORING_VERSION = "attribute-gmnps-v1"
NOT_CALCULATED_TOKEN = "__GMNPS_NOT_CALCULATED_V1__"
NOT_CALCULATED_SCHEMA_VERSION = "gmnps-not-calculated-csv-v1"
PRIMARY_METHOD = "attribute_recomposition"
LEGACY_METHOD = "legacy_final_score_offset"
_ALLOWED_METHODS = frozenset({PRIMARY_METHOD, LEGACY_METHOD})
_ACTIVE_ATTRIBUTES = tuple(name for name, rule in FCS2_RULES.items() if rule.active)
_ALL_DOMAINS = tuple(dict.fromkeys(FCS2_RULES[name].domain for name in _ACTIVE_ATTRIBUTES))
_TARGET_ATTRIBUTES = tuple(
    dict.fromkeys(
        row.attribute for row in PRIMARY_ATTRIBUTE_MAPPINGS if row.role == "effect"
    )
)
PRIMARY_RECOMPUTED_DOMAINS = tuple(
    domain
    for domain in _ALL_DOMAINS
    if any(FCS2_RULES[attribute].domain == domain for attribute in _TARGET_ATTRIBUTES)
)
_REQUIRED_SOURCE_HASHES = frozenset(
    {
        "official_fcs",
        "food_metadata",
        "baseline_attribute_points",
        "food_exposures",
        "effective_attribute_weights",
        "input_manifest",
    }
)
_HEX = frozenset("0123456789abcdef")
_PRIMARY_INDIVIDUAL_COLUMNS = (
    "individual_id",
    "food_id",
    "food_name",
    "food_group",
    "FCS2",
    "GMNPS_delta",
    "GMNPS_score",
    "MAC_delta",
    "LIPID_delta",
    "channel_interaction_delta",
    "top_positive_drivers",
    "top_negative_drivers",
    "driver_basis",
    "mask_version",
    "mapping_version",
    "scoring_version",
    "method_role",
)
_LEGACY_INDIVIDUAL_COLUMNS = tuple(
    "legacy_top_positive_nutrient_drivers"
    if column == "top_positive_drivers"
    else "legacy_top_negative_nutrient_drivers"
    if column == "top_negative_drivers"
    else column
    for column in _PRIMARY_INDIVIDUAL_COLUMNS
)
_PRODUCTION_ARTIFACT_HASH_KEYS = (
    "official_fcs",
    "food_metadata",
    "baseline_attribute_points",
    "food_exposures",
    "effective_attribute_weights",
)
_TRUSTED_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/fcs2_fndds_release_registry.json"
)
FCS2_FNDDS_REGISTRY_SCHEMA_VERSION = "fcs2-fndds-release-registry-schema-v1"
FCS2_FNDDS_REGISTRY_VERSION = "fcs2-fndds-release-registry-v1"
FCS2_FNDDS_REGISTRY_DIGEST_ALGORITHM = "sha256"
_CANONICAL_PRODUCTION_RELEASES = tuple(
    f"FNDDS {start}-{start + 1}" for start in range(2001, 2018, 2)
)
_VERIFIED_BYTE_LOADER_TOKEN = object()


@dataclass(frozen=True)
class ReleaseRegistrySnapshot:
    """One immutable run-level release-registry byte snapshot."""

    raw_bytes: bytes
    snapshot_sha256: str
    schema_version: str
    registry_version: str
    digest_algorithm: str
    canonical_release_set: tuple[str, ...]
    approved_artifacts_json: str

    @property
    def approved_artifacts(self) -> tuple[dict[str, object], ...]:
        return tuple(json.loads(self.approved_artifacts_json))


def load_release_registry_snapshot(raw: bytes) -> ReleaseRegistrySnapshot:
    """Parse and validate a registry without treating its current bytes as method code."""

    if not isinstance(raw, bytes):
        raise TypeError("release registry snapshot must be immutable bytes")
    try:
        registry = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("release registry snapshot is unreadable") from error
    if not isinstance(registry, dict):
        raise ValueError("release registry snapshot must be an object")
    required = {
        "schema_version",
        "registry_version",
        "digest_algorithm",
        "canonical_release_set",
        "approved_artifact_entry_schema",
        "approved_artifacts",
    }
    if set(registry) != required:
        raise ValueError("release registry snapshot schema is invalid")
    if registry["schema_version"] != FCS2_FNDDS_REGISTRY_SCHEMA_VERSION:
        raise ValueError("release registry schema version is not supported by this method")
    if registry["registry_version"] != FCS2_FNDDS_REGISTRY_VERSION:
        raise ValueError("release registry version is not supported by this method")
    if registry["digest_algorithm"] != FCS2_FNDDS_REGISTRY_DIGEST_ALGORITHM:
        raise ValueError("release registry digest algorithm must be sha256")
    canonical = registry["canonical_release_set"]
    if (
        not isinstance(canonical, list)
        or tuple(canonical) != _CANONICAL_PRODUCTION_RELEASES
    ):
        raise ValueError("release registry canonical release set is invalid")
    entries = registry["approved_artifacts"]
    if not isinstance(entries, list):
        raise ValueError("release registry approved_artifacts must be a list")
    entry_schema = registry["approved_artifact_entry_schema"]
    if not isinstance(entry_schema, dict) or set(entry_schema) != {
        "required_fields",
        "artifact_sha256_keys",
        "digest_algorithm",
    }:
        raise ValueError("release registry approved entry schema is invalid")
    if entry_schema["digest_algorithm"] != FCS2_FNDDS_REGISTRY_DIGEST_ALGORITHM:
        raise ValueError("release registry approved entry digest algorithm must be sha256")
    required_fields_value = entry_schema["required_fields"]
    artifact_keys_value = entry_schema["artifact_sha256_keys"]
    if not isinstance(required_fields_value, list) or not isinstance(
        artifact_keys_value, list
    ):
        raise ValueError("release registry approved entry schema is invalid")
    required_entry_fields = set(required_fields_value)
    if len(required_entry_fields) != len(required_fields_value):
        raise ValueError("release registry approved entry schema is invalid")
    if required_entry_fields != {
        "bundle_id",
        "fndds_releases",
        "artifact_sha256",
        "food_source_linkage_sha256",
        "nutrient_units",
        "exposure_basis",
    }:
        raise ValueError("release registry approved entry schema is invalid")
    if len(set(artifact_keys_value)) != len(artifact_keys_value) or set(
        artifact_keys_value
    ) != set(_PRODUCTION_ARTIFACT_HASH_KEYS):
        raise ValueError("release registry artifact digest schema is invalid")
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != required_entry_fields:
            raise ValueError("release registry contains an invalid approved artifact entry")
        if not isinstance(entry["bundle_id"], str) or not entry["bundle_id"]:
            raise ValueError("release registry bundle_id is invalid")
        if not isinstance(entry["fndds_releases"], list) or tuple(
            entry["fndds_releases"]
        ) != _CANONICAL_PRODUCTION_RELEASES:
            raise ValueError("release registry approved release set is invalid")
        if not isinstance(entry["artifact_sha256"], dict) or set(
            entry["artifact_sha256"]
        ) != set(_PRODUCTION_ARTIFACT_HASH_KEYS):
            raise ValueError("release registry artifact digest keys are invalid")
        if any(not _is_sha256(value) for value in entry["artifact_sha256"].values()):
            raise ValueError("release registry contains an invalid artifact digest")
        if not _is_sha256(entry["food_source_linkage_sha256"]):
            raise ValueError("release registry contains an invalid linkage digest")
        if not isinstance(entry["nutrient_units"], dict) or not entry[
            "nutrient_units"
        ]:
            raise ValueError("release registry nutrient units are invalid")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(unit, str)
            or not unit
            for name, unit in entry["nutrient_units"].items()
        ):
            raise ValueError("release registry nutrient units are invalid")
        if entry["exposure_basis"] != "per_100_kcal":
            raise ValueError("release registry exposure basis is invalid")
    return ReleaseRegistrySnapshot(
        raw_bytes=raw,
        snapshot_sha256=sha256(raw).hexdigest(),
        schema_version=str(registry["schema_version"]),
        registry_version=str(registry["registry_version"]),
        digest_algorithm=str(registry["digest_algorithm"]),
        canonical_release_set=tuple(canonical),
        approved_artifacts_json=json.dumps(
            entries,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _read_trusted_release_registry_snapshot(
    *,
    expected_sha256: str,
    require_approved_entry: bool,
) -> ReleaseRegistrySnapshot:
    """Read and validate the repository trust root for this single operation."""

    if not _is_sha256(expected_sha256):
        raise ValueError(
            "expected_release_registry_sha256 must be a lowercase SHA-256 digest"
        )
    try:
        trusted_bytes = _TRUSTED_REGISTRY_PATH.read_bytes()
    except OSError as error:
        raise ValueError("trusted release registry is missing or unreadable") from error
    snapshot = load_release_registry_snapshot(trusted_bytes)
    if snapshot.snapshot_sha256 != expected_sha256:
        raise ValueError(
            "trusted release registry snapshot does not match the pre-label/method-lock "
            "expected SHA-256"
        )
    if require_approved_entry and not snapshot.approved_artifacts:
        raise ValueError(
            "no approved production bundle is registered; Phase 2 must verify official "
            "FNDDS 2001-2018 digests and food/source linkage"
        )
    return snapshot


def _json_scalar(value: object) -> object:
    if value is NOT_CALCULATED:
        return {"not_calculated": True}
    if isinstance(value, NotCalculated):
        raise ValueError("noncanonical NotCalculated value")
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isnan(number):
            return {"nan": True}
        if not isfinite(number):
            raise ValueError("fingerprinted values must be finite")
        return number
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"unsupported fingerprint value: {type(value).__name__}")


def _frame_payload(frame: pd.DataFrame | pd.Series) -> dict[str, object]:
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    if isinstance(frame.index, pd.MultiIndex):
        index = [[_json_scalar(item) for item in row] for row in frame.index.tolist()]
    else:
        index = [[_json_scalar(item)] for item in frame.index.tolist()]
    return {
        "index_names": [_json_scalar(name) for name in frame.index.names],
        "index": index,
        "columns": [_json_scalar(column) for column in frame.columns],
        "values": [
            [_json_scalar(item) for item in row]
            for row in frame.to_numpy(dtype=object).tolist()
        ],
    }


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX




def _mapping_rows(version: str):
    if version == PRIMARY_MAPPING_VERSION:
        return PRIMARY_ATTRIBUTE_MAPPINGS
    try:
        return SENSITIVITY_ATTRIBUTE_MAPPINGS[version]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown reviewed mapping version: {version}") from error


def _effect_nutrients_by_channel(mapping_version: str) -> dict[str, tuple[str, ...]]:
    """Return the exact reviewed effect nutrients for each supported channel."""

    grouped: dict[str, list[str]] = {"MAC": [], "LIPID": []}
    for row in _mapping_rows(mapping_version):
        if row.role != "effect":
            continue
        if row.channel not in grouped:
            raise ValueError(
                f"reviewed effect mapping channel must be MAC or LIPID: {row.channel}"
            )
        if row.nutrient not in grouped[row.channel]:
            grouped[row.channel].append(row.nutrient)
    if not all(grouped.values()):
        raise ValueError("reviewed mapping must contain both MAC and LIPID effect nutrients")
    overlap = set(grouped["MAC"]) & set(grouped["LIPID"])
    if overlap:
        raise ValueError(f"reviewed MAC/LIPID effect nutrients overlap: {sorted(overlap)}")
    return {channel: tuple(nutrients) for channel, nutrients in grouped.items()}


def _target_domains(version: str) -> tuple[str, ...]:
    targets = {row.attribute for row in _mapping_rows(version) if row.role == "effect"}
    return tuple(
        domain
        for domain in _ALL_DOMAINS
        if any(FCS2_RULES[attribute].domain == domain for attribute in targets)
    )


@dataclass(frozen=True)
class AttributeGMNPSConfig:
    """Named method choices only; numerical values live in locked Task 3/4 registries."""

    method: str = PRIMARY_METHOD
    mapping_version: str = PRIMARY_MAPPING_VERSION
    attribute_point_mode: str = "primary"
    final_cap_mode: str = "primary"
    recomputed_domains: tuple[str, ...] = PRIMARY_RECOMPUTED_DOMAINS
    scoring_version: str = ATTRIBUTE_GMNPS_SCORING_VERSION
    mask_version: str = PRIMARY_MASK_VERSION
    method_role: str | None = None
    development_smoke_test: bool = False
    expected_release_registry_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.method not in _ALLOWED_METHODS:
            raise ValueError(f"method must be one of {sorted(_ALLOWED_METHODS)}")
        _mapping_rows(self.mapping_version)
        if self.attribute_point_mode not in ATTRIBUTE_POINT_FRACTION_MODES:
            raise ValueError("attribute_point_mode must be primary, low, or high")
        if self.final_cap_mode not in FINAL_DEVIATION_CAP_MODES:
            raise ValueError("final_cap_mode must be primary, low, or high")
        if not isinstance(self.recomputed_domains, tuple):
            raise ValueError("recomputed_domains must be an immutable tuple")
        expected_domains = _target_domains(self.mapping_version)
        if (
            self.mapping_version != PRIMARY_MAPPING_VERSION
            and self.recomputed_domains == PRIMARY_RECOMPUTED_DOMAINS
        ):
            object.__setattr__(self, "recomputed_domains", expected_domains)
        elif self.recomputed_domains != expected_domains:
            raise ValueError(
                "recomputed_domains must exactly cover the reviewed mapping targets"
            )
        if self.scoring_version != ATTRIBUTE_GMNPS_SCORING_VERSION:
            raise ValueError("scoring_version is not the locked attribute GMNPS version")
        if self.mask_version != PRIMARY_MASK_VERSION:
            raise ValueError("mask_version is not the locked expert-reviewed mask")
        if not isinstance(self.development_smoke_test, bool):
            raise ValueError("development_smoke_test must be boolean")
        if (
            self.expected_release_registry_sha256 is not None
            and not _is_sha256(self.expected_release_registry_sha256)
        ):
            raise ValueError(
                "expected_release_registry_sha256 must be a lowercase SHA-256 digest"
            )
        if self.method == LEGACY_METHOD:
            if self.method_role != "sensitivity":
                raise ValueError("legacy_final_score_offset is available only as sensitivity")
            if not self.development_smoke_test:
                raise ValueError("production rejects legacy_final_score_offset")
        else:
            expected_role = (
                "primary"
                if self.mapping_version == PRIMARY_MAPPING_VERSION
                and self.attribute_point_mode == "primary"
                and self.final_cap_mode == "primary"
                else "sensitivity"
            )
            if self.method_role is None:
                object.__setattr__(self, "method_role", expected_role)
            elif self.method_role != expected_role:
                raise ValueError(
                    f"the selected named modes require method_role {expected_role}"
                )


def _validate_string_index(index: pd.Index, label: str) -> None:
    if isinstance(index, pd.MultiIndex) or not index.is_unique or index.hasnans:
        raise ValueError(f"{label} must have unique, nonmissing one-level labels")
    if any(not isinstance(value, str) or not value.strip() for value in index):
        raise ValueError(f"{label} labels must be nonempty strings")


def _finite_frame(frame: pd.DataFrame, label: str, *, nonnegative: bool = False) -> None:
    if frame.empty or not frame.columns.is_unique:
        raise ValueError(f"{label} must be nonempty with unique columns")
    if any(not isinstance(column, str) or not column for column in frame.columns):
        raise ValueError(f"{label} columns must be nonempty strings")
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} values must be finite numbers") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{label} values must be finite numbers")
    if nonnegative and np.any(values < 0.0):
        raise ValueError(f"{label} values must be nonnegative")


def _release_kind(releases: tuple[str, ...]) -> str:
    if not releases or any(not isinstance(value, str) or not value.strip() for value in releases):
        raise ValueError("fndds_releases must contain nonempty release labels")
    if releases == _CANONICAL_PRODUCTION_RELEASES:
        return "production"
    if releases == ("FNDDS 2021-2023",):
        return "non-production"
    raise ValueError(
        "production requires the exact trusted FCS2-aligned FNDDS 2001-2018 "
        "canonical release set; "
        "only FNDDS 2021-2023 is recognized for development smoke tests"
    )


def _matching_registry_entries(
    *,
    snapshot: ReleaseRegistrySnapshot,
    fndds_releases: tuple[str, ...],
    source_hashes: Mapping[str, str],
    food_source_linkage_sha256: str,
    nutrient_units: Mapping[str, str],
    exposure_basis: str,
) -> list[dict[str, object]]:
    actual_artifacts = {
        key: source_hashes[key] for key in _PRODUCTION_ARTIFACT_HASH_KEYS
    }
    return [
        entry
        for entry in snapshot.approved_artifacts
        if tuple(entry["fndds_releases"]) == fndds_releases
        and entry["artifact_sha256"] == actual_artifacts
        and entry["food_source_linkage_sha256"] == food_source_linkage_sha256
        and entry["nutrient_units"] == dict(nutrient_units)
        and entry["exposure_basis"] == exposure_basis
    ]


@dataclass(frozen=True)
class ProductionBundleAttestation:
    """Source bytes and canonical parsed-content proofs minted by the loader."""

    source_bytes: tuple[tuple[str, bytes], ...]
    input_manifest_bytes: bytes
    release_registry_bytes: bytes
    source_byte_sha256: tuple[tuple[str, str], ...]
    parsed_content_fingerprints: tuple[tuple[str, str], ...]
    input_manifest_sha256: str
    release_registry_snapshot_sha256: str
    approved_bundle_id: str | None
    approved_entry_json: str | None
    fingerprint: str


def _attestation_fingerprint(
    *,
    source_byte_sha256: tuple[tuple[str, str], ...],
    parsed_content_fingerprints: tuple[tuple[str, str], ...],
    input_manifest_sha256: str,
    release_registry_snapshot_sha256: str,
    approved_bundle_id: str | None,
    approved_entry_json: str | None,
) -> str:
    return _fingerprint(
        {
            "source_byte_sha256": list(source_byte_sha256),
            "parsed_content_fingerprints": list(parsed_content_fingerprints),
            "input_manifest_sha256": input_manifest_sha256,
            "release_registry_snapshot_sha256": release_registry_snapshot_sha256,
            "approved_bundle_id": approved_bundle_id,
            "approved_entry_json": approved_entry_json,
        }
    )


def _validate_approved_production_bundle(
    bundle: "FoodAttributeBundle",
) -> ReleaseRegistrySnapshot:
    snapshot = _read_trusted_release_registry_snapshot(
        expected_sha256=bundle.release_registry_snapshot_sha256,
        require_approved_entry=True,
    )
    if bundle._factory_token is not _VERIFIED_BYTE_LOADER_TOKEN:
        raise ValueError(
            "production FoodAttributeBundle must be created by the verified byte loader"
        )
    if bundle.production_attestation is None:
        raise ValueError("production bundle is missing verified byte attestation")
    if (
        bundle.production_attestation.release_registry_snapshot_sha256
        != snapshot.snapshot_sha256
    ):
        raise ValueError("production attestation trusted registry snapshot hash mismatch")
    matches = _matching_registry_entries(
        snapshot=snapshot,
        fndds_releases=bundle.fndds_releases,
        source_hashes=bundle.source_hashes,
        food_source_linkage_sha256=bundle.food_source_linkage_sha256,
        nutrient_units=bundle.nutrient_units,
        exposure_basis=bundle.exposure_basis,
    )
    if len(matches) != 1:
        raise ValueError(
            "production bundle does not exactly match one trusted registry entry"
        )
    if bundle.production_attestation.approved_bundle_id != matches[0]["bundle_id"]:
        raise ValueError("production attestation approved bundle ID mismatch")
    canonical_entry = json.dumps(
        matches[0],
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if bundle.production_attestation.approved_entry_json != canonical_entry:
        raise ValueError("production attestation approved registry entry mismatch")
    return snapshot


@dataclass(frozen=True)
class FoodAttributeBundle:
    """Official food baseline and composition inputs bound by a content fingerprint."""

    official_fcs: pd.Series
    baseline_points: pd.DataFrame
    food_exposures: pd.DataFrame
    effective_attribute_weights: pd.DataFrame
    food_metadata: pd.DataFrame
    fndds_releases: tuple[str, ...]
    source_hashes: Mapping[str, str]
    nutrient_units: Mapping[str, str]
    exposure_basis: str
    reconstruction_status: pd.DataFrame
    production_label: str
    registry_version: str
    release_registry_snapshot_sha256: str
    release_registry_canonical_release_set: tuple[str, ...]
    food_source_linkage_sha256: str
    production_attestation: ProductionBundleAttestation | None = None
    fingerprint: str = ""
    _factory_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.validate(check_fingerprint=False)
        expected = self.compute_fingerprint()
        if self.fingerprint:
            if self.fingerprint != expected:
                raise ValueError("food attribute bundle fingerprint does not match its contents")
        else:
            object.__setattr__(self, "fingerprint", expected)

    def compute_fingerprint(self) -> str:
        return _fingerprint(
            {
                "official_fcs": _frame_payload(self.official_fcs),
                "baseline_points": _frame_payload(self.baseline_points),
                "food_exposures": _frame_payload(self.food_exposures),
                "effective_attribute_weights": _frame_payload(
                    self.effective_attribute_weights
                ),
                "food_metadata": _frame_payload(self.food_metadata),
                "fndds_releases": list(self.fndds_releases),
                "source_hashes": dict(sorted(self.source_hashes.items())),
                "nutrient_units": dict(sorted(self.nutrient_units.items())),
                "exposure_basis": self.exposure_basis,
                "reconstruction_status": _frame_payload(self.reconstruction_status),
                "production_label": self.production_label,
                "registry_version": self.registry_version,
                "release_registry_snapshot_sha256": (
                    self.release_registry_snapshot_sha256
                ),
                "release_registry_canonical_release_set": list(
                    self.release_registry_canonical_release_set
                ),
                "food_source_linkage_sha256": self.food_source_linkage_sha256,
                "production_attestation_fingerprint": (
                    None
                    if self.production_attestation is None
                    else self.production_attestation.fingerprint
                ),
            }
        )

    @property
    def effective_weights_fingerprint(self) -> str:
        return _fingerprint(_frame_payload(self.effective_attribute_weights))

    def validate(self, *, check_fingerprint: bool = True) -> None:
        if not isinstance(self.official_fcs, pd.Series) or self.official_fcs.empty:
            raise ValueError("official_fcs must be a nonempty pandas Series")
        _validate_string_index(self.official_fcs.index, "official_fcs food IDs")
        if self.official_fcs.index.name != "food_id":
            raise ValueError("official_fcs index must be named food_id")
        official_values = pd.to_numeric(self.official_fcs, errors="coerce").to_numpy(float)
        if not np.isfinite(official_values).all() or np.any((official_values < 1) | (official_values > 100)):
            raise ValueError("official_fcs values must be finite and between 1 and 100")

        frames = {
            "baseline_points": self.baseline_points,
            "food_exposures": self.food_exposures,
            "effective_attribute_weights": self.effective_attribute_weights,
            "food_metadata": self.food_metadata,
        }
        for label, frame in frames.items():
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                raise ValueError(f"{label} must be a nonempty pandas DataFrame")
            _validate_string_index(frame.index, f"{label} food IDs")
            if frame.index.name != "food_id":
                raise ValueError(f"{label} index must be named food_id")
            if set(frame.index) != set(self.official_fcs.index):
                raise ValueError(f"{label} must contain the exact official food IDs")

        if tuple(self.baseline_points.columns) != _ACTIVE_ATTRIBUTES:
            raise ValueError("baseline_points must contain every active FCS2 attribute in canonical order")
        _finite_frame(self.food_exposures, "food_exposures", nonnegative=True)
        if self.exposure_basis != "per_100_kcal":
            raise ValueError("exposure_basis must be per_100_kcal")
        if self.food_exposures.attrs.get("basis") != self.exposure_basis:
            raise ValueError("food_exposures attrs['basis'] must match exposure_basis")

        required_metadata = {"food_name", "food_group", "is_dairy"}
        missing_metadata = required_metadata - set(self.food_metadata.columns)
        if missing_metadata:
            raise ValueError(f"food_metadata missing required columns: {sorted(missing_metadata)}")
        for column in ("food_name", "food_group"):
            if self.food_metadata[column].isna().any() or any(
                not isinstance(value, str) or not value.strip()
                for value in self.food_metadata[column]
            ):
                raise ValueError(f"food_metadata {column} values must be nonempty strings")
        if any(not isinstance(value, (bool, np.bool_)) for value in self.food_metadata["is_dairy"]):
            raise ValueError("food_metadata is_dairy values must be boolean")
        if "FCS2" in self.food_metadata:
            aligned = self.food_metadata.loc[self.official_fcs.index, "FCS2"].to_numpy(float)
            if not np.array_equal(aligned, official_values):
                raise ValueError("food_metadata FCS2 must exactly match official_fcs")

        if not isinstance(self.fndds_releases, tuple):
            raise ValueError("fndds_releases must be an immutable tuple")
        release_kind = _release_kind(self.fndds_releases)
        if self.production_label != release_kind:
            raise ValueError(
                "production requires FNDDS 2001-2018; FNDDS 2021-2023 must be "
                "labelled non-production"
            )
        if self.registry_version != FCS2_FNDDS_REGISTRY_VERSION:
            raise ValueError("bundle registry_version does not match the trusted registry")
        if not _is_sha256(self.release_registry_snapshot_sha256):
            raise ValueError(
                "release_registry_snapshot_sha256 must be a lowercase SHA-256 digest"
            )
        if self.release_registry_canonical_release_set != _CANONICAL_PRODUCTION_RELEASES:
            raise ValueError("bundle release registry canonical set is invalid")
        if not _is_sha256(self.food_source_linkage_sha256):
            raise ValueError("food_source_linkage_sha256 must be a lowercase SHA-256 digest")

        if not isinstance(self.source_hashes, Mapping):
            raise ValueError("source_hashes must be a mapping")
        missing_hashes = _REQUIRED_SOURCE_HASHES - set(self.source_hashes)
        if missing_hashes:
            raise ValueError(f"source_hashes missing required sources: {sorted(missing_hashes)}")
        if any(not _is_sha256(value) for value in self.source_hashes.values()):
            raise ValueError("source_hashes values must be lowercase SHA-256 digests")

        _finite_frame(
            self.effective_attribute_weights,
            "effective_attribute_weights",
        )
        if tuple(self.effective_attribute_weights.columns) != _ACTIVE_ATTRIBUTES:
            raise ValueError(
                "effective_attribute_weights must contain every active attribute in canonical order"
            )
        if np.any(self.effective_attribute_weights.to_numpy(dtype=float) <= 0.0):
            raise ValueError("effective_attribute_weights must be greater than zero")
        for food_id in self.official_fcs.index:
            dairy = bool(self.food_metadata.loc[food_id, "is_dairy"])
            for attribute in _ACTIVE_ATTRIBUTES:
                expected_weight = float(FCS2_RULES[attribute].weight)
                if attribute == "unsaturated_to_saturated_fat_ratio" and dairy:
                    expected_weight = 0.5
                actual_weight = float(
                    self.effective_attribute_weights.loc[food_id, attribute]
                )
                if actual_weight != expected_weight:
                    raise ValueError(
                        "effective_attribute_weights do not match published per-food "
                        f"weights for {food_id}, {attribute}"
                    )

        ratio_attributes = tuple(
            attribute
            for attribute in _ACTIVE_ATTRIBUTES
            if FCS2_RULES[attribute].kind == "log_ratio"
        )
        for food_id in self.official_fcs.index:
            exposure = self.food_exposures.loc[food_id]
            for attribute in ratio_attributes:
                gate_passes = ratio_gate_passes_from_exposures(attribute, exposure)
                point = self.baseline_points.loc[food_id, attribute]
                if not gate_passes and point is not NOT_CALCULATED:
                    raise ValueError(
                        f"ratio gate fail requires canonical NOT_CALCULATED for {food_id}, {attribute}"
                    )
                if gate_passes and point is NOT_CALCULATED:
                    raise ValueError(
                        f"ratio gate pass requires a finite point for {food_id}, {attribute}"
                    )

        # Task 4 validates bounds, sentinels, weights, and complete domain state.
        decompose_official_baseline(
            self.official_fcs,
            self.baseline_points,
            recomputed_domains=_ALL_DOMAINS,
            effective_attribute_weights=self.effective_attribute_weights,
        )

        if not isinstance(self.nutrient_units, Mapping):
            raise ValueError("nutrient_units must be a mapping")
        if set(self.nutrient_units) != set(self.food_exposures.columns):
            raise ValueError("nutrient_units must exactly cover food_exposures columns")
        if any(not isinstance(value, str) or not value.strip() for value in self.nutrient_units.values()):
            raise ValueError("nutrient_units values must be nonempty strings")

        expected_status = reconstruction_status_table()
        try:
            pd.testing.assert_frame_equal(
                self.reconstruction_status.reset_index(drop=True),
                expected_status,
                check_exact=True,
            )
        except AssertionError as error:
            raise ValueError("reconstruction_status must match the locked Task 2 table") from error

        trusted_snapshot = None
        if release_kind == "production":
            trusted_snapshot = _validate_approved_production_bundle(self)
        if self.production_attestation is not None:
            _validate_bundle_attestation(self, trusted_snapshot=trusted_snapshot)

        if check_fingerprint:
            expected = self.compute_fingerprint()
            if not _is_sha256(self.fingerprint) or self.fingerprint != expected:
                raise ValueError("food attribute bundle fingerprint does not match its contents")


def _parse_json_bytes(raw: bytes, label: str) -> dict[str, object]:
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be a readable JSON object") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _read_indexed_csv_bytes(raw: bytes, index_name: str, label: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(BytesIO(raw), dtype={index_name: str})
    except Exception as error:
        raise ValueError(f"unable to read {label} from immutable bytes") from error
    if index_name not in frame.columns:
        raise ValueError(f"{label} must contain index column {index_name}")
    if frame[index_name].isna().any() or frame[index_name].duplicated().any():
        raise ValueError(f"{label} {index_name} values must be unique and nonmissing")
    frame = frame.set_index(index_name)
    frame.index = frame.index.astype(str)
    return frame.sort_index(kind="mergesort")


def _restore_not_calculated_from_manifest(
    baseline: pd.DataFrame,
    serialization: object,
) -> pd.DataFrame:
    if not isinstance(serialization, dict):
        raise ValueError("input manifest must declare not_calculated_serialization")
    if serialization.get("schema_version") != NOT_CALCULATED_SCHEMA_VERSION:
        raise ValueError("NOT_CALCULATED serialization schema version is invalid")
    if serialization.get("token") != NOT_CALCULATED_TOKEN:
        raise ValueError("NOT_CALCULATED serialization token is invalid")
    mask_rows = serialization.get("baseline_attribute_points_mask")
    if not isinstance(mask_rows, list):
        raise ValueError("NOT_CALCULATED serialization mask must be a list")
    canonical_mask: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in mask_rows:
        if not isinstance(row, dict) or set(row) != {"food_id", "attribute"}:
            raise ValueError("NOT_CALCULATED mask rows must name food_id and attribute")
        key = (row["food_id"], row["attribute"])
        if key in seen:
            raise ValueError("NOT_CALCULATED mask contains duplicate rows")
        seen.add(key)
        canonical_mask.append(key)
    if canonical_mask != sorted(canonical_mask):
        raise ValueError("NOT_CALCULATED mask must use canonical sorted order")
    token_positions = sorted(
        (food_id, attribute)
        for food_id, row in baseline.iterrows()
        for attribute, value in row.items()
        if value == NOT_CALCULATED_TOKEN
    )
    if token_positions != canonical_mask:
        raise ValueError(
            "NOT_CALCULATED token positions must exactly match the explicit manifest mask"
        )
    restored = baseline.copy()
    for food_id, attribute in token_positions:
        if attribute not in FCS2_RULES or FCS2_RULES[attribute].kind != "log_ratio":
            raise ValueError("NOT_CALCULATED token is allowed only for ratio attributes")
        restored[attribute] = restored[attribute].astype(object)
        restored.loc[food_id, attribute] = NOT_CALCULATED
    for attribute in restored.columns:
        for food_id, value in restored[attribute].items():
            if value is NOT_CALCULATED:
                continue
            try:
                restored.loc[food_id, attribute] = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"baseline_attribute_points[{food_id}, {attribute}] must be numeric "
                    "or the declared NOT_CALCULATED token"
                ) from error
    return restored


def _parse_bundle_source_bytes(
    source_bytes: Mapping[str, bytes],
    manifest: Mapping[str, object],
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected = {
        "food_metadata",
        "baseline_attribute_points",
        "food_exposures",
        "effective_attribute_weights",
    }
    if set(source_bytes) != expected or any(
        not isinstance(data, bytes) for data in source_bytes.values()
    ):
        raise ValueError(
            "food bundle source bytes must contain exactly the four immutable artifacts"
        )
    metadata = _read_indexed_csv_bytes(
        source_bytes["food_metadata"], "food_id", "food_metadata"
    )
    baseline = _read_indexed_csv_bytes(
        source_bytes["baseline_attribute_points"],
        "food_id",
        "baseline_attribute_points",
    )
    baseline = _restore_not_calculated_from_manifest(
        baseline, manifest.get("not_calculated_serialization")
    )
    exposures = _read_indexed_csv_bytes(
        source_bytes["food_exposures"], "food_id", "food_exposures"
    )
    exposures.attrs["basis"] = "per_100_kcal"
    effective_weights = _read_indexed_csv_bytes(
        source_bytes["effective_attribute_weights"],
        "food_id",
        "effective_attribute_weights",
    )
    if "FCS2" not in metadata:
        raise ValueError("food_metadata must contain official FCS2")
    official = metadata["FCS2"].copy()
    official.name = "FCS2"
    return official, baseline, exposures, effective_weights, metadata


def _parsed_bundle_fingerprints(
    official: pd.Series,
    baseline: pd.DataFrame,
    exposures: pd.DataFrame,
    effective_weights: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                "official_fcs": _fingerprint(_frame_payload(official)),
                "food_metadata": _fingerprint(_frame_payload(metadata)),
                "baseline_attribute_points": _fingerprint(_frame_payload(baseline)),
                "food_exposures": _fingerprint(_frame_payload(exposures)),
                "effective_attribute_weights": _fingerprint(
                    _frame_payload(effective_weights)
                ),
            }.items()
        )
    )


def _validate_bundle_attestation(
    bundle: FoodAttributeBundle,
    *,
    trusted_snapshot: ReleaseRegistrySnapshot | None,
) -> None:
    attestation = bundle.production_attestation
    if attestation is None:
        raise ValueError("bundle attestation is missing")
    source_bytes = dict(attestation.source_bytes)
    observed_source_hashes = tuple(
        sorted((name, sha256(data).hexdigest()) for name, data in source_bytes.items())
    )
    if observed_source_hashes != attestation.source_byte_sha256:
        raise ValueError("attestation source byte digests are inconsistent")
    observed_hash_map = dict(observed_source_hashes)
    expected_bundle_hashes = {
        **observed_hash_map,
        "official_fcs": observed_hash_map["food_metadata"],
        "input_manifest": attestation.input_manifest_sha256,
    }
    if any(
        bundle.source_hashes.get(name) != digest
        for name, digest in expected_bundle_hashes.items()
    ):
        raise ValueError("bundle source hashes do not match attested immutable bytes")
    if sha256(attestation.input_manifest_bytes).hexdigest() != attestation.input_manifest_sha256:
        raise ValueError("attestation input manifest bytes are inconsistent")
    attested_snapshot = load_release_registry_snapshot(attestation.release_registry_bytes)
    if (
        attested_snapshot.snapshot_sha256
        != attestation.release_registry_snapshot_sha256
    ):
        raise ValueError("attestation release registry bytes are inconsistent")
    if trusted_snapshot is not None:
        if attestation.release_registry_bytes != trusted_snapshot.raw_bytes:
            raise ValueError(
                "attested release registry bytes do not match the live trusted registry"
            )
        if attested_snapshot.snapshot_sha256 != trusted_snapshot.snapshot_sha256:
            raise ValueError("attestation trusted release registry snapshot mismatch")
        snapshot = trusted_snapshot
    else:
        snapshot = attested_snapshot
    manifest = _parse_json_bytes(attestation.input_manifest_bytes, "input manifest")
    if manifest.get("release_registry_snapshot_sha256") != snapshot.snapshot_sha256:
        raise ValueError("attested input manifest release registry snapshot mismatch")
    if manifest.get("registry_version") != snapshot.registry_version:
        raise ValueError("attested input manifest release registry version mismatch")
    if tuple(manifest.get("fndds_releases", ())) != bundle.fndds_releases:
        raise ValueError("attested input manifest FNDDS releases mismatch")
    if manifest.get("production_label") != bundle.production_label:
        raise ValueError("attested input manifest production label mismatch")
    if manifest.get("food_source_linkage_sha256") != bundle.food_source_linkage_sha256:
        raise ValueError("attested input manifest food/source linkage mismatch")
    if manifest.get("exposure_basis") != bundle.exposure_basis:
        raise ValueError("attested input manifest exposure basis mismatch")
    if manifest.get("nutrient_units") != dict(bundle.nutrient_units):
        raise ValueError("attested input manifest nutrient units mismatch")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        raise ValueError("attested input manifest files object is missing")
    for name, digest in observed_source_hashes:
        entry = manifest_files.get(name)
        declared = (
            entry
            if isinstance(entry, str)
            else entry.get("sha256")
            if isinstance(entry, dict)
            else None
        )
        if declared != digest:
            raise ValueError(
                "attested input manifest source digest does not match immutable bytes"
            )
    parsed = _parse_bundle_source_bytes(source_bytes, manifest)
    observed_parsed = _parsed_bundle_fingerprints(*parsed)
    if observed_parsed != attestation.parsed_content_fingerprints:
        raise ValueError("attestation parsed content fingerprints are inconsistent")
    official, baseline, exposures, effective_weights, metadata = parsed
    current_food_ids = list(bundle.official_fcs.index)
    source_subset = (
        official.loc[current_food_ids].copy(),
        baseline.loc[current_food_ids].copy(),
        exposures.loc[current_food_ids].copy(),
        effective_weights.loc[current_food_ids].copy(),
        metadata.loc[current_food_ids].copy(),
    )
    source_subset[2].attrs["basis"] = "per_100_kcal"
    expected_current_parsed = _parsed_bundle_fingerprints(*source_subset)
    current_parsed = _parsed_bundle_fingerprints(
        bundle.official_fcs,
        bundle.baseline_points,
        bundle.food_exposures,
        bundle.effective_attribute_weights,
        bundle.food_metadata,
    )
    if current_parsed != expected_current_parsed:
        raise ValueError("bundle parsed content does not match its source byte attestation")
    expected_attestation_fingerprint = _attestation_fingerprint(
        source_byte_sha256=attestation.source_byte_sha256,
        parsed_content_fingerprints=attestation.parsed_content_fingerprints,
        input_manifest_sha256=attestation.input_manifest_sha256,
        release_registry_snapshot_sha256=attestation.release_registry_snapshot_sha256,
        approved_bundle_id=attestation.approved_bundle_id,
        approved_entry_json=attestation.approved_entry_json,
    )
    if (
        not _is_sha256(attestation.fingerprint)
        or attestation.fingerprint != expected_attestation_fingerprint
    ):
        raise ValueError("bundle attestation fingerprint is inconsistent")


def load_food_attribute_bundle_from_bytes(
    source_bytes: Mapping[str, bytes],
    *,
    input_manifest_bytes: bytes,
    release_registry_bytes: bytes | None = None,
) -> FoodAttributeBundle:
    """Load, hash, parse and attest one immutable food-bundle byte snapshot."""

    if not isinstance(input_manifest_bytes, bytes):
        raise TypeError("input manifest must be supplied as immutable bytes")
    manifest = _parse_json_bytes(input_manifest_bytes, "input manifest")
    if release_registry_bytes is None:
        try:
            release_registry_bytes = _TRUSTED_REGISTRY_PATH.read_bytes()
        except OSError as error:
            raise ValueError("trusted release registry is missing or unreadable") from error
    elif manifest.get("production_label") == "production":
        try:
            trusted_bytes = _TRUSTED_REGISTRY_PATH.read_bytes()
        except OSError as error:
            raise ValueError("trusted release registry is missing or unreadable") from error
        if release_registry_bytes != trusted_bytes:
            raise ValueError(
                "production release registry bytes do not match the trusted registry path"
            )
    snapshot = load_release_registry_snapshot(release_registry_bytes)
    if manifest.get("release_registry_snapshot_sha256") != snapshot.snapshot_sha256:
        raise ValueError("input manifest release registry snapshot SHA-256 mismatch")
    if manifest.get("registry_version") != snapshot.registry_version:
        raise ValueError("input manifest registry version mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("input manifest must contain source file SHA-256 declarations")
    source_copy = {name: bytes(data) for name, data in source_bytes.items()}
    for name, data in source_copy.items():
        entry = files.get(name)
        declared = entry if isinstance(entry, str) else entry.get("sha256") if isinstance(entry, dict) else None
        observed = sha256(data).hexdigest()
        if declared != observed:
            raise ValueError(
                f"SHA-256 mismatch for {name}: expected {declared}, observed {observed}; "
                "the same immutable bytes must be hashed and parsed"
            )
    parsed = _parse_bundle_source_bytes(source_copy, manifest)
    official, baseline, exposures, effective_weights, metadata = parsed
    fndds_releases = tuple(manifest.get("fndds_releases", ()))
    production_label = manifest.get("production_label")
    nutrient_units = manifest.get("nutrient_units")
    if not isinstance(nutrient_units, dict):
        raise ValueError("input manifest must declare nutrient_units")
    linkage = manifest.get("food_source_linkage_sha256")
    if not _is_sha256(linkage):
        raise ValueError("input manifest must declare food_source_linkage_sha256")
    source_hashes = {
        name: sha256(data).hexdigest() for name, data in source_copy.items()
    }
    source_hashes["official_fcs"] = source_hashes["food_metadata"]
    source_hashes["input_manifest"] = sha256(input_manifest_bytes).hexdigest()
    matches = _matching_registry_entries(
        snapshot=snapshot,
        fndds_releases=fndds_releases,
        source_hashes=source_hashes,
        food_source_linkage_sha256=linkage,
        nutrient_units=nutrient_units,
        exposure_basis=str(manifest.get("exposure_basis")),
    )
    if production_label == "production" and not snapshot.approved_artifacts:
        raise ValueError(
            "no approved production bundle is registered; Phase 2 must verify official "
            "FNDDS 2001-2018 digests and food/source linkage"
        )
    if production_label == "production" and len(matches) != 1:
        raise ValueError("production bundle does not exactly match one trusted registry entry")
    approved_bundle_id = matches[0]["bundle_id"] if len(matches) == 1 else None
    approved_entry_json = (
        json.dumps(
            matches[0],
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(matches) == 1
        else None
    )
    source_byte_sha256 = tuple(
        sorted((name, sha256(data).hexdigest()) for name, data in source_copy.items())
    )
    parsed_fingerprints = _parsed_bundle_fingerprints(*parsed)
    input_manifest_sha256 = sha256(input_manifest_bytes).hexdigest()
    attestation = ProductionBundleAttestation(
        source_bytes=tuple(sorted(source_copy.items())),
        input_manifest_bytes=input_manifest_bytes,
        release_registry_bytes=release_registry_bytes,
        source_byte_sha256=source_byte_sha256,
        parsed_content_fingerprints=parsed_fingerprints,
        input_manifest_sha256=input_manifest_sha256,
        release_registry_snapshot_sha256=snapshot.snapshot_sha256,
        approved_bundle_id=approved_bundle_id,
        approved_entry_json=approved_entry_json,
        fingerprint=_attestation_fingerprint(
            source_byte_sha256=source_byte_sha256,
            parsed_content_fingerprints=parsed_fingerprints,
            input_manifest_sha256=input_manifest_sha256,
            release_registry_snapshot_sha256=snapshot.snapshot_sha256,
            approved_bundle_id=approved_bundle_id,
            approved_entry_json=approved_entry_json,
        ),
    )
    return FoodAttributeBundle(
        official_fcs=official,
        baseline_points=baseline,
        food_exposures=exposures,
        effective_attribute_weights=effective_weights,
        food_metadata=metadata,
        fndds_releases=fndds_releases,
        source_hashes=source_hashes,
        nutrient_units=nutrient_units,
        exposure_basis=str(manifest.get("exposure_basis")),
        reconstruction_status=reconstruction_status_table(),
        production_label=str(production_label),
        registry_version=snapshot.registry_version,
        release_registry_snapshot_sha256=snapshot.snapshot_sha256,
        release_registry_canonical_release_set=snapshot.canonical_release_set,
        food_source_linkage_sha256=linkage,
        production_attestation=attestation,
        _factory_token=_VERIFIED_BYTE_LOADER_TOKEN,
    )


@dataclass(frozen=True)
class AttributeGMNPSModel:
    config: AttributeGMNPSConfig
    normalization_state: BetaNormalizationState
    normalization_fingerprint: str
    config_fingerprint: str
    fingerprint: str

    def validate(self) -> None:
        if not isinstance(self.config, AttributeGMNPSConfig):
            raise ValueError("model config must be AttributeGMNPSConfig")
        if not self.config.development_smoke_test:
            if self.config.expected_release_registry_sha256 is None:
                raise ValueError(
                    "production fit/score requires explicit "
                    "expected_release_registry_sha256 from the pre-label/method lock"
                )
            _read_trusted_release_registry_snapshot(
                expected_sha256=self.config.expected_release_registry_sha256,
                require_approved_entry=True,
            )
        if self.normalization_fingerprint != self.normalization_state.state_fingerprint:
            raise ValueError("model normalization fingerprint does not match state")
        expected_config = _fingerprint(asdict(self.config))
        if self.config_fingerprint != expected_config:
            raise ValueError("model config fingerprint does not match config")
        expected = _fingerprint(
            {
                "config_fingerprint": self.config_fingerprint,
                "normalization_fingerprint": self.normalization_fingerprint,
            }
        )
        if self.fingerprint != expected:
            raise ValueError("model fingerprint does not match its contents")


@dataclass(frozen=True)
class AttributeGMNPSResult:
    individual_food: pd.DataFrame
    food_summary: pd.DataFrame
    attribute_attribution: pd.DataFrame
    domain_attribution: pd.DataFrame
    run_manifest: dict[str, object]


def fit_attribute_gmnps(
    development_beta: pd.DataFrame,
    config: AttributeGMNPSConfig | None = None,
) -> AttributeGMNPSModel:
    """Fit only the locked Task 3 normalization on development beta."""

    resolved = config or AttributeGMNPSConfig()
    if not isinstance(resolved, AttributeGMNPSConfig):
        raise TypeError("config must be an AttributeGMNPSConfig")
    state = fit_beta_normalization(development_beta)
    config_fingerprint = _fingerprint(asdict(resolved))
    model_fingerprint = _fingerprint(
        {
            "config_fingerprint": config_fingerprint,
            "normalization_fingerprint": state.state_fingerprint,
        }
    )
    model = AttributeGMNPSModel(
        config=resolved,
        normalization_state=state,
        normalization_fingerprint=state.state_fingerprint,
        config_fingerprint=config_fingerprint,
        fingerprint=model_fingerprint,
    )
    model.validate()
    return model


def _run_primary_pipeline(
    model: AttributeGMNPSModel,
    beta: pd.DataFrame,
    bundle: FoodAttributeBundle,
):
    config = model.config
    responses = attribute_response(
        model.normalization_state,
        beta,
        bundle.food_exposures,
        mapping_version=config.mapping_version,
    )
    calibration = calibrate_attribute_points(
        bundle.baseline_points,
        responses,
        mode=config.attribute_point_mode,
        effective_attribute_weights=bundle.effective_attribute_weights,
    )
    selected_attributes = tuple(
        attribute
        for attribute in _ACTIVE_ATTRIBUTES
        if FCS2_RULES[attribute].domain in config.recomputed_domains
    )
    baseline = decompose_official_baseline(
        bundle.official_fcs,
        bundle.baseline_points.loc[:, selected_attributes],
        recomputed_domains=config.recomputed_domains,
        effective_attribute_weights=bundle.effective_attribute_weights.loc[
            :, selected_attributes
        ],
    )
    recomputed = recompute_personalized_domains(baseline, calibration)
    final = compose_personalized_fcs(
        baseline,
        recomputed,
        cap_mode=config.final_cap_mode,
    )
    return responses, calibration, baseline, recomputed, final


def _counterfactual_beta(
    beta: pd.DataFrame,
    state: BetaNormalizationState,
    *,
    frozen_channel: str,
    mapping_version: str,
) -> pd.DataFrame:
    channels = _effect_nutrients_by_channel(mapping_version)
    if frozen_channel not in channels:
        raise ValueError("frozen_channel must be MAC or LIPID")
    frozen_nutrients = set(channels[frozen_channel])
    frozen = beta.loc[:, state.nutrient_order].copy()
    for nutrient in state.nutrient_order:
        if nutrient in frozen_nutrients:
            frozen.loc[:, nutrient] = state.medians[nutrient]
    return frozen


def _channel_for_attributes(mapping_version: str) -> dict[str, str]:
    channels: dict[str, set[str]] = {}
    for row in _mapping_rows(mapping_version):
        if row.role == "effect":
            channels.setdefault(row.attribute, set()).add(row.channel)
    output = {}
    for attribute in _ACTIVE_ATTRIBUTES:
        values = channels.get(attribute, set())
        output[attribute] = next(iter(values)) if len(values) == 1 else ("MIXED" if values else "NONE")
    return output


def _driver_strings(calibration, top_n: int = 3) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    positive: dict[tuple[str, str], str] = {}
    negative: dict[tuple[str, str], str] = {}
    for pair, row in calibration.deltas.iterrows():
        values = []
        for order, (attribute, value) in enumerate(row.items()):
            if value is NOT_CALCULATED:
                continue
            numeric = float(value)
            if numeric != 0.0:
                values.append((attribute, numeric, order))
        positives = sorted(
            (item for item in values if item[1] > 0),
            key=lambda item: (-item[1], item[2]),
        )[:top_n]
        negatives = sorted(
            (item for item in values if item[1] < 0),
            key=lambda item: (item[1], item[2]),
        )[:top_n]
        positive[pair] = "; ".join(
            f"{name}={value:.6g}" for name, value, _ in positives
        ) or "none"
        negative[pair] = "; ".join(
            f"{name}={value:.6g}" for name, value, _ in negatives
        ) or "none"
    return positive, negative


def summarize_attribute_gmnps_foods(individual_food: pd.DataFrame) -> pd.DataFrame:
    """Build the deterministic article-facing per-food summary."""

    grouped = individual_food.groupby(
        ["food_id", "food_name", "food_group"], sort=True, dropna=False
    )
    summary = grouped.agg(
        FCS2=("FCS2", "first"),
        GMNPS_mean=("GMNPS_score", "mean"),
        GMNPS_sd=("GMNPS_score", "std"),
        GMNPS_p05=("GMNPS_score", lambda values: float(np.percentile(values, 5))),
        GMNPS_p95=("GMNPS_score", lambda values: float(np.percentile(values, 95))),
        MAC_variance=("MAC_delta", "var"),
        LIPID_variance=("LIPID_delta", "var"),
        channel_interaction_variance=("channel_interaction_delta", "var"),
        source_cohort_n=("individual_id", "nunique"),
    ).reset_index()
    variance_columns = [
        "GMNPS_sd",
        "MAC_variance",
        "LIPID_variance",
        "channel_interaction_variance",
    ]
    summary.loc[:, variance_columns] = summary.loc[:, variance_columns].fillna(0.0)
    variances = summary.loc[
        :, ["MAC_variance", "LIPID_variance", "channel_interaction_variance"]
    ].to_numpy(float)
    labels = np.asarray(["MAC", "LIPID", "interaction"], dtype=object)
    summary["dominant_channel"] = labels[np.argmax(variances, axis=1)]
    return summary.sort_values("food_id", kind="mergesort").reset_index(drop=True)


def _attribution_tables(calibration, recomputed, final, config: AttributeGMNPSConfig):
    channels = _channel_for_attributes(config.mapping_version)
    membership = recomputed.membership_audit
    attribute_rows = []
    for (individual_id, food_id), point_row in calibration.points.iterrows():
        for attribute in _ACTIVE_ATTRIBUTES:
            baseline = calibration.diagnostics.loc[
                (individual_id, food_id, attribute), "baseline_points"
            ]
            personalized = point_row[attribute]
            delta = calibration.deltas.loc[(individual_id, food_id), attribute]
            not_calculated = personalized is NOT_CALCULATED
            membership_key = (individual_id, food_id, attribute)
            if membership_key in membership.index:
                membership_row = membership.loc[membership_key]
                calculated = bool(membership_row["calculated"])
                baseline_selected = bool(membership_row["baseline_selected"])
                personalized_selected = bool(
                    membership_row["personalized_selected"]
                )
                effective_weight = float(membership_row["published_weight"])
                baseline_denominator = float(
                    membership_row["baseline_active_weight_denominator"]
                )
                active_denominator = float(
                    membership_row["active_weight_denominator"]
                )
            else:
                calculated = not not_calculated
                baseline_selected = False
                personalized_selected = False
                effective_weight = float(
                    calibration.effective_attribute_weights.loc[food_id, attribute]
                )
                baseline_denominator = 0.0
                active_denominator = 0.0
            attribute_rows.append(
                {
                    "individual_id": individual_id,
                    "food_id": food_id,
                    "attribute": attribute,
                    "domain": FCS2_RULES[attribute].domain,
                    "channel": channels[attribute],
                    "baseline_points": (
                        NOT_CALCULATED_TOKEN if not_calculated else float(baseline)
                    ),
                    "personalized_points": (
                        NOT_CALCULATED_TOKEN
                        if not_calculated
                        else float(personalized)
                    ),
                    "attribute_point_delta": (
                        NOT_CALCULATED_TOKEN if not_calculated else float(delta)
                    ),
                    "not_calculated": not_calculated,
                    "effective_attribute_weight": effective_weight,
                    "calculated": calculated,
                    "active": personalized_selected,
                    "baseline_selected": baseline_selected,
                    "personalized_selected": personalized_selected,
                    "baseline_active_domain_denominator": baseline_denominator,
                    "active_domain_denominator": active_denominator,
                    "mapping_version": config.mapping_version,
                    "calibration_fingerprint": calibration.calibration_fingerprint,
                }
            )
    attribute = pd.DataFrame(attribute_rows).sort_values(
        ["individual_id", "food_id", "attribute"], kind="mergesort"
    ).reset_index(drop=True)
    domain = final.domain_audit.reset_index()
    membership_rows = membership.reset_index()
    domain_membership = (
        membership_rows.groupby(
            ["individual_id", "food_id", "domain"],
            sort=False,
            as_index=False,
        )
        .agg(
            baseline_active_domain_denominator=(
                "baseline_active_weight_denominator",
                "first",
            ),
            active_domain_denominator=("active_weight_denominator", "first"),
            calculated_attribute_count=("calculated", "sum"),
            baseline_selected_attribute_count=("baseline_selected", "sum"),
            active_attribute_count=("personalized_selected", "sum"),
        )
    )
    domain = domain.merge(
        domain_membership,
        on=["individual_id", "food_id", "domain"],
        how="left",
        validate="one_to_one",
    )
    for column in (
        "calculated_attribute_count",
        "baseline_selected_attribute_count",
        "active_attribute_count",
    ):
        domain[column] = domain[column].astype(int)
    domain["mapping_version"] = config.mapping_version
    domain["recomposition_fingerprint"] = final.recomposition_fingerprint
    domain = domain.sort_values(
        ["individual_id", "food_id", "domain"], kind="mergesort"
    ).reset_index(drop=True)
    return attribute, domain


def _primary_result(
    model: AttributeGMNPSModel,
    beta: pd.DataFrame,
    bundle: FoodAttributeBundle,
) -> AttributeGMNPSResult:
    full = _run_primary_pipeline(model, beta, bundle)
    _, calibration, baseline, recomputed, final = full
    mac_only_beta = _counterfactual_beta(
        beta,
        model.normalization_state,
        frozen_channel="LIPID",
        mapping_version=model.config.mapping_version,
    )
    lipid_only_beta = _counterfactual_beta(
        beta,
        model.normalization_state,
        frozen_channel="MAC",
        mapping_version=model.config.mapping_version,
    )
    mac_only = _run_primary_pipeline(model, mac_only_beta, bundle)
    lipid_only = _run_primary_pipeline(model, lipid_only_beta, bundle)
    mac_final = mac_only[-1]
    lipid_final = lipid_only[-1]
    positive, negative = _driver_strings(calibration)

    rows = []
    for pair in final.scores.index:
        individual_id, food_id = pair
        official = float(bundle.official_fcs.loc[food_id])
        total_delta = float(final.deltas.loc[pair])
        mac_delta = float(mac_final.scores.loc[pair] - official)
        lipid_delta = float(lipid_final.scores.loc[pair] - official)
        metadata = bundle.food_metadata.loc[food_id]
        rows.append(
            {
                "individual_id": individual_id,
                "food_id": food_id,
                "food_name": metadata["food_name"],
                "food_group": metadata["food_group"],
                "FCS2": official,
                "GMNPS_delta": total_delta,
                "GMNPS_score": float(final.scores.loc[pair]),
                "MAC_delta": mac_delta,
                "LIPID_delta": lipid_delta,
                "channel_interaction_delta": total_delta - mac_delta - lipid_delta,
                "top_positive_drivers": positive[pair],
                "top_negative_drivers": negative[pair],
                "driver_basis": "attribute_point_delta",
                "mask_version": model.config.mask_version,
                "mapping_version": model.config.mapping_version,
                "scoring_version": model.config.scoring_version,
                "method_role": model.config.method_role,
            }
        )
    individual = pd.DataFrame(rows, columns=_PRIMARY_INDIVIDUAL_COLUMNS).sort_values(
        ["individual_id", "food_id"], kind="mergesort"
    ).reset_index(drop=True)
    attribute, domain = _attribution_tables(
        calibration, recomputed, final, model.config
    )
    manifest = _base_manifest(model, bundle)
    manifest.update(
        {
            "baseline_fingerprint": baseline.fingerprint,
            "response_fingerprint": full[0].response_fingerprint,
            "calibration_fingerprint": calibration.calibration_fingerprint,
            "recomposition_fingerprint": recomputed.fingerprint,
            "final_fingerprint": final.fingerprint,
            "counterfactual_provenance": {
                "MAC_delta": {
                    "definition": "score_with_LIPID_frozen_minus_FCS2",
                    "frozen_channel": "LIPID",
                    "raw_beta_replacement": "development_median",
                    "frozen_effect_nutrients": list(
                        _effect_nutrients_by_channel(model.config.mapping_version)["LIPID"]
                    ),
                    "response_fingerprint": mac_only[0].response_fingerprint,
                    "calibration_fingerprint": mac_only[1].calibration_fingerprint,
                    "recomposition_fingerprint": mac_only[3].fingerprint,
                    "final_fingerprint": mac_final.fingerprint,
                },
                "LIPID_delta": {
                    "definition": "score_with_MAC_frozen_minus_FCS2",
                    "frozen_channel": "MAC",
                    "raw_beta_replacement": "development_median",
                    "frozen_effect_nutrients": list(
                        _effect_nutrients_by_channel(model.config.mapping_version)["MAC"]
                    ),
                    "response_fingerprint": lipid_only[0].response_fingerprint,
                    "calibration_fingerprint": lipid_only[1].calibration_fingerprint,
                    "recomposition_fingerprint": lipid_only[3].fingerprint,
                    "final_fingerprint": lipid_final.fingerprint,
                },
                "channel_interaction_delta": {
                    "definition": "GMNPS_delta_minus_MAC_delta_minus_LIPID_delta"
                },
            },
        }
    )
    return AttributeGMNPSResult(
        individual_food=individual,
        food_summary=summarize_attribute_gmnps_foods(individual),
        attribute_attribution=attribute,
        domain_attribution=domain,
        run_manifest=manifest,
    )


def _base_manifest(
    model: AttributeGMNPSModel, bundle: FoodAttributeBundle
) -> dict[str, object]:
    config = model.config
    not_calculated_mask = [
        {"food_id": food_id, "attribute": attribute}
        for food_id, row in bundle.baseline_points.iterrows()
        for attribute, value in row.items()
        if value is NOT_CALCULATED
    ]
    not_calculated_mask.sort(key=lambda row: (row["food_id"], row["attribute"]))
    return {
        "method": config.method,
        "method_role": config.method_role,
        "scoring_version": config.scoring_version,
        "mapping_version": config.mapping_version,
        "mask_version": config.mask_version,
        "attribute_point_mode": config.attribute_point_mode,
        "attribute_point_fraction": float(
            ATTRIBUTE_POINT_FRACTION_MODES[config.attribute_point_mode]
        ),
        "final_cap_mode": config.final_cap_mode,
        "final_delta_cap": float(FINAL_DEVIATION_CAP_MODES[config.final_cap_mode]),
        "recomputed_domains": list(config.recomputed_domains),
        "model_fingerprint": model.fingerprint,
        "normalization_fingerprint": model.normalization_fingerprint,
        "normalization_fit_n": model.normalization_state.fit_n,
        "normalization_fit_id_sha256": model.normalization_state.fit_id_sha256,
        "bundle_fingerprint": bundle.fingerprint,
        "effective_attribute_weights_sha256": bundle.effective_weights_fingerprint,
        "source_hashes": dict(sorted(bundle.source_hashes.items())),
        "fndds_releases": list(bundle.fndds_releases),
        "nutrient_units": dict(sorted(bundle.nutrient_units.items())),
        "exposure_basis": bundle.exposure_basis,
        "input_bundle_production_label": bundle.production_label,
        "production_label": (
            "non-production"
            if config.development_smoke_test or config.method == LEGACY_METHOD
            else bundle.production_label
        ),
        "development_smoke_test": config.development_smoke_test,
        "software_version": config.scoring_version,
        "reconstruction_status_sha256": _fingerprint(
            _frame_payload(bundle.reconstruction_status)
        ),
        "missing_exposure_values": 0,
        "score_centering": "none",
        "beta_centering": "development_median",
        "candidate_selection": "none",
        "driver_basis": (
            "raw_nutrient_contribution"
            if config.method == LEGACY_METHOD
            else "attribute_point_delta"
        ),
        "trusted_registry_version": FCS2_FNDDS_REGISTRY_VERSION,
        "trusted_registry_schema_version": FCS2_FNDDS_REGISTRY_SCHEMA_VERSION,
        "trusted_registry_digest_algorithm": (
            FCS2_FNDDS_REGISTRY_DIGEST_ALGORITHM
        ),
        "expected_release_registry_sha256": (
            config.expected_release_registry_sha256
        ),
        "release_registry_snapshot_sha256": (
            bundle.release_registry_snapshot_sha256
        ),
        "release_registry_canonical_release_set": list(
            bundle.release_registry_canonical_release_set
        ),
        "approved_registry_entry": (
            None
            if bundle.production_attestation is None
            or bundle.production_attestation.approved_entry_json is None
            else {
                **json.loads(bundle.production_attestation.approved_entry_json),
                "entry_sha256": sha256(
                    bundle.production_attestation.approved_entry_json.encode("utf-8")
                ).hexdigest(),
                "attestation_sha256": bundle.production_attestation.fingerprint,
            }
        ),
        "food_source_linkage_sha256": bundle.food_source_linkage_sha256,
        "not_calculated_serialization": {
            "schema_version": NOT_CALCULATED_SCHEMA_VERSION,
            "token": NOT_CALCULATED_TOKEN,
            "allowed_attributes": [
                attribute
                for attribute in _ACTIVE_ATTRIBUTES
                if FCS2_RULES[attribute].kind == "log_ratio"
            ],
            "baseline_attribute_points_mask": not_calculated_mask,
        },
    }


def _legacy_result(
    model: AttributeGMNPSModel,
    beta: pd.DataFrame,
    bundle: FoodAttributeBundle,
) -> AttributeGMNPSResult:
    legacy_config = AnchoredScoringConfig(
        mask_version=model.config.mask_version,
        scoring_version=model.config.scoring_version,
    )
    individual, _, legacy_manifest = score_individual_foods(
        beta,
        bundle.food_exposures,
        bundle.food_metadata.assign(FCS2=bundle.official_fcs),
        legacy_config,
    )
    individual["channel_interaction_delta"] = (
        individual["GMNPS_delta"] - individual["MAC_delta"] - individual["LIPID_delta"]
    )
    individual["mapping_version"] = model.config.mapping_version
    individual["method_role"] = "sensitivity"
    individual["scoring_version"] = model.config.scoring_version
    individual = individual.rename(
        columns={
            "top_positive_drivers": "legacy_top_positive_nutrient_drivers",
            "top_negative_drivers": "legacy_top_negative_nutrient_drivers",
        }
    )
    legacy_driver_columns = [
        "legacy_top_positive_nutrient_drivers",
        "legacy_top_negative_nutrient_drivers",
    ]
    individual[legacy_driver_columns] = individual[legacy_driver_columns].replace(
        "", "none"
    )
    individual["driver_basis"] = "raw_nutrient_contribution"
    individual = individual.loc[:, _LEGACY_INDIVIDUAL_COLUMNS].sort_values(
        ["individual_id", "food_id"], kind="mergesort"
    ).reset_index(drop=True)
    manifest = _base_manifest(model, bundle)
    manifest["legacy_manifest"] = legacy_manifest
    manifest["scientific_boundary"] = (
        "Legacy centered final-score offset; sensitivity comparator only."
    )
    attribute = pd.DataFrame(
        columns=[
            "individual_id",
            "food_id",
            "attribute",
            "domain",
            "channel",
            "baseline_points",
            "personalized_points",
            "attribute_point_delta",
            "mapping_version",
            "calibration_fingerprint",
        ]
    )
    domain = pd.DataFrame(
        columns=[
            "individual_id",
            "food_id",
            "domain",
            "baseline_contribution",
            "personalized_contribution",
            "domain_delta",
            "mapping_version",
            "recomposition_fingerprint",
        ]
    )
    return AttributeGMNPSResult(
        individual_food=individual,
        food_summary=summarize_attribute_gmnps_foods(individual),
        attribute_attribution=attribute,
        domain_attribution=domain,
        run_manifest=manifest,
    )


def score_attribute_gmnps(
    model: AttributeGMNPSModel,
    score_beta: pd.DataFrame,
    food_bundle: FoodAttributeBundle,
) -> AttributeGMNPSResult:
    """Score held-out beta through the locked primary or explicit legacy path."""

    if not isinstance(model, AttributeGMNPSModel):
        raise TypeError("model must be an AttributeGMNPSModel")
    model.validate()
    if not isinstance(food_bundle, FoodAttributeBundle):
        raise TypeError("food_bundle must be a FoodAttributeBundle")
    food_bundle.validate()
    if food_bundle.production_label == "production":
        if (
            model.config.expected_release_registry_sha256
            != food_bundle.release_registry_snapshot_sha256
        ):
            raise ValueError(
                "production bundle registry snapshot does not match the model's "
                "pre-label/method-lock expected snapshot"
            )
    if food_bundle.production_label == "non-production" and not model.config.development_smoke_test:
        raise ValueError(
            "FNDDS 2021-2023 is non-production and requires development_smoke_test=true"
        )
    if model.config.method == LEGACY_METHOD:
        return _legacy_result(model, score_beta, food_bundle)
    return _primary_result(model, score_beta, food_bundle)


__all__ = [
    "ATTRIBUTE_GMNPS_SCORING_VERSION",
    "FCS2_FNDDS_REGISTRY_DIGEST_ALGORITHM",
    "FCS2_FNDDS_REGISTRY_SCHEMA_VERSION",
    "FCS2_FNDDS_REGISTRY_VERSION",
    "PRIMARY_RECOMPUTED_DOMAINS",
    "AttributeGMNPSConfig",
    "AttributeGMNPSModel",
    "AttributeGMNPSResult",
    "FoodAttributeBundle",
    "ProductionBundleAttestation",
    "ReleaseRegistrySnapshot",
    "fit_attribute_gmnps",
    "load_food_attribute_bundle_from_bytes",
    "load_release_registry_snapshot",
    "score_attribute_gmnps",
    "summarize_attribute_gmnps_foods",
]
