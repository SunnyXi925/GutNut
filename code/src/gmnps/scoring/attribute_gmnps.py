"""Public, provenance-bound API for attribute-level GMNPS scoring."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
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
from gmnps.scoring.fcs2_attribute_rules import FCS2_RULES, NOT_CALCULATED, NotCalculated
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
)
_TRUSTED_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/fcs2_fndds_release_registry.json"
)


def _load_trusted_registry() -> tuple[dict[str, object], str]:
    try:
        raw = _TRUSTED_REGISTRY_PATH.read_bytes()
        registry = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("trusted FCS2/FNDDS release registry is unreadable") from error
    if not isinstance(registry, dict):
        raise RuntimeError("trusted FCS2/FNDDS release registry must be an object")
    required = {
        "schema_version",
        "registry_version",
        "canonical_release_set",
        "approved_artifact_entry_schema",
        "approved_artifacts",
    }
    if set(registry) != required:
        raise RuntimeError("trusted FCS2/FNDDS release registry schema is invalid")
    if not isinstance(registry["registry_version"], str):
        raise RuntimeError("trusted registry version is invalid")
    canonical = registry["canonical_release_set"]
    if not isinstance(canonical, list) or not canonical or any(
        not isinstance(value, str) or not value for value in canonical
    ):
        raise RuntimeError("trusted registry canonical release set is invalid")
    entries = registry["approved_artifacts"]
    if not isinstance(entries, list):
        raise RuntimeError("trusted registry approved_artifacts must be a list")
    required_entry_fields = set(
        registry["approved_artifact_entry_schema"]["required_fields"]
    )
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != required_entry_fields:
            raise RuntimeError("trusted registry contains an invalid approved artifact entry")
        if set(entry["artifact_sha256"]) != set(_PRODUCTION_ARTIFACT_HASH_KEYS):
            raise RuntimeError("trusted registry artifact digest keys are invalid")
        if any(not _is_sha256(value) for value in entry["artifact_sha256"].values()):
            raise RuntimeError("trusted registry contains an invalid artifact digest")
        if not _is_sha256(entry["food_source_linkage_sha256"]):
            raise RuntimeError("trusted registry contains an invalid linkage digest")
    return registry, sha256(raw).hexdigest()


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


_TRUSTED_FCS2_FNDDS_REGISTRY, _TRUSTED_FCS2_FNDDS_REGISTRY_SHA256 = (
    _load_trusted_registry()
)
FCS2_FNDDS_REGISTRY_VERSION = str(
    _TRUSTED_FCS2_FNDDS_REGISTRY["registry_version"]
)
_CANONICAL_PRODUCTION_RELEASES = tuple(
    _TRUSTED_FCS2_FNDDS_REGISTRY["canonical_release_set"]
)


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


def _validate_approved_production_bundle(bundle: "FoodAttributeBundle") -> None:
    if bundle.registry_version != FCS2_FNDDS_REGISTRY_VERSION:
        raise ValueError("production registry version does not match the trusted registry")
    entries = _TRUSTED_FCS2_FNDDS_REGISTRY["approved_artifacts"]
    if not entries:
        raise ValueError(
            "no approved production bundle is registered; Phase 2 must verify official "
            "FNDDS 2001-2018 digests and food/source linkage"
        )
    actual_artifacts = {
        key: bundle.source_hashes[key] for key in _PRODUCTION_ARTIFACT_HASH_KEYS
    }
    matches = [
        entry
        for entry in entries
        if tuple(entry["fndds_releases"]) == bundle.fndds_releases
        and entry["artifact_sha256"] == actual_artifacts
        and entry["food_source_linkage_sha256"]
        == bundle.food_source_linkage_sha256
        and entry["nutrient_units"] == dict(bundle.nutrient_units)
        and entry["exposure_basis"] == bundle.exposure_basis
    ]
    if len(matches) != 1:
        raise ValueError(
            "production bundle does not exactly match one trusted registry entry"
        )


@dataclass(frozen=True)
class FoodAttributeBundle:
    """Official food baseline and composition inputs bound by a content fingerprint."""

    official_fcs: pd.Series
    baseline_points: pd.DataFrame
    food_exposures: pd.DataFrame
    food_metadata: pd.DataFrame
    fndds_releases: tuple[str, ...]
    source_hashes: Mapping[str, str]
    nutrient_units: Mapping[str, str]
    exposure_basis: str
    reconstruction_status: pd.DataFrame
    production_label: str
    registry_version: str
    food_source_linkage_sha256: str
    fingerprint: str = ""

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
                "food_metadata": _frame_payload(self.food_metadata),
                "fndds_releases": list(self.fndds_releases),
                "source_hashes": dict(sorted(self.source_hashes.items())),
                "nutrient_units": dict(sorted(self.nutrient_units.items())),
                "exposure_basis": self.exposure_basis,
                "reconstruction_status": _frame_payload(self.reconstruction_status),
                "production_label": self.production_label,
                "registry_version": self.registry_version,
                "food_source_linkage_sha256": self.food_source_linkage_sha256,
            }
        )

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
        # Task 4 validation enforces bounds, canonical sentinels, and domain completeness.
        decompose_official_baseline(
            self.official_fcs,
            self.baseline_points,
            recomputed_domains=_ALL_DOMAINS,
        )

        _finite_frame(self.food_exposures, "food_exposures", nonnegative=True)
        if self.exposure_basis != "per_100_kcal":
            raise ValueError("exposure_basis must be per_100_kcal")
        if self.food_exposures.attrs.get("basis") != self.exposure_basis:
            raise ValueError("food_exposures attrs['basis'] must match exposure_basis")

        required_metadata = {"food_name", "food_group"}
        missing_metadata = required_metadata - set(self.food_metadata.columns)
        if missing_metadata:
            raise ValueError(f"food_metadata missing required columns: {sorted(missing_metadata)}")
        for column in required_metadata:
            if self.food_metadata[column].isna().any() or any(
                not isinstance(value, str) or not value.strip()
                for value in self.food_metadata[column]
            ):
                raise ValueError(f"food_metadata {column} values must be nonempty strings")
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
        if not _is_sha256(self.food_source_linkage_sha256):
            raise ValueError("food_source_linkage_sha256 must be a lowercase SHA-256 digest")

        if not isinstance(self.source_hashes, Mapping):
            raise ValueError("source_hashes must be a mapping")
        missing_hashes = _REQUIRED_SOURCE_HASHES - set(self.source_hashes)
        if missing_hashes:
            raise ValueError(f"source_hashes missing required sources: {sorted(missing_hashes)}")
        if any(not _is_sha256(value) for value in self.source_hashes.values()):
            raise ValueError("source_hashes values must be lowercase SHA-256 digests")

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

        if release_kind == "production":
            _validate_approved_production_bundle(self)

        if check_fingerprint:
            expected = self.compute_fingerprint()
            if not _is_sha256(self.fingerprint) or self.fingerprint != expected:
                raise ValueError("food attribute bundle fingerprint does not match its contents")


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


def _attribution_tables(calibration, final, config: AttributeGMNPSConfig):
    channels = _channel_for_attributes(config.mapping_version)
    attribute_rows = []
    for (individual_id, food_id), point_row in calibration.points.iterrows():
        for attribute in _ACTIVE_ATTRIBUTES:
            baseline = calibration.diagnostics.loc[
                (individual_id, food_id, attribute), "baseline_points"
            ]
            personalized = point_row[attribute]
            delta = calibration.deltas.loc[(individual_id, food_id), attribute]
            not_calculated = personalized is NOT_CALCULATED
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
                    "mapping_version": config.mapping_version,
                    "calibration_fingerprint": calibration.calibration_fingerprint,
                }
            )
    attribute = pd.DataFrame(attribute_rows).sort_values(
        ["individual_id", "food_id", "attribute"], kind="mergesort"
    ).reset_index(drop=True)
    domain = final.domain_audit.reset_index()
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
    attribute, domain = _attribution_tables(calibration, final, model.config)
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
        "centering": "none",
        "candidate_selection": "none",
        "driver_basis": (
            "raw_nutrient_contribution"
            if config.method == LEGACY_METHOD
            else "attribute_point_delta"
        ),
        "trusted_registry_version": FCS2_FNDDS_REGISTRY_VERSION,
        "trusted_registry_sha256": _TRUSTED_FCS2_FNDDS_REGISTRY_SHA256,
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
    if food_bundle.production_label == "non-production" and not model.config.development_smoke_test:
        raise ValueError(
            "FNDDS 2021-2023 is non-production and requires development_smoke_test=true"
        )
    if model.config.method == LEGACY_METHOD:
        return _legacy_result(model, score_beta, food_bundle)
    return _primary_result(model, score_beta, food_bundle)


__all__ = [
    "ATTRIBUTE_GMNPS_SCORING_VERSION",
    "PRIMARY_RECOMPUTED_DOMAINS",
    "AttributeGMNPSConfig",
    "AttributeGMNPSModel",
    "AttributeGMNPSResult",
    "FoodAttributeBundle",
    "fit_attribute_gmnps",
    "score_attribute_gmnps",
    "summarize_attribute_gmnps_foods",
]
