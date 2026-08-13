"""Fail-closed population-safety design audits for attribute-level GMNPS."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from itertools import combinations, product
import json
import os
from pathlib import Path
import stat
from typing import Mapping

import numpy as np
import pandas as pd

from gmnps.validation.method_lock_gate import (
    MethodLockArtifactPaths,
    validate_method_lock_manifest,
)


POPULATION_EVIDENCE_ROLE = "population_safety_design_audit"
POPULATION_SCHEMA_VERSION = "gmnps-attribute-validation-input-v2"
_CATEGORY_ORDER = ("minimize", "moderate", "encourage")
_CATEGORY_INDEX = {value: index for index, value in enumerate(_CATEGORY_ORDER)}
_REQUIRED_COLUMNS = {
    "individual_id",
    "food_id",
    "food_group",
    "food_subgroup",
    "FCS2",
    "GMNPS_delta",
    "GMNPS_score",
}
_FORBIDDEN_COLUMNS = frozenset(
    {
        "disease",
        "disease_label",
        "phenotype_label",
        "outcome",
        "y_true",
        "glucose_iauc_2h",
        "tg_6h_rise",
        "c_peptide_iauc_2h",
        "postprandial_response",
    }
)
_ROOT = Path(__file__).resolve().parents[4]
_TRUSTED_TASK4_REGISTRY_PATH = (
    _ROOT / "code/src/configs/attribute_validation_trust_registry.json"
)
_POPULATION_IMPLEMENTATION = Path(__file__).resolve()
_BIOLOGICAL_IMPLEMENTATION = Path(__file__).with_name("biological_consistency.py")
_RUNNER_IMPLEMENTATION = _ROOT / "code/src/scripts/run_attribute_validation.py"
_SCORING_IMPLEMENTATION = _ROOT / "code/src/scripts/run_attribute_gmnps.py"


@dataclass(frozen=True)
class PopulationSafetyConfig:
    """Frozen analysis, resampling, and minimum-information parameters."""

    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 20260813
    confidence_level: float = 0.95
    minimum_valid_replicates: int = 800
    minimum_clusters: int = 20

    def __post_init__(self) -> None:
        for name in (
            "bootstrap_replicates",
            "bootstrap_seed",
            "minimum_valid_replicates",
            "minimum_clusters",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a strict integer")
        if self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")
        if not 1 <= self.minimum_valid_replicates <= self.bootstrap_replicates:
            raise ValueError(
                "minimum_valid_replicates must be between 1 and bootstrap_replicates"
            )
        if self.minimum_clusters < 2:
            raise ValueError("minimum_clusters must be at least 2")
        if not isinstance(self.confidence_level, (int, float)) or isinstance(
            self.confidence_level, bool
        ):
            raise ValueError("confidence_level must be numeric")
        if not 0.0 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must be between zero and one")

    def binding_payload(self) -> dict[str, object]:
        return {
            "schema_version": "population-safety-config-v2",
            **asdict(self),
            "category_thresholds": {
                "minimize_upper_exclusive": 31,
                "encourage_lower_inclusive": 70,
            },
            "ci_method": "independent_cluster_percentile",
            "panel_policy": "complete_rectangular_identical_food_panel",
        }


@dataclass(frozen=True)
class PopulationArtifactPaths:
    """Every live Phase 1 and Task 4 artifact required for production."""

    task4_input_table: Path
    task4_provenance: Path
    phase1_individual_food: Path
    phase1_run_manifest: Path
    phase1_success_marker: Path
    upstream_input_manifest: Path
    method_lock_manifest: Path
    method_lock_artifact_locator: Path
    subgroup_enrichment: Path
    method_lock_artifacts: MethodLockArtifactPaths

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "method_lock_artifacts":
                if not isinstance(value, MethodLockArtifactPaths):
                    raise TypeError("method_lock_artifacts must be MethodLockArtifactPaths")
            else:
                object.__setattr__(self, item.name, Path(value))


@dataclass(frozen=True)
class PopulationSafetyAudit:
    evidence_role: str
    panel_audit: dict[str, object]
    global_summary: dict[str, object]
    group_convergence: pd.DataFrame
    subgroup_convergence: pd.DataFrame
    between_group_discrimination: pd.DataFrame
    transition_matrix: pd.DataFrame
    reversal_audit: pd.DataFrame
    bootstrap_summary: pd.DataFrame
    verified_input_hashes: Mapping[str, str] | None = None


def _canonical_json_sha256(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _immutable_read(path: Path, label: str) -> bytes:
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is missing or is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        live = os.stat(path, follow_symlinks=False)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        if identity(before) != identity(after) or identity(after) != identity(live):
            raise ValueError(f"{label} changed during immutable read")
        return b"".join(chunks)
    except OSError as error:
        raise ValueError(f"{label} is missing or unreadable") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _json_object(raw: bytes, label: str) -> dict[str, object]:
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def task4_binding_hashes(config: PopulationSafetyConfig) -> dict[str, str]:
    """Hash installed Task 4 implementations and all analysis thresholds."""

    return {
        "attribute_population_safety_implementation_sha256": sha256(
            _immutable_read(_POPULATION_IMPLEMENTATION, "population implementation")
        ).hexdigest(),
        "biological_consistency_implementation_sha256": sha256(
            _immutable_read(_BIOLOGICAL_IMPLEMENTATION, "biological implementation")
        ).hexdigest(),
        "run_attribute_validation_implementation_sha256": sha256(
            _immutable_read(_RUNNER_IMPLEMENTATION, "Task 4 runner implementation")
        ).hexdigest(),
        "population_safety_config_sha256": _canonical_json_sha256(
            config.binding_payload()
        ),
    }


def _validate_base_population_provenance(manifest: Mapping[str, object]) -> bool:
    expected = {
        "schema_version": POPULATION_SCHEMA_VERSION,
        "analysis_kind": "population_safety",
        "evidence_role": POPULATION_EVIDENCE_ROLE,
        "method": "attribute_recomposition",
        "method_role": "primary",
        "score_centering": "none",
    }
    if not isinstance(manifest, Mapping):
        raise ValueError("provenance manifest must be an object")
    for field, required in expected.items():
        if manifest.get(field) != required:
            raise ValueError(f"provenance {field} must equal {required!r}")
    if manifest.get("scoring_frozen_before_labels") is not True:
        raise ValueError("scoring must be frozen before labels are inspected")
    if manifest.get("independent_unit") not in {
        "individual_id",
        "component_id",
        "cohort_component_id",
    }:
        raise ValueError("provenance independent_unit is invalid")
    if not isinstance(manifest.get("bundle_id"), str) or not manifest["bundle_id"]:
        raise ValueError("provenance bundle_id must be nonempty")
    production = manifest.get("production_label") == "production"
    if production:
        if manifest.get("synthetic") is not False:
            raise ValueError("production provenance synthetic must be explicitly false")
        if manifest.get("testing_only") is not False:
            raise ValueError("production provenance testing_only must be explicitly false")
        if manifest.get("data_class") != "locked_attribute_level_gmnps":
            raise ValueError("production data_class is invalid")
        if manifest.get("n_foods") != 9234:
            raise ValueError("production input must contain 9,234 foods")
    else:
        if manifest.get("synthetic") is not True or manifest.get("testing_only") is not True:
            raise ValueError("test provenance must explicitly declare synthetic/testing_only true")
        if manifest.get("data_class") != "synthetic_test_fixture":
            raise ValueError("test data_class must be synthetic_test_fixture")
    return production


def validate_population_provenance(
    manifest: Mapping[str, object],
    *,
    allow_test_data: bool = False,
) -> None:
    production = _validate_base_population_provenance(manifest)
    if production:
        raise ValueError(
            "DataFrame API is testing/in-memory-only; production requires artifact paths"
        )
    elif not allow_test_data:
        raise ValueError("testing/synthetic provenance is rejected by default")


def verify_phase1_run_artifacts(
    phase1_individual_food: Path,
    phase1_run_manifest: Path,
    phase1_success_marker: Path,
    upstream_input_manifest: Path,
) -> dict[str, str]:
    """Verify live Phase 1 output bytes against `_SUCCESS` and run provenance."""

    table_bytes = _immutable_read(Path(phase1_individual_food), "Phase 1 individual_food")
    run_bytes = _immutable_read(Path(phase1_run_manifest), "Phase 1 run manifest")
    success_bytes = _immutable_read(Path(phase1_success_marker), "Phase 1 _SUCCESS marker")
    upstream_bytes = _immutable_read(Path(upstream_input_manifest), "Phase 1 upstream manifest")
    run = _json_object(run_bytes, "Phase 1 run manifest")
    success = _json_object(success_bytes, "Phase 1 _SUCCESS marker")
    table_hash = sha256(table_bytes).hexdigest()
    run_hash = sha256(run_bytes).hexdigest()
    upstream_hash = sha256(upstream_bytes).hexdigest()
    files = success.get("files")
    if success.get("status") != "complete" or not isinstance(files, dict):
        raise ValueError("Phase 1 _SUCCESS marker is incomplete")
    if files.get("individual_food.csv") != table_hash:
        raise ValueError("Phase 1 _SUCCESS individual_food hash mismatch")
    if files.get("run_manifest.json") != run_hash:
        raise ValueError("Phase 1 _SUCCESS run_manifest hash mismatch")
    expected_run = {
        "method": "attribute_recomposition",
        "method_role": "primary",
        "score_centering": "none",
        "production_label": "production",
        "development_smoke_test": False,
    }
    for field, value in expected_run.items():
        if run.get(field) != value:
            raise ValueError(f"Phase 1 run manifest {field} is not production-locked")
    if run.get("input_manifest_sha256") != upstream_hash:
        raise ValueError("Phase 1 upstream manifest hash mismatch")
    source_hashes = run.get("source_hashes")
    if not isinstance(source_hashes, dict) or source_hashes.get("input_manifest") != upstream_hash:
        raise ValueError("Phase 1 run source_hashes do not bind the upstream manifest")
    return {
        "phase1_individual_food_sha256": table_hash,
        "phase1_run_manifest_sha256": run_hash,
        "phase1_success_marker_sha256": sha256(success_bytes).hexdigest(),
        "upstream_input_manifest_sha256": upstream_hash,
    }


def load_method_lock_artifact_locator(path: Path) -> MethodLockArtifactPaths:
    """Parse an exact locator from immutable bytes; the locator grants no trust."""

    raw = _immutable_read(Path(path), "method-lock artifact locator")
    locator = _json_object(raw, "method-lock artifact locator")
    names = {item.name for item in fields(MethodLockArtifactPaths)}
    if set(locator) != names:
        raise ValueError("method-lock artifact locator fields are not exact")
    return MethodLockArtifactPaths(
        **{name: Path(str(locator[name])) for name in names}
    )


def _trusted_task4_entry(bundle_id: str) -> tuple[dict[str, object], str]:
    try:
        raw = _immutable_read(_TRUSTED_TASK4_REGISTRY_PATH, "trusted Task 4 registry")
    except ValueError as error:
        raise ValueError(
            "production blocked: repository-trusted Task 4 registry is unavailable"
        ) from error
    registry = _json_object(raw, "trusted Task 4 registry")
    if registry.get("schema_version") != "gmnps-attribute-validation-trust-v1":
        raise ValueError("trusted Task 4 registry schema is invalid")
    entries = registry.get("population_bundles")
    matches = [
        entry
        for entry in entries if isinstance(entry, dict) and entry.get("bundle_id") == bundle_id
    ] if isinstance(entries, list) else []
    if len(matches) != 1:
        raise ValueError("production bundle has no unique repository-trusted Task 4 entry")
    return dict(matches[0]), sha256(raw).hexdigest()


def _verify_enrichment(
    task4_bytes: bytes, phase1_bytes: bytes, subgroup_bytes: bytes
) -> pd.DataFrame:
    from io import BytesIO

    try:
        task4 = pd.read_csv(BytesIO(task4_bytes))
        phase1 = pd.read_csv(BytesIO(phase1_bytes))
        subgroup = pd.read_csv(BytesIO(subgroup_bytes))
    except Exception as error:
        raise ValueError("production score or subgroup table is not readable CSV") from error
    required = {"food_id", "food_group", "food_subgroup"}
    if not required.issubset(subgroup.columns) or subgroup["food_id"].duplicated().any():
        raise ValueError("subgroup enrichment must uniquely map food_id to group/subgroup")
    if "food_subgroup" in phase1.columns:
        raise ValueError("Phase 1 table unexpectedly contains food_subgroup")
    merged = phase1.merge(
        subgroup.loc[:, ["food_id", "food_group", "food_subgroup"]],
        on=["food_id", "food_group"],
        how="left",
        validate="many_to_one",
    )
    if merged["food_subgroup"].isna().any():
        raise ValueError("subgroup enrichment does not cover the Phase 1 food panel")
    if list(task4.columns) != list(merged.columns):
        raise ValueError("Task 4 table columns differ from Phase 1 plus subgroup enrichment")
    try:
        pd.testing.assert_frame_equal(task4, merged, check_dtype=False, check_like=False)
    except AssertionError as error:
        raise ValueError("Task 4 table is not the exact Phase 1 table plus subgroup enrichment") from error
    return task4


def _load_population_artifacts(
    paths: PopulationArtifactPaths, config: PopulationSafetyConfig
) -> tuple[pd.DataFrame, dict[str, object], dict[str, str]]:
    """Verify the complete live artifact chain and return the same read bytes."""

    provenance_bytes = _immutable_read(paths.task4_provenance, "Task 4 provenance")
    provenance = _json_object(provenance_bytes, "Task 4 provenance")
    if not _validate_base_population_provenance(provenance):
        raise ValueError("live verification is only valid for production provenance")
    entry, registry_hash = _trusted_task4_entry(str(provenance["bundle_id"]))
    bindings = task4_binding_hashes(config)
    for field, observed in bindings.items():
        if entry.get(field) != observed:
            raise ValueError(f"trusted Task 4 registry {field} mismatch")

    phase1_hashes = verify_phase1_run_artifacts(
        paths.phase1_individual_food,
        paths.phase1_run_manifest,
        paths.phase1_success_marker,
        paths.upstream_input_manifest,
    )
    for field, observed in phase1_hashes.items():
        if entry.get(field) != observed:
            raise ValueError(f"trusted Task 4 registry {field} mismatch")

    method_bytes = _immutable_read(paths.method_lock_manifest, "method-lock manifest")
    method_manifest = _json_object(method_bytes, "method-lock manifest")
    method_hash = sha256(method_bytes).hexdigest()
    if entry.get("method_lock_manifest_sha256") != method_hash:
        raise ValueError("trusted Task 4 registry method-lock hash mismatch")
    expected_registry = entry.get("phase1_release_registry_sha256")
    if not isinstance(expected_registry, str):
        raise ValueError("trusted Task 4 registry lacks Phase 1 release-registry hash")
    validate_method_lock_manifest(
        method_manifest,
        paths.method_lock_artifacts,
        expected_release_registry_sha256=expected_registry,
    )
    if _immutable_read(paths.method_lock_manifest, "method-lock manifest") != method_bytes:
        raise ValueError("method-lock manifest changed during verification")

    scoring_beta_bytes = _immutable_read(
        paths.method_lock_artifacts.scoring_beta, "scoring beta"
    )
    scoring_impl_bytes = _immutable_read(_SCORING_IMPLEMENTATION, "scoring implementation")
    locator_bytes = _immutable_read(
        paths.method_lock_artifact_locator, "method-lock artifact locator"
    )
    if load_method_lock_artifact_locator(
        paths.method_lock_artifact_locator
    ) != paths.method_lock_artifacts:
        raise ValueError("method-lock artifact locator changed or does not match paths")
    subgroup_bytes = _immutable_read(paths.subgroup_enrichment, "subgroup enrichment")
    task4_bytes = _immutable_read(paths.task4_input_table, "Task 4 input table")
    phase1_bytes = _immutable_read(paths.phase1_individual_food, "Phase 1 individual_food")
    if sha256(phase1_bytes).hexdigest() != phase1_hashes[
        "phase1_individual_food_sha256"
    ]:
        raise ValueError("Phase 1 individual_food changed during complete verification")
    if sha256(
        _immutable_read(paths.upstream_input_manifest, "Phase 1 upstream manifest")
    ).hexdigest() != phase1_hashes["upstream_input_manifest_sha256"]:
        raise ValueError("Phase 1 upstream manifest changed during complete verification")
    validate_method_lock_manifest(
        method_manifest,
        paths.method_lock_artifacts,
        expected_release_registry_sha256=expected_registry,
    )
    live_hashes = {
        **phase1_hashes,
        "task4_input_table_sha256": sha256(task4_bytes).hexdigest(),
        "task4_provenance_sha256": sha256(provenance_bytes).hexdigest(),
        "method_lock_manifest_sha256": method_hash,
        "method_lock_artifact_locator_sha256": sha256(locator_bytes).hexdigest(),
        "scoring_beta_file_sha256": sha256(scoring_beta_bytes).hexdigest(),
        "scoring_implementation_sha256": sha256(scoring_impl_bytes).hexdigest(),
        "subgroup_enrichment_sha256": sha256(subgroup_bytes).hexdigest(),
        "task4_trust_registry_sha256": registry_hash,
        **bindings,
    }
    for field in (
        "task4_input_table_sha256",
        "task4_provenance_sha256",
        "method_lock_artifact_locator_sha256",
        "scoring_beta_file_sha256",
        "scoring_implementation_sha256",
        "subgroup_enrichment_sha256",
    ):
        if entry.get(field) != live_hashes[field]:
            raise ValueError(f"trusted Task 4 registry {field} mismatch")
    frame = _verify_enrichment(task4_bytes, phase1_bytes, subgroup_bytes)
    scoring_payload = _json_object(scoring_beta_bytes, "scoring beta")
    scoring_ids = scoring_payload.get("participant_ids")
    if (
        not isinstance(scoring_ids, list)
        or set(map(str, scoring_ids)) != set(frame["individual_id"].astype(str))
    ):
        raise ValueError("scoring beta participant IDs do not match Task 4 people")
    return frame, provenance, live_hashes


def _invariant(frame: pd.DataFrame, unit: str, column: str) -> None:
    counts = frame.groupby(unit, dropna=False)[column].nunique(dropna=False)
    if (counts > 1).any():
        raise ValueError(f"{column} is inconsistent within {unit}")


def _validate_frame(
    frame: pd.DataFrame, manifest: Mapping[str, object]
) -> tuple[pd.DataFrame, str, dict[str, object]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("individual_food must be a nonempty DataFrame")
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"individual_food missing required columns: {sorted(missing)}")
    normalized = {str(column).strip().lower() for column in frame.columns}
    leaked = sorted(normalized & _FORBIDDEN_COLUMNS)
    if leaked:
        raise ValueError(f"population input contains outcome/label leakage: {leaked}")
    if frame.duplicated(["individual_id", "food_id"]).any():
        raise ValueError("individual_food must contain one row per individual_id-food_id")
    working = frame.copy()
    for column in ("individual_id", "food_id", "food_group", "food_subgroup"):
        if working[column].isna().any() or working[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must be nonempty")
        working[column] = working[column].astype(str).str.strip()
    for column in ("FCS2", "GMNPS_score", "GMNPS_delta"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
        if not np.isfinite(working[column]).all():
            raise ValueError(f"{column} must contain only finite values")
    if not working["FCS2"].between(0, 100).all() or not working["GMNPS_score"].between(0, 100).all():
        raise ValueError("FCS2 and GMNPS_score must lie in [0, 100]")
    if not np.allclose(
        working["GMNPS_delta"],
        working["GMNPS_score"] - working["FCS2"],
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("GMNPS_delta must equal GMNPS_score - FCS2")
    if "method_role" in working and not working["method_role"].eq("primary").all():
        raise ValueError("method_role must be primary")
    for column in ("FCS2", "food_group", "food_subgroup"):
        _invariant(working, "food_id", column)
    parents = working[["food_subgroup", "food_group"]].drop_duplicates().groupby(
        "food_subgroup"
    )["food_group"].nunique()
    if (parents != 1).any():
        raise ValueError("each food_subgroup must have a unique parent food_group")

    panel = tuple(sorted(working["food_id"].unique()))
    panels = working.groupby("individual_id")["food_id"].apply(
        lambda values: tuple(sorted(values))
    )
    expected_rows = len(panels) * len(panel)
    if len(working) != expected_rows or not panels.map(lambda value: value == panel).all():
        raise ValueError("all people must have an identical complete food panel")
    if int(manifest.get("n_foods", -1)) != len(panel):
        raise ValueError("manifest n_foods does not match the complete panel")

    cluster_column = str(manifest["independent_unit"])
    if cluster_column != "individual_id":
        if cluster_column not in working.columns:
            raise ValueError(f"declared independent unit {cluster_column} is missing")
        if (
            working[cluster_column].isna().any()
            or working[cluster_column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"{cluster_column} must be nonempty")
        working[cluster_column] = working[cluster_column].astype(str).str.strip()
        _invariant(working, "individual_id", cluster_column)
    effective = working.groupby("food_id")[cluster_column].nunique()
    panel_audit = {
        "panel_status": "complete_rectangular",
        "n_individuals": int(working["individual_id"].nunique()),
        "n_independent_units": int(working[cluster_column].nunique()),
        "n_foods": int(len(panel)),
        "expected_rows": int(expected_rows),
        "observed_rows": int(len(working)),
        "min_effective_units_per_food": int(effective.min()),
        "max_effective_units_per_food": int(effective.max()),
    }
    return working, cluster_column, panel_audit


def _category(value: float) -> str:
    return "encourage" if value >= 70 else "moderate" if value >= 31 else "minimize"


def _spearman(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.rank(method="average").corr(right.rank(method="average")))


def _food_level(frame: pd.DataFrame, cluster_column: str) -> pd.DataFrame:
    return frame.groupby("food_id", sort=True).agg(
        FCS2=("FCS2", "first"),
        GMNPS_population_mean=("GMNPS_score", "mean"),
        food_group=("food_group", "first"),
        food_subgroup=("food_subgroup", "first"),
        n_individuals=("individual_id", "nunique"),
        n_effective_units=(cluster_column, "nunique"),
    ).reset_index()


def _iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


def _convergence(food: pd.DataFrame, levels: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = levels[0] if len(levels) == 1 else levels
    for key, group in food.groupby(grouper, sort=True):
        keys = (key,) if len(levels) == 1 else tuple(key)
        baseline = group["FCS2"].to_numpy(float)
        personalized = group["GMNPS_population_mean"].to_numpy(float)
        row: dict[str, object] = {
            "stratum_level": "+".join(levels),
            "stratum": " :: ".join(str(value) for value in keys),
            "n_foods": int(len(group)),
            "n_individuals": int(group["n_individuals"].min()),
            "n_effective_units_per_food_min": int(group["n_effective_units"].min()),
            "n_effective_units_per_food_max": int(group["n_effective_units"].max()),
            "fcs2_median": float(np.median(baseline)),
            "gmnps_population_median": float(np.median(personalized)),
            "median_shift": float(np.median(personalized - baseline)),
            "mean_absolute_food_shift": float(np.mean(np.abs(personalized - baseline))),
            "fcs2_iqr": _iqr(group["FCS2"]),
            "gmnps_population_iqr": _iqr(group["GMNPS_population_mean"]),
            "distribution_l1_quantile_distance": float(
                np.mean(np.abs(np.sort(personalized) - np.sort(baseline)))
            ),
            "within_stratum_spearman": _spearman(
                group["FCS2"], group["GMNPS_population_mean"]
            ),
            "evidence_role": POPULATION_EVIDENCE_ROLE,
        }
        row.update(dict(zip(levels, keys)))
        rows.append(row)
    return pd.DataFrame(rows)


def _discrimination(food: pd.DataFrame) -> pd.DataFrame:
    means = food.groupby("food_group", sort=True).agg(
        FCS2=("FCS2", "mean"),
        GMNPS=("GMNPS_population_mean", "mean"),
        n_foods=("food_id", "nunique"),
    )
    rows = []
    for left, right in combinations(means.index, 2):
        baseline = float(means.loc[left, "FCS2"] - means.loc[right, "FCS2"])
        personalized = float(means.loc[left, "GMNPS"] - means.loc[right, "GMNPS"])
        if np.isclose(baseline, 0.0):
            status = "not_estimable"
        elif np.isclose(personalized, 0.0):
            status = "collapsed"
        elif np.sign(baseline) != np.sign(personalized):
            status = "reversed"
        else:
            status = "preserved"
        rows.append(
            {
                "stratum_a": left,
                "stratum_b": right,
                "n_foods_a": int(means.loc[left, "n_foods"]),
                "n_foods_b": int(means.loc[right, "n_foods"]),
                "fcs2_mean_difference": baseline,
                "gmnps_population_mean_difference": personalized,
                "discrimination_status": status,
                "direction_preserved": status == "preserved",
                "absolute_difference_ratio": (
                    abs(personalized) / abs(baseline)
                    if status != "not_estimable"
                    else float("nan")
                ),
                "evidence_role": POPULATION_EVIDENCE_ROLE,
            }
        )
    return pd.DataFrame(rows)


def _transitions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["fcs2_category"] = work["FCS2"].map(_category)
    work["gmnps_category"] = work["GMNPS_score"].map(_category)
    rows = []
    for origin, destination in product(_CATEGORY_ORDER, repeat=2):
        selected = work[
            work["fcs2_category"].eq(origin) & work["gmnps_category"].eq(destination)
        ]
        denominator = int(work["fcs2_category"].eq(origin).sum())
        movement = _CATEGORY_INDEX[destination] - _CATEGORY_INDEX[origin]
        rows.append(
            {
                "fcs2_category": origin,
                "gmnps_category": destination,
                "direction": "up" if movement > 0 else "down" if movement < 0 else "stable",
                "category_steps": abs(movement),
                "n_individual_foods": int(len(selected)),
                "origin_fraction": len(selected) / denominator if denominator else 0.0,
                "mean_delta": float(selected["GMNPS_delta"].mean()) if len(selected) else np.nan,
                "median_absolute_delta": float(selected["GMNPS_delta"].abs().median()) if len(selected) else np.nan,
                "evidence_role": POPULATION_EVIDENCE_ROLE,
            }
        )
    extreme = work[
        (work["fcs2_category"].eq("encourage") & work["gmnps_category"].eq("minimize"))
        | (work["fcs2_category"].eq("minimize") & work["gmnps_category"].eq("encourage"))
    ].copy()
    extreme["reversal_type"] = extreme["fcs2_category"] + "_to_" + extreme["gmnps_category"]
    extreme["absolute_delta"] = extreme["GMNPS_delta"].abs()
    extreme["audit_status"] = "flagged_for_clinical_nutritional_review"
    extreme["evidence_role"] = POPULATION_EVIDENCE_ROLE
    columns = [
        "individual_id", "food_id", "food_group", "food_subgroup", "FCS2",
        "GMNPS_score", "GMNPS_delta", "fcs2_category", "gmnps_category",
        "reversal_type", "absolute_delta", "audit_status", "evidence_role",
    ]
    return pd.DataFrame(rows), extreme.loc[:, columns].reset_index(drop=True)


def _bootstrap(
    frame: pd.DataFrame,
    cluster_column: str,
    config: PopulationSafetyConfig,
) -> pd.DataFrame:
    clusters = sorted(frame[cluster_column].unique())
    food = _food_level(frame, cluster_column)
    estimate = _spearman(food["FCS2"], food["GMNPS_population_mean"])
    base = {
        "metric": "spearman_fcs2_vs_population_gmnps",
        "estimand": "food-level population mean",
        "estimate": estimate,
        "confidence_level": config.confidence_level,
        "ci_method": "independent_cluster_percentile",
        "cluster_column": cluster_column,
        "n_clusters": len(clusters),
        "n_individuals": frame["individual_id"].nunique(),
        "replicates_requested": config.bootstrap_replicates,
        "bootstrap_seed": config.bootstrap_seed,
        "minimum_clusters": config.minimum_clusters,
        "evidence_role": POPULATION_EVIDENCE_ROLE,
    }
    if len(clusters) < config.minimum_clusters:
        return pd.DataFrame(
            [{
                **base,
                "analysis_status": "not_estimable",
                "reason": "fewer_than_minimum_independent_clusters",
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "replicates_valid": 0,
            }]
        )
    rng = np.random.default_rng(config.bootstrap_seed)
    values = []
    for _ in range(config.bootstrap_replicates):
        sampled = rng.choice(clusters, len(clusters), replace=True)
        replicate = pd.concat(
            [frame[frame[cluster_column].eq(cluster)] for cluster in sampled],
            ignore_index=True,
        )
        replicate_food = _food_level(replicate, cluster_column)
        value = _spearman(
            replicate_food["FCS2"], replicate_food["GMNPS_population_mean"]
        )
        if np.isfinite(value):
            values.append(value)
    if len(values) < config.minimum_valid_replicates:
        return pd.DataFrame(
            [{
                **base,
                "analysis_status": "not_estimable",
                "reason": "fewer_than_minimum_valid_bootstrap_replicates",
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "replicates_valid": len(values),
            }]
        )
    alpha = (1 - config.confidence_level) / 2
    lower, upper = np.quantile(values, [alpha, 1 - alpha])
    return pd.DataFrame(
        [{
            **base,
            "analysis_status": "estimable",
            "reason": "",
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "replicates_valid": len(values),
        }]
    )


def _audit_population_frame(
    individual_food: pd.DataFrame,
    provenance_manifest: Mapping[str, object],
    *,
    config: PopulationSafetyConfig,
    verified_input_hashes: Mapping[str, str] | None,
) -> PopulationSafetyAudit:
    frame, cluster_column, panel_audit = _validate_frame(
        individual_food, provenance_manifest
    )
    food = _food_level(frame, cluster_column)
    transition, reversal = _transitions(frame)
    global_summary = {
        "evidence_role": POPULATION_EVIDENCE_ROLE,
        "analysis_boundary": "design_audit_only",
        "estimand": "food-level population mean",
        "score_centering": "none",
        **panel_audit,
        "spearman_fcs2_vs_population_gmnps": _spearman(
            food["FCS2"], food["GMNPS_population_mean"]
        ),
        "mean_absolute_food_shift": float(
            (food["GMNPS_population_mean"] - food["FCS2"]).abs().mean()
        ),
        "implausible_reversal_count": len(reversal),
        "implausible_reversal_fraction": len(reversal) / len(frame),
    }
    return PopulationSafetyAudit(
        evidence_role=POPULATION_EVIDENCE_ROLE,
        panel_audit=panel_audit,
        global_summary=global_summary,
        group_convergence=_convergence(food, ["food_group"]),
        subgroup_convergence=_convergence(food, ["food_group", "food_subgroup"]),
        between_group_discrimination=_discrimination(food),
        transition_matrix=transition,
        reversal_audit=reversal,
        bootstrap_summary=_bootstrap(frame, cluster_column, config),
        verified_input_hashes=(dict(verified_input_hashes) if verified_input_hashes else None),
    )


def audit_population_safety(
    individual_food: pd.DataFrame,
    provenance_manifest: Mapping[str, object],
    *,
    config: PopulationSafetyConfig | None = None,
    allow_test_data: bool = False,
) -> PopulationSafetyAudit:
    """Testing/in-memory-only population audit; production is path-only."""

    validate_population_provenance(
        provenance_manifest, allow_test_data=allow_test_data
    )
    return _audit_population_frame(
        individual_food,
        provenance_manifest,
        config=config or PopulationSafetyConfig(),
        verified_input_hashes=None,
    )


def audit_population_safety_from_artifacts(
    paths: PopulationArtifactPaths,
    *,
    config: PopulationSafetyConfig | None = None,
) -> PopulationSafetyAudit:
    """Verify production paths and analyze the exact verified bytes in one call."""

    resolved = config or PopulationSafetyConfig()
    frame, provenance, hashes = _load_population_artifacts(paths, resolved)
    return _audit_population_frame(
        frame,
        provenance,
        config=resolved,
        verified_input_hashes=hashes,
    )


__all__ = [
    "POPULATION_EVIDENCE_ROLE",
    "PopulationArtifactPaths",
    "PopulationSafetyAudit",
    "PopulationSafetyConfig",
    "audit_population_safety",
    "audit_population_safety_from_artifacts",
    "load_method_lock_artifact_locator",
    "task4_binding_hashes",
    "validate_population_provenance",
    "verify_phase1_run_artifacts",
]
