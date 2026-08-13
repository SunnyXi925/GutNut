"""Supporting biological and adjudicated-path consistency analyses."""
from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import stat
from typing import Mapping

import numpy as np
import pandas as pd

from gmnps.validation.attribute_population_safety import (
    load_method_lock_artifact_locator,
    verify_phase1_run_artifacts,
)
from gmnps.validation.method_lock_gate import (
    MethodLockArtifactPaths,
    validate_method_lock_manifest,
)


SUPPORTING_SCHEMA_VERSION = "gmnps-supporting-evidence-v2"
BIOLOGICAL_CONSISTENCY = "biological_consistency"
MECHANISTIC_CONSISTENCY = "mechanistic_consistency"
_ROLE_BY_SOURCE = {
    "gmrepo_disease_labels": BIOLOGICAL_CONSISTENCY,
    "zoe_aggregate_ranks": BIOLOGICAL_CONSISTENCY,
    "microbiome_shuffle_rerun": BIOLOGICAL_CONSISTENCY,
    "knowledge_paths": MECHANISTIC_CONSISTENCY,
}
_PRODUCTION_CLASS_BY_SOURCE = {
    "gmrepo_disease_labels": "observed_supporting_labels",
    "zoe_aggregate_ranks": "published_aggregate_ranks",
    "microbiome_shuffle_rerun": "locked_microbiome_shuffle_rerun",
    "knowledge_paths": "curated_knowledge_paths",
}
_OUTCOME_COLUMNS = frozenset(
    {
        "outcome",
        "y_true",
        "glucose_iauc_2h",
        "tg_6h_rise",
        "c_peptide_iauc_2h",
        "postprandial_response",
    }
)
_ROOT = Path(__file__).resolve().parents[4]
_TRUSTED_SUPPORTING_REGISTRY_PATH = (
    _ROOT / "code/src/configs/supporting_evidence_trust_registry.json"
)
_SCORING_IMPLEMENTATION = _ROOT / "code/src/scripts/run_attribute_gmnps.py"


@dataclass(frozen=True)
class BiologicalConsistencyConfig:
    bootstrap_replicates: int = 1000
    shuffle_replicates: int = 1000
    seed: int = 20260813
    confidence_level: float = 0.95
    minimum_valid_replicates: int = 800
    minimum_arm_units: int = 10

    def __post_init__(self) -> None:
        for name in (
            "bootstrap_replicates",
            "shuffle_replicates",
            "seed",
            "minimum_valid_replicates",
            "minimum_arm_units",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a strict integer")
        if self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")
        if self.shuffle_replicates <= 0:
            raise ValueError("shuffle_replicates must be positive")
        if not 1 <= self.minimum_valid_replicates <= self.bootstrap_replicates:
            raise ValueError(
                "minimum_valid_replicates must be between 1 and bootstrap_replicates"
            )
        if self.minimum_arm_units < 2:
            raise ValueError("minimum_arm_units must be at least 2")
        if not isinstance(self.confidence_level, (int, float)) or isinstance(
            self.confidence_level, bool
        ):
            raise ValueError("confidence_level must be numeric")
        if not 0 < float(self.confidence_level) < 1:
            raise ValueError("confidence_level must be between zero and one")


@dataclass(frozen=True)
class MicrobiomeShuffleArtifactPaths:
    """Paths needed to verify two complete Phase 1 scoring runs."""

    provenance: Path
    original_score_table: Path
    original_run_manifest: Path
    original_success_marker: Path
    original_input_manifest: Path
    rerun_score_table: Path
    rerun_run_manifest: Path
    rerun_success_marker: Path
    rerun_input_manifest: Path
    mapping_sidecar: Path
    original_scoring_beta: Path
    rerun_scoring_beta: Path
    method_lock_manifest: Path
    method_lock_artifact_locator: Path
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
class DiseaseConsistencyResult:
    evidence_role: str
    estimates: pd.DataFrame
    inference: pd.DataFrame
    label_permutation: pd.DataFrame
    microbiome_shuffle: pd.DataFrame
    verified_input_hashes: Mapping[str, str] | None = None


def canonical_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash a table with explicit columns, scalar values, and row order."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    def scalar(value: object) -> object:
        if pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    payload = {
        "columns": [str(column) for column in frame.columns],
        "rows": [[scalar(value) for value in row] for row in frame.itertuples(index=False, name=None)],
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value) - set("0123456789abcdef")
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_mapping_sha256(payload: Mapping[str, object]) -> str:
    return sha256(
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _nonempty(manifest: Mapping[str, object], field: str) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"provenance {field} must be nonempty")
    return value.strip()


def _base_evidence_provenance(manifest: Mapping[str, object]) -> bool:
    if not isinstance(manifest, Mapping):
        raise ValueError("provenance manifest must be an object")
    if manifest.get("schema_version") != SUPPORTING_SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SUPPORTING_SCHEMA_VERSION}")
    source = manifest.get("source_kind")
    if source not in _ROLE_BY_SOURCE:
        raise ValueError("source_kind is unsupported")
    if manifest.get("evidence_role") != _ROLE_BY_SOURCE[str(source)]:
        raise ValueError("evidence_role does not match the source-specific schema")
    _nonempty(manifest, "source_id")
    _nonempty(manifest, "accession")
    _sha(manifest.get("table_sha256"), "table_sha256")
    if manifest.get("source_verified") is not True:
        raise ValueError("source_verified must be true")
    if manifest.get("scoring_frozen_before_labels") is not True:
        raise ValueError("scoring must be frozen before labels")
    if manifest.get("outcome_columns") != []:
        raise ValueError("outcome_columns must be an empty list")
    if source == "gmrepo_disease_labels" and manifest.get("label_access_authorized") is not True:
        raise ValueError("GMrepo label access must be authorized")
    production = manifest.get("production_label") == "production"
    if production:
        if manifest.get("synthetic") is not False:
            raise ValueError("production provenance synthetic must be explicitly false")
        if manifest.get("testing_only") is not False:
            raise ValueError("production provenance testing_only must be explicitly false")
        if manifest.get("data_class") != _PRODUCTION_CLASS_BY_SOURCE[str(source)]:
            raise ValueError("production data_class does not match source_kind")
    else:
        if manifest.get("synthetic") is not True or manifest.get("testing_only") is not True:
            raise ValueError("test evidence must explicitly declare synthetic/testing_only true")
        if manifest.get("data_class") != "synthetic_test_fixture":
            raise ValueError("test data_class must be synthetic_test_fixture")
    return production


def validate_evidence_provenance(
    manifest: Mapping[str, object],
    *,
    allow_test_data: bool = False,
) -> None:
    production = _base_evidence_provenance(manifest)
    if production:
        raise ValueError(
            "DataFrame API is testing/in-memory-only; production requires artifact paths"
        )
    elif not allow_test_data:
        raise ValueError("testing/synthetic supporting evidence is rejected by default")


def _immutable_read(path: Path, label: str) -> bytes:
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        chunks = []
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


def _load_verified_supporting_evidence(
    table_path: Path,
    provenance_path: Path,
    expected_source_kind: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, str], dict[str, object]]:
    """Read, registry-check, and parse the exact production bytes once."""

    table_path = Path(table_path)
    provenance_path = Path(provenance_path)
    table_bytes = _immutable_read(table_path, "supporting evidence table")
    provenance_bytes = _immutable_read(provenance_path, "supporting evidence provenance")
    try:
        manifest = json.loads(provenance_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("supporting evidence provenance is invalid") from error
    if not isinstance(manifest, dict):
        raise ValueError("supporting evidence provenance must be a JSON object")
    if not _base_evidence_provenance(manifest):
        raise ValueError("live source verification is only for production evidence")
    if expected_source_kind is not None and manifest.get("source_kind") != expected_source_kind:
        raise ValueError(f"production analysis requires {expected_source_kind} provenance")
    table_hash = sha256(table_bytes).hexdigest()
    if table_hash != manifest.get("table_sha256"):
        raise ValueError("supporting evidence table content hash mismatch")
    provenance_hash = _canonical_mapping_sha256(manifest)
    try:
        registry_bytes = _immutable_read(
            _TRUSTED_SUPPORTING_REGISTRY_PATH, "trusted supporting-evidence registry"
        )
    except ValueError as error:
        raise ValueError(
            "production blocked: trusted supporting-evidence registry is unavailable"
        ) from error
    try:
        registry = json.loads(registry_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("trusted supporting-evidence registry is invalid") from error
    if (
        not isinstance(registry, dict)
        or registry.get("schema_version") != "gmnps-supporting-evidence-trust-v1"
        or not isinstance(registry.get("sources"), list)
    ):
        raise ValueError("trusted supporting-evidence registry schema is invalid")
    matches = [
        entry
        for entry in registry["sources"]
        if isinstance(entry, dict)
        and entry.get("source_kind") == manifest.get("source_kind")
        and entry.get("source_id") == manifest.get("source_id")
        and entry.get("accession") == manifest.get("accession")
        and entry.get("table_sha256") == table_hash
        and entry.get("provenance_sha256") == provenance_hash
    ]
    if len(matches) != 1:
        raise ValueError("supporting source has no unique repository-trusted digest entry")
    # Detect a path swap or edit between trust resolution and analysis.
    if _immutable_read(table_path, "supporting evidence table") != table_bytes:
        raise ValueError("supporting evidence table changed during verification")
    if _immutable_read(provenance_path, "supporting evidence provenance") != provenance_bytes:
        raise ValueError("supporting evidence provenance changed during verification")
    try:
        frame = pd.read_csv(BytesIO(table_bytes))
    except Exception as error:
        raise ValueError("supporting evidence table is not readable CSV") from error
    hashes = {
        "table_sha256": table_hash,
        "provenance_file_sha256": sha256(provenance_bytes).hexdigest(),
        "provenance_sha256": provenance_hash,
        "trusted_registry_sha256": sha256(registry_bytes).hexdigest(),
    }
    return frame, manifest, hashes, dict(matches[0])


def verify_supporting_evidence_file(
    table_path: Path, provenance_path: Path
) -> dict[str, str]:
    """Path-only live verification; the result is descriptive, not authority."""

    _, _, hashes, _ = _load_verified_supporting_evidence(table_path, provenance_path)
    return hashes


def _scoring_contract(manifest: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "method",
        "method_role",
        "mapping_version",
        "scoring_version",
        "score_centering",
        "beta_centering",
        "bundle_fingerprint",
        "model_fingerprint",
    )
    missing = [field for field in fields if field not in manifest]
    if missing:
        raise ValueError(f"scoring manifest lacks locked contract fields: {missing}")
    return {field: manifest[field] for field in fields}


def _json_bytes_object(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _manifest_file_digest(manifest: Mapping[str, object], name: str) -> str:
    files = manifest.get("files")
    if not isinstance(files, dict) or name not in files:
        raise ValueError(f"input manifest lacks files.{name}")
    entry = files[name]
    digest = entry if isinstance(entry, str) else entry.get("sha256") if isinstance(entry, dict) else None
    return _sha(digest, f"input manifest files.{name}")


def _canonical_beta(raw: bytes, label: str) -> tuple[list[str], list[str], np.ndarray]:
    payload = _json_bytes_object(raw, label)
    if payload.get("schema_version") != "gmnps-canonical-beta-v1":
        raise ValueError(f"{label} schema_version is invalid")
    people = payload.get("participant_ids")
    nutrients = payload.get("nutrient_order")
    values = payload.get("values")
    if not isinstance(people, list) or not people or len(set(map(str, people))) != len(people):
        raise ValueError(f"{label} participant_ids must be unique and nonempty")
    if not isinstance(nutrients, list) or not nutrients or len(set(map(str, nutrients))) != len(nutrients):
        raise ValueError(f"{label} nutrient_order must be unique and nonempty")
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} values are invalid") from error
    if matrix.shape != (len(people), len(nutrients)) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} values must be a finite participant-by-nutrient matrix")
    return [str(value) for value in people], [str(value) for value in nutrients], matrix


def _verify_deranged_beta(
    original_scores: pd.DataFrame,
    sidecar: pd.DataFrame,
    original_beta_bytes: bytes,
    rerun_beta_bytes: bytes,
) -> None:
    original_people, nutrients, original_values = _canonical_beta(
        original_beta_bytes, "original scoring beta"
    )
    rerun_people, rerun_nutrients, rerun_values = _canonical_beta(
        rerun_beta_bytes, "rerun scoring beta"
    )
    if nutrients != rerun_nutrients:
        raise ValueError("rerun beta must preserve nutrient order")
    score_people = set(original_scores["individual_id"].astype(str))
    normalized, _ = _validate_person_mapping_sidecar(sidecar, score_people)
    if set(original_people) != score_people or set(rerun_people) != score_people:
        raise ValueError(
            "target/source individual sets must exactly equal score-table and beta person IDs"
        )
    original_index = {person: index for index, person in enumerate(original_people)}
    rerun_index = {person: index for index, person in enumerate(rerun_people)}
    for row in normalized.itertuples(index=False):
        if not np.array_equal(
            rerun_values[rerun_index[row.target_individual_id]],
            original_values[original_index[row.source_individual_id]],
        ):
            raise ValueError(
                "rerun beta target individual is not exactly the declared source individual"
            )


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{label} must be a nonempty DataFrame")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")


def _reject_outcomes(frame: pd.DataFrame) -> None:
    leaked = sorted({str(column).strip().lower() for column in frame.columns} & _OUTCOME_COLUMNS)
    if leaked:
        raise ValueError(f"supporting input contains outcome/label leakage: {leaked}")


_PERSON_MAPPING_SIDECAR_COLUMNS = (
    "target_individual_id",
    "source_individual_id",
    "target_component_id",
    "source_component_id",
)


def _validate_person_mapping_sidecar(
    sidecar: pd.DataFrame,
    score_people: set[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    _require_columns(sidecar, set(_PERSON_MAPPING_SIDECAR_COLUMNS), "mapping sidecar")
    _reject_outcomes(sidecar)
    if set(sidecar.columns) != set(_PERSON_MAPPING_SIDECAR_COLUMNS):
        raise ValueError(
            "mapping sidecar must contain exactly target/source individual/component columns"
        )
    work = sidecar.loc[:, _PERSON_MAPPING_SIDECAR_COLUMNS].copy()
    for column in _PERSON_MAPPING_SIDECAR_COLUMNS:
        if work[column].isna().any() or work[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"mapping sidecar {column} must be nonempty")
        work[column] = work[column].astype(str).str.strip()
    target_people = set(work["target_individual_id"])
    source_people = set(work["source_individual_id"])
    if target_people != score_people or source_people != score_people:
        raise ValueError(
            "target/source individual sets must exactly equal scorer table/beta person IDs"
        )
    if (
        work["target_individual_id"].duplicated().any()
        or work["source_individual_id"].duplicated().any()
        or target_people != source_people
    ):
        raise ValueError("mapping sidecar target-to-source people must be bijective")
    if work["target_individual_id"].eq(work["source_individual_id"]).any():
        raise ValueError("mapping sidecar must be a person-level derangement")

    membership = dict(
        zip(work["target_individual_id"], work["target_component_id"])
    )
    expected_source_components = work["source_individual_id"].map(membership)
    if not expected_source_components.eq(work["source_component_id"]).all():
        raise ValueError(
            "source_component_id must equal invariant component membership of source individual"
        )
    component_pairs = work[
        ["target_component_id", "source_component_id"]
    ].drop_duplicates()
    if (
        component_pairs["target_component_id"].duplicated().any()
        or component_pairs["source_component_id"].duplicated().any()
        or set(component_pairs["target_component_id"])
        != set(component_pairs["source_component_id"])
    ):
        raise ValueError("mapping sidecar component mapping must be bijective")
    if component_pairs["target_component_id"].eq(
        component_pairs["source_component_id"]
    ).any():
        raise ValueError("mapping sidecar must be a component-level derangement")
    component_map = dict(
        zip(
            component_pairs["target_component_id"],
            component_pairs["source_component_id"],
        )
    )
    component_sizes = work.groupby("target_component_id")[
        "target_individual_id"
    ].size()
    for target_component, source_component in component_map.items():
        if int(component_sizes[target_component]) != int(component_sizes[source_component]):
            raise ValueError("deranged components must have compatible person counts")
    return work, component_map


def _person_invariant(frame: pd.DataFrame, column: str) -> None:
    counts = frame.groupby("individual_id", dropna=False)[column].nunique(dropna=False)
    if (counts > 1).any():
        raise ValueError(f"inconsistent {column} within individual_id")


def _stable_seed(seed: int, *parts: object) -> int:
    raw = "|".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(sha256(raw).digest()[:8], "big") % (2**32)


def _prepare_gmrepo(frame: pd.DataFrame, value_column: str):
    required = {
        "individual_id", "food_id", "cohort_id", "disease_label",
        "label_source", "label_provenance", value_column,
    }
    _require_columns(frame, required, "GMrepo table")
    _reject_outcomes(frame)
    if frame.duplicated(["individual_id", "food_id"]).any():
        raise ValueError("GMrepo table must contain unique person-food rows")
    work = frame.copy()
    for column in (
        "individual_id", "food_id", "cohort_id", "disease_label",
        "label_source", "label_provenance",
    ):
        if work[column].isna().any() or work[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must be nonempty")
        work[column] = work[column].astype(str).str.strip()
        if column != "food_id":
            _person_invariant(work, column)
    panel = tuple(sorted(work["food_id"].unique()))
    panels = work.groupby("individual_id")["food_id"].apply(
        lambda values: tuple(sorted(values))
    )
    if len(work) != len(panels) * len(panel) or not panels.map(lambda value: value == panel).all():
        raise ValueError("GMrepo people/components must share an identical predeclared food panel")
    cluster = "component_id" if "component_id" in work.columns else "individual_id"
    if cluster != "individual_id":
        if work[cluster].isna().any() or work[cluster].astype(str).str.strip().eq("").any():
            raise ValueError("component_id must be nonempty")
        work[cluster] = work[cluster].astype(str).str.strip()
        _person_invariant(work, cluster)
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    if np.isinf(work[value_column]).any():
        raise ValueError(f"{value_column} cannot contain infinite values")

    # Freeze the independent-unit complete-case set before estimating any
    # cohort-by-food location or scale. A component is excluded if any person
    # assigned to it has any non-finite panel value.
    person_complete = work.groupby("individual_id")[value_column].apply(
        lambda values: bool(values.notna().all())
    )
    work["person_complete"] = work["individual_id"].map(person_complete)
    unit_complete = work.groupby(cluster)["person_complete"].all()
    all_unit_metadata = work.groupby(cluster, sort=True).agg(
        cohort_id=("cohort_id", "first"),
        disease_label=("disease_label", "first"),
        n_individuals=("individual_id", "nunique"),
        n_rows=("food_id", "size"),
    ).reset_index().rename(columns={cluster: "cluster_id"})
    all_unit_metadata["complete_case"] = all_unit_metadata["cluster_id"].map(
        unit_complete
    )
    analysis_work = work[work[cluster].map(unit_complete)].copy()

    def standardize(values: pd.Series) -> pd.Series:
        finite = values.dropna()
        result = pd.Series(np.nan, index=values.index, dtype=float)
        if finite.empty:
            return result
        scale = float(finite.std(ddof=0))
        result.loc[finite.index] = 0.0 if np.isclose(scale, 0) else (finite - finite.mean()) / scale
        return result

    analysis_work["food_standardized_value"] = analysis_work.groupby(
        ["cohort_id", "food_id"], group_keys=False
    )[value_column].apply(standardize)
    person = analysis_work.groupby("individual_id", sort=True).agg(
        cohort_id=("cohort_id", "first"),
        disease_label=("disease_label", "first"),
        label_source=("label_source", "first"),
        label_provenance=("label_provenance", "first"),
        cluster_id=(cluster, "first"),
        n_valid_foods=("food_standardized_value", "count"),
        value=("food_standardized_value", "mean"),
        n_rows=("food_id", "size"),
    ).reset_index()
    if not person.empty and not person["n_valid_foods"].eq(len(panel)).all():
        raise AssertionError("complete-case filtering failed to preserve a finite panel")
    cluster_meta = person.groupby("cluster_id").agg(
        n_cohorts=("cohort_id", "nunique"), n_labels=("disease_label", "nunique")
    )
    if (cluster_meta["n_cohorts"] > 1).any() or (cluster_meta["n_labels"] > 1).any():
        raise ValueError("independent components cannot span cohorts or labels")
    units = person.groupby("cluster_id", sort=True).agg(
        cohort_id=("cohort_id", "first"),
        disease_label=("disease_label", "first"),
        value=("value", "mean"),
        n_individuals=("individual_id", "nunique"),
        n_rows=("n_rows", "sum"),
    ).reset_index()
    return analysis_work, person, units, all_unit_metadata, cluster, panel


def _difference(units: pd.DataFrame, disease: str, reference: str) -> float:
    disease_values = units.loc[units["disease_label"].eq(disease), "value"].dropna()
    reference_values = units.loc[units["disease_label"].eq(reference), "value"].dropna()
    if disease_values.empty or reference_values.empty:
        return np.nan
    return float(disease_values.mean() - reference_values.mean())


def _bootstrap(units, disease, reference, config):
    disease_units = units[units["disease_label"].eq(disease) & units["value"].notna()]
    reference_units = units[units["disease_label"].eq(reference) & units["value"].notna()]
    rng = np.random.default_rng(config.seed)
    values = []
    for _ in range(config.bootstrap_replicates):
        left = disease_units.iloc[rng.integers(0, len(disease_units), len(disease_units))]
        right = reference_units.iloc[rng.integers(0, len(reference_units), len(reference_units))]
        values.append(float(left["value"].mean() - right["value"].mean()))
    alpha = (1 - config.confidence_level) / 2
    lower, upper = np.quantile(values, [alpha, 1 - alpha])
    return values, float(lower), float(upper)


def _label_permutation(units, disease, reference, cohort, config):
    rng = np.random.default_rng(_stable_seed(config.seed, cohort, disease, "label"))
    values = []
    for _ in range(config.shuffle_replicates):
        shuffled = units.copy()
        shuffled["disease_label"] = rng.permutation(shuffled["disease_label"].to_numpy())
        value = _difference(shuffled, disease, reference)
        if np.isfinite(value):
            values.append(value)
    alpha = (1 - config.confidence_level) / 2
    lower, upper = np.quantile(values, [alpha, 1 - alpha]) if values else (np.nan, np.nan)
    return {
        "cohort_id": cohort,
        "disease_label": disease,
        "reference_label": reference,
        "control": "shuffled_label",
        "analysis_status": "estimable" if values else "not_estimable",
        "null_mean_difference": float(np.mean(values)) if values else np.nan,
        "null_ci_lower": float(lower),
        "null_ci_upper": float(upper),
        "shuffle_replicates_requested": config.shuffle_replicates,
        "shuffle_replicates_valid": len(values),
        "n_analysis_units": int(len(units)),
        "seed": _stable_seed(config.seed, cohort, disease, "label"),
        "evidence_role": BIOLOGICAL_CONSISTENCY,
    }


def disease_cohort_consistency(
    individual_food: pd.DataFrame,
    provenance_manifest: Mapping[str, object],
    *,
    config: BiologicalConsistencyConfig | None = None,
    value_column: str = "GMNPS_delta",
    allow_test_data: bool = False,
) -> DiseaseConsistencyResult:
    """Testing/in-memory-only GMrepo analysis."""

    validate_evidence_provenance(provenance_manifest, allow_test_data=allow_test_data)
    return _disease_cohort_consistency_frame(
        individual_food,
        provenance_manifest,
        config=config,
        value_column=value_column,
        verified_input_hashes=None,
    )


def _disease_cohort_consistency_frame(
    individual_food: pd.DataFrame,
    provenance_manifest: Mapping[str, object],
    *,
    config: BiologicalConsistencyConfig | None,
    value_column: str,
    verified_input_hashes: Mapping[str, str] | None,
) -> DiseaseConsistencyResult:
    if provenance_manifest.get("source_kind") != "gmrepo_disease_labels":
        raise ValueError("disease analysis requires GMrepo provenance")
    work, person, units, all_units, cluster, panel = _prepare_gmrepo(
        individual_food, value_column
    )
    reference = _nonempty(provenance_manifest, "reference_label")
    resolved = config or BiologicalConsistencyConfig()
    estimates = []
    inference = []
    permutations = []
    for cohort, cohort_all_units in all_units.groupby("cohort_id", sort=True):
        labels = sorted(set(cohort_all_units["disease_label"]) - {reference})
        for disease in labels:
            cohort_people = person[person["cohort_id"].eq(cohort)]
            selected_people = cohort_people[cohort_people["disease_label"].isin([reference, disease])]
            selected_units = units[
                units["cohort_id"].eq(cohort)
                & units["disease_label"].isin([reference, disease])
            ]
            selected_all_units = cohort_all_units[
                cohort_all_units["disease_label"].isin([reference, disease])
            ]
            n_reference = int(
                selected_units.loc[
                    selected_units["disease_label"].eq(reference), "value"
                ].notna().sum()
            )
            n_disease = int(
                selected_units.loc[
                    selected_units["disease_label"].eq(disease), "value"
                ].notna().sum()
            )
            estimate = _difference(selected_units, disease, reference)
            original = work[
                work["cohort_id"].eq(cohort)
                & work["disease_label"].isin([reference, disease])
            ]
            base = {
                "cohort_id": cohort,
                "disease_label": disease,
                "reference_label": reference,
                "estimate": estimate,
                "estimand": "difference in mean food-standardized independent-unit summary",
                "summary_method": "within_cohort_food_zscore_then_person_mean",
                "n_reference_units": n_reference,
                "n_disease_units": n_disease,
                "n_independent_units": n_reference + n_disease,
                "n_analysis_units": n_reference + n_disease,
                "n_excluded_units": int((~selected_all_units["complete_case"]).sum()),
                "n_individuals": int(selected_people["individual_id"].nunique()),
                "n_rows": int(len(original)),
                "n_foods": len(panel),
                "label_source": "|".join(sorted(original["label_source"].unique())),
                "label_provenance": "|".join(sorted(original["label_provenance"].unique())),
                "evidence_role": BIOLOGICAL_CONSISTENCY,
                "analysis_boundary": "supporting_cohort_stratified_summary_only",
            }
            estimates.append(base)
            if n_reference < resolved.minimum_arm_units or n_disease < resolved.minimum_arm_units:
                inference.append(
                    {
                        **base,
                        "analysis_status": "not_estimable",
                        "reason": "fewer_than_minimum_units_in_one_or_both_arms",
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "ci_method": "independent_component_percentile",
                        "cluster_column": cluster,
                        "bootstrap_seed": resolved.seed,
                        "bootstrap_replicates_requested": resolved.bootstrap_replicates,
                        "bootstrap_replicates_valid": 0,
                        "minimum_arm_units": resolved.minimum_arm_units,
                    }
                )
                permutations.append(
                    {
                        "cohort_id": cohort,
                        "disease_label": disease,
                        "reference_label": reference,
                        "control": "shuffled_label",
                        "analysis_status": "not_estimable",
                        "null_mean_difference": np.nan,
                        "null_ci_lower": np.nan,
                        "null_ci_upper": np.nan,
                        "shuffle_replicates_requested": resolved.shuffle_replicates,
                        "shuffle_replicates_valid": 0,
                        "n_analysis_units": n_reference + n_disease,
                        "seed": _stable_seed(resolved.seed, cohort, disease, "label"),
                        "evidence_role": BIOLOGICAL_CONSISTENCY,
                    }
                )
            else:
                boot, lower, upper = _bootstrap(
                    selected_units, disease, reference, resolved
                )
                inference.append(
                    {
                        **base,
                        "analysis_status": "estimable",
                        "reason": "",
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "ci_method": "independent_component_percentile",
                        "cluster_column": cluster,
                        "bootstrap_seed": resolved.seed,
                        "bootstrap_replicates_requested": resolved.bootstrap_replicates,
                        "bootstrap_replicates_valid": len(boot),
                        "minimum_arm_units": resolved.minimum_arm_units,
                    }
                )
                permutations.append(
                    _label_permutation(
                        selected_units, disease, reference, cohort, resolved
                    )
                )
    if not estimates:
        raise ValueError("no within-cohort disease/reference contrast exists")
    microbiome = pd.DataFrame(
        [{
            "analysis_status": "not_estimable",
            "reason": "production_rerun_artifact_paths_not_provided",
            "control": "microbiome_profile_derangement_rerun",
            "evidence_role": BIOLOGICAL_CONSISTENCY,
        }]
    )
    return DiseaseConsistencyResult(
        evidence_role=BIOLOGICAL_CONSISTENCY,
        estimates=pd.DataFrame(estimates),
        inference=pd.DataFrame(inference),
        label_permutation=pd.DataFrame(permutations),
        microbiome_shuffle=microbiome,
        verified_input_hashes=(
            dict(verified_input_hashes) if verified_input_hashes is not None else None
        ),
    )


def disease_cohort_consistency_from_paths(
    table_path: Path,
    provenance_path: Path,
    *,
    config: BiologicalConsistencyConfig | None = None,
    value_column: str = "GMNPS_delta",
) -> DiseaseConsistencyResult:
    """Verify a trusted GMrepo table and analyze those exact bytes."""

    frame, manifest, hashes, _ = _load_verified_supporting_evidence(
        table_path, provenance_path, "gmrepo_disease_labels"
    )
    return _disease_cohort_consistency_frame(
        frame,
        manifest,
        config=config,
        value_column=value_column,
        verified_input_hashes=hashes,
    )


def evaluate_microbiome_shuffle_rerun(
    original_scores: pd.DataFrame,
    rerun_scores: pd.DataFrame,
    mapping_sidecar: pd.DataFrame,
    provenance_manifest: Mapping[str, object],
    *,
    allow_test_data: bool = False,
) -> dict[str, object]:
    """Testing/in-memory-only evaluator for a deterministic toy rerun."""

    validate_evidence_provenance(provenance_manifest, allow_test_data=allow_test_data)
    if provenance_manifest.get("source_kind") != "microbiome_shuffle_rerun":
        raise ValueError("shuffle evaluator requires microbiome_shuffle_rerun provenance")
    for field, observed in {
        "original_table_sha256": canonical_frame_sha256(original_scores),
        "rerun_table_sha256": canonical_frame_sha256(rerun_scores),
        "mapping_sidecar_sha256": canonical_frame_sha256(mapping_sidecar),
    }.items():
        if provenance_manifest.get(field) != observed:
            raise ValueError(f"{field} does not bind the supplied toy rerun")
    for field in (
        "original_scoring_manifest_sha256",
        "rerun_scoring_manifest_sha256",
        "locked_scoring_contract_sha256",
    ):
        _sha(provenance_manifest.get(field), field)
    return _evaluate_microbiome_shuffle_frames(
        original_scores, rerun_scores, mapping_sidecar, production=False
    )


def _evaluate_microbiome_shuffle_frames(
    original_scores: pd.DataFrame,
    rerun_scores: pd.DataFrame,
    mapping_sidecar: pd.DataFrame,
    *,
    production: bool,
) -> dict[str, object]:
    for frame, label in ((original_scores, "original"), (rerun_scores, "rerun")):
        _reject_outcomes(frame)
        _require_columns(
            frame,
            {"individual_id", "food_id", "FCS2", "GMNPS_score", "GMNPS_delta"},
            f"{label} score table",
        )
        if frame.duplicated(["individual_id", "food_id"]).any():
            raise ValueError(f"{label} score table contains duplicate person-food rows")
        for column in ("individual_id", "food_id"):
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"{label} score table {column} must be nonempty")
        numeric = frame[["FCS2", "GMNPS_score", "GMNPS_delta"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise ValueError(f"{label} score table values must be finite")
        if not np.allclose(
            numeric["GMNPS_delta"],
            numeric["GMNPS_score"] - numeric["FCS2"],
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                f"{label} GMNPS_delta must equal GMNPS_score - FCS2"
            )
        if "method_role" in frame and not frame["method_role"].eq("primary").all():
            raise ValueError(f"{label} score table method_role must be primary")
    original_people = set(original_scores["individual_id"].astype(str))
    rerun_people = set(rerun_scores["individual_id"].astype(str))
    if original_people != rerun_people:
        raise ValueError("original and rerun score tables must contain the same people")
    sidecar, component_map = _validate_person_mapping_sidecar(
        mapping_sidecar, original_people
    )
    keys = ["individual_id", "food_id"]
    if original_scores[keys].sort_values(keys).reset_index(drop=True).equals(
        rerun_scores[keys].sort_values(keys).reset_index(drop=True)
    ) is False:
        raise ValueError("original and rerun score tables must share the same panel")
    original_ordered = original_scores.sort_values(keys).reset_index(drop=True)
    rerun_ordered = rerun_scores.sort_values(keys).reset_index(drop=True)
    if not np.array_equal(
        original_ordered["FCS2"].to_numpy(float),
        rerun_ordered["FCS2"].to_numpy(float),
    ):
        raise ValueError("original and rerun score tables must share identical FCS2")
    panel = tuple(sorted(original_scores["food_id"].astype(str).unique()))
    panels = original_scores.assign(
        individual_id=original_scores["individual_id"].astype(str),
        food_id=original_scores["food_id"].astype(str),
    ).groupby("individual_id")["food_id"].apply(lambda values: tuple(sorted(values)))
    if not panels.map(lambda values: values == panel).all():
        raise ValueError("score-table people must share an identical complete food panel")
    left = original_ordered["GMNPS_delta"].to_numpy(float)
    right = rerun_ordered["GMNPS_delta"].to_numpy(float)
    return {
        "analysis_status": "verified" if production else "contract_verified_test_only",
        "mapping_status": "bijective_derangement",
        "n_independent_units": len(component_map),
        "n_individuals": int(sidecar["target_individual_id"].nunique()),
        "n_foods": int(original_scores["food_id"].nunique()),
        "mean_absolute_score_change": float(np.mean(np.abs(right - left))),
        "original_table_sha256": canonical_frame_sha256(original_scores),
        "rerun_table_sha256": canonical_frame_sha256(rerun_scores),
        "mapping_sidecar_sha256": canonical_frame_sha256(mapping_sidecar),
        "evidence_role": BIOLOGICAL_CONSISTENCY,
        "analysis_boundary": "profile_derangement_locked_rescoring_control_only",
    }


def toy_profile_attribute_rescore(
    profiles: pd.DataFrame,
    foods: pd.DataFrame,
    *,
    mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Deterministic test-only profile→beta→attribute rescoring fixture."""

    _require_columns(
        profiles,
        {"independent_unit_id", "mac_profile", "lipid_profile"},
        "toy profiles",
    )
    _require_columns(
        foods,
        {"food_id", "FCS2", "mac_attribute", "lipid_attribute"},
        "toy foods",
    )
    if profiles["independent_unit_id"].duplicated().any():
        raise ValueError("toy profiles must have unique independent_unit_id")
    if foods["food_id"].duplicated().any():
        raise ValueError("toy foods must have unique food_id")
    profile = profiles.copy()
    food = foods.copy()
    for column in ("mac_profile", "lipid_profile"):
        profile[column] = pd.to_numeric(profile[column], errors="coerce")
    for column in ("FCS2", "mac_attribute", "lipid_attribute"):
        food[column] = pd.to_numeric(food[column], errors="coerce")
    if not np.isfinite(profile[["mac_profile", "lipid_profile"]]).all().all():
        raise ValueError("toy profile values must be finite")
    if not np.isfinite(food[["FCS2", "mac_attribute", "lipid_attribute"]]).all().all():
        raise ValueError("toy food values must be finite")
    units = profile["independent_unit_id"].astype(str).tolist()
    if mapping is None:
        source_by_target = dict(zip(units, units))
    else:
        _require_columns(
            mapping,
            set(_PERSON_MAPPING_SIDECAR_COLUMNS),
            "toy mapping sidecar",
        )
        normalized, _ = _validate_person_mapping_sidecar(mapping, set(units))
        source_by_target = dict(
            zip(
                normalized["target_individual_id"],
                normalized["source_individual_id"],
            )
        )
    profile = profile.set_index(profile["independent_unit_id"].astype(str))
    rows: list[dict[str, object]] = []
    for target in units:
        source = source_by_target[target]
        source_row = profile.loc[source]
        # Explicit deterministic profile-to-beta transform, then attribute scoring.
        mac_beta = 2.0 * float(source_row["mac_profile"]) - float(source_row["lipid_profile"])
        lipid_beta = float(source_row["lipid_profile"]) + 0.5 * float(source_row["mac_profile"])
        for food_row in food.itertuples(index=False):
            raw_delta = mac_beta * float(food_row.mac_attribute) + lipid_beta * float(food_row.lipid_attribute)
            score = float(np.clip(float(food_row.FCS2) + raw_delta, 1.0, 100.0))
            rows.append(
                {
                    "individual_id": target,
                    "food_id": str(food_row.food_id),
                    "FCS2": float(food_row.FCS2),
                    "GMNPS_delta": score - float(food_row.FCS2),
                    "GMNPS_score": score,
                }
            )
    return pd.DataFrame(rows)


def evaluate_microbiome_shuffle_rerun_from_paths(
    paths: MicrobiomeShuffleArtifactPaths,
) -> dict[str, object]:
    """Live-verify both Phase 1 runs and evaluate the exact rerun bytes."""

    rerun, provenance, source_hashes, registry_entry = (
        _load_verified_supporting_evidence(
            paths.rerun_score_table,
            paths.provenance,
            "microbiome_shuffle_rerun",
        )
    )
    raw = {
        "original_table": _immutable_read(paths.original_score_table, "original score table"),
        "rerun_table": _immutable_read(paths.rerun_score_table, "rerun score table"),
        "mapping_sidecar": _immutable_read(
            paths.mapping_sidecar, "person/component mapping sidecar"
        ),
        "original_run_manifest": _immutable_read(paths.original_run_manifest, "original run manifest"),
        "rerun_run_manifest": _immutable_read(paths.rerun_run_manifest, "rerun run manifest"),
        "original_input_manifest": _immutable_read(paths.original_input_manifest, "original input manifest"),
        "rerun_input_manifest": _immutable_read(paths.rerun_input_manifest, "rerun input manifest"),
        "original_scoring_beta": _immutable_read(paths.original_scoring_beta, "original scoring beta"),
        "rerun_scoring_beta": _immutable_read(paths.rerun_scoring_beta, "rerun scoring beta"),
        "method_lock_manifest": _immutable_read(paths.method_lock_manifest, "method-lock manifest"),
        "method_lock_locator": _immutable_read(paths.method_lock_artifact_locator, "method-lock artifact locator"),
        "scoring_implementation": _immutable_read(_SCORING_IMPLEMENTATION, "scoring implementation"),
    }
    if sha256(raw["rerun_table"]).hexdigest() != source_hashes["table_sha256"]:
        raise ValueError("rerun score table changed after supporting-source verification")
    try:
        original = pd.read_csv(BytesIO(raw["original_table"]))
        mapping_sidecar = pd.read_csv(BytesIO(raw["mapping_sidecar"]))
    except Exception as error:
        raise ValueError("shuffle run score table or mapping sidecar is unreadable") from error

    original_phase1 = verify_phase1_run_artifacts(
        paths.original_score_table,
        paths.original_run_manifest,
        paths.original_success_marker,
        paths.original_input_manifest,
    )
    rerun_phase1 = verify_phase1_run_artifacts(
        paths.rerun_score_table,
        paths.rerun_run_manifest,
        paths.rerun_success_marker,
        paths.rerun_input_manifest,
    )
    if load_method_lock_artifact_locator(paths.method_lock_artifact_locator) != paths.method_lock_artifacts:
        raise ValueError("method-lock artifact locator does not match supplied paths")
    method_lock = _json_bytes_object(raw["method_lock_manifest"], "method-lock manifest")
    release_registry_hash = sha256(
        _immutable_read(paths.method_lock_artifacts.release_registry, "trusted release registry")
    ).hexdigest()
    validate_method_lock_manifest(
        method_lock,
        paths.method_lock_artifacts,
        expected_release_registry_sha256=release_registry_hash,
    )
    if raw["original_input_manifest"] != _immutable_read(
        paths.method_lock_artifacts.input_manifest, "locked input manifest"
    ):
        raise ValueError("original run input manifest is not the method-locked manifest")
    if raw["original_scoring_beta"] != _immutable_read(
        paths.method_lock_artifacts.scoring_beta, "locked scoring beta"
    ):
        raise ValueError("original scoring beta is not the method-locked scoring beta")

    original_run = _json_bytes_object(raw["original_run_manifest"], "original run manifest")
    rerun_run = _json_bytes_object(raw["rerun_run_manifest"], "rerun run manifest")
    for run, label in ((original_run, "original"), (rerun_run, "rerun")):
        for field, expected in {
            "method": "attribute_recomposition",
            "method_role": "primary",
            "score_centering": "none",
            "production_label": "production",
            "development_smoke_test": False,
        }.items():
            if run.get(field) != expected:
                raise ValueError(f"{label} run {field} is not production locked")
    if _scoring_contract(original_run) != _scoring_contract(rerun_run):
        raise ValueError("original and rerun do not share the locked scoring contract")

    original_input = _json_bytes_object(raw["original_input_manifest"], "original input manifest")
    rerun_input = _json_bytes_object(raw["rerun_input_manifest"], "rerun input manifest")
    original_files = original_input.get("files")
    rerun_files = rerun_input.get("files")
    if not isinstance(original_files, dict) or not isinstance(rerun_files, dict):
        raise ValueError("both input manifests must contain files objects")
    required_phase1_inputs = {
        "development_beta": paths.method_lock_artifacts.development_beta,
        "score_beta": paths.method_lock_artifacts.scoring_beta,
        "food_metadata": paths.method_lock_artifacts.food_metadata,
        "baseline_attribute_points": paths.method_lock_artifacts.baseline_attribute_points,
        "food_exposures": paths.method_lock_artifacts.food_exposures,
        "effective_attribute_weights": paths.method_lock_artifacts.effective_attribute_weights,
    }
    if (
        set(original_files) != set(rerun_files)
        or not set(required_phase1_inputs).issubset(original_files)
    ):
        raise ValueError("original and rerun input artifact sets must be identical")
    for name, artifact_path in required_phase1_inputs.items():
        if _manifest_file_digest(original_input, name) != sha256(
            _immutable_read(artifact_path, f"method-locked {name}")
        ).hexdigest():
            raise ValueError(f"original input manifest does not bind method-locked {name}")
    for name in original_files:
        if name != "score_beta" and _manifest_file_digest(original_input, name) != _manifest_file_digest(rerun_input, name):
            raise ValueError("original and rerun must use identical production food/method artifacts")
    sidecar_hash = sha256(raw["mapping_sidecar"]).hexdigest()
    if "microbiome_shuffle_sidecar_sha256" in original_input:
        raise ValueError("original input manifest must not declare a shuffle sidecar")
    if rerun_input.get("microbiome_shuffle_sidecar_sha256") != sidecar_hash:
        raise ValueError("rerun input manifest does not bind the mapping sidecar")
    original_nonfiles = {
        key: value for key, value in original_input.items() if key != "files"
    }
    rerun_nonfiles = {
        key: value
        for key, value in rerun_input.items()
        if key not in {"files", "microbiome_shuffle_sidecar_sha256"}
    }
    if original_nonfiles != rerun_nonfiles:
        raise ValueError("original and rerun input method contracts differ")
    if _manifest_file_digest(original_input, "score_beta") != sha256(raw["original_scoring_beta"]).hexdigest():
        raise ValueError("original input manifest does not bind original scoring beta")
    if _manifest_file_digest(rerun_input, "score_beta") != sha256(raw["rerun_scoring_beta"]).hexdigest():
        raise ValueError("rerun input manifest does not bind rerun scoring beta")

    observed = {
        "original_table_sha256": sha256(raw["original_table"]).hexdigest(),
        "rerun_table_sha256": sha256(raw["rerun_table"]).hexdigest(),
        "mapping_sidecar_sha256": sidecar_hash,
        "original_scoring_manifest_sha256": sha256(raw["original_run_manifest"]).hexdigest(),
        "rerun_scoring_manifest_sha256": sha256(raw["rerun_run_manifest"]).hexdigest(),
        "locked_scoring_contract_sha256": _canonical_mapping_sha256(_scoring_contract(original_run)),
        "original_input_manifest_sha256": sha256(raw["original_input_manifest"]).hexdigest(),
        "rerun_input_manifest_sha256": sha256(raw["rerun_input_manifest"]).hexdigest(),
        "original_scoring_beta_sha256": sha256(raw["original_scoring_beta"]).hexdigest(),
        "rerun_scoring_beta_sha256": sha256(raw["rerun_scoring_beta"]).hexdigest(),
        "method_lock_manifest_sha256": sha256(raw["method_lock_manifest"]).hexdigest(),
        "method_lock_artifact_locator_sha256": sha256(raw["method_lock_locator"]).hexdigest(),
        "phase1_release_registry_sha256": release_registry_hash,
        "scoring_implementation_sha256": sha256(raw["scoring_implementation"]).hexdigest(),
        "original_success_marker_sha256": original_phase1["phase1_success_marker_sha256"],
        "rerun_success_marker_sha256": rerun_phase1["phase1_success_marker_sha256"],
    }
    for field, digest in observed.items():
        if provenance.get(field) != digest:
            raise ValueError(f"production provenance {field} does not match live artifacts")
        if registry_entry.get(field) != digest:
            raise ValueError(f"trusted supporting registry {field} does not match live artifacts")
    # Analysis starts only after every live byte has matched provenance and the
    # repository-trusted registry entry.
    result = _evaluate_microbiome_shuffle_frames(
        original, rerun, mapping_sidecar, production=True
    )
    _verify_deranged_beta(
        original,
        mapping_sidecar,
        raw["original_scoring_beta"],
        raw["rerun_scoring_beta"],
    )
    result["verified_input_hashes"] = {**source_hashes, **observed}
    return result


def _spearman(left: pd.Series, right: pd.Series) -> float:
    return float(left.rank(method="average").corr(right.rank(method="average")))


def zoe_aggregate_rank_consistency(
    aggregate_ranks: pd.DataFrame,
    provenance_manifest: Mapping[str, object],
    *,
    allow_test_data: bool = False,
) -> dict[str, object]:
    """Testing/in-memory-only aggregate-rank summary."""

    validate_evidence_provenance(provenance_manifest, allow_test_data=allow_test_data)
    return _zoe_aggregate_rank_frame(aggregate_ranks, provenance_manifest)


def _zoe_aggregate_rank_frame(
    aggregate_ranks: pd.DataFrame,
    provenance_manifest: Mapping[str, object],
    verified_input_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    if provenance_manifest.get("source_kind") != "zoe_aggregate_ranks":
        raise ValueError("ZOE analysis requires zoe_aggregate_ranks provenance")
    _require_columns(
        aggregate_ranks,
        {"food_id", "gmnps_rank", "zoe_rank", "rank_source"},
        "ZOE aggregate ranks",
    )
    _reject_outcomes(aggregate_ranks)
    if aggregate_ranks["food_id"].duplicated().any():
        raise ValueError("ZOE aggregate ranks must have unique food_id")
    work = aggregate_ranks.copy()
    work["gmnps_rank"] = pd.to_numeric(work["gmnps_rank"], errors="coerce")
    work["zoe_rank"] = pd.to_numeric(work["zoe_rank"], errors="coerce")
    valid = work[np.isfinite(work["gmnps_rank"]) & np.isfinite(work["zoe_rank"])]
    base = {
        "n_total": int(len(work)),
        "n_valid": int(len(valid)),
        "n_excluded": int(len(work) - len(valid)),
        "rank_sources": sorted(work["rank_source"].astype(str).unique()),
        "evidence_role": BIOLOGICAL_CONSISTENCY,
        "analysis_boundary": "aggregate_rank_summary_only",
        "verified_input_hashes": (
            dict(verified_input_hashes) if verified_input_hashes is not None else None
        ),
    }
    if len(valid) < 3:
        return {
            **base,
            "analysis_status": "not_estimable",
            "reason": "fewer_than_three_valid_rank_pairs",
            "spearman_rank_correlation": None,
        }
    if valid["gmnps_rank"].nunique() < 2 or valid["zoe_rank"].nunique() < 2:
        return {
            **base,
            "analysis_status": "not_estimable",
            "reason": "constant_rank_vector",
            "spearman_rank_correlation": None,
        }
    return {
        **base,
        "analysis_status": "estimable",
        "reason": "",
        "spearman_rank_correlation": _spearman(
            valid["gmnps_rank"], valid["zoe_rank"]
        ),
    }


def zoe_aggregate_rank_consistency_from_paths(
    table_path: Path, provenance_path: Path
) -> dict[str, object]:
    """Verify trusted ZOE aggregate ranks and analyze those exact bytes."""

    frame, manifest, hashes, _ = _load_verified_supporting_evidence(
        table_path, provenance_path, "zoe_aggregate_ranks"
    )
    return _zoe_aggregate_rank_frame(frame, manifest, hashes)


def knowledge_path_consistency(
    paths: pd.DataFrame,
    provenance_manifest: Mapping[str, object],
    *,
    allow_test_data: bool = False,
) -> dict[str, object]:
    """Testing/in-memory-only adjudicated-path summary."""

    validate_evidence_provenance(provenance_manifest, allow_test_data=allow_test_data)
    return _knowledge_path_frame(paths, provenance_manifest)


def _knowledge_path_frame(
    paths: pd.DataFrame,
    provenance_manifest: Mapping[str, object],
    verified_input_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    if provenance_manifest.get("source_kind") != "knowledge_paths":
        raise ValueError("knowledge summary requires knowledge_paths provenance")
    required = {
        "path_id", "source_node", "target_node", "direction",
        "is_direction_consistent", "path_provenance",
    }
    _require_columns(paths, required, "knowledge paths")
    _reject_outcomes(paths)
    if paths["path_id"].duplicated().any():
        raise ValueError("path_id must be unique")
    for column in ("path_id", "source_node", "target_node", "path_provenance"):
        if paths[column].isna().any() or paths[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must contain nonempty values")
    if not paths["direction"].isin(["positive", "negative"]).all():
        raise ValueError("direction must be positive or negative")
    consistent = paths["is_direction_consistent"]
    if not consistent.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("is_direction_consistent must be boolean")
    return {
        "n_paths": int(len(paths)),
        "n_direction_consistent": int(consistent.sum()),
        "direction_consistent_fraction": float(consistent.mean()),
        "n_path_provenance_sources": int(paths["path_provenance"].nunique()),
        "evidence_role": MECHANISTIC_CONSISTENCY,
        "analysis_boundary": "adjudicated_path_summary_only",
        "verified_input_hashes": (
            dict(verified_input_hashes) if verified_input_hashes is not None else None
        ),
    }


def knowledge_path_consistency_from_paths(
    table_path: Path, provenance_path: Path
) -> dict[str, object]:
    """Verify trusted knowledge paths and summarize those exact bytes."""

    frame, manifest, hashes, _ = _load_verified_supporting_evidence(
        table_path, provenance_path, "knowledge_paths"
    )
    return _knowledge_path_frame(frame, manifest, hashes)


__all__ = [
    "BIOLOGICAL_CONSISTENCY",
    "MECHANISTIC_CONSISTENCY",
    "BiologicalConsistencyConfig",
    "DiseaseConsistencyResult",
    "MicrobiomeShuffleArtifactPaths",
    "canonical_frame_sha256",
    "disease_cohort_consistency",
    "disease_cohort_consistency_from_paths",
    "evaluate_microbiome_shuffle_rerun",
    "evaluate_microbiome_shuffle_rerun_from_paths",
    "knowledge_path_consistency",
    "knowledge_path_consistency_from_paths",
    "toy_profile_attribute_rescore",
    "validate_evidence_provenance",
    "verify_supporting_evidence_file",
    "zoe_aggregate_rank_consistency",
    "zoe_aggregate_rank_consistency_from_paths",
]
