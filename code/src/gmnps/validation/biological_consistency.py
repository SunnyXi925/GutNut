"""Supporting biological and adjudicated-path consistency analyses."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Mapping

import numpy as np
import pandas as pd


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
_VERIFICATION_TOKEN = object()


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
class VerifiedSupportingEvidence:
    source_kind: str
    source_id: str
    accession: str
    table_sha256: str
    provenance_sha256: str
    registry_sha256: str
    _token: object


@dataclass(frozen=True)
class VerifiedMicrobiomeShuffleRerun:
    """Live binding of profiles, derangement, scoring manifests, and rerun."""

    source: VerifiedSupportingEvidence
    original_table_sha256: str
    rerun_table_sha256: str
    mapping_sha256: str
    original_scoring_manifest_sha256: str
    rerun_scoring_manifest_sha256: str
    locked_scoring_contract_sha256: str
    _token: object


@dataclass(frozen=True)
class DiseaseConsistencyResult:
    evidence_role: str
    estimates: pd.DataFrame
    inference: pd.DataFrame
    label_permutation: pd.DataFrame
    microbiome_shuffle: pd.DataFrame


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
    verified_source: VerifiedSupportingEvidence | None = None,
) -> None:
    production = _base_evidence_provenance(manifest)
    if production:
        if (
            not isinstance(verified_source, VerifiedSupportingEvidence)
            or verified_source._token is not _VERIFICATION_TOKEN
            or verified_source.source_kind != manifest.get("source_kind")
            or verified_source.source_id != manifest.get("source_id")
            or verified_source.accession != manifest.get("accession")
            or verified_source.table_sha256 != manifest.get("table_sha256")
        ):
            raise ValueError("production requires repository-trusted source verification")
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


def verify_supporting_evidence_file(
    table_path: Path, manifest: Mapping[str, object]
) -> VerifiedSupportingEvidence:
    """Resolve a source only through the repository-installed digest registry."""

    if not _base_evidence_provenance(manifest):
        raise ValueError("live source verification is only for production evidence")
    table_bytes = _immutable_read(Path(table_path), "supporting evidence table")
    observed = sha256(table_bytes).hexdigest()
    if observed != manifest.get("table_sha256"):
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
        and entry.get("table_sha256") == observed
        and entry.get("provenance_sha256") == provenance_hash
    ]
    if len(matches) != 1:
        raise ValueError("supporting source has no unique repository-trusted digest entry")
    return VerifiedSupportingEvidence(
        source_kind=str(manifest["source_kind"]),
        source_id=str(manifest["source_id"]),
        accession=str(manifest["accession"]),
        table_sha256=observed,
        provenance_sha256=provenance_hash,
        registry_sha256=sha256(registry_bytes).hexdigest(),
        _token=_VERIFICATION_TOKEN,
    )


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
        "source_hashes",
    )
    missing = [field for field in fields if field not in manifest]
    if missing:
        raise ValueError(f"scoring manifest lacks locked contract fields: {missing}")
    return {field: manifest[field] for field in fields}


def verify_microbiome_shuffle_rerun_files(
    original_table_path: Path,
    rerun_table_path: Path,
    mapping_path: Path,
    original_scoring_manifest_path: Path,
    rerun_scoring_manifest_path: Path,
    provenance_manifest: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, VerifiedMicrobiomeShuffleRerun]:
    """Verify a production derangement and both locked scoring-run manifests."""

    source = verify_supporting_evidence_file(rerun_table_path, provenance_manifest)
    if provenance_manifest.get("source_kind") != "microbiome_shuffle_rerun":
        raise ValueError("rerun verifier requires microbiome_shuffle_rerun provenance")
    blobs = {
        "original": _immutable_read(Path(original_table_path), "original score table"),
        "rerun": _immutable_read(Path(rerun_table_path), "rerun score table"),
        "mapping": _immutable_read(Path(mapping_path), "shuffle mapping"),
        "original_manifest": _immutable_read(
            Path(original_scoring_manifest_path), "original scoring manifest"
        ),
        "rerun_manifest": _immutable_read(
            Path(rerun_scoring_manifest_path), "rerun scoring manifest"
        ),
    }
    from io import BytesIO

    try:
        original = pd.read_csv(BytesIO(blobs["original"]))
        rerun = pd.read_csv(BytesIO(blobs["rerun"]))
        mapping = pd.read_csv(BytesIO(blobs["mapping"]))
        original_manifest = json.loads(blobs["original_manifest"])
        rerun_manifest = json.loads(blobs["rerun_manifest"])
    except (UnicodeError, json.JSONDecodeError, pd.errors.ParserError) as error:
        raise ValueError("shuffle rerun artifacts are unreadable") from error
    if not isinstance(original_manifest, dict) or not isinstance(rerun_manifest, dict):
        raise ValueError("scoring manifests must be JSON objects")
    original_contract = _scoring_contract(original_manifest)
    rerun_contract = _scoring_contract(rerun_manifest)
    if original_contract != rerun_contract:
        raise ValueError("original and rerun do not share the locked scoring contract")
    observed = {
        "original_table_sha256": canonical_frame_sha256(original),
        "rerun_table_sha256": canonical_frame_sha256(rerun),
        "mapping_sha256": canonical_frame_sha256(mapping),
        "original_scoring_manifest_sha256": sha256(
            blobs["original_manifest"]
        ).hexdigest(),
        "rerun_scoring_manifest_sha256": sha256(blobs["rerun_manifest"]).hexdigest(),
        "locked_scoring_contract_sha256": _canonical_mapping_sha256(
            original_contract
        ),
    }
    for field, value in observed.items():
        if provenance_manifest.get(field) != value:
            raise ValueError(f"{field} does not match the live shuffle artifact")
    verification = VerifiedMicrobiomeShuffleRerun(
        source=source,
        **observed,
        _token=_VERIFICATION_TOKEN,
    )
    return original, rerun, mapping, verification


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

    def standardize(values: pd.Series) -> pd.Series:
        finite = values.dropna()
        result = pd.Series(np.nan, index=values.index, dtype=float)
        if finite.empty:
            return result
        scale = float(finite.std(ddof=0))
        result.loc[finite.index] = 0.0 if np.isclose(scale, 0) else (finite - finite.mean()) / scale
        return result

    work["food_standardized_value"] = work.groupby(
        ["cohort_id", "food_id"], group_keys=False
    )[value_column].apply(standardize)
    person = work.groupby("individual_id", sort=True).agg(
        cohort_id=("cohort_id", "first"),
        disease_label=("disease_label", "first"),
        label_source=("label_source", "first"),
        label_provenance=("label_provenance", "first"),
        cluster_id=(cluster, "first"),
        n_valid_foods=("food_standardized_value", "count"),
        value=("food_standardized_value", "mean"),
        n_rows=("food_id", "size"),
    ).reset_index()
    person.loc[person["n_valid_foods"] != len(panel), "value"] = np.nan
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
    return work, person, units, cluster, panel


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
    verified_source: VerifiedSupportingEvidence | None = None,
) -> DiseaseConsistencyResult:
    validate_evidence_provenance(
        provenance_manifest,
        allow_test_data=allow_test_data,
        verified_source=verified_source,
    )
    if provenance_manifest.get("source_kind") != "gmrepo_disease_labels":
        raise ValueError("disease analysis requires GMrepo provenance")
    work, person, units, cluster, panel = _prepare_gmrepo(individual_food, value_column)
    reference = _nonempty(provenance_manifest, "reference_label")
    resolved = config or BiologicalConsistencyConfig()
    estimates = []
    inference = []
    permutations = []
    for cohort, cohort_people in person.groupby("cohort_id", sort=True):
        labels = sorted(set(cohort_people["disease_label"]) - {reference})
        for disease in labels:
            selected_people = cohort_people[cohort_people["disease_label"].isin([reference, disease])]
            selected_units = units[
                units["cohort_id"].eq(cohort)
                & units["disease_label"].isin([reference, disease])
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
            "reason": "verified_rerun_not_provided",
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
    )


def evaluate_microbiome_shuffle_rerun(
    original_scores: pd.DataFrame,
    rerun_scores: pd.DataFrame,
    mapping: pd.DataFrame,
    provenance_manifest: Mapping[str, object],
    *,
    allow_test_data: bool = False,
    verified_rerun: VerifiedMicrobiomeShuffleRerun | None = None,
) -> dict[str, object]:
    """Validate a profile derangement followed by a locked full GMNPS rerun."""

    production = _base_evidence_provenance(provenance_manifest)
    if production and (
        not isinstance(verified_rerun, VerifiedMicrobiomeShuffleRerun)
        or verified_rerun._token is not _VERIFICATION_TOKEN
    ):
        return {
            "analysis_status": "not_estimable",
            "reason": "repository_trusted_shuffle_rerun_unavailable",
            "evidence_role": BIOLOGICAL_CONSISTENCY,
        }
    validate_evidence_provenance(
        provenance_manifest,
        allow_test_data=allow_test_data,
        verified_source=(verified_rerun.source if verified_rerun else None),
    )
    if provenance_manifest.get("source_kind") != "microbiome_shuffle_rerun":
        raise ValueError("shuffle evaluator requires microbiome_shuffle_rerun provenance")
    _require_columns(
        mapping,
        {"independent_unit_id", "shuffled_profile_unit_id"},
        "shuffle mapping",
    )
    if (
        mapping["independent_unit_id"].duplicated().any()
        or mapping["shuffled_profile_unit_id"].duplicated().any()
        or set(mapping["independent_unit_id"]) != set(mapping["shuffled_profile_unit_id"])
    ):
        raise ValueError("microbiome shuffle mapping must be bijective")
    if mapping["independent_unit_id"].eq(mapping["shuffled_profile_unit_id"]).any():
        raise ValueError("microbiome shuffle mapping must be a derangement")
    for frame, label in ((original_scores, "original"), (rerun_scores, "rerun")):
        _require_columns(
            frame,
            {"individual_id", "component_id", "food_id", "GMNPS_delta"},
            f"{label} score table",
        )
        if frame.duplicated(["individual_id", "food_id"]).any():
            raise ValueError(f"{label} score table contains duplicate person-food rows")
    if "profile_source_unit_id" not in rerun_scores:
        raise ValueError("rerun scores must record profile_source_unit_id")
    keys = ["individual_id", "component_id", "food_id"]
    if original_scores[keys].sort_values(keys).reset_index(drop=True).equals(
        rerun_scores[keys].sort_values(keys).reset_index(drop=True)
    ) is False:
        raise ValueError("original and rerun score tables must share the same panel")
    map_dict = dict(
        zip(mapping["independent_unit_id"], mapping["shuffled_profile_unit_id"])
    )
    expected_source = rerun_scores["component_id"].map(map_dict)
    if expected_source.isna().any() or not expected_source.eq(
        rerun_scores["profile_source_unit_id"]
    ).all():
        raise ValueError("rerun profile_source_unit_id does not match the shuffle mapping")
    hash_fields = {
        "original_table_sha256": canonical_frame_sha256(original_scores),
        "rerun_table_sha256": canonical_frame_sha256(rerun_scores),
        "mapping_sha256": canonical_frame_sha256(mapping),
    }
    for field, observed in hash_fields.items():
        if provenance_manifest.get(field) != observed:
            raise ValueError(f"{field} does not bind the supplied rerun contract")
        if production and getattr(verified_rerun, field) != observed:
            raise ValueError(f"verified production {field} does not match supplied data")
    for field in (
        "original_scoring_manifest_sha256",
        "rerun_scoring_manifest_sha256",
        "locked_scoring_contract_sha256",
    ):
        _sha(provenance_manifest.get(field), field)
        if production and getattr(verified_rerun, field) != provenance_manifest.get(field):
            raise ValueError(f"verified production {field} does not match provenance")
    left = original_scores.sort_values(keys)["GMNPS_delta"].to_numpy(float)
    right = rerun_scores.sort_values(keys)["GMNPS_delta"].to_numpy(float)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("original and rerun GMNPS values must be finite")
    return {
        "analysis_status": "verified" if production else "contract_verified_test_only",
        "mapping_status": "bijective_derangement",
        "n_independent_units": int(mapping["independent_unit_id"].nunique()),
        "n_foods": int(original_scores["food_id"].nunique()),
        "mean_absolute_score_change": float(np.mean(np.abs(right - left))),
        "original_table_sha256": hash_fields["original_table_sha256"],
        "rerun_table_sha256": hash_fields["rerun_table_sha256"],
        "mapping_sha256": hash_fields["mapping_sha256"],
        "evidence_role": BIOLOGICAL_CONSISTENCY,
        "analysis_boundary": "profile_derangement_locked_rescoring_control_only",
    }


def _spearman(left: pd.Series, right: pd.Series) -> float:
    return float(left.rank(method="average").corr(right.rank(method="average")))


def zoe_aggregate_rank_consistency(
    aggregate_ranks: pd.DataFrame,
    provenance_manifest: Mapping[str, object],
    *,
    allow_test_data: bool = False,
    verified_source: VerifiedSupportingEvidence | None = None,
) -> dict[str, object]:
    validate_evidence_provenance(
        provenance_manifest,
        allow_test_data=allow_test_data,
        verified_source=verified_source,
    )
    if provenance_manifest.get("source_kind") != "zoe_aggregate_ranks":
        raise ValueError("ZOE analysis requires zoe_aggregate_ranks provenance")
    _require_columns(
        aggregate_ranks,
        {"food_id", "gmnps_rank", "zoe_rank", "rank_source"},
        "ZOE aggregate ranks",
    )
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


def knowledge_path_consistency(
    paths: pd.DataFrame,
    provenance_manifest: Mapping[str, object],
    *,
    allow_test_data: bool = False,
    verified_source: VerifiedSupportingEvidence | None = None,
) -> dict[str, object]:
    validate_evidence_provenance(
        provenance_manifest,
        allow_test_data=allow_test_data,
        verified_source=verified_source,
    )
    if provenance_manifest.get("source_kind") != "knowledge_paths":
        raise ValueError("knowledge summary requires knowledge_paths provenance")
    required = {
        "path_id", "source_node", "target_node", "direction",
        "is_direction_consistent", "path_provenance",
    }
    _require_columns(paths, required, "knowledge paths")
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
    }


__all__ = [
    "BIOLOGICAL_CONSISTENCY",
    "MECHANISTIC_CONSISTENCY",
    "BiologicalConsistencyConfig",
    "DiseaseConsistencyResult",
    "VerifiedSupportingEvidence",
    "VerifiedMicrobiomeShuffleRerun",
    "canonical_frame_sha256",
    "disease_cohort_consistency",
    "evaluate_microbiome_shuffle_rerun",
    "knowledge_path_consistency",
    "validate_evidence_provenance",
    "verify_supporting_evidence_file",
    "verify_microbiome_shuffle_rerun_files",
    "zoe_aggregate_rank_consistency",
]
