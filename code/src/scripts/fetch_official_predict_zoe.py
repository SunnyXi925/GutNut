#!/usr/bin/env python3
"""Fetch checksum-bound official predictor resources without outcome access.

The allowlist contains only microbiome profiles, sequencing metadata, public
record metadata, and historical USDA food-composition releases. It never
discovers or downloads clinical response, label, or supplementary ranking
tables. Large raw FASTQ files are deliberately outside this script.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import md5, sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ACCESSED_ON = "2026-08-13"
USER_AGENT = "GMNPS-predictor-reconstruction/1.0 (+official-source-audit)"
DEFAULT_CACHE = Path(
    "data/project_data/predict_multi/L2_phenotype/predictor_reconstruction/cache"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_TOKENS = frozenset(
    {
        "outcome",
        "response",
        "label",
        "glucose",
        "triglyceride",
        "c-peptide",
        "c_peptide",
        "ranking",
        "supplementary",
    }
)


@dataclass(frozen=True)
class OfficialArtifact:
    artifact_id: str
    group: str
    source: str
    stable_url: str
    filename: str
    content_scope: str
    expected_sha256: str | None = None
    expected_size: int | None = None

    def __post_init__(self) -> None:
        if self.group not in {"predictor", "fndds"}:
            raise ValueError("artifact group must be predictor or fndds")
        if not self.stable_url.startswith("https://"):
            raise ValueError("official artifacts require HTTPS stable URLs")
        if Path(self.filename).name != self.filename:
            raise ValueError("artifact filename must not contain directories")
        searchable = f"{self.artifact_id} {self.filename} {self.content_scope}".lower()
        present = sorted(token for token in _FORBIDDEN_TOKENS if token in searchable)
        if present:
            raise ValueError(f"outcome/label-bearing artifact is forbidden: {present}")
        if self.expected_sha256 is not None and _SHA256.fullmatch(
            self.expected_sha256
        ) is None:
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
        if self.expected_size is not None and self.expected_size <= 0:
            raise ValueError("expected_size must be positive")


def _ena_metadata(accession: str) -> OfficialArtifact:
    fields = (
        "study_accession,sample_accession,run_accession,instrument_platform,"
        "library_strategy,library_source,library_layout"
    )
    url = (
        "https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&"
        f"query=study_accession%3D%22{accession}%22&fields={fields}&"
        "format=tsv&limit=0"
    )
    return OfficialArtifact(
        artifact_id=f"ena_{accession.lower()}_sequencing_metadata",
        group="predictor",
        source="European Nucleotide Archive",
        stable_url=url,
        filename=f"{accession}_read_run_metadata.tsv",
        content_scope="sequencing accession and library metadata only",
    )


PREDICTOR_ARTIFACTS = (
    OfficialArtifact(
        artifact_id="experimenthub_eh5458_relative_abundance",
        group="predictor",
        source="Bioconductor ExperimentHub EH5458",
        stable_url="https://experimenthub.bioconductor.org/fetch/5501",
        filename="2021-03-31.AsnicarF_2021.relative_abundance.rda",
        content_scope="microbiome relative-abundance profiles only",
        expected_sha256="89c635061b357583351ca33f520a72d0efced4a2963e73182e529228c8395c54",
        expected_size=755545,
    ),
    OfficialArtifact(
        artifact_id="zenodo_15308000_public_predictor_record_metadata",
        group="predictor",
        source="Zenodo record 15308000",
        stable_url="https://zenodo.org/api/records/15308000",
        filename="zenodo_15308000_record_metadata.json",
        content_scope="public record and file metadata only",
    ),
    *tuple(
        _ena_metadata(accession)
        for accession in (
            "PRJEB39223",
            "PRJEB75460",
            "PRJEB75462",
            "PRJEB75463",
            "PRJEB75464",
        )
    ),
)


_ARS_APPS = "https://www.ars.usda.gov/ARSUserFiles/80400530/apps"


def _ars_artifact(
    cycle: str,
    filename: str,
    local_filename: str | None = None,
) -> OfficialArtifact:
    encoded = filename.replace(" ", "%20")
    return OfficialArtifact(
        artifact_id=f"fndds_{cycle.replace('-', '_')}_{Path(filename).stem.lower()}",
        group="fndds",
        source="USDA ARS Food Surveys Research Group",
        stable_url=f"{_ARS_APPS}/{encoded}",
        filename=local_filename or filename.replace(" ", "_"),
        content_scope=f"FNDDS {cycle} food-composition release artifact",
    )


FNDDS_ARTIFACTS = (
    _ars_artifact("2001-2002", "FNDDS1_ASCII.EXE"),
    _ars_artifact("2003-2004", "FNDDS2_ASCII.EXE"),
    _ars_artifact("2005-2006", "FNDDS3_ASCII.EXE"),
    _ars_artifact("2007-2008", "FNDDS4_ASCII.EXE"),
    _ars_artifact("2009-2010", "FNDDS5_ASCII.EXE"),
    _ars_artifact("2011-2012", "FNDDS_2011-2012_ASCII.EXE"),
    _ars_artifact("2013-2014", "FNDDS_2013-2014_ACCESS.EXE"),
    *tuple(
        _ars_artifact(cycle, f"{cycle} FNDDS At A Glance - {table}.xlsx")
        for cycle in ("2015-2016", "2017-2018")
        for table in (
            "Foods and Beverages",
            "Portions and Weights",
            "FNDDS Ingredients",
            "Ingredient Nutrient Values",
            "FNDDS Nutrient Values",
        )
    ),
)


def _digest_file(path: Path) -> tuple[str, str, int]:
    sha = sha256()
    legacy = md5(usedforsecurity=False)
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
            legacy.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), legacy.hexdigest(), size


def _load_previous_manifest(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("existing acquisition manifest is unreadable") from error
    records = payload.get("artifacts") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError("existing acquisition manifest schema is invalid")
    return {
        str(record["artifact_id"]): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("artifact_id"), str)
    }


def _known_digest(
    artifact: OfficialArtifact,
    previous: dict[str, dict[str, object]],
) -> str | None:
    if artifact.expected_sha256 is not None:
        return artifact.expected_sha256
    prior = previous.get(artifact.artifact_id, {})
    if prior.get("stable_url") != artifact.stable_url:
        return None
    value = prior.get("sha256")
    return value if isinstance(value, str) and _SHA256.fullmatch(value) else None


def _cached_record(
    artifact: OfficialArtifact,
    destination: Path,
    expected_sha256: str,
) -> dict[str, object] | None:
    if not destination.is_file():
        return None
    digest, md5_digest, size = _digest_file(destination)
    if digest != expected_sha256:
        return None
    if artifact.expected_size is not None and size != artifact.expected_size:
        return None
    return {
        **asdict(artifact),
        "status": "cached_verified",
        "sha256": digest,
        "md5": md5_digest,
        "size_bytes": size,
        "cache_path": str(destination),
        "final_url": artifact.stable_url,
        "etag": None,
        "last_modified": None,
        "upstream_checksum_basis": (
            "fixed_expected_sha256"
            if artifact.expected_sha256 is not None
            else "prior_acquisition_manifest_sha256"
        ),
    }


def fetch_artifact(
    artifact: OfficialArtifact,
    *,
    cache_dir: Path,
    previous: dict[str, dict[str, object]],
    timeout_seconds: int,
) -> dict[str, object]:
    """Fetch one allowlisted artifact into a content-verified cache."""

    destination = cache_dir / artifact.group / artifact.filename
    known_digest = _known_digest(artifact, previous)
    if known_digest is not None:
        cached = _cached_record(artifact, destination, known_digest)
        if cached is not None:
            return cached

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        request = Request(artifact.stable_url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=timeout_seconds) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".part",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temporary.write(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())
            final_url = response.geturl()
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")
        digest, md5_digest, size = _digest_file(temporary_path)
        if artifact.expected_sha256 is not None and digest != artifact.expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {artifact.artifact_id}: expected "
                f"{artifact.expected_sha256}, observed {digest}"
            )
        if artifact.expected_size is not None and size != artifact.expected_size:
            raise ValueError(
                f"size mismatch for {artifact.artifact_id}: expected "
                f"{artifact.expected_size}, observed {size}"
            )
        if known_digest is not None and digest != known_digest:
            raise ValueError(
                f"remote bytes changed from prior manifest for {artifact.artifact_id}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        return {
            **asdict(artifact),
            "status": "downloaded_verified",
            "sha256": digest,
            "md5": md5_digest,
            "size_bytes": size,
            "cache_path": str(destination),
            "final_url": final_url,
            "etag": etag,
            "last_modified": last_modified,
            "upstream_checksum_basis": (
                "fixed_expected_sha256"
                if artifact.expected_sha256 is not None
                else "official_url_plus_response_metadata_and_local_sha256"
            ),
        }
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _select_artifacts(group: str) -> tuple[OfficialArtifact, ...]:
    if group == "predictor":
        return PREDICTOR_ARTIFACTS
    if group == "fndds":
        return FNDDS_ARTIFACTS
    return PREDICTOR_ARTIFACTS + FNDDS_ARTIFACTS


def run_acquisition(
    artifacts: Iterable[OfficialArtifact],
    *,
    cache_dir: Path,
    manifest_path: Path,
    timeout_seconds: int,
    max_workers: int = 4,
) -> dict[str, object]:
    previous = _load_previous_manifest(manifest_path)
    selected = tuple(artifacts)
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    by_id: dict[str, dict[str, object]] = {}

    def attempt(artifact: OfficialArtifact) -> dict[str, object]:
        try:
            return fetch_artifact(
                artifact,
                cache_dir=cache_dir,
                previous=previous,
                timeout_seconds=timeout_seconds,
            )
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            return {
                **asdict(artifact),
                "status": "blocked",
                "error_type": type(error).__name__,
                "error": str(error),
            }

    with ThreadPoolExecutor(max_workers=min(max_workers, len(selected))) as executor:
        futures = {executor.submit(attempt, artifact): artifact for artifact in selected}
        for future in as_completed(futures):
            artifact = futures[future]
            by_id[artifact.artifact_id] = future.result()
            checkpoint = _acquisition_payload(
                [by_id[item.artifact_id] for item in selected if item.artifact_id in by_id]
            )
            _write_acquisition_manifest(manifest_path, checkpoint)
    payload = _acquisition_payload([by_id[item.artifact_id] for item in selected])
    _write_acquisition_manifest(manifest_path, payload)
    return payload


def _acquisition_payload(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "predictor-acquisition-manifest-v1",
        "accessed_on": ACCESSED_ON,
        "nature_data_boundary": "predictor_only_no_outcome_or_label_tables",
        "raw_fastq_downloaded": False,
        "artifacts": records,
    }


def _write_acquisition_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch official PREDICT/ZOE predictors and FNDDS source releases"
    )
    parser.add_argument("--group", choices=("predictor", "fndds", "all"), default="predictor")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Return success while retaining exact blocked records in the manifest.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.timeout_seconds <= 0 or args.max_workers <= 0:
        raise SystemExit("--timeout-seconds and --max-workers must be positive")
    manifest_path = args.manifest or args.cache_dir / f"{args.group}_acquisition_manifest.json"
    payload = run_acquisition(
        _select_artifacts(args.group),
        cache_dir=args.cache_dir,
        manifest_path=manifest_path,
        timeout_seconds=args.timeout_seconds,
        max_workers=args.max_workers,
    )
    blocked = [
        record for record in payload["artifacts"] if record["status"] == "blocked"
    ]
    summary = {
        "manifest": str(manifest_path),
        "n_artifacts": len(payload["artifacts"]),
        "n_verified": len(payload["artifacts"]) - len(blocked),
        "n_blocked": len(blocked),
        "blocked_ids": [record["artifact_id"] for record in blocked],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if blocked and not args.allow_partial:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
