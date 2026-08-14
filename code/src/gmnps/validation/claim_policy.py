"""Machine-enforced claims bound to the repository's path-only evidence gate."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from gmnps.validation.evidence_gate import (
        EvidenceGateArtifactPaths,
        EvidenceGateOutcome,
    )


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_TRUSTED_DECISION_RELATIVE = Path(
    "results/phase2/evidence-gate/current_gate_decision.json"
)
_TRUSTED_POLICY_RELATIVE = Path("results/phase2/evidence-gate/claim_policy.json")
_TRUSTED_REGISTRY_RELATIVE = Path("code/src/configs/claim_policy_registry.json")
_TRUSTED_DECISION_PATH = _REPOSITORY_ROOT / _TRUSTED_DECISION_RELATIVE
_TRUSTED_POLICY_PATH = _REPOSITORY_ROOT / _TRUSTED_POLICY_RELATIVE
_TRUSTED_POLICY_REGISTRY_PATH = _REPOSITORY_ROOT / _TRUSTED_REGISTRY_RELATIVE
_CURRENT_BUNDLE_ID = "phase2-current-production-claim-bundle-v9"

_FORBIDDEN_PATTERNS = (
    r"\bvalidat(?:e|es|ed|ing|ion)\b",
    r"\bexternal\b",
    r"\bindependent\s+cohort\b",
    r"\bgenerali[sz]\w*\b",
    r"\b(?:subject[- ]?)?held[- ]out\b",
    r"\bimprov\w*(?:\s+\w+){0,4}\s+prediction\b",
    r"\bprediction(?:\s+\w+){0,4}\s+improv\w*\b",
    r"\bmetabolic\s+response\b",
    r"\bguidance\b",
    r"\brecommendation\w*\b",
    r"\bbiological\s+consistency\b",
    r"\bmechanistic\s+validation\b",
    r"\bclinical\w*\b",
    r"\bcausal\w*\b",
    r"\bprecision[- ]ready\b",
)
_CLAIM_SUBJECT_PATTERNS = (
    r"\bGMNPS\b",
    r"\b(?:model|framework|approach|method|system|platform|algorithm|implementation)\b",
    r"\bpersonal(?:i[sz]ation|i[sz]ed(?:\s+GMNPS)?\s+scores?)\b",
    r"\bmicrobiome[- ]informed(?:\s+\w+){0,2}\s+scores?\b",
    r"\b(?:GMNPS|personal(?:i[sz]ed)?|microbiome[- ]informed)\s+"
    r"(?:model|framework|approach|method|system|platform|algorithm|scores?)\b",
    r"\b(?:scores?|findings|results|analysis)\b",
)
_ASSERTION_PATTERNS = (
    r"\bpredict(?:s|ed|ing)?\b|\bprediction\b",
    r"\boutperform(?:s|ed|ing)?\b",
    r"\bimprov(?:e|es|ed|ing|ement|ements)\b",
    r"\bgenerali[sz]\w*\b",
    r"\bvalidat(?:e|es|ed|ing|ion|ions)\b",
    r"\baccur(?:ate|ately|acy)\b",
    r"\bestablish(?:es|ed|ing|ment)?\b",
    r"\bdemonstrat(?:e|es|ed|ing|ion|ions)\b",
    r"\bsupport(?:s|ed|ing)?\b",
    r"\benabl(?:e|es|ed|ing)\b",
    r"\bguid(?:e|es|ed|ing|ance)\b",
    r"\brecommend(?:s|ed|ing|ation|ations)?\b",
    r"\bassociate(?:s|d|ing)?\b|\bassociation(?:s)?\b",
    r"\brecover(?:s|ed|ing|y)?\b",
    r"\bresponses?\b",
)
_POSITIVE_ASSERTION_PATTERNS = (
    r"\bpredict(?:s|ed|ing)\b",
    r"\boutperform(?:s|ed|ing)?\b",
    r"\bimprov(?:e|es|ed|ing|ement|ements)\b",
    r"\bgenerali[sz](?:e|es|ed|ing)\b",
    r"\bvalidat(?:e|es|ed|ing)\b",
    r"\baccur(?:ate|ately)\b",
    r"\bestablish(?:es|ed|ing)?\b",
    r"\bdemonstrat(?:e|es|ed|ing)\b",
    r"\bsupport(?:s|ed|ing)?\b",
    r"\benabl(?:e|es|ed|ing)\b",
    r"\bguid(?:e|es|ed|ing)\b",
    r"\brecommend(?:s|ed|ing)?\b",
    r"\bassociate(?:s|d|ing)?\b",
    r"\brecover(?:s|ed|ing)?\b",
)
_ANAPHORIC_SUBJECT_PATTERN = r"\A(?:it|this|that|these|those|they)\b"
_EMPIRICAL_TARGET_PATTERN = (
    r"\b(?:outcomes?|performance|predictions?|responses?|validity|accuracy|"
    r"rmse|auroc)\b"
)
_TEX_SEMANTIC_VIEW_RULE = "explicit_tex_semantic_view_v1"
_ZERO_WIDTH_BRACED_COMMANDS = frozenset({"index", "label"})
_ZERO_WIDTH_BARE_COMMANDS = frozenset({"phantomsection"})
_LATEX_STRUCTURAL_COMMANDS = frozenset(
    {
        "abstract",
        "author",
        "backmatter",
        "begin",
        "bibliography",
        "bottomrule",
        "caption",
        "centering",
        "clearpage",
        "documentclass",
        "end",
        "graphicspath",
        "include",
        "input",
        "keywords",
        "maketitle",
        "midrule",
        "newpage",
        "raggedbottom",
        "renewcommand",
        "section",
        "setcounter",
        "subsection",
        "subsubsection",
        "title",
        "toprule",
        "unnumbered",
        "usepackage",
    }
)
_SUBJECT_DEFAULT_DENY = True
_DIRECT_SCOPED_SENTENCE = (
    "For the locked primary endpoints glucose_iAUC_2h and tg_6h_rise, "
    "locked_attribute_gmnps had lower subject-held-out RMSE than fcs_microbiome."
)
_COMPUTATIONAL_POSITIVE_TEMPLATES = (
    "The Gut Microbiome-informed Nutrient Profiling System (GMNPS) uses bounded "
    "attribute-level calibration with locked inputs.",
    "The locked implementation recovered the programmed mapping in a correctly "
    "specified synthetic positive-control.",
    "Eligible observed participant-by-meal outcomes are unavailable in the audited data.",
    "The evidence gate implements a computational, fail-closed design.",
)
_EXACT_NON_EMPIRICAL_TEMPLATES = (
    "The microbiome is used here as a feature source rather than a causal "
    "mechanism or a sufficient basis for dietary decisions.",
    "The prespecified evidence chain separates method establishment, evidence "
    "authorization, data availability, a correctly specified synthetic "
    "positive-control and the boundary on empirical interpretation.",
    "The current decision did not pass because the required real validation "
    "artifacts were absent.",
    "This content review does not establish outcome validity or superiority of "
    "the revised channel assignment.",
    r"The 20\% native-attribute cap, \(\pm 12\)-point final cap, absence of score "
    "centring and exact zero-response identity are safety properties of the design.",
    "Because generator and implementation share the mapping, this evidence "
    "diagnoses implementation fidelity but not biological misspecification or "
    "real-world response prediction.",
    "It supports content review of four nutrient-channel decisions, but it does "
    "not establish outcome validity, external validity, readiness for clinical "
    "use or superiority of an expert-revised mask.",
    "Eligible observed participant-by-meal outcomes, a trusted production food "
    "run and registry-bound supporting analyses are absent, so the main "
    "quantitative evidence needed for response validity, population behaviour "
    "and generalization cannot be assembled.",
    "A future empirical programme should preserve the frozen method while adding "
    "a trusted production food bundle, eligible observed outcomes and held-out "
    "analyses whose splits, inference units, intervals, multiplicity control, "
    "leakage checks and provenance are fixed before outcome inspection.",
    "Clinical deployment would additionally require prospective safety "
    "assessment, decision-curve analysis and reporting appropriate to high-risk "
    r"prediction tools \cite{collins2015tripod,wolff2019probast,collins2024tripodAi}.",
    "The locked method version was fixed before validation access and separated "
    "development-only normalization from later scoring inputs.",
    "Available external feedback informed revisions concerning carbohydrate, "
    "zinc, copper and vitamin A as retinol activity equivalents.",
    "The evidence gate requires path-bound artifacts, a valid run-level method "
    "lock, disjoint development and test subjects, eligible observed outcomes "
    "and the prespecified subject-held-out comparisons before any direct-response "
    "claim can be authorized.",
    "The locked implementation, evidence-gate code and audit scripts are "
    "maintained in the project repository.",
    r"Its primary method is \texttt{attribute\_recomposition}, its primary mapping "
    r"is \texttt{expert\_reviewed\_attribute\_mapping\_v1}, and its recomposition "
    r"rule is \texttt{native\_domain\_fixed\_residual\_v1}.",
    "Mapping roles, allocation weights, attribute rules, temperatures, cap modes, "
    "domain membership and score centring were locked before outcome access.",
    "The nutrient order, medians, scales, scale-source methods, development fit "
    "count, SHA-256 of sorted development identifiers, method version and "
    "normalization temperature are immutable normalization state.",
    "Held-out or later scoring participants are transformed with this state and "
    "are never appended to the development fit.",
    "The trusted registry snapshot must have the fixed schema, registry version "
    "and SHA-256 digest algorithm and must contain the single approved artifact "
    "entry bound to the run.",
    "Hash-bound external C1 workbooks contained available item-level feedback; "
    "available comments informed the treatment of carbohydrate, zinc, copper and "
    "vitamin A as retinol activity equivalents.",
    "For future direct-response analysis, the complete frozen predictor "
    "opportunity universe is split before outcomes are joined.",
    "Food-held-out, meal-held-out and cohort-held-out modes are secondary and "
    "descriptive unless valid multiway or cohort-cluster inference is implemented.",
    r"\item population food analysis: a trusted rectangular production person-food "
    "panel, with food as the analysis and inference unit for population summaries; "
    r"\item direct response: participant-meal observations with family/twin connected "
    "components as the inferential unit; "
    r"\item GMrepo: independent people or connected components within cohort; "
    r"\item ZOE aggregate ranks: one source-bound food/rank-table unit; and "
    r"\item knowledge paths: one registry-bound adjudicated path unit.",
    "No blocked analysis was replaced by a test fixture, synthetic output or "
    "placeholder statistic.",
    "Because the generator uses the programmed mapping and domain aggregation "
    "evaluated by the implementation, these findings test numerical and "
    "assignment fidelity under correct specification.",
    "No eligible observed participant-by-meal outcome artifact, run-level "
    "method-lock instance, paired metric table, prediction table, split audit or "
    "analysis-status artifact was available.",
)
_EXACT_NEGATIVE_LIMITATION_TEMPLATES = (
    "The framework provides an auditable basis for future empirical testing, but "
    "GMNPS does not establish external validity.",
    "No analysis evaluated clinical utility or dietary recommendations.",
)
_NEGATIVE_LIMITATION_RULE = "strict_direct_governance_v1"
_DIRECT_NEGATIVE_CUE_PATTERN = (
    r"(?:\b(?:does|do|did|is|are|was|were|has|have|had|can|could|will|would)"
    r"\s+not|\bcannot|\bcan't|\bno\s+evidence(?:\s+(?:of|for))?)\s*\Z"
)
_NEGATIVE_NOMINAL_TAIL_PATTERN = (
    r"(?:\s+(?:validity|validation|evidence|claim|claims|performance|effect|"
    r"effects|response|responses|utility|readiness|association|consistency)){0,2}"
)
_NEGATIVE_PASSIVE_LINK_PATTERN = (
    rf"\A{_NEGATIVE_NOMINAL_TAIL_PATTERN}\s+"
    r"(?:is|are|was|were|has|have|had)\s+not(?:\s+been)?\s*\Z"
)
_NEGATIVE_PASSIVE_PATTERN = (
    rf"\A{_NEGATIVE_NOMINAL_TAIL_PATTERN}\s+"
    r"(?:is|are|was|were|has|have|had)\s+not(?:\s+been)?\s+"
    r"(?:established|demonstrated|shown|supported|assessed|evaluated|performed|"
    r"available|validated)\b"
)
_REMAINS_UNVALIDATED_PATTERN = (
    r"\A[^.!?,;:]+\bremains?\s+unvalidated\.?\Z"
)
_NEGATION_INVERSION_PATTERNS = (
    r"\bnot\s+fail(?:s|ed|ing)?\s+to\b",
    r"\b(?:cannot|can't)\s+fail(?:s|ed|ing)?\s+to\b",
    r"\bnot\s+only\b",
)
_LOCAL_CLAUSE_BOUNDARY = re.compile(
    r"[,;:]|\b(?:but|however|yet|although|though|whereas|while)\b",
    flags=re.IGNORECASE,
)
_METHODS_ACTOR = (
    r"(?:the\s+)?(?:GMNPS(?:\s+(?:model|framework|approach|method|system|"
    r"platform|algorithm|implementation))?|model|framework|approach|method|"
    r"system|platform|algorithm|implementation|evidence\s+gate)"
)
_METHODS_INFRASTRUCTURE_PATTERNS = (
    rf"\A{_METHODS_ACTOR}\s+(?:is\s+)?implemented\s+as\s+(?:an?\s+)?"
    r"(?:(?:locked|deterministic|computational|fail-closed|bounded|"
    r"attribute-level|repository-local|path-only)\s+){0,4}"
    r"(?:pipeline|workflow|procedure|implementation|scoring\s+pipeline|"
    r"evidence\s+gate|software\s+component)\.?\Z",
    rf"\A{_METHODS_ACTOR}\s+computes\s+(?:bounded\s+)?"
    r"(?:attribute(?:-level)?|component|domain|GMNPS)\s+scores?"
    r"(?:\s+from\s+(?:locked|verified|repository-local)\s+"
    r"(?:inputs|configuration|artifacts))?\.?\Z",
    rf"\A{_METHODS_ACTOR}\s+loads\s+(?:the\s+)?"
    r"(?:(?:fixed|locked|verified|registry-bound|repository-local|trusted)\s+)?"
    r"(?:inputs|configuration|registry|manifest|artifacts?|source\s+files?)\.?\Z",
    rf"\A{_METHODS_ACTOR}\s+verifies\s+(?:the\s+)?"
    r"(?:(?:fixed|locked|trusted|source|artifact)\s+)?"
    r"(?:sha-?256\s+)?(?:hashes|digests|checksums|manifests|registries|"
    r"artifacts|configuration|file\s+paths|inputs)\.?\Z",
    rf"\A{_METHODS_ACTOR}\s+hash-binds\s+(?:its\s+|the\s+)?"
    r"(?:artifacts|outputs|claim\s+policy|configuration|source\s+files?)\s+to\s+"
    r"(?:its\s+|the\s+|their\s+)?(?:sha-?256\s+)?"
    r"(?:digests|hashes|source\s+manifest|registry|configuration|gate\s+decision)"
    r"\.?\Z",
    rf"\A{_METHODS_ACTOR}\s+uses\s+bounded\s+attribute(?:-level)?\s+calibration"
    r"(?:\s+with\s+locked\s+(?:inputs|configuration))?\.?\Z",
    rf"\A{_METHODS_ACTOR}\s+source\s+hash\s+"
    r"(?:supports|enables)\s+reproducible\s+artifact\s+verification\.?\Z",
)
_METHODS_INFRASTRUCTURE_EXACT_IDENTIFIERS = ("attribute-gmnps-v1",)
_METHODS_FORBIDDEN_SEMANTICS_PATTERN = (
    r"\b(?:outcomes?|performance|validat\w*|validity|guid\w*|recommend\w*|"
    r"predict\w*|forecast\w*|stratif\w*|responses?|rmse|mae|auroc|superior|"
    r"better|lower|improv\w*|achiev\w*|yield\w*|glycaem\w*|glucos\w*|"
    r"triglyceride\w*|postprandial|metabolic|clinical\w*|biological\w*|"
    r"causal\w*|external|cohort|participant|patient|individual|personal(?:i[sz]ed|"
    r"i[sz]ation)|microbiome|nutrition|health|dietary|food\s+compass)\b"
)
_METHODS_FORBIDDEN_SEMANTICS = re.compile(
    _METHODS_FORBIDDEN_SEMANTICS_PATTERN,
    flags=re.IGNORECASE,
)
_NEGATION_SCOPE_BREAK = re.compile(
    r"\b(?:and|or|but|yet|however)\b",
    flags=re.IGNORECASE,
)
_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ClaimViolation:
    path: Path
    pattern: str


def canonical_payload_sha256(payload: dict[str, object]) -> str:
    core = {key: value for key, value in payload.items() if key != "payload_sha256"}
    return sha256(
        json.dumps(
            core,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _build_claim_policy_payload(
    outcome: "EvidenceGateOutcome",
    *,
    production_authorized: bool,
) -> dict[str, object]:
    """Build policy bytes; only the path-only gate wrapper may authorize them."""

    direct = outcome.tier == "direct_external_validity" and outcome.passed is True
    positive_templates = list(_COMPUTATIONAL_POSITIVE_TEMPLATES)
    if direct:
        positive_templates.append(_DIRECT_SCOPED_SENTENCE)
    payload: dict[str, object] = {
        "schema_version": "claim-policy-v9",
        "tier": outcome.tier,
        "source_state": outcome.source_state,
        "allowed_claims": list(outcome.allowed_claims),
        "forbidden_claims": list(outcome.prohibited_claims),
        "forbidden_patterns": list(_FORBIDDEN_PATTERNS),
        "claim_subject_patterns": list(_CLAIM_SUBJECT_PATTERNS),
        "assertion_patterns": list(_ASSERTION_PATTERNS),
        "positive_assertion_patterns": list(_POSITIVE_ASSERTION_PATTERNS),
        "anaphoric_subject_pattern": _ANAPHORIC_SUBJECT_PATTERN,
        "empirical_target_pattern": _EMPIRICAL_TARGET_PATTERN,
        "tex_semantic_view_rule": _TEX_SEMANTIC_VIEW_RULE,
        "negative_limitation_rule": _NEGATIVE_LIMITATION_RULE,
        "negative_limitation_direct_cue_pattern": _DIRECT_NEGATIVE_CUE_PATTERN,
        "negative_limitation_passive_link_pattern": (
            _NEGATIVE_PASSIVE_LINK_PATTERN
        ),
        "negative_limitation_passive_pattern": _NEGATIVE_PASSIVE_PATTERN,
        "negative_limitation_remains_unvalidated_pattern": (
            _REMAINS_UNVALIDATED_PATTERN
        ),
        "negative_inversion_patterns": list(_NEGATION_INVERSION_PATTERNS),
        "subject_default_deny": _SUBJECT_DEFAULT_DENY,
        "methods_infrastructure_patterns": list(_METHODS_INFRASTRUCTURE_PATTERNS),
        "methods_infrastructure_exact_identifiers": list(
            _METHODS_INFRASTRUCTURE_EXACT_IDENTIFIERS
        ),
        "methods_forbidden_semantics_pattern": _METHODS_FORBIDDEN_SEMANTICS_PATTERN,
        "direct_scope_whitelist": [_DIRECT_SCOPED_SENTENCE] if direct else [],
        "positive_claim_templates": positive_templates,
        "exact_non_empirical_templates": list(_EXACT_NON_EMPIRICAL_TEMPLATES),
        "negative_limitation_sentences_allowed": True,
        "exact_negative_limitation_templates": list(
            _EXACT_NEGATIVE_LIMITATION_TEMPLATES
        ),
        "scope": "Phase 3 manuscript and build inputs",
        "phase3_must_consume": True,
        "authorization": (
            "production_path_only_evidence_gate"
            if production_authorized
            else "testing_only_unauthorized"
        ),
    }
    payload["payload_sha256"] = canonical_payload_sha256(payload)
    return payload


def _write_claim_artifacts(
    output_directory: Path,
    outcome: "EvidenceGateOutcome",
    *,
    production_authorized: bool,
) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    policy = _build_claim_policy_payload(
        outcome,
        production_authorized=production_authorized,
    )
    policy_bytes = _json_bytes(policy)
    policy_path = output_directory / "claim_policy.json"
    policy_path.write_bytes(policy_bytes)
    decision: dict[str, object] = {
        "schema_version": "evidence-gate-decision-v2",
        "gate_version": outcome.gate_version,
        "tier": outcome.tier,
        "passed": outcome.passed,
        "blockers": list(outcome.blockers),
        "allowed_claims": list(outcome.allowed_claims),
        "forbidden_claims": list(outcome.prohibited_claims),
        "claim_policy_file": policy_path.name,
        "claim_policy_file_sha256": sha256(policy_bytes).hexdigest(),
        "source_state": outcome.source_state,
        "authorization": policy["authorization"],
        "manuscript_checked_or_revised": False,
    }
    decision["payload_sha256"] = canonical_payload_sha256(decision)
    decision_path = output_directory / "current_gate_decision.json"
    decision_path.write_bytes(_json_bytes(decision))
    return {"decision": decision_path, "policy": policy_path}


def _write_testing_only_claim_artifacts(
    output_directory: str | Path,
    outcome: "EvidenceGateOutcome",
) -> dict[str, Path]:
    """Materialize an explicitly unauthorized fixture from an in-memory outcome."""

    return _write_claim_artifacts(
        Path(output_directory),
        outcome,
        production_authorized=False,
    )


def build_claim_policy_from_evidence_gate(
    paths: "EvidenceGateArtifactPaths | None" = None,
) -> dict[str, Path]:
    """Rerun the path-only production gate and write only the fixed policy bundle."""

    from gmnps.validation.evidence_gate import (
        EvidenceGateArtifactPaths,
        evaluate_evidence_gate,
    )

    if paths is not None:
        if not isinstance(paths, EvidenceGateArtifactPaths):
            raise TypeError("claim policy production builder accepts only gate paths")
    outcome = evaluate_evidence_gate(paths)
    generated = _write_claim_artifacts(
        _TRUSTED_DECISION_PATH.parent,
        outcome,
        production_authorized=True,
    )
    policy_raw = generated["policy"].read_bytes()
    decision_raw = generated["decision"].read_bytes()
    registry = {
        "schema_version": "claim-policy-registry-v1",
        "approved_bundles": [
            {
                "bundle_id": _CURRENT_BUNDLE_ID,
                "tier": outcome.tier,
                "source_state": outcome.source_state,
                "decision_path": _TRUSTED_DECISION_RELATIVE.as_posix(),
                "policy_path": _TRUSTED_POLICY_RELATIVE.as_posix(),
                "decision_sha256": sha256(decision_raw).hexdigest(),
                "policy_sha256": sha256(policy_raw).hexdigest(),
            }
        ],
    }
    _TRUSTED_POLICY_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TRUSTED_POLICY_REGISTRY_PATH.write_bytes(_json_bytes(registry))
    generated["registry"] = _TRUSTED_POLICY_REGISTRY_PATH
    return generated


def _read_regular_bytes(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing or unreadable") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{label} changed during immutable open")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"{label} changed during live reread")
    return b"".join(chunks)


def _load_verified_json(path: Path, label: str) -> tuple[bytes, dict[str, object]]:
    raw = _read_regular_bytes(path, label)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    expected = payload.get("payload_sha256")
    if not isinstance(expected, str) or expected != canonical_payload_sha256(payload):
        raise ValueError(f"{label} payload hash is invalid")
    return raw, payload


def _validate_registry(
    decision_raw: bytes,
    policy_raw: bytes,
    decision: dict[str, object],
    policy: dict[str, object],
) -> None:
    raw = _read_regular_bytes(_TRUSTED_POLICY_REGISTRY_PATH, "claim policy registry")
    try:
        registry = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("claim policy registry is not valid JSON") from error
    if not isinstance(registry, dict) or set(registry) != {
        "schema_version",
        "approved_bundles",
    }:
        raise ValueError("claim policy registry schema is invalid")
    if registry.get("schema_version") != "claim-policy-registry-v1":
        raise ValueError("claim policy registry version is invalid")
    approved = registry.get("approved_bundles")
    if not isinstance(approved, list) or not approved:
        raise ValueError("claim policy registry has no approved bundle")
    required = {
        "bundle_id",
        "tier",
        "source_state",
        "decision_path",
        "policy_path",
        "decision_sha256",
        "policy_sha256",
    }
    valid_rows: list[dict[str, object]] = []
    for row in approved:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("claim policy registry approved entry is invalid")
        if not isinstance(row["bundle_id"], str) or not row["bundle_id"]:
            raise ValueError("claim policy registry bundle_id is invalid")
        if row["decision_path"] != _TRUSTED_DECISION_RELATIVE.as_posix():
            raise ValueError("claim policy registry decision path is not fixed")
        if row["policy_path"] != _TRUSTED_POLICY_RELATIVE.as_posix():
            raise ValueError("claim policy registry policy path is not fixed")
        if not _DIGEST.fullmatch(str(row["decision_sha256"])) or not _DIGEST.fullmatch(
            str(row["policy_sha256"])
        ):
            raise ValueError("claim policy registry digest is invalid")
        valid_rows.append(row)
    decision_digest = sha256(decision_raw).hexdigest()
    policy_digest = sha256(policy_raw).hexdigest()
    matches = [
        row
        for row in valid_rows
        if row["decision_sha256"] == decision_digest
        and row["policy_sha256"] == policy_digest
        and row["tier"] == decision.get("tier") == policy.get("tier")
        and row["source_state"]
        == decision.get("source_state")
        == policy.get("source_state")
    ]
    if len(matches) != 1:
        raise ValueError("decision/policy pair is not approved by the fixed registry")


def _validate_bound_bundle() -> dict[str, object]:
    policy_raw, policy = _load_verified_json(_TRUSTED_POLICY_PATH, "claim policy")
    decision_raw, decision = _load_verified_json(
        _TRUSTED_DECISION_PATH, "gate decision"
    )
    if decision.get("schema_version") != "evidence-gate-decision-v2":
        raise ValueError("gate decision schema version is invalid")
    if policy.get("schema_version") != "claim-policy-v9":
        raise ValueError("claim policy schema version is invalid")
    if decision.get("authorization") != "production_path_only_evidence_gate" or policy.get(
        "authorization"
    ) != "production_path_only_evidence_gate":
        raise ValueError("claim bundle was not produced by the production path-only gate")
    if decision.get("claim_policy_file") != _TRUSTED_POLICY_PATH.name:
        raise ValueError("claim policy filename does not match fixed path")
    if decision.get("claim_policy_file_sha256") != sha256(policy_raw).hexdigest():
        raise ValueError("claim policy hash does not match gate decision")
    for field in ("tier", "source_state", "allowed_claims", "forbidden_claims"):
        if decision.get(field) != policy.get(field):
            raise ValueError(f"claim policy {field} does not match gate decision")
    patterns = policy.get("forbidden_patterns")
    if patterns != list(_FORBIDDEN_PATTERNS):
        raise ValueError("claim policy forbidden_patterns is invalid or incomplete")
    if policy.get("claim_subject_patterns") != list(_CLAIM_SUBJECT_PATTERNS):
        raise ValueError("claim policy claim_subject_patterns is invalid")
    if policy.get("assertion_patterns") != list(_ASSERTION_PATTERNS):
        raise ValueError("claim policy assertion_patterns is invalid")
    if policy.get("positive_assertion_patterns") != list(
        _POSITIVE_ASSERTION_PATTERNS
    ):
        raise ValueError("claim policy positive_assertion_patterns is invalid")
    if policy.get("anaphoric_subject_pattern") != _ANAPHORIC_SUBJECT_PATTERN:
        raise ValueError("claim policy anaphoric-subject rule is invalid")
    if policy.get("empirical_target_pattern") != _EMPIRICAL_TARGET_PATTERN:
        raise ValueError("claim policy empirical-target rule is invalid")
    if policy.get("tex_semantic_view_rule") != _TEX_SEMANTIC_VIEW_RULE:
        raise ValueError("claim policy TeX semantic-view rule is invalid")
    expected_negative_policy = {
        "negative_limitation_rule": _NEGATIVE_LIMITATION_RULE,
        "negative_limitation_direct_cue_pattern": _DIRECT_NEGATIVE_CUE_PATTERN,
        "negative_limitation_passive_link_pattern": _NEGATIVE_PASSIVE_LINK_PATTERN,
        "negative_limitation_passive_pattern": _NEGATIVE_PASSIVE_PATTERN,
        "negative_limitation_remains_unvalidated_pattern": (
            _REMAINS_UNVALIDATED_PATTERN
        ),
        "negative_inversion_patterns": list(_NEGATION_INVERSION_PATTERNS),
    }
    if any(policy.get(key) != value for key, value in expected_negative_policy.items()):
        raise ValueError("claim policy negative-limitation rule is invalid")
    if policy.get("subject_default_deny") is not _SUBJECT_DEFAULT_DENY:
        raise ValueError("claim policy subject-default-deny rule is invalid")
    if policy.get("methods_infrastructure_patterns") != list(
        _METHODS_INFRASTRUCTURE_PATTERNS
    ):
        raise ValueError("claim policy Methods infrastructure patterns are invalid")
    if policy.get("methods_infrastructure_exact_identifiers") != list(
        _METHODS_INFRASTRUCTURE_EXACT_IDENTIFIERS
    ):
        raise ValueError("claim policy Methods exact identifiers are invalid")
    if (
        policy.get("methods_forbidden_semantics_pattern")
        != _METHODS_FORBIDDEN_SEMANTICS_PATTERN
    ):
        raise ValueError("claim policy Methods semantic exclusions are invalid")
    whitelist = policy.get("direct_scope_whitelist")
    expected_whitelist = (
        [_DIRECT_SCOPED_SENTENCE]
        if decision.get("tier") == "direct_external_validity"
        and decision.get("passed") is True
        else []
    )
    if whitelist != expected_whitelist:
        raise ValueError("claim policy direct scope whitelist is invalid")
    expected_templates = list(_COMPUTATIONAL_POSITIVE_TEMPLATES)
    if decision.get("tier") == "direct_external_validity" and decision.get("passed") is True:
        expected_templates.append(_DIRECT_SCOPED_SENTENCE)
    if policy.get("positive_claim_templates") != expected_templates:
        raise ValueError("claim policy positive_claim_templates is invalid")
    if policy.get("exact_non_empirical_templates") != list(
        _EXACT_NON_EMPIRICAL_TEMPLATES
    ):
        raise ValueError("claim policy exact non-empirical templates are invalid")
    if policy.get("negative_limitation_sentences_allowed") is not True:
        raise ValueError("claim policy negative-limitation rule is invalid")
    if policy.get("exact_negative_limitation_templates") != list(
        _EXACT_NEGATIVE_LIMITATION_TEMPLATES
    ):
        raise ValueError("claim policy exact negative limitations are invalid")
    _validate_registry(decision_raw, policy_raw, decision, policy)
    return policy


def _strip_unescaped_tex_comment(line: str) -> tuple[str, bool]:
    for index, character in enumerate(line):
        if character != "%":
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and line[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return line[:index], True
    return line, False


def _balanced_group_end(text: str, start: int) -> int | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    index = start
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def _remove_zero_width_tex_commands(line: str) -> str:
    rendered: list[str] = []
    index = 0
    while index < len(line):
        command = re.match(r"\\([A-Za-z]+)", line[index:])
        if command is None:
            rendered.append(line[index])
            index += 1
            continue
        name = command.group(1)
        command_end = index + command.end()
        if name in _ZERO_WIDTH_BARE_COMMANDS:
            index = command_end
            while index < len(line) and line[index].isspace():
                index += 1
            continue
        if name in _ZERO_WIDTH_BRACED_COMMANDS:
            argument_start = command_end
            while argument_start < len(line) and line[argument_start].isspace():
                argument_start += 1
            argument_end = _balanced_group_end(line, argument_start)
            if argument_end is not None:
                index = argument_end
                continue
        rendered.append(line[index:command_end])
        index = command_end
    return "".join(rendered)


def _is_semantic_line_boundary(line: str) -> bool:
    stripped = line.strip()
    if (
        not stripped
        or stripped in {r"\[", r"\]", "$$"}
        or stripped.endswith(r"\\")
    ):
        return True
    command = re.match(r"\\([A-Za-z]+)\*?", stripped)
    return command is not None and command.group(1) in _LATEX_STRUCTURAL_COMMANDS


def _build_tex_semantic_view(text: str) -> str:
    semantic: list[str] = []
    prose_run: list[str] = []
    pending_separator = ""

    def flush_prose_run() -> None:
        nonlocal pending_separator
        if prose_run:
            semantic.append(re.sub(r"\s+", " ", "".join(prose_run)).strip())
            prose_run.clear()
        pending_separator = ""

    for raw_line in text.splitlines():
        if not raw_line.strip():
            flush_prose_run()
            semantic.append("")
            continue
        uncommented_line, comment_suppresses_newline = (
            _strip_unescaped_tex_comment(raw_line)
        )
        visible_line = _remove_zero_width_tex_commands(
            uncommented_line
        )
        if visible_line.strip() and _is_semantic_line_boundary(visible_line):
            flush_prose_run()
            semantic.append(visible_line)
        else:
            prose_run.extend((pending_separator, visible_line))
            pending_separator = "" if comment_suppresses_newline else " "
    flush_prose_run()
    return "\n".join(semantic)


def _sentences(text: str, *, tex_semantic_view_rule: str) -> tuple[str, ...]:
    if tex_semantic_view_rule != _TEX_SEMANTIC_VIEW_RULE:
        raise ValueError("unsupported TeX semantic-view rule")
    semantic_text = _build_tex_semantic_view(text)
    return tuple(
        match.group(0).strip()
        for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", semantic_text)
        if match.group(0).strip()
    )


def _pattern_matches(
    patterns: Iterable[str], sentence: str
) -> list[tuple[str, re.Match[str]]]:
    return [
        (pattern, match)
        for pattern in patterns
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE)
    ]


def _is_methods_infrastructure_sentence(
    sentence: str,
    policy: dict[str, object],
) -> bool:
    exact_identifier_sentences = {
        rf"The frozen specification is \texttt{{{identifier}}}."
        for identifier in policy.get("methods_infrastructure_exact_identifiers", [])
    }
    return (
        _METHODS_FORBIDDEN_SEMANTICS.search(sentence) is None
        and (
            sentence in exact_identifier_sentences
            or any(
                re.fullmatch(pattern, sentence, flags=re.IGNORECASE) is not None
                for pattern in _METHODS_INFRASTRUCTURE_PATTERNS
            )
        )
    )


def _is_negative_limitation_sentence(
    sentence: str,
    forbidden_matches: list[tuple[str, re.Match[str]]],
    assertion_matches: list[tuple[str, re.Match[str]]],
) -> bool:
    if (
        _LOCAL_CLAUSE_BOUNDARY.search(sentence) is not None
        or _NEGATION_SCOPE_BREAK.search(sentence) is not None
        or any(
            re.search(pattern, sentence, flags=re.IGNORECASE) is not None
            for pattern in _NEGATION_INVERSION_PATTERNS
        )
    ):
        return False
    matches = [
        ("forbidden", pattern, match) for pattern, match in forbidden_matches
    ] + [
        ("assertion", pattern, match) for pattern, match in assertion_matches
    ]
    if not matches:
        return (
            re.fullmatch(
                _REMAINS_UNVALIDATED_PATTERN,
                sentence,
                flags=re.IGNORECASE,
            )
            is not None
        )

    def overlaps(left: re.Match[str], right: re.Match[str]) -> bool:
        return left.start() < right.end() and right.start() < left.end()

    for _, _, anchor in matches:
        if re.search(
            _DIRECT_NEGATIVE_CUE_PATTERN,
            sentence[: anchor.start()],
            flags=re.IGNORECASE,
        ) is None:
            continue
        governed = True
        for kind, _, current in matches:
            if overlaps(anchor, current):
                continue
            if current.start() >= anchor.end():
                if kind == "assertion":
                    governed = False
                    break
                continue
            if current.end() <= anchor.start() and re.fullmatch(
                _NEGATIVE_PASSIVE_LINK_PATTERN,
                sentence[current.end() : anchor.start()],
                flags=re.IGNORECASE,
            ) is not None:
                continue
            governed = False
            break
        if governed:
            return True

    return all(
        re.search(
            _NEGATIVE_PASSIVE_PATTERN,
            sentence[match.end() :],
            flags=re.IGNORECASE,
        )
        is not None
        for _, _, match in matches
    )


def _is_exact_positive_template(sentence: str, policy: dict[str, object]) -> bool:
    normalized = re.sub(r"\s+", " ", sentence).strip().casefold()
    return normalized in {
        str(template).casefold()
        for template in policy.get("positive_claim_templates", [])
    }


def _is_exact_non_empirical_template(
    sentence: str,
    policy: dict[str, object],
) -> bool:
    normalized = re.sub(r"\s+", " ", sentence).strip()
    return normalized in policy.get("exact_non_empirical_templates", [])


def check_claim_inputs(
    input_paths: Iterable[str | Path],
) -> tuple[ClaimViolation, ...]:
    """Check inputs against the fixed, registry-approved production claim bundle."""

    policy = _validate_bound_bundle()
    forbidden_patterns = policy["forbidden_patterns"]
    subject_patterns = policy["claim_subject_patterns"]
    assertion_patterns = policy["assertion_patterns"]
    positive_assertion_patterns = policy["positive_assertion_patterns"]
    violations: list[ClaimViolation] = []
    for value in input_paths:
        path = Path(value)
        text = path.read_text(encoding="utf-8")
        for sentence in _sentences(
            text,
            tex_semantic_view_rule=str(policy["tex_semantic_view_rule"]),
        ):
            if _is_exact_positive_template(sentence, policy):
                continue
            if _is_exact_non_empirical_template(sentence, policy):
                continue
            if sentence in policy.get("exact_negative_limitation_templates", []):
                continue
            forbidden_matches = _pattern_matches(forbidden_patterns, sentence)
            assertion_matches = _pattern_matches(assertion_patterns, sentence)
            if _is_negative_limitation_sentence(
                sentence,
                forbidden_matches,
                assertion_matches,
            ):
                continue
            positive_assertion_matches = _pattern_matches(
                positive_assertion_patterns,
                sentence,
            )
            if positive_assertion_matches:
                anaphoric_subject = re.search(
                    str(policy["anaphoric_subject_pattern"]),
                    sentence,
                    flags=re.IGNORECASE,
                )
                empirical_target = re.search(
                    str(policy["empirical_target_pattern"]),
                    sentence,
                    flags=re.IGNORECASE,
                )
                if anaphoric_subject is not None:
                    violations.append(
                        ClaimViolation(
                            path=path,
                            pattern=str(policy["anaphoric_subject_pattern"]),
                        )
                    )
                    continue
                if empirical_target is not None:
                    violations.append(
                        ClaimViolation(
                            path=path,
                            pattern=str(policy["empirical_target_pattern"]),
                        )
                    )
                    continue
            if forbidden_matches:
                violations.append(
                    ClaimViolation(path=path, pattern=forbidden_matches[0][0])
                )
                continue
            subject_matches = _pattern_matches(subject_patterns, sentence)
            if not subject_matches:
                continue
            if _is_methods_infrastructure_sentence(sentence, policy):
                continue
            violations.append(
                ClaimViolation(path=path, pattern=subject_matches[0][0])
            )
    return tuple(violations)


__all__ = [
    "ClaimViolation",
    "build_claim_policy_from_evidence_gate",
    "canonical_payload_sha256",
    "check_claim_inputs",
]
