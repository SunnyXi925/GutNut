from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_EDGE_COLUMNS = (
    "source",
    "target",
    "source_type",
    "target_type",
    "relation",
    "weight",
    "evidence_source",
)


@dataclass(frozen=True)
class EvidenceGraph:
    """Validated lightweight KEGG/GMMAD2-style edge table."""

    edges: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", _validate_edges(self.edges))

    @classmethod
    def from_paths(cls, paths: list[Path]) -> "EvidenceGraph":
        return cls(load_evidence_edges(paths))

    def score_nutrient_disease_paths(
        self,
        beta: pd.DataFrame,
        top_n: int = 5,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return score_nutrient_disease_paths(beta, self.edges, top_n=top_n)


def load_evidence_edges(paths: list[Path]) -> pd.DataFrame:
    """Load and validate lightweight evidence edge tables.

    Supported formats are CSV, TSV/TXT, JSON/JSONL, and parquet. The returned
    frame preserves the table edge types and normalizes weights to float.
    """
    if not paths:
        raise ValueError("at least one evidence edge path is required")

    frames = [_read_edge_table(Path(path)) for path in paths]
    edges = pd.concat(frames, ignore_index=True)
    return _validate_edges(edges)


def score_nutrient_disease_paths(
    beta: pd.DataFrame,
    edges: pd.DataFrame,
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one-hop and KEGG/GMMAD2 nutrient-to-disease evidence paths."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    validated_edges = _validate_edges(edges)
    path_templates = _enumerate_path_templates(validated_edges)

    path_rows: list[dict[str, object]] = []
    beta_frame = beta.astype(float).copy()
    beta_frame.index = beta_frame.index.astype(str)
    beta_frame.columns = beta_frame.columns.astype(str)

    for sample_id, row in beta_frame.iterrows():
        for template in path_templates:
            nutrient = str(template["nutrient"])
            if nutrient not in row.index:
                continue
            beta_value = float(row[nutrient])
            path_score = beta_value * float(template["path_weight"])
            path_rows.append(
                {
                    "sample_id": str(sample_id),
                    "nutrient": nutrient,
                    "disease": str(template["disease"]),
                    "path": template["path"],
                    "path_weight": float(template["path_weight"]),
                    "beta_value": beta_value,
                    "path_score": path_score,
                    "evidence_sources": template["evidence_sources"],
                    "relations": template["relations"],
                }
            )

    paths = pd.DataFrame(
        path_rows,
        columns=[
            "sample_id",
            "nutrient",
            "disease",
            "path",
            "path_weight",
            "beta_value",
            "path_score",
            "evidence_sources",
            "relations",
        ],
    )
    if paths.empty:
        return _empty_scores(), paths

    ranked = paths.assign(_abs_score=paths["path_score"].abs())
    ranked = ranked.sort_values(
        ["sample_id", "disease", "_abs_score", "path"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    top_paths = ranked.groupby(["sample_id", "disease"], as_index=False, group_keys=False).head(top_n)
    scores = (
        top_paths.groupby(["sample_id", "disease"], as_index=False)
        .agg(
            evidence_score=("path_score", "sum"),
            n_paths=("path_score", "size"),
            top_paths=("path", lambda values: " | ".join(map(str, values))),
            evidence_sources=("evidence_sources", _merge_semicolon_values),
        )
        .sort_values(["sample_id", "disease"], kind="mergesort")
        .reset_index(drop=True)
    )
    return scores, paths.reset_index(drop=True)


def _read_edge_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported evidence edge format: {path}")


def _validate_edges(edges: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_EDGE_COLUMNS).difference(edges.columns)
    if missing:
        raise ValueError(f"evidence edges missing required columns: {sorted(missing)}")

    frame = edges.loc[:, REQUIRED_EDGE_COLUMNS].copy()
    frame["weight"] = pd.to_numeric(frame["weight"], errors="raise").astype(float)
    for column in REQUIRED_EDGE_COLUMNS:
        if column != "weight":
            frame[column] = frame[column].astype(str)
    if frame["evidence_source"].str.strip().eq("").any():
        raise ValueError("evidence_source values must be non-empty")
    return frame


def _enumerate_path_templates(edges: pd.DataFrame) -> list[dict[str, object]]:
    one_hop = edges.loc[
        edges["source_type"].eq("nutrient") & edges["target_type"].eq("disease")
    ]
    templates = [_template_from_edges([edge]) for _, edge in one_hop.iterrows()]

    nutrient_to_compound = edges.loc[
        edges["source_type"].eq("nutrient") & edges["target_type"].eq("kegg_compound")
    ]
    compound_to_microbe = edges.loc[
        edges["source_type"].eq("kegg_compound") & edges["target_type"].eq("microbe")
    ]
    microbe_to_metabolite = edges.loc[
        edges["source_type"].eq("microbe") & edges["target_type"].eq("metabolite")
    ]
    metabolite_to_disease = edges.loc[
        edges["source_type"].eq("metabolite") & edges["target_type"].eq("disease")
    ]

    for _, e1 in nutrient_to_compound.iterrows():
        e2_candidates = compound_to_microbe.loc[compound_to_microbe["source"].eq(e1["target"])]
        for _, e2 in e2_candidates.iterrows():
            e3_candidates = microbe_to_metabolite.loc[microbe_to_metabolite["source"].eq(e2["target"])]
            for _, e3 in e3_candidates.iterrows():
                e4_candidates = metabolite_to_disease.loc[metabolite_to_disease["source"].eq(e3["target"])]
                for _, e4 in e4_candidates.iterrows():
                    templates.append(_template_from_edges([e1, e2, e3, e4]))

    return sorted(templates, key=lambda item: (str(item["nutrient"]), str(item["disease"]), str(item["path"])))


def _template_from_edges(edge_rows: Iterable[pd.Series]) -> dict[str, object]:
    rows = list(edge_rows)
    path_nodes = [str(rows[0]["source"])] + [str(edge["target"]) for edge in rows]
    path_weight = 1.0
    relations = []
    evidence_sources = []
    for edge in rows:
        path_weight *= float(edge["weight"])
        relations.append(str(edge["relation"]))
        evidence_sources.append(str(edge["evidence_source"]))
    return {
        "nutrient": str(rows[0]["source"]),
        "disease": str(rows[-1]["target"]),
        "path": "->".join(path_nodes),
        "path_weight": path_weight,
        "relations": ";".join(relations),
        "evidence_sources": _merge_semicolon_values(evidence_sources),
    }


def _merge_semicolon_values(values: Iterable[object]) -> str:
    unique: list[str] = []
    for value in values:
        for part in str(value).split(";"):
            clean = part.strip()
            if clean and clean not in unique:
                unique.append(clean)
    return ";".join(unique)


def _empty_scores() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["sample_id", "disease", "evidence_score", "n_paths", "top_paths", "evidence_sources"]
    )
