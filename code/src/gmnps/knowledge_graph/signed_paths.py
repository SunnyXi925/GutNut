from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


REQUIRED_SIGNED_EDGE_COLUMNS = (
    "source",
    "target",
    "source_type",
    "target_type",
    "relation",
    "sign",
    "weight",
    "evidence_source",
)


def validate_signed_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Validate a signed KG edge table without losing its direction or sign."""
    missing = set(REQUIRED_SIGNED_EDGE_COLUMNS).difference(edges.columns)
    if missing:
        raise ValueError(f"signed edges missing required columns: {sorted(missing)}")

    frame = edges.loc[:, REQUIRED_SIGNED_EDGE_COLUMNS].copy()
    frame["sign"] = pd.to_numeric(frame["sign"], errors="raise")
    if not frame["sign"].isin((-1, 1)).all():
        raise ValueError("sign values must be either -1 or 1")
    frame["sign"] = frame["sign"].astype(int)

    frame["weight"] = pd.to_numeric(frame["weight"], errors="raise").astype(float)
    if not np.isfinite(frame["weight"]).all() or frame["weight"].lt(0).any():
        raise ValueError("weight values must be finite and non-negative")

    for column in REQUIRED_SIGNED_EDGE_COLUMNS:
        if column not in {"sign", "weight"}:
            frame[column] = frame[column].astype(str).str.strip()
            if frame[column].eq("").any():
                raise ValueError(f"{column} values must be non-empty")
    return frame


def enumerate_signed_paths(edges: pd.DataFrame, max_paths_per_nutrient: int) -> pd.DataFrame:
    """Enumerate directed nutrient-to-disease paths and multiply their edge signs."""
    if max_paths_per_nutrient <= 0:
        raise ValueError("max_paths_per_nutrient must be positive")
    validated_edges = validate_signed_edges(edges)
    adjacency = _build_adjacency(validated_edges)
    rows: list[dict[str, object]] = []

    nutrients = sorted(validated_edges.loc[validated_edges["source_type"].eq("nutrient"), "source"].unique())
    for nutrient in nutrients:
        paths = _paths_from_nutrient(("nutrient", nutrient), adjacency)
        paths.sort(key=lambda path: (str(path[-1]["target"]), _path_string(path)))
        rows.extend(_path_row(path) for path in paths[:max_paths_per_nutrient])

    return pd.DataFrame(
        rows,
        columns=[
            "nutrient",
            "disease",
            "path",
            "path_sign",
            "path_weight",
            "n_hops",
            "relations",
            "evidence_sources",
        ],
    )


def score_signed_paths(
    beta: pd.DataFrame,
    paths: pd.DataFrame,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score signed nutrient-to-disease paths for each sample."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    required = {"nutrient", "disease", "path", "path_sign", "path_weight"}
    missing = required.difference(paths.columns)
    if missing:
        raise ValueError(f"signed paths missing required columns: {sorted(missing)}")

    beta_frame = beta.astype(float).copy()
    beta_frame.index = beta_frame.index.astype(str)
    beta_frame.columns = beta_frame.columns.astype(str)
    rows: list[dict[str, object]] = []
    for path in paths.to_dict("records"):
        nutrient = str(path["nutrient"])
        if nutrient not in beta_frame.columns:
            continue
        signed_weight = int(path["path_sign"]) * float(path["path_weight"])
        values = beta_frame[nutrient]
        for sample_id, beta_value in values.items():
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "nutrient": nutrient,
                    "disease": str(path["disease"]),
                    "path": str(path["path"]),
                    "path_sign": int(path["path_sign"]),
                    "path_weight": float(path["path_weight"]),
                    "beta_value": float(beta_value),
                    "path_score": float(beta_value) * signed_weight,
                }
            )
    detail = pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "nutrient",
            "disease",
            "path",
            "path_sign",
            "path_weight",
            "beta_value",
            "path_score",
        ],
    )
    if detail.empty:
        return pd.DataFrame(columns=["sample_id", "disease", "evidence_score", "n_paths"]), detail
    ranked = detail.assign(_magnitude=detail["path_score"].abs()).sort_values(
        ["sample_id", "disease", "_magnitude", "path"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    selected = ranked.groupby(["sample_id", "disease"], group_keys=False).head(top_n)
    scores = (
        selected.groupby(["sample_id", "disease"], as_index=False)
        .agg(evidence_score=("path_score", "sum"), n_paths=("path_score", "size"))
        .sort_values(["sample_id", "disease"], kind="mergesort")
        .reset_index(drop=True)
    )
    return scores, detail


def _build_adjacency(edges: pd.DataFrame) -> dict[tuple[str, str], list[dict[str, object]]]:
    adjacency: dict[tuple[str, str], list[dict[str, object]]] = {}
    for edge in edges.to_dict("records"):
        source_key = (str(edge["source_type"]), str(edge["source"]))
        adjacency.setdefault(source_key, []).append(edge)
    for edge_list in adjacency.values():
        edge_list.sort(key=lambda edge: (str(edge["target"]), str(edge["relation"]), str(edge["evidence_source"])))
    return adjacency


def _paths_from_nutrient(
    nutrient: tuple[str, str],
    adjacency: dict[tuple[str, str], list[dict[str, object]]],
) -> list[list[dict[str, object]]]:
    found: list[list[dict[str, object]]] = []

    def visit(node: tuple[str, str], path: list[dict[str, object]], visited: set[tuple[str, str]]) -> None:
        for edge in adjacency.get(node, []):
            target = (str(edge["target_type"]), str(edge["target"]))
            if target in visited:
                continue
            next_path = [*path, edge]
            if edge["target_type"] == "disease":
                found.append(next_path)
            else:
                visit(target, next_path, visited | {target})

    visit(nutrient, [], {nutrient})
    return found


def _path_row(path: Iterable[dict[str, object]]) -> dict[str, object]:
    edges = list(path)
    sign = 1
    weight = 1.0
    for edge in edges:
        sign *= int(edge["sign"])
        weight *= float(edge["weight"])
    return {
        "nutrient": str(edges[0]["source"]),
        "disease": str(edges[-1]["target"]),
        "path": _path_string(edges),
        "path_sign": sign,
        "path_weight": weight,
        "n_hops": len(edges),
        "relations": ";".join(str(edge["relation"]) for edge in edges),
        "evidence_sources": _merge_semicolon_values(edge["evidence_source"] for edge in edges),
    }


def _path_string(path: Iterable[dict[str, object]]) -> str:
    edges = list(path)
    return "->".join([str(edges[0]["source"]), *(str(edge["target"]) for edge in edges)])


def _merge_semicolon_values(values: Iterable[object]) -> str:
    unique: list[str] = []
    for value in values:
        for item in str(value).split(";"):
            clean = item.strip()
            if clean and clean not in unique:
                unique.append(clean)
    return ";".join(unique)
