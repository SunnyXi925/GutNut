import pandas as pd

from gmnps.knowledge_graph.signed_paths import (
    enumerate_signed_paths,
    score_signed_paths,
    validate_signed_edges,
)


def test_enumerate_signed_paths_preserves_negative_risk_direction():
    edges = pd.DataFrame(
        [
            {"source": "Fiber", "target": "C00031", "source_type": "nutrient", "target_type": "kegg_compound", "relation": "maps_to", "sign": 1, "weight": 1.0, "evidence_source": "KEGG"},
            {"source": "C00031", "target": "Faecalibacterium", "source_type": "kegg_compound", "target_type": "microbe", "relation": "supports", "sign": 1, "weight": 0.8, "evidence_source": "KEGG"},
            {"source": "Faecalibacterium", "target": "butyrate", "source_type": "microbe", "target_type": "metabolite", "relation": "produces", "sign": 1, "weight": 0.9, "evidence_source": "GMMAD2"},
            {"source": "butyrate", "target": "IBD", "source_type": "metabolite", "target_type": "disease", "relation": "protective_for", "sign": -1, "weight": 0.7, "evidence_source": "GMMAD2"},
        ]
    )
    paths = enumerate_signed_paths(validate_signed_edges(edges), 10)
    assert paths.iloc[0]["path_sign"] == -1
    assert "KEGG" in paths.iloc[0]["evidence_sources"]
    assert "GMMAD2" in paths.iloc[0]["evidence_sources"]


def test_validate_signed_edges_rejects_invalid_signs():
    edges = pd.DataFrame(
        [
            {"source": "Fiber", "target": "IBD", "source_type": "nutrient", "target_type": "disease", "relation": "risk_for", "sign": 0, "weight": 1.0, "evidence_source": "GMMAD2"}
        ]
    )

    try:
        validate_signed_edges(edges)
    except ValueError as exc:
        assert "sign" in str(exc)
    else:
        raise AssertionError("invalid sign should fail validation")


def test_enumerate_signed_paths_does_not_traverse_same_id_across_node_types():
    edges = pd.DataFrame(
        [
            {"source": "Fiber", "target": "X", "source_type": "nutrient", "target_type": "kegg_compound", "relation": "maps_to", "sign": 1, "weight": 1.0, "evidence_source": "KEGG"},
            {"source": "X", "target": "IBD", "source_type": "microbe", "target_type": "disease", "relation": "causes", "sign": 1, "weight": 1.0, "evidence_source": "GMMAD2"},
        ]
    )

    paths = enumerate_signed_paths(edges, 10)

    assert paths.empty


def test_score_signed_paths_applies_path_sign_to_evidence():
    paths = pd.DataFrame(
        [
            {
                "nutrient": "Fiber",
                "disease": "IBD",
                "path": "Fiber->IBD",
                "path_sign": -1,
                "path_weight": 0.5,
            }
        ]
    )

    scores, detail = score_signed_paths(
        pd.DataFrame({"Fiber": [2.0]}, index=["sample"]),
        paths,
    )

    assert detail.iloc[0]["path_score"] == -1.0
    assert scores.iloc[0]["evidence_score"] == -1.0
