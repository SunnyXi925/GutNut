import pandas as pd

from gmnps.knowledge_graph.evidence_graph import load_evidence_edges, score_nutrient_disease_paths


def test_score_nutrient_disease_paths_uses_kegg_and_gmmad2_edges(tmp_path):
    edges = pd.DataFrame(
        [
            {
                "source": "Fiber",
                "target": "C00031",
                "source_type": "nutrient",
                "target_type": "kegg_compound",
                "relation": "maps_to",
                "weight": 1.0,
                "evidence_source": "KEGG",
            },
            {
                "source": "C00031",
                "target": "Bifidobacterium",
                "source_type": "kegg_compound",
                "target_type": "microbe",
                "relation": "supports",
                "weight": 0.8,
                "evidence_source": "KEGG",
            },
            {
                "source": "Bifidobacterium",
                "target": "butyrate",
                "source_type": "microbe",
                "target_type": "metabolite",
                "relation": "produces",
                "weight": 0.7,
                "evidence_source": "GMMAD2",
            },
            {
                "source": "butyrate",
                "target": "Health",
                "source_type": "metabolite",
                "target_type": "disease",
                "relation": "protective_for",
                "weight": 0.9,
                "evidence_source": "GMMAD2",
            },
            {
                "source": "Saturated fat",
                "target": "T2D",
                "source_type": "nutrient",
                "target_type": "disease",
                "relation": "risk_for",
                "weight": 0.6,
                "evidence_source": "GMMAD2",
            },
        ]
    )
    path = tmp_path / "edges.csv"
    edges.to_csv(path, index=False)
    loaded = load_evidence_edges([path])
    beta = pd.DataFrame({"Fiber": [2.0], "Saturated fat": [1.0]}, index=["s1"])

    scores, paths = score_nutrient_disease_paths(beta, loaded, top_n=3)

    assert scores.set_index(["sample_id", "disease"]).loc[("s1", "Health"), "evidence_score"] > 0
    assert scores.set_index(["sample_id", "disease"]).loc[("s1", "T2D"), "evidence_score"] > 0
    assert {"KEGG", "GMMAD2"}.issubset(set(paths["evidence_sources"].str.split(";").explode()))


def test_load_evidence_edges_requires_evidence_source(tmp_path):
    path = tmp_path / "bad_edges.csv"
    pd.DataFrame(
        [
            {
                "source": "Fiber",
                "target": "Health",
                "source_type": "nutrient",
                "target_type": "disease",
                "relation": "protective_for",
                "weight": 1.0,
            }
        ]
    ).to_csv(path, index=False)

    try:
        load_evidence_edges([path])
    except ValueError as exc:
        assert "evidence_source" in str(exc)
    else:
        raise AssertionError("missing evidence_source column should fail validation")
