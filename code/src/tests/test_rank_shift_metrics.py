import pandas as pd

from gmnps.validation.rank_shift import (
    RankShiftThresholds,
    evaluate_rank_shift_thresholds,
    individual_rank_shift_metrics,
    population_consensus_metrics,
)


def _scores():
    rows = []
    fcs = {"a": 90, "b": 80, "c": 70, "d": 30, "e": 20}
    personalized = {
        "healthy_1": {"a": 91, "b": 79, "c": 71, "d": 29, "e": 21},
        "disease_1": {"a": 30, "b": 40, "c": 65, "d": 86, "e": 92},
    }
    labels = {"healthy_1": "Health", "disease_1": "T2D"}
    for individual_id, values in personalized.items():
        for food_id, score in values.items():
            rows.append(
                {
                    "individual_id": individual_id,
                    "food_id": food_id,
                    "FCS2": fcs[food_id],
                    "GMNPS_score": score,
                    "GMNPS_delta": score - fcs[food_id],
                    "phenotype_label": labels[individual_id],
                }
            )
    return pd.DataFrame(rows)


def test_individual_rank_shift_metrics_detect_disease_reranking():
    metrics = individual_rank_shift_metrics(_scores(), top_k=2).set_index("individual_id")
    assert metrics.loc["healthy_1", "rank_spearman_vs_fcs2"] > 0.95
    assert metrics.loc["disease_1", "rank_spearman_vs_fcs2"] < -0.80
    assert metrics.loc["disease_1", "top_k_jaccard_vs_fcs2"] == 0.0
    assert metrics.loc["disease_1", "delta_span_p95_p05"] > 50


def test_population_consensus_uses_mean_personalized_scores():
    consensus = population_consensus_metrics(_scores())
    assert consensus["n_foods"] == 5
    assert -1.0 <= consensus["spearman_fcs2_gmnps_mean"] <= 1.0


def test_thresholds_are_group_aware():
    metrics = individual_rank_shift_metrics(_scores(), top_k=2)
    report = evaluate_rank_shift_thresholds(
        metrics,
        RankShiftThresholds(
            disease_max_jaccard=0.70,
            disease_max_rank_spearman=0.75,
            healthy_min_rank_spearman=0.60,
            min_delta_span=12.0,
        ),
        group_col="phenotype_label",
    )
    by_group = report.set_index("group")
    assert by_group.loc["Health", "passes_healthy_rank_preservation"]
    assert by_group.loc["T2D", "passes_disease_reranking"]
