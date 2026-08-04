import pandas as pd

from gmnps.knowledge_graph.adjudicator_eval import (
    adjudicate_signed_evidence_labels,
    class_balanced_metrics,
)


def test_class_balanced_metrics_penalizes_single_class_collapse():
    truth = pd.DataFrame({"sample_id": ["a", "b", "c"], "true_label": ["Health", "IBD", "CRC"]})
    pred = pd.DataFrame({"sample_id": ["a", "b", "c"], "predicted_label": ["CRC", "CRC", "CRC"]})
    metrics = class_balanced_metrics(pred, truth)
    assert metrics["accuracy"] == 1 / 3
    assert metrics["balanced_accuracy"] == 1 / 3
    assert metrics["single_class_collapse"] is True


def test_class_balanced_metrics_detects_non_collapsed_predictions():
    truth = pd.DataFrame({"sample_id": ["a", "b", "c", "d"], "true_label": ["Health", "IBD", "CRC", "CRC"]})
    pred = pd.DataFrame({"sample_id": ["a", "b", "c", "d"], "predicted_label": ["Health", "IBD", "CRC", "IBD"]})

    metrics = class_balanced_metrics(pred, truth)

    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 5 / 6
    assert metrics["single_class_collapse"] is False


def test_signed_evidence_adjudicator_is_named_as_deterministic_not_ai():
    scores = pd.DataFrame(
        {
            "sample_id": ["a", "a"],
            "disease": ["Health", "IBD"],
            "evidence_score": [0.2, 0.7],
        }
    )

    predictions = adjudicate_signed_evidence_labels(scores, ["Health", "IBD"])

    assert predictions.loc[0, "predicted_label"] == "IBD"
    assert predictions.loc[0, "adjudicator"] == "deterministic_signed_kg_evidence"
