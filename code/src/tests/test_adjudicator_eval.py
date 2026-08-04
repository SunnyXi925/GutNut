import pandas as pd

from gmnps.knowledge_graph.adjudicator_eval import class_balanced_metrics


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
