import pandas as pd
import pytest

from gmnps.knowledge_graph.label_adjudication import (
    adjudicate_labels_rule_ai,
    build_adjudication_prompt_table,
    label_metrics,
)


def test_rule_ai_adjudication_blocks_label_leakage_and_scores_hits():
    path_scores = pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2", "s2"],
            "disease": ["Health", "T2D", "Health", "T2D"],
            "evidence_score": [4.0, 0.5, 0.2, 3.0],
            "top_paths": ["Fiber->C00031->Bifidobacterium->butyrate->Health", "fat->T2D", "Fiber->Health", "fat->T2D"],
        }
    )
    prompt = build_adjudication_prompt_table(path_scores, leakage_columns=[])
    pred = adjudicate_labels_rule_ai(prompt, ["Health", "T2D"])
    truth = pd.DataFrame({"sample_id": ["s1", "s2"], "true_label": ["Health", "T2D"]})
    metrics = label_metrics(pred, truth)

    assert pred.set_index("sample_id").loc["s1", "predicted_label"] == "Health"
    assert pred.set_index("sample_id").loc["s2", "predicted_label"] == "T2D"
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["majority_baseline_accuracy"] == 0.5
    assert metrics["random_baseline_expected_accuracy"] == 0.5
    assert metrics["no_kg_baseline_accuracy"] == metrics["majority_baseline_accuracy"]
    assert metrics["rule_only_baseline_accuracy"] == metrics["accuracy"]

    leaked = path_scores.assign(true_label=["Health", "Health", "T2D", "T2D"])
    with pytest.raises(ValueError, match="leakage"):
        build_adjudication_prompt_table(leaked, leakage_columns=["true_label"])
