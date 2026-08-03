import json
from pathlib import Path

import pandas as pd

from scripts.build_provenance_audit_tables import build_input_data_provenance, build_summary_report


def test_provenance_prefers_reproducible_beta_i_bundle(tmp_path):
    root = tmp_path / "repo"
    out = tmp_path / "out"
    out.mkdir()
    beta = root / "data/project_data/predict_multi/L7_nutrient_bridge_beta_i"
    beta.mkdir(parents=True)
    pd.DataFrame([[0.1]], index=["s1"], columns=["Fiber, total dietary (g)"]).to_parquet(beta / "W_personalized.parquet")
    (beta / "beta_i_manifest.json").write_text(json.dumps({"method": "GMWI2-style health-index finite-difference beta_i"}))

    required = [
        root / "data/project_data/predict_multi/L6_layer_c_fndds/raw",
        root / "data/project_data/predict_multi/L6_layer_c_fndds/processed",
        root / "data/project_data/predict_multi/L9_food_compass",
        root / "data/project_data/predict_multi/L1_microbiome/cmd_processed",
        root / "data/project_data/predict_multi/L4_knowledge_graph/gmmad2_processed",
        root / "data/project_data/predict_multi/L7_nutrient_bridge",
    ]
    for path in required:
        path.mkdir(parents=True, exist_ok=True)
    for path in [
        root / "data/project_data/predict_multi/L6_layer_c_fndds/raw/FNDDS_Nutrient_Values.xlsx",
        root / "data/project_data/predict_multi/L6_layer_c_fndds/processed/N_food_nutrient_full.parquet",
        root / "data/project_data/predict_multi/L6_layer_c_fndds/processed/N_food_nutrient_imputed.parquet",
        root / "data/project_data/predict_multi/L9_food_compass/food_compass_scores.csv",
        root / "data/project_data/predict_multi/L1_microbiome/cmd_processed/layer_a_master.parquet",
        root / "data/project_data/predict_multi/L4_knowledge_graph/gmmad2_processed/B_genus_metabolite.parquet",
        root / "data/project_data/predict_multi/L7_nutrient_bridge/W_personalized.parquet",
    ]:
        path.write_bytes(b"x")

    provenance = build_input_data_provenance(root, out)
    row = provenance.set_index("dataset_id").loc["W_personalized_beta_i"]
    assert row["source"] == "GMWI2-style health-index finite-difference beta_i"
    assert "finite-difference change" in row["notes"]
    assert "improvement" not in row["notes"]


def test_summary_report_names_beta_i_method_and_boundary(tmp_path):
    root = tmp_path / "repo"
    out = tmp_path / "out"
    out.mkdir()
    processed = root / "data/project_data/predict_multi/L6_layer_c_fndds/processed"
    beta = root / "data/project_data/predict_multi/L7_nutrient_bridge_beta_i"
    processed.mkdir(parents=True)
    beta.mkdir(parents=True)
    (processed / "layer_c_manifest.json").write_text(json.dumps({"n_foods": 1}))
    (processed / "layer_c_full_manifest.json").write_text(
        json.dumps({"releases": ["2021-2023"], "fcs_audit": {"coverage_pct": 100.0}})
    )
    (processed / "imputation_manifest.json").write_text(
        json.dumps({"fcs_audit_after_imputation": {"coverage_pct": 100.0, "still_missing": 0}})
    )
    (beta / "beta_i_manifest.json").write_text(
        json.dumps(
            {
                "method": "GMWI2-style health-index finite-difference beta_i",
                "formula": "beta_i,k = (H(M_i + dose * P_k) - H(M_i)) / dose",
                "n_samples": 2,
                "n_nutrients": 1,
                "health_model_training_summary": {"train_auc": 0.8},
                "health_label_diagnostics": {"n_excluded_contradictory_sample_ids": 1},
                "interpretation_boundary": "not a causal nutrient effect",
            }
        )
    )
    food_audit = pd.DataFrame({"included_primary_run": [True]})
    beta_audit = pd.DataFrame({"reconstruction_r2": [0.7], "sparsity": [0.2]})

    build_summary_report(root, out, food_audit, beta_audit)

    report = json.loads((out / "provenance_audit_report.json").read_text())
    assert report["beta_i_method"] == "GMWI2-style health-index finite-difference beta_i"
    assert report["beta_i_interpretation_boundary"] == "not a causal nutrient effect"
    assert report["beta_i_manifest_path"] == "data/project_data/predict_multi/L7_nutrient_bridge_beta_i/beta_i_manifest.json"
    assert report["historical_bridge_reconstruction_r2_median"] == 0.7
