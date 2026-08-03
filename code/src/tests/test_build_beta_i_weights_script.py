import json
from pathlib import Path

import pandas as pd

from scripts.build_beta_i_weights import run


class Args:
    def __init__(self, root: Path, output_dir: Path):
        self.root = str(root)
        self.output_dir = str(output_dir)
        self.c_value = 10.0
        self.max_iter = 500
        self.random_state = 11
        self.dose = 0.1
        self.clip_abs_beta = 12.0
        self.batch_size = 2


def test_build_beta_i_weights_script_writes_expected_bundle(tmp_path):
    root = tmp_path / "repo"
    l5 = root / "data/project_data/predict_multi/L5_gmnps_pipeline_v2"
    l1 = root / "data/project_data/predict_multi/L1_microbiome/cmd_processed"
    l7 = root / "data/project_data/predict_multi/L7_nutrient_bridge"
    l8 = root / "data/project_data/predict_multi/L8_inference_bundle_v2"
    for path in [l5, l1, l7, l8]:
        path.mkdir(parents=True)
    pd.DataFrame(
        {
            "Akkermansia": [2.0, 1.8, 1.5, -1.0, -1.2, -1.4],
            "Faecalibacterium": [1.5, 1.2, 1.0, -1.1, -0.9, -1.3],
            "Escherichia": [-1.2, -1.0, -0.8, 1.4, 1.6, 1.2],
        },
        index=["s1", "s2", "s3", "s4", "s5", "s6"],
    ).to_parquet(l5 / "M_clr.parquet")
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "phenotype_label": ["Health", "Health", "Health", "IBD", "CRC", "T2D"],
            "disease": ["healthy", "healthy", "healthy", "IBD", "CRC", "T2D"],
        }
    ).to_parquet(l1 / "layer_a_master.parquet")
    pd.DataFrame(
        {
            "Akkermansia": {"Fiber, total dietary (g)": 1.0, "Total Fat (g)": -0.2},
            "Faecalibacterium": {"Fiber, total dietary (g)": 0.6, "Total Fat (g)": -0.1},
            "Escherichia": {"Fiber, total dietary (g)": -0.4, "Total Fat (g)": 0.8},
        }
    ).to_parquet(l7 / "B_nutrient_genus.parquet")
    (l8 / "nutrient_index.json").write_text(json.dumps(["Fiber, total dietary (g)", "Total Fat (g)"]))

    out_dir = tmp_path / "out"
    run(Args(root, out_dir))

    weights = pd.read_parquet(out_dir / "W_personalized.parquet")
    assert weights.shape == (6, 2)
    assert list(weights.columns) == ["Fiber, total dietary (g)", "Total Fat (g)"]
    assert (out_dir / "health_index.joblib").exists()
    assert (out_dir / "nutrient_perturbations.parquet").exists()
    assert (out_dir / "sample_beta_diagnostics.csv").exists()
    manifest = json.loads((out_dir / "beta_i_manifest.json").read_text())
    assert manifest["method"] == "GMWI2-style health-index finite-difference beta_i"
    assert manifest["n_samples"] == 6
    assert manifest["n_nutrients"] == 2
