from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet

from .utils import read_config, resolve_path, project_root, ensure_dir

def run_nutrient_weight(config_path: str = None) -> pd.DataFrame:
    """
    使用 Elastic Net 建模营养素对 Δmicrobiome 风险的影响，输出权重并保存热力图。
    这里将目标变量设为模拟基线与原始 CLR 向量之差的 L2 范数（风险降低幅度的代理）。
    """
    cfg = read_config(config_path)
    # 读取营养素矩阵（subjects × nutrients）
    nut_fp = resolve_path(cfg["paths"]["nutrients_csv"]) 
    nutrients = pd.read_csv(nut_fp).set_index("SubjectID")
    # 读取原始 CLR 与 baseline CLR
    from .preprocessing import prepare_xy
    X, _, _ = prepare_xy(config_path)
    sim_dir = project_root() / "outputs" / "simulation"
    # 为简单起见，选择 config.simulation.target_label 的 baseline 文件，若不存在则回退为零差异
    label = cfg["simulation"]["target_label"]
    deltas = []
    for sid in X.index:
        fp = sim_dir / f"{sid}_{label}_baseline.csv"
        if fp.exists():
            bl = pd.read_csv(fp).set_index("SubjectID").loc[sid].values
            orig = X.loc[sid].values
            delta = np.linalg.norm(bl - orig)
        else:
            delta = 0.0
        deltas.append(delta)
    y = np.array(deltas)
    # 拟合 Elastic Net
    en_cfg = cfg["nutrient"]["elastic_net"]
    model = ElasticNet(alpha=float(en_cfg["alpha"]), l1_ratio=float(en_cfg["l1_ratio"]))
    model.fit(nutrients.values, y)
    weights = pd.Series(model.coef_, index=nutrients.columns, name="omega")
    out_dir = project_root() / "outputs" / "nutrient"
    ensure_dir(out_dir)
    weights.to_csv(out_dir / "weights.csv")
    plt.figure(figsize=(10, 2))
    sns.heatmap(weights.to_frame().T, cmap="viridis", cbar=True)
    plt.title("Nutrient Weights (Elastic Net)")
    plt.tight_layout()
    plt.savefig(project_root() / "outputs" / "figures" / "nutrient_weights_heatmap.png", dpi=150)
    plt.close()
    return weights.to_frame()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MiNPS nutrient weight estimation (Elastic Net)")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    run_nutrient_weight(args.config)
    print("Nutrient weights computed and saved.")