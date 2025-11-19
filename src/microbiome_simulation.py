from typing import Tuple

import numpy as np
import pandas as pd
from joblib import load

from .preprocessing import prepare_xy
from .utils import read_config, project_root, ensure_dir

def simulate_candidates(original_clr: np.ndarray, sigma_scale: float, n_candidates: int, clip_low: float, clip_high: float) -> np.ndarray:
    """
    在 CLR 空间进行高斯扰动，生成候选矩阵，并进行裁剪。
    """
    rng = np.random.default_rng(123)
    scale = sigma_scale * (np.abs(original_clr) + 1e-6)
    noise = rng.normal(loc=0.0, scale=scale, size=(n_candidates, original_clr.shape[0]))
    M_pert = original_clr + noise
    M_pert = np.clip(M_pert, clip_low, clip_high)
    return M_pert

def choose_baseline(M_pert: np.ndarray, model, risk_type: str = "proba") -> Tuple[np.ndarray, float, int]:
    """
    使用 SVM 模型评估风险并选择最低风险的候选作为 baseline。
    risk_type: 使用概率（二类概率）作为风险。
    返回：(baseline_vector, min_risk, idx)
    """
    if risk_type == "proba":
        risks = model.predict_proba(M_pert)[:, 1]
    else:
        risks = model.decision_function(M_pert)
    idx = int(np.argmin(risks))
    return M_pert[idx], float(risks[idx]), idx

def run_simulation(config_path: str = None, subject_id: int = 1, label: str = None) -> None:
    """
    对指定受试者的 CLR 向量进行 Monte Carlo 模拟，选择风险最低的候选。
    默认标签使用 config.simulation.target_label。
    需先训练并保存对应标签的 SVM 模型。
    """
    cfg = read_config(config_path)
    X, _, _ = prepare_xy(config_path)
    label = label or cfg["simulation"]["target_label"]
    model_fp = project_root() / "outputs" / "models" / f"{label}_svm.pkl"
    if not model_fp.exists():
        raise FileNotFoundError(f"Model not found: {model_fp}. Please run src.model_train first.")
    model = load(model_fp)
    original_clr = X.loc[subject_id].values
    M_pert = simulate_candidates(
        original_clr,
        sigma_scale=float(cfg["simulation"]["sigma_scale"]),
        n_candidates=int(cfg["simulation"]["n_candidates"]),
        clip_low=float(cfg["simulation"]["clip_low"]),
        clip_high=float(cfg["simulation"]["clip_high"]),
    )
    baseline, min_risk, idx = choose_baseline(M_pert, model, risk_type="proba")
    out_dir = project_root() / "outputs" / "simulation"
    ensure_dir(out_dir)
    out_df = pd.DataFrame(baseline.reshape(1, -1), columns=X.columns)
    out_df.insert(0, "SubjectID", subject_id)
    out_df.to_csv(out_dir / f"{subject_id}_{label}_baseline.csv", index=False)
    print(f"Chosen baseline idx={idx} with risk={min_risk:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MiNPS microbiome Monte Carlo simulation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--subject_id", type=int, default=1)
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()
    run_simulation(args.config, args.subject_id, args.label)