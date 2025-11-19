from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .preprocessing import prepare_xy
from .utils import read_config, resolve_path, ensure_dir, save_json, project_root

def xgb_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    使用 XGBoost 进行分类并返回特征重要性（gain）。
    """
    cfg = read_config()
    params = cfg["feature_selection"]["xgboost"]
    model = XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        random_state=int(params["random_state"]),
        n_jobs=-1,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X.values, y.values)
    importance = model.feature_importances_
    return pd.Series(importance, index=X.columns).sort_values(ascending=False)

def bootstrap_stability(X: pd.DataFrame, y: pd.Series, n_bootstrap: int, top_k: int) -> Dict[str, float]:
    """
    进行 n 次 Bootstrap。每次采样后用 XGBoost 训练，统计进入 Top-K 的频次，得到稳定性分数（频率）。
    返回 dict: feature -> stability_score。
    """
    rng = np.random.default_rng(42)
    counts = {f: 0 for f in X.columns}
    n = X.shape[0]
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        Xi = X.iloc[idx]
        yi = y.iloc[idx]
        imp = xgb_feature_importance(Xi, yi)
        top = set(imp.head(top_k).index)
        for f in top:
            counts[f] += 1
    stability = {f: counts[f] / n_bootstrap for f in X.columns}
    return stability

def run_feature_selection(config_path: str = None, label: str = "obesity") -> Tuple[List[str], Dict[str, float]]:
    """
    针对指定标签（默认 obesity）进行特征选择，输出 Top-K 与稳定性。
    """
    cfg = read_config(config_path)
    n_bootstrap = int(cfg["feature_selection"]["n_bootstrap"])
    top_k = int(cfg["feature_selection"]["top_k"])
    X, pheno_z, _ = prepare_xy(config_path)
    y = pheno_z[label].astype(int)
    stability = bootstrap_stability(X, y, n_bootstrap=n_bootstrap, top_k=top_k)
    imp = xgb_feature_importance(X, y)
    selected = list(imp.head(top_k).index)
    out = {
        "label": label,
        "top_k": top_k,
        "selected_features": selected,
        "stability": {f: stability[f] for f in selected},
    }
    out_dir = project_root() / "outputs" / "feature_selection"
    ensure_dir(out_dir)
    save_json(out, out_dir / "selected_features.json")
    return selected, stability

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MiNPS feature selection (XGBoost + Bootstrap)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--label", type=str, default="obesity")
    args = parser.parse_args()
    selected, stability = run_feature_selection(args.config, args.label)
    print(f"Selected Top-{len(selected)} features for {args.label}: {selected[:5]}...")