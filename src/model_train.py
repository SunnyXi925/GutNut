from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Real
import matplotlib.pyplot as plt
from joblib import dump

from .preprocessing import prepare_xy
from .utils import read_config, project_root, ensure_dir

def train_svm_for_label(X: pd.DataFrame, y: pd.Series, cfg: Dict) -> Dict:
    """
    使用贝叶斯优化搜索 RBF-SVM 的 C 和 gamma，返回最佳模型与 ROC/AUC 信息。
    """
    space_cfg = cfg["model"]["svm_bayes_search"]["space"]
    search = BayesSearchCV(
        estimator=SVC(kernel="rbf", probability=True, class_weight="balanced"),
        search_spaces={
            "C": Real(float(space_cfg["C"]["low"]), float(space_cfg["C"]["high"]), prior="log-uniform"),
            "gamma": Real(float(space_cfg["gamma"]["low"]), float(space_cfg["gamma"]["high"]), prior="log-uniform"),
        },
        n_iter=int(cfg["model"]["svm_bayes_search"]["n_iter"]),
        cv=StratifiedKFold(n_splits=int(cfg["model"]["svm_bayes_search"]["cv_folds"]), shuffle=True, random_state=int(cfg["model"]["svm_bayes_search"]["random_state"])) ,
        n_jobs=int(cfg["model"]["svm_bayes_search"]["n_jobs"]),
        random_state=int(cfg["model"]["svm_bayes_search"]["random_state"]),
        verbose=0,
    )
    search.fit(X.values, y.values)
    best = search.best_estimator_
    proba = best.predict_proba(X.values)[:, 1]
    fpr, tpr, _ = roc_curve(y.values, proba)
    roc_auc = auc(fpr, tpr)
    return {"model": best, "fpr": fpr, "tpr": tpr, "auc": roc_auc}

def run_training(config_path: str = None) -> None:
    """
    针对多个标签分别训练模型，保存模型与 ROC 曲线。
    """
    cfg = read_config(config_path)
    X, pheno_z, _ = prepare_xy(config_path)
    labels = list(cfg["model"]["labels"])
    out_models = project_root() / "outputs" / "models"
    out_fig = project_root() / "outputs" / "figures"
    ensure_dir(out_models)
    ensure_dir(out_fig)
    for label in labels:
        y = pheno_z[label].astype(int)
        res = train_svm_for_label(X, y, cfg)
        dump(res["model"], out_models / f"{label}_svm.pkl")
        plt.figure(figsize=(6, 5))
        plt.plot(res["fpr"], res["tpr"], label=f"AUC={res['auc']:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC - {label}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_fig / f"{label}_roc.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MiNPS model training (Bayesian Optimized SVM)")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    run_training(args.config)
    print("Training done. Models and ROC figures saved.")