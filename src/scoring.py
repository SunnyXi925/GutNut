from typing import List, Optional

import pandas as pd

from .utils import read_config, resolve_path, project_root, ensure_dir

def filter_foods(foods: pd.DataFrame, allergies: List[str], max_gi: Optional[float]) -> pd.DataFrame:
    """
    根据过敏与 GI 阈值过滤食物。
    过敏：若 allergens 列包含任一过敏原，则剔除。
    GI：若 GI 超过阈值，则剔除。
    """
    out = foods.copy()
    if allergies:
        mask = out["allergens"].fillna("")
        out = out[~mask.str.contains("|".join(allergies), case=False, regex=True)]
    if max_gi is not None:
        out = out[out["GI"] <= max_gi]
    return out

def run_scoring(config_path: str = None, output_csv: Optional[str] = None) -> pd.DataFrame:
    """
    读取营养素权重与食物成分，计算 S=Σ ω_i N_i 得分并输出排序表。
    """
    cfg = read_config(config_path)
    foods = pd.read_csv(resolve_path(cfg["paths"]["foods_csv"]))
    weights_fp = project_root() / "outputs" / "nutrient" / "weights.csv"
    if not weights_fp.exists():
        raise FileNotFoundError("Nutrient weights not found. Please run src.nutrient_weight first.")
    weights = pd.read_csv(weights_fp, index_col=0).iloc[:, 0]
    nut_cols = [c for c in foods.columns if c.startswith("N")]
    S = foods[nut_cols].dot(weights.loc[nut_cols])
    foods["Score"] = S
    # 过滤
    filt = cfg["scoring"]["filters"]
    foods_f = filter_foods(foods, allergies=list(filt.get("allergies", [])), max_gi=filt.get("max_gi", None))
    foods_f = foods_f.sort_values(by="Score", ascending=False)
    out_dir = project_root() / "outputs" / "scores"
    ensure_dir(out_dir)
    out_fp = output_csv or (out_dir / "food_scores.csv")
    foods_f.to_csv(out_fp, index=False)
    return foods_f

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MiNPS personalized food scoring")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    df = run_scoring(args.config, args.output)
    print(df.head())