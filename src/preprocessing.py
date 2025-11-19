from typing import Tuple, List

import numpy as np
import pandas as pd

from .utils import read_config, resolve_path

def aggregate_taxonomy(df: pd.DataFrame, level: str = "Species") -> pd.DataFrame:
    """
    根据指定层级（Genus/Species）聚合微生物丰度。
    输入 df 期望包含列：SubjectID, Genus, Species, Abundance。
    返回列：SubjectID, Taxon(指定层级), Abundance。
    """
    assert level in {"Genus", "Species"}
    taxon = df[level]
    agg = df.groupby(["SubjectID", taxon]).agg({"Abundance": "sum"}).reset_index()
    agg = agg.rename(columns={level: "Taxon"})
    return agg

def to_matrix(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    将长表（SubjectID, Taxon, Abundance）转为矩阵（subjects × taxa）。
    缺失填 0。
    """
    mat = agg_df.pivot_table(index="SubjectID", columns="Taxon", values="Abundance", fill_value=0.0)
    mat = mat.sort_index(axis=0).sort_index(axis=1)
    return mat

def clr_transform(matrix: pd.DataFrame, pseudo_count: float = 1e-6) -> pd.DataFrame:
    """
    在组合数据上进行 CLR 转换：CLR(x) = log(x) - mean(log(x))。
    为避免 log(0)，对所有条目加上 pseudo_count。
    """
    X = matrix.values.astype(float)
    X_pc = X + pseudo_count
    logX = np.log(X_pc)
    gm = logX.mean(axis=1, keepdims=True)
    clr = logX - gm
    return pd.DataFrame(clr, index=matrix.index, columns=matrix.columns)

def zscore_phenotypes(pheno: pd.DataFrame, phenotype_cols: List[str]) -> pd.DataFrame:
    """
    对健康表型进行 z-score 标准化。二分类示例保持 0/1 不变，这里仍返回数值型列。
    如果为连续变量，进行标准化；如果为二分类（0/1），原样返回。
    """
    out = pheno.copy()
    for c in phenotype_cols:
        vals = out[c].astype(float)
        unique = set(vals.unique())
        if unique.issubset({0.0, 1.0}):
            out[c] = vals
        else:
            mean = vals.mean()
            std = vals.std(ddof=0) if vals.std(ddof=0) > 0 else 1.0
            out[c] = (vals - mean) / std
    return out

def load_example_data(config_path: str = None, level: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载示例数据并返回：(原始长表, 聚合长表, CLR 矩阵)
    """
    cfg = read_config(config_path)
    level = level or cfg["preprocessing"]["taxonomy_level"]
    micro_fp = resolve_path(cfg["paths"]["microbiome_csv"])  # SubjectID, Genus, Species, Abundance
    df = pd.read_csv(micro_fp)
    agg = aggregate_taxonomy(df, level=level)
    mat = to_matrix(agg)
    clr = clr_transform(mat, pseudo_count=float(cfg["preprocessing"]["pseudo_count"]))
    return df, agg, clr

def prepare_xy(config_path: str = None, level: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    准备 X（CLR 矩阵）与 y（标准化表型）以及原始 phenotype 表。
    """
    cfg = read_config(config_path)
    _, _, clr = load_example_data(config_path, level)
    pheno_fp = resolve_path(cfg["paths"]["phenotypes_csv"])  # SubjectID, obesity, diabetes, ldl
    pheno = pd.read_csv(pheno_fp).set_index("SubjectID").loc[clr.index]
    pheno_z = zscore_phenotypes(pheno, phenotype_cols=list(pheno.columns))
    return clr, pheno_z, pheno

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MiNPS preprocessing")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    df, agg, clr = load_example_data(args.config)
    print(f"Raw records: {len(df)}; Aggregated: {len(agg)}")
    print(f"CLR shape: {clr.shape}")