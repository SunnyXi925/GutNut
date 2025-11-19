# MiNPS: Microbiome-informed Nutrient Profiling System

MiNPS 是一个完整、可运行的微生物组驱动营养评分系统，包含从数据预处理、特征选择、SVM 建模、微生物组蒙特卡罗模拟、营养素影响参数估计到个性化食物评分的全流程代码与示例数据。

## 功能模块
- 数据预处理：合并到属/种层级、非零加常数、CLR 转换、表型 z-score 标准化
- 特征选择：XGBoost 特征重要性 + 100 次 Bootstrap 稳定性，输出 Top-20 物种与稳定性
- 模型训练：使用 BayesSearchCV 对 RBF-SVM 的 `C`、`gamma` 做贝叶斯优化，输出最佳模型和 ROC/AUC 图
- 微生物组模拟：在 CLR 空间进行 σ=0.1 的高斯扰动，裁剪到 [-5,5]，挑选风险最低候选作为 baseline
- 营养素参数：Elastic Net (alpha=0.5, l1_ratio=0.7) 建模营养素对 Δmicrobiome 的影响，输出权重与热力图
- 食物评分：根据 S=Σ ω_i N_i 为每个食物生成个性化评分，支持过敏、高 GI、特殊疾病过滤

## 项目结构
```
MiNPS/
  data/
    microbiome/
      microbiome_example.csv
      phenotypes_example.csv
    nutrient_table/
      nutrients_example.csv
    food_composition/
      food_example.csv
  notebooks/
    01_data_overview.ipynb
    02_feature_selection.ipynb
    03_model_training.ipynb
    04_microbiome_simulation.ipynb
    05_nutrient_weighting.ipynb
    06_food_scoring.ipynb
  src/
    preprocessing.py
    feature_selection.py
    model_train.py
    microbiome_simulation.py
    nutrient_weight.py
    scoring.py
    utils.py
  tests/
    test_preprocessing.py
    test_feature_selection.py
    test_scoring.py
  config.yaml
  requirements.txt
  README.md
  LICENSE
```

## 快速开始

### 1. 创建 Conda 环境
```
conda create -n minps python=3.11 -y
conda activate minps
pip install -r requirements.txt
```

### 2. 运行全流程（示例）
```
python -m src.feature_selection --config config.yaml
python -m src.model_train --config config.yaml
python -m src.microbiome_simulation --config config.yaml --subject_id 1 --label obesity
python -m src.nutrient_weight --config config.yaml
python -m src.scoring --config config.yaml --output outputs/scores/food_scores.csv
```

### 3. 打开笔记本
使用 JupyterLab 或 VS Code 打开 `notebooks/` 目录下的 `.ipynb` 文件，逐步运行各步骤。

## 示例数据说明
- `microbiome_example.csv`：模拟 30 名受试者 × 25 个物种的相对丰度数据（含属、种信息），用于构建矩阵
- `phenotypes_example.csv`：包含 `obesity`、`diabetes`、`ldl` 三个二分类亚健康指标（示例）
- `nutrients_example.csv`：模拟 63 种营养素 × 30 个受试者的摄入量矩阵
- `food_example.csv`：10 种食物 × 63 营养素组成，含 GI 和过敏原字段

## 配置说明
参见 `config.yaml`，包含数据路径、超参数、输出目录等。

## 运行产物
- `outputs/feature_selection/selected_features.json`
- `outputs/models/<label>_svm.pkl`
- `outputs/figures/<label>_roc.png`
- `outputs/simulation/<subject_id>_<label>_baseline.csv`
- `outputs/nutrient/weights.csv` 与 `outputs/figures/nutrient_weights_heatmap.png`
- `outputs/scores/food_scores.csv`

## 测试
```
python -m unittest discover -s MiNPS/tests -v
```

## 许可
MIT License