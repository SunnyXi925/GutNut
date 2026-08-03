# GMNPS Nature Food 投稿优化计划与多 Agent 执行设计

生成日期：2026-08-03

## 1. 总体判断

当前 GMNPS 稿件已经适合作为高级预审稿，但不建议直接投 Nature Food。现有证据能够支撑：

- GMNPS 是 anchored nutrient-profiling transform，而不是替代 NPS 的新临床推荐模型。
- Population-mean GMNPS 在 9,234 个食物上高度保留 Food Compass 2.0 baseline ranking。
- GMNPS 产生 bounded、model-attributed person-food heterogeneity。
- 六位专家两轮评审支持 nutrient-channel reporting boundary 的 content validity。
- Digital-gut-twin benchmark 支持 anchoring 在 preservation-personalization trade-off 中的设计必要性。
- GMrepo 只支持 cardiometabolic microbiome plausibility boundary。

当前证据不能支撑：

- GMNPS 已经临床验证。
- GMNPS 能预测真实 individual dietary response。
- Microbiome-derived deviations 是因果效应。
- Expert-revised mask 在现有 synthetic benchmark 中具有预测性能优势。
- GMNPS 可直接用于疾病诊断、健康预测或患者饮食处方。

推荐主线改为：

> Personalized calibration provides an auditable way to extend nutrient profiling toward precision nutrition without replacing the population food-quality prior.

推荐标题方向：

> Personalized calibration extends nutrient profiling toward precision nutrition

## 2. 使用的 scientific-agent-skills 方法框架

本轮参考并应用了本地已安装的 K-Dense scientific-agent-skills 中下列模块：

- `pdf`：抽取 Barrera-Suarez et al. 2026 综述全文。
- `literature-review`：将综述和参考文献线索转化为可复现的证据主题，而不是单篇罗列。
- `experimental-design`：把新增实验按 unit、response、nuisance factor、negative control 和 failure downgrade 设计。
- `statistical-analysis` / `statistical-power` 思路：统一 bootstrap CI、seed-level uncertainty、FDR、independent unit 定义。
- `citation-management`：后续用于补充 Barrera-Suarez 2026、GMWI2/GMHI/DI-GM、MICOM/AGORA、CAMI、SHAP/LIME、FAIR/compositional microbiome 文献。
- `nature-writing`：将稿件定位为 research manuscript + Nature 子刊 + 中英资料写作，并强制 claim-evidence-boundary 对齐。
- `nature-reviewer`：按 originality、significance、technical soundness、interdisciplinary readability 和 claim moderation 做预审风险评估。

GitHub 主分支归档下载不完整，因此本轮执行以本地已安装 skills 和项目内完整稿件/审阅报告/PDF 抽取文本为准。

## 3. 从 Barrera-Suarez 2026 提取的可用方法增量

Barrera-Suarez et al. 2026 的主要价值不是重复说明 microbiome 重要，而是为 GMNPS 提供四个补强方向：

1. Multi-omics 和 multimodal provenance：diet、microbiome、host phenotype、clinical metadata 必须透明记录。
2. Composite microbiome health metrics：GMWI2/GMHI/hiPCA 说明高维 microbiome 可压缩为 interpretable score，但 GMNPS 不能被写成诊断性 microbiome health index。
3. Diet-specific microbiome indices：DI-GM 可作为 diet-gut microbiome index 的背景对照，但不是 food-level NPS。
4. GSMM/MCMM/digital twins：支持 GMNPS 的 digital-gut-twin-inspired benchmark 和未来 mechanistic validation，但当前若未跑 GSMM，不能写成 true digital twin。
5. Trustworthy AI / benchmarking：需要 nested validation、external validation、domain shift、uncertainty、interpretability 和 transparent driver attribution。

## 4. P0 投稿前必须完成

### P0.1 数据 provenance 与 beta_i 可审计化

目标：消除 Nature Food 审稿人对 15,492 profiles、microbiome-derived weights 和数据来源的硬性质疑。

关键动作：

- 将摘要和正文中的 `15,492 microbiome profiles` 改为 `15,492 microbiome-derived calibration profiles`。
- 将 `microbiome-derived nutrient-response weights` 改为 `regularized microbiome-derived nutrient calibration weights`。
- 明确这些 profiles 不是 matched food-response cohort。
- 补写 Supplementary Methods：profile 来源、QC、metadata、beta_i 生成、bridge construction、reconstruction diagnostics、软件版本、随机种子。
- 生成 provenance audit tables：
  - `input_data_provenance`
  - `microbiome_profile_audit`
  - `beta_weight_audit`
  - `food_alignment_audit`
  - `nutrient_processing_mask_audit`
  - `source_data_reproducibility_manifest`

失败降级：

- 若 beta_i 来源无法完整审计，GMNPS 只能写成 computational calibration demonstration，不能写成 microbiome response model。

### P0.2 Mask-sensitive synthetic twin v2

目标：测试 expert-revised mask 是否在可区分场景下减少 spurious driver 或改善 variance control。

实现建议：

- 扩展 `code/src/gmnps/validation/synthetic_twin.py`，新增 `run_mask_sensitive_benchmark()`。
- 使用真实 FNDDS nutrient covariance。
- 显式建模被 v4 排除的 carbohydrate、zinc、copper、vitamin A RAE 作为 broad/proxy/noisy features。
- 输出：
  - `figure4_mask_sensitive_repeated_seeds.csv`
  - `figure4_mask_false_driver_rate.csv`

评价指标：

- personalized residual Spearman
- NPS preservation Spearman
- mask-specific delta MSE
- excluded-nutrient false-driver rate
- paired seed difference with 95% CI
- BH-FDR across scenarios

失败降级：

> Expert revision improved content validity and variance control, whereas mask-specific predictive superiority was not established in simulation.

### P0.3 Channel attribution stress test

目标：证明 MAC/LIPID reporting channels 至少在 selected food groups 中稳健。

实现建议：

- 新增 `code/src/scripts/run_channel_attribution_stress.py`。
- 输入 existing delta/MAC/LIPID matrix chunks 和 food summary。
- 输出：
  - `figure3_food_group_channel_ci.csv`
  - `figure3_channel_stress_test.csv`

评价指标：

- food-group MAC/LIPID fraction bootstrap 95% CI
- dominant-channel stability
- drop-one-nutrient sensitivity
- channel-swap negative control
- top-k high-variance food overlap

失败降级：

> Model-attributed channel decomposition was robust for selected groups but exploratory for mixed categories.

### P0.4 Individual-level top-driver interpretability

目标：让 Fig. 1/Fig. 3 的个体化偏移可解释，而不是只展示分数变化。

实现建议：

- 复用 `anchored.py` 中 `_driver_strings()` 思路，但只对 figure examples 和 top heterogeneity foods 计算，避免全量长表爆炸。
- 输出 `figure1_3_top_driver_examples.csv`。

字段：

- individual_id_hash
- food_id
- food_name
- FCS2
- GMNPS_delta
- GMNPS_score
- MAC_delta
- LIPID_delta
- top_positive_drivers
- top_negative_drivers
- driver_channel_concordance

失败降级：

- 只作为 selected-case interpretability source data，不写成真实机制或临床解释。

### P0.5 Category shift detail table

目标：主动解释 79 个 category shifts 都是阈值附近相邻移动。

输出 `figure2_category_shift_detail.csv`。

字段：

- food_id
- food_name
- food_group
- FCS2
- GMNPS_mean
- baseline_category
- GMNPS_category
- absolute_shift
- nearest_threshold
- distance_to_threshold
- adjacent_shift_flag
- dominant_channel

### P0.6 统计不确定性总表

目标：让 preservation、simulation、GMrepo、channel attribution 的不确定性集中呈现。

输出 `statistical_summary_table.csv`。

字段：

- analysis_family
- comparison
- independent_unit
- n
- metric
- estimate
- ci_low
- ci_high
- p_value
- fdr
- bootstrap_or_seed_count
- interpretation_boundary

## 5. P1 显著增强但不应阻塞 P0

1. GMrepo 主文定位强化：T2D/CAD 作为 cardiometabolic plausibility，IBD/CD/UC 作为 boundary conditions。
2. ZOE public rank concordance：只做 directionality anchor，不做 GMNPS validation。
3. CRA013939 mechanism-anchor concordance：只做 hypothesis-generating mechanism plausibility。
4. American Gut stress test：只放 Supplement，作为 noisy transportability stress test。
5. Food-group finer taxonomy sensitivity：拆分 legumes/nuts、mixed foods、grains，避免过强 food-group 叙事。

## 6. P2 图表、source data 和稿件包装

1. Fig. 1d 若是 schematic，图注必须写 schematic；若写 source-data-derived，必须用真实 source data。
2. Fig. 2 保留 NPS preservation，但将其写成 architecture verification，不写成 clinical performance。
3. Fig. 3 强调 same food, different calibration profiles；减少容易争议的 food-group 过度解释。
4. Fig. 4 以 digital-gut-twin benchmark 为主，GMrepo 做 boundary panel；若 mask-sensitive simulation 成功，替换 original vs expert identical 面板。
5. 同步 `outputs/main_figures_v4/source_data/` 与 `manuscript/nature_food_submission/source_data/`，补 SHA256、生成命令、软件版本、git commit。
6. 修正 author metadata、funding、ethics、competing interests placeholder。

## 7. 多 Agent 执行设计

### Agent A：Data Provenance and Methods Reproducibility

已执行 agent：`019fc5f4-303b-7473-ba15-365274fa494f`

职责：

- 审计 15,492 profiles、W_personalized、beta_i、food/nutrient alignment。
- 生成 provenance audit 表。
- 补 Supplementary Methods 数据来源和可复现性部分。

交付物：

- provenance audit scripts
- six supplementary audit tables
- Supplementary Methods sections 1-8 and 15-16
- revised safe wording for Methods/Data availability

### Agent B：Literature and Reference Strengthening

已执行 agent：`019fc5f4-5521-7ce1-bbd8-2c36fb15dc9b`

职责：

- 从 Barrera-Suarez 2026 和现有 bibliography 中补齐文献主题。
- 补 GMWI2/GMHI/DI-GM、GSMM/MCMM/MICOM/AGORA、benchmarking、FAIR、interpretability 文献。
- 更新 Introduction/Discussion 的文献定位。

交付物：

- literature matrix
- candidate BibTeX additions
- Introduction/Discussion citation map
- forbidden overclaim list

### Agent C：Experiment Design and Statistical Validation

已执行 agent：`019fc5f4-82da-7521-bf09-b8fd82fed3f7`

职责：

- 实现 mask-sensitive simulation。
- 实现 channel attribution stress test。
- 生成 category-shift detail table 和 statistical summary table。
- 设计 GMrepo/ZOE/CRA/American Gut 的主图/补图定位。

交付物：

- new analysis scripts
- updated source data tables
- Methods statistical paragraph
- result interpretation downgrade rules

### Agent D：Nature Food Reviewer and Narrative Strategy

已执行 agent：`019fc5f4-b1bf-7d20-8a8c-c39ec2ce17f0`

职责：

- 模拟 Nature Food reviewer。
- 将 major concerns 转化为标题、摘要、Results 小标题、图注和 claim boundary 修改。
- 审查 Fig. 1d/Fig. 4 evidence status。

交付物：

- pre-submission reviewer report
- revised title/abstract/section-heading package
- figure legend risk audit
- final claim-evidence-boundary checklist

### Agent E：Manuscript Integration and Submission Package QA

下一步建议新增。

职责：

- 合并 A-D 输出。
- 更新 `sections.tex`、`sn-article.tex`、source data README、Supplementary Methods。
- 检查引用、图号、source data、availability statements、author metadata。

交付物：

- revised Nature Food submission package
- clean source-data manifest
- preflight report

## 8. 执行顺序

### Phase 1：一票否决风险清除

1. Agent A 完成 provenance audit 和 beta_i Methods。
2. Agent D 同步降级标题、摘要、Results 小标题和 claim boundary。
3. 主线程检查所有 clinical/causal/validation 过强表述。

### Phase 2：实验辨识度增强

1. Agent C 实现 mask-sensitive simulation。
2. Agent C 实现 channel stress test、top-driver examples、category shift table。
3. Agent B 更新相关文献和 Discussion framing。

### Phase 3：投稿包整合

1. Agent E 合并正文、补充方法、source data、references。
2. 主线程运行测试、figure/source-data preflight、LaTeX 编译。
3. Agent D 做最终 mock review。

## 9. 预计投稿前最低完成标准

最低可投稿标准：

- `15,492 microbiome-derived calibration profiles` 来源和 beta_i 生成可审计。
- Supplementary Methods 能复现 scoring、alignment、preprocessing、GMrepo、simulation。
- Fig. 1d evidence status 修正。
- Mask-sensitive simulation 已完成，或完全删除 expert mask performance 优势暗示。
- Channel attribution 至少对 selected food groups 有 bootstrap/stress-test 支持。
- Category shifts 79 个有补表。
- GMrepo 弱/负结果和 CAD small-n caveat 被透明呈现。
- Source data manifest 有 checksum、生成命令和软件版本。
- 标题、摘要、Results、Discussion 全部不越过 computational proof-of-concept 边界。

## 10. 本轮执行结果（2026-08-03）

### 已落地代码与输出

- 新增 provenance audit：`code/src/scripts/build_provenance_audit_tables.py`。
- 新增 post-hoc validation assets：`code/src/scripts/build_posthoc_validation_assets.py`。
- 新增 PREDICT1-derived added-value stress test：`code/src/scripts/run_predict1_gmnps_added_value.py`。
- 扩展 synthetic twin：新增 mask-sensitive twin 和 `run_mask_sensitive_benchmark()`。
- 新增测试：mask-sensitive benchmark 能识别 original mask 的 excluded-nutrient false drivers。

### 已生成审计/验证资产

- `outputs/provenance_audit/`
  - `input_data_provenance.csv`
  - `food_alignment_audit.csv`
  - `beta_weight_audit.csv`
  - `nutrient_processing_mask_audit.csv`
  - `source_data_reproducibility_manifest.csv`
  - `provenance_audit_report.json`
- `outputs/posthoc_validation/`
  - `figure2_category_shift_detail.csv`
  - `figure3_food_group_channel_ci.csv`
  - `figure1_3_top_driver_examples.csv`
  - `figure4_mask_sensitive_repeated_seeds.csv`
  - `figure4_mask_sensitive_summary.csv`
  - `statistical_summary_table.csv`
  - `posthoc_source_data_manifest.csv`
- `outputs/predict1_added_value/`
  - `metrics.csv`
  - `fold_metrics.csv`
  - `fold_predictions.parquet`
  - `model_comparison.csv`
  - `null_controls.csv`
  - `analysis_manifest.json`

### 关键事实

- 食物成分数据链路：本地存在 USDA/FDC FNDDS 2021-2023 原始文件；当前 v4 primary scoring 使用 `N_food_nutrient_imputed.parquet`，来源为 2015-2016 至 2021-2023 FNDDS harmonised/imputed matrix，而不是单纯 2021-2023 原始矩阵。
- FCS2 对齐：9,237 个 FCS2 rows；v4 primary run 纳入 9,234 个；imputation 后缺失 `35001000`, `35002000`, `35003000`。
- 人群一致性：post-hoc report 记录 79 个 category shifts，0 个 non-adjacent shifts。
- beta 权重边界：15,492 个 microbiome-derived calibration profiles；median reconstruction R2 约 0.0897，median sparsity 0.40。必须写成 regularized calibration features，不可写成 validated nutrient-response effects。
- PREDICT1-derived added-value stress test：本地 `glucose_iAUC_2h`, `tg_6h_rise`, `c_peptide_iAUC_2h` 在 provenance 中标注为 statistically anchored synthetic fields。该分析只能作为 design/provenance stress test，不可作为真实外部验证。

### PREDICT1-derived stress test 结果解读

在 5-fold group CV（`family_id else subject_id`）中，加入 GMNPS deltas 未显示稳定预测增益：

- glucose：`G2` 相比 `B2` 的 `delta_rmse = +0.0395`，`delta_r2 = -0.000894`。
- triglyceride：`G2` 相比 `B2` 的 `delta_rmse = +0.000251`，`delta_r2 = -0.000882`。
- C-peptide：`G2` 相比 `B2` 的 `delta_rmse = +0.001081`，`delta_r2 = -0.002768`。

这不是失败，而是审稿风险的提前暴露：现有 synthetic response 不应被用于支撑“真实个体饮食反应预测增益”。Nature Food 投稿前若要保留“差异有价值”主张，需要接入真实 individual-level diet/microbiome/postprandial 或 intervention response 数据，或把主张降级为 computational feasibility with stress-tested ablations。

### beta_i 方法修正

beta_i 的主方法不再表述为 `regularized nutrient-genus bridge reconstruction`。新的主方法为：参考 GMWI2 构建 GMWI2-style gut microbiome health index `H(M_i)`，参考 DI-GM 的有益/不利证据方向思想并结合 GMMAD/L7 nutrient-genus bridge，将专家评审通过的 MAC/LIPID 双通道设计约束到 65 个营养素层面，随后以有限差分计算 `beta_i,k = (H(M_i + dose * P_k) - H(M_i)) / dose`。该 beta_i 是个体 i 的 microbiome-derived nutrient response calibration weight，不是因果营养效应。

### 计划调整

- 主文可以保留：population NPS preservation、bounded personalization、interpretable channel attribution、known-ground-truth synthetic/digital-twin stress tests。
- 主文不应保留为核心证据：PREDICT1-derived added-value 真实验证。
- 下一阶段优先级：
  1. 获取或接入真实个体饮食反应数据，重跑 added-value experiment。
  2. 将 GMMAD/GMMAD-style evidence categories 接到 channel driver examples。
  3. 写 Supplementary Methods：USDA/FDC FNDDS provenance、harmonisation/imputation boundary、beta calibration boundary、PREDICT1-derived response boundary。
  4. 更新 manuscript claim language，明确 GMNPS 是 Food Compass 2.0-anchored microbiome-informed calibration framework。
