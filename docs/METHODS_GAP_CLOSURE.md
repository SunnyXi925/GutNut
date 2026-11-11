# 方法学缺口与 arXiv 补全对照

检索日期 2026-11-08,arXiv API,检索记录 `provenance/arxiv_method_search.json`。
原则:只用文献为**本文已有的分析**提供形式化框架、术语与对照方法,不引入本文未做的实验。

---

## G1 没有方法学框架定位 → 域变系数迁移学习

**缺口**:v37 把方法描述为"经验贝叶斯 + 营养掩码",没有放进任何已有的统计框架,
审稿人因此只能判断为"把标准收缩用在营养数据上"。

**补全**:
- **arXiv:2602.17967** *Minimax optimal adaptive structured transfer learning through semi-parametric
  domain-varying coefficient model* (2026-02)。提出 DVCM:用可观测的 domain identifier 编码
  domain-relatedness,在 invariant 与 fully heterogeneous 两个极端之间插值,并给出
  **防止负迁移 (negative transfer)** 的自适应估计量与 minimax 最优性。
- **arXiv:2607.23304** *Context-Adaptive Inference: A Unified Statistical and Foundation-Model View*
  (2026-07)。把 varying-coefficient / local regression / hierarchical sharing、meta-learning、
  foundation model 的 prompting-retrieval-routing 统一为 `context c -> θ(c) -> f(x; θ(c))`。

**如何用**:
1. GutNut 正是一个 **participant-as-domain 的结构化变系数迁移估计量**:共享预测器承载
   invariant 部分,三个个人系数承载 heterogeneous 部分,营养通道掩码就是"结构"。
   这给了方法一个有文献根基的名字和位置,而不是自造术语。
2. **负迁移概念直接解释本文的实测现象**:T0 显示零餐冷启动(仅先验)RMSE 2.3917,
   比共享预测器 2.3639 **更差 0.0278**。在 DVCM 语言里这就是典型的负迁移——
   在没有目标域观测时无差别借力反而有害。本文的闭式后验更新正是"用目标域证据
   自适应地决定借多少",这一点现在可以被形式化地陈述。
3. Context-Adaptive Inference 给 Introduction 提供了把"菌群先验"与"少样本适应"放进
   同一张图的合法方式:菌群/宿主是 context,个人系数是 θ(c)。

**不能用的**:本文没有证明 minimax 最优性,不得声称最优率;只能说方法属于该类并引用其框架。

---

## G2 置换负对照没有统计形式 → 内部负对照的置换 p 值

**缺口**:审稿意见要求"做 microbe–participant link 的 permutation test",但没有说明
如何把置换分布转成合法的推断陈述。

**补全**:
- **arXiv:2303.01552** *Simultaneous Hypothesis Testing Using Internal Negative Controls*
  (2023-03, v2)。给出用内部负对照构造**置换 p 值**的充分条件,以及基于负对照的 FDR 与
  local-fdr 估计。

**如何用**:T3 按该逻辑在两个角色上同时打断菌群—个体对应关系(先验估计端的训练参与者、
先验条件化端的测试参与者),重估经验贝叶斯,得到参考分布,报告
`p_perm = (1 + #{perm >= observed}) / (1 + B)`。这把"菌群没用"从一句描述变成一个有定义的检验。

---

## G3 不确定性只报后验 SD 缩小 → 后验预测检查而非校准误差

**缺口**:v37 只报后验标准差下降 36.1/18.9/16.5%,这不是校准证据;
审稿意见要求覆盖率,但直接用 ECE 会把 epistemic 与 aleatoric 混淆。

**补全**:
- **arXiv:2112.01477** *Why Calibration Error is Wrong Given Model Uncertainty: Using Posterior
  Predictive Checks with Deep Learning*。论证在存在模型不确定性时 calibration error 及其变体
  几乎总是错误选择,应使用后验预测检查 (PPC)。

**如何用**:T8/T9 报告 **50/80/95% 后验预测区间覆盖率** 与 PPC,而不是 ECE;
并在 T9 中用已知真值分别给出先验正确与先验错误指定两种情形下的覆盖率,
把"区间是否诚实"与"先验是否正确"分开陈述。

---

## G4 参与者级 bootstrap 与个体内变异 → 异方差函数型混合效应

**缺口**:审稿人会问为什么用 participant-balanced RMSE 与参与者自助法,
以及为何只建模条件均值而不管个体内变异。

**补全**:
- **arXiv:2611.07937** *Joint Modeling of Conditional Mean and Variability in Multilevel Functional
  Data: Application to Postprandial Glucose Trajectories in the AEGIS Study* (2026-11)。
  提出异方差函数型混合效应框架,在均值与方差两部分都放参与者随机效应,并开发
  **participant-level bootstrap** 量化不确定性;发现在调整协变量后参与者之间的残差变异
  仍有实质差异。

**如何用**:
1. 为本文的参与者级自助法提供同期同领域的方法学先例。
2. 为"个体内残差变异本身是异质的"提供文献支撑,从而解释本文 responder 分析中
   `target_sd` 与通道增益的关系,并作为"只建模条件均值"的明确局限与未来工作。

---

## G5 少样本贝叶斯适应的对照方法 → ALPaCA

**缺口**:v37 引了 MAML/Grant/Harrison 但无任何实证对照。

**补全**:
- **arXiv:1807.08912** *Meta-Learning Priors for Efficient Online Bayesian Regression* (ALPaCA)。
  学习有限维特征编码与权重先验,在该特征空间内做在线贝叶斯线性回归。

**如何用**:ALPaCA 是与 GutNut 结构最接近的方法(学到的先验 + 固定特征上的闭式贝叶斯更新)。
差别在于 GutNut 的特征不是学出来的神经编码,而是**预先指定、营养学上可解释的饱和暴露坐标**;
这正是可归因性与跨食物迁移的来源。T7 的 last-layer fine-tune 基线就是这一对照的简化实现。

---

## G6 同期竞争工作缺失 → 2025–2026 的餐后血糖预测文献

**缺口**:v37 的 Related work 停在 Zeevi/Berry/Asnicar,缺同期方法。

**补全**(均需在 Introduction/Discussion 正面比较):
| arXiv | 年月 | 内容 | 与本文关系 |
|---|---|---|---|
| 2609.24453 | 2026-09 | 从餐图像 + 临床变量 + 菌群预测 PPGR,注意力建模膳食与宿主交互 | 最接近的同期工作;它关注**输入模态替代**(图像替代手工记录),本文关注**少样本个体适应与跨食物迁移**,二者正交 |
| 2605.30865 | 2026-05 | GlucoFM,双流 CGM 基础模型,477 人 109,066 小时预训练 | 代表"大规模预训练"路线;本文是"固定共享预测器 + 三参数适应"路线,对照的是参数量与可归因性 |
| 2511.03986 | 2025-11 | CGM + ML 识别代谢亚表型以指导精准营养 | 亚表型分层 vs 连续个体系数,属于互补的个体化粒度 |
| 2608.13581 | 2026-07 | 从预测到干预:LLM agent 的餐级血糖调控 | 支撑 Discussion 中"预测改善 ≠ 决策收益"的论点,并指出前瞻验证方向 |

---

## 补全后仍然存在的方法学缺口(必须写进 Limitations)

1. **无 minimax/收敛率理论**:只能引用 DVCM 框架,不能声称率最优。
2. **无保形预测 (conformal) 覆盖保证**:T8/T9 的覆盖率是模型内后验预测覆盖,
   在先验错误指定时会反保守(T9 显示 95% 名义覆盖降到 ~0.82–0.87)。
   若要分布无关保证,需要 split-conformal,本文未做。
3. **只建模条件均值**:未建模个体内方差(见 G4)。
4. **通道掩码是专家预设而非学习所得**:可识别性受限于预设坐标;
   T9 给出的是在该坐标下的可识别边界,不是最优坐标的存在性证明。
