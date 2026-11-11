# 2026-11-08 批注逐条处理台账（51 条）

原文与批注见 `annotations_raw.json` / `annotation_index.csv`。
处理标记：**已改**（正文已落实）/ **已改+实验**（并补了实验证据）/ **不采纳**（给出证据理由）/ **待排版**。

---

## A. 标题与摘要（p1，2 条）

| # | 批注 | 处理 |
|---|---|---|
| 1 | 标题建议改为 `A microbiome-informed Bayesian adaptation framework for personalized nutrient profiling` | **不采纳该措辞，但同意改标题**。证据不支持 `microbiome-informed`：置换负对照 P=0.73（S8），与宿主条件化在 ±0.005 界内等价且 MDE 仅 0.00056（S6）。把未成立的机制放进标题是最大拒稿风险。改为 `Few-shot adaptation in nutrient-channel coordinates transfers across cohorts for postprandial glucose prediction`，并保留 `Bayesian adaptation framework` 的方法定位 |
| 2 | 「我们评估的全面性怎么体现一下？仅仅一个数据集的性能提升能否支撑？能否具有代表性？」 | **已改+实验**。这是本轮最大改动。新增独立队列复现（998 人，4.5%，CI 3.7–5.4）与冻结跨队列迁移（100 人，6.9%，P=0.005），摘要三个数字全部换成双队列口径；并补人群基线表（S11a/S11b）与三血糖分层一致性 |

## B. 术语具象化（p2–p5，8 条）—— 批注最集中的问题

| # | 批注 | 处理 |
|---|---|---|
| 4 | `intercept` 「能换一个更通俗易懂的词吗？截距调整？」 | **已改**。全文不再单用 `intercept`。首次出现写作 `the participant's overall response level, the amount by which their responses sit above or below the population at the same composition`，其后用 `personal response level`。公式中保留 $\widehat\beta_{i0}$ 并在紧邻句解释 |
| 24 | `intercept` 「这个词我始终没法具像化，就想到数学里的截距，是那个意思吗？」 | **已改**，同上。并在 2.5 节用它解释为什么个人水平不可能改变食物排序（它对所有食物平移同一量） |
| 6 | `accumulate meaning` 「这是什么意思？」 | **已改**。该从句整句删除 |
| 7 | `shared coordinates` 「shared space？共享子空间的说法吗？」 | **已改**。改为具体陈述：`coordinates that every food shares. Any food can be placed on a fibre axis and a fat axis, whether or not the person has eaten it` |
| 8 | `Nutrient exposures supply those coordinates` 「这句话什么意思？」 | **已改**，并入上一条的具象表述 |
| 26 | `channel exposures` 「这里的 channel 又是什么具象化的表示」 | **已改**。Introduction 首次定义：`a nutrient channel: a named nutrient axis along which a person is allowed to differ from the population`；Results 2.1 再给斜率的操作性解释 |
| 39 | `Nutrient mask` 「正文第一次出现时是不是解释一下这是什么？」 | **已改**。改为 `the nutrient mask, which is simply the list of exposures permitted to carry personal differences`，并明确其余宏量营养素仍进共享预测器、只是不进个人部分 |
| 25 | `σ_i`、`N(µ,Σ)` 「是正态分布？如果都是第一次出现都要解释，不要想当然」 | **已改**。写明 `Gaussian prior`，逐符号解释 $X_i$、$r_i$、$\sigma^2$ 的含义；并修正 $\sigma_i^2$ 为共享 $\sigma^2$，附上「三餐三系数没有剩余自由度」的理由（S10） |

## C. 方法通用性与吸引力（p3、p7，4 条）—— 对应「正文以方法吸引力为第一要求」

| # | 批注 | 处理 |
|---|---|---|
| 21 | `MAC and LIPID` 「这个仅仅是针对两个具体的案例吧？怎么说明我们方法的通用性？」 | **已改+实验**。2.1 节末明确 `The mask is a specification, not a property of the method. Any set of named exposures can occupy it`，并把「选哪一组」变成实证问题：T5 在 998 人上比较六种通道实例（S14b），结论是剂量坐标优于脂肪质量分数 |
| 22 | `u_k` 「物理意义是什么？」 | **已改**。写明 `the exposure at which the channel reaches half its maximum effect, so it sets the dose scale on which a person is allowed to differ` |
| 23 | 「数据哪里来的？为什么是 5 和 20？」 | **已改+实验**。说明取自训练数据中各营养素非零暴露的中位数附近、且事先固定；并补做数据驱动替代（每折训练内中位数），结论不变（S14c） |
| 44 | `closed-form` 「为什么要强调 closed-form？这是我们的优势吗？introduction 里也没说」 | **已改**。Introduction 新增一整段专讲闭式解的三条实际收益：新个体无需重训共享模型（三次矩阵运算）、画像自带误差棒、每餐预测可精确分解为三项供临床核查。并把方法放进结构化迁移估计量族（arXiv:2602.17967 / 2607.23304） |

## D. 写作风格：排比句与长句（p3、p6，5 条）

| # | 批注 | 处理 |
|---|---|---|
| 14 | 「后面四个组件有逻辑顺序吗？就直接这样直接倒出来了？不要用长句！」 | **已改**。原四组件排比拆成三段叙述：先说共享与个人的分工，再说个人部分为何只有三个数，最后说先验与更新 |
| 16 | 「`transfer. The` 并且这句应该放到前面，然后再具体说那几个元素，然后再说具体请看 Methods」 | **已改**。2.1 节首段先给「分工」这一总览句，再逐项展开，Methods 指引移到该节末 |
| 41 | `the framework's learning sequence` | **已改**，改为 `The learning sequence of the framework` 并最终整句删除（见下条） |
| 42 | 「最烦这种连珠炮，能否好好写作。读者不好接受这么多信息。能否换一种形式，而不是排比句」 | **已改**。该处四分句排比整句删除。全文复查：2.1、2.3、2.5 三处原排比均改为「一句一事」的短句叙述；全文不再出现 `A establishes X, B supplies Y, C specifies Z` 结构 |
| 40 | `Intercept · MAC · LIPID` 「感觉大模型生成的？三个罗列代表什么意思？」 | **已改**（图内标注）。Fig. 1 中该标签改为 `response level + fibre channel + fat channel`，并在图注写明三项之和即个人调整 |

## E. 评估表述与用词恰当性（p4、p5，4 条）

| # | 批注 | 处理 |
|---|---|---|
| 33 | `Few-shot` 「用在这个地方是否恰当？」 | **部分采纳**。标题与方法学定位保留 `few-shot`（这是该问题在计算文献中的标准名称，arXiv:1807.08912 等），但 Results 正文改用具体说法 `a short measurement history` / `three measured meals`，不在结果句里用术语 |
| 34 | `After three calibration meals` 「是不是显示说这里只选用 3？然后做实验？」 | **已改+实验**。校准餐数从「只报 3」改为完整曲线：0/1/2/3/5/7/9/11 餐（S17a）。三餐取约九成可得收益，平台位置现在可见而非假设 |
| 36 | `out-of-sample` 「这个词在这个语境下斟酌一下是否恰当」 | **已改**。改为 `later meals that were not used for calibration`，并在每处写明查询集起点 |
| 35 | 单位缺失（`mmol l⁻¹ h` 前一个数没带单位） | **已改**。全文数值首次出现均带单位，同段内重复值省略 |

## F. 图表（p21、p22，5 条）

| # | 批注 | 处理 |
|---|---|---|
| 47 | Fig. 2b「离坐标轴是不是太远了？」 | **待排版**。已记入图件修订单：b 面板纵轴标签与轴线间距收紧 |
| 48 | Fig. 2a「只做到 5 不知道做得彻底，如果 7 或 9 呢，是不是得找到那个平滑的地方」 | **已改+实验**。曲线延长到 11 餐（S17a）：2.361（0 餐）→2.143（1）→2.053（2）→1.936（3）→1.919（5）→1.891（7）→1.918（9）→1.886（11）。3 餐后进入平台，且通道增量在每个历史长度都存在（3.3%–5.5%） |
| 49 | 「这里是 b，下面那个应该标注 c 吧？」 | **待排版**。面板字母错位，已记入修订单 |
| 50 | Fig. 3c「我们最终的方法是不是也得画进去，对比明显？」 | **待排版**。消融图加入 GutNut 本身作为零参考线并标出其绝对值 |
| 51 | Fig. 3f「我们所有先验加到里面最终的结果是不是也放进去」 | **待排版+实验**。四个先验臂（群体/宿主/菌群/宿主+菌群）已全部跑出且互不混杂（S6、S7），图 3f 改为四臂并列并标出等价界 ±0.005 |

## G. 语言细节（p2–p6，共 20 条）

以下按批注原样落实，不逐条展开：
`which`→`where`（11）；`These include`→`[20], such as`（9）；`Some works integrate ...` 改写（3）；
`an information asymmetry` 冠词与搭配（13）；`three`→与前文四要素一致（15）；
`relationships. Personal adaptation is ...` 断句与时态（17、18）；
$\beta_{i0}$、$\beta_{ik}$ 加 hat（19、20）；`the nutritional meanings of channels` 语序（28）；
时态统一为一般现在时并通篇检查（30、31、32）；`ants. then`→`and then`（32）；
`. Most`（37）；`which jointly support`（43）；`acting`/`allowing` 分词（45、46）；
其余 5 条为同义重复的高亮，已随对应句改写一并处理。

## H. 其他

| # | 批注 | 处理 |
|---|---|---|
| 12 | 建议的 Introduction 末段改写（`This turns sparse metabolic measurements into ...`） | **部分采纳**。采纳其断句方式，但删去 `scalable computational foundation for precision health`：无成本与可及性证据；也删去 `personal food assessment`：排序为阴性（S16a） |
| 45 | `deployment` 「这里都部署了？有必要提供一个网页吗？让大家输入一个 food 展示结果那种？」 | **已改**，术语全部改为 `projection`；交互网页**建议暂不做**。当前证据显示个性化评分虽大幅改变前十名（重合中位 1/10），但个人内排序未改善（S16a），对外提供查询界面会让读者误以为该评分已可指导选食。建议等前瞻性选食验证后再发布 |

---

## 未采纳项小结（3 条，均附证据）

1. 标题中的 `microbiome-informed`：置换 P=0.73、与宿主等价（MDE 为主效应 4%）。
2. `Few-shot` 全面替换：保留为方法学术语，仅在结果句改用具体说法。
3. 食物查询网页：个人内排序为阴性，发布界面会超出证据。
