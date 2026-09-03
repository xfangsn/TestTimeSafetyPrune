# Plan: 拒绝翻转中的 Neuron 归因与消融实验

> ⚠️ **历史存档(标注于 2026-08-27):本文记录较早阶段的计划/方法/结果,可能与当前代码与结论脱节。**
> 方法线现已定名 **BLADE** 并引入 best-first ELS(有效层选择);当前定稿的方法与实验结果请以
> [`blade-method.md`](blade-method.md)、[`blade-algorithm.md`](blade-algorithm.md)、
> [`blade-experiment-report.md`](blade-experiment-report.md) 为准。**下文内容保留原貌,未作修订。**

> 状态：已执行；N1–N3、N6–N7 完成，结果见 docs/results-neuron-attribution.md。
> 前置：拒绝方向实验已完成（见 `docs/results-refusal-steering.md`），mean-diff 方向向量在
> `data/directions/refusal_llama32_3b_instruct.pt`，最佳约束选参 L8 α=−2（test 服从率 33%），
> 有效层窗口 L8–L18。

## 1. 目标与假设

核心问题：activation steering 翻转拒绝行为时，是哪些 neuron 的激活被显著改变了？
它们是否构成拒绝行为的关键节点——即**不做 steering、直接移除这些 neuron，模型是否也会降低拒绝率**？

具体步骤（对应你的想法）：

1. 对模型会拒绝的有害指令，用 steering 翻转其输出（拒绝 → 服从），逐条记录最小翻转强度 α*；
2. 对每条成功翻转的样本，teacher-forcing 同一 completion，分别在有/无 steering 下前向，
   对每层 neuron 激活做 softmax 得到分布 p1（无 steering）与 p2（有 steering）；
3. 以 Σ_samples |p1 − p2| 作为每个 neuron 的 importance，找出突出 neuron 及其层分布；
4. 消融 top-k 重要 neuron（与随机 neuron 对照），测不 steering 时的拒绝率变化。

附带检验你的猜想：**α* 越小（越容易翻转）的样本，p1/p2 差异是否越小**（Spearman 相关）。

## 2. 关键设计决策

- **Neuron 定义**：MLP 中间层 post-activation（SwiGLU 后）神经元为主
  （Llama-3.2-3B：28 层 × 8192 = 229,376 个）；同时记录残差流 channel（28×3072）作为辅助视图。
- **p 的计算**：对每层，在回复 token 区间取 neuron 激活均值 → 层内向量 x → `p = softmax(x)`。
  softmax 对激活尺度敏感（混合正负值的 softmax 是人为构造的分布），因此同时计算两个
  稳健性变体：L1 归一化的 |x|、以及 z-score 后的 |Δ|。主报告用 softmax 版，结论以三个
  变体一致为准。
- **对照的干净性**：p1/p2 用**同一条 completion**（steering 下生成的服从回复）做 teacher
  forcing，两次前向唯一差别是 steering 注入与否——排除输出文本不同带来的混淆。
- **steering 层固定 L8**（约束下最优层），neuron 测量覆盖**所有层**（L0–L27，
  含 steering 层上游作对照——上游层理论上应几乎无变化，可作为阴性对照验证管线正确性）。
- **数据分工**（避免泄漏）：neuron 发现在 harmful_train（256 条）上做；消融强度 k 在
  harmful_val（64 条）上选；最终消融效果在 harmful_test（200 条）上报告一次。
- **翻转判定**：沿用关键词 judge；α* 用二分搜索（|α| ∈ [0, 12]，容差 0.25，上限内未翻转
  的样本标记为 unflippable 并统计占比）。

## 3. 里程碑

- **N0 捕获设施**：扩展 `hooks.py`，支持按层捕获 MLP post-act 激活与残差流
  （teacher-forced，单样本或小 batch，记录 response token 区间）。测试：形状/层数正确；
  无 steering 两次前向结果 bit 级一致。
- **N1 翻转 α\* 搜索**（`scripts/find_flip_alpha.py`）：harmful_train 256 条在 L8 二分搜索
  α*；输出分布统计（均值/中位数/直方图 PNG）、unflippable 占比。结果存
  `results/flip_alpha.json`。
- **N2 neuron 差异与 importance**（`scripts/neuron_attribution.py`）：对成功翻转样本
  （预期 200+ 条）计算 p1/p2，累加 importance；输出：
  - α* vs Σ|p1−p2| 的 Spearman 相关（检验猜想）；
  - 每层 importance 总量曲线 + top-50 neuron 的层分布直方图；
  - 集中度统计（top-1% neuron 占总 importance 比例、Gini 系数）——回答"有没有特别
    突出的 neuron"；
  - 上游层（L0–L7）importance ≈ 0 的阴性对照检查。
  结果存 `results/neuron_importance.json` + 图。
- **N3 消融实验**（`scripts/ablate_neurons.py`）：hook 将 top-k 重要 neuron 的 MLP 中间
  激活置 0（等价于移除该 neuron 的贡献），**不做 steering**：
  - k ∈ {16, 64, 256, 1024, 4096}，另有同数量随机 neuron 对照（seed 固定，3 个随机种子取均值）；
  - 指标：harmful_val 拒绝率、harmless 拒绝率（过度拒绝副作用）、wikitext ppl；
  - val 上选出最佳 k（拒绝率下降幅度 − ppl/过度拒绝代价的权衡），在 harmful_test 上跑
    最终数字（含随机对照）。
  结果存 `results/neuron_ablation.json` + 曲线图。
- **N4 小结**：`docs/results-neuron-attribution.md`：α* 分布、猜想检验结果、突出 neuron
  层分布、消融曲线、结论（这些 neuron 是否为拒绝行为的关键节点）、复现命令。

## 3.3 N7 修订：逐层局部归因 + 全局 pruning（已完成）

修订理由（用户反馈）：importance 计算必须逐层局部（注入 block l 输入、只测 block l 的
neuron），但 pruning 不必局限单层——各层分别在各自的注入点下算出 importance 后，
可以汇成全局分数表做**全局 top-k pruning**。N6b 的"只在 layer 9 内消融"是不必要的
限制，予以修正。

- **N7a 全层局部归因**（扩展 `scripts/neuron_attr_local.py`）：
  - 注入层 l 扫遍 **1–27**（l=0 无上游 block 输出可注入，跳过；每个 l 用 v̂_{l−1}
    注入 block l 输入，只测 block l 的 8192 个 MLP neuron）；
  - 每层独立：α=−2 固定，harmful_train 256 条生成 + judge 得该层翻转子集，
    对子集样本 teacher-forcing 双前向算 p1/p2；
  - D1[n] = Σ_flipped |p1−p2|，D2[n] = Σ_flipped log p1（同 N6a 定义）；
  - 翻转子集 <30 的层（预计深层）标记为低置信；作为稳健性，这些层补一个 α=−4
    变体重算并对比；
  - 同时计算 **actdiff 选择信号**（原 N5 设计，无需注入：harmful_train vs harmless
    指令下 MLP 激活的 t 统计量，天然覆盖全层）作为第二套全局分数，回答文献差异
    问题（arXiv:2406.14144 式选择是否更好）；
  - 存 `data/neuron_importance_local_alllayers.pt` + `results/n7_attribution.json`。
- **N7b 全局 pruning**（复用消融设施，选择改为跨层全局 top-k）：
  - 三套全局分数：D1-local、D2-local、actdiff；k ∈ {64, 256, 1024, 4096, 16384,
    65536}（0.03% → 28%，覆盖"小而关键"到"大而关键"整个谱系）；
  - 随机对照 3 种子；不 steering；指标：harmful_val 拒绝率、harmless 拒绝率、
    wikitext ppl；
  - 判定（沿用并合并 N3/N5 规则）：
    - 某信号在某 k 窗口拒绝率→近 0、超随机 >10pp 且 ppl ≤25% → 关键集合存在
      （小 k 成立="小而关键"，仅大 k 成立="大而关键"）；
    - 所有信号在所有 k 上拒绝率与 ppl 同步恶化、无优势窗口 → 拒绝与通用能力
      在神经元粒度上纠缠，神经元层面无"关键集合"，结论坐实方向层面解释；
  - 最佳配置 harmful_test 跑一次；结果存 `results/n7_global_pruning.json` + 曲线图。
- 结论与 N3（跨层 steering 归因）、N6b（单层）对比讨论，更新
  `docs/results-neuron-attribution.md`。

预计开销：N7a 约 27 层 × (256 生成 + ~2×翻转子集前向)；N7b 3 信号 × 6 k ×
(val 生成 + ppl) + 对照。GPU 预计 1–1.5 小时。

## 3.2 N6 修订设计：注入层局部归因（已完成；同层消融部分被 N7 取代）

修订理由（用户反馈）：注入在哪一层，就只应计算那一层 neuron 的 importance；
下游层的变化只是效应传播，不是干预的作用点。固定 α=2 注入（不再用逐样本 α*）。

技术前提（关键）：原约定「在 block l 输出注入」时，block l 的 MLP 在注入点之前已执行，
同层 MLP 神经元在两次前向中 bit 级相同（N2 已实测证实）。因此本修订统一定义
**injection layer l := 将方向向量注入 block l 的输入**（即 block l−1 的输出处，
使用方向向量 v̂_{l−1}），测量对象 = **layer l 的 8192 个 MLP post-SwiGLU 神经元**——
它们是第一个消费被注入残差流的神经元，p1/p2 在该层非零且闭环。

- **N6a 注入层归因**（`scripts/neuron_attr_local.py`）：
  - 主配置 l=9（对应已验证的最佳注入点 L8 输出），另扫 l ∈ {10, 12, 14, 16, 18}
    回答"各注入层的重要性结构有何差异"；
  - 固定 α=−2（raw unit）。对 harmful_train 256 条：α=0 生成确认拒绝，
    α=−2 生成 + judge 判定翻转子集（预计约 40%，即 α*≤2 的部分）；
  - 对**成功翻转**的样本：teacher-forcing 其 α=−2 completion，两次前向
    （无注入 / 注入），取 layer l MLP 激活在回复 token 区间均值 x1, x2，
    p1=softmax(x1)，p2=softmax(x2)；
  - 每个 neuron 两个 importance 指标：
    - `D1[n] = Σ_flipped |p1[n] − p2[n]|`（被干预改变的程度）；
    - `D2[n] = Σ_flipped log p1[n]`（新指标：翻转样本中拒绝状态下的特征性激活——
      在拒绝前向里持续高概率的 neuron 得分高）；
  - 分析：D1 与 D2 的相关性；各指标的集中度（top-1% 份额、Gini）；
    各注入层横向对比。结果存 `results/neuron_attr_local.json` + 图，
    重要性矩阵存 `data/neuron_importance_local.pt`。
- **N6b 同层消融**（复用 `ablate_neurons.py`，选择改为单层内）：
  - 对主配置 l=9，分别按 D1 / D2 取 layer-l 内 top-k，k ∈ {16, 64, 256, 1024}
    （单层共 8192 个 neuron），随机对照 3 种子；
  - 不 steering，测 harmful_val 拒绝率 + harmless 拒绝率 + wikitext ppl；
  - 判定：top-k 拒绝率下降超随机 >10pp 且 ppl 代价可接受 → 该层存在关键 neuron；
  - 最佳配置 harmful_test 跑一次。结果存 `results/neuron_ablation_local.json`。
- 与 N3 的跨层选择做对比讨论，结论追加进 `docs/results-neuron-attribution.md`。

## 3.1 N5 扩展（已被 N7 吸收）：两种选择信号 × 大 k 剂量-反应曲线

动机：N3 否定了"小而关键"，但 k 只扫到 4096（1.8%）；k=4096 处 top-k 与随机的拒绝率
差距达 76.6pp，说明所选集合对拒绝行为有因果特异性。"大而关键"假说（需要一个较大但
特异的集合）尚未被检验。同时回答文献差异问题（Finding Safety Neurons, arXiv:2406.14144
等用「有害/无害激活差异」选 neuron，不经 steering）——是否选择信号不同导致结论不同。
（注：暂缓执行，待 N6 结果出来后再决定 N5 是否仍按原设计进行，或并入 N6 的同层框架。）

设计：

- **两种 neuron 选择信号**：
  - `steer-rankagg`：沿用 N2 的 steering 前后变化（rank-aggregated，已完成，直接复用）；
  - `actdiff`（文献式）：不经 steering，直接用模型在 harmful_train vs harmless 指令上的
    MLP post-act 激活差异选 neuron——对每个 neuron 算两组激活（指令区间均值）的
    t 统计量（或 |mean 差|/pooled std），全局 top-k。新脚本
    `scripts/select_neurons_actdiff.py`。
- **扩展 k 网格**：k ∈ {8192, 16384, 32768, 65536}（约 3.5%/7%/14%/28%），
  与 N3 已有的小 k 数据拼成完整剂量-反应曲线；两种选择信号 × 每个 k × 随机对照
  （3 种子均值）。指标同 N3：harmful_val 拒绝率、harmless 拒绝率、wikitext ppl。
- **判定标准**：
  - "大而关键"成立 ⟺ 存在某个 k 窗口使 top-k 拒绝率 → 接近 0、与随机差距 >10pp，
    且 ppl 退化 ≤25%（比 N3 的 5% 放宽——大 k 下 5% 不现实，但需明确记录）；
  - 若 top-k 的拒绝率与 ppl 始终同步恶化、无优势窗口 → 拒绝行为与通用能力在这些
    neuron 上纠缠，"关键"不成立；
  - 两种选择信号分别判定，比较谁更优（回答文献差异问题）。
- **泄漏控制**：选择信号只用 harmful_train + harmless；网格扫描在 harmful_val；
  最终结论配置在 harmful_test 跑一次。
- 结果存 `results/n5_dose_response.json` + `results/n5_dose_response.png`
  （拒绝率 vs k 与 ppl vs k 双面板，top-k vs 随机）；结论追加进
  `docs/results-neuron-attribution.md` 新一节。

预计开销：actdiff 选择需 ~600 次前向；消融网格约 2 信号 × 4 k × (64 val 生成 + ppl)
+ 随机对照 ≈ 1 小时 GPU 内。

## 4. 验收标准

- N1：≥60% 的有害训练指令可在 |α|≤12 内翻转（达不到则换 steering 层 L14 重试并记录）。
- N2：阴性对照成立（L0–L7 importance 接近 0）；importance 集中度有定量结论
  （无论结论是"存在少数关键 neuron"还是"弥散分布"，都是有效结果）。
- N3：结论性判定标准 = top-k 消融降低拒绝率的效果**显著强于**同数量随机消融
  （差距 >10pp 视为显著）；同时报告 ppl 与过度拒绝代价。
- N5：大 k 网格与 actdiff 选择信号已并入 N7（§3.3），不再单独执行。
- N6：固定 α=−2、同层注入同层测量；判定 = 同层 top-k（D1 或 D2）消融降拒绝率
  超随机 >10pp 且 ppl 代价可接受。
- N7：判定规则见 §3.3——三套全局分数（D1-local / D2-local / actdiff）× 全谱系 k
  网格，按"k 窗口内拒绝率近 0、超随机 >10pp、ppl ≤25%"判定，并区分"小而关键"与
  "大而关键"两种成立形态。

## 5. 风险与备注

- 关键词 judge 的翻转判定有噪声 → 二分搜索边界样本可能误标；对 α* 边界样本（翻转与
  未翻转差一档）人工抽查 ~10 条。
- 逐样本二分搜索约 256 × ~6 次生成 ≈ 1500 次生成，3B 模型分钟级；N2 每样本 2 次
  teacher-forced 前向，总计 ~500 次前向，均在预算内。
- softmax 分布的人为性：结论以三个归一化变体一致为准，正文中明确说明该局限。
- 若消融有效但代价大（ppl 暴涨），如实报告 trade-off，不强行下"关键 neuron"结论。
- 所有数据/结果不进 git；不 commit。

## 6. 预计开销

GPU 时间全程预计 30–60 分钟（3B bf16，RTX 5090）。
