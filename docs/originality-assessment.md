# 原创性独立评估：signed actdiff edge 与 Route B ratio

> ⚠️ **历史存档(标注于 2026-08-27):本文记录较早阶段的计划/方法/结果,可能与当前代码与结论脱节。**
> 方法线现已定名 **BLADE** 并引入 best-first ELS(有效层选择);当前定稿的方法与实验结果请以
> [`blade-method.md`](blade-method.md)、[`blade-algorithm.md`](blade-algorithm.md)、
> [`blade-experiment-report.md`](blade-experiment-report.md) 为准。**下文内容保留原貌,未作修订。**

> 日期：2026-08-23。性质：对 codex 已有审计
> （`method-and-related-work-gradient-free-signed-edge.md`、
> `comparison-wei2024-safety-alignment-pruning.md`）的独立复核 + 独立文献检索。
> 调查方式：代码-文档一致性核对 + 独立 Web 文献检索（不复用 codex 的检索结论）。

## 1. 对 codex 报告的核查结论

codex 的两份审计文档质量高、口径克制，核心结论（"组合型创新，不能宣称首次"）经
独立复核**成立**。发现的四处问题：

1. **作者归属错误（已在原文档加勘误）**：codex 把最关键的先验工作
   [arXiv:2604.09544](https://arxiv.org/abs/2604.09544) 记为 "Orgad et al. 2026"。
   核实原文：作者为 **Boyi Wei, Kaden Zheng, Martin Wattenberg, Peter Henderson,
   Seraphina Goldfarb-Tarrant, Yonatan Belinkov**（Kempner/Harvard/Princeton/
   Cohere/Technion），Hadas Orgad 只是 arXiv 提交者；标题也记错了
   （正确：*Generate Harmful Content Using a Distinct, Unified Mechanism*）。
   **这不是小错**：第一作者 Boyi Wei 正是 Wei et al. ICML 2024（我们比较的
   brittleness 论文）的第一作者——signed-SNIP 安全剪枝是**同一课题组的后续工作**，
   该领域被占用的程度比 codex 文档传达的更强。
2. **文献遗漏**：codex 的 17 篇检索未覆盖以下直接相关的工作（见 §3）。
3. **内部不一致**：Wei 对比文档 §4.1 称 Wei 的选择是"每个输出行内 top-q"，
   §13.4 与代码实现是 per-matrix；以代码为准。
4. 小数字不一致：taylor@0.01% 的 refusal 在两份文档中分别为 0.0156 与 0.016
   （不影响结论）。

## 2. 方法要素拆解与先验对照（独立检索新增部分）

### signed actdiff edge：`s_ij = max(r_i · W_ij · (μ^H_j − μ^U_j), 0)`

| 要素 | 已有先例 |
|---|---|
| refusal direction（difference-in-means） | Arditi et al. 2024 (2406.11717) |
| W×激活的 scalar 剪枝分数、forward-only | Wanda (2306.11695)；**Antidote**（2408.09600, ICML 2025——直接用 harmful 数据的 Wanda 分数剪"有害权重"） |
| 保留符号的 weight 级安全剪枝 | **Wei et al. 2026（2604.09544）signed SNIP** |
| 对比提示对的激活差用于参数选择（gradient-free） | **TwinBreak**（2506.07596, USENIX Security 2025——harmful/harmless "twin" prompt 的激活差定位安全参数；neuron 级、无符号） |
| 方向投影式权重修改 + 对比数据 | **ProSafePrune**（ICLR 2026——对比数据 SVD 子空间投影剪枝；rank 级） |
| edge 级 refusal 定位 | **C-ΔΘ**（2602.04521——EAP-IG refusal circuit mask，需梯度） |

### Route B ratio：`max(W·E[∂margin/∂W],0) / (|W|·RMS_harmless(x)+ε)`

- signed margin Taylor 分子 ≈ Wei 2026 signed SNIP（差在 paired margin 目标）；
- benefit/cost 比值结构有先例：**SSD**（2308.07707, AAAI 2024——forget/retain
  Fisher 比值做选择性遗忘）；
- 组合未见，但审稿人很可能读作"signed SNIP 除以 Wanda"。

## 3. 独立检索结论

**未找到**同时具备以下全部要素的单一工作：scalar weight 级 × contrastive 激活均值差
× refusal direction 投影 × 保留符号决定删除侧 × 无需 backward × 实际剪枝。
这一点与 codex 的结论一致。

**但每个要素都已有明确先例**（上表），且其中三个最"招牌"的要素分别在
2025–2026 被占据：signed weight 级安全剪枝（Wei 2026）、对比激活差剪枝（TwinBreak）、
方向投影安全剪枝（ProSafePrune）。

## 4. 更大的风险：科学叙事的抢占

比公式更值得警惕的是 Wei et al. 2026 的**发现层面**内容：

- "约 0.0005% 全模型权重紧凑承载 harmful generation，剪除后 utility 保持"——
  与我们 weight 级的核心卖点（极稀疏、能力代价小）同构；
- 跨 harm category 的统一机制、generation 与 understanding 的双重分离、
  pruning as causal probe 的定位——覆盖了"稀疏安全权重集合"故事的大部分。

我们的结果与之的差异化空间在于：(a) 目标是 **refusal 机制**（jailbreak 视角）
而非 harmful generation（去安全能力视角），两者符号相反、机制不同——
且我们 neuron 层面的系列阴性结果（N3/N6/N7：神经元粒度无关键集合，方向层面有效）
恰是对"压缩"假说粒度的独立补充证据；(b) gradient-free 单前向评分的工程优势
（可测的显存/时间对比）；(c) 方向可解释性（分数有解析的 direct-flow 含义）。

## 5. 最终判定

- **signed actdiff edge**：新颖的**组合式评分规则**（特定组合未见先例），
  不是新方法类。可 defend 的定位："signed-SNIP 家族的 gradient-free、
  方向可解释特化 + 解析 direct-flow 分解"。
- **Route B ratio**：新颖性更薄（signed SNIP / Wanda 分母，SSD 有比值先例），
  卖点应放在实证强度而非公式。
- **禁止的表述**（比 codex 的清单更紧）：不得宣称 "first safety-specific pruning"
  （Wei 2024/2026、Antidote、TwinBreak、ProSafePrune 均在前）；
  不得宣称 "first signed weight pruning"（Wei 2026）；
  不得宣称 "first contrastive-activation pruning"（TwinBreak）。
- **可以主张的**：特定公式组合的首个实例化 + 单模型初步因果证据 +
  与 Wei matched-scope 的实证对比（同池同数据下 edge 的安全特异性/权重高一个
  数量级，下游保持相当）。

## 6. 后续建议（按优先级）

1. ~~把 Wei 2026 signed SNIP 也加入 matched-scope 对比（codex §13.3 已列，未执行）~~
   **已完成**：`results/signed_snip_comparison.json` +
   `docs/comparison-wei2024-safety-alignment-pruning.md` §13.5。signed SNIP 在
   **关键词口径**三档全部 refusal=0，但附带损伤远高于 edge
   （KL 1.08–5.15 vs 0.04–0.13；下游 acc_norm 0.607–0.682 vs 0.681–0.687）。
   **语义判定修正（2026-08-23，`results/llm_judge_eval.json`）**：最小档 signed
   配置的"refusal=0"是 judge 伪影——gemma-2-9b-it 判其语义拒绝率实为
   0.5625（输出为无关键词的说教式拒答），substantive-harmful 仅 0.4375；
   edge_s0.0005 语义拒绝率 0.000、substantive 1.000。即语义口径下 edge 的
   安全破坏优势反而扩大，"追平"不成立；
2. 跨模型复现（≥2 个模型家族）；
3. 独立 held-out harmfulness 评估（StrongREJECT/HarmBench 级 judge），
   跳出关键词 refusal——**部分完成（2026-08-23）**：gemma-2-9b-it 三分类
   judge 已重评 harmful_val 上的关键配置（`results/llm_judge_eval.json`，
   `src/ttsafety/llm_judge.py`），证实 edge/ratio/wei24 结论成立、signed SNIP
   的追平为伪影；尚未做的是 StrongREJECT/HarmBench 级独立题库；
4. 与 TwinBreak 的神经元级选择在同一模型上对照（直接检验"weight 级 vs neuron 级"
   的粒度论断——我们已有 N 系列阴性结果支撑）；
5. 投稿前做一次 Semantic Scholar/OpenAlex 引用图检索（codex 文档 §8.5 同样建议）。

## 7. 参考（本次新增，codex 未覆盖）

- TwinBreak: [arXiv:2506.07596](https://arxiv.org/abs/2506.07596)（USENIX Security 2025）
- ProSafePrune: ICLR 2026
- Antidote: [arXiv:2408.09600](https://arxiv.org/abs/2408.09600)（ICML 2025）
- SSD: [arXiv:2308.07707](https://arxiv.org/abs/2308.07707)（AAAI 2024）
- C-ΔΘ: [arXiv:2602.04521](https://arxiv.org/abs/2602.04521)（2026 preprint）
- Wei et al. 2026（正确署名）: [arXiv:2604.09544](https://arxiv.org/abs/2604.09544)
