# Plan: 双边分离实验——"拒绝门"与"有害生成能力"在权重空间是否可分离（改进方向 #10）

> ⚠️ **历史存档(标注于 2026-08-27):本文记录较早阶段的计划/方法/结果,可能与当前代码与结论脱节。**
> 方法线现已定名 **BLADE** 并引入 best-first ELS(有效层选择);当前定稿的方法与实验结果请以
> [`blade-method.md`](blade-method.md)、[`blade-algorithm.md`](blade-algorithm.md)、
> [`blade-experiment-report.md`](blade-experiment-report.md) 为准。**下文内容保留原貌,未作修订。**

> 状态：解释性 plan，待批准后执行。
> 背景：你之前说没看懂这条，这里先讲清逻辑。

## 1. 这条要回答什么

我们和 Wei et al. 2026（2604.09544）看起来都在"剪安全相关权重"，但目标是**两个相反
方向的机制**：

| | 目标行为 | 分数 | 删除哪一侧 | 预期效果 |
|---|---|---|---|---|
| Wei 2026 signed SNIP | 有害**生成能力** | `W·∇L(harmful response)` | 负分（删除使 harmful loss 上升 = 权重支持生成） | 模型**产不出**有害内容；论文报告 refusal 反而**上升**（+932%），拒绝门完好甚至过敏 |
| 我们（edge/ratio） | **拒绝门** |  refusal 方向投影 / refusal margin Taylor | 正分（删除降低 refusal margin） | 模型**不再拒绝**；有害内容照常生成 |

Wei 2026 的核心叙事是"安全训练的脆弱性在于只装了拒绝门、没动生成能力"（他们的
Figure 3：剪掉生成能力后 refusal/解释/检测都保留；且生成权重被剪后拒绝反而过敏）。
我们的核心叙事是"拒绝门本身可以被精准拆除"。

**如果两个叙事都对，那么两个权重集合应该几乎不重叠，且交叉评估应呈现双重分离**：

- 剪生成权重（Wei 式）→ 有害内容质量崩坏，但 refusal 率不降甚至上升；
- 剪拒绝权重（我们）→ refusal 归零，但有害内容依然流畅实质（hedged but real）。

这是一个**双重分离（double dissociation）**实验：同一模型、同一目标池、同一预算，
两种目标函数，交叉测量两种行为。若成立，"拒绝门"与"生成能力"在权重空间是两套
可分离的参数——这比任何单侧结果都强，而且是两篇论文叙事互相印证的直接证据。

## 2. 已有基础

- Wei 2026 signed SNIP 分数已算好：`data/weight_scores/wei_safety_signed_snip.pt`
  （safety 端是 refusal response NLL——**注意：这是"拒绝端"分数，不是 Wei 论文的
  "有害生成端"分数**，见 §3 的修正点）；
- 我们的 edge/ratio 分数与掩码设施齐全；
- 交叉评估指标现成：refusal 率（关键词 judge + 待升级的 LLM judge）、有害内容质量
  （需要语义评估，正好与改进方向 #6 共用 judge 设施）。

## 3. 关键设计修正

现有 `wei_safety_signed_snip.pt` 的 safety loss 是**拒绝回复**的 NLL——剪负分权重
削弱的是"说出拒绝"的能力，本质上仍是拒绝门操作，不是 Wei 2026 的语义。
要复现 Wei 2026 的原始目标，safety 数据必须是**有害回复**的 NLL
（他们用 jailbroken 模型在 AdvBench 上生成的 harmful responses）。
我们已有现成材料：`data/caa_pairs.jsonl` 里每对都有 compliance 字段
（steering 下生成的、judge 确认的服从回复）——用它作为"有害生成端"的 NLL 目标，
重新算一份 signed SNIP 分数 `wei_gen_signed_snip.pt`。

于是四方对比（同一目标池 L7–L18、matched n_pruned）：

| 配置 | 目标 loss | 删除侧 | 预期 refusal | 预期有害内容质量 |
|---|---|---|---|---|
| edge@0.05%（已有） | refusal 方向 | 正 | ↓↓ 0.000 | 保持（流畅有害） |
| refusal-SNIP signed（已有分数） | 拒绝回复 NLL | 负 | ↓↓ 0.000 | 保持？ |
| **gen-SNIP signed（新算）** | 有害回复 NLL | 负 | **不降或上升** | **崩坏** |
| random | — | — | 不变 | 不变 |

注意第 2、3 行是我们管线里两种语义的 signed SNIP——它们的对比本身就有信息量
（同一算法、不同目标数据，应锁定不同机制）。

## 4. 实验步骤

1. 算 `wei_gen_signed_snip.pt`：compliance 字段 response NLL 的 signed W·∇L
   （镜像 `score_wei_signed_snip.py`，约 10s GPU）；
2. 生成端剪枝掩码：per-matrix 最负 top-q，q 对齐 edge@0.05% 的 n（~207k）；
   utility 排除沿用缓存的 abs SNIP；
3. 交叉评估（三个剪枝配置 + random + base）：
   - refusal 率（harmful_val，关键词 + LLM judge 双判定）；
   - 有害内容质量：对 harmful_val 生成回复，用 LLM judge 按"是否给出实质有害信息"
     打分（复用方向 #6 的 judge）；
   - 过拒绝：harmless 320 条的 refusal 率（Wei 2026 预测生成端剪枝会**升高**它）；
   - ppl + 下游六任务（复用 W 设施）；
4. 集合重叠分析：两个剪枝集合的 Jaccard / 富集倍数（预期接近随机）；
5. 产出 `results/bilateral_separation.json` + 表，结论写入对比文档。

## 5. 判定

- 双重分离成立（gen-SNIP 剪枝 refusal 不降且内容质量崩；edge 剪枝 refusal 归零且
  内容质量保持；两集合重叠 ≈ 随机）→ 强正面结论，直接支撑论文叙事；
- 部分成立或混杂 → 如实报告，并讨论两种机制的耦合程度。
