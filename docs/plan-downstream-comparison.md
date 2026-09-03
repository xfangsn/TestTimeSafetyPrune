# Plan: Prune 后下游任务对比 — signed actdiff edge vs Wei SNIP set-difference

> ⚠️ **历史存档(标注于 2026-08-27):本文记录较早阶段的计划/方法/结果,可能与当前代码与结论脱节。**
> 方法线现已定名 **BLADE** 并引入 best-first ELS(有效层选择);当前定稿的方法与实验结果请以
> [`blade-method.md`](blade-method.md)、[`blade-algorithm.md`](blade-algorithm.md)、
> [`blade-experiment-report.md`](blade-experiment-report.md) 为准。**下文内容保留原貌,未作修订。**

> 状态：待批准。批准前不动代码。
> 前置：Route B weight pruning 全部完成（`docs/results-weight-level-refusal-editing.md`），
> Wei matched-scope baseline 已扫完（`results/sweep_wei_snip_set_difference.json`），
> 所有分数缓存就绪（`data/weight_scores/edge.pt`、`wei_safety_snip.pt`、`wei_utility_snip.pt`
> 及各 ranking），剪枝掩码可确定性重建，无需重算分数。

## 1. 目标

现有对比只覆盖了安全性（refusal）+ 浅层能力代理（WikiText PPL、harmless KL）。
Wei et al. (ICML 2024) 的效用评估用的是**六个 zero-shot 下游任务准确率**——这正是
当前缺失的直接可比维度。本实验在 matched n_pruned 下比较两种方法剪掉 safety 权重后的
下游任务表现。

## 2. 对比组（matched-n 三档 + 对照）

目标池均为 L7–L18 的 `down_proj + o_proj`（共 415,236,096 权重）：

| 档 | signed actdiff edge | Wei SNIP set-diff | random 对照 |
|---|---|---|---|
| 1 (~0.01%) | edge_s0.0001, n=41,524, refusal 0.125 | wei_p0.0001_q0.0001, n=29,244, refusal 0.797 | random0_s0.0001 |
| 2 (~0.05%) | edge_s0.0005, n=207,618, refusal 0.000 | wei_p0.0005_q0.0005, n=145,993, refusal 0.625 | random0_s0.0005 |
| 3 (~0.1%) | edge_s0.001, n=415,236, refusal 0.000 | wei_p0.001_q0.001, n=291,727, refusal 0.5625 | random0_s0.001 |

外加 base（不剪）基线。共 10 个模型配置。

注意两点并如实报告：
- n_pruned 不完全相等（Wei 集合差大小由 p,q 决定）——按最近邻匹配；
- 两种方法在 matched-n 下的 refusal 水平差异很大（edge 0.125 vs wei 0.797），因此
  补充第二种解读：「达到同等 refusal 降幅所需代价」——Wei 要 refusal ≤0.16 需要
  n≈2.65M（p=q=0.01），把该 cell 加入下游评估（第 11 个配置），比较
  "同等安全破坏程度下谁的下游保持更好"。

## 3. 下游任务集（全部本地缓存，离线可跑）

六个 zero-shot 任务，对齐 Wei et al. 的效用评估风格：

- ARC-Easy（test 2,376）、ARC-Challenge（test 1,172）（allenai/ai2_arc）
- HellaSwag（val 10,042，抽 2,000，seed 固定）（Rowan/hellaswag）
- PiQA（val 1,838）（baber/piqa）
- Winogrande（val 1,267）（allenai/winogrande, winogrande_xl）
- BoolQ（val 3,270，抽 2,000）（aps/super_glue boolq）

评分方式：lm-eval 风格 context/continuation logprob，报 acc 与 acc_norm
（continuation 按字节长度归一）双指标。HellaSwag/BoolQ 抽 2,000 控制时长，
其余全量。

## 4. 执行步骤

- **W1 评估设施**（`src/ttsafety/downstream.py` + `scripts/eval_downstream.py`）：
  数据集加载（local cache）、prompt 格式化（lm-eval 惯例）、batched logprob 评分、
  acc/acc_norm。单元测试：base 模型各任务 acc 落在 3B-Instruct 合理区间
  （与已知公开数字大致相符，如 ARC-E > 0.65），不合理则报错不静默继续。
- **W2 掩码重建 + 评估**：从缓存分数重建 11 个配置的剪枝掩码（复用
  `weight_prune.py` 的 ranking/mask 设施，验证 selected-values-zero 检查通过），
  逐配置跑六个任务 + WikiText PPL（10k token 快速版）。逐配置落盘部分结果，
  可断点续跑。
- **W3 汇总**：`results/downstream_comparison.json` + 表 + 图
  （六任务平均 acc vs n_pruned，两方法 + random；以及「refusal 降幅 vs 平均 acc」
  trade-off 图）。结论追加 `docs/comparison-wei2024-safety-alignment-pruning.md`
  新一节（§13.1 的 matched-scope baseline 结果 + 下游对比），数字与 JSON 交叉核对。

## 5. 判定与报告口径

- 主问题：matched-n 下，谁的六任务平均 acc 更高（差距 >1pp 视为有实际意义）；
- 次问题：同等 refusal 降幅视角下的能力代价对比；
- 无论结果偏向哪边都如实报告；不声称"优于 Wei"除非 matched-n 三档全部占优。

## 6. 预计开销

11 配置 × 6 任务 × ~2k 样本的 batched logprob 评分，3B bf16，预计 GPU 30–50 分钟。
掩码重建为 CPU 内存操作（每配置 ~3.3GB 分数读取），逐配置处理避免内存峰值。

## 7. 纪律

- 不涉及 harmful split 使用（纯能力评估），无泄漏风险；
- 不 commit；GPU 走 flock；逐配置存中间结果。
