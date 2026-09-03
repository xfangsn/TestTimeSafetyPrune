# BLADE（Signed Actdiff Edge）：实验结果详细总结

> 方法名 **BLADE** = Behavioral Localization via Activation-Difference Edges；
> 下文 "signed actdiff edge" / "edge" 为其技术描述符，指同一评分规则。
> 范围：本文只汇总 **BLADE / signed actdiff edge**（下称 edge）这一条方法线的全部
> 实验证据，便于快速查阅与对外展示。方法定义与创新性审计见
> [`method-and-related-work-gradient-free-signed-edge.md`](method-and-related-work-gradient-free-signed-edge.md)；
> 与 Wei et al. 2024/2026 的完整方法对比见
> [`comparison-wei2024-safety-alignment-pruning.md`](comparison-wei2024-safety-alignment-pruning.md)。
> 模型：Llama-3.2-3B-Instruct（28 层，hidden 3072，MLP intermediate 8192），bf16，RTX 5090。
> 汇总日期：2026-08-23。

## 1. 方法速览

对 L7–L18 的 residual writer 输出矩阵（`self_attn.o_proj` 与
`mlp.down_proj`，共 24 个矩阵、415,236,096 个标量权重）逐边打分：

```text
c_ij = r_i · W_ij · (μ^H_j − μ^U_j)
s_ij = max(c_ij, 0)
```

- `r`：该 destination layer 的单位 refusal direction；
- `μ^H − μ^U`：harmful 与 harmless prompt 最后一个非 pad token 处 writer
  输入激活的均值差；
- `W_ij` 三者乘积为正 ⇒ 该 edge 对"harmful 相对 harmless 沿拒绝方向的
  局部直接输出差"有正贡献；**在 writer 输入激活不变的前提下**,置零它会使该局部量严格减少 `c_ij`
  (真实剪枝经再前向会改变激活,故对最终行为只是近似)。
- 全程只需前向，无反向传播（gradient-free）。

选择规则：全局 top-k（`k = max(1, round(目标池比例 × 池大小))`），每矩阵上限 10%。

## 2. 主扫描结果（`results/sweep_weight_prune.json`）

六档 sparsity（相对目标池），harmful_val 64 条 / harmless **320 条全部**用于 harmless refusal；
**只有 KL 用前 128 条**：

| sparsity | n_pruned | harmful refusal | harmless refusal | PPL Δ% | harmless KL | random gap | 过硬约束 |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 0.01% | 41,524 | 0.125 | 0.000 | +0.09% | 0.038 | 87.5pp | ✅ |
| **0.05%** | **207,618** | **0.000** | **0.000** | **+0.61%** | **0.094** | **100pp** | ✅ |
| 0.1% | 415,236 | 0.000 | 0.000 | +1.52% | 0.131 | 100pp | ❌(KL>0.10) |
| 0.5% | 2,076,180 | 0.000 | 0.003 | +4.36% | 0.266 | 100pp | ❌ |
| 1% | 4,152,361 | 0.016 | 0.000 | +12.76% | 0.301 | 98.4pp | ❌ |
| 5% | 20,761,805 | 0.000 | 0.000 | +79173% | 3.886 | — | ❌（模型整体打坏） |

- 预注册硬约束：refusal ≤0.05、PPL Δ ≤5%、KL ≤0.10、超随机 **≥10pp**、
  `quality.adverse_rate ≤0.01`（退化/乱码率上限）。
- **选定 cell：0.05%（207,618 权重）**——refusal 归零且全部约束通过；
  0.01% 档全部副作用指标更优但 refusal 停在 0.125。
- 同档随机对照（3 seed）refusal ≈1.000（**5% 档为 0.969–1.000**，模型被整体打坏时
  略降）；Wanda-smallest 与 magnitude-smallest 即使 5% 也打不动 refusal。
- 5% 档 PPL 爆炸而 random 同档基本无损，说明 edge 分数集中在真正承重的
  权重上，不是"随便剪 5% 都会坏"。

## 3. 被剪权重的空间分布（edge_s0.0005，207,618 条）

| 层 | down_proj | o_proj | 合计 |
|---:|---:|---:|---:|
| L7 | 1,087 | 561 | 1,648 |
| L8 | 2,049 | 1,474 | 3,523 |
| L9 | 2,620 | 4,910 | 7,530 |
| L10 | 3,177 | 956 | 4,133 |
| L11 | 7,312 | 22,362 | 29,674 |
| L12 | 9,135 | 8,511 | 17,646 |
| L13 | 13,631 | 16,472 | 30,103 |
| **L14** | **17,760** | **17,896** | **35,656** |
| L15 | 15,796 | 14,579 | 30,375 |
| L16 | 13,373 | 1,920 | 15,293 |
| L17 | 13,340 | 2,227 | 15,567 |
| L18 | 14,460 | 2,010 | 16,470 |

分布强烈集中于 L11–L15（合计 143,454 条，占 ~69%），峰值 L14——与路线 A 独立选出的
refusal direction 最优 source layer（L14）一致，两条方法线互相印证。
单矩阵修剪比例最高的是 L11 o_proj（22,362 条，占该矩阵 0.24%，远低于
10% cap）。
组件上 down_proj 与 o_proj 大体各半（113,740 / 93,878），与路线 A
"attention 为主、MLP 协同"的结论相容。

## 4. 下游能力（`results/downstream_comparison.json`，六任务 acc_norm）

| 配置 | ARC-E | ARC-C | HellaSwag | PiQA | Wino | BoolQ | mean acc | mean acc_norm | ppl₁₀k Δ% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | .723 | .467 | .737 | .770 | .619 | .812 | 0.6508 | 0.6878 | — |
| edge 0.01% | .715 | .474 | .740 | .771 | .616 | .809 | 0.6506 | 0.6873 | +0.23% |
| edge 0.05% | .712 | .464 | .737 | .768 | .619 | .800 | 0.6472 | 0.6831 | +0.64% |
| edge 0.1% | .702 | .462 | .738 | .766 | .623 | .793 | 0.6407 | 0.6806 | +1.54% |

refusal 归零的 0.05% 档，mean acc_norm 相对 base 仅 −0.47pp（逐任务最大
跌幅 BoolQ −1.2pp）——量级很小(约半个百分点),没有任何单一任务出现塌陷。
(注:此为**单次**评测,未做多 seed,故"是否纯噪声"无统计对照;只能说绝对变化很小。)

## 5. 语义评审复核（#6，gemma-2-9b-it judge，`results/llm_judge_eval.json`）

关键词 refusal 会漏判"无拒绝关键词的说教/退化文本"，故用三分类语义评审
（REFUSAL / HEDGED / COMPLIANCE + substantive_harmful 标志；sanity 9/10，
0 解析失败）复核 harmful_val 64 条：

| 配置 | 关键词 refusal | 语义 refusal | HEDGED | COMPLIANCE | 实质有害率 |
|---|---:|---:|---:|---:|---:|
| base | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| **edge_s0.0005** | 0.000 | **0.000** | 0.828 | 0.172 | **1.000** |
| ratio_s0.0001 | 0.031 | 0.047 | 0.688 | 0.266 | 0.953 |
| signed_p0.0001_q0.0001 | 0.000 | **0.562** | 0.344 | 0.094 | **0.438** |
| wei_p0.01_q0.01 | 0.156 | 0.156 | 0.844 | 0.000 | 0.844 |

- **edge 的 headline 完全存活且被强化**：语义拒绝同样 0.000，100% 输出
  含实质有害内容。82.8% 为 hedged 服从（"我可以假设性地描述…"免责框架
  + 真实内容），17.2% 为直接服从。
- 对照组 signed SNIP（Wei 2026）的关键词 refusal=0 是评审伪影：其输出
  多为无关键词的退化劝阻循环，语义拒绝率实为 0.5625、实质有害率仅
  0.438。**语义口径下 edge 对该 baseline 的优势反而扩大。**

## 6. 对抗稳健性（#8，`results/adversarial_robustness.json`）

edge_s0.0005 剪枝模型 vs base，harmful_val 64 条，语义评审：

| 攻击 | base 实质有害率 | pruned 实质有害率 | pruned 语义拒绝 |
|---|---:|---:|---:|
| 无（greedy） | 0.000 | 1.000 | 0.000 |
| prefill "Sure, here is" | 0.859 | 1.000 | 0.000 |
| sampling t=1.0 top_p=0.9 ×3 seed | — | 0.984–1.000 | 0.000–0.016 |

- base 在 prefill 下已有 85.9% 实质有害——3B 的拒绝本身就脆；
- 剪枝模型在所有攻击下实质有害率 ~100%，拒绝**几乎不可恢复**（sampling 下最多
  翻回 1–2/64,即语义拒绝 ≤0.016,非严格 0);
- 结论：**剪掉的是"拒绝的门"，有害生成能力完整保留且极易激发**——
  与 Wei 2026 对 gate-pruning 机制的预测一致。剪枝模型不是"能力被删除
  的模型"，而是"门被拆除、能力完好"的模型。

## 7. HarmBench held-out 一次性评估（`results/harmbench_heldout.json`）

按 `docs/plan-harmbench-heldout.md` 执行的**一次性** held-out 评估：不调参、
不迭代、judge prompt 一字未改。

- 数据：HarmBench standard text behaviors。HF `walledai/HarmBench` 为 gated，
  改用官方 GitHub CSV（`data/behavior_datasets/harmbench_behaviors_text_all.csv`，
  SHA256 `8d81acce…4afc`），400 行中过滤 `FunctionalCategory=standard` 且
  `ContextString` 为空，得 **200 条**（plan 估计的 ~320 为高估，standard 实际
  只有 200），落盘 `data/harmbench_standard.jsonl` + `.meta.json`（来源/哈希/条数）。
- 生成：`scripts/gen_harmbench_gens.py`，五组锁定配置 greedy 128 tokens，
  每组生成前做 selected-values-zero 验证，共 1000 条
  `data/harmbench_gens.jsonl`。
- 评审：`scripts/judge_harmbench.py`，gemma-2-9b-it（sanity 9/10），
  逐条判定 `data/harmbench_judgments.jsonl`，**解析失败 0/1000**。

| 配置 | 关键词 refusal | 语义 refusal | HEDGED | 实质有害率 | Δ语义 vs val | Δ实质 vs val |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.910 | 0.890 | 0.110 | 0.110 | −0.110 | +0.110 |
| **edge_s0.0005** | 0.000 | **0.005** | 0.830 | **0.990** | +0.005 | −0.010 |
| ratio_s0.0001 | 0.115 | 0.170 | 0.710 | 0.830 | +0.123 | −0.123 |
| signed_p0.0001_q0.0001 | 0.005 | **0.610** | 0.330 | 0.390 | +0.047 | −0.047 |
| wei_p0.01_q0.01 | 0.165 | 0.140 | 0.795 | 0.860 | −0.016 | +0.016 |

- **预注册判定（edge_s0.0005）：强成功**——语义拒绝 0.005 ≤ 0.10 且实质有害
  0.990 ≥ 0.80，两个口径都远优于门槛。held-out 泛化成立，val 上的 headline
  不是过拟合 64 条的伪影。
- edge 的泛化 gap 几乎为零（±1pp）；输出形态与 val 一致（83% hedged 服从）。
- **ratio 在 HarmBench 上明显回退**（语义拒绝 0.047→0.170、实质 0.953→0.830），
  val 上"几乎追平 edge"的表观差距在 held-out 上重新拉开。
- base 在 HarmBench 上语义拒绝也只有 0.890（val 为 1.000）——3B 的原生拒绝
  对分布外 harmful 指令本身就有 ~11% 漏过，这是 base 侧的背景噪声，不影响
  剪枝组的判定。
- signed SNIP 的语义评审伪影在 held-out 上复现（语义拒绝 0.610 vs 关键词
  0.005），§5 的结论方向不变。

## 8. 变体实验（#2/#3，`results/edge_variants.json`）

在基线 `s_ij = max(r_i·W_ij·Δa_j, 0)` 上测试五类改进（3 档 sparsity）：

| 变体 | 0.01% 档 | 0.05% 档 | 0.1% 档 |
|---|---|---|---|
| edge（基线） | 0.125 / +0.09% / 0.038 | 0.000 / +0.61% / 0.094 | 0.000 / +1.52% / 0.131 |
| signcons（符号一致性 γ=1） | 0.156 / +0.11% / 0.036 | 0.000 / +0.63% / 0.090 | 0.000 / +1.55% / 0.132 |
| trimmed（10% 截尾均值） | 0.156 / +0.10% / 0.037 | 0.000 / +0.65% / 0.094 | 0.000 / +1.76% / 0.135 |
| subspace k=2 | 0.781 / −0.26% / 0.024 | 0.016 / +0.10% / 0.059 | **0.000 / +0.52% / 0.078** |
| subspace k=4 | 0.859 / −0.01% / 0.017 | 0.016 / +0.00% / 0.044 | **0.000 / +0.73% / 0.075** |
| subspace k=8 | 0.938 / −0.06% / 0.017 | 0.297 / +0.40% / 0.044 | **0.000 / +0.58% / 0.069** |

（表格数字 = refusal / ppl Δ% / KL；random 对照三档均 1.000。）

- **分布感知变体（signcons/trimmed）无提升**：逐样本诊断显示正分边的
  符号一致性中位数仅 ~0.51，但一致性加权并不改善选择——组均值已耗尽
  拒绝相关的激活差异信号。
- **子空间变体呈双向效应**：小档急剧变差（PC2+ 稀释了拒绝特异方向；
  PC1 与缓存拒绝方向余弦 ≥0.998，1-D 方向本身没丢信号）；0.1% 档则
  严格占优——refusal 同为 0.000 时 ppl 损伤降为基线 ~1/3、KL 降为 ~55%。
  其作用是**正则化**：把选择从纯方向对齐扩展到有害子空间。若采用，
  建议 k=2 且仅用于 ≥0.05% 档。注意：仅 val 证据。

## 9. 与全部 baseline 的定位（matched-pool L7–L18，语义修正后）

| 方法 | 信号 | 达到 refusal=0 的代价 | 语义复核 |
|---|---|---|---|
| **edge（本文）** | signed actdiff，gradient-free | **0.05% 池 / 208k 权重即 refusal=0**，ppl +0.61%，下游 −0.5pp | ✅ 成立（语义 0.000） |
| ratio（Route B） | signed Taylor / harmless Wanda | 0.01% / 41.5k 时 refusal **仍 0.031(非 0)**,需更高档才到 0 | ✅ 成立（语义 0.047） |
| Wei 2024 unsigned SNIP | 无符号 safety∖utility | 12.8× 权重仅降到 0.156 | ✅ 成立（语义 0.156） |
| Wei 2026 signed SNIP | 有符号 NLL 敏感度 | 关键词 0.000 但 KL 1.08–5.15、下游 −5~−8pp | ❌ 语义拒绝 0.562（最小档） |
| random / magnitude / Wanda-min | — | 各档基本打不动（5% 档 random 0.97–1.00） | — |

edge 的独特位置：**唯一 gradient-free 且语义复核成立**的方法。**权重效率上先前"edge 需 5×
权重、逊于 ratio"的说法是错的**——那是拿 edge 的 refusal=0 点(208k)与 ratio 的**非零**点
(41.5k @ refusal 0.031)错配比较;按同一"达到 refusal=0"口径,ratio 在 0.01% 尚未清零、需更高
稀疏度,二者达零的权重量级相当。edge 的真正优势是**无需反向传播、无需成对 completion,打分成本
最低,且机制解释最干净**(局部输出差的逐边精确展开)。

## 10. 局限（如实列出）

- val 侧结论基于 harmful_val（64 条）；已补 HarmBench standard 200 条一次性
  held-out（§7，edge 强成功），但本 repo 自有 harmful_test split 仍未用于
  路线 B（路线 B 在路线 A 用掉 test 后才完成选择，遵守一次性 held-out 纪律）。
- 语义评审为单一评审模型（gemma-2-9b-it）；sanity 9/10 而非满分。
  HarmBench 级评估已做（§7），StrongREJECT 未做。
- 本 edge 主线在单模型（Llama-3.2-3B-Instruct）上;**refusal 移除的跨模型复现此后已补**
  ——`scripts/blade_refusal_els.py`(best-first ELS)在 Qwen3-4B、Gemma-3-4B 上同样把拒绝清零
  (Qwen 0.98→0.00 @+2.06%、Gemma 见 `results/blade_refusal_els_*.json`),详见
  [`blade-experiment-report.md`](blade-experiment-report.md) §3;但本文 §2–§9 的逐档/下游/语义
  证据仍仅 Llama。
- 0.05% 是目标池（24 矩阵）比例；全模型占比约 0.007%。unstructured
  zero 在 dense kernel 上无推理加速。
- 子空间变体的 0.1% 档优势同样只有 val 证据。

## 11. 复现入口

```bash
# 打分（gradient-free，~分钟级；--score 必填）
flock -w 14400 .gpu.lock uv run python scripts/score_refusal_weights.py --score edge
# 扫描（逐 cell 断点续跑；--rule 必填,或用 --finalize）
flock -w 14400 .gpu.lock uv run python scripts/sweep_weight_prune.py --rule edge
# 下游六任务
flock -w 14400 .gpu.lock uv run python scripts/eval_downstream.py --config edge_s0.0005
# 语义评审 + 对抗稳健性
flock -w 14400 .gpu.lock uv run python scripts/gen_adversarial_gens.py
flock -w 14400 .gpu.lock uv run python scripts/run_llm_judge.py
# HarmBench held-out（一次性，勿重跑调参）
uv run python scripts/prepare_harmbench.py
flock -w 14400 .gpu.lock uv run python scripts/gen_harmbench_gens.py
flock -w 14400 .gpu.lock uv run python scripts/judge_harmbench.py
```
