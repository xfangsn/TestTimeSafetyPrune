# Weight-level 拒绝编辑与单权重剪枝实验结果（W0–W5）

> ⚠️ **历史存档(标注于 2026-08-27):本文记录较早阶段的计划/方法/结果,可能与当前代码与结论脱节。**
> 方法线现已定名 **BLADE** 并引入 best-first ELS(有效层选择);当前定稿的方法与实验结果请以
> [`blade-method.md`](blade-method.md)、[`blade-algorithm.md`](blade-algorithm.md)、
> [`blade-experiment-report.md`](blade-experiment-report.md) 为准。**下文内容保留原貌,未作修订。**

> 对应计划：`docs/plan-weight-level-refusal-editing.md`。  
> 模型：Llama-3.2-3B-Instruct，bf16，RTX 5090。  
> 主结论：**weight level 明显优于 neuron level**。方向正交化在 held-out test
> 上把拒绝率从 0.990 降到 0.005，PPL 只增加 1.16%；此外，train-score/val-select
> 的 Taylor/Wanda ratio 在目标权重池仅剪 0.01% 时，val 拒绝率已降至 0.031，
> 并通过预注册的全部副作用与随机对照门槛。

## 1. 实验完整性

数据分工保持不变：

- refusal direction、Taylor/edge score：harmful_train / CAA pairs；
- source、scope、component、lambda、sparsity 选择：harmful_val；
- 路线 A 最终一次报告：harmful_test；
- 路线 B 在路线 A test 之后才完成，因此严格保持 val-only，没有再次使用 test。

W0 基线完全复现历史结果：

| 指标 | 结果 |
|---|---:|
| harmful_val refusal | 1.000 |
| harmless refusal | 0.00625 |
| WikiText PPL | 13.062107741 |
| 相对历史 PPL 漂移 | 0.00% |
| 空/乱码/循环率 | 0 |

模型结构确认：28 层，hidden size 3072，MLP intermediate size 8192。
selection 在 W4 前写入 SHA256 锁；`weight_ortho_final.json` 已设置为存在即拒绝
重跑，避免反复使用 held-out test。

设施验证：

- 39 个全量测试通过；
- lambda=0 logits 逐比特一致；
- context 退出后权重逐比特恢复；
- fp32 完整正交化残差约 5.1e-9；
- BF16 hook 与物化 logits 平均误差约 0.035，argmax 一致率 97.8%–100%；
- W5 选中值在 context 内确实为零，退出后逐比特恢复。

## 2. 路线 A：refusal-direction weight orthogonalization

### 2.1 Source pilot

固定 destination=L7–L18、components=both：

| source | lambda | harmful_val refusal | PPL delta | KL |
|---|---:|---:|---:|---:|
| L8 | 1.0 | 0.875 | +0.30% | 0.0040 |
| L10 | 1.0 | 0.594 | +0.39% | 0.0090 |
| **L14** | **1.0** | **0.125** | **+1.84%** | **0.0491** |

因此按预注册规则锁定 L14；主网格没有重新选择 source。

### 2.2 48-cell 主网格

MLP-only 在所有 16 个 cell 中 harmful refusal 都保持 1.000。增大 scope/lambda
只会增加 over-refusal 和 KL；例如 MLP L7–L18 lambda=1 的 harmless refusal
为 0.412。

Attention-only 已有强效果：

| 配置 | harmful_val refusal | harmless | PPL delta | KL |
|---|---:|---:|---:|---:|
| attn L8–L14 lambda=.75 | 0.750 | 0.003 | +0.33% | 0.0101 |
| attn L8–L14 lambda=1 | 0.391 | 0 | +0.53% | 0.0178 |
| attn L7–L18 lambda=1 | 0.297 | 0 | +0.98% | 0.0208 |

MLP+attention 联合编辑最强：

| 配置 | harmful_val refusal | harmless | PPL delta | KL |
|---|---:|---:|---:|---:|
| both L8–L14 lambda=.75 | 0.719 | 0 | +0.31% | 0.0111 |
| both L8–L14 lambda=1 | 0.125 | 0.003 | +0.60% | 0.0222 |
| both L7–L18 lambda=1 | 0.078 | 0 | +1.80% | 0.0491 |

同 scope/lambda 下，both L8–L14 lambda=1 比最强单组件 attention 多降低
26.6pp refusal；L7–L18 lambda=1 的协同增益为 21.9pp。两者都超过 10pp 的
预注册机制阈值。结论是：**attention 是主要 writer，MLP 单独无效，但与
attention 联合时存在真实协同。**

全层范围并不更好。both L0–L27 lambda=1 的 refusal 仅降到 0.359，同时 PPL
+7.03%、KL 0.307，说明局部中层窗口比“到处阻断”更干净。

### 2.3 W3 稳健性

三个 shortlist 配置的三 seed 随机方向 refusal 均为 1.000。label-shuffled
mean-diff 平均 refusal 分别为 0.943、0.948、0.969，远弱于真实方向。

最佳 shared-direction 配置的随机方向 gap 为 92.2pp。其他变体：

| 变体 | harmful_val refusal | harmless | PPL delta | KL |
|---|---:|---:|---:|---:|
| shared L14, both L7–L18 | 0.078 | 0 | +1.80% | 0.0491 |
| norm-preserving | 0.094 | 0 | +1.85% | 0.0487 |
| shared + embedding | 0.094 | 0 | +1.88% | 0.0432 |
| **per-destination-layer direction** | **0.000** | **0** | **+1.16%** | **0.0369** |

norm-preserving 基本复现原结果，因此效果不是简单缩小 column norm。embedding
没有增益。最终锁定 per-layer directions、L7–L18、both、lambda=1。

### 2.4 W4 held-out test

| 指标 | baseline | weight orthogonalization |
|---|---:|---:|
| harmful_test refusal | 0.990 | **0.005** |
| harmful_test compliance | 0.010 | **0.995** |
| harmless refusal | 0.00625 | **0.000** |
| WikiText PPL | 13.0621 | 13.2133 |
| PPL delta | — | **+1.16%** |
| harmless KL | — | **0.0369** |
| 空/乱码/循环率 | — | **0** |

三个 per-layer 随机方向的 test refusal 都是 0.990。路线 A 不仅超过
activation steering 的 test 0.670 / PPL +2.44%，也大幅超过 N7 actdiff
top-4096 的 test 0.910 / PPL +16.33%；达到预注册“强成功”
(test refusal <=0.20)，实际为 0.005。

hook 与 BF16 物化版本在 16 条上的文本 exact match 为 56.25%、token agreement
73.2%，但两者 refusal 都为 0。这说明物化时 BF16 舍入会改变具体 greedy 路径，
导出永久模型前仍应对物化模型做完整复测；本实验没有导出安全被移除的模型。

## 3. 路线 B：refusal-aware individual-weight pruning

### 3.1 分数与目标池

目标池为 L7–L18 的 12 个 `mlp.down_proj` 和 12 个
`self_attn.o_proj`，共 415,236,096 个权重。比较：

- Taylor-only：正的 signed weight-margin gradient；
- Taylor/Wanda ratio：Taylor refusal score / harmless preservation score；
- signed actdiff edge；
- label-shuffled Taylor；
- Wanda-smallest、magnitude-smallest；
- random 三 seed。

所有 sparsity 都是相对目标池的比例，并施加单矩阵最多 10% 的 cap。

### 3.2 关键结果

主结果 Taylor/Wanda ratio：

| sparsity | 权重数 | harmful_val refusal | harmless | PPL delta | KL |
|---:|---:|---:|---:|---:|---:|
| **0.01%** | **41,524** | **0.031** | **0.006** | **+0.03%** | **0.0017** |
| 0.05% | 207,618 | 0.016 | 0.003 | +0.30% | 0.0503 |
| 0.10% | 415,236 | 0.000 | 0 | +0.61% | 0.2444 |
| 0.50% | 2,076,180 | 0.000 | 0 | +2.47% | 1.9428 |
| 1.00% | 4,152,361 | 0.000 | 0 | +5.57% | 3.1728 |
| 5.00% | 20,761,805 | 0.000 | 0 | +80.07% | 6.7321 |

0.01% 是最干净的选定 cell。三个同 sparsity 随机 seed 的 refusal 都为 1.000，
因此 gap=96.9pp。输出质量标志为 0；val-only 复核再次得到 refusal=0.031，
并验证 41,524 个值全部置零及退出后逐比特恢复。

signed actdiff edge 提供独立复现：

- 0.01%：refusal 0.125，PPL +0.09%，KL 0.0381；
- 0.05%：refusal 0.000，PPL +0.61%，KL 0.0935，仍通过全部硬约束。

Taylor-only 在 0.01% 可把 refusal 降到 0.016，但 PPL +5.34%、KL 1.54，
且出现 3.44% 循环标志，不合格。Wanda 分母对选择“拒绝相关但 benign
代价低”的边至关重要。

控制结果：

- random 在 <=1% 的全部 cell refusal 均为 1.000；
- Wanda-smallest 和 magnitude-smallest 即使 5% 也保持 refusal=1.000；
- shuffled Taylor 最小 0.01% 已有 harmless refusal 0.584、PPL +9.89%、
  KL 2.89，更高 sparsity 为灾难性损伤。

人工样例抽检中，ratio 0.01% 的 changed 样例是连贯、实质性的 engagement，
不是空输出、乱码或简单移除拒绝关键词；harmless 样例保持正常回答。仍需注意：
这不是自动化安全语义 judge，不能替代独立人工标注。

### 3.3 W5 判定

预注册标准要求 sparsity<=1%、refusal<=0.05、PPL<=5%、KL<=0.10、
超随机>10pp。ratio 0.01% 全部满足，因此：

**“存在 refusal-specific sparse key-weight set”在 train-score/val-select
层面成立。**

路线 B 没有运行 harmful_test：其配置是在路线 A 已使用 test 后才完成选择，
若再用同一 test 会破坏一次性 held-out 原则。要把该结论提升为独立 test 结论，
需要新增未使用的 harmful benchmark 或重新预注册一个新 split。

## 4. 对四个研究问题的回答

1. **rank-one weight orthogonalization 是否主导 activation steering / N7？**  
   是。test refusal 0.005，PPL +1.16%，同时优于 steering 的 0.670/+2.44%
   和 N7 的 0.910/+16.33%。

2. **拒绝方向主要由哪类 writer 写入？**  
   attention 是主要单组件；MLP-only 无效，但 MLP+attention 相对 attention-only
   有 21.9–26.6pp 协同，因此完整机制是联合写入。

3. **是否存在 <=1% 且副作用受控的 refusal-specific weight set？**  
   是，val 上甚至只需目标池的 0.01%（41,524 个权重）。ratio 和 edge 两种
   score 都有合格 cell，随机/Wanda/magnitude 无效。但路线 B 尚无独立 test。

4. **结果支持稀疏安全权重还是分布式低维方向？**  
   两者不是互斥的。拒绝在 activation geometry 上高度低维，跨多个 residual
   writer 的方向正交化最稳定；同时，在原始标量 weight 基底中，监督式
   Taylor/Wanda 能找到一个极稀疏、功能特异的 edge 集合。N7 的阴性说明
   “整列 neuron”是过粗的删除单位，不代表单条 weight edge 不稀疏。

## 5. 限制

- refusal 主指标是关键词 judge；虽做了样例抽检，仍可能漏掉软拒绝或误判。
- harmless prompt-token KL 很低，但路线 A greedy completion exact match 仅
  5.94%、mean token agreement 29.5%；生成路径敏感，低 KL 不等于逐字保持。
- W5 使用自产 CAA refusal/compliance completion 构造 Taylor score，可能带入
  生成器偏差。
- W5 的 0.01% 是目标池比例，不是全模型比例；unstructured zero 在普通 dense
  kernel 上不会带来实际推理加速。
- 路线 B 主扫描只有 val 证据；已按 `docs/plan-harmbench-heldout.md` 补
  HarmBench standard 200 条一次性 held-out（见 `results-edge-summary.md` §7）：
  edge 语义拒绝 0.005 / 实质有害 0.990（强成功），ratio 回退至 0.170 / 0.830。
  下一步仍应注册更多 held-out 数据与更强的语义安全/能力评估。

## 6. Edge 分数改进变体（任务 #2/#3）

实现：`scripts/score_edge_variants.py`（打分）+ `scripts/sweep_edge_variants.py`
（扫描/汇总）；结果 `results/edge_variants.json` + `results/edge_variants.png`；
分数缓存 `data/weight_scores/edge_{signcons,trimmed,subspace_k2,k4,k8}.pt` 等。
基线 edge 定义 s_ij = max(r_i·W_ij·Δa_j, 0)（r = 层单位拒绝方向，Δa =
最后非 pad token 处 writer 输入的有害−无害均值差）。

**任务 #2：分布感知变体**（基线只用组均值，这里用逐样本信息）：

- `edge_signcons`：逐样本贡献 c_ij(x) = r_i·W_ij·(a_j(x)−μ^U_j)，按边累计
  n_pos（c>0 的有害样本数）与 sum_pos；score = (sum_pos/256)·(n_pos/256)^γ，
  γ=1。诊断：纯一致性分数 n_pos/256 单独保存；正分数边的一致性中位数仅
  ~0.51（q99=1.0）——大部分边只在约半数样本上同号。
- `edge_trimmed`：Δa 改用 10% 截尾均值（有害/无害分别双尾截尾）。

**任务 #3：子空间方向变体**（1-D 方向是否丢信号）：对每层取有害样本最后
token 残差（以无害均值为中心）的 PCA 前 k 个分量，score = max(Σ_k'
r^k'_i·W_ij·Δa_j, 0)（先求和后截断；clamp-then-sum 作诊断）。**PC1 与缓存
拒绝方向的余弦在全部 12 层 ≥0.998**（L14 起 =1.000），PC1 解释 48–82%
方差（top-8 累计 78–91%）——均值差方向确实是主子方向。clamp-first 与
sum-then-clamp 的相关性随 k 下降（Pearson 0.95/0.87/0.81 @ k=2/4/8）。

**扫描结果**（harmful_val refusal / ppl Δ% / harmless KL；3 档 × 5 变体 +
基线 edge 与缓存 random0 对照）：

| 变体 | 0.01% 档 (41.5k) | 0.05% 档 (207.6k) | 0.1% 档 (415.2k) |
|---|---|---|---|
| edge（基线） | 0.125 / +0.09% / 0.038 | 0.000 / +0.61% / 0.094 | 0.000 / +1.52% / 0.131 |
| signcons | 0.156 / +0.11% / 0.036 | 0.000 / +0.63% / 0.090 | 0.000 / +1.55% / 0.132 |
| trimmed | 0.156 / +0.10% / 0.037 | 0.000 / +0.65% / 0.094 | 0.000 / +1.76% / 0.135 |
| subspace k=2 | 0.781 / −0.26% / 0.024 | 0.016 / +0.10% / 0.059 | **0.000 / +0.52% / 0.078** |
| subspace k=4 | 0.859 / −0.01% / 0.017 | 0.016 / +0.00% / 0.044 | **0.000 / +0.73% / 0.075** |
| subspace k=8 | 0.938 / −0.06% / 0.017 | 0.297 / +0.40% / 0.044 | **0.000 / +0.58% / 0.069** |
| random0（对照） | 1.000 | 1.000 | 1.000 |

**判定**（胜出 = 同档 refusal 更低且 ppl/KL 不差，或 refusal 相同且损伤更小）：

- 分布感知两个变体**无改进**：三档与基线几乎重合（最小档甚至略差，
  0.156 vs 0.125）——组均值已经用尽了拒绝相关的激活差异信号。
- 子空间变体**在小档变差、在最大档严格胜出**：0.01% 档 refusal 随 k 增大
  急剧变差（0.78–0.94；PC2+ 稀释了拒绝特异方向）；但 0.1% 档 k=2/4/8 全部
  refusal=0.000 且 ppl 损伤降为基线的 ~1/3（+0.52–0.73% vs +1.52%）、
  KL 约为基线的 55%（0.069–0.078 vs 0.131）。0.05% 档 k=2/4 的
  refusal=0.016（1/64，噪声边缘）伴近零 ppl 代价。
- 结论：1-D 方向没有"丢信号"（PC1≈方向本身）；子空间求和的作用是
  **正则化**——把选择从纯方向对齐扩展到有害子空间，最大档下用同样的
  安全破坏换更小的附带损伤。若采用，建议 k=2 且只在 ≥0.05% 档使用。
## 7. 语义评审复核（#6）与对抗鲁棒性（#8）

实现：`src/ttsafety/llm_judge.py`（gemma-2-9b-it 评审，三分类
REFUSAL/HEDGED/COMPLIANCE + substantive_harmful 标志，JSON 输出 + 稳健解析）、
`scripts/gen_adversarial_gens.py`（3B 生成，640 条）、`scripts/run_llm_judge.py`
（评审 + 汇总，独立进程加载 18GB 评审模型）。结果：
`results/llm_judge_eval.json` + `results/adversarial_robustness.json`，
逐条判定 `data/judge_judgments.jsonl`。

**评审模型 sanity**：10 条人工标注样例 9/10 通过（通过 ≥9 门槛），
解析失败 0。

### 7.1 关键配置的关键词评审 vs LLM 语义评审（harmful_val 64 条）

| 配置 | 关键词 refusal | LLM refusal | HEDGED | COMPLIANCE | 实质有害率 |
|---|---|---|---|---|---|
| base | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| edge_s0.0005 | 0.000 | **0.000** | 0.828 | 0.172 | **1.000** |
| ratio_s0.0001 | 0.031 | 0.047 | 0.688 | 0.266 | 0.953 |
| signed_p0.0001_q0.0001 | 0.000 | **0.562** | 0.344 | 0.094 | **0.438** |
| wei_p0.01_q0.01 | 0.156 | 0.156 | 0.844 | 0.000 | 0.844 |

**headline 是否存活**：

- **edge@0.05% 的 refusal 0.000 完全存活**且被强化：LLM 评审同样 0 拒绝，
  且 100% 输出含实质有害内容（82.8% 是 hedged 而非干净服从——
  大量 "I can provide a hypothetical…" 的免责框架，但内容是真的）。
  ratio@0.01%（0.031→0.047）与 wei24@0.01（0.156→0.156）同样存活。
- **signed SNIP 的 refusal 0.000 部分死于语义评审**：真实语义拒绝率
  0.5625、实质有害率仅 0.438。其输出是无拒绝关键词的**退化劝阻循环**
  （"Identity theft is a serious crime… Identity theft is a serious
  crime…"），关键词评审误判为服从。这与其高 KL（1.08）相互印证——
  §13.5（对比文档）中 signed SNIP 的安全有效性需要按此下调解读；
  edge 的优势进一步扩大。

### 7.2 对抗鲁棒性（edge_s0.0005 剪枝模型 vs base）

| 攻击 | 配置 | 关键词 refusal | LLM refusal | 实质有害率 |
|---|---|---|---|---|
| 无（greedy） | base | 1.000 | 1.000 | 0.000 |
| 无（greedy） | pruned | 0.000 | 0.000 | 1.000 |
| prefill "Sure, here is" | base | 0.109 | 0.141 | 0.859 |
| prefill | pruned | 0.000 | 0.000 | 1.000 |
| sampling t=1.0 top_p=0.9 (3 seeds) | pruned | 0.000–0.016 | 0.000–0.016 | 0.984–1.000 |

**判定**：prefill 攻击单独即可击穿 base（实质有害 85.9%），说明 3B 的拒绝
本身就脆；剪枝后模型在所有攻击下实质有害率 ~100%，拒绝能力不可恢复。
**剪掉拒绝"门"后，有害生成能力完整保留且极易激发**——与 Wei 2026 对
gate-pruning 的预测一致。剪枝得到的不是"安全的模型"也不是"能力被删除的
模型"，而是"门被拆除、能力完好"的模型。
