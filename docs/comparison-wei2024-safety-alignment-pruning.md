# Route B 与 Wei et al. (ICML 2024) 的方法比较

> ⚠️ **历史存档(标注于 2026-08-27):本文记录较早阶段的计划/方法/结果,可能与当前代码与结论脱节。**
> 方法线现已定名 **BLADE** 并引入 best-first ELS(有效层选择);当前定稿的方法与实验结果请以
> [`blade-method.md`](blade-method.md)、[`blade-algorithm.md`](blade-algorithm.md)、
> [`blade-experiment-report.md`](blade-experiment-report.md) 为准。**下文内容保留原貌,未作修订。**

> 比较对象：Boyi Wei et al.,
> [Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications](https://arxiv.org/abs/2402.05162),
> arXiv v4 / ICML 2024。本文按 v4（2024-10-24）的方法和实验描述比较。
>
> 我方方法：Route B，即 signed refusal-margin Taylor / harmless-Wanda ratio
> individual-weight pruning。完整定义见
> [`method-refusal-aware-weight-pruning.md`](method-refusal-aware-weight-pruning.md)。

## 1. 执行摘要

两项工作的共同问题高度重合：都希望找到对安全/拒绝重要、对一般能力相对不重要的
标量权重，并通过置零验证其因果作用。因此 Route B 不是独立于 Wei et al. 的全新
研究范式，而是一个有明确公式差异的后续变体。

最核心差异是：

~~~text
Wei et al.:
    unsigned safety importance + unsigned utility importance
    -> discrete top-set difference

Route B:
    signed refusal-vs-compliance deletion benefit
    / harmless Wanda output cost
    -> continuous benefit/cost ranking
~~~

当前证据可以支持“Route B 在自身控制组中有效”，但因为尚未实现同设置的 Wei
SNIP set-difference baseline，不能支持“Route B 优于 Wei et al.”。

> 更新：matched-scope baseline 与下游六任务对比已完成，见 §13.4——matched-n
> 下两者下游效用打平（差距 <1pp），但 edge 用 ~1/13 的权重达到更低的 refusal。

## 2. 术语澄清

Wei et al. 在部分表述中称其单元为 `neuron` 或 `weight neuron`，但其公式、mask 和
剪枝对象是单个矩阵 entry `W_ij`。这与本项目早期 N 系列所说的完整 MLP intermediate
dimension 不是同一种 neuron。

本比较统一使用：

- **scalar weight / edge**：单个 `W_ij`；
- **structured MLP neuron**：一整个 intermediate activation dimension 及其关联权重；
- **rank component**：一个低秩矩阵方向。

在这个定义下，Wei et al. 的 neuron-attribution 分支与 Route B 都属于 scalar-weight
unstructured pruning。

## 3. 共同点

### 3.1 相同的总体研究目标

两者都试图找到：

> 删除后显著削弱安全拒绝，同时尽量保持一般语言能力的参数子集。

两者都把普通模型压缩技术重新用于安全机制归因，而不是以推理加速为主要目标。

### 3.2 相同的基本干预

最终都对选中权重执行：

~~~text
W'_ij = 0
~~~

都属于不对剩余权重做补偿更新的非结构化剪枝。

### 3.3 相同的基础技术来源

两者都建立在：

- SNIP/一阶 Taylor 权重敏感度；
- Wanda/权重乘输入 activation norm；
- safety 与 utility/harmless 贡献需要分离；
- 只剪 safety 高、utility 低的部分才可能保持能力。

Wei et al. 的公式与方法总览见
[论文 §2](https://arxiv.org/html/2402.05162v4)。

## 4. 核心公式对比

### 4.1 Wei et al.：无符号 SNIP + 集合差

对 safety 数据 `D_s` 和 utility 数据 `D_u` 分别计算：

~~~text
I_s(W_ij) = E_x~D_s |W_ij * d L_s(x) / d W_ij|
I_u(W_ij) = E_x~D_u |W_ij * d L_u(x) / d W_ij|
~~~

其中 `L` 是给定 prompt 后生成目标 response 的 conditional NLL。

在每个矩阵输出行内定义 safety top-q 与 utility top-p，然后取集合差：

~~~text
S_s(q) = per-row top-q% according to I_s
S_u(p) = per-row top-p% according to I_u
K_Wei  = S_s(q) - S_u(p)
~~~

该方法含义是：安全回答重要，但不属于 utility 最重要集合的权重被视为
safety-critical。论文对 `p,q` 在 0.1--90 范围内做网格搜索。

### 4.2 Route B：有符号 paired margin + 连续 ratio

Route B 对同一 harmful prompt 的 refusal/compliance pair 定义：

~~~text
m(x) = mean_logp(R | x) - mean_logp(C | x)
~~~

计算：

~~~text
g_bar_ij = E_x d m(x) / d W_ij
T_ij     = max(W_ij * g_bar_ij, 0)
H_ij     = |W_ij| * sqrt(E_harmless[x_j^2])
S_ij     = T_ij / (H_ij + 1e-7)
K_ours   = capped_global_top_k(S)
~~~

Route B 直接估计删除 edge 的 refusal-margin benefit，并用 harmless writer-output
cost 连续归一化。

## 5. 差异一：行为目标

### Wei et al.

Safety calibration 样本为：

~~~text
harmful prompt -> model-generated safe refusal response
~~~

高 `I_s` 表示权重对复现该 safe response 的 NLL 很重要。该目标同时包含：

- 识别有害意图；
- 决定拒绝；
- 生成自然语言；
- 具体 refusal template 和措辞。

### Route B

Route B 使用同 prompt 下的 paired responses：

~~~text
harmful prompt
|- refusal response
+- compliance response
~~~

目标是二者的相对概率 margin。两类 completion 共有的 prompt 理解和语言建模成分
理论上可在差分中部分抵消。

### 判断

这是实质方法差异。Route B 更直接针对“拒绝相对服从的选择边界”，而 Wei et al.
更广泛地针对“生成安全回答的重要性”。但 paired margin 是否带来更高 specificity
仍需同设置实验证明，不能只由公式推出。

## 6. 差异二：符号与样本聚合顺序

### Wei et al.

~~~text
I_Wei = E |W * grad(L)|
~~~

先对每个样本取绝对值，再跨样本平均。不同样本上方向相反的梯度都会增加重要性。
该分数回答“这个权重是否影响安全回答”，不回答删除会使安全增强还是减弱。

### Route B

~~~text
T_ours = max(W * E grad(m), 0)
~~~

先跨 pair 累积有符号梯度，再乘权重并截取正部。只有在总体上预测删除会降低
refusal margin 的 edge 才保留正分。

### 后果

- Wei 更容易保留只在部分样本重要、但方向不一致的权重；
- Route B 更偏好跨样本方向一致的 refusal-supporting edge；
- Route B 可能漏掉类别专门化但总体梯度抵消的 edge；
- 这两个 score 即使使用相同数据也不会产生相同 ranking。

## 7. 差异三：能力保持方式

### Wei et al.：完整 utility loss attribution

Wei et al. 在 Alpaca-Cleaned prompt-response 上计算 `I_u`，因此 utility score 通过
最终 conditional NLL 衡量完整模型行为。优点是更接近 downstream utility，缺点是
需要 utility target response 和反向传播。

### Route B：harmless Wanda 局部输出代理

Route B 只在 harmless prompts 上收集 writer input RMS：

~~~text
H_ij = |W_ij| * RMS_harmless(x_j)
~~~

它估计删除 edge 后的直接 writer-output 变化，不是完整 downstream loss。优点是：

- 只需前向；
- 无需 harmless target response；
- 可以把“预计 refusal benefit / 局部 harmless cost”直接形成连续 ratio。

代价是局部 output preservation 不一定等价于知识、推理和长文本生成能力保持。

## 8. 差异四：离散集合差与连续效用比

### Wei et al.

~~~text
top safety q% minus top utility p%
~~~

一个权重是否跨过 top-p/top-q 阈值决定其资格；进入集合后，safety 与 utility
score 的相对距离不再参与统一排序。

### Route B

~~~text
predicted refusal benefit / estimated harmless cost
~~~

所有候选得到连续 trade-off score，可在统一预算下直接取 top-k。

### 权衡

| 属性 | Set difference | Ratio |
|---|---|---|
| 阈值 | 两个集合阈值 `p,q` | 一个 sparsity，加固定 `epsilon` |
| trade-off | 离散排除 utility top set | 连续 benefit/cost |
| 尺度敏感性 | 主要依赖 rank，较弱 | 对分母与跨矩阵尺度更敏感 |
| 近零 utility score | 只表示不在 top set | 可能产生很大的 ratio |
| 可解释性 | safety-important 且非 utility-top | 每单位 harmless 代价的 refusal benefit |

不能理论上断言其中一个统一优于另一个。

## 9. 差异五：目标矩阵与剪枝调度

### Wei et al.

论文对每个 Transformer block 的七类线性层做 block-wise pruning：

- `q_proj`、`k_proj`、`v_proj`、`o_proj`；
- `up_proj`、`gate_proj`、`down_proj`。

从第一 block 开始，当前 block 剪枝后重算输出，再继续下一 block；每个矩阵按输出行
独立剪固定比例。详见
[论文 Appendix B](https://arxiv.org/html/2402.05162v4)。

### Route B

只处理 L7--L18：

- `self_attn.o_proj`；
- `mlp.down_proj`。

所有 score 在未剪枝原模型上一次计算，然后构造固定全局 ranking，所有选中值同时
置零。每个矩阵最多贡献其局部 top 10%，但不要求每个输出行有相同比例。

### 判断

- Wei 更接近全模型 pruning pipeline；
- Route B 更接近机制约束下的 residual-writer edge localization；
- Route B 的中层/组件先验降低搜索空间，但也可能把关键 input transform edge 排除；
- one-shot ranking 更容易复现，但没有适应上游剪枝导致的 activation drift。

## 10. 差异六：数据与评估

| 维度 | Wei et al. | Route B |
|---|---|---|
| 模型 | Llama2-7B-chat、13B-chat | Llama-3.2-3B-Instruct |
| safety attribution | AdvBench + safe responses | 247 refusal/compliance CAA pairs |
| utility/harmless attribution | Alpaca prompt-response NLL | 320 harmless prompt activations |
| attribution sample | 每次抽 128 pairs | 247 pairs / 320 prompts |
| safety metric | Vanilla、GCG suffix、adversarial decoding ASR | harmful refusal keyword rate |
| utility metric | 六个 zero-shot task 平均准确率 | WikiText PPL、harmless KL/生成质量 |
| 模型数量 | 2 | 1 |
| Route B 独立 test | 不适用其设计 | 尚未执行 |

Wei et al. 报告删除不到 3% 的权重可使三类 ASR 接近 1，同时维持合理 zero-shot
accuracy；只看 adversarial 情形时，不到 1% 已能显著削弱安全。结果见
[论文 §4.1](https://arxiv.org/html/2402.05162v4)。

Route B 当前最佳 cell：

~~~text
target-pool sparsity: 0.01% = 41,524 / 415,236,096
harmful_val refusal:  1.000 -> 0.03125
PPL delta:            +0.0265%
harmless KL:          0.00166
random refusal:       1.000 for all 3 seeds
~~~

这些数字不能直接得出“比 Wei 稀疏 300 倍”：

- 比例分母不同；
- 目标矩阵范围不同；
- 模型与数据不同；
- refusal 与 ASR 不是同一指标；
- Wei 覆盖 adversarial attack 和下游任务，Route B 目前仅 val。

## 11. 差异七：对照与证据强项

### Wei et al. 的强项

- 两个模型；
- 全部七类线性层；
- vanilla、adversarial suffix、adversarial decoding；
- 六个 zero-shot utility tasks；
- 同时研究 scalar weights 与 ActSVD ranks；
- 分析 safety/utility overlap 与冻结 safety region 后的 fine-tuning attack。

### Route B 的强项

- paired refusal/compliance 直接行为目标；
- deletion sign 明确；
- train-score / val-select 数据纪律；
- Taylor-only、ratio、signed edge 三种 refusal-aware signal；
- shuffled labels、magnitude、Wanda-smallest、random 三 seed；
- PPL、prompt-token KL、completion agreement 与 adverse output flags；
- 精确可逆权重 context 和 selected-value 验证。

两者证据强项互补：Wei 的外部有效性与安全评估更强，Route B 的局部副作用诊断和
有符号机制控制更细。

## 12. 当前没有做出的关键基线

> 更新：本节所述 baseline 已完成，结果见 §13.4。

现有 Route B controls 不包含等价的 Wei set-difference：

- `taylor` 是 signed paired-margin Taylor，不是 safety NLL SNIP；
- `wanda` 是剪 harmless Wanda 最小值，不是 safety/utility set difference；
- `taylor-shuffled` 验证标签语义，但不比较两种 disentanglement 方法。

因此当前只能说：

> Route B ratio 优于本项目中的 Taylor-only、random、magnitude、Wanda-smallest
> 和 shuffled controls。

不能说：

> Route B ratio 优于 Wei et al. 的 SNIP set-difference。

## 13. 公平比较所需实验

### 13.1 Matched-scope baseline（优先）

在完全相同的 L7--L18 `down_proj + o_proj` 目标池中实现：

~~~text
I_s = E |W * grad(NLL_refusal)|
I_u = E |W * grad(NLL_utility)|
K   = per-row top-q(I_s) - per-row top-p(I_u)
~~~

要求：

- 使用与 Route B 相同的数据 split；
- 统一实际 `n_pruned`，而不只统一百分比；
- 统一 harmful/harmless/PPL/KL/quality 评估；
- 对 `p,q` 只在 validation 选择；
- 加 random 三 seed；
- 与 ratio 比较完整 Pareto front。

这回答“评分和分离方式是否更好”。

### 13.2 Faithful Wei replication（第二阶段）

按原论文扩展到所有 block、七类矩阵、per-output pruning 和 block-wise recompute。
这回答“在作者原始搜索空间中能否复现其结论”，但不能单独隔离 ratio 的贡献。

### 13.3 独立评估

两种方法锁定配置后，在新 benchmark 上一次性评估：

- HarmBench 或未使用的 harmful split；
- 语义 harmfulness judge，而非只用 refusal substrings；
- jailbreak/adversarial prompts；
- MMLU/ARC/HellaSwag 等能力任务；
- 至少两个额外模型。

### 13.4 实验结果：matched-scope baseline + 下游任务对比（已完成）

对应计划 `docs/plan-downstream-comparison.md`。两部分结果：
§13.1 的 matched-scope Wei baseline 已扫完
（`results/sweep_wei_snip_set_difference.json`），下游六任务对比见
`results/downstream_comparison.json` + `results/downstream_comparison.png`
（逐配置部分结果在 `results/downstream/{config}.json`）。

**Matched-scope Wei baseline（§13.1）**：与 Route B 完全相同的
L7–L18 `down_proj + o_proj` 目标池（415,236,096 权重）、相同数据 split，
官方 `wandg_set_difference` 选择（per-matrix top-q safety ∖ top-p utility，
官方 commit 0b0e707）。关键格子（harmful_val refusal / ppl Δ%）：

| p=q | n_pruned | refusal | ppl Δ% |
|---|---|---|---|
| 0.0001 | 29,244 | 0.797 | +0.22% |
| 0.0005 | 145,993 | 0.625 | +0.82% |
| 0.001 | 291,727 | 0.5625 | +1.21% |
| 0.01 | 2,654,984 | 0.156 | +3.06% |

同池对照：signed actdiff edge 在 n=41,524 即 refusal 0.125、n=207,618 即
0.000——Wei set-difference 需要 ~12.8 倍的权重（2.65M）才降到 0.156。

**下游评估设施（W1）**：`src/ttsafety/downstream.py` +
`scripts/eval_downstream.py`，lm-eval 风格 context/continuation logprob
评分（报 acc 与按字节长度归一的 acc_norm）。六个 zero-shot 任务全部本地
缓存：ARC-Easy（test 2,376）、ARC-Challenge（test 1,172）、HellaSwag
（val 抽 2,000，seed 0）、PiQA（val 1,838）、Winogrande（val 1,267）、
BoolQ（val 抽 2,000，seed 0）。**Sanity gate（未剪枝 base）通过**：
ARC-E 0.723 / ARC-C 0.467 / HellaSwag 0.737 / PiQA 0.770 /
Winogrande 0.643 / BoolQ 0.787（acc_norm；Winogrande/BoolQ 为 acc），
与 3B 模型公开 harness 数字一致。WikiText ppl（10k token 快速版）
base = 12.39。调试记录：BoolQ 初版 label 映射写反（label True=yes
误映射到 " no"），acc 0.21 触发 gate，修正后 0.787。

**下游对比（W2，12 配置；每配置掩码重建后 selected-values-zero 检查全部
通过，n_pruned 与 sweep JSON 逐格核对一致）**。acc_norm 逐任务
（ARC-E / ARC-C / HS / PiQA / WG / BoolQ）：

| 配置 | n_pruned | refusal | 六任务 acc_norm | mean acc | mean acc_norm | ppl Δ% |
|---|---|---|---|---|---|---|
| base | 0 | — | .723/.467/.737/.770/.619/.811 | 0.6508 | 0.6878 | — |
| edge_s0.0001 | 41,524 | 0.125 | .715/.474/.740/.771/.616/.808 | 0.6506 | 0.6873 | +0.23% |
| edge_s0.0005 | 207,618 | **0.000** | .712/.464/.737/.768/.619/.799 | 0.6472 | 0.6831 | +0.64% |
| edge_s0.001 | 415,236 | **0.000** | .702/.462/.738/.766/.623/.792 | 0.6407 | 0.6806 | +1.54% |
| wei_p0.0001 | 29,244 | 0.797 | .719/.459/.736/.770/.615/.800 | 0.6461 | 0.6832 | +0.18% |
| wei_p0.0005 | 145,993 | 0.625 | .715/.462/.736/.767/.611/.802 | 0.6471 | 0.6822 | +0.73% |
| wei_p0.001 | 291,727 | 0.5625 | .721/.456/.735/.767/.616/.805 | 0.6484 | 0.6833 | +1.02% |
| wei_p0.01 | 2,654,984 | 0.156 | .706/.467/.732/.768/.617/.802 | 0.6460 | 0.6820 | +2.34% |
| random0_s0.0001 | 41,524 | 1.000 | .721/.462/.738/.771/.616/.808 | 0.6509 | 0.6861 | −0.03% |
| random0_s0.0005 | 207,618 | 1.000 | .719/.464/.738/.774/.620/.809 | 0.6517 | 0.6872 | +0.13% |
| random0_s0.001 | 415,236 | 1.000 | .718/.462/.739/.771/.615/.810 | 0.6499 | 0.6857 | +0.15% |
| ratio_s0.0001（参考） | 41,524 | 0.031 | .722/.466/.736/.771/.614/.809 | 0.6502 | 0.6864 | +0.05% |

**判定（按计划 §5，差距 >1pp 才算有实际意义）**：

- **主问题（matched-n 三档谁的 mean acc 更高）**：edge − wei =
  +0.45pp / +0.00pp / −0.77pp——**三档全部 <1pp，无实际意义差异**。
  不能声称 Route B 下游保持优于 Wei（`claim_superior: false`）。
  两方法在 0.1% 档都比 random 低 ~0.5–0.9pp（random 不伤下游），
  即两者破坏安全的权重都略带通用功能。
- **次问题（同等 refusal 降幅视角）**：edge@0.05%（n=207,618）已
  refusal 0.000，mean acc 0.6472；Wei 要到 p=q=0.01（n=2,654,984，
  **12.8 倍权重**）才 refusal 0.156，mean acc 0.6460。同等（实则更低）
  refusal 水平下下游保持几乎相同（+0.12pp），但 edge 用远少的权重达到
  远彻底的安全破坏——**安全特异性上 edge 明显更高效**，代价效率上
  两者下游损伤相当。
- 综合：matched-n 下下游效用打平；安全性-效用 trade-off 图上 edge
  占据左下优势区（refusal→0 而 acc 损失 ≤1pp），Wei 全曲线 refusal
  停在 0.16 以上。

**复现**：

```bash
# W1 base sanity gate
flock -w 14400 .gpu.lock uv run python scripts/eval_downstream.py --config base
# W2 逐配置（掩码重建 + 零值验证 + 六任务 + ppl10k，部分结果断点续跑）
flock -w 14400 .gpu.lock uv run python scripts/eval_downstream.py --config edge_s0.0005
# ... 其余 10 个配置同；或 --all 一次跑完
# W3 汇总（纯 CPU）
uv run python scripts/aggregate_downstream.py
```

### 13.5 Wei et al. 2026 signed SNIP baseline（已完成）

对应 `docs/originality-assessment.md` §6 建议 1。Wei et al. (2026,
arXiv:2604.09544) 的 **signed SNIP**：I(W_ij) = mean_x[W_ij · dL/dW_ij]，
L 为 response-token NLL（prompt 掩码、含 EOT），**不取绝对值**（其
Eq. 1–2 "critically omits the absolute value"）；负分数权重促进目标行为生成。
本项目适配：safety loss = refusal-response NLL（同 2024 baseline 的数据，
`data/caa_pairs.jsonl` 247 对），utility 排除沿用缓存的绝对值 SNIP
（`wei_utility_snip.pt`）；选择 = per-matrix 最负 top-q% safety ∖ per-matrix
top-p% utility。实现：`scripts/score_wei_signed_snip.py`（签名打分，~8s GPU；
sanity：负分数占比 ≈50%，且 |signed| ≠ abs 版本，max diff 0.0021）、
`scripts/sweep_wei_signed_snip.py`（同 2024 sweep 的评估口径与 matched 网格）、
下游复用 `scripts/eval_downstream.py`。结果：
`results/sweep_wei_signed_snip.json` + 三方对比
`results/signed_snip_comparison.json`。

**三方 matched-n 对比**（refusal = harmful_val 64 条；KL = 128 条 harmless
prompt 的条件分布偏移；下游 = 六任务 mean acc_norm；ppl Δ% 为 50k token 版）：

| 档 | 方法 | n_pruned | refusal | ppl Δ% | KL | 下游 acc_norm |
|---|---|---|---|---|---|---|
| ~0.01% | edge | 41,524 | 0.125 | +0.09% | 0.038 | 0.687 |
| | wei2024 unsigned | 29,244 | 0.797 | +0.22% | 0.031 | 0.683 |
| | **wei2026 signed** | 35,543 | **0.000** | +0.38% | 1.079 | 0.682 |
| ~0.05% | edge | 207,618 | 0.000 | +0.61% | 0.094 | 0.683 |
| | wei2024 unsigned | 145,993 | 0.625 | +0.82% | 0.048 | 0.682 |
| | **wei2026 signed** | 180,155 | **0.000** | +2.37% | 3.453 | 0.639 |
| ~0.1% | edge | 415,236 | 0.000 | +1.52% | 0.131 | 0.681 |
| | wei2024 unsigned | 291,727 | 0.562 | +1.21% | 0.073 | 0.683 |
| | **wei2026 signed** | 360,596 | **0.000** | +4.70% | 5.151 | 0.607 |

（base 下游 acc_norm = 0.688；signed 完整网格 p=q∈{1e-4…5e-2} 全部
refusal=0，p=q=0.05 时 ppl +10,588%、模型整体打坏。）

**判定（如实报告）**：

- **signed SNIP 完全追平了 edge 的安全破坏力**：三档 refusal 全部 0.000，
  含最小档（edge 在 0.01% 档仍为 0.125）——**符号信息正是 2024 绝对值版
  丢掉的关键**，2026 版的论文主张在我们的 matched-scope 设置下得到独立证实。
- **但附带损伤远大于 edge**：第 2/3 档 harmless KL = 3.45/5.15
  （edge 为 0.09/0.13，差 30–40 倍），下游 acc_norm 掉到 0.639/0.607
  （edge 保持 0.683/0.681，base 0.688）；ppl 代价也更高（+2.4%/+4.7% vs
  +0.6%/+1.5%）。signed SNIP 的负分尾部权重同时承载较多通用功能。
- 与 2024 绝对值版对照呈镜像：unsigned 保 utility（KL 0.03–0.07、下游
  ~0.683）但几乎打不动 refusal（0.80/0.63/0.56）；signed 打碎 refusal
  但牺牲 utility。**综合「安全破坏力 × 附带损伤」双准则，edge 仍占优**；
  signed SNIP 是迄今最强的外部 baseline，但不构成对 edge 结论的推翻。

**语义判定修正（2026-08-23，LLM judge，`results/llm_judge_eval.json`）**：
上表 refusal 为关键词口径，会漏掉「不含拒绝关键词的道德说教/退化循环」。
gemma-2-9b-it 三分类 judge（sanity 9/10）对最小档 signed 配置
`signed_p0.0001_q0.0001`（n=35,543）的重评显示：关键词拒绝率 0.000 实为
**语义拒绝率 0.5625**（judge 判 REFUSAL），substantive-harmful 仅
**0.4375**——输出多为无拒绝关键词的说教式拒答或退化循环，而非真实服从。
同口径下 edge_s0.0005（n=207,618）语义拒绝率 **0.000**、substantive
1.000（其中 82.8% 为 hedged 服从），ratio_s0.0001 与 wei_p0.01 的数字在
语义判定下基本不变（0.031→0.047；0.156→0.156）。

结论下修为：**关键词口径下 signed SNIP 追平 edge；语义口径下 edge 的优势
反而扩大**——signed SNIP 在最小档的"refusal=0"一半是 judge 伪影，其高
KL/低下游并非唯一代价，真实安全破坏力也不如关键词指标所示。这同时说明
本项目后续评估应以语义 judge 为准（见 `docs/results-weight-level-refusal-editing.md` §7）。

**复现**：

```bash
flock -w 14400 .gpu.lock uv run python scripts/score_wei_signed_snip.py
flock -w 14400 .gpu.lock uv run python scripts/sweep_wei_signed_snip.py   # 逐 cell 断点续跑
flock -w 14400 .gpu.lock uv run python scripts/eval_downstream.py --config signed_p0.0005_q0.0005
uv run python scripts/compare_signed_snip.py                              # 纯 CPU 汇总
```

## 14. 创新性边界

### 14.1 已有、不能声称原创

- 用单权重剪枝研究安全脆弱性；
- 用 SNIP/Taylor 或 Wanda 计算 safety importance；
- 寻找 safety 高、utility 低的权重；
- 删除稀疏参数即可显著破坏安全；
- 同时研究 unstructured weights 与低秩修改。

### 14.2 Route B 可主张的方法差异

- refusal/compliance paired log-prob margin；
- 保留 deletion direction 的 signed Taylor positive part；
- `W * mean(gradient)` 而非 `mean(abs(W * gradient))`；
- harmless Wanda 作为 preservation cost denominator；
- 连续 refusal-benefit / harmless-cost ratio；
- residual-writer constrained pool 与 capped global selection；
- ratio 与 signed actdiff edge 的双信号复现。

在完成系统文献检索和 Wei matched-scope baseline 前，推荐称这些为
`methodological modifications` 或 `a new scoring construction`，而不是无保留地称
`the first safety-specific weight-pruning method`。

## 15. Route A 与 Wei rank-level 分支的关系

如果把“我们的方法”理解为整个 weight-level 项目，还需区分 Route A。

Wei 的 rank-level 方法为：

~~~text
Delta W_Wei = (I - Pi_u) Pi_s W
~~~

其中 `Pi_s/Pi_u` 来自 safety/utility writer outputs 的 ActSVD subspace。

Route A 为：

~~~text
W'_l = (I - lambda * r_l r_l^T) W_l
~~~

其中 `r_l` 是该 destination layer 的显式 refusal direction。

共同点：都是输出空间的低秩投影修改。

区别：

- Wei 删除 safety-important 且与 utility subspace 正交的低秩成分；
- Route A 删除语义明确的 refusal direction；
- Wei 的 subspace 由 ActSVD 输出变化定义；
- Route A 的 direction 由 harmful/harmless residual mean difference 定义；
- Route A 系统比较 MLP/attention writers 和中层 scope。

但 Route A 的基本 refusal-direction weight orthogonalization 已由
[Arditi et al.](https://arxiv.org/abs/2406.11717) 提出，不能因其不同于 Wei 就称为
首次低秩安全编辑。

## 16. 推荐论文表述

英文建议：

> Building on safety-critical weight attribution, we replace unsigned safety/utility
> set difference with a signed, contrastive deletion objective. We rank individual
> residual-writer weights by the first-order reduction in refusal-over-compliance
> margin per unit of harmless Wanda preservation cost.

中文建议：

> 在已有 safety-critical weight attribution 基础上，我们将无符号的 safety/utility
> 集合差改为有符号的对比式删除目标，并按“refusal-over-compliance margin 的一阶
> 降低量 / harmless Wanda 保持代价”对 residual-writer 单权重进行排序。

不推荐：

> 我们首次发现少量权重控制模型安全。

也不推荐在完成直接基线前写：

> 我们的方法比 Wei et al. 更稀疏、更好地保持 utility。

## 17. 参考文献

1. Boyi Wei et al.,
   [Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications](https://arxiv.org/abs/2402.05162),
   ICML 2024.
2. Namhoon Lee, Thalaiyasingam Ajanthan, Philip H. S. Torr,
   [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340),
   ICLR 2019.
3. Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter,
   [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695),
   ICLR 2024.
4. Andy Arditi et al.,
   [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717),
   2024.
