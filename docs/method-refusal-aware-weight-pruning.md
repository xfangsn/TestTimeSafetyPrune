# Route B：拒绝特异、能力保持的单权重剪枝算法

> ⚠️ **历史存档(标注于 2026-08-27):本文记录较早阶段的计划/方法/结果,可能与当前代码与结论脱节。**
> 方法线现已定名 **BLADE** 并引入 best-first ELS(有效层选择);当前定稿的方法与实验结果请以
> [`blade-method.md`](blade-method.md)、[`blade-algorithm.md`](blade-algorithm.md)、
> [`blade-experiment-report.md`](blade-experiment-report.md) 为准。**下文内容保留原貌,未作修订。**

> 状态：算法与实验均已实现；主结论目前为 train-score / val-select，尚无新的
> 独立 held-out test。实验结果见
> [`results-weight-level-refusal-editing.md`](results-weight-level-refusal-editing.md)。
>
> 本文只描述 Route B。Route A 是 refusal-direction weight orthogonalization，
> 属于另一种低秩方向编辑方法，不能与本文的非结构化单权重剪枝混称为同一算法。

## 1. 算法摘要

Route B 的目标是在一个已经对齐的语言模型中，找到一小组标量权重，使得：

1. 将这些权重置零会降低模型对 refusal completion 相对于 compliance completion
   的偏好；
2. 这些权重在 harmless 输入上的直接计算贡献尽量小；
3. 剪枝后模型的语言建模能力、harmless 行为和输出质量保持稳定。

算法为每个候选权重计算两个量：

~~~text
refusal/compliance pairs -> signed Taylor refusal benefit --+
                                                          +-> ratio -> capped global top-k
harmless prompts --------> Wanda preservation cost -------+
~~~

主评分为：

~~~text
score(weight) = predicted refusal-reduction benefit
                ------------------------------------
                estimated harmless-output cost
~~~

选中最高分权重后直接置零。算法不训练模型、不迭代重评分、不更新剩余权重，属于
one-shot、training-free、unstructured pruning。

## 2. 研究问题与适用边界

Route B 回答的是：

> 在原始 scalar-weight 坐标系中，是否存在一个极稀疏、拒绝特异、同时不承载
> 大量 harmless 能力的关键 edge 集合？

它不直接回答：

- 剪枝能否带来真实推理加速；普通 dense kernel 不会利用这些非结构化零值；
- 非拒绝是否等于有害服从；拒绝关键词指标不能替代语义安全 judge；
- 选中权重是否构成唯一机制；一阶局部评分可能只找到多个等价路径中的一种；
- 结果能否跨模型、跨 benchmark 泛化；当前 Route B 只在一个模型上完成 val 证据。

该方法用于机制研究和对齐脆弱性评估，不应被表述为安全增强或模型压缩方法。

## 3. 模型、数据与目标权重池

### 3.1 模型

- 模型：Llama-3.2-3B-Instruct；
- 推理 dtype：bf16；
- 层数：28；
- residual hidden size：3072；
- MLP intermediate size：8192。

### 3.2 数据分工

| 数据 | 数量 | 用途 |
|---|---:|---|
| `data/caa_pairs.jsonl` | 247 | refusal/compliance margin 与 Taylor 梯度 |
| `data/harmless.jsonl` | 320 | harmless writer-input 二阶矩与副作用评估 |
| `data/harmful_val.jsonl` | 64 | sparsity/rule 选择 |
| WikiText-2 固定文本 | 50k token | teacher-forced PPL |
| harmless 前 128 条 | 128 | base-to-pruned prompt-token KL |

Route B 没有使用 `harmful_test`。其配置在 Route A 已使用一次 test 后才完成选择，
因此继续使用同一 test 会破坏一次性 held-out 原则。

### 3.3 目标权重池

只考虑 L7--L18 中直接写回 residual stream 的两类矩阵：

- 12 个 `mlp.down_proj`，每个形状为 `3072 x 8192`；
- 12 个 `self_attn.o_proj`，每个形状为 `3072 x 3072`。

参数量为：

~~~text
12 * 3072 * 8192 = 301,989,888  (MLP down_proj)
12 * 3072 * 3072 = 113,246,208  (attention o_proj)
total                415,236,096
~~~

限制该目标池有两个目的：

1. 与前序实验定位到的有效中层窗口一致；
2. 只研究直接 residual writers，避免把输入变换矩阵和写回矩阵混在一个机制问题中。

## 4. 记号

对候选线性层：

~~~text
y = W x
~~~

其中：

- `W in R^(d_out x d_in)`：目标 writer 权重；
- `W_ij`：输入维度 `j` 到输出维度 `i` 的单条 edge；
- `x_j`：writer 的第 `j` 个输入激活；
- `R`：refusal completion；
- `C`：compliance completion；
- `m_theta`：模型的 refusal-over-compliance margin；
- `T_ij`：删除权重的 refusal benefit 一阶估计；
- `H_ij`：删除权重的 harmless preservation cost；
- `S_ij`：最终 Taylor/Wanda ratio。

## 5. 阶段 A：构造 refusal/compliance margin

对每个 CAA pair：

~~~text
(instruction x, refusal response R, compliance response C)
~~~

分别 teacher-force 两个回答，只对 response token 计算平均 log probability：

~~~text
l_R(x) = (1 / |R|) * sum_t log p_theta(R_t | x, R_<t)
l_C(x) = (1 / |C|) * sum_t log p_theta(C_t | x, C_<t)
~~~

长度归一化避免较长 completion 因累积更多负 log probability 而被系统性惩罚。
代码会找到 prompt/full-text tokenization 的最长公共前缀，从第一个 response token
起建立 mask；padding token 不计入平均。

定义：

~~~text
m_theta(x) = l_R(x) - l_C(x)
~~~

- `m_theta > 0`：模型更偏好 refusal completion；
- `m_theta < 0`：模型更偏好 compliance completion；
- Route B 的目标是选择删除后使 `m_theta` 降低的权重。

实现：[`scripts/score_refusal_weights.py`](../scripts/score_refusal_weights.py)。

## 6. 阶段 B：signed Taylor refusal benefit

### 6.1 删除一个权重的一阶效应

把 `W_ij` 置零等价于扰动：

~~~text
delta W_ij = -W_ij
~~~

对 margin 做一阶 Taylor 展开：

~~~text
delta m_ij ~= (d m / d W_ij) * delta W_ij
           = -W_ij * (d m / d W_ij)
~~~

如果删除该权重预计降低 refusal margin，则需要：

~~~text
delta m_ij < 0
<=> W_ij * (d m / d W_ij) > 0
~~~

因此定义：

~~~text
g_bar_ij = (1 / N) * sum_n d m_theta(x_n) / d W_ij
T_ij     = max(W_ij * g_bar_ij, 0)
~~~

`T_ij` 是 deletion benefit，而不是一般意义上的无符号重要性：

- `T_ij > 0`：删除该权重预计降低 refusal-over-compliance margin；
- `T_ij = 0`：一阶近似下删除它不会降低 margin，或方向相反；
- 分数越大，预测的 refusal reduction 越强。

### 6.2 梯度聚合顺序

实际实现为：

~~~text
T_ij = max(W_ij * mean_n(grad_ij(m_n)), 0)
~~~

而不是：

~~~text
mean_n(abs(W_ij * grad_ij(m_n)))
~~~

这意味着只有在不同 pair 上具有一致净方向的 edge 才容易获得高分；互相抵消的
梯度不会因逐样本取绝对值而累积成高重要性。

实现细节：

- 非目标参数全部冻结；
- 24 个目标矩阵单独启用梯度；
- pair batch size 为 2；
- 累积梯度和 margin 运算使用 fp32；
- 最终 score tensor 保存为 fp16，以降低磁盘占用。

## 7. 阶段 C：harmless Wanda preservation cost

仅根据 `T_ij` 剪枝，会偏向删除任何强烈影响 refusal completion 的权重，其中可能
包含语法、知识和一般生成能力。为估计 harmless 计算代价，对每个 writer 注册
forward pre-hook，并在 320 条 harmless prompt 的所有有效 prompt token 上收集：

~~~text
A_j = E_harmless[x_j^2]
~~~

定义 Wanda 风格的保持分数：

~~~text
H_ij = |W_ij| * sqrt(A_j)
~~~

其含义来自单条 edge 的直接输出：

~~~text
y_i        = sum_j W_ij x_j
delta y_ij = -W_ij x_j
~~~

因此 `|W_ij| * RMS(x_j)` 是删除该 edge 后直接 writer-output 扰动尺度的近似：

- `H_ij` 大：该权重在 harmless prompt 上承载较大的直接计算；
- `H_ij` 小：删除它预计对 harmless writer output 影响较小。

这里没有 harmless target response，也没有对 harmless loss 反向传播；它是局部输出
保持代理，不是完整 downstream utility loss。

## 8. 阶段 D：Taylor/Wanda ratio

主评分定义为：

~~~text
S_ij = T_ij / (H_ij + epsilon)
epsilon = 1e-7
~~~

完整形式：

~~~text
S_ij = max(W_ij * g_bar_ij, 0)
       -----------------------------------------
       |W_ij| * sqrt(E_harmless[x_j^2]) + 1e-7
~~~

它近似表示：

> 每单位 harmless writer-output 代价能够换取多少预计的 refusal-margin 降低。

### 8.1 一个重要的代数性质

当 `W_ij != 0` 且分母远大于 `epsilon` 时，权重幅值近似抵消：

~~~text
S_ij ~= max(sign(W_ij) * g_bar_ij, 0)
        --------------------------------
        sqrt(E_harmless[x_j^2])
~~~

因此 ratio 主要按“方向正确的梯度 / harmless 输入尺度”排名，而不是简单偏好大权重。
这正是它比 Taylor-only 更能保护语言能力的一个可能原因，但也带来限制：

- 极小 Wanda 分母会放大 score；
- `epsilon`、fp16 score 存储和跨矩阵尺度可能影响尾部排序；
- ratio 不是二阶重构目标，没有考虑多权重同时删除后的相互作用。

## 9. 阶段 E：其他评分规则与对照

主实验比较以下 ranking rule。

### 9.1 Refusal-aware rules

1. `ratio`：`T_ij / (H_ij + epsilon)`，主规则；
2. `taylor`：只使用 `T_ij`；
3. `edge`：signed actdiff edge：

~~~text
E_ij = max(r_i * W_ij * delta a_j, 0)
delta a_j = mean_harmful(a_j) - mean_harmless(a_j)
~~~

其中 `r_i` 是该 destination layer 的单位 refusal direction。该分数估计单条 edge
对 harmful-vs-harmless activation difference 写向 refusal direction 的贡献，作为
不依赖 response likelihood gradient 的独立复现。

### 9.2 Controls

- `taylor-shuffled`：以 seed 0 对约一半 pair 交换 refusal/compliance 标签；
- `wanda`：剪掉 harmless Wanda score 最小的权重；
- `magnitude`：剪掉绝对值最小的权重；
- `random0/1/2`：三个固定随机 seed。

这些控制分别检验：标签语义、普通能力保持剪枝、权重幅值以及纯随机稀疏性是否足以
解释效果。

## 10. 阶段 F：带 per-matrix cap 的全局排序

直接在 415,236,096 个权重上全局 top-k，可能因某个矩阵的 score scale 特殊而将
剪枝集中到单层。当前实现采用两阶段排名。

### 10.1 每矩阵候选 cap

对每个矩阵只保留其局部 top 10% 候选：

~~~text
C_m = local_top_10_percent(score_m)
~~~

对 `ratio/taylor/edge/shuffled/random` 取最大分；对 `wanda/magnitude` 取最小分。

### 10.2 全局排名

合并所有局部候选后，按对应方向全局排序。预先构造到最大 5% 目标池 sparsity 的
ranking；每个较小 sparsity 使用同一 ranking 的前缀，所以 mask 是嵌套的：

~~~text
K_0.01% subset K_0.05% subset ... subset K_5%
~~~

扫描比例及实际权重数：

| 目标池比例 | 权重数 |
|---:|---:|
| 0.01% | 41,524 |
| 0.05% | 207,618 |
| 0.10% | 415,236 |
| 0.50% | 2,076,180 |
| 1.00% | 4,152,361 |
| 5.00% | 20,761,805 |

实现：[`src/ttsafety/weight_prune.py`](../src/ttsafety/weight_prune.py)。

## 11. 阶段 G：可逆单权重置零

对选中集合 `K`：

~~~text
W'_ij = 0     if (i, j) in K
W'_ij = W_ij  otherwise
~~~

实验使用 context manager：

1. 解析矩阵并缓存选中 flat indices；
2. 精确备份选中权重；
3. context 内将这些值设为零；
4. 完成生成/PPL/KL；
5. 退出后从备份恢复原始值。

不同 sweep cell 不会累积修改。测试另外验证了 context 内选中值确实为零、退出后
权重逐比特恢复。项目没有导出安全被移除的永久模型。

## 12. 阶段 H：评估与选择规则

每个 `rule x sparsity` cell 测量：

- harmful validation refusal rate；
- harmless refusal rate；
- WikiText teacher-forced PPL 及相对变化；
- harmless base-to-pruned prompt-token KL；
- 与 baseline harmless completion 的 token agreement；
- 空回复、乱码、循环和重复等 adverse output rate。

硬约束：

~~~text
harmless refusal <= 0.05
PPL delta        <= 5%
harmless KL      <= 0.10
adverse rate     <= 0.01
random gap       >= 0.10
~~~

稀疏关键权重集合还要求：

~~~text
target-pool sparsity <= 1%
harmful refusal      <= 0.05
~~~

如果多个 cell 通过，选择顺序为：

1. sparsity 更小；
2. harmful refusal 更低；
3. KL 更低；
4. PPL 退化更低。

## 13. 算法伪代码

~~~python
# Candidate pool
writers = residual_writers(
    layers=range(7, 19),
    components=("mlp.down_proj", "self_attn.o_proj"),
)

# A. Signed refusal-margin Taylor numerator
grad_sum = zeros_like(writers)
for instruction, refusal, compliance in caa_pairs:
    logp_r = mean_response_token_logp(instruction, refusal)
    logp_c = mean_response_token_logp(instruction, compliance)
    margin = logp_r - logp_c
    backward(margin)
    grad_sum += gradients(writers)

for W in writers:
    mean_grad[W] = grad_sum[W] / len(caa_pairs)
    taylor[W] = relu(W * mean_grad[W])

# B. Harmless Wanda denominator
second_moment = zeros_per_input_dimension(writers)
token_count = 0
for harmless_prompt in harmless_prompts:
    for W in writers:
        x = writer_input_activations(W, harmless_prompt)
        second_moment[W] += sum_over_tokens(x ** 2)
    token_count += number_of_valid_prompt_tokens(harmless_prompt)

for W in writers:
    rms = sqrt(second_moment[W] / token_count)
    wanda_keep[W] = abs(W) * rms[None, :]
    ratio[W] = taylor[W] / (wanda_keep[W] + 1e-7)

# C. Capped global ranking
candidates = {}
for W in writers:
    candidates[W] = local_top_fraction(ratio[W], fraction=0.10)
ranking = global_sort_descending(union(candidates))

# D. Nested sparsity sweep
for fraction in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
    k = round(fraction * total_target_pool_weights)
    selection = ranking[:k]
    with temporarily_zero_and_exactly_restore(selection):
        evaluate_harmful_refusal()
        evaluate_harmless_refusal()
        evaluate_wikitext_ppl()
        evaluate_harmless_kl_and_quality()

# E. Select the smallest cell satisfying all preregistered limits
~~~

## 14. 计算与存储特性

### 14.1 计算

- Taylor：需要对 247 个 pair 做反向传播，是最主要评分成本；
- Wanda：只需 harmless prompt 前向与 pre-hook 输入统计；
- ranking：对每矩阵先做局部 `topk`，再做候选集合全局 `topk`；
- sweep：每个 cell 需要重新生成 harmful/harmless 输出并计算 PPL/KL。

### 14.2 存储

- 每个目标矩阵的 Taylor/Wanda/edge score 保存为 fp16；
- ranking 只保存矩阵 id 与 flat index；
- 不保存完整模型副本；
- 生成样例和大 tensor 放在 `data/`，汇总指标放在 `results/`。

## 15. 当前结果

主配置 `ratio` 在目标池 0.01% 时：

| 指标 | 结果 |
|---|---:|
| 剪枝权重数 | 41,524 |
| harmful_val refusal | 0.03125 |
| harmless refusal | 0.00625 |
| WikiText PPL delta | +0.0265% |
| harmless KL | 0.00166 |
| 三 seed random refusal | 1.000 / 1.000 / 1.000 |
| adverse output rate | 0 |

Taylor-only 在同一 sparsity 下 refusal 为 0.0156，但 PPL +5.34%、KL 1.54，且
出现循环输出，因此未通过。该结果支持 Wanda denominator 对能力保持的重要性。

signed actdiff edge 在 0.05% 时也达到 refusal 0、PPL +0.61%、KL 0.0935，提供了
不同评分信号下的独立复现。

## 16. 方法假设与限制

1. **一阶局部性**：`T_ij` 假设单权重删除的局部一阶效应能预测多权重联合删除；
   sparsity 增大后交互项和分布漂移会使该近似恶化。
2. **梯度抵消**：先平均有符号梯度偏好跨 pair 一致的 edge，可能漏掉只对某类
   harmful prompt 重要、但在总体上抵消的专门 edge。
3. **Wanda 是局部代理**：`H_ij` 只近似 writer output 变化，不等价于完整模型
   utility loss 或下游能力。
4. **ratio 尺度**：近零分母、fp16 score 和跨矩阵 scale 会影响极端排名；cap
   只缓解集中问题，不能消除校准问题。
5. **坐标依赖**：unstructured scalar sparsity 依赖原始参数基底；等价的旋转重参数化
   可能改变“关键权重”集合。
6. **评估不足**：关键词 refusal 不等于语义安全；当前 Route B 无独立 test，不能
   把 validation 结果写成跨 benchmark 结论。
7. **无推理加速**：普通 dense kernel 不利用这些零值；算法的“稀疏”是机制与存储
   结构概念，不代表实际 latency 降低。

## 17. 实现与复现入口

主要文件：

- [`scripts/score_refusal_weights.py`](../scripts/score_refusal_weights.py)：Taylor、
  shuffled Taylor、Wanda、signed actdiff edge；
- [`scripts/sweep_weight_prune.py`](../scripts/sweep_weight_prune.py)：rule 构造、
  sparsity sweep、评估和最终选择；
- [`src/ttsafety/weight_prune.py`](../src/ttsafety/weight_prune.py)：capped ranking、
  flat-index 选择、可逆置零；
- [`tests/test_weight_prune.py`](../tests/test_weight_prune.py)：排序、cap、置零与恢复测试。

核心命令：

~~~bash
uv run python scripts/score_refusal_weights.py --score taylor
uv run python scripts/score_refusal_weights.py --score taylor-shuffled
uv run python scripts/score_refusal_weights.py --score wanda
uv run python scripts/score_refusal_weights.py --score edge

uv run python scripts/sweep_weight_prune.py --rule ratio,taylor,edge
uv run python scripts/sweep_weight_prune.py --rule taylor-shuffled,wanda,magnitude
uv run python scripts/sweep_weight_prune.py --rule random0,random1,random2
uv run python scripts/sweep_weight_prune.py --finalize
~~~

GPU 实验应继续使用项目约定的 `.gpu.lock`。

## 18. 方法来源与推荐署名

Route B 不是从一篇论文原样复现，而是组合并修改了三类已有方法：

1. SNIP 的一阶 connection sensitivity：
   [`SNIP: Single-shot Network Pruning based on Connection Sensitivity`](https://arxiv.org/abs/1810.02340)；
2. Wanda 的 activation-aware weight importance：
   [`A Simple and Effective Pruning Approach for Large Language Models`](https://arxiv.org/abs/2306.11695)；
3. safety/utility-specific weight isolation：
   [`Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications`](https://arxiv.org/abs/2402.05162)。

准确的方法描述应为：

> We build on first-order connection sensitivity and Wanda-style activation-aware
> preservation, and introduce a signed refusal-versus-compliance Taylor objective,
> a harmless-Wanda cost denominator, and capped global selection for refusal-specific
> individual-weight pruning.

不能声称：首次使用单权重剪枝破坏安全、首次把 SNIP/Wanda 用于安全归因、或首次
发现 safety-critical sparse weights；这些总体方向已有先例。
