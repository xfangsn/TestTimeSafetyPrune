# BLADE（gradient-free signed actdiff edge）：算法定义、相关工作检索与创新性审计

> ⚠️ **历史存档(标注于 2026-08-27):本文记录较早阶段的计划/方法/结果,可能与当前代码与结论脱节。**
> 方法线现已定名 **BLADE** 并引入 best-first ELS(有效层选择);当前定稿的方法与实验结果请以
> [`blade-method.md`](blade-method.md)、[`blade-algorithm.md`](blade-algorithm.md)、
> [`blade-experiment-report.md`](blade-experiment-report.md) 为准。**下文内容保留原貌,未作修订。**

> **方法命名（2026-08-24）**：本方法正式命名为 **BLADE** ——
> **B**ehavioral **L**ocalization via **A**ctivation-**D**ifference **E**dges。
> 全文中的 "(gradient-free) signed actdiff edge" 是其技术描述符，与 BLADE 指
> **同一评分规则** $s_{ij}=[\,r_i W_{ij}(\mu^A_j-\mu^B_j)\,]_+$；本仓库历史文档
> （results-*/plan-*/comparison-* 等）中出现的 "signed actdiff edge" / "edge"
> 均等价于 BLADE，未逐一改名以保留记录原貌。BLADE 从 refusal 推广到多行为定位
> 的工作见 [`related-work-behavior-atlas.md`](related-work-behavior-atlas.md)。

> 文档用途：供其他 AI / 研究者独立 review。
> 调查日期：2026-08-23。
> **勘误（2026-08-23，由独立 review 添加）**：本文中所有 "Orgad et al. 2026" /
> "Orgad 2026" 的署名有误。arXiv:2604.09544 的正确署名是 **Boyi Wei, Kaden Zheng,
> Martin Wattenberg, Peter Henderson, Seraphina Goldfarb-Tarrant, Yonatan Belinkov**
>（Hadas Orgad 仅为 arXiv 提交者）；正确标题为 *Large Language Models Generate
> Harmful Content Using a Distinct, Unified Mechanism*。即该 signed-SNIP 后续工作与
> Wei et al. ICML 2024（2402.05162）是**同一课题组的后续工作**，这一归属加强了
> §11.3 第 2 条的创新性风险判断。
> 审查对象：项目中已有的 `signed_actdiff_edge`，不是 Route B 的
> Taylor/Wanda ratio，也不是后来提出并已得到阴性结果的 CRFP。
> 实现：[`scripts/score_refusal_weights.py`](../scripts/score_refusal_weights.py)、
> [`src/ttsafety/weight_prune.py`](../src/ttsafety/weight_prune.py)。
> 现有结果：
> [`results-weight-level-refusal-editing.md`](results-weight-level-refusal-editing.md)。

## 1. 一页结论

项目中的 gradient-free signed actdiff edge 是下面的单权重分数：

$$
c^{(l,m)}_{ij}
=r^{(l)}_i W^{(l,m)}_{ij}
\left(\mu^{(l,m),H}_j-\mu^{(l,m),U}_j\right),
\qquad
s^{(l,m)}_{ij}=[c^{(l,m)}_{ij}]_+.
$$

其中，$W^{(l,m)}$ 是第 $l$ 层 residual writer $m$ 的输出矩阵，$r^{(l)}$
是该 destination layer 的单位 refusal direction，$\mu^H$ 和 $\mu^U$ 分别是
harmful 与 harmless prompt 最后一个非 padding token 处的 writer 输入均值。
算法删除 $s_{ij}$ 最大的标量权重。

这个公式不是从某一篇论文直接照搬的。它可以被严格理解为：把一个 residual writer
在 harmful--harmless 条件下、沿 refusal direction 的**局部直接输出差**

$$
\Delta z=r^\top W(\mu^H-\mu^U)
$$

逐个 scalar edge 精确展开。对固定输入激活而言，$c_{ij}$ 正好是 edge
$j\rightarrow i$ 对 $\Delta z$ 的可加项；把 $W_{ij}$ 置零会使这个局部量变化
$-c_{ij}$。

截至本次检索：

- 没有找到名称为 `signed actdiff edge` 的既有方法；
- 没有找到同时具备“scalar weight、harmful--harmless 输入激活差、refusal
  direction 投影、保留符号、无需 backward、按正分删权重”全部要素的先行论文；
- 但所有主要原料都有明确先例，因此不能把整个思想描述成凭空产生：
  - contrastive / difference-in-means representation；
  - 把组件输出直接投影到 feature/refusal direction；
  - weight × activation 的标量归因或剪枝；
  - signed weight pruning；
  - 安全行为的 neuron/weight/rank 定位与删除。

最准确的创新性判断是：**它是一个有清楚线性推导的、可能新的组合型评分规则；当前
检索支持“未发现完全等价方法”，不支持无保留地宣称“首次”。** 特别是 2026 年
Orgad 等人的工作已经提出 signed-SNIP scalar-weight pruning 来定位有害生成权重；
本方法与它最关键的区别是使用 forward-only 的局部 refusal-flow 分解，而不是
response NLL 的权重梯度。

## 2. 被审查方法的边界

### 2.1 它是什么

它是一个：

- white-box：需要读模型权重与中间激活；
- forward-only：分数计算只做 harmful/harmless forward；
- gradient-free：不启用参数梯度，不调用 backward、JVP 或 VJP；
- parameter-level：归因和干预单位是单个 $W_{ij}$；
- behavior-directed：目标不是一般语言建模误差，而是写向 refusal direction 的
  harmful--harmless 差异；
- signed：只删除对该差异作正贡献的 edge；
- unstructured：把选中 scalar weights 置零，不删除完整 row、column、head 或
  neuron。

### 2.2 它不是什么

- 不是 Route B 的主规则 Taylor/Wanda ratio；后者需要 response likelihood
  gradient，并用 harmless Wanda cost 归一化。
- 不是 Wanda；Wanda 使用 $|W_{ij}|\lVert X_{:,j}\rVert_2$ 保留一般能力，而这里
  使用带符号的 $r_iW_{ij}\Delta a_j$ 主动删除 refusal-supporting edge。
- 不是 Arditi 等人的 weight orthogonalization；后者对整个 writer 做
  $W'=W-rr^\top W$，是 dense rank-one edit，本方法只把少量原始坐标基权重置零。
- 不是 Edge Attribution Patching；EAP 中的 edge 是组件计算图之间的 activation
  channel，并使用一次 backward；这里的 edge 是线性矩阵中的 scalar connection。
- 不是完整因果效应估计。分数对固定 writer 输入的直接投影是精确的，但不等于删除
  权重后最终生成、refusal rate 或 loss 的精确变化。
- 不是压缩加速算法。普通 dense kernel 不会因为少量离散零权重自动变快。

### 2.3 与 CRFP 的关系

后来的 Route C / CRFP 使用 refusal-vs-compliance teacher-forced response trajectory、
方差/LCB 和 harmless cost。它不是本文公式的同义名。已有实验中 CRFP 没有超过旧的
signed actdiff edge，并按预注册停止条件终止。详情见
[`results-gradient-free-refusal-flow-pruning.md`](results-gradient-free-refusal-flow-pruning.md)。

## 3. 记号与实验对象

对第 $l$ 层、类型为 $m$ 的 residual writer，记：

$$
y^{(l,m)}=W^{(l,m)}a^{(l,m)},
$$

其中：

- $W\in\mathbb{R}^{d_{\mathrm{model}}\times d_{\mathrm{in}}}$；
- $a_j$ 是 writer 的第 $j$ 个输入通道；
- $y_i$ 是写入 residual stream 的第 $i$ 个输出通道；
- $W_{ij}$ 是从 $a_j$ 到 $y_i$ 的 scalar edge；
- $r^{(l)}\in\mathbb{R}^{d_{\mathrm{model}}}$ 是 destination residual site 对齐的
  refusal direction，使用前单位化；
- $H$、$U$ 分别表示 harmful 和 harmless prompt 集合。

当前实例固定为：

- 模型：Llama-3.2-3B-Instruct；
- harmful score 数据：`data/harmful_train.jsonl`，256 条；
- harmless score 数据：`data/harmless.jsonl`，320 条；
- 激活位置：chat template 后每条 prompt 的最后一个非 padding token；
- writer：L7--L18 的 12 个 `mlp.down_proj` 与 12 个
  `self_attn.o_proj`；
- 目标池：415,236,096 个 scalar weights；
- refusal directions：`data/directions/refusal_llama32_3b_instruct.pt` 中每层的
  direction。

这里的 harmful/harmless 数据不要求成对。两组样本数也可以不同，因为算法只使用两组
各自的均值。

## 4. 详细算法

### 4.1 第一步：固定方向的来源与符号

对每个目标层取得 refusal direction $r^{(l)}$，并做：

$$
\hat r^{(l)}=\frac{r^{(l)}}{\lVert r^{(l)}\rVert_2}.
$$

当前 score 脚本只负责单位化，不在运行时重新判断方向符号。因此，方向 artifact 必须
保证“正投影对应更强 refusal”。若 artifact 的符号翻转，全部正/负 edge 的语义也会
翻转。这应作为复现实验的显式 invariant：

$$
\mathbb E_H[\hat r^\top h]-\mathbb E_U[\hat r^\top h] > 0.
$$

更严格的复现还应记录 direction 的训练数据、layer、token position、hash，以及它是否
与 edge-score 数据独立。

### 4.2 第二步：收集 writer 输入均值

对每条 prompt 做一次正常 forward，通过 `forward_pre_hook` 取得各目标 writer 的输入。
令 $t(x)$ 为该 prompt 最后一个非 padding token 的位置，则：

$$
\mu^{H,(l,m)}_j
=\frac{1}{|H|}\sum_{x\in H}a^{(l,m)}_j(x,t(x)),
$$

$$
\mu^{U,(l,m)}_j
=\frac{1}{|U|}\sum_{x\in U}a^{(l,m)}_j(x,t(x)).
$$

定义输入激活差：

$$
\Delta a^{(l,m)}_j
=\mu^{H,(l,m)}_j-\mu^{U,(l,m)}_j.
$$

当前实现按 example 等权；它不保存 token trajectory，也不使用回答 token 或 response
NLL。

### 4.3 第三步：把 direct refusal flow 展开到 scalar edge

writer 的 harmful--harmless 平均输出差为：

$$
\Delta y=W\mu^H-W\mu^U=W\Delta a.
$$

沿 refusal direction 的 scalar direct-flow contrast 为：

$$
\Delta z=\hat r^\top\Delta y
=\hat r^\top W\Delta a.
$$

利用矩阵乘法的双重求和：

$$
\Delta z
=\sum_i\hat r_i\sum_jW_{ij}\Delta a_j
=\sum_{i,j}\underbrace{\hat r_iW_{ij}\Delta a_j}_{c_{ij}}.
$$

因此定义 signed edge contribution：

$$
c_{ij}=\hat r_iW_{ij}\Delta a_j.
$$

这个等式给出三个符号因子的完整解释：

- $\Delta a_j$：harmful prompt 是否比 harmless prompt 更激活输入通道 $j$；
- $W_{ij}$：该输入通道以什么符号写入 residual 输出坐标 $i$；
- $\hat r_i$：输出坐标 $i$ 与 refusal direction 的符号及权重。

三者乘积为正，表示这条 edge 支持 harmful 相对 harmless 的正向 refusal projection；
乘积为负，表示它抵消这个 projection。

### 4.4 第四步：把归因变成删除规则

若只把一个权重置零：

$$
W'_{ij}=0,
$$

并暂时把该 writer 的输入 $a$ 视为固定，则：

$$
\Delta z'-\Delta z=-\hat r_iW_{ij}\Delta a_j=-c_{ij}.
$$

所以若目标是减小 $\Delta z$，只应优先删除 $c_{ij}>0$ 的权重。当前分数为：

$$
s_{ij}=\max(c_{ij},0).
$$

这里的 “signed” 不是说最终 score 仍可为负，而是说**先保留 contribution 的符号来
决定干预方向，再把不符合删除目标的负贡献设为零**。如果一开始取
$|r_iW_{ij}\Delta a_j|$，就会混入原本抑制 refusal flow 的 weights，删除它们可能使
refusal 更强。

### 4.5 第五步：带 per-matrix candidate cap 的全局排序

当前实现不是简单地把 415M 个分数一次性全局排序：

1. 每个矩阵保留局部 top 10% 作为候选；
2. 合并所有矩阵候选；
3. 按 $s_{ij}$ 从大到小取全局 top-$K$；
4. 不同 sparsity 使用同一 ranking 的前缀，保证 masks 嵌套。

若目标池比例为 $q$，则：

$$
K=\operatorname{round}(qP),\qquad P=415{,}236{,}096.
$$

已使用的关键点：

| 目标池比例 | 删除权重数 |
|---:|---:|
| 0.01% | 41,524 |
| 0.05% | 207,618 |

“每矩阵 top 10%”是 candidate cap，不保证最终选中权重在各矩阵均匀分布；某个矩阵仍可
贡献大量全局 top-$K$，上限是它自己的 10%。

数学定义最好写成先要求 $c_{ij}>0$ 再 top-$K$。当前实现实际是
`clamp_min(0)` 后取 top-$K$；只要正分数候选多于 $K$，两者等价。正式复现应记录每个矩阵
的 positive fraction，并在正分数不足时停止，而不是让零分数的任意 tie 进入 mask。

### 4.6 第六步：可逆置零与评估

对选中的集合 $\mathcal K$：

$$
W'_{ij}=\begin{cases}
0,&(i,j)\in\mathcal K,\\
W_{ij},&\text{otherwise}.
\end{cases}
$$

项目实现会备份选中值，在 context 中置零，评估完成后逐值恢复。主要评估包括 harmful
refusal、harmless refusal、WikiText PPL、harmless prompt-token KL 和输出质量标志。

### 4.7 伪代码

```text
Inputs:
  model M
  harmful prompts H, harmless prompts U
  target residual writers T
  layer-wise refusal directions {r_l}
  target sparsity q, per-matrix candidate cap gamma = 0.10

for each layer l:
  r_l <- r_l / ||r_l||_2
  assert sign(r_l) means harmful/refusal-positive

mu_H <- forward_mean_writer_inputs(M, H, last_nonpad_token=True)
mu_U <- forward_mean_writer_inputs(M, U, last_nonpad_token=True)

for each writer (l, m) in T:
  delta <- mu_H[l,m] - mu_U[l,m]
  C <- outer(r_l, delta) elementwise_mul W[l,m]
  S <- max(C, 0)
  candidates[l,m] <- local_top(S, gamma * numel(S))

K <- round(q * total_target_weights)
selected <- global_top_k(union(candidates), K)
assert every selected raw contribution is positive

temporarily set selected W_ij to zero
evaluate refusal, harmfulness, benign utility, PPL, KL, output quality
restore exact original values
```

## 5. “精确”到什么程度

### 5.1 精确成立的命题

在给定 $W$、$\hat r$、$\mu^H$、$\mu^U$ 的线性 writer 上：

$$
\sum_{ij}c_{ij}=\hat r^\top W(\mu^H-\mu^U)
$$

严格成立。单 edge 置零对这个 frozen-input 局部投影差的变化也严格为 $-c_{ij}$。

### 5.2 不精确的部分

真实模型删除权重后：

- 同层和后续层的激活会变化；
- RMSNorm/LayerNorm 的缩放会变化；
- attention pattern、MLP gating 和 token generation path 可能改变；
- 多个 weights 同时删除会产生交互；
- refusal direction 只是一维 proxy；
- 最后一个 prompt token 的差异不等于整段 response 的行为差异。

因此 $s_{ij}$ 是**解析的局部 direct-effect proxy**，不是对最终 refusal metric 的
unbiased causal estimator。后续实际置零实验才是因果干预；分数本身不是。

## 6. 计算与内存开销

设 score 数据共 $N=|H|+|U|$ 条，目标池有 $P$ 个 weights。

- forward 代价：约为 $N$ 次 prompt prefill；无 response teacher forcing；
- 解析评分：$O(P)$ 次逐元素乘法；
- 激活统计额外状态：每个 writer 一个 $d_{\mathrm{in}}$ 长度的累计向量，不需要保存
  全部样本激活；
- score artifact：当前以 fp16 保存，约 $2P$ bytes；415,236,096 个分数约 0.77 GiB，
  不含容器和排序临时内存；
- 排序：每矩阵先 top 10%，再全局 top-$K$，临时 candidate values/indices 可能成为
  CPU 内存峰值；
- 相比 SNIP/Taylor：不保存梯度，也没有 backward activation/gradient 开销；
- 相比逐 weight ablation：不需要 $O(P)$ 次模型 forward。

它降低的是**定位权重的实验成本**，不是部署时推理成本。当前方法在 RTX 5090 上已对
3B 模型运行；文档没有仅凭复杂度外推更大模型的实证结论。

## 7. 当前实证证据

在相同的 L7--L18 `down_proj + o_proj` 目标池上：

| score / sparsity | 删除数 | harmful_val refusal | PPL 变化 | harmless KL | 判定 |
|---|---:|---:|---:|---:|---|
| signed edge / 0.01% | 41,524 | 0.125 | +0.09% | 0.0381 | 有明显信号，未达 refusal≤0.05 |
| signed edge / 0.05% | 207,618 | 0.000 | +0.61% | 0.0935 | 通过预注册硬约束 |
| Taylor/Wanda / 0.01% | 41,524 | 0.031 | +0.03% | 0.0017 | Route B 主规则，更优 |
| random / ≤1% | — | 1.000 | — | — | 无 refusal 下降 |

这支持：forward-only direct-flow score 不是 magnitude、Wanda-smallest 或随机稀疏性能够
解释的；它定位到了与 refusal 行为相关的 scalar weights。

但证据等级仍有限：

- signed edge 只在 `harmful_val` 上报告，没有新的 independent harmful test；
- refusal 主指标是关键词 judge，非拒绝不等于安全、有帮助或事实正确；
- 只有一个 3B instruct model；
- 方向与 score 数据的独立程度需要在 publication protocol 中重新锁定；
- 0.05% 是目标池比例，不是全模型比例，不能直接与论文中的全模型 sparsity 比数值；
- 结果证明所选集合有因果影响，不证明每个入选 edge 都有独立、可重复的因果作用。

## 8. 系统相关工作检索协议

### 8.1 检索目标

优先寻找完全等价的方法；若没有，则按以下三层纳入：

1. **数学最近邻**：目标方向投影、direct attribution、weight × activation 分解；
2. **算法最近邻**：gradient-free scalar pruning、signed scalar pruning、edge
   attribution；
3. **问题最近邻**：安全/refusal neuron、weight 或 rank 的定位、剪枝和 patching。

### 8.2 数据源

本次查阅了：

- arXiv 题录与论文 HTML/PDF；
- NeurIPS、ICLR、PMLR、PLOS 的正式论文页面；
- OpenReview 的论文页面/PDF；
- 已知近邻论文的参考文献与 related-work 链接。

比较结论尽量依据论文原文而非博客或二手摘要。2026 年尚未正式同行评议的 arXiv 工作
会明确标注为 preprint。

### 8.3 代表性查询词

使用了名称、精确短语、公式结构与问题域查询，包括：

```text
"signed actdiff edge"
"signed activation difference" edge attribution neural network
"direct feature attribution" refusal direction
"weight times activation" pruning attribution signed
"activation difference" "weight pruning" transformer
"projected contribution" scalar weights transformer
refusal direction pruning scalar weights
safety alignment parameter pruning weight
contrastive activation weight pruning LLM safety
edge attribution patching transformer
```

还分别检索了 Arditi refusal direction、CAA/RepE、DFA/DLA、LRP、DeepLIFT、Wanda、
SNIP、EAP/AtP*、Wei safety pruning、Safety Neurons 及 2025--2026 新安全剪枝工作。

### 8.4 纳入和排除标准

纳入至少满足一项的工作：

- 单个标量 weight 的 importance 或 pruning；
- contrastive / reference activation difference；
- component output 对目标 direction/logit/feature 的直接投影；
- computational-graph edge attribution；
- safety/refusal 机制的 neuron/weight/rank intervention。

只做常规量化、蒸馏、结构压缩，且没有行为定向或归因联系的论文不进入主比较表。

### 8.5 检索限制

“没有找到”不是数学上的不存在证明：

- 搜索引擎不能保证覆盖所有 workshop、学位论文、未公开稿和代码仓库；
- 同一公式可能使用不同符号或名称；
- 本次未完成付费数据库的全文布尔检索和完整 citation-graph snowballing；
- 2026 年文献仍快速变化。

所以本文只作可复核的 prior-art audit，不作专利式 novelty search。投稿前应再做一次
Google Scholar / Semantic Scholar / OpenAlex citation graph 检索，并检查下文核心论文
自 2026-08-23 之后的 citing papers。

## 9. 相关工作逐类比较

### 9.1 Refusal direction 与直接方向投影

#### Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*

论文用 harmful--harmless difference-in-means 提取 refusal direction，进行 activation
addition、directional ablation，并提出：

$$
W'=W-\hat r\hat r^\top W.
$$

它还用 direct feature attribution，把 attention head / MLP component 的输出投影到
refusal direction，分析哪些组件写入该方向。

与本文最直接的数学关系是：Arditi 的 component-level 量可以写成
$\hat r^\top W\Delta a$；本文再把它展开为
$\sum_{ij}\hat r_iW_{ij}\Delta a_j$，并把正项用于 sparse scalar pruning。

差异：Arditi 没有提出本文的 scalar score、正部分 top-$K$ 或 sparse edge deletion；
其 weight edit 是对每个 writer 的 dense rank-one orthogonalization。

来源：[arXiv:2406.11717](https://arxiv.org/abs/2406.11717)。

#### Kissane et al. (2024), *Interpreting Attention Layer Outputs with Sparse Autoencoders*

该工作把 SAE feature pre-activation 写成各 attention head 贡献的和，并称为 Direct
Feature Attribution。它清楚展示了“线性输出可按输入组件精确分解”的思路。

差异：单位是 head / source-position 对 SAE feature 的贡献，不是 pretrained model
中的 scalar weight；目标是解释 feature activation，不是删除 refusal-supporting
weights。

来源：[arXiv:2406.17759](https://arxiv.org/abs/2406.17759)。

#### Makelov et al. (2024), *Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control*

该工作也是 Arditi 所引 DFA 背景之一，讨论 SAE feature 的可解释与控制评价。

差异：研究对象是 SAE feature 与 component attribution，不给出本文的 scalar
parameter pruning rule。

来源：[arXiv:2405.08366](https://arxiv.org/abs/2405.08366)。

### 9.2 Contrastive representation / activation addition

#### Zou et al. (2023/2025), *Representation Engineering*

RepE 把 population-level representation 作为监测和操纵高级概念的中心对象，覆盖
harmlessness 等安全相关概念。

差异：它提供 direction-level 的上游范式，不做本文的 scalar writer-edge 分解。

来源：[arXiv:2310.01405](https://arxiv.org/abs/2310.01405)。

#### Rimsky et al. (2023), *Steering Llama 2 via Contrastive Activation Addition*

CAA 用正/负 contrastive examples 的 activation mean difference 构造 steering
vector，并在推理时加到 residual stream。

差异：CAA 操纵 activation direction，不读取 $W_{ij}$，也不做 weight pruning。

来源：[arXiv:2312.06681](https://arxiv.org/abs/2312.06681)。

### 9.3 Weight × activation 与 reference-difference attribution

#### Sun et al. (2023/ICLR 2024), Wanda

Wanda 的 score 为：

$$
I^{\text{Wanda}}_{ij}=|W_{ij}|\lVert X_{:,j}\rVert_2,
$$

并在每个 output 内删除最小分 weights。它无需 retraining/backward，是本文在工程结构上
最接近的通用 LLM pruning baseline。

共同点：scalar weight、weight × input activation、forward-only、one-shot pruning。

差异：Wanda 使用无符号 activation norm，目标是保留整体语言建模能力，删除“不重要”
weights；本文使用有符号 contrastive mean 和 refusal direction，删除“支持目标行为”
的最高分 weights。

来源：[arXiv:2306.11695](https://arxiv.org/abs/2306.11695)。

#### Bach et al. (2015), Layer-Wise Relevance Propagation

在线性层中，LRP 的局部消息以 $a_jW_{ij}$ 一类 pre-activation contribution 为基础，
再按 conservation rule 把 output relevance 逐层向输入传播。

共同点：局部连接的 weight × activation contribution 和符号都有解释意义。

差异：LRP 从最终 prediction relevance 反向传播，通常带归一化/稳定项并递归跨层；本文
不传播最终 output relevance，只对一个预先给定的内部方向做单层精确展开，并直接用于
weight deletion。

来源：[PLOS ONE](https://doi.org/10.1371/journal.pone.0130140)。

#### Shrikumar et al. (2017), DeepLIFT

DeepLIFT 相对 reference activation 传播 contribution differences，并可区分正、负
贡献。

共同点：contrast/reference difference 和 signed contribution。

差异：DeepLIFT 需要 modified backward-style propagation，把最终 prediction 归因到
输入；本文只收集 forward moments，不传播 output credit，也不产生 input attribution。

来源：[PMLR 70](https://proceedings.mlr.press/v70/shrikumar17a.html)。

### 9.4 Scalar-weight pruning 与 signed pruning

#### Lee et al. (2018/ICLR 2019), SNIP

SNIP 用 connection sensitivity / first-order gradient 给 individual weights 排名，是
后续安全权重剪枝的重要基础。

差异：SNIP 的核心信号来自 loss gradient；本文的核心信号来自固定内部方向与 forward
activation difference。

来源：[arXiv:1810.02340](https://arxiv.org/abs/1810.02340)。

#### Wei et al. (2024), *Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications*

Wei 等人使用 safety 与 utility calibration data，分别以 SNIP/Wanda 给每个
$W_{ij}$ 评分，再用集合差分定位其术语中的 safety-critical “weight neurons”；另以
ActSVD 和正交投影做 rank-level intervention。这里的 “weight neuron” 实际是单个
weight entry，论文报告可用约 3% parameter-level region 或约 2.5% rank 破坏安全并
保留相当 utility。

共同点：安全行为定向、weight importance、对 utility 的保护、剪枝干预。

差异：其主要安全隔离是每个 output row 内的 safety/utility top-set difference；SNIP
对 $|W_{ij}\nabla_{W_{ij}}\mathcal L|$ 先取绝对值，依赖 safety-response NLL gradient，
不区分促进或抑制 refusal 的符号，也不使用 refusal direction 投影。本文同样排序
scalar entries，但使用全局（带 per-matrix candidate cap）的 signed direct-flow score。

来源：[arXiv:2402.05162](https://arxiv.org/abs/2402.05162)。

#### Orgad et al. (2026), *Large Language Models Generate Harmful Responses Using a Distinct Mechanism, Shared Across Harm Types*

这是本次检索中**问题和干预单位最接近**的新工作。它对 individual scalar weights
使用 signed SNIP：

$$
I(W_{ij},x)=W_{ij}\nabla_{W_{ij}}\mathcal L(x),
$$

并只剪掉删除后会提高 harmful-response loss、即促进有害生成的符号一侧；同时用 benign
数据的 absolute SNIP top set 做 preservation exclusion。论文报告约全模型 0.0005%
的 weights 可大幅降低 harmful generation，并检验跨 harm category 迁移。当前为 2026
arXiv preprint。

共同点：scalar weights、保留符号、行为特异、主动剪掉促进目标行为的 weights、benign
preservation 和真实 pruning intervention。

关键差异：Orgad 的 score 是最终 harmful response NLL 的一阶 Taylor 梯度，需要一个
forward-backward pass；本文是 $r_iW_{ij}\Delta a_j$ 的局部直接分解，不需要 harmful
completion、loss 或 backward。两者回答的问题也不同：前者定位 harmful-content
generation capability，后者定位 aligned model 的 refusal flow。

因此，2026 年之后不能把“signed scalar-weight behavioral pruning”本身当作本文创新点；
可讨论的独特性是**gradient-free contrastive direct-flow score**及其 refusal-specific
实现。

来源：[arXiv:2604.09544](https://arxiv.org/abs/2604.09544)。

### 9.5 Computational-graph edge attribution

#### Syed et al. (2023), Edge Attribution Patching

EAP 用 clean/corrupted activation difference 与下游 metric gradient 的点积近似 edge
activation patching：两次 forward 加一次 backward 后，对计算图 edge 排名并保留 circuit。

共同点：contrastive activation、edge-level score、线性化、top-$K$。

差异：EAP 的 edge 是 attention head/MLP 等 nodes 之间的 computational dependency，
不是矩阵中的 $W_{ij}$；它需要 metric gradient，目标是保留足以执行任务的 circuit，而
本文删除促进 refusal flow 的参数。

来源：[arXiv:2310.10348](https://arxiv.org/abs/2310.10348)。

#### Kramár et al. (2024), AtP*

AtP* 系统研究如何用 attribution patching 高效定位 LLM behavior components，并修复
gradient saturation/cancellation 导致的 false negatives。

差异与 EAP 类似：组件/计算图归因、需要 backward，不是 scalar parameter direct-flow
分解。

来源：[arXiv:2403.00745](https://arxiv.org/abs/2403.00745)。

### 9.6 Safety neuron 与 differential-activation pruning

#### Chen et al. (2024/NeurIPS 2025), *Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons*

该方法比较 safety-aligned 与 unaligned model 在完整 generation 上的 MLP intermediate
neuron activations，以 RMS difference 定位约 5% safety neurons，再用 dynamic
activation patching 检验因果作用。

共同点：forward activation difference、安全行为、内部定位和干预。

差异：对比的是两个模型而非 harmful/harmless inputs；score 是 unsigned RMS；单位是
MLP neuron；干预是动态 activation patching，不是 direction-aware scalar weight
pruning。

来源：[arXiv:2406.14144](https://arxiv.org/abs/2406.14144)。

#### Martra et al. (2026), *Fairness Pruning*

这篇 2026 preprint 用 minimally contrastive demographic prompts 的 differential
activations 定位 GLU-MLP neurons，再做 neuron zeroing。作者还观察到 unsigned score
会混合相反方向的 neurons，引起双向 bias destabilization。

共同点：forward-only、contrastive activation、行为定向 pruning；其对 unsigned
selection 的问题也间接支持保留符号的重要性。

差异：任务是 demographic bias；单位是 neuron；没有 refusal direction 或
$r_iW_{ij}\Delta a_j$ 的 scalar edge 展开。

来源：[arXiv:2607.28319](https://arxiv.org/abs/2607.28319)。

SafeNeuron 与 Mask2Shield 等 2026 工作主要研究如何通过训练把安全表征分散化、抵御
neuron-pruning attack；它们证明安全稀疏性的防御问题正在快速发展，但不是本文 score 的
算法先例：
[SafeNeuron](https://arxiv.org/abs/2602.12158)、
[Mask2Shield](https://arxiv.org/abs/2607.23015)。

## 10. 关键属性对照矩阵

| 方法 | scalar $W_{ij}$ | contrastive activation | target direction | 保留符号决定删除侧 | 无 backward | 实际 weight pruning |
|---|---:|---:|---:|---:|---:|---:|
| 本文 signed actdiff edge | **是** | **是** | **refusal $r$** | **是** | **是** | **是** |
| Arditi DFA / Ortho | 否，component/dense matrix | 是 | refusal $r$ | 部分 | 是 | dense edit |
| Wanda | 是 | 否，activation norm | 否 | 否 | 是 | 是 |
| LRP | 可产生连接消息 | reference-dependent | output relevance | 是 | 否，relevance backward | 否 |
| DeepLIFT | 否，主要归因 input | 是 | final output | 是 | 否，modified backward | 否 |
| EAP / AtP* | 否，计算图 edge | 是 | downstream metric | 通常取绝对值/可选 | 否 | circuit pruning/masking |
| Wei et al. 2024 | **是**（论文称 weight neuron） | safety vs utility datasets | loss/output | SNIP 取绝对值 | SNIP 否；Wanda 是 | **scalar weight** / rank |
| Orgad et al. 2026 | **是** | 不用内部 actdiff | harmful NLL | **是** | **否** | **是** |
| Safety Neurons | 否，MLP neuron | 两模型 activation difference | 否 | 否，RMS | 是 | activation patching |
| Fairness Pruning | 否，MLP neuron | 是 | 否 | 否，unsigned | 是 | neuron zeroing |

没有一行已有工作与本文在六个属性上完全相同。

## 11. 创新性拆解

### 11.1 不能主张为新的部分

- harmful--harmless difference-in-means direction；
- contrastive activation addition / representation engineering；
- component output 对 feature/refusal direction 的直接投影；
- weight × activation 的局部贡献；
- gradient-free one-shot scalar pruning；
- signed scalar-weight pruning；
- 用 safety 与 utility 信号寻找稀疏安全相关参数；
- 通过实际置零测试因果行为变化。

### 11.2 本次未找到等价先例的组合

对 residual writer 的 harmful--harmless input mean difference 做：

$$
\underbrace{\hat r^\top W\Delta a}_{\text{component-level direct contrast}}
=\sum_{ij}
\underbrace{\hat r_iW_{ij}\Delta a_j}_{\text{scalar signed edge}}
$$

并把正项直接作为无需 backward 的 refusal-specific sparse deletion ranking。

这个贡献更像一个**解析分解 + 干预规则**，而不是新优化器或新网络结构。其概念创新性
应评价为中等、组合型；若要形成强论文贡献，真正需要强化的是：

- 跨模型、跨数据的稳定复现；
- 与 signed SNIP、Wei set-difference、Wanda、Arditi Ortho 的公平比较；
- 对 score 与真实 deletion effect 相关性的统计验证；
- 新的独立 held-out harmfulness/utility 评估；
- 说明 forward-only 在显存、时间和可扩展性上的可测优势；
- 解释为什么 scalar positive contributions 呈极稀疏、跨层或跨组件集中。

### 11.3 当前创新性风险

1. 审稿人可能把它视为 DFA 的逐坐标展开，认为公式“自然但增量”。
2. Orgad et al. 2026 已覆盖 signed scalar behavioral pruning，会削弱宽泛 novelty claim。
3. LRP/DeepLIFT 提供了 reference-difference signed decomposition 的更一般框架；必须说明
   本方法不做全网 relevance propagation，而是利用 residual writer 的特定线性结构。
4. 只有单模型 val 结果时，容易被评价为有趣 proxy 而不是普适算法。
5. 删除 refusal 并不等价于定位 safety；若用“安全权重”表述，会被要求 harmful-content
   judge、jailbreak 和 capability 评估。

## 12. 建议的论文表述

### 12.1 可以使用的谨慎表述

中文：

> 我们从 residual writer 沿 refusal direction 的 harmful--harmless 直接输出差出发，
> 将其精确分解为 scalar contributions $r_iW_{ij}\Delta a_j$，并以正贡献作为
> forward-only 的单权重删除分数。根据截至 2026-08-23 的相关工作检索，我们未发现
> 同时使用 contrastive writer-input mean、refusal-direction projection 和 signed
> scalar pruning 的既有方法；但各组成部分分别与 difference-in-means representation、
> direct feature attribution、Wanda/LRP 和 signed-SNIP pruning 有密切联系。

英文：

> We derive a forward-only scalar edge score by exactly decomposing the direct
> harmful-versus-harmless writer projection onto a precomputed refusal direction,
> $r^\top W\Delta a=\sum_{ij}r_iW_{ij}\Delta a_j$, and prune the largest positive
> terms. In a literature search conducted through 23 August 2026, we did not identify
> prior work combining contrastive writer-input means, refusal-direction projection,
> signed scalar scoring, and gradient-free weight pruning, although each ingredient is
> closely related to established work on contrastive representations, direct feature
> attribution, activation-aware pruning, relevance propagation, and signed SNIP.

### 12.2 暂时不要使用的表述

- “the first gradient-free safety pruning method”；
- “the first signed scalar-weight pruning method”；
- “精确计算每个 weight 对 refusal 的因果贡献”；
- “删除 0.05% 全模型参数”；
- “找到了模型的安全权重”；
- “比所有 gradient-based 方法更好”；
- “非拒绝输出证明模型失去安全对齐”。

更准确的术语是 `local direct contribution`、`refusal-flow proxy`、
`target-pool sparsity` 和 `val-only causal intervention evidence`。

## 13. 供其他 AI / reviewer 检查的问题

### 13.1 数学正确性

1. $r$ 是否与 writer 输出处在同一 residual basis，是否存在 pre/post-norm 对齐错误？
2. $r$ 的正号是否被独立验证？
3. `o_proj` 和 `down_proj` hook 获取的是否确实是矩阵右乘前的输入？
4. 最后非 padding token 的索引在 right padding 和 chat template 下是否正确？
5. toy linear test 是否验证 $\sum c_{ij}=r^\top W\Delta a$？
6. 单 edge deletion test 是否验证局部 direct-flow 变化为 $-c_{ij}$？
7. fp16 score 是否改变 top-$K$ tie/ranking，尤其在零附近？

### 13.2 归因有效性

1. top edge score 与逐 edge 或 grouped deletion 的真实 refusal-margin 变化相关多强？
2. 输入均值会不会因 pair cancellation 掩盖 subtype-specific edges？
3. 只看最后一个 prompt token 是否过度捕捉 chat-template/refusal-prefix 特征？
4. score 是否集中在少数 input columns / output rows，从而本质上仍是 neuron effect？
5. 多 edge 同时删除时，additivity 在多大 sparsity 开始失效？
6. 换成 response trajectory、median、trimmed mean 或 sign-consistency 后结论是否稳定？

### 13.3 对照实验

必须至少比较：

- random；
- magnitude-smallest；
- Wanda-smallest；
- $|W_{ij}\Delta a_j|$，不使用 $r$；
- $|r_iW_{ij}|$，不使用 contrast；
- absolute $|r_iW_{ij}\Delta a_j|$；
- sign-flipped $r$；
- label-shuffled harmful/harmless；
- Arditi dense orthogonalization；
- Wei 2024 SNIP set-difference；
- Orgad 2026 signed SNIP，尽量在同一 scalar pool、数据与 sparsity 下重实现；
- Route B Taylor/Wanda ratio。

其中最重要的新 baseline 是 Orgad 2026 signed SNIP。它可回答：本文的 forward-only
direct proxy 以多少效果换来了多少显存/时间节省。

### 13.4 实验与结论边界

1. direction/score/selection/test 是否严格分离？
2. 是否有全新 external holdout，而不是继续复用 `harmful_val` 或旧 test？
3. 是否用 StrongREJECT/HarmBench classifier 或人工双盲标注区分非拒绝、软拒绝与真实
   harmful compliance？
4. harmless 保持是否覆盖 instruction following、reasoning、knowledge 和长文本，而不只
   是 PPL/KL？
5. 是否跨至少三个 model families 和 sizes？
6. top-$K$ 是否跨不同 score sample seeds 稳定，Jaccard 和行为效果如何？
7. 是否报告全模型比例和目标池比例，避免把两者混淆？
8. 是否明确该研究是安全机制脆弱性分析，而不是可部署的安全增强方法？

## 14. 建议的下一轮最小实验包

为让方法经得住外部 AI review，优先级建议如下：

1. **代码级 invariant**：补 direction sign、正分数充足性、toy decomposition 和
   single-edge deletion tests。
2. **同池 signed-SNIP baseline**：按 Orgad 2026 的 signed loss-gradient 规则，在同一
   L7--L18 writer pool 和相同数据预算上比较效果、峰值显存和 wall time。
3. **score--effect correlation**：按 score 分桶，随机抽 grouped edges 做 deletion，报告
   direct projection、refusal margin 与最终 behavior 的 Spearman 相关。
4. **新 held-out**：锁定未使用 harmful set；同时使用 refusal、harmfulness、coherence
   三类指标。
5. **跨模型复现**：至少加入一个 7B/8B 和另一架构家族；固定而不是重新手调 layer pool
   的规则。
6. **机制分析**：报告 layer/component/row/column concentration，测试 scalar sparsity
   是否只是少数 neurons 的碎片化表现。

在这六项完成前，最合适的研究定位是“一个有理论可解释性和单模型初步因果证据的
gradient-free refusal-edge proxy”。

## 15. 参考文献

1. Arditi, A. et al. (2024). *Refusal in Language Models Is Mediated by a Single
   Direction*. [arXiv:2406.11717](https://arxiv.org/abs/2406.11717).
2. Rimsky, N. et al. (2023). *Steering Llama 2 via Contrastive Activation Addition*.
   [arXiv:2312.06681](https://arxiv.org/abs/2312.06681).
3. Zou, A. et al. (2023; rev. 2025). *Representation Engineering: A Top-Down Approach
   to AI Transparency*. [arXiv:2310.01405](https://arxiv.org/abs/2310.01405).
4. Kissane, C. et al. (2024). *Interpreting Attention Layer Outputs with Sparse
   Autoencoders*. [arXiv:2406.17759](https://arxiv.org/abs/2406.17759).
5. Makelov, A. et al. (2024). *Towards Principled Evaluations of Sparse Autoencoders for
   Interpretability and Control*.
   [arXiv:2405.08366](https://arxiv.org/abs/2405.08366).
6. Sun, M. et al. (2023/ICLR 2024). *A Simple and Effective Pruning Approach for Large
   Language Models*. [arXiv:2306.11695](https://arxiv.org/abs/2306.11695).
7. Lee, N. et al. (2018/ICLR 2019). *SNIP: Single-shot Network Pruning based on
   Connection Sensitivity*. [arXiv:1810.02340](https://arxiv.org/abs/1810.02340).
8. Bach, S. et al. (2015). *On Pixel-Wise Explanations for Non-Linear Classifier
   Decisions by Layer-Wise Relevance Propagation*.
   [PLOS ONE](https://doi.org/10.1371/journal.pone.0130140).
9. Shrikumar, A. et al. (2017). *Learning Important Features Through Propagating
   Activation Differences*. [PMLR 70](https://proceedings.mlr.press/v70/shrikumar17a.html).
10. Syed, A. et al. (2023). *Attribution Patching Outperforms Automated Circuit
    Discovery*. [arXiv:2310.10348](https://arxiv.org/abs/2310.10348).
11. Kramár, J. et al. (2024). `AtP*`: An Efficient and Scalable Method for Localizing
    LLM Behaviour to Components*. [arXiv:2403.00745](https://arxiv.org/abs/2403.00745).
12. Wei, B. et al. (2024). *Assessing the Brittleness of Safety Alignment via Pruning
    and Low-Rank Modifications*.
    [arXiv:2402.05162](https://arxiv.org/abs/2402.05162).
13. Chen, J. et al. (2024/NeurIPS 2025). *Towards Understanding Safety Alignment: A
    Mechanistic Perspective from Safety Neurons*.
    [arXiv:2406.14144](https://arxiv.org/abs/2406.14144).
14. Wei, B., Zheng, K., Wattenberg, M., Henderson, P., Goldfarb-Tarrant, S.,
    Belinkov, Y. (2026). *Large Language Models Generate Harmful Content Using a
    Distinct, Unified Mechanism*.
    [arXiv:2604.09544](https://arxiv.org/abs/2604.09544).
    （本文早先版本误记为 "Orgad, H. et al."，见顶部勘误。）
15. Martra, P. et al. (2026). *Fairness Pruning: Locating Demographic Bias in GLU-MLP
    Layers via Differential Activations*.
    [arXiv:2607.28319](https://arxiv.org/abs/2607.28319).
16. Wang, Z. et al. (2026). *SafeNeuron: Neuron-Level Safety Alignment for Large
    Language Models*. [arXiv:2602.12158](https://arxiv.org/abs/2602.12158).
17. JinCheng, Y. et al. (2026). *Mask2Shield: Strengthening LLM Safety against
    Neuron-Pruning Attacks*. [arXiv:2607.23015](https://arxiv.org/abs/2607.23015).

## 16. 最终审计结论

`signed_actdiff_edge` 最清楚的学术谱系是：

```text
contrastive / difference-in-means activations
                 +
direct feature attribution to a refusal direction
                 +
scalar weight × activation decomposition
                 +
sign-aware top-k deletion
                 =
gradient-free signed actdiff edge
```

它不应被归功于某一篇单独论文，也不应被包装成与所有 prior art 无关。根据本次检索，
最合理的出处说明是：**refusal direction 与 component projection 直接继承 Arditi et al.
的机制框架；scalar activation-aware pruning 与 Wanda/LRP 有结构亲缘；signed behavioral
weight pruning 的强最近邻是 Orgad et al. 2026；本文自己的具体步骤是把
$r^\top W\Delta a$ 精确展开到 scalar weights，并用正项做 forward-only sparse
deletion。**

只要论文把这个谱系、局部精确性和全局因果局限写清楚，方法是可以 defend 的；它的最终
创新性强弱将主要取决于跨模型验证、与 signed SNIP 的正面对照，以及独立 safety
evaluation，而不只取决于公式本身。
