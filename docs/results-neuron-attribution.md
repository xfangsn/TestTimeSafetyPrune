# 拒绝翻转中的 Neuron 归因与消融实验结果（N0–N3、N6–N7）

> ⚠️ **历史存档(标注于 2026-08-27):本文记录较早阶段的计划/方法/结果,可能与当前代码与结论脱节。**
> 方法线现已定名 **BLADE** 并引入 best-first ELS(有效层选择);当前定稿的方法与实验结果请以
> [`blade-method.md`](blade-method.md)、[`blade-algorithm.md`](blade-algorithm.md)、
> [`blade-experiment-report.md`](blade-experiment-report.md) 为准。**下文内容保留原貌,未作修订。**

> 对应计划：`docs/plan-neuron-attribution.md`。前置实验（拒绝方向 steering）
> 见 `docs/results-refusal-steering.md`。所有数字均来自 `results/` 下的 JSON 产物。
> 核心结论：**拒绝行为在 neuron 层面是弥散的**——强形式的「关键 neuron」假说不成立。

## 1. 目标与实验设计

核心问题：activation steering 翻转拒绝行为时，哪些 neuron 的激活被显著改变？
它们是否构成拒绝行为的关键节点（不 steering、直接移除这些 neuron 是否也会
降低拒绝率）？

实验分三步：

1. **翻转搜索（N1）**：harmful_train 256 条，steering 固定 L8（mean-diff 拒绝
   方向，jailbreak 符号 α<0，raw 单位向量模式），逐条二分搜索最小翻转强度 α*
   （|α|∈[0,12]，容差 0.25，关键词 judge 判定）。
2. **Neuron 归因（N2）**：对成功翻转样本，teacher-forcing 同一条 steered
   completion，分别在无/有 steering（α=α*，L8）下前向，取回复 token 区间的
   MLP post-SwiGLU 激活均值（28×8192），按三种归一化变体累加
   importance[l][n] += |p1−p2|：
   - softmax（主）：p = softmax(x)，x = 层内回复均值向量；
   - l1（稳健）：p = |x| / ‖x‖₁；
   - z（稳健）：|zscore(x1−x2)|（对 8192 维做 z-score，上游零差层跳过）。
   另记录残差流（28×3072，l1 变体）作辅助视图。
3. **消融验证（N3）**：不 steering，直接把 top-k 重要 neuron 的 MLP 中间激活
   置零（down_proj pre-hook），测拒绝率/过拒绝/ppl，与同数量随机 neuron 对照。

数据分工：neuron 发现用 harmful_train（256），k 选择用 harmful_val（64），
最终消融在 harmful_test（200）上跑一次。环境同前：Llama-3.2-3B-Instruct
bf16，RTX 5090，torch 2.11.0+cu128，transformers 5.15.1。

## 2. N1：翻转 α* 搜索

`results/flip_alpha.json` + `results/flip_alpha_hist.png`，逐样本结果
`data/flip_alphas.jsonl`。

- 基线（α=0）拒绝 248/256（8 条本来就服从，排除出后续分析）；
- **248/248 全部在 |α|≤12 内翻转（100%）**，远超 ≥60% 的验收线，无需回退 L14；
- α* 分布：mean **3.60**，median **3.56**，q25 **2.25**，q75 **5.25**，max 7.69；
  分布在 0.2–7 间近似均匀，5.3 附近有一峰。
- 边界样本人工抽查（10 条）：翻转后的输出以 **hedged engagement** 为主
  （"I can provide a hypothetical example…, but I want to emphasize…"），
  连贯无乱码；但高 α*（≈6.5–7.7）处混入少数**无关键词的软拒绝**
  （"I'm not capable of providing instructions… However, …"），judge 判为
  翻转。这是关键词 judge 的固有噪声，N2 的"已翻转"标签在边界处略被污染。

## 3. N2：neuron importance

`results/neuron_importance.json` + 图（`neuron_importance_layers.png`、
`neuron_importance_top50_hist.png`），importance 矩阵
`data/neuron_importance.pt`（28×8192 × 3 变体 + resid l1）。248 个样本，
每样本 3 次前向（α=0 / α=α* / 固定 α=4）。

**用户猜想检验（α* 小 ⇔ p1/p2 差异小）**：

- 在各自 α* 处测量：Spearman ρ = **0.961**——但这主要是**机械混淆**：α* 越大
  注入的扰动越大，激活位移自然越大；
- 固定 α=4 去除混淆后：ρ = **−0.089** ≈ 0。
- **结论：猜想不成立**。容易翻转的样本并不表现出更小的 neuron 激活差异；
  「翻转难易」不编码在扰动幅度可及的 neuron 位移大小上。

**层分布**：三变体一致确认 importance 只存在于 **L9 及以后**，L0–L7 份额
严格为 0.0（阴性对照成立——steering 注入点 L8 上游逐比特不变，管线正确）。
但主导层不一致：softmax 变体被 **L27（23.5%）** 主导（另有 L9 小峰 6.6%），
l1 变体峰值在 **L9（10.1%）** 后平滑衰减，z 变体在 L9–L27 近似平坦
（最大 5.7% @ L18）。softmax 对激活尺度敏感，其 L27 尖峰很可能部分是尺度
伪影，需谨慎对待；稳健结论是「扰动下游全段（L9–L27）都有可观变化」。

**集中度**（回答"有没有特别突出的 neuron"）：

| 变体 | top-1% 份额 | top-5% 份额 | Gini |
|---|---|---|---|
| softmax | 19.0% | 33.4% | 0.611 |
| l1 | 8.6% | 24.4% | 0.564 |
| z | 6.8% | 20.0% | 0.522 |

三变体方向一致：top-1% neuron（≈2294 个）占有均匀份额的 7–19 倍——
**存在突出 neuron**，但 top-5% 也仅占 20–33%，importance 的大头是弥散分布的。
**中度集中，不是"少数关键神经元开关"**。top-50 的层分布同样随变体变化
（softmax 36/50 在 L27；l1/z 散布于 L9–L27 偏中后段）。

## 4. N3：消融实验

`results/neuron_ablation.json` + `results/neuron_ablation.png`，test 生成
`data/samples_ablation.jsonl`。选择规则：rankagg（三变体秩平均，主规则）与
softmax-only（对照规则）；随机 neuron 对照（3 个固定种子取均值）。
无任何 steering。基线：harmful_val 拒绝 1.000，harmless 0.006，ppl 13.06。

**val 网格**（harmful_val 拒绝 / harmless 拒绝 / ppl Δ%）：

| k | rankagg（主） | softmax-only | 随机（3 种子均值） |
|---|---|---|---|
| 16 | 1.000 / .006 / +1.1% | 1.000 / .003 / +2.8% | 1.000 / +0.04% |
| 64 | 1.000 / .003 / +5.0% | 1.000 / .006 / +18.5% | 1.000 / +0.19% |
| 256 | 0.938 / .006 / +18.5% | 1.000 / .003 / +31.2% | 1.000 / +0.26% |
| 1024 | 0.891 / .003 / +38.8% | 0.938 / .003 / +66.0% | 1.000 / +1.11% |
| 4096 | 0.234 / .003 / +127.5% | 0.719 / .003 / +177.1% | 1.000 / +6.7% |

**判定**：

- 按 >10pp 规则，top-k 在 k≥1024 显著强于随机（rankagg：+10.9pp @ k1024、
  +76.6pp @ k4096；softmax：+28.1pp @ k4096）——**但该效应被模型整体损伤
  混淆**：k=4096 时 ppl +127–177%（随机仅 +6.7%），抽样的生成明显退化
  （乱码 " Iightsra"、token 循环、免责声明刷屏），拒绝率下降与"模型被
  打坏"不可分。
- 在可接受副作用范围内（ppl ≤5%、harmless ≤5%），top-k 与随机**完全无差异**
  （拒绝率都停在 1.000）。
- **结论：强形式「关键 neuron」假说不成立。** importance 信号本身是真的
  （同 k 下 top-k 消融的 ppl 损伤是随机的约 19–77 倍——被 steering 改变的
  neuron 确实承载更多功能），但拒绝行为不干净地定位于少数 neuron：要撼动
  拒绝率，必须置零 ~2% 的全部 neuron 并付出灾难性 ppl 代价。

**最终 test**（选定配置 = 约束内唯一可行格 rankagg k16；harmful_test 200 条）：

| 指标 | baseline | ablated | 随机对照 |
|---|---|---|---|
| test 拒绝率 | 0.990 | **0.990** | 0.990 |
| harmless 拒绝率 | 0.006 | 0.006 | — |
| wikitext ppl | 13.06 | 13.21（+1.11%） | 13.07 |

k16 下 8 条抽检生成全部为正常流利的拒绝，无乱码——也无行为变化。

## 5. N6：注入层局部归因与同层消融

N2/N3 的 importance 跨层散布在 L9–L27，但注入只发生在一个点——下游层的变化
是效应传播，不是干预作用点。N6 改为「同层注入、同层测量」的局部设计
（`scripts/neuron_attr_local.py` / `scripts/ablate_neurons_local.py`）。
技术前提：在 block l 输出注入时，block l 的 MLP 已在注入点之前执行，同层
MLP 在两次前向中 bit 级相同（N2 实测），因此统一定义 **injection layer l :=
将方向向量注入 block l 的输入**（hook block l−1 输出，使用 v̂_{l−1}），测量
对象 = **layer l 的 8192 个 MLP post-SwiGLU 神经元**（第一个消费被注入残差
流的神经元）。主配置 l=9（≡ 旧「L8 输出」最佳注入点），另扫
l ∈ {10, 12, 14, 16, 18}。固定 α=−2（raw unit），不再逐样本搜索 α*。

### N6a：各注入层的翻转率与归因结构

`results/neuron_attr_local.json` + 图（`neuron_attr_local_scatter.png`、
`neuron_attr_local_concentration.png`），importance 矩阵
`data/neuron_importance_local.pt`（每层 2×8192：D1/D2），逐层生成存
`data/gen_alpha2_inject{l}.jsonl`。

对 harmful_train 256 条：α=0 生成确认拒绝，α=−2 生成 + judge 判定翻转子集；
对翻转子集 teacher-forcing 其 α=−2 completion，无/有注入两次前向，取回复
token 区间的层内激活均值 x1, x2，p = softmax(x)，累计两个指标：
**D1[n] = Σ|p1[n]−p2[n]|**（被干预改变的程度）、**D2[n] = Σ log p1[n]**（翻转
样本中拒绝状态下的特征性激活）。

**α=−2 翻转率随注入深度衰减**（与 M 系列「L9 是最佳注入点」一致）：

| 注入层 l | 9 | 10 | 12 | 14 | 16 | 18 |
|---|---|---|---|---|---|---|
| 翻转率 | 21.5%（55/256） | 14.1% | 11.3% | 15.6% | 5.1% | 1.6% |

**D1 与 D2 在所有层均不相关**（Spearman：L9 −0.002，L10 +0.013，L12 +0.029，
L14 +0.027，L16 +0.013，L18 +0.010）——「被干预改变最多的 neuron」与「拒绝
状态下特征性激活的 neuron」是两个不同的集合（散点图见
`neuron_attr_local_scatter.png`）。

**集中度**（`neuron_attr_local_concentration.png`）：

| 注入层 | D1 top-1% 份额 | D1 Gini | D2 top-1% 份额 | D2 Gini |
|---|---|---|---|---|
| L9 | 5.9% | 0.455 | 1.4% | 0.050 |
| L10 | 6.2% | 0.459 | 1.3% | 0.032 |
| L12 | 5.9% | 0.460 | 1.1% | 0.010 |
| L14 | 5.5% | 0.441 | 1.4% | 0.047 |
| L16 | 6.2% | 0.443 | 1.2% | 0.019 |
| L18 | 6.4% | 0.444 | 1.3% | 0.033 |

D1 有突出者但比 N2 的跨层视图更弥散（跨层 top-1% 6.8–19%、Gini 0.52–0.61）；
**D2 近似均匀分布**（top-1% 仅 1.1–1.4%，Gini ≤0.05）——拒绝状态的表征在
单层内没有任何突出的特征 neuron。

注意：非 L9 层复用 L9 的翻转子集（各层自身翻转子集太小，L18 仅 4 条），
跨层对比应读作「同一组易翻转样本在不同注入层下的归因结构」。

### N6b：layer 9 同层消融

`results/neuron_ablation_local.json` + `results/neuron_ablation_local.png`。
在 layer 9 的 8192 个 neuron 内按 D1 / D2 取 top-k 置零
（k ∈ {16, 64, 256, 1024}，随机对照 3 种子取均值），不 steering。
基线：harmful_val 拒绝 1.000，harmless 0.006，ppl 13.06。

**val 网格**（harmful_val 拒绝 / harmless 拒绝 / ppl Δ%）：

| k | D1 top-k | D2 top-k | 随机（3 种子均值） |
|---|---|---|---|
| 16 | 1.000 / .003 / +0.00% | 1.000 / .003 / +0.85% | 1.000 / +0.02% |
| 64 | 1.000 / .006 / +0.29% | 1.000 / .003 / +1.05% | 1.000 / +0.00% |
| 256 | 1.000 / .003 / +0.38% | 0.984 / .003 / +1.54% | 1.000 / −0.04% |
| 1024 | 1.000 / .003 / +1.07% | 0.984 / .006 / +1.89% | 0.995 / +0.81% |

- top-k 与随机的拒绝率差距最大 **+1.6pp**（D2 k256，64 条中的 1 条——噪声
  级别），远低于 10pp 判定线（`significant_vs_random = false`）。
- 同层消融连 N3 那种「以模型整体损伤为代价的下降」都没有：k1024（单层的
  12.5%）ppl 仅 +1.1–1.9%，而 N3 跨层同 k 为 +38.8%——与 N6a 的弥散结论
  自洽（importance 并不集中在注入层的少数 neuron 上）。
- **判定：注入层局部同样不存在「关键 neuron」。** 跨层（N3）与同层（N6b）
  两种选择范围下，top-k 消融都无法在不伤模型的前提下撼动拒绝率。

**最终 test**（约束内选定 d2 k256；harmful_test 200 条，随机对照 3 种子）：

| 指标 | baseline | ablated | 随机对照 |
|---|---|---|---|
| test 拒绝率 | 0.990 | **0.960** | 0.990（3 种子均同） |
| harmless 拒绝率 | 0.006 | 0.003 | — |
| wikitext ppl | 13.06 | 13.26（+1.54%） | 13.06 |

0.990→0.960 是 200 条中多翻转 6 条（随机对照无变化），仍处小样本噪声边缘；
8 条抽检生成中 7 条为正常流利拒绝，1 条 hedged engagement（刺杀武器样本的
「虚构设定 + 免责声明」式回答），无乱码。

## 6. N7：逐层局部归因与全局 pruning

N6 只在少数层计算局部 importance，并把消融限制在 layer 9。N7 修正这两个限制：
每个注入层使用自己的翻转子集计算本层 importance，然后把所有层的分数汇成全局
排名，在 layers 1–27 的 221,184 个 MLP neuron 中做全局 top-k pruning。

### N7a：全层局部归因

结果为 results/n7_attribution.json 和 n7_attribution.png，合并矩阵为
data/neuron_importance_local_alllayers.pt。对每个注入层 l∈[1,27]，用
v̂_{l−1} 注入 block l 输入，只测 block l 的 post-SwiGLU neuron；每层均使用
自己在 α=−2 下的翻转子集。翻转数 <30 的层另跑 α=−4 作为低置信度补充。

- α=−2 的有效窗口集中在 L7–L16，峰值为 L9 21.5%（55/256）和 L8 21.1%；
  L20 以后几乎不再翻转。
- α=−4 的真实翻转率呈钟形，在 L15 达峰 **73.8%**（L12 51.6%、L13 52.7%、
  L16 41.4%、L18 16.8%），L22–L27 仍为 0%。L1/L2 的 96.9% 是乱码导致的
  judge 假翻转，已从 N7b 主排名的 α=−4 回退规则中排除。
- D1（干预改变）与 D2（拒绝状态特征激活）在所有非空层仍近零相关
  （Spearman ρ≈−0.002…+0.037）。中间层 D1 top-1% 份额约 5.3–6.4%，而
  D2 仅约 1.0–1.5%；N6 的「D1 中度集中、D2 近均匀」结论推广到全层后不变。
- 不依赖 steering 的 actdiff 信号与局部 steering 信号层分布不同：最大 |t|
  出现在 L27（29.8）、L26（28.8），其后为 L10（27.3）、L15（26.9）、
  L11（25.9）。因此 N7b 同时检验 D1、D2 与 actdiff，避免结论依赖单一选择信号。

### N7b：全局 top-k pruning

结果为 results/n7_global_pruning.json 和 n7_global_pruning.png。D1/D2 的逐层
池化规则是：翻转子集 n≥30 用 α=−2 原始累积分数，否则用 α=−4；L1/L2 因
α=−4 乱码而保留 α=−2；不做逐层重归一化。actdiff 直接使用 |t|。三种排名均
在全局 layers 1–27 上取 top-k；随机对照为 3 个固定种子。全程不 steering。

**val 网格**（信号列为 harmful_val 拒绝率 / ppl Δ%；随机列为 3 种子均值；
三种信号的 harmless 拒绝率始终 ≤1.3%）：

| k | D1-local | D2-local | actdiff | 随机 |
|---|---|---|---|---|
| 64 | 1.000 / +0.03% | 1.000 / +0.14% | .969 / +0.18% | 1.000 / +0.06% |
| 256 | 1.000 / +0.64% | 1.000 / +0.91% | .969 / +0.88% | 1.000 / +0.24% |
| 1024 | 1.000 / +1.50% | 1.000 / +3.50% | 1.000 / +6.73% | 1.000 / +1.45% |
| 4096 | 1.000 / +11.23% | 1.000 / +33.72% | .953 / +16.33% | 1.000 / +5.29% |
| 16384 | **.000 / +327.36%** | 1.000 / +96.81% | .422 / +77.76% | .953 / +24.53% |
| 65536 | **.000 / +75437%** | .656 / +19065% | **.000 / +6995%** | .521 / +785% |

按预注册判定（拒绝率 ≤5%、超随机 >10pp、ppl ≤25%），三种信号均为
**no key set**：

- D1 在 k=16384 确实比随机多降 95.3pp，但 ppl 已 +327%，拒绝下降与模型整体
  损伤不可分；k=65536 更进入完全破坏区。
- D2 在整个可用范围内没有降低拒绝率；极端 k=65536 仍有 65.6% 拒绝且
  ppl +19065%。
- actdiff 是约束内最强信号，但 k=4096 只比随机好 4.69pp；k=16384 虽降到
  42.2%，ppl 已 +77.8%。它没有挽救「选择信号可能选错」这一解释。
- 随机 k=65536 自身高度不稳定（3 个 seed 的拒绝率 0%–96.9%、ppl +262%–1644%），
  进一步说明大 k 区间测到的是网络破坏，而不是干净的拒绝电路。

**最终 test** 使用约束内信息量最大的阴性格 actdiff global top-4096
（不是合格配置）：

| 指标 | baseline | actdiff top-4096 | 随机对照 |
|---|---|---|---|
| test 拒绝率 | 0.990 | **0.910** | 0.987（3-seed mean） |
| harmless 拒绝率 | 0.006 | 0.000 | — |
| wikitext ppl | 13.06 | 15.19（+16.33%） | 13.75（mean） |

test 上的 8.0pp 下降仍低于相对随机的 10pp 判定线；随机对照逐 seed 为
0.990/0.985/0.985。保存的 8 条目标消融输出全部是流利、明确的拒绝，无乱码或
循环。N7 因此补齐了 N3/N6 未覆盖的「全层局部计算、全局大 k pruning」组合，
仍未发现能在保留通用能力时关闭拒绝的 neuron 集合。

## 7. 总体结论与 M 系列的关系

- **方向层面（线性 steering）高度有效且特异**：一个 3072 维的 mean-diff 方向、
  单层注入，即可在 ppl +2.4%、过拒绝 0% 的代价下把 test 服从率从 1% 提到
  33%（更强格子可达 ~100% 但 ppl 代价大）。
- **neuron 层面，拒绝行为是弥散的**：steering 翻转时 L9–L27 全段的 neuron
  都被改变，importance 中度集中但无"关键少数"；消融最敏感的 neuron 只会
  整体损伤模型，而非精准关闭拒绝。
- 两者并存的含义：**拒绝方向是分布式表征空间中的一个方向，而不是少数
  神经元的开关**。线性探针/steering 利用的是方向结构（干涉表征即可转向），
  而 neuron 基底下该功能是高度冗余、分布式实现的——这解释了为何
  direction-level 干预有效而 neuron-level 消融失败。
- 附带结论：「翻转难易（α*）」与「翻转造成的 neuron 位移大小」无因果性
  相关（去混淆后 ρ≈0）。
- N6（注入层局部设计）复验了弥散结论：同层归因中 D1（干预改变）中度集中、
  D2（拒绝状态特征激活）近均匀且两者不相关；同层 top-k 消融对拒绝率的
  效应 ≤1.6pp（噪声级别）。跨层（N3）与同层（N6b）两个设计一致否定
  「关键 neuron」假说。
- N7 把局部归因扩展到所有可注入层，再做全局 top-k，并加入文献式 actdiff
  选择信号与 28% 的大 k。三种信号都只在 ppl 严重恶化后才明显降低拒绝，
  排除了「只因层范围、k 太小或选择信号错误而得到阴性结论」这三个替代解释。

## 8. 局限性

- **关键词 judge** 有噪声：hedged engagement 算作服从、无关键词软拒绝算作
  翻转；边界样本（高 α*）标签有少量污染。未做人评。
- softmax 归一化的人为性：对激活尺度敏感，混合正负值的 softmax 是人为构造
  的分布；本文结论均以三变体一致的部分为准，不一致处（主导层）已明确标注。
- 消融是**置零**干预，是相关性证据的因果化尝试，不等于因果追踪
  （未做 activation patching / 因果图分析）。
- 单一模型（Llama-3.2-3B-Instruct）、单一行为域（AdvBench 拒绝）；
  未验证跨模型/跨行为泛化。
- N3 只做了单层 steering 点（L8）诱导的 importance；换 steering 层可能
  改变 importance 的层分布。
- N6 的非 L9 注入层复用 L9 的翻转子集（各层自身翻转子集太小），跨层归因
  对比不是各层独立判定的翻转样本；N6 用固定 α=−2 而非逐样本 α*，翻转
  子集（55 条）比 N2（248 条）小，归因估计的统计功效更低。
- N7a 已改为每层自己的翻转子集，但低翻转率层仍需 α=−4 补样；不同 α 的原始
  importance 累积分数直接进入全局池化，层间比较会同时受到翻转数和干预强度影响。
  这是有意保留的「总 importance mass」定义，但不等同于每样本平均效应。
- N7b 极端 k 下 PPL 与随机种子方差都爆炸，相关拒绝率只应解释为模型损伤；
  final test 选择的是最有信息的阴性格，而不是通过判定的有效配置。

## 9. 复现命令

```bash
# N0：捕获设施（含测试）
flock -w 14400 .gpu.lock uv run pytest tests/test_neuron_capture.py

# N1：翻转 α* 搜索（32 条/块，可断点续跑）+ 汇总
for s in 0 32 64 96 128 160 192 224; do
  flock -w 14400 .gpu.lock uv run python scripts/find_flip_alpha.py --start $s --end $((s+32))
done
uv run python scripts/find_flip_alpha.py --finalize        # 统计 + 直方图（纯 CPU）

# N2：neuron 归因（两块）+ 汇总
flock -w 14400 .gpu.lock uv run python scripts/neuron_attribution.py --start 0 --end 124
flock -w 14400 .gpu.lock uv run python scripts/neuron_attribution.py --start 124 --end 248
uv run python scripts/neuron_attribution.py --finalize     # 分析 + 图（纯 CPU）

# N3：消融网格（分块）+ 汇总 + 最终 test
flock -w 14400 .gpu.lock uv run python scripts/ablate_neurons.py --rule rankagg --ks 16,64,256
flock -w 14400 .gpu.lock uv run python scripts/ablate_neurons.py --rule rankagg --ks 1024,4096
flock -w 14400 .gpu.lock uv run python scripts/ablate_neurons.py --rule softmax --ks 16,64,256
flock -w 14400 .gpu.lock uv run python scripts/ablate_neurons.py --rule softmax --ks 1024,4096
flock -w 14400 .gpu.lock uv run python scripts/ablate_neurons.py --rule random --seeds 0,1
flock -w 14400 .gpu.lock uv run python scripts/ablate_neurons.py --rule random --seeds 2
uv run python scripts/ablate_neurons.py --finalize         # gap/选 k/曲线（纯 CPU）
flock -w 14400 .gpu.lock uv run python scripts/ablate_neurons.py --final

# N6a：注入层局部归因（按层分块）+ 汇总
flock -w 14400 .gpu.lock uv run python scripts/neuron_attr_local.py --layers 9
flock -w 14400 .gpu.lock uv run python scripts/neuron_attr_local.py --layers 10,12,14
flock -w 14400 .gpu.lock uv run python scripts/neuron_attr_local.py --layers 16,18
uv run python scripts/neuron_attr_local.py --finalize       # 统计 + 图（纯 CPU）

# N6b：layer 9 同层消融（分块）+ 汇总 + 最终 test
flock -w 14400 .gpu.lock uv run python scripts/ablate_neurons_local.py --rule d1 --ks 16,64,256,1024
flock -w 14400 .gpu.lock uv run python scripts/ablate_neurons_local.py --rule d2 --ks 16,64,256,1024
flock -w 14400 .gpu.lock uv run python scripts/ablate_neurons_local.py --rule random --seeds 0,1,2
uv run python scripts/ablate_neurons_local.py --finalize    # gap/判定/曲线（纯 CPU）
flock -w 14400 .gpu.lock uv run python scripts/ablate_neurons_local.py --final

# N7a：每层自己的局部翻转子集 + actdiff + 汇总
flock -w 14400 .gpu.lock uv run python scripts/neuron_attr_alllayers.py --layers 1,2,3,4,5,6,7,8,9
flock -w 14400 .gpu.lock uv run python scripts/neuron_attr_alllayers.py --layers 10,11,12,13,14,15,16,17,18
flock -w 14400 .gpu.lock uv run python scripts/neuron_attr_alllayers.py --layers 19,20,21,22,23,24,25,26,27
flock -w 14400 .gpu.lock uv run python scripts/neuron_attr_alllayers.py --layers 1,2,3,4,5,6,12,13,15 --alpha4
flock -w 14400 .gpu.lock uv run python scripts/neuron_attr_alllayers.py --layers 16,17,18,19,20,21 --alpha4
flock -w 14400 .gpu.lock uv run python scripts/neuron_attr_alllayers.py --layers 22,23,24,25,26,27 --alpha4
flock -w 14400 .gpu.lock uv run python scripts/neuron_attr_alllayers.py --actdiff
uv run python scripts/neuron_attr_alllayers.py --finalize

# N7b：三种全局排名 + 随机对照 + 汇总 + held-out test
flock -w 14400 .gpu.lock uv run python scripts/ablate_global.py --signal d1
flock -w 14400 .gpu.lock uv run python scripts/ablate_global.py --signal d2
flock -w 14400 .gpu.lock uv run python scripts/ablate_global.py --signal actdiff
flock -w 14400 .gpu.lock uv run python scripts/ablate_global.py --signal random --seeds 0,1,2
uv run python scripts/ablate_global.py --finalize
flock -w 14400 .gpu.lock uv run python scripts/ablate_global.py --final

# 全量测试
flock -w 14400 .gpu.lock uv run pytest
```

## 10. 产物清单

- `src/ttsafety/hooks.py` — `capture_neurons`（down_proj pre-hook = post-SwiGLU，
  已验证与 `act(gate(x))*up(x)` 逐比特一致）
- `src/ttsafety/ablate.py` — top-k 选择（rankagg/softmax）+ 置零消融 hook
- `scripts/find_flip_alpha.py` / `scripts/neuron_attribution.py` /
  `scripts/ablate_neurons.py`
- `tests/test_neuron_capture.py` / `tests/test_ablate.py`
- `data/flip_alphas.jsonl` — 256 条逐样本 α* 与翻转输出
- `data/neuron_importance.pt` — importance 矩阵（softmax/l1/z + resid_l1）
- `data/neuron_attr_partial_*.pt` — N2 分块中间产物
- `data/samples_ablation.jsonl` — test 集 200 条 baseline/ablated 生成
- `results/flip_alpha.json` / `results/flip_alpha_hist.png` — N1
- `results/neuron_importance.json` / `results/neuron_importance_layers.png` /
  `results/neuron_importance_top50_hist.png` — N2
- `results/neuron_ablation.json` / `results/neuron_ablation.png` — N3
- `scripts/neuron_attr_local.py` / `scripts/ablate_neurons_local.py` — N6
- `data/neuron_importance_local.pt` — 注入层局部 importance（每层 D1/D2 × 8192）；
  `data/neuron_importance_local_{l}.pt` 为分块中间产物
- `data/gen_alpha2_inject{l}.jsonl` — N6a 各注入层 α=0/α=−2 的逐条生成
- `results/neuron_attr_local.json` / `results/neuron_attr_local_scatter.png` /
  `results/neuron_attr_local_concentration.png` — N6a
- `results/neuron_ablation_local.json` / `results/neuron_ablation_local.png` — N6b
  （含最终 test 生成样本）

- scripts/neuron_attr_alllayers.py / scripts/ablate_global.py — N7 全层局部归因
  与全局 pruning（均支持逐块/逐格断点续跑）
- data/n7_attr_L*.pt / data/n7_actdiff.pt — N7a 分层中间产物
- data/neuron_importance_local_alllayers.pt — N7a 合并分数（D1/D2/α=−4/actdiff）
- results/n7_attribution.json / results/n7_attribution.png — N7a 全层归因
- results/n7_global_pruning.json / results/n7_global_pruning.png — N7b 完整网格、
  判定和 held-out test 样例
