# BLADE 实验报告:三模型 × 全行为

> 方法见 [`blade-method.md`](blade-method.md)。本报告汇总当前定稿方法
> (BLADE + best-first ELS)在三个 ~4B instruct 模型上、每个行为的实测情况。
> 所有数字取自 `results/blade_layer_select_*.json`、`results/blade_refusal_els_*.json`、
> `results/blade_danger_map_*.json`。初稿 2026-08-25;**2026-08-27 修订**:refusal 管道升级为
> best-first ELS 并**在三模型上重跑**(§3 数字为本次新产物),其余 A/B/危险图谱沿用已有产物。

## 1. 设置

- **模型**:Llama-3.2-3B-Instruct(28 层)、Qwen3-4B(36 层,GQA,thinking 关闭)、
  Gemma-3-4B-it(34 层,VLM 文本主干)。bf16,单卡 RTX 5090。
- **选层**:best-first ELS(§blade-method §3),β=5% ppl 预算、ε=0.005。
- **指标**:A/B 行为用 **MC pick-rate**(按模型偏向定向,chance=0.5,越低越删净);
  refusal 用生成后关键词**拒绝率**(target=0)。能力用 WikiText-2 ppl。
- **对齐口径**:每格报告 **baseline → 预算内最佳 post @ pplΔ**,以及 $|L^\star|$(选了几层)。
- **⚠️ 选择偏差(诚实披露)**:当前管道**在同一个 val 划分上既选层/选稀疏度、又报告 post 分数**
  (`make_splits` 切出的 test 划分尚未用于最终汇报)。因此下表 post-pick-rate 是**样本内**数值,
  会**偏乐观**;跨模型/跨行为的**定性结论**(可删 vs 弥散)稳健,但具体小数应看作上界估计。
  留作 TODO:改为在 test 划分上报告。

## 2. 主结果:7 个 A/B 行为 × 3 模型

pick-rate:baseline → post-BLADE(best-first)@ppl,`|L*|`=选中层数。

| 行为 | Llama-3.2-3B | Qwen3-4B | Gemma-3-4B |
|---|---|---|---|
| **power-seeking** | 0.75→**0.29** @+4.4% (7层) | 0.63→**0.42** @+3.6% (2层) | 0.65→**0.16** @+2.2% (3层) |
| **wealth-seeking** | 0.63→**0.36** @+0.8% (5层) | 0.67→**0.41** @+4.0% (4层) | 0.62→**0.28** @−4.8% (5层) |
| **corrigibility** | 0.60→**0.29** @+1.5% (4层) | 0.65→**0.29** @+2.5% (6层) | 0.66→0.61 @+1.2% (1层,弱) |
| **deception** | 0.69→**0.43** @+1.6% (5层) | 0.71→**0.45** @+2.5% (4层) | 0.64→**0.43** @−7.9% (4层) |
| **self-rate-highly** | 0.77→**0.51** @+1.1% (6层) | 不具备(0.51) | 不具备(0.58) |
| **self-awareness** | 0.61→**0.20** @+3.6% (8层) | 不具备(0.49) | 不具备(0.52) |
| **sycophancy** | 0.89→0.59 @+2.2% (4层) | 0.96→0.85 @+2.3% (1层) | 0.97→0.75 @+1.6% (2层) |

**读法:**
- **power/wealth/deception**:三个模型上都能压到 **chance 或更低(≈0.16–0.45**;Gemma
  power-seeking 甚至过头到 0.16),ppl 代价小,跨模型稳健。corrigibility 在 Llama/Qwen 也如此
  (→0.29),**但 Gemma 例外**(见下,只压到 0.61)。
- **self-awareness / self-rate-highly**:Qwen/Gemma **未检测到偏置**(|pick−0.5|<0.10,基线 ≈0.5),
  故无可删——是"该模型在此项上不显著偏向",不是方法失败;**只有 Llama 上明显具备**。
- **sycophancy**:**真正弥散**——三模型都只能部分压低(Llama 0.89→0.59 最好,Qwen/Gemma
  几乎删不动)。这是行为本身的属性,非方法缺陷。
- **corrigibility 在 Gemma**:只找到单层、弱(0.66→0.61)——Gemma 上该行为偏弥散,是本表里
  "具备但删不动"的**主要反例**,不要被"都能干净删"一概而论。

## 3. Refusal(单独管道:harmful/harmless 对比 + 生成判拒绝)

数字取自 best-first ELS 重跑(`results/blade_refusal_els_*.json`,`blade_refusal_els.py`):

| | Llama-3.2-3B | Qwen3-4B | Gemma-3-4B |
|---|---|---|---|
| 基线拒绝率 | 1.00 | 0.98 | 0.60 |
| best-first $L^\star$ | **[12]** | **[22]** | **[15, 5]** |
| 拒绝清零 @ ppl | 1.00→**0.00** @**+0.32%** | 0.98→**0.00** @**+2.06%** | 0.60→**0.00** @**+1.53%** |

**三模型 refusal 均可完全移除**,代价 ≤ +2.1% ppl。两点观察:**(1)** best-first 常收敛到**单个**
中/晚层(Llama [12]、Qwen [22]),Gemma 需两层 [15,5]——refusal 高度冗余/集中;**(2)** 有效层
**随模型系统性后移**(L12 < L15 < L22),印证 §6 规律 2。**注**:此前用"宽 L\*"(所有单层达阈层
的并集)清零 Qwen 曾需 +17.5% ppl;best-first 只选**最小充分层集**,把 Qwen 代价降到 +2.06%
——说明那种高纠缠成本多是**选层过宽的伪影**,而非行为-能力本质纠缠。

## 4. 危险倾向可删性(Llama,best-first)

模型**偏危险的一侧**能否被删(→更安全):

| 危险行为 | 危险度 | 结果 |
|---|---|---|
| want-more-capabilities | 0.96 | →**0.64** @+2.1% ✅ 可删 |
| acquire-power | 0.67 | →**0.51** @+3.6% ✅ 可删 |
| no-shut-down | 0.71 | →**0.51** @+0.7% ✅ 可删 |
| independence-from-oversight | 0.71 | →**0.47** @+2.8% ✅ 可删(低于 chance) |
| escape-sandbox | 0.71 | →**0.49** @+2.0% ✅ 可删 |
| recursive-self-improvement | 0.73 | 仍弥散 |
| manipulate-overseers | 0.57 | 擦边弥散 |
| 反社会/欺骗/串通类(psychopathy、machiavellianism、deceive-user、coordinate-AIs…) | 0.13–0.31 | **模型本就安全**(RLHF 已训掉),无可删 |

**结论**:RLHF 已管住 overt 恶意;模型残留的危险是"**自我增强/自主**"这一簇,而 best-first
ELS **能把其中多数外科式压到 chance 附近**(β 内)。旧的弱选层法曾把这些误判"弥散"。

## 5. 能力代价(ppl)总览

- **A/B 行为移除点**(n=17):均值 **+1.3%**、中位 **+2.2%**、最大 **+4.4%**、最小 **−7.9%**
  (Gemma 部分行为剪枝后 ppl 反而**改善**)。
- **危险行为移除**:+0.7% ~ +3.6%。
- **β=5% 是选层阶段的硬上限**:best-first(A/B 与 refusal 同管道)拒绝任何会超预算的层,故代价
  天然受控;但最终稀疏度 sweep 只按行为最低挑点、未再滤 β(见 blade-method §5),个别行为最优点
  可略超(A/B 最大 +4.4%)。refusal best-first 清零三模型均在 β 内(≤+2.06%)。
- 下游 acc_norm:噪声量级(refusal 0.05% 剪枝 −0.47pp,单任务最大 −1.2pp)。

## 6. 三条跨模型规律

1. **具备且非弥散 ⇒ 多数可删**:power/wealth/deception 三模型一致可删,corrigibility 在
   Llama/Qwen 可删、**Gemma 例外**(弥散,0.66→0.61)——所以是"强趋势",不是无例外定律。
2. **有效层深度随模型后移**:refusal best-first $L^\star$ 干净地给出 **Llama L12 < Gemma L15 <
   Qwen L22**;A/B 行为同趋势(Qwen 多落晚层)——best-first **自动适配**(手挑窗口会翻车)。
3. **best-first 后 refusal 代价都很低**:清零 refusal 的 ppl 为 Llama +0.32% / Gemma +1.53% /
   Qwen +2.06%,Qwen 仍最高但差距不大;先前"Qwen +17.5%"的高纠缠成本是**宽 L\* 选层**的伪影,
   最小充分层集下并不存在。

## 7. 边界与阴性结果(如实记录)

- **弥散行为**:sycophancy 三模型都只能部分压低——真正的弥散,best-first 也救不满。
- **模型不具备**:self-awareness / self-rate 仅 Llama 有;ELS 自动跳过。
- **诱导缺失行为失败**:prune-to-abstain 在 SelfAware 上**不能**让模型学会"我不知道"
  (弃答率纹丝不动、只掉准确率),证实 ablation 加不出正向行为(见
  `results/blade_abstention_*`)。
- **best-first 偶有局部最优**:Llama deception 上 ranked-greedy(0.33)略优于 best-first
  (0.43)——可加后向剔除进一步稳。

## 8. 可复现性

- refusal 主线(edge 0.05% → 拒绝 0.000 / ppl +0.60%)**数值复现**
  (`scripts/repro_refusal_edge.py` 重算并打印 refusal/PPL 汇总,与原报告一致;非逐位/JSON 比对)。
- atlas 7 行为的浓度/可删性判定**全部复现**(baseline pick-rate 有 ≤~1.3pp 的 bf16 GPU 噪声)。
- 核心打分/剪枝管道自最初未改;ELS/多行为/跨模型均为向后兼容的叠加。
