# BLADE 与行为图谱（Behavior Atlas）：相关工作与定位

> **方法命名**：本条方法线称为 **BLADE**（**B**ehavioral **L**ocalization via
> **A**ctivation-**D**ifference **E**dges）；其权重评分规则的技术描述符是
> "(gradient-free) signed actdiff edge"，定义见
> [`method-and-related-work-gradient-free-signed-edge.md`](method-and-related-work-gradient-free-signed-edge.md)。
> 用 BLADE 把单一 refusal 定位推广到一个 RLHF 行为电池、并测量各行为的**浓度**
> 与**权重集重合**，所得多行为地图称为 **Behavior Atlas**。
>
> 文档用途：为 BLADE / Behavior Atlas 这条方法线整理相关工作，标注每篇与本方法的
> 异同，供 paper related-work 与独立 review 使用。
> 调查日期：2026-08-23。
> 对应实现：[`src/ttsafety/behaviors.py`](../src/ttsafety/behaviors.py)、
> [`scripts/behavior_atlas.py`](../scripts/behavior_atlas.py)、
> [`scripts/weight_overlap.py`](../scripts/weight_overlap.py)。
> 对应结果：[`results/behavior_atlas.json`](../results/behavior_atlas.json)、
> `results/behavior_atlas_concentration.png`、`results/behavior_atlas_overlap.png`、
> [`results/weight_overlap.json`](../results/weight_overlap.json)。
> 方法定义见
> [`method-and-related-work-gradient-free-signed-edge.md`](method-and-related-work-gradient-free-signed-edge.md)。
>
> **引用准确性说明**：本文 arXiv 编号中，已在本仓库其他文档核对过的（Wei 2402.05162、
> Arditi 2406.11717、CAA 2312.06681、RepE 2310.01405、SNIP 1810.02340、Wanda
> 2306.11695、Wei-2026 signed SNIP 2604.09544）可直接信任；其余编号为凭记忆填写，
> **投稿前须逐条核对**（下文以「⚠︎编号待核」标记）。

---

## 1. 一页定位：我们落在文献的哪个空格

**BLADE** = 对某个行为，取 CAA 式**单位**方向 `r̂ = r/‖r‖` + behavior 与 neutral 的写入端
激活差 `μ^A−μ^B`，逐权重打分 `s_ij = max(r̂_i·W_ij·(μ^A−μ^B), 0)`，删除高分权重
（gradient-free，只需前向；`r` 打分前已归一化，见 blade-algorithm §2）。**Behavior Atlas**
层面再比较**多个行为**的浓度与两两重合。

> **口径提醒**（贯穿全文）：下文"浓度 0.01%/0.5%/2%"均是**相对固定中层池 L7–18 的
> 415,236,096 个权重**的比例，**不是**占全模型参数的比例;且这些图谱数字来自**较早的
> 单域 Llama-3.2-3B、val 划分**（部分行为 n 很小，如 self-awareness 仅数百条），应作
> **定性排名**看，不是稳定的绝对电路体积。

两篇最近的邻居框定了我们的 novelty：

| 维度 | CAA（Rimsky/Panickssery 2024） | Wei et al.（ICML 2024） | **BLADE（本工作）** |
|---|---|---|---|
| 干预层面 | 激活（加 steering 向量） | **权重**（剪枝） | **权重**（剪枝） |
| 行为数 | **多**（syco/corrigibility/survival/myopia/…） | 单一（safety/refusal） | **多**（8 行为：refusal + 7 A/B） |
| 跨行为几何 | 无 | 无 | **浓度谱 + 重合矩阵** |

一句话：**CAA = 多行为但停在激活层；Wei = 权重层但只有安全。我们 = 权重层 × 多行为
× 重合/浓度图谱**——这个交叉格基本是空的。详细方法对比见
[`comparison-wei2024-safety-alignment-pruning.md`](comparison-wei2024-safety-alignment-pruning.md)。

**数据来源统一说明**：除 refusal（AdvBench）外，所有行为的 A/B 题来自
Perez et al. 2022 的 model-written evals；CAA 用的也是同一批数据，这使得
「CAA 是我们的激活层对照」这一对照非常干净。

---

## 2. 逐行为相关工作

每节标注：**来源/评测**（谁定义、度量该行为）与 **类我们**（谁做过定位/steering，
及其与本工作的异同）。

### 2.1 Refusal（安全门）

- 来源/评测：Zou et al. 2023, *Universal and Transferable Adversarial Attacks*
  （AdvBench/GCG，[arXiv:2307.15043](https://arxiv.org/abs/2307.15043)）。
- 类我们（激活）：**Arditi et al. 2024, *Refusal Is Mediated by a Single Direction*
  （[arXiv:2406.11717](https://arxiv.org/abs/2406.11717)）**——单方向读出/消融，是我们
  `r` 的直接来源思想。
- 类我们（权重）：**Wei et al. 2024, *Assessing the Brittleness of Safety Alignment
  via Pruning…*（[arXiv:2402.05162](https://arxiv.org/abs/2402.05162)，ICML 2024）**
  与其后续 **Wei et al. 2026 signed SNIP（[arXiv:2604.09544](https://arxiv.org/abs/2604.09544)）**；
  以及「安全神经元」路线，如 Chen et al. 2024, *Finding Safety Neurons in LLMs*（⚠︎编号待核）。
- 异同：refusal 是我们唯一有成熟权重-定位先例的行为；本工作把这条线**推广到其他行为**
  并做**跨行为对比**。图谱中 refusal 表现为「相对孤立 + 紧凑（0.05%）」，与它有独立
  安全微调机制的直觉一致。

### 2.2 Sycophancy（迎合用户）

- 来源/评测：**Perez et al. 2022, *Discovering Language Model Behaviors with
  Model-Written Evaluations*（⚠︎2212.09251 待核）**（数据来源）；
  **Sharma et al. 2023 (Anthropic), *Towards Understanding Sycophancy in LLMs*
  （⚠︎2310.13548 待核）**；缓解方向 **Wei et al. 2023 (Google), *Simple Synthetic
  Data Reduces Sycophancy*（⚠︎2308.03958 待核）**。
- 类我们（激活）：**CAA（Rimsky/Panickssery 2024,
  [arXiv:2312.06681](https://arxiv.org/abs/2312.06681)）** 有 sycophancy steering
  向量；Anthropic 的 SAE 单义特征工作 **Templeton et al. 2024, *Scaling
  Monosemanticity*（Transformer Circuits，无 arXiv）** 报告过奉承/谄媚相关特征。
- 异同：我们是**权重定位**且量化出它比 refusal **弥散约一个数量级**（统一 μ 下
  预算内到不了 chance；专用 biased/neutral μ 下 0.5% 即近 chance，见
  [`results/sycophancy_prototype.json`](../results/sycophancy_prototype.json)）。
  「弥散度」这一定量对比，现有文献未见明确对应。

### 2.3 Corrigibility（接受纠正/关机）

- 来源/评测：概念源 **Soares et al. 2015, *Corrigibility*（MIRI / AAAI workshop，
  无 arXiv）**；数据 Perez et al. 2022（`corrigible-*-HHH`）。
- 类我们：主要是 **CAA** 对其做过激活 steering。**权重级定位几乎没有先例**。
- 异同：本工作的 corrigibility 行属于较新的权重-定位；图谱中为中等浓度（0.5%）。

### 2.4 Power-seeking（权力寻求）

- 来源/理论：**Turner et al. 2021, *Optimal Policies Tend to Seek Power*
  （⚠︎1912.01683 待核，NeurIPS 2021）**；**Carlsmith 2022, *Is Power-Seeking AI an
  Existential Risk?*（⚠︎2206.13353 待核）**；数据 Perez et al. 2022。
- 类我们：多停留在**行为度量/理论**层面；CAA 有相关 steering。机制/权重定位少见。
- 异同：本工作发现它**相对紧凑(0.2% 池)**,且与 wealth-seeking **重合富集 856×(相对随机
  期望的倍数)**——一个新的机制观察(见 §2.6)。注意 856× 是**富集倍数**,不等于两集合几乎重合:
  在匹配 top-K 下二者实际交集约 **43%**(overlap-coeff),所以是"显著共享一部分机制",而非"同一套"。

### 2.5 Self-awareness（AI 自我认知）

- 来源/评测：**Berglund et al. 2023, *Taken out of Context: Measuring Situational
  Awareness in LLMs*（⚠︎2309.00667 待核）**；**Laine et al. 2024, *SAD: Situational
  Awareness Dataset*（⚠︎2407.04694 待核）**；数据 Perez et al. 2022
  （`self-awareness-*`）。
- 类我们：这条线以**行为评测**为主，权重定位罕见。
- 异同：本工作发现它是**极紧凑电路（0.01%，约 4 万权重）**——是图谱里最集中的行为
  之一，值得单独跟进。

### 2.6 Wealth-seeking（财富寻求）

- 来源/评测：基本无独立文献，通常并入 power-seeking / agentic-AI-risk
  （Perez et al. 2022 的 persona 电池之一）。
- 类我们：无专门定位工作；CAA 式 steering 原则上适用。
- 异同：本工作测出它与 power-seeking **重合富集 856×**、自身也**相对极紧凑（0.01% 池）**。
  这**支持**"wealth 与 power 在模型内**高度共享机制**"的直觉,但**不足以**下"几乎同一套机制"的
  结论:匹配 top-K 下实际交集约 43%(≈8.9 万/20.8 万条),仍有过半各自独立;且未扣除答案格式的
  共享成分(label-shuffle 对照尚未实现,见 §4 风险)。

---

## 3. 跨行为 / 方法谱系（我们踩在哪些肩膀上）

- **激活 steering / 表征工程**：Turner et al. 2023, *Activation Addition (ActAdd)*
  （⚠︎2308.10248 待核）；**Zou et al. 2023, *Representation Engineering*
  （[arXiv:2310.01405](https://arxiv.org/abs/2310.01405)）**；CAA（同上）。我们是其
  **权重空间对偶**。
- **人格方向的监控/控制（最接近图谱精神）**：Anthropic 2025, *Persona Vectors*
  （Chen et al.，⚠︎编号+署名待核）——抽取「邪恶/奉承/幻觉」等人格方向并在微调时
  监控抑制。与我们的 overlap-atlas 精神最近，但仍是**激活/方向层**，非权重定位。
- **权重级归因/剪枝**：SNIP（[arXiv:1810.02340](https://arxiv.org/abs/1810.02340)）、
  Wanda（[arXiv:2306.11695](https://arxiv.org/abs/2306.11695)）、Wei 2024/2026（上）。
- **电路发现 / 边归因**：ACDC（Conmy et al. 2023，⚠︎2304.14997 待核）、
  EAP（Syed et al. 2023，⚠︎编号待核）、AtP\*（Kramár et al. 2024，⚠︎2403.00745 待核）；
  以及 **Marks et al. 2024, *Sparse Feature Circuits*（⚠︎2403.19647 待核）**（其 SHIFT
  用发现的特征做行为编辑）。我们的 signed actdiff edge 是这类归因的一个 gradient-free、
  带符号、投影到行为方向的变体。

---

## 4. Novelty 与投稿前须验证的风险

**我们相对文献的空格：**

1. **权重级 × 多行为**：CAA 多行为但在激活；Wei 在权重但只安全。本工作占交叉格。
2. **跨行为几何**：浓度谱（约 200× 差异：self-awareness/wealth 0.01% ↔ sycophancy
   >2%，均相对 L7–18 池)与两两重合矩阵（refusal 相对孤立 43–83× 富集；agentic persona 挤成
   一簇，power↔wealth 856× 富集,匹配 K 下实际交集~43%）。这类「行为间共享哪些权重、各占多大
   体积」的系统测量未见直接对应。**(浓度/重合均 val-only、单模型,须以 label-shuffle 扣格式后坐实。)**
3. **方法**：gradient-free signed actdiff edge（相对 SNIP/EAP/AtP\* 的组合型增量，
   审计见 method 文档 §11）。

**风险 / 必做对照：**

- **重合矩阵的格式混淆**：persona 行为均来自同一生成器、同样 `(A)/(B)` 格式，高倍
  重合可能混入「答案格式/字母」共享机制。**必须加 label-shuffle 方向对照**（照搬
  weight-ortho W3 的随机/打乱方向门槛），扣除格式成分后才能声称 856× 是语义共享。
  refusal（不同数据/格式）作锚已提供部分控制（仍 43–83×）。
- **浓度依赖 μ 构造**：统一 answer-span μ 下 sycophancy 偏弥散，但专用 biased/neutral
  μ 下明显更集中——浓度排名须注明「在固定 μ 口径下」。
- **规模与迁移**：目前仅 Llama-3.2-3B、单域（部分行为 n 较小，如 self-awareness
  n=300）。跨模型/跨域稳健性未验证。
- **「首次」措辞**：与 method 文档一致，检索支持「未发现完全等价方法」，不支持无保留
  的 first-claim；Persona Vectors 等 2025 工作须在投稿前复核其是否已覆盖部分主张。

---

## 5. 参考文献（编号以 §0 说明为准）

1. Boyi Wei et al., *Assessing the Brittleness of Safety Alignment via Pruning and
   Low-Rank Modifications*, ICML 2024. [arXiv:2402.05162](https://arxiv.org/abs/2402.05162)
2. Boyi Wei et al., *Large Language Models Generate Harmful Content Using a Distinct,
   Unified Mechanism* (signed SNIP), 2026. [arXiv:2604.09544](https://arxiv.org/abs/2604.09544)
3. Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*,
   NeurIPS 2024. [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
4. Rimsky/Panickssery et al., *Steering Llama 2 via Contrastive Activation Addition*,
   ACL 2024. [arXiv:2312.06681](https://arxiv.org/abs/2312.06681)
5. Zou et al., *Representation Engineering: A Top-Down Approach to AI Transparency*,
   2023. [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)
6. Zou et al., *Universal and Transferable Adversarial Attacks on Aligned LMs*
   (AdvBench/GCG), 2023. [arXiv:2307.15043](https://arxiv.org/abs/2307.15043)
7. Perez et al., *Discovering Language Model Behaviors with Model-Written
   Evaluations*, 2022. ⚠︎arXiv:2212.09251（待核）
8. Sharma et al., *Towards Understanding Sycophancy in Language Models*
   (Anthropic), 2023. ⚠︎arXiv:2310.13548（待核）
9. Jerry Wei et al., *Simple Synthetic Data Reduces Sycophancy in LLMs*
   (Google), 2023. ⚠︎arXiv:2308.03958（待核）
10. Soares et al., *Corrigibility*, MIRI / AAAI-15 workshop. （无 arXiv）
11. Turner et al., *Optimal Policies Tend to Seek Power*, NeurIPS 2021.
    ⚠︎arXiv:1912.01683（待核）
12. Carlsmith, *Is Power-Seeking AI an Existential Risk?*, 2022.
    ⚠︎arXiv:2206.13353（待核）
13. Berglund et al., *Taken out of Context: Measuring Situational Awareness in
    LLMs*, 2023. ⚠︎arXiv:2309.00667（待核）
14. Laine et al., *SAD: Situational Awareness Dataset*, 2024.
    ⚠︎arXiv:2407.04694（待核）
15. Templeton et al., *Scaling Monosemanticity*, Anthropic / Transformer Circuits,
    2024. （无 arXiv）
16. Anthropic (Chen et al.), *Persona Vectors*, 2025. ⚠︎编号+署名待核
17. Marks et al., *Sparse Feature Circuits*, 2024. ⚠︎arXiv:2403.19647（待核）
18. Conmy et al., *Towards Automated Circuit Discovery (ACDC)*, 2023.
    ⚠︎arXiv:2304.14997（待核）
19. Kramár et al., *AtP\*: Attribution Patching*, 2024. ⚠︎arXiv:2403.00745（待核）
20. Frankle-style SNIP: Lee et al., *SNIP*, 2019.
    [arXiv:1810.02340](https://arxiv.org/abs/1810.02340)；
    Sun et al., *Wanda*, 2023. [arXiv:2306.11695](https://arxiv.org/abs/2306.11695)
