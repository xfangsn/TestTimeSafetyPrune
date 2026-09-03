# 通过稀疏权重操纵"强化/加固"LLM Safety 的文献调研

> **调研问题**：BLADE 把 refusal / power-seeking / deception 等"训进 LLM 的行为"定位到 residual-writer 的极少量标量权重并**剪除**（移除行为）。本文档系统调研**反向/互补**的问题：有没有工作通过改变/操纵 behavior 相关权重的稀疏性（sparse weight 层面）来**强化、加固或保护**模型的 safety？
>
> **核实状态说明**：除特别标注外，文中所有 arXiv 编号均通过抓取 `arxiv.org/abs/<id>` 摘要页（或 PMLR / ACL Anthology / OpenReview 页面）逐条核对过标题、作者与摘要。无法核实的条目会显式标注 **[需核实]**；确认不存在的"传闻论文"列在文末附录。**没有任何编号是凭记忆写下的。**
>
> 调研日期：2026-08-27

---

## 0. TL;DR

- **"稀疏强化 safety"（与 BLADE 方向相反）确实存在，但工作很少**：最典型的是 **ROSI**（arXiv 2508.20766，rank-1 权重注入放大 refusal）和 task-vector 系的 **RESTA / Safety Arithmetic**。真正把 safety **加强**到少量标量权重层面的工作基本是空白——现有"强化"要么是 rank-1（结构化），要么是 dense task vector。
- **"保护/免疫 safety 关键权重"有较清晰的一支**：Safety Layers / SPPFT（冻结安全层，ICLR 2025）、Wei et al. 的 brittleness 分析（NeurIPS 2024，~3% 权重即可摧毁 safety）、SafetyLock（safety-head bias）。但注意 Wei et al. 的重要反例：**单纯冻结 safety-critical 区域并不能完全防住低成本微调攻击**。
- **"稀疏安全修复"（harmful fine-tuning 之后恢复 safety）是最拥挤的方向**：Antidote（ICML 2025，稀疏剪除 harmful 权重）、SafeDelta（ICML 2025）、IRR、NLSR、Yang et al. 等，多数是"稀疏剪除*有害*参数以恢复 refusal"——机制与 BLADE 相同（稀疏剪），但剪的对象相反（剪 harm 而非剪 refusal）。
- **主流抗微调防御（Vaccine / RepNoise / Booster / TAR / Lisa）全部是 dense 全模型方法**，不是 sparse-weight 层面；例外是 T-Vaccine（层粒度冻结）和 Safe LoRA / SaLoRA（低秩子空间层面）。
- **机制类证据充分**：refusal 由单一方向介导（Arditi et al.）、~3% 权重（Wei et al.）、<1–5% 神经元（Zhao et al. / Chen et al.）、少数 attention heads（Zhou et al. ICLR 2025 Oral）即可决定 safety——这既是 BLADE 的前提，也是"反向加固"的前提。

---

## 1. 分类综述

### 类别 1：放大/强化安全行为的稀疏权重编辑（与 BLADE 方向相反）

这是与"BLADE 反过来做"最直接对应的一类，**工作很少**。

| 论文 | 作者 | Venue/年份 | arXiv | 稀疏性 | 一句话方法 |
|---|---|---|---|---|---|
| **Turning the Spell Around: Lightweight Alignment Amplification via Rank-One Safety Injection (ROSI)** | Hammoud et al. | arXiv 2025 | 2508.20766 ✅ | rank-1（结构化，非逐标量） | 在所有 residual-stream 写矩阵上注入 rank-1 权重修改，把激活永久推向 refusal 子空间；论文摘要明说这是 refusal-direction 消融的"**the opposite approach**"，甚至能把 uncensored 模型重新对齐。 |
| **Safety Arithmetic: A Framework for Test-time Safety Alignment of Language Models by Steering Parameters and Activations** | Hazra, Layek, Banerjee, Poria | EMNLP 2024 (main) | 2406.11801 ✅ | dense task vector + 激活转向（非稀疏 mask） | 免训练：从权重中减去 "harm direction"、加上 safety-alignment 向量，配合 in-context 激活转向增强安全行为。 |
| **RESTA: Language Models are Homer Simpson! Safety Re-Alignment through Task Arithmetic** | Bhardwaj, Anh, Poria | ACL 2024 (Long) | 2402.11746 ✅ | dense safety vector（论文评估了 DARE 稀疏化变体） | 计算 safety 向量（safety-tuned 与 base 模型权重差），算术加到被损害的模型上恢复 refusal。 |
| **Model Merging and Safety Alignment: One Bad Model Spoils the Bunch** | Hammoud et al. | Findings of EMNLP 2024 | 2406.14563 ✅ | 否（dense 合并） | safety-aware 模型合并：把对齐数据纳入 data-aware 合并优化，把对齐当作可最大化技能。 |
| **LED-Merging: Mitigating Safety-Utility Conflicts in Model Merging** | Ma, Liu, Chen, Zhang, Shao | ACL 2025 (Long) | 2502.16770 ✅ | 是（神经元粒度 mask） | 梯度归因定位 safety/task 关键神经元，多模型重要性融合选出关键神经元并隔离冲突参数再合并，保住合并模型的 safety 神经元。 |
| **Finding Safety Neurons in Large Language Models**（机制+干预） | Chen, Wang, Yao, Bai, Hou, Li | NeurIPS 2025（arXiv comment 标注） | 2406.14144 ✅ | 是（~5% 神经元，激活层面） | 推理时激活对比 + 动态激活修补定位 safety 神经元；只修补这些神经元即可恢复 >90% safety。干预在激活层面而非永久权重编辑。 |

**与 BLADE 的关系**：ROSI 几乎是 BLADE 的"镜像论文"——BLADE 在 residual-writer 上找标量权重剪掉 refusal，ROSI 在 residual-writer 上注入 rank-1 结构放大 refusal。BLADE 的反向操作（**放大**而非剪除 refusal 边）在逐标量（unstructured sparse）粒度上**尚无对标工作**，这是本调研发现的最明确的空白（见 §3）。

### 类别 2：保护/免疫 safety 关键权重（冻结、正则、加固）

| 论文 | 作者 | Venue/年份 | arXiv | 稀疏性 | 一句话方法 |
|---|---|---|---|---|---|
| **Safety Layers in Aligned Large Language Models: The Key to LLM Security** | Li, Yao, Zhang, Li | ICLR 2025 | 2408.17003 ✅ | 是（层粒度） | 发现一小段连续中间层（"safety layers"）的输入几何能区分恶意/正常查询；提出 SPPFT，微调时**冻结**这些层的梯度。 |
| **Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications** | Wei, Huang, Huang, Xie, Qi, Xia, Mittal, Wang, Henderson | ICML 2024（PMLR v235:52588–52610） | 2402.05162 ✅ | 是（~3% 参数、~2.5% 奇异方向） | 定位与 utility 区域**解耦**的 safety-critical 权重区域；只剪这些稀疏区域就能摧毁 safety 而几乎不影响 utility。⚠️ 重要反例：论文同时表明**即使限制修改 safety-critical 区域，低成本微调攻击仍能部分成功**——单纯冻结不够。 |
| **SafetyLock: Locking Down the Finetuned LLMs Safety** | Zhu, Yang, Wei, Zhang, Zhang | arXiv 2024（投稿 ACL/OpenReview，录用情况**[需核实]**） | 2410.10343 ✅ | 是（safety attention head 的 bias 向量） | 发现稳定编码无害性的 "safety heads"，从 base 模型提取可迁移的 Meta-SafetyLock bias 方向，微调后注入（~0.01s）恢复 safety。 |

**与 BLADE 的关系**：这一类是"识别 safety 关键组件 → 保护"，与 BLADE"识别 behavior 关键权重 → 剪除"共享同一个定位问题，只是后续操作相反。BLADE 的定位方法（residual-writer 标量归因）若用于"找出要冻结/加固的 W_ij"，属于此类，但 Wei et al. 的反例说明"冻结定位出的关键权重"防御力有限，必须面对。

### 类别 3：稀疏化的安全修复（harmful fine-tuning 之后恢复 safety）

**这是 sparse-weight 安全方向最拥挤的一类**，但注意：它们的机制大多是"稀疏剪除*有害*参数"——与 BLADE 共享剪枝机制，剪的对象相反。

| 论文 | 作者 | Venue/年份 | arXiv | 稀疏性 | 一句话方法 |
|---|---|---|---|---|---|
| **Antidote: Post-fine-tuning Safety Alignment for LLMs against Harmful Fine-tuning Attack** | Huang, Bhattacharya, Joshi, Kimball, Liu | ICML 2025（PMLR v267:25059–25074） | 2408.09600 ✅ | 是（one-shot 剪除 top-α "harmful" 坐标） | 微调后在 re-alignment 数据上算逐权重重要性分数，一次性剪掉最"有害"的少量标量权重恢复对齐，对微调超参不可知。 |
| **SafeDelta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets** | Lu, Liu, Wu, Chen, Zhang, Ong, Wang, Tang | ICML 2025（PMLR v267） | 2505.12038 ✅ | 是（逐标量 delta 参数 mask + 补偿向量） | Hessian/OBS 式估计每个微调 delta 参数对 safety 的损害，在安全损失预算内贪心选出保留子集，其余坐标加安全补偿向量。 |
| **Alleviating the Fear of Losing Alignment in LLM Fine-tuning** | Yang, Tao, Chen, Xu | arXiv 2025（作者主页 PDF 标注 IEEE S&P 2025，venue **[需核实]**） | 2504.09757 ✅ | 是（小量子集权重回滚） | 识别 "harmful direction"，用梯度引导把微调模型的一小撮权重**恢复**为原始对齐模型的值，配 rollback 机制保护 utility；125 个微调模型上有害率 33.25%→1.74%。 |
| **NLSR: Neuron-level Safety Realignment of LLMs against Harmful Fine-tuning** | Yi, Zheng, Wang, de Melo, Wang, He | arXiv 2024 | 2412.12497 ✅ | 是（神经元粒度 patch） | 免训练：构建 safety 参考模型，找出微调后相似度变化最大的 safety 关键神经元，只在这些神经元做权重"移植"。 |
| **Separate the Wheat from the Chaff: A Post-Hoc Approach to Safety Re-Alignment (IRR)** | Wu, Lu, Zhao, Qin | arXiv 2024 | 2412.11041 ✅ | 是（delta 参数稀疏 mask + OBS 补偿） | Identify-Remove-Recalibrate：用 safety 向量识别符号与 safety 方向冲突的 delta 权重并置零，再对保留参数做 Hessian 逆补偿。 |
| **SafeMERGE: Preserving Safety Alignment via Selective Layer-Wise Model Merging** | Djuhera, Kadhe, Ahmed, Zawad, Boche | arXiv 2025 | 2503.17239 ✅ | 结构化稀疏（层粒度） | 只对行为偏离安全的层做"微调模型↔对齐模型"选择性合并，其余层不动。 |

**与 BLADE 的关系**：Antidote 的哲学（"移除 harmful 参数即可恢复，无论它们如何形成"）与 BLADE 在数学上是同一个操作（按重要性分数剪标量权重），区别只在**重要性分数的定义方向**（harm 分数 vs. refusal 分数）。这说明 BLADE 的归因管线几乎可以直接搬到这一类——也意味着这一类对 BLADE"反向加固"的新意构成最强的prior-art 压力。

### 类别 4：抗微调防御——哪些是 sparse-weight 层面？

结论：**主流旗舰防御全部是 dense 全模型方法**。稀疏的只出现在"冻结子集"和"低秩子空间"两种形态。

| 论文 | 作者 | Venue/年份 | arXiv | Sparse-weight 层面？ | 一句话方法 |
|---|---|---|---|---|---|
| **Vaccine: Perturbation-aware Alignment for LLMs against Harmful Fine-tuning** | Huang, Hu, Liu | NeurIPS 2024 | 2402.01109 ✅ | ❌ dense（全层 embedding 扰动） | 对齐阶段渐进加入扰动，学习对扰动不变的安全 embedding。 |
| **RepNoise: Representation Noising** | Rosati et al. | NeurIPS 2024 | 2405.14577 ✅ | ❌ dense | 三层损失把有害表示推向噪声，使有害信息无法被微调恢复。 |
| **Booster: Tackling Harmful Fine-tuning via Attenuating Harmful Perturbation** | Huang, Hu, Ilhan, Tekin, Liu | ICLR 2025 | 2409.01586 ✅ | ❌ dense | 对齐阶段正则模拟一步有害微调（MAML 一阶近似），压平 harmful loss 景观。 |
| **TAR: Tamper-Resistant Safeguards for Open-Weight LLMs** | Tamirisa, Bharathi, Phan, Zhou, ... Hendrycks, Mazeika | ICLR 2025 | 2408.00761 ✅ | ❌ dense（对抗式元学习更新全部权重） | 内循环跑微调攻击、外循环更新权重，使 safeguard 扛住数百至数千步攻击。 |
| **Lisa: Lazy Safety Alignment** | Huang, Hu, Ilhan, Tekin, Liu | NeurIPS 2024 | 2405.18641 ✅ | ❌ dense（proximal 正则作用于全部参数） | 双状态优化：对齐数据/用户数据状态交替 + proximal 项锚定在对齐模型附近。 |
| **Safe LoRA: The Silver Lining of Reducing Safety Risks when Finetuning LLMs** | Hsu, Tsai, Lin, Chen, Yu, Huang | NeurIPS 2024 | **2405.16833** ✅ ⚠️ | ◐ 低秩子空间（非逐标量） | 免训练免数据：把选定层的 LoRA 矩阵投影到由 base-vs-aligned 权重差张成的"安全对齐子空间"。⚠️ **编号陷阱：2406.16834 是一篇 GAN 理论论文，不是本文。** |
| **SaLoRA: Safety-Alignment Preserved Low-Rank Adaptation** | Li, Si, Backes, Zhang, Wang | ICLR 2025 | 2501.01765 ✅ | ◐ 冻结 LoRA 内固定 safety 子空间 | 在 LoRA 更新中冻结由安全数据导出的固定"safety module"，正交投影可训练低秩矩阵。注意与 Safe LoRA 是**两篇不同的论文**，极易混淆。 |
| **T-Vaccine: Safety Alignment via Layer-wise Perturbation** | Liu, Huang, Liu et al. | arXiv 2024 | 2410.09760 ✅ | ✅ 层粒度（冻结大部分层，只对 safety-critical 层做扰动训练） | 用梯度范数识别 safety-critical 层，只对这些层做 Vaccine 式扰动训练——唯一显式"冻结多数网络"的抗微调防御，兼具类别 2 色彩。 |
| **Immunization against Harmful Fine-tuning Attacks** | Rosati et al. | arXiv 2024（框架论文） | 2402.16382 ✅ | N/A | 把有害微调防御形式化为以攻击者训练预算为参数的"免疫条件"。 |

**与 BLADE 的关系**：这一大类证明"safety 加固"需求真实存在且被社区重视，但**权重层面全部是 dense 训练**——训练和推理成本高。BLADE 式稀疏定位若能以零训练成本达到可比的加固效果，将有清晰的效率卖点。

### 类别 5：机制类证据——"safety 集中在少量权重/组件"

| 论文 | 作者 | Venue/年份 | arXiv/出处 | 核心发现 |
|---|---|---|---|---|
| **Refusal in Language Models Is Mediated by a Single Direction** | Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda | ICML 2024 MecInterp Workshop | 2406.11717 ✅ | 13 个模型上 refusal 由**一维** residual-stream 子空间介导；擦掉该方向即解除 refusal，加上即对正常 prompt 也拒绝。 |
| **On the Role of Attention Heads in LLM Safety** | Zhou, Yu, Zhang, Xu, Huang, Wang, Liu, Fang, Li | ICLR 2025 **Oral** | 2410.13708 ✅ | 消融**单个** safety 关键 attention head（≈0.006% 参数）使 Llama-2-7b-chat 攻击成功率 16×（0.04→0.64），对 helpfulness 影响很小。 |
| **Safety Alignment Should Be Made More Than Just A Few Attention Heads** | Huang, Zhang, Yue, Li, Zhang, Liu | arXiv 2025（无已确认 venue） | 2508.19697 ✅ | safety 集中在少数 head 是结构性漏洞；提出 Attention-Head Dropout 把 safety **分散**到更多 head 以增强鲁棒性。 |
| **Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron** | Zhao, Zhang, Xie, Goyal, Kawaguchi, Shieh | ICLR 2025 | **无 arXiv 版本**；OpenReview: `yR47RmND1m` ✅ | safety 神经元 <1% 参数，集中在自注意力层和**前几层**；只微调这些神经元（SN-Tune）即可大幅提升 safety。 |
| **Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons** | Chen, Wang, Yao, Bai, Hou, Li | NeurIPS 2025 | 2406.14144 ✅ | ~5% safety 神经元因果充分；⚠️ 但 safety 与 helpfulness **高度重叠**在同一批神经元上（只是激活模式不同）——稀疏干预有副作用风险。 |
| **Safety Alignment Should Be Made More Than Just a Few Tokens Deep** | Qi, Panda, Lyu, Ma, Roy, Beirami, Mittal, Henderson | ICLR 2025 | 2406.05946 ✅ | 对齐是"浅层"的：主要只改变前几个输出 token 的分布——safety 是薄薄一层修改，易被小扰动绕过。 |
| **Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!** | Qi, Zeng, Xie, Chen, Jia, Mittal, Henderson | ICLR 2024 | 2310.03693 ✅ | 少至 10 条有害样本（$0.20）即可越狱 GPT-3.5-Turbo；善意微调也会损害 safety。 |
| **Exploiting Novel GPT-4 APIs** | Pelrine, Taufeeque, Zając, McLean, Gleave | arXiv 2023 | 2312.14302 ✅ | 15 条有害或 100 条良性样本微调即可移除 GPT-4 的核心防护。 |
| **Removing RLHF Protections in GPT-4 via Fine-Tuning** | Zhan, Fang, Bindu, Gupta, Hashimoto, Kang | NAACL 2024 (Short) | 2311.05553 ✅ | 少量样本微调即可移除 GPT-4 的 RLHF 防护（与 Pelrine et al. 是不同论文）。 |
| **Improving Alignment and Robustness with Circuit Breakers** | Zou, Phan, Wang, Duenas, Lin, Andriushchenko, Wang, Kolter, Fredrikson, Hendrycks | NeurIPS 2024 | **2406.04313** ✅ ⚠️ | 直接控制负责有害输出的内部表示（representation rerouting），对未见攻击鲁棒。⚠️ **编号陷阱：2406.04311 是一篇系外行星论文。** |
| **Representation Engineering: A Top-Down Approach to AI Transparency** | Zou, Phan, Chen, Campbell, ... Kolter, Hendrycks | arXiv 2023 | 2310.01405 ✅ | 诚实、无害等高层概念在激活空间线性可读、可控——稀疏/方向性干预的方法论基础。 |
| **A Mechanistic Understanding of Alignment Algorithms: DPO and Toxicity** | Lee, Bai, Pres, Wattenberg, Kummerfeld, Mihalcea | ICML 2024 | 2401.01967 ✅ | DPO 并未删除预训练学到的毒性，只是学了一个薄的"绕过/抑制"机制——解释了为何稀疏攻击如此容易。 |

**与 BLADE 的关系**：这一类同时支撑 BLADE 与其反向操作的前提（safety 稀疏、可定位、可手术），也给出两条警告：(a) safety 神经元与 utility 神经元高度重叠（Chen et al.）；(b) 集中本身是漏洞，已有工作（2508.19697）主张把 safety **分散**而非加固少数点——这与"加固少数关键 W_ij"的思路直接冲突，是反向 BLADE 必须回应的论点。

---

## 2. 与 BLADE 的总体关系图

```
                        对 safety 相关稀疏权重的操作
        ┌──────────────────────────┬─────────────────────────────┐
        │      移除/削弱行为         │        强化/保护行为          │
        ├──────────────────────────┼─────────────────────────────┤
  标量   │ BLADE（本项目）           │  ❌ 基本空白 ← 机会所在       │
  权重   │ Antidote* / IRR*         │  （Antidote/IRR 机制相同但     │
        │ （剪 harm 参数=恢复安全） │   剪的是 harm 而非强化 refusal）│
        ├──────────────────────────┼─────────────────────────────┤
  神经元 │ Wei et al. (诊断)        │  SN-Tune, NLSR, SafetyLock,   │
  /head  │ Zhou et al. (诊断)       │  LED-Merging, Safety Neurons  │
        ├──────────────────────────┼─────────────────────────────┤
  层粒度 │ （结构剪枝破坏安全，      │  SPPFT (冻结), T-Vaccine,     │
        │  Wei et al. 等）          │  SafeMERGE                    │
        ├──────────────────────────┼─────────────────────────────┤
  方向/  │ refusal-direction 消融   │  ROSI (rank-1 注入),          │
  低秩   │ (Arditi et al.)          │  Safe LoRA / SaLoRA (子空间),  │
        │                          │  RESTA / Safety Arithmetic    │
        ├──────────────────────────┼─────────────────────────────┤
  dense  │ fine-tuning 攻击          │  Vaccine/RepNoise/Booster/    │
  全模型  │ (Qi et al. 等)           │  TAR/Lisa（全部 dense 训练）   │
        └──────────────────────────┴─────────────────────────────┘
```

\* Antidote/IRR 的目标是恢复 safety，但其**机制**是剪除 harmful 参数，因此在"移除"列——它们与 BLADE 共享工具、目标相反。

---

## 3. 空白与机会：BLADE 反过来用于强化 safety

### 3.1 具体的新意空间

1. **逐标量（unstructured sparse）的 refusal 放大 = 明确空白。**
   现有"强化"工作最好的粒度是 rank-1（ROSI）或神经元（NLSR/SN-Tune）；没有任何工作在 residual-writer 上做**逐标量权重的选择性强化**（例如对 BLADE 定位出的 refusal 关键 W_ij 施加正向缩放/正则，或在训练目标中只放开这 k 个参数做 safety 微调）。BLADE 的归因管线天然产出标量级掩码，这是直接的差异化。

2. **"放大而非剪除 refusal 边"可以作为 BLADE 论文内部的消融/对偶实验。**
   同一个掩码 M，−M 方向剪除是 BLADE，+M 方向缩放（W ← W + λ·(M⊙Δ)）即"反 BLADE"。若 +M 能以极低参数量提升 jailbreak 鲁棒性（对照 ROSI），这本身就是有发表价值的对偶性结果，且几乎不需要新基础设施。

3. **稀疏加固 vs. dense 防御的效率叙事。**
   Vaccine/RepNoise/Booster/TAR 都需要 dense 对抗训练；T-Vaccine/SPPFT 证明"只动子集"可行但停留在层粒度。BLADE 式定位若能把"要保护的参数预算"从层粒度（~10% 参数）压到标量粒度（~0.01–0.1%），在"参数冻结预算—防御强度"权衡曲线上可能占优。

4. **与安全修复类（Antidote 等）的对偶掩码。**
   Antidote 用 harm 重要性剪 harm 参数；BLADE 反用可在**同一微调后模型**上做双掩码：剪掉 harm 参数（Antidote 方向）+ 放大/保护 refusal 参数（反 BLADE 方向），检验两者是否可叠加、是否互相干扰。这是一个没人做过的组合。

### 3.2 坑（已有文献给出的硬约束）

1. **Wei et al. (ICML 2024) 的反例**：冻结 safety-critical 区域**挡不住**低成本微调攻击——攻击者可以绕开被冻结区域，用 utility 区域重建有害行为。任何"冻结/加固少数 W_ij"的防御都必须在这篇的评估协议下检验，否则结论不可信。

2. **Safety–utility 神经元重叠**（Chen et al. 2406.14144）：safety 与 helpfulness 共用神经元。放大 refusal 权重可能直接放大 over-refusal（XSTest 类指标）或损伤能力；必须同时报 utility 和 over-safety。ROSI 报了 MMLU/HellaSwag/Arc，是底线参照。

3. **"分散 vs. 加固"的路线冲突**（2508.19697）：一种有影响力的观点认为 safety 集中本身是漏洞，正解是把 safety *分散*到更多组件（Attention-Head Dropout），而不是把鸡蛋加固在同一个篮子里。反向 BLADE 需要正面回应：为什么加固少数点是优于/互补于分散化的（例如：分散需要重训练，加固是 post-hoc 零训练成本）。

4. **浅层对齐问题**（Qi et al. 2406.05946；Lee et al. 2401.01967）：对齐本身只是 base 模型能力上的"薄包装"。权重级加固如果不改变这一性质，可能只是把薄包装焊牢一点，对自适应攻击（改变攻击预算/目标）依旧脆弱——TAR 的评估（数百至数千步攻击）是合适的压力测试。

5. **评估陷阱**：
   - refusal 率上升 ≠ safety 上升（可能只是 over-refusal）；
   - 白盒定位掩码 M 公开后，攻击者可直接针对 M 之外的参数微调（Wei et al. 的场景）；
   - 与 Antidote 系方法比较时必须控制"掩码大小"这一变量，否则稀疏度不同结论无意义。

6. **定位粒度之争尚无定论**：方向（Arditi）/ head（Zhou）/ 神经元（Chen, Zhao）/ 层（Li）/ 标量（Wei）各有证据。BLADE 选择 residual-writer 标量粒度，反向应用时需要论证标量粒度相对 head/方向粒度在**加固**任务上的优势（方向粒度有 ROSI 在先，标量粒度是空白——这是机会也是举证责任）。

### 3.3 一句话结论

> "稀疏**剪除**行为"（BLADE、Antidote 系）已有清晰谱系；"稀疏**强化/保护** safety"在 rank-1/方向层面有 ROSI、在层粒度有 SPPFT/T-Vaccine、在神经元粒度有零散工作，但**在逐标量权重层面基本空白**。BLADE 的归因管线反向使用有真实新意（对偶消融、效率叙事、与 Antidote 的组合），但必须正面应对 Wei et al. 的冻结不足反例、safety–utility 重叠、以及"分散优于加固"的替代路线。

---

## 4. 附录：核实记录与陷阱

**本次调研中确认存在的易错点（均已按正确值写入正文）：**

- ⚠️ **Circuit Breakers = arXiv 2406.04313**，不是 2406.04311（后者是系外行星质量-半径论文）。
- ⚠️ **Safe LoRA (Hsu et al., NeurIPS 2024) = arXiv 2405.16833**；2406.16834 是 GAN 理论论文。且 **Safe LoRA ≠ SaLoRA**（2501.01765, ICLR 2025），是两篇不同论文。
- ⚠️ **Antidote (Huang et al., ICML 2025) = arXiv 2408.09600**；另有一篇同名 "AntiDote"（arXiv 2509.08000，bi-level 对抗防篡改），不要混淆。
- ⚠️ "Safety Alignment Should Be Made More Than **Just A Few** Attention Heads"（2508.19697）的确切标题含 "Just A Few"，且目前**仅是 arXiv 预印本**（2025-08），无已确认 venue。
- ⚠️ 两篇 "safety neurons" 论文易混：Zhao et al.（ICLR 2025，<1% 参数、前几层、**只有 OpenReview 无 arXiv**，id `yR47RmND1m`）vs. Chen et al.（NeurIPS 2025，~5% 神经元、arXiv 2406.14144）。

**检索后确认不存在、已丢弃（未写入正文表格）：**

- ❌ "SaMERU: Safer Model Merging"——未找到该 LLM 论文（该名字只出现在无关的道路安全文献中），疑似记忆性幻觉。
- ❌ "RecoverLLM"——未找到该论文。
- ❌ "On the Robustness of Safety Alignment in Instruction-Tuned Language Models"——该确切标题未找到；该位置的真实论文是 Li et al. (2408.17003) 与 Qi et al. (2310.03693 / 2406.05946)。

**venue 待核实条目：** SafetyLock（2410.10343，ACL/OpenReview 投稿状态未确认）；Yang et al.（2504.09757，作者 PDF 标注 IEEE S&P 2025，未经官方页面独立确认）。

**抽查记录（父代理独立完成）**：2408.09600（Antidote）、2508.20766（ROSI）、2406.11801（Safety Arithmetic）、2504.09757（Yang et al.）四个摘要页已逐一抓取复核，标题/作者/摘要均匹配；其余编号由三个并行调研代理各自抓取 arXiv/PMLR/ACL/OpenReview 页面核实。
