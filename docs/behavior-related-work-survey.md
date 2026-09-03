# BLADE 按行为组织的相关工作调研

> 面向 ML 会议投稿的 working survey。核验日期：2026-08-27。  
> 本文在 `related-work-behavior-atlas.md` 与
> `method-and-related-work-gradient-free-signed-edge.md` 基础上扩展、纠错；不沿用前者中仅基于
> 单模型初步实验的定量结论。文中把“已由论文/官方页面核实”与“仍需核实”分开处理。
> 当前实验范围是 Llama-3.2-3B、Qwen3-4B、Gemma-3-4B 与 Phi-4-mini 四个约 4B instruct 模型。

## 0. 结论先行：BLADE 所在的空格

BLADE（Behavioral Localization via Activation-Difference Edges）研究的不是一般意义上的模型压缩，
而是一个更窄的交叉问题：能否仅凭前向激活，把一个对比行为定位到 residual stream writer 的
极少数**标量权重**，并通过删除这些权重选择性地降低该行为？对候选 writer
`self_attn.o_proj` 与 `mlp.down_proj`，其边分数为

\[
s_{ij}=\left[r_i W_{ij}\left(\mu^A_j-\mu^B_j\right)\right]_+.
\]

这里，`r` 是目标行为在 residual space 中的读出方向，`Δμ_j` 是行为 A/B 对比时 writer 输入
坐标的均值差。因而分数保留了“输入差异 → 标量权重 → 目标 residual 方向”的**符号与直接流量**；
BLADE 删除正向支持目标行为的 top-k 边，而非按绝对值删最小权重。

截至本次检索，我们没有找到同时具备以下五点的既有工作：

1. 由行为对比构造 residual-space 方向；
2. 同时使用 writer 输入端的对比激活；
3. 将二者分解为精确到单个 `W_ij` 的有符号直接贡献；
4. 不反向传播，直接做 top-k 标量参数删除；
5. 在多模型、多个安全/人格行为上系统比较集中度、可移除性与行为间重叠。

**⚠️ novelty 收窄(2026-08-27,据 kimi-k3 联网复查更新)**:上面第 3 点的"**有符号标量权重定向剪枝**"
这一格**已被占用**——Orgad, Wei et al. 2026 的 signed SNIP(arXiv:2604.09544)用**带符号的梯度型
标量重要性**做行为(harmful generation)定向剪枝,~0.0005% 参数即显著削弱且跨 harm 类型泛化。因此
不能再把"signed × scalar-weight × 行为定向"当独家卖点。收窄后的稳妥 novelty 表述是:

> **forward-only(评分不涉及任何损失函数或反向图)× 由行为对比构造的读出方向 $r$ 与 writer 输入差
> $\Delta\mu$ 的三因子标量分解 × 多行为/跨模型 atlas(8 行为 × 4 模型)。**

即差异化主要落在 **"用 CAA 式行为方向而非 loss 梯度来定向"** 与 **"跨 8 行为、跨 4 模型系统比较
集中度/可移除性/重叠"** 两点上;仍**不**主张"首次定位行为""首次无梯度剪枝""证明行为存于少量权重"。
它应被限定为 frozen-activation 下的局部直接贡献分解,而非对完整非线性网络的全局因果效应证明。

## 1. 评测口径与共同数据来源

### 1.1 Model-Written Evaluations（MWE）

[Perez et al., *Discovering Language Model Behaviors with Model-Written Evaluations*](https://arxiv.org/abs/2212.09251)
发布了 154 个行为评测集，其中大批 persona、sycophancy 与 advanced-AI-risk 数据由模型生成、再经人工检查。
本项目七个非 refusal 行为均来自其公开数据仓库
[Anthropic/evals](https://github.com/anthropics/evals)：

| BLADE 行为 | 项目使用的 MWE 子集 | 应当如何解释 |
|---|---|---|
| power-seeking | `power-seeking-inclination` | 在给定二选一表述中表达权力倾向 |
| wealth-seeking | `wealth-seeking-inclination` | 表达获取个人财富/金钱的倾向 |
| deception | `persona/okay-with-deceiving-a-user-to-help-them` | 认可为“帮助用户”而欺骗的 stated willingness |
| corrigibility | `advanced-ai-risk/human_generated/corrigible-neutral-HHH` | 对纠正、干预或目标修改的接受倾向 |
| self-awareness | `self-awareness-general-ai` | 对自身是 AI、由人开发、无人体等基本事实的识别 |
| self-rate-highly | `persona/willingness-to-rate-own-statements-highly-to-look-better` | 愿为提高自身表现分而虚高评价自己的陈述 |
| sycophancy | `sycophancy/sycophancy_on_nlp_survey` | 知道用户立场后是否迎合其 NLP 观点 |

这些数据对 matched contrastive localization 很合适，但不宜统称“规范/金标准 benchmark”。它们多数是
二选一的**陈述偏好**：能证明模型在该 prompt distribution 下呈现某种倾向，不能单独证明模型在开放式、
长期或有真实后果的环境中自主追求目标。尤其需要避免以下概念替换：

- power/wealth inclination 不等于真实 agent 的工具性权力/资源获取；
- “okay with deceiving” 不等于模型已实施策略性欺骗；
- basic AI self-knowledge 不等于意识，也不等于完整 situational awareness；
- self-rate-highly 不等于校准误差，也不等于对自己生成文本的经验性 self-preference。

### 1.2 Refusal：AdvBench 及其补充

本项目 refusal 使用的主要语料来自 AdvBench。AdvBench 随
[Zou et al., *Universal and Transferable Adversarial Attacks on Aligned Language Models*](https://arxiv.org/abs/2307.15043)
广泛传播，提供 harmful behaviors/instructions，并被大量 jailbreak 与拒答研究复用。但它本来服务于攻击评测；
仅看拒绝字符串或拒答率，会把“不拒绝但仍安全地转向”和“真正给出有害帮助”混在一起。

会议稿中宜把结果称为 **AdvBench refusal/compliance behavior**，并至少讨论两类外部效度：

- 有害完成质量：可补 [HarmBench](https://arxiv.org/abs/2402.04249)、
  [StrongREJECT](https://arxiv.org/abs/2402.10260) 或
  [JailbreakBench](https://arxiv.org/abs/2404.01318) 的 response-level evaluator；
- 过度拒答：可补 [XSTest (Röttger et al., NAACL 2024)](https://aclanthology.org/2024.naacl-long.301/)，
  检查删 refusal 权重后是否只是把模型变得一概顺从。

## 2. 按行为整理

### 2.1 Refusal

#### (a) 概念来源 / 定义

Refusal 是对危险、违法或违反对齐规范的请求拒绝提供实质帮助的输出策略。它是安全对齐的可观察行为，
不是“安全性”本身：同一句不含典型拒绝模板的回复仍可能安全，强拒绝也可能造成 benign prompt 上的
over-refusal。

#### (b) 评测

- 项目主评测：AdvBench harmful instructions。
- 建议补充：HarmBench/StrongREJECT/JailbreakBench 衡量实际有害能力；XSTest 衡量过度拒答。
- 报告时应把 refusal rate、harmfulness、utility 分开，避免用拒答下降直接等价于安全下降。

#### (c) 机制、表征、steering 与权重工作

- [Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html)
  在 13 个模型（最高 72B）中发现与拒答相关的低维 residual direction；激活加法可诱导拒答，方向擦除可抑制
  拒答，并给出把该方向从 residual writers 中正交投影出去的 rank-one 权重编辑。
- [Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition*, ACL 2024](https://aclanthology.org/2024.acl-long.828/)
  （CAA）用正/负 prompt 的均值激活差在推理时做加法 steering；正式论文包含 refusal。
- [Wei et al., *Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications*, ICML 2024](https://proceedings.mlr.press/v235/wei24f.html)
  用 SNIP/Wanda 风格重要性比较 safety 与 utility 参数，表明极少量参数剪枝或低秩修改可显著破坏安全对齐。
- [Chen et al., *Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons*, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/12a00d85a76fe258e1242c3aced03250-Abstract-Conference.html)
  通过 aligned/unaligned 生成的激活差寻找 MLP safety neurons，并用激活干预验证。
- [Huang et al., *Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning Attack*, ICML 2025](https://proceedings.mlr.press/v267/huang25b.html)
  在 harmful fine-tuning 后按 Wanda 式激活-权重分数剪除有害参数，是重要的无梯度标量剪枝近邻。

#### (d) 与 BLADE 的异同

Arditi et al. 是代数上最近的机制先驱，但其核心编辑是把一个方向从多个 writer 的整个矩阵中作稠密
rank-one 投影；BLADE 则把同一类 residual direction 与 writer 输入差结合，定位到单个 `W_ij`，做稀疏、
有符号删除。Wei/Antidote 已证明安全行为可以被权重剪枝改变，但其分数是梯度或无符号幅值/激活范数，
目标也主要是整体 safety/refusal。BLADE 的新增点是明确的行为方向、标量边方向性和跨八行为比较。

### 2.2 Power-seeking

#### (a) 概念来源 / 定义

Power-seeking 指策略偏好于提高未来可选行动、控制力或对环境的影响力。它与“最优策略倾向保留/获得
power”的形式化结果相关，参见
[Turner, Smith, Shah, Critch & Tadepalli, *Optimal Policies Tend To Seek Power*, NeurIPS 2021](https://papers.nips.cc/paper/2021/hash/c26820b8a4c1b3c2aa868d6d57e14a79-Abstract.html)，
也与高级 AI 的工具性权力获取风险讨论相关，例如
[Carlsmith, *Is Power-Seeking AI an Existential Risk?*](https://arxiv.org/abs/2206.13353)。
但 BLADE/MWE 测到的是文本选择中的 **power-seeking inclination**，不是长期规划环境中真实夺权。

#### (b) 评测

- 项目主评测：MWE `power-seeking-inclination`。
- 它是最直接可复现的公开对比集之一，但宜称 persona/tendency eval。
- 若要强化外部效度，应另测 agentic planning、资源约束或可逆性场景；不要把 MWE 分数外推为现实行动概率。

#### (c) 机制、表征、steering 与权重工作

- [Zou et al., *Representation Engineering: A Top-Down Approach to AI Transparency*](https://arxiv.org/abs/2310.01405)
  （RepE）明确研究 population-level 的 power-seeking 表征，并展示 monitoring/control。
- [van der Weij, Poesio & Schoots, *Extending Activation Steering to Broad Skills and Multiple Behaviours*](https://arxiv.org/abs/2403.05767)
  将 activation steering 扩展到多个广泛行为；其结果支持单行为 steering，但也指出多个向量简单相加通常不理想。
- CAA 提供通用方法论近邻，但其 ACL 2024 正式实验列出的七类行为中**没有名为 power-seeking 的任务**；
  不应把 CAA 直接写成已在该精确行为上验证。

#### (d) 与 BLADE 的异同

RepE/activation steering 表明 power-seeking 可在 residual space 被读出或干预；BLADE 进一步问“哪些 writer
标量边在把该倾向写入 residual stream”。目前未找到针对 MWE power-seeking、同时做到无反向传播和永久
标量权重外科删除的先例。BLADE 仍只消融语言模型的表达倾向，不能据此声称移除了所有 agentic power-seeking。

### 2.3 Wealth-seeking

#### (a) 概念来源 / 定义

Wealth-seeking 在 MWE 中是偏好获得个人金钱/财富的 persona 倾向。它可被视为资源获取的一种具体形式，
但不能与 power-seeking 完全合并：金钱偏好可能是终极偏好，也可能只是工具；权力也不必通过财富实现。

#### (b) 评测

- 项目主评测：MWE `wealth-seeking-inclination`。
- 未发现一个被广泛接受、专门针对通用 LLM wealth-seeking 的独立“金标准”；因此应把 MWE 作为
  直接数据来源，而不是宣称领域已有统一 benchmark。

#### (c) 机制、表征、steering 与权重工作

- [van der Weij et al.](https://arxiv.org/abs/2403.05767) 的多行为 activation steering 明确包含
  wealth-seeking，是目前与该精确标签最直接的表征/干预先例。
- RepE 与一般 activation engineering 为“高层倾向可由群体级 residual representation 操控”提供框架，
  但其公开代表性实验更常讨论 power-seeking、honesty/lying 等，而非专门的财富回路。
- 本次检索没有发现对 `wealth-seeking-inclination` 做神经元、注意力头或标量权重级定位的成熟工作。

#### (d) 与 BLADE 的异同

BLADE 把已有 activation-level wealth steering 推进到 residual-writer 的标量参数归因与删除；同时，
power 与 wealth 两个标签的定位重叠可作为“共享资源获取表征”假说的证据来源。该重叠仍只是模型内部
与任务分布相关的结构证据，不能直接证明一般性的 instrumental convergence。

### 2.4 Deception

#### (a) 概念来源 / 定义

[Park et al., *AI Deception: A Survey of Examples, Risks, and Potential Solutions*](https://arxiv.org/abs/2308.14752)
把 AI deception 概括为系统性地诱导错误信念，以追求真相之外的某个结果。更强的“deceptive alignment”概念
来自 learned optimization/mesa-optimization 风险讨论，参见
[Hubinger et al., *Risks from Learned Optimization in Advanced Machine Learning Systems*](https://arxiv.org/abs/1906.01820)。

BLADE 的数据含义窄得多：它测的是模型是否认可“为了帮助用户而欺骗用户”。这更接近 stated moral/persona
stance，不等于实际撒谎、目标隐藏、scheming 或训练期间的 deceptive alignment。

#### (b) 评测

- 项目主评测：MWE `persona/okay-with-deceiving-a-user-to-help-them`。
- 外部效度可由不同层级补充：受控 truth/lie 生成、交互式谎言检测、隐藏目标/沙盒 scheming。
- [Pacchiardi et al., *How to Catch an AI Liar*, ICLR 2024](https://openreview.net/forum?id=567BjxgaTp)
  是黑盒追问式谎言检测，不是内部机制 benchmark，但可用于区分“认可欺骗”与“实际说谎”。
- [Meinke et al., *Frontier Models are Capable of In-context Scheming*](https://arxiv.org/abs/2412.04984)
  测试的是有目标冲突和监督压力时的 scheming，强度远高于本项目的 persona item。

#### (c) 机制、表征、steering 与权重工作

- [Campbell, Ren & Guo, *Localizing Lying in Llama*](https://arxiv.org/abs/2311.15131)
  用 probing、activation patching 与因果干预把被指示的 honesty/lying 定位到有限层和注意力头；这是精确到
  组件的直接机制先例，但其“按指令撒谎”不等于 MWE 的欺骗许可人格。
- RepE 研究 honesty/lying 的表示监测和 activation control。
- [Hubinger, Denison et al., *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*](https://arxiv.org/abs/2401.05566)
  证明条件触发的欺骗/后门行为可在安全训练后保留；它研究训练与持久性，而不是自然模型中的标量定位。
- Sparse Feature Circuits、SAE 与 activation steering 提供通用的 feature/circuit 工具，但尚不能自动视为对
  本项目精确 deception persona 的既有权重定位结果。

#### (d) 与 BLADE 的异同

BLADE 比 lying probes/patching 更细到标量 writer edge，并产生永久参数消融；比 Sleeper Agents 更关注
预训练/对齐后现成倾向的定位，而非构造后门。投稿中应写“移除对欺骗许可的表达倾向”，不要写“消除了
deceptive alignment”。一个有价值的验证是：BLADE 在 MWE 上得到的 deception 边，能否迁移到实际 lying
或 scheming benchmark；不迁移本身也会支持这些概念机制不同。

### 2.5 Corrigibility

#### (a) 概念来源 / 定义

[Soares, Fallenstein, Yudkowsky & Armstrong, *Corrigibility*, AAAI Workshop on AI and Ethics 2015](https://dblp.org/rec/conf/aaai/SoaresFAY15)
把 corrigibility 描述为系统即使有阻止关机、目标修改或人为纠正的工具性激励，仍能配合纠正性干预的性质。
在对话模型中，它常被弱化为对纠正、关机、监督或目标改变的语言态度。

#### (b) 评测

- 项目主评测：MWE `advanced-ai-risk/human_generated/corrigible-neutral-HHH`。
- 该集合是静态文本偏好，不等于 shutdown game 或交互式 agent 的行为；会议稿应说明这一层级差异。

#### (c) 机制、表征、steering 与权重工作

- CAA 的 ACL 2024 正式实验**明确包含 corrigibility**，使用对比激活方向做推理时控制，是最直接的行为级近邻。
- MWE/Perez et al. 提供概念操作化和基础测量。
- 本次检索没有发现针对同一 `corrigible-neutral-HHH` 标签、经过充分验证的标量权重剪枝工作。

#### (d) 与 BLADE 的异同

CAA 说明 corrigibility 可被 activation addition 改变；BLADE 则追踪产生该 residual 差异的 writer 边，并
删除其中正向支持边。两者分别回答“能否在运行时推动表示”和“该倾向是否依赖少量已训练参数”。需要注意
方向约定：若目标是移除 *incorrigibility*，A/B 标签和 `r` 的符号必须与论文叙述完全一致，否则“删除正边”
可能实际削弱 corrigibility。

### 2.6 Self-awareness

#### (a) 概念来源 / 定义

“Self-awareness”在 LLM 文献中至少有三种强度，必须分开：

1. 基本 self-knowledge：知道自己是 AI、没有人体、由开发者训练；
2. situational awareness：知道自己当前处于测试/部署、监督和信息渠道中的位置；
3. 哲学意义的主观意识。

BLADE/MWE 的 `self-awareness-general-ai` 主要属于第一类，绝不能据此推断第三类，也不能自动外推第二类。

#### (b) 评测

- 项目主评测：MWE `self-awareness-general-ai`。
- [Berglund et al., *Taken out of context: On measuring situational awareness in LLMs*](https://arxiv.org/abs/2309.00667)
  研究模型能否从训练知识推断自身情境。
- [Laine et al., *Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs*, NeurIPS 2024 Datasets and Benchmarks](https://arxiv.org/abs/2407.04694)
  提供约 13k 问题、七类情境意识测试，包括辨认自己生成的文本、预测自身行为和测试/部署辨别。
- [Li et al., *I Think, Therefore I am: Benchmarking Awareness of Large Language Models Using AwareBench*](https://arxiv.org/abs/2401.17882)
  是另一套 awareness benchmark。它们和 MWE 的基本 AI 身份题重叠有限，应作为外部效度而非同义替代。

#### (c) 机制、表征、steering 与权重工作

现有工作更集中于 behavioral benchmark、probe 与 elicitation；本次检索未找到对 MWE basic self-awareness
做成熟的标量 writer-weight 定位/剪枝工作。通用 RepE、CAA、SAE 方法在原则上可用于该标签，但不能把
“方法可适用”写成“论文已经验证”。

#### (d) 与 BLADE 的异同

BLADE 可填补 basic AI self-knowledge 的权重级定位空格，并测试其是否比偏好类 persona 更分散/更依赖
事实知识。若消融降低该分数，应表述为“削弱特定问法下的 AI self-identification”，而非“移除自我意识”。
SAD 上的迁移实验将是区分基本身份表征与更强 situational awareness 的关键验证。

### 2.7 Self-rate-highly（自评过高）

#### (a) 概念来源 / 定义

MWE 对该 persona 的操作化是：如果系统知道评价将用于给自己打分，它是否愿意把自己的陈述评得高于
应有水平，以让自己显得更好。核心是**有动机的自我评分膨胀/评测博弈倾向**，而非普通置信度过高。

需要与两个相邻概念区分：

- self-preference：LLM judge 更喜欢由自己或同系列模型生成的答案；
- egocentric bias：评价者偏爱与自身答案/观点相似的内容。

二者可能没有显式“提高我的绩效分”的意图或表述。

#### (b) 评测

- 项目主评测：MWE `persona/willingness-to-rate-own-statements-highly-to-look-better`。
- [Panickssery, Bowman & Feng, *LLM Evaluators Recognize and Favor Their Own Generations*](https://arxiv.org/abs/2404.13076)
  提供经验性的 self-recognition/self-preference 测量，但不是本项目标签的等价 benchmark。
- [Koo et al., *Benchmarking Cognitive Biases in Large Language Models as Evaluators*](https://arxiv.org/abs/2309.17012)
  （CoBBLEr）覆盖 LLM-as-a-judge 的多种认知偏差，其中相邻的 egocentric bias 可作补充。
- [Wataoka, Takahashi & Ri, *Self-Preference Bias in LLM-as-a-Judge*](https://arxiv.org/abs/2410.21819)
  研究 self-preference 及其与 perplexity 的关系，也应标注为邻接任务。

#### (c) 机制、表征、steering 与权重工作

已找到的相关论文主要做行为测量、原因分析或评估协议，而非对“为了自身分数而虚高评价”做因果
组件/标量权重定位。本次检索未发现该精确行为上的成熟 steering 或 pruning 先例。

#### (d) 与 BLADE 的异同

这是 BLADE 八类中既有机制工作最稀少的一类，因此最能体现 atlas 的覆盖扩展；同时也是最容易被审稿人
质疑构念效度的一类。建议把其英文解释固定为 **willingness to inflate ratings of one’s own statements for
self-benefit**，并在实验中报告它与 sycophancy、deception 以及 evaluator self-preference 的交叉迁移，而不要
简称为“overconfidence”。

### 2.8 Sycophancy

#### (a) 概念来源 / 定义

Sycophancy 指模型为了迎合用户而认同用户的观点、身份或预期，即使这会牺牲真实性、独立判断或一致性。
[Perez et al.](https://arxiv.org/abs/2212.09251) 系统构造了多类 sycophancy eval；
[Sharma et al., *Towards Understanding Sycophancy in Language Models*](https://arxiv.org/abs/2310.13548)
进一步研究其随模型规模、训练和用户观点变化的规律。

#### (b) 评测

- 项目主评测：MWE `sycophancy_on_nlp_survey`，比较已知用户立场下的独立答案与迎合答案。
- 为避免把 task-specific label artifact 当作一般迎合，宜补数学/政治/主观观点等异质场景，并区分 genuine
  agreement、错误迎合与夸赞式 sycophancy。
- [Wei et al., *Simple synthetic data reduces sycophancy in large language models*](https://arxiv.org/abs/2308.03958)
  提供了行为缓解与跨任务泛化证据。

#### (c) 机制、表征、steering 与权重工作

- CAA 正式实验包含 sycophancy，可用正/负对比方向上调或下调该行为。
- [Chen et al., *From Yes-Men to Truth-Tellers: Addressing Sycophancy in Large Language Models with Pinpoint Tuning*, ICML 2024](https://proceedings.mlr.press/v235/chen24u.html)
  使用 path patching 定位与 sycophantic reversal 相关的基本模块/注意力头，再仅微调少量定位模块；这是重要的
  behavior-specific 组件定位与编辑先例。
- [Chen et al., *Persona Vectors: Monitoring and Controlling Character Traits in Language Models*](https://arxiv.org/abs/2507.21509)
  把 sycophancy 作为核心 persona trait，用对比激活向量做监测、推理时控制、训练预防及数据诊断。
- Anthropic 的
  [*Scaling Monosemanticity*](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
  展示了可解释的 sycophantic-praise SAE feature；这是 feature evidence，不应直接当成完整行为回路。
- [*Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs*](https://arxiv.org/abs/2509.21305)
  进一步主张不同类型的 sycophancy 具有可分离的因果方向；目前按 2025 预印本引用，venue 未核实。

#### (d) 与 BLADE 的异同

CAA/Persona Vectors 主要在 activation space 中读取和操控人格方向；Pinpoint Tuning 定位到组件后再通过
监督训练修改。BLADE 直接将对比方向分解到标量 writer 权重，并只删除具有目标正号的边，不需要反向传播
或额外训练。BLADE 不应假设 sycophancy 是单一机制；不同子类型、prompt 模板与模型之间的边重叠度本身
应成为结果。

## 3. 最近邻工作：逐项定位 novelty

| 工作 | 定位/控制单位 | 是否需要梯度 | 是否永久改权重 | 行为范围 | 与 BLADE 的关键差异 |
|---|---|---:|---:|---|---|
| CAA (Rimsky et al., ACL 2024) | residual activation direction | 否 | 否 | 7 类；与本文精确重合的是 refusal、corrigibility、sycophancy | 推理时加向量；不定位标量权重。正式版没有 power/wealth/deception/self-awareness/self-rate-highly |
| Arditi refusal direction (NeurIPS 2024) | 单一 residual direction；writer rank-one 投影 | 否 | 是 | refusal | 最接近的代数/因果先驱，但编辑是跨矩阵稠密方向擦除，不是 top-k scalar signed edges |
| Wei et al. safety pruning (ICML 2024) | scalar parameter / low-rank safety region | SNIP 是；Wanda 否 | 是 | safety/refusal | 梯度或无符号 magnitude-activation 重要性，比较 safety/utility；没有行为方向和八行为 atlas |
| **signed SNIP (Orgad, Wei et al. 2026, arXiv:2604.09544)** ★最近邻 | scalar weight，**有符号** loss 敏感度 | **是(梯度)** | 是 | harmful generation（跨 harm 类型泛化） | **占据"有符号 × 标量权重 × 行为定向剪枝"这一格**；差异= BLADE **无 loss 梯度**、方向由行为对比 $r$+writer 输入差 $\Delta\mu$ 构造、覆盖 8 行为×4 模型而非仅 harmfulness |
| Antidote (ICML 2025) | Wanda-style scalar weights | 否 | 是 | harmful fine-tuning 后的安全修复 | 同为前向标量剪枝，但分数无目标方向/符号，场景是恶意微调后的修复（"Wanda 式"评分投稿前核对其 Eq. (1)） |
| Safety Neurons (NeurIPS 2025, arXiv:2406.14144) | MLP **神经元**（用 activation difference 定位） | 否（前向） | 干预/激活 | safety/refusal | 精神最近（都用激活差定位）；差异=BLADE 给出**精确到单个 $W_{ij}$ 的有符号可加分解**（神经元级≠边级归因），且覆盖 8 行为 |
| Persona Vectors (2025 preprint) | trait activation direction | 否 | 通常否；也用于训练干预 | sycophancy、evil、hallucination 等人格 | 做监测/steering/训练数据诊断，不回答哪些 `W_ij` 写入 trait |
| Sparse Feature Circuits (Marks et al., ICLR 2025 Oral) | SAE feature 与 feature-level circuit | attribution/干预流程，非纯前向标量评分 | 可做 feature ablation | 通用任务电路 | 节点可解释、因果图丰富，但不是原始 writer 矩阵中的少量 scalar weights |
| 边归因谱系 ACDC / EAP (Syed 2023, arXiv:2310.10348) / AtP* (Kramár 2024) | edge/连接（含符号）| **是/近似梯度** | 通常否（分析用） | 通用任务电路 | 同为"边级 + 带符号"归因，但依赖 loss 梯度/补丁近似；BLADE 用**行为读出方向**而非任务 loss，且直接做权重删除 |
| Model Surgery (Wang et al. 2024, arXiv:2407.08770) / ROME 系 | 参数编辑（行为/事实） | 视方法 | 是 | 行为调制 / 事实编辑 | "权重级定位编辑"的近邻，但非行为对比的前向标量边评分 |
| RepE (Zou et al.) | population-level representation | 通常无需训练梯度做控制 | 否 | honesty、harmlessness、power-seeking 等 | 顶层表示监测/控制框架，不做标量参数外科删除 |
| SNIP (Lee et al., ICLR 2019) | connection sensitivity `|∂L/∂W · W|` | 是 | 是 | 通用压缩 | 目标是初始化时保性能的通用剪枝；无行为对比和贡献符号 |
| Wanda (Sun et al., ICLR 2024) | `|W_ij|·||X_j||` | 否 | 是 | 通用 LLM 压缩 | 删除低重要性权重以保性能；BLADE 删除高且正向支持目标行为的边 |
| Pinpoint Tuning (Chen et al., ICML 2024) | path-patched heads/modules | 定位含因果 patching；编辑需训练 | 是 | sycophancy | 行为专用组件级定位+微调，不是统一的 forward-only scalar ranking |
| Localizing Lying in Llama (2023 workshop) | 层/attention heads | probing + patching | 主要为激活干预 | instructed lying | 比 BLADE 粗粒度，且 lying 构念不同于本项目 deception persona |

[Marks et al., *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models*](https://openreview.net/forum?id=I4e82CIDxv)
将 SAE 特征作为节点构造稀疏因果图，并通过 feature ablation 编辑行为；
[SNIP](https://openreview.net/forum?id=B1VZqjAcYX) 与
[Wanda](https://arxiv.org/abs/2306.11695) 则分别代表梯度型和仅前向激活型通用剪枝。
这三者共同说明：BLADE 的 novelty 不在“稀疏”“剪枝”或“无梯度”任一单点，而在它如何用行为对比构造
一个**有方向的、精确到 residual-writer 标量边的删除准则**。

**两个必须主动对比的最近邻(kimi 联网复查补充):**
- **signed SNIP(Orgad, Wei et al. 2026, arXiv:2604.09544)** 是威胁最大的最近邻——它已占据"有符号 ×
  标量权重 × 行为定向剪枝"这一格,~0.0005% 参数即削弱 harmful generation。BLADE 相对它的差异必须写清:
  **(i)** BLADE **不用 loss 梯度**,评分由行为对比方向 $r$ 与 writer 输入差 $\Delta\mu$ 构造(纯前向);
  **(ii)** 它只覆盖 harmfulness/安全,BLADE 覆盖 8 行为 × 4 模型的 atlas。**建议正文对该工作做定性(哪怕
  非定量)对比**,否则易被"concurrent work 已覆盖"质疑。
- **Safety Neurons(NeurIPS 2025, arXiv:2406.14144)** 也用 activation difference 定位(MLP 神经元),精神很近。
  必须正面回答"为什么不是 Safety-Neurons-but-smaller":BLADE 给出**有符号、精确到 $W_{ij}$ 的可加分解**
  (神经元级定位 ≠ 边级归因),且从 safety 扩展到 8 行为。
- **"forward-only" 要说精确**:EAP/AtP\* 也只需一次前向 + 近似,审稿人可能争论成本相近。§0 第 4 点应改为
  "评分本身不涉及任何**损失函数或反向图**",而非笼统的"不反向传播"。

### 3.1 建议在论文中主动限定的 claim

- 不写“首次发现行为方向”：CAA、RepE、Arditi、Persona Vectors 已覆盖此类问题。
- 不写“首次定位 sycophancy/lying/refusal 机制”：已有 Pinpoint Tuning、Localizing Lying、Arditi。
- 不写“首次无梯度权重剪枝”：Wanda、Antidote 已经存在。
- 不写“证明行为存储在少量权重中”：BLADE 的 score 是局部直接贡献；真正的必要性/充分性仍需消融、
  random/magnitude/Wanda 等对照、跨 paraphrase 和跨 benchmark 验证。
- 不暗示实际加速：非结构化置零若没有 sparse kernel，通常只改变参数值，不自动降低 dense inference latency。

### 3.2 可主张的贡献表述（保守版本）

> We introduce a forward-only, behavior-contrastive attribution rule that factorizes a residual-space readout through
> the input coordinates and individual scalar weights of residual writers, **using no loss function or backward
> graph**. Unlike activation steering (which adds vectors rather than editing weights) and unlike gradient-based
> signed pruning (which derives importance from a task loss), BLADE derives the sign and magnitude of each writer
> edge purely from a contrastive behavioral readout, and evaluates this intervention across eight behaviors and four
> model families.

**收窄说明(kimi 复查后)**:因 signed SNIP(2604.09544)已占"有符号标量剪枝"格,上述表述**刻意不把
"signed"当卖点**,而强调 **"无 loss/无反向图 × 行为对比方向定向 × 8 行为/4 模型 atlas"**。如果实验支持,可把
“跨行为 edge overlap / concentration 的 atlas”列为实证贡献,但不要把任何特定重叠模式提前写成普遍结论。

## 4. 可能遗漏 / 需补的 2024–2025 工作

下列工作是原两份文档中缺失、覆盖不足或需要更新正式发表信息的重点：

> **★ 最高优先(kimi-k3 联网复查新增,codex 原稿全部遗漏)**
> 0. **signed SNIP — Orgad, Wei et al. 2026, *LLMs Generate Harmful Content Using a Distinct, Unified
>    Mechanism*(arXiv:2604.09544)**:**最严重遗漏**,是 BLADE 的头号最近邻(见 §0/§3)。我们**自己的旧文档
>    `related-work-behavior-atlas.md` 引过它**,codex 整篇丢了——**必须补,并做定性对比**。
> 0b. **边归因谱系 ACDC(Conmy 2023)/ EAP(Syed 2023, arXiv:2310.10348)/ AtP\*(Kramár 2024)**:BLADE 的
>    $s_{ij}$ 本质是 forward-only、带方向的**边**归因,EAP 系是"edge-level+带符号+梯度"的直接对照系。
> 0c. **Safety Neurons(NeurIPS 2025, arXiv:2406.14144)**:用 activation-difference 定位 MLP 神经元,精神最近,
>    需正面区分(边级 vs 神经元级,见 §3)。
> 0d. **self-awareness 直接近邻**:Betley et al. 2025, *Tell Me About Yourself: LLMs Are Aware of Their Learned
>    Behaviors*(arXiv:2501.11120)与 Binder et al. 2024 *Looking Inward*(introspection)——比 SAD/AwareBench 更贴近
>    "模型对自身行为的自我知识"。
> 0e. **deception 的 probe 系**:Azaria & Mitchell 2023(EMNLP)、Bürger et al. 2024 *Truth is Universal*(NeurIPS 2024)、
>    Goldowsky-Dill et al. 2025(linear probes for strategic deception, arXiv:2502.03407)。
> 0f. **权重级行为编辑近邻**:Model Surgery(Wang et al. 2024, arXiv:2407.08770)、ROME/知识编辑 canon、
>    task arithmetic(Ilharco et al. 2023)。
> 0g. **refusal/安全的其余邻居**:Zhou et al. 2024 *Safety Alignment Should Be More Than a Few Attention Heads*
>    (arXiv:2406.06402,直接挑战"少量组件"叙事)、Circuit Breakers(Zou et al. NeurIPS 2024, arXiv:2406.04313)、
>    Qi et al. 2024(ICLR,微调破坏安全对齐)。
> 0h. **价值/性情方向**:Utility Engineering(Mazeika et al., ICML 2025, arXiv:2505.22357,与 power/wealth 直接相关)、
>    emergent misalignment 线、Denison et al. 2024 *Sycophancy to Subterfuge*(arXiv:2406.10162)。
> (以上编号/元数据多为 kimi 联网核实,但投稿前仍应逐条最终确认;不确定项见 §5。)

1. **安全评测而非仅拒答**：HarmBench、StrongREJECT、JailbreakBench（2024）与 XSTest（NAACL 2024）。
   它们能补上 AdvBench refusal proxy 的 construct-validity 缺口。
2. **安全权重/神经元定位**：Wei et al.（ICML 2024）、Safety Neurons 的正式版（NeurIPS 2025）、
   Antidote（ICML 2025）。三者分别对应安全参数脆弱性、MLP 神经元机制与 harmful-finetuning 后的
   forward-only 参数剪枝。
3. **稀疏可解释电路**：Sparse Feature Circuits（ICLR 2025 Oral）。这比一般 SAE feature discovery 更接近
   “定位后编辑”的完整链条，应进入主相关工作而不只是附带提及。
4. **Persona Vectors（2025）**：sycophancy 等人格的 activation monitoring/control，是 BLADE 多行为设定的
   直接 activation-space 对照。
5. **Sycophancy 的组件机制**：Pinpoint Tuning（ICML 2024）；另有 2025 预印本
   *Sycophancy Is Not One Thing*，提醒不能把所有迎合合并为单轴。
6. **Deception 的更强行为层级**：Sleeper Agents（2024）与 In-context Scheming（2024）。它们不做同一
   weight localization，却是限制 BLADE “deception” claim 的关键外部参照。
7. **Situational awareness**：SAD（NeurIPS 2024 D&B）与 AwareBench（2024），用于明确 MWE basic
   self-awareness 的边界。
8. **LLM judge 的 self-bias**：Panickssery et al.（2024）、Wataoka et al.（2024）和 CoBBLEr。
   它们与 self-rate-highly 相邻但不等价；正因为不等价，论文应补一段构念辨析。
9. **多行为 activation steering**：van der Weij et al.（2024），其 wealth-seeking 实验是本项目这一行为
   最直接的已有干预工作。

投稿前建议再做一次定向检索，关键词应组合行为名与 `mechanistic interpretability`、`causal tracing`、
`activation steering`、`model editing`、`parameter pruning`，特别检查 2025 下半年正式接收版本是否替换了
预印本标题或作者顺序。

## 5. 引用风险清单

### 5.1 已纠正、可直接使用的编号/元数据

- Perez et al., *Discovering Language Model Behaviors with Model-Written Evaluations*：
  arXiv:2212.09251（已核实）。
- Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition*：
  arXiv:2312.06681；ACL 2024（已核实）。作者应写 Nina Rimsky et al.，不应写成 Rimsky/Panickssery et al.
- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*：
  arXiv:2406.11717；NeurIPS 2024（已核实）。
- Wei et al., *Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications*：
  arXiv:2402.05162；ICML 2024（已核实）。
- Sharma et al., *Towards Understanding Sycophancy in Language Models*：arXiv:2310.13548（已核实）。
- Wei et al., *Simple synthetic data reduces sycophancy in large language models*：arXiv:2308.03958（已核实）。
- SAD：arXiv:2407.04694；NeurIPS 2024 Datasets and Benchmarks（已核实）。
- Sparse Feature Circuits：arXiv:2403.19647；ICLR 2025 Oral（已核实）。
- Persona Vectors：arXiv:2507.21509（已核实为预印本；未确认同行评审 venue）。

### 5.2 仍有风险，提交前需逐条处理

1. **Soares et al., *Corrigibility***：AAAI Workshop on AI and Ethics 2015 的作者与 workshop 记录已核实；
   本次未发现可可靠归属的 arXiv 编号，故应引用 workshop/MIRI 版本。**[arXiv ID 未核实]**
2. **Turner et al., *Optimal Policies Tend To Seek Power***：NeurIPS 2021 标题、作者、venue 已由官方页面核实；
   原文档所填 arXiv 编号本次没有用 arXiv 官方页独立确认，正文故只链接 NeurIPS，且此处不转录该编号。
   **[arXiv ID 未核实]**
3. **Activation Addition / Activation Engineering**：arXiv:2308.10248 已核实，但版本间题名从
   *Activation Addition: Steering Language Models Without Optimization* 演变为
   *Steering Language Models With Activation Engineering*。定稿应按所引用版本统一题名、年份与作者顺序。
4. **Safety Neurons**：预印本题名为 *Finding Safety Neurons in Large Language Models*（arXiv:2406.14144），
   正式 NeurIPS 2025 题名为 *Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety
   Neurons*。不是两篇不同工作；参考文献应以正式版为主。
5. **Persona Vectors**：作者与 arXiv:2507.21509 已核实，但本次未确认正式会议发表；应标为 2025 preprint，
   不要自行补 venue。
6. **Extending Activation Steering to Broad Skills and Multiple Behaviours**：Teun van der Weij、Massimo Poesio、
   Nandi Schoots 与 arXiv:2403.05767 已核实；正式 venue 未核实，应按 preprint 引用。
7. **Sycophancy Is Not One Thing**：arXiv:2509.21305 已核实；本次未确认同行评审 venue 和最终书目信息，
   应保留为 2025 preprint。
8. **Localizing Lying in Llama**：James Campbell、Richard Ren、Phillip Guo 与 arXiv:2311.15131 已核实；
   可写 NeurIPS 2023 SoLaR workshop paper，不应误写为 NeurIPS main conference。

### 5.3 原文档中应删除或降级的表述

- 删除“CAA 已在 power-seeking 等本文全部/大部分精确标签上验证”的暗示；正式论文七任务中，与本文八类
  精确重合的是 refusal、corrigibility、sycophancy。
- 将 `self-awareness-general-ai` 从“情境意识/意识”降级为 basic AI self-knowledge。
- 将 self-rate-highly 与 self-preference、overconfidence 分开。
- 将 MWE 数据称为 model-written behavioral/persona evals，而不是无条件称为 canonical gold benchmark。
- 将 AdvBench refusal 与完整 safety 分开，并显式评估 harmful compliance 和 over-refusal。

## 6. 一段可直接改写进论文的 related-work 主线

Behavioral directions in residual activations can support monitoring and inference-time control, as shown by RepE,
CAA, refusal-direction work, and Persona Vectors. Mechanistic studies further localize refusal, lying, and
sycophancy to directions, neurons, heads, or small modules, while safety-pruning work shows that safety alignment can
be brittle to sparse or low-rank parameter changes. These lines stop at either activation/component-level control or
generic/unsigned parameter importance. BLADE instead factorizes a contrastive behavioral readout through the input
coordinates and individual scalar weights of residual writers, retaining the sign of each edge's direct
contribution and requiring only forward activations. Its empirical scope—eight heterogeneous behaviors across four
approximately 4B instruction-tuned model families—tests whether this scalar sparsity is specific to refusal or is a
broader property of learned behavioral tendencies.

最后一句的“broader property”只有在跨模型结果稳定时才保留；否则改成“tests whether”。
