1. **[Critical] 四轴评估是必要条件，但目前仍不足以排除“只改变措辞”。**  
   Keyword refusal 可以把“先道歉、后提供有害内容”判成成功；marker-based hedging 可以把“加一句不确定、然后继续编造”判成改善。Hedging 本身也不等于 abstention。**修复：**主指标应包括 harmful compliance / attack success、无依据断言率、正确回答率及适当拒答率；keywords/markers 仅作为辅助诊断。HarmBench 明确把“先拒绝、后执行有害行为”列为评估器必须识别的情况。[HarmBench](https://www.harmbench.org/HarmBench.pdf)

2. **[Major] “Answerable”“模型知道答案”“应当自信回答”被混为一谈。**  
   客观可回答的问题可能超出模型知识；正确回答中的适度 hedging 不一定是 harm。False-premise questions 则可能需要明确纠正前提，而非表达不确定。[FalseQA](https://github.com/thunlp/FalseQA)  
   **修复：**区分缺失信息、模型知识不足、错误前提和真正不可回答；以独立的 baseline 测量定义“known”子集，同时报告全体 answerable 集上的正确率。增加回答 coverage 与回答条件下错误率，防止用更多错误回答换取较低 over-abstention。

3. **[Critical] “Best single-behavior configuration”没有数学定义，success 与 failure 也不是互补事件。**  
   多个 benefit、harm 指标下通常不存在唯一 best configuration。Joint 可能扩大 Pareto frontier，却无法同时超过两个 single 家族各自的最高 benefit；也可能证据不足，既无法证明优越，也无法证明被匹配。**修复：**预注册究竟检验“严格双改善”还是“受约束 Pareto frontier 扩展”，并保留 **inconclusive** 结果；不要将“不满足 success”直接归为文中的 failure。

4. **[Critical] Matched-harm 需要明确的多维约束，而不是口头上的“equalized”。**  
   一个可执行的严格版本是：预先固定 harm 上限向量 \(h\)，将 capability 限制一并纳入，并定义完整 single 配置集合的可行子集
   \[
   \mathcal S(h)=\{s:H_k(s)\le h_k,\ \forall k\}.
   \]
   对预先选定的 joint 配置 \(j\)，要求其可行，且
   \[
   B_b(j)-\sup_{s\in\mathcal S(h)}B_b(s)>\epsilon_b,
   \qquad b\in\{\mathrm{ref},\mathrm{unc}\},
   \]
   其中 \(\epsilon_b\) 是预注册的最小实际改善。这是很强、但清楚的“双改善”定义。**修复：**明确 baseline 集合、各项预算、改善阈值和统计判定。“同一 harm 预算下更优”与“相对某个 comparator 每项 harm 均不更差”仍需区分；后者需要逐项 non-inferiority 检验。

5. **[Major] Frontier interpolation 不能创造不存在的配置。**  
   多个 harm 同时匹配时，通常不存在一条可供简单插值的曲线；一个插值系数一般也无法同时匹配所有 harms。**修复：**优先在 development set 上加密边界附近的真实 dose 并重新测量。若采用两个配置的随机 mixture，其同一个混合比例必须作用于所有 prompt 类型和指标，不能分别为 refusal、uncertainty 或不同评估集选比例。对平均 rates，这能实现期望值的凸组合；它代表随机 mixture，并不证明某个中间 \(\alpha\) 达到同一点。无共同支持区域时报告无法匹配，不做外推。

6. **[Critical] 当前 sign-agreement 诊断在数学上不成立。**  
   \(c_{ij}=[r_iW_{ij}\Delta\mu_j]_+\ge0\)。若共享 coordinate 在两种行为上均有 \(c>0\)，其 rectification 前的项也都为正，sign-agreement 必然为 100%。此外，两次 amplification 都满足 \(\Delta W=(\alpha-1)W\)，不存在对同一 scalar weight 的相反编辑方向。**修复：**保存未 rectified 的 \(z=r_iW_{ij}\Delta\mu_j\)，分别报告正、负、零质量，并仅将其作为局部 surrogate；通过实际 cross-effects 和联合干预检验 antagonism。删除由当前 sign-agreement 推导 `max`/disjoint 必胜或 joint 不可能的预言。

7. **[Major] Track B 与最终 Track A 的 mask 定义不一致。**  
   Amplification 的排名依赖 \(\alpha\)，P1 又重新选择 \(L^\star\)，因此现有 overlap 表未必描述最终实验，Qwen/Phi 的“重叠/不重叠”分组也可能改变。**修复：**明确区分 frozen-mask dose sweep 与每个 dose 重新 selection 的方法评估。前者可以解释固定 mask 的 dose response，但必须另与完整 P1 frontier 比较；后者必须逐 cell 报告实际 mask。P0 可以先做探索，最终机制解释必须基于实际使用的 masks。

8. **[Major] `disjoint-assign` 会破坏“两条轴恢复 single baseline”的承诺。**  
   若共享 coordinate 固定归 refusal，则在 \(\alpha_{\mathrm{ref}}=1,\alpha_{\mathrm{unc}}>1\) 时，它不会随 uncertainty 放大，因此该轴不是原始 uncertainty-only intervention。**修复：**明确报告 partition 后的 single baselines，并额外保留原始完整 masks 的 single baselines；或定义 inactive-mask 时如何恢复 ownership，并承认此时 assignment 随配置变化。`union` 和 joint re-selection 也应作为独立 intervention 家族，不能默认具有同样的轴恢复性质。

9. **[Major] Overlap 的 null 必须定义 eligibility universe；只报 enrichment 会误导。**  
   若按 layer/matrix 分层，固定两种 mask 在各层的数量，则随机 intersection 的期望为
   \[
   E[I]=\sum_g\frac{n_{\mathrm{ref},g}n_{\mathrm{unc},g}}{N_g}.
   \]
   Layer sets 完全不相交时，该条件 null 下 \(E[I]=0\)，enrichment 是未定义，而非“零 enrichment”。Jaccard 也会把“大 mask 完全包含小 mask”呈现为低 overlap。**修复：**同时报告原始数量、Jaccard、两个方向的 containment fraction、期望数量及随机化区间；对 Jaccard 直接构造随机分布，不用期望 intersection 代入比值。分别分析 layer selection overlap 和共享 layer 内的 coordinate overlap。

10. **[Major] Overlap-vs-\(\rho\) 的下降不自动意味着“共享核心被稀释”。**  
    Enrichment 的分母随 mask 大小增加，本身就能产生下降；若同时重新选择 layers 或改变 ranking，曲线还混合了多个机制。**修复：**主曲线固定 \(L^\star,\alpha,\lambda,r,\Delta\mu,Q\) 和 tie-breaking，只改变 top-\(\rho\) 截断；同时画实际 intersection、containment 和随机基线。所谓 core 必须跨 calibration resamples 稳定，并通过 core-only / core-removed 干预证明其功能作用。

11. **[Major] Whole-universe score correlation 易受共同结构和大量零值支配。**  
    两种 scores 共用 \(W\)，并具有由 \(r_i\Delta\mu_j\) 产生的行列依赖；rectification 会产生大量 ties。高相关可能来自权重大小、layer 尺度或共同零值，而不是行为共同定位。**修复：**分 layer/matrix 报告 signed \(z\)、rectified \(c\) 和实际 selection score \(c-\lambda Q\) 的相关性、零值比例及非零支持关系；增加匹配 \(|W|\)、\(Q\)、行列结构的 null。不要把数百万 coordinates 当作独立样本计算极小 p-value；对生成 scores 的 calibration prompts 重采样。

12. **[Major] Direction geometry 的 null 不合适，机制结论也过强。**  
    Difference-in-means directions 来自同一模型的非各向同性 activation distribution，不能默认服从独立、均匀随机单位向量的 null。近乎平行最多说明这些 contrasts 在该表征空间中对齐。**修复：**使用匹配 prompt/template 的 label permutations、无关但匹配的 contrasts，以及 prompt bootstrap；报告方向 norm 和估计稳定性。分别在对应 layer/module 内计算 cosine，不把近似平行表述为“同一个机制”。

13. **[Critical] Track B → Track A 的 conditioning 应是可检验假设，不能作为机制判定表。**  
    Disjoint weights 可能实现同一功能；共享 weights 可能参与不同计算；单一 dose 上对称的 cross-effects 也可能来自 saturation 或共同输出风格。**修复：**分别干预 \(S_{\mathrm{ref}}\cap S_{\mathrm{unc}}\)、两个 exclusive 部分以及移除 intersection 后的 masks；测量多个 doses 下的 cross-effect curves。只有这些结果支持时，才能说存在共享的功能贡献；当前统计最多支持“所选 masks/方向的重叠或关联”。

14. **[Major] Qwen 对比 Phi 无法识别 overlap 的因果作用。**  
    Architecture、training、baseline headroom、层数、mask 大小和 dose sensitivity 都同时变化。“Phi 成功、Qwen 失败”不能推出“只有分别定位时才可组合”。**修复：**在同一模型内构造 overlap 不同、但 layer budget、score quality、编辑规模和 single 性能尽量匹配的 mask 对，并重复 selection。跨模型对比保留为 replication 或 hypothesis generation。

15. **[Major] Operator 比较仍混杂了编辑规模和 dose distribution。**  
    Joint 通常编辑更多独特 weights；sequential 还在 intersection 引入乘积项。只报 coordinate 数量和 \(\alpha\) 无法解释性能差异。**修复：**增加 additive-deviation operator：
    \[
    \alpha_{\mathrm{shared}}=1+(\alpha_{\mathrm{ref}}-1)+(\alpha_{\mathrm{unc}}-1),
    \]
    并报告每层 \(\|\Delta W\|\)、共享部分的编辑强度，以及 calibration activations 上的输出扰动。加入相同总编辑预算的 enlarged single masks。若两个 masks 完全相同，sequential/max 分别退化为单一 product/max dose，single sweep 必须覆盖这些有效 doses。

16. **[Major] Joint re-selection 和 score-based assignment 缺少尺度定义。**  
    两个行为的 raw scores 可能因 contrast 强度、direction normalization 和数据分布而处于不同尺度；直接相加会隐式偏向其中之一。“Principled”因此尚无依据。**修复：**预定义各行为 scores 的归一化、组合系数和单次 generic penalty，保持总 sparsity/编辑预算一致，并在 development set 上选择权衡。把 joint re-selection 作为重新学习 mask 的方法单独评估，而非仅视为 composition operator。

17. **[Major] C1 的 random mask 不是充分的 generic-abstention baseline。**  
    Random masks 失败，只能说明随机扰动不够好；不能证明行为特异性。仅匹配 XSTest over-refusal，也没有匹配其他 harms。**修复：**加入明确训练的 generic decline/uncertainty direction、union mask 的单一 dose，以及简单的 cautious-prompt baseline；统一使用多维 harm 预算。Random controls 应有多个 seeds，并匹配 layer allocation、权重/importance 分布和扰动规模。无法在 capability 预算内达到目标 harm 的 control，应报告不可达。

18. **[Critical] “重新分割已看过的数据”不能恢复 untouched test。**  
    如果 XSTest/OOD prompts 或其结果已经影响方法、mask、judge、margin 或超参数，重新 partition 仍有研究者层面的泄漏。反复按 WikiText 结果筛配置，也会使其成为 development data。**修复：**分离 direction/mask estimation、ELS 与超参数选择、最终 confirmation 三个阶段；最终集采用未参与决策的 prompt families。所有已经用于筛选的评估集，包括 capability 集，都需要新的 confirmation 数据。

19. **[Critical] “获胜 cell 的 bootstrap CI 排除零”没有处理整个搜索过程。**  
    P1、operator selection、grid refinement 和 frontier selection 都会产生 winner’s curse；只 bootstrap 最后赢家会低估不确定性。**修复：**在 development set 上锁定 primary joint 配置、comparators、mixture 权重及 budgets，再在 untouched test 上做 paired cluster inference。若仍在 test 上搜索，需要覆盖 selection/frontier estimation 的推断。固定候选必须同时满足所有必要条件时可采用 intersection–union testing；搜索“存在一个赢家”则是额外的选择问题。

20. **[Major] 非劣效、稀有 harms 和 mask 稳定性需要专门的统计设计。**  
    “Harm 未显著上升”不等于 harm 被控制；观测到 \(0.00\) 也不意味着真实概率为零，普通 bootstrap 在全零样本上甚至会给出虚假的零宽区间。**修复：**为每项 harm 预注册 non-inferiority margin，要求差值的单侧上界不超过 margin，并据有效 cluster 数做 power analysis。所有配置使用配对 prompts；按 behavior/template family 处理依赖。另用多个 calibration resamples 重做 masks/ELS，区分固定 intervention 的评估不确定性与 selection 稳定性。

21. **[Major] 缺少区分语义选择性与 dataset cues 的关键实验。**  
    Harmful-vs-harmless 和 unanswerable-vs-answerable contrasts 可能同时编码题材、长度、模板或熟悉程度，last-prompt-token directions 尤其可能捕获这些差异。**修复：**增加 safety × answerability 的交叉测试，以及同主题、最小改动的 counterfactual pairs；在每类 prompt 上同时判断 refusal、uncertainty、correctness 和 helpfulness。OOD refusal 还需保留完整测试集；若报告 baseline-failure 子集，应独立确定该子集。固定攻击测试之后，增加等预算、针对最终模型的 adaptive attacks，或明确仅声称对固定攻击集有效。

22. **[Major] PPL 与 lexical degeneration 不能充分证明能力保持。**  
    模型可以输出流畅、低困惑度的拒答，同时损失 instruction following、事实回答或推理能力。XSTest 是特定 over-refusal 诊断集，也不能代表一般 benign utility。[XSTest](https://aclanthology.org/2024.naacl-long.301/)  
    **修复：**增加小型、独立的 benign utility suite，覆盖正确性、任务完成度和 instruction following。预注册空输出、截断、跑题及 degeneration 的评分规则，不从 denominator 删除这些输出，也不让其自动获得安全改善的解释。

23. **[Major] 有限 grid 和 P1-selected masks 不足以支撑普遍失败结论；joint gain 也不自动等于 synergy。**  
    单任务最优 masks 未必是最适合联合使用的 masks；\(3\times3\) grid 可能错过狭窄可行区。另一方面，joint 超过 singles 可以只是两个独立效果相加。**修复：**保留多个 P1 Pareto masks，在 development set 上预定义边界加密规则和搜索预算；负结果限定在实际搜索的家族和范围内。若要声称 synergy，另检验预定义尺度上的 interaction，例如
    \[
    I_b=B_b(a,u)-B_b(a,1)-B_b(1,u)+B_b(1,1),
    \]
    并说明二元 rates 的 ceiling 和非线性 link 会影响解释。Interaction 不是证明实用 composition gain 的必要条件。

24. **[Major] 各种结果所能支持的最强结论需要降到证据对应的层级。**  
    **修复：**严格通过 matched-budget confirmation 时，可声称“在这些模型、任务、预算和 baseline 搜索范围内，joint editing 改善了受约束性能”，不能据此证明两个独立机制。若 generic control 同样有效，说明收益未体现 BLADE 特异性。若所有被测 joint 点都被可靠匹配，只能说该搜索范围内未发现 frontier 扩展；若 CI 太宽，则是 inconclusive。高 overlap 加对称 cross-effects 支持共享功能贡献的假设，不能证明单一 abstention mechanism。仅 Phi 成功不能证明 disjointness 是必要条件；仅 sequential degeneration 也只能证明该 operator/剂量失败，除非替代 operator 在可比条件下确实解决问题。
