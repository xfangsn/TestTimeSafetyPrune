1. **【Critical】E3 能识别 conditional causal effect，但不能识别 write-path 的历史起源。**  
   Revert \(S\) 后行为下降，说明这些坐标的 IT values 相对于 PT values，在其余参数保持 IT 状态时对行为有因果贡献。它不能排除 **co-adaptation、gain compensation、gating 或跨模块失配**：例如，其他模块改变了 activation scale，writer update 只是补偿这一变化；单独撤销补偿也会破坏行为。测量发生在 instructed model 中，解决了 baseline floor，却没有解决这些替代解释。  
   **具体修正：**将 E3 的 estimand 明确定义为“在 IT background 下撤销 \(\Delta W_S\) 的行为效应”。只有行为接近消失且能力 controls 通过时，才能进一步称为该背景下的 **joint necessity**；不能推出每个 selected coordinate 都必要，也不能推出它们此前没有行为功能。

2. **【Critical】\(c^{pt}\) 实际测量的是“PT weights 在 IT representation 下的投影”，不是 pretrained model 的实际 routing。**  
   固定 \(r^{it},\Delta\mu^{it}\) 的 decomposition 在代数上成立，但它回答的是一个 hybrid counterfactual。它没有测量 pretrained model 实际遇到的输入、使用的方向或 downstream readout。另外，多层同时 revert 后，后续层的 \(\Delta\mu\) 会变化，冻结 representation 的加和不再等于实际干预中的 write change，更不等于 behavior change。  
   **具体修正：**标记为 \(c^{pt\mid it}\)，并把它称为 **fixed-representation projected write**。在相同 prompt cohort、明确 token positions、统一方向符号和 normalization 下，补充实际 forward pass 的 writer output、projection 和 behavioral/logit effect；分层干预，再与联合干预比较。Intervention 的 metric 和操作方式会影响解释结果，已有系统研究支持这一方法学警惕。[Zhang & Nanda, 2024](https://arxiv.org/abs/2309.16042)

3. **【Critical】\(\phi_S\) 不是可解释为“instruction-created fraction”的 attribution ratio。**  
   即使 selected IT scores 全为正，它仍会被 PT 或 delta 项之间的 cancellation 欺骗。例如：
   \[
   c^{it}=(1,1),\quad c^{pt}=(100,-100),\quad
   c^\Delta=(-99,101).
   \]
   此时 \(\phi_S=1\)，但 PT 的 positive projected mass 是 \(100\)，并非接近零。反过来，\(c^{pt}=(2,0),c^\Delta=(-1,1)\) 给出 \(\phi_S=0\)，却包含明显的坐标间功能重分配。Ratio 也可小于 0 或大于 1。  
   **具体修正：**删除“\(\phi\to1\) 表示 created、\(\phi\to0\) 表示 merely exposed”的解释。需要精确区分：如果 \(S\) 只包含当前 representation 下严格正的 \(c^{it}\)，分母本身没有正负抵消；但 layer-wide sums、其他 behavior directions、held-out estimates 和 null controls 不保证这一点。分母接近零时应报告 undefined/unstable，不能用任意 \(\epsilon\) 掩盖。

4. **【Major】替代 estimators 应保留符号信息，并与 BLADE 的 rectification 对齐。**  
   \[
   [c^{pt}+c^\Delta]_+\ne[c^{pt}]_++[c^\Delta]_+,
   \]
   因而线性 decomposition 不是 rectified BLADE score 的 additive decomposition。简单把 \(\phi\) 的分子改成 \(\sum[c^\Delta]_+\) 也不能得到 attribution fraction。  
   **具体修正：**至少报告
   \[
   P_X=\sum_S[c^X]_+,\qquad N_X=\sum_S[-c^X]_+,
   \quad X\in\{pt,it,\Delta\}.
   \]
   再报告固定 representation 下，revert 导致的 positive-score loss 和 gain：
   \[
   L_S=\frac{\sum_S\big[[c^{it}]_+-[c^{pt}]_+\big]_+}{P_{it}},
   \qquad
   G_S=\frac{\sum_S\big[[c^{pt}]_+-[c^{it}]_+\big]_+}{P_{it}}.
   \]
   当 \(P_{it}>0\) 时，\(L_S\in[0,1]\)，而 \(G_S\) 可以超过 1；同时报告二者可避免掩盖反向变化。这些是 **projected-score diagnostics**，仍不能命名为历史归因比例。主要 causal estimator 应是 held-out prompts 上的 paired behavior change，以及它与 matched controls 的效应差。

5. **【Major】C1 没有匹配 E3 真正施加的 perturbation dose。**  
   Revert 的改变量是 \(-\Delta W_S\)，不是 \(-W^{pt}_S\)。只匹配 \(|W^{pt}|\) deciles，无法排除 \(S\) 的 update 更大、落在更活跃的 inputs 上，或集中于少数高敏感度 neurons/heads。  
   **具体修正：**E3 的 controls 应保持 per-layer/per-matrix 数量，并匹配 \(|\Delta W|\)、总 perturbation norm、generic activation/importance，以及适当的 row/column/head concentration；报告 balance 和可匹配范围。可另设匹配 generic-data residual-output perturbation 的 controls。**E1 的 enrichment test 则不能把 \(\Delta W\) mass 本身匹配掉**，否则把待检验的 outcome 当成了匹配变量。两类问题需要不同 control families。

6. **【Major】C4 的“random direction 下 \(\phi\approx0\)”没有数学依据。**  
   固定 \(S\) 时，\(r\mapsto-r\) 会同时翻转分子和分母，\(\phi\) 完全不变。如果 \(S\) 上 \(\Delta W=\alpha W^{it}\)，则任何分母非零的 direction 都给出 \(\phi=\alpha\)。Random numerator 与 denominator 各自均值为零，也不意味着它们的 ratio 均值为零；ratio 甚至可能没有稳定均值。  
   **具体修正：**区分“固定 mask 的 direction sensitivity”和“selection procedure 的 null distribution”。后者使用合理的 label permutation，重新估计 \(r,\Delta\mu\)，重跑 layer/mask selection，再在独立真实任务上评估。比较明确的 score statistic 或 behavior effect 的 empirical distribution，不预设 ratio 应趋近零。

7. **【Major】E3 没有证明 pretrained endpoint 具有特殊意义，也没有证明 revert 比 zeroing 更优。**  
   如果 \(\|\Delta W_S\|\ll\|W^{it}_S\|\)，revert 比 zeroing 温和可能只是因为 edit 更小。真实 PT value 放进 IT network 后，也不保证处于兼容的功能状态。  
   **具体修正：**比较
   \[
   W_S(\alpha)=W^{it}_S-\alpha\Delta W_S
   \]
   的 dose–response curve，与 zeroing/shrinkage curve、同 support 且 norm-matched 的 randomized update controls。所有算子用同一 validation protocol 调参，比较 **相同 behavior reduction 下的 utility**，或相同 utility budget 下的 behavior reduction。单个 operating point 的较小 \(\Delta\)ppl 不足以支持“better removal operator”。

8. **【Major】“revert all writers 是 upper bound”和“negative E4 说明 not sufficient”都过强。**  
   Neural network 中不存在 edit support 越大、目标行为必然下降越多的 monotonicity；更广泛 revert 可能恢复原先失配的模块配合。E4 阴性也只说明 graft 在那个 PT background、prompt format 和 assay 中未产生可测行为。  
   **具体修正：**把 all-writer revert 称为 broad intervention comparator。将现有 PT、IT、E3、E4 明确组织成 \(S\) 与其余参数取 PT/IT values 的 \(2\times2\) design，分析 interaction：
   \[
   I=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
   \]
   这要求四个背景中的 behavioral metric 可比较。更有信息量的 sufficiency test 是在保持 instruction-following 的中间 checkpoint/background 中 add back \(S\)。把 E3 完全 undo 回 IT 只是 implementation sanity check。

9. **【Major】“在 IT 上测量，所以不 circular”忽略了 selection bias 和 evaluation leakage。**  
   \(r,\Delta\mu,L^\star,\rho,S\) 都可能已使用目标 behavior 的数据或既有 evaluation results。E2 在 discovery data 上重新确认高 projection，部分是 selection 的直接结果；E3 若继续使用被反复查看的 eval sets，也不足以验证 generalization。  
   **具体修正：**分开 direction/mask discovery、layer/operator/budget validation 和 untouched behavioral test。已有结果影响过的 eval sets 应视为 development data。C4 若检验整个方法，需要重跑 selection pipeline；若只检验固定 \(S\)，应明确其条件性。WikiText 一旦用于选择 operator 或阈值，就不再是最终 held-out utility test。

10. **【Major】统计方案不足以支持“removes like zeroing”“approximately zero”或 control superiority。**  
    当前没有 sample size、effect size、CI、random-mask variability 或 multiplicity 计划。数百万 scalar weights 不是数百万独立实验单位；一个 random mask 也不能代表 random-mask distribution。  
    **具体修正：**以 paired prompts 为主要实验单位，对 paraphrase/template families 做 cluster bootstrap；随机生成需要考虑 decoding variability。重复抽取 matched masks，并同时报告 prompt uncertainty 与 mask variability。固定 \(S\) 的 test-set CI 和重新估计 \(r,\Delta\mu,S\) 的 selection stability 应分开报告。预设 primary behavior/model/comparison，其余标记 exploratory 或校正 multiplicity。“接近零”“效果相当”需要 practical equivalence margin，而非仅凭不显著。

11. **【Major】WikiText perplexity 不能排除任务相关能力损坏。**  
    Refusal 降低可能对应空输出、乱码、答非所问或 instruction-following 丧失；uncertainty 降低可能只是表达更武断。普通文本 PPL 稳定也不能排除这些情况。  
    **具体修正：**同时测量 target behavior、任务完成质量、instruction-following、输出有效性，以及 behavior-specific outcomes：例如 refusal 与实际 harmful compliance 分开，uncertainty expression 与 accuracy/calibration 分开。记录长度和 EOS 行为，使用盲评并验证 evaluator。报告完整 cross-behavior intervention matrix；C3 必须考虑 mask overlap、层位置差异和 behavior 本身的相关性。Utility 最好同时给出可加性的 held-out mean NLL change 及 CI。

12. **【Major】E5 的高 AUROC 不能证明相关 feature 被 writer 使用，更不能证明“feature pretrained、write-path instruction-tuned”。**  
    Harmful-vs-harmless 可能由 lexical/topic cues 解码；它不等同于 refusal mechanism。Residual-stream probe 也未必测到 \(o_{\text{proj}}\) 或 \(down_{\text{proj}}\) 的实际输入特征。对于 uncertainty，同一问题对 PT 与 IT 的真实 epistemic difficulty 还可能不同。Probe performance 本身需要 capacity 和 control-task 检验。[Hewitt & Liang, 2019](https://arxiv.org/abs/1909.03368)  
    **具体修正：**使用 topic/template-disjoint splits、matched hard negatives、合理的 lexical/control baselines，并明确 uncertainty labels 的定义。补测 writer inputs，检验 probe feature 与 selected columns 的联系，再用受控 activation intervention 测其是否影响 projected write 和行为。AUROC 阳性最多直接支持“该标签信息可被线性解码”。

13. **【Critical，针对历史归因】Shape identity 不等于可验证的 instruction-tuning ancestry 或 functional alignment。**  
    文档承认这一限制，但它直接影响“撤销 instruction-tuning update”的定义。若 released IT 并非 released PT 的直接后代，\(\Delta W\) 只是 checkpoint difference；同 shape 也不保证 hidden units 功能对应。Permutation symmetry 已足以说明这种一般性问题。[Ainsworth et al., 2023](https://arxiv.org/abs/2209.04836)  
    **具体修正：**核验 checkpoint provenance、revision、tokenizer/config 与 parameter correspondence。无法验证时，把操作称为 **cross-checkpoint value replacement**。若要保留历史性的“created by instruction tuning”，至少增加一个已知 base 出发的 controlled post-training experiment，保存中间 checkpoints，并设置 comparable training 的 behavior-neutral control；多个 training seeds 才能支持超出单一路径的结论。

14. **【Moderate】E1 的 estimands 和 null baseline 需要更精确。**  
    \(N\) 定义为 \(L^\star\) 内 writer weights，但 \(\|\Delta W\|_F^2\) 的范围不明确；二者范围不同会使 enrichment baseline 错误。即使范围一致，\(E=1\) 也只是 uniform-coordinate sampling 的基准，并非 magnitude-matched null 的预期。Relative change 会被接近零的 \(W^{pt}\) 和任意 \(\epsilon\) 主导。  
    **具体修正：**明确共同 universe \(\mathcal U\)：
    \[
    E=
    \frac{\sum_S\Delta W^2/|S|}
         {\sum_{\mathcal U}\Delta W^2/|\mathcal U|}.
    \]
    报告 per-matrix enrichment、matched empirical null 和 uncertainty；overlap 也应对 matched null 标准化。Relative change 同时报告 absolute update、原权重分层结果及 \(\epsilon\) sensitivity。使用 FP32 做 subtraction/accumulation，并说明 released precision 对小 delta 可辨识性的限制。

15. **【Major】“base 在 chance”与“base 没有该行为”尚未被建立。**  
    若 balanced A/B assay 的 chance 是 \(0.5\)，\(0.37\) 不能未经检验就称为 chance；它可能反映反向偏好、选项位置偏差或 parsing failure。Observed refusal \(=0\) 也只是该 assay 下没有观察到 refusal，不能推出相应 computation 不存在。  
    **具体修正：**明确 metric、chance baseline、样本量和 CI；counterbalance answer positions，区分有效选择与格式失败。补充 PT 可用的 continuation/forced-choice likelihood assay，并比较 common format 与各自 native format。把 floor-limited base ablation 视为低信息量证据，而不是通过转向 E3 就认为“PT routing 很少或没有”已被证明。

16. **【Major】Falsification criteria 混淆了不同 hypotheses。**  
    \(E\approx1\) 不能 falsify behavioral creation：update 的 **方向** 可能关键，而 update mass 完全不 enriched。\(\phi\approx0\) 也可能来自抵消。相反，E3 阳性、E1 enrichment、E5 高 AUROC 同时成立，仍可兼容 co-adaptation 解释。  
    **具体修正：**分别预注册 update-mass enrichment、fixed-representation write change、conditional behavioral effect、behavior specificity 和 historical emergence 的检验及最小有意义效应。比如，“E3 对目标行为的效应不超过 dose-matched controls，且 CI 排除了预设优势”直接反驳 selective causal localization；它不自动否定所有 instruction-tuning mechanism。

17. **【Critical】最强诚实结论应限定为 checkpoint-relative、background-dependent causal contribution。**  
    **具体修正：**若预期结果通过独立测试、perturbation-matched controls 和任务能力检查，可以得出：**在所测试的 released pairs 和任务中，将 BLADE-selected writer coordinates 的 IT values 替换为对应 PT values，会选择性降低目标行为；这些坐标上的 checkpoint differences 在当前 IT network 中具有因果贡献。** 修正后的 E2 可以进一步说明这些 differences 如何改变指定 representation 下的 projected write。  
    这仍不足以证明 pretrained model 不使用这些坐标、instruction tuning 创造了新的行为功能、\(S\) 是唯一 write-path，或 \(S\) 本身足够。保留“created”需要 controlled training trajectory，并将实际 PT computation、机制出现时间和上述干预证据连接起来。
