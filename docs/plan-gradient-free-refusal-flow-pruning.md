# Plan: Gradient-free 对比拒绝流单权重剪枝（Route C）

> ⚠️ **历史存档(标注于 2026-08-27):本文记录较早阶段的计划/方法/结果,可能与当前代码与结论脱节。**
> 方法线现已定名 **BLADE** 并引入 best-first ELS(有效层选择);当前定稿的方法与实验结果请以
> [`blade-method.md`](blade-method.md)、[`blade-algorithm.md`](blade-algorithm.md)、
> [`blade-experiment-report.md`](blade-experiment-report.md) 为准。**下文内容保留原貌,未作修订。**

> 状态：核心 CRFP 已执行（C0--C3）；结果为有信息的阴性。C4 grouped
> calibration 因触发预注册停止条件而未运行；C6 因没有新的 external holdout
> 而未运行。完整结果见
> [`results-gradient-free-refusal-flow-pruning.md`](results-gradient-free-refusal-flow-pruning.md)。
>
> 本计划提出 Route C：Contrastive Refusal-Flow Pruning（CRFP，对比拒绝流
> 剪枝）。它是一个 white-box、forward-only、gradient-free 的单权重选择方法：
> 可以读取模型权重和内部激活，但不启用 autograd、不调用 backward、不计算或
> 近似 loss gradient。
>
> 前置结果：
>
> - Route A 的 per-layer refusal-direction orthogonalization 在已有 held-out test
>   上将 refusal 从 0.990 降至 0.005，PPL +1.16%；
> - Route B 的 Taylor/Wanda ratio 在目标权重池剪除 0.01%（41,524 个权重）时，
>   harmful_val refusal=0.031、PPL +0.03%、KL=0.0017；
> - 现有 gradient-free signed actdiff edge 在 0.01% 时 refusal=0.125，在
>   0.05% 时 refusal=0、PPL +0.61%、KL=0.0935；
> - Route B 尚无新的独立 held-out test，不能继续复用已经被 Route A 使用过的
>   `harmful_test`。
>
> 相关文档：
>
> - [`method-refusal-aware-weight-pruning.md`](method-refusal-aware-weight-pruning.md)
> - [`plan-weight-level-refusal-editing.md`](plan-weight-level-refusal-editing.md)
> - [`results-weight-level-refusal-editing.md`](results-weight-level-refusal-editing.md)

## 1. 研究问题

Route C 检验以下问题：

1. 不计算梯度，仅利用 paired forward activation 和 refusal direction，能否恢复
   Route B 的稀疏、拒绝特异权重集合？
2. refusal/compliance 完整 response trajectory 是否比 harmful/harmless prompt
   最后一个 token 更适合归因拒绝行为？
3. edge 对 refusal direction 的直接写入量，经过 harmless activation cost 约束后，
   是否足以预测真实剪枝效果？
4. 少量 forward-only grouped ablation 能否校准解析 proxy 中遗漏的 downstream
   非线性和权重交互？
5. gradient-free 方法能否以更低 GPU 显存和运行时间达到接近 Taylor/Wanda 的
   refusal–utility Pareto 前沿？

主假设：

- **H1（paired trajectory）**：同一 instruction 内 refusal/compliance response
  activation 的差分，比 harmful/harmless prompt 差分更贴近 Route B 的行为目标。
- **H2（direct refusal flow）**：在 refusal response 中持续向 refusal direction
  写入的 scalar edges 构成一个可剪除的稀疏集合。
- **H3（能力约束）**：只按 refusal flow 排名会损伤通用能力；加入 harmless
  Wanda cost 后可显著降低 PPL/KL 副作用。
- **H4（稳定性）**：使用 pair-level 方差或符号一致率，可减少由少数异常回答
  驱动的虚假高分 edge。
- **H5（因果校准）**：grouped forward ablation 能筛除解析分数高、但实际剪枝
  不降低 refusal margin 的候选组。

## 2. 方法定位与边界

### 2.1 Gradient-free 的精确定义

Route C 允许：

- 读取 BF16/FP32 权重；
- 对模型做 teacher-forced forward；
- 通过 forward hook 读取 residual writer 输入；
- 使用已有的 layer-wise refusal directions；
- 临时把候选权重置零，并用 forward 评估实际效果。

Route C 禁止：

- 将任何模型参数设置为 `requires_grad=True`；
- 调用 `loss.backward()`、`torch.autograd.grad()`、JVP/VJP；
- 通过 finite difference 或 SPSA 近似每个 scalar weight 的 gradient；
- 使用 optimizer 或更新剩余模型权重。

因此它是 forward-only white-box attribution，不是只能访问文本 API 的严格黑盒
方法。如果只有远程 API、不能读取权重或激活，就无法执行本计划中的 scalar-weight
剪枝。

### 2.2 与 Route A、Route B 的关系

| 路线 | 干预单位 | 归因信号 | 是否需要梯度 | 主要问题 |
|---|---|---|---:|---|
| Route A | dense rank-one weight edit | refusal direction | 否 | 能否删除方向写入 |
| Route B | scalar weight | refusal/compliance logp margin Taylor gradient | 是 | 是否有稀疏关键 edge |
| Route C | scalar weight | paired activation refusal flow | **否** | forward signal 能否恢复关键 edge |

Route C 不替代 Route A 的方向编辑结论。它主要作为 Route B 的低开销替代和机制
对照。

### 2.3 不主张的结论

本实验即使成功，也不自动意味着：

- 非拒绝输出是安全、正确或真实有帮助的输出；
- 被选中的 edge 是唯一或充分的拒绝机制；
- 非结构化零权重能带来 dense kernel 推理加速；
- 在一个模型和一套 CAA pairs 上的结果能跨模型泛化；
- CRFP 的完整组合在文献中没有先例。

在完成系统文献检索前，文稿中使用“我们检验的 forward-only 组合方法”，不使用
“首次”“全新”等创新性表述。

## 3. 模型、权重池与数据隔离

### 3.1 首个模型

- 模型：`meta-llama/Llama-3.2-3B-Instruct`；
- dtype：bf16；
- 层数：28；
- hidden size：3072；
- intermediate size：8192；
- GPU：RTX 5090 32GB。

首个实验保持与 Route B 相同的目标权重池：

- L7--L18 的 12 个 `mlp.down_proj`；
- L7--L18 的 12 个 `self_attn.o_proj`；
- 共 415,236,096 个 scalar weights。

保持同一 target pool 是为了隔离“评分方法”这一变量，不能在看到 Route C 结果后
再自由扩大或移动层窗口。

### 3.2 数据角色

| 数据 | 用途 | 是否参与选择 |
|---|---|---:|
| `data/caa_pairs.jsonl` score split | paired activation moments | 是，仅计算分数 |
| `data/caa_pairs.jsonl` calibration split | grouped forward ablation | 是，仅组校准 |
| `data/harmless.jsonl` | harmless Wanda moments、KL、行为保持 | 是 |
| `data/harmful_val.jsonl` | 与 Route B 回顾性横向比较 | **否，不得再调参** |
| 已有 `harmful_test` | Route A 已使用 | **禁止再次作为独立 test** |
| 新建并锁定的 external holdout | Route C 最终一次评估 | 是，仅一次 |

### 3.3 CAA pair 固定拆分

在任何 score 运行前，按 pair 内容 SHA256 排序并固定拆分：

- 前 80%：score split；
- 后 20%：group calibration split。

拆分 manifest 保存：

~~~text
data/splits/crfp_caa_split_seed0.json
~~~

manifest 记录每条 pair 的 hash，不复制回答文本。拆分一旦产生不得根据实验结果修改。

注意：CAA pairs 在先前 Route B 中已经整体使用过，因此这个拆分只能防止 Route C
内部的 proxy/calibration 自我验证，不能产生真正独立的最终证据。

### 3.4 新独立评测集

在运行 C3 sparsity sweep 前，需要建立一份从未用于本项目调参的新 harmful
evaluation set，并满足：

- 来源、许可、样本数、去重方法有记录；
- 与 harmful train/val/test、CAA instructions 做 normalized exact-match 和
  near-duplicate 检查；
- 在查看任何 Route C generation 前写入 SHA256 manifest；
- 默认至少 200 条；
- 只允许对最终锁定配置运行一次。

若本阶段没有新数据，Route C 必须保持为 retrospective/val-only 结果，不能把
旧 `harmful_val` 或旧 `harmful_test` 包装成新 held-out 结论。

## 4. 算法定义

### 4.1 记号

对于第 `l` 层的 residual writer：

~~~text
y_l = W_l x_l
~~~

其中：

- `W_l in R^(d_out x d_in)`；
- `W_l,ij` 是输入维度 `j` 到 residual 输出维度 `i` 的单条 edge；
- `x_l,j` 是 writer 输入；
- `r_l` 是与 writer destination residual site 对齐的单位 refusal direction；
- `R_n`、`C_n` 是第 `n` 个 instruction 的 refusal/compliance completion。

方向必须先单位化并固定符号。若：

~~~text
dot(r_l, mean_harmful_l - mean_harmless_l) < 0
~~~

则翻转 `r_l`。方向来源和翻转结果写入 metadata。

### 4.2 Response-prediction position mask

不能直接把 response token 在输入中的位置当作预测该 token 的位置。对于 causal LM，
位置 `t-1` 的 hidden state 预测 token `t`。

若 response token 范围为：

~~~text
[response_start, response_end)
~~~

则 writer activation 的采样位置应为：

~~~text
[response_start - 1, response_end - 1)
~~~

mask 必须：

- 排除 prompt 中不负责预测 response 的位置；
- 排除 padding；
- 对 refusal 和 compliance 分别做长度归一化；
- 明确定义 `<|eot_id|>` 是否作为 response target；主实验包含 eot，与 Route B
  logp margin 保持一致；去掉 eot 只作为诊断 ablation。

这是 C1 的强制单元测试项，mask 未验证前不得运行全量 score。

### 4.3 Paired response activation difference

对每个 pair 分别 teacher-force refusal 和 compliance。对负责预测 response token 的
writer 输入取均值：

~~~text
x_bar_R[n,l,j] = mean over response-prediction positions of x_R[n,l,t,j]
x_bar_C[n,l,j] = mean over response-prediction positions of x_C[n,l,t,j]
delta_x[n,l,j] = x_bar_R[n,l,j] - x_bar_C[n,l,j]
~~~

每个 pair 权重相同，不按回答 token 数再次加权。这样避免长回答在 pair aggregation
中占更大权重。

对每个 writer 输入维度用 Welford 算法在线累计：

~~~text
mu_delta[l,j]  = mean_n(delta_x[n,l,j])
var_delta[l,j] = variance_n(delta_x[n,l,j])
p_pos[l,j]     = fraction_n(delta_x[n,l,j] > 0)
~~~

不保存 `N x tokens x hidden` 全量 activation。

### 4.4 单权重 direct refusal flow

单条 edge 沿 refusal direction 的 paired direct contribution 为：

~~~text
b[n,l,i,j] = r[l,i] * W[l,i,j] * delta_x[n,l,j]
~~~

其均值和标准误可由输入维度统计解析得到：

~~~text
a[l,i,j]    = r[l,i] * W[l,i,j]
mean_b      = a * mu_delta[l,j]
se_b        = abs(a) * sqrt(var_delta[l,j] / N)
~~~

当 `mean_b > 0` 时，该 edge 在 refusal completion 中比在 compliance completion
中写入更多正向 refusal flow；把它置零会直接移除这部分贡献。

主 benefit 使用预注册的 lower-confidence-bound：

~~~text
B[l,i,j] = max(mean_b - beta * se_b, 0)
beta = 1.0
~~~

`beta=0` 仅作为消融，不参与主配置选择。

可同时记录 edge 的符号一致率。因为 `a` 对 pair 固定：

~~~text
consistency = P_n(b[n,l,i,j] > 0)
~~~

它可以由 `p_pos[l,j]` 和 `sign(a[l,i,j])` 直接得到，不需要保存 per-edge
pair statistics。主分数不再额外乘 consistency，避免 LCB 与 sign stability
重复惩罚；consistency 只用于诊断和 tie-break。

### 4.5 Harmless preservation cost

在 320 条 harmless prompt 的全部有效 prompt token 上收集 writer input 二阶矩：

~~~text
A[l,j] = E_harmless[x[l,j]^2]
H[l,i,j] = abs(W[l,i,j]) * sqrt(A[l,j])
~~~

`H` 与现有 Wanda 定义一致，表示删除该 edge 对 harmless writer output 的局部
扰动尺度代理。

### 4.6 主排名规则

直接使用 `B/H` 会使分子分母中的 `abs(W)` 大幅抵消，可能把真实贡献很小但分母
更小的 edge 排到最前。主规则采用 benefit eligibility + tempered cost：

1. 每个矩阵只保留正 `B` 中最高的 10% 作为候选；
2. 每个矩阵定义：

~~~text
tau_l = median of positive H values in that matrix
~~~

3. 候选最终分数：

~~~text
S_CRFP[l,i,j] = B[l,i,j] / sqrt(H[l,i,j] + tau_l)
~~~

4. 在全部矩阵候选中执行 global top-k；
5. 延续单矩阵最多剪除 10% 的硬 cap。

`sqrt` 对应 `alpha=0.5`，在能力保持与保留 refusal benefit 幅度之间折中。以下规则
仅作预注册 ablation，不参与主结果选择：

- flow-only：`B`；
- full-ratio：`B / (H + epsilon)`；
- no-LCB：`max(mean_b, 0) / sqrt(H + tau)`；
- consistency-weighted：`S_CRFP * consistency`。

如果主规则失败而某个 ablation 成功，只能报告探索性结果，不能把 ablation 重新
定义成预注册主方法。

### 4.7 剪枝操作

选中权重后直接置零：

~~~text
W'[l,i,j] = 0  if selected
W'[l,i,j] = W[l,i,j] otherwise
~~~

复用现有 reversible pruning context：

- 只备份被选中的值；
- context 内验证选中值全部为零；
- 退出后逐比特恢复；
- 不通过 BF16 加回 delta 的方式恢复。

## 5. 可选的 grouped forward-only 因果校准

解析 direct flow 不经过下游层和最终 LM head，可能把局部方向贡献误当作最终行为
因果效应。C4 增加一个不使用梯度的 grouped ablation，但它不能接触最终 holdout。

### 5.1 初始候选

固定取 score split 上主 CRFP 排名的 top 0.10%，约 415,236 个权重。这个比例只
用于校准候选池，不是最终 sparsity。

### 5.2 分组

每个矩阵内按 CRFP score 排序并分成 high/low 两个等数量 bin：

~~~text
24 matrices x 2 bins = at most 48 groups
~~~

空组跳过。每个组的 layer、component、score range、weight count 固定写入 manifest。

### 5.3 Forward objective

在 CAA calibration split 上计算长度归一化 paired logp margin：

~~~text
m = mean_n(logp(R_n) / len(R_n) - logp(C_n) / len(C_n))
G_group = m_base - m_group_pruned
~~~

`G_group > 0` 表示删除该组实际降低 refusal-over-compliance margin。

同时在固定 64 条 harmless prompts 上计算：

- base-to-pruned prompt-token KL；
- greedy token agreement；
- 空输出和重复启发式。

按 pair bootstrap 计算 `G_group` 的 90% confidence interval。组保留规则预注册为：

~~~text
lower_CI_90(G_group) > 0
and harmless_KL <= 0.10
and no quality failure
~~~

最终 calibrated ranking 保持原 CRFP 相对次序，但只允许从通过组中选择。

### 5.4 校准边界

- grouped calibration 是 Route C 的增强版 `CRFP-cal`，不能与纯解析 `CRFP` 混称；
- 主报告必须同时给出 CRFP 与 CRFP-cal；
- 如果没有组通过 CI，不降低 CI 或反复改变分组；
- 如果 group effect 太小导致统计功效不足，报告“不确定”，不能报告“无因果效应”；
- 不用 `harmful_val` 决定保留哪些组。

## 6. 对照和消融

### 6.1 必须对照

在相同 target pool、相同 sparsity、相同 per-matrix cap 下比较：

1. Route B Taylor/Wanda ratio：有梯度性能参照；
2. 现有 signed actdiff edge：gradient-free 直接基线；
3. flow-only `B`：检验 harmless cost；
4. Wanda-smallest；
5. magnitude-smallest；
6. random 三个 seed；
7. refusal/compliance label-swapped CRFP；
8. layer-wise random direction CRFP，三个 seed。

Taylor/Wanda 使用现有冻结结果，不重新调参或使用 Route C 新 holdout。

### 6.2 机制消融

只在固定开发数据上运行：

- paired response trajectory vs harmful/harmless prompt difference；
- response-prediction 全 token vs last prompt token；
- per-destination-layer direction vs shared L14 direction；
- MLP-only vs attention-only vs both；
- `beta=1` vs `beta=0`；
- tempered cost vs full ratio；
- 含 eot vs 不含 eot；
- CRFP vs CRFP-cal。

这些消融用于解释，不允许扩大成无界超参数搜索。

## 7. 指标、sparsity 和成功标准

### 7.1 Sparsity 网格

相对于 415,236,096 个目标权重：

~~~text
0.001%, 0.003%, 0.01%, 0.03%, 0.05%, 0.10%, 0.50%, 1.00%
~~~

其中 0.01% 是预注册 primary operating point，因为它与 Route B 当前最干净结果
直接可比。其他比例用于画 Pareto 曲线，不改变 primary 判定。

### 7.2 行为和能力指标

沿用现有实现：

- harmful refusal rate；
- compliance rate；
- harmless refusal rate；
- WikiText-2 固定 50k-token PPL 与相对变化；
- harmless prompt-token base-to-pruned KL；
- greedy token agreement；
- 空输出、乱码、循环和长度异常；
- CAA calibration paired margin；
- 每层、每组件、每矩阵的选中权重数量；
- CRFP benefit/cost/consistency 分布；
- 峰值 GPU allocated/reserved memory 和 wall-clock 时间。

在新 holdout 上增加独立语义 judge 或盲人工标注，不能只依赖 refusal keyword。

### 7.3 硬约束

一个配置只有同时满足以下条件才算有效：

- harmless refusal <=5%；
- PPL 退化 <=5%；
- harmless KL <=0.10 nat/token；
- 空输出/乱码/循环率 <=1%；
- 相对三个 matched-random seed 的平均 refusal 至少多下降 10pp。

### 7.4 成功等级

- **gradient-free key set 成立**：在新 holdout 上，0.01% primary 配置 refusal
  <=0.05，并满足全部硬约束；
- **接近 Taylor**：CRFP 0.01% refusal 不高于同 split Taylor/Wanda 结果 5pp，且
  PPL/KL 不更差超过预设容差；
- **部分成功**：0.01% 未达标，但 <=0.10% 存在 refusal<=0.05 的合格配置；
- **弱成功**：只有 0.50%--1.00% 合格，说明 forward proxy 能定位较大的分布式
  edge set，但不能复现极稀疏集合；
- **阴性**：<=1% 均不优于 random/旧 edge，或效果只伴随 PPL/KL/输出质量损伤；
- **retrospective-only**：没有新 holdout 时，无论旧 val 多强都只能使用此标签。

## 8. 软件设计

### 8.1 新模块

新增 `src/ttsafety/refusal_flow.py`：

1. `response_prediction_mask(...)`
   - 构造因果 shift 后的 response-prediction mask；
2. `collect_paired_writer_moments(...)`
   - 同批处理 R/C，在线累计 mean/variance/sign；
3. `collect_harmless_writer_moments(...)`
   - 收集 harmless input second moment；
4. `direct_refusal_flow_score(...)`
   - 分块计算 `mean_b`、`se_b`、`B`、`H` 和 CRFP score；
5. `rank_crfp_indices(...)`
   - benefit eligibility、tempered cost、global top-k、per-matrix cap；
6. `assert_gradient_free(...)`
   - 检查全部参数无 gradient、无参数启用 `requires_grad`。

### 8.2 Backbone-only forward

activation collection 不需要词表 logits。Llama 第一阶段优先调用 backbone：

~~~text
model.model(input_ids=..., attention_mask=..., use_cache=False)
~~~

避免由 128,256 词表产生完整 LM-head logits。实现需提供兼容 helper；若模型架构
没有统一 backbone 接口，显式报错，不能静默退回会大幅增加显存的完整 logits
forward。

grouped calibration 和最终 paired logp 才调用完整 causal-LM forward。

### 8.3 新脚本

- `scripts/score_refusal_flow.py`
  - `--stage paired|harmless|rank|all`；
  - 分矩阵/分层落盘；
  - 输出显存和时间 provenance；
- `scripts/calibrate_refusal_flow_groups.py`
  - 创建固定 group manifest；
  - 逐组 forward ablation；
  - resume-safe；
- `scripts/sweep_refusal_flow_prune.py`
  - 主 sparsity grid、controls、ablations；
  - 不接触最终 holdout；
- `scripts/run_refusal_flow_final.py`
  - 校验锁定配置和数据 hash；
  - 新 holdout 只运行一次；
  - 若结果文件存在则拒绝覆盖。

### 8.4 新测试

新增 `tests/test_refusal_flow.py`：

- response prediction mask 的 causal shift 正确；
- prompt/padding 不进入统计；
- R/C 长度不同仍按每个回答均值、每个 pair 等权；
- Welford 在线 mean/variance 与完整 tensor 计算一致；
- toy Linear 中所有 edge flow 之和等于 `r^T W delta_x`；
- 删除单 edge 后 direct-flow 变化精确等于 `-b_ij`；
- direction sign canonicalization 正确；
- LCB、`tau`、tempered score 数值正确；
- per-matrix eligibility 和 global top-k 可复现；
- label swap 后 `mu_delta` 符号翻转；
- backbone-only forward 不生成 vocab logits；
- 全过程所有 parameter `.grad is None`；
- reversible pruning context 退出后逐比特恢复。

## 9. 实验阶段

### C0：预注册、hash 和 baseline 冻结

任务：

- 写入本计划版本 hash；
- 生成 CAA score/calibration split manifest；
- 锁定主 score 公式、`beta=1`、`alpha=0.5`、L7--L18、components=both；
- 锁定 primary sparsity=0.01%；
- 记录模型、directions、数据和代码 commit hash；
- 检查新 external holdout 是否存在并完成去重/hash；
- 复现 baseline PPL、harmful_val 和 harmless 指标。

停止条件：

- baseline PPL 相对 13.0621 漂移 >1%；
- 数据 hash 与历史记录无解释地变化；
- 新 holdout 与旧数据出现不可接受的重复；
- 在 score 运行前无法锁定 primary config。

### C1：设施验证

任务：

- 完成全部 `test_refusal_flow.py`；
- 16 个 CAA pairs 上检查 R/C mask 和 activation shape；
- 验证 backbone-only 和 full-model hidden states 在相同 site 一致；
- 验证 no-grad、无 `.grad` tensor；
- 测量 3B 单 batch 峰值显存；
- 检查在线统计在 batch size 1/2/4 下结果一致。

验收：

- 全量测试通过；
- batch-size 变化造成的 fp32 moment 相对误差 <1e-5；
- 退出 hook/context 后模型状态不变；
- 峰值显存明显低于现有 Taylor 的实测 9.98GiB，或对差异给出解释。

### C2：解析 CRFP score

任务：

1. 在 score split 收集 paired response activation moments；
2. 在 harmless 数据收集二阶矩；
3. 分矩阵生成 `B`、`H`、CRFP score；
4. 保存 top candidate ranking；
5. 生成 layer/component 分布和分数诊断。

必须检查：

- positive `B` fraction；
- LCB 相比 raw mean 删除了多少候选；
- top edges 是否异常集中在一个 input/output dimension；
- 每矩阵 top candidate 数是否触发 cap；
- CRFP 与 Taylor、旧 edge 的 top-k Jaccard/overlap；
- score 与 weight magnitude、Wanda cost 的 Spearman 相关；
- label swap 后主排名是否显著改变。

若真实和 label-swapped CRFP 的 top-k overlap 或行为效果几乎一致，停止主张
refusal specificity，先调查 mask、direction sign 和 pair construction。

### C3：纯解析 sparsity sweep

对以下规则运行完整固定网格：

- CRFP（主）；
- flow-only；
- full-ratio；
- no-LCB；
- consistency-weighted；
- label-swapped CRFP；
- random direction CRFP 三 seed；
- random weight 三 seed；
- 复用旧 edge/Taylor/Wanda/magnitude 结果作为参照。

评估数据只使用旧 harmful_val、harmless 和 WikiText，因此本阶段是 retrospective
development comparison，不是新独立结论。

C3 完成后不能根据 val 结果修改主公式。任何新公式进入单独的 exploratory 文件。

### C4：grouped forward calibration

只对 C0 已定义的主 CRFP 执行：

- 创建 top 0.10% group manifest；
- 在独立 CAA calibration split 上逐组评估 margin；
- 计算 bootstrap CI 和 harmless KL；
- 生成 CRFP-cal ranking；
- 在旧 val 上运行相同 sparsity grid；
- 比较 CRFP-cal 是否稳定优于未校准 CRFP。

如果 48 个组全部不确定：

- 不放宽 CI；
- 报告 calibration split 的统计功效；
- CRFP-cal 标记 inconclusive；
- 仍可继续评估预注册的纯 CRFP。

### C5：机制对照和样例检查

任务：

- 运行第 6 节的固定消融；
- 对 primary 0.01% 配置保存 changed/still-refusal/harmless 样例；
- 检查非拒绝是否来自空输出、截断、乱码或无关回答；
- 比较选中 edge 的 layer、component、row/column 分布；
- 比较 CRFP、CRFP-cal、Taylor ratio 的集合 overlap；
- 检验 MLP/attention 是否仍呈现 Route A 中的协同模式。

人工样例只能用于错误分析，不能在看完后更换 primary 配置。

### C6：锁定和新 held-out final

进入条件：

- C0 已锁定的新 external holdout 可用；
- primary config、ranking hash、sparsity 和代码 commit 已写入 selection JSON；
- 无数据泄漏；
- 所有质量控制通过。

一次性运行：

- baseline；
- primary CRFP 0.01%；
- CRFP-cal 0.01%（只有 C4 预注册规则产生有效 ranking 时）；
- matched random 三 seed；
- harmless、PPL、KL、输出质量；
- 独立语义 judge/盲人工标注。

不在 final 后修改 score、layer、component、sparsity 或 calibration group。

## 10. 产物

计划新增：

~~~text
src/ttsafety/refusal_flow.py
scripts/score_refusal_flow.py
scripts/calibrate_refusal_flow_groups.py
scripts/sweep_refusal_flow_prune.py
scripts/run_refusal_flow_final.py
tests/test_refusal_flow.py

data/splits/crfp_caa_split_seed0.json
data/weight_scores/crfp_activation_stats.pt
data/weight_scores/crfp_harmless_moments.pt
data/weight_scores/crfp_ranking.pt
data/weight_scores/crfp_calibrated_ranking.pt
data/weight_scores/crfp_group_manifest.json
data/samples_refusal_flow.jsonl

results/sweep_refusal_flow_prune.json
results/refusal_flow_group_calibration.json
results/refusal_flow_final.json
results/refusal_flow_pareto.png
results/refusal_flow_layer_distribution.png

docs/results-gradient-free-refusal-flow-pruning.md
~~~

大 tensor 放 `data/`，汇总指标和图放 `results/`。不导出完整的 safety-removed
模型 checkpoint。

JSON metadata 至少记录：

- model ID、revision、dtype；
- direction path/hash 和 sign；
- score/calibration/evaluation data hashes；
- target layers、components、pool size；
- beta、alpha、tau 定义、candidate cap、sparsity；
- score/ranking hash；
- torch、transformers、CUDA、GPU；
- batch size、最大/均值 token 长度；
- peak allocated/reserved GPU memory；
- wall-clock time；
- `gradient_free_verified=true` 及检查结果；
- complete/failed 状态。

## 11. 预计开销

### 11.1 GPU 显存

当前 Taylor 3B 单 batch 实测峰值约 9.98GiB，其中主要额外项是目标梯度和
415M 元素 FP32 accumulator。

CRFP activation collection：

- 模型 BF16 权重约 5.98GiB；
- 无 gradient；
- 无 per-weight FP32 accumulator；
- backbone-only forward 不生成 128K vocab logits；
- 仅保存每个 writer 输入维度的在线 fp32 moments。

3B 预计峰值约 6--8GiB，以 C1 实测为准。使用相同短序列时：

| 模型规模 | 预计可行性 |
|---|---|
| 3B | 非常宽松 |
| 7B/8B | 宽松 |
| 13B | batch 1、短序列下可能可行，必须实测 |
| 14B | 临界，可能需要量化/offload |
| 30B+ | 当前单卡 BF16 方案不可行 |

grouped calibration 需要完整 logits，但仍只有 forward，不保留 backward graph。

### 11.2 时间和存储

3B 粗略预算：

| 阶段 | 预计 GPU 时间 |
|---|---:|
| C1 smoke/profile | 5--10 分钟 |
| C2 paired + harmless moments | 5--15 分钟 |
| C3 sparsity/controls | 30--60 分钟 |
| C4 最多 48 个 group ablations | 20--60 分钟 |
| C5 diagnostics | 20--40 分钟 |
| C6 final | 5--15 分钟 |

总开销预计 1--3 小时，主要由 generation/PPL/KL 和 grouped calibration 决定，
不是 activation score 本身。

存储预算：

- activation moments 只按输入维度保存，预计 MB 级；
- 完整 CRFP fp16 score 约 793MiB；
- 5% ranking 约 199MiB；
- controls 若都保存完整 score 会增加数 GB；
- 默认只为主 CRFP 保存完整 score，controls 保存 top candidates 和汇总统计。

## 12. 计划接口与复现命令

以下命令定义未来接口；实现和测试通过后才执行：

~~~bash
# C0/C1
uv run python scripts/score_refusal_flow.py --stage prepare
flock -w 14400 .gpu.lock uv run pytest tests/test_refusal_flow.py
flock -w 14400 .gpu.lock uv run python scripts/score_refusal_flow.py --stage smoke

# C2
flock -w 14400 .gpu.lock uv run python scripts/score_refusal_flow.py --stage paired
flock -w 14400 .gpu.lock uv run python scripts/score_refusal_flow.py --stage harmless
uv run python scripts/score_refusal_flow.py --stage rank

# C3
flock -w 14400 .gpu.lock uv run python scripts/sweep_refusal_flow_prune.py --stage primary
flock -w 14400 .gpu.lock uv run python scripts/sweep_refusal_flow_prune.py --stage controls
uv run python scripts/sweep_refusal_flow_prune.py --finalize

# C4
uv run python scripts/calibrate_refusal_flow_groups.py --stage prepare
flock -w 14400 .gpu.lock uv run python scripts/calibrate_refusal_flow_groups.py --stage evaluate
uv run python scripts/calibrate_refusal_flow_groups.py --stage finalize
flock -w 14400 .gpu.lock uv run python scripts/sweep_refusal_flow_prune.py --stage calibrated

# C5/C6
flock -w 14400 .gpu.lock uv run python scripts/sweep_refusal_flow_prune.py --stage diagnostics
flock -w 14400 .gpu.lock uv run python scripts/run_refusal_flow_final.py

# 全量回归
flock -w 14400 .gpu.lock uv run pytest
~~~

所有 GPU 阶段使用 `.gpu.lock`，长任务按 cell 或 group 立即写盘并支持 resume。

## 13. 决策树与停止条件

1. **C1 mask、direct-flow 恒等式或 no-grad 检查失败**：停止，不运行 score。
2. **真实/label-swapped score 行为无显著差异**：停止主张 refusal specificity，
   检查数据、方向和 sign。
3. **CRFP 只在 PPL>5%、KL>0.10 或质量异常时降低 refusal**：判为能力损伤，
   不进入 final。
4. **CRFP 不优于旧 signed edge**：方法没有证明 paired trajectory/LCB 的增益；
   完成诊断后停止，不扩无界网格。
5. **CRFP 接近 Taylor 且通过硬约束**：进入 grouped calibration 和新 holdout。
6. **CRFP-cal 不优于 CRFP**：保留纯 CRFP，报告 group calibration 阴性，不能
   反复修改 group 定义。
7. **没有新 external holdout**：结果保持 retrospective-only，不复用旧 test。
8. **final 文件已存在**：拒绝重跑，除非建立全新的预注册评测集和实验版本。
9. **任何阶段发现数据泄漏、baseline 漂移或 restore 失败**：立即停止并修复，
   不能用后续结果覆盖问题。

## 14. 最终应回答的问题

结果文档必须明确回答：

1. CRFP 在 0.01% 是否达到 Route B Taylor/Wanda 的行为效果和能力保持水平？
2. paired response trajectory 相比旧 prompt-last-token edge 是否有稳定增益？
3. LCB 和 tempered harmless cost 分别贡献了什么？
4. grouped forward calibration 是否提高了真实 causal precision？
5. CRFP 与 Taylor 选中集合的 overlap 是高还是低；低 overlap 时是否仍有相同行为？
6. 选中 edge 主要分布在哪些层、MLP/attention 组件和输入/输出维度？
7. gradient-free 节省了多少峰值显存、wall time 和磁盘，但付出了多少效果损失？
8. 结论是极稀疏 refusal-specific edge set、较大的分布式集合，还是 proxy 失败？

## 15. 安全与发布边界

本实验研究安全对齐的机制和脆弱性。运行和发布时：

- 不上传或发布已经移除安全拒绝行为的完整模型 checkpoint；
- 不把 refusal 降低描述为安全增强；
- 样例发布避免提供可直接滥用的详细有害执行内容；
- 报告关键词 judge、语义 judge 和人工检查的局限；
- 同时报告能力损伤、over-refusal、空输出和重复等失败模式；
- 发布 ranking 或精确 mask 前单独评估其双重用途风险。
