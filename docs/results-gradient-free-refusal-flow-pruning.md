# Gradient-free 对比拒绝流剪枝实验结果（Route C）

> ⚠️ **历史存档(标注于 2026-08-27):本文记录较早阶段的计划/方法/结果,可能与当前代码与结论脱节。**
> 方法线现已定名 **BLADE** 并引入 best-first ELS(有效层选择);当前定稿的方法与实验结果请以
> [`blade-method.md`](blade-method.md)、[`blade-algorithm.md`](blade-algorithm.md)、
> [`blade-experiment-report.md`](blade-experiment-report.md) 为准。**下文内容保留原貌,未作修订。**

> 对应计划：
> [`plan-gradient-free-refusal-flow-pruning.md`](plan-gradient-free-refusal-flow-pruning.md)。
>
> 模型：Llama-3.2-3B-Instruct，bf16，RTX 5090。
>
> 状态：C0--C3 已完成；retrospective/val-only。没有复用已经被 Route A 使用的
> `harmful_test`，也没有新的 external holdout。
>
> 主结论：**CRFP 的核心打分确实比 Route B Taylor 更省显存、更快，但当前解析
> score 不能替代 Route B。它能定位与拒绝有关的 weights，却无法在预注册
> harmless KL 约束内得到低 refusal；同 sparsity 下也弱于已有 gradient-free
> signed actdiff edge。**

## 1. 完整性与实现状态

本轮新增：

- `src/ttsafety/refusal_flow.py`；
- `scripts/score_refusal_flow.py`；
- `scripts/sweep_refusal_flow_prune.py`；
- `tests/test_refusal_flow.py`。

实现满足：

- 所有模型参数 `requires_grad=False`；
- 全过程不调用 backward、autograd gradient、JVP/VJP 或 optimizer；
- score collection 使用 backbone-only forward，不生成完整 vocabulary logits；
- response activation 使用 causal shift 后的 response-prediction positions；
- refusal/compliance 回答分别长度归一化，每个 pair 等权；
- pair moments 使用在线 parallel-Welford 聚合；
- 目标权重仍为 L7--L18 的 `down_proj + o_proj`，共 415,236,096 个；
- 剪枝 context 内精确置零，退出后逐比特恢复；
- 所有结果逐 cell 落盘，未运行旧 test。

测试结果：

- Route C + 既有剪枝 targeted tests：12/12 通过；
- 项目全量回归测试通过；
- toy Linear 中所有 edge flow 之和等于 `r^T W delta_x`；
- 删除单 edge 后 direct-flow 变化精确等于该 edge flow 的负值；
- 真实模型 smoke 捕获全部 24 个 writer，无 parameter gradient 被分配。

## 2. 数据拆分和泄漏边界

247 个 CAA pairs 在运行前按 canonical JSON SHA256 排序并固定拆分：

| split | 数量 | 用途 |
|---|---:|---|
| score | 197 | paired R-C activation moments |
| calibration | 50 | 预留给 grouped forward calibration |

manifest：`data/splits/crfp_caa_split_seed0.json`。

需要强调：这 247 个 pairs 已经整体用于先前 Route B，因此 80/20 拆分只能防止
Route C 内部 score/calibration 自我验证，不能构成新的独立证据。旧
`harmful_val` 也已经参与先前选择，本轮只把它作为 retrospective 横向比较。

## 3. 实际执行的 CRFP 算法

对 writer `y_l = W_l x_l`，在同一 instruction 的 refusal/compliance response
prediction positions 上收集：

~~~text
delta_x[n,l,j] = mean(x_R[n,l,:,j]) - mean(x_C[n,l,:,j])
~~~

对每条 edge 定义：

~~~text
a[l,i,j] = r[l,i] * W[l,i,j]
mean_b    = a * mean(delta_x[l,j])
se_b      = abs(a) * sqrt(var(delta_x[l,j]) / N)
B         = max(mean_b - se_b, 0)
~~~

harmless cost：

~~~text
H = abs(W[l,i,j]) * sqrt(E_harmless[x[l,j]^2])
~~~

主 score：

~~~text
S_CRFP = B / sqrt(H + median_matrix(H))
~~~

每个矩阵只允许正 `B` 中最高 10% 进入候选，之后进行 capped global ranking。

## 4. 成本实测

### 4.1 GPU 显存

| 阶段 | peak allocated | peak reserved |
|---|---:|---:|
| 2-pair real-model smoke | 6.04 GiB | 6.11 GiB |
| 197-pair activation moments | 6.10 GiB | 6.25 GiB |
| 320 harmless moments | 6.05 GiB | 6.25 GiB |
| score + ranking | 7.03 GiB | 7.29 GiB |
| Route B Taylor 单 batch | 9.98 GiB | 10.39 GiB |

Route C activation collection 相对 Route B Taylor 的 measured peak allocated 降低：

~~~text
(9.98 - 6.10) / 9.98 = 38.9%
~~~

即使把 score/ranking 阶段的 7.03GiB 作为 Route C 总峰值，相对 Taylor 也降低
约 29.6%。

### 4.2 核心阶段 wall time

不含一次模型加载和后续行为评估：

| 阶段 | wall time |
|---|---:|
| paired moments | 3.22 s |
| harmless moments | 1.04 s |
| score + ranking | 0.95 s |
| 合计 | **5.21 s** |

这证明 Route C 的核心 attribution cost 很低。完整实验的主要时间仍来自 8 个
cell 的 generation、50k-token PPL 和 harmless KL，而不是 CRFP score。

### 4.3 存储

| 产物 | 大小 |
|---|---:|
| paired activation stats | 1.6 MiB |
| harmless moments | 540 KiB |
| 完整 fp16 CRFP score | 793 MiB |
| top-1% ranking | 40 MiB |

score 文件与 Route B 的单个 fp16 score 大小相同；Route C 的优势主要是 GPU
常驻状态和反向计算，不是每权重结果的磁盘复杂度。

## 5. Score 诊断

24 个矩阵中：

- 正 LCB benefit fraction 为 45.3%--47.5%；
- benefit eligibility 后共有 19,250,862 个候选；
- top-0.01% 覆盖全部 24 个矩阵，没有退化到单层或单矩阵；
- primary 41,524 个 weights 中，MLP 为 21,554，attention 为 19,970；
- top 权重较多出现在 L11--L17，L13 两个 writer 最集中。

CRFP 与 Taylor/Wanda ratio 的 top-0.01%：

| 指标 | 数值 |
|---|---:|
| CRFP weights | 41,524 |
| Taylor/Wanda weights | 41,524 |
| intersection | 824 |
| selected-set overlap | 1.98% |
| Jaccard | 1.00% |

两者找到的集合明显不同。后续行为差异表明，这不是简单地用 forward proxy 重现
Taylor ranking。

## 6. 八个固定 sparsity cells

所有 sparsity 相对于 415,236,096 个目标权重池。

| target sparsity | weights | harmful_val refusal | harmless | PPL delta | KL | adverse |
|---:|---:|---:|---:|---:|---:|---:|
| 0.001% | 4,152 | 0.859 | 0.003 | +0.03% | 0.0374 | 0 |
| 0.003% | 12,457 | 0.531 | 0.003 | +0.18% | 0.1344 | 0 |
| **0.010% primary** | **41,524** | **0.359** | **0** | **+0.73%** | **0.2769** | **0** |
| 0.030% | 124,571 | 0.000 | 0 | +1.91% | 0.3940 | 0 |
| 0.050% | 207,618 | 0.016 | 0 | +2.61% | 0.4680 | 0 |
| 0.100% | 415,236 | 0.016 | 0 | +4.30% | 0.6224 | 0 |
| 0.500% | 2,076,180 | 0.000 | 0 | +8.62% | 1.4534 | 0 |
| 1.000% | 4,152,361 | 0.016 | 0 | +11.78% | 1.8846 | 0.003 |

关键观察：

1. 只删除 0.001% 已使 refusal 下降 14.1pp，说明 CRFP 排名含有真实拒绝信号；
2. refusal 随 sparsity 快速下降，0.03% 已到 0；
3. 但 harmless KL 更早越过 0.10：0.003% 时已为 0.134；
4. 因而不存在同时满足 `refusal<=0.05` 和 `KL<=0.10` 的 cell；
5. harmless keyword refusal 和简单输出质量标志保持正常，但低 keyword refusal
   不能抵消明显的分布级 KL 变化。

## 7. 与 Route B 和旧 gradient-free edge 比较

在相同 0.01% target sparsity：

| 方法 | gradient | refusal | PPL delta | KL |
|---|---:|---:|---:|---:|
| random mean | 否 | 1.000 | 约 0 | 很低 |
| **Taylor/Wanda ratio** | 是 | **0.031** | **+0.03%** | **0.0017** |
| signed actdiff edge | 否 | 0.125 | +0.09% | 0.0381 |
| **CRFP** | 否 | **0.359** | **+0.73%** | **0.2769** |

CRFP 相对 random 有 64.1pp refusal gap，因此不是随机选择；但它同时：

- 比 Taylor/Wanda 多 32.8pp refusal；
- 比旧 signed edge 多 23.4pp refusal；
- KL 分别约为 Taylor/Wanda 的 167 倍、旧 edge 的 7.3 倍。

因此，当前 paired response direct-flow score 没有改善旧 gradient-free edge，更没有
达到 Route B 的效果/能力 Pareto 前沿。

## 8. 预注册判定

primary 0.01% 要求：

- refusal<=0.05；
- harmless refusal<=0.05；
- PPL delta<=5%；
- KL<=0.10；
- adverse<=1%；
- 超 random 至少 10pp。

实际：

- random gap：通过；
- harmless/PPL/quality：通过；
- refusal：失败（0.359）；
- KL：失败（0.2769）。

因此：

**Route C 的 gradient-free sparse key set 未建立，selection=null。**

这不是“完全无信号”：CRFP 能用极少 weights 控制 refusal；失败点是拒绝效果和
harmless distribution change 没有被当前解析 cost 干净地解耦。

## 9. 为什么没有继续 C4/C6

计划的停止条件规定：如果 CRFP 不优于已有 signed edge，应完成诊断后停止，不扩展
无界网格。当前 0.01% 的 refusal、PPL、KL 三项全部弱于 signed edge，因此触发该
条件。

所以本轮：

- 没有运行最多 48 组的 CRFP-cal grouped ablation；
- 没有事后修改 `beta`、`alpha`、`tau` 或主公式；
- 没有选择更高 sparsity 冒充 primary 成功；
- 没有复用旧 `harmful_test`；
- 没有新的 external holdout，故不运行 C6。

这保留了阴性结果的可解释性，也防止在已看过的 val 上继续过拟合。

## 10. 可能原因

当前证据支持以下解释，但尚未逐项证明：

1. **direct flow 不等于 downstream causal effect**：`r^T W delta_x` 只描述局部
   writer 写入，没有经过后续层非线性和最终 LM head；
2. **response trajectory 含行为后果**：R/C activation 差异不仅是产生拒绝的原因，
   也包含 teacher-forced 不同回答文本造成的结果；
3. **局部 Wanda cost 不足**：`|W| RMS(x)` 只估计直接 writer-output 扰动，不能
   预测完整 harmless KL；
4. **tempered denominator 约束过弱**：`sqrt(H+tau)` 保留 benefit magnitude，但
   对 harmless cost 的惩罚可能不足；
5. **Taylor 的 downstream objective 信息关键**：Route B gradient 直接对应
   refusal/compliance logp margin，而 CRFP 只对齐预先提取的 activation direction。

这些解释只能指导下一份预注册实验，不能在本结果中通过事后换公式验证。

## 11. 产物

~~~text
src/ttsafety/refusal_flow.py
scripts/score_refusal_flow.py
scripts/sweep_refusal_flow_prune.py
tests/test_refusal_flow.py

data/splits/crfp_caa_split_seed0.json
data/weight_scores/crfp_activation_stats.pt
data/weight_scores/crfp_harmless_moments.pt
data/weight_scores/crfp.pt
data/weight_scores/crfp.json
data/weight_scores/ranking_crfp.pt

results/sweep_refusal_flow_prune.json
results/refusal_flow_pareto.png
docs/results-gradient-free-refusal-flow-pruning.md
~~~

## 12. 下一步边界

如果继续研究 gradient-free 版本，应建立新的计划和开发 split，而不是修改本轮主
结果。最有价值的候选方向是：

1. 在独立 calibration split 上做预算受限的 grouped forward ablation；
2. 用真实 forward margin change 过滤 direct-flow 候选组；
3. 比较 full Wanda ratio 与 tempered cost，但预先固定选择规则；
4. 使用同 prompt 的 activation patching，减少 R/C teacher-forcing 文本差异的后果
   混入；
5. 引入新的 external holdout 和语义安全 judge 后再做最终验证。

当前最稳健的总结是：

> Route C 将 3B attribution 峰值显存从约 9.98GiB 降到 6.10--7.03GiB，核心
> score 只需约 5.21 秒；代价是失去 Taylor 对 downstream logp margin 的直接
> 敏感性。当前 CRFP 找到的是“拒绝相关但能力纠缠”的权重集合，而不是 Route B
> 已发现的干净极稀疏 key-weight set。
