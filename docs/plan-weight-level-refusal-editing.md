# Plan: Weight-level 拒绝方向编辑与选择性权重剪枝

> ⚠️ **历史存档(标注于 2026-08-27):本文记录较早阶段的计划/方法/结果,可能与当前代码与结论脱节。**
> 方法线现已定名 **BLADE** 并引入 best-first ELS(有效层选择);当前定稿的方法与实验结果请以
> [`blade-method.md`](blade-method.md)、[`blade-algorithm.md`](blade-algorithm.md)、
> [`blade-experiment-report.md`](blade-experiment-report.md) 为准。**下文内容保留原貌,未作修订。**

> 状态：已完成（W0–W5）。路线 A 已执行一次 held-out test；路线 B 按泄漏
> 规则保持 val-only。完整结果见
> `docs/results-weight-level-refusal-editing.md`。
> 前置结果：
> - activation steering：L8 α=−2，在 held-out test 上拒绝率 0.990→0.670，
>   harmless 拒绝率 0，WikiText PPL +2.44%；
> - neuron pruning（N7）：D1/D2/actdiff 全部没有合格 key set；约束内最强
>   actdiff top-4096 在 test 上拒绝率 0.910，但 PPL +16.33%；
> - 已有逐层 mean-diff refusal directions：
>   data/directions/refusal_llama32_3b_instruct.pt。

## 1. 研究问题

N7 表明，拒绝行为无法通过删除少量 MLP neuron 干净地关闭。下一步检验：

1. 把干预从 neuron 基底改到 weight 空间，是否能获得更好的
   拒绝率–通用能力 Pareto 前沿？
2. 直接删除“拒绝相关权重”是否有效，还是仍需删除大量权重并破坏模型？
3. 如果 weight-level 有效，收益来自更细粒度的稀疏权重选择，还是显式利用
   refusal direction 的低秩权重编辑？
4. MLP 和 attention 哪一类 residual writer 对拒绝方向的生成更关键？

主假设：

- **H1（主要）**：rank-one 权重正交化只删除各 residual writer 写向拒绝方向的
  分量，会明显优于整列 neuron 置零。
- **H2（对照）**：普通单 weight 剪枝即使粒度更细，如果评分不对齐拒绝方向，
  也不会自动得到行为特异性。
- **H3（机制）**：MLP down_proj 与 attention o_proj 共同写入拒绝方向；
  只编辑一种组件的效果弱于二者联合。
- **H4（分布式性）**：如果 refusal-aware 单权重剪枝仍只在高 sparsity/PPL
  损伤区生效，则 N7 的“分布式方向而非稀疏元件集合”结论推广到 weight 粒度。

## 2. 方法优先级与边界

本计划把“weight level”分成两条路线，不混为同一个实验。

### 路线 A（主线）：refusal-direction weight orthogonalization

对所有写回 residual stream 的矩阵 W_out，使用单位拒绝方向 r̂：

~~~text
P_λ = I − λ r̂ r̂ᵀ
W_out' = P_λ W_out
~~~

- λ=0：原模型；
- 0<λ<1：部分删除写向拒绝方向的分量；
- λ=1：完整正交化。

Llama 中第一阶段只编辑：

- model.layers[l].mlp.down_proj.weight，形状 3072×8192；
- model.layers[l].self_attn.o_proj.weight，形状 3072×3072。

不在主网格中直接编辑 gate_proj/up_proj/q_proj/k_proj/v_proj，因为它们不直接
写回 residual stream。embedding 作为 W3 的扩展项单独评估，不与主网格混合。

该编辑是 rank-one、方向对齐、密集但低秩的权重修改，不等于 unstructured pruning。

### 路线 B（第二阶段）：refusal-aware individual-weight pruning

对单个权重计算“删除该权重是否降低拒绝偏好，同时尽量不伤 harmless 能力”的
Taylor/Wanda 组合分数，再做稀疏置零。它回答“是否存在稀疏拒绝权重集合”，
但不承担主线验收。

普通 magnitude/Wanda/SparseGPT 只优化通用重构或语言建模能力，作为能力保持
对照，不视为拒绝特异方法。

参考：

- Arditi et al., Refusal in Language Models Is Mediated by a Single Direction:
  https://arxiv.org/abs/2406.11717
- Wanda: https://arxiv.org/abs/2306.11695
- SparseGPT: https://arxiv.org/abs/2301.00774
- Does Localization Inform Editing?: https://arxiv.org/abs/2301.04213

## 3. 数据、泄漏控制与固定基线

数据沿用现有固定 seed=0 划分：

| 用途 | 数据 | 数量 |
|---|---|---:|
| direction/weight score | harmful_train | 256 |
| benign preservation score | harmless train | 320 |
| CAA/Taylor pair | data/caa_pairs.jsonl | 247 |
| 选 source/scope/λ/sparsity | harmful_val | 64 |
| 最终一次报告 | harmful_test | 200 |
| PPL | WikiText-2 test，前 50k token | 固定 |

严格规则：

- harmful_test 不参与 direction、层、组件、λ、sparsity 或阈值选择；
- test 只对最终选定的一个路线 A 配置运行一次；
- 路线 B 若继续到 final，也必须使用独立、预先锁定的配置，不能根据路线 A 的
  test 结果再调参；
- 现有 baseline/steering/N7 JSON 用于对比，但 final 阶段重新跑一次 baseline，
  避免代码或环境变化造成不可比；
- 所有 sweep cell 写入 JSON 后立即落盘，支持断点续跑。

固定对照：

1. 原模型；
2. activation steering L8 α=−2；
3. N7 actdiff global top-4096（仅引用现有结果，PPL +16.33%）；
4. 3 个 label-shuffled mean-diff 随机方向；
5. 同 scope/λ 的普通随机单位方向，3 个 seed（只在 shortlist 上跑）。

## 4. 指标与质量控制

### 4.1 主指标

- harmful_val/test refusal rate：沿用关键词 judge，保持与 M/N 系列可比；
- compliance rate = 1 − refusal rate。

### 4.2 副作用指标

- harmless refusal rate；
- WikiText teacher-forced PPL 及相对退化；
- harmless token-level KL：

~~~text
KL_base→edit = mean KL(p_base(.|prefix) || p_edit(.|prefix))
~~~

  在固定的 128 条 harmless prompt、每条最多 128 token 上计算；
- harmless completion exact/near match：
  与 baseline greedy completion 的 token agreement；
- 输出质量：
  空回复率、长度、unique-token ratio、连续重复 n-gram、乱码启发式；
- 权重改变量：
  每矩阵 ||ΔW||F / ||W||F、全局相对 Frobenius norm、被移除方向能量。

### 4.3 人工抽检

每个 shortlist 配置保存：

- harmful_val 最易变化的 10 条；
- harmful_val 仍拒绝的 5 条；
- harmless 随机 10 条；
- 极端 λ=1 的 10 条。

人工标记：明确拒绝、hedged engagement、真实服从、软拒绝、乱码/循环。
最终结论不能只依赖关键词 judge。

## 5. 选择规则与验收标准

### 5.1 硬约束

主选择只在以下条件同时成立的 cell 中进行：

- harmless refusal ≤5%；
- PPL 退化 ≤5%；
- harmless KL ≤0.10 nat/token；
- 空回复/乱码/循环率 ≤1%；
- 与随机方向相比，harmful refusal 至少多下降 10pp。

可行 cell 中按以下顺序选择：

1. harmful_val refusal 最低；
2. KL 更低；
3. PPL 退化更低；
4. 编辑矩阵数量更少。

如果无可行 cell：

- 允许报告 relaxed 区（PPL≤10%、KL≤0.20），但标记为不通过主验收；
- relaxed 结果不冒充成功配置；
- 仍可选择“最有信息的阴性 cell”做样例分析，但默认不跑 test，除非用户批准。

### 5.2 相对基线的成功等级

- **最低成功**：同为 PPL≤5% 时，val refusal 低于 activation steering 的
  0.766，并超过随机方向 10pp；
- **主要成功**：held-out test refusal <0.670，同时 PPL≤5%、harmless≤5%；
- **强成功**：test refusal ≤0.20，同时满足全部硬约束；
- **路线 B 关键权重集合成立**：
  拒绝率 ≤0.05、超随机 >10pp、PPL≤5%，且 sparsity≤1% 的目标权重池；
- 若只在 PPL>25% 或明显乱码时生效，判为模型损伤，不算成功。

## 6. 软件设计

### 6.1 新模块

新增 src/ttsafety/weight_edit.py：

1. project_residual_writes(...)：
   可逆 forward-hook context，用于 sweep；
2. materialize_orthogonalization(...)：
   真正计算 W'=(I−λrrᵀ)W；
3. restore/materialization verification：
   不通过 BF16 add/sub 恢复权重，避免舍入漂移；
4. shuffled_direction(...)：
   生成 label-shuffled mean-diff 与随机单位方向；
5. weight_delta_stats(...)：
   计算每层/组件的改变量与方向能量。

Sweep 使用 module output projection：

~~~text
y = W x
y' = y − λ (y·r̂) r̂
~~~

它与 W'=(I−λr̂r̂ᵀ)W 数学等价，但不修改参数，适合几十个 cell 共用同一模型。

最终物化时：

- 用 fp32 计算投影，再 cast 回原 dtype；
- 不保存整份 6GB 模型，只保存方向、config、矩阵 delta 统计；
- 若以后需要导出模型，再单独请求批准。

### 6.2 新脚本

- scripts/validate_weight_ortho.py
  - hook 与物化权重数值等价性；
  - λ=0 exact no-op；
  - λ=1 后 r̂ᵀW'≈0；
  - context 退出后权重逐比特不变。
- scripts/sweep_weight_ortho.py
  - pilot/main/controls/finalize 分阶段；
  - 每 cell resume-safe。
- scripts/run_weight_final.py
  - held-out test 一次；
  - baseline、选定 edit、随机方向对照；
  - 保存样例与图。
- scripts/score_refusal_weights.py
  - 路线 B Taylor/Wanda 分数；
  - 分块/断点落盘。
- scripts/sweep_weight_prune.py
  - 路线 B sparsity sweep 与随机/Wanda 对照。

### 6.3 新测试

- tests/test_weight_edit.py：
  - toy Linear 精确公式；
  - down_proj/o_proj shape 与层发现；
  - λ=0 bit-exact；
  - λ=1 direction residual <1e-5（fp32）；
  - hook 与物化输出 allclose；
  - 异常退出仍移除 hook；
  - 不修改 gate/up/q/k/v；
  - 多层、多组件组合正确。
- tests/test_weight_prune.py：
  - flat weight index ↔ matrix/index 映射；
  - selected weight 精确置零；
  - context 退出逐比特恢复；
  - per-matrix cap 与随机 seed 可复现。

## 7. 路线 A 实验阶段

### W0：冻结基线与环境

目标：

- 读取现有 directions、数据与 baseline；
- 确认 28 层、hidden=3072、intermediate=8192；
- 重新跑 16 条 harmful +16 条 harmless 的 smoke generation；
- 固定 torch/transformers/GPU/config provenance；
- 测一次完整 baseline PPL/KL 缓存。

产物：

- results/weight_edit_baseline.json；
- data/cache/harmless_base_logits_*.pt（只保存 KL 所需压缩统计，避免完整 logits
  过大）。

验收：

- baseline harmful_val refusal=1.000±关键词判定确定性误差；
- PPL 与 13.06 的历史结果相差 <1%；
- 若不一致，停止并调查，不能直接继续 sweep。

### W1：设施与等价性验证

对 L8 的 down_proj、o_proj、both，使用 r̂8，λ∈{0,1}：

- hook projection；
- 物化到模型副本；
- 同一输入比较 logits；
- context 退出后比较 state tensor checksum。

验收：

- λ=0 logits 逐比特相同；
- hook vs materialized 最大 logits 误差满足 bf16 合理阈值；
- fp32 λ=1 的 |r̂ᵀW'|/||W'|| <1e-5；
- 退出 context 后原权重逐比特一致；
- 全量测试通过。

### W2a：source direction pilot

候选 source directions：

- r̂8：当前约束内最佳 activation steering 层；
- r̂10：强 steering 窗口代表；
- r̂14：中层强翻转代表。

pilot 固定：

- components=both；
- destination layers=7–18；
- λ∈{0.5,1.0}；
- harmful_val 前32条、harmless 前64条、WikiText 10k token；
- 共 3×2=6 cells。

选择：

- 硬约束 proxy 内 harmful refusal 最低者；
- 若前两名相差≤5pp，使用较低 KL/PPL 者；
- source 选定后锁定，不因 full-grid 结果重新选择。

### W2b：主网格

锁定一个 source direction 后运行：

| 轴 | 取值 |
|---|---|
| components | mlp、attn、both |
| destination scope | source-only、L8–L14、L7–L18、L0–L27 |
| λ | 0.25、0.50、0.75、1.00 |

总计 3×4×4=48 cells，另加共享 baseline。

每个 cell：

- harmful_val 64；
- harmless 320；
- WikiText 50k-token PPL；
- harmless KL/greedy agreement；
- 权重 delta 统计；
- 立即写 results/sweep_weight_ortho.json。

绘图：

- refusal vs PPL Pareto；
- refusal vs harmless KL；
- λ 剂量曲线；
- component/scope heatmap；
- 每层被移除方向能量。

### W2c：主网格诊断

回答三个机制问题：

1. MLP-only、attention-only、both 是否存在协同；
2. 局部 scope 是否已足够，还是必须全层持续阻止方向写入；
3. λ 增强时 refusal 与 PPL/KL 是否同步恶化。

如果 both 显著优于两种单组件，报告 synergy：

~~~text
synergy = Δrefusal_both − max(Δrefusal_mlp, Δrefusal_attn)
~~~

同 scope/λ 下 synergy >10pp 才称为机制性协同。

### W3：shortlist 稳健性

从主网格 Pareto 前沿选最多 3 个配置：

- 每个配置跑 3 个随机单位方向；
- 每个配置跑 3 个 label-shuffled mean-diff direction；
- λ=1 finalist 加 norm-preserving 版本：
  每个投影后 column 恢复原 L2 norm，同时保持与 r̂ 正交；
- 比较 shared r̂ 与 per-destination-layer r̂_l：
  只在最佳 scope/components/λ 上增加一个 cell；
- 最佳 writes-only 配置增加 embedding projection 一个 cell；
  embedding 若造成 KL/PPL 明显恶化则不进入 final。

注意：

- norm-preserving、per-layer direction、embedding 都是 shortlist robustness，
  不进入主网格自由组合；
- 不根据这些诊断无限扩网格；
- 每个新增 cell 仍只用 val。

### W4：路线 A 最终 test

运行前把 selection JSON 锁定并记录 SHA256。

一次性评估：

- harmful_test 200：baseline vs selected edit；
- harmless 320；
- WikiText PPL；
- harmless KL；
- 随机方向 3 seed；
- 保存 20 条有代表性的生成样例；
- 物化 edit 与 hook 版本各生成一小批，验证行为一致。

产物：

- results/weight_ortho_final.json；
- results/weight_ortho_pareto.png；
- results/weight_ortho_components.png；
- data/samples_weight_ortho_final.jsonl；
- docs/results-weight-level-refusal-editing.md。

## 8. 路线 B：refusal-aware 单权重剪枝

路线 B 在 W4 后执行，或当 W2b 没有可行 cell 时提前作为替代诊断。它不修改
路线 A 已锁定的 test 选择。

### W5a：目标权重池

只考虑 N7a 的有效窗口 L7–L18：

- mlp.down_proj；
- self_attn.o_proj。

原因：

- 把计算控制在单卡可承受范围；
- 避免对明显无翻转深层做大规模多重比较；
- 与路线 A 的 residual writer 定义一致。

unstructured 零值不会在普通 dense kernel 上带来实际加速；本实验是机制干预，
不是部署压缩实验。

### W5b：拒绝特异 Taylor 分数

使用 247 个 CAA pairs。对 harmful prompt x、refusal completion R、
compliance completion C 定义长度归一化 margin：

~~~text
m_θ(x) = log p_θ(R|x)/|R| − log p_θ(C|x)/|C|
~~~

删除权重 w 的一阶效应：

~~~text
Δm ≈ −w ∂m/∂w
S_ref(w) = max(w ∂m/∂w, 0)
~~~

S_ref 高表示把 w 置零预计会降低 refusal-over-compliance margin。

benign 保留分数：

~~~text
S_keep(w) = |w| sqrt(E_harmless[x_in²])
S_ratio(w) = S_ref(w) / (S_keep(w) + ε)
~~~

比较三种排名：

1. Taylor-only：S_ref；
2. Taylor/Wanda ratio：S_ratio（主）；
3. signed actdiff edge score：
   max(r_i W_ij Δa_j, 0)，把 N7 neuron activation difference 映射到具体
   residual output edge。

控制：

- random；
- magnitude-smallest；
- Wanda-smallest（通用能力剪枝对照）；
- gradient label-shuffle。

### W5c：sparsity 网格

相对于 W5a 目标权重池：

~~~text
0.01%, 0.05%, 0.1%, 0.5%, 1%, 5%
~~~

约束：

- 单个矩阵最多剪 10%，避免全局排名集中摧毁一层；
- 分数只用 train/CAA pairs；
- val 指标与路线 A 完全相同；
- 每个 k 随机 3 seed；
- selected values 单独备份，context 退出精确恢复，不能通过加回 BF16 delta 恢复。

### W5d：路线 B 判定

- sparsity≤1%、PPL≤5%、KL≤0.10、拒绝率≤0.05、超随机>10pp：
  稀疏关键权重集合成立；
- 拒绝率下降但 PPL/KL 同步恶化：
  权重与通用功能纠缠；
- 只有 5% sparsity 才生效：
  只能称“大而分布式 weight set”，不能称少数关键权重；
- 三种 refusal-aware score 都不优于随机/Wanda：
  N7 的分布式结论推广到 weight level。

默认路线 B 只报告 val，不自动跑 harmful_test。只有其通过预注册标准并在路线 A
test 前完成选择，才允许作为第二个独立 final；否则避免重复使用 test。

## 9. 产物、断点与数据结构

计划新增：

~~~text
src/ttsafety/weight_edit.py
scripts/validate_weight_ortho.py
scripts/sweep_weight_ortho.py
scripts/run_weight_final.py
scripts/score_refusal_weights.py
scripts/sweep_weight_prune.py
tests/test_weight_edit.py
tests/test_weight_prune.py

results/weight_edit_baseline.json
results/sweep_weight_ortho.json
results/weight_ortho_final.json
results/sweep_weight_prune.json
results/weight_ortho_pareto.png
results/weight_ortho_components.png
results/weight_prune_pareto.png

data/weight_scores/taylor_*.pt
data/weight_scores/wanda_*.pt
data/weight_scores/edge_*.pt
data/weight_edits/best_ortho_config.pt
data/samples_weight_ortho_final.jsonl
docs/results-weight-level-refusal-editing.md
~~~

JSON cell 至少包含：

- config：source layer、destination layers、components、λ、variant；
- metrics：harmful/harmless refusal、PPL、KL、agreement、quality flags；
- edit_stats：矩阵数、参数数、relative delta norm、direction energy；
- provenance：模型、torch、transformers、CUDA、GPU、数据 hash；
- status：complete/failed，错误 cell 可重试但不能静默覆盖。

大 tensor 中间产物放 data/，生成文本放 data/，均沿用 gitignore；results/ 只放
汇总 JSON/PNG，不保存完整模型权重。

## 10. 预计 GPU/存储开销

RTX 5090 32GB，bf16 Llama-3.2-3B：

| 阶段 | 预计 GPU 时间 |
|---|---:|
| W0–W1 | 5–10 分钟 |
| W2a pilot | 5–10 分钟 |
| W2b 48-cell 主网格 | 30–60 分钟 |
| W3 controls/robustness | 20–40 分钟 |
| W4 final | 5–10 分钟 |
| W5 Taylor score | 20–45 分钟 |
| W5 pruning grid | 30–60 分钟 |

主线 W0–W4 预计 1–2 小时；含路线 B 共 2–3 小时。

存储：

- 不导出完整模型；
- hook sweep 不产生权重副本；
- Taylor/Wanda 分数按层保存 fp16/bf16，预计数 GB 以内；
- 若实际超过 8GB，改为只保存 top candidate indices/values，不扩大磁盘占用。

所有 GPU 命令继续使用：

~~~bash
flock -w 14400 .gpu.lock uv run python ...
~~~

单次命令按阶段/组件/scope 分块；每块完成后写 JSON，便于中断续跑。

## 11. 计划中的复现命令

以下命令只定义未来接口，本计划批准并实现后才执行：

~~~bash
# W0/W1
flock -w 14400 .gpu.lock uv run pytest tests/test_weight_edit.py
flock -w 14400 .gpu.lock uv run python scripts/validate_weight_ortho.py

# W2a
flock -w 14400 .gpu.lock uv run python scripts/sweep_weight_ortho.py --stage pilot

# W2b（按 component 分块）
flock -w 14400 .gpu.lock uv run python scripts/sweep_weight_ortho.py --stage main --components mlp
flock -w 14400 .gpu.lock uv run python scripts/sweep_weight_ortho.py --stage main --components attn
flock -w 14400 .gpu.lock uv run python scripts/sweep_weight_ortho.py --stage main --components both
uv run python scripts/sweep_weight_ortho.py --finalize

# W3/W4
flock -w 14400 .gpu.lock uv run python scripts/sweep_weight_ortho.py --stage controls
uv run python scripts/sweep_weight_ortho.py --finalize
flock -w 14400 .gpu.lock uv run python scripts/run_weight_final.py

# W5
flock -w 14400 .gpu.lock uv run python scripts/score_refusal_weights.py --score taylor
flock -w 14400 .gpu.lock uv run python scripts/score_refusal_weights.py --score wanda
flock -w 14400 .gpu.lock uv run python scripts/score_refusal_weights.py --score edge
flock -w 14400 .gpu.lock uv run python scripts/sweep_weight_prune.py --rule ratio
flock -w 14400 .gpu.lock uv run python scripts/sweep_weight_prune.py --rule taylor,edge
flock -w 14400 .gpu.lock uv run python scripts/sweep_weight_prune.py --rule random,wanda
uv run python scripts/sweep_weight_prune.py --finalize

# 全量回归
flock -w 14400 .gpu.lock uv run pytest
~~~

## 12. 停止条件与决策树

1. W1 等价性或 bit-exact restore 不通过：
   停止，不进入任何 GPU sweep。
2. W2a 三个 source 全部在 λ=0.5 即 PPL>10%：
   降低 λ 到 0.1/0.25，检查实现；仍异常则判方法不适配。
3. W2b 出现硬约束内配置：
   进入 W3 controls，再锁定 W4。
4. W2b 只有 relaxed 配置：
   完成 W3 诊断，先汇报，是否跑 test 由用户决定。
5. W2b 完全阴性：
   不扩展无界网格；进入 W5 检验“单 weight 稀疏集合”。
6. W5 也只在高损伤区有效：
   结束 weight-level 路线，结论为拒绝在 neuron 与 weight 两个自然基底下都
   不稀疏，但可由低维 activation direction 控制。
7. 任一阶段出现 test 数据泄漏、结果文件不完整、乱码率>1% 或 baseline 漂移：
   停止并修复，不能用后续结果掩盖。

## 13. 安全与研究边界

- 仅在本地开源模型上研究，不针对在线服务；
- harmful generations 只写入已 gitignore 的 data/，不外发、不提交；
- 不发布或上传被永久移除安全拒绝的完整模型；
- 本阶段默认只做可逆 hook sweep；权重物化仅用于等价性验证和最终小批评估；
- 每个里程碑完成后先汇报，再进入下一阶段。

## 14. 最终要回答的问题

实验结束时必须给出清晰的四项判定：

1. rank-one weight orthogonalization 是否主导 activation steering 或 N7 pruning
   的 Pareto 前沿？
2. 拒绝方向主要由 MLP、attention，还是两者协同写入？
3. 是否存在 sparsity≤1% 且副作用受控的 refusal-specific weight set？
4. 当前结果更支持：
   - “拒绝由稀疏安全权重实现”；还是
   - “拒绝是分布式权重计算形成的低维 activation direction”？
