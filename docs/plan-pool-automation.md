# Plan: 目标池自动化与组件覆盖分析（改进方向 #4）

> ⚠️ **历史存档(标注于 2026-08-27):本文记录较早阶段的计划/方法/结果,可能与当前代码与结论脱节。**
> 方法线现已定名 **BLADE** 并引入 best-first ELS(有效层选择);当前定稿的方法与实验结果请以
> [`blade-method.md`](blade-method.md)、[`blade-algorithm.md`](blade-algorithm.md)、
> [`blade-experiment-report.md`](blade-experiment-report.md) 为准。**下文内容保留原貌,未作修订。**

> 状态：解释性 plan，待批准后执行。
> 背景：当前所有 weight 级方法的剪枝池是手工固定的 L7–L18 `down_proj + o_proj`
> （415M 权重），依据是 steering 的有效层窗口。这引入两个人为先验：(a) 层范围；
> (b) 只剪 writer（写残差流的矩阵），不剪 reader（q/k/v/up/gate，读残差流的矩阵）。

## 1. 为什么这是个问题

- 层窗口 L7–L18 来自 activation steering 实验（哪里注入有效）。但"steering 有效的层"
  与"承载可剪拒绝权重的层"不必相同：steering 测的是表征对扰动的敏感性，pruning 测的是
  权重对行为的因果贡献。手工窗口可能漏掉窗口外的高分权重（我们没有全池分数的对照）。
- writer-only 的限制同样来自方向投影公式：edge 分数需要一个 destination 方向 r，
  天然只适用于写残差流的矩阵。reader 矩阵（如 `down_proj` 上游的 `up_proj/gate_proj`、
  attention 的 `q/k/v_proj`）被整个排除，但它们同样参与拒绝行为（N 系列显示拒绝相关的
  MLP 激活差异是真实存在的，那些激活正是 up/gate 算出来的）。

## 2. 提议的实验

### 2a. 池外检验（便宜，先做）

把 edge 分数推广到**全部 28 层**的 writer（L0–L27 的 down_proj + o_proj，公式不变，
r 取对应层的 refusal direction——已有全层向量），回答两个问题：

1. L7–L18 之外的 writer 是否含有可进全局 top-K 的高分权重？占比多少？
2. 若有，用全层池重新做 0.01%/0.05% 档剪枝，与现有结果对比 refusal/ppl/KL。

若池外高分权重占比 <1%，手工窗口事后被验证合理（这本身是个可报告的结果）；
若不可忽略，说明当前结果低估了方法效果或池定义需要修正。

### 2b. reader 侧扩展（需要新分数，第二阶段）

reader 矩阵没有 destination 方向可用。候选设计：

- **能量版**：`s = max(Δa_in 方向投影贡献...)` 不适用——reader 的"输出"不进残差流。
  可行定义：对 `up_proj/gate_proj`，其输出即 MLP 神经元激活本身，可以用 N7 的
  actdiff t 统计量作为该神经元（= 该输出行）的分数，行内按 |W·Δ输入| 分配；
- 或直接用 unsigned 的 actdiff 行分数做 per-row 候选，再用 sign 规则过滤。

这部分有真实的设计不确定性，建议先跑 2a，2a 的结果会提示 reader 扩展是否值得。

## 3. 判定与产出

- 2a 产出：全层分数缓存、池外占比统计、（如有必要）全层池的对照 sweep 行；
- 判定：池外 top-K 占比 <1% → 现池定义成立，文档记录；≥1% → 重新评估主结果；
- 结果写入 `results/pool_audit.json`，结论追加到 weight-editing 结果文档。
