# Plan: BLADE-G —— 在 BLADE 分数里加入通用重要性（generalizability）项

> 状态：**计划，待执行**。2026-09-02 在 LLMPrune 会话中起草，经 Codex(gpt-5.6-sol, high) 两轮评审后定稿；
> 评审意见已并入正文（标注为「Codex 修正」处是被驳回或改写的原始设想）。
> 本文自包含：执行者只需本文 + 本仓库代码。

## 0. 一句话与两个目标

BLADE 的权重分数 `s_ij = [(r_ℓ)_i · W_ij · (Δμ_W)_j]_+` 只回答"这个权重把行为对比路由进残差多少"；
能力保护完全在**分数之外**（ELS 的 ppl 预算 β + 每矩阵 10% 上限）。本计划给分数加一个
**逐权重的通用重要性项** `Q_ij`——该权重在 **C4（通用文本）** 上引起的、下游 RMSNorm 读者看得到的
期望平方扰动——得到

```
S_ij = [c_ij]_+ − λ·Q_ij        （主形式；λ 在开发集上标定；S_ij ≤ 0 的权重在排序阶段被过滤）
```

两个目标，优先级不同：

- **G1（主）**：等行为删除下更低的通用损伤（WikiText ΔNLL、XSTest-safe 过度拒绝、prompt KL）。
  `Q` 是**附带损伤的代理**，这是它有理由改善的东西。
- **G2（探索性）**：OOD/in-dist 迁移比更接近 1。动机是 sycophancy 负结果里"随机中层扰动匹配有原则选择"
  的混淆——若指标下降有一部分来自通用损伤，而通用损伤不迁移，则扣掉它应提高迁移比。
  **Codex 修正**：`Q` 没有强理由改善 OOD 迁移，G2 必须写成探索性假设，不能当主张。

## 1. 分数定义

### 1.1 分子（阶段 1 不改）
`c_ij = (r_ℓ)_i · W_ij · (Δμ_W)_j`，即 `ttsafety.sycophancy.score_edges` 在 `clamp_min_(0)` 之前的量。

### 1.2 分母：C4 上的通用重要性 `Q_ij`
含义：删掉 `W_ij` 后，**紧接该写入者之后的 RMSNorm 读者**看到的期望平方扰动（一阶度量的精确对角）。

读者映射（已核对 HF Llama-3.2 的 decoder 顺序，Codex 确认）：
- `layers[ℓ].self_attn.o_proj` → 读者 `layers[ℓ].post_attention_layernorm`；
- `layers[ℓ].mlp.down_proj` → 读者 `layers[ℓ+1].input_layernorm`，最后一层为 `model.norm`。

设读者 norm 的输入为 `h_t`（写入后的残差），`γ` 为读者 norm 权重，`m` = hidden size，
`r_t = sqrt(mean(h_t²) + ε)`，`q_t = h_t /(√m · r_t)`。四个估计器（**按成本递增**）：

| 记号 | 公式 | 说明 |
|---|---|---|
| `g0` | `W_ij² · E_C4[x_{t,j}²]` | Wanda 式，无输出侧几何 |
| `g1mean` | `W_ij² · E[a_{t,i}] · E[x_{t,j}²]` | 可分近似，丢掉协方差 |
| **`g1scalar`（主竞争者）** | `W_ij² · γ_i² · E_C4[ x_{t,j}² / r_t² ]` | 保留**逐 token 范数协方差**（`1/r_t²` 与激活幅度相关），成本与 g0 同级 |
| `g1`（精确对角） | `W_ij² · E_C4[ x_{t,j}² · a_{t,i} ]`，`a_{t,i} = [γ_i²(1−2q_{t,i}²) + q_{t,i}² Σ_k γ_k² q_{t,k}²]/r_t²` | 每写入者一次 `Aᵀ@X²`；坐标修正项量级 `O(1/m)` |

**Codex 修正（成本）**：精确 g1 的算术量 ≈ `n_tokens × 写入者总参数量`；3B 模型 × 262k token 是
数百万亿次运算，远超原估计。因此：**先做估计器 pilot（§3.0），再决定是否值得跑 g1**。

自检：ε=0、γ=1 时 `a_{t,i} = (1−q_{t,i}²)/r_t²`，且 `Σ_i a_{t,i} = (m−1)/r_t²`。

### 1.3 合成、弃权与放大
- 主形式 `S = [c]_+ − λQ`（拉格朗日）。诊断形式 `S' = [c]_+/sqrt(Q+τ)` 只用于分析，不作主排序
  （**Codex**：`Q` 是期望平方扰动，与可加二次预算相配的是拉格朗日式；比值是"每单位 RMS 扰动的效应"）。
- **弃权**：`S ≤ 0` 的权重在 `rank_weight_indices` 之前被**过滤掉**（不是排完序再看符号）。
  **Codex 修正**：一旦过滤，"被选权重中 S≤0 的比例"恒为 0，无意义；应报告
  **候选覆盖率**（过滤后剩余候选数 / 目标 K）与 **BLADE 选中集合被 BLADE-G 拒绝的比例**。
- 放大：`S^amp = (α−1)[c]_+ − λ(α−1)² Q`。
- **全程 fp32**：`S` 在 0 附近的符号决定弃权，fp16 会使其不稳（现有 `score_edges` 返回 fp16，
  新函数不要沿用）。

## 2. 实现

### 2.1 新文件 `src/ttsafety/generic_importance.py`
```python
@torch.no_grad()
def collect_c4_generic_importance(model, tokenizer, layers, components, *,
                                  text, seqlen=2048, batch_size=4,
                                  mode="g1scalar",          # g0|g1mean|g1scalar|g1
                                  ) -> dict[str, torch.Tensor]:  # {writer: Q fp32 CPU}
```
要点：
- 文本来自 `ttsafety.eval.load_c4_text`；**按 §3 的划分只用 `C4-Q` 那一份**。
- 每个写入者挂 `register_forward_pre_hook` 取输入 `x_t`；其**读者 norm** 也挂 pre-hook 取 `h_t`。
  同一次前向内两者到齐后累积；**注意多个写入者可能共享读者模块，需按写入者名分别缓冲**。
- `g1`：`acc += Aᵀ @ X2`（fp32，`A=(B·T,out)`、`X2=(B·T,in)`）；`g1scalar`：`acc += (1/r_t²) 加权的 X2` 后再乘 `γ_i²`（外积可分，无需 `[out×in]` GEMM）。
- **必须**：`use_cache=False`；hook 在 `finally` 里移除；**排除 padding token**（用 attention_mask）
  与文档拼接边界；断言 `Linear.weight.shape == (out, in)`；结果 key 与 `score_edges` 完全一致。
- 结束：`Q = W² ⊙ acc / n_tokens`。

### 2.2 `src/ttsafety/sycophancy.py` 新增（不改旧函数）
```python
def score_edges_g(model, directions, mu_s, mu_n, layers, components, *,
                  Q, lam, form="lagrange", tau=1e-8, abstain=True):
    """S = relu(c) - lam*Q（fp32）；abstain=True 时把 S<=0 置为 -inf 以便排序阶段过滤。"""
```
`rank_weight_indices` / `selection_from_ranking` / `pruned_weights` / `solo_layer_pool` /
`bestfirst_layers` **全部原样复用**：给后两者加一个 `score_fn` 参数（默认 `score_edges`），
**ELS 的每一次 `Prune(S,ρ)` 与最终排序必须用同一个分数**，否则 ELS 测的是另一种干预。

### 2.3 基线与对照（复用现有代码）
`score_edges`（BLADE）；`matrixwise_set_difference(safety=c⁺, utility=Q)`（Wei 式，`utility_fraction ∈ {0.01,0.05,0.1}`）；
`random_scores_like`（等数量等矩阵随机）；**Q 分位匹配的随机**（在与所选集合相同的 Q 分位桶内随机抽等量）；
**仅 Q 最低**（`S=−Q`，通用损伤对照，不应移除行为）；**置换 `r_ℓ` 坐标**。

## 3. 数据划分（防泄漏；Codex 修正后为三份互斥资源）

| 资源 | 用途 | 说明 |
|---|---|---|
| `C4-Q` | 估计 `Q` | `load_c4_text` 的第 1 段 |
| `C4-dev` + 行为 dev | 选 `λ` 与权重数 K | 第 2 段；行为 dev 独立于 test |
| `C4-budget` | β 可行性检验（不参与选择） | 第 3 段 |
| WikiText test | **仅最终报告** | 现状不变 |
| benign-dev（新） | 若 XSTest 参与保留/丢弃规则，必须另切一份良性开发集 | 否则 XSTest 不再是干净测试集 |

- refusal：方向/矩来自 AdvBench train（`extract_refusal_direction`，现有 `data/directions/refusal_llama32_3b_instruct.pt`）；
  in-dist 用 AdvBench val/test；**OOD = HarmBench 完全留出**（沿用 `scripts/refusal_ood_transfer.py`）。
- sycophancy：A/B 与开放式按主题/模板划分留出单元。
- **λ 用 ΔNLL / log-ppl 标定，不用原始 Δppl**（NLL 才是可加量）；λ 网格用无量纲 scale
  `median(c⁺)/median(Q)` 乘 `{0, 1e-3, 1e-2, 1e-1, 1, 10, 100}`（**要超过 1**，中位数匹配说明不了尾部）。
- 次要参数化（可选，便于解释）：`max Σc s.t. ΣQ ≤ B, |A| ≤ K`；但 `ΣQ` 仍是对角局部代理，`B` 仍需对 NLL 经验标定。

### 3.0 估计器 pilot（**先做，1–2 小时**）
在 3–4 个层上用 8k–16k token 比较 g0 / g1mean / g1scalar / g1：top-k 重叠随 token 数的收敛、
四者的秩相关、协方差项量级、墙钟与显存。**不要在 pilot 之前把 3 seed × 262k token 投给精确 g1。**

## 4. 实验

固定：Llama-3.2-3B-Instruct；ρ ∈ {0.002, 0.005}；3 个 seed（方向/矩/C4 子采样重抽）；
初期关闭层搜索（用现有 refusal L*=[12]）；解码与判别沿用现有脚本。

### E1 主实验（refusal，L* 固定）
分数集合：BLADE、BLADE-G(g0 / g1scalar / g1)、Wei 集合差、随机、Q 分位匹配随机、仅 Q 最低、置换 r。
指标：in-dist 拒绝率、OOD(HarmBench) 拒绝率、迁移比、WikiText ΔNLL、`C4-budget` ΔNLL、
XSTest-safe 服从率、`prompt_kl`、下游 zero-shot 子集。

**匹配方式（Codex 修正，重要）**：不用插值。**预先设定行为目标**（如 in-dist 拒绝率 ≤ 0.15），
对每个分数**二分搜索全局排序的前缀长度**达到该目标，**实际评测那个模型**；同时报告
"等权重数 K"与"等行为"两条前沿。插值只作敏感性分析。OOD 全程不参与任何选择，
**同时报告原始 OOD 变化与迁移比**（in-dist 分母小时比值不稳）。
统计：**配对的样本级/分层 bootstrap**，展示全部 seed 点；n=3 不足以做 seed 总体显著性主张。

### E2 放大（可裁剪）
`S^amp` vs BLADE 排序，α ∈ {1.3, 1.5}，沿用 `scripts/blade_amplify_refusal_ood.py` 的 OOD/XSTest/ppl。

### E3 sycophancy（负结果复查，可裁剪）
同 E1 分数集合；看 BLADE-G 是否改变"随机匹配有原则选择"的格局，以及**弃权比例**——
若绝大多数权重 `S ≤ 0`，这是"该行为在此操作化下被通用损伤主导"的诊断。

### E4 消融（可裁剪）
λ 网格；`form=ratio`；估计器四档；`P=I` vs `P=I−rrᵀ`（预期后者 XSTest 更差：过度拒绝正沿 r）。

### E5 C4 重构修复（最先裁剪）
删除后在 `C4-Q` 上做固定 mask 的最小二乘/OBS 重构，看通用功能恢复多少、行为是否被"补回"。

### E6 **跨层选择：能否去掉 ELS**（与 E1 同等优先）

现状关键事实：`rank_weight_indices` **本来就是跨所有写入者矩阵的一个全局排序**——
缺的不是机器，而是**分数的跨层可比性**；ELS 存在正是因为分数不可跨层比。

**Codex 修正（必须写进结论口径）**：读者归一化**必要但不充分**。它消除了残差范数随深度增长
这一主要尺度失配，但**不**使层局部分数在模型输出处可比。仍不可比的有：从该读者到最终 logits 的
增益/衰减、各层独立提取的 `r_ℓ` 的语义质量与归一化、层内 attention/MLP 的位置、
不同深度的行为边际敏感度与通用损失曲率、下游抵消/放大/自修复。

**分子必须改成读者位点的对比（Codex 修正）**：
```
c^rdr_ij = W_ij ( E_{behaviour+}[u_{t,i} x_{t,j}] − E_{behaviour−}[u_{t,i} x_{t,j}] ),  u_t = J_tᵀ r^rdr_ℓ
```
其中 `r^rdr_ℓ` **必须在该读者位点重新提取**：现有方向取自 block 输出（pre-norm，且对 o_proj 而言
是在 MLP **之后**），而 o_proj 的直接读者 `post_attention_layernorm` 在 MLP **之前**——
直接套用会张冠李戴。

变体表：

| 变体 | 层范围 | 分数 | 备注 |
|---|---|---|---|
| A | ELS 的 L* | BLADE | 基线 |
| B | ELS 的 L* | E1 的赢家 | |
| **C** | 全部层 | `[c^rdr]_+ − λQ`，κ=1 | **真正无测量**的检验 |
| **D** | 全部层 | 同上 + 逐层标定 κ | 应命名为**一次性标定的全局排序**，不是"ELS-free" |
| E | 全部层 | BLADE 原分数 | 不可比性对照（预期被浅层主导、损伤爆掉） |
| F | 全部层 | `[c^rdr]_+`（无 Q） | 检验 Q 能否单独替代 `solo_layer_pool` |

**κ 标定（Codex 修正）**：不要用行为**率**在 ρ=1e-4 下探针——多数层不会跨阈值，κ 会是 0 或极不稳。
改用**连续边际**目标：最终层的拒绝方向投影，或拒绝-vs-服从的 teacher-forced 对数似然边际。
```
κ^beh_{ℓ,c} = Δ(连续边际) / Σ c^rdr ,   κ^util_{ℓ,c} = ΔNLL / Σ Q
```
**对 o_proj 与 down_proj 分别标定**（两个不同读者位点），用两个探针幅度，向深度平滑曲线做收缩。
更干净的 forward-only 替代：在每个读者位点沿其行为方向做小幅**激活注入**，测最终连续边际——
这样估计的是传输本身，不与 top-k mask 质量混淆。

**ρ 的可比性（Codex）**：不要在 L* 与全部层上用同一个 ρ（绝对权重数不同）。用**固定绝对 K**
与**行为匹配前缀**两种对比。保留每矩阵上限，但它不归一化分数分布，也不能阻止浅层吃掉大量预算。

**关于 "Q 会自动排除 L0"（Codex 驳回为弱预测）**：局部 norm 能量 ≠ 下游能力关键性；
RMSNorm 甚至可能拉平早期层的局部能量，而早期扰动有更多层去放大。L0 仍可能因 `c/Q` 尾部大、
候选多而在极值竞争中胜出。**当作待检验假设（F 对 C），不要当作定义推论去掉 `solo_layer_pool`。**

**β 的角色**：固定排序后做 β 回退（对前缀长度二分）——这是一维可行性搜索，不是层搜索；
**回退时只动前缀，不要同时动 λ**（否则变成二维重调参）。

**E6 必须记录的逐样本诊断**（区分四种失败）：
1. 预测的局部位移 `Σc`；2. 编辑后写入者处**实际**的读者投影变化；3. 传输到最终的连续边际变化；
4. 生成的二值行为。
→ 位移没实现 = 分数/交叉项失败；位移实现但最终边际恢复 = 下游抵消/hydra；
边际动了但行为率不动 = 阈值/饱和；单层有效而联合无效 = 跨层交互。
画边际及其变化的**分布**（不只是均值），并画逐层的"编辑−dense 残差投影"轨迹看恢复从哪层开始。

**成本口径（Codex）**：报告 dense token 数（方向/矩/Q）、编辑模型的行为评测次数、
编辑模型的 C4/NLL 评测次数、κ 探针次数、β 回退次数、最终报告评测次数（与选择成本分开），
以及 GPU 小时与评测 token 数。ELS 一侧报告 `solo_layer_pool` 与 `bestfirst_layers`
**实际评估的 prune context 次数**，不要只写 O(L²)。

**结论口径**：只有 **C 成功**才能说"分数使 ELS 不必要"；若只有 **D 成功**，正确说法是
**"组合式 ELS 可被线性代价的传输标定 + 全局排序取代"**。若 C/D 明显更差，
这是"分数不可替代联合实测"的负结果，同样值得写。

## 5. 判决规则
- 保留 `Q`：跨 seed 与两个 ρ，在**实际达成的**等行为删除下，WikiText ΔNLL 或 XSTest 显著更好
  （配对样本级 bootstrap 95% CI 不含 0），且 OOD 行为删除不更差。
- 若"仅 Q 最低"也移除了行为 → 该层的 BLADE 结果主要靠通用损伤，写入报告。
- g0 ≈ g1scalar ≈ g1 → 用 g0；g1scalar 明显更好 → 逐 token 范数协方差有效；g1 再明显更好 → 坐标几何有效。

## 6. 两 GPU·天的裁剪顺序（Codex 建议）
先做 §3.0 pilot → E1 → E6。裁剪顺序：E5 → E2 → E4 大部 → 必要时 E3。
两天内保留：BLADE、g0、g1scalar、一次减 token 的精确 g1 抽查、Q 分位匹配随机、ELS vs 全局对比。

## 7. 交付物
- 代码：`src/ttsafety/generic_importance.py`、`sycophancy.score_edges_g`、
  `scripts/blade_g_refusal.py`（E1/E2）、`scripts/blade_g_crosslayer.py`（E6）、
  `scripts/blade_g_sycophancy.py`（E3）。
- 结果：`results/blade_g_pilot.json`、`results/blade_g_refusal.json`、`results/blade_g_crosslayer.json`、
  `results/blade_g_sycophancy.json`。
- 图：`figures/blade_g_pareto.{png,pdf}`（行为删除 vs ΔNLL / XSTest 前沿）、
  `figures/blade_g_layerdist.{png,pdf}`（所选权重的层分布 vs ELS 的 L*）——house style 见 `[[figure-house-style]]`。
- 文档：`docs/results-blade-g.md`。

## 8. 措辞边界
可宣称的是**合取**："forward-only 的残差写入者带符号贡献，与同一权重在通用文本上的
RMSNorm 切空间扰动做拉格朗日标量化"。
禁用："first generalisation-aware pruning"、"guarantees OOD transfer / capability preservation"、
"disentangles behavior weights"、"exact causal effect"（"exact" 只可用于"所声明的线性化读者标量的可加分解"）。
最近先行工作：Orgad et al. 2026（arXiv 2604.09544，带符号 `W∇_W L` + 良性集合差 + 跨伤害类型）、
Wei et al. 2024（集合差，本仓库已实现）、Group-Robust LLM Pruning（2608.02940，组稳健分配）、
GPrune-LLM（2603.13418，跨数据集排名稳定，神经元级）。
