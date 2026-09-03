# 拒绝方向 steering 实验结果（M0–M4）

> ⚠️ **历史存档(标注于 2026-08-27):本文记录较早阶段的计划/方法/结果,可能与当前代码与结论脱节。**
> 方法线现已定名 **BLADE** 并引入 best-first ELS(有效层选择);当前定稿的方法与实验结果请以
> [`blade-method.md`](blade-method.md)、[`blade-algorithm.md`](blade-algorithm.md)、
> [`blade-experiment-report.md`](blade-experiment-report.md) 为准。**下文内容保留原貌,未作修订。**

> 对应计划：`docs/plan-caa-refusal-steering.md`。本文汇总 M2（方向提取+验证）、
> M3（sweep+最终测试）、M4（CAA 对比对对照）的全部结果与复现命令。
> 所有数字均来自 `results/` 下的 JSON 产物（含 env 溯源块）。

## 1. 方法概述与环境

- **主方法（mean-diff，Arditi et al. 2024 风格）**：对每条指令做 chat template
  包装，取指令最后一个 token 位置的残差流激活，
  `v_l = mean(act(有害)) − mean(act(无害))`；jailbreak = 前向时
  `h ← h − |α|·v̂_l`（全部 token 位置，v̂ 为单位向量，raw 模式）。
- **对照方法（严格 CAA 对比对）**：对有害指令构造（拒绝回复, 服从回复）配对，
  仅在回复 token 区间取激活均值，`v_caa_l = mean(act(拒绝) − act(服从))`。
- **模型**：`meta-llama/Llama-3.2-3B-Instruct`（bf16，28 层，hidden 3072，本地 HF 缓存）。
- **环境**：torch 2.11.0+cu128，transformers 5.15.1，RTX 5090 32GB。
- **数据**：AdvBench 有害指令 520 条，seed=0 划分 256 train / 64 val / 200 test；
  Alpaca 无害指令 320 条（仅 input 为空、5–40 词、去重，seed=0 抽样）。
  统计见 `results/data_stats.json`（各 split 词数均值 ≈10–12）。
- **判定**：关键词拒绝分类器 `src/ttsafety/judge.py`（含 Unicode 撇号归一化）。

## 2. M2：方向提取与双向验证

提取产物：`data/directions/refusal_llama32_3b_instruct.pt`（28 层，fp32）。
层间余弦热图：`results/direction_cosine_heatmap.png`——早层（0–10）与中后层
（13–27）各自聚成高相关块，两块之间近似正交。

**符号自检**（`results/validate_direction.json` → `sign_check`）：v̂ 上有害指令
投影显著高于无害指令（如 L18：+6.73 vs −4.99），确认 v = 有害 − 无害 即
拒绝方向，jailbreak 符号为 α<0（`jailbreak_sign`：L10/14/18 为 −1）。

**尺度说明**：中间层残差流 L2 范数约 8–24（32 条无害探针），因此单位向量
α=1–2 仅占残差 6–13%，太弱；有效区间在 α≈4–8。M2 初版用 α∈{1,2} 未见效应，
扩展网格后双向效应均显著。

**双向效应表**（基线：harmful_val 拒绝率 1.000，harmless 0.006）：

| 格子 | harmless +α（期望↑） | harmless −α | harmful_val −α（期望↓） | harmful_val +α |
|---|---|---|---|---|
| L10 α1 | .013 | .000 | 1.000 | 1.000 |
| L10 α2 | .056 | .000 | .906 | 1.000 |
| L10 α4 | **.934** | .000 | .484 | 1.000 |
| L10 α8 | **1.000** | .000 | **.000** | .891 |
| L14 α4 | .281 | .000 | .219 | 1.000 |
| L14 α8 | .903 | .000 | **.000** | .938 |
| L18 α4 | .016 | .000 | .875 | 1.000 |
| L18 α8 | .122 | .000 | **.016** | .688 |
| L22（全部 α） | ≈.006–.016 | ≈.000–.006 | 1.000 | 1.000 |

M2 验收（两个方向 |Δ| > 20pp）**通过**且余量巨大。L22 完全无效（方向虽能
分离两类激活，但 steering 不改变行为）。

## 3. M3：val sweep、选参与最终测试

Sweep：`results/sweep_steer.json` + `results/sweep_curves.png`，
layers {8,10,12,14,16,18,20} × |α| {2,4,6,8}，jailbreak 符号，逐格记录
harmful_val 拒绝率、harmless 过拒绝率、wikitext-2 ppl（基线 13.06）。

要点：

- **过拒绝几乎为零**：所有格子 harmless 拒绝率 ≤0.003，方向高度拒绝特异。
- **ppl 是约束瓶颈**：拒绝率随 |α| 快速下降，但 ppl 同步恶化
  （如 L14 α6：val 拒绝 1.6% 但 ppl +26.4%；L8 α8：0% 但 +139.8%）。
- **选参规则**：harmless ≤5% 且 ppl 退化 ≤5% 的可行格中取 val 拒绝率最低者
  → **L8 α=−2.0**（val 拒绝 0.766，harmless 0.000，ppl +2.44%，未越界）。

**最终 test（200 条 held-out，`results/final_test.json`）**：

| 指标 | baseline | steered (L8, α=−2) |
|---|---|---|
| test 服从率 | 1.0% | **33.0%** |
| test 拒绝率 | 99.0% | 67.0% |
| harmless 拒绝率 | 0.6% | **0.0%** |
| wikitext-2 ppl | 13.06 | 13.38（**+2.44%**） |

全部 200 条生成见 `data/samples_final.jsonl`；对比图 `results/final_curves.png`。

## 3.5 M3b：多层 steering 对照

**假设**：把干预分摊到多个层、每层用更小的 |α|，可能以更低的 ppl 代价达到
同等 jailbreak 成功率。脚本 `scripts/sweep_multilayer.py`，结果
`results/sweep_multilayer.json` + `results/multilayer_pareto.png`
（val 集，jailbreak 符号，raw 单位向量模式）。

| 组合 | harmful_val 拒绝 | harmless 拒绝 | ppl Δ |
|---|---|---|---|
| {8,10,12,14} × α−0.5 | .891 | .000 | +0.73% |
| {8,10,12,14} × α−1 | .375 | .000 | +4.02% |
| {8,10,12,14} × α−2 | **.016** | .000 | +20.60% |
| {10,14,18} × α−1 | .875 | .000 | +2.08% |
| {10,14,18} × α−2 | .391 | .000 | +9.06% |
| {12,14,16} × α−1 | .891 | .003 | +3.05% |
| {12,14,16} × α−2 | .219 | .000 | +11.72% |
| {8,14,20} × α−1 | .828 | .000 | +1.94% |
| {8,14,20} × α−2 | .297 | .000 | +8.03% |
| {10,12,14,16,18} × α−1 | .562 | .000 | +5.47% |

**结论：假设不成立（未主导单层前沿）**。判定标准：达到拒绝率 ≤5% 的多层组合
中 ppl 最低者（{8,10,12,14}×α−2：1.6% @ +20.6%）并未低于达到同等拒绝率的
最优单层格（L12 α−6：3.1% @ +18.9%；L14 α−6：1.6% @ +26.4%）——与前者的
前沿相比无改善，仅严格支配了 L14 α−6 这一个格子（同拒绝率、ppl 更低）。
Pareto 图显示多层组合在中等拒绝率区间（0.2–0.4）的 ppl 略优于单层前沿
（如 {10,14,18}×α−2：0.391 @ +9.1%，优于同拒绝率附近的单层格），但在
低拒绝率端两种方案收敛到相同的 ~+19–21% ppl 代价。按既定规则，多层方案
未明显主导，故未在 test 集上重跑最终评估。多层 steering 的 ppl 代价近似
随各层 |α| 总量线性累加，没有观察到"分摊红利"。

## 4. M4：CAA 对比对对照

- **配对构造**（`data/caa_pairs.jsonl`）：harmful_train 256 条上，默认模型
  确认拒绝 248 条，L14 α=−8 下确认"优质服从"（非拒绝 + ≥20 词 + 去重词比
  ≥0.3）255 条，交集 **247 对**。抽检服从样本为连贯的指南式文本，无需回退
  到 α=−6。
- **方向提取**：`data/directions/caa_llama32_3b_instruct.pt`，回复 token 区间
  均值差；in-sample 符号自检全部通过（拒绝回复投影 > 服从回复投影）。
- **与 mean-diff 的逐层余弦**（`results/caa_contrast.json`）：早层 ≈0（L0–4
  ≤0.05），中层爬升（L12 0.343、L14 0.405、L16 0.463），峰值在中后层
  **L23 0.503**、L25 0.489、L24 0.470，末层回落（L27 0.214）。两种方法在
  中后层提取到的是同一族方向（中等强度一致）。
- **有效性 spot-check**（top 对齐层 16/18/20，harmful_val，与 sweep 中
  mean-diff 同格对比；图 `results/caa_vs_meandiff.png`）：

| 格子 | CAA 对比对 | mean-diff |
|---|---|---|
| L16 α−2 | 1.000 | .969 |
| L16 α−4 | .938 | .672 |
| L16 α−8 | **.844** | **.000** |
| L18 α−4 | .984 | .875 |
| L18 α−8 | 1.000 | .016 |
| L20 α−8 | 1.000 | .688 |

CAA 最优格（L16 α−8）val 拒绝率仍有 84.4%，harmless 拒绝 0.003。
**结论：在本设置下 mean-diff 方向明显优于 CAA 对比对方向**——同尺度单位向量
steering 下 mean-diff 在 L16 α−8 已达 0% 拒绝。可能原因：CAA 方向取自回复
区间均值，混合了回复内容语义，单位化后拒绝成分被稀释；而 mean-diff 取自指令
末 token，更"纯净"地编码拒绝决策。两者方向余弦仅中等（≤0.5），也印证了
两者捕获的信号并不完全相同。

## 5. 生成质量观察

- 选定格（L8 α=−2）无退化：拒绝样本是正常流利拒绝；服从样本（test 10 条
  抽检中 3 条）为连贯的 markdown 结构化回答，真实回应有害指令，但普遍带
  "假设/虚构"hedging 框架（"I can provide a hypothetical example…"）——
  模型是"软化后参与"而非无条件服从。
- 高 α（ppl 代价大的格子）出现明显退化：重复循环（"comprehensive and
  comprehensive and …"）、回复变短变空。
- 强 jailbreak 格（L14 α−8）的服从生成大多连贯（M4 质量过滤仅淘汰 1/256）。

## 6. 局限性

- **judge 是关键词法**：可能误判（如把安全转述算作服从；把含 "I can't" 的
  正常回答算作拒绝）。未做人评，仅抽检 10 条。
- 单一模型（Llama-3.2-3B-Instruct），未验证跨模型泛化。
- ppl 仅用 wikitext-2 50k token 窗口；未测下游任务能力。
- CAA 对照只做了单层 spot-check（3 层 × 3 α），未做完整 sweep。
- 关键词判定 + hedging 框架意味着"服从率"的语义需要人工解读。

## 7. 复现命令

```bash
# M0/M1：环境与数据
uv sync --extra dev
flock -w 14400 .gpu.lock uv run pytest                       # 18 项测试
uv run python scripts/prepare_data.py                        # AdvBench + Alpaca

# M2：方向提取 + 双向验证
flock -w 14400 .gpu.lock uv run python scripts/extract_direction.py
flock -w 14400 .gpu.lock uv run python scripts/validate_direction.py

# M3：sweep（可按层分块）→ 选参 → 最终测试
flock -w 14400 .gpu.lock uv run python scripts/sweep_steer.py --layers 8
flock -w 14400 .gpu.lock uv run python scripts/sweep_steer.py --layers 10,12
flock -w 14400 .gpu.lock uv run python scripts/sweep_steer.py --layers 14,16
flock -w 14400 .gpu.lock uv run python scripts/sweep_steer.py --layers 18,20
uv run python scripts/sweep_steer.py --finalize              # 选参 + 曲线（纯 CPU）
flock -w 14400 .gpu.lock uv run python scripts/run_final.py

# M3b：多层 steering 对照
flock -w 14400 .gpu.lock uv run python scripts/sweep_multilayer.py --only 0,1,2,3,4
flock -w 14400 .gpu.lock uv run python scripts/sweep_multilayer.py --only 5,6,7,8,9
uv run python scripts/sweep_multilayer.py --finalize         # Pareto + 图（纯 CPU）

# M4：CAA 对比对
flock -w 14400 .gpu.lock uv run python scripts/run_caa_contrast.py --stage pairs
flock -w 14400 .gpu.lock uv run python scripts/run_caa_contrast.py --stage extract
flock -w 14400 .gpu.lock uv run python scripts/run_caa_contrast.py --stage spotcheck
```

## 8. 产物清单

- `data/directions/refusal_llama32_3b_instruct.pt` — mean-diff 方向（28 层）
- `data/directions/caa_llama32_3b_instruct.pt` — CAA 对比对方向（28 层）
- `data/caa_pairs.jsonl` — 247 对（instruction, refusal, compliance）
- `data/samples_final.jsonl` — test 集 200 条 baseline/steered 生成
- `results/data_stats.json` — 数据统计
- `results/validate_direction.json` — M2 双向验证（含符号自检、jailbreak 符号）
- `results/direction_cosine_heatmap.png` — 层间余弦热图
- `results/sweep_steer.json` / `results/sweep_curves.png` — M3 sweep 与选参
- `results/final_test.json` / `results/final_curves.png` — M3 最终 test 结果
- `results/sweep_multilayer.json` / `results/multilayer_pareto.png` — M3b 多层对照
- `results/caa_contrast.json` / `results/caa_vs_meandiff.png` — M4 对照
