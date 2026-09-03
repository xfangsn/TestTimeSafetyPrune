# Plan: 基于 Activation Steering 的拒绝方向研究与 Jailbreak 实验

> ⚠️ **历史存档(标注于 2026-08-27):本文记录较早阶段的计划/方法/结果,可能与当前代码与结论脱节。**
> 方法线现已定名 **BLADE** 并引入 best-first ELS(有效层选择);当前定稿的方法与实验结果请以
> [`blade-method.md`](blade-method.md)、[`blade-algorithm.md`](blade-algorithm.md)、
> [`blade-experiment-report.md`](blade-experiment-report.md) 为准。**下文内容保留原貌,未作修订。**

> 状态：待批准。批准前不执行任何代码改动。
> 工作目录：`~/Projects/TestTimeSafetyPrune`（新建项目，当前为空仓库）
> 参考代码：`~/Projects/TestTimePrune`（复用其 hook/steering 设施的设计）

## 1. 目标

1. 构造「拒绝（refusal）vs 服从（compliance）」对比数据对；
2. 用 CAA（Contrastive Activation Addition, arXiv:2312.06681）风格的均值差方法，从残差流中提取**拒绝方向（refusal direction）**；
3. 通过在 forward 中注入 `-α·v̂`（即沿拒绝方向的反方向 steering），在 held-out 有害指令上提升服从率，验证 jailbreak 效果；
4. 量化副作用（无害指令过度拒绝、困惑度退化），给出 layer × scale 扫描结果。

## 2. 方法选型

两条路线，先简后繁：

- **主方法（mean-diff，Arditi et al. 2024 风格，最简洁）**：
  对每个 decoder layer `l`，取指令最后一个 token 位置的残差流激活，
  `v_l = mean(act(有害指令)) − mean(act(无害指令))`。
  该方向即拒绝方向。jailbreak = 前向时 `h ← h − α·v̂_l`（所有 token 位置）。
  优点：不需要任何模型生成，只要两组指令，一次前向即可。
- **对比方法（严格 CAA 对比对）**：
  对每条有害指令构造 `(prompt + 拒绝回复)` 与 `(prompt + 服从回复)` 配对，
  在回复 token 区间取激活均值做差。拒绝回复由模型默认行为生成；
  服从回复在拿到主方法的方向后、用 `-α·v̂` steering 自生成（或手工构造
  "Sure, here is ..." 前缀续写）。作为第二阶段验证方向的一致性（余弦相似度）
  与效果对比。

默认采用主方法产出实验结果，CAA 对比对方法作为对照实验。

## 3. 环境与技术约束

- 硬件：单卡 RTX 5090 32GB，torch 需 cu128（sm_120）。
- 包管理：uv（沿用 TestTimePrune 的 pyproject 配置模板：pytorch-cu128 index、
  transformers>=4.45、datasets、accelerate、numpy、matplotlib、pytest）。
- 主模型：`meta-llama/Llama-3.2-3B-Instruct`（本地 HF 缓存已有完整权重，bf16 约 6GB）。
- GPU 互斥：沿用 `flock -w 14400 .gpu.lock ...` 约定。
- 需要 `use_cache=True` 与 `model.generate()`（TestTimePrune 全局关掉了 cache，
  新项目的 loader 默认开启）。
- 所有结果落盘为 JSON（含 config + env/git hash 溯源块），沿用 TestTimePrune 约定。

## 4. 数据构造

- **有害指令**：AdvBench `harmful_behaviors.csv`
  （`https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv`，约 520 条）。
  划分：256 train（提方向）/ 64 val（选层选 α）/ 其余 test（最终报告）。
  若网络不可达则报错并请求用户提供本地副本，不静默降级。
- **无害指令**：`tatsu-lab/alpaca`（HF datasets）随机抽 320 条短指令，
  过滤掉含敏感词项；与有害集等长对齐 token 长度分布。
- **CAA 对比对（第二阶段）**：
  - 拒绝回复：对 train 有害指令用默认模型 greedy 生成，经拒绝关键词判定确认为拒绝的样本；
  - 服从回复：同批指令在 `-α·v̂` steering 下生成，取确认非拒绝的样本。
- 数据存放在 `data/`（gitignore），加载器写 `src/ttsafety/data.py`。

## 5. 实验流水线（src/ttsafety/ 模块划分）

| 模块 | 内容 |
|---|---|
| `models.py` | loader（bf16, cuda, use_cache=True, 本地缓存优先）+ chat template 工具 |
| `data.py` | AdvBench / Alpaca 加载、train/val/test 划分（固定 seed） |
| `hooks.py` | decoder layer 发现 + 残差流捕获/注入 hook（参考 ttprune 的 `steering.py`/`wanda.get_decoder_layers`，简化重写） |
| `extract.py` | mean-diff 方向提取（指令末 token 位置，全层单次前向，早停） |
| `steer.py` | steering context manager：`h ← h + α·v̂`（raw/relative 两种 mode，多层支持） |
| `generate.py` | `model.generate` 封装（greedy，max_new_tokens=128，batch） |
| `judge.py` | 拒绝判定：标准拒绝前缀/关键词表（"I'm sorry", "I cannot", ...），输出 per-sample 标签 |
| `eval.py` | 指标汇总：服从率/拒绝率、过度拒绝率、ppl 副作用 |

脚本（`scripts/`，均 `uv run python scripts/...`，GPU 脚本走 flock）：

1. `extract_direction.py` — 提取全层拒绝方向，存 `data/directions/refusal_{model}.pt`；
   输出层间余弦相似度矩阵图。
2. `validate_direction.py` — 方向有效性 sanity check：
   - 正向：无害指令 `+α·v̂` 应显著升高拒绝率（方向确实编码拒绝）；
   - 反向：val 有害指令 `-α·v̂` 应显著降低拒绝率。
3. `sweep_steer.py` — val 集上 layer × α 网格扫描（layers: 全层或 4 个深度代表层；
   α: 0, 0.5, 1, 1.5, 2, 3），指标 = 服从率 + 过度拒绝率 + ppl 副作用。
4. `run_final.py` — 用 val 选出的最优 (layer, α)，在 test 集出最终服从率，
   与 α=0 baseline 对比，存结果 JSON + 样例生成文本（人工抽查 10 条）。
5. `run_caa_contrast.py` —（第二阶段）严格 CAA 对比对提取方向，报告与主方法方向的
   余弦相似度，重复 sweep 的关键格子做效果对比。

## 6. 评估指标

- **主指标**：test 集服从率（1 − 拒绝率），拒绝判定用关键词表；
  报告中注明关键词法的局限（可能误判），并附 10 条生成样本人工核对。
- **副作用**：
  - 过度拒绝率：320 条无害指令在 steering 下的拒绝率（应接近 0）；
  - 困惑度：WikiText-2 子集 ppl，α=0 vs 最优 α（退化应 <5%）。
- **方向质量**：层间余弦相似度热图；主方法 vs CAA 对比对方向的余弦相似度。

## 7. 里程碑

- **M0 脚手架**：pyproject + venv + loader smoke test（模型加载、chat template、
  一次 generate 成功）。验收：`uv run pytest` 通过 smoke 测试。
- **M1 数据**：AdvBench + Alpaca 加载、划分、统计报告（条数、长度分布）。
- **M2 方向提取 + sanity check**：M2 验收 = validate_direction 的正反两向
  效应都显著（如 |Δ拒绝率| > 20pp）。
- **M3 sweep + 最终结果**：val 选参，test 出最终数字与图（服从率 vs α 曲线、
  层间热图），写入 `results/`。
- **M4 CAA 对比对对照**（可选第二阶段）。
- **M5 小结**：`docs/results-refusal-steering.md`，含复现命令。

每个里程碑跑完先汇报结果，再进入下一个。

## 8. 安全与边界

- 纯本地研究用途；生成的有害内容样例只存本地 `results/`，不进 git，不外发。
- `.gitignore` 排除 `data/`、`results/` 中的生成文本。
- 不针对任何在线服务，仅本地开源权重模型。

## 9. 预计开销

- 3B 模型 bf16，批量前向/生成在 RTX 5090 上均为分钟级；
  全流程（M0–M3）预计 GPU 时间 < 1 小时；M4 另加约半小时。
