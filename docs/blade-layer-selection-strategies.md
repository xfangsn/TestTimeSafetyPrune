# ELS 选层策略:S0 / S2 / S3 的区别

> BLADE 的有效层选择(ELS)需要**数据驱动地选出要剪哪些层**。我们比较过三种策略,
> 最终定稿用 **S3**。本文讲清三者区别、各自毛病、以及为何 S3 胜出。
> 方法总览见 [`blade-method.md`](blade-method.md);实测见
> [`blade-experiment-report.md`](blade-experiment-report.md)。实现:选层函数
> `src/ttsafety/behaviors.py` 的 `solo_layer_pool` / `bestfirst_layers`;三策略并排对比脚本
> [`scripts/blade_layer_select.py`](../scripts/blade_layer_select.py)。
> (曾有 **S1 = 固定窗口 + 全局 top-k**,窗口写死、非 data-driven,在 Qwen/Gemma 上几乎全
> 失效,已弃用,不在本文比较范围。)
>
> **术语提醒**:下文的 SCREEN_FRAC / test_frac(均 **0.005**)是**权重稀疏度**(剪某层池的
> 多少比例权重);S2 的 **top-12** 是**层数**上限。两者不是一回事。所有对比在同一组常量下跑:
> `SCREEN_FRAC=0.005, DELTA(δ)=0.10, BETA(β)=0.05, GREEDY_MARGIN=0.02, EPS(ε)=0.005,
> GREEDY_TESTFRAC=0.005`。

## 一句话区别

| | 每层怎么评估 | 怎么组多层 | 判定超参 |
|---|---|---|---|
| **S0 单层 ELS** | **孤立**(单剪该层 SCREEN_FRAC) | 把各自达标的层**并起来**(按层号排序,无效率排名) | δ=0.10, β=0.05, SCREEN_FRAC=0.005 |
| **S2 ranked-greedy** | **联合**(剪已选∪候选 的 test_frac) | 在 ppl 可行层里取**前 12**、按此**固定顺序**加,联合改进 **>0.02** 才收 | top-12, 固定顺序, margin=0.02, β, (test_frac) |
| **S3 best-first(定稿)** | **联合** | 每轮在**全部剩余可行层**里选联合降幅最大者,改进 **>ε=0.005** 才收 | β, ε=0.005, (SCREEN_FRAC, test_frac) |

**核心分界**:S0 在**孤立**下评估每层;S2/S3 评估**联合**效果。S2 靠"按单层排序 + 固定顺序 +
较大 margin(0.02)",S3 靠"全池 best-first + 更小 margin(ε=0.005)"。

---

## S0 — 单层 ELS(solo screen)

对每个候选层 $l$**单独**剪其 top-SCREEN_FRAC 权重,量行为下降 $\Delta\pi_l$ 与能力代价
$\Delta\text{ppl}_l$,选择

$$
\mathcal{L}^\star=\{\,l:\Delta\pi_l\ge\delta\ \wedge\ \Delta\text{ppl}_l\le\beta\,\}\quad(\text{按层号排序})。
$$

(对比脚本里 S0 只取"过阈层的并集",**不做效率排名**。)**致命缺陷:协同盲区。** 一个
"**单独剪没用、和别的层一起剪才生效**"的层,$\Delta\pi_l<\delta$ 被漏掉;若**没有任何单层达标**,
$\mathcal{L}^\star=\varnothing$——即便这些层**联合**能删掉行为。实测:corrigibility 在 Llama/Gemma
上 S0 判空(误报"删不掉")。

## S2 — ranked-greedy(有序贪心联合)

先在**ppl 可行层**(单剪 $\Delta\text{ppl}\le\beta$)里按单层效应排序、取**前 12** 作候选,**按此顺序**
逐个试加:把候选层加进已选集 $L$、剪 $L\cup\{l\}$ 的 test_frac、量**联合**行为分;若比当前
**再降 > 0.02(严格)** 则保留,否则跳过。

- **进步**:评估的是**联合**效果 → 能捕捉一部分协同(corrigibility 从"空"变可删)。
- **三个任意超参(毛病)**:
  1. **top-12**——凭空砍的**层数**上限,会把"单层效应弱、协同强"的层(排 13+)**排除**,
     和"抓协同"自相矛盾;
  2. **固定顺序**——按单层效应加,顺序影响结果(贪心路径依赖);
  3. **margin 0.02**——联合改进不足 0.02 的层被拒 → **欠选**(sycophancy 只选到单层 [12]、
     比 S0 的 [10,12] 还差)。

## S3 — best-first(最优优先贪心联合,定稿)

候选池 = **全部**"单剪后 $\Delta\text{ppl}\le\beta$"的层(`solo_layer_pool`,只做**能力过滤**、
不按行为效应砍数量)。从 $L=\varnothing$ 起,**每轮遍历全部剩余可行候选**,在**满足 $\Delta\text{ppl}\le\beta$
的候选中**选联合行为分最低者 $l^\star$;若它使联合分**再降 > ε(0.005,严格)** 则加入,否则停:

$$
l^\star=\arg\min_{\substack{l\in\text{pool}\setminus L\\ \Delta\text{ppl}(L\cup\{l\})\le\beta}}
\pi\!\bigl(\text{prune }L\cup\{l\}\bigr),\qquad
\text{加入 iff }\ \pi(L\cup\{l^\star\})<\pi(L)-\varepsilon 。
$$

**相对 S2 去掉了三个任意超参**:
- 无 top-12 → **全池候选**(不提前砍层);
- 无固定顺序 → **每轮全池 argmin**;
- 无 0.02 的大 margin → **换成更小的 ε=0.005**(所以能补层,不欠选:sycophancy 补成
  [12,10,16,3])。

**判定层面的超参只剩 β 与 ε**(能力预算 + 收敛阈值);但请注意**并非"零超参"**:仍有**固定的**
SCREEN_FRAC / test_frac(0.005,评估用的权重稀疏度);评估时 ELS 调用方还传了个 `max(frac,0.01)`
的排名下限(是**调用方**给 `rank_weight_indices` 的参数,不是该函数自带的常量);而且
**ε 本身就是一个(更小的)硬改进阈值**——只是比 S2 的 0.02 小,不是"无阈值"。另注:选层用的
行为指标是**离散**的——A/B 是 val pick-rate(分辨率 ≈1/条数,≤150 条时 ≳0.0067),refusal 是
48 条样本的拒绝率(步长 ≈0.0208),**两者最小步长都已大于 ε=0.005**;所以实践中 ε 主要起
"拒绝零/负改进"的作用,近似"接受任何能降一格的层",并非在某个连续指标上做实质阈值筛选。

## 三个必须说清的边界

1. **S3 仍是贪心,不是全局最优。** 精确并列时由遍历顺序决定(比较用严格 `<`),所以严格说
   **不是完全"顺序无关"**;而且 best-first 能踩到**局部最优**(实测 Llama deception 上 S2 的路径
   反而更好)。
2. **"抓协同"不是无条件保证。** S3 要求**每次新加的层在当步就让联合分再降 > ε**;若存在一组
   "**任何单层先加都不够 ε、必须多层同时才生效**"的**纯协同**,S3 也发现不了。此外 solo-ppl 预筛
   可能剔除"单独略超预算、但联合其实可行"的层。
3. **报告的分数来自单独的最终 sweep。** 选层用 test_frac=0.005;选完后 `sweep_on` 在选中层上
   **重新**按 `[0.0005, 0.002, 0.005, 0.02]` 扫一遍取最优——所以**表里报告的 pick-rate 不一定等于
   选层时用的分数**。

---

## 并排看一个例子(Llama,报告=最终 sweep 最优)

| 行为 | S0 | S2 | S3 |
|---|---|---|---|
| corrigibility(协同型) | **空 → 判"删不掉"** | [6,13,15,17] 0.60→0.318 | [13,17,15,4] 0.60→**0.294** |
| sycophancy(弥散) | [12] →0.753 | [10,12] →0.740 | [12,10,16,3] →**0.593** |
| power-seeking | [1,10,11,12] →0.447 | [1,5,11,14] →0.393 | [11,1,14,5,3,6,4] →**0.293** |

(数字取自当前 `results/blade_layer_select_llama-32-3b-instruct.json`;因 bf16 GPU
非确定性,不同批次里 sycophancy 的 S0/S2 谁选到 [12]、谁选到 [10,12] 会互换,两者效果
都在 0.74–0.75、都删不动——sycophancy 是**弥散**行为,三法都只能部分压低。)

## 实测结论(3 模型 × 7 行为 = 17 个"被具备"的案例)

**最优或并列次数**(容差 0.015,三策略同图对比):

| 策略 | 次数 |
|---|---|
| S0 单层 ELS | **1 / 17** |
| S2 ranked-greedy | **5 / 17** |
| **S3 best-first** | **16 / 17**(容差 0.015);严格相等口径下 **15 / 17** |

- **S3 在协同型上最强、且从不判空**:corrigibility 等 S0 判空的行为,S3 都能删。注意
  **S2 也能解**其中若干(Llama/Gemma corrigibility、Gemma wealth),只是 S3 普遍更深——所以是
  "S3 一致更好",不是"S3 独家";Gemma wealth 上 S0 其实也**非空**([12,13,14,16]),只是预算内删不动。
- **S3 修好 S2 的欠选**(sycophancy、self-rate);
- 唯一例外 Llama deception(best-first 局部最优,S2 恰好路径更好)。

**定稿:ELS = S3 best-first**(`bestfirst_layers`);S0 的单层 screen 保留为
"候选池过滤 + 诊断"(`solo_layer_pool`),不再用于最终判定。

> 本文经 codex(gpt-5.6-sol)对照代码 review 并据其发现修订(2026-08-27):补全常量、
> 修正 S0 无效率排名/无层级 top-k、S2 top-12 限于 ppl 可行层、严格不等号(>)、S3 的可行集
> argmin、以及"非零超参/贪心非全局最优/协同非无条件"等边界。
