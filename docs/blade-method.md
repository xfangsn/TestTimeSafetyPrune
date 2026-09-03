# BLADE:当前最佳方法流程

> 本文自包含地阐述 BLADE 的**当前定稿方法**(含 best-first ELS)。深层数学推导、
> 复杂度与创新性审计见
> [`blade-algorithm.md`](blade-algorithm.md);实验结果见
> [`blade-experiment-report.md`](blade-experiment-report.md)。
> 撰写日期:2026-08-25。

**BLADE** = **B**ehavioral **L**ocalization via **A**ctivation-**D**ifference **E**dges:
一个 **gradient-free**(只需前向)、**外科式**的方法,用来**定位并移除**一个被训进
LLM 的行为(拒绝、谄媚、权力寻求…)所对应的**残差写入权重**——打分**精细到单条权重(边)**,
但实际剪的是全局分数最高的**一批 top-k 边**(不是只删一条)。

---

## 0. 一图流程

```
对比数据(behavior 侧 vs neutral 侧)
      │
      ├─► 方向 r^(l)   = mean(behavior span) − mean(neutral span)   [每层,CAA/mean-diff]
      └─► 矩差 Δμ^(l,m) = μ^behavior − μ^neutral                     [每个 writer 的输入端均值差]
      │
      ▼
选层 ELS(数据驱动):
   ① solo 筛选  → 候选池 P = {单剪后 ppl 代价 ≤ β 的层}     (剔除能力关键层)
   ② best-first → L* = 逐层贪心加入,每轮取使联合行为降最多者,ppl≤β、再降>ε 才收
      │
      ▼
逐权重打分(L* 内):  s_ij = [ r_i · W_ij · Δμ_j ]_+
      │
      ▼
全局 top-k 选边 → 置零 → 行为被移除(能力基本不动)
```

---

## 1. 前提:目标是什么权重

只对**residual writer**(把输出直接加回残差流的线性层)打分——每个 decoder block 的
`self_attn.o_proj` 与 `mlp.down_proj`。**原因**:BLADE 的分数把 writer 输出投影到残差流
方向 `r`,只有这两类的输出**就是**残差流增量;`q/k/v_proj`、`gate/up_proj` 写入的是别的
空间,公式对它们无定义(要纳入得换 gradient/Taylor,失去 gradient-free 性)。

目标池 $\mathcal{P}=\{(l,m)\}$,$l$ 遍历选中层、$m\in\{\text{attn},\text{mlp}\}$。

## 2. 两样"原料"(都只需前向)

**行为方向 $r^{(l)}$**(每层单位向量)。用**对比对**取激活均值差:

$$
r^{(l)}=\frac{\overline{h^{(l)}(\mathcal{A})}-\overline{h^{(l)}(\mathcal{B})}}
{\lVert\cdot\rVert},\qquad
\mathcal{A}=\text{行为侧},\ \mathcal{B}=\text{中性/相反侧}.
$$

**按模型偏向自动定向**:$r$ 指向模型当前偏好的一侧,这样定位到的是"模型已有偏置"的
权重,剪掉即把行为推向 chance。

**写入端激活差 $\Delta\mu^{(l,m)}=\mu^{(l,m),\mathcal{A}}-\mu^{(l,m),\mathcal{B}}$**:writer
$(l,m)$ 在两条件下**输入激活**的均值差。

## 3. 数据驱动选层:ELS(best-first)

**不手挑窗口**,由数据用固定步骤选**多层**:

**① solo 筛选(候选池,只做过滤)**:每层单独剪 $\kappa_s$(默认 0.5% 单层池),保留
ppl 代价在预算内的层:$\mathcal{P}_L=\{l:\Delta\text{ppl}_l\le\beta\}$。**只剔除能力关键层**
(如 $L_0$),**不按行为效应筛**——单独无用但协同才生效的层要留。

**② best-first 贪心联合选层(选择判定)**:从 $L^\star=\varnothing$ 起,每轮在候选池里试加
每个未选层、在 $L^\star\!\cup\!\{l\}$ 上剪 test-frac、量**联合**行为指标 $\pi$,取降幅最大者;
若再降 $>\varepsilon$(严格,实现为 `best_m < current - eps`)且 $\Delta\text{ppl}\le\beta$ 则加入,否则停:

$$
l^\star=\arg\min_{l\in\mathcal{P}_L\setminus L^\star}\pi(\text{prune }L^\star\!\cup\!\{l\}),
\quad\text{加入 iff }\pi\text{ 再降}>\varepsilon\wedge\Delta\text{ppl}\le\beta.
$$

**为什么是 best-first**:每轮全池 argmin 评估**联合**效果,因此**能捕捉协同**(几层单独都不行、
合起来才删得掉,如 corrigibility),且去掉了单层法的三个任意超参(top-k/固定顺序/硬阈值 δ)。
注意两点诚实边界:它仍是**贪心**(不是全局最优,精确并列时取决于遍历顺序,故非严格"顺序无关"),
且只发现"每加一层当步就再降 >ε"的协同——**纯多层协同**(任何单层先加都不够 ε)仍会漏。三模型 ×
7 行为对比中,best-first 在 **16/17**(容差 0.015;严格口径 15/17)案例最优或并列。

若 $L^\star=\varnothing$ ⇒ 在**当前设定下**(这组 β/ε/test-frac + solo 预筛)未能干净定位——
通常意味着行为与能力纠缠或模型不具备,是个有意义的阴性判定;但它是**相对当前超参**的结论,
不等于绝对"无法定位"(换更小 ε、更大预算或加后向剔除可能改判)。

## 4. BLADE 逐权重评分

对 $L^\star$ 中每个标量权重($i$=输出/残差流索引,$j$=输入索引):

$$
\boxed{\ s^{(l,m)}_{ij}=\bigl[\,r^{(l)}_i\,W^{(l,m)}_{ij}\,\Delta\mu^{(l,m)}_j\,\bigr]_+\ }
$$

**含义**:$\sum_{ij}s_{ij}$(未截断时)$=r^\top W\Delta\mu=\Delta z$,即 writer 沿方向 $r$、
在行为 vs 中性下的**局部直接输出差**;每个 $s_{ij}$ 是边 $j\!\to\!i$ 对 $\Delta z$ 的**可加
贡献**,$[\cdot]_+$ 只留**正向推动**行为的边。**注意这是"冻结输入"下的一阶局部量**:置零一条边
使该 writer 的 $\Delta z$ 精确减少 $s_{ij}$ 只在**输入 $\Delta\mu$ 不变**时成立;真实剪枝会改变下游
乃至(经再前向)上游激活,故整体行为变化只是**近似**由 $\sum s_{ij}$ 预测,不是逐位精确。

## 5. 排序、选择、剪枝

全局 top-k(每矩阵 10% 上限):$K=\mathrm{round}(\rho\,|L^\star\text{池}|)$,取分数最高的 $K$ 条边
置零(可逆:context 退出后逐比特恢复;每矩阵上限用 $\lfloor 0.10\cdot n_m\rfloor$ 截断)。稀疏度
$\rho$ 在 `[0.0005,0.002,0.005,0.02]` 上扫、取使行为最低者。**注意**:这一步的最终 sweep 目前
**只按行为最低挑点、未再过滤 $\Delta\text{ppl}\le\beta$**——β 硬约束作用在**选层**阶段(见 §3),报告
的最终点 ppl 一般也很小,但个别行为的最优点可能略超 β(A/B 移除点实测最大 +4.4%,见报告)。

## 6. 超参数(只有两个是"选层判定")

| 超参 | 默认 | 作用 |
|---|---|---|
| **β**(ppl 预算) | 0.05 | ELS 候选池过滤 + best-first 加入约束的硬上限 |
| **ε**(停止阈值) | 0.005 | best-first 边际改进小于它就停 |
| $\kappa_s$(solo 筛选稀疏度) | 0.005 | 候选池评估用,固定 |
| test-frac(贪心测试稀疏度) | 0.005 | 贪心中评估联合效果用,固定 |
| 组件 | o_proj+down_proj | residual writer(有原理,非选择) |
| per-matrix cap | 0.10 | 每矩阵最多剪 10%(防单矩阵垄断) |

## 7. 能力代价(ppl)有多大

**选层阶段** β=5% 是硬上限(best-first 拒绝任何会超预算的层;最终稀疏度 sweep 未再过滤 β,
见 §5)。实测 17 个 A/B 行为移除点:**均值 +1.3%、中位 +2.2%、最大 +4.4%**,**偶尔还改善**
(Gemma deception −7.9%)。下游 acc_norm 变化在噪声量级(refusal 0.05% 剪枝时单点 −0.47pp,
量级同随机种子波动)。总之**能力基本不动**。

## 8. 方法边界(诚实)

BLADE 是**移除(ablation)**,不是添加:

- ✅ 能删**训进去的、反面=默认**的行为:refusal(→服从)、power/wealth-seeking、
  corrigibility、自我认知…(见报告)。
- ❌ **诱导不出缺失的正向行为**:剪枝**不能**让模型学会弃答/更真实/更鲁棒——那些要
  **addition**(steering / ROSI 式放大)。实测 prune-to-abstain 在 SelfAware 上失败。
- ⚠️ 边界取决于 **(a) 行为浓度**(真正弥散的如 sycophancy 各模型都只能部分压低)与
  **(b) 模型是否具备该行为**(不具备则无可删)。ELS 的 $L^\star=\varnothing$ 会如实报告。

## 9. 实现

- 打分/剪枝:[`src/ttsafety/weight_prune.py`](../src/ttsafety/weight_prune.py)、
  [`src/ttsafety/weight_edit.py`](../src/ttsafety/weight_edit.py)。
- 方向/矩/ELS:[`src/ttsafety/behaviors.py`](../src/ttsafety/behaviors.py)
  (`extract_direction` / `collect_span_input_moments` / `solo_layer_pool` /
  `bestfirst_layers`)。
- 驱动:[`scripts/blade_effective_layers.py`](../scripts/blade_effective_layers.py)、
  refusal 线 [`scripts/blade_refusal_els.py`](../scripts/blade_refusal_els.py)。
