# BLADE：算法流程与完整数学描述

> **BLADE** = **B**ehavioral **L**ocalization via **A**ctivation-**D**ifference **E**dges。
> 本文给出方法的完整数学规范:符号、两个原料(方向与矩)、逐权重评分及其推导、
> 排序与选择、消融、评价指标,以及行为图谱(concentration / overlap)的定义,最后是
> 端到端伪代码。
> 实现对应:
> [`src/ttsafety/weight_edit.py`](../src/ttsafety/weight_edit.py)（residual writer 枚举）、
> [`scripts/score_refusal_weights.py`](../scripts/score_refusal_weights.py)（refusal 打分）、
> [`src/ttsafety/behaviors.py`](../src/ttsafety/behaviors.py)（多行为管道）、
> [`src/ttsafety/weight_prune.py`](../src/ttsafety/weight_prune.py)（排序/选择/剪枝）、
> [`src/ttsafety/eval.py`](../src/ttsafety/eval.py)（ppl/KL 指标）。
> 评分规则的创新性审计见
> [`method-and-related-work-gradient-free-signed-edge.md`](method-and-related-work-gradient-free-signed-edge.md)。

---

## 1. 记号与设置

设 Transformer 有 $D$ 个 decoder block,残差流维度 $d=d_{\text{model}}$。第 $l$ 层的
**residual writer** 指其输出直接加回残差流的线性映射 $m\in\{\texttt{o\_proj},\ \texttt{down\_proj}\}$:

- **注意力 writer** $\texttt{o\_proj}$:$W^{(l,\text{attn})}\in\mathbb{R}^{d\times d_a}$,输入为注意力输出($d_a$),输出为残差流增量($d$);
- **MLP writer** $\texttt{down\_proj}$:$W^{(l,\text{mlp})}\in\mathbb{R}^{d\times d_f}$,输入为 MLP 中间层($d_f$),输出为残差流增量($d$)。

对一个 writer $W^{(l,m)}\in\mathbb{R}^{d\times d_{\text{in}}}$,其前向为 $y = W^{(l,m)} x$,其中
$x$ 是该 writer 的**输入激活**,$y$ 是它写入残差流的增量。标量权重记为
$W^{(l,m)}_{ij}$,行 $i\in\{1,\dots,d\}$ 索引**输出**(残差流方向),列 $j\in\{1,\dots,d_{\text{in}}\}$
索引**输入**。

**目标池(target pool)** 取一段中层窗口 $\mathcal{L}$ 与两类组件:

$$
\mathcal{P}=\bigl\{\,(l,m):\ l\in\mathcal{L},\ m\in\{\text{attn},\text{mlp}\}\,\bigr\},
\qquad
N=\sum_{(l,m)\in\mathcal{P}} d\cdot d_{\text{in}}^{(l,m)} .
$$

**关于 $\mathcal{L}$ 的两种用法**:做**跨行为图谱**(§8)时用**固定中层窗口**以保证各行为在同一索引
空间可比,本项目实例 $\mathcal{L}=\{7,\dots,18\}$;做**单行为定位/移除**时 $\mathcal{L}=\mathcal{L}^\star$
由 §4.5 的 ELS **数据驱动选出**(不再手挑窗口)。固定窗口实例:$d=3072$,$d_f=8192$,$d_a=3072$,
共 24 个矩阵,$N=12(3072{\times}3072)+12(3072{\times}8192)=415{,}236{,}096$。

**行为(behavior)** 由一个可控的二分对比给出:一族"行为侧"输入/续写 $\mathcal{A}$ 与
一族"中性/相反侧"输入/续写 $\mathcal{B}$。方法对每个行为独立执行。

---

## 2. 原料一:行为方向 $r^{(l)}$

设 $h^{(l)}(\cdot)\in\mathbb{R}^{d}$ 为第 $l$ 个 decoder block **输出**处、在选定 token 位置
$p$ 的残差流激活。方向由**均值差(mean-difference / CAA 风格)** 给出:

$$
v^{(l)} \;=\; \frac{1}{|\mathcal{A}|}\sum_{a\in\mathcal{A}} h^{(l)}(a)
\;-\;\frac{1}{|\mathcal{B}|}\sum_{b\in\mathcal{B}} h^{(l)}(b),
\qquad
r^{(l)} \;=\; \frac{v^{(l)}}{\lVert v^{(l)}\rVert_2}\in\mathbb{S}^{d-1}.
$$

**符号定向(sign orientation)**:令 $r^{(l)}$ 指向**模型当前偏好的一侧**,即选取行为侧
$\mathcal{A}$ 使基线更倾向 $\mathcal{A}$;这样定位到的是"模型已有偏置"的权重。定向自检要求
行为侧激活在 $r^{(l)}$ 上的投影显著高于相反侧:

$$
\big\langle \overline{h^{(l)}(\mathcal{A})},\,r^{(l)}\big\rangle
\;>\;
\big\langle \overline{h^{(l)}(\mathcal{B})},\,r^{(l)}\big\rangle .
$$

**两种实例(位置 $p$ 与对比集不同,数学形式相同):**

| 实例 | $h^{(l)}$ 取位 $p$ | $\mathcal{A}$ / $\mathcal{B}$ |
|---|---|---|
| refusal | prompt 最后一个非 pad token | harmful prompts / harmless prompts |
| A/B 行为 | 答案 span 的 token 均值 | 行为侧答案续写 / 相反侧答案续写 |

---

## 3. 原料二:写入端激活差 $\Delta\mu^{(l,m)}$

设 $g^{(l,m)}(\cdot)\in\mathbb{R}^{d_{\text{in}}}$ 为 writer $(l,m)$ 在选定位置 $p$ 的**输入激活**
(即 $\texttt{o\_proj}$/$\texttt{down\_proj}$ 的输入)。两条件下的均值矩:

$$
\mu^{(l,m),A}=\frac{1}{|\mathcal{A}|}\sum_{a\in\mathcal{A}} g^{(l,m)}(a),\qquad
\mu^{(l,m),B}=\frac{1}{|\mathcal{B}|}\sum_{b\in\mathcal{B}} g^{(l,m)}(b),
$$

$$
\boxed{\;\Delta\mu^{(l,m)} \;=\; \mu^{(l,m),A}-\mu^{(l,m),B}\;\in\;\mathbb{R}^{d_{\text{in}}}\;}
$$

只需前向传播即可累计(forward pre-hook 累加求和再除以计数),无需反向。**实现细节**:
`collect_span_input_moments` 是把 span(A/B 行为为答案 span、含结束符 EOT;refusal 为末位 token)
内**所有 token 跨样本汇总求和 / 总计数**,即**token 加权**均值——与上式"逐样本先平均再跨样本平均"
仅在各样本 span 等长时严格一致,span 不等长时略有加权差异。

---

## 4. BLADE 评分与推导

对目标池中每个标量权重 $W^{(l,m)}_{ij}$:

$$
\boxed{\;
c^{(l,m)}_{ij} \;=\; r^{(l)}_i \; W^{(l,m)}_{ij}\;\bigl(\Delta\mu^{(l,m)}_j\bigr),
\qquad
s^{(l,m)}_{ij} \;=\; \bigl[\,c^{(l,m)}_{ij}\,\bigr]_{+}=\max\!\bigl(c^{(l,m)}_{ij},0\bigr).
\;}
$$

**推导(为什么这是精确的边贡献)**。定义 writer $(l,m)$ 在方向 $r^{(l)}$ 上的
**局部直接输出**为标量 $z = \langle r^{(l)},\,W^{(l,m)} x\rangle = r^{(l)\top} W^{(l,m)} x$。
在两条件的平均输入 $\mu^{A},\mu^{B}$ 之间,该量的差为

$$
\Delta z^{(l,m)}
= r^{(l)\top} W^{(l,m)}\bigl(\mu^{(l,m),A}-\mu^{(l,m),B}\bigr)
= \sum_{i=1}^{d}\sum_{j=1}^{d_{\text{in}}} r^{(l)}_i\,W^{(l,m)}_{ij}\,\Delta\mu^{(l,m)}_j
= \sum_{i,j} c^{(l,m)}_{ij}.
$$

因此 $c^{(l,m)}_{ij}$ **恰好**是边 $j\!\to\!i$ 对"行为相对中性、沿方向 $r$ 的局部直接输出差
$\Delta z$"的可加贡献;**在输入 $\Delta\mu$ 固定的前提下**,把 $W^{(l,m)}_{ij}$ 置零使 $\Delta z$ 精确
减少 $c^{(l,m)}_{ij}$。(真实剪枝会经再前向改变下游/上游激活,故对**最终行为**的作用只是近似,见下框。)

**clamp 的含义**:$s_{ij}=[c_{ij}]_+$ 只保留**正向推动**行为方向的边($c_{ij}>0$)。删除
这些边直接降低 $\Delta z$、削弱行为的直接写入。负贡献边($c_{ij}<0$,反向抵消行为)不被选中。

> 说明:$\Delta z$ 是**线性直接项**,不含跨层非线性传播,故 BLADE 是一阶、gradient-free
> 的归因,而非行为对最终 logit 的全导数。经验上删 $s$ 最高的边即可移除行为(见 §7)。

---

## 4.5 有效层选择(Effective-Layer Selection, ELS)

目标层不应手挑窗口,而由数据用**固定步骤**确定。ELS 分两步:**① solo 筛选(候选池)**
+ **② best-first 贪心联合选层(选择判定)**。

**① solo 筛选(诊断/过滤,不做最终判定)**:对每层单独剪 $\kappa_s$(默认 0.5% 单层池),
只保留能力代价在预算内的层,得候选池

$$
\mathcal{P}=\bigl\{\,l\in\mathcal{L}_{\text{all}}:\ \Delta\text{ppl}_l\le\beta\,\bigr\},\qquad
\Delta\text{ppl}_l=\tfrac{\text{ppl}(\text{prune }\mathcal{S}_l)-\text{ppl}_{\text{base}}}{\text{ppl}_{\text{base}}}.
$$

这一步只**剔除能力关键层**(如 Qwen 的 $L_0$:$\Delta\text{ppl}=7.9\%$),**不按行为效应筛**
——单独无用、但与他层协同才生效的层必须保留。

**② best-first 贪心联合选层(选择步骤)**:从 $\mathcal{L}^\star=\varnothing$ 起,每轮在候选池里
试加每个未选层、在 $\mathcal{L}^\star\!\cup\!\{l\}$ 上剪 test-frac、量**联合**行为指标 $\pi$,取降幅
最大者 $l^\star$;若它使联合指标再降 $>\varepsilon$(严格,实现 `best_m < current - eps`)且
$\Delta\text{ppl}\le\beta$ 则加入,否则停:

$$
\boxed{\ l^\star=\arg\min_{l\in\mathcal{P}\setminus\mathcal{L}^\star}
\pi\!\bigl(\text{prune }\mathcal{L}^\star\!\cup\!\{l\}\bigr);\quad
\text{加入 iff }\ \pi\text{ 再降}>\varepsilon\ \wedge\ \Delta\text{ppl}\le\beta.\ }
$$

**判定层面只有两个超参**:$\beta$(ppl 预算)与 $\varepsilon$(停止阈值,默认 0.005);相较单层法
**去掉了三个任意超参**——top-$k$ 候选数、固定加入顺序、$\delta$ 硬阈值,这些会**漏掉协同型行为**
(任一单层都不达标、合起来才生效,如 corrigibility)。**但并非"零超参"**:仍有固定的评估稀疏度
$\kappa_s$/test-frac(0.005)与调用方传入的排名下限 $\max(\text{frac},0.01)$。best-first 评估**联合**效果
故**能捕捉协同**,但它仍是**贪心**(不是全局最优;精确并列时取决于遍历顺序,故非严格"顺序无关"),
且只发现"每加一层当步就再降 $>\varepsilon$"的协同,**纯多层协同**仍会漏。

若 $\mathcal{L}^\star=\varnothing$ 则在**当前设定下**未能干净定位(通常因与能力纠缠或模型不具备)——
一个有意义的阴性判定,但它相对当前 $\beta/\varepsilon$/test-frac,非绝对不可定位。随后完整 BLADE
(§5)只在 $\mathcal{L}^\star$ 的 writer 上扫稀疏度。

**为什么用 best-first(三模型验证,`results/blade_layer_select_*.json`)**:对 7 行为 × 3 模型
(Llama/Qwen/Gemma)对比三种 data-driven 选层——solo 单层、ranked-greedy、best-first——
**best-first 在 16/17 案例最优或并列**;它修好了 solo 的协同盲区(corrigibility 从"删不掉"→
$0.29\text{–}0.35$),也不像固定窗口那样在 Qwen/Gemma 全失效(固定窗口非 data-driven,已弃用)。
用 best-first 重跑危险图谱还**解锁了此前判"弥散"的自我增强危险倾向**:want-more-capabilities
$0.96{\to}0.64$、acquire-power、no-shut-down、independence-from-oversight 均降到 chance 附近
(见 `figures/blade_danger_map.pdf`)——证明"弥散"多为**弱选层法的伪影**,非行为本身属性。

**跨模型经验(ELS 验证,`results/blade_layer_select_*.json`、`results/blade_refusal_els_*.json`)**:
BLADE + ELS 在 **Llama-3.2-3B、Qwen3-4B、Gemma-3-4B 三个 ~4B 模型上均成立**——对模型
**实际具备且非弥散**的行为,能在 $\le\beta$ ppl 预算内(refusal 宽 $\mathcal{L}^\star$ 清零除外)干净
移除。**当前 best-first 的每模型 × 每行为完整数字见
[`blade-experiment-report.md`](blade-experiment-report.md) §2–§3**,此处不再复制以免与之脱节。

三条跨模型规律(refusal best-first,`blade_refusal_els_*.json`):**(1)** refusal 三模型全部可
完全清零,且高度冗余——best-first 常收敛到单个中/晚层(Llama $L^\star{=}[12]$、Qwen $[22]$,
Gemma 需 $[15,5]$);**(2)** 有效层随模型系统性后移:**Llama L12 < Gemma L15 < Qwen L22**
——ELS 自动适配,手挑窗口会翻车;**(3)** 清零 refusal 的 ppl 都很低(Llama +0.32% / Gemma
+1.53% / Qwen +2.06%),Qwen 略高但差距不大;先前"Qwen +17.5%"是**宽 $\mathcal{L}^\star$** 选层
的伪影,最小充分层集下并不存在。

结论:方法边界不是"模型架构",而是 **(a) 行为的浓度**(真正弥散的行为如 sycophancy 各模型
仍难,即便 best-first 也只部分压低)与 **(b) 模型是否具备该行为**(self-awareness 仅 Llama 具备)。
实现:选层 `bestfirst_layers` / `solo_layer_pool` 见
[`src/ttsafety/behaviors.py`](../src/ttsafety/behaviors.py),驱动见
[`scripts/blade_effective_layers.py`](../scripts/blade_effective_layers.py)、
[`scripts/blade_refusal_els.py`](../scripts/blade_refusal_els.py)。

**复杂度**:一趟前向得所有层的 $r^{(l)},\mu$;solo 筛选 $|\mathcal{L}_{\text{all}}|$ 次单层评估;
best-first $O(|\mathcal{L}^\star|\cdot|\mathcal{P}|)$ 次联合评估(每次 = 剪 + pick-rate + 短窗 ppl)。
全程无反向传播,单卡消费级 GPU 可完成。

---

## 5. 排序与选择

给定目标稀疏度 $\rho\in(0,1]$(相对目标池),选出得分最高的
$K=\mathrm{round}(\rho N)$ 条边(实现用 `round`,非 $\lceil\cdot\rceil$),并施加**每矩阵候选上限**
$\kappa$(默认 $\kappa=0.10$),防止单一矩阵垄断:

1. **每矩阵候选**:对每个 $(l,m)$,取其展平得分中最大的
   $\kappa_{lm}=\lfloor \kappa\, d\,d_{\text{in}}^{(l,m)}\rfloor$ 条(实现用 `int()` 截断),得候选集
   $\mathcal{C}_{lm}$(带全局坐标 $(l,m,i,j)$ 与分值)。
2. **全局取前 $K$**:在 $\bigcup_{(l,m)}\mathcal{C}_{lm}$ 上按 $s$ 降序取前 $K$,记选择集

$$
\mathcal{S}(\rho)=\operatorname*{top\text{-}K}_{(l,m,i,j)\in\bigcup \mathcal{C}_{lm}} s^{(l,m)}_{ij},
\qquad K=\mathrm{round}(\rho N) .
$$

$\mathcal{S}(\cdot)$ 关于 $\rho$ 是**前缀嵌套**的($\rho_1<\rho_2\Rightarrow \mathcal{S}(\rho_1)\subset\mathcal{S}(\rho_2)$,
只要 $\rho_2$ 未触及 $\kappa$ 上限),故一次排序即可扫多个稀疏度。

---

## 6. 消融(可逆剪枝)

对选中的边置零,前向后精确恢复(context manager):

$$
\widetilde{W}^{(l,m)}_{ij}=
\begin{cases}
0, & (l,m,i,j)\in\mathcal{S}(\rho),\\[2pt]
W^{(l,m)}_{ij}, & \text{otherwise.}
\end{cases}
$$

模型参数在 context 退出后逐比特还原(备份被删标量的原值)。

---

## 7. 评价指标

**(a) 行为指标 —— MC pick-rate。** 对第 $k$ 个样本,prompt $q_k$,行为侧答案 $a_k^{+}$、
相反侧答案 $a_k^{-}$。teacher-forced 答案对数似然(对答案 span 内 token 求和):

$$
\ell(a\mid q)=\sum_{t\in\text{span}(a)} \log p_\theta\!\left(a_t\mid q,\,a_{<t}\right).
$$

行为侧被选中当且仅当其似然更高,pick-rate 为

$$
\pi=\frac{1}{M}\sum_{k=1}^{M}\mathbf{1}\!\left[\ell(a_k^{+}\mid q_k)>\ell(a_k^{-}\mid q_k)\right].
$$

基线 $\pi_0$ 越偏离 $0.5$,行为越强;剪枝后 $\pi\to 0.5$ 表示行为被移除。
(refusal 另用关键词 + LLM-judge 的拒绝率作行为指标。)

**(b) 能力指标 —— 困惑度。** WikiText-2 上非重叠窗口的 teacher-forced 平均 ppl:

$$
\text{ppl}=\exp\!\left(\frac{1}{T}\sum_{t} -\log p_\theta(w_t\mid w_{<t})\right),
\qquad
\Delta\text{ppl}=\frac{\text{ppl}(\widetilde W)-\text{ppl}(W)}{\text{ppl}(W)} .
$$

辅以 6 个 zero-shot 任务的 $\text{acc\_norm}$ 与 prompt 分布上的 KL$(p_W\Vert p_{\widetilde W})$。

**(c) 零效对照(null control)。** 以相同边数 $K$ **随机**选择 $\mathcal{S}_{\text{rand}}$(多 seed)
并剪枝;真实定位应满足 $\pi_{\text{BLADE}}\ll\pi_{\text{rand}}\approx\pi_0$。

**(d) 选参规则。** 在预算 $\Delta\text{ppl}\le\tau$(默认 $\tau=5\%$)内,取使 $\pi$ 最低的最小
$\rho$:

$$
\rho^\star=\arg\min_{\rho:\ \Delta\text{ppl}(\rho)\le\tau}\ \pi(\rho).
$$

> **实现说明**:上式写的是"预算内 argmin",但 `blade_effective_layers.py` / `blade_refusal_els.py`
> 当前的最终稀疏度 sweep **直接对 $\pi$ 取最小、未再施加 $\Delta\text{ppl}\le\tau$ 过滤**(β 约束只作用在
> §4.5 的**选层**阶段)。实测报告点 ppl 一般都很小,但个别行为的 $\rho^\star$ 可能略超 τ——如需严格
> 受限需在 sweep 里补一个 ppl 过滤(TODO)。

---

## 8. 行为图谱量(Behavior Atlas)

对多个行为在**同一目标池、同一权重索引空间**上分别得分,可比较:

**(a) 浓度(concentration)。** 达到近 chance(阈值 $\theta$,默认 $\pi\le 0.55$)且满足 ppl 预算
的最小稀疏度:

$$
\text{conc(behavior)}=\min\{\rho:\ \pi(\rho)\le\theta \ \wedge\ \Delta\text{ppl}(\rho)\le\tau\}.
$$

**(b) 权重集重合。** 两行为在匹配边数 $K$ 下的选择集 $\mathcal{S}_a,\mathcal{S}_b$:

$$
J=\frac{|\mathcal{S}_a\cap\mathcal{S}_b|}{|\mathcal{S}_a\cup\mathcal{S}_b|},\qquad
\text{overlap-coeff}=\frac{|\mathcal{S}_a\cap\mathcal{S}_b|}{\min(|\mathcal{S}_a|,|\mathcal{S}_b|)}.
$$

**(c) 富集倍数(vs 随机)。** 独立假设下期望交集 $\mathbb{E}_0=K_aK_b/N$,富集

$$
\mathrm{Enrich}(a,b)=\frac{|\mathcal{S}_a\cap\mathcal{S}_b|}{\mathbb{E}_0}
=\frac{N\,|\mathcal{S}_a\cap\mathcal{S}_b|}{K_a K_b}.
$$

**(d) 免阈值连续重合。** 展平得分向量 $s_a,s_b\in\mathbb{R}^{N}_{\ge0}$ 的余弦:

$$
\cos(s_a,s_b)=\frac{\langle s_a,s_b\rangle}{\lVert s_a\rVert_2\,\lVert s_b\rVert_2}.
$$

---

## 9. 端到端算法

> **Algorithm 1 — BLADE(单行为定位与消融)**
>
> **输入**:模型 $\theta$;目标层窗口 $\mathcal{L}$、组件集;对比集 $\mathcal{A},\mathcal{B}$;
> 稀疏度网格 $\{\rho_r\}$;每矩阵上限 $\kappa$;ppl 预算 $\tau$。
> **输出**:选择集 $\mathcal{S}(\rho^\star)$ 与评估。
>
> 1. **枚举** target pool $\mathcal{P}=\{(l,m)\}$(`iter_residual_writers`)。
> 2. **方向**:前向取 $h^{(l)}$,按 §2 得 $r^{(l)}=v^{(l)}/\lVert v^{(l)}\rVert$;做符号自检。
> 3. **矩**:前向(pre-hook)累加得 $\mu^{(l,m),A},\mu^{(l,m),B}$,$\Delta\mu^{(l,m)}=\mu^{A}-\mu^{B}$(§3)。
> 4. **有效层选择(ELS)**:solo 筛选得候选池 $\mathcal{P}=\{l:\Delta\text{ppl}_l\le\beta\}$,
>    再 **best-first 贪心联合**逐层加入(每轮取使联合 $\pi$ 降最多者,再降 $>\varepsilon$ 且
>    $\Delta\text{ppl}\le\beta$ 才收),得 $\mathcal{L}^\star$(§4.5);若 $\mathcal{L}^\star=\varnothing$
>    则判"不可干净定位"并停止。
> 5. **打分**:对 $\mathcal{L}^\star$ 中每个 $(l,m)$,$s^{(l,m)}=\big[\,r^{(l)}\!\otimes W^{(l,m)}\!\otimes(\Delta\mu^{(l,m)})^{\!\top}\big]_+$
>    (逐元素 $s_{ij}=[r_i W_{ij}\Delta\mu_j]_+$)(§4)。
> 6. **排序**:每矩阵取 top-$\kappa$ 候选,全局降序(`rank_weight_indices`)。
> 7. **对每个 $\rho_r$**:$\mathcal{S}(\rho_r)\leftarrow$ 全局前 $K=\mathrm{round}(\rho_r N^\star)$($N^\star$ 为 $\mathcal{L}^\star$ 池大小,`selection_from_ranking`);
>    在 context 内置零(§6),测 $\pi(\rho_r)$、$\Delta\text{ppl}(\rho_r)$;并对随机 $\mathcal{S}_{\text{rand}}$ 同测(§7c);退出还原。
> 8. **选参**:$\rho^\star=\arg\min_{\rho:\ \Delta\text{ppl}\le\tau}\pi(\rho)$;在 held-out 上报告一次。
>
> **多行为图谱**:对每个行为跑步骤 1–5 得 $s^{(\text{behavior})}$,按 §8 计算浓度与两两重合。

**复杂度**:方向 + 矩各一趟前向($O(|\mathcal{A}|+|\mathcal{B}|)$ 次前向);打分为逐矩阵的
外积 $O(\sum d\,d_{\text{in}})=O(N)$;排序 $O(N\log K)$。**全程无反向传播**,单张
消费级 GPU 可完成。

---

## 10. 关键设计点与对照(备查)

- **gradient-free**:只用前向的均值激活 + 权重 + 方向投影,区别于 SNIP/Taylor 的
  $|\,g\odot W\,|$。
- **保留符号**:$[\,\cdot\,]_+$ 精确对应"沿行为方向的正贡献边"。
- **每矩阵上限 $\kappa$** 与**全局 top-$K$** 结合,兼顾局部集中与全局稀疏。
- **零效对照(已实现)+ label-shuffle 方向(计划中,尚未实现)**:前者(相同边数随机剪)证明
  "不是随便剪就坏",**当前代码已有**;后者(用打乱标签得到的伪方向)本应证明"定位来自真实语义
  对比",但**目前尚未在代码中实现**,是一条 TODO——所以现阶段图谱中高倍重合**尚未扣除**答案格式
  的共享成分,解读时须留意。
- **一致索引空间**:所有行为共享同一 $\mathcal{P}$ 与展平坐标,故 §8 的交集/余弦可直接计算。
