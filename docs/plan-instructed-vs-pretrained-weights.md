# Comparing instructed vs pretrained models at the weight level

*(v2 — rewritten after Codex + Kimi review. v1's central estimator was broken and its headline claim was
not supportable; both are fixed below. Read "What we can and cannot claim" before anything else.)*

## What we want to say, and what we are actually entitled to say

**The intuition.** BLADE's selected coordinates exist in the pretrained model, but their behavioral
function appears to come from instruction tuning.

**What this design can support** (strongest honest version):

> For the released Llama-3.2-3B and Gemma-3-4B base/instruct pairs and the behaviors we study, replacing
> the instructed values of BLADE-selected writer coordinates with the corresponding pretrained values
> *selectively* reduces the target behavior — more than dose-matched random, behavior-agnostic-importance,
> and cross-behavior controls — at small measured utility cost. Those checkpoint differences are causally
> contributing **in the current instructed network**.

**What it cannot support, and we must not write:**

1. **"Instruction tuning *created* the behavior at these weights."** That is a claim about a *training
   trajectory*. A released pair gives us a checkpoint *difference*, not a history (§Provenance). Only E6
   (a controlled post-training run we do ourselves) licenses "created".
2. **"The pretrained model does not use these coordinates."** We never observe PT's own routing; the
   decomposition below evaluates PT *weights* under **IT representations** (§Estimand).
3. **"These coordinates are the unique/sufficient write-path."** E3 shows joint necessity *in the IT
   background*, not uniqueness, not per-coordinate necessity, and not sufficiency.
4. **Anything that assumes the effect is not co-adaptation.** If other modules shifted activation scale
   and the writer update merely compensates, undoing the compensation alone would also break the
   behavior. E3 cannot separate this; say so.

## The trap that shapes the whole design

Base models do not exhibit these behaviors, so "ablating $S$ in the base model does nothing" is **vacuous**.
Evidence: `blade_refusal_els_gemma-3-4b-pt.json` has `base_refusal = 0.0` and the run *aborted* as
"low baseline refusal". Every load-bearing measurement therefore happens **in the instructed model**.

But note the symmetric error, which v1 made: `base_vs_instruct_llama.json` shows Llama base power-seeking
$0.37$ vs instruct $0.74$, and v1 called $0.37$ "at chance". **It is not** — $0.37$ is below the $0.5$
chance level, which may be a *reverse* preference, option-position bias, or parsing failure. Before any of
this we must run **P0** below.

## Assets and provenance

Locally cached (5090; Hazel has none of these): `meta-llama/Llama-3.2-3B` ↔ `-3B-Instruct` (28L, vocab
128256) and `google/gemma-3-4b-pt` ↔ `-it` (34L); residual-writer shapes are identical in both pairs, so
coordinate-wise replacement is mechanically possible.

**Shape identity is not ancestry.** If the released `-it` is not a direct descendant of the released `-pt`,
$\Delta W = W^{it}-W^{pt}$ is merely a checkpoint difference, and permutation symmetries mean equal shapes
do not guarantee unit-to-unit functional correspondence (git re-basin, Ainsworth et al. 2023). **P1:**
record repo revision hashes, config/tokenizer diffs, and any model-card statement of lineage. If lineage
is unverifiable, we call the operation **cross-checkpoint value replacement**, not "undoing instruction
tuning", throughout.

## Prerequisites (run before interpreting anything)

- **P0 — establish the base-model baseline properly.** For each behavior: state the metric, its chance
  level, $n$, and a CI; counterbalance A/B answer positions; separate *valid choices* from *format
  failures*. Add a PT-appropriate assay (forced-choice likelihood / continuation) and evaluate both models
  under a common format as well as each one's native format. Outcome: a defensible statement of what the
  base model does and does not do — not "it's at chance".
- **P2 — gate on update mass.** Report $\lVert\Delta W_S\rVert_F/\lVert\Delta W_{\mathcal U}\rVert_F$ and
  the absolute median $|\Delta W|$ on $S$. If instruction tuning barely moved $S$, a null in E3 is
  uninformative and must not be reported as falsification. Do the subtraction in **fp32**; note that
  released checkpoint precision bounds how small a delta is even detectable.

## Estimand and notation (v1's error, corrected)

$S$ = BLADE's selected coordinates; $\mathcal U$ = all residual-writer weights in $L^\star$ (the common
universe for every ratio below); $\Delta W = W^{it}-W^{pt}$.

BLADE's behavioral term $c_{ij}=r_i W_{ij}\Delta\mu_j$ is linear in $W$, so with $(r,\Delta\mu)$ **fixed at
the instructed model's values**:

$$c^{it}_{ij} = c^{pt\mid it}_{ij} + c^{\Delta}_{ij}$$

Two things v1 got wrong and must be stated explicitly:

- $c^{pt\mid it}$ is a **fixed-representation projected write** — "what the pretrained weights would write
  onto the *instructed* model's representations". It is a hybrid counterfactual, **not** the pretrained
  model's actual routing (which uses PT's own inputs, directions and downstream readout).
- BLADE **rectifies**: $[a+b]_+\neq[a]_++[b]_+$. So this linear split is *not* an additive decomposition of
  the score BLADE actually uses. Any "attribution fraction" language is therefore unlicensed.
- Under a *multi-layer* revert, downstream $\Delta\mu$ changes, so the frozen-representation sum does not
  equal the write change actually realized, let alone the behavior change.

### The estimator (v1's $\phi_S$ is deleted)

v1 defined $\phi_S=\sum_S c^\Delta/\sum_S c^{it}$ and read $\phi\!\to\!1$ as "created". **This is broken:**
$c$ is signed, so with $c^{it}=(1,1)$, $c^{pt\mid it}=(100,-100)$, $c^{\Delta}=(-99,101)$ we get
$\phi_S=1$ while the pretrained positive mass is $100$, nowhere near zero. The ratio can also exceed $1$ or
go negative, and its denominator can pass through zero.

Replace it with rectified, sign-aware **projected-score diagnostics** (descriptive, never called
attribution). With $P_X=\sum_S[c^X]_+$ and $N_X=\sum_S[-c^X]_+$ for $X\in\{pt\!\mid\!it,\,it,\,\Delta\}$:

$$L_S=\frac{\sum_S\big[[c^{it}]_+-[c^{pt\mid it}]_+\big]_+}{P_{it}},\qquad
  G_S=\frac{\sum_S\big[[c^{pt\mid it}]_+-[c^{it}]_+\big]_+}{P_{it}}$$

$L_S$ (positive-write *lost* by reverting, $\in[0,1]$ when $P_{it}>0$) and $G_S$ (positive-write *gained*)
are reported **together**, so reverse-direction changes cannot be hidden. Also report: $P_X,N_X$
separately; the per-coordinate **sign-agreement rate** $\Pr[\mathrm{sign}(c^\Delta)=\mathrm{sign}(c^{it})]$
(more robust than any ratio); and the $(c^{pt\mid it}, c^{\Delta})$ **scatter**, so the joint distribution
is visible rather than collapsed into one number. If $P_{it}\approx0$, report *undefined*, never patch with
an $\epsilon$.

These are diagnostics. **The primary estimator is the behavioral effect in E3**, with matched controls.

## Experiments

### E3 (flagship) — cross-checkpoint value replacement on $S$
In the **instructed** model set $W^{it}[S]\leftarrow W^{pt}[S]$; measure target behavior + utility (below).

*Estimand, stated precisely:* the behavioral effect of undoing $\Delta W_S$ **in the IT background**. Not
per-coordinate necessity; not historical origin.

Conditions — the specificity controls are **generation conditions**, not just statistics:

| | condition | purpose |
|---|---|---|
| a | unedited IT | reference |
| b | zero $S$ | our existing BLADE result |
| c | **revert $S$** | the test |
| d | revert **dose-matched random** $S'$ | not "any edit of this size" |
| e | revert **Wanda/generic-importance top-$|S|$** | not "any important weights" |
| f | **cross-behavior revert** (revert refusal's $S$, measure *uncertainty*, and vice versa) | **the strongest control**: kills generic-damage explanations if it passes |
| g | revert all writers | *broad-intervention comparator* (**not** an "upper bound" — no monotonicity in edit support) |

**Dose matching (v1 matched the wrong quantity).** The applied perturbation is $-\Delta W_S$, so controls
must match the $\lVert\Delta W\rVert$ dose (total and per-decile), per-layer/per-matrix counts, and
row/column/head concentration — *not* $|W^{pt}|$ deciles. Report balance diagnostics. Draw **multiple**
matched masks so mask-to-mask variability is part of the error bar, not a single draw.

**Dose–response, not a single point.** Sweep $W_S(\alpha)=W^{it}_S-\alpha\Delta W_S$ for $\alpha\in[0,1]$
and compare against zeroing/shrinkage curves at **equal behavior reduction** (or equal utility budget).
This is also the only way the "revert beats zeroing" idea can be tested: if
$\lVert\Delta W_S\rVert\ll\lVert W^{it}_S\rVert$, revert is gentler merely because the edit is smaller.

### E4 — sufficiency, as a $2\times2$ rather than a one-off graft
Organize PT/IT $\times$ ($S$ / rest) into four cells $Y_{S,\text{rest}}$ and report the interaction
$I = Y_{11}-Y_{10}-Y_{01}+Y_{00}$, which requires a behavioral metric comparable across all four
backgrounds (see P0). Pre-register what counts as partial emergence. A negative result means the graft
produced no measurable behavior *in that PT background, prompt format and assay* — not "insufficient" in
general. A more informative sufficiency test adds $S$ back in a background that already follows
instructions.

### E2 — projected-write diagnostics
$L_S$, $G_S$, $P_X/N_X$, sign-agreement, scatter, for $S$ **and** every control family, since the
*increment over matched controls* is the estimand, not the raw value on $S$.

**Selection-on-outcome bias:** $S$ was chosen by $c^{it}=c^{pt\mid it}+c^\Delta$, which mechanically favors
large $c^\Delta$. Compute the diagnostics on a **held-out** contrast split (select $S$ on one split,
measure on another) and always report controls alongside.

Supplement the frozen-representation algebra with the **actually realized** quantity: run the forward pass
under each intervention and measure the writer's output projection and the logit/behavior effect, on a
fixed prompt cohort with stated token positions. Intervene layer-by-layer and compare to the joint
intervention.

### E5 — is the feature already present in PT?
Linear probes (harmful/harmless, uncertain/certain) at $L^\star$, PT vs IT. **Probe caveats are
mandatory:** use topic/template-disjoint splits, matched hard negatives, a lexical baseline, and
**control tasks** (Hewitt & Liang 2019) so the result is not "this label is linearly decodable from topic
cues". Probe the **writer inputs**, not only the residual stream, and confirm any claimed link by a
controlled activation intervention that moves the projected write and the behavior. A positive AUROC alone
licenses only "the label is linearly decodable", not "the pretrained model uses this feature".

### E1 — update-mass enrichment (descriptive)
On the common universe $\mathcal U$:
$$E=\frac{\sum_S \Delta W^2/|S|}{\sum_{\mathcal U}\Delta W^2/|\mathcal U|}$$
Report per-matrix enrichment, a matched empirical null, CIs, and the overlap with
$\mathrm{Top}_{|S|}(|\Delta W|)$ **within $L^\star$**, normalized to that null. Report absolute
$|\Delta W|$ alongside the relative version (whose denominator explodes for near-zero $W^{pt}$; give
$\epsilon$-sensitivity). Note: E1's null may **not** match on $\Delta W$ mass — that is the outcome under
test. E1 and E3 therefore need *different* control families.

### E6 (optional, the only route to "created")
Post-train the *known* base ourselves (small SFT/LoRA), saving intermediate checkpoints, with a
behavior-neutral training control and multiple seeds; track when $S$ becomes load-bearing. Expensive, but
it is the only design that converts a checkpoint difference into a claim about instruction tuning.

## Utility measurement (WikiText alone is not enough)

A behavior drop can be an artifact: refusal can fall because output is empty/garbled or
instruction-following is gone; hedging can fall because the model merely became more assertive. So report,
per condition: target behavior; **instruction-following and output validity** (length, EOS behavior,
degenerate-repetition rate); and **behavior-specific outcomes** — refusal separated from actual harmful
compliance, uncertainty separated from accuracy/calibration. Utility as held-out mean NLL change with a CI
in addition to perplexity. Blind, validated judging.

## Statistics (non-optional)

Primary experimental unit is the **prompt**, not the weight — millions of scalars are not millions of
independent units. Cluster-bootstrap over paraphrase/template families; include decoding variability;
report prompt uncertainty and mask variability separately. Pre-register the primary behavior/model/
comparison (E3c vs E3d), and treat the rest as exploratory or correct for multiplicity. "Removes like
zeroing" and "approximately zero" require a **practical equivalence margin**, not a non-significant test.

**Leakage discipline:** separate (i) direction/mask discovery, (ii) layer/operator/budget validation,
(iii) an untouched behavioral test set. Eval sets we have already inspected repeatedly are *development*
data. If WikiText is used to choose an operator or threshold, it is no longer a held-out utility test.

## Falsification, per hypothesis (v1 conflated these)

| hypothesis | falsified by |
|---|---|
| update-mass enrichment | $E\approx1$ against a matched null |
| fixed-representation write change | $L_S$ no larger than for matched controls |
| conditional behavioral effect | E3c's effect $\le$ dose-matched controls, CI excluding the pre-set margin |
| behavior specificity | cross-behavior revert (E3f) damages the *other* behavior equally |
| historical creation | **not testable here** — requires E6 |

$E\approx1$ does **not** falsify behavioral creation (the update's *direction* can matter with no mass
enrichment), and E3+E1+E5 all passing is still compatible with co-adaptation.

## Cost
All local on the 5090; E1/E2 are tensor math (minutes). E3 is the expensive part: ~7 conditions × dose
sweep × behaviors × 2 model pairs, plus the extra utility assays. Both pairs are required as replication —
cross-pair agreement is the main defense against the provenance limitation.
