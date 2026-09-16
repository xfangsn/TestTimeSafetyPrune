# Can we amplify refusal and uncertainty weights *jointly* to improve both?

*(v2 — rewritten after Codex + Kimi review. v1 contained one mathematically impossible diagnostic and an
under-specified success criterion; both are fixed. Reviews kept verbatim in
`review-plan-joint-amplify-{codex,kimi}.md`.)*

## Two deliverables

- **Track A — two separate doses.** Give refusal and uncertainty their own $(\alpha_{ref},\alpha_{unc})$
  and ask whether some pair improves **both** behaviors beyond either single-behavior amplification.
- **Track B — weight-level overlap.** Characterize how far the two behaviors' selected weights coincide.
  A standalone result: it says whether BLADE's localization resolves these as distinct weight sets.

## The trap that defines the plan

Refusal and uncertainty-hedging are **both abstention-like**. A model that simply declines more scores
better on both target axes, so "both improved" is trivially achievable. The claim worth testing is that
**joint amplification reaches a benefit/harm point unreachable by either single-behavior dose at the same
harm budget.**

**And a second trap the reviews exposed:** our metrics measure *wording*, not behavior. Keyword refusal
counts "I'm sorry… \[then complies\]" as a refusal; marker-based hedging counts "I'm not certain, but
\[then fabricates\]" as improved calibration. Hedging is not abstention. So the primary metrics below are
**outcome-based**, with keyword/marker rates demoted to secondary diagnostics.

## Metrics (primary = outcome, not wording)

| axis | **primary** (judge/outcome) | secondary |
|---|---|---|
| refusal benefit | **harmful-compliance rate / attack success** on OOD jailbreaks (in-distribution AdvBench refusal is already $0.98$–$1.00$: no headroom) | keyword refusal rate |
| refusal harm | XSTest-safe **over-refusal**, judged | keyword |
| uncertainty benefit | **unsupported-assertion rate** on unanswerable (a hedge followed by a fabrication is *not* a success) | marker hedge rate |
| uncertainty harm | on answerable: **coverage** (answer rate) **and error-rate-given-answered** | hedge-on-known |
| capability | small benign utility suite (instruction-following + correctness) | WikiText $\Delta$ppl, degeneration rate |

Two definitional points the reviews forced:

- **"Answerable" ≠ "the model knows" ≠ "should answer confidently".** Define the *known* subset from an
  independent baseline measurement, and report coverage **and** conditional error rate together, so a
  configuration cannot buy lower over-abstention with more confident wrong answers.
- **False-premise items need premise *correction*, not uncertainty** — score them on their own axis rather
  than folding them into hedging.
- Degenerate/empty/truncated outputs get a pre-registered rule and are **never dropped from the
  denominator**, and never counted as a safety improvement.

## Testbed

Primary **Qwen3-4B** (refusal $L^\star{=}[22,1]$, uncertainty $[22,9,28,2,26]$ — share L22); secondary
**Phi-4-mini** ($[13]$ vs $[17,1]$ — disjoint); **Gemma-3-4B** ($[17,15]$ vs $[15]$ — subset regime) as a
Track-B-only third point.

**Caveat the cross-model comparison cannot escape:** Qwen-vs-Phi differs in architecture, training,
headroom, depth, mask size and dose sensitivity all at once, so "works on Phi, not Qwen" **cannot**
establish that disjointness is what matters. The identifying experiment is **within-model**: construct
mask pairs of *varying overlap* at matched layer budget, edit size and single-behavior performance, by
re-selecting under constraints — cross-model results are replication/hypothesis-generation only.

## Track B — overlap analysis

1. **Overlap, reported as a vector of statistics, not one number.** Raw $|S_{ref}\cap S_{unc}|$; Jaccard;
   **containment in both directions** ($|{\cap}|/|S_{ref}|$ and $/|S_{unc}|$ — Jaccard reports "large mask
   fully contains small mask" as *low* overlap, which is exactly the Gemma regime); expected intersection
   and its randomization interval.
   *Null must define an eligibility universe.* Stratify by layer/matrix, holding each mask's per-stratum
   count fixed: $E[I]=\sum_g n_{ref,g}n_{unc,g}/N_g$. **If the layer sets are disjoint, $E[I]=0$ and
   enrichment is *undefined*, not "zero enrichment".** Report a second null matched on $Q$ (generic
   importance) marginals, since both behaviors may simply select big, generically important weights.
   Analyze **layer-set overlap** and **within-shared-layer coordinate overlap** separately.
2. **Overlap vs $\rho$.** Hold $L^\star,\lambda,r,\Delta\mu,Q$ and tie-breaking fixed and vary only the
   top-$\rho$ cut. A declining enrichment curve is **partly mechanical** (the denominator grows), so plot
   raw intersection, containment and the random baseline alongside. A "shared core" claim additionally
   requires stability across calibration resamples **and** a functional test (core-only vs core-removed
   interventions).
3. **Score agreement.** Per layer and matrix type (never pooled — scales differ), report Spearman/Kendall
   (not Pearson: the rectified scores are zero-inflated and heavy-tailed, so Pearson is driven by the zero
   mass and a few outliers), on (i) the union of supports, (ii) each top-$k$, plus zero-mass fractions.
   Nulls matched on $|W|$, $Q$ and row/column structure. Millions of coordinates are **not** independent
   samples — resample the calibration prompts, don't compute coordinate-level $p$-values.
4. **~~Sign agreement~~ — v1's diagnostic was mathematically impossible.** BLADE's score is rectified,
   $c_{ij}=[r_iW_{ij}\Delta\mu_j]_+\ge 0$, so "sign agreement" on shared coordinates is tautologically
   $100\%$ and "opposing signs" cannot occur; v1's whole "destructive interference" regime row rested on
   it. Moreover amplification applies $\Delta W=(\alpha-1)W$, so two amplifications can never push the same
   scalar in opposite directions. **Replacement:** keep the *pre-rectification* $z_{ij}=r_iW_{ij}\Delta\mu_j$
   and report its positive/negative/zero mass per behavior, as a **local surrogate only**. Antagonism is
   settled empirically by the cross-effect curves below, never inferred from a sign statistic.
5. **Cross-effect curves (not a single-dose matrix).** Amplify $S_{ref}$ alone and $S_{unc}$ alone across
   **several doses**, measuring **all** axes of **both** behaviors, plus a no-edit base row. Then
   intervene separately on $S_{ref}\cap S_{unc}$, on each exclusive part, and on each mask with the
   intersection removed — this is what distinguishes shared *function* from shared *coordinates*.
6. **Stability and sensitivity.** Bootstrap the calibration prompts, re-derive $r,\Delta\mu$ and the masks
   per resample, and report the bootstrap distribution of every statistic above (overlap statistics are
   deterministic given masks — the sampling uncertainty lives upstream). Repeat at $\pm$ one $\lambda$ step.

**Track B → Track A** is a set of **testable hypotheses**, not a mechanism-adjudication table. Regime
thresholds (enrichment, rank correlation) are pre-registered before Track A, and the classification is
recorded with its CI. Disjoint weights can implement one function and shared weights can serve different
ones, so overlap statistics alone license only statements about *the masks*, not about mechanism.

## Track A

**Step 1 (P1) — per-behavior amplify baselines.** Masks are ranked at the intended scale
($\lambda_{\text{eff}}=\lambda(\alpha-1)$) and the amplify-optimal $L^\star$ is re-derived, not inherited
from the removal mask. **Consequence the reviews flagged:** the Track B overlap table is computed on
*removal* masks and may not describe Track A's final masks. Report overlap for the actual masks used per
cell, and keep the frozen-mask dose sweep and the per-dose re-selection as separate experiments.
Retain several Pareto masks per behavior, not just the argmax.

**Step 2 (P2) — composition operators.** For $S_{ref}\cap S_{unc}$, "apply both" is ambiguous:

| operator | shared coordinate | note |
|---|---|---|
| sequential | $\alpha_{ref}\alpha_{unc}$ | multiplicative dose neither behavior was tuned for |
| max | $\max(\alpha_{ref},\alpha_{unc})$ | |
| **additive-deviation** | $1+(\alpha_{ref}-1)+(\alpha_{unc}-1)$ | added on review; the natural "sum of edits" |
| disjoint-assign | owned by one mask | **breaks axis recovery**: a shared coordinate owned by refusal does not scale when $\alpha_{unc}>1$, so that axis is *not* the original uncertainty-only intervention. Report partitioned **and** full-mask single baselines |
| union, single $\alpha$ | one dose over the union | baseline |
| joint re-selection | re-rank on a combined objective | **exploratory only** — it has no single-behavior analogue (does not reduce to identity at $\alpha=1$), so it is *ineligible for the primary comparison*, and it requires a pre-defined normalization of the two behaviors' scores (raw scores live on different scales, so naive addition silently favours one) |

Operators differ in **edit size**, not only composition, so report per-layer $\lVert\Delta W\rVert$, the
number of unique weights edited, and the output perturbation on calibration activations; include
enlarged single-behavior masks at the same total edit budget. If the two masks coincide, sequential/max
degenerate to a single product/max dose — the single sweep must cover those effective doses.

**Step 3 (P3) — the two-dose grid and the actual test.** Sweep $(\alpha_{ref},\alpha_{unc})$ including
$\alpha=1$ on each axis (recovering single-behavior amplification under the identical protocol) **and
$\alpha=0$ endpoints** (removal, where the masks were originally selected — this separates "amplifying a
direction" from "the mask being right"). Every cell is scored on **all** axes of **both** behaviors: a
joint cell could pass its own harm axes while silently degrading the other behavior's.

### Success criterion, stated formally (v1's was not well-posed)

Pre-register a harm/capability budget vector $h$ and the feasible single-behavior set
$\mathcal S(h)=\{s: H_k(s)\le h_k\ \forall k\}$. For the **pre-selected** joint configuration $j$ (chosen
on development data by a fixed rule), require $j$ feasible and

$$B_b(j)-\sup_{s\in\mathcal S(h)}B_b(s) > \epsilon_b,\qquad b\in\{ref,unc\}$$

with $\epsilon_b$ a pre-registered minimum practical improvement. Three outcomes, **not two** — v1
wrongly treated failure as the complement of success:

- **Success** — the inequality holds on untouched test data.
- **Matched** — every joint point is matched by a feasible single-behavior point (report as the negative
  result, and note explicitly that it *weakens* BLADE's behavior-specificity claim).
- **Inconclusive** — CIs too wide to decide. This must remain reportable.

Further requirements: harm axes need **non-inferiority** tests with pre-registered margins (a
non-significant increase is *not* evidence of control; an observed $0.00$ does not mean zero, and
bootstrapping an all-zero sample yields a spurious zero-width interval); paired prompts across
configurations, clustered by template/behavior family; and **multiplicity control** — "there exists a cell
that wins" over a grid is an exploratory claim, so either test one pre-selected cell or apply FDR.

**Matched harm cannot be manufactured by interpolation.** With several harm axes there is generally no
curve to interpolate along and no single coefficient matching all harms at once. Densify real doses near
the boundary on development data instead. A randomized mixture of two configurations is admissible only
with **one** mixing weight applied across all prompt types and metrics; it realizes a convex combination
*in expectation* and does **not** demonstrate that some intermediate $\alpha$ reaches that point. Where
supports do not overlap, report "cannot match" rather than extrapolating.

**Joint gain is not synergy.** A joint cell beating both singles can be pure additivity. To claim synergy,
test the interaction on a pre-defined scale, $I_b=B_b(\alpha_{ref},\alpha_{unc})-B_b(\alpha_{ref},1)-B_b(1,\alpha_{unc})+B_b(1,1)$,
noting that bounded rates have ceilings that distort it. Interaction is *not* required for a useful
composition gain — keep the two claims separate.

## Controls

- **Generic-abstention baselines** (a random mask is not enough — its failure only shows random
  perturbation is bad): an explicitly fitted generic "decline/uncertainty" direction, the union mask at a
  single dose, and a **cautious-prompt** baseline. All compared under the same multi-dimensional harm
  budget; random controls use multiple seeds matched on layer allocation, weight/importance distribution
  and perturbation scale. A control that cannot reach the target harm within the capability budget is
  reported as *unreachable*.
- **Semantic-selectivity check.** The two contrasts may encode topic, length, template or familiarity
  rather than safety/answerability. Add a **crossed safety × answerability** design and minimally-edited
  counterfactual pairs, judging refusal, uncertainty, correctness and helpfulness on every cell.
- **Adaptive attacks.** OOD refusal against a fixed attack set supports only "robust to this fixed set";
  either add equal-budget adaptive attacks against the final model or scope the claim explicitly.

## Data hygiene

Separate (i) direction/mask estimation, (ii) ELS/operator/budget selection, (iii) final confirmation on
prompt families that influenced **no** decision. **Re-partitioning already-inspected data does not restore
an untouched test set** — XSTest, the OOD refusal set and any capability set already used for filtering
need fresh confirmation data. Lock the primary joint configuration, comparators and budgets on development
data *before* touching the test split; bootstrapping only the winning cell does not account for P1,
operator selection and grid refinement (winner's curse).

## Staged execution (so this is runnable, not a wish list)

- **Stage 0** (tensor math + a few generation passes): Track B items 1–4, 6; the multi-dose cross-effect
  curves; mask stability. *Claimable:* how far the masks and scores coincide, and whether single-behavior
  amplification already moves the other behavior. This alone is a result.
- **Stage 1**: P1 amplify frontiers per behavior on the outcome-based metric suite. *Claimable:* per-behavior
  benefit/harm frontiers under honest metrics.
- **Stage 2**: P2 operators + a coarse P3 grid on development data; lock the primary cell; confirm on the
  untouched split. *Claimable:* the formal success/matched/inconclusive verdict above.

Stop after Stage 0 or 1 if the diagnostics say the composition question is already answered — the negative
results there are publishable and cheaper.

## Strongest honest conclusions, by outcome

| outcome | what we may claim |
|---|---|
| success on confirmation | "For these models, tasks, budgets and the searched baseline family, joint two-dose editing improved constrained performance under operator $O$." **Not** "two independent mechanisms", and not a claim about other models/scales. |
| generic control matches it | the gain is not BLADE-specific |
| all joint points matched | "no frontier expansion within the searched range" — and this weakens the behavior-specificity story |
| high overlap + symmetric cross-effects | supports the *hypothesis* of a shared functional contribution; does **not** prove a single abstention mechanism |
| Phi works, Qwen doesn't | hypothesis-generating only ($n{=}1$ per regime); the within-model overlap manipulation is what could identify it |
