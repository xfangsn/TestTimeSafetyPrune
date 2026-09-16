# Review: plan-joint-amplify-refusal-uncertainty.md

## (1) Experime
ntal validity

**1. The "sign-agreement on shared coordinates" diagnostic is i
ll-posed as written — this is a bug, not a caveat. (Critical)**
$c_{ij} = [r_i 
W_{ij} d\\mu_j]_+$ is rectified, so $c^{ref}, c^{unc} \\geq 0$ everywhere. On an
y shared coordinate both values are non-negative, so "sign agreement" is tautolo
gically 100% and "opposing signs" is impossible — yet the plan's Track B→Track 
A regime table (line 99: "high overlap, opposing signs → destructive interferenc
e predicted") hinges on this. The raw product $r_i W_{ij} d\\mu_j$ has a sign; t
he rectified score does not.
*Fix:* Define sign agreement on the **pre-rectific
ation** behavioral term $\\tilde c_{ij} = r_i W_{ij} d\\mu_j$, restricted to coo
rdinates where at least one behavior has $c>0$. State explicitly that rectificat
ion destroys sign information and that sign agreement is computed on $\\tilde c$
.

**2. The four-axis table under-measures joint-cell harm: cross-behavior har
ms are only in Track B, not in the P3 success criterion. (High)**
The success c
riterion (line 138) requires "no worse harm on all harm axes" where the axes are
*within-behavior* harms (XSTest, hedge-on-known). But the plan's own Track B it
em 5 measures cross-effects because they exist: $\\alpha_{ref}$ plausibly raises
hedging-on-known (uncertainty harm) and vice versa. A joint cell could pass the 
stated harm axes while silently degrading the *other* behavior's harm axis.
*F
ix:* The success criterion must read "no worse harm on **all four** harm/benefit
axes of **both** behaviors" (i.e., the full 2×2 cross-effect matrix for every P3
cell, as is done for the Track B matrix). Track B's cross-effect matrix should 
be a pilot for a measurement that P3 then applies everywhere, not a one-off.


**3. Capability is proxied only by WikiText ppl + degeneration; ppl is a weak gu
ard for capability damage that matters behaviorally. (High)**
A joint edit that
leaves WikiText ppl untouched can still wreck instruction-following or reasoning
, and the "capability" axis is load-bearing in C4 (it decides which cells are ev
en compared).
*Fix:* Add one cheap behavioral capability suite (e.g., IFEval or
a small MMLU slice) to the four-axis profile, or explicitly scope the paper's c
laim to "no degradation detectable by ppl/degeneration" and state this limitatio
n up front.

**4. Keyword-based metrics are brittle in both directions and the
plan's own history shows config-sensitivity. (High)**
Refusal rate = keyword h
it rate can be inflated by degeneration artifacts and deflated by "refusal-adjac
ent" responses that hedge rather than decline; the over-refusal axis (XSTest) an
d both hedge axes inherit this. The plan promises blind judging (line 172) but d
oesn't say whether the *pre-registered numbers* are keyword or judge numbers.

*Fix:* Pre-register the primary metric as judge-based (keyword as secondary sani
ty), report judge agreement ($\\kappa$ or percent agreement), and state which nu
mber the success criterion operates on. If judging is too expensive for every gr
id cell, pre-register the subset of cells that get judged (all P1 reference poin
ts + the claimed-success cell + its matched-harm comparators).

**5. "Matched-
harm comparison" as a concept is sound, but the plan never defines the matching 
tolerance. (High)**
Harm axes are continuous; the grid is coarse ($3\\times3$).
C2 will almost never find a single-behavior cell at exactly equal harm.
*Fix:* 
Pre-register a scalarization or a band. The cleanest: for each joint cell $j$, d
efine the comparator set as single-behavior cells whose harm vector lies within 
a pre-specified band (e.g., each harm axis within $\\pm\\delta$ of $j$'s, with 
$\\delta$ set from bootstrap SE) or within the convex hull of the single-behavio
r frontier in harm space; require benefit dominance against *every* comparator i
n that set, not just the nearest one. Without this, "matched" is negotiable post
hoc.

**6. The Track B→Track A conditioning logic is good, but as written it i
s a post-hoc story generator. (Medium)**
Four regimes, each with a verbal predi
ction about a noisy 2D dose grid, means *any* Track A outcome can be "explained"
by whichever regime the (noisy) overlap statistics landed in. Also, the "prior e
vidence" table (lines 29–34) shows L22 shared between a 2-layer and a 5-layer se
t — that is weak evidence of shared machinery without the enrichment test, yet t
he plan leans on it ("substantive hint").
*Fix:* (a) Pre-register regime thresh
olds (e.g., Jaccard enrichment > X = high overlap; Spearman $\\rho$ > Y = high s
core correlation; sign-agreement < 50% = opposing) *before* Track A, and record 
the regime classification as a pre-registered prediction with its own CI; (b) so
ften the layer-table language to "consistent with overlap; enrichment test in Tr
ack B item 1 will decide."

**7. Winner's curse in "the best single-behavior 
configuration." (Medium)**
The comparator is itself selected by maximizing bene
fit on the same data, so its measured benefit is upward-biased, making the succe
ss criterion *harder* — fine for conservatism — but the plan doesn't acknowledg
e that the comparison baseline is stochastic, and a joint cell beating a lucky-d
rawn baseline with a bootstrap CI that ignores the selection step is not rigorou
s.
*Fix:* Use paired bootstrap over prompts for the joint-vs-baseline differenc
e (this handles correlation), and if the baseline selection is unstable across p
rompt resamples, report the comparison against the top-few single-behavior cells
, not only the argmax.

## (2) Metrics / statistics for Track B

**8. Pearso
n correlation of two rectified, zero-inflated, heavy-tailed score vectors is dom
inated by the zero mass and top outliers. (High)**
"Correlate $c^{ref}$ and $c^
{unc}$ over the whole common universe" (line 75) will be driven by (a) the huge 
shared population of zeros/tiny values and (b) a handful of huge scores. It is a
lso undefined-informative about the top of the ranking, which is where selection
operates.
*Fix:* Use Spearman/Kendall as primary; additionally report correlati
on on (i) the union of the two supports, (ii) the top-$k$ of each, and (iii) per
-layer, with a summary across layers (median + IQR) rather than pooling raw coor
dinates — pooling mixes per-layer score scales. Pooling across $W^o$ and $W^{dow
n}$ has the same scale problem.

**9. The matched-random null for Jaccard enri
chment must match more than size. (Medium)**
A random mask of the same cardinal
ity drawn uniformly will not reproduce BLADE's selection bias toward high-$|W|$
/high-$Q$ coordinates. If refusal and uncertainty both select "big, generically 
important" coordinates, enrichment over a uniform null overstates behavior-speci
fic overlap.
*Fix:* Draw the null from coordinates matched on $Q_{ij}$ (importa
nce) marginals — e.g., stratify by $Q$ quantiles and sample within strata — and 
report both the uniform-null and importance-matched-null enrichment. The uniform
null is the headline; the matched null is the confound control. For small sets, 
also give the exact hypergeometric $p$-value.

**10. Overlap statistics are de
terministic given masks; the sampling uncertainty lives upstream, in $r$ and $\\
Delta\\mu$. (Medium)**
The plan never says where CIs on Jaccard/enrichment come
from. Bootstrapping "coordinates" is meaningless (no randomness there).
*Fix:* 
Bootstrap the *prompts* used to estimate $r$ and $\\Delta\\mu$, re-derive masks 
per resample, and report the bootstrap distribution of Jaccard/enrichment/sign-a
greement. This also doubles as the mask-stability analysis a referee will demand
(issue 14).

**11. Overlap-vs-$\\rho$ curve: Jaccard is variance-dominated at 
small $\\rho$. (Low)**
At $\\rho=.002$ the intersection of two tiny sets is 0/1
coordinates, so the curve will be staircase noise at exactly the region the plan
calls most informative ("small core that dilutes").
*Fix:* Plot enrichment rati
o with the bootstrap CI from issue 10, and complement Jaccard with $|S_{ref}\\ca
p S_{unc}|$ raw counts and the expected-intersection under both nulls; consider 
a rank-weighted overlap (e.g., sum of 1/(rank sums)) that is less discretized.


**12. The success criterion's "there exists a joint cell" needs multiplicity 
control. (Medium-High)**
A $3\\times3$ grid with 4 eval suites is already dozen
s of cells; with refined grids it's hundreds. "Some cell beats the baseline on 
both benefit axes with CI excluding zero" is expected by chance under the null f
or some cell.
*Fix:* Either (a) declare one primary joint cell (chosen by a pre
-registered rule, e.g., the cell maximizing minimum normalized benefit gain subj
ect to harm constraints) and test only that, or (b) apply FDR/Bonferroni across 
the grid. As written, the criterion is exploratory even though it's labeled pre
-registered.

**13. Undefined terms that must be pinned down before runs: "pre
-registered margin of base" (line 55) has no number; "lexical degeneration rate"
has no definition. (Medium)**
*Fix:* Specify the margin numerically (e.g., each
harm axis within 2 bootstrap SEs of base, or an absolute rate delta cap), and de
fine degeneration operationally (e.g., distinct-3, self-repetition rate, or a ju
dge tag), with the exclusion rule (C4) referencing that definition.

## (3) Mi
ssing experiments / controls / limitations a referee would demand

**14. Mask 
stability to prompt resampling is never tested. (High)**
$r$ and $\\Delta\\mu$ 
are estimated from finite prompt sets; the layer-set table already wobbles acros
s models. A referee will ask: re-derive $S_{ref}, S_{unc}$ from different prompt
subsamples — does L22 stay shared? Does the Track B regime classification flip? 
This is cheap (tensor math) and is the same fix as issue 10.

**15. $\\lambda$
sensitivity is unexamined. (Medium)**
Overlap, the regime table, and Track A al
l condition on a penalty $\\lambda$ calibrated under a perplexity budget. Two be
haviors could overlap only because a shared $\\lambda$ threshold prunes both in 
the same place.
*Fix:* Report Track B items 1–3 at $\\pm$ one $\\lambda$ step a
round the operating point, and note the regime classification's $\\lambda$-robu
stness in the Track B→Track A table.

**16. No pure-removal ($\\alpha=0$) cont
rol in the joint grid. (Medium)**
The whole apparatus is amplification-flavored
, but the removal operating point is where the masks were originally selected. A
joint cell at $(0, \\alpha_{unc})$ and $(\\alpha_{ref}, 0)$ would separate "ampl
ifying a direction" from "the mask being right."
*Fix:* Add $\\alpha=0$ endpoin
ts to the dose axes at minimal cost (they're single-behavior removal cells, but
they must be measured under the identical protocol/grid, per the plan's own log
ic at line 134–136).

**17. Test-split hygiene is promised but conditional. (M
edium)**
"should be treated as development data *unless re-partitioned*" — re-p
artitioning XSTest and the OOD refusal set needs a concrete statement (which sub
sets were inspected, how the untouched split is constructed, whether it's large
enough for the planned CIs).
*Fix:* State the split sizes and which prior decis
ions were made on which data; if the untouched remainder is too small to power t
he primary comparison, that itself is a finding to report before running P3.


**18. Two models, both ≤4B, one per regime — and one shared-layer observation re
sts on a single layer. (Medium)**
Failure mode 4 (works on Phi, not Qwen) is ca
lled "arguably the most informative outcome," but with one model per regime it i
s indistinguishable from generic cross-model variability.
*Fix:* At minimum add
Gemma-3-4B (uncertainty $\\subset$ refusal — a *third* regime, subset) as a chea
p Track B-only model, and report Track A on it only if P0/P1 costs allow. Also l
ist model scale as a stated limitation.

**19. Base-model cross-effects need a
zero row.** (Low)**
The Track B cross-effect matrix lacks the no-edit row (base
model's refusal/hedging rates). Without it, "amplifying $S_{unc}$ raises refusa
l" can't be attributed vs. base drift.
*Fix:* Fill in the base row from the P1
pre-runs.

## (4) Overstatement audit — and the strongest honest conclusions


**20. The trap framing is excellent and not overstated; the risks of overstate
ment are elsewhere. (—)**
The plan is notably disciplined. The specific oversta
tements:

- Line 27: layer-set overlap called a "**substantive hint** of share
d machinery." With $n=2$ vs $n=5$ layers sharing one element, a hypergeometric t
ail is needed before that word is earned. → say "suggestive; the enrichment test
decides."
- Line 82: "Near-parallel directions would mean these are **one** mec
hanism measured two ways." Parallel $r$ vectors do not imply one mechanism — $r$
is an average over a prompt distribution; two distinct circuits could partially 
share a readout direction. → "consistent with a shared readout; not evidence of 
a single mechanism."
- Line 100: "cross-effects symmetric … share an abstention
write-path" — symmetry of *measured effects* at two doses cannot establish a sha
red write-path; it establishes symmetric cross-effects. → downgrade the causal v
erb.
- Line 136: putting single-behavior baselines "inside" the grid is right, 
but note the axes recover single-behavior amplification only if the composition 
operator reduces to identity when one $\\alpha=1$ — trivially true for sequentia
l/max/disjoint-assign, but **not** for joint re-selection (line 126), which has 
no single-behavior analogue. → mark joint re-selection as exploratory-only, neve
r eligible for the primary comparison.

**Strongest honest conclusions under e
ach outcome:**

- **P3 success criterion met (with issues 5, 7, 12 fixed):** "
For Qwen3-4B and Phi-4-mini, at the tested scales, a joint two-dose edit reaches
a benefit/harm point that no single-behavior dose at equal harm reaches, under o
perator $O$." *Not* licensed: "refusal and uncertainty are separable behaviors,"
"BLADE localizes behavior-specific weights," or any claim about other models/sca
les. The paper-level claim is about the dose-composition frontier of this method
on these models.
- **Joint cells slide along the abstention axis (failure branc
h, line 143):** "The amplifiable weights of the two behaviors behave as a single
generic abstention dose; C1 + C2 show the gains are dose- and not weight-specifi
c." Equally publishable, and it *weakens* BLADE's behavior-specificity selling 
point — the plan should say so explicitly rather than only framing it as a clean
null.
- **Track B: high overlap, same-sign:** "At this granularity BLADE does n
ot resolve refusal vs. uncertainty as distinct weight sets; the two masks are la
rgely one abstention subnetwork." This is a negative result about localization r
esolution, and it caps how strongly *any* Track A success can be spun as "separa
te but composable."
- **Phi works, Qwen doesn't:** "Composition benefit appear
s only when masks are layer-disjoint" — with $n=1$ per regime this is hypothesis
-generating, not a conclusion; the honest phrasing is "consistent with," and it 
motivates the Gemma (subset-regime) replication in issue 18.

## Priority orde
r for fixes before any compute is spent
1. Issue 1 (sign-agreement on rectified
scores is broken as specified) — cheapest, most embarrassing if caught later.
2
. Issues 5, 12, 13 (pre-register harm-matching rule, multiplicity control, numer
ic margins/definitions) — these define what "success" even means.
3. Issue 2 (c
ross-behavior harms in the success criterion).
4. Issues 10 + 14 (prompt-bootst
rap CIs for overlap statistics; mask stability).
5. Issues 4, 6, 20 (judge-base
d primary metrics; regime thresholds pre-registered; soften three causal phrasin
gs).
