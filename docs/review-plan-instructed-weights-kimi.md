# Kimi review — plan-instructed-vs-pretrained-weights (v1)

I've read only `docs/plan-instructed-vs-pretrained-weights.md`. Here 
is my referee-style review.

---

## (1) EXPERIMENTAL VALIDITY

**1. E3's
controls are necessary but not sufficient — the specificity control is missing f
rom the causal conditions. (Severity: Critical)**
E3(c) vs E3(d) rules out "rev
erting *any* |S|-many coordinates breaks the model," but it does not rule out "r
everting any *behaviorally-important, large-magnitude* coordinates breaks the mo
del" — which is exactly the confound C1–C3 were invented for. The fix is trivial
and high-value: add two E3 conditions — (f) revert **Wanda/C2's** top-|S| set (
behavior-agnostic importance), and (g) the **cross-behavior revert**: revert ref
usal's S in a model and measure *uncertainty*, and vice versa. Condition (g) is
the single strongest control in the whole design: if reverting refusal-S kills r
efusal but leaves uncertainty intact (with disjoint masks), generic-damage expla
nations are dead. C3 currently exists only as a *statistics* control; it must be
promoted to an E3 generation condition.

**2. The magnitude confound is matche
d on the wrong quantity in E3(d). (Severity: High)**
C1 matches |W^pt| deciles,
but what E3(c) actually removes is the *pretrained value*, i.e. the perturbation
applied is −ΔW restricted to S. The damage of a revert scales with ‖ΔW‖ on the r
everted set, not with |W^pt|. If BLADE-S happens to have large ‖ΔW‖ and the rand
om S' doesn't, (c) beats (d) for reasons unrelated to the claim. Fix: make E3(
d) (and ideally a Wanda-set revert) match the **‖ΔW‖-mass distribution** of S (t
otal, or per-decile of |ΔW|), in addition to or instead of the |W^pt| deciles. A
lso report behavior-drop **per unit Δppl** for every condition, so "removes the 
behavior" is always a tradeoff statement, not an absolute.

**3. E3 has an uns
tated precondition that determines whether a null is informative. (Severity: Hig
h)**
If instruction tuning moved S only slightly in absolute terms (‖ΔW_S‖ ≪ ‖W
_S‖), then reverting does nothing and "E3c fails to remove behavior" cannot fals
ify anything. Conversely the falsification criterion as written ("revert S fails
while random revert succeeds") is unreachable if ‖ΔW_S‖ ≈ 0. The doc never gates
on this. Fix: report ‖ΔW_S‖_F / ‖ΔW_{L*}‖_F and median |ΔW|/|W^pt| on S *first* 
(this is partially in E1, but it must be an explicit go/no-go before E3 is inter
preted), and state that E3 is only interpretable when ΔW_S is a non-trivial frac
tion of the total update.

**4. Selection-on-the-outcome circularity in φ_S (a
lso contaminates E2's interpretation). (Severity: High)**
S was selected by ex
actly the quantity being decomposed: c^it = c^pt + c^Δ. Selecting the top-ρ coor
dinates by c^it mechanically biases S toward coordinates where c^Δ is large (sin
ce c^pt for a random coordinate is ~0-centered noise while c^Δ carries the selec
ted signal). φ_S is therefore an overestimate by construction, and the same bias
inflates every E1/E2 statistic. The doc asserts "This is not circular: the measu
rement is in the model that has the behavior," but that sentence only defends E3
against the base-model-vacuity trap; it does not address estimator selection bia
s. Fix: (a) compute φ on a **held-out selection** — select S on one behavior-eva
l split, measure φ on another; or (b) report φ for C1/C2/C3 sets alongside, so t
he *increment* of φ_S over the matched controls is the estimand, not φ_S itself;
(c) scatter (c^pt, c^Δ) per coordinate so readers see the joint distribution rat
her than a ratio of sums.

**5. The fixed-representation decomposition is soun
d, but its scope should be stated more sharply. (Severity: Medium)**
At fixed (
r, Δμ) from the *instructed* model, c^pt = r^T W^pt Δμ is a well-defined counter
factual ("what the pretrained weights would write onto the instructed model's r
epresentations"). That's fine and the doc mostly says this. The residual gap: t
his decomposition attributes the write under the *instructed* model's features;
it says nothing about whether the same coordinates would have been a write-path 
in the pretrained model's own representational geometry (E5 is the right comple
ment, and the doc should say E5 is what licenses the word "created" vs "re-expre
ssed"). Also: because r and Δμ are computed from behavior-contrast data on the i
nstruct model, r itself is a post-IT direction — a fully pre-IT story would need
r^pt, which base-model behavior metrics can't give (correctly flagged as Limita
tion 3, but connect it here).

**6. E4's predicted weakness is fine, but its 
null needs a success criterion. (Severity: Low)**
"Almost certainly partial at 
best" with no threshold makes the negative result unfalsifiable in both directio
ns. Fix: pre-register what partial emergence counts as (e.g., any significant li
ft over base-rate on the power-seeking metric), and include the converse graft —
instruct model with *non-S* coordinates reverted — as a matched baseline.

---


## (2) STATISTICS / METRICS

**7. φ_S = Σc^Δ / Σc^it is ill-posed on signe
d sums — this is a genuine failure mode, not a formality. (Severity: Critical)**

c is signed, and over the top-ρ selected coordinates Σc^it can be small or nea
r-zero due to cancellation between large positive and negative terms, making φ a
rbitrarily large, negative, or sign-flipping. This is exactly why BLADE itself r
ectifies via [c]_+. A ratio of raw signed sums inherits the worst property of th
e unrectified score. Fix — pick one and pre-register:
- Compute φ over BLADE's
actual selected set, i.e. coordinates with [c^it]_+ > 0 (the ones the method rea
lly selects). Then both sums are dominated by positives, though cancellation in 
the numerator remains possible.
- Better: define φ on **rectified components ma
tched by sign**: φ_S = Σ_S [c^Δ]_+·1[c^it>0] / Σ_S [c^it]_+, i.e. "of the positi
ve behavioral write BLADE selects, what fraction is attributable to ΔW."
- Or a
bandon the ratio for a projection/paired formulation: numerator r^T ΔW_S Δμ, den
ominator r^T W^it_S Δμ, reported with a bootstrap CI over eval prompts, plus the
per-coordinate sign-agreement rate (fraction of S where sign(c^Δ) = sign(c^it)) 
and the (c^pt, c^Δ) scatter. The sign-agreement rate is more robust than any rat
io and reviewers will ask for it.

Also: the doc's interpretive mapping "φ→1 
created, φ→0 merely exposed" is a false dichotomy. φ near 1 combined with E3 rev
ert *failing* would indicate redundancy/compensation (other weights carry the be
havior once S is restored), which is a third hypothesis the design cannot curren
tly distinguish. Add a one-line escape hatch: if E3(c) fails but φ≈1, test rever
t of S plus Top-|S|(c) outside S, or state the claim is only "created *and non-r
edundant*."

**8. E's normalization is ambiguous and its control is underspec
ified. (Severity: Medium)**
E = [Σ_S ΔW²/‖ΔW‖²_F] / (|S|/N): is ‖ΔW‖²_F over al
l residual writers in the model, or only L*? If the former, E mixes "IT concentr
ated its update in L*" with "S is enriched within L*" — two different claims. Fi
x: define ‖ΔW‖²_F over the same writer set as N (L*), and optionally report the 
full-model version separately. The overlap statistic should be Top_|S|(|ΔW|) **w
ithin L***, not global, for the same reason.

**9. The relative-change median 
has an unstable denominator. (Severity: Low)**
|ΔW_ij|/(|W^pt_ij|+ε) explodes f
or near-zero pretrained weights and its median is sensitive to ε. Fix: report al
ongside the absolute version (median |ΔW| on S vs controls) and a log-scaled var
iant; the absolute one is what E3 actually manipulates.

**10. C4's null is n
ot a real null as stated. (Severity: Medium)**
"Recompute with a random unit r 
— φ should be unstable/≈0" is vague: for random r, r^T ΔW Δμ is a random variabl
e with scale ‖ΔW Δμ‖/√d — small but not zero, and its *distribution* is the corr
ect null. "Unstable/≈0" gives no pass/fail. Fix: formalize as a permutation test
— shuffle Δμ (or sample r from the empirical r's null via spherical Gaussian), 
compute the null distribution of φ over ≥1000 draws, and report an empirical p-v
alue for the real φ_S.

**11. No uncertainty quantification anywhere. (Severit
y: High)**
Behavior π, Δppl, φ, E — none have replicate/CI plans. Behavior eval
s at these scales have large sampling noise, and the entire E3 prediction is a c
omparison of condition means. Fix: bootstrap CIs over eval prompts/seeds for eve
ry reported quantity; pre-register the comparison as significant only if the E3(
c) vs E3(d) behavior gap's CI excludes zero. Cheap and non-optional.

---


## (3) MISSING EXPERIMENTS / LIMITATIONS

**12. No replication requirement. (S
everity: High)**
Two pairs are listed as assets, but nothing makes Gemma a repl
ication rather than an optional extra. A referee will require the flagship (E3 +
φ + specificity control) on **both** pairs, especially given Limitation 1 (unkno
wn SFT/RLHF mixture; Gemma -it not guaranteed a pure descendant). The cross-pair
agreement *is* the main defense against that limitation — say so and commit to i
t.

**13. Missing limitation: chat-template / token mismatch. (Severity: Mediu
m)**
The pairs differ in embeddings and added chat tokens (Limitation 2 mention
s writers only), but it also means the *inputs* to the writers — Δμ — are comput
ed on chat-formatted prompts that have no clean base-model counterpart. This wid
ens the gap between "the released pair" and "instruction tuning's effect" and s
hould be stated in Limitation 1/2, since it bounds how much of ΔW can be attribu
ted to behavior-relevant training vs. format/token adaptation.

**14. Missing 
experiment: revert-vs-zero efficiency comparison is proposed but not made a firs
t-class result. (Severity: Low)**
The "revert is a better removal operator than
zeroing" secondary prediction is genuinely interesting and should be elevated: r
eport the full behavior-vs-Δppl frontier for (b) vs (c) across ρ. If (c) dominat
es (b), that's a standalone contribution and strengthens the paper independent 
of the attribution claim.

**15. Missing limitation: L* itself may be post-IT.
(Severity: Low)**
L* ([13], [14,1]) was selected on the instruct model. The lay
er-set selection is part of the pipeline whose "creation by IT" is being asserte
d; a strict reviewer could ask for L* re-derived on the PT model's geometry (ev
en if behavior is vacuous there, the perplexity-budget selection is not). At min
imum state that the analysis is conditioned on the instruct model's L*.

---\
n
## (4) IS THE CLAIM OVERSTATED?

**16. "Created" is stronger than what E3 +
φ can jointly deliver. (Severity: High for the abstract; the fix is a wording do
wngrade)**
What the design can actually support, at best:
> *For the released 
Llama-3.2-3B and Gemma-3-4b base/instruct pairs and the studied behaviors, BLADE
's selected coordinates carry a behavioral write whose fixed-representation com
ponent is dominated by the instruction-tuning weight delta; surgically reverting
only those coordinates to their pretrained values removes the behavior at small 
perplexity cost, while ‖ΔW‖-matched random and behavior-agnostic-importance reve
rts do not, and the effect is behavior-specific.*

Three hedges are mandatory 
and currently implicit at best: (i) **released pair, not controlled fine-tuning*
* (Limitation 1 — keep it in the claim's first clause); (ii) **necessary-within
-context, not sufficient and not exclusive** — E3 shows S's ΔW is load-bearing 
in the IT network; it does not show IT didn't *also* create redundant paths els
ewhere, nor that these coordinates are the unique locus (the redundancy escape h
atch in #7); (iii) **the attribution is to the weight delta at fixed instructed-
model representations**, not to "instruction tuning" as a training process — dat
a mixture, format adaptation, and token additions all live inside ΔW (tie to #13
). If any of E3(f)/(g) fail, the claim degrades to "BLADE selects coordinates th
at IT changed and that are behaviorally important" — a much weaker but still pub
lishable finding, and the paper should say so in one sentence.

---

**Summa
ry of must-fixes before running:** #1 (add Wanda-revert + cross-behavior-revert 
E3 conditions), #2 (match ‖ΔW‖, report drop-per-Δppl), #7 (replace/robustify φ_S
; add sign-agreement), #3 (gate E3 on ΔW_S mass), #11 (CIs), #4 (control for sel
ection bias via held-out split or control-set deltas), #12 (commit to both pairs
). Everything else is polish that will survive review if the must-fixes land.
