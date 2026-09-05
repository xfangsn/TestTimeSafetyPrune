# Plan (draft, for codex review) — pair-harness direction-building to make backtracking BLADE-editable

## 0. Why
Our current backtracking direction is built from a **keyword-span vs rest** contrast within one trace.
Measured failure (Qwen3-14B, blind-annotated manipulation check):
- keyword lexicon precision only 0.23–0.33 (keywords over-fire massively);
- **remove** drops keyword rate (7.1→~2) but semantic backtracking barely moves (0.28→0.17, Δ ns);
- **amplify** null at α=1.5; α-sweep {2,3,4} does NOT raise semantic backtracking (α=4 even collapses
  it, Δ−0.38, while ppl stays <1%).
Diagnosis: the direction tracks the *surface expression* of backtracking, not the behavior, so edits
move words, not the computation. Fix: build the direction from a **construct-valid matched-pair
contrast** — same problem/prefix, one trace that genuinely backtracks vs one that doesn't — and
BLADE-score writer weights against that gap. Goal: make BOTH remove and amplify have a real
*semantic/functional* effect (not just keyword change).

## 1. Primary method: ① error-injection matched pairs
The cleanest construct for backtracking = *error-triggered self-correction*. So we manufacture the
opportunity and contrast correction vs no-correction on the SAME prefix:
1. Take MATH problems with known answers. For each, build a prefix
   `P = <problem> <think> <partial correct reasoning up to step k> <one INJECTED wrong step>`.
2. Sample N continuations from the identical prefix P (temperature).
3. Auto-classify each continuation:
   - **backtrack+** = it notices and corrects the injected error (executor: final answer correct
     ⇒ it must have overturned the wrong step; AND an LLM judge confirms it references/repairs that
     specific step);
   - **backtrack−** = it continues from the wrong step without correcting (executor: wrong final
     answer consistent with the injected error; judge: no repair).
4. **gap direction**: with the prefix held fixed, `r_l = mean_act(backtrack+ continuations) −
   mean_act(backtrack− continuations)` at the *continuation* tokens, per layer; and the BLADE writer
   moment shift `Δμ_W = μ_W(backtrack+) − μ_W(backtrack−)`. This slots into the existing dirs format
   (dirs=r, muC=backtrack+ moments, muG=backtrack− moments) so ELS/mask/eval reuse unchanged.
5. **BLADE**: score writers `s=[r·W·Δμ]_+` → ELS (BLADE-B layers) → BLADE-G mask. Predictions:
   remove (α=0) → **lower correction rate**; amplify (α>1) → **higher correction rate**.

## 2. Evaluation (its own construct metric — self-correction, ProcessBench-style)
Held-out corrupted prefixes (disjoint problems). Primary endpoint = **error-correction rate**:
P(final answer corrects the injected error). Report remove vs base vs amplify(α-sweep) vs controls.
Also: the blind **semantic manipulation-check** (does the edit change annotated backtracking, not just
keywords), accuracy on clean problems, WikiText Δppl, thinking length. Controls (BLADE-G edit only):
random fixed-sparsity, **fixed-damage** (control tuned to match ΔNLL incl. math-domain), shuffled-r.

## 3. Confounds & controls (critical — these decide if the result means anything)
1. **Correctness vs backtracking confound (biggest).** backtrack+ continuations are, by construction,
   more often *correct*; the gap may encode "correctness/quality," not the *act of backtracking*.
   Controls:
   - (a) **outcome-matched pairs**: also form pairs where BOTH continuations end WRONG but one
     genuinely backtracked (attempted a correction) and one didn't — the direction must still emerge.
   - (b) **correctness control direction**: build a separate direction from correct-vs-wrong
     continuations of *un-corrupted* prefixes (no injected error, so no backtracking opportunity);
     the backtracking direction should be distinguishable from this correctness direction (report
     their cosine; ideally edit effects dissociate).
2. **Matched prefix**: identical tokens up to injection; extract activations ONLY on continuation
   tokens (not the shared prefix).
3. **Auto-label reliability**: executor + LLM-judge agreement; hand-audit a sample; report inter-rater.
4. **Distributed-computation risk** (codex): a single mean-diff may capture only an entry signal.
   Pre-register: (i) predict backtrack+/− from pre-correction activations; (ii) dimensionality sweep
   (rank 1/2/4/8 — if it keeps improving, reject single-direction); (iii) confirm the **weight edit
   reproduces an activation-level intervention** and weight-restoration rescues the effect.
5. **Amplify behavior-collapse**: sweep α with the semantic manip-check AND ppl AND accuracy (we saw
   α=4 collapse backtracking with ppl <1% — ppl alone won't warn us).
6. **Injected-error quality**: the wrong step must be *plausible but wrong* (else the model trivially
   ignores it or is always fooled). Generate corruptions by perturbing a correct step (numeric/logic)
   and validate: base model corrects it at an intermediate rate (~30–70%), giving both classes.

## 4. Error-injection construction
Per problem: (a) get a correct step-by-step solution (model or dataset); (b) pick a step k; (c) corrupt
it (change a number/operation/sign, or a wrong lemma) via a rule + an LLM rewrite to keep it fluent;
(d) validate the corruption is wrong (executor) and non-trivial (base correction rate in a target
band). Keep prefixes where base correction rate ∈ [0.2, 0.8] so pairs are balanced.

## 5. Phasing (Qwen3-14B first)
- **P0a**: build pairs on ~120 MATH problems (×N sampled continuations), auto-label, build the gap
  direction; sanity: does a probe on r separate backtrack+/− on held-out (linear decodability)?
- **P0b**: ELS + BLADE-G remove/amplify(α-sweep); measure error-correction rate + semantic
  manip-check + ppl + accuracy, vs base + controls; run confound controls §3(1a,1b).
- **Compare** head-to-head with the keyword-direction: does the pair-direction make remove AND amplify
  work where the keyword-direction failed? That comparison is the deliverable.
- If linear decodability / edit effects fail → honest negative: backtracking not single-direction
  BLADE-editable even from a construct-valid contrast (consistent with distributed computation).

## 6. Fallback pairings (if ① labeling is too noisy)
- ② **natural pairs**: sample N per problem, label backtrack+/− semantically, pair by problem
  (no injection) — easier but less controlled (content diverges).
- ③ **forced-continuation minimal pairs**: same prefix, force "Wait, reconsider—" vs "So, continuing,"
  then generate — most minimal but the forced token is artificial.

## 7. Wording bounds
Claim only: "a matched-pair (error-injection) contrast yields a backtracking direction; we test whether
BLADE weights selected from it edit self-correction rate, vs correctness/random/shuffle controls, on
Qwen3-14B/MATH." Do NOT claim we edit "the backtracking faculty" or that a single direction is the
mechanism unless the dimensionality + activation-patch + outcome-matched controls pass.
