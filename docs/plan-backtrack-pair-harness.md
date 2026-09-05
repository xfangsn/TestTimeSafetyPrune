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

## 6b. REVISION after codex review (supersedes §1–§5 where they conflict)
**Decisive flaw (fatal to the mechanistic claim): post-treatment contrast.** For a fixed corrupted
prefix, the hidden state *before generation is identical* for all sampled continuations; backtrack+/−
only diverges *after* sampling. So a whole-continuation mean-diff encodes correction language, later
correct computation, outcome quality, length, confidence, termination — NOT a pre-branch "decision to
backtrack." A successful BLADE edit off this direction would only show "BLADE can exploit weights
**correlated with successful corrupted-prefix recovery trajectories**," not "localization/editing of the
act of correcting." (This also kills the "predict backtrack+/− from pre-correction activations" test —
at the shared prefix those activations are identical.)

**Reframe the construct**: what we can honestly study is **"recovery from an externally-supplied
corrupted prefix"**, NOT "intrinsic self-correction / backtracking faculty / mechanism." (Externally
inserted error ⇒ not intrinsic self-correction anyway — Huang ICLR'24, Kamoi TACL'24; natural
self-generated-error eval is required before that term.)

**Must-have design changes (codex, ranked by how likely they invalidate a positive result):**
1. **Event-aligned windows, not whole continuations**: decompose into Detection / Revocation / Repair;
   build directions from short fixed-width windows around each event; outcome = separate variable. If
   event-alignment infeasible → explicitly call it a "successful-recovery trajectory direction."
2. **Correctness confound → 2×2 design** (BT+/− × correct/wrong), incl. the neglected **BT−/correct**
   cell (ignores corruption / restarts / guesses right). Estimate a backtracking main-effect vector
   β_BT via cross-fitted, **prefix-clustered regression** controlling outcome + interaction + length;
   use β_BT (residualized to correctness) as the direction, NOT the raw success-vs-fail diff. Require a
   double dissociation (backtracking edit moves repair conditional on opportunity; a correctness edit
   moves outcome more than repair). Cosine-to-correctness alone proves nothing.
3. **Corrupted-vs-sham control** to separate error *detection* from *repair*: per prefix build
   {original correct step, meaning-preserving sham rewrite, corrupted rewrite, (natural error)};
   corrupted-vs-sham = detection, BT+/− within corrupted = handling.
4. **Length/position/event-window matching**; equal weight per prefix; don't token-weight Δμ while
   example-weighting r.
5. **Lexical-template-disjoint eval splits** + marker-only negatives (fluent "wait" that doesn't
   revoke; implicit repairs w/o markers); probe decodability ≠ causal use (Hewitt & Liang, EMNLP'19).
6. **Labeling fix**: "final correct ⇒ it corrected the step" is FALSE. Executor → OUTCOME label only.
   Process label via mechanical reuse/revoke/replace check where symbolic execution allows + **≥2 blind
   human annotators** (blind to condition + outcome flag); LLM judge only triages spans (biased). CoT
   is not faithful (Turpin NeurIPS'23; Paul EMNLP'24-findings).
7. **Selection/collider**: the [0.2,0.8] correction-band filter selects on a post-injection variable —
   use an independent pilot for correction propensity, stratify, and EVALUATE on an unfiltered test.
8. **Pseudoreplication**: unit = prefix, not continuation (~120 independent units); problem-clustered
   bootstrap + power analysis at that level.

**Remove ≠ amplify (don't assume symmetry)**: 1-D control exists for refusal (Arditi NeurIPS'24) but
correction is multi-stage; zeroing may break a needed component while scaling need not raise frequency
(and can collapse it — our α=4 result). Respect BLADE-G's scale penalty λ|α−1|; do NOT reuse a
removal-selected mask for large amplify.

**Minimum evidence before any 1-direction/mechanism claim**: nested rank sweep (1/2/4/8, magnitude &
layers chosen on val, one locked test); **stage-specific causal patching** (Meng NeurIPS'22 causal
tracing; preregister corruption+metric per Zhang & Nanda ATTRIB'23); convergent act-add ↔ weight-edit
effects; selective (vs random/neighbor) restoration; necessity+sufficiency patching. If rank>1 wins,
the honest conclusion is "recovery is a subspace/staged computation," not failure.

**Evaluation (codex)**: "ProcessBench-style" was overstated (ProcessBench = earliest-error ID w/ human
labels; Zheng ACL'25). Two co-primary endpoints: (i) **specific process-repair success** (detect+revoke+
validly replace, downstream no longer depends on the error), (ii) **final-answer recovery** (executor).
Report the full transition matrix incl. **sham-correct → unnecessarily revised / broken** (essential
for amplify — raising "correction" by making the model chronically doubt correct reasoning is NOT an
improvement); aggregate = **net correction utility = fixed − broken**. Add natural self-generated-error
eval, detection/repair latency, clean-MATH accuracy, math-domain NLL/prompt-KL, semantic false-positive
backtracking (ppl alone can't see collapse — α=4 proved it).

**Go/no-go gates (proceed to BLADE only if):** (1) process labels high classwise agreement vs blind
human gold; (2) ≥3 (ideally 4) of the 2×2 cells populated; (3) event-aligned, outcome-controlled
direction generalizes to lexical-disjoint problems; (4) activation intervention changes *repair*, not
only final correctness or correction vocabulary; (5) effect survives length/position/sham/
correctness-residualization/prefix-clustered analysis.

## 7. Wording bounds
Claim only: "a matched-pair (error-injection) contrast yields a backtracking direction; we test whether
BLADE weights selected from it edit self-correction rate, vs correctness/random/shuffle controls, on
Qwen3-14B/MATH." Do NOT claim we edit "the backtracking faculty" or that a single direction is the
mechanism unless the dimensionality + activation-patch + outcome-matched controls pass.
