# Plan (rethink) — bidirectional weight-level control as the correctness criterion for behavior localization

## 0. The theoretical problem (user's point)
If we have truly located the weights that *control* a thinking behavior, editing them should be
**bidirectional**: scaling down (α<1, incl. α=0) should REDUCE the behavior and scaling up (α>1) should
INCREASE it, both monotonically. Our results violate this:
- uncertainty: remove works (semantic −0.31), amplify null;
- backtracking: remove borderline/null, amplify null (α=4 collapses it);
- adding-knowledge: remove null.
The remove-but-not-amplify asymmetry is diagnostic: **we located weights whose *removal disrupts* the
behavior (a necessary bottleneck / one component of a conjunction), NOT weights that set the behavior's
*magnitude* (a knob).** So "we located the behavior's weights" is not yet established. Fixing this is
the point of the rethink.

## 1. Why the current method is intrinsically asymmetric (diagnosis)
BLADE score `s=[r·W·Δμ]_+` = a weight's **first-order DIRECT contribution** to the behavior-difference
along r, rectified to positive-pushing weights. Consequences:
1. **Zeroing** removes that direct contribution → reliably disrupts (remove "works" even for a mere
   bottleneck).
2. **Scaling up** does NOT linearly increase the behavior: the writer's output passes through RMSNorm
   (scale-normalizing) + attention routing + downstream nonlinear gates; other writers also contribute;
   superposition means the same weight carries other features. So α>1 saturates / compensates / breaks
   rather than amplifies (matches our α-sweep: α=4 collapses backtracking with ppl <1%).
3. Selecting only `[c]_+` (positive-pushing) is a removal-oriented criterion; the magnitude knob may be
   a *different*, possibly bidirectional, set — or may not exist as a single writer set at all.
Corollary: **remove-effectiveness is weak evidence of localization; bidirectional monotonic control is
strong evidence.** We should demand the latter.

## 2. Reframed goal & success criterion
Goal: find an intervention that gives **bidirectional, monotonic, SEMANTIC control** of a thinking
behavior. Success criterion (pre-registered), per behavior:
- an α-sweep (e.g. α ∈ {0, 0.5, 1, 1.5, 2, 3}) produces a **monotonic** semantic behavior response
  (blind-annotated, not keyword), down for α<1 and up for α>1;
- with capability preserved (ppl + task accuracy) and beyond random/reversed/shuffled controls;
- and the **weight edit reproduces an activation-level intervention** in BOTH directions.
If only one direction works, we report "necessary-but-not-controlling weights," not "located."

## 3. New method: activation-steering-FIRST, then locate weights that reproduce it
The current pipeline jumps straight to weights. Instead, establish controllability at the activation
level first, because a weight edit can only be bidirectional if the underlying activation direction is.
1. **Direction from matched pairs** (not keyword spans): behavior-present vs behavior-absent activations
   contrasted under matched context (see §4). Gives a candidate direction v_ℓ.
2. **Bidirectional activation-steering test (the precondition/gate)**: add +c·v (amplify) and −c·v
   (remove) at the ELS layers during generation; sweep c; measure the SEMANTIC behavior rate
   (blind manip-check). Require a **monotonic bidirectional** response with bounded ppl. If activation
   steering is not bidirectional, weight editing cannot be — stop and report (honest negative; likely
   for functional behaviors like backtracking).
3. **Locate weights** that write v: BLADE `s=[r·W·Δμ]_+` gives the removal set; for the AMPLIFY set,
   test alternatives (see §5) rather than assuming the same weights scale up.
4. **Convergence gate**: the weight edit must move the same semantic metric in the same direction as the
   activation steering, both ways; necessity+sufficiency activation patching; selective (vs
   random/neighbor) restoration.

## 4. Direction construction (matched pairs, replacing keyword-span-vs-rest)
Keyword-span vs all-thinking is lexical + phase-confounded (codex; our P-1). Options, cleanest first:
- **(a) prompt-matched pairs**: same problem, two system/prefix conditions that elicit the behavior vs
  suppress it (e.g. "think very carefully, reconsider and hedge" vs "answer directly, commit"); extract
  at matched RESPONSE positions; subtract within-problem. Controls the problem/content.
- **(b) benchmark-derived pairs** (from the survey): DeltaBench corrected-vs-error / natural
  behavior-present vs absent segments — but codex flagged authorship/style confounds; only with the
  style-matched replacement layer.
- **(c) contrastive within-model natural pairs**: sample N per problem, blind-label behavior +/−, pair
  by problem.
All must pass: incremental validity beyond keyword/length/position baselines; lexical-disjoint eval;
reversed/random controls. The direction is validated by the §3.2 bidirectional STEERING test before any
weight claim.

## 5. Amplify may need a different operation than α·W
Zeroing and up-scaling the same [c]_+ weights are not symmetric operations. For amplification test, in
order of increasing departure from current:
1. α·W on the removal set (current) — expected to saturate; keep as baseline.
2. **Weight-level vector addition / task-arithmetic**: add a scaled outer-product that writes +v into
   the residual (the weight-space analogue of activation steering), rather than rescaling existing
   weights.
3. **Select an amplify-specific weight set**: weights by *sensitivity* (small Δ → large behavior Δ,
   i.e. a gradient/finite-difference bidirectional importance), not by direct positive contribution.
Report which operation (if any) yields monotonic semantic amplification with bounded ppl.

## 6. Evaluation, controls, gates
- **Primary**: semantic behavior rate (blind annotation) across the α/c-sweep → monotonicity in both
  directions; capability (ppl + task accuracy); random/reversed/shuffled + fixed-damage controls.
- **Convergence**: activation-steering vs weight-edit agreement; necessity+sufficiency patching;
  rank sweep (1/2/4/8 — if k>1 wins, the behavior is a subspace, report as such).
- **Stats**: problem-clustered bootstrap; pre-register the monotonicity test + minimum effect.
- **Honest outcomes**: "bidirectionally controllable" (both directions monotonic+semantic) vs
  "removal-only / necessary-not-controlling" vs "not linearly controllable."

## 7. Behavior & model
Start on Qwen3-14B with **uncertainty** (our only current semantic-remove success) as the test case for
the bidirectional criterion: does a matched-pair direction + activation-steering-first give BIDIRECTIONAL
control where the keyword direction gave remove-only? If even uncertainty (a surface behavior) is not
bidirectionally weight-controllable, that bounds the method. Then attempt one functional behavior.

## 8. Wording bounds
Claim "located the weights controlling behavior X" ONLY under bidirectional monotonic semantic control +
activation-patch convergence. Otherwise: "weights whose removal disrupts X" or "an activation direction
that steers X" — never conflate necessity with control.
