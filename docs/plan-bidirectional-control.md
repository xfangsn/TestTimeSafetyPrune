# Plan (v2, rewritten after Fable-5.1 + codex reviews + the construction-difference point)
## Bidirectional / controllable weight-level editing of thinking behaviors — done correctly

This supersedes v1. Three independent critiques reshaped it: (A) the user noted our refusal and
uncertainty **directions are built by different methods**, so "refusal amplify worked" is NOT evidence
the reasoning-behavior method's amplify works; (B) Fable-5.1 and (C) codex both showed the v1 premise
and criterion were wrong. Full reviews: `scratch_bidir_codex.md`, `scratch_bidir_brief.md`, and the
Fable task output.

## 0. What the reviews overturned
1. **Premise wrong**: "amplify fails" is false in general — BLADE α-scaling amplified **refusal**
   (0.083→0.50 @α1.5, inverted-U) and **uncertainty keyword-rate** (Qwen3-4B +19%, 8B +11%). The 14B
   *semantic* amplify-null is confounded by config + metric, not proof of a "bottleneck."
2. **"remove works + amplify fails ⇒ necessary-not-controlling" is under-determined**: equally
   consistent with ablation damage (superposition), ceiling/measurement saturation, downstream
   thresholding, conjunctive mechanisms, ceiling/headroom, **and the wrong amplify parameterization**.
3. **Criterion wrong**: monotonic-to-α=3 bidirectionality is (a) too strong — no known intervention
   (CAA/ActAdd/SAE, and our own **refusal is inverted-U**) is monotone that far; (b) too weak — a
   rank-1 `W+=λvuᵀ` is bidirectional at any layer and localizes nothing.
4. **Amplify operation wrong**: raw `αW` moves toward the arbitrary parameter origin (and scales every
   superposed function), NOT along a learned behavior direction; it goes OOD by α≈3. Amplify must be a
   **signed edit along a learned direction**, β centered at 0.
5. **Capability metric wrong**: WikiText ppl missed the α=4 collapse; use held-out **thinking-trace ppl
   + task accuracy**.
6. **Construction confound (the user's point)**: refusal = **prompt-set contrast (harmful vs harmless)
   at the last prompt token (fixed, pre-generation)**; uncertainty/reasoning = **keyword-span vs
   rest, on generated tokens**. Different pipelines → cross-behavior comparison is confounded, AND it
   suggests construction quality (not behavior type) may drive the asymmetry.

## 1. Claim taxonomy — pre-register WHICH claim per experiment (codex)
1. **Causal mediator**: intervening changes behavior.  2. **Necessary**: selective suppression reduces
it (beating damage-matched controls).  3. **Sufficient**: inserting/activating it elicits behavior when
absent.  4. **Control axis**: a signed move along ONE parameter direction gives predictable opposing
changes. Current evidence supports at most "candidate mediators whose ablation disrupts uncertainty."
Bidirectionality is the criterion for **(4) control-axis**, NOT for localization; **selective causal
mediation** (necessity+sufficiency, interchange) is the criterion for mechanism localization.

## 2. Primary factor to control FIRST: direction-construction method (the user's point)
2×N factorial: **construction × behavior**, so we don't reattribute a construction effect to behavior.
- **C1 refusal-style**: contrast two prompt SETS (behavior-eliciting vs behavior-suppressing) at a
  FIXED pre-generation position (e.g. last prompt token / a fixed decision point). Clean CAA
  difference-of-means (Arditi/CAA).
- **C2 keyword-span** (current reasoning method) — baseline to beat.
- **C3 matched natural pairs**: same problem, blind-labeled behavior+/− generations, subtract
  within-problem (residualize position/length/correctness/difficulty/entropy/seed).
Direction-validation before any editing: incremental validity beyond keyword/length/position baselines;
probe AUROC on held-out; lexical-disjoint + domain-shifted eval; cross-construction cosine (do C1/C3
agree?). **Hypothesis to test directly**: does a C1 (refusal-style) direction for a reasoning behavior
make amplify work where C2 failed? (Separates construction from behavior type + gain-vs-bias.)

## 3. Amplify done right (Fable + codex)
Separate **which weights (support S)** from **which operation**. A "knob" needs ONE shared support +
ONE signed direction, both sides evaluated in a local trust region before curvature/damage. Operations,
ranked:
1. **Suppressor removal** `s⁻=[−r·W·Δμ]_+`, zero top-ρ — amplify via the *removal* op BLADE is reliable
   at; directly tests the "we omitted suppressors" hypothesis (Lee DPO/toxicity ICLR'24). Cheapest — do
   first.
2. **Sparse edit restricted to S along a learned direction**: signed `θ(β)=θ0+βδθ`, β∈{−b..0..+b},
   δθ from a constrained least-squares/Fisher/LoRA objective targeting the POST-RMSNorm projection onto
   v with a KL-preservation trust region — vs a random-mask-of-equal-size control. Most principled
   sufficiency test.
3. Raw `αW` on S — **negative baseline only** (arbitrary origin, superposition, OOD).
4. Rank-1 `W+=λv uᵀ` — **upper-bound reference only** ("steering baked into weights"); NOT localization,
   and note it's a ROME-style edit, not "task arithmetic" (which is a checkpoint diff, Ilharco ICLR'23).
**Re-select ELS for amplify separately** (refusal: removal L12 ≠ amplify L14; Hase NeurIPS'23:
localization ≠ where edits succeed). If different supports are needed up vs down → report a **push–pull
circuit** ("two actuator sets"), not "one knob"; quantify support overlap + cross-direction transfer.

## 4. Metrics & headroom (Fable)
- **Frequency vs intensity separately**: fraction of responses with ≥1 event (frequency) AND events per
  1k tokens / per-event length (intensity). The keyword↑-but-semantic-flat split under amplify is an
  intensity/frequency *dissociation* — a result, not a null.
- **Functional consequence** per behavior (accuracy, calibration/ECE, answer-change-after-backtrack) —
  not marker presence.
- **Capability**: held-out **thinking-trace ppl + task accuracy + boxed-answer rate + trace length**
  (NOT WikiText — it missed the α=4 collapse).
- **Headroom**: evaluate removal on high-base-rate prompts, amplify on **low-base-rate** prompts
  (else amplify has no room). Autoregressive feedback: measure first-occurrence hazard (position-
  controlled) separately from persistence.

## 5. Activation steering = SCREEN, not gate (Fable + codex)
Signed steering (±c·v̂ at ELS layers, c in units of the natural std of ⟨h,v̂⟩) is a good first screen
for "a linear mediator exists," but: a steerable direction does NOT guarantee a weight edit exists
(steering adds a position-constant bias; a weight edit is input-gated Δh=ΔW·φ(x,t) — needs a key/gate),
and steering FAILURE does not rule out weight control (readers/gates/routing). So it's reported, never a
stop-rule. Steering can also be a **bypass** (supplies a signal the model never computes).
**Mechanism-equivalence battery** (required before "same mechanism"): representational alignment (edit
induces Δh∥v with low orthogonal energy); **dose correspondence** (induced signed projection onto v,
not edit magnitude, predicts effect across prompts/doses — the single most diagnostic plot); mediation
(project out/clamp v after the edit → effect vanishes); rescue (matched steering restores after
suppressive edit); interchange patching (edited→base reproduces, base→edited attenuates); downstream
trajectory/distribution equivalence closer than equally-effective controls.

## 6. Verdict classes (report honestly, don't force "knob")
(A) graded removal in [0,1] + bounded amplify in a local trust region on the SAME support,
mechanism-equivalent ⇒ "S is a control axis for X"; (B) graded removal only ⇒ "graded gain on X's
expression"; (C) threshold removal only ⇒ "necessary component"; (D) up needs a different support ⇒
"push–pull, two actuator sets"; (E) activation-steerable, no weight edit reproduces ⇒ "linearly
steerable, not sparsely weight-localized."

## 7. Behaviors (Fable + codex)
- **Refusal = calibration/positive control that ANY criterion must pass first** (Arditi: two-sided
  activation control + removal-oriented rank-1 weight edit; it is inverted-U, so a monotone-to-α=3
  criterion is disqualified).
- **Uncertainty = debug positive control, but split the construct**: expressed (hedging language) vs
  epistemic (calibrated confidence vs evidence strength) vs reasoning (competing-hypothesis
  maintenance). Vary ambiguity/evidence, measure calibration+accuracy — surface hedging control is
  uninformative about "thinking."
- **First AMPLIFY test = example-testing/verification** (low base rate → headroom, checkable functional
  consequence), or backtracking **conditional on planted recoverable errors** (measure effective
  path/answer change, not "Wait" spam). No behavior declared inherently remove-only on current evidence.

## 8. Design & model
Phase A: pre-register constructs, primary continuous + thresholded metrics, ceiling/floor exclusions,
min effects, capability equivalence margins, claim type. Fresh test problems (avoid winner's curse from
picking uncertainty because it worked). Split by problem/template/lexical-family/domain into
direction-train / layer-select / edit-train / untouched-eval. Phase B: build C1/C2/C3 directions +
validate. Phase C: activation dose-response (screen). Phase D: weight ops §3 with §3 controls +
mechanism battery §5. Phase E: verdict §6. **Model: Qwen3-8B first** (cheap, ELS-tested, keyword-amplify
already positive), then 14B; validate the criterion on **refusal** before applying to thinking behaviors.

## 9. Wording bounds
"Located weights controlling X" ⇒ verdict (A) only. Else use "necessary component" / "graded gain" /
"push–pull actuators" / "steerable not weight-localized." Never conflate necessity with control, keyword
with semantics, or expressed with epistemic uncertainty. Refusal's rank-1 removal edit is not evidence
of bidirectional WEIGHT control.
