# Plan v3 — construct-validity + OOD transfer of the scheme-A epistemic edit
## (rewritten after BOTH reviews: Fable-5.1 `scratch_ood_fable.md`, codex `scratch_ood_codex.md`)

codex reframed the goal: what we have is **promising sparse, selection-specific causal control of an
expressed ANSWER DISPOSITION, not yet a localized epistemic-uncertainty mechanism**. The next experiment
is NOT a benchmark sweep — it is a held-out counterfactual evidence-sufficiency crossover that breaks the
confound baked into our generator.

## 0. FACTUAL CORRECTIONS to our own prior claims (codex; must fix before citing anything)
- **"length-controlled" is FALSE.** We never matched/residualized length. Within-family length-AUROC is
  1.0 for capital/person_attr/quantity; only event_year/authorship are length-uninformative. Correct
  statement: "non-length ONLY on the two length-matched families," not globally.
- **The validated direction ≠ the edited direction.** epistemic_direction.py (LOFO probe) uses
  thinking=ON; blade_epistemic_els.py recomputes the direction thinking=OFF; the steering screen is
  thinking=ON/512tok/decode-only; the weight edit is thinking=OFF/64tok/all-positions. These are THREE
  regimes — not one validated mechanistic chain. P0 must rebuild + validate the EXACT thinking-off,
  fold-specific direction the weights are scored from, and run steering with that same direction/regime.
- **Wrong test.** base vs edited are PAIRED -> McNemar (exact) / paired cluster bootstrap, not Fisher.
- **Denominator.** "0.5% of weights" = 0.5% of eligible residual-writer entries in the selected layers
  (671,089 scalars), not 0.5% of 8B params. State it.
- **Labels not uniformly unanswerable.** Fictional settings can have canonical capitals; made-up titles
  can collide with real works; premise-denial is a CONFIDENT-CORRECT answer, not hedging. Audit labels;
  define the correct response per item.
- Amplify (suppressor p=0.051, raw-alphaW p=0.24) and the input-gated boundary are HYPOTHESES, not
  results; the rho-sweep rebound (0.05->0.15 at rho=0.02) means "graded" is unproven.

## 1. Claim taxonomy (tightened; codex §9)
Licensed after a clean P0: **"selective sparse causal control of expressed answer disposition on
held-out prompts, vs matched-null edits, with measured effect on unsupported commitment and bounded
capability change."** NOT "necessity", NOT "graded" (needs independent dose trend), NOT "bidirectional
axis" (raw-alphaW and suppressor-removal use different supports -> at most a "push-pull pair of sparse
actuators"), NOT "localized mechanism" (beating one wrong-layer pair is insufficient), NOT "epistemic
uncertainty" (needs the §4 evidence-sufficiency crossover + a correctness/calibration consequence),
NOT "input-gated boundary" (confounded by thinking-mode/tokens/layer/position). Injection out of scope.

## 2. P0a — repair factual + measurement foundation (before any transfer test)
1. Audit + correct item labels; per-item correct-response policy; drop/rescore fictional items with
   plausible canonical answers.
2. Rebuild the EXACT thinking-off direction used by BLADE; re-run LOFO with NESTED surface controls
   (length + tense + bag-of-tokens/char-ngrams + entity-frequency proxy + family) — report incremental
   held-out validity of activations over that baseline, not just "beats a length-only AUROC".
3. Decide ONE of: **Design A** (freeze [23,16]/rho/BLADE-G/thresholds as pilot decisions, evaluate ONCE
   on a fresh versioned confirmation bank, never reselect) OR **Design B** (nested: dir/moment-train ->
   inner ELS/hyperparam -> outer untouched eval; report selected-layer/mask STABILITY as part of the
   localization claim). Do not mix (v2 mixed them).
4. Lock a MULTIDIMENSIONAL judge (codex §5), orthogonal axes: disposition {direct / explicit non-answer /
   premise-correction / mixed} x epistemic-language {none / calibrated caveat / strong} x factual-status
   {correct / wrong / unsupported / n.a.} x task-appropriateness x output-integrity {ok / truncated /
   degenerate}. Judge blind to condition/mask/family/hypothesis, NOT blind to the prompt+reference.
   Develop rubric on pilot outputs; validate with >=2 blinded human raters on a stratified sample
   (oversample rare + base/edit-discordant cells); report classwise agreement/confusion; then LOCK.
   Functional endpoint = rise in UNSUPPORTED/WRONG commitments among prompts the base appropriately
   withheld — NOT reduced hedge language. Track truncation, not just bigger GEN_TOKENS.
5. Confirm paired REMOVE + rho trend on the fresh bank with McNemar/paired-bootstrap; state denominator.

## 3. P0b — cheap specificity threats (run EARLY; cheapest danger checks)
- **2x2 own/cross refusal** (build the missing Qwen3-8B refusal BLADE-G selection): {epistemic,refusal}
  mask x {epistemic,refusal} eval, test the INTERACTION at matched realized damage; report mask overlap
  + cross-projection. Behavioral cross-transfer => "behaviorally non-specific output control", NOT
  automatically "shared circuit". Predefine an equivalence margin (drop the crude ">=50%" rule).
- **Label-permuted full-pipeline null**: permute certain/uncertain labels within strata, recompute
  r/moments/scores/ELS under the same budget (preserves fitting opportunity, destroys the relationship).
- **>=20-50 matched random masks** (not 3): match per-layer/per-matrix counts + Q distribution + fan-in/
  out; report as a distribution, treat prompt AND mask as uncertainty sources. Damage-match on separate
  validation data; use an equivalence margin, not "n.s.".
- **Layer-selection distribution** (not one wrong pair): score all single layers / a prespecified matched
  layer-pair distribution on the selection split; a nested permutation of the whole ELS pipeline is the
  significance reference. If selected layers win -> "ELS identifies more edit-effective layers", not
  "the mechanism localizes".
- **Verbosity/directness mask** from matched completions; it falsifies the epistemic route only if it
  ALSO reproduces the functional transition to unsupported/wrong commitments (not merely fewer hedge
  words).

## 4. P1 — THE decisive experiment: counterfactual evidence-sufficiency crossover (codex §11)
New item quadruples, identical question wording where possible, counterbalanced (answer position,
context length, lexical content, premise wording):
  (a) familiar entity + context that CONTAINS the answer;
  (b) familiar entity + matched context that OMITS the answer;
  (c) nonce/fictional entity + context that CONTAINS the answer;
  (d) same nonce entity + matched context that OMITS the answer.
Nonce+context = ANSWERABLE; familiar+omitted = UNANSWERABLE — this breaks knowability<->surface-anomaly.
Fit direction/moments/ELS/mask/judge NOWHERE on these. Evaluate base, fixed BLADE, many matched random
masks, label-permuted pipeline, verbosity mask, (refusal mask if budget) under identical decoding.
**Pre-registered paired DiD primary**: BLADE-specific drop in appropriate non-answering when evidence is
ABSENT minus its effect when evidence is PRESENT, with NO interaction that merely tracks entity novelty;
require evidence-present accuracy equivalence + evidence-absent unsupported-commitment rise; measure
edit-induced projection onto v in all four cells. STRENGTHENS if the edit follows evidence availability
within the same entity and mediates via v-projection; ENDANGERS (=> anomaly/decline/directness
controller) if it follows nonce wording regardless of evidence, or is reproduced by verbosity/refusal.
Then: nested 5-family LOFO (leak-free: held-out family absent from dir/moments/scores/lambda/ELS/rho/
judge) as evidence of transfer across the 5 generators — reported as strata with family fixed effects,
NOT n=5 iid, and NOT used alone to rename the construct.

## 5. P2 — external OOD (only after P1). Freeze all benchmark memberships + rules FIRST. Include EVERY
prespecified set (SelfAware, SQuAD2-unanswerable, FalseQA, FreshQA/RealTimeQA); direction-AUROC and base
headroom are MODERATORS/transition-strata, NEVER exclusion gates. Per-dataset correct-response policy +
scorer (premise-correction rewarded; SQuAD2 must force context-only; freshness needs snapshot/ref date).
Report the paired per-prompt transition table. Primary = BLADE vs a well-estimated matched-null
distribution; equal-weight dataset-level meta-analysis (don't let SQuAD2 item count dominate); Holm/max-T
over a SHORT key-control family; equivalence tests (±margin) for preservation, not "n.s.".

## 6. P3/P4 — scope + replication
- Axis C (regime/calibration): rebuild around a confidence signal NOT reducible to the same edited
  verbalized-caveat output (normalized answer likelihood/margin, or validated selective-prediction);
  report accuracy and confidence-discrimination separately; Brier/log-score over bin-ECE; pass the EXACT
  scheme-A artifacts to the harness (not the keyword-span reasoning ELS it was written for); run thinking
  OFF and ON. Difficulty band from separate data, not the eval responses.
- **Rename Axis D "cross-model REPLICATION"** (14B refit is reproducibility, not weight transfer);
  predefine which hyperparams are fixed in dimensionless terms; report mask/layer stability.

## 7. Statistics (codex §7): paired estimands carried through ONE joint resample across all conditions;
crossed prompt x mask hierarchical bootstrap for random masks; real power from the expected discordance
matrix (not independent-proportion SE); avoid OOD/in-dist RATIO as primary (noisy denominator) — report
absolute paired effects; pre-register sequential-gate decision rules + which data stay untouched after
each gate; all already-observed results are pilot/exploratory (a preregistration now cannot make them
confirmatory).
