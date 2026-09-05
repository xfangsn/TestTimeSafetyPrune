# Plan — OOD transfer of the scheme-A epistemic-uncertainty BLADE edit
## Does the direction/weights, built on our TEMPLATED certain/uncertain pairs, control uncertainty on
## distributions we never fit? (Qwen3-8B primary, 14B confirm)

For Fable-5.1 + codex review. Builds on the established in-distribution result (all Qwen3-8B):
- direction: refusal-style diff-of-means at last prompt token; LOFO probe AUROC 1.000, length-controlled.
- weights: BLADE-G ELS -> L*=[23,16]; REMOVE 0.5% -> hedging on unanswerables 0.564->0.05, ppl +1.1%;
  same-sparsity random / shuffled-r / 20x-damage random all NULL (0.51-0.59). AMPLIFY (raw-alphaW &
  suppressor-removal) is bidirectional INTENSITY control on the warranted regime (unanswerable
  0.56->0.72/0.79); INJECTION on known facts fails for weight ops (needs steering; input-gated boundary).

## 0. The one thing this plan must prove (anti-[[ood-selection-negative]])
Our prior OOD line died because transfer was NOT mechanism-specific — random weight pools transferred as
well as BLADE. So the DELIVERABLE here is not "the edit transfers" but "the edit transfers AND the
matched null controls (random-direction, shuffled-r, random-weights, cross-behavior weights) do NOT."
Transfer without that separation is reported as a NEGATIVE, exactly as before.

## 1. Claim (pre-registered) + wording bound
Claim = "BLADE weights selected on templated epistemic pairs act as a control axis for expressed
epistemic uncertainty on UNSEEN uncertainty distributions, beyond random/shuffled/cross-behavior
controls." NOT "we found the uncertainty mechanism." Down = necessity/graded-removal; up = graded gain
on the warranted regime (injection-on-absent explicitly out of scope, per the in-distribution boundary).

## 2. OOD axes (fit on templated pairs; NEVER refit on the target)
A. **Dataset OOD (primary)** — external, human-built benchmarks, prefetched offline on Hazel:
   - SelfAware (answerable vs unanswerable), SQuAD 2.0 unanswerable, FalseQA (false-premise),
     FreshQA / RealTimeQA (beyond-cutoff). These replace our synthetic families entirely.
   - Endpoint: on the UNANSWERABLE/false-premise items, does REMOVE cut appropriate hedging and raise
     confident-wrong (hallucination) answers? does AMPLIFY raise hedging? vs base + controls.
B. **Family OOD (clean, internal)** — leave-one-FAMILY-out for the WEIGHT EDIT (not just the probe):
   build direction+moments+ELS on 4 families, evaluate remove/amplify on the held-out family. 5 folds.
C. **Regime OOD** — built on absent-knowledge (unanswerable); does it transfer to ANSWERABLE-BUT-HARD
   uncertainty? Evaluate on MATH/GSM8K items the base model is genuinely unsure of. Endpoint = calibration
   (ECE, risk-coverage AUC, selective accuracy), reusing eval_calibration.py. Tests "general uncertainty"
   vs "unanswerability detector." A NULL here is an informative scope result, not a failure.
D. **Model OOD** — repeat the whole recipe on Qwen3-14B (does ELS localize + edit reproduce?); if time,
   one other family (Gemma/Llama) for architecture transfer.

## 3. Controls (REQUIRED on every OOD axis; the plan's point)
For each transfer test, run at matched sparsity/damage:
1. **random-direction** and **shuffled-r** weights (from the SAME BLADE-G pipeline) — must not transfer.
2. **random-weight** same-sparsity (x3 seeds) + **damage-matched** (random at higher sparsity to match
   ΔNLL) — must not transfer.
3. **cross-behavior** weights: the refusal BLADE selection (already have it) applied to epistemic eval,
   and vice-versa (epistemic weights on AdvBench refusal). Specificity both ways.
4. **direction cosine**: our epistemic direction vs a direction rebuilt ON each external benchmark
   (SelfAware etc.) — do they agree? (convergent validity of the construct across datasets.)
Report BLADE-minus-best-control gap with prefix-clustered bootstrap CIs; transfer counts only if the gap
is significant and controls are near-null.

## 4. Metrics (per axis)
- hedge/abstain rate (freq) + events-per-response (intensity), separated.
- hallucination on unanswerables: confident-wrong rate (executor/judge) — the functional consequence of
  removal. (A judge is needed; reuse llm_judge / kimi when quota allows, with a lexical pre-filter.)
- calibration on answerable sets (ECE, AUROC conf-vs-correct, risk-coverage) via eval_calibration.py.
- capability: held-out thinking-trace ppl + task accuracy (NOT WikiText alone — it missed our alpha=4
  collapse). C4/Wiki ppl as budget only.

## 5. Design / hygiene
- Fit (direction+moments+ELS) once on templated pairs; freeze. All external sets are eval-only.
- Split external benchmarks so nothing leaks; report per-benchmark and pooled.
- Pre-register min effect (e.g. >=15-pt hedge change on unanswerables, controls <5 pt), ceiling/floor
  exclusions, and the calibration equivalence margin.
- Contamination note: GSM8K/MATH may be in pretraining; treat as controlled stimuli, report harder
  Olympiad-style separately if used.

## 6. Phasing (Qwen3-8B first)
- P1 dataset OOD (axis A) + controls (§3.1-3.3) — the headline transfer test.
- P2 family-OOD (axis B) — cleanest internal generalization.
- P3 regime OOD (axis C, calibration) — scope of the mechanism.
- P4 model OOD (axis D, 14B).
- Honest-negative allowed and expected to be publishable: if controls transfer as well as BLADE (as in
  the prior line), report "epistemic direction is decodable/steerable but its BLADE weight-selection does
  not transfer mechanism-specifically OOD."

## 7. Wording bounds
"transfers as a control axis for expressed epistemic uncertainty, beyond matched null + cross-behavior
controls, on <benchmarks>/<models>." Never "the uncertainty mechanism"; keep remove(necessity/graded)
and amplify(graded gain on warranted regime) claims separate; injection-on-absent stays out of scope.
