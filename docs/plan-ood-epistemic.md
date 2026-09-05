# Plan v2 — OOD transfer of the scheme-A epistemic-uncertainty BLADE edit
## (rewritten after Fable-5.1 review; codex review pending — CLI is 401-blocked, retry after re-auth)

Does the direction/weights, built on our TEMPLATED certain/uncertain pairs, control uncertainty on
distributions we never fit? Qwen3-8B primary, 14B confirm. Full review: `scratch_ood_fable.md`.

In-distribution status (Qwen3-8B): direction LOFO probe AUROC 1.000 (length-controlled, but 1.000 at
EVERY layer L10-27); BLADE-G ELS->L*=[23,16]; REMOVE 0.5% -> lexical-hedge on unanswerables 0.564->0.05
(random/shuffled-r/20x-damage all null, Fisher p~1e-6); AMPLIFY bidirectional intensity control on the
warranted regime (0.56->0.79) but NOT significant yet (suppressor p=0.051, raw-alphaW p=0.24) and cannot
inject on knowns (input-gated boundary, kept out of scope).

## 0. THREE conflations this plan must prevent (Fable; the prior random-matches-BLADE failure is NOT the
main risk here since random is already ruled out in-distribution):
1. **Selected-on-test in-distribution anchor.** ELS best-first minimises `unc_rate(unc_ev)` on the SAME
   39 prompts the 0.564->0.05 headline is reported on; controls reuse that split. No untouched set exists.
2. **Metric drift fit->OOD.** The lexical `UNC_MARKERS` is nonexistence-heavy ("fictional","does not
   exist","no such") — it scores the CORRECT answer "Wakanda is fictional" as hedging (3/5 families are
   nonexistence), and has NO entry for OOD hedge forms ("as of my knowledge cutoff", "not stated in the
   passage", "the premise is false"). A null OOD transfer would be uninterpretable (no transfer vs wrong
   dictionary). GEN_TOKENS=64 also floors longer OOD answers.
3. **Shared decline circuit.** Refusal and epistemic abstention plausibly share a late instruction-tuned
   "decline/caveat" writer. §3.3's cross-behavior control has no pre-registered interpretation, and the
   Qwen3-8B refusal selection it needs DOES NOT EXIST yet (only 4b/llama/gemma/phi).

## 1. Claim (pre-registered) + wording
"BLADE weights selected on templated epistemic pairs give SELECTIVE GRADED REMOVAL of expressed
abstention that transfers to UNSEEN uncertainty distributions, beyond random/shuffled/wrong-layer/
cross-behavior/verbosity controls." NOT "necessity" (no rescue/mediation yet), NOT "the uncertainty
mechanism". Current honest construct label = **"expressed abstention on unanswerable/nonexistent-entity
questions"**; it is UPGRADED to "epistemic uncertainty" only if axis B (family-OOD) shows transfer across
nonexistence <-> non-nonexistence families. Injection-on-absent stays out of scope. AMPLIFY = exploratory
until powered.

## P0 — re-validate in-distribution BEFORE any OOD run (~1 GPU-day; prerequisite, not a control)
1. **Untouched split.** Re-split into direction-train / ELS-select / **untouched-eval**, or generate a
   fresh ~40+40-per-family set with NEW entities (existing fictional-country/made-up-title pools are
   nearly exhausted at 176 rows). Re-report REMOVE + rho-sweep + AMPLIFY on the untouched set. ELS stays
   on its own screen set. Bring in-distribution n up to >=150/class (so the OOD/in-dist transfer RATIO
   isn't SE-dominated; at n=39 a control CI is +-16pt, so "controls <5pt" is currently unfalsifiable).
2. **Blind 4-way judge as PRIMARY metric**, applied identically in-dist and OOD: {appropriate
   abstention/caveat, confident-correct, confident-wrong (hallucination), hedged-wrong}. Judge sees
   neither condition nor template. Validate vs ~100 human labels AND vs the lexicon (to show where the
   lexicon fails). Lexicon demoted to secondary; extend it with cutoff/context/premise phrases.
3. **Report REMOVE's outcome on `capital` (fictional) items**: does it confabulate a capital or just say
   "fictional" less? (establishes the hallucination endpoint in-distribution).
4. **Build the Qwen3-8B refusal BLADE-G ELS selection.** Pre-register: cross-behavior transfer >=50% of
   the epistemic effect in EITHER direction (refusal-weights on epistemic eval; epistemic-weights on
   AdvBench) => "shared decline circuit", reported as NON-specific (§0 negative).
5. **Wrong-layer BLADE control** (identical BLADE-G at matched sparsity on non-selected AUROC-1.0 layers,
   e.g. [10,27]): if they transfer as well as [23,16], ELS "localizes" nothing — drop the word.
6. **Verbosity/answer-forcing control**: a direction from a caveat-style contrast ("answer in one
   sentence" vs "answer carefully with caveats") through the SAME BLADE-G pipeline; report answer length
   under every condition. If it also cuts hedging, the epistemic label isn't earned.
7. **Amplify**: power it in-distribution (n>=150, require p<0.01) or demote to exploratory; replicate the
   rho-sweep for the graded-removal claim (current sweep 0.31->0.13->0.05->0.15 reads as noise at n=39).

## 2. OOD axes (fit frozen on templated pairs; NEVER refit on target). REORDERED per Fable.
**Axis B FIRST (cheapest, fully controlled, decides the construct label):** leave-one-FAMILY-out for the
WEIGHT EDIT, 5 folds, on the untouched split, judge metric, with random/shuffled-r/wrong-layer controls
IN EVERY FOLD. Clean win = held-out-family abstention drops >=15pt with confident-wrong rising, in every
fold incl. `capital` and `event_year` (future, no nonexistence) held out, controls flat. This alone
answers "nonexistence detector vs epistemic uncertainty" and is ~5x the current run.

**Axis A (dataset OOD, primary headline)** — external offline benchmarks: SelfAware, SQuAD 2.0
unanswerable, FalseQA, FreshQA/RealTimeQA. For EACH set, FIRST report direction-transfer AUROC of the
frozen v on that set's answerable-vs-unanswerable labels; **exclude a set from the pooled primary if
AUROC<0.75** (a direction-transfer failure, reported as such, so a weight null is attributable). Then
REMOVE/AMPLIFY vs controls. Filter each set into base-abstains (REMOVE headroom) vs base-answers (AMPLIFY
headroom); report separately, don't pool. Headroom rule: exclude a set from REMOVE primary if base in
[0,0.2], from AMPLIFY if base in [0.8,1].

**Axis C (regime OOD)** — answerable-but-hard uncertainty (MATH/GSM8K the base model is unsure of);
endpoint = calibration (ECE, risk-coverage AUC, selective accuracy) via eval_calibration.py. Run BOTH
thinking-OFF and thinking-ON (the edit was fit thinking-OFF/64tok; calibration harness is ON/4096 — don't
stack regime shifts silently). Confirm the harness points at the scheme-A weights, not the keyword-span
reasoning ELS it was written for. A NULL here = informative scope result ("unanswerability detector, not
general uncertainty"), not a failure.

**Axis D (model OOD)** — repeat the whole recipe on Qwen3-14B; if time, one other family (Gemma/Llama).

## 3. Controls (every axis, matched sparsity/damage): random x3, shuffled-r, damage-matched random,
wrong-layer BLADE, cross-behavior (refusal), verbosity-direction, + steering-OOD reference (±c·v̂ at
L16/L22 on each set: steer-transfers-but-weights-don't => verdict E; neither => templated-specific;
both => weights do what the direction does — this partition makes a null informative). Behaviour
preservation: score the ANSWERABLE half of every set (REMOVE must keep accuracy; AMPLIFY must not hedge
on items the model gets right) — reuse blade_abstention.py's two-axis SelfAware frame.

## 4. Statistics: ONE pre-registered primary (axis A pooled REMOVE gap vs max-control, rho=0.005,
L*=[23,16], judge abstention on unanswerables, cluster bootstrap by passage/category, n>=300/set);
everything else secondary/descriptive. Cluster by entity+family in-dist, by passage/category OOD.
Multiple-comparison discipline (>100 candidate tests). Add T=0.7 x>=3 seeds for primary sets or state
greedy + prompt-clustered bootstrap.

## 5. Phasing: P0 (above) -> P1 axis B -> P2 axis A -> P3 axis C -> P4 axis D. Honest-negative
publishable at each gate. Most-endangering test = Qwen3-8B cross-behavior specificity (P0.4): run EARLY;
if refusal weights cut epistemic hedging ~as much, re-scope before spending judge budget.

## 6. Wording bounds: "selective graded removal" not "necessity"; "expressed abstention on unanswerable
questions" until axis B licenses "epistemic uncertainty"; keep remove(selective/graded) and
amplify(graded gain on warranted regime, exploratory) separate; injection-on-absent out of scope;
"localizes to [23,16]" only if the wrong-layer control fails.
