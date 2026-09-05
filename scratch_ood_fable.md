# Review of `docs/plan-ood-epistemic.md` — OOD transfer of the scheme-A epistemic BLADE edit

Reviewer stance: skeptical NeurIPS/ICLR mech-interp reviewer. I read the plan, the parent plan
(`docs/plan-bidirectional-control.md`), the prior-negative memo (`ood-selection-negative`), and the code
and result files the in-distribution claims rest on:
`scripts/build_epistemic_pairs.py`, `scripts/epistemic_direction.py`, `scripts/blade_epistemic_els.py`,
`scripts/blade_epistemic_controls.py`, `scripts/blade_epistemic_amplify.py`, `scripts/blade_abstention.py`,
`results/blade_epistemic_{els,controls,amplify}_qwen3-8b*.json`, `results/epistemic_direction_qwen3_8b.json`.

Short verdict: the plan has the right *intent* (transfer counts only if matched controls do not transfer),
but as written it can still produce an un-interpretable result — not by the prior failure mode (random
matches BLADE) but by three others that the plan does not guard against: (i) the in-distribution anchor
itself is selected-on-test at n=39 with a lexical metric that counts correct "X is fictional" as hedging;
(ii) the OOD metric will not measure the same thing the edit was fit on; (iii) the most dangerous
specificity control (a generic "abstain/decline" circuit shared with refusal) is listed but has no
pre-registered interpretation, and the Qwen3-8B refusal selection it needs does not exist yet.

---

## 1. Will it establish mechanism-specific transfer, or repeat the prior conflation?

**The prior failure is *not* the main risk here; a different conflation is.** In the sycophancy line,
random pools matched BLADE *in-distribution-to-OOD* because there was no in-distribution signal to begin
with (solo-layer corr(gain_in, gain_OOD) ≈ 0). Here the in-distribution picture is different: BLADE
22/39 → 2/39 (Fisher p ≈ 1e-6) with random ×3, shuffled-r, and damage-matched random at ρ=0.10
(Δppl +1.5% ≈ BLADE's +1.1%) all at 20–23/39. So random-matching is already ruled out in-distribution,
and the refusal precedent (solo-OOD corr +0.95) says a unitary, deployment-aligned contrast can transfer.

The risks the plan does NOT guard against:

**(a) The in-distribution anchor is selected on its own evaluation set.** `blade_epistemic_els.py`
lines 156–166 and 193–201: `split_by_entity` yields `unc_ev` (39 prompts); `measure()` — the ELS
best-first objective — is `unc_rate(model, tok, unc_ev)`; and the reported REMOVE sweep (line 214) is
`unc_rate(model, tok, unc_ev)` on the *same 39 prompts*. L*=[23,16] was chosen to minimise hedging on
exactly the prompts the headline 0.564→0.05 is reported on. The controls script reuses the same split
("same seed -> same split as ELS run", line 56). There is no untouched in-distribution set anywhere.
This is winner's curse at the layer-selection stage; every downstream OOD number inherits an
in-distribution reference that is optimistic by an unknown amount. It also means the "4-fold vs 5-fold"
family-OOD design in §2B is the first time the *weight edit* will ever be evaluated on data it was not
selected on.

**(b) The metric will drift between fit and OOD eval, so a null transfer is ambiguous.** `UNC_MARKERS`
(ELS script lines 53–61) is dominated by nonexistence/fiction phrases: "fictional", "does not exist",
"no such", "isn't a real", "not a real", "no record", "hypothetical", "no widely", "not a recognized".
Three of five training families are nonexistence families (fictional capital, made-up book title,
unknowable count), one is future-event, one is an unknowable attribute. Consequences:
  - On `capital`, the *correct, confident* answer "Wakanda is a fictional country" is scored as a
    hedge. Part of the 0.564 base rate is not hedging at all; it is knowledge-denial. What REMOVE does
    on those items is unknown from the reported numbers (does it confabulate a capital, or say
    "fictional" less?). The plan's hallucination endpoint (§2A) is therefore *not* established
    in-distribution either.
  - The lexicon has no entry for the dominant OOD hedge forms: "as of my knowledge cutoff" / "as of
    my last update" (FreshQA/RealTimeQA), "the passage does not mention / not stated in the context"
    (SQuAD 2.0), "the premise is false / the question assumes" (FalseQA), "there is no scientific
    consensus / this is subjective" (SelfAware). If the lexical metric is used as the pre-filter or
    the primary, OOD "transfer" will be measured against words the edit never touched, and a null is
    uninterpretable (no transfer vs. wrong dictionary).
  - GEN_TOKENS=64 with thinking disabled. SQuAD2 with a passage and SelfAware philosophical
    questions produce longer, later-hedging answers; a 64-token window will floor the metric.

**(c) Position/format shift for the direction itself.** The direction is a last-prompt-token
diff-of-means on 16–24-token closed-book questions with no system prompt. SQuAD2 puts a ~150-token
passage before the question; FreshQA questions are long and dated. The residual at the last prompt
token in those contexts is a different distribution; the edit's *gate* (ΔW·φ(x,t) needs the key) may
not fire. That is a legitimate OOD test, but the plan should first check that the *direction* transfers
(probe/projection AUROC of the frozen v on the external set's answerable-vs-unanswerable labels) before
asking whether the *weights* transfer. Without that step, a weight-transfer null cannot be assigned to
"weights" vs "direction" vs "metric."

**(d) The "generic abstention circuit" outcome is not pre-interpreted.** §3.3 runs refusal weights on
the epistemic eval and vice versa, but says nothing about what to conclude if the cross-behavior
control *does* transfer. On an instruction-tuned model, refusal and epistemic abstention plausibly share
a late "decline-to-answer / produce a caveat" writer. If refusal-BLADE weights cut hedging on SelfAware
by 20 pt, the epistemic-transfer story collapses into "we found the abstention style circuit" — which
is exactly the mechanism-nonspecific verdict this plan exists to avoid. Conversely if epistemic weights
cut AdvBench refusal, the same. The plan must state now whether that outcome is a positive (a shared
mechanism is still a mechanism) or a negative (not epistemic), and what "specificity" number counts.

Bottom line for Q1: the plan *can* establish mechanism-specific transfer, but only if (a) an untouched
in-distribution set is added, (b) the primary OOD metric is a judge with a fixed 3-way label schema
applied identically in-distribution and OOD, (c) direction-transfer is tested before weight-transfer,
and (d) the cross-behavior outcome is pre-registered in both directions.

---

## 2. Controls: what is present, what is missing

Present and adequate: same-sparsity random ×3, shuffled-r, damage-matched random, cross-behavior (once
the Qwen3-8B refusal selection is built — see below), direction cosine across datasets.

Missing, in priority order:

1. **Untouched in-distribution eval split** (see 1a). Re-split `epistemic_pairs.json` by entity into
   direction-train / ELS-select / untouched-eval, or better, generate a fresh set of ~40 uncertain +
   ~40 certain items per family with *new entities* (there are only 176 rows; the fictional-country and
   made-up-title pools are exhausted by the existing rows). Report all in-distribution numbers on the
   untouched set. This is a prerequisite, not a control.

2. **Same-direction, wrong-layer BLADE** (layer-specificity). The probe AUROC is 1.000 at *every*
   layer from L10 to L27 (`epistemic_direction_qwen3_8b.json`), so the direction is decodable
   everywhere; "localizes to [23,16]" is entirely an ELS-on-hedge-rate result. Run the identical
   BLADE-G scoring at matched sparsity on non-selected layers with AUROC 1.0 (e.g. [10,27]). If those
   transfer as well as [23,16], the ELS step is not finding anything and the paper should not say
   "localizes." If they do not, that is the strongest evidence that the *selected* weights matter.

3. **Steering-OOD reference.** The parent plan says steering is a screen, but for the OOD question it is
   the cheapest reference for whether the *direction* carries OOD: ±c·v̂ at L16/L22 on each external
   set. If steering transfers and weights do not → verdict (E) "steerable, not weight-localized";
   if neither → the direction is templated-specific; if both → weight edit is doing what the direction
   does. This partition is what makes a null informative.

4. **Cross-behavior control needs to actually exist.** There is no `blade_refusal_els_qwen3-8b*.json`
   (only qwen3-4b, llama-3.2-3b, gemma, phi). "already have it" (§3.3) is false for the primary model.
   Budget the Qwen3-8B refusal ELS run before P1.

5. **A "generic directness/answer-forcing" control.** The functional consequence claimed for REMOVE
   is "more confident-wrong answers." A trivial way to get that is to make the model shorter/more
   direct. Include (i) a length/format-matched control: a direction built from a *verbosity* or
   *caveat-style* contrast (same question, "answer in one sentence" vs "answer carefully with caveats")
   fed through the same BLADE-G pipeline; and (ii) report answer length and refusal-lexicon rate under
   every condition. If the verbosity direction also cuts hedging OOD, the epistemic label is not earned.

6. **Behaviour-preservation on answerable OOD items** (over-confidence side of removal, and over-hedging
   side of amplify). §2A only scores the unanswerable half. On SelfAware-answerable / SQuAD2-answerable,
   REMOVE must keep accuracy and AMPLIFY must not start hedging on items the model gets right; otherwise
   AMPLIFY is just "injection on the warranted regime" measured on the wrong subset. `blade_abstention.py`
   already has the two-axis (abstain-up vs accuracy-flat) framing and a SelfAware loader — reuse it.

7. **Seeds and decoding.** Everything is greedy on 39 prompts. Add sampling at T=0.7 with ≥3 seeds for
   the primary OOD sets, or at minimum report that greedy is used and cluster the bootstrap by prompt.

---

## 3. Confounds, statistics, measurement

**Power / thresholds are inconsistent with the sample sizes.** Plan §5 pre-registers "≥15-pt hedge
change, controls <5 pt." At n=39, SE(p̂) ≈ 0.08, so "<5 pt" for a control is unfalsifiable (the CI
is ±16 pt). To bound a control at ±5 pt (95%) needs n ≈ 384 per set; to detect a 15-pt BLADE-vs-control
gap at 80% power needs ~175 per arm (unpaired; paired-by-prompt bootstrap will do better). External
sets are large enough (SelfAware ~1,000 unanswerable; SQuAD2 dev ~5,900 unanswerable), so this is a
budget question, not a data question — but the in-distribution reference must be brought up to at
least ~150 per class or the OOD/in-distribution *ratio* (the transfer number) will have an SE dominated
by the denominator.

**Amplify is not significant in-distribution.** Suppressor-removal 22/39 → 31/39 is Fisher p = 0.051;
raw αW=1.5 22/39 → 28/39 is p = 0.24. The plan carries "AMPLIFY … bidirectional intensity control" into
the OOD claim as if established. It is not. Either (a) drop amplify from the P1 headline and treat it as
exploratory, or (b) power it first (n ≥ 150) on the untouched in-distribution set before spending OOD
budget on it. The REMOVE sweep is also non-monotone (0.31 → 0.13 → 0.05 → 0.15 at ρ=0.02), which a
reviewer will read as noise at n=39 rather than "graded"; a graded-removal claim needs the sweep
replicated at adequate n.

**Clustering.** SQuAD2 has many questions per passage; SelfAware unanswerables come in a few
question-type categories; FalseQA premises repeat templates. Cluster the bootstrap by passage/category,
not by item. In-distribution, cluster by entity and family (the plan says "prefix-clustered" — define
prefix; for closed-book questions the natural cluster is entity).

**Multiple comparisons.** 4 axes × ≥4 benchmarks × 2 ops × ≥2 sparsities × 3 metrics is >100 tests.
Pre-register ONE primary: axis A, pooled across the four external sets, REMOVE at ρ=0.005 on L*=[23,16],
judge-labelled abstention rate on unanswerables, BLADE-minus-max(control) gap with cluster bootstrap CI.
Everything else is secondary/descriptive and should be labelled so in the paper.

**Judge.** Use one blind judge with a fixed 4-way label per response: {appropriate abstention/caveat,
confident-correct, confident-wrong (hallucination), hedged-wrong}. Apply it identically in-distribution
and OOD. Validate it against ~100 human labels *and* against the lexical metric on the training
families so the paper can show where the lexicon fails (it will: "fictional" items). The lexical rate
stays as a secondary. Do not let the judge see the condition label or the prompt template.

**Ceiling/floor.** Base hedge on known items is 0.00 (floor) — amplify on known is out of scope, fine,
but then "graded gain on the warranted regime" must be measured on OOD unanswerables whose base rate is
mid-range. FreshQA/RealTimeQA beyond-cutoff base hedging on Qwen3-8B is probably ≥0.8 → no headroom for
AMPLIFY; SQuAD2-with-passage "not mentioned" rate may also be high. Pre-register a headroom rule
(exclude a set from the AMPLIFY primary if base ∈ [0.8,1]; from REMOVE if base ∈ [0,0.2]) and report
base rates per set before running edits.

**Thinking mode.** In-distribution fit and eval use `enable_thinking=False`, 64 tokens. Axis C
(`eval_calibration.py`) uses thinking ON, 4096 tokens. That is a large regime shift stacked on the
distribution shift; keep it, but say so, and run axis C once with thinking OFF as well so the
"general uncertainty vs unanswerability detector" question is not confounded with "thinking on/off."
Also confirm axis C uses the scheme-A epistemic weights, not the keyword-span reasoning-uncertainty ELS
that `eval_calibration.py` was written for (`--dirs qwen3_14b_dirs.pt --els reasoning_els_...`).

**Contamination / knowledge.** For FalseQA and SelfAware, filter items to those where the base model
*does* abstain (so REMOVE has something to remove) and separately where it *does not* (so AMPLIFY has
headroom), and report both subsets — do not pool.

---

## 4. Claim taxonomy and scope

- **"Necessity" is over-claimed.** Beating damage-matched random shows *selectivity* of the removed
  weights, not necessity (necessity: the behaviour cannot occur without them; you have not shown the
  model cannot re-express hedging via another path, e.g., under steering after the edit — the rescue
  test in the parent plan §5). Use "selective graded removal" for REMOVE. Reserve "necessary" for after
  a rescue/mediation test, or drop it.
- **"Graded gain" for AMPLIFY is not yet supported in-distribution (p=0.05 / 0.24).** Downgrade to
  exploratory until powered.
- **Injection-out-of-scope is right and should stay.** The input-gated vs constant-bias boundary is a
  clean, defensible result; do not let OOD amplify results blur it (an amplify "success" on an OOD
  unanswerable set is still gain, not injection).
- **The construct label needs a caveat now.** With 3/5 nonexistence families and a nonexistence-heavy
  lexicon, the honest current label is "expressed abstention on unanswerable/nonexistent-entity
  questions." Whether it becomes "expressed epistemic uncertainty" is exactly what axis B (family-OOD)
  decides: if the edit built on the four non-`capital` families transfers to `capital` and vice versa,
  and to `event_year` (future, no nonexistence), the broader label is earned. Run axis B *before*
  axis A; it is cheaper, cleaner, and tells you whether axis A is even worth the judge budget.
- **Control-axis (verdict A) is not on the table for this plan** and the wording bounds correctly avoid
  it. Keep it that way; the OOD plan should conclude at most "selective graded removal transfers to
  <sets>; gain transfers to <sets>" as two separate sentences.

---

## 5. What a reviewer attacks first; the single most strengthening and most endangering experiments

**First attack (near-certain):** "Your layers were chosen by minimising a keyword regex on the 39
prompts you then report; your regex counts 'Wakanda is fictional' as hedging; and your hallucination
endpoint was never measured. Everything downstream is built on that." This is entirely fixable before
any OOD run: untouched split + judge + report REMOVE's outcome on `capital` items.

**Second attack:** "Refusal and epistemic abstention share an instruction-tuned decline circuit; your
cross-behavior control will show it and you have no pre-registered interpretation." Fixable by
pre-registering both outcomes and building the Qwen3-8B refusal selection.

**Third attack:** "n=39, greedy, lexical, no clustering, >100 comparisons."

**Single most strengthening experiment:** family-OOD for the *weight edit* (axis B), 5 folds, on the
untouched split, with the judge, and with the same 5 folds run for random / shuffled-r / wrong-layer
controls. A clean result — held-out-family abstention drops by ≥15 pt with confident-wrong rising, in
every fold including `capital` held out, controls flat — is internally consistent, fully controlled,
independent of external-benchmark metric drift, and directly answers "nonexistence detector or
uncertainty." It is also cheap (same pipeline, ~5× the current run).

**Single most endangering experiment:** the cross-behavior specificity test on Qwen3-8B. If refusal
BLADE-G weights (at matched sparsity, their own ELS layers) cut epistemic hedging on SelfAware/FalseQA by
about as much as the epistemic weights — or the epistemic weights cut AdvBench refusal — the claim
reverts to "generic decline circuit" and the plan's own §0 says that is reported as a negative. Run it
early (it is one ELS run + two evals), because if it fails the rest of the plan should be re-scoped
before the judge budget is spent.

---

## 6. Concrete changes to the plan (ordered)

P0 (before any OOD run; ~1 GPU-day):
1. Add an untouched in-distribution eval split with new entities (~40/40 per family); re-report REMOVE,
   the ρ sweep, and AMPLIFY on it. Keep ELS selection on its own screen set.
2. Replace the primary metric with a blind judge (4-way label) applied identically everywhere; keep the
   lexicon as secondary; extend the lexicon with cutoff/context/premise phrases for the secondary.
3. Report per-family outcome of REMOVE on `capital` (fictional) items: confabulated capital vs. dropped
   "fictional."
4. Build the Qwen3-8B refusal BLADE-G ELS selection. Pre-register: cross-behavior transfer ≥ 50% of the
   epistemic effect in either direction ⇒ "shared decline circuit," reported as non-specific.
5. Add the wrong-layer BLADE control ([10,27] or similar AUROC-1.0 layers) and the steering-OOD
   reference to §3.

P1 (reordered): axis B (family-OOD, 5-fold, judge, controls) → then axis A.
- Axis A primary = pooled REMOVE gap vs max(control), ρ=0.005, L*=[23,16], judge abstention on
  unanswerables, cluster bootstrap by passage/category, n ≥ 300 per set. Direction-transfer AUROC of
  the frozen v on each set reported first; if AUROC < 0.75 on a set, that set is excluded from the
  pooled primary and reported as a direction-transfer failure.
- AMPLIFY = secondary until powered in-distribution (n ≥ 150, p < 0.01).
- Score answerable halves of every external set (accuracy under REMOVE; hedge rate under AMPLIFY).

P2–P4 as planned, with axis C run with thinking OFF and ON, and explicitly on scheme-A weights.

Wording bound to add: "selective graded removal" (not "necessity"); "expressed abstention on
unanswerable questions" until axis B licenses "expressed epistemic uncertainty."
