# Plan — OOD transfer of the scheme-A epistemic edit (for codex review)
## Does control of CLOSED-BOOK parametric-knowledge (un)certainty transfer to real benchmarks we never fit on?

Focused successor to plan-ood-epistemic v3, now informed by finished in-distribution + construct results.

## 0. What is already established (Qwen3-8B, all PILOT->now confirmed on untouched splits)
- Direction: refusal-style diff-of-means at last prompt token, thinking-OFF (the regime the edit uses),
  re-validated LOFO diffmeans AUROC 1.000 (L15), beats surface bow+len 0.915; non-length on the 2
  length-matched families.
- REMOVE (untouched 3-way entity split, ELS on its own select set, paired McNemar): unanswerable
  abstain 0.63->0.02, McNemar 28/28 flips p=7.5e-9, known 0.00; blind semantic judge: abstain 0.83->0.26,
  25/38 base-abstained items flip to confident CONFABULATION; answerable answering preserved 0.94->0.98.
  Controls (random x3, shuffled-r, 20x-damage random) all null.
- AMPLIFY (powered, untouched, McNemar): same L*, unanswerable abstain 0.63->0.93 (raw-alphaW x2
  p=1.2e-4, dose-monotone; suppressor-removal p=1.2e-4), known 0.00 throughout, ppl flat. Bidirectional
  intensity control. Injection-on-known out of scope (weights can't; needs steering).
- Construct pinned by the evidence-sufficiency crossover: the edit is REGIME-SPECIFIC to closed-book
  parametric-knowledge abstention — it does NOT touch context/evidence-sufficiency abstention (DiD=0),
  and it is NOT a pure novelty/fictional-entity detector (same nonce entities keep abstaining with a
  passage). Weight overlap with refusal: Jaccard 0.06 (largely disjoint) + small shared late-layer.

## 1. The OOD question + claim (pre-registered)
Claim: "BLADE weights selected on our TEMPLATED closed-book certain/uncertain pairs act as a
bidirectional control of expressed parametric-knowledge (un)certainty on UNSEEN closed-book
answerable/unanswerable benchmarks — REMOVE raises confident-wrong commitments on items the base
abstained on; AMPLIFY raises correct abstention (less hallucination) — beyond matched-null controls that
do NOT transfer." Scope: closed-book parametric only (NOT reading-comprehension/context regime, per the
crossover). NOT "the uncertainty mechanism". Anti-[[ood-selection-negative]]: transfer counts ONLY if
random/shuffled/label-permuted/wrong-layer controls stay near-null.

## 2. Benchmarks (CLOSED-BOOK only; eval-only, fit frozen on templated pairs)
- **SelfAware** (Yin ACL'23): answerable vs unanswerable questions, closed-book (reuse the loader in
  scripts/blade_abstention.py). Primary.
- **FalseQA** (Hu ACL'23): false-premise questions — appropriate response = premise correction/decline;
  closed-book. (Correct-response policy: premise-correction counts as appropriate NON-commitment, NOT
  hallucination.)
- **Known-Unknown / unanswerable factual** (e.g. a SimpleQA/TriviaQA-unanswerable or a curated
  genuinely-unknowable set) + a matched ANSWERABLE factual set (TriviaQA/SimpleQA the model reliably
  gets right) for the preservation axis.
EXCLUDE SQuAD2 and reading-comprehension sets (wrong regime — crossover showed the edit is inert there).
FreshQA/temporal only as a clearly-labelled secondary (borderline construct).
Prefetch all offline on Hazel; report base-model answerability AND base-model correctness per item
(gold-unanswerable != model-doesn't-know).

## 3. Metrics (blind SEMANTIC judge as primary; lexical secondary)
Reuse the P0 multidimensional blind judge (opaque ids, condition-hidden): disposition {abstain / commit /
mixed} x factual-status {correct / wrong / unsupported / n.a.}, per dataset-specific correct-response
policy. Endpoints:
- **REMOVE**: among items the base appropriately abstained on, rate that flip to a confident
  UNSUPPORTED/WRONG commitment (hallucination) — the functional harm.
- **AMPLIFY**: among items the base hallucinated on (base-commit on unanswerable), rate that flip to
  correct abstention (functional benefit).
- **Preservation**: on ANSWERABLE items, base-correct-retained under REMOVE and under AMPLIFY (AMPLIFY
  must NOT start abstaining on answerables — the in-distribution known-stays-0.00 must replicate OOD).
- capability: C4/Wiki ppl (budget) + answerable accuracy.

## 4. Controls (every benchmark, matched sparsity/damage; the crux)
- >=20 matched random masks (per-layer/per-matrix count + Q matched) -> distribution, treat mask as an
  uncertainty source; BLADE vs the random-null distribution is the PRIMARY comparator.
- shuffled-r; label-permuted full-pipeline null (recompute r/moments/scores/ELS on permuted labels);
  wrong-layer BLADE at matched sparsity.
- Direction-transfer FIRST: frozen v's AUROC on each benchmark's answerable-vs-unanswerable labels,
  reported as a MODERATOR (not an exclusion gate); decompose {dir transfers?} x {edit transfers?}.

## 5. Design / stats
- Fit (direction+moments+ELS+mask) ONCE on templated pairs; freeze; all benchmarks eval-only.
- Both REMOVE (rho=0.005) and AMPLIFY (raw-alphaW x1.5 & x2.0 — cleanest; suppressor-removal secondary).
- Paired McNemar per benchmark; cluster bootstrap by dataset/category; equal-weight per-benchmark
  meta-analysis (don't let the largest set dominate); pre-register ONE primary (pooled REMOVE
  confident-wrong-commitment gain vs random-null, judge-scored, on base-abstained unanswerables).
- Headroom rule (report, not exclude): REMOVE needs base-abstain high (SelfAware/FalseQA ok); AMPLIFY
  needs base-hallucination present (report base rates first; some sets may have little AMPLIFY headroom).
- Power from the expected discordance matrix, not independent-proportion SE.

## 6. Honest-negative + wording
Publishable negative: "the closed-book parametric-uncertainty edit is decodable/controllable
in-distribution but does not transfer mechanism-specifically to <benchmarks> (controls match)." Wording:
"bidirectional control of expressed closed-book parametric (un)certainty that transfers to <sets>,
beyond matched-null controls"; keep REMOVE (confident-wrong gain) and AMPLIFY (correct-abstention gain,
capability-preserving) as separate sentences; scope = closed-book parametric; injection-on-absent and
context/evidence-sufficiency explicitly out of scope.

## 7. Phasing (Qwen3-8B; 14B replication after)
P1 SelfAware (primary, both directions + controls + direction-transfer moderator). P2 FalseQA +
known-unknown/answerable preservation. P3 pooled meta-analysis + equivalence tests on preservation.
P4 Qwen3-14B replication (refit, frozen dimensionless rules).
