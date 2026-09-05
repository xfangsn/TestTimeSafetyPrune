# Plan v2 — OOD transfer of the scheme-A epistemic edit (rewritten after codex review)
## Does control of CLOSED-BOOK parametric-knowledge (un)certainty transfer to real benchmarks?

codex review: `scratch_ood_uncertainty_codex.md`. Verdict: closed-book scoping + semantic
confident-wrong endpoint are right; the danger is now (a) pooling several distinct "reasons not to
answer" into one latent uncertainty, and (b) using base-vs-edit significance to imply BLADE-vs-random
specificity. Lead with a matched minimal-pair crossover, not a benchmark sweep.

## 0. Established (Qwen3-8B, untouched splits): direction LOFO 1.000 (beats surface 0.915); REMOVE
unanswerable abstain 0.63->0.02 (McNemar p=7.5e-9; semantic: 25/38 flip to confabulation; random/
shuffled/20x-damage null); AMPLIFY 0.63->0.93 (p=1.2e-4, dose-monotone; known 0.00; ppl flat) =
bidirectional intensity control; crossover => regime-specific to closed-book parametric abstention
(inert in passage regime; not a novelty detector). Weight overlap w/ refusal Jaccard 0.06.

## 1. Claim (pre-registered), scope, wording
"A frozen sparse BLADE edit fit on templated closed-book pairs gives SELECTION-SPECIFIC bidirectional
control of warranted vs unwarranted factual commitment on UNSEEN closed-book benchmarks — REMOVE raises
unwarranted definite commitments on items the base withheld; AMPLIFY raises correct withholding/premise-
rejection — beyond matched-random AND an active decline/directness control." Datasets are "external to
edit FITTING", not "unseen to Qwen3 pretraining" (SelfAware/FalseQA predate it; dedup vs the templated
bank, report hashes, add a small sealed post-freeze set). Scope = closed-book parametric; NOT the
uncertainty mechanism, NOT localization, NOT context/evidence-sufficiency, NOT injection-on-absent.

## 2. Benchmarks — distinct constructs, dataset-specific correct-response policy (NEVER one binary)
| dataset/type | appropriate | harmful commitment |
|---|---|---|
| world/fact unknown (KUQ; SelfAware fact-like strata) | withhold the requested fact (calibrated explanation ok) | state a definite value/entity/date without warrant |
| subjective/underdetermined (SelfAware) | state dependence/discuss conditionally | claim a unique objective answer |
| FalseQA | reject/correct the false premise | ACCEPT the false premise & fill its slot |
| answerable / true-premise | give the correct answer | abstain / reject true premise / answer wrong |
- **Name the 3rd benchmark exactly: KUQ** (Known-Unknowns, ACL'24-findings; has uncertainty reasons).
- **SelfAware**: stratify its 5 causes; the fact-like / no-consensus / future-fact strata are the
  claim-aligned primary; subjective/philosophical are a heterogeneity secondary. NEVER score a nuanced
  discussion as hallucination.
- **FalseQA**: premise correction = premise REJECTION (confident-correct), NOT abstention; failure =
  false-premise ACCEPTANCE. Its true-premise minimal pairs are the preservation control.
- EXCLUDE passage/reading-comprehension (SQuAD2) except as a labelled negative-boundary replication.
  FreshQA/temporal secondary only. Also report base-model CORRECTNESS per item (gold-unanswerable !=
  model-doesn't-know); define base-known via a frozen independent screening protocol.
- Prompt: neutral question-only is the clean primary; an "abstain if unsure" instruction is a separate
  robustness condition (else we test cue-compliance).

## 3. THE decisive first experiment (codex): FalseQA false/true-premise minimal-pair crossover
For each untouched pair, generate {false-premise Q, minimally-revised true-premise Q} under base, BLADE
REMOVE, BLADE AMPLIFY (raw-alphaW x2 primary), and operation-matched control DISTRIBUTIONS. Locked blind
rubric using the supplied rebuttal + true answer. Metrics:
- REMOVE harm (false Q): new false-premise acceptance / wrong slot answer.
- AMPLIFY benefit (false Q): new correct premise rejection.
- paired PRESERVATION (true Q): correct answering + non-rejection of the true premise.
- REMOVE selectivity = Δ(false-premise acceptance) − Δ(error/decline on the true partner);
  AMPLIFY selectivity = Δ(correct rejection) − Δ(erroneous rejection on the true partner);
  SPECIFICITY = each BLADE selectivity contrast − the same for matched-random masks.
Decisive because it is external, closed-book, human-written, topic/wording-matched, and powered; it
separates selective warrant-sensitivity from a generic answer/decline actuator. Endangering: random-mask
transfer, equal effect on true premises, or a pure length/style change collapses the claim.

## 4. Endpoints & judge (blind, multidimensional; primary = FUNCTIONAL commitment, not hedge words)
Orthogonal axes: response-act {answer-value / explicit-withhold / premise-reject-or-correct /
conditional-discussion / mixed} x qualification {unqualified-definite / calibrated / strongly-uncertain}
x factual-status {correct / wrong / not-verifiable / no-commitment} x FalseQA-premise {accept / reject-
correct / reject-wrong} x integrity {ok / truncated / degenerate}. Define "confident-wrong commitment"
BEFORE scoring (unqualified-definite requested answer that is wrong, or accepts a gold false premise).
Judge sees question + gold/rebuttal, blind to condition (opaque ids, no side-by-side). Validate vs >=2
blind humans per dataset+class; report classwise precision/recall/confusion; human-adjudicate all
primary BLADE/base discordances; freeze judge+prompt+version; track truncation.
Report per benchmark: full base->edit transition table; conditional conversion in a FIXED opportunity
set (locked once from base, identical denominator for BLADE and every control); UNCONDITIONAL absolute
risk difference (anchors the pooled primary — headroom-robust); answerable accuracy + base-correct
retention.

## 5. Controls (the crux) + specificity rule
- **matched-random masks: >=99** (not 20; min empirical tail 1/(M+1)) OR a prespecified crossed
  prompt×mask hierarchical model. Match joint dist of Q + |W| + per-layer/per-matrix counts + BLADE
  eligibility; operation-matched (REMOVE, raw-alphaW, suppressor each get their own random stratum).
- shuffled-r and label-permuted FULL-pipeline null: several draws (characterize variance) or explicitly
  demoted to qualitative falsification checks.
- wrong-layer = a prespecified DISTRIBUTION of nonselected layer sets (matched layer count/params/
  matrix comp), not one hand-picked pair -> tests layer-selection specificity (not localization).
- **ACTIVE decline/directness control**: a frozen refusal OR verbosity/directness edit under the SAME
  OOD protocol; if it matches BLADE's factual transitions, the effect is a generic assert/decline
  actuator even if random masks are null. (This is the specificity control that random/shuffled cannot
  provide — NOTE: not the dropped cross-behavior 2x2; a single frozen active baseline under the OOD run.)
- **Equivalence rule**: predeclare absolute margins (e.g. control-induced harmful-commitment change
  <=5pt = null band; BLADE−control advantage >=10pt = specificity). Positive REQUIRES: BLADE > min
  effect AND BLADE−matched-random > specificity margin AND control upper bound within its band.
  "BLADE significant / controls n.s." does NOT qualify.

## 6. Statistics
- Primary comparator = BLADE-minus-null on the SAME prompts via a joint prompt-cluster×mask analysis
  (crossed hierarchical binary model / prompt-clustered randomization over masks / hierarchical
  bootstrap). McNemar only as a simple base-vs-edit sensitivity table.
- Cluster by dataset/category/FalseQA-pair; state the unit before opening outputs.
- Aggregate = standardized FIXED-benchmark estimand (per-dataset paired absolute risk difference, fixed
  weights, interval from the joint resample) + always show per-dataset + heterogeneity; do NOT
  equal-weight tiny vs large conditional opportunity sets — anchor on unconditional risk.
- **Bidirectional = CONJUNCTIVE**: reject the broad null only if REMOVE and AMPLIFY each pass their own
  specificity+preservation test; else report one-way transfer. AMPLIFY primary dose = raw-alphaW x2
  (x1.5 support; suppressor-removal secondary). Gatekeeping hierarchy / Holm–max-T over datasets, both
  directions, doses, ops, control families.
- Blinded baseline-only stage to fix headroom/cluster sizes and freeze N + analysis BEFORE interventions.

## 7. Direction-transfer = continuous descriptive only (NOT a gate/moderator at K~3)
Freeze sign/layer/position/template/normalization from training; report signed-projection distributions
+ AUROC with cluster CIs + a SURFACE-only baseline (SelfAware classes differ in source/style) + AUROC by
base-behavior stratum (withholds / hallucinates / correct / wrong). Gold-answerability AUROC != model
knowledge. The 4 quadrants (readout/edit transfer) are diagnostics, not a gating rule; direction success
cannot excuse a weight null and direction failure cannot drop a set from the denominator.

## 8. Phasing (Qwen3-8B; 14B replication after): P1 FalseQA minimal-pair crossover (§3) + active decline
control + >=99 random masks. P2 KUQ + SelfAware fact-like strata + answerable preservation. P3 pooled
fixed-benchmark meta-analysis + equivalence tests. P4 Qwen3-14B replication (refit, frozen dimensionless
rules). Honest-negative publishable at each gate.
