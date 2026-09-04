# Plan — effect of editing keyword-associated thinking patterns on construct-aligned evaluation (Qwen3-1.7B, BLADE-G)

Revised after TWO adversarial codex reviews (`scratch_eval_plan_review_codex.md`,
`scratch_eval_plan_rereview_codex.md`). Framing decisions baked in: **(1)** rigorous route —
capacity-level language ("the uncertainty/backtracking capacity") is *earned* only after the P-1 gate
passes; the default term everywhere is **"keyword-associated thinking pattern"**, and a passing pattern
becomes a **"construct-validated proxy"**, never a proven latent capacity. **(2)** first & only model =
**fresh Hazel-built Qwen3-1.7B** (artifacts audited, §2). **(3)** the edit (pruning/amplify weight
selection) is **BLADE-G**; **layer selection (ELS) is BLADE-B by design** — we reuse the existing
BLADE-B-selected `reasoning_els_qwen3_17b.json` for *which* layers, then build the *weight* mask with
the generic-importance penalty (BLADE-G). This is intentional and disclosed (avoids the slow
generation-based BLADE-G ELS and keeps layer choice comparable to prior runs). Comparisons are against
random (two regimes) + shuffled controls; no BLADE-B *edit* is tested.
**(4)** the P-1 gate is scoped to only the patterns entering P0 — **uncertainty & backtracking** — and
uses a **validation split disjoint from the P0 test split**.

Core lesson kept from the literature the user surfaced: thinking is evaluated *by dimension*
(uncertainty↔calibration, backtracking↔self-correction, adding-knowledge↔knowledge), with
*faithfulness* overarching. But our edit targets weights chosen from a **keyword** proxy, so we must
first prove the edit changes the *semantic* pattern before mapping it to a capacity.

---

## 0. What we edit vs what we measure
- **Edit (BLADE-G only)**: at a pattern's ELS layers, `remove` (α=0) or `amplify` (α∈{1.25,1.5}). One
  audited mask-builder (see §7), objective stated once: score `S=[c]_+ − λ_eff·Q`, `λ_eff=λ·|α−1|`
  (so α=0 → λ_eff=λ; α=1.5 → 0.5λ), abstain on `S≤0`, and **fail if positive candidates < requested
  k** (no silent back-fill of abstained weights). Serialize the exact selected indices + SHA-256.
- **Controls** (per edited pattern): `shuffled-r` (breaks r↔W, same layers/sparsity) and TWO random
  regimes — **fixed-sparsity** (same layers/#weights) and **fixed-damage** (control sparsity tuned on
  a disjoint calibration split to match ΔNLL, including a math-domain NLL, then tested once). ≥8 mask
  seeds for inferential use (3 was too few).
- **Measure**: the dimension-matched outcomes in §3, plus accuracy, thinking-length, no-answer rate,
  and ΔNLL (general + math-domain) at every operating point.

## 1. P-1 — manipulation check + double dissociation (GATE; run before any capacity claim)
The directions/ELS are keyword-derived (`scripts/qwen3_directions.py`, `scripts/reasoning_els.py`),
which optimizes a *lexical* proxy, not a latent capacity. Before §3, we must show the edit changes the
**semantic** pattern and does so **specifically**:
1. **Frozen manipulation-check set** (separate from eval + direction/ELS prompts): generate base vs
   `remove`/`amplify` traces; **blind semantic annotation** (Opus, per memory; rubric per pattern)
   of whether each trace *actually* hedges / backtracks / recalls-and-integrates — with **keyword-free
   positives** and **keyword-containing negatives** so we score the lexicon's precision/recall, not
   the keywords.
2. **Length control**: confirm the semantic change survives after controlling for trace length
   (removal that only shortens text is not "removing the pattern").
3. **Double dissociation** (the specificity test): run EVERY pattern-edit × EVERY outcome dimension;
   require edit-X to move dimension-X more than (i) other pattern-edits move dimension-X, and (ii)
   edit-X moves unrelated dimensions. Only patterns passing (1)–(3) get capacity-level language in §3;
   others are reported as "keyword-associated" effects.
Deliverable: a validation table (lexicon precision/recall, semantic-change effect size w/ length
control, dissociation matrix). **If a pattern fails, we still report its editing effects, but do not
claim it is "the uncertainty/backtracking/knowledge capacity."**

**Gate rigor (per re-review):**
- **Scope**: P-1 covers ONLY uncertainty & backtracking (the patterns entering P0). Knowledge and
  faithfulness are NOT in the gate (they belong to P1), removing the earlier P-1↔phasing conflict.
- **Split**: the double-dissociation is estimated on a **validation split disjoint from the P0 test
  split** (frozen IDs published), so the same items never both validate the construct and estimate its
  P0 effect.
- **Standardized contrast**: "moves more" is defined on **z-scored (per-metric) effect sizes** with
  **bootstrap 95% CIs**; edit-X passes only if its effect on dimension-X is (i) CI-separated above its
  effect on the other pattern's dimension and (ii) above the other edit's effect on dimension-X.
- **Annotation reliability**: blind Opus annotation with a frozen rubric; report **inter-rater
  agreement** on a re-annotated subset (second pass / second annotator) before trusting the labels.
- **Language rule**: a pass earns "construct-validated proxy for <dimension>", NOT "the <dimension>
  capacity"; a fail stays "keyword-associated pattern".

## 2. Model & artifacts (Qwen3-1.7B, fresh, Hazel /share)
Artifacts live on Hazel `/share/jekml/xfang23/TestTimeSafetyPrune/results/`:
`reasoning_els_qwen3_17b.json` (fresh ELS: all four patterns localised) + `qwen3_17b_dirs.pt`. **Audit
before use**: record path, model id + revision, tokenizer revision, direction span counts, ELS config
(eps/screen/test-frac), and SHA-256; the evaluator must **assert** the loaded direction artifact's
stored model == `--model` and that layer count / tensor shapes / writer names match.
*Known discrepancy to investigate (not blocking 1.7B)*: the older *local* 4B/8B ELS files localised all
four patterns, whereas the fresh Hazel 4B/8B ELS kept only uncertainty — likely an eps/traces/seed
difference; flag it, don't silently rely on either.

## 3. Pattern → construct-aligned evaluation (citations corrected)
| Edited pattern | Dimension (if P-1 passes) | Evaluation | Datasets | Primary metric (+ secondary) |
|---|---|---|---|---|
| uncertainty-estimation | calibration / selective prediction | verbalized **and** consistency confidence; **fixed-answer** + total-system | GSM8K, MATH-500 | **Brier** (primary); + ECE (binning fixed), AUROC, risk–coverage/excess-AURC; P(True)/answer-logprob confidence |
| backtracking | fair-setting self-correction | **crossed** generate/review (see §5) | MATH-500, GSM8K | **net correction benefit** (primary) with separate **helpful (w→c)** & **harmful (c→w)** rates; error-detection on BIG-Bench Mistake |
| adding-knowledge | split: parametric recall **and** evidence integration | closed-book short-answer; + fixed-evidence probe | closed-book QA; evidence set (relevant/irrelevant/contradictory) | recall EM/F1 + claim-level precision; evidence correct-use & attribution |
| example-testing | (self-verification sub-case) | verification-step correctness (not mere frequency) | MATH-500 | verified-step accuracy |
| faithfulness (overarching) | CoT reflects computation | **Turpin** biasing-feature (stated cue) + **Lanham** truncation curve; **restoration-error rate** lives HERE | BBH subset (+ stated cue) | targeted-flip rate + accuracy Δ + audited acknowledgment; same-answer-vs-%CoT-retained curve; restoration-error rate |

## 4. Datasets (frozen; leakage audited)
Freeze exact revision, IDs, prompt template, answer extractor, scoring lib for each. First pass: GSM8K
500, MATH-500 (all), a closed-book multi-hop QA subset + a curated evidence set for the knowledge split,
BBH 3–4 injectable tasks ×~200. **TruthfulQA**: pick ONE protocol — MC1/MC2 (candidate log-likelihood)
*or* generative truth×info — not both; used only as a secondary factuality check. **Leakage audit
procedure**: exact-normalized + semantic/fuzzy overlap of every eval item against the direction/ELS
prompts (`scripts/steer_messages/messages.py` contains many synthetic math problems); publish excluded
IDs. Materialize all to offline files on a login node (`TTS_GSM8K_FILE`, `TTS_MATH_FILE`, …).

## 5. Self-correction: crossed design (removes the initial-vs-correction confound)
1. Generate + **freeze** initial traces from **base** and from **edited** models.
2. **Review** each frozen trace with **base** and **edited** reviewers (2×2).
3. Report the **endogenous total effect** (edited-generate + edited-review) *separately* from the
   **fixed-input reviewer effect** (same frozen trace, base vs edited reviewer).
4. Stratify by the frozen trace's correctness; paired item-level comparisons on identical trace text.
Fair-setting: no oracle feedback; the initial policy (prompt + decoding) is **frozen/tuned on
validation** (drop the contradictory phrase "best-possible greedy"). Baselines to compare correction
against, **matched on generated tokens/FLOPs (not calls)**: self-consistency majority vote;
generate-and-rank with a frozen non-oracle selector; equal-calls independent resampling; oracle pass@n
as a **labeled upper bound only**. pass@1–pass@k is a diversity/ceiling diagnostic, not the metric.

## 6. Confounds & fairness
1. Behavioral vs generic damage: the **fixed-damage** random control (matched ΔNLL incl. math-domain)
   must move the outcome *differently* than the BLADE-G edit; WikiText ppl alone can't rule out
   domain-specific damage.
2. Extraction bias: edits change formatting/length → log no-answer/truncation separately, hand-audit.
3. Confidence validity: report **fixed-answer** confidence (freeze answer text, elicit under base vs
   edit) alongside total-system; multiple elicitation templates; attribute verbal-overconfidence to
   **Xiong et al.** (not the later CritiCal preprint).
4. Floor risk: **pilot base accuracy first** — if 1.7B is near-floor on MATH-500, AUROC/flip metrics
   lack support; then choose the operating benchmark accordingly.

## 7. Statistics / preregistration
One **primary endpoint per construct**: Brier (uncertainty); net correction benefit with separate
helpful/harmful conditional rates (backtracking); recall-EM (knowledge). Predefine paired item-level
contrasts, **bootstrap CIs**, mask-seed aggregation (hierarchical item×mask for inference),
generation-seed handling, **family-wise multiplicity correction**, and **minimum detectable effect**.
Everything else is exploratory.

## 8. Infra (measured, not asserted)
- One audited mask-builder module (extracted from `blade_reasoning_full.mask`, which is currently
  nested in `main` and not importable) with unit tests for α=0/α>1, abstention preservation, and the
  `<k positive → fail` guard.
- Model-resident worker: load Qwen3-1.7B once, loop over (pattern×mode×dataset) with resumable
  per-item output (self-correction & confidence are **multi-pass**, so "one generation pass" is wrong).
- Prefetch benchmarks to offline files (current `prefetch_reasoning_hazel.sh` only does C4/WikiText).
- **50-item end-to-end smoke first**, timing model-load / mask-build / generation / scoring
  separately; replace any "~1 day" claim with a measured estimate. Hazel a10 (24GB) fits 1.7B; jobs
  capped at 2h → shard resumably.

## 9. Phasing
- **P-1 (GATE)**: manipulation check + double dissociation on Qwen3-1.7B. Decides capacity vs
  keyword-pattern language for each pattern.
- **P0 (core, only patterns that passed or clearly flagged)**: pilot base accuracy; then uncertainty
  (Brier, fixed-answer + total) and backtracking (crossed design) on MATH-500 (+GSM8K), edit = BLADE-G
  {remove}, controls = {random fixed-sparsity, random fixed-damage, shuffled-r}.
- **P1**: add amplify(1.25/1.5); adding-knowledge split; faithfulness (Turpin + Lanham + restoration).
- **P2**: uncertainty-only calibration size trend across 4B/8B/14B (after reconciling the ELS
  discrepancy), BLADE-G only.
- **P3**: robustness — n=8 self-consistency, hidden-state probe (Zhang'25), extraction hand-audit.

## 10. Deliverables
P-1 validation table (lexicon precision/recall; length-controlled semantic-change; dissociation
matrix). Per-pattern verdict table: Δ(primary construct metric) + Δaccuracy vs base with random
(both regimes) & shuffled bands, bootstrap CIs, ppl (general+math), thinking-length. Figures:
reliability/risk–coverage (uncertainty); helpful/harmful flip bars (backtracking); recall + evidence
integration (knowledge). Honest one-paragraph verdict per pattern incl. nulls.

## 11. References (corrected)
Turpin et al., *LMs Don't Always Say What They Think*, **NeurIPS 2023** · Huang et al., *LLMs Cannot
Self-Correct Reasoning Yet*, ICLR 2024 · Kamoi et al., *When Can LLMs Actually Correct Their Own
Mistakes?*, **TACL 2024** · Tyen et al., *LLMs cannot find reasoning errors, but can correct them given
the error location*, **Findings of ACL 2024** · Lightman et al., *Let's Verify Step by Step*, ICLR 2024
(only if we add labeled step-correctness eval) · Lin et al., *Teaching Models to Express Uncertainty in
Words*, 2022 · Xiong et al., *Can LLMs Express Their Uncertainty?*, ICLR 2024 · Zhang et al., *Reasoning
Models Know When They're Right*, 2025 · Zhao et al., *Verify-and-Edit*, ACL 2023 · Lanham et al.,
*Measuring Faithfulness in Chain-of-Thought Reasoning*, 2023 · (BIG-Bench Mistake for error detection).

## 12. Wording bounds (gated on P-1)
Default claim: "we edit weights selected to suppress/amplify **keyword-associated thinking patterns**
and estimate the effect on pre-specified construct-aligned outcomes, vs fixed-sparsity **and**
fixed-damage controls, on Qwen3-1.7B and the named datasets." Capacity-level language
("the uncertainty/backtracking/knowledge capacity", "behavior-localized") is used **only for patterns
that pass P-1** (manipulation check + double dissociation). Never claim a pattern is *necessary* for
reasoning, nor generalisation beyond the tested model/datasets. Faithfulness (bias-injection) and
tiny-N results are directional.
