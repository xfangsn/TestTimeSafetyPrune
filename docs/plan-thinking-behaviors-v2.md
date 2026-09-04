# Plan (draft, for codex discussion) — can BLADE edit higher-level thinking behaviors?
## Target behaviors: Deliberation, Hypothesis Testing, Intermediate Verification, Search

## 0. What we already learned (constrains this)
From the P-1 manipulation check on Qwen3 (uncertainty/backtracking, keyword-span directions):
- keyword-span directions are a **noisy lexical proxy** (lexicon precision 0.23–0.74 vs blind semantic
  labels); editing them moves keywords far more than semantics.
- At 14B, **uncertainty REMOVAL** produced a real semantic change (−0.28, CI-excl-0); but **backtracking
  removal was null**, and **amplify was null for both** (adds surface markers, not the behavior).
- Reading: the more *surface/linguistic* a behavior (uncertainty ≈ hedging words), the more editable;
  the more *functional/load-bearing* (backtracking = actual self-correction), the more the edit either
  fails or would damage capability.

The four target behaviors are **more functional/structural** than uncertainty. So the honest prior is:
BLADE may *localize a direction* for them, but clean **removal is likely hard** (load-bearing, like
backtracking) and **amplify likely adds only surface form**. The design below is built to TEST this
per behavior, not assume success — and, critically, to fix the direction-building weakness.

## 1. The key change vs the keyword pipeline: SEMANTIC-annotation directions
For these behaviors keyword lexicons are hopeless (no reliable trigger words). So directions must come
from **LLM-annotated spans**, not keyword matches:
1. Generate reasoning traces on a diverse prompt set (held out from all eval sets).
2. An LLM annotator (codex/kimi; blind to purpose) labels, at the **sentence/segment** level, whether
   each segment instantiates the behavior — with a per-behavior rubric + positive/negative exemplars.
3. Build the CAA-style direction r_c and writer moment shift Δμ_c from the **annotated** behavior spans
   vs the rest of the thinking (exactly the BLADE recipe, but spans are semantic not lexical).
4. This makes the direction a *construct* proxy, and it lets us reuse the same manipulation-check gate
   (does the edit change the annotated-semantic behavior?) without the circularity we hit before.

## 2. Per-behavior: definition, direction contrast, gate metric, construct metric, feasibility prior
| Behavior | Working definition (rubric seed) | Contrast for r / Δμ | Manip-check (P-1) label | Construct metric (P0, its own dimension) | Prior |
|---|---|---|---|---|---|
| **Deliberation** | extended weighing of multiple considerations before committing (vs snap answer) | high-deliberation segments / long weighing vs terminal commit | fraction of trace that is genuine multi-option weighing | thinking-length & **accuracy-vs-length (overthinking)**: does removing it shorten traces and drop accuracy on hard items? (efficient-reasoning lit) | medium — deliberation is partly a global/length property, not a local span; a single direction may under-capture it |
| **Hypothesis Testing** | propose a candidate hypothesis, then check it against constraints/data | hypothesis-propose-and-check segments vs linear derivation | segment does propose→test | accuracy on **induction / rule-finding** tasks (e.g. ARC, list-function induction, Zendo/Mastermind) where H-testing is required | medium — annotatable, but likely load-bearing on those tasks |
| **Intermediate Verification** | check a partial result mid-reasoning before continuing | verification segments (re-derive/plug-back/sanity-check) vs none | segment verifies a partial result | **step-level correctness / error-catch rate** (PRM-style) + accuracy on multi-step problems | lower for removal (verification is load-bearing; expect backtracking-like null or capability damage) |
| **Search** | explore multiple branches of a solution space, enumerate/try alternatives | branching/enumeration segments vs single-path | segment explores ≥2 alternatives | accuracy on **search/planning** tasks (Countdown/Game-of-24, mazes, blocksworld) + branch count in traces | medium — structural; direction may be diffuse |

## 3. Pipeline (reuses existing machinery)
- Direction build: extend `qwen3_directions.py` to accept **semantic span labels** (from annotation)
  instead of keyword matches → `<tag>_dirs.pt` per behavior.
- ELS (BLADE-B layer selection) + edit (BLADE-G remove/amplify): unchanged (`reasoning_els.py`,
  `reasoning_mask.py`).
- **P-1 gate per behavior** (mandatory, from our last iteration): blind semantic annotation of
  base vs edited traces on a frozen held-out set; a behavior earns capacity language only if its edit
  changes the annotated-semantic rate (paired bootstrap CI excl-0). No cross-behavior dissociation.
- **P0 construct metric** per behavior as in the table; controls = random(fixed-sparsity),
  fixed-damage, shuffled-r; edit = BLADE-G only.
- Model: start Qwen3-14B (where uncertainty removal reached semantics); the smaller sizes were lexical.

## 4. Feasibility summary (honest prior, to be tested)
- Most likely to show a REAL removal effect: **Deliberation** and **Search** (they have the strongest
  behavioral signature in traces and clear task metrics). **Intermediate Verification** likely behaves
  like backtracking (null removal or capability damage). **Hypothesis Testing** is in between.
- Amplify: expect surface-form increase without genuine behavior (as observed) unless a stronger
  config; treat as secondary.
- Real risk: these behaviors may not be **linearly localized** as a single residual-writer direction
  at all — they are compositions. If a direction's manip-check fails for a behavior, we report it as
  "not BLADE-localizable at this config" (a legitimate negative), not force it.

## 5b. REVISION after codex discussion (supersedes §2/§4 where they conflict)
codex's core correction: all four targets are **sequential multi-stage policies, not semantic
attributes**. A single residual-writer direction can at most gate *entry* into a mode or encode its
*surface language*; the computation is distributed. So expect low absolute success; a positive result
must be a **functional change on an opportunity-conditioned metric with little change on matched
no-opportunity tasks** — not fewer labeled sentences, not just lower accuracy.

**Feasibility ranking (codex, corrected — replaces §2's "prior" column):**
1. **Hypothesis testing** (narrowed to "evaluate/reject the current candidate") — best first bet.
2. **Intermediate verification** — moderate; risk of damaging arithmetic/comparison, not verification.
3. **Deliberation** — low unless narrowed to comparative option-evaluation (else length/phase confound).
4. **Search** — lowest (trajectory-level algorithm; branch *narration* is linear but systematic
   exploration/pruning need not be; often lives in the harness, not one direction). Observability ≠
   localizability.

**Semantic annotation does NOT remove circularity** (annotator + edit both key on surface form). Required:
- meaning-preserving **paraphrases that strip canonical markers**; **marker-only negatives** (fluent
  "let me verify" that performs no check); positives/negatives **matched** on task, length, position,
  uncertainty, correctness, syntax.
- **incremental validity**: show the direction predicts the semantic label *beyond* keyword-count +
  length + discourse-position baselines; cross-domain + lexical-adversarial splits (not random split).
- build directions from **pre-onset activations / matched counterfactual continuations**, NOT
  "behavior spans vs rest of thinking" (the rest is mixed phases/false-negatives).
- **linked-event annotation**, not binary spans: HT = `hypothesis→test-evidence→result→action`;
  Search = `parent-state→branches→pruned/selected`.
- pre-edit necessity tests: predict onset from activation *before* the behavior text; dimensionality
  sweep (rank-1/2/4/8 — if it keeps improving, reject single-direction); confirm the **weight edit
  reproduces the activation intervention** and weight-restoration rescues the effect.

**Construct metrics + datasets (codex, concrete, top-venue):**
- Deliberation → **DeLLMa** (Liu, ICLR 2025) normalized-utility/optimal-action, report BOTH natural-
  and fixed-token-budget; length is a resource not the construct (Snell, ICLR 2025).
- Hypothesis testing → **List Functions** with symbolic execution of every proposed rule (Qiu, ICLR
  2024; Hypothesis Search, Wang, ICLR 2024); **MIRAGE** (Li, ICLR 2025) as induction-vs-deduction control.
- Intermediate verification → **ProcessBench** (Zheng, ACL 2025) harmonic-mean detection on
  clean/corrupted twins; adapt to *spontaneous* (truncate after error, ask to continue).
- Search → **Game-of-24** with exact state graph, equal node/token/call budget (ToT, Yao, NeurIPS
  2023); PlanBench Blocksworld secondary.

**Keep recording off-target behaviors** (multi-label event graph + conditional effects): we drop the
strict double-dissociation *requirement* (correlated constructs), but NOT the recording — else a
"hypothesis-testing edit" may just be a backtracking/verbosity edit.

**Taxonomy caveat**: {Deliberation, HT, Verification, Search} is NOT standard and mixes abstraction
levels; heavy overlap with our existing set (HT ≈ example-testing almost directly; Verification ≈
example-testing + backtracking; Search generates+tests alternatives). Frame results as refinements of
existing constructs, not new faculties.

## 5c. Recommended FIRST experiment (codex)
**Counterexample-driven hypothesis revision on fresh List Functions.** Per item, two paired conditions
with the SAME candidate rule: (a) a minimally-changed **falsifying** example, (b) a matched
**supporting** example. Without telling the model to "verify", measure
`D = P(reject/revise | falsifying) − P(reject/revise | supporting)`.
BLADE removal passes only if it **reduces D**, reduces the revised rule's held-out pass rate, and
reduces whole-task accuracy *specifically where revision is needed*, while leaving valid-rule
application, matched deduction, trace budget, and a generic-capability battery intact — with an
automatic executor as ground truth and **weight-restoration rescue**. Report as editing
"counterexample-sensitive hypothesis revision", not "the hypothesis-testing faculty".

## 5. Open questions for codex (answered above in §5b/§5c)
1. Are these four behaviors the right/standard taxonomy, and are they *linearly* representable enough
   for a CAA-style single direction + residual-writer edit to work — or are they inherently
   compositional (needing multi-direction / circuit-level edits)?
2. Best **construct metric + dataset** per behavior (esp. Hypothesis Testing and Deliberation), from
   the ML literature — what do top-venue papers actually measure?
3. Is semantic-annotation direction-building enough to escape the lexical-proxy trap, or is there a
   residual circularity (annotator + edit both keying on surface form)?
4. Which behavior is the best *first* bet (highest chance of a clean, interpretable result)?
5. Any confound specific to these (e.g. Deliberation ≡ length, so any length-changing edit fakes it;
   Search ≡ compute, etc.) and how to control it.
