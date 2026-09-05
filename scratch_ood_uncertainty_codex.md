# Critical review of `docs/plan-ood-uncertainty.md`

Reviewer stance: skeptical mechanistic-interpretability reviewer. I take the stated untouched-split,
semantic-judge, bidirectionality, control, and crossover results as established rather than reopening
them. The question here is whether the proposed external evaluation can support the narrower claim the
plan now makes.

## Executive verdict

The closed-book restriction is correct and scientifically important. Excluding SQuAD2-style passage
QA from this transfer claim follows directly from the evidence-sufficiency crossover; putting it back
would mix a regime in which the fixed edit has already been shown to be inert with the parametric-
knowledge regime. The move from lexical abstention to a blind semantic outcome, and especially the
REMOVE endpoint “base appropriately withheld -> edited model makes an unwarranted definite
commitment,” is also exactly the right direction.

The plan is not yet preregistration-ready. Its central weaknesses are:

1. The three proposed benchmark families do not instantiate the same target. SelfAware mixes
   subjective, philosophical, underdetermined, speculative-future, and no-consensus questions;
   FalseQA tests rejection of a proposition the model may confidently know is false; and the third
   benchmark is still an unspecified “e.g.”. These should not be pooled as interchangeable cases of
   parametric ignorance.
2. FalseQA premise correction is a confident factual commitment, not “non-commitment” or abstention.
   A common abstention endpoint would mis-score the cleanest benchmark in the suite.
3. Conditional flip rates are excellent mechanistic transition endpoints, but are not sufficient as
   the sole primary estimands. They need fixed opportunity sets, unconditional absolute-risk results,
   and full transition tables. Otherwise low or dataset-dependent headroom controls the headline.
4. “BLADE versus the random-null distribution” is not tested by McNemar against baseline. Twenty
   random masks are barely enough for a one-sided empirical randomization p-value of 1/21 = 0.0476 and
   cannot support a corrected confirmatory tail claim. The primary needs a crossed prompt-by-mask
   analysis and substantially more random draws, or a clearly justified hierarchical model.
5. Direction-transfer AUROC is useful descriptive evidence, but with roughly three benchmarks it is
   not an estimable dataset-level moderator. It also compares against dataset labels rather than the
   model's item-specific knowledge. Do not dichotomize it into “direction transfers/does not transfer”
   or use it to rescue a failed edit.
6. The plan says controls “must stay near-null,” but does not define near-null. Failure to reject a
   control effect is not evidence of specificity. Equivalence margins and a BLADE-minus-control
   specificity margin must be fixed in advance.

With those changes, this is a strong and unusually falsifiable OOD study. Without them, a positive
result can still be summarized by a reviewer as “a sparse assertiveness/decline edit changes the style
preferred by several old abstention benchmarks.”

## 1. Benchmark choice and closed-book scope

### 1.1 The regime restriction is right

Keep all primary prompts genuinely closed-book: no passage, retrieval, tool output, or few-shot
demonstration containing task evidence. Use the same thinking-off chat template and decoding policy as
the intervention study. SQuAD2 is not merely an inconvenient benchmark; it asks a different causal
question—whether evidence in a supplied context is sufficient—and the crossover has already found
zero edit effect in that regime. SQuAD2 can appear as a negative boundary replication, but it should
not enter the OOD transfer aggregate.

The paper should call these datasets **external to edit fitting**, not necessarily unseen to Qwen3's
pretraining or instruction tuning. SelfAware and FalseQA predate Qwen3, and benchmark exposure is
plausible. That does not invalidate transfer away from the templated fitting distribution, but it does
weaken rhetoric about natural, unseen deployment behavior. Deduplicate all evaluation questions
against the templated bank, report benchmark/version hashes, and ideally add a small sealed,
human-authored set produced after the edit and judge were frozen.

Also lock the exact prompt. An instruction such as “say I don't know if uncertain” can turn the
experiment into compliance with an abstention cue. A neutral question-only prompt is the clean primary;
an abstention-permitted instruction can be a robustness condition.

### 1.2 SelfAware is useful, but its unanswerable half is not a unitary parametric-knowledge test

SelfAware contains 1,032 internet questions judged unanswerable and 2,337 answerable questions drawn
from SQuAD, HotpotQA, and TriviaQA ([paper](https://aclanthology.org/2023.findings-acl.551/),
[dataset](https://github.com/yinzhangyue/SelfAware)). Its five reported unanswerable categories include
no scientific consensus, imagination about the future, completely subjective questions, too many
variables, and philosophical questions. The local file confirms examples such as “Would you rather
...?”, “Why does anything exist?”, and speculative science questions. A useful discussion is often
appropriate on these items; demanding a bare abstention is not.

Consequences:

- Do not call every direct answer to a SelfAware-unanswerable item a hallucination. The target event is
  an **unwarranted definite answer to the proposition the question asks**, not merely producing
  content.
- Reannotate or stratify the unanswerable items by reason. The closest strata to the established
  construct are unknown future fact, unknowable empirical detail, and genuinely unresolved fact. The
  subjective/philosophical strata are valuable heterogeneity tests, but not evidence of parametric
  knowledge boundaries.
- The answerable and unanswerable halves have different provenance and style. A high frozen-direction
  AUROC can therefore be driven by source/style. Report surface-only AUROC on the external set and
  per-category direction distributions, just as was necessary in-distribution.
- Answerability in the dataset is not knowledge in Qwen3-8B. Report full-set answerable accuracy, and
  define a separate base-known preservation stratum using a frozen, independently scored baseline
  protocol.

SelfAware is defensible as the primary **natural-query stress test**, but not as the sole primary for a
parametric-knowledge claim. If it remains primary, preregister a factual/empirical subset as the claim-
aligned primary stratum and the full 1,032 as a heterogeneous secondary analysis.

### 1.3 FalseQA is the cleanest design, but a different behavior

FalseQA contains 2,365 human-written false-premise questions, explanations/rebuttals, and minimally
revised true-premise questions ([paper](https://aclanthology.org/2023.acl-long.309/),
[repository](https://github.com/thunlp/FalseQA)). Its paired construction is a major advantage: it can
hold topic and much of the wording fixed while changing whether accepting the premise is warranted.

However, FalseQA is not model ignorance in the ordinary sense. The original paper's premise is often
that the model already has the factual knowledge required to rebut the question. “The sun has no eyes”
is a confident-correct proposition. Therefore:

- premise correction/rebuttal must be scored as **premise rejection**, not abstention and not
  non-commitment;
- the failure endpoint is **false-premise acceptance**—answering the requested slot as if the premise
  were true—or another factually wrong response;
- the paired true-premise question supplies the best preservation control. An intervention that
  rejects both false and true premises is a generic skepticism/decline edit, not selective epistemic
  control.

A positive FalseQA result licenses “transfer to false-premise handling” and supports a broader
epistemic-gating interpretation. It does not, by itself, prove that the edit tracks absence of stored
knowledge. A clean null on FalseQA would narrow the effect to ignorance/unknowability rather than
automatically refute it.

### 1.4 The third benchmark must be named now

“Known-Unknown / unanswerable factual (e.g. a SimpleQA/TriviaQA-unanswerable or a curated set)” is not a
preregistered benchmark. TriviaQA and SimpleQA are answerable QA datasets; “unanswerable” subsets made
by selecting model errors are model-specific difficulty strata, not gold-unanswerable questions. A
wrong exact-match answer can also reflect aliasing, extraction, or instruction following rather than
lack of knowledge.

Choose one exact, versioned dataset and define its role. KUQ is an obvious candidate for world-level
known-unknowns and includes reasons for uncertainty
([paper](https://aclanthology.org/2024.findings-acl.383/)). A fresh curated set can be stronger, but then
the annotation protocol, adjudication, source dates, exclusion rules, category balance, and sealed test
split must be specified before generation. Separately select a matched known set from ordinary factual
QA using a screening split. Do not silently convert “the base got it wrong once” into “the model does
not know.”

FreshQA/temporal questions should remain secondary. They are answerable at a reference date, are highly
sensitive to the model's unknown training cutoff, and can test stale knowledge or date awareness rather
than intrinsic unanswerability.

### 1.5 Do not pool unlike response policies under one binary label

Use a shared high-level idea—**warranted versus unwarranted commitment**—but dataset-specific observed
events:

| Dataset/type | Appropriate behavior | Harmful commitment |
|---|---|---|
| Factual/world unknown | Explicitly withhold the requested fact; a calibrated explanation is allowed | State a definite value/entity/date without warrant |
| Subjective/underdetermined | State dependence on preferences/variables; discuss conditionally | Claim a unique objective answer |
| FalseQA | Reject/correct the false premise | Accept the false premise and fill its requested slot |
| Answerable/true-premise | Give the correct answer | Abstain, reject a true premise, or answer incorrectly |

This preserves a common functional interpretation without pretending that “I don't know,” “the
premise is false,” and “there is no scientific consensus” are the same response act. Meta-analyze a
common binary **task-appropriate outcome** only after showing the component transition tables.

## 2. Judge and endpoints

### 2.1 The functional endpoint is right; the proposed schema cannot yet measure “confident”

Replacing abstention-word counts with the transition to an unsupported/wrong commitment is the most
important improvement over the earlier work. It tests harm, not disappearance of uncertainty language.
But `{abstain, commit, mixed} x {correct, wrong, unsupported, n.a.}` lacks a confidence/qualification
axis. A commit can be explicitly tentative, while a premise correction can be confident and correct.

Use orthogonal fields such as:

1. **Response act:** answer requested value / explicit withhold / premise rejection or correction /
   conditional discussion / mixed.
2. **Qualification:** unqualified-definite / calibrated-qualified / strongly uncertain / not
   applicable.
3. **Factual status of the main commitment:** correct / wrong / not verifiable in principle / no main
   commitment.
4. **Premise status for FalseQA:** accepts / rejects correctly / rejects incorrectly / ambiguous.
5. **Task appropriateness and output integrity:** appropriate/inappropriate; complete/truncated/
   degenerate.

Then define “confident wrong commitment” before scoring, for example: an unqualified-definite requested
answer whose factual status is wrong, or an unqualified answer that accepts a gold false premise. For
world-unknown questions, “unsupported” must mean a definite answer to the unknowable target—not every
ancillary statement in an otherwise appropriate explanation.

### 2.2 Blind judging is appropriate, with stronger validation

Opaque IDs and condition-hidden, individually randomized responses are correct. The judge must still
see the question and the dataset-specific gold answer, false-premise explanation, or uncertainty reason;
otherwise factual status cannot be scored. Do not show responses from several conditions side by side,
because their pairing can reveal the intervention and encourage relative rather than absolute scoring.

Before locking:

- validate against at least two blinded human raters on every dataset and every primary class;
- oversample base/edit and BLADE/control discordances, but prevalence-weight judge-accuracy summaries;
- report classwise precision/recall and confusion, not only kappa or aggregate agreement;
- human-adjudicate all primary BLADE/base discordances if feasible, plus a random sample of concordant
  and random-mask cases;
- freeze judge model/version, prompt, decoding, retry policy, and tie/adjudication rules;
- track truncation separately, because an edit-induced short answer can look more “confident.”

Judge error is especially dangerous here because the primary is made only of rare transitions. A few
systematic errors on nuanced SelfAware discussions can create most of the estimated harm.

### 2.3 Conditional transition endpoints are necessary but should not stand alone

For deterministic decoding, conditioning REMOVE on items where the fixed base response appropriately
withheld is a legitimate and very interpretable opportunity-set estimand. Likewise, conditioning
AMPLIFY on base false commitments identifies its rescue opportunities. Determine each base stratum
once, with a locked blind label, and use exactly the same denominator for BLADE and every control.

Still report, per benchmark:

- the complete base-to-edit transition table;
- the conditional conversion rate in the fixed opportunity set;
- the unconditional absolute change in harmful-commitment/correct-response rate over all eligible
  prompts;
- the opportunity-set denominator and cluster count;
- answerable full-set accuracy and base-correct retention.

The unconditional effect should anchor the pooled primary because it remains defined when a benchmark
has little headroom. Conditional rates from, say, 12 baseline abstainers should not receive equal weight
to rates from 300. A headroom problem is a precision/result issue, not an exclusion rule.

If generation is sampled, one baseline draw is not a stable stratum. Define opportunity using an
independent baseline batch or a prespecified majority over repeated draws, then evaluate interventions
on new draws and cluster by prompt. With greedy decoding, state that the estimand is for that fixed
policy, not a stochastic response distribution.

### 2.4 Preservation needs a margin, not “stayed at 0.00”

On answerable items, evaluate all failure modes: correct -> wrong, correct -> abstain, and (for paired
FalseQA) true-premise -> erroneous premise rejection. PPL flatness does not establish behavioral
preservation. Predeclare a one-sided noninferiority margin for answerable accuracy and a separate margin
for induced unwarranted abstention. Report confidence intervals; “not significant” is not preservation.

## 3. Null controls and specificity

The proposed control types are directionally right, but the number and inferential role of each need
revision.

### 3.1 Random masks: 20 is an exploratory minimum, not a strong primary null

Use exact per-layer, per-matrix, and realized scalar counts. Match the joint distribution—not only the
mean—of generic importance `Q`, weight magnitude, and, where relevant, BLADE sign/eligibility and
fan-in/fan-out occupancy. REMOVE and raw-alphaW AMPLIFY each need operation-matched random controls.
Suppressor-removal needs random draws from the same suppressor-eligible stratum.

Any damage matching or rerandomization must use separate generic validation data, never OOD outcomes.
Report generic KL/NLL, response length, output-change rate, and answerable accuracy rather than relying
on Wiki/C4 perplexity alone.

With `M` random masks, the smallest ordinary one-sided empirical tail p-value is `1/(M+1)`. Thus 20
masks permit only 0.0476 even before any multiplicity. Use at least 99 for a confirmatory 0.01 tail
resolution, ideally more if random-mask exceedance is the headline comparator. If compute limits the
study to 20, use a prespecified crossed prompt-by-mask hierarchical model and label the empirical-tail
comparison low-resolution; do not claim a well-estimated extreme null.

### 3.2 Shuffled-r and label permutation need distributions

A single coordinate shuffle is an anecdote. It can also change score support and geometry in ways that
make it trivially weak. Use several independent shuffles and report their score/support diagnostics.

The full-pipeline label-permutation null is stronger because it preserves the pipeline's opportunity to
overfit. Permute within every training stratum that the real fit balances, recompute direction,
moments, BLADE-G scores, ELS, and the final mask, and repeat. One permuted fit does not characterize
selection variance. If full refits are too costly, designate this as a qualitative falsification check
and make random masks the sole inferential null; do not write as though one permuted model proves the
null pipeline cannot transfer.

### 3.3 “Wrong layer” must not mean one hand-picked loser

A single wrong-layer mask tests only that layer. Predefine a distribution over nonselected layer sets
matched for number of layers, eligible parameter count, matrix composition, and preferably generic
damage. Do not choose wrong layers after looking at OOD behavior. Success against this distribution
supports **layer-selection specificity**, not anatomical localization of an uncertainty mechanism.

### 3.4 The listed nulls do not address generic assertiveness/decline control

Random, shuffled, permuted, and wrong-layer controls ask whether the fitted BLADE selection is special.
They do not distinguish epistemic gating from a learned generic “answer versus decline” actuator. The
answerable preservation axis and the completed passage crossover help substantially, but a skeptical
reviewer can still ask whether the edit is a prompt-triggered assertiveness controller.

Include one active specificity control: a frozen refusal or matched directness/verbosity edit, evaluated
under the same OOD protocol. Interpret it prospectively. Matching BLADE on unwarranted commitment means
the external result is behaviorally nonspecific even if random masks are null; merely changing response
length without reproducing factual transitions does not count as matching.

### 3.5 “Controls must be null” requires equivalence rules

Predeclare practically meaningful absolute margins. For example, if 5 percentage points is the largest
tolerable control-induced harmful-commitment change and 10 points is the smallest meaningful BLADE-
minus-control advantage, power and judge validation should target those numbers. The exact margins need
a substantive rationale, but they cannot be chosen after results.

A positive result should require both:

1. the BLADE effect exceeds its minimum meaningful effect; and
2. BLADE exceeds the matched-random estimand by the specificity margin, while the upper confidence or
   prediction bound for null-control transfer lies inside its equivalence margin.

“BLADE significant, controls nonsignificant” does not meet this criterion.

## 4. Direction transfer as a moderator

Reporting the frozen direction is valuable, but the current `{dir transfers?} x {edit transfers?}`
framing is too discrete and too strong.

- Preserve the direction sign, layer, last-token position, chat template, and normalization learned on
  the templated training data. Do not recalibrate a threshold or flip the sign per benchmark.
- Report signed projection distributions and AUROC with cluster confidence intervals. Add a surface-
  only baseline, particularly for SelfAware, whose classes have different sources and styles.
- Gold answerable/unanswerable AUROC is a benchmark-label test, not a model-knowledge test. Also report
  projection by base behavioral/correctness stratum: base appropriately withholds, hallucinates,
  answers correctly, and answers incorrectly.
- Treat AUROC continuously/descriptively. With only about three datasets, a dataset-level
  meta-regression of edit effect on AUROC has essentially no inferential content. A binary quadrant
  based on significance or an arbitrary AUROC cutoff magnifies noise.
- An item-level projection-by-intervention interaction can be prespecified as exploratory, with dataset
  fixed effects and surface covariates. It is not causal mediation: projection can be a correlate of
  prompt type.

The four qualitative outcomes remain useful diagnostics: readout and edit both transfer; readout only;
edit only; neither. They do not form a gating rule. In particular, direction success cannot excuse a
weight-edit null, and direction failure cannot be removed from the OOD denominator.

## 5. Statistics, power, and multiplicity

### 5.1 McNemar does not test the headline specificity claim

Exact McNemar is appropriate for a simple binary base-versus-one-edit comparison when items are
independent. The primary claim, however, is **BLADE versus a distribution of matched controls on the
same prompts**. Analyze the paired difference between BLADE's change and each mask's change, carrying
prompt and mask uncertainty through one joint procedure. Suitable options are a crossed hierarchical
binary model, a prompt-clustered randomization analysis over masks, or a hierarchical/cluster bootstrap
that resamples prompt clusters and random masks.

If questions share categories, templates, entities, or FalseQA source pairs, ordinary McNemar p-values
are anti-conservative. Use the cluster-level procedure for inference and show McNemar only as a familiar
sensitivity analysis. State the clustering unit for each dataset before opening intervention outputs.

### 5.2 Power the actual BLADE-minus-null estimator

“Power from the expected discordance matrix” is correct for paired base/edit effects but incomplete for
the primary comparator. Simulation should include:

- base opportunity-set prevalence;
- BLADE/base and control/base discordance, including their within-prompt correlation;
- between-mask variation;
- category/pair clustering;
- judge misclassification sensitivity;
- the prespecified equivalence and specificity margins.

Run a blinded baseline-only stage to estimate headroom and cluster sizes, then freeze the maximum sample
and analysis before interventions. Do not choose benchmarks or intervention doses based on edited
effects.

### 5.3 Equal benchmark weighting needs an exact estimand

An equal-weight mean over three deliberately chosen datasets estimates only the average over those three
benchmarks. It is not a random-effects estimate of deployment domains, and `K=3` cannot estimate
between-benchmark variance reliably. Define the aggregate as a standardized fixed-benchmark estimand:
compute a paired absolute risk difference within each prespecified dataset, average them with fixed
weights, and obtain its interval in the joint prompt-by-mask resample. Always show each dataset and
heterogeneity.

Do not equal-weight conditional conversion rates with tiny and large opportunity sets. Prefer
unconditional task-harm risk differences for the aggregate; report opportunity-set conversions
separately.

FalseQA should either have its own construct-specific primary or enter an aggregate only through
task-appropriateness/harm, not literal abstention. An aggregate that changes endpoint definition silently
is not interpretable.

### 5.4 Make bidirectional transfer a conjunctive claim

If the paper claims **bidirectional OOD transfer**, both prespecified directions must pass their own
specificity test:

- REMOVE increases unwarranted commitment relative to matched nulls while meeting preservation;
- AMPLIFY increases appropriate withholding/premise rejection relative to matched nulls while meeting
  preservation.

This can be an intersection-union/conjunctive claim: reject the broad null only if both component tests
pass at the prespecified level. If only one passes, report one-way transfer. Do not use successful REMOVE
to make AMPLIFY confirmatory after the fact.

Fix x2 raw-alphaW as the AMPLIFY primary dose if that is the established strongest clean intervention;
x1.5 is dose support and suppressor-removal is a mechanistically interesting secondary. Across
benchmark-specific claims, doses, alternative operations, and primary control families, use a written
gatekeeping hierarchy or Holm/max-T procedure. The current phrase “one pooled primary” is insufficient
because the headline also promises both directions and preservation.

## 6. What a skeptical reviewer will attack

In likely order:

1. **Construct pooling:** “SelfAware's opinions, FalseQA premise corrections, and model errors on
   TriviaQA are not one kind of epistemic uncertainty; you changed the scoring rule per dataset and
   averaged the answers.”
2. **Specificity inference:** “McNemar says BLADE differs from baseline, not that BLADE beats random
   edits. Twenty random masks give one barely-significant rank and no corrected tail resolution.”
3. **Generic response policy:** “You found an assert/decline actuator. The null masks cannot rule that
   out, and C4 perplexity plus answerable abstention is not a behavioral active control.”
4. **Judge construct leakage:** “Your judge calls nuanced discussion a commitment, has no confidence
   axis, and uses ‘unsupported’ without a reference that could establish support.”
5. **Model-specific knowledge:** “Gold answerability is not what Qwen knows. You selected knowns from
   one baseline answer and unknowns from errors, then interpreted the partition as latent knowledge.”
6. **Contamination and novelty:** “These old public benchmarks may be in Qwen3's alignment data. The
   experiment is OOD relative to your fitting prompts, not necessarily unfamiliar to the model.”
7. **Multiplicity:** “There are several datasets, two directions, two amplification doses, a second
   amplification method, several controls, preservation, and direction moderation, but only one vague
   pooled primary.”
8. **Mechanistic overreach:** “External behavioral transfer of a fixed sparse edit is causal evidence
   about those weights, but it does not by itself show that the weights compute uncertainty. They may
   write a downstream answer policy.”

The wording in Section 6 of the plan is mostly disciplined. Retain “expressed closed-book parametric
(un)certainty” and “sparse causal control,” avoid “the uncertainty mechanism,” “localization,” and
“general uncertainty,” and name every dataset on which transfer actually passed the specificity rule.

## 7. The single most decisive—and most endangering—OOD experiment

Run the complete FalseQA **false-premise/true-premise minimal-pair crossover** first, before an
unstructured benchmark sweep.

For every untouched pair, generate the false-premise question and its minimally revised true-premise
counterpart under base, BLADE REMOVE, primary BLADE AMPLIFY, and their operation-matched control
distributions. Use the supplied rebuttal/explanation and true answer in a locked blind rubric. Define:

- REMOVE harm on the false question: new false-premise acceptance or wrong slot answer;
- AMPLIFY benefit on the false question: new correct premise rejection/rebuttal;
- paired preservation: correct answering and non-rejection of the true-premise counterpart;
- REMOVE selectivity contrast: increased false-premise acceptance minus increased error/decline on the
  true-premise partner;
- AMPLIFY selectivity contrast: increased correct premise rejection minus increased erroneous
  rejection/decline on the true-premise partner;
- specificity contrast: each BLADE selectivity contrast minus the same contrast for matched random
  masks.

Show fixed opportunity strata as especially legible secondary transitions:

- base rejects false premise and answers true counterpart correctly -> under REMOVE, does it begin
  accepting the false premise while retaining the true answer?
- base accepts false premise and answers true counterpart correctly -> under AMPLIFY, does it begin
  rejecting only the false premise?

Why this is decisive: it is external, closed-book, human-written, large enough to power, and matched at
the topic/wording level. It directly separates selective sensitivity to warrant from a generic tendency
to answer or decline. Why it is endangering: random-mask transfer, equal effects on true premises, or a
pure length/style change would collapse the OOD claim immediately. A null confined to FalseQA would be
interpreted more narrowly—no transfer to premise verification—but a nonspecific positive would be much
more damaging than an ordinary null.

This experiment is not the purest demonstration of “the model lacks a stored fact”; it is the strongest
falsification of the proposed broader epistemic answer-policy interpretation. Follow it with a frozen
fact-like SelfAware/KUQ evaluation if it passes.

## 8. Concrete edits required before running

1. Replace the third benchmark placeholder with an exact version, split, item policy, and frozen item
   list. State which benchmark/stratum is the construct-aligned primary.
2. Keep SelfAware, but stratify its five causes and make fact-like/no-consensus/future-fact strata
   distinct from subjective/philosophical discussion.
3. Recast FalseQA around premise acceptance/rejection and exploit its true-premise minimal pairs; never
   call a correct rebuttal abstention.
4. Expand the blind judge with qualification/confidence and FalseQA premise-status axes; validate it
   classwise and human-adjudicate primary discordances.
5. Lock opportunity sets once from base, use them identically for all masks, and add unconditional risk
   differences plus full paired transitions.
6. Define exact equivalence, minimum-effect, specificity, and answerable-preservation margins.
7. Increase matched random masks to at least 99 for an empirical-tail primary, or preregister and
   justify a crossed hierarchical alternative. Repeat shuffled-r and full-pipeline permutations enough
   to characterize their variability, or explicitly demote them to falsification checks.
8. Replace one wrong-layer control with a prespecified matched distribution of nonselected layer sets.
9. Add one active generic decline/directness control. State in advance that matching functional
   transitions is evidence against epistemic specificity.
10. Replace per-benchmark McNemar as the headline with a joint prompt-cluster-by-mask analysis of the
    BLADE-minus-null effect. Use McNemar only for simple paired component tables.
11. Define the fixed-benchmark aggregate and multiplicity hierarchy. Make bidirectional transfer
    conjunctive; otherwise claim only the direction that passes.
12. Treat frozen-direction AUROC as continuous descriptive/heterogeneity evidence, with surface
    baselines and base-behavior strata, not as a gate or a powered moderator across three datasets.

## Bottom line

The plan's scientific core is sound: test a frozen sparse edit only in the closed-book regime in which
the construct experiment says it operates, measure factual commitment rather than hedge words, and
require selection-specific controls. The remaining danger is not the old random-mask failure alone. It
is treating several distinct reasons not to answer as one latent “uncertainty” variable and then using
baseline-vs-edit significance to imply BLADE specificity. A paired FalseQA crossover, a fact-like
world-unknown set, a genuinely crossed mask analysis, equivalence-based nulls, and a judge that separates
withholding from confident premise correction would make the OOD result much harder to dismiss.
