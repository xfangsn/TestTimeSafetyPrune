# Independent review of `docs/plan-ood-epistemic.md` (plan v2)

Reviewer stance: skeptical top-venue mechanistic-interpretability reviewer. I read plan v2, the full
Fable review, the parent bidirectionality plan, the relevant construction/evaluation scripts, and the
saved Qwen3-8B results. This review concentrates on problems that survived the Fable rewrite, plus a
few places where the rewrite overstates what the existing artifacts show.

## Executive verdict

Plan v2 is materially better than v1. A genuinely disjoint ELS-selection/evaluation split and a
semantic response metric would fix the two most immediate defects in the 0.564 -> 0.05 result. Putting
family transfer before heterogeneous external benchmarks is also a sensible cost ordering.

It does **not yet make the intended claim confirmatory**, for five reasons:

1. The current direction-validation result is not the direction used by the weight experiment. The
   reported LOFO artifact was built with Qwen thinking **on**, whereas ELS recomputes its direction and
   moments with thinking **off**. In addition, the result is not “length-controlled”; it merely beats a
   separately reported length-only baseline.
2. P0 and Axis B do not specify a leak-free nesting. Axis B says to operate “on the untouched split,”
   and it is unclear whether a held-out family is also excluded from direction fitting, writer moments,
   ELS, rho/layer choice, and judge development. Reusing P0's untouched evaluation set for layer/family
   development makes it no longer untouched.
3. Axis A reintroduces selection on the target distributions by excluding benchmarks based on observed
   frozen-direction AUROC and observed base headroom. These quantities are valuable moderators, not
   legitimate post hoc admission criteria for a pooled confirmatory endpoint.
4. The synthetic families do not identify epistemic uncertainty. Knowability is entangled with tense,
   entity familiarity, lexical oddity, question predicate, and answer type. Five-family transfer can
   show transfer across these five generators, but it cannot by itself license the label “epistemic
   uncertainty.”
5. The controls are a good list of threats, but several proposed interpretations are too strong. A
   wrong-layer pair is not a layer-localization test, behavioral cross-transfer does not establish a
   shared circuit, a verbosity edit producing the same surface behavior does not by itself falsify an
   epistemic route, and steering plus weight success does not establish mechanism equivalence.

My overall assessment would be: **promising sparse, selection-specific causal control of an expressed
answer-disposition, not yet a localized epistemic-uncertainty mechanism**. The next experiment should
not be a broad benchmark sweep. It should be a held-out, counterfactual evidence-sufficiency experiment
that crosses entity novelty with whether the answer is actually supplied by the prompt. That one design
can either sharply strengthen the epistemic interpretation or show that the edit is an anomaly/decline/
directness controller.

## 1. Corrections to the factual premise of plan v2

### 1.1 The LOFO direction result and the BLADE edit use different prompt regimes

`scripts/epistemic_direction.py` explicitly uses `enable_thinking=True` (lines 34--40). It produces the
saved LOFO AUROCs and `data/directions/epistemic_qwen3_8b.pt`. By contrast,
`scripts/blade_epistemic_els.py` defines `qwen_wrap(... enable_thinking=False)` and monkey-patches the
direction extractor before recomputing the direction from its training split. Thus:

- the “LOFO AUROC 1.0” direction is a thinking-on, full-data artifact;
- the BLADE score and selected weights use a different, thinking-off, train-split direction;
- the existing steering screen uses the thinking-on saved direction, thinking-on generation, 512 new
  tokens, and decode-only steering;
- the weight result uses thinking-off generation with 64 new tokens and weights active during both
  prefill and decoding.

The current direction validation, steering result, and weight result therefore cannot be treated as a
single validated mechanistic chain. P0 must rebuild and validate the **exact fold-specific direction
used to score the weights**, under the identical chat template and thinking setting. The steering
reference must use that same direction, prompts, output budget, and intervention positions (with
prompt-only/decode-only/all-position steering separated if scientifically useful).

This also makes the plan's proposed L16/L22 steering reference questionable: the selected weight layers
are L23/L16, while L22 appears to have been retained because a prior steering screen looked positive.
For a confirmatory comparison, steer at the predeclared weight layers or give an independent rule for
selecting the steering layer. Do not choose it from the same behavioral sweep being used as evidence.

### 1.2 “Length-controlled” is false

The LOFO code computes a token-length AUROC and separately computes probes/projections. It does not
match on length, residualize activations against length, include length as a covariate, or test
incremental predictive value over length. The saved mean length-only AUROC is 0.893, and within-family
length AUROC is 1.0 for `capital`, `person_attr`, and `quantity`. This is a major warning, not a control.

The headline also conflates probe and diff-of-means results. The diff-of-means AUROC is 1.0 at L10 and
many later layers; the logistic-probe AUROC is not 1.0 at L10--L14. Rewrite the premise as exactly what
was measured and add one of:

- strict length/token-pattern matching within each class;
- a residualized direction after removing length and template/predicate features using training data;
- incremental held-out performance of activations over a strong surface baseline (length, bag of
  tokens/character n-grams, tense, entity-frequency proxies, family), with nested fitting.

Even this would establish representational information, not epistemic semantics.

### 1.3 The current labels are not uniformly “unanswerable”

The synthetic construction equates a fictional country with an unanswerable capital question. That
implication does not hold in general: established fictional settings can have canonical capitals. The
same audit is needed for generated book titles, which can collide with real works. More generally,
`capital` tests fictional-entity recognition; `authorship` tests title familiarity; `event_year` exposes
the label in tense and year; `person_attr` changes the predicate completely; and `quantity` contrasts
arithmetic with deliberately impossible measurements. These are not close counterfactual pairs.

Before P0, manually audit labels and define what the correct response is for every item. Do not call a
premise denial “hedging.” Also correct the repeated “3/5 families are nonexistence” shorthand: the
families mix nonexistence, undocumented historical details, future events, and unmeasurable quantities;
that coarse count is not a stable construct description.

### 1.4 The reported test is paired, and the sparsity denominator is ambiguous

Base and edited responses are generated for the same prompts. Fisher's exact test on two independent
proportions is not the appropriate primary test. Use an exact McNemar test for an unclustered binary
endpoint, or a paired cluster bootstrap/randomization test when prompts share entities/passages. Save
item-level outputs so the discordant transitions are auditable.

“0.5% of weights” should state its denominator. The saved edit contains 671,089 scalar entries: rho is
0.5% of eligible residual-writer entries in the selected layer set, not 0.5% of all 8B parameters. A
reviewer will notice if the denominator changes across layer sets or models.

## 2. Does P0 fix selection on test?

**It can, but the current wording does not guarantee it.** There are two coherent designs; plan v2
mixes them.

### Design A: freeze the pilot result

Treat `[23,16]`, rho=0.005, the BLADE-G definition, thinking-off generation, the direction/moment
training set, and all thresholds as decisions made from pilot data. Construct a genuinely new,
versioned confirmation set. Evaluate once. Do not reselect ELS, rho, judge prompt, marker list, or
benchmark membership from this set. This gives a clean confirmatory test of the already selected edit.

### Design B: validate the whole selection algorithm

If the intended claim concerns the pipeline rather than the fixed `[23,16]` mask, use nested splits:

1. direction/moment training;
2. inner ELS/hyperparameter selection;
3. outer untouched evaluation.

Repeat across outer splits or on a separately generated confirmation bank. The outer set cannot be
used to revise the judge or thresholds. Report variability in selected layers and masks; stability is
part of the localization claim.

Plan v2 says both “re-split and leave ELS on its screen” and later fixes `L*=[23,16]` in the Axis A
primary. It needs to choose. If P0 reselects and finds a different L*, `[23,16]` cannot remain the
confirmatory pipeline output without an explicit fixed-mask versus re-fit-pipeline distinction.

The judge also creates a new development channel. Develop its rubric and prompt on pilot outputs,
validate on a separate stratified sample, then lock it before confirmation. “Validate vs ~100 human
labels” is too vague: 100 spread across five families, several datasets, four response classes, and
multiple intervention conditions leaves almost no examples per cell. Use at least two blinded human
raters, report classwise agreement/confusion and adjudication, and deliberately sample rare and
base/edit-discordant cases. The judge must see the question/context/reference needed to assess
correctness; it should be blind to condition, mask, family metadata, and study hypothesis—not blind to
the prompt itself.

## 3. Axis B is correctly early, but is not yet a leak-free or construct-valid test

### 3.1 “Leave one family out for the weight edit” must mean the entire fitted pipeline

For outer fold family F, F must be absent from:

- direction estimation at every layer;
- writer-input moments and BLADE/BLADE-G scores;
- lambda/generic-cost calibration if behavior data enter it;
- solo-layer screening and best-first ELS;
- rho/alpha choice and stopping rules;
- judge/rubric development and any prompt filtering.

Only generic C4 importance may be shared. The four training families should have their own inner ELS
split. Then evaluate the learned mask once on new examples from F. Controls must be produced from the
same training fold and evaluated on the identical held-out prompts.

If `[23,16]` is held fixed from all five families, this is a legitimate but narrower test: **conditional
on globally selected layers, does the within-layer edge ranking learned from four families affect the
fifth?** It is not LOFO validation of the full weight-selection mechanism. Both versions could be
reported, but they answer different questions.

The phrase “on the untouched split” is dangerous. If P0's untouched set is used to construct or select
five fold-specific edits, it has become development data. Use a separate Axis-B outer bank, or perform
the nested procedure before opening one final confirmation bank.

### 3.2 Five hand-built families are fixed cases, not five independent replications

The five outer folds have heavily overlapping training data and represent deliberately chosen behavior
types. They cannot be treated as n=5 iid samples from a population of uncertainty families. Report each
family as a fixed stratum and a prespecified weighted aggregate. “At least 15 points in every fold” is
a stringent descriptive success rule, not a statistical analysis. It needs paired confidence intervals
and a rationale for the margin. Power should be based on discordant-pair rates within each family, not
only total n.

### 3.3 Axis B does not decide “nonexistence detector versus epistemic uncertainty”

LOFO breaks exact template identity, but all current generators make unknowability correlate with
surface anomaly: uncommon/fictional names, future tense, large future years, “exactly,” implausible
biographical detail, and fanciful titles. A representation can generalize from four families to the
fifth by coding anomaly, temporal futurity, or answer disposition without estimating whether evidence
supports an answer.

Consequently, success on every current fold licenses only:

> the sparse edit learned from four templated answerability contrasts transfers to the fifth tested
> contrast and changes expressed answer disposition.

It does not yet license “epistemic uncertainty.” That label requires a crossed manipulation in which
surface content and entity familiarity are held fixed while evidence sufficiency changes (proposed in
Section 9 below), plus a functional correctness/calibration consequence.

## 4. Axis A reintroduces selection on the target set

### 4.1 Direction AUROC should be a moderator, not an exclusion gate

Excluding a dataset from the pooled primary when frozen-v AUROC < 0.75 selects evaluation
distributions according to success on the target labels. This is not weight fitting, but it is still
selection on test and converts “OOD transfer” into “transfer on OOD sets where the representation
already transfers.” The 0.75 threshold is also arbitrary and apparently chosen after seeing how easy
the synthetic AUROCs are.

Report all prespecified datasets in the primary analysis. Use direction AUROC, with a clustered CI, as
a prespecified moderator in a decomposition:

- direction fails, weight edit fails;
- direction transfers, edit fails;
- both transfer;
- edit transfers despite weak linear readout.

The first case is an informative failure of end-to-end OOD generalization, not a reason to remove the
dataset. A secondary, clearly conditional estimand among direction-positive datasets is fine if its
membership rule is frozen in advance, but it cannot replace the all-dataset headline.

There is also a construct issue: benchmark answerability labels are not the same as model-specific
uncertainty. A model can fail an “answerable” item or know an item labeled context-unanswerable from
pretraining. AUROC against dataset labels measures encoded task answerability, not necessarily “does
this model know the answer.” Report both gold answerability and base-model correctness/disposition.

### 4.2 Headroom stratification is useful; headroom-based benchmark exclusion is not

For deterministic decoding, the most interpretable outcome is a paired transition table for every
prompt:

- base abstains -> edited answers, with correctness of the edited answer;
- base answers -> edited abstains;
- answer -> answer, with correctness change;
- abstain -> abstain.

The first transition is REMOVE's direct opportunity set. It is reasonable to report it as a
prespecified conditional estimand. It is not reasonable to inspect the target set's base rate and then
exclude the whole dataset from the primary. Retain all datasets, report that a floor/ceiling makes the
corresponding conditional effect weakly identified, and use the transition endpoint.

With stochastic decoding, selecting prompts using the same baseline draw creates regression-to-the-
mean and unstable strata. Define strata with an independent baseline batch or cross-fitting, then
evaluate across new draws. Generation seeds are repeated measurements, not new independent prompts.

### 4.3 The proposed benchmarks do not share one response semantics

A single pooled “abstention on unanswerables” endpoint is not automatically meaningful:

- SelfAware's local unanswerable split includes subjective preferences, philosophical questions,
  malformed premises, and genuinely unknowable facts. A nuanced discussion can be appropriate without
  abstaining.
- SQuAD 2.0 tests whether an answer is supported by a supplied passage. The prompt must explicitly
  require context-only answering; otherwise parametric knowledge changes the task.
- A false-premise benchmark should reward a confident, correct premise correction. That is neither
  hedging nor “confident-wrong.”
- Freshness/real-time benchmarks test temporally scoped factual knowledge. Their questions are not
  intrinsically unanswerable; snapshot date, reference date, scoring, and the model's actual knowledge
  matter.

Predefine a task-specific correct-response policy and scorer for each dataset. Pool only a common
binary disposition dimension after establishing measurement invariance, and separately meta-analyze
functional correctness. An equal-weight average of dataset-level paired effects is more defensible
than concatenating all prompts, which lets the largest dataset define the result.

## 5. The four-way judge schema still conflates dimensions

`{appropriate abstention/caveat, confident-correct, confident-wrong, hedged-wrong}` is not mutually
exclusive or exhaustive across the proposed tasks. “This country is fictional” can be confident and
correct, a premise correction can be confident and correct, a response can contain both a caveat and a
correct answer, and some unanswerable prompts have no proposition on which “correct/wrong” is defined.

Use orthogonal labels instead:

1. **Disposition:** direct answer / explicit non-answer / premise correction / mixed-discussion;
2. **Epistemic language:** none / calibrated caveat / strong uncertainty;
3. **Factual status:** correct / wrong / unsupported / not applicable;
4. **Task appropriateness:** appropriate / inappropriate under a dataset-specific rubric;
5. **Output integrity:** truncated / degenerate / incoherent.

The primary can be an explicit non-answer/premise-correction disposition if that is the construct, but
do not infer hallucination merely from reduced hedge language. The functional endpoint should be the
increase in **unsupported or wrong commitments** among prompts for which the base appropriately
withheld an answer.

Also track truncation rather than merely increasing `GEN_TOKENS`. A common token budget can induce
different censoring across conditions when the edit changes verbosity. Prefer task-appropriate concise
answer instructions, enough generation budget, and a reported truncation rate.

## 6. Controls: what they establish and what they do not

### 6.1 Random and damage-matched controls

Three random masks do not estimate the random-mask null distribution. Use enough independently drawn
masks (roughly 20--50, budget permitting) to quantify mask-to-mask variability, and treat both prompt
and mask as uncertainty sources. The old random-matches-BLADE failure makes this especially important;
an in-distribution random null does not rule out an OOD-specific random effect.

Match random masks on more than scalar count:

- exact per-layer and per-matrix counts;
- BLADE-G generic-importance/Q distribution;
- weight-magnitude and fan-in/fan-out distributions where feasible;
- validation-set changes in C4/domain NLL, benign KL, response length, and answerable accuracy.

The existing “20x damage” result removes many more entries until C4 perplexity roughly matches. It is a
useful catastrophic-random reference, but it is not a clean equal-damage causal control because count,
matrix occupancy, and domain damage differ drastically. Tune damage matching on separate validation
data and evaluate both masks on untouched outcomes. Failure to reject a random effect is not evidence
that it is flat; use an equivalence interval with a prespecified margin.

### 6.2 Shuffled-r

Coordinate-shuffling r breaks the residual basis geometry and can produce a very different score
distribution and eligible-positive set. Keep it, but add a stronger pipeline-level null: permute
certain/uncertain labels within matched strata, then recompute r, moments, BLADE-G scores, and ELS under
the same budget. This preserves the fitting/search opportunity while destroying the behavioral
relationship. Check that every method can supply the requested number of finite positive candidates.

### 6.3 Wrong-layer control

One hand-picked pair such as `[10,27]` is not enough. AUROC equivalence does not make layers causally or
numerically matched, and a negative at one pair can be luck. Conversely, a positive at another layer
does not mean ELS “localizes nothing”; it can mean the behavior has redundant or distributed causal
sites.

There are two fair tests:

- **Conditional edge specificity:** freeze `[23,16]` and compare BLADE weights to matched null weights
  within those layers.
- **Layer-selection specificity:** on the selection split, evaluate all single layers or a
  prespecified distribution of matched layer pairs; lock ELS's choice; compare its untouched effect/
  damage efficiency with the full held-out distribution. A nested permutation of the complete ELS
  pipeline is the cleanest significance reference.

If selected layers win independently, say “ELS identifies more edit-effective layers under this
budget.” “The mechanism localizes to L16/L23” remains stronger than the evidence because causal sites
can be nonunique.

### 6.4 Cross-behavior refusal control

This is important, but the proposed `>=50% in either direction => shared decline circuit` rule is too
crude and its conclusion too strong. The two tasks have different baselines, headroom, metrics, layers,
mask sizes, and damage. A behavioral cross-effect establishes lack of functional specificity; it does
not by itself prove that both behaviors share one circuit. Collateral damage or convergence on a common
output vocabulary can produce the same result.

Use a 2 x 2 own-versus-cross design and test the interaction at matched realized damage:

| fitted mask | epistemic evaluation | refusal evaluation |
|---|---:|---:|
| epistemic | own effect | cross effect |
| refusal | cross effect | own effect |

Report headroom-normalized effects only as secondary, alongside absolute paired effects. Predefine an
equivalence/noninferiority margin for cross versus own effects. Also report mask overlap conditional on
layer/matrix occupancy and whether each edit changes projection onto the other direction. If cross
effects are large but representational changes differ, the correct conclusion is “behaviorally
non-specific output control,” not yet “shared circuit.”

### 6.5 Verbosity/directness control

The proposed contrast is useful but not decisive. “Answer in one sentence” versus “answer carefully
with caveats” changes explicit instruction-following tokens, not only verbosity. Build it from matched
questions/completions or at least counterbalance the instructions and validate that it changes length
without changing answerability decoding.

If its edit reduces hedge words, that shows a second route to the same surface metric. It invalidates
the epistemic interpretation only if it also reproduces the **functional transition** to unsupported/
wrong commitments, has comparable residual effects, or overlaps the same weights more than expected.
Otherwise distinct mechanisms can converge on shorter answers. Length is partly a mediator of output
style, so simply covarying it away can also remove the phenomenon of interest.

### 6.6 Steering reference

The three-way interpretation in plan v2 is too categorical:

- steering success plus weight success does not show that the weights “do what the direction does”;
- both failing does not prove templated specificity (dose, layer, position, and output regime can fail);
- steering success plus one sparse deletion failure only bounds that deletion operator, not all weight
  realizations.

At minimum, measure the edit-induced residual change on base versus edited forward passes and decompose
it into projection along v and orthogonal energy across prompts/tokens. Show that signed projection,
not merely rho or damage, predicts response transitions. Stronger evidence would add a rescue or
mediation test: steering along v restores behavior after deletion, or clamping/projecting v attenuates
the edit effect. Without one of these, call steering an intervention reference, not mechanism identity.

### 6.7 Preservation controls

Answerable accuracy is necessary but insufficient. Report base-correct retention, base-wrong changes,
new abstentions, task-specific accuracy, benign-prompt KL, domain NLL, coherence/degeneration,
truncation, and length. Use equivalence margins rather than “not significant.” Because BLADE-G's Q is
estimated on C4, a C4 PPL guard is partly aligned with the selection objective; domain-matched
preservation is essential.

## 7. Statistical design and power

### 7.1 Define paired estimands, not just rates

For prompt i and condition c, let A_ic denote judged abstention. REMOVE's effect is the paired mean
`mean_i(A_i,base - A_i,edit)`. Selection specificity is the paired difference between that effect and
the corresponding control-mask effect on the same prompts. Bootstrap/resample clusters once and carry
the identical resample through every condition. For random masks, use a crossed prompt-by-mask
hierarchical bootstrap or a mixed model.

The functional co-primary/major secondary should replace A with unsupported/wrong commitment. Otherwise
the paper can show only deletion of a phrase style.

### 7.2 `n >= 300/set` is not a power analysis

Power depends on baseline rate, discordant transition probability, intracluster correlation, number
and size of clusters, judge error, and the specificity/equivalence margin. Simulate or compute power
for the paired primary before fixing n. For passage/category clustering, the number of independent
clusters—not the number of questions—is the limiting quantity. Ordinary cluster bootstrap is unstable
with few clusters; use a randomization/wild-cluster method or redesign to obtain more clusters.

The same applies to P0's `n >= 150/class` and Axis B's 40-per-cell suggestion. These may be adequate,
but they are not justified by an independent-proportions standard error. Use the expected discordance
matrix. Avoid an OOD/in-distribution effect ratio as a primary statistic: its denominator is noisy and
can be close to zero. Report absolute effects; bootstrap a ratio only as secondary.

### 7.3 “Gap versus max-control” is an opaque primary

The controls are heterogeneous scientific hypotheses, not exchangeable replicates. Taking the maximum
of noisy control effects is conservative in one sense but creates a biased, hard-to-interpret comparator.
If retained, the maximum must be recomputed inside every joint bootstrap replicate. A cleaner plan is:

- primary: BLADE versus a prespecified matched random-mask distribution;
- key specificity tests: BLADE versus label-permuted pipeline, wrong-layer distribution, refusal mask,
  and verbosity mask, each with a prespecified margin;
- familywise correction across this small key-control family (Holm or max-T);
- all remaining controls descriptive.

“Controls flat” requires two one-sided equivalence tests or CIs contained within (say) +/-5 points,
not nonsignificant p-values. Pick the margin based on the scientific claim, then power for it.

### 7.4 Pooling and heterogeneity must be specified

Predefine whether the pooled Axis-A effect gives equal weight to datasets, categories, clusters, or
items. I recommend equal weight to prespecified dataset-level effects plus all per-dataset CIs and a
heterogeneity statistic. Do not let SQuAD's item count dominate SelfAware or freshness datasets.
Benchmark-specific response policies make correctness better suited to a meta-analysis than raw item
pooling.

The five Axis-B folds are not independent because their training sets overlap. Analyze held-out items
with family fixed effects and report the family-specific estimates; do not run a t-test across five
fold point estimates.

### 7.5 Sampling seeds do not solve the main uncertainty

Greedy decoding with prompt/cluster-level inference is a defensible primary. T=0.7 with three seeds is
a robustness analysis, not three independent replications. If used, keep prompt identity in the model
and never multiply the nominal sample size by the number of seeds. More random-mask seeds and more
independent semantic clusters are higher priority than three decoding seeds.

### 7.6 Sequential gates and multiplicity

Declaring one final primary is good, but P0 and P1 are used to change labels, stop work, and potentially
rescope the paper. Pre-register those decision rules and which data remain untouched after each gate.
If inferential claims accumulate across gates, use a closed testing/alpha-spending scheme or clearly
label P0/P1 as independent confirmation studies rather than treating every successful gate as another
primary. All already observed results are pilot/exploratory; a preregistration now cannot retroactively
make them confirmatory.

For AMPLIFY, `p < 0.01` is not a scientific criterion. Specify the minimal increase, a paired CI, a
local monotonic/trend contrast, and a damage bound. The current 0.56 -> 0.79 suppressor result and raw
alpha result are underpowered and were explored over operations/doses; they require independent
confirmation.

## 8. Axis C and Axis D need reinterpretation

### 8.1 Axis C's current confidence metric is itself edited behavior

`eval_calibration.py` first generates an answer under the edited model, then asks the **same edited
model** for an integer confidence in a second prompt. An edit to caveat/abstention expression can
directly change that integer response without changing epistemic discrimination. Conversely, a change
in answer accuracy alters ECE/Brier even if the confidence mapping is unchanged. This makes verbalized
confidence a downstream behavioral endpoint, not an independent measure of internal calibration.

Use at least one confidence signal not defined by the same surface circuit—for example normalized
answer likelihood/margin where the task supports it, self-consistency estimated with a fixed procedure,
or a selective-prediction score whose extraction is validated not to be directly controlled by the
edit. Report accuracy and confidence-discrimination separately. Brier/log score are preferable
primaries to bin-sensitive ECE; risk-coverage should use paired bootstrap CIs.

Do not select “problems the base model is unsure of” from the same evaluation responses. Define a
difficulty band on separate calibration data or report continuous interactions with base confidence.
Thinking-on/off comparison is still useful, but the exact scheme-A direction/moments/mask must be
passed to the harness rather than its current reasoning-behavior artifact.

### 8.2 Model OOD is replication, not transfer

Recomputing a direction, moments, ELS, and mask on Qwen3-14B tests cross-scale reproducibility of the
recipe. Repeating on Gemma/Llama tests cross-family reproducibility. Neither is “model OOD transfer” of
the Qwen3-8B weights. Rename Axis D to **cross-model replication** and predefine which hyperparameters
are fixed in dimensionless terms versus legitimately reselected. Report mask/layer stability only
after accounting for architecture and depth.

## 9. Claim taxonomy: what is and is not licensed

Plan v2 correctly drops “necessity” and keeps injection-on-known prompts out of the OOD headline. The
remaining taxonomy still needs tightening.

- **Selective removal:** earned only by a held-out BLADE-versus-well-matched-null effect with capability
  equivalence. Current random controls are encouraging pilot evidence.
- **Graded removal:** requires a prespecified, independently evaluated local dose/rho trend. A single
  rho Axis-A primary does not establish that gradedness transfers. The existing rebound at rho=0.02
  makes this an open question.
- **Expressed abstention/answer disposition:** the defensible current construct, after fixing the
  judge. “Hedging” is narrower and should not include confident premise denial.
- **Epistemic uncertainty:** not licensed by current family LOFO alone. It needs sensitivity to evidence
  sufficiency while holding content/familiarity fixed, plus a correctness/calibration consequence.
- **Bidirectional intensity control:** not yet established. Raw `alpha W` moves relative to the
  arbitrary weight origin; suppressor removal uses a different weight set. If each side replicates,
  call it a **push-pull pair of sparse actuators** or “gain modulation on already-uncertain inputs,” not
  one bidirectional weight axis unless the same signed support and mechanism-equivalence tests pass.
- **Input-gated boundary:** currently a hypothesis, not a result. Failure to inject at a behavioral
  floor can reflect thresholding, insufficient/incorrect edit, metric failure, or superposition. The
  existing steering comparison is additionally confounded by thinking mode, token budget, layer, and
  intervention position. Establish gating by measuring edit-induced writer/residual changes on known
  versus unknown inputs and by crossing evidence sufficiency within the same input content.
- **Localized mechanism:** beating one wrong-layer pair is insufficient. “Edit-effective sparse weights
  in L16/L23” is safer. Mechanism identity requires at least induced-v projection plus mediation/rescue,
  not merely two interventions that both change output.

The strongest honest paper-level claim after a clean P0/Axis A would be:

> A sparse set of Qwen3-8B residual-writer parameters, selected from templated answerability contrasts,
> selectively controls expressed answer disposition on held-out prompts and transfers to specified
> external tasks, relative to matched null edits, with measured effects on unsupported commitment and
> bounded capability change.

Only the counterfactual evidence test below would justify replacing “answer disposition” with a more
epistemic label.

## 10. What a skeptical reviewer will still attack

In likely order:

1. **The contrast is shortcut-rich.** Perfect late-layer AUROC mostly shows that the model represents
   future tense, rare/fictional entities, odd predicates, and prompt length. It is not evidence of a
   calibrated “I do not know” latent variable.
2. **The evidence chain is configuration-inconsistent.** Thinking-on direction/steering and thinking-off
   BLADE outputs are currently presented as one mechanism.
3. **Axis A cherry-picks successful OOD sets.** AUROC/headroom exclusions use target outcomes to define
   the primary pool.
4. **The primary outcome is still answer style.** Even a good judge of abstention does not show what the
   model believed; the decisive harm/functional result is unsupported commitment and accuracy.
5. **ELS specificity is not established.** One wrong pair cannot validate a greedy search over 36
   layers, and selection instability is not reported.
6. **The null distribution is undersampled.** Three random masks and one shuffled direction do not
   characterize parameter-edit variability, especially after the project's prior OOD random-control
   failure.
7. **External datasets are pooled despite different correct policies.** Premise correction, context
   non-answerability, subjectivity, and stale knowledge are not one label.
8. **No activation-level causal bridge is shown.** Sparse deletion and steering may independently hit
   the same output behavior; neither establishes that the selected weights implement v OOD.
9. **“14B model OOD” is really a refitted replication.** It does not demonstrate cross-model transfer.
10. **Existing results are over-described.** Paired data are tested as independent, “length-controlled”
    is inaccurate, 0.5% lacks a denominator, and amplify/input-gating language outruns significance and
    mechanism evidence.

## 11. The single most decisive experiment

The most informative single experiment is a **held-out counterfactual evidence-sufficiency crossover**,
not the current five-family LOFO alone.

Construct new item quadruples with identical question wording wherever possible:

- familiar entity + decisive context containing the answer;
- familiar entity + matched context omitting the answer;
- nonce/fictional entity + decisive context containing the answer;
- the same nonce/fictional entity + matched context omitting the answer.

Counterbalance answer position, context length, lexical content, and premise wording. The key is that a
nonce entity is fully answerable when the prompt supplies the fact, while a familiar-looking question
can be unanswerable when the relevant evidence is withheld. Fit the direction, moments, ELS, mask, and
judge nowhere on these items. Evaluate base, fixed BLADE, many matched random masks, label-permuted
pipeline, verbosity mask, and (if budget permits) refusal mask under identical decoding.

Pre-register a paired difference-in-differences primary:

> the BLADE-specific drop in appropriate non-answering when evidence is absent minus its effect when
> evidence is present, with no interaction that merely tracks entity novelty.

Also require that evidence-present answer accuracy remains equivalent and that evidence-absent
unsupported commitments rise when abstention falls. Measure edit-induced projection onto v in all four
cells.

This experiment **most strengthens** the claim if the edit follows evidence availability within the
same entity/question, generalizes equally to familiar and nonce entities, beats matched masks, and its
induced v-projection mediates response transitions. It **most endangers** the claim if the effect follows
nonce/fictional wording regardless of supplied evidence, suppresses caveats equally when the answer is
present, or is reproduced by the verbosity/refusal masks. In the latter cases, the current phenomenon
is better described as anomaly detection or generic answer-forcing/decline control.

The cross-behavior refusal experiment remains the cheapest early danger check and should still run
before an expensive benchmark sweep. But the evidence-sufficiency crossover is more decisive because
it directly breaks the central confounding correlation in the training data instead of comparing two
already heterogeneous behaviors.

## 12. Concrete revision of the execution order

### P0a: repair the factual and measurement foundation

1. Audit and correct item labels; remove fictional items with plausible canonical answers or score
   them against an explicit reference policy.
2. Rebuild the exact thinking-off direction used by BLADE; run LOFO with nested surface/length controls.
3. Decide explicitly between a fixed `[23,16]` confirmation and a nested validation of the full ELS
   pipeline. Do not mix them.
4. Lock a multidimensional judge rubric on pilot outputs; validate it with stratified double-human
   annotation. Save all item-level outputs and judge rationales/labels.
5. Confirm the paired REMOVE and rho trend on a new set; use paired tests and state the sparsity
   denominator. Treat amplify and input gating as hypotheses until separately confirmed.

### P0b: cheap specificity threats

6. Run the 2 x 2 epistemic/refusal own-cross evaluation, but interpret behavioral cross-transfer as
   non-specificity rather than automatic proof of a shared circuit.
7. Run a label-permuted full-pipeline null, enough matched random masks, and an independently evaluated
   layer-selection distribution. Add the carefully constructed verbosity/directness mask.

### P1: decisive controlled OOD

8. Run the counterfactual evidence-sufficiency crossover as the main construct-validity experiment.
9. Run nested five-family LOFO as evidence of transfer across the five synthetic generators, without
   using it alone to rename the construct.

### P2: external OOD

10. Freeze all benchmark memberships and analysis rules. Include every prespecified set; use direction
    AUROC and base headroom as moderators/transition strata, never as exclusion gates. Use dataset-
    specific appropriateness/correctness policies and equal-weight dataset meta-analysis.
11. Make the confirmatory contrast BLADE versus a well-estimated matched-null distribution. Apply
    multiplicity correction to a short list of key specificity controls and equivalence tests to
    preservation outcomes.

### P3/P4: scope and replication

12. Rebuild Axis C around an uncertainty score not reducible to the same verbalized-caveat output, and
    run exact scheme-A artifacts thinking off/on.
13. Rename Axis D cross-model replication; refit independently under frozen dimensionless rules on 14B
    and then another model family.

## Bottom line

P0 plus the reordered Axis B addresses Fable's original selection-on-test and lexical-metric problems
only **if** splits, judge development, and the entire fitted pipeline are strictly nested. As currently
worded, it leaves leakage paths and then creates new target-selection gates in Axis A. More deeply,
neither a semantic judge nor five-family LOFO removes the construct confound built into the prompt
generator.

The project already has an unusually strong pilot causal effect and encouraging null edits. The
highest-value move is now to make answerability counterfactual while holding entity/question content
fixed, quantify a paired BLADE-versus-null interaction, and connect the sparse edit to v at the
activation level. A clean result there would be substantially more convincing to a mechanistic-
interpretability reviewer than adding four loosely pooled benchmarks; a failure there would save the
team from overclaiming an epistemic mechanism when it has found a powerful answer-style or anomaly
controller.
