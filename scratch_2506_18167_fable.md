# arXiv 2506.18167 vs our epistemic-uncertainty construct

**Paper**: Venhoff, Arcuschin, Torr, Conmy, Nanda. *Understanding Reasoning in Thinking Language Models via Steering Vectors.* ICLR 2025 Workshop on Reasoning and Planning for LLMs (v4, 22 Oct 2025). Code: github.com/cvenhoff/steering-thinking-llms.

Source read: arXiv abs page + full PDF (18 pp., incl. appendices A-E).

## 1. What their "uncertainty estimation" is

A **reasoning behavior expressed inside the chain-of-thought**, one of six GPT-4o-annotated sentence categories in DeepSeek-R1-Distill thinking traces:

| label | definition (Sec. 3 / App. A annotation prompt) |
|---|---|
| initializing | rephrases the task, states initial thoughts |
| deduction | derives conclusions from current approach |
| adding-knowledge | "enriching the current approach with recalled facts" |
| example-testing | generates examples to test hypothesis |
| **uncertainty-estimation** | "The model explicitly states its confidence or uncertainty regarding its reasoning" / annotation prompt: "The model is stating its own uncertainty" |
| backtracking | abandons current approach, tries alternative |

Concrete spans labeled uncertainty-estimation in their appendices:
- "That might be a stretch, though." (App. B, riddle trace)
- "Hmm, let's see." (App. E, probability trace)
- "52 divided by 6 is approximately 8.666..., but maybe I should compute it step by step." (App. E)
- steered output: "Hmm, probability problems can be tricky," (App. E, positive steering)

So it is **hedging/meta-cognitive commentary about the ongoing reasoning process** on tasks that all have answers (math, spatial, verbal logic, riddles, probability, etc.; 500 Claude-3.5-Sonnet-generated tasks across 10 categories, Table 1). No task is unanswerable; there is no notion of "does the model know this fact". Zero mentions of hallucination, abstention, unanswerable questions, calibration, nonexistent entities, or closed-book QA anywhere in the paper.

Models: DeepSeek-R1-Distill-Qwen-14B, -Qwen-1.5B, -Llama-8B (steering); five distills + five non-thinking baselines for the prevalence survey (Fig. 2: thinking models ~9% uncertainty-estimation sentences vs ~2% for baselines).

## 2. How their direction is constructed

- **Data**: model-generated reasoning chains (greedy, max 1000 tokens) on the 500 tasks; GPT-4o splits each chain into labeled spans.
- **Position**: activations at **generated reasoning tokens** — for each annotated span of category c, the token *preceding* the span plus the span itself (up to 10 tokens); per-prompt mean over those positions.
- **Contrast**: Difference-of-Means, but D+ = prompts containing at least one span of category c, D- = the **full dataset** (no matched counterfactual). Normalized to the norm of the mean overall activation.
- **Layer selection**: attribution patching (KL of next-token dist. when adding u at the position preceding a labeled span); choose max-KL middle layer, skipping early layers that correlate with embeddings (Table 2: L12 Llama-8B, L18 Qwen-1.5B, L29 Qwen-14B for uncertainty-estimation).
- **Intervention**: activation addition/subtraction of the vector at inference time (no weight editing, no localization of which weights write the direction).
- **Evaluation**: fraction of sentences GPT-4o labels as the target behavior on 50 unseen reasoning tasks, under positive vs negative steering (Fig. 4). No accuracy, no abstention rate, no hallucination, no ppl reported.
- **Result for uncertainty vector**: positive steering raises the uncertainty-estimation sentence fraction (e.g. Qwen-14B ~19% -> ~46%), negative steering drives it to ~1-2%. Behavior frequency only.
- **App. C**: cosine between behavior vectors; uncertainty-estimation and backtracking are "moderately" correlated (~0.68 Llama-8B, ~0.78 Qwen-14B per heatmap; our docs/plan-thinking-eval-impact.md already cites these numbers), and in Qwen-14B uncertainty also sits ~0.7 with deduction. They still claim the behaviors are distinct because steering effects are behavior-specific.

## 3. Same construct as ours? **No — different construct.**

| axis | Venhoff et al. 2506.18167 | Ours (Qwen3-8B epistemic parametric-knowledge uncertainty) |
|---|---|---|
| **Construct** | *Process* meta-commentary: "I'm not sure about this step" while solving an answerable reasoning task. Verbal hedging as a CoT style/behavior. | *Epistemic state about parametric knowledge*: does the model know the fact? Whole-response answer disposition (abstain vs answer) on genuinely unknowable / nonexistent-entity questions. |
| **Locus / position** | Generated reasoning tokens mid-CoT (span + preceding token, up to 10 tokens), thinking models only. | Last prompt token, pre-generation, fixed position; no CoT needed; works on the answer disposition before any token is emitted. |
| **Contrast** | Same task distribution; D+ = traces containing a hedging span vs D- = all traces. Contrast is over *which sentences the model happened to emit*, not over inputs. | Two prompt sets that differ in the *input's* epistemic status (known fact vs unanswerable); balanced difference-of-means over inputs. |
| **Ground truth** | GPT-4o sentence labels (they flag false positives/negatives in Limitations). | Whether the question is objectively answerable; behavioral outcome judged as abstain / correct / hallucinated. |
| **Intervention** | Activation addition/subtraction at one layer at inference time. | BLADE: localize the residual-writer weights and rescale (α=0 removal, α>1 amplification); weight-level, persists without hooks. |
| **What the edit changes** | Frequency of hedging sentences in the CoT (more/fewer "Hmm, that might be a stretch"). Correctness not measured. | Removal: model stops abstaining and confabulates confident fake answers (hallucination); amplification: more appropriate abstention. Changes the *truthfulness disposition*, not the verbal style. |
| **Evaluation** | GPT-4o-labeled behavior fraction on 50 held-out reasoning tasks. | Abstention rate / hallucination rate on unanswerable vs answerable sets, OOD transfer, WikiText ppl, evidence-sufficiency crossover. |
| **Regime specificity** | Not studied. | Shown regime-specific: closed-book parametric only; does NOT touch context-grounded (passage) uncertainty. |
| **Models** | DeepSeek-R1-Distill 1.5B/8B/14B (thinking). | Qwen3-8B (and non-thinking generation path). |

Sharpest differences: (a) *hedging-as-CoT-behavior* vs *knowledge-aware abstention*; (b) generated mid-CoT spans vs fixed last-prompt-token, pre-generation; (c) contrast over emitted sentences within one task distribution vs contrast over input epistemic status; (d) steering the *rate of hedging sentences* vs editing weights to flip *abstain/hallucinate*.

A model can express a lot of Venhoff-style uncertainty ("hmm, let me double check") while confidently answering an answerable math problem, and can abstain on "capital of Kesteria" with zero mid-CoT hedging. The two are orthogonal constructs that happen to share the word "uncertainty".

Caveat worth being honest about: their "adding-knowledge" behavior (recalling facts into the CoT) is closer in spirit to parametric-knowledge access than their "uncertainty-estimation" is, but it is still a CoT-span behavior on answerable tasks and is not an epistemic-state direction.

## 4. Would a reviewer say "same thing, already done"?

Unlikely on the merits, but the overlap in *naming* ("uncertainty", "steering vector", "difference of means", "Nanda group") is enough that we must pre-empt it explicitly. Their result is: "thinking models have a linear direction that turns hedging sentences on/off in the CoT." Ours is: "there is a pre-generation direction encoding whether the model knows a fact; BLADE-editing its writer weights flips abstention vs hallucination, and it is regime-specific to parametric knowledge." No shared metric, no shared data type, no shared position, no shared intervention. What *would* be shared if we ran a thinking-behaviors study (docs/plan-thinking-eval-impact.md) is the CoT-behavior construct — there the Venhoff vector IS the prior work and must be compared against, not this epistemic line.

## 5. Citation framing / relationships worth citing

- Cite as prior evidence that **CoT-expressed uncertainty is a linearly steerable *reasoning behavior*** in thinking models, and then delimit: "Venhoff et al. steer the frequency with which a thinking model *verbalizes* uncertainty mid-chain-of-thought on answerable reasoning tasks; we instead target the model's *epistemic state about its own parametric knowledge*, read at the prompt's final token before generation, and show that removing it converts abstention into confident hallucination. The two constructs are dissociable: verbal hedging is a property of the reasoning trace, whereas knowledge-awareness is a property of the input/knowledge pair."
- Their App. C cosines (uncertainty-estimation vs backtracking ~0.68-0.78) are useful for the thinking-behavior manipulation plan, not for the epistemic line. They report no comparison to any knowledge/abstention direction, so no cosine or behavioral overlap with our direction exists in the paper. If we want an empirical dissociation for the paper, the cheap check is: extract a Venhoff-style CoT-hedging vector on Qwen3-8B thinking traces and report (i) cosine with our epistemic direction at matched layer and (ii) that steering with it does not change abstention rate on unanswerable questions (and our α=0 edit does not change hedging-sentence fraction on MATH). That would make the "different construct" claim quantitative rather than definitional.
- Adjacent citations they lean on that are closer to our construct and should be cited alongside: Li et al. 2023 (ITI, truthfulness direction) and Arditi et al. 2024 (refusal direction) — both are input-conditioned, last-token-style directions like ours; Venhoff is the CoT-behavior branch.

## Bottom line

DIFFERENT construct. Venhoff et al.'s "uncertainty estimation" = in-CoT verbal hedging as a reasoning behavior, extracted from GPT-4o-labeled generated spans in DeepSeek-R1-Distill traces and modulated by activation addition, evaluated as sentence-frequency. Ours = pre-generation epistemic knowledge-awareness, extracted at the last prompt token from input-contrasted prompt sets, localized to weights by BLADE, evaluated as abstain-vs-hallucinate. No overlap in construct, locus, extraction, intervention, or evaluation; cite and delimit in one sentence.
