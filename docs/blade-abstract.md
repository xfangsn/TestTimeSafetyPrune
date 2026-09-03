# BLADE — abstract (synthesized from Claude + codex gpt-5.6-sol + kimi k3)

## Final (synthesized)

Post-training instills dispositional behaviors in large language models---refusing harmful
requests, agreeing with users, seeking power, deceiving---yet localizing and controlling them
without degrading general capability remains difficult. We introduce **BLADE** (Behavioral
Localization via Activation-Difference Estimation), a gradient-free, forward-only method that
scores each scalar weight of a layer's residual-writer matrices by its signed,
direction-projected contribution to the output shift along a difference-in-means behavior
direction, and uses best-first Effective-Layer Selection to choose a sparse layer set under a
perplexity budget (calibrated on C4, reported on held-out WikiText). Scaling only the selected
weights gives a single control knob: α=0 removes the behavior, 0<α<1 attenuates it, and α>1
amplifies it. Across four model families and up to eight behaviors, BLADE edits only
≈0.002–0.09% of weights while holding held-out perplexity within a few percent and six zero-shot
tasks nearly unchanged; the selected weights are modulatory-like---behavior-selective and
capability-sparing---not the driving bulk of the matrices. These behaviors are post-training
artifacts (the base model sits near chance), and at matched sycophancy BLADE incurs essentially
no downstream accuracy loss versus a 3.8-point drop for weight-space task-vector steering,
without training. Transfer depends on the contrast: refusal removal generalizes across harmful
benchmarks (HarmBench 0.91→0.09), whereas format-bound sycophancy does not; editing also composes
with activation steering.

---

## Draft — codex (gpt-5.6-sol)

Post-training can instill behaviors whose control without degrading general capabilities remains
difficult. We introduce Behavioral Localization via Activation-Difference Estimation (BLADE), a
gradient-free, forward-only method for locating and controlling such behaviors. BLADE scores
scalar weights in residual-writer matrices by their signed direct contribution to an output shift
along a difference-in-means behavior direction, then uses best-first Effective-Layer Selection to
choose a sparse layer set under a perplexity budget. Scaling only the selected weights provides a
single control parameter: zero removes the localized behavior, intermediate values attenuate it,
and values above one amplify it. Across four model families and up to eight behaviors---including
refusal, sycophancy, power-seeking, deception, and self-awareness---BLADE edits approximately
0.002–0.09% of model weights while keeping held-out WikiText perplexity within a few percent and
performance on six zero-shot tasks nearly unchanged. At matched sycophancy, it incurs
approximately no downstream accuracy loss versus a 3.8-point drop for weight-space task-vector
steering, without training. Transfer depends on the behavioral contrast: refusal removal
generalizes across harmful benchmarks (HarmBench refusal rate 0.91→0.09), whereas A/B sycophancy
does not transfer to a different format. BLADE also composes with activation steering.

---

## Draft — kimi (k3)

Post-training instills behaviors such as refusal, sycophancy, power-seeking, and deception into
language models, yet localizing and controlling these behaviors remains difficult. We introduce
BLADE (Behavioral Localization via Activation-Difference Estimation), a gradient-free,
forward-only method that identifies the scalar weights of a layer's residual-writer matrices most
responsible for a target behavior. Each weight is scored by its signed, direction-projected
contribution to the output shift along a difference-in-means behavior direction, and best-first
Effective-Layer Selection picks a sparse layer set under a perplexity budget. BLADE then edits
only the selected weights with a single scaling knob: alpha=0 removes the behavior, intermediate
values attenuate, and alpha>1 amplifies it. Across four models and seven to eight behaviors, BLADE
removes behaviors by editing roughly 0.002–0.09% of all weights with near-zero capability cost on
held-out perplexity and six zero-shot tasks, and it matches weight-steering baselines on behavior
removal with no downstream degradation. The selected weights behave as modulatory,
behavior-selective components rather than the matrices' driving bulk. Transfer is strong for
refusal but weaker for sycophancy, and editing composes with activation steering.

---

## Conference version (~120 words)

We introduce **BLADE** (Behavioral Localization via Activation-Difference Estimation), a
gradient-free, forward-only method that localizes and controls behaviors instilled by
post-training in LLMs. BLADE scores each residual-writer weight by its signed contribution to a
difference-in-means behavior direction and, via best-first layer selection under a perplexity
budget, edits only ≈0.002–0.09% of weights; scaling them removes (α=0), attenuates, or amplifies
(α>1) the behavior. Across four models and eight behaviors---refusal, sycophancy, power-seeking,
deception, and more---BLADE removes the behavior at near-zero capability cost (held-out perplexity
and six zero-shot tasks nearly unchanged), beating a task-vector baseline at far lower cost with no
training. Removal transfers for refusal (HarmBench 0.91→0.09) but not for format-bound sycophancy,
and composes with activation steering.
