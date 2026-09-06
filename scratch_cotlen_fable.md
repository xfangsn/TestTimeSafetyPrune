# Can BLADE / scheme-A control reasoning-chain length? — skeptical assessment

Source doc: `docs/idea-cot-length.md`. Reference paper fetched: arXiv 2507.04742.

## 0. What 2507.04742 actually does

**"Activation Steering for Chain-of-Thought Compression" (ASC)** — Azizi, Baghaei Potraghloo, Pedram (USC), July 2025.

| Aspect | ASC |
|---|---|
| Mechanism | Single **difference-of-means steering vector** in the residual stream. Training-free. No weight edit, no RL, no prompt budget. |
| Contrast data | 50 paired examples per dataset: the model's own **verbose** CoT vs a **GPT-4o-written concise** CoT for the same question ("minimal English, math-centric reasoning"). |
| Position | **Final token of `[question ⊕ CoT]`** — i.e. the end of the completed trace, a *generated-token* contrast, not a prompt-position one. `v = mean_i h(q_i⊕short_i)[-1] − h(q_i⊕long_i)[-1]`. |
| Layer | One mid layer, chosen by hand: L21 for 7B/8B, L57 for QwQ-32B. |
| Application | Additive `h ← h + γ v` at **every decoding step** until EOS. γ ∈ 0.275–0.50, chosen via a KL-divergence bound on the next-token distribution (their theoretical contribution). |
| Models | DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Llama-8B, QwQ-32B. |
| Results | MATH500: −61% tokens (3984→1543) at 89.0 vs 88.8 acc (Qwen-7B); −34% (Llama-8B); −51% (QwQ). GSM8K: −67% (Qwen-7B, 1080→536), −67.5% (Llama-8B), −46% (QwQ). Accuracy flat or +0.2–0.4. 2.73× wall-clock speedup. Cross-dataset vector cosine 0.92. |
| Bidirectional? | **No** — compression only. Lengthening never tested. |
| Locus analysis? | **None** — t-SNE separation only; no per-position, no termination-signal, no layer sweep beyond the heuristic. |
| Baselines | Vanilla CoT, CoD (prompted brevity), DEER (early exit), TCC, SEAL (prior steering). |
| Hard benchmarks? | No AIME/GPQA — only MATH500 and GSM8K, where the models are near ceiling and heavy re-verification is mostly waste. |

Two things to notice, because they define our delta:
1. Their contrast is a **style** contrast (verbose English vs terse math), not a "how many reasoning steps" contrast. A large share of the 60% is plausibly linguistic compression (fewer filler tokens per step), which is why accuracy is untouched. That is the *safe* kind of shortening.
2. Their intervention is applied continuously; the vector is read at the end of a trace. So ASC neither claims nor tests that length is a **pre-generation** commitment. Our recipe implicitly does.

**Adjacent prior work I did not fetch but am fairly confident exists (verify before writing anything):**
- **ThinkEdit** (Sun, Yan, Weng, arXiv 2503.22048): finds an "overly-short reasoning" direction in DS-R1-distills, identifies ~2–4% of attention heads that write it, and **edits their `o_proj` weights** (projecting the direction out) → longer reasoning and higher accuracy on the short-response subset. This is a *sparse weight edit for length*, lengthening-only, projection-based.
- **Overclocking LLM reasoning** (Eisenstadt, Zimerman, Wolf, arXiv 2506.07240): a linear **"thinking progress" direction** decodable per token mid-generation; steering it shortens/lengthens. Direct evidence that (part of) length is tracked during generation.
- **CoT-Valve** (Ma et al., arXiv 2502.09601): a *trained* parameter-space direction (LoRA delta) whose scale continuously controls CoT length — a weight-space α knob, but requires training.
- **SEAL** (Chen et al., 2504.07986): steers "reflection/transition thought" directions to calibrate reasoning.

So "steering to shorten" (ASC, SEAL, Overclocking), "weight edit to lengthen" (ThinkEdit), and "a weight-space length knob" (CoT-Valve, trained) all exist. The unclaimed square is **training-free, sparse, bidirectional, one-knob, with a capability frontier** — and a mechanistic answer to *where the knob lives*.

---

## 1. Contrast / construct design

Candidates from the doc, plus what ASC does:

| Construct | Position | Confounds | Verdict |
|---|---|---|---|
| (a) Same problems, "think briefly" vs "think at length / verify every step" instruction | last prompt token (our recipe, unchanged) | Same problems → no difficulty/topic confound. **Risk: instruction-detection ≠ realized length.** R1-style thinkers are notoriously poor at obeying brevity instructions inside `<think>` (this is the entire motivation for budget forcing / L1). If the instruction moves length only weakly, the direction encodes "brevity was requested", not "will be brief". | **Best first try, gated on a behavioral prerequisite** (instruction must actually move median think-tokens ≥ ~2×). |
| (b) Naturally short vs long *correct* traces on the same problem (sample K, take min/max length) | **cannot be prompt-position** — the prompts are identical, so last-prompt-token activations are identical by construction | Forces a generated-token contrast. Sampling-induced length variance is largely path noise (a lucky early answer), so the "direction" may be a progress/confidence feature, not a mode feature. | Not a scheme-A construct. Useful as the *mid-generation* comparison direction (see §2). |
| (c) Difficulty-graded problems | last prompt token | Fully confounded with a **difficulty** direction. Removing "hard-ness" makes the model treat hard problems as easy → shorter *and* dumber. | Reject as the construct; keep as a **confound control** (report cos(r_length, r_difficulty)). |
| (d) ASC: verbose vs terse rewrite, end-of-CoT token | generated | Style, not step count. Requires an external concise writer. | Their construct; our comparison point. |

**Is a fixed pre-generation position the right locus?** Two separate questions get conflated here:

- *Estimation* locus: r_ℓ and Δμ_j are estimated at the last prompt token.
- *Application* locus: the BLADE edit is a permanent change to `down_proj`/`o_proj`; the edited weights fire at **every** position whose writer-input resembles Δμ_j. The steering screen, as we run it, is also applied at every position.

So the recipe does not *require* that length is decided at the prompt. It requires that the feature present at the last prompt token under a "be brief" instruction is the **same feature** (same r, same writer-input pattern) that is re-expressed at thinking positions and biases the continue/stop decision. That is plausible for a persistent "mode" feature (the instruction is attended to throughout the trace; ASC's cross-task cosine of 0.92 says verbosity is a global direction), and implausible for the actual **termination event** ("I have the answer, confidence high → emit `</think>`"), which is a local, state-dependent computation that a last-prompt-token difference-of-means will not see.

Length = (prompt-time scale/mode) × (trajectory-dependent stochastic termination). Under sampling, the same prompt gives think-token CVs of ~0.5+, so a large fraction of length is not set at the prompt. A prompt-position direction can at best move the first factor. That is not nothing — it is what "reason briefly" instructions do — but it bounds the achievable effect and means the honest name for what we would be controlling is **deliberation mode**, not length per se.

Also: for Qwen3 specifically, instruction sensitivity of the *answer* is much higher than of the *thinking block*. The (a) direction may shorten the final answer while leaving `<think>` untouched. Count think-tokens and answer-tokens separately.

Sign convention: define the behavior as the **long pole** (extended deliberation), r = μ_long − μ_short. α=0 → shorter (compression, the commercially interesting side); α>1 → longer (test-time-scaling side, where ThinkEdit says accuracy can go *up* on problems the model under-thinks).

## 2. Where does length live?

Priors, from mechanism and literature:

- **For prompt-time commitment:** difficulty is linearly decodable from prompt activations and models pre-allocate effort by perceived difficulty; `/think` vs `/no_think` is a prompt-token gate the model honors perfectly; brevity instructions work partially; ASC's direction is global across tasks.
- **For during-generation:** R1-distill length is dominated by re-verification loops ("Wait, let me double-check") triggered by intermediate uncertainty; Overclocking finds a per-token progress direction; `</think>` emission is a local decision; the repo's own backtracking/uncertainty directions (`scripts/blade_reasoning*.py`) are generated-token contrasts precisely because those behaviors are not visible at the prompt.

My expectation: **both, split**. A mode component (prompt-decodable, persistent) and a termination component (mid-generation, local). The uncertainty-crossover finding in the doc ("decodable everywhere ≠ steerable everywhere") is the same lesson: the locus at which you estimate the direction determines which component you touch.

The direct test (cheap, no generation needed beyond one base pass): estimate three directions from the same 200 problems and compare cosines per layer —
1. r_prompt: last prompt token, brief vs thorough instruction (construct a);
2. r_think: mean over thinking tokens of short vs long traces (construct b);
3. r_term: last ~20 tokens before `</think>` vs matched mid-trace tokens (a pure termination direction).

If cos(r_prompt, r_think) is high (>0.6) at the ELS layers and cos(r_prompt, r_term) is low, the mode component is real and separable, and BLADE-from-prompt should move length without touching the stop decision. If r_prompt is orthogonal to both, construct (a) is picking up "instruction present" and the whole line is null at this locus.

## 3. Feasibility of the weight edit vs capability damage

Concerns, in order of likelihood:

1. **"Dumber, not shorter."** The writer weights for the long pole are likely the weights that implement re-verification and backtracking (we already have a backtracking BLADE mask; check its overlap with the length mask — if the top-ρ sets overlap heavily, we are removing checking, not verbosity). Removing checking shortens *and* drops accuracy on level-4/5 MATH and AIME, while looking fine on GSM8K/MATH500 where checking is mostly redundant. ASC's benchmarks would hide this; ours must not.
2. **Degenerate termination.** α=0 could produce immediate `</think>` (no reasoning → accuracy collapse) rather than *tighter* reasoning; α>1 could produce runaway traces that never terminate (truncation at max_new). Report the truncation rate and the fraction of traces with <50 think tokens as first-class numbers.
3. **Style vs substance.** A shortened trace could just drop English (ASC's effect) — benign — or drop steps — risky. Count steps (newline/`Wait`/`So` segments) as well as tokens.
4. **Answer-migration.** Thinking shrinks but the answer balloons with the same content. Report total tokens.
5. **Regime specificity** (as with uncertainty): the edit calibrated on MATH may only shorten math. Test GPQA / a code task.

Controls that separate length control from damage:
- Random mask at matched ρ **and matched Δppl** (we know BLADE-G masks can hide behind low ppl).
- **Budget-forcing curve as the null frontier**: take the *base* model, truncate its trace at the edited model's median length, force the answer (s1-style). If the edit's accuracy at length L equals truncate-at-L accuracy, the edit is no better than a token cap. This is the single most important baseline and it is free.
- Accuracy at matched difficulty (MATH level bins; or bin by base-model think length).
- Retention/gain: on problems the base got right, does α=0 keep them (retention)? On problems the base got wrong, does α>1 fix them (gain)? Bidirectional monotonicity in *both* length and these two rates is the result.
- WikiText ppl (necessary, not sufficient) plus a no-think MMLU/GSM8K-no-think sanity check that non-reasoning capability is unchanged.

## 4. Metrics

- Think tokens: **median and distribution** (heavy-tailed; means are dominated by runaways), per α.
- Total tokens; tokens-to-first-`\boxed`; truncation rate at max_new (use 4096+, cf. the P0 calibration fix); step count.
- Accuracy: MATH500 (by level), GSM8K (ceiling check), **AIME24/25 and GPQA-diamond** (where shortening should hurt if it is going to). Use the existing verifier in `scripts/eval_calibration.py` (`last_boxed`, `norm_math`, sympy equality; `qwen_wrap(..., think=True)` already reports lengths).
- The frontier plot: accuracy vs median think tokens, curves for α ∈ {0, 0.5, 1, 1.5, 2, 3}, overlaid with budget-forcing, brevity prompting, ASC γ-sweep (reimplement — 50 pairs, one layer, trivial), random mask. Summary number: area between our curve and the budget-forcing curve.
- Bidirectionality: Spearman(α, median tokens) and monotone accuracy retention.

## 5. Relation to 2507.04742

Same family: difference-of-means, residual stream, mid layer, one direction, training-free. Three real differences:

| | ASC | Ours (proposed) |
|---|---|---|
| Direction locus | end-of-CoT token, generated contrast, verbose-vs-terse *style* | last prompt token, instruction contrast, deliberation *mode* |
| Intervention | activation addition every step, KL-bounded γ | sparse permanent weight edit, top-ρ of `down_proj`/`o_proj`, α knob; ELS picks layers |
| Direction of control | shorten only | shorten (α=0) **and** lengthen (α>1) with one edit |

Is that a real delta? Honestly: **moderate, and conditional.**
- The primitives are all anticipated (ASC shorten-by-steering; ThinkEdit lengthen-by-o_proj-edit; CoT-Valve trained weight knob). A pure "we can also shorten CoT" result is a weaker ASC unless we match ~50% compression at flat MATH500 accuracy, which a prompt-locus direction probably will not.
- The delta is real if at least one of these lands: (i) **bidirectional monotone control from one sparse edit** (nobody has shown both signs with one mechanism, training-free); (ii) the **frontier claim** — the edit beats budget forcing / ASC at matched length on *hard* sets, i.e. it removes waste rather than reasoning; (iii) the **locus finding** — a mode component of length is prompt-decodable and weight-editable, separable from the termination component. (iii) is the scientifically interesting one and fits the paper's thesis ("one recipe, many behaviors, and here is where it stops working").
- Sparse weight edit is a deployment advantage over ASC (ship a checkpoint, no hooks), and a mechanistic advantage (which weights) — but ThinkEdit already has "which heads" for the lengthening side.

## 6. Verdict

**Promising-but-narrow extension; the plain length-control result is largely already done; the pre-generation locus is the crux and I put ~35% on it working well enough (≥30% median think-token reduction at ≤2 pt MATH500 and ≤5 pt AIME drop) directly from the last-prompt-token recipe, ~60% if Δμ_j is re-estimated at thinking positions (a small change to the recipe that keeps r and the scoring intact).** The expected honest outcome is a "deliberation-mode" knob with modest compression and a clean lengthening effect, plus a locus result — not an ASC-beating compressor.

Do not frame it as "length control". Frame it as "does the unified knob reach a whole-trajectory behavior, and at which locus".

### The single most decisive first experiment

**A locus-resolved steering screen on Qwen3-8B (thinking mode), 200 MATH500 + 100 AIME problems, before any weight edit.**

0. Prerequisite (1 hour): measure median think tokens under "reason in as few tokens as possible" vs "reason very thoroughly, verify every step" instructions. If the ratio is <2×, construct (a) is dead at the prompt locus; go straight to a generated-token contrast (repo already has this machinery in `blade_reasoning.py` / `reasoning_els.py`).
1. Build r_ℓ from construct (a) at the last prompt token, all layers, ELS selection (`reasoning_els.py` pattern). Also compute r_think and r_term (§2) for the cosine table.
2. Apply ±k·v_raw (per memory: not σ·v̂ at deep layers) in **three application modes**: (i) prompt positions only, (ii) generated positions only, (iii) all positions (ASC-style). Measure median think tokens, accuracy, truncation rate, <50-token rate.
3. Overlay the budget-forcing curve of the base model.

Readout:
- (i) ≈ (iii) and length moves ≥30% with accuracy on the budget-forcing curve or above → length has a prompt-time mode component that a fixed-position direction captures; proceed to BLADE-G weight edit with α sweep, random-mask and ThinkEdit/ASC comparisons.
- Only (ii)/(iii) work → the feature is mid-generation; re-estimate Δμ_j at thinking positions before scoring (recipe change is minimal; the finding is itself reportable).
- Nothing moves although the instruction does → the instruction effect is not a single linear direction at any of these loci; write the null and stop.
- Length moves but accuracy falls **below** the budget-forcing curve → we are removing reasoning, not verbosity; stop.

Cost: one GPU-day. It answers points 1–2, previews 3, and decides whether the weight edit deserves the ρ/α/ELS sweep.

### Existing repo assets to reuse
- `scripts/eval_calibration.py`: `qwen_wrap(tok, instr, think=True)`, `last_boxed`/`norm_math`/sympy verifier, accuracy + lengths (MAX_NEW already raised to 4096 for thinking traces).
- `scripts/gen_reasoning_traces.py`, `scripts/blade_reasoning.py`: `parse_think`, generated-token contrast directions (`reasoning_dirs.pt`, muC/muG), BLADE-remove of backtracking/uncertainty on DS-R1-Distill-1.5B — use the backtracking mask for the overlap check in §3.
- `scripts/reasoning_els.py`, `scripts/steer_epistemic_screen.py`: ELS and the ±c steering screen.
