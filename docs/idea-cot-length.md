# Idea to assess — can our BLADE / scheme-A design control REASONING-CHAIN LENGTH?

## Our method (established on Qwen3, uncertainty behavior)
- **Direction (refusal/CAA-style)**: contrast two PROMPT SETS at a FIXED pre-generation position (last
  prompt token), difference-of-means → a residual-stream direction r_ℓ per layer. For uncertainty:
  certain (model knows) vs uncertain (unanswerable) prompts.
- **BLADE weight scoring**: s_ij = [ r_i · W_ij · Δμ_j ]_+ over residual-writer weights (down_proj, o_proj),
  where Δμ_j = mean writer-input shift between the two sets at the same fixed position; BLADE-G subtracts a
  generic-importance penalty. ELS auto-selects the effective layers.
- **Bidirectional edit via one α knob**: α=0 zeroes the top-ρ weights (REMOVE the behavior); α>1 scales
  them (AMPLIFY). Also an activation-steering screen (±c·v̂) as a diagnostic.
- **Results (Qwen3-8B, uncertainty)**: REMOVE → model stops abstaining on unanswerable Qs (hallucinates);
  AMPLIFY → more appropriate abstention; monotone, capability-preserving (Δppl ≤ ~1.6% up to α=4);
  transfers OOD (SelfAware/FalseQA), beats ITI/DoLa on the hallucination-vs-capability tradeoff.
- Construct caveat we already found: the edit is REGIME-SPECIFIC (closed-book parametric knowledge), and
  "direction is decodable everywhere" ≠ "steerable/weight-editable everywhere" (layer choice matters;
  dose parameterization matters — use k·v_raw not σ·v̂ at deep layers).

## The question
Can the SAME design control the LENGTH of the reasoning chain (CoT) — i.e., make a thinking model
produce systematically SHORTER or LONGER reasoning before its answer, on demand, via a sparse weight edit
(and/or the activation-steering screen)? Reference (a paper that reportedly does length control):
https://arxiv.org/abs/2507.04742 — read it and relate their mechanism to ours (do they use a steering
vector / a learned length direction / RL / prompting? what is our delta or overlap?).

## What to assess (be concrete)
1. **Construct/contrast design**: what two prompt SETS (or paired generations) give a clean "long vs short
   reasoning" direction at a fixed position? Options: (a) same problems with "think briefly" vs "think
   step by step at length" instructions (expressed, prompt-set, our refusal-style); (b) matched
   naturally-long vs naturally-short correct CoTs on the same problems (generated-token contrast);
   (c) difficulty-graded problems. Which is least confounded (difficulty/correctness/topic/verbosity)?
   Note length is a CONTINUOUS, autoregressive, whole-trajectory property, unlike our fixed-position
   binary uncertainty decision — is a last-prompt-token direction even the right locus, or is this a
   "when to emit </think>/stop" decision better captured mid-generation?
2. **Where the behavior lives**: is CoT length gated at the prompt (a pre-commitment the model sets before
   thinking) or continuously during generation (a termination/So-let-me-stop signal)? Our direction is
   pre-generation; if length is decided during generation, our fixed-position recipe may miss it (cf. our
   uncertainty crossover finding that regime/locus matters).
3. **Feasibility of the weight edit**: would REMOVE/AMPLIFY of length-writer weights shorten/lengthen CoT,
   or just degrade fluency? Confounds: length is entangled with task difficulty and answer correctness —
   shortening could just make it dumber. What controls separate "length control" from "capability
   damage"? (accuracy at matched difficulty; the </think> position; tokens-to-answer.)
4. **Metrics**: thinking-token count, tokens-to-first-answer, accuracy (must be preserved / measured as a
   function of induced length — the whole point is length↓ WITHOUT accuracy↓, the efficiency frontier).
5. **Relation to 2507.04742**: same or different mechanism? Is our sparse-weight, bidirectional,
   capability-preserving angle a real contribution over theirs, or subsumed?
6. **Verdict**: is this a promising extension of our method, a likely-null (length isn't a
   fixed-position linear direction), or already-done by the paper? Give the single most decisive first
   experiment.
