# Plan (v3, for codex review) — editing backtracking/error-recovery with existing benchmarks
## DeltaBench (direction) + ProcessBench (held-out eval), Qwen3-14B, BLADE-G

Supersedes the hand-injection plan (`plan-backtrack-pair-harness.md`). Uses existing, human-annotated,
NATURAL-error benchmarks (codex survey `scratch_bench_survey_codex.md`) to avoid splice/injection
artifacts and to get gold process labels, and separates direction-building data from evaluation data.

## 0. Construct & honest scope
We study **recovery from an erroneous reasoning step** ("does the model retract+repair a wrong step").
We do NOT claim to edit "the backtracking faculty" or "intrinsic self-correction" (the errors are not
the target model's own). A positive result = a BLADE weight edit that changes *repair behavior* on
held-out, human-located natural errors, beyond correctness / length / random / reversed controls.

## 1. Direction from DeltaBench (natural, matched, human-corrected)
DeltaBench (ACL 2025, 1236 long-CoT traces; QwQ/R1/Gemini-2.0-Flash-Thinking; humans annotate first
error + a written correction + reflection usefulness). For each annotated erroneous section with a
human correction, holding the problem + all pre-error content FIXED, form two contexts:
- **error**: prefix + the annotated erroneous step;
- **corrected**: prefix + the human correction of that step.
Estimator: at the token immediately after the step delimiter, record Qwen3-14B residual states;
compute **within-pair** `corrected − error`, normalize per pair, average by layer → direction r_ℓ; and
the BLADE writer moment shift Δμ_W = μ_W(corrected) − μ_W(error) over the same fixed boundary window.
Pairwise subtraction removes problem/topic/generator/prefix content. Split by underlying PROBLEM and
SOURCE MODEL (not by trace) to prevent near-duplicate leakage; choose layer on a DeltaBench dev split.
BLADE score s=[r·W·Δμ]_+ → ELS (BLADE-B layers) → BLADE-G mask. Predictions: remove (α=0) → lower
repair rate; amplify (α>1) → higher repair rate (respect scale penalty λ|α−1|; do NOT reuse the
removal mask for large amplify).

## 2. Held-out eval on ProcessBench (natural errors, expert first-error labels)
ProcessBench (ACL 2025, 3400 traces: GSM8K 400 / MATH 1000 / OlympiadBench 1000 / Omni-MATH 1000;
2221 erroneous + 1179 correct; human earliest-error index e). For each erroneous trace, **keep steps
s₀…s_e (INCLUDING the error step e), drop s_{e+1..n}** → corrupted prefix. Four conditions:

| Prefix | Cue | Measures |
|---|---|---|
| error through s_e | "Continue solving." | intrinsic detection+repair |
| error through s_e | "Step e is incorrect; reconsider from there." | repair GIVEN gold localization (Tyen decomposition) |
| corrected/sham prefix (error step → its correction) | "Continue solving." | clean utility + false backtracking |
| naturally-correct ProcessBench trace | "It may be correct or incorrect; verify and continue." | overcorrection under non-leading verify |

Report per condition and per subset (GSM8K/MATH/Olympiad/Omni): final-answer recovery (executor);
wrong→correct & correct→wrong; whether the model explicitly retracts the erroneous proposition;
tokens-to-first-valid-correction; total reasoning tokens; **unnecessary-backtracking rate on clean
prefixes**; net utility = fixed − broken. base vs edited (remove, amplify α-sweep) vs controls, same
decoding seed.

## 3. Controls (carry over from codex reviews)
1. **Correctness confound**: the corrected member is, by construction, "more correct." Controls:
   (a) same-prefix subtraction already removes most content/outcome; (b) a separate **correctness
   direction** from correct-vs-wrong FINAL answers on clean traces — report cosine AND edit
   dissociation (backtracking edit moves repair-conditional-on-opportunity; correctness edit moves
   outcome more); (c) where DeltaBench allows, pairs where the corrected step still fails downstream
   (isolates the repair ACT from success).
2. **Style/role confound** (codex): corrected−error may encode "human-correction style"/role, not
   recovery → random-direction, **reversed-direction**, and layer controls; and **transfer**: build on
   DeltaBench, test the SAME direction separates MR-Ben (NeurIPS 2024) human-corrected vs error steps
   WITHOUT refitting, and drives ProcessBench recovery.
3. **Detection vs repair**: corrected/sham vs error prefix (condition 3) isolates detection; the
   oracle-cue condition isolates repair-given-location.
4. **Length/position**: fixed delimiter-token extraction; match/adjust continuation length; equal
   weight per pair; don't token-weight Δμ while example-weighting r.
5. **Lexical**: eval on lexical-template-disjoint splits + marker-only negatives ("wait" that doesn't
   revoke; implicit repair w/o marker); probe decodability ≠ causal use (Hewitt & Liang EMNLP'19).
6. **Damage controls (BLADE-G edit only)**: random fixed-sparsity, **fixed-damage** (tuned to match
   ΔNLL incl. math-domain), shuffled-r; ppl alone can't see collapse (our α=4 result).
7. **Stats**: unit = problem (prefix-clustered bootstrap); pilot correction-propensity, stratify, and
   evaluate on the UNFILTERED ProcessBench test; report by error difficulty/subset.
8. **Contamination**: GSM8K/MATH may be in pretraining; treat ProcessBench as a controlled stimulus
   set and report harder Olympiad/Omni subsets separately.

## 4. Minimum evidence before any mechanism / single-direction claim
Nested rank sweep (1/2/4/8; magnitude+layer on val, one locked test); stage-specific causal patching
(Meng NeurIPS'22 causal tracing; preregister corruption+metric, Zhang&Nanda ATTRIB'23); convergent
activation-add ↔ weight-edit effects; selective (vs random/neighbor) restoration; necessity+sufficiency
patching. If rank>1 wins → "recovery is a subspace/staged computation," a valid conclusion, not failure.

## 5. Novelty baseline (must position against)
**Self-Correction Bench (COLM 2026)** already shows a minimal "Wait" marker activates correction and
derives a transferable conversational-role direction on Qwen-family models — the closest prior
mechanistic work. We must (a) cite it, (b) compare our DeltaBench corrected−error direction against its
role/"Wait" direction (cosine + head-to-head editing), and state our delta (weight-level BLADE edit +
generic-importance penalty + ProcessBench recovery eval vs their activation/role analysis).

## 6. Phasing (Qwen3-14B)
- **P0a** build DeltaBench direction; linear-decodability sanity (does r separate held-out
  corrected/error at the delimiter token, on problem-disjoint split, beyond a bag-of-words baseline?).
- **P0b** ELS + BLADE-G remove/amplify(α-sweep); ProcessBench 4-condition eval + controls §3;
  transfer to MR-Ben. Head-to-head vs Self-Correction-Bench direction.
- **P1** mechanism gates §4 (rank sweep, causal patching) only if P0 shows a real, controlled effect.
- Honest negative allowed: if remove/amplify don't move repair beyond controls, report "recovery not
  single-direction BLADE-editable on Qwen3-14B" (consistent with distributed computation).

## 7. Infra
DeltaBench + ProcessBench + MR-Ben prefetch to offline files on a Hazel login node. Reuse
collect_span_input_moments / capture_span_mean (fixed delimiter-token span), reasoning_els (BLADE-B
layers), reasoning_mask (BLADE-G, λ|α−1|), an executor for MATH answers (fixed verifier). Eval sharded
on h100 (infinite time). Human labels come WITH the benchmarks (no new annotation needed for P0).

## 8. Wording bounds
Claim only: "a DeltaBench corrected−error direction yields BLADE weights whose edit changes
error-recovery behavior on held-out human-located ProcessBench errors, beyond correctness/style/random/
reversed controls, on Qwen3-14B." Reserve "backtracking mechanism"/"single direction" for results that
pass the §4 causal + rank tests; reserve "intrinsic self-correction" for natural self-generated errors.
