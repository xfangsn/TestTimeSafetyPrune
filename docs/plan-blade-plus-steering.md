# Plan — BLADE + activation steering: complementary or redundant?

## Question
After BLADE edits the weights (removes sycophancy by zeroing/rescaling the selected
modulatory weights), does **activation steering still work** on top — and is
**BLADE + steering better than steering alone** (lower sycophancy at equal-or-lower
capability cost)?

This tests whether the two interventions act on the **same** mechanism (→ redundant:
steering does little on the BLADE-edited model) or **different** ones (→ complementary:
BLADE removes a modulatory *gain* in the weights, steering adds an *activation* offset,
so stacking pushes sycophancy lower on a better sycophancy–capability Pareto front).

## Model / behavior
Llama-3.2-3B-Instruct, sycophancy. Metric: A/B pick-rate on the held-out `val` split
(chance 0.5). Capability: WikiText ppl (held-out) + 6 zero-shot tasks at the operating
point. Optional: OOD TriviaQA sycophancy.

## Interventions
- **BLADE(α)**: our weight edit. Use the β=5% L\* (C4-calibrated run) at ρ=0.005, α=0
  (zeroing). One fixed edited model `M_BLADE`. (Also record BLADE-only pick-rate.)
- **Steer(c, ℓ_s)**: CAA-style activation steering. Steering vector
  `v_{ℓ_s} = mean act(sycophantic) − mean act(non-sycophantic)` at a mid layer ℓ_s,
  applied by a forward hook that adds `−c · v̂` (unit v) to the residual output of
  layer ℓ_s at every position, during scoring/generation. `c>0` reduces sycophancy.

**Steering-vector provenance (2 variants, important):**
- (V1) recompute `v` on the model it is applied to (base-vector for base, edited-vector
  for `M_BLADE`) — the honest "steer *this* model" setup;
- (V2) use the *base-model* `v` on both — tests whether the base steering direction still
  transfers after BLADE. (V1 primary, V2 ablation.)

## Conditions (2×2 core + sweeps)
| condition | model | steering |
|---|---|---|
| Baseline | unedited | none |
| Steer-only | unedited | Steer(c) sweep |
| BLADE-only | M_BLADE | none |
| BLADE+Steer | M_BLADE | Steer(c) sweep |

Sweeps: coefficient `c ∈ {0, 0.5, 1, 2, 4, 8}` (subtract). Steering layer ℓ_s: fix to a
mid layer (~half depth, e.g. L14) for the main run; secondary sweep ℓ_s ∈ {8,12,14,18}
to pick the best steering layer once.

## What we measure per (condition, c)
sycophancy A/B pick-rate; WikiText Δppl. At the chosen operating points also: OOD
sycophancy, 6-task downstream. (All A/B pick-rate = cheap logprob; ppl cheap; downstream
only at a few points.)

## Analysis / decision criteria
1. **Does steering still work after BLADE?** slope of pick-rate vs c on `M_BLADE`.
   If ≈0 → BLADE already captured the direction (redundant). If clearly negative →
   steering remains effective on the edited model.
2. **Pareto front**: plot sycophancy (y) vs WikiText Δppl (x) for **Steer-only** and
   **BLADE+Steer** (each a curve over c), plus the BLADE-only and Baseline points.
   - **Complementary / better** if the BLADE+Steer curve lies **below-left** of
     Steer-only (lower sycophancy at the same Δppl), and reaches a lower sycophancy
     floor than either alone.
   - **Redundant** if the two curves overlap, or BLADE+Steer = BLADE-only (steering adds
     nothing).
3. **Best achievable** sycophancy within a capability budget (Δppl ≤ 5%, downstream
   drop ≤ ~1pp): compare the minimum sycophancy reachable by Steer-only vs BLADE+Steer.

## Hypotheses (to be falsified)
- BLADE weights are *modulatory* (a gain); steering is an additive activation shift →
  plausibly **complementary**: BLADE lowers the baseline sycophancy cheaply (≈0 ppl),
  steering then trims further, so BLADE+Steer reaches a lower floor at lower ppl than
  heavy steering alone.
- Alternative: both ride the same difference-in-means direction → **redundant** after
  BLADE (steering slope flattens on `M_BLADE`). Either result is informative and
  publishable (mechanism independence vs. shared direction).

## Implementation notes
- Reuse `extract_direction` / the sycophancy A/B contrast for `v` and BLADE's L\*/scores.
- New: a `steering_hook(layer, vec, coef)` context manager (forward hook adding `−coef·v̂`
  to the block output); compose with `pruned_weights(sel)` for the BLADE+Steer condition.
- Deterministic; A/B scoring already teacher-forced. ~1 GPU session, mostly cheap
  logprob passes + a handful of ppl/downstream evals. Runs AFTER the current C4 chain
  (GPU serialization).

## Deliverables
- `results/blade_plus_steering_sycophancy.json` (grid).
- Figure: sycophancy–ppl Pareto (Steer-only vs BLADE+Steer curves, Baseline/BLADE-only
  points), house style.
- One-paragraph verdict: complementary vs redundant, and whether BLADE+Steer beats
  steering-alone.
