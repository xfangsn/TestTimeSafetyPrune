## Bottom line

The plan is executable after several corrections, but §9 currently overclaims what reader normalization buys. Pulling through RMSNorm removes a major scale mismatch; it does **not** make layer-local scores fully comparable at the model output. Treat E6 as testing a hierarchy:

1. raw global score;
2. reader-normalized global score;
3. transport-calibrated global score;
4. ELS.

Also narrow the main hypothesis: `Q` is a collateral-damage proxy. There is no strong reason it should improve OOD/in-distribution transfer; that should remain exploratory.

## A. Execution-plan review

### 1. The RMSNorm diagonal and reader mapping

For Hugging Face Llama’s RMSNorm,

\[
N(h)=\gamma\odot h/r,\quad
J=\frac{D_\gamma}{r}(I-qq^\top),\quad
q=\frac{h}{\sqrt m\,r},
\]

your diagonal is correct:

\[
\|Je_i\|^2 =
\frac{\gamma_i^2(1-2q_i^2)+q_i^2\sum_k\gamma_k^2q_k^2}{r^2}.
\]

With `ε=0, γ=1`, it reduces to `(1-q_i²)/r²`, and its sum is `(m−1)/r²`.

The mapping is also correct for standard Llama decoder ordering:

- `o_proj` output → residual addition → `post_attention_layernorm`.
- `down_proj` output → residual addition → next layer’s `input_layernorm`.
- Last `down_proj` → final `model.norm`.

But call this the **exact diagonal of the first-order RMSNorm metric**, not the exact deletion cost. It ignores:

- finite-step curvature of RMSNorm;
- cross terms between jointly deleted weights;
- subsequent-layer transport.

Implementation details currently missing:

- Exclude padding and document-packing artifacts from token moments.
- Disable KV cache and remove hooks in `finally`.
- Accumulate in fp32 and keep the final `S` in fp32; fp16 around the `S=0` boundary will make abstention unstable.
- Explicitly assert `Linear.weight.shape == (out, in)`.
- Specify hook buffering when several writers share reader modules.
- Make abstention a filter in ranking, not merely negative scores at the end.
- If abstention is enforced, “fraction of selected weights with `S≤0`” is identically zero. Report candidate coverage or the fraction of the **BLADE-selected** set rejected by BLADE-G instead.
- “New code limited to two places” conflicts with adding `score_fn` plumbing, scripts, tests, and result-schema changes.

### 2. Exact `AᵀX²` versus separable moments

`Aᵀ @ X²` is mathematically the correct diagonal estimator:

\[
Q_{ij}=W_{ij}^2 E[a_i x_j^2].
\]

But it is much more expensive than the plan suggests. Across every `o_proj` and `down_proj`, its arithmetic is approximately `n_tokens × total_writer_parameters`; with 262k tokens on a 3B model, that is hundreds of trillions of operations. The 100 MB accumulator is not the main problem—the repeated enormous GEMMs and GPU-to-CPU handling are.

The covariance can matter because the dominant part of `a_i` contains the tokenwise scalar `1/r_t²`, which can correlate with attention/MLP activation magnitude. Conversely, the coordinate-specific `q_i²` correction is generally `O(1/m)`.

Add an intermediate estimator:

\[
Q^{scalar}_{ij}
= W_{ij}^2\gamma_i^2 E[x_j^2/r_t^2].
\]

It captures the important tokenwise norm covariance at Wanda-like cost while dropping the small coordinate correction. This is likely a better primary competitor than `E[a_i]E[x_j²]`.

Before full experiments, use 8k–16k tokens on several layers to measure:

- convergence of top-k overlap;
- rank correlation among g0, g1mean, g1scalar, and exact g1;
- covariance magnitude;
- wall time and memory.

Do not commit three seeds × 262k tokens to exact g1 before that pilot.

### 3. Lambda calibration

The Lagrangian is sound, but the present calibration description mixes roles: C4 can provide `Q` and utility loss, but not the behavior objective.

Use three disjoint resources:

- `C4-Q`: estimate `Q`;
- behavior-dev + `C4-dev`: select `λ` and weight count;
- `C4-budget`: untouched feasibility check.

Tune using `ΔNLL` or log-ppl, not raw `Δppl`, because NLL is the additive quantity. Expand the dimensionless λ grid beyond 1; median matching says little about the selected tail.

A predicted-collateral constraint is a useful secondary parameterization:

\[
\max_A\sum_{w\in A}c_w
\quad\text{s.t.}\quad
\sum_{w\in A}Q_w\le B,\ |A|\le K.
\]

But `ΣQ` is only a diagonal local proxy, so `B` still needs empirical calibration to NLL. It does not eliminate validation. Report both the Lagrangian frontier and `ΣQ` for interpretability.

### 4. Two-GPU-day scope

Run the estimator pilot, E1, and a corrected E6. Cut, in order:

1. E5;
2. E2 amplification;
3. most of E4;
4. E3 sycophancy if necessary.

For two GPU-days I would retain BLADE, g0, g1scalar, one reduced-token exact-g1 check, Q-matched random, and ELS/global comparisons. The full baseline zoo and three-seed exact g1 are less valuable than establishing whether E6 works.

### 5. Leakage and matched comparisons

Interpolation between two realized pruning interventions is not itself a realized model. It is acceptable for plotting a Pareto curve, but not for a claim such as “at equal behavior removal” unless interpreted as randomized mixing.

Prefer:

- a prespecified behavior target;
- search the discrete prefix to attain it and actually evaluate that model;
- report full budget-matched and behavior-matched frontiers;
- use interpolation only as a sensitivity analysis.

Matching on in-distribution behavior can favor a method tuned to that distribution. Keep OOD fully untouched and report raw OOD changes alongside the transfer ratio; the ratio becomes unstable when the in-distribution denominator is small.

Three seeds do not support a convincing seed-level bootstrap. Use paired example-level or hierarchical bootstrap, show all seed points, and avoid claiming seed-population significance from `n=3`.

If XSTest participates in the keep/drop rule, it is no longer a clean test set. Split off a benign-development set or label the analysis exploratory.

## B. Can the score replace ELS?

### 1. What reader normalization does—and does not—solve

Raw BLADE scores are not reliably cross-layer comparable. Reader pullback removes residual-scale growth and places each contribution in the local norm-output geometry. That is necessary, but not sufficient.

Still incomparable are:

- downstream gain or attenuation from that reader to final logits;
- differing semantic quality and normalization of independently extracted `r_ℓ`;
- attention/MLP position within a layer;
- behavior-margin sensitivity at different depths;
- generic-loss curvature at different depths;
- cancellation, amplification, and self-repair downstream.

More importantly, §6/§9’s numerator needs correction. If `u_t=J_t^\top r`, the behavioral contrast should be something like

\[
c^{rdr}_{ij}
= W_{ij}\left(
E_{\mathrm{behavior+}}[u_{t,i}x_{t,j}]
-
E_{\mathrm{behavior-}}[u_{t,i}x_{t,j}]
\right),
\]

not an unspecified `E_t[u_i x_j]`. The direction `r` must be defined at that particular reader output. Existing block-output directions cannot simply be called post-norm covectors. For `o_proj` especially, its immediate reader occurs before the MLP, whereas the existing block-output direction occurs after it. Re-extract directions at each reader site.

Thus “reader units” means comparable local scalar projections—not comparable final behavioral effects.

### 2. Existing global ranking makes E6 riskier

Yes: only the candidate layer set changes. But admitting all layers creates:

- extreme-value competition across many more weights;
- more opportunities for noisy score tails to win;
- different total edit counts if `ρ` is interpreted relative to all eligible weights.

Do not compare the same `ρ` over `L*` and all layers: that edits different absolute numbers of weights. Compare both fixed absolute `K` and behavior-matched prefixes. Keep the per-matrix cap, but recognize that it does not normalize score distributions or prevent early layers from consuming substantial budget.

### 3. The proposed κ calibration

A single probe at `ρ=10⁻⁴` using behavior **rate** is likely too noisy: most layers may produce no threshold crossings, yielding zero or unstable `κ_beh`.

Use a continuous transport target instead:

- final-layer refusal-direction projection;
- refusal-versus-compliance teacher-forced log-likelihood margin;
- continuous judge/classifier logit, if already validated.

Estimate

\[
\kappa_{\ell,c}^{beh}
=
\frac{\Delta\text{continuous margin}}
     {\sum c^{rdr}},
\qquad
\kappa_{\ell,c}^{util}
=
\frac{\Delta\mathrm{NLL}}
     {\sum Q},
\]

separately for `o_proj` and `down_proj`, over two small probe magnitudes, with shrinkage toward a smooth depthwise curve. A single κ per layer incorrectly combines two different reader sites.

This is layer calibration, not ELS-free operation. It is still materially cheaper than greedy joint selection, but variant D should be named **one-shot calibrated global ranking**. Variant C is the genuinely measurement-free test.

A cleaner forward-only alternative is small activation injection at each reader along its behavior direction, measuring the continuous final margin. This estimates transport without confounding κ with the top-k mask quality.

### 4. Q probably will not automatically eliminate L0

This prediction is weak. Immediate local norm energy is not the same as downstream capability criticality. RMSNorm may actually equalize early-layer local energy, while an early perturbation has many layers in which to propagate. L0 can still win if:

- its `c/Q` tail is large;
- it has many candidates and wins by extreme values;
- downstream amplification is large but absent from `Q`;
- cross-weight interference makes summed `Q` underestimate damage;
- the matrix cap permits a damaging allocation.

Test the prediction, but do not motivate removal of `solo_layer_pool` as though it follows from the definition.

### 5. Detecting threshold failure versus transport failure

For every E6 model record, per example:

1. predicted local shift `Σc`;
2. actual reader projection changes immediately after edited writers;
3. transported continuous final margin change;
4. generated binary behavior.

This distinguishes:

- predicted shift not realized → score approximation/cross-term failure;
- local shift realized but final margin recovers → downstream cancellation or hydra self-repair;
- final margin shifts but behavior rate does not → threshold/saturation or insufficient margin;
- individual-layer effects work but joint effect does not → cross-layer interaction.

Plot the distribution of baseline margins and margin changes, not just their means. A layerwise trace of the edited-minus-dense residual projection will reveal where recovery begins.

### 6. Beta verification and honest cost accounting

Keep fixed-ranking β backoff. Bisection over prefix length is a one-dimensional feasibility search, but it is not ELS-style layer search. Say exactly that. Do not change both λ and prefix during backoff, or it becomes a two-dimensional retuning loop.

Report:

- dense tokens used for direction/moment and `Q` construction;
- number of edited-model behavior evaluations;
- number of edited-model C4/NLL evaluations;
- κ probe evaluations;
- β-backoff evaluations;
- final reporting evaluations, separately excluded from selection cost;
- GPU-hours and evaluated tokens.

For ELS, report the actual number of prune contexts evaluated by solo-pool and best-first, not merely `O(L²)`. The defensible conclusion is not “a score makes ELS unnecessary” unless C succeeds; if only D succeeds, the conclusion is “combinatorial ELS can be replaced by linear-cost transport calibration plus global ranking.”