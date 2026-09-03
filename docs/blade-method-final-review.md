# Final review — the "difference-in-means direction" attribution

**Sentence under review** (in `docs/blade-method.tex`, Behavior representation):

> This is the difference-in-means contrastive direction used for refusal by
> \citet{arditi2024refusal} and, more broadly, in representation engineering and
> contrastive activation steering \citep{zou2023repe,rimsky2024caa,marks2024geometry}.

**Question:** is this direction specifically Arditi's, or equally the others'? What does each
actually do?

## What BLADE computes
$r_\ell = \operatorname{mean}_{x\in A} h_\ell(x) - \operatorname{mean}_{x\in B} h_\ell(x)$,
normalized — a **plain difference of class means** (diff-in-means) of residual-stream
activations, per layer.

## What each cited work computes as its "direction"

| Work | Direction estimator | = BLADE's diff-in-means? |
|---|---|---|
| **Arditi et al. 2024** (refusal, 2406.11717) | diff-in-means: `mean act(harmful) − mean act(harmless)` at a selected layer/token, then directional ablation / addition | **Yes — identical.** BLADE's *refusal* $r_\ell$ (harmful−harmless mean diff) is exactly this. |
| **Rimsky et al. 2024** (CAA, 2312.06681) | steering vector = **average of `act(positive) − act(negative)`** over contrastive A/B pairs at a layer; added at inference | **Yes — diff-in-means**, and over the *same* A/B multiple-choice contrast BLADE uses for its A/B behaviors. |
| **Marks & Tegmark 2024** (Geometry of Truth, 2310.06824) | **"mass-mean" probe** = `mean(true) − mean(false)` activations | **Yes — diff-in-means**; the foundational reference for this estimator as a probe direction. |
| **Zou et al. 2023** (RepE, 2310.01405) | LAT "reading vector" = **first PCA component of normalized paired-difference vectors** | **No — PCA of differences, not a raw mean difference.** RepE is the broad read/control *framework*, not this specific estimator. |

(Estimators verified via the papers' method descriptions, Aug 2026.)

## Verdict
1. **Arditi is the correct *primary* citation for the refusal instantiation** — BLADE's refusal
   direction is the harmful−harmless diff-in-means, which is exactly Arditi's refusal direction.
   So "used for refusal by Arditi" is accurate and appropriately specific.
2. **But the diff-in-means *estimator* is not uniquely Arditi's.** For BLADE's A/B behaviors, the
   closest precedent is **CAA** (averaged activation difference over the same A/B contrast), and
   the estimator itself is canonically **Marks & Tegmark's mass-mean** direction. These two should
   be credited *on equal footing with* Arditi for the estimator, not merely "more broadly".
3. **RepE is mis-grouped.** Its signature direction (LAT reading vector) is **PCA of paired
   differences**, not a mean difference. It belongs as the broader representation
   reading/steering *framework*, with a note that its estimator differs.

So: not "Arditi's method that others also use", but rather **a diff-in-means direction that Arditi
applied to refusal, CAA applied to A/B behaviors, and Marks & Tegmark established as a probe** —
while RepE is the umbrella agenda using a *different* (PCA) estimator.

## Recommended rewrite (precise)
> BLADE's $r_\ell$ is the difference-in-means (a.k.a. mass-mean) contrastive direction: the mean
> activation on side $A$ minus side $B$~\citep{marks2024geometry}. For refusal it is exactly the
> harmful$-$harmless refusal direction of \citet{arditi2024refusal}; for the A/B behaviors it is
> the contrastive steering-vector construction of \citet{rimsky2024caa} (averaged activation
> difference over the same multiple-choice contrast). It sits within the broader representation
> reading/steering framework of \citet{zou2023repe}, whose reading vector instead uses PCA of
> paired differences.

## Also flagged
- **Bib key/year:** the entry is keyed `marks2023geometry` but the venue is COLM **2024** (the
  quoted line already uses `marks2024geometry`). Rename the key to `marks2024geometry` for
  consistency and update the citation.
