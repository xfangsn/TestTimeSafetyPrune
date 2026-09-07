# Kimi Code review — docs/fig_uncertainty_method_cmp_both.tex

Reviewer: Kimi Code (agent mode; verified every cited number against
`scripts/plot_method_comparison.py` (Qwen3-8B) and `scripts/plot_llama_method_cmp.py` (Llama-3.2-3B)).
LaTeX drafted by Fable 5.1 (`--model claude-fable-5-1`). Both figures + interpretive body cover
`uncertainty_method_cmp.pdf` (Qwen) and `uncertainty_method_cmp_llama.pdf` (Llama).

## Findings

1. **[Important] "strict/clean Pareto improvement" is technically false.** On Qwen, BLADE *worsens* both
   matched preserve-behaviors vs base (answerable answering 87→85%, valid-premise 84→75%). A Pareto
   improvement requires no axis to get worse. **Applied:** retitled paragraph to "a clean
   capability–behavior win"; cross-model reading now says "near-Pareto improvement … at the cost of only
   2–9 percentage points on the matched preserve-behaviors (87→85% answerable, 84→75% valid-premise)".

2. **[Important] Selective reporting on Llama ITI c=1 FalseQA.** Passage said "ITI reaches only 21% at
   c=2" but omitted that at c=1 ITI *raises* false-premise acceptance above base (29.0→31.4%) at +8.4%
   ppl. **Applied:** added "at the mild c=1 it actually raises false-premise acceptance above base
   (29%→31%) at +8.4% perplexity".

3. **[Important] "each model needs its own α/ρ" — ρ was never swept on Llama** (fixed at .005 in the
   script/METHODS; only α re-swept). **Applied:** "ρ fixed at 0.005 throughout … its own α (and possibly
   ρ, untested here)".

4. **[Important] "near-zero capability cost" for Llama BLADE contradicts +5.7% ppl.** **Applied:** changed
   to "at modest capability cost (+3.5–5.7% perplexity)".

5. **[Optional] "pass/n-noisy" unclear.** **Applied:** "quantized at 1/70 granularity here, so
   single-case differences fall within sampling noise".

6. **[Optional] Panel refs render as "(1b)"** (`\ref{...}b`, ~10 uses). Left as-is (house convention;
   readable). Could adopt `Fig.~\ref{...}(b)` later.

7. **[Optional] "the honest summary" editorializes in paper voice.** **Applied:** "the summary is that…".

8. **[OK — verified] All cited numbers match ground truth.** Qwen: 20→10, −0.3%, 85/87, +17.5%, 20→16,
   +57.3%, 87→47, 23→15, 75, 14/84, 9/60. Llama: 47→33/37, 73→40, 39/29 vs 65/44, 29→13, 71=base,
   +5.7/+3.5, 21/55/+46.5, ~1.5% incorrect. Panel mapping (a,b,d ↓ / c,e ↑), c∈{4,6} on Qwen, n=70 per
   split, both figure PDFs present, cites (li2023iti/selfaware/falseqa) defined in blade-method.bib.

**Verdict:** numerically sound; issues 1–4 were overclaim/omission a referee would flag. All applied
(1–5, 7); 6 left as house-style choice.
