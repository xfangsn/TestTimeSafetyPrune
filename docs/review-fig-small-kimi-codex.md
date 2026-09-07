# Reviews — docs/fig_uncertainty_method_cmp_small.tex (Qwen3-4B + Llama-3.2-3B)

LaTeX drafted by Fable 5.1 (`--model claude-fable-5-1`). Reviewed by **Kimi Code** (agent mode, verified
numbers against plot scripts + `results/q4b_cmp{,2}_judged.json` + sweep JSONs) and **Codex** (VS Code
extension). Both confirmed all Qwen3-4B prose numbers match the judged data; issues below reconciled and
applied.

## Applied

1. **[Kimi/Codex] Pareto wording overstated ties.** "clean separation" / "genuine Pareto" / "better on
   every panel" ignored two exact ties (SAans 81=81, FQfa 24=24 for ρ.002 vs c2; FQfa 20=20 for ρ.02 vs
   c4). → "no worse on any of the four behaviors—better on two, tied on two"; dropped "genuine/clean";
   added an explicit small-margin caveat (n=70, gaps of a few cases).
2. **[Codex] "far more" matched behavior → concrete pp.** → "+4 pp SelfAware answering (71 vs 67), +3 pp
   true-premise (73 vs 70)".
3. **[Codex] Perplexity ≠ capability.** "≈30% capability tax" / "at no cost" / "essentially for free" →
   phrased as perplexity increase / "no perplexity cost" throughout.
4. **[Codex] Sweep-superiority/optimality overclaim.** "far harder than any ITI dose" → "harder than any
   ITI dose we tried"; "operating sweet spot" → "selected operating point"; "ITI cannot" → "ITI, among the
   doses we swept, reaches ... only with".
5. **[Codex] Cross-model "removes"/"preserving" overstated.** "removes the targeted failure" → "reduces";
   restricted the full-preservation claim to Llama FalseQA at α=2.0 (71%=base).
6. **[Codex High / Kimi disagreed] "base already abstains on much of that [SelfAware] split."** Codex: a
   misplaced SimpleQA fact; Kimi: technically true (Llama base abstains ~53% of unanswerable). Resolved by
   removing the contested clause and re-justifying FalseQA as "the clearer axis" via consistency +
   n-granularity, not base-abstention.
7. **[Kimi] ACT taxonomy undefined.** → added the label set \{answer, abstain, reject-premise, discuss,
   mixed\} in Benchmarks.
8. **[Kimi] "thinking off" doesn't apply to Llama-3.2-3B.** → "thinking-off on the thinking-capable
   Qwen3-4B".
9. **[Kimi] Llama caption not standalone.** → added "doses re-swept ... ITI c∈{1,2}, BLADE α∈{1.75,2.0}".
10. **[Kimi] Provenance.** Committed `results/iti_ppl_qwen3-4b.json`, `iti_ppl_llama-3.2-3b-instruct.json`,
    and `q4b_cmp{,2}_judged.json`. (Llama behavior rates remain hard-coded in `plot_llama_method_cmp.py`
    from the earlier session's judged run; noted for future backfill.)

## Not changed (verified accurate)
- Protocol claims (blind Opus judge, teacher-forced C4 ppl, same fit contrast/last-token, BLADE-G/ELS):
  Codex flagged as unverifiable from the brief, but they are accurate to the actual pipeline and Kimi
  confirmed ELS + the ρ sweep set {.002,.005,.01,.02} against the code. Kept.
- All 20 per-model prose numbers verified against plot scripts; Qwen rates verified against judged JSONs;
  panel→metric→direction mapping, n=70, 1.4 pp, and bib keys all confirmed by both reviewers.
