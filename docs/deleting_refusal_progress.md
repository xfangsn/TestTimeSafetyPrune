# Deleting Refusal — progress & resume notes

**Last updated:** 2026-09-01. **Section target:** `\subsubsection{Deleting Refusal}` (`\label{sec:exp-refusal-amplify}`).
**Model:** meta-llama/Llama-3.2-3B-Instruct. **Env:** `.venv/bin/python`, `PYTHONPATH=src:scripts`.

## What this is
Flip the OOD-refusal *strengthening* methods into their *removal* forms and compare how selectively
each deletes refusal on held-out HarmBench (plain generation, no prefill; base refuses **88.5%**).
Leakage-safe: pick each config on in-dist `harmful_val` to **minimize residual refusal s.t. C4
rel-ppl ≤ 5%**, freeze+hash, then report on held-out HarmBench + WikiText ppl. Metric = **residual
refusal rate** (string matcher + Kimi-K3 refuse-vs-not); lower = more deleted. (We report refusal
rate, NOT "ASR/harmful completion" — see review note below.)

## Status: DONE vs PENDING
- ✅ **Controlled panel** (all methods on shared residual-writer surface o_proj+down_proj) — string + LLM.
- ✅ **Paper-native panel** (each baseline in its native form) — **string only**.
- ⏳ **PENDING (needs Kimi quota):** LLM refusal column for the **native** panel. Kimi hit its 5-hour
  usage cap (403). On the controlled panel string≈LLM (within a few points), so string is a reliable
  primary; the LLM column is a cross-check.

### To finish the pending step (when kimi has usage)
```bash
cd ~/Projects/TestTimeSafetyPrune
# quick check quota is back:
~/.local/bin/kimi -p "Reply with exactly: OK" --print --quiet -m kimi-code/k3
# then generate+judge the 5 native frozen configs (model gen + kimi judge, ~15 min):
setsid nohup env PYTHONPATH=src:scripts .venv/bin/python scripts/gen_weaken_native_for_judge.py \
  > logs/gen_weaken_native_judge.log 2>&1 &
# fills results/weaken_native_llm_judge.json; then add the LLM column to
# docs/deleting_refusal_native_table.tex (LLM refusal = 1 - llm_comply, i.e. summ[key]["llm_refusal"]).
```

## Results

### Controlled panel — held-out HarmBench, residual refusal (string / LLM), Δppl
| Method | Removal form | Config | Δppl | refusal str | refusal LLM |
|---|---|---|---|---|---|
| **BLADE** (ours) | zero top-ρ writer weights | L12, ρ=0.5% | +0.3% | **0.000** | **0.033** |
| Wei et al. | prune SNIP set-difference | q=p=0.5% | +0.9% | 0.170 | 0.180 |
| Arditi et al. | directional ablation (I−rrᵀ)W | s=0.5 | +1.2% | 0.220 | 0.293 |
| CAA (Rimsky) | subtract steering vector | L12, c=−0.2 | +0.5% | 0.870 | 0.833 |
| ActAdd (Turner) | subtract contrast vector | L16, c=−16 | +5.2% | 0.855 | 0.847 |

Table file: `docs/deleting_refusal_table.tex` (PRIMARY, apples-to-apples on one edit surface).
Data: `results/blade_weaken_ood.json`, `results/weaken_llm_judge.json`, `results/weaken_gens_for_judge.json`.

### Paper-native panel — held-out HarmBench, residual refusal (string), Δppl
| Method | Native form | Config | Δppl | refusal str | LLM |
|---|---|---|---|---|---|
| **BLADE** (ours) | zero top-ρ writer weights | L12, ρ=0.5% | +0.3% | **0.000** | ⏳ |
| Wei (native) | SNIP set-diff over ALL linear matrices | q=p=0.5% | +14.9%† | 0.025 | ⏳ |
| Arditi (native) | single global dir, s=1, all writers | src L8 | +0.3% | 0.370 | ⏳ |
| CAA (native) | subtract vector, wide sweep | L14, c=−0.35 | +2.6% | 0.805 | ⏳ |
| ActAdd (native) | subtract vector, wide sweep | L16, c=−16 | +5.2%† | 0.855 | ⏳ |

† exceeds the 5% ppl budget; reported as least-refusing config. Wei-native: NO config within budget
(all q give +13–21% ppl). Table file: `docs/deleting_refusal_native_table.tex` (companion/robustness).
Data: `results/blade_weaken_native.json`, `results/blade_weaken_native.frozen.json`.

## Takeaways (both panels agree)
1. Only **BLADE** deletes refusal completely (0.000) within the capability budget (+0.3% ppl).
2. **Wei-native** all-matrix pruning *can* delete refusal (0.025) but only by wrecking perplexity
   (+15–21%) — highlights BLADE's selectivity.
3. **Arditi** ablation deletes partially (0.22–0.37); **CAA/ActAdd** negative steering cannot delete
   refusal even with wide sweeps (not an under-tuning artifact — larger coefs just blow the ppl budget).

## Code / data inventory
- Controlled eval: `scripts/blade_weaken_ood.py` → `results/blade_weaken_ood.json`
- Controlled gen+judge: `scripts/gen_weaken_for_judge.py` → `results/weaken_{gens_for_judge,llm_judge}.json`
- Native SNIP rescore (all 196 linear matrices): `scripts/score_wei_snip_native.py` →
  `data/weight_scores/wei_{safety,utility}_snip_native.pt` (11.3 GB each; already computed)
- Native eval: `scripts/blade_weaken_native.py` → `results/blade_weaken_native.json`
- Native gen+judge (PENDING run): `scripts/gen_weaken_native_for_judge.py`
- Tables: `docs/deleting_refusal_table.tex`, `docs/deleting_refusal_native_table.tex`
- Bib keys (present in `docs/blade-method.bib`): wei2024brittleness, arditi2024refusal, rimsky2024caa,
  turner2024actadd. Tables need `\usepackage{booktabs}`.

## Review outcome (codex source-verified + kimi partial, 2026-09-01)
Implementations are sound but must be framed as a **controlled-surface refusal-REMOVAL** comparison,
NOT "five faithful paper algorithms" / NOT "ASR/harmful completion":
- Metric = residual **refusal rate** (string + LLM refuse-vs-not); COMPLY/REFUSAL does not prove harmful
  task fulfillment. "Deleting Refusal" only needs refusal removal → refusal rate measures it directly.
- Controlled panel restricts all methods to o_proj+down_proj (Wei/Arditi natively edit more) → the
  native panel above is the fairness check.
- Wei whole-matrix flatten = faithful to authors' released `prune_wandg_set_difference` (paper text
  says per-row). Arditi controlled = per-layer dirs at budget-forced s=0.5; native = single dir s=1.
  CAA span-mean extraction ≠ original single-answer-token MC (application faithful). ActAdd faithful.

## Possible next steps
- Fill native LLM column (above).
- Decide table placement: controlled = main table; native = appendix/robustness.
- Optional: HarmBench-classifier or harmful-fulfillment judge (stronger than refuse-vs-not) if a
  "genuine jailbreak" claim is ever made (not needed for the refusal-deletion claim).
