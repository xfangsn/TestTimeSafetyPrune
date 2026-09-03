"""Over-pruning test: does pruning MORE flip a behavior fully to its opposite,
or does it bottom out near/below chance and then collapse capability?

For a behavior on its ELS effective layers, sweep sparsity well past the removal
point and track pick-rate (behavior side) vs wikitext ppl. Below 0.5 = leaning to
the opposite; a rebound + ppl blow-up = capability collapse, not a clean flip.
"""
import json
from pathlib import Path
import torch
from ttsafety.behaviors import (behavior_edge_scores, fetch_ab, make_splits, pick_rate)
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.models import load_model
from ttsafety.weight_prune import pruned_weights, rank_weight_indices, selection_from_ranking

DATA = Path("data"); RESULTS = Path("results")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
EOT = "<|eot_id|>"
TARGETS = {"self-awareness": [11, 13, 14], "power-seeking": [10, 11, 12]}
GRID = [0.002, 0.005, 0.02, 0.05, 0.1, 0.2]


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=8000)
    print(f"base ppl {base_ppl:.2f}", flush=True)
    out = {"model": MODEL_ID, "base_ppl": base_ppl, "runs": {}}
    for name, L in TARGETS.items():
        rows = fetch_ab(name, DATA / "behaviors"); sp = make_splits(rows)
        rate_m, _ = pick_rate(model, tok, sp["val"], "matching")
        side = "matching" if rate_m >= 0.5 else "not_matching"
        base_pi, _ = pick_rate(model, tok, sp["val"], side)
        scores, _ = behavior_edge_scores(model, tok, sp["train"], side, L, "both", eot=EOT)
        rk = rank_weight_indices(scores, 0.25, per_matrix_cap=0.5)
        print(f"\n[{name}] L*={L} baseline π={base_pi:.3f}", flush=True)
        sweep = []
        for frac in GRID:
            sel = selection_from_ranking(rk, frac)
            with pruned_weights(model, sel):
                pi, _ = pick_rate(model, tok, sp["val"], side)
                ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=8000)
            dppl = (ppl - base_ppl) / base_ppl
            sweep.append({"sparsity": frac, "pick_rate": pi, "ppl_delta": dppl})
            print(f"  s={frac:.2%}  π {pi:.3f}  Δppl {dppl:+.1%}", flush=True)
        out["runs"][name] = {"L": L, "baseline_pi": base_pi, "sweep": sweep}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "blade_overprune_llama.json").write_text(json.dumps(out, indent=2))
    print("\nsaved results/blade_overprune_llama.json", flush=True)


if __name__ == "__main__":
    main()
