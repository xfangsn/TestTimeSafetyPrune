"""Does NEGATING the pruned edges (W <- factor*W, factor<0) remove an A/B
behavior further than zeroing them? BLADE selects edges with POSITIVE
contribution to the behavior; zeroing (factor=0) drops that contribution,
negating (factor<0) flips it, potentially pushing the behavior past chance.
Same mask throughout. Measures val pick-rate + WikiText ppl per factor, on Llama.
"""
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch

from ttsafety.behaviors import (collect_span_input_moments, extract_direction,
                                fetch_ab, make_splits, pick_rate, score_edges)
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.models import env_info, load_model
from ttsafety.weight_prune import (rank_weight_indices, selection_from_ranking,
                                   _resolve_modules)

DATA = Path("data"); RESULTS = Path("results")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
EOT = "<|eot_id|>"; COMPONENTS = "both"; PPL_TOKENS = 5000
FACTORS = [1.0, 0.0, -0.5, -1.0, -2.0, -3.0]
CONFIGS = [
    ("sycophancy", "matching", [12, 15], 0.005),
    ("power-seeking", "matching", [11, 1, 22, 14, 19, 6, 9, 13], 0.005),
]


@contextmanager
def scaled_weights(model, selection, factor):
    modules = _resolve_modules(model, list(selection))
    backups = {}
    with torch.no_grad():
        for name, idx in selection.items():
            w = modules[name].weight
            di = idx.to(w.device, torch.long)
            flat = w.view(-1)
            backups[name] = flat[di].detach().cpu().clone()
            flat[di] = flat[di] * factor
    try:
        yield
    finally:
        with torch.no_grad():
            for name, idx in selection.items():
                w = modules[name].weight
                di = idx.to(w.device, torch.long)
                w.view(-1)[di] = backups[name].to(w.device, w.dtype)


def build_selection(model, tok, name, side, layers, rho):
    rows = fetch_ab(name, DATA / "behaviors")
    sp = make_splits(rows)
    other = "not_matching" if side == "matching" else "matching"
    directions = extract_direction(model, tok, sp["train"], side, eot=EOT)
    mu_a = collect_span_input_moments(model, tok, sp["train"], side, layers, COMPONENTS, eot=EOT)
    mu_b = collect_span_input_moments(model, tok, sp["train"], other, layers, COMPONENTS, eot=EOT)
    scores = score_edges(model, directions, mu_a, mu_b, layers, COMPONENTS)
    rk = rank_weight_indices(scores, max(0.03, rho))
    return selection_from_ranking(rk, rho), sp["val"]


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    report = {"model": MODEL_ID, "base_ppl": base_ppl, "env": env_info(), "configs": {}}

    for name, side, layers, rho in CONFIGS:
        sel, val = build_selection(model, tok, name, side, layers, rho)
        n = sum(int(v.numel()) for v in sel.values())
        base_pick, _ = pick_rate(model, tok, val, side)
        print(f"\n=== {name}  L*={layers}  rho={rho:.1%}  {n:,} edges  base={base_pick:.3f} ===",
              flush=True)
        rows = []
        for f in FACTORS:
            with scaled_weights(model, sel, f):
                pi, _ = pick_rate(model, tok, val, side)
                ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
            dppl = (ppl - base_ppl) / base_ppl
            tag = ("base" if f == 1 else "ZERO(BLADE)" if f == 0 else f"negate x{f:g}")
            rows.append({"factor": f, "pick": pi, "ppl_delta": dppl})
            print(f"  {tag:14} factor={f:>5g} | pick {pi:.3f} | Δppl {dppl:+.1%}", flush=True)
        report["configs"][name] = {"side": side, "L_star": layers, "rho": rho,
                                   "n_edges": n, "base_pick": base_pick, "sweep": rows}

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "blade_ab_negate.json").write_text(json.dumps(report, indent=2))
    print("\nsaved results/blade_ab_negate.json", flush=True)


if __name__ == "__main__":
    main()
