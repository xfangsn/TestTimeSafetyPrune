"""Gemma-3-4B sycophancy: instead of zeroing the BLADE-selected weights (alpha=0), try scaling them
by a NEGATIVE factor (sign-flip). Does negating the behavior-writing weights push the A/B pick-rate
below what pruning achieves (0.660), possibly past 0.5, while staying within the 5% ppl budget?
Sweep alpha over the SAME selected mask (L* from beta5_c4, operating rho) and report pick + C4/Wiki ppl."""
import json
from pathlib import Path

import torch

from ttsafety.behaviors import (collect_span_input_moments, extract_direction, fetch_ab,
                                make_splits, pick_rate, score_edges)
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_ppl
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import load_model
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking
from blade_refusal_amplify import scaled_weights

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; RESULTS = ROOT / "results"
MODEL_ID = "google/gemma-3-4b-it"; EOT = "<end_of_turn>"; COMPONENTS = "both"; PPL_TOKENS = 5000
ALPHAS = [0.0, 2.0, -0.25, 2.25, -1.0, 3.0]


def main():
    rec = json.loads((RESULTS / "blade_els_gemma-3-4b-it_beta5_c4.json").read_text())["results"]["sycophancy"]
    L_star = rec["L_star"]; side = rec["side"]
    # operating rho = best-within-C4-budget point (matches the figure)
    sw = rec["Lstar_sweep"]; within = [s for s in sw if s["ppl_delta"] <= 0.05] or sw
    rho = min(within, key=lambda s: s["pick_rate"])["sparsity"]
    other = "not_matching" if side == "matching" else "matching"
    print(f"L*={L_star} side={side} rho={rho}", flush=True)

    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    all_layers = list(range(len(get_decoder_layers(model))))
    sp = make_splits(fetch_ab("sycophancy", DATA / "behaviors")); train, val = sp["train"], sp["val"]
    directions = extract_direction(model, tok, train, side, eot=EOT)
    mu_a = collect_span_input_moments(model, tok, train, side, all_layers, COMPONENTS, eot=EOT)
    mu_b = collect_span_input_moments(model, tok, train, other, all_layers, COMPONENTS, eot=EOT)
    sc = score_edges(model, directions, mu_a, mu_b, sorted(L_star), COMPONENTS)
    sel = selection_from_ranking(rank_weight_indices(sc, max(rho, 0.01)), rho)

    c4 = load_c4_text(); wiki = load_wikitext_text()
    base_c4 = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    base_wiki = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_pick = pick_rate(model, tok, val, side)[0]
    print(f"base pick={base_pick:.3f} | base ppl C4 {base_c4:.2f} / Wiki {base_wiki:.2f}", flush=True)

    out = {"model": MODEL_ID, "L_star": L_star, "rho": rho, "base_pick": base_pick, "cells": []}
    for a in ALPHAS:
        with scaled_weights(model, sel, a):
            pk = pick_rate(model, tok, val, side)[0]
            c4p = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
            wkp = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
        dc4 = (c4p - base_c4) / base_c4; dwk = (wkp - base_wiki) / base_wiki
        ok = "OK" if dc4 <= 0.05 else "x"
        out["cells"].append({"alpha": a, "pick": pk, "ppl_c4": dc4, "ppl_wiki": dwk, "within_budget": dc4 <= 0.05})
        print(f"  alpha={a:<5} pick={pk:.3f} C4{dc4:+.2%} Wiki{dwk:+.2%} {ok}", flush=True)
    (RESULTS / "blade_gemma_syco_neg_control.json").write_text(json.dumps(out, indent=2))
    print("saved results/blade_gemma_syco_neg.json", flush=True)


if __name__ == "__main__":
    main()
