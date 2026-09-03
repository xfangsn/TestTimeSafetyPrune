"""E3: does BLADE-G help REMOVE sycophancy at lower collateral than BLADE? Llama-3.2-3B, A/B sycophancy.
Fix L* (the multi-layer sycophancy selection), compute Q on C4, compare BLADE vs BLADE-G(g0,g1scalar)
over a rho sweep: A/B pick-rate (lower=more removed, chance 0.5) + WikiText ΔNLL + C4 ppl. Sycophancy
is the hard case (BLADE capped ~0.66 within budget, high WikiText ppl) -> tests whether Q lets us
remove as much (or more) at lower WikiText damage. Writes results/blade_g_sycophancy.json."""
import json
import math
from pathlib import Path

import torch

from ttsafety.behaviors import (collect_span_input_moments, extract_direction, fetch_ab,
                                make_splits, pick_rate)
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_ppl
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import load_model
from ttsafety.sycophancy import score_edges, score_edges_g
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.weight_prune import (pruned_weights, random_scores_like, rank_weight_indices,
                                   selection_from_ranking)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; RESULTS = ROOT / "results"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"; EOT = "<|eot_id|>"; COMP = "both"
SCREEN = 0.03; PPL_TOKENS = 5000; Q_TOKENS = 65536
RHOS = [0.002, 0.005, 0.01, 0.02]


def clean(S):
    return {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in S.items()}


def med_pos(d):
    t = torch.cat([v.flatten() for v in d.values()]).float(); return t[t > 0].median().item()


def main():
    # L* = the multi-layer sycophancy selection (gives Q room to reroute)
    L = json.loads((RESULTS / "blade_els_llama-32-3b-instruct_syco_multi.json").read_text()
                   )["results"]["sycophancy"]["L_star"]
    print(f"sycophancy L*={L}", flush=True)
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    allL = list(range(len(get_decoder_layers(model))))
    sp = make_splits(fetch_ab("sycophancy", DATA / "behaviors")); train, val = sp["train"], sp["val"]
    side = "matching"; other = "not_matching"
    dirs = extract_direction(model, tok, train, side, eot=EOT)
    mu_a = collect_span_input_moments(model, tok, train, side, allL, COMP, eot=EOT)
    mu_b = collect_span_input_moments(model, tok, train, other, allL, COMP, eot=EOT)
    c4 = load_c4_text(); wiki = load_wikitext_text()
    Qg0, _ = collect_c4_generic_importance(model, tok, L, COMP, text=c4, seqlen=2048, batch_size=2, mode="g0", max_tokens=Q_TOKENS)
    Qg1, meta = collect_c4_generic_importance(model, tok, L, COMP, text=c4, seqlen=2048, batch_size=2, mode="g1scalar", max_tokens=Q_TOKENS)

    blade = score_edges(model, dirs, mu_a, mu_b, sorted(L), COMP)
    cmed = med_pos(blade); s1 = cmed / med_pos(Qg1); s0 = cmed / med_pos(Qg0)
    SCORES = {"BLADE": blade,
              "BLADE-G_g0": clean(score_edges_g(model, dirs, mu_a, mu_b, sorted(L), COMP, Q=Qg0, lam=s0)),
              "BLADE-G_g1scalar": clean(score_edges_g(model, dirs, mu_a, mu_b, sorted(L), COMP, Q=Qg1, lam=s1)),
              "random": random_scores_like(blade, seed=0)}

    base_ppl_c4 = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    base_ppl_wiki = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_pick = pick_rate(model, tok, val, side)[0]
    print(f"base pick {base_pick:.3f} (chance .5) | ppl C4 {base_ppl_c4:.2f} Wiki {base_ppl_wiki:.2f} "
          f"| scale_g1 {s1:.2e} scale_g0 {s0:.2e}", flush=True)

    out = {"model": MODEL_ID, "L": L, "base": {"pick": base_pick, "ppl_c4": base_ppl_c4,
           "ppl_wiki": base_ppl_wiki}, "rhos": RHOS, "cells": [], "env": {}}
    for name, S in SCORES.items():
        rk = rank_weight_indices(S, SCREEN)
        for rho in RHOS:
            sel = selection_from_ranking(rk, rho)
            with pruned_weights(model, sel):
                pk = pick_rate(model, tok, val, side)[0]
                pc4 = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
                pwk = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
            dc4 = (pc4 - base_ppl_c4) / base_ppl_c4; dwk = (pwk - base_ppl_wiki) / base_ppl_wiki
            out["cells"].append({"score": name, "rho": rho, "pick": pk, "ppl_c4": dc4, "ppl_wiki": dwk,
                                 "within_c4_budget": dc4 <= 0.05})
            print(f"  {name:18} rho={rho:<5} pick {pk:.3f} C4{dc4:+.1%} Wiki{dwk:+.1%} "
                  f"{'OK' if dc4<=0.05 else 'x'}", flush=True)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "blade_g_sycophancy.json").write_text(json.dumps(out, indent=2))
    print("saved results/blade_g_sycophancy.json", flush=True)


if __name__ == "__main__":
    main()
