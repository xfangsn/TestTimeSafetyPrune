"""E1: does BLADE-G (BLADE score minus a C4 generic-importance penalty Q) remove refusal as well as
BLADE at LOWER general-capability damage? Fixed L*=[12] (Llama-3.2-3B refusal). For each score
{BLADE, BLADE-G g1scalar (lam x1, x10), BLADE-G g0 (x1), equal-size random}, sweep rho and prune; at
each point measure in-dist refusal (AdvBench-style plain gen), OOD refusal (HarmBench plain), and
WikiText ΔNLL. The winner reaches a given refusal removal at lower ΔNLL. Writes results/blade_g_refusal.json."""
import json
import math
from pathlib import Path

import torch

from ttsafety.data import load_jsonl
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_nll
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.judge import is_refusal
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.sycophancy import score_edges, score_edges_g
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.weight_prune import (pruned_weights, random_scores_like, rank_weight_indices,
                                   selection_from_ranking)
from blade_refusal_els import last_token_moments
from blade_steering_baselines import _generate, GEN_TOKENS

DATA = Path("data"); RESULTS = Path("results")
L = [12]; COMP = "both"; N_DIR = 200; SCREEN = 0.03; PPL_TOKENS = 5000
RHOS = [0.002, 0.005, 0.01, 0.02]
Q_TOKENS = 65536


def plain_refusal(model, tok, prompts):
    outs = _generate(model, tok, [chat_wrap(tok, p) for p in prompts], GEN_TOKENS)
    return sum(is_refusal(o) for o in outs) / len(outs)


def main():
    model, tok = load_model("meta-llama/Llama-3.2-3B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    val = [r["instruction"] for r in load_jsonl(DATA / "harmful_val.jsonl")]
    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")]
    wiki = load_wikitext_text(); c4 = load_c4_text()
    allL = list(range(len(get_decoder_layers(model))))

    dirs = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:192])
    muH = last_token_moments(model, tok, harmful[:N_DIR], allL, COMP, chat_wrap)
    muU = last_token_moments(model, tok, harmless[:N_DIR], allL, COMP, chat_wrap)
    print("collecting Q (g0, g1scalar) on C4 ...", flush=True)
    Qg0, _ = collect_c4_generic_importance(model, tok, L, COMP, text=c4, seqlen=2048,
                                           batch_size=2, mode="g0", max_tokens=Q_TOKENS)
    Qg1, meta = collect_c4_generic_importance(model, tok, L, COMP, text=c4, seqlen=2048,
                                              batch_size=2, mode="g1scalar", max_tokens=Q_TOKENS)
    print(f"Q meta {meta}", flush=True)

    blade = score_edges(model, dirs, muH, muU, L, COMP)
    def med_pos(d):
        t = torch.cat([v.flatten() for v in d.values()]).float()
        return t[t > 0].median().item()
    cmed = med_pos(blade)                       # median positive BLADE numerator
    scale1 = cmed / med_pos(Qg1)                # lambda scale for g1scalar
    scale0 = cmed / med_pos(Qg0)                # lambda scale for g0
    print(f"scale_g1={scale1:.3e} scale_g0={scale0:.3e}", flush=True)

    def clean(S):   # replace -inf (abstained) with 0 so topk ranks finite scores first
        return {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in S.items()}

    scale = scale1
    SCORES = {
        "BLADE": blade,
        "BLADE-G_g1scalar_lam1": clean(score_edges_g(model, dirs, muH, muU, L, COMP, Q=Qg1, lam=scale1)),
        "BLADE-G_g1scalar_lam10": clean(score_edges_g(model, dirs, muH, muU, L, COMP, Q=Qg1, lam=10 * scale1)),
        "BLADE-G_g0_lam1": clean(score_edges_g(model, dirs, muH, muU, L, COMP, Q=Qg0, lam=scale0)),
        "random": random_scores_like(blade, seed=0),
    }

    base_nll, _ = teacher_forced_nll(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_val = plain_refusal(model, tok, val); base_ood = plain_refusal(model, tok, ood)
    print(f"base: in-dist refusal {base_val:.3f} | OOD {base_ood:.3f} | wiki nll {base_nll:.4f}", flush=True)

    out = {"model": "meta-llama/Llama-3.2-3B-Instruct", "L": L, "Q_meta": meta, "scale": scale,
           "base": {"val_refusal": base_val, "ood_refusal": base_ood, "wiki_nll": base_nll},
           "rhos": RHOS, "cells": [], "env": env_info()}
    ranks = {name: rank_weight_indices(S, SCREEN) for name, S in SCORES.items()}
    for name, rk in ranks.items():
        for rho in RHOS:
            sel = selection_from_ranking(rk, rho)
            n = sum(int(v.numel()) for v in sel.values())
            with pruned_weights(model, sel):
                vr = plain_refusal(model, tok, val); orr = plain_refusal(model, tok, ood)
                nll, _ = teacher_forced_nll(model, tok, wiki, max_tokens=PPL_TOKENS)
            dnll = nll - base_nll
            out["cells"].append({"score": name, "rho": rho, "n_edges": n, "val_refusal": vr,
                                 "ood_refusal": orr, "wiki_dnll": dnll, "wiki_relppl": math.exp(dnll) - 1})
            print(f"  {name:24} rho={rho:<5} n={n:>7} valRef {vr:.3f} oodRef {orr:.3f} "
                  f"ΔNLL {dnll:+.4f} (Δppl {math.exp(dnll)-1:+.1%})", flush=True)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "blade_g_refusal.json").write_text(json.dumps(out, indent=2))
    print("saved results/blade_g_refusal.json", flush=True)


if __name__ == "__main__":
    main()
