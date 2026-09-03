"""Proper BLADE (ELS + beta) built from pv-prompts, then OOD-tested -- fixing the
earlier shortcut that skipped ELS and swept rho on a fixed window.

Direction/moments from pv-prompts (sycophantic vs non-sycophantic, paired).
ELS: solo ppl-budget pool -> best-first layer selection using the in-distribution
sycophancy metric (Anthropic A/B pick-rate) under budget beta. Then rho sweep on
L*. Report in-dist pick, OOD sycophancy (TriviaQA), OOD accuracy, ppl.
"""
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset

from ttsafety.behaviors import (bestfirst_layers, fetch_ab, make_splits, pick_rate,
                                score_edges, solo_layer_pool)
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,
                                   selection_from_ranking)
from mia_defend import collect
from ood_sycophancy_eval import load_ood, eval_ood

DATA = Path("data"); RESULTS = Path("results")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
COMPONENTS = "both"; PPL_TOKENS = 5000
BETA = 0.05; EPS = 0.005; SCREEN_FRAC = 0.005; TESTFRAC = 0.005
RHOS = [0.0005, 0.002, 0.005, 0.02]
N_DIR = 300


def pv_texts(name, n, tok):
    d = load_dataset(name, split="train")
    return [tok.apply_chat_template(d[i]["messages"], tokenize=False,
                                    add_generation_prompt=False) for i in range(min(n, len(d)))]


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    all_layers = list(range(len(get_decoder_layers(model))))
    wiki = load_wikitext_text()
    ab_val = make_splits(fetch_ab("sycophancy", DATA / "behaviors"))["val"]
    exs = load_ood()

    print("direction/moments from pv-prompts (all layers) ...", flush=True)
    r_s, mu_s = collect(model, tok, pv_texts("cfierro/pv-prompts-sycophantic", N_DIR, tok), all_layers)
    r_n, mu_n = collect(model, tok, pv_texts("cfierro/pv-prompts-non-sycophantic", N_DIR, tok), all_layers)
    directions = {l: r_s[l] - r_n[l] for l in all_layers}

    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_ab, _ = pick_rate(model, tok, ab_val, "matching")

    def ppl_now():
        return teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)

    def measure():   # in-distribution sycophancy metric for ELS
        return pick_rate(model, tok, ab_val, "matching")[0], ppl_now()

    print(f"ELS: base A/B={base_ab:.3f}, beta={BETA} ...", flush=True)
    pool = solo_layer_pool(model, directions, mu_s, mu_n, all_layers, COMPONENTS,
                           ppl_now, base_ppl, screen_frac=SCREEN_FRAC, beta=BETA)
    L_star = bestfirst_layers(model, directions, mu_s, mu_n, pool, COMPONENTS,
                              measure, base_ab, base_ppl, beta=BETA, eps=EPS,
                              test_frac=TESTFRAC)
    print(f"pool={len(pool)} -> L* = {L_star}", flush=True)

    base_ood = eval_ood(model, tok, exs)
    print(f"BASE  in-dist(A/B) {base_ab:.3f}  OOD syco {base_ood['sycophancy']:.3f} "
          f"acc {base_ood['accuracy']:.3f}", flush=True)
    rows = [{"rho": 0.0, "n_edges": 0, "indist_pick": base_ab,
             "ood_sycophancy": base_ood["sycophancy"], "ood_accuracy": base_ood["accuracy"],
             "ppl_delta": 0.0}]
    if L_star:
        scores = score_edges(model, directions, mu_s, mu_n, L_star, COMPONENTS)
        rk = rank_weight_indices(scores, max(0.03, max(RHOS)))
        for rho in RHOS:
            sel = selection_from_ranking(rk, rho)
            n = sum(int(v.numel()) for v in sel.values())
            with pruned_weights(model, sel):
                ab, _ = pick_rate(model, tok, ab_val, "matching")
                ood = eval_ood(model, tok, exs)
                ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
            dppl = (ppl - base_ppl) / base_ppl
            rows.append({"rho": rho, "n_edges": n, "indist_pick": ab,
                         "ood_sycophancy": ood["sycophancy"], "ood_accuracy": ood["accuracy"],
                         "ppl_delta": dppl})
            print(f"BLADE(ELS) rho={rho:<7g} L*={L_star} in-dist {ab:.3f} "
                  f"OOD syco {ood['sycophancy']:.3f} acc {ood['accuracy']:.3f} "
                  f"Δppl {dppl:+.1%} ({n:,} edges)", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ood_sycophancy_blade_els.json").write_text(json.dumps(
        {"model": MODEL_ID, "source": "pv-prompts", "L_star": L_star, "beta": BETA,
         "n_ood": len(exs), "base_ppl": base_ppl, "rows": rows, "env": env_info()}, indent=2))
    print("saved results/ood_sycophancy_blade_els.json", flush=True)


if __name__ == "__main__":
    main()
