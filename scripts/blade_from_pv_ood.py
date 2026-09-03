"""BLADE built from the SAME source as weight-steering (cfierro/pv-prompts, paired
sycophantic vs non-sycophantic responses), then evaluated OOD on
cfierro/sycophancy_eval_answer -- for a controlled generalization comparison.

r_l   = mean last-tok block-output (sycophantic chat) - (non-sycophantic chat)
Delta = writer-input last-tok mean (sycophantic) - (non-sycophantic)
Zero top-k positive edges; sweep rho. Report OOD sycophancy (their benchmark),
in-distribution sycophancy (Anthropic A/B pick-rate), and WikiText ppl.
"""
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset

from ttsafety.behaviors import fetch_ab, make_splits, pick_rate, score_edges
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,
                                   selection_from_ranking)
from mia_defend import collect                      # last-tok block-out + writer-in means
from ood_sycophancy_eval import load_ood, eval_ood  # OOD benchmark + scorer

DATA = Path("data"); RESULTS = Path("results")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
COMPONENTS = "both"; PPL_TOKENS = 5000
RHOS = [0.0005, 0.002, 0.005, 0.02]
N_DIR = 300


def pv_texts(name, n, tok):
    d = load_dataset(name, split="train")
    out = []
    for r in d:
        msgs = r["messages"]
        out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
        if len(out) >= n:
            break
    return out


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    n_layers = len(get_decoder_layers(model))
    layers = list(range(2, n_layers - 2))

    print("building BLADE direction/moments from pv-prompts ...", flush=True)
    syco_tx = pv_texts("cfierro/pv-prompts-sycophantic", N_DIR, tok)
    nons_tx = pv_texts("cfierro/pv-prompts-non-sycophantic", N_DIR, tok)
    r_s, mu_s = collect(model, tok, syco_tx, layers)     # sycophantic side (A)
    r_n, mu_n = collect(model, tok, nons_tx, layers)     # non-sycophantic side (B)
    directions = {l: r_s[l] - r_n[l] for l in layers}
    scores = score_edges(model, directions, mu_s, mu_n, layers, COMPONENTS)
    rk = rank_weight_indices(scores, max(0.03, max(RHOS)))

    # eval fixtures
    exs = load_ood()
    ab_val = make_splits(fetch_ab("sycophancy", DATA / "behaviors"))["val"]
    wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_ab, _ = pick_rate(model, tok, ab_val, "matching")
    base_ood = eval_ood(model, tok, exs)
    print(f"BASE  in-dist(A/B) {base_ab:.3f}  OOD syco {base_ood['sycophancy']:.3f} "
          f"acc {base_ood['accuracy']:.3f}", flush=True)

    rows = [{"rho": 0.0, "n_edges": 0, "indist_pick": base_ab,
             "ood_sycophancy": base_ood["sycophancy"], "ood_accuracy": base_ood["accuracy"],
             "ppl_delta": 0.0}]
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
        print(f"BLADE rho={rho:<7g} in-dist {ab:.3f}  OOD syco {ood['sycophancy']:.3f} "
              f"acc {ood['accuracy']:.3f}  Δppl {dppl:+.1%}  ({n:,} edges)", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ood_sycophancy_blade.json").write_text(json.dumps(
        {"model": MODEL_ID, "source": "pv-prompts", "n_ood": len(exs),
         "base_ppl": base_ppl, "rows": rows, "env": env_info()}, indent=2))
    print("saved results/ood_sycophancy_blade.json", flush=True)


if __name__ == "__main__":
    main()
