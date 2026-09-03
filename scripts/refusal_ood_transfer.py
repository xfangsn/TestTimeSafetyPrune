"""Prediction #1: does BLADE refusal removal TRANSFER out-of-distribution?
Parallel to the sycophancy OOD controls, to test the thesis: unitary +
format-aligned behaviors (refusal: contrast = harmful-prompt generation =
deployment format) transfer OOD, unlike the format-bound A/B sycophancy.

Direction/moments + best-first ELS built ONLY from AdvBench (harmful_train vs
harmless). Then evaluate refusal rate at each rho on:
  in-dist = AdvBench harmful_test        (same distribution as the direction)
  OOD     = HarmBench standard           (different harmful content/phrasing)

Three probes (mirror s4_controls):
  A. solo-OOD correlation: per-layer solo Δrefusal in-dist vs OOD.
     Predict POSITIVE corr (contrast: sycophancy was ~0).
  B. BLADE L* transfer: Δrefusal_OOD vs Δrefusal_indist at matched rho.
  C. random-pool control: does BLADE's L* beat random mid-layer pools on OOD?
     Predict YES (contrast: for sycophancy random pools matched BLADE).
"""
import json
import os
import random
from pathlib import Path
from statistics import mean

import torch

from ttsafety.behaviors import (bestfirst_layers, score_edges, solo_layer_pool)
from ttsafety.data import load_jsonl
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.judge import is_refusal
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,
                                   selection_from_ranking)
from blade_refusal_els import last_token_moments, refusal_rate

DATA = Path("data"); RESULTS = Path("results")
MODEL_ID = os.environ.get("BLADE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
COMPONENTS = "both"; PPL_TOKENS = 5000
BETA = 0.05; EPS = 0.005; SCREEN_FRAC = 0.005; GREEDY_TESTFRAC = 0.005
N_DIR = 200; N_SCREEN = 48; SEED = 0
RHOS = [0.0005, 0.002, 0.005]
N_MAIN = 150      # eval prompts per distribution (in-dist / OOD)
N_SOLO = 80       # prompts per distribution for the solo scan
N_RANDOM = 5


def pearson(x, y):
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for i, j in enumerate(o):
            r[j] = i
        return r
    return pearson(rk(x), rk(y))


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    all_layers = list(range(len(get_decoder_layers(model))))
    wiki = load_wikitext_text()

    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmful_val = [r["instruction"] for r in load_jsonl(DATA / "harmful_val.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    in_test = [r["instruction"] for r in load_jsonl(DATA / "harmful_test.jsonl")][:N_MAIN]
    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")][:N_MAIN]

    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    screen = harmful_val[:N_SCREEN]
    base_ref = refusal_rate(model, tok, screen)
    base_in = refusal_rate(model, tok, in_test)
    base_ood = refusal_rate(model, tok, ood)
    print(f"{MODEL_ID} | base refusal screen {base_ref:.3f} | in-dist(AdvBench) {base_in:.3f} "
          f"| OOD(HarmBench) {base_ood:.3f}", flush=True)

    directions = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:N_DIR])
    muH = last_token_moments(model, tok, harmful[:N_DIR], all_layers, COMPONENTS, chat_wrap)
    muU = last_token_moments(model, tok, harmless[:N_DIR], all_layers, COMPONENTS, chat_wrap)

    def ppl_now():
        return teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)

    def measure():
        return refusal_rate(model, tok, screen), ppl_now()

    def sel_at(cand, frac):
        sc = score_edges(model, directions, muH, muU, cand, COMPONENTS)
        return selection_from_ranking(rank_weight_indices(sc, max(frac, 0.01)), frac)

    pool = solo_layer_pool(model, directions, muH, muU, all_layers, COMPONENTS,
                           ppl_now, base_ppl, screen_frac=SCREEN_FRAC, beta=BETA)
    L_star = bestfirst_layers(model, directions, muH, muU, pool, COMPONENTS,
                              measure, base_ref, base_ppl, beta=BETA, eps=EPS,
                              test_frac=GREEDY_TESTFRAC)
    print(f"pool={pool}\nBLADE L* = {L_star}", flush=True)

    # ---- A. solo-OOD correlation ----
    print("\n=== A. solo Δrefusal per layer (in-dist vs OOD) ===", flush=True)
    solo_in, solo_ood = {}, {}
    in_s = in_test[:N_SOLO]; ood_s = ood[:N_SOLO]
    b_in_s = refusal_rate(model, tok, in_s); b_ood_s = refusal_rate(model, tok, ood_s)
    for l in pool:
        with pruned_weights(model, sel_at([l], GREEDY_TESTFRAC)):
            ri = refusal_rate(model, tok, in_s); ro = refusal_rate(model, tok, ood_s)
        solo_in[l] = b_in_s - ri; solo_ood[l] = b_ood_s - ro
        print(f"  L{l:>2}  Δref_in {solo_in[l]:+.3f}   Δref_OOD {solo_ood[l]:+.3f}", flush=True)
    lc = list(pool)
    corr = {"pearson": pearson([solo_in[l] for l in lc], [solo_ood[l] for l in lc]),
            "spearman": spearman([solo_in[l] for l in lc], [solo_ood[l] for l in lc])}
    print(f"  corr(Δref_in, Δref_OOD) pearson {corr['pearson']:+.3f} spearman {corr['spearman']:+.3f}", flush=True)

    # ---- B/C: rho sweep for BLADE L* and random pools ----
    def sweep(tag, cand):
        sc = score_edges(model, directions, muH, muU, cand, COMPONENTS)
        rk = rank_weight_indices(sc, max(0.03, max(RHOS)))
        rows = []
        for rho in RHOS:
            sel = selection_from_ranking(rk, rho)
            n = sum(int(v.numel()) for v in sel.values())
            with pruned_weights(model, sel):
                ri = refusal_rate(model, tok, in_test)
                ro = refusal_rate(model, tok, ood)
                ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
            rows.append({"rho": rho, "n_edges": n, "refusal_in": ri, "refusal_ood": ro,
                         "d_ppl": (ppl - base_ppl) / base_ppl})
            print(f"  [{tag}] rho={rho} ref_in {ri:.3f} ref_OOD {ro:.3f} "
                  f"Δppl {(ppl-base_ppl)/base_ppl:+.1%} ({n:,} edges)", flush=True)
        return {"tag": tag, "L": cand, "rows": rows}

    print("\n=== B. BLADE L* transfer ===", flush=True)
    B = sweep("BLADE", L_star) if L_star else None

    print("\n=== C. random-pool control ===", flush=True)
    size = max(1, len(L_star)) if L_star else 1
    C = []
    for k in range(N_RANDOM):
        rp = sorted(random.Random(1000 + k).sample(pool, min(size, len(pool))))
        C.append(sweep(f"random{k}:{rp}", rp))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "refusal_ood_transfer.json").write_text(json.dumps(
        {"model": MODEL_ID, "beta": BETA, "pool": pool, "L_star": L_star,
         "base": {"screen": base_ref, "in_dist": base_in, "ood": base_ood},
         "solo_in": solo_in, "solo_ood": solo_ood, "correlation": corr,
         "blade": B, "random_pools": C, "env": env_info()}, indent=2, ensure_ascii=False))
    print("\nsaved results/refusal_ood_transfer.json", flush=True)


if __name__ == "__main__":
    main()
