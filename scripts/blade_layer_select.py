"""Compare 3 data-driven layer strategies for BLADE removal, on behaviors where
single-layer ELS disagrees with the window method.
  S0 = solo ELS  : per-layer solo screen -> L* = layers passing (delta, beta).
  S1 = window+topk: score a broad window, global top-k prune (no explicit select).
  S2 = greedy joint: rank layers by solo effect, greedily add measuring the JOINT
       removal, keep a layer only if it improves the joint result within ppl.
For each strategy: best within-budget pick-rate (base -> post) + layers used.
"""
import json
import os
from collections import Counter
from pathlib import Path

import torch

import ttsafety.behaviors as B
from ttsafety.behaviors import (collect_span_input_moments, extract_direction,
                                fetch_ab, make_splits, pick_rate, score_edges, CATALOG)
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,
                                   selection_from_ranking)

DATA = Path("data"); RESULTS = Path("results")
MODEL_ID = os.environ.get("BLADE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")


def qwen_wrap(tok, s):
    m = [{"role": "user", "content": s}]
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
BEHAVIORS = ["sycophancy", "corrigibility", "power-seeking", "self-awareness",
             "wealth-seeking", "deception", "self-rate-highly"]
COMPONENTS = "both"
SCREEN_FRAC = 0.005
DELTA, BETA, EPS = 0.10, 0.05, 0.005
MIN_BIAS = 0.10
SPARS = [0.0005, 0.002, 0.005, 0.02]
GREEDY_TESTFRAC = 0.005
GREEDY_MARGIN = 0.02
PPL_TOKENS = 6000


def layer_hist(sel):
    c = Counter()
    for name, idx in sel.items():
        c[int(name.split(".")[1])] += len(idx)
    return dict(sorted(c.items()))


def sweep_on(model, tok, directions, mu_a, mu_b, layers, side, val, wiki, base_ppl):
    """score->rank->sweep on a layer set; return best within-budget."""
    scores = score_edges(model, directions, mu_a, mu_b, layers, COMPONENTS)
    rk = rank_weight_indices(scores, 0.03)
    best = {"pick": None, "ppl_delta": None, "sparsity": None, "layers": None}
    base_pick, _ = pick_rate(model, tok, val, side)
    best["pick"] = base_pick
    for frac in SPARS:
        sel = selection_from_ranking(rk, frac)
        with pruned_weights(model, sel):
            pi, _ = pick_rate(model, tok, val, side)
            ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
        dppl = (ppl - base_ppl) / base_ppl
        if dppl <= BETA and pi < best["pick"]:
            best = {"pick": pi, "ppl_delta": dppl, "sparsity": frac,
                    "layers": layer_hist(sel)}
    return best


def main():
    mid = MODEL_ID.lower()
    is_qwen = "qwen" in mid
    is_gemma = "gemma" in mid
    is_phi = "phi" in mid
    eot = ("<|im_end|>" if is_qwen else "<end_of_turn>" if is_gemma
           else "<|end|>" if is_phi else "<|eot_id|>")
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if is_qwen:
        B.chat_wrap = qwen_wrap
    n = len(get_decoder_layers(model))
    all_layers = list(range(n))
    win = list(range(4, n - 4))     # broad "reasonable" window (drop edges)
    wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    print(f"{MODEL_ID}: {n} layers | window S1 = L{win[0]}-L{win[-1]} | base ppl {base_ppl:.2f}\n",
          flush=True)

    report = {"model": MODEL_ID, "env": env_info(), "window_s1": [win[0], win[-1]],
              "base_ppl": base_ppl, "behaviors": {}}
    for name in BEHAVIORS:
        rows = fetch_ab(name, DATA / "behaviors")
        sp = make_splits(rows)
        rm, _ = pick_rate(model, tok, sp["val"], "matching")
        side = "matching" if rm >= 0.5 else "not_matching"
        base_pick, _ = pick_rate(model, tok, sp["val"], side)
        if abs(rm - 0.5) < MIN_BIAS:
            report["behaviors"][name] = {"side": side, "baseline": base_pick,
                                         "status": "not-exhibited"}
            print(f"=== {name} | baseline={base_pick:.3f} -> not exhibited (skip) ===",
                  flush=True)
            continue
        print(f"=== {name} | side={side} baseline={base_pick:.3f} ===", flush=True)

        directions = extract_direction(model, tok, sp["train"], side, eot=eot)
        mu_a = collect_span_input_moments(model, tok, sp["train"], side, all_layers, COMPONENTS, eot=eot)
        mu_b = collect_span_input_moments(model, tok, sp["train"], "not_matching" if side == "matching" else "matching",
                                          all_layers, COMPONENTS, eot=eot)

        # solo screen (for S0 + S2 ranking)
        solo = []
        for l in all_layers:
            sc = score_edges(model, directions, mu_a, mu_b, [l], COMPONENTS)
            sel = selection_from_ranking(rank_weight_indices(sc, 0.01), SCREEN_FRAC)
            with pruned_weights(model, sel):
                pi, _ = pick_rate(model, tok, sp["val"], side)
                ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
            solo.append({"l": l, "d": base_pick - pi, "dppl": (ppl - base_ppl) / base_ppl})

        # S0: solo ELS
        L0 = sorted(r["l"] for r in solo if r["d"] >= DELTA and r["dppl"] <= BETA)
        s0 = sweep_on(model, tok, directions, mu_a, mu_b, L0, side, sp["val"], wiki, base_ppl) if L0 else None

        # S1: window + global top-k
        s1 = sweep_on(model, tok, directions, mu_a, mu_b, win, side, sp["val"], wiki, base_ppl)

        # S2: greedy joint (rank by solo effect, add if improves joint within budget)
        ranked = [r["l"] for r in sorted(solo, key=lambda r: -r["d"]) if r["dppl"] <= BETA][:12]
        L2, best_pick = [], base_pick
        for l in ranked:
            cand = sorted(L2 + [l])
            sc = score_edges(model, directions, mu_a, mu_b, cand, COMPONENTS)
            sel = selection_from_ranking(rank_weight_indices(sc, 0.01), GREEDY_TESTFRAC)
            with pruned_weights(model, sel):
                pi, _ = pick_rate(model, tok, sp["val"], side)
                ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
            if pi < best_pick - GREEDY_MARGIN and (ppl - base_ppl) / base_ppl <= BETA:
                L2, best_pick = cand, pi
        s2 = sweep_on(model, tok, directions, mu_a, mu_b, L2, side, sp["val"], wiki, base_ppl) if L2 else None

        # S3: best-first greedy (all layers w/ solo ppl<=beta; no top-k, no fixed order/margin)
        pool = [r["l"] for r in solo if r["dppl"] <= BETA]
        L3, cur = [], base_pick
        while True:
            bl, bpi = None, cur
            for l in pool:
                if l in L3:
                    continue
                cand = sorted(L3 + [l])
                sc = score_edges(model, directions, mu_a, mu_b, cand, COMPONENTS)
                sel = selection_from_ranking(rank_weight_indices(sc, 0.01), GREEDY_TESTFRAC)
                with pruned_weights(model, sel):
                    pi, _ = pick_rate(model, tok, sp["val"], side)
                    ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
                if (ppl - base_ppl) / base_ppl <= BETA and pi < bpi:
                    bl, bpi = l, pi
            if bl is not None and bpi < cur - 0.005:
                L3.append(bl); cur = bpi
            else:
                break
        s3 = sweep_on(model, tok, directions, mu_a, mu_b, L3, side, sp["val"], wiki, base_ppl) if L3 else None

        report["behaviors"][name] = {"side": side, "baseline": base_pick,
                                     "S0_solo_ELS": {"L": L0, "result": s0},
                                     "S1_window_topk": {"win": [win[0], win[-1]], "result": s1},
                                     "S2_greedy_joint": {"L": L2, "result": s2},
                                     "S3_bestfirst": {"L": L3, "result": s3}}
        def fmt(r):
            return (f"{base_pick:.3f}->{r['pick']:.3f} @{r['ppl_delta']:+.1%} (s={r['sparsity']:.3%})"
                    if r and r["pick"] < base_pick else "no removal / L* empty")
        print(f"  S0 solo-ELS   L*={L0 if L0 else 'EMPTY'}  {fmt(s0) if s0 else 'L* EMPTY -> diffuse'}")
        print(f"  S2 greedy      L*={L2 if L2 else 'EMPTY'}  {fmt(s2) if s2 else 'no layer improved'}")
        print(f"  S3 best-first L*={L3 if L3 else 'EMPTY'}  {fmt(s3) if s3 else 'no layer improved'}\n")

    RESULTS.mkdir(exist_ok=True)
    tag = MODEL_ID.split("/")[-1].replace(".", "").lower()
    (RESULTS / f"blade_layer_select_{tag}.json").write_text(json.dumps(report, indent=2))
    print(f"saved results/blade_layer_select_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
