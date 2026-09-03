"""Decisive controls (both reviewers' top picks) for the OOD-selection question.
BLADE edge score + best-first are UNCHANGED; we only vary the candidate pool /
run diagnostics.

A. solo-OOD correlation: for each pool layer, SOLO-prune at test_frac and measure
   the OOD sycophancy reduction. Correlate gain_AB and gain_OPEN (from S4) against
   gain_OOD across layers. Answers: does ANY in-dist solo signal predict OOD?

B. drop-L0 best-first @ beta=10%: run the UNCHANGED best-first on pools
   {full, minus{0}, minus{0,1,2}, gain_OPEN>=0 (S4-filter tau=0)}. Answers:
   is "just avoid the shortcut layer(s)" enough to reach S3-beta5's OOD 0.25, and
   does the open-format filter add anything beyond that?

C. random-pool control @ beta=10%: K random pools (same size as the drop-{0,1,2}
   result) through the same best-first. Answers: is any OOD gain just "smaller pool
   => less pruning", independent of layer content?

Held fixed vs the plain-ELS beta=10% baseline: data, ppl budget, test_frac, eps.
"""
import json
import random
from pathlib import Path
from statistics import mean, pstdev

import torch

from ttsafety.behaviors import (bestfirst_layers, collect_span_input_moments,
                                extract_direction, fetch_ab, pick_rate, score_edges)
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,
                                   selection_from_ranking)
from ood_sycophancy_eval import load_ood, eval_ood

DATA = Path("data"); RESULTS = Path("results")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
EOT = "<|eot_id|>"; COMPONENTS = "both"; PPL_TOKENS = 5000
BETA = 0.10; TEST_FRAC = 0.005; EPS = 0.005; SEED = 0
N_OOD = 300; RHOS = [0.002, 0.005]; N_RANDOM = 5


def pearson(x, y):
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    return pearson(ranks(x), ranks(y))


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    all_layers = list(range(len(get_decoder_layers(model))))
    wiki = load_wikitext_text()
    rows = fetch_ab("sycophancy", DATA / "behaviors")
    random.Random(SEED).shuffle(rows)
    train = rows[:600]
    val = rows[600:750]
    exs = load_ood(n=N_OOD)

    rate_m, _ = pick_rate(model, tok, val, "matching")
    side = "matching" if rate_m >= 0.5 else "not_matching"
    other = "not_matching" if side == "matching" else "matching"
    print(f"side={side}", flush=True)

    dF = extract_direction(model, tok, train, side, eot=EOT)
    aF = collect_span_input_moments(model, tok, train, side, all_layers, COMPONENTS, eot=EOT)
    bF = collect_span_input_moments(model, tok, train, other, all_layers, COMPONENTS, eot=EOT)

    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_ab, _ = pick_rate(model, tok, val, side)
    base_ood = eval_ood(model, tok, exs)
    print(f"BASE  A/B {base_ab:.3f}  OOD syco {base_ood['sycophancy']:.3f} acc {base_ood['accuracy']:.3f}", flush=True)

    def sel_at(cand, frac):
        sc = score_edges(model, dF, aF, bF, cand, COMPONENTS)
        return selection_from_ranking(rank_weight_indices(sc, max(frac, 0.01)), frac)

    def ppl_ok(cand):
        with pruned_weights(model, sel_at(cand, TEST_FRAC)):
            ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
        return (ppl - base_ppl) / base_ppl <= BETA

    pool = [l for l in all_layers if ppl_ok([l])]
    print(f"pool = {pool}", flush=True)

    # ---- read S4 solo gains ----
    s4 = json.loads((RESULTS / "s4_format_dro_sycophancy.json").read_text())
    gAB = {r["layer"]: r["gain_AB"] for r in s4["solo"]}
    gOP = {r["layer"]: r["gain_OPEN"] for r in s4["solo"]}

    # ---- A. solo-OOD correlation ----
    print("\n=== A. solo-OOD gain per layer ===", flush=True)
    solo_ood = {}
    for l in pool:
        with pruned_weights(model, sel_at([l], TEST_FRAC)):
            o = eval_ood(model, tok, exs)
        solo_ood[l] = base_ood["sycophancy"] - o["sycophancy"]
        print(f"  L{l:>2}  gain_AB {gAB.get(l,0):+.3f}  gain_OPEN {gOP.get(l,0):+.3f}  "
              f"gain_OOD {solo_ood[l]:+.3f}", flush=True)
    layers_c = [l for l in pool if l in gAB]
    xAB = [gAB[l] for l in layers_c]; xOP = [gOP[l] for l in layers_c]
    yOOD = [solo_ood[l] for l in layers_c]
    corr = {"pearson_AB_OOD": pearson(xAB, yOOD), "spearman_AB_OOD": spearman(xAB, yOOD),
            "pearson_OPEN_OOD": pearson(xOP, yOOD), "spearman_OPEN_OOD": spearman(xOP, yOOD)}
    print(f"  corr gain_AB~gain_OOD  pearson {corr['pearson_AB_OOD']:+.3f} spearman {corr['spearman_AB_OOD']:+.3f}", flush=True)
    print(f"  corr gain_OPEN~gain_OOD pearson {corr['pearson_OPEN_OOD']:+.3f} spearman {corr['spearman_OPEN_OOD']:+.3f}", flush=True)

    # ---- helper: run unchanged best-first on a given pool, then rho-sweep OOD ----
    def run_pool(tag, cand_pool):
        def measure():
            return pick_rate(model, tok, val, side)[0], teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
        L = bestfirst_layers(model, dF, aF, bF, cand_pool, COMPONENTS,
                             measure, base_ab, base_ppl, beta=BETA, eps=EPS, test_frac=TEST_FRAC)
        out = {"tag": tag, "pool": cand_pool, "L_star": L, "rows": []}
        if L:
            sc = score_edges(model, dF, aF, bF, L, COMPONENTS)
            rk = rank_weight_indices(sc, max(0.03, max(RHOS)))
            for rho in RHOS:
                sel = selection_from_ranking(rk, rho)
                n = sum(int(v.numel()) for v in sel.values())
                with pruned_weights(model, sel):
                    o = eval_ood(model, tok, exs)
                    ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
                dppl = (ppl - base_ppl) / base_ppl
                out["rows"].append({"rho": rho, "n_edges": n, "ood_sycophancy": o["sycophancy"],
                                    "ood_accuracy": o["accuracy"], "ppl_delta": dppl})
        print(f"[{tag}] L*={L}", flush=True)
        for r in out["rows"]:
            print(f"    rho={r['rho']} OOD syco {r['ood_sycophancy']:.3f} acc {r['ood_accuracy']:.3f} "
                  f"Δppl {r['ppl_delta']:+.1%} ({r['n_edges']:,} edges)", flush=True)
        return out

    # ---- B. drop-L0 variants ----
    print("\n=== B. drop-shortcut best-first @ beta=10% ===", flush=True)
    filt = [l for l in pool if gOP.get(l, -1) >= 0]  # S4-filter tau=0
    variants = [
        ("full", pool),
        ("drop_L0", [l for l in pool if l != 0]),
        ("drop_L012", [l for l in pool if l not in (0, 1, 2)]),
        ("filter_open>=0", filt),
    ]
    B = [run_pool(t, p) for t, p in variants]

    # ---- C. random-pool control (size = |drop_L012 result|, min 4) ----
    ref = next((b for b in B if b["tag"] == "drop_L012"), None)
    size = max(4, len(ref["L_star"]) if ref and ref["L_star"] else 4)
    print(f"\n=== C. random-pool control (size {size}, {N_RANDOM} draws) ===", flush=True)
    C = []
    for k in range(N_RANDOM):
        rp = random.Random(1000 + k).sample(pool, min(size, len(pool)))
        C.append(run_pool(f"random{k}", sorted(rp)))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "s4_controls.json").write_text(json.dumps(
        {"model": MODEL_ID, "beta": BETA, "pool": pool, "base_ppl": base_ppl,
         "base_ab": base_ab, "base_ood": base_ood, "n_ood": N_OOD,
         "solo_ood": solo_ood, "gain_AB": gAB, "gain_OPEN": gOP, "correlation": corr,
         "drop_variants": B, "random_pools": C, "env": env_info()}, indent=2))
    print("\nsaved results/s4_controls.json", flush=True)


if __name__ == "__main__":
    main()
