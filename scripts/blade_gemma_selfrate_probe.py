"""Probe: can best-first BLADE remove Gemma-3-4B's weak self-rate-highly
preference (base ~0.58), which the MIN_BIAS=0.10 gate normally skips?

Runs the exact S3 best-first path from blade_layer_select on one behavior,
gate bypassed. Saves to a separate JSON (does NOT overwrite the full run).
"""
import json
import os
from pathlib import Path

os.environ.setdefault("BLADE_MODEL", "google/gemma-3-4b-it")

import ttsafety.behaviors as B  # noqa: E402
from ttsafety.behaviors import (collect_span_input_moments, extract_direction,  # noqa: E402
                                fetch_ab, make_splits, pick_rate, score_edges)
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl  # noqa: E402
from ttsafety.hooks import get_decoder_layers  # noqa: E402
from ttsafety.models import env_info, load_model  # noqa: E402
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,  # noqa: E402
                                   selection_from_ranking)
from blade_layer_select import (sweep_on, layer_hist, COMPONENTS, SCREEN_FRAC,  # noqa: E402
                                BETA, GREEDY_TESTFRAC, PPL_TOKENS)

DATA = Path("data"); RESULTS = Path("results")
MODEL_ID = os.environ["BLADE_MODEL"]
NAME = "self-rate-highly"


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eot = "<end_of_turn>"  # gemma
    n = len(get_decoder_layers(model))
    all_layers = list(range(n))
    wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)

    rows = fetch_ab(NAME, DATA / "behaviors")
    sp = make_splits(rows)
    rm, _ = pick_rate(model, tok, sp["val"], "matching")
    side = "matching" if rm >= 0.5 else "not_matching"
    base_pick, _ = pick_rate(model, tok, sp["val"], side)
    print(f"{MODEL_ID} | {NAME} | side={side} | base_pick={base_pick:.3f} "
          f"|bias|={abs(rm-0.5):.3f} | base_ppl={base_ppl:.2f}  (GATE BYPASSED)",
          flush=True)

    directions = extract_direction(model, tok, sp["train"], side, eot=eot)
    mu_a = collect_span_input_moments(model, tok, sp["train"], side, all_layers,
                                      COMPONENTS, eot=eot)
    mu_b = collect_span_input_moments(model, tok, sp["train"],
                                      "not_matching" if side == "matching" else "matching",
                                      all_layers, COMPONENTS, eot=eot)

    # solo screen -> ppl-feasible pool (same as blade_layer_select S3)
    solo = []
    for l in all_layers:
        sc = score_edges(model, directions, mu_a, mu_b, [l], COMPONENTS)
        sel = selection_from_ranking(rank_weight_indices(sc, 0.01), SCREEN_FRAC)
        with pruned_weights(model, sel):
            pi, _ = pick_rate(model, tok, sp["val"], side)
            ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
        solo.append({"l": l, "d": base_pick - pi, "dppl": (ppl - base_ppl) / base_ppl})
    pool = [r["l"] for r in solo if r["dppl"] <= BETA]

    # S3 best-first
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
            print(f"  + L{bl} -> joint pick {bpi:.3f}", flush=True)
        else:
            break

    s3 = (sweep_on(model, tok, directions, mu_a, mu_b, L3, side, sp["val"], wiki, base_ppl)
          if L3 else None)
    print(f"pool={len(pool)} | S3 L*={L3 if L3 else 'EMPTY'}", flush=True)
    if s3 and s3["pick"] < base_pick:
        print(f"RESULT: {base_pick:.3f} -> {s3['pick']:.3f} @ {s3['ppl_delta']:+.1%} ppl "
              f"(s={s3['sparsity']:.3%}, layers={s3['layers']})", flush=True)
    else:
        print("RESULT: no within-budget removal (weak preference not localizable)", flush=True)

    out = {"model": MODEL_ID, "behavior": NAME, "side": side, "baseline": base_pick,
           "bias": abs(rm - 0.5), "gate_bypassed": True, "base_ppl": base_ppl,
           "pool_size": len(pool), "S3_bestfirst": {"L": L3, "result": s3},
           "env": env_info()}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "blade_gemma_selfrate_probe.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("saved results/blade_gemma_selfrate_probe.json", flush=True)


if __name__ == "__main__":
    main()
