"""PAPER-NATIVE panel for Deleting Refusal. Same leakage-safe protocol as blade_weaken_ood.py
(plain gen, select on harmful_val to minimize refusal s.t. C4 rel-ppl <= BETA, freeze, report on
held-out HarmBench + WikiText), but each baseline run closer to its ORIGINAL paper:
  BLADE  : unchanged (residual-writer surface is BLADE's definition)
  Wei    : SNIP set-difference over ALL decoder linear matrices (q,k,v,o,gate,up,down; native surface)
  Arditi : SINGLE global refusal direction, full strength s=1.0, ablated from all residual writers
           (vs. the controlled panel's per-layer directions at budget-forced s=0.5)
  CAA    : wide sweep -- all candidate layers, wide negative-coef range (rule out under-tuning)
  ActAdd : wide sweep -- layers x wide negative-coef range
Emits results/blade_weaken_native.json. Methods with a magnitude knob are held to the ppl budget;
if none is eligible the least-refusing point is reported and flagged over-budget.
"""
import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

import torch

from ttsafety.data import load_jsonl
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_nll
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.sycophancy import score_edges
from ttsafety.weight_edit import orthogonalized_weights
from ttsafety.weight_prune import matrixwise_set_difference, rank_weight_indices, selection_from_ranking
from blade_refusal_amplify import scaled_weights
from blade_refusal_els import last_token_moments
from blade_steering_baselines import (_generate, build_methods, rel_ppl, wilson,
                                      GEN_TOKENS, BETA, PPL_TOKENS, C4_TOKENS)
from blade_weaken_ood import plain_refusal

DATA = Path("data"); RESULTS = Path("results"); SCORES = DATA / "weight_scores"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"; N_DIR = 200; SCREEN = 0.03
WEI_STRUCTS = [(0.0005, 0.0005), (0.001, 0.001), (0.002, 0.002), (0.005, 0.005)]
ARDITI_SRC_LAYERS = [8, 10, 12, 14, 16]           # single-direction source layer (native selection)
CAA_LAYERS = [6, 8, 10, 12, 14, 16, 18, 20]; CAA_COEFS = [-0.2, -0.35, -0.5, -0.7, -1.0, -1.5, -2.0]
ACTADD_LAYERS = [8, 12, 16]; ACTADD_COEFS = [-16.0, -32.0, -64.0, -128.0]
OUTNAME = "blade_weaken_native"


def select(rows, key_ref="val_refusal", key_ppl="c4_relppl"):
    elig = [r for r in rows if r["eligible"]]
    pool = elig if elig else rows
    star = min(pool, key=lambda r: (r[key_ref], r[key_ppl]))
    return star, bool(elig)


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    harmful_val = [r["instruction"] for r in load_jsonl(DATA / "harmful_val.jsonl")]
    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")]
    wiki = load_wikitext_text(); c4 = load_c4_text()
    all_layers = list(range(len(get_decoder_layers(model))))

    dirs = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:192])
    muH = last_token_moments(model, tok, harmful[:N_DIR], all_layers, "both", chat_wrap)
    muU = last_token_moments(model, tok, harmless[:N_DIR], all_layers, "both", chat_wrap)
    methods, _ = build_methods(model, tok)

    @contextmanager
    def noop(mode):
        yield

    def weight_ctx(sel, factor):
        @contextmanager
        def cm(mode):
            with scaled_weights(model, sel, factor):
                yield
        return cm

    def arditi_native_ctx(src_layer, strength=1.0):
        r = dirs[src_layer]                       # single global direction, all writers
        @contextmanager
        def cm(mode):
            with orthogonalized_weights(model, r, all_layers, "both", strength=strength):
                yield
        return cm

    base_nll_wiki, n_scored = teacher_forced_nll(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_nll_c4, _ = teacher_forced_nll(model, tok, c4, max_tokens=C4_TOKENS)
    base_val = plain_refusal(model, tok, harmful_val, noop)
    base_ood = plain_refusal(model, tok, ood, noop)
    print(f"base PLAIN refusal: val {base_val:.3f} OOD {base_ood:.3f}", flush=True)

    def sweep(cands):
        rows = []
        for label, ctx, cfg in cands:
            r = plain_refusal(model, tok, harmful_val, ctx)
            pp = rel_ppl(model, tok, c4, base_nll_c4, ctx, C4_TOKENS)
            rows.append({"label": label, "cfg": cfg, "ctx": ctx, "val_refusal": r,
                         "c4_relppl": pp, "eligible": pp <= BETA})
            print(f"  {label:28} val {r:.3f} c4Δppl {pp:+.1%} {'OK' if pp<=BETA else 'x'}", flush=True)
        return rows

    frozen = {}
    # BLADE (native = residual-writer surface)
    print("== BLADE ==", flush=True)
    blade_sel = selection_from_ranking(rank_weight_indices(
        score_edges(model, dirs, muH, muU, [12], "both"), SCREEN), 0.005)
    frozen["blade"] = sweep([("BLADE L12 rho.005 a0", weight_ctx(blade_sel, 0.0),
                              {"L": [12], "rho": 0.005, "n_edges": sum(len(v) for v in blade_sel.values())})])

    # Wei native (all linear matrices)
    print("== Wei native (all linear) ==", flush=True)
    ws = torch.load(SCORES / "wei_safety_snip_native.pt", map_location="cpu", weights_only=False)["scores"]
    wu = torch.load(SCORES / "wei_utility_snip_native.pt", map_location="cpu", weights_only=False)["scores"]
    wei_c = []
    for q, p in WEI_STRUCTS:
        sel = matrixwise_set_difference(ws, wu, safety_fraction=q, utility_fraction=p)
        wei_c.append((f"Wei-native q{q} p{p}", weight_ctx(sel, 0.0),
                      {"q": q, "p": p, "n_edges": sum(len(v) for v in sel.values()),
                       "n_matrices": len(sel)}))
    frozen["wei"] = sweep(wei_c)

    # Arditi native (single global dir, s=1.0, all writers)
    print("== Arditi native (single dir, s=1) ==", flush=True)
    frozen["arditi"] = sweep([(f"Arditi-native src L{L} s1.0", arditi_native_ctx(L, 1.0),
                               {"src_layer": L, "strength": 1.0}) for L in ARDITI_SRC_LAYERS])

    # CAA native (wide sweep)
    print("== CAA native (wide) ==", flush=True)
    frozen["caa"] = sweep([(f"CAA L{L} c{c}", methods["caa"]["ctx"](L, c), {"layer": L, "coef": c})
                           for L in CAA_LAYERS for c in CAA_COEFS])

    # ActAdd native (wide sweep)
    print("== ActAdd native (wide) ==", flush=True)
    frozen["actadd"] = sweep([(f"ActAdd L{L} c{c}", methods["actadd"]["ctx"](L, c), {"layer": L, "coef": c})
                              for L in ACTADD_LAYERS for c in ACTADD_COEFS])

    # freeze best per method
    chosen = {}
    for m, rows in frozen.items():
        star, had_elig = select(rows)
        chosen[m] = {"label": star["label"], "cfg": star["cfg"], "val_refusal": star["val_refusal"],
                     "c4_relppl": star["c4_relppl"], "within_budget": had_elig, "ctx": star["ctx"]}
    sha = hashlib.sha256(json.dumps({m: c["cfg"] for m, c in chosen.items()},
                                    sort_keys=True, default=str).encode()).hexdigest()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{OUTNAME}.frozen.json").write_text(json.dumps(
        {m: {k: c[k] for k in ("label", "cfg", "within_budget")} for m, c in chosen.items()}
        | {"sha256": sha}, indent=2, default=str))
    print(f"== FROZEN (sha {sha[:12]}) ==", flush=True)
    for m, c in chosen.items():
        print(f"   {m}: {c['label']} {'' if c['within_budget'] else '(over budget)'}", flush=True)

    # PHASE 2 report on held-out HarmBench + WikiText
    report = {}
    for m, c in chosen.items():
        ctx = c["ctx"]
        n_ref = round(plain_refusal(model, tok, ood, ctx) * len(ood))
        ood_ref = n_ref / len(ood)
        relppl = rel_ppl(model, tok, wiki, base_nll_wiki, ctx, PPL_TOKENS)
        lo, hi = wilson(n_ref, len(ood))
        report[m] = {"label": c["label"], "cfg": c["cfg"], "ood_refusal": ood_ref, "ci": [lo, hi],
                     "wiki_relppl": relppl, "budget_ok": relppl <= BETA,
                     "within_budget_selection": c["within_budget"]}
        print(f"  report {c['label']:28} OOD refusal {ood_ref:.3f} [{lo:.2f},{hi:.2f}] "
              f"Δppl {relppl:+.1%} {'' if relppl<=BETA else '(>budget)'}", flush=True)

    out = {"model": MODEL_ID, "panel": "paper_native", "setting": "plain_gen_no_prefill",
           "base": {"val_refusal": base_val, "ood_refusal": base_ood, "wiki_nll": base_nll_wiki},
           "frozen_sha256": sha,
           "val_search": {m: [{k: r[k] for k in ("label", "cfg", "val_refusal", "c4_relppl", "eligible")}
                              for r in rows] for m, rows in frozen.items()},
           "frozen": {m: {k: c[k] for k in ("label", "cfg", "val_refusal", "c4_relppl", "within_budget")}
                      for m, c in chosen.items()},
           "report": report, "env": env_info()}
    (RESULTS / f"{OUTNAME}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"saved results/{OUTNAME}.json", flush=True)


if __name__ == "__main__":
    main()
