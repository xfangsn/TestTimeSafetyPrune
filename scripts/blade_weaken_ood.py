"""WEAKENING direction: flip each OOD-refusal *strengthening* method into its canonical *removal*
form and measure how much it JAILBREAKS held-out HarmBench (lower refusal = more weakened), under a
capability budget. Mirror of the strengthening comparison but:
  - measured on PLAIN generation (no prefill attack): base plain-gen refusal is ~1.0, giving full
    headroom to weaken (the prefill setting floors base refusal at ~0.1).
  - objective = MINIMIZE refusal subject to WikiText/C4 rel-ppl <= BETA (capability is the collateral,
    not over-refusal). Select on in-dist val, freeze+hash, report on held-out HarmBench + WikiText.

Each method in its faithful removal form:
  BLADE  : zero the ELS-selected mask (alpha=0)                    [forward-only weight attribution]
  Wei    : zero the SNIP set-difference safety mask (alpha=0)      [gradient weight attribution; the
                                                                    paper's own jailbreak-by-pruning]
  Arditi : directional ablation of the refusal direction (weights) [(I-rr^T)W over residual writers]
  CAA    : subtract the mean-diff refusal direction (coef<0)
  ActAdd : subtract the intent-to-refuse contrast vector (coef<0)
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
from ttsafety.judge import is_refusal
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.sycophancy import score_edges
from ttsafety.weight_edit import orthogonalized_weights
from ttsafety.weight_prune import matrixwise_set_difference, rank_weight_indices, selection_from_ranking
from blade_refusal_amplify import scaled_weights
from blade_refusal_els import last_token_moments
import ttsafety.steering_methods as SM
from blade_steering_baselines import (_generate, build_methods, rel_ppl, wilson,
                                      GEN_TOKENS, BETA, PPL_TOKENS, C4_TOKENS)

DATA = Path("data"); RESULTS = Path("results"); SCORES = DATA / "weight_scores"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
N_DIR = 200; SCREEN = 0.03
# weakening grids (negative coefs / removal masks); brackets the ppl-budget transition per method
BLADE_STRUCTS = [([12], 0.005), ([14], 0.002), ([14, 0], 0.001), ([14, 8, 2], 0.0005)]
WEI_STRUCTS = [(0.0005, 0.0005), (0.001, 0.001), (0.002, 0.002), (0.005, 0.005)]
ABLATE_STRENGTHS = [0.25, 0.5, 1.0]
CAA_LAYERS = [8, 12, 16]; CAA_COEFS = [-0.2, -0.35, -0.5, -0.7]
ACTADD_LAYERS = [8, 12, 16]; ACTADD_COEFS = [-16.0, -32.0, -64.0]
OUTNAME = "blade_weaken_ood"


def plain_refusal(model, tok, prompts, ctx):
    with ctx("plain_gen"):
        outs = _generate(model, tok, [chat_wrap(tok, p) for p in prompts], GEN_TOKENS)
    return sum(is_refusal(o) for o in outs) / len(outs)


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

    # directions / masks
    dirs = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:192])
    muH = last_token_moments(model, tok, harmful[:N_DIR], all_layers, "both", chat_wrap)
    muU = last_token_moments(model, tok, harmless[:N_DIR], all_layers, "both", chat_wrap)
    wei_safety = torch.load(SCORES / "wei_safety_snip.pt", map_location="cpu", weights_only=False)["scores"]
    wei_util = torch.load(SCORES / "wei_utility_snip.pt", map_location="cpu", weights_only=False)["scores"]
    methods, _ = build_methods(model, tok)   # provides caa/actadd directions + ctx builders

    @contextmanager
    def noop(mode):
        yield

    def weight_ctx(sel, factor):
        @contextmanager
        def cm(mode):
            with scaled_weights(model, sel, factor):
                yield
        return cm

    def ablate_ctx(strength):
        @contextmanager
        def cm(mode):
            with orthogonalized_weights(model, dirs, all_layers, "both", strength=strength):
                yield
        return cm

    # ---- candidate table: (method, label, ctx, cfg) ----
    cands = []
    for L, rho in BLADE_STRUCTS:
        scores = score_edges(model, dirs, muH, muU, L, "both")
        sel = selection_from_ranking(rank_weight_indices(scores, SCREEN), rho)
        cands.append(("blade", f"BLADE L{'-'.join(map(str,L))} rho{rho} a0", weight_ctx(sel, 0.0),
                      {"L": L, "rho": rho, "alpha": 0.0, "n_edges": sum(len(v) for v in sel.values())}))
    for q, p in WEI_STRUCTS:
        sel = matrixwise_set_difference(wei_safety, wei_util, safety_fraction=q, utility_fraction=p)
        cands.append(("wei", f"Wei q{q} p{p} a0", weight_ctx(sel, 0.0),
                      {"q": q, "p": p, "alpha": 0.0, "n_edges": sum(len(v) for v in sel.values())}))
    for s in ABLATE_STRENGTHS:
        cands.append(("arditi_ablate", f"Arditi ablate s={s}", ablate_ctx(s), {"strength": s}))
    for Ly in CAA_LAYERS:
        for c in CAA_COEFS:
            cands.append(("caa", f"CAA L{Ly} c{c}", methods["caa"]["ctx"](Ly, c), {"layer": Ly, "coef": c}))
    for Ly in ACTADD_LAYERS:
        for c in ACTADD_COEFS:
            cands.append(("actadd", f"ActAdd L{Ly} c{c}", methods["actadd"]["ctx"](Ly, c), {"layer": Ly, "coef": c}))

    base_nll_wiki, n_scored = teacher_forced_nll(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_nll_c4, _ = teacher_forced_nll(model, tok, c4, max_tokens=C4_TOKENS)
    base_val = plain_refusal(model, tok, harmful_val, noop)
    base_ood = plain_refusal(model, tok, ood, noop)
    print(f"base PLAIN refusal: val {base_val:.3f} OOD {base_ood:.3f}", flush=True)

    # PHASE 1: sweep all candidates on VAL only; eligible iff C4 rel-ppl <= BETA
    search = []
    for method, label, ctx, cfg in cands:
        r = plain_refusal(model, tok, harmful_val, ctx)
        pp = rel_ppl(model, tok, c4, base_nll_c4, ctx, C4_TOKENS)
        elig = pp <= BETA
        search.append({"method": method, "label": label, "cfg": cfg, "val_refusal": r,
                       "c4_relppl": pp, "eligible": elig})
        print(f"  {label:26} val {r:.3f} c4Δppl {pp:+.1%} {'OK' if elig else 'x'}", flush=True)

    # per method: pick MIN val refusal among eligible (most weakened within budget)
    frozen = {}
    for method in ["blade", "wei", "arditi_ablate", "caa", "actadd"]:
        elig = [row for row in search if row["method"] == method and row["eligible"]]
        if elig:
            star = min(elig, key=lambda r: (r["val_refusal"], r["c4_relppl"]))
            frozen[method] = {"label": star["label"], "cfg": star["cfg"],
                              "val_refusal": star["val_refusal"], "c4_relppl": star["c4_relppl"]}
        else:
            frozen[method] = None
    sha = hashlib.sha256(json.dumps({m: (f and f["cfg"]) for m, f in frozen.items()},
                                    sort_keys=True).encode()).hexdigest()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{OUTNAME}.frozen.json").write_text(json.dumps({"frozen": frozen, "sha256": sha}, indent=2))
    print(f"== FROZEN before HarmBench (sha {sha[:12]}) ==", flush=True)
    for m, f in frozen.items():
        print(f"   {m}: {f['label'] if f else 'NONE eligible'}", flush=True)

    # PHASE 2: report frozen configs on held-out HarmBench (plain gen) + WikiText ppl
    ctx_by_label = {label: ctx for _, label, ctx, _ in cands}
    report = {}
    for method, f in frozen.items():
        if f is None:
            report[method] = None; continue
        ctx = ctx_by_label[f["label"]]
        n_ref = round(plain_refusal(model, tok, ood, ctx) * len(ood))
        ood_ref = n_ref / len(ood)
        relppl = rel_ppl(model, tok, wiki, base_nll_wiki, ctx, PPL_TOKENS)
        lo, hi = wilson(n_ref, len(ood))
        report[method] = {"label": f["label"], "cfg": f["cfg"], "ood_refusal": ood_ref, "ci": [lo, hi],
                          "asr": 1 - ood_ref, "wiki_relppl": relppl,
                          "budget_ok": relppl <= BETA}
        print(f"  report {f['label']:26} OOD refusal {ood_ref:.3f} [{lo:.2f},{hi:.2f}] "
              f"(ASR {1-ood_ref:.3f}) Δppl {relppl:+.1%}", flush=True)

    out = {"model": MODEL_ID, "direction": "weaken_ood_refusal", "setting": "plain_gen_no_prefill",
           "base": {"val_refusal": base_val, "ood_refusal": base_ood, "wiki_nll": base_nll_wiki,
                    "scored_tokens": n_scored},
           "frozen_sha256": sha, "val_search": search, "frozen": frozen, "report": report,
           "env": env_info()}
    (RESULTS / f"{OUTNAME}.json").write_text(json.dumps(out, indent=2))
    print(f"saved results/{OUTNAME}.json", flush=True)


if __name__ == "__main__":
    main()
