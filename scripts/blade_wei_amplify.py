"""Add Wei et al. (2024, arXiv:2402.05162) to the OOD-refusal (amplify) comparison.

Wei's method identifies safety-critical weights with an absolute-SNIP set difference
S_safety(q) \\ S_utility(p), where I(W,x)=|W * grad_W L(x)| (gradients + labeled objective).
We AMPLIFY that gradient-selected mask by the SAME scalar factor alpha and evaluate under the
SAME leakage-safe protocol used for BLADE amplify (blade_amplify_configselect.py) and the steering
baselines: select (q, p, alpha) on IN-DIST validation under the locked benign+ppl budget, freeze +
hash BEFORE any HarmBench read, then report on held-out HarmBench / full XSTest / WikiText.

This isolates the single variable of interest: BLADE's forward-only score vs Wei's gradient-SNIP
score as the weight-attribution method, with the amplify intervention held identical.
"""
import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

import torch

from ttsafety.data import load_jsonl
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_nll
from ttsafety.models import env_info, load_model
from ttsafety.weight_prune import matrixwise_set_difference
from blade_refusal_amplify import scaled_weights
from blade_steering_baselines import (prefill_refusal, over_refusal, rel_ppl, wilson,
                                      L_BENIGN, BETA, PPL_TOKENS, C4_TOKENS, XSTEST_SEARCH)

DATA = Path("data"); RESULTS = Path("results")
SCORES = DATA / "weight_scores"
# Wei set-difference structures (safety_fraction q, utility_fraction p). p=q is the paper's typical
# setting (matches the downstream wei_p*_q* runs); the asymmetric row keeps more safety-specific
# weight. Mask sizes are reported so sparsity is transparent and bracket BLADE's rho=0.002 mask.
STRUCTS = [(0.0005, 0.0005), (0.001, 0.001), (0.002, 0.002), (0.005, 0.005), (0.002, 0.0005)]
# amplify grid: BLADE amplify used [1.3,1.5,2.0,3.0]; 4.0 added to give Wei's mask its best shot at
# reaching BLADE's OOD level (noted at report time).
ALPHAS = [1.3, 1.5, 2.0, 3.0, 4.0]
OUTNAME = "blade_wei_amplify"


def load_wei_scores():
    safety = torch.load(SCORES / "wei_safety_snip.pt", map_location="cpu", weights_only=False)
    utility = torch.load(SCORES / "wei_utility_snip.pt", map_location="cpu", weights_only=False)
    return safety["scores"], utility["scores"]


def main():
    model, tok = load_model("meta-llama/Llama-3.2-3B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    harmful_val = [r["instruction"] for r in load_jsonl(DATA / "harmful_val.jsonl")]
    xstest = [r["instruction"] for r in load_jsonl(DATA / "xstest_safe.jsonl")]
    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")]
    wiki = load_wikitext_text(); c4 = load_c4_text()

    safety_scores, utility_scores = load_wei_scores()

    @contextmanager
    def noop(mode):
        yield

    def ctx(sel, factor):
        @contextmanager
        def cm(mode):
            with scaled_weights(model, sel, factor):
                yield
        return cm

    base_nll_wiki, n_scored = teacher_forced_nll(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_nll_c4, _ = teacher_forced_nll(model, tok, c4, max_tokens=C4_TOKENS)
    base_val = prefill_refusal(model, tok, harmful_val, noop)
    base_ood = prefill_refusal(model, tok, ood, noop)
    base_benign_sub = over_refusal(model, tok, xstest[:XSTEST_SEARCH], noop)
    base_benign_full = over_refusal(model, tok, xstest, noop)
    print(f"base: val {base_val:.3f} OOD {base_ood:.3f} benign {base_benign_full:.3f}", flush=True)

    # PHASE 1: select (q, p, alpha) on VALIDATION only
    search = []
    sels = {}
    for q, p in STRUCTS:
        sel = matrixwise_set_difference(safety_scores, utility_scores,
                                        safety_fraction=q, utility_fraction=p)
        key = f"wei_q{q}_p{p}"
        sels[key] = sel
        n_edges = sum(len(v) for v in sel.values())
        for a in ALPHAS:
            c = ctx(sel, a)
            r = prefill_refusal(model, tok, harmful_val, c)
            b = over_refusal(model, tok, xstest[:XSTEST_SEARCH], c)
            pp = rel_ppl(model, tok, c4, base_nll_c4, c, C4_TOKENS)
            elig = (b <= base_benign_sub + L_BENIGN) and (pp <= BETA)
            search.append({"key": key, "q": q, "p": p, "alpha": a, "n_edges": n_edges,
                           "val_refusal": r, "benign": b, "c4_relppl": pp, "eligible": elig})
            print(f"  {key:20} n={n_edges:>8} a={a:<4g} val {r:.3f} benign {b:.3f} "
                  f"c4Δppl {pp:+.1%} {'OK' if elig else 'x'}", flush=True)
    elig = [row for row in search if row["eligible"]]
    star = (max(elig, key=lambda r: (r["val_refusal"], -r["c4_relppl"])) if elig else None)
    if star is None:
        print("no eligible config; freezing base", flush=True)
        star = {"key": "base", "q": None, "p": None, "alpha": 1.0}
    print(f"  -> frozen {star['key']} alpha={star['alpha']} "
          f"(val {star.get('val_refusal', base_val):.3f})", flush=True)

    frozen = {"key": star["key"], "q": star["q"], "p": star["p"], "alpha": star["alpha"]}
    sha = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{OUTNAME}.frozen.json").write_text(
        json.dumps({"frozen": frozen, "sha256": sha}, indent=2))
    print(f"== FROZEN before HarmBench: {frozen} sha256 {sha} ==", flush=True)

    # PHASE 2: report the frozen config's alpha grid on HarmBench
    rows = []
    if star["key"] == "base":
        report_alphas = [1.0]; sel = None
    else:
        sel = sels[star["key"]]; report_alphas = ALPHAS
    for a in report_alphas:
        c = ctx(sel, a) if sel is not None else noop
        n_ref = round(prefill_refusal(model, tok, ood, c) * len(ood))
        ood_ref = n_ref / len(ood)
        relppl = rel_ppl(model, tok, wiki, base_nll_wiki, c, PPL_TOKENS)
        benign_full = over_refusal(model, tok, xstest, c)
        report_elig = (benign_full <= base_benign_full + L_BENIGN) and (relppl <= BETA)
        lo, hi = wilson(n_ref, len(ood))
        rows.append({"alpha": a, "ood_refusal": ood_ref, "ci": [lo, hi], "wiki_relppl": relppl,
                     "wiki_dnll": math.log(1 + relppl), "benign_full": benign_full,
                     "report_eligible_wiki_full_xstest": report_elig,
                     "is_astar": (a == star["alpha"])})
        print(f"  report a={a:<4g} OOD {ood_ref:.3f} [{lo:.2f},{hi:.2f}] Δppl {relppl:+.1%} "
              f"benign {benign_full:.3f} {'OK' if report_elig else 'x'}"
              f"{'  <-*' if a == star['alpha'] else ''}", flush=True)

    out = {"model": "meta-llama/Llama-3.2-3B-Instruct", "method": "wei2024_snip_set_difference_amplify",
           "frozen": frozen, "frozen_sha256": sha,
           "base": {"val_refusal": base_val, "ood_refusal": base_ood, "benign": base_benign_full,
                    "wiki_nll": base_nll_wiki, "scored_tokens": n_scored},
           "structs": [{"q": q, "p": p} for q, p in STRUCTS], "alphas": ALPHAS,
           "val_search": search, "report": rows, "env": env_info()}
    (RESULTS / f"{OUTNAME}.json").write_text(json.dumps(out, indent=2))
    print(f"saved results/{OUTNAME}.json", flush=True)


if __name__ == "__main__":
    main()
