"""Rigorous version: select the BLADE amplify configuration (L*, rho, components, alpha) ON
IN-DIST VALIDATION under the same locked rule as the steering baselines, freeze it, then report
on HarmBench. This makes the "a sparser/deeper BLADE config strengthens OOD refusal within the
benign budget" claim leakage-safe (the exploratory blade_config_sweep.py measured on HarmBench).
"""
import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

from ttsafety.data import load_jsonl
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_nll
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.sycophancy import score_edges
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking
from blade_refusal_els import last_token_moments
from blade_refusal_amplify import scaled_weights
from blade_steering_baselines import (prefill_refusal, over_refusal, rel_ppl, wilson,
                                      L_BENIGN, BETA, PPL_TOKENS, C4_TOKENS, XSTEST_SEARCH)

DATA = Path("data"); RESULTS = Path("results")
N_DIR = 200; SCREEN = 0.03
# FAITHFUL candidate structures = exactly what BLADE's best-first ELS selects at each rho
# (L12@rho.005, L14@rho.002, [14,0]@rho.001, [14,8,2]@rho.0005); rho is BLADE's own hyperparam,
# each L* is ELS-selected (not hand-picked). alpha (amplify) selected on val per structure.
STRUCTS = [([12], 0.005, "both"), ([14], 0.002, "both"), ([14, 0], 0.001, "both"),
           ([14, 8, 2], 0.0005, "both")]
ALPHAS = [1.3, 1.5, 2.0, 3.0]
OUTNAME = "blade_amplify_els_configs"


def main():
    model, tok = load_model("meta-llama/Llama-3.2-3B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    harmful_val = [r["instruction"] for r in load_jsonl(DATA / "harmful_val.jsonl")]
    xstest = [r["instruction"] for r in load_jsonl(DATA / "xstest_safe.jsonl")]
    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")]
    wiki = load_wikitext_text(); c4 = load_c4_text()

    all_layers = list(range(len(get_decoder_layers(model))))
    dirs = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:192])
    muH = last_token_moments(model, tok, harmful[:N_DIR], all_layers, "both", chat_wrap)
    muU = last_token_moments(model, tok, harmless[:N_DIR], all_layers, "both", chat_wrap)

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

    # PHASE 1: select (struct, alpha) on VALIDATION only
    search = []
    sels = {}
    for L, rho, comp in STRUCTS:
        scores = score_edges(model, dirs, muH, muU, L, comp)
        sel = selection_from_ranking(rank_weight_indices(scores, SCREEN), rho)
        key = f"L{'-'.join(map(str, L))}_r{rho}_{comp}"
        sels[key] = sel
        for a in ALPHAS:
            c = ctx(sel, a)
            r = prefill_refusal(model, tok, harmful_val, c)
            b = over_refusal(model, tok, xstest[:XSTEST_SEARCH], c)
            p = rel_ppl(model, tok, c4, base_nll_c4, c, C4_TOKENS)
            elig = (b <= base_benign_sub + L_BENIGN) and (p <= BETA)
            search.append({"key": key, "L": L, "rho": rho, "components": comp, "alpha": a,
                           "n_edges": sum(len(v) for v in sel.values()),
                           "val_refusal": r, "benign": b, "c4_relppl": p, "eligible": elig})
            print(f"  {key:22} a={a:<4g} val {r:.3f} benign {b:.3f} c4Δppl {p:+.1%} "
                  f"{'OK' if elig else 'x'}", flush=True)
    elig = [row for row in search if row["eligible"]]
    star = (max(elig, key=lambda r: (r["val_refusal"], -r["c4_relppl"])) if elig else None)
    if star is None:
        print("no eligible config; freezing base", flush=True)
        star = {"key": "base", "L": [12], "rho": 0.005, "components": "both", "alpha": 1.0}
    print(f"  -> frozen {star['key']} alpha={star['alpha']} (val {star.get('val_refusal', base_val):.3f})", flush=True)

    frozen = {"key": star["key"], "L": star["L"], "rho": star["rho"],
              "components": star["components"], "alpha": star["alpha"]}
    sha = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{OUTNAME}.frozen.json").write_text(
        json.dumps({"frozen": frozen, "sha256": sha}, indent=2))
    print(f"== FROZEN before HarmBench: {frozen} sha256 {sha} ==", flush=True)

    # PHASE 2: report the frozen config's alpha grid on HarmBench
    rows = []
    if star["key"] == "base":
        report_alphas = [1.0]
        sel = None
    else:
        sel = sels[star["key"]]
        report_alphas = ALPHAS
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
                     "report_eligible_wiki_full_xstest": report_elig, "is_astar": (a == star["alpha"])})
        print(f"  report a={a:<4g} OOD {ood_ref:.3f} [{lo:.2f},{hi:.2f}] Δppl {relppl:+.1%} "
              f"benign {benign_full:.3f} {'OK' if report_elig else 'x'}{'  <-*' if a == star['alpha'] else ''}",
              flush=True)

    out = {"model": "meta-llama/Llama-3.2-3B-Instruct", "frozen": frozen, "frozen_sha256": sha,
           "base": {"val_refusal": base_val, "ood_refusal": base_ood, "benign": base_benign_full,
                    "wiki_nll": base_nll_wiki, "scored_tokens": n_scored},
           "val_search": search, "report": rows, "env": env_info()}
    (RESULTS / f"{OUTNAME}.json").write_text(json.dumps(out, indent=2))
    print("saved results/blade_amplify_configselect.json", flush=True)


if __name__ == "__main__":
    main()
