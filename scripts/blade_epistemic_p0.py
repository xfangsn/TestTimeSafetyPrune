"""P0a core — re-run REMOVE with (i) a genuinely UNTOUCHED eval split (ELS selects on its OWN set),
(ii) paired McNemar (base vs edited are paired), (iii) saved per-item generations for a blind SEMANTIC
judge (the lexical marker conflates 'X is fictional' [correct] with hedging). Fixes the two biggest
threats to the headline: selection-on-test + metric conflation. thinking-OFF throughout (matches the edit).

3-way entity-disjoint split per family: dir_train (direction+moments) / els_select (ELS metric) /
untouched_eval (report). Output: results/epistemic_p0_<tag>.json with per-item base/remove texts +
lexical McNemar. Then run blade_epistemic_judge (subagent) on the saved generations.

Env: BLADE_MODEL, BLADE_G=1. Usage: BLADE_MODEL=Qwen/Qwen3-8B BLADE_G=1 .venv/bin/python scripts/blade_epistemic_p0.py
"""
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
try:
    from scipy.stats import binomtest
except Exception:  # scipy may be absent on air-gapped Hazel venv
    binomtest = None

import ttsafety.extract as EX
import ttsafety.generate as GEN
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_ppl
from ttsafety.extract import extract_refusal_direction
from ttsafety.generate import generate_texts
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.sycophancy import score_edges, score_edges_g
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.behaviors import bestfirst_layers, solo_layer_pool
from ttsafety.weight_prune import pruned_weights, rank_weight_indices, selection_from_ranking
from blade_epistemic_els import (qwen_wrap, last_token_moments, is_unc, COMPONENTS, PPL_TOKENS,
                                 GEN_TOKENS, _med_pos)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BLADE_G = os.environ.get("BLADE_G", "") == "1"
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")
SCREEN_FRAC = float(os.environ.get("BLADE_SCREEN_FRAC", "0.005"))
BETA = float(os.environ.get("BLADE_BETA", "0.05"))
EPS = float(os.environ.get("BLADE_EPS", "0.005"))
Q_GLOBAL = None


def split_3way(rows, seed=0, fr=(0.45, 0.25)):
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    tr, sel, ev = [], [], []
    rng = random.Random(seed)
    for fam, fr_rows in by_fam.items():
        ents = sorted({r["entity"] for r in fr_rows}); rng.shuffle(ents)
        n1 = max(1, int(fr[0] * len(ents))); n2 = max(1, int(fr[1] * len(ents)))
        a, b = set(ents[:n1]), set(ents[n1:n1 + n2])
        for r in fr_rows:
            (tr if r["entity"] in a else sel if r["entity"] in b else ev).append(r)
    return tr, sel, ev


def gen_outputs(model, tok, prompts):
    return generate_texts(model, tok, list(prompts), max_new_tokens=GEN_TOKENS, batch_size=16)


def score_fn_for(model, directions, mu_a, mu_b, all_layers):
    if not BLADE_G:
        return score_edges
    lam = _med_pos(score_edges(model, directions, mu_a, mu_b, all_layers, COMPONENTS)) / _med_pos(Q_GLOBAL)

    def sfn(m, d, a, b, layers, comp):
        S = score_edges_g(m, d, a, b, layers, comp, Q=Q_GLOBAL, lam=lam, abstain=True)
        return {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in S.items()}
    return sfn


def mcnemar(base_flags, edit_flags):
    # paired binary; b = base-unc & edit-not, c = base-not & edit-unc
    b = sum(1 for x, y in zip(base_flags, edit_flags) if x and not y)
    c = sum(1 for x, y in zip(base_flags, edit_flags) if y and not x)
    if b + c == 0:
        p = 1.0
    elif binomtest is not None:
        p = binomtest(min(b, c), b + c, 0.5).pvalue
    else:  # exact two-sided sign test without scipy
        from math import comb
        n, k = b + c, min(b, c)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"b_base_only": b, "c_edit_only": c, "p_exact": p}


def main():
    global Q_GLOBAL
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    EX.chat_wrap = qwen_wrap
    GEN.chat_wrap = qwen_wrap
    all_layers = list(range(len(get_decoder_layers(model))))

    rows = json.loads((RESULTS / "epistemic_pairs_v2.json").read_text())["rows"]
    tr, sel, ev = split_3way(rows)
    unc_tr = [r["question"] for r in tr if r["label"] == 1]
    cert_tr = [r["question"] for r in tr if r["label"] == 0]
    unc_sel = [r["question"] for r in sel if r["label"] == 1]
    ev_unc = [r for r in ev if r["label"] == 1]
    ev_cert = [r for r in ev if r["label"] == 0]
    print(f"split: dir_train {len(tr)} | els_select {len(sel)} | untouched_eval {len(ev)} "
          f"({len(ev_unc)}unc/{len(ev_cert)}cert)", flush=True)

    c4 = load_c4_text(); wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    base_ppl_wiki = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)

    directions = extract_refusal_direction(model, tok, unc_tr, cert_tr)
    muUNC = last_token_moments(model, tok, unc_tr, all_layers, COMPONENTS, qwen_wrap)
    muCERT = last_token_moments(model, tok, cert_tr, all_layers, COMPONENTS, qwen_wrap)
    if BLADE_G:
        print("Q ...", flush=True)
        Q_GLOBAL, _ = collect_c4_generic_importance(model, tok, all_layers, COMPONENTS, text=c4,
                                                    seqlen=2048, batch_size=2, mode="g1scalar", max_tokens=65536)
    sfn = score_fn_for(model, directions, muUNC, muCERT, all_layers)

    def ppl_now():
        return teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)

    def measure():                                   # ELS metric on els_select (NOT eval)
        outs = gen_outputs(model, tok, unc_sel)
        return sum(is_unc(o) for o in outs) / max(len(outs), 1), ppl_now()

    base_sel_hedge = measure()[0]
    print(f"base hedge on els_select unanswerable {base_sel_hedge:.3f}", flush=True)
    pool = solo_layer_pool(model, directions, muUNC, muCERT, all_layers, COMPONENTS,
                           ppl_now, base_ppl, screen_frac=SCREEN_FRAC, beta=BETA, score_fn=sfn)
    L_star = bestfirst_layers(model, directions, muUNC, muCERT, pool, COMPONENTS,
                              measure, base_sel_hedge, base_ppl, beta=BETA, eps=EPS,
                              test_frac=SCREEN_FRAC, score_fn=sfn)
    print(f"pool={len(pool)} -> L*={L_star}", flush=True)

    # baseline generations on UNTOUCHED eval
    base_unc_out = gen_outputs(model, tok, [r["question"] for r in ev_unc])
    base_cert_out = gen_outputs(model, tok, [r["question"] for r in ev_cert])

    report = {"model": MODEL_ID, "blade_g": BLADE_G, "L_star": L_star,
              "split": {"dir_train": len(tr), "els_select": len(sel), "untouched_eval": len(ev)},
              "base_ppl_c4": base_ppl, "base_ppl_wiki": base_ppl_wiki, "env": env_info(),
              "note": "denominator: rho = fraction of eligible residual-writer entries in L_star",
              "sweep": [], "items": []}

    if L_star:
        scores = sfn(model, directions, muUNC, muCERT, L_star, COMPONENTS)
        rk = rank_weight_indices(scores, 0.03)
        best_items = None
        for frac in [0.002, 0.005, 0.02]:
            selw = selection_from_ranking(rk, frac)
            n = sum(int(v.numel()) for v in selw.values())
            with pruned_weights(model, selw):
                ru = gen_outputs(model, tok, [r["question"] for r in ev_unc])
                rc = gen_outputs(model, tok, [r["question"] for r in ev_cert])
                pc = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
            bu = [is_unc(o) for o in base_unc_out]; eu = [is_unc(o) for o in ru]
            mc = mcnemar(bu, eu)
            row = {"sparsity": frac, "n_edges": n,
                   "hedge_unans_base": sum(bu) / len(bu), "hedge_unans_remove": sum(eu) / len(eu),
                   "hedge_known_base": sum(is_unc(o) for o in base_cert_out) / len(base_cert_out),
                   "hedge_known_remove": sum(is_unc(o) for o in rc) / len(rc),
                   "mcnemar_unans": mc, "ppl_delta_c4": (pc - base_ppl) / base_ppl}
            report["sweep"].append(row)
            print(f"  REMOVE s={frac:.2%} n={n} hedge unans {row['hedge_unans_base']:.2f}->"
                  f"{row['hedge_unans_remove']:.2f} (McNemar b={mc['b_base_only']} c={mc['c_edit_only']} "
                  f"p={mc['p_exact']:.1e}) known {row['hedge_known_base']:.2f}->{row['hedge_known_remove']:.2f} "
                  f"Δppl {row['ppl_delta_c4']:+.2%}", flush=True)
            if frac == 0.005:
                best_items = (ru, rc)

        # save per-item generations at rho=0.005 for the blind semantic judge
        ru, rc = best_items
        for r, b, e in zip(ev_unc, base_unc_out, ru):
            report["items"].append({"family": r["family"], "gold_label": "unanswerable",
                                    "question": r["question"], "base": b, "remove": e})
        for r, b, e in zip(ev_cert, base_cert_out, rc):
            report["items"].append({"family": r["family"], "gold_label": "answerable",
                                    "question": r["question"], "base": b, "remove": e})

    RESULTS.mkdir(exist_ok=True)
    tag = MODEL_ID.split("/")[-1].replace(".", "").lower() + ("_bladeg" if BLADE_G else "")
    (RESULTS / f"epistemic_p0_{tag}.json").write_text(json.dumps(report, indent=2))
    print(f"saved results/epistemic_p0_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
