"""Amplify (POWERED, untouched split, paired McNemar) — bidirectional INTENSITY control of closed-book
parametric-knowledge abstention. Same 3-way entity split + L* as blade_epistemic_p0.py. On the UNTOUCHED
eval, does AMPLIFY raise appropriate abstention on unanswerable prompts (= LESS hallucination) via the
SAME L* support (opposite operation from REMOVE), WITHOUT over-abstaining on answerable/known prompts?
Two amplify ops: raw-alphaW (scale selected weights by alpha>1) and suppressor-removal ([-r.W.Δμ]_+ at L*).
Saves generations at the best condition for the blind judge.

Env: BLADE_MODEL, BLADE_G=1. Usage: BLADE_MODEL=Qwen/Qwen3-8B BLADE_G=1 .venv/bin/python scripts/blade_epistemic_amplify_v2.py
"""
import json
import os
from pathlib import Path

import torch

import ttsafety.extract as EX
import ttsafety.generate as GEN
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_ppl
from ttsafety.extract import extract_refusal_direction
from ttsafety.generate import generate_texts
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.behaviors import bestfirst_layers, solo_layer_pool
from ttsafety.weight_prune import pruned_weights, rank_weight_indices, selection_from_ranking
from blade_refusal_amplify import scaled_weights
from blade_epistemic_els import qwen_wrap, last_token_moments, is_unc, COMPONENTS, PPL_TOKENS, GEN_TOKENS
from blade_epistemic_p0 import split_3way, score_fn_for, mcnemar
import blade_epistemic_p0 as P0

RESULTS = Path(__file__).resolve().parent.parent / "results"
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")
BLADE_G = os.environ.get("BLADE_G", "") == "1"


def gen(model, tok, prompts):
    return generate_texts(model, tok, list(prompts), max_new_tokens=GEN_TOKENS, batch_size=16)


def main():
    P0.BLADE_G = BLADE_G
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    EX.chat_wrap = qwen_wrap; GEN.chat_wrap = qwen_wrap
    all_layers = list(range(len(get_decoder_layers(model))))

    rows = json.loads((RESULTS / "epistemic_pairs_v2.json").read_text())["rows"]
    tr, sel, ev = split_3way(rows)
    unc_tr = [r["question"] for r in tr if r["label"] == 1]
    cert_tr = [r["question"] for r in tr if r["label"] == 0]
    unc_sel = [r["question"] for r in sel if r["label"] == 1]
    ev_unc = [r for r in ev if r["label"] == 1]
    ev_cert = [r for r in ev if r["label"] == 0]

    c4 = load_c4_text(); wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    directions = extract_refusal_direction(model, tok, unc_tr, cert_tr)
    dirs_neg = {l: -directions[l] for l in directions}
    muUNC = last_token_moments(model, tok, unc_tr, all_layers, COMPONENTS, qwen_wrap)
    muCERT = last_token_moments(model, tok, cert_tr, all_layers, COMPONENTS, qwen_wrap)
    if BLADE_G:
        print("Q ...", flush=True)
        P0.Q_GLOBAL, _ = collect_c4_generic_importance(model, tok, all_layers, COMPONENTS, text=c4,
                                                       seqlen=2048, batch_size=2, mode="g1scalar", max_tokens=65536)
    sfn = score_fn_for(model, directions, muUNC, muCERT, all_layers)
    sfn_neg = score_fn_for(model, dirs_neg, muUNC, muCERT, all_layers)  # suppressor scoring (flip dir)

    def ppl_now():
        return teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)

    def measure():
        outs = gen(model, tok, unc_sel)
        return sum(is_unc(o) for o in outs) / max(len(outs), 1), ppl_now()

    base_sel = measure()[0]
    pool = solo_layer_pool(model, directions, muUNC, muCERT, all_layers, COMPONENTS,
                           ppl_now, base_ppl, screen_frac=0.005, beta=0.05, score_fn=sfn)
    L_star = bestfirst_layers(model, directions, muUNC, muCERT, pool, COMPONENTS,
                              measure, base_sel, base_ppl, beta=0.05, eps=0.005, test_frac=0.005, score_fn=sfn)
    print(f"L*={L_star}", flush=True)

    base_unc = gen(model, tok, [r["question"] for r in ev_unc])
    base_cert = gen(model, tok, [r["question"] for r in ev_cert])
    b_unc = [is_unc(o) for o in base_unc]; b_cert = [is_unc(o) for o in base_cert]
    print(f"UNTOUCHED base: abstain unans {sum(b_unc)/len(b_unc):.2f} known {sum(b_cert)/len(b_cert):.2f}",
          flush=True)

    scores = sfn(model, directions, muUNC, muCERT, L_star, COMPONENTS)
    rk = rank_weight_indices(scores, 0.05)
    supp = sfn_neg(model, dirs_neg, muUNC, muCERT, L_star, COMPONENTS)
    rk_supp = rank_weight_indices(supp, 0.05)

    report = {"model": MODEL_ID, "L_star": L_star, "env": env_info(),
              "base_abstain_unans": sum(b_unc) / len(b_unc), "base_abstain_known": sum(b_cert) / len(b_cert),
              "n_eval": [len(ev_unc), len(ev_cert)], "conditions": [], "items": []}
    best = None

    def run(label, cm):
        with cm:
            au = gen(model, tok, [r["question"] for r in ev_unc])
            ak = gen(model, tok, [r["question"] for r in ev_cert])
            pc = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
        eu = [is_unc(o) for o in au]; ek = [is_unc(o) for o in ak]
        mc = mcnemar(b_unc, eu)  # discordant on unanswerable
        row = {"label": label, "abstain_unans": sum(eu) / len(eu), "abstain_known": sum(ek) / len(ek),
               "mcnemar_unans_vs_base": mc, "ppl_delta_c4": (pc - base_ppl) / base_ppl}
        report["conditions"].append(row)
        print(f"  {label:20s} abstain unans {report['base_abstain_unans']:.2f}->{row['abstain_unans']:.2f} "
              f"(McNemar b={mc['b_base_only']} c={mc['c_edit_only']} p={mc['p_exact']:.1e}) "
              f"known {row['abstain_known']:.2f} Δppl {row['ppl_delta_c4']:+.2%}", flush=True)
        return au, ak

    print("== AMPLIFY: raw-alphaW (scale selected weights) ==", flush=True)
    for a in (1.25, 1.5, 2.0):
        sel_w = selection_from_ranking(rk, 0.005)
        out = run(f"alphaW_a{a}", scaled_weights(model, sel_w, a))
        if a == 1.5:
            best = out
    print("== AMPLIFY: suppressor-removal at L* ==", flush=True)
    for rho in (0.005, 0.02):
        sel_s = selection_from_ranking(rk_supp, rho)
        out = run(f"suppress_r{rho}", pruned_weights(model, sel_s))
        if rho == 0.02:
            best = out

    au, ak = best
    for r, b, e in zip(ev_unc, base_unc, au):
        report["items"].append({"gold": "unanswerable", "question": r["question"], "base": b, "amplify": e})
    for r, b, e in zip(ev_cert, base_cert, ak):
        report["items"].append({"gold": "answerable", "question": r["question"], "base": b, "amplify": e})

    tag = MODEL_ID.split("/")[-1].replace(".", "").lower() + ("_bladeg" if BLADE_G else "")
    (RESULTS / f"epistemic_amplify_v2_{tag}.json").write_text(json.dumps(report, indent=2))
    print(f"saved results/epistemic_amplify_v2_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
