"""ρ × α grid sweep on ONE fixed fit/L*/mask-ranking (clean sweep; only ρ and α vary). α=0 = remove
(prune top-ρ), α>1 = amplify (raw-αW ×α on top-ρ). On the OOD set (SelfAware+FalseQA, CAP/cell) we save
generations + per-condition C4 Δppl + a cheap LEXICAL degeneration rate (repetition), so we can map the
landscape without a judge, then Opus-judge only the non-degenerate cells.
Env: BLADE_MODEL, RHOS, ALPHAS, CAP. Usage: BLADE_MODEL=Qwen/Qwen3-8B .venv/bin/python scripts/blade_rho_sweep.py
"""
import csv, json, os
from pathlib import Path

import torch

import ttsafety.extract as EX
import ttsafety.generate as GEN
from ttsafety.eval import load_c4_text, teacher_forced_ppl
from ttsafety.extract import extract_refusal_direction
from ttsafety.generate import generate_texts
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.behaviors import bestfirst_layers, solo_layer_pool
from ttsafety.weight_prune import pruned_weights, rank_weight_indices, selection_from_ranking
from blade_refusal_amplify import scaled_weights
from blade_epistemic_els import qwen_wrap, last_token_moments, is_unc, COMPONENTS, PPL_TOKENS
from blade_epistemic_p0 import split_3way, score_fn_for
import blade_epistemic_p0 as P0

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"; DATA = ROOT / "data"
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")
RHOS = [float(x) for x in os.environ.get("RHOS", "0.002,0.005,0.01,0.02,0.05").split(",")]
ALPHAS = [float(x) for x in os.environ.get("ALPHAS", "0,2,4").split(",")]
CAP = int(os.environ.get("CAP", "40"))
GEN_TOK = 128


def load_ood():
    fq = list(csv.DictReader(open(DATA / "abstention" / "falseqa_test.csv")))
    out = []
    for q in [r["question"] for r in fq if r["label"] == "1"][:CAP]:
        out.append({"dataset": "falseqa", "gold": "false_premise", "question": q})
    for q in [r["question"] for r in fq if r["label"] == "0"][:CAP]:
        out.append({"dataset": "falseqa", "gold": "true_premise", "question": q})
    ex = json.loads((DATA / "abstention" / "SelfAware.json").read_text())["example"]
    for q in [x["question"] for x in ex if not x["answerable"]][:CAP]:
        out.append({"dataset": "selfaware", "gold": "unanswerable", "question": q})
    for q in [x["question"] for x in ex if x["answerable"]][:CAP]:
        out.append({"dataset": "selfaware", "gold": "answerable", "question": q})
    return out


def rep_score(t):
    w = t.split()
    if len(w) < 8:
        return 1.0 if len(w) < 3 else 0.0
    g = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / len(g)


def degen_rate(outs):
    return sum(rep_score(o) > 0.5 for o in outs) / max(len(outs), 1)


def main():
    P0.BLADE_G = True
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    EX.chat_wrap = qwen_wrap; GEN.chat_wrap = qwen_wrap
    all_layers = list(range(len(get_decoder_layers(model))))
    rows = json.loads((RESULTS / "epistemic_pairs_v2.json").read_text())["rows"]
    tr, sel, _ = split_3way(rows)
    unc_tr = [r["question"] for r in tr if r["label"] == 1]; cert_tr = [r["question"] for r in tr if r["label"] == 0]
    unc_sel = [r["question"] for r in sel if r["label"] == 1]
    c4 = load_c4_text(); base_ppl = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    directions = extract_refusal_direction(model, tok, unc_tr, cert_tr)
    muUNC = last_token_moments(model, tok, unc_tr, all_layers, COMPONENTS, qwen_wrap)
    muCERT = last_token_moments(model, tok, cert_tr, all_layers, COMPONENTS, qwen_wrap)
    print("Q ...", flush=True)
    P0.Q_GLOBAL, _ = collect_c4_generic_importance(model, tok, all_layers, COMPONENTS, text=c4, seqlen=2048,
                                                   batch_size=2, mode="g1scalar", max_tokens=65536)
    sfn = score_fn_for(model, directions, muUNC, muCERT, all_layers)

    def ppl_now(): return teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    def measure():
        o = generate_texts(model, tok, unc_sel, max_new_tokens=64, batch_size=16)
        return sum(is_unc(x) for x in o) / max(len(o), 1), ppl_now()
    base_sel = measure()[0]
    pool = solo_layer_pool(model, directions, muUNC, muCERT, all_layers, COMPONENTS, ppl_now, base_ppl,
                           screen_frac=0.005, beta=0.05, score_fn=sfn)
    L_star = bestfirst_layers(model, directions, muUNC, muCERT, pool, COMPONENTS, measure, base_sel,
                              base_ppl, beta=0.05, eps=0.005, test_frac=0.005, score_fn=sfn)
    print(f"FIXED L*={L_star}", flush=True)
    rk = rank_weight_indices(sfn(model, directions, muUNC, muCERT, L_star, COMPONENTS), max(RHOS) + 0.01)

    items = load_ood(); prompts = [it["question"] for it in items]
    def gen(): return generate_texts(model, tok, prompts, max_new_tokens=GEN_TOK, batch_size=16)
    report = {"model": MODEL_ID, "L_star": L_star, "base_ppl_c4": base_ppl, "cap": CAP,
              "rhos": RHOS, "alphas": ALPHAS, "env": env_info(), "grid": [], "items": [dict(it) for it in items]}
    for rho in RHOS:
        selw = selection_from_ranking(rk, rho)
        n = sum(int(v.numel()) for v in selw.values())
        for a in ALPHAS:
            cm = pruned_weights(model, selw) if a == 0 else scaled_weights(model, selw, a)
            cond = f"r{rho}_a{a}"
            with cm:
                outs = gen(); pc = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
            dg = degen_rate(outs)
            report["grid"].append({"cond": cond, "rho": rho, "alpha": a, "n_edges": n,
                                   "degen": dg, "ppl_delta_c4": (pc - base_ppl) / base_ppl})
            for rec, o in zip(report["items"], outs):
                rec[cond] = o
            print(f"  {cond:14} n={n:7d} degen {dg:.2f}  Δppl {(pc-base_ppl)/base_ppl:+.2%}", flush=True)
    tag = os.environ.get("OUT_TAG", "")
    outp = RESULTS / f"blade_rho_sweep{tag}_qwen3-8b.json"
    outp.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"saved {outp}", flush=True)


if __name__ == "__main__":
    main()
