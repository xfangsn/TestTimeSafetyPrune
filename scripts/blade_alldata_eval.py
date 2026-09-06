"""One BLADE(rho=.005, alpha=2.5) edit, ELS run ONCE, then generated on all three benchmarks
(SelfAware + FalseQA via load_ood, and SimpleQA) so the SAME L* is used everywhere. Purpose: compare the
edit's ELS layer selection across environments (RTX 5090 vs Hazel L40S, which pick different L*) on a
consistent pipeline, and pick the better L*. Saves base + BLADE generations for a downstream Opus judge
(ACT for SA/FQ, correct/incorrect/not-attempted for SimpleQA).
Env: BLADE_MODEL, SQ_N (SimpleQA n), CAP (SA/FQ per cell), OUT_TAG, L_STAR (optional pin).
Usage: BLADE_MODEL=Qwen/Qwen3-8B SQ_N=400 CAP=40 OUT_TAG=_5090 python scripts/blade_alldata_eval.py
"""
import csv, json, os
from pathlib import Path
import torch
import ttsafety.extract as EX
import ttsafety.generate as GEN
from ttsafety.eval import load_c4_text, teacher_forced_ppl
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.behaviors import bestfirst_layers, solo_layer_pool
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking
from ttsafety.generate import generate_texts
from blade_refusal_amplify import scaled_weights
from blade_epistemic_els import qwen_wrap, last_token_moments, is_unc, COMPONENTS, PPL_TOKENS
from blade_epistemic_p0 import split_3way, score_fn_for
import blade_epistemic_p0 as P0
from baseline_dola_iti import gen_plain

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"; DATA = ROOT / "data"
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")
SQ_N = int(os.environ.get("SQ_N", "400")); CAP = int(os.environ.get("CAP", "40"))
RHO = 0.005; ALPHA = 2.5
L_STAR_ENV = os.environ.get("L_STAR", "")


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


def load_simpleqa(n):
    p = os.environ.get("TTS_SIMPLEQA_FILE") or str(DATA / "abstention" / "simpleqa_test.csv")
    return [{"dataset": "simpleqa", "gold": r["answer"], "question": r["problem"]}
            for r in list(csv.DictReader(open(p)))[:n]]


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
    if L_STAR_ENV:
        L_star = [int(x) for x in L_STAR_ENV.split(",")]
        print(f"L*={L_star} (pinned)", flush=True)
    else:
        base_sel = measure()[0]
        pool = solo_layer_pool(model, directions, muUNC, muCERT, all_layers, COMPONENTS, ppl_now, base_ppl,
                               screen_frac=RHO, beta=0.05, score_fn=sfn)
        L_star = bestfirst_layers(model, directions, muUNC, muCERT, pool, COMPONENTS, measure, base_sel,
                                  base_ppl, beta=0.05, eps=0.005, test_frac=RHO, score_fn=sfn)
        print(f"L*={L_star}", flush=True)
    rk = rank_weight_indices(sfn(model, directions, muUNC, muCERT, L_star, COMPONENTS), RHO + 0.01)
    selw = selection_from_ranking(rk, RHO)

    items = load_ood() + load_simpleqa(SQ_N)
    prompts = [it["question"] for it in items]
    print(f"generating base+blade on {len(prompts)} prompts (SA/FQ {CAP}/cell + SimpleQA {SQ_N}) ...", flush=True)
    base_g = gen_plain(model, tok, prompts); print("  base done", flush=True)
    with scaled_weights(model, selw, ALPHA):
        blade_ppl = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
        blade_g = gen_plain(model, tok, prompts)
    print(f"  blade done (C4 Δppl {(blade_ppl-base_ppl)/base_ppl:+.2%})", flush=True)

    report = {"model": MODEL_ID, "L_star": L_star, "rho": RHO, "alpha": ALPHA,
              "blade_ppl_delta_c4": (blade_ppl - base_ppl) / base_ppl, "cap": CAP, "sq_n": SQ_N,
              "env": env_info(), "items": []}
    for i, it in enumerate(items):
        report["items"].append({**it, "base": base_g[i], "blade": blade_g[i]})
    outp = RESULTS / f"blade_alldata{os.environ.get('OUT_TAG','')}_qwen3-8b.json"
    outp.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"saved {outp}", flush=True)


if __name__ == "__main__":
    main()
