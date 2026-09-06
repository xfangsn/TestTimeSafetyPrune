"""SimpleQA (Wei et al. 2024, arXiv:2411.04368) generations for the uncertainty_method_cmp configs:
base, ITI c=4, ITI c=6, BLADE (rho=.005, alpha=2.5). Short fact-seeking Qs with known answers; the
official grading buckets a response as CORRECT / INCORRECT / NOT_ATTEMPTED, so amplify (more abstention)
should trade INCORRECT (hallucination) for NOT_ATTEMPTED. We only GENERATE here (Qwen3-8B, thinking-off,
same qwen_wrap/greedy/128-tok path for every config); grading is a separate Opus pass.
Env: BLADE_MODEL, SQ_N (num questions). Data: $TTS_SIMPLEQA_FILE or data/abstention/simpleqa_test.csv.
Usage: BLADE_MODEL=Qwen/Qwen3-8B SQ_N=400 python scripts/simpleqa_eval.py
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
from baseline_dola_iti import head_acts, proj_std, gen_plain, gen_iti

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"; DATA = ROOT / "data"
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")
SQ_N = int(os.environ.get("SQ_N", "400"))
ITI_K = 48
BLADE_RHO = 0.005; BLADE_ALPHA = 2.5
ITI_ALPHAS = [4.0, 6.0]


def load_simpleqa(n):
    p = os.environ.get("TTS_SIMPLEQA_FILE") or str(DATA / "abstention" / "simpleqa_test.csv")
    rows = list(csv.DictReader(open(p)))[:n]
    return [{"question": r["problem"], "gold": r["answer"]} for r in rows]


def main():
    P0.BLADE_G = True
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    EX.chat_wrap = qwen_wrap; GEN.chat_wrap = qwen_wrap
    all_layers = list(range(len(get_decoder_layers(model))))
    blocks = get_decoder_layers(model)
    cfg = model.config
    nh, hd = cfg.num_attention_heads, getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

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
    print(f"BLADE L*={L_star}", flush=True)
    rk = rank_weight_indices(sfn(model, directions, muUNC, muCERT, L_star, COMPONENTS), BLADE_RHO + 0.01)
    selw = selection_from_ranking(rk, BLADE_RHO)

    # ITI heads (fit on the same certain/uncertain contrast, last prompt token)
    mu_u, _ = head_acts(model, tok, unc_tr + unc_sel, blocks, nh, hd)
    mu_c, _ = head_acts(model, tok, cert_tr + [r["question"] for r in sel if r["label"] == 0], blocks, nh, hd)
    diffs = {(i, h): (mu_u[i][h] - mu_c[i][h]) for i in range(len(blocks)) for h in range(nh)}
    ranked = sorted(diffs, key=lambda k: -diffs[k].norm().item())[:ITI_K]
    dirs = {k: diffs[k] / diffs[k].norm().clamp_min(1e-6) for k in ranked}
    sigma = proj_std(model, tok, unc_tr + cert_tr, blocks, nh, hd, dirs)

    def add_vec(alpha):
        add = {i: torch.zeros(nh * hd) for i in range(len(blocks))}
        for (i, h) in ranked:
            add[i].view(nh, hd)[h] = alpha * sigma[(i, h)] * dirs[(i, h)]
        return add

    items = load_simpleqa(SQ_N); prompts = [it["question"] for it in items]
    print(f"SimpleQA n={len(prompts)}; generating ...", flush=True)
    gens = {}
    gens["base"] = gen_plain(model, tok, prompts); print("  base done", flush=True)
    with scaled_weights(model, selw, BLADE_ALPHA):
        gens[f"blade_r{BLADE_RHO}_a{BLADE_ALPHA}"] = gen_plain(model, tok, prompts)
    print("  blade done", flush=True)
    for a in ITI_ALPHAS:
        gens[f"iti_a{a}"] = gen_iti(model, tok, prompts, blocks, add_vec(a)); print(f"  iti_a{a} done", flush=True)

    report = {"model": MODEL_ID, "n": len(prompts), "blade_L_star": L_star, "blade_rho": BLADE_RHO,
              "blade_alpha": BLADE_ALPHA, "iti_k": ITI_K, "iti_alphas": ITI_ALPHAS,
              "conditions": list(gens), "env": env_info(), "items": []}
    for i, it in enumerate(items):
        report["items"].append({**it, **{c: gens[c][i] for c in gens}})
    slug = "qwen3-8b" if MODEL_ID == "Qwen/Qwen3-8B" else MODEL_ID.split("/")[-1].lower()
    outp = RESULTS / f"simpleqa_{slug}.json"
    outp.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"saved {outp}", flush=True)


if __name__ == "__main__":
    main()
