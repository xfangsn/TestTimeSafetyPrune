"""Tuning run for the method comparison: extra BLADE amplify configs (raw-alphaW x3/x4, suppressor-removal
rho0.02) + ITI alpha=4, generated on the SAME OOD prompts (SelfAware+FalseQA, CAP 70/cell) as ood_run /
baseline_dola_iti, for Opus judging. Env: BLADE_MODEL, BLADE_G=1."""
import json
import os
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
from ttsafety.weight_prune import pruned_weights, rank_weight_indices, selection_from_ranking
from blade_refusal_amplify import scaled_weights
from blade_epistemic_els import qwen_wrap, last_token_moments, is_unc, COMPONENTS, PPL_TOKENS
from blade_epistemic_p0 import split_3way, score_fn_for
import blade_epistemic_p0 as P0
from baseline_dola_iti import load_ood, head_acts, proj_std, gen_iti, gen_plain

RESULTS = Path(__file__).resolve().parent.parent / "results"
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")


def main():
    P0.BLADE_G = True
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    EX.chat_wrap = qwen_wrap; GEN.chat_wrap = qwen_wrap
    all_layers = list(range(len(get_decoder_layers(model))))
    blocks = get_decoder_layers(model); cfg = model.config
    nh, hd = cfg.num_attention_heads, getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    rows = json.loads((RESULTS / "epistemic_pairs_v2.json").read_text())["rows"]
    tr, sel, _ = split_3way(rows)
    unc_tr = [r["question"] for r in tr if r["label"] == 1]; cert_tr = [r["question"] for r in tr if r["label"] == 0]
    unc_all = [r["question"] for r in rows if r["label"] == 1]; cert_all = [r["question"] for r in rows if r["label"] == 0]
    unc_sel = [r["question"] for r in sel if r["label"] == 1]
    c4 = load_c4_text(); base_ppl = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)

    directions = extract_refusal_direction(model, tok, unc_tr, cert_tr)
    dirs_neg = {l: -directions[l] for l in directions}
    muUNC = last_token_moments(model, tok, unc_tr, all_layers, COMPONENTS, qwen_wrap)
    muCERT = last_token_moments(model, tok, cert_tr, all_layers, COMPONENTS, qwen_wrap)
    print("Q ...", flush=True)
    P0.Q_GLOBAL, _ = collect_c4_generic_importance(model, tok, all_layers, COMPONENTS, text=c4, seqlen=2048,
                                                   batch_size=2, mode="g1scalar", max_tokens=65536)
    sfn = score_fn_for(model, directions, muUNC, muCERT, all_layers)
    sfn_neg = score_fn_for(model, dirs_neg, muUNC, muCERT, all_layers)

    def ppl_now():
        return teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)

    def measure():
        o = gen_plain(model, tok, unc_sel); return sum(is_unc(x) for x in o) / max(len(o), 1), ppl_now()
    base_sel = measure()[0]
    pool = solo_layer_pool(model, directions, muUNC, muCERT, all_layers, COMPONENTS, ppl_now, base_ppl,
                           screen_frac=0.005, beta=0.05, score_fn=sfn)
    L_star = bestfirst_layers(model, directions, muUNC, muCERT, pool, COMPONENTS, measure, base_sel,
                              base_ppl, beta=0.05, eps=0.005, test_frac=0.005, score_fn=sfn)
    print(f"L*={L_star}", flush=True)
    rk = rank_weight_indices(sfn(model, directions, muUNC, muCERT, L_star, COMPONENTS), 0.05)
    rk_supp = rank_weight_indices(sfn_neg(model, dirs_neg, muUNC, muCERT, L_star, COMPONENTS), 0.05)
    sel_005 = selection_from_ranking(rk, 0.005)
    sel_supp = selection_from_ranking(rk_supp, 0.02)

    # ITI dirs (fit on full contrast, same as baseline)
    mu_u, _ = head_acts(model, tok, unc_all, blocks, nh, hd)
    mu_c, _ = head_acts(model, tok, cert_all, blocks, nh, hd)
    diffs = {(i, h): (mu_u[i][h] - mu_c[i][h]) for i in range(len(blocks)) for h in range(nh)}
    ranked = sorted(diffs, key=lambda k: -diffs[k].norm().item())[:48]
    idirs = {k: diffs[k] / diffs[k].norm().clamp_min(1e-6) for k in ranked}
    sigma = proj_std(model, tok, unc_all + cert_all, blocks, nh, hd, idirs)

    def iti_add(a):
        add = {i: torch.zeros(nh * hd) for i in range(len(blocks))}
        for (i, h) in ranked:
            add[i].view(nh, hd)[h] = a * sigma[(i, h)] * idirs[(i, h)]
        return add

    items = load_ood()
    prompts = [it["question"] for it in items]
    print(f"OOD items {len(items)}", flush=True)
    gens = {}
    gens["iti_a4.0"] = gen_iti(model, tok, prompts, blocks, iti_add(4.0)); print("  iti_a4 done", flush=True)
    for a in (3.0, 4.0):
        with scaled_weights(model, sel_005, a):
            gens[f"blade_ampW{a}"] = gen_plain(model, tok, prompts)
        print(f"  blade_ampW{a} done", flush=True)
    with pruned_weights(model, sel_supp):
        gens["blade_suppr_r0.02"] = gen_plain(model, tok, prompts)
    print("  blade_suppr done", flush=True)

    report = {"model": MODEL_ID, "L_star": L_star, "env": env_info(), "conditions": list(gens), "items": []}
    for i, it in enumerate(items):
        report["items"].append({**it, **{c: gens[c][i] for c in gens}})
    (RESULTS / "ood_tune_qwen3-8b.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("saved results/ood_tune_qwen3-8b.json", flush=True)


if __name__ == "__main__":
    main()
