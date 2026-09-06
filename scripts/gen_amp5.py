"""Generate BLADE amplify alpha=5 (raw-alphaW x5) on the OOD prompts + its C4 Δppl, for the method
comparison. Env: BLADE_MODEL. -> results/ood_amp5_qwen3-8b.json (+ ppl field)."""
import json, os
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
from blade_refusal_amplify import scaled_weights
from blade_epistemic_els import qwen_wrap, last_token_moments, is_unc, COMPONENTS, PPL_TOKENS
from blade_epistemic_p0 import split_3way, score_fn_for
import blade_epistemic_p0 as P0
from baseline_dola_iti import load_ood, gen_plain

RESULTS = Path(__file__).resolve().parent.parent / "results"
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")
ALPHAS = [float(x) for x in os.environ.get("AMP_ALPHAS", "5,6").split(",")]


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
    P0.Q_GLOBAL, _ = collect_c4_generic_importance(model, tok, all_layers, COMPONENTS, text=c4, seqlen=2048,
                                                   batch_size=2, mode="g1scalar", max_tokens=65536)
    sfn = score_fn_for(model, directions, muUNC, muCERT, all_layers)

    def ppl_now(): return teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    def measure():
        o = gen_plain(model, tok, unc_sel); return sum(is_unc(x) for x in o) / max(len(o), 1), ppl_now()
    base_sel = measure()[0]
    pool = solo_layer_pool(model, directions, muUNC, muCERT, all_layers, COMPONENTS, ppl_now, base_ppl,
                           screen_frac=0.005, beta=0.05, score_fn=sfn)
    L_star = bestfirst_layers(model, directions, muUNC, muCERT, pool, COMPONENTS, measure, base_sel,
                              base_ppl, beta=0.05, eps=0.005, test_frac=0.005, score_fn=sfn)
    sel005 = selection_from_ranking(rank_weight_indices(sfn(model, directions, muUNC, muCERT, L_star, COMPONENTS), 0.05), 0.005)
    items = load_ood(); prompts = [it["question"] for it in items]
    report = {"model": MODEL_ID, "L_star": L_star, "env": env_info(),
              "conditions": [f"blade_ampW{a}" for a in ALPHAS], "ppl_delta_c4": {}, "items": [dict(it) for it in items]}
    for a in ALPHAS:
        cond = f"blade_ampW{a}"
        with scaled_weights(model, sel005, a):
            gens = gen_plain(model, tok, prompts)
            pc = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
        report["ppl_delta_c4"][cond] = (pc - base_ppl) / base_ppl
        for rec, g in zip(report["items"], gens):
            rec[cond] = g
        print(f"{cond} Δppl {(pc-base_ppl)/base_ppl:+.2%}", flush=True)
    (RESULTS / "ood_ampextra_qwen3-8b.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("saved results/ood_ampextra_qwen3-8b.json", flush=True)


if __name__ == "__main__":
    main()
