"""C4 Δppl for BLADE raw-alphaW amplify at alpha in {2,3,4} (same L*/mask as ood_tune). -> results/amp_ppl.json"""
import json, os
from pathlib import Path
import torch
import ttsafety.extract as EX
from ttsafety.eval import load_c4_text, teacher_forced_ppl
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import load_model
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.behaviors import bestfirst_layers, solo_layer_pool
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking
from blade_refusal_amplify import scaled_weights
from blade_epistemic_els import qwen_wrap, last_token_moments, is_unc, COMPONENTS, PPL_TOKENS
from blade_epistemic_p0 import split_3way, score_fn_for
import blade_epistemic_p0 as P0
from blade_epistemic_els import _med_pos  # noqa

RESULTS = Path(__file__).resolve().parent.parent / "results"
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")


def main():
    P0.BLADE_G = True
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    EX.chat_wrap = qwen_wrap
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
        from blade_epistemic_els import GEN_TOKENS
        from ttsafety.generate import generate_texts
        o = generate_texts(model, tok, unc_sel, max_new_tokens=GEN_TOKENS, batch_size=16)
        return sum(is_unc(x) for x in o) / max(len(o), 1), ppl_now()
    base_sel = measure()[0]
    pool = solo_layer_pool(model, directions, muUNC, muCERT, all_layers, COMPONENTS, ppl_now, base_ppl,
                           screen_frac=0.005, beta=0.05, score_fn=sfn)
    L_star = bestfirst_layers(model, directions, muUNC, muCERT, pool, COMPONENTS, measure, base_sel,
                              base_ppl, beta=0.05, eps=0.005, test_frac=0.005, score_fn=sfn)
    sel005 = selection_from_ranking(rank_weight_indices(sfn(model, directions, muUNC, muCERT, L_star, COMPONENTS), 0.05), 0.005)
    out = {"base_ppl_c4": base_ppl, "amp_ppl_delta": {}}
    import contextlib
    for a in (2.0, 3.0, 4.0):
        with scaled_weights(model, sel005, a):
            p = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
        out["amp_ppl_delta"][f"ampW{a}"] = (p - base_ppl) / base_ppl
        print(f"ampW{a}: Δppl {(p-base_ppl)/base_ppl:+.2%}", flush=True)
    (RESULTS / "amp_ppl.json").write_text(json.dumps(out, indent=2))
    print("saved results/amp_ppl.json")


if __name__ == "__main__":
    main()
