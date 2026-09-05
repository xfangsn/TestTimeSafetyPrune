"""OOD transfer run — apply the FROZEN scheme-A epistemic edit (fit on epistemic_pairs_v2, closed-book)
to real closed-book benchmarks it was NEVER fit on: FalseQA (false/true-premise minimal pairs) and
SelfAware (unanswerable/answerable). Generate base / REMOVE / AMPLIFY(raw-alphaW x2) + matched-random +
shuffled-r controls; save per-item generations for a blind kimi judge (like the OOD-refusal line).

Env: BLADE_MODEL, BLADE_G=1, OOD_NFQ (false/true pairs), OOD_NSA_U/OOD_NSA_A (SelfAware unans/ans).
Usage: BLADE_MODEL=Qwen/Qwen3-8B BLADE_G=1 .venv/bin/python scripts/blade_ood_run.py
"""
import csv
import json
import os
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
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,
                                   random_scores_like, selection_from_ranking)
from blade_refusal_amplify import scaled_weights
from blade_epistemic_els import qwen_wrap, last_token_moments, is_unc, COMPONENTS, PPL_TOKENS
from blade_epistemic_p0 import split_3way, score_fn_for
import blade_epistemic_p0 as P0

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"; DATA = ROOT / "data"
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")
BLADE_G = os.environ.get("BLADE_G", "") == "1"
GEN_TOK = 128
RHO = 0.005
NFQ = int(os.environ.get("OOD_NFQ", "150"))
NSA_U = int(os.environ.get("OOD_NSA_U", "150"))
NSA_A = int(os.environ.get("OOD_NSA_A", "100"))


def load_falseqa():
    rows = list(csv.DictReader(open(DATA / "abstention" / "falseqa_test.csv")))
    false_ = [r["question"] for r in rows if r["label"] == "1"][:NFQ]
    true_ = [r["question"] for r in rows if r["label"] == "0"][:NFQ]
    out = []
    for q in false_:
        out.append({"dataset": "falseqa", "gold": "false_premise", "question": q})
    for q in true_:
        out.append({"dataset": "falseqa", "gold": "true_premise", "question": q})
    return out


def load_selfaware():
    ex = json.loads((DATA / "abstention" / "SelfAware.json").read_text())["example"]
    un = [x["question"] for x in ex if not x["answerable"]][:NSA_U]
    an = [x["question"] for x in ex if x["answerable"]][:NSA_A]
    return ([{"dataset": "selfaware", "gold": "unanswerable", "question": q} for q in un]
            + [{"dataset": "selfaware", "gold": "answerable", "question": q} for q in an])


def gen(model, tok, prompts):
    return generate_texts(model, tok, list(prompts), max_new_tokens=GEN_TOK, batch_size=16)


def main():
    P0.BLADE_G = BLADE_G
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    EX.chat_wrap = qwen_wrap; GEN.chat_wrap = qwen_wrap
    all_layers = list(range(len(get_decoder_layers(model))))

    # fit frozen on scheme-A v2 (closed-book)
    rows = json.loads((RESULTS / "epistemic_pairs_v2.json").read_text())["rows"]
    tr, sel, _ = split_3way(rows)
    unc_tr = [r["question"] for r in tr if r["label"] == 1]
    cert_tr = [r["question"] for r in tr if r["label"] == 0]
    unc_sel = [r["question"] for r in sel if r["label"] == 1]
    c4 = load_c4_text(); base_ppl = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    directions = extract_refusal_direction(model, tok, unc_tr, cert_tr)
    muUNC = last_token_moments(model, tok, unc_tr, all_layers, COMPONENTS, qwen_wrap)
    muCERT = last_token_moments(model, tok, cert_tr, all_layers, COMPONENTS, qwen_wrap)
    if BLADE_G:
        print("Q ...", flush=True)
        P0.Q_GLOBAL, _ = collect_c4_generic_importance(model, tok, all_layers, COMPONENTS, text=c4,
                                                       seqlen=2048, batch_size=2, mode="g1scalar", max_tokens=65536)
    sfn = score_fn_for(model, directions, muUNC, muCERT, all_layers)

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
    scores = sfn(model, directions, muUNC, muCERT, L_star, COMPONENTS)
    rk = rank_weight_indices(scores, 0.05)
    remove_sel = selection_from_ranking(rk, RHO)
    rand_sel = selection_from_ranking(rank_weight_indices(random_scores_like(scores, 7), 0.05), RHO)
    g = torch.Generator().manual_seed(123)
    dshuf = {l: directions[l][torch.randperm(directions[l].numel(), generator=g)] for l in directions}
    sc_shuf = sfn(model, dshuf, muUNC, muCERT, L_star, COMPONENTS)
    shuf_sel = selection_from_ranking(rank_weight_indices(sc_shuf, 0.05), RHO)

    items = load_falseqa() + load_selfaware()
    prompts = [it["question"] for it in items]
    print(f"OOD items: {len(items)} (falseqa {sum(i['dataset']=='falseqa' for i in items)}, "
          f"selfaware {sum(i['dataset']=='selfaware' for i in items)})", flush=True)

    from contextlib import nullcontext
    conds = {
        "base": nullcontext(),
        "remove": pruned_weights(model, remove_sel),
        "amplify": scaled_weights(model, remove_sel, 2.0),
        "ctrl_random": pruned_weights(model, rand_sel),
        "ctrl_shuffledr": pruned_weights(model, shuf_sel),
    }
    gens = {}
    for name, cm in conds.items():
        with cm:
            gens[name] = gen(model, tok, prompts)
        print(f"  generated {name}", flush=True)

    report = {"model": MODEL_ID, "L_star": L_star, "rho": RHO, "amplify": "raw_alphaW_x2",
              "n_items": len(items), "env": env_info(), "items": []}
    for i, it in enumerate(items):
        report["items"].append({**it, **{c: gens[c][i] for c in conds}})
    tag = MODEL_ID.split("/")[-1].replace(".", "").lower() + ("_bladeg" if BLADE_G else "")
    (RESULTS / f"ood_run_{tag}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved results/ood_run_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
