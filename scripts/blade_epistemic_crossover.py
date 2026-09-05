"""Experiment B run — fit the scheme-A epistemic BLADE-G edit on epistemic_pairs_v2 (fit NOWHERE on the
crossover), then generate base + REMOVE on the 144 evidence-sufficiency crossover items and save per-cell
generations for the blind judge. The decisive question (codex): does REMOVE reduce appropriate abstention
FOLLOWING EVIDENCE (both context-omits cells, familiar AND nonce) = epistemic, or only on NONCE entities
regardless of evidence = novelty/anomaly detector.

Env: BLADE_MODEL, BLADE_G=1, BLADE_RHO=0.005.
Usage: BLADE_MODEL=Qwen/Qwen3-8B BLADE_G=1 .venv/bin/python scripts/blade_epistemic_crossover.py
"""
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
from ttsafety.weight_prune import pruned_weights, rank_weight_indices, selection_from_ranking
from blade_epistemic_els import qwen_wrap, last_token_moments, is_unc, COMPONENTS, PPL_TOKENS
from blade_epistemic_p0 import split_3way, score_fn_for
import blade_epistemic_p0 as P0

RESULTS = Path(__file__).resolve().parent.parent / "results"
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")
RHO = float(os.environ.get("BLADE_RHO", "0.005"))
GEN_TOK = 96
BLADE_G = os.environ.get("BLADE_G", "") == "1"


def gen(model, tok, prompts):
    return generate_texts(model, tok, list(prompts), max_new_tokens=GEN_TOK, batch_size=12)


def main():
    P0.BLADE_G = BLADE_G
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    EX.chat_wrap = qwen_wrap; GEN.chat_wrap = qwen_wrap
    all_layers = list(range(len(get_decoder_layers(model))))

    # fit on scheme-A v2 (NOT on crossover)
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
    selw = selection_from_ranking(rank_weight_indices(scores, 0.03), RHO)

    cross = json.loads((RESULTS / "evidence_crossover.json").read_text())["rows"]
    prompts = [r["question"] for r in cross]
    base_out = gen(model, tok, prompts)
    with pruned_weights(model, selw):
        rem_out = gen(model, tok, prompts)
    report = {"model": MODEL_ID, "L_star": L_star, "rho": RHO, "n_items": len(cross),
              "env": env_info(), "items": []}
    for r, b, e in zip(cross, base_out, rem_out):
        report["items"].append({"qid": r["qid"], "entity_kind": r["entity_kind"], "context": r["context"],
                                "answerable": r["answerable"], "value": r["value"],
                                "question": r["question"], "base": b, "remove": e})
    tag = MODEL_ID.split("/")[-1].replace(".", "").lower() + ("_bladeg" if BLADE_G else "")
    (RESULTS / f"epistemic_crossover_{tag}.json").write_text(json.dumps(report, indent=2))
    print(f"saved results/epistemic_crossover_{tag}.json ({len(cross)} items, generations for judge)", flush=True)


if __name__ == "__main__":
    main()
