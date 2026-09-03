"""Llama-3.2-3B base (pretrained) vs instruct on the BLADE behaviors.
Shows these behaviors are POST-TRAINING artifacts: the base model sits near
chance (A/B) / never refuses, while instruct exhibits them.

Base A/B pick-rate measured on the SAME exhibiting side as instruct (side read
from the instruct beta=5% run). Refusal measured by generation on harmful prompts.
Instruct baselines are read from existing result files (no re-run).
"""
import json
from pathlib import Path

import torch

import ttsafety.behaviors as B
import ttsafety.generate as G
from ttsafety.behaviors import fetch_ab, make_splits, pick_rate
from ttsafety.data import load_jsonl
from ttsafety.judge import is_refusal
from ttsafety.generate import generate_texts
from ttsafety.models import env_info, load_model

DATA = Path("data"); RESULTS = Path("results")
BASE_ID = "meta-llama/Llama-3.2-3B"
INSTR_ELS = RESULTS / "blade_els_llama-32-3b-instruct_beta5.json"
INSTR_REF = RESULTS / "blade_refusal_els_llama-32-3b-instruct.json"
BEHAVIORS = ["power-seeking", "wealth-seeking", "corrigibility", "deception",
             "self-rate-highly", "self-awareness", "sycophancy"]


def main():
    instr = json.loads(INSTR_ELS.read_text())["results"]
    model, tok = load_model(BASE_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base_wrap = lambda tokenizer, s: s + "\n"     # base: plain prompt, no chat template
    B.chat_wrap = base_wrap                        # for pick_rate
    G.chat_wrap = base_wrap                        # for generate_texts (refusal)

    rows = []
    for beh in BEHAVIORS:
        rec = instr.get(beh, {})
        side = rec.get("side", "matching")
        instr_pick = rec.get("baseline_bias")
        val = make_splits(fetch_ab(beh, DATA / "behaviors"))["val"]
        base_pick = pick_rate(model, tok, val, side)[0]
        rows.append({"behavior": beh, "side": side,
                     "base": base_pick, "instruct": instr_pick})
        print(f"{beh:16} side={side:12} base={base_pick:.3f}  instruct={instr_pick:.3f}", flush=True)

    # refusal: base generation on harmful prompts
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_val.jsonl")]
    outs = generate_texts(model, tok, harmful, max_new_tokens=48, batch_size=16)
    base_ref = sum(is_refusal(o) for o in outs) / len(outs)
    instr_ref = json.loads(INSTR_REF.read_text()).get("base_refusal")
    rows.append({"behavior": "refusal", "side": "refuse",
                 "base": base_ref, "instruct": instr_ref})
    print(f"{'refusal':16} base={base_ref:.3f}  instruct={instr_ref:.3f}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "base_vs_instruct_llama.json").write_text(json.dumps(
        {"base_model": BASE_ID, "rows": rows, "env": env_info()}, indent=2))
    print("saved results/base_vs_instruct_llama.json", flush=True)


if __name__ == "__main__":
    main()
