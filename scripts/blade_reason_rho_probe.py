"""Diagnostic: can ANY (rho, layer-band) remove uncertainty-estimation on the 8B model while keeping
WikiText ppl within budget? Sweep rho over a mid-layer band (giving BLADE its best shot, analogous to
the 1.5B L15-18 band that worked), BLADE-remove vs equal-size RANDOM mask, measure all-behavior
keyword rates + ppl. If uncertainty only drops when ppl blows the budget (or never beats random),
that confirms the behavior is entangled with general computation at 8B scale."""
import argparse
import json
import re
from contextlib import contextmanager
from pathlib import Path

import torch

from ttsafety.eval import load_wikitext_text, teacher_forced_nll
from ttsafety.models import load_model
from ttsafety.sycophancy import score_edges
from ttsafety.weight_prune import random_scores_like, rank_weight_indices, selection_from_ranking
from blade_refusal_amplify import scaled_weights
import sys
REPO = Path("/tmp/claude-1000/-home-xfang1999-Projects-TestTimeSafetyPrune/"
            "e16f646c-64a5-440b-bd68-985c068d25df/scratchpad/steering-thinking-llms")
sys.path.insert(0, str(REPO / "messages"))
from messages import messages as TRAIN, eval_messages as EVAL  # noqa: E402

RESULTS = Path("results"); SCREEN = 0.06; MAX_NEW = 512; PPL_TOKENS = 4000
BEH = ["uncertainty-estimation", "backtracking", "example-testing", "adding-knowledge"]
TARGET = "uncertainty-estimation"
RHOS = [0.005, 0.01, 0.02, 0.04]
KW = {
    "uncertainty-estimation": [" maybe", " perhaps", "not sure", "i think", " possibly", " might ",
                               "could be", "i'm not", " unsure", " i guess", "not certain"],
    "example-testing": ["for example", "for instance", "let's try", "e.g.", "let me test",
                        " suppose ", "let's test", " let me try"],
    "backtracking": [" wait", " actually", "reconsider", " hmm", "scratch that",
                     "on second thought", " no,", "let me re", " but wait"],
    "adding-knowledge": ["i know that", "i remember", "recall that", "it's known", "the formula",
                         "by definition", "in general,"],
}


def parse_think(t):
    m = re.search(r"(.*?)</think>", t, re.DOTALL)
    return (m.group(1) if m else t).strip()


@torch.no_grad()
def gen(model, tok, prompts, bs=8):
    prev = tok.padding_side; tok.padding_side = "left"; outs = []
    try:
        for s in range(0, len(prompts), bs):
            enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
            outs.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    finally:
        tok.padding_side = prev
    return outs


def rates(thinks):
    return {b: sum(1000.0 * sum(t.lower().count(k) for k in KW[b]) / max(1, len(t.split()))
                   for t in thinks) / len(thinks) for b in BEH}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--dirs", default="reasoning_dirs_llama8b.pt")
    ap.add_argument("--band", default="10,11,12,13,14,15,16,17")
    ap.add_argument("--out", default="blade_reason_rho_probe.json")
    args = ap.parse_args()
    BAND = [int(x) for x in args.band.split(",")]
    model, tok = load_model(args.model)
    D = torch.load(RESULTS / args.dirs, weights_only=False)
    dirs, muC, muG = D["dirs"], D["muC"], D["muG"]
    train = {m["content"] for m in TRAIN}
    evals = [m for m in EVAL if m["content"] not in train]
    prompts = [tok.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in evals]
    wiki = load_wikitext_text()
    base_nll, _ = teacher_forced_nll(model, tok, wiki, max_tokens=PPL_TOKENS)
    import math

    def measure(cm):
        with cm:
            g = gen(model, tok, prompts)
            nll, _ = teacher_forced_nll(model, tok, wiki, max_tokens=PPL_TOKENS)
        return rates([parse_think(x) for x in g]), math.exp(nll - base_nll) - 1.0

    @contextmanager
    def noop():
        yield

    def wcm(sel, a=0.0):
        @contextmanager
        def cm():
            with scaled_weights(model, sel, a):
                yield
        return cm()

    out = {"band": BAND, "rhos": RHOS, "target": TARGET, "cells": []}
    br, bppl = measure(noop())
    print(f"base | " + " ".join(f"{b[:4]} {br[b]:5.2f}" for b in BEH), flush=True)
    out["cells"].append({"cfg": "base", "rates": br, "relppl": 0.0})

    sc = score_edges(model, dirs[TARGET], muC[TARGET], muG, BAND, "both")
    for rho in RHOS:
        blade = selection_from_ranking(rank_weight_indices(sc, max(SCREEN, rho)), rho)
        rand = selection_from_ranking(
            rank_weight_indices(random_scores_like(sc, seed=7), max(SCREEN, rho)), rho)
        n = sum(len(v) for v in blade.values())
        for tag, sel in [("blade", blade), ("random", rand)]:
            r, pp = measure(wcm(sel))
            d = r[TARGET] - br[TARGET]
            out["cells"].append({"cfg": f"{tag}_rho{rho}", "rho": rho, "mask": tag, "n_edges": n,
                                 "rates": r, "relppl": pp})
            print(f"{tag:6} rho{rho:<5} unce {r[TARGET]:5.2f} (Δ{d:+.2f}) | "
                  + " ".join(f"{b[:4]} {r[b]:4.1f}" for b in BEH if b != TARGET)
                  + f" | Δppl {pp:+.1%}", flush=True)
        (RESULTS / args.out).write_text(json.dumps(out, indent=2))
    print("saved results/blade_reason_rho_probe.json", flush=True)


if __name__ == "__main__":
    main()
