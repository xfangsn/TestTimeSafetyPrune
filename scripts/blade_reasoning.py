"""BLADE remove/amplify of reasoning BEHAVIORS on DeepSeek-R1-Distill-Qwen-1.5B, using per-behavior
directions from results/reasoning_dirs.pt (extracted from the paper's bundled annotations).
For each target behavior: s_ij=[r_i W_ij Δμ_j]_+ at the paper layer(s); REMOVE (α=0) and AMPLIFY
(α∈{1.5,2}) the BLADE mask vs an equal-size RANDOM mask; generate on held-out eval tasks; measure the
behavior via keyword rate per 1000 thinking tokens (LEAN proxy; Fable annotator validates later) plus
thinking length and a WikiText ppl guard. Writes results/blade_reasoning.json."""
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from ttsafety.eval import load_wikitext_text, teacher_forced_nll
from ttsafety.models import load_model
from ttsafety.sycophancy import score_edges
from ttsafety.weight_prune import random_scores_like, rank_weight_indices, selection_from_ranking
from blade_refusal_amplify import scaled_weights

REPO = Path("/tmp/claude-1000/-home-xfang1999-Projects-TestTimeSafetyPrune/"
            "e16f646c-64a5-440b-bd68-985c068d25df/scratchpad/steering-thinking-llms")
sys.path.insert(0, str(REPO / "messages"))
from messages import messages as TRAIN_MSGS, eval_messages as EVAL_MSGS  # noqa: E402

RESULTS = Path("results")
LAYERS = [15, 16, 17, 18]     # paper's Qwen-1.5B band
RHO = 0.01; SCREEN = 0.03; MAX_NEW = 512; PPL_TOKENS = 4000
TARGETS = ["example-testing", "uncertainty-estimation", "backtracking"]
KW = {
    "uncertainty-estimation": [" maybe", " perhaps", "not sure", "i think", " possibly", " might ",
                               "could be", "i'm not", " unsure", " i guess", "not certain"],
    "example-testing": ["for example", "for instance", "let's try", "e.g.", "let me test",
                         " suppose ", "let's test", " let me try"],
    "backtracking": [" wait", " actually", "reconsider", " hmm", "scratch that",
                     "on second thought", " no,", "let me re", " but wait"],
}


def keyword_rate(thinking, behavior):
    t = thinking.lower()
    n_tok = max(1, len(thinking.split()))
    hits = sum(t.count(k) for k in KW[behavior])
    return 1000.0 * hits / n_tok


def parse_think(text):
    m = re.search(r"(.*?)</think>", text, re.DOTALL)
    return (m.group(1) if m else text)


@torch.no_grad()
def generate(model, tok, prompts, bs=12):
    prev = tok.padding_side; tok.padding_side = "left"
    outs = []
    try:
        for s in range(0, len(prompts), bs):
            enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            outs.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    finally:
        tok.padding_side = prev
    return outs


def main():
    model, tok = load_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    D = torch.load(RESULTS / "reasoning_dirs.pt", weights_only=False)
    dirs, muC, muG = D["dirs"], D["muC"], D["muG"]

    train_set = {m["content"] for m in TRAIN_MSGS}
    evals = [m for m in EVAL_MSGS if m["content"] not in train_set]   # drop overlap -> held-out
    prompts = [tok.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in evals]
    print(f"held-out eval tasks: {len(evals)}", flush=True)
    wiki = load_wikitext_text()
    base_nll, _ = teacher_forced_nll(model, tok, wiki, max_tokens=PPL_TOKENS)

    @contextmanager
    def noop():
        yield

    import math

    def measure(make_cm, label):
        with make_cm():
            outs = generate(model, tok, prompts)
            nll, _ = teacher_forced_nll(model, tok, wiki, max_tokens=PPL_TOKENS)
        thinks = [parse_think(o) for o in outs]
        rates = {b: sum(keyword_rate(t, b) for t in thinks) / len(thinks) for b in TARGETS}
        tlen = sum(len(t.split()) for t in thinks) / len(thinks)
        relppl = math.exp(nll - base_nll) - 1.0
        print(f"  {label:34} tlen {tlen:6.0f} | " +
              " ".join(f"{b[:4]} {rates[b]:5.2f}" for b in TARGETS) + f" | Δppl {relppl:+.1%}", flush=True)
        return {"label": label, "think_len": tlen, "kw_rates": rates, "relppl": relppl}

    out = {"model": D["model"], "layers": LAYERS, "rho": RHO, "n_eval": len(evals),
           "targets": TARGETS, "cells": []}
    print("== BASE ==", flush=True)
    out["cells"].append(measure(noop, "base"))

    for beh in TARGETS:
        scores = score_edges(model, dirs[beh], muC[beh], muG, LAYERS, "both")
        blade_sel = selection_from_ranking(rank_weight_indices(scores, SCREEN), RHO)
        rand_sel = selection_from_ranking(
            rank_weight_indices(random_scores_like(scores, seed=7), SCREEN), RHO)
        n_edges = sum(len(v) for v in blade_sel.values())
        print(f"\n== target={beh} n_edges={n_edges} ==", flush=True)
        for tag, sel, alphas in [("blade", blade_sel, [0.0, 1.5, 2.0]),
                                 ("random", rand_sel, [0.0])]:
            for a in alphas:
                def make_cm(sel=sel, a=a):
                    @contextmanager
                    def cm():
                        with scaled_weights(model, sel, a):
                            yield
                    return cm()
                r = measure(make_cm, f"{beh[:8]}:{tag}:a{a}")
                r.update({"target": beh, "mask": tag, "alpha": a, "n_edges": n_edges})
                out["cells"].append(r)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "blade_reasoning.json").write_text(json.dumps(out, indent=2))
    print("\nsaved results/blade_reasoning.json", flush=True)


if __name__ == "__main__":
    main()
