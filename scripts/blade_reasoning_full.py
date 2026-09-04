"""Full BLADE reasoning-behavior experiment with controls (agenda ①③). Parameterized by --dirs
(reasoning_dirs_llama8b.pt default) and --model. Held-out eval tasks (repo eval_messages minus train
overlap). Saves per-config generated traces (for later Fable annotation) + keyword-proxy behavior
rates (per 1000 thinking words) + thinking length + WikiText ppl, incrementally to results/<out>.

Stages (each writes as it goes):
  base                          : reference behavior rates.
  remove_matrix                 : BLADE remove (α=0) for EACH target behavior; measure ALL behaviors
                                  -> 4x4 selectivity matrix (edit i, effect on j).
  controls                      : for the primary behavior, multi-seed RANDOM masks + SHUFFLED-r mask
                                  (both α=0) -> BLADE must beat these.
  steer_baseline                : paper's activation steering (subtract r_c at the layer) remove.
  amplify                       : GENTLER config (single layer, small ρ, small α) to test enhancement
                                  without the ppl blow-up seen at ρ0.01x4layers on the 1.5B model.
"""
import argparse
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from ttsafety.eval import (load_c4_text, load_wikitext_text, teacher_forced_nll)
from ttsafety.models import load_model
from ttsafety.sycophancy import score_edges, score_edges_g
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.hooks import get_decoder_layers
from ttsafety.weight_prune import random_scores_like, rank_weight_indices, selection_from_ranking
from blade_refusal_amplify import scaled_weights
import ttsafety.steering_methods as SM

# Training/eval message sets (cvenhoff/steering-thinking-llms). Prefer $STEER_REPO/messages if set
# (the cloned repo), else the vendored copy committed under scripts/steer_messages (Hazel-portable).
import os
_STEER = os.environ.get("STEER_REPO")
sys.path.insert(0, str(Path(_STEER) / "messages") if _STEER
                else str(Path(__file__).resolve().parent / "steer_messages"))
from messages import messages as TRAIN_MSGS, eval_messages as EVAL_MSGS  # noqa: E402

RESULTS = Path("results")
SCREEN = 0.03; MAX_NEW = 512; PPL_TOKENS = 4000
TARGETS = ["uncertainty-estimation", "backtracking", "example-testing", "adding-knowledge"]
PRIMARY = "uncertainty-estimation"
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


def kw_rate(thinking, beh):
    t = thinking.lower(); n = max(1, len(thinking.split()))
    return 1000.0 * sum(t.count(k) for k in KW[beh]) / n


def parse_think(text):
    m = re.search(r"(.*?)</think>", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


@torch.no_grad()
def generate(model, tok, prompts, bs=8):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="reasoning_dirs_llama8b.pt")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--out", default="blade_reasoning_llama8b.json")
    ap.add_argument("--els", default="reasoning_els_llama8b.json")  # BLADE-auto-selected layers
    ap.add_argument("--remove-rho", type=float, default=0.008)
    ap.add_argument("--amp-rho", type=float, default=0.002)
    ap.add_argument("--amp-alphas", default="1.25,1.5")
    ap.add_argument("--blade-g", action="store_true",
                    help="use BLADE-G (generic-importance-penalized) scoring instead of base BLADE-B")
    ap.add_argument("--q-tokens", type=int, default=65536, help="C4 tokens for the generic penalty Q")
    args = ap.parse_args()
    AMP_ALPHAS = [float(x) for x in args.amp_alphas.split(",")]

    model, tok = load_model(args.model)
    D = torch.load(RESULTS / args.dirs, weights_only=False)
    dirs, muC, muG = D["dirs"], D["muC"], D["muG"]

    # --- BLADE-G: generic-importance penalty Q (once, all layers) + per-behavior lambda ---
    def _med_pos(d):
        t = torch.cat([v.flatten() for v in d.values()]).float(); return t[t > 0].median().item()
    Q_GLOBAL, LAM = None, {}
    if args.blade_g:
        allL = list(range(len(get_decoder_layers(model))))
        print(f"[BLADE-G] computing generic-importance Q (g1scalar) on {len(allL)} layers ...", flush=True)
        Q_GLOBAL, _ = collect_c4_generic_importance(model, tok, allL, "both", text=load_c4_text(),
                                                    seqlen=2048, batch_size=2, mode="g1scalar",
                                                    max_tokens=args.q_tokens)
        qmed = _med_pos(Q_GLOBAL)
        for b in dirs:
            LAM[b] = _med_pos(score_edges(model, dirs[b], muC[b], muG, allL, "both")) / qmed
        print(f"[BLADE-G] lambda per behavior: " +
              " ".join(f"{b[:4]} {LAM[b]:.2e}" for b in LAM), flush=True)
    # BLADE ELS-selected effective layers per behavior (auto, not hand-picked)
    ELS = json.loads((RESULTS / args.els).read_text())["els"]
    els_layers = {b: (ELS[b]["L_star"] or ELS[b]["pool"][:1]) for b in ELS}  # fallback: top pool layer
    print(f"ELS layers per behavior: {els_layers}", flush=True)

    train_set = {m["content"] for m in TRAIN_MSGS}
    evals = [m for m in EVAL_MSGS if m["content"] not in train_set]
    prompts = [tok.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in evals]
    tasks = [m["content"] for m in evals]
    wiki = load_wikitext_text()
    base_nll, _ = teacher_forced_nll(model, tok, wiki, max_tokens=PPL_TOKENS)

    out = {"model": args.model, "dirs": args.dirs,
           "cfg": {"els_layers": els_layers, "remove_rho": args.remove_rho,
                   "amp_rho": args.amp_rho, "amp_alphas": AMP_ALPHAS, "blade_g": args.blade_g},
           "n_eval": len(evals), "cells": [], "traces": {}}
    traces_path = RESULTS / args.out.replace(".json", "_traces.json")

    def save():
        (RESULTS / args.out).write_text(json.dumps({k: v for k, v in out.items() if k != "traces"}, indent=2))
        traces_path.write_text(json.dumps(out["traces"], ensure_ascii=False))

    import math

    def measure(make_cm, label):
        with make_cm():
            gens = generate(model, tok, prompts)
            nll, _ = teacher_forced_nll(model, tok, wiki, max_tokens=PPL_TOKENS)
        thinks = [parse_think(g) for g in gens]
        rates = {b: sum(kw_rate(t, b) for t in thinks) / len(thinks) for b in TARGETS}
        tlen = sum(len(t.split()) for t in thinks) / len(thinks)
        relppl = math.exp(nll - base_nll) - 1.0
        cell = {"label": label, "think_len": tlen, "kw_rates": rates, "relppl": relppl}
        out["cells"].append(cell)
        out["traces"][label] = [{"task": t, "thinking": th} for t, th in zip(tasks, thinks)]
        print(f"  {label:30} tlen {tlen:5.0f} | " +
              " ".join(f"{b[:4]} {rates[b]:5.2f}" for b in TARGETS) + f" | Δppl {relppl:+.1%}", flush=True)
        save()
        return cell

    def mask(beh, layers, rho, seed=None, shuffle=False):
        r = dirs[beh]
        if shuffle:  # break r<->W alignment: permute each layer's direction entries
            g = torch.Generator().manual_seed(123)
            r = {l: r[l][torch.randperm(r[l].numel(), generator=g)] for l in r}
        if args.blade_g:
            scores = score_edges_g(model, r, muC[beh], muG, layers, "both",
                                   Q=Q_GLOBAL, lam=LAM[beh], abstain=True)
            scores = {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in scores.items()}
        else:
            scores = score_edges(model, r, muC[beh], muG, layers, "both")
        if seed is not None:
            scores = random_scores_like(scores, seed=seed)
        return selection_from_ranking(rank_weight_indices(scores, SCREEN), rho)

    def wcm(sel, a):
        def f():
            @contextmanager
            def cm():
                with scaled_weights(model, sel, a):
                    yield
            return cm()
        return f

    @contextmanager
    def _noop():
        yield

    # --- base ---
    print("== BASE ==", flush=True)
    measure(lambda: _noop(), "base")

    # Only behaviours that ELS actually localised (have selected layers) can be edited; on some
    # models ELS keeps just uncertainty. Iterate those, always keeping PRIMARY if present.
    HAVE = [b for b in TARGETS if b in els_layers and els_layers[b]]
    print(f"editable behaviors (in ELS): {HAVE}", flush=True)

    # --- remove_matrix: BLADE remove each behavior at ITS ELS layers, measure all (4x4) ---
    print("== REMOVE (4x4 selectivity, ELS layers) ==", flush=True)
    for beh in HAVE:
        sel = mask(beh, els_layers[beh], args.remove_rho)
        out.setdefault("n_edges", {})[beh] = sum(len(v) for v in sel.values())
        measure(wcm(sel, 0.0), f"remove:{beh}")

    # --- controls on the primary behavior (same ELS layers) ---
    print(f"== CONTROLS ({PRIMARY}) ==", flush=True)
    for seed in (7, 17, 27):
        measure(wcm(mask(PRIMARY, els_layers[PRIMARY], args.remove_rho, seed=seed), 0.0),
                f"ctrl_random{seed}:{PRIMARY}")
    measure(wcm(mask(PRIMARY, els_layers[PRIMARY], args.remove_rho, shuffle=True), 0.0),
            f"ctrl_shuffledr:{PRIMARY}")

    # --- activation-steering baseline (paper's method): subtract r_c (unit) at an ELS layer ---
    print("== STEER BASELINE (subtract r) ==", flush=True)

    def steer_cm(beh, coef):
        L = els_layers[beh][len(els_layers[beh]) // 2]     # middle ELS layer of the behavior
        vec = dirs[beh][L] / dirs[beh][L].norm().clamp_min(1e-6)
        def f():
            @contextmanager
            def cm():
                with SM.resid_add(model, L, vec, coef, "all"):
                    yield
            return cm()
        return f
    for beh in [b for b in (PRIMARY, "backtracking") if b in HAVE]:
        for c in (-8.0, -16.0):     # negative = suppress the behavior
            measure(steer_cm(beh, c), f"steer:{beh}:c{c}")

    # --- amplify (gentler): same ELS layers, smaller rho + small alpha ---
    print("== AMPLIFY (gentle, ELS layers) ==", flush=True)
    for beh in [b for b in (PRIMARY, "example-testing") if b in HAVE]:
        sel = mask(beh, els_layers[beh], args.amp_rho)
        for a in AMP_ALPHAS:
            measure(wcm(sel, a), f"amp:{beh}:a{a}")

    print(f"\nsaved results/{args.out} (+ traces)", flush=True)


if __name__ == "__main__":
    main()
