"""Can a BLADE-style weight edit change membership leakage?

Localize a "membership behavior": direction r_l = mean last-token block-output
(member) - (non-member); writer-input moment diff mu_member - mu_nonmember.
Score residual-writer edges, take top-k, then SCALE that mask by `factor`
(0 = zero/remove, 1 = baseline, >1 = amplify) and re-measure MIA.

CRITICAL: direction/moments built on a TRAIN split; MIA (TPR @ low FPR) evaluated
on a disjoint TEST split -- no circularity. Metric = TPR@1%/5%FPR (AUC reference).
"""
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.sycophancy import score_edges
from ttsafety.weight_edit import iter_residual_writers
from ttsafety.weight_prune import (rank_weight_indices, selection_from_ranking,
                                   _resolve_modules)
from mia_wikimia import token_logprobs, scores_for, tpr_at_fpr, auc

MODEL_ID = os.environ.get("BLADE_MODEL", "EleutherAI/pythia-2.8b")
SPLIT = os.environ.get("WIKIMIA_SPLIT", "WikiMIA_length64")
COMPONENTS = "both"
SPARSITY = float(os.environ.get("MIA_SPARSITY", "0.005"))
FACTORS = [float(x) for x in os.environ.get("MIA_FACTORS","1.0,1.05,1.1,1.2").split(",")]
RESULTS = Path("results")


@contextmanager
def scaled_weights(model, selection, factor):
    modules = _resolve_modules(model, list(selection))
    backups = {}
    with torch.no_grad():
        for name, idx in selection.items():
            w = modules[name].weight
            di = idx.to(w.device, torch.long)
            flat = w.view(-1)
            backups[name] = flat[di].detach().cpu().clone()
            flat[di] = flat[di] * factor
    try:
        yield
    finally:
        with torch.no_grad():
            for name, idx in selection.items():
                w = modules[name].weight
                di = idx.to(w.device, torch.long)
                w.view(-1)[di] = backups[name].to(w.device, w.dtype)


@torch.no_grad()
def collect(model, tok, texts, layers):
    """Mean last-token block-output (per layer) + writer-input (per writer)."""
    blocks = get_decoder_layers(model)
    writers = dict(iter_residual_writers(model, layers, COMPONENTS))
    out_sum = {l: None for l in layers}
    win_sum = {n: None for n in writers}
    state = {}
    cnt = 0

    def blk_hook(l):
        def h(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            state.setdefault("blk", {})[l] = hs[torch.arange(hs.shape[0]), state["last"]].float().sum(0)
        return h

    def wr_hook(n):
        def h(_m, args):
            v = args[0].float()
            state.setdefault("win", {})[n] = v[torch.arange(v.shape[0]), state["last"]].sum(0)
        return h

    hs = [blocks[l].register_forward_hook(blk_hook(l)) for l in layers]
    hs += [writers[n].register_forward_pre_hook(wr_hook(n)) for n in writers]
    try:
        for s in range(0, len(texts), 8):
            enc = tok(texts[s:s + 8], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(model.device)
            state["last"] = enc["attention_mask"].sum(1) - 1
            state["blk"] = {}; state["win"] = {}
            model(**enc)
            for l in layers:
                out_sum[l] = state["blk"][l] if out_sum[l] is None else out_sum[l] + state["blk"][l]
            for n in writers:
                win_sum[n] = state["win"][n] if win_sum[n] is None else win_sum[n] + state["win"][n]
            cnt += enc["input_ids"].shape[0]
    finally:
        for h in hs:
            h.remove()
    return ({l: (v / cnt).cpu() for l, v in out_sum.items()},
            {n: (v / cnt).cpu() for n, v in win_sum.items()})


def eval_mia(model, tok, rows):
    per = {m: [] for m in ("loss", "mink", "minkpp", "zlib")}
    labels = []
    for r in rows:
        o = token_logprobs(model, tok, r["input"])
        if o is None:
            continue
        sc = scores_for(*o)
        for m in per:
            per[m].append(sc[m])
        labels.append(r["label"])
    return {m: {"auc": auc(per[m], labels),
                "tpr1": tpr_at_fpr(per[m], labels, 0.01),
                "tpr5": tpr_at_fpr(per[m], labels, 0.05)} for m in per}


def main():
    rows = list(load_dataset("swj0419/WikiMIA", split=SPLIT))
    random.seed(0); random.shuffle(rows)
    mem = [r for r in rows if r["label"] == 1]
    non = [r for r in rows if r["label"] == 0]
    half_m, half_n = len(mem) // 2, len(non) // 2
    train_mem, test_mem = mem[:half_m], mem[half_m:]
    train_non, test_non = non[:half_n], non[half_n:]
    test = test_mem + test_non
    print(f"{MODEL_ID} | {SPLIT} | train: {len(train_mem)}m/{len(train_non)}n "
          f"| test: {len(test_mem)}m/{len(test_non)}n", flush=True)

    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    nl = len(get_decoder_layers(model))
    layers = list(range(2, nl - 2))

    print("building membership direction + moments on TRAIN ...", flush=True)
    r_mem, mu_mem = collect(model, tok, [r["input"] for r in train_mem], layers)
    r_non, mu_non = collect(model, tok, [r["input"] for r in train_non], layers)
    directions = {l: r_mem[l] - r_non[l] for l in layers}
    scores = score_edges(model, directions, mu_mem, mu_non, layers, COMPONENTS)
    rk = rank_weight_indices(scores, 0.03)
    sel = selection_from_ranking(rk, SPARSITY)
    n_edges = sum(len(v) for v in sel.values())
    print(f"membership mask = {n_edges:,} edges (top {SPARSITY:.1%})", flush=True)

    wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=5000)
    sweep = []
    for f in FACTORS:
        if f == 1.0:
            rep = eval_mia(model, tok, test); ppl = base_ppl
        else:
            with scaled_weights(model, sel, f):
                rep = eval_mia(model, tok, test)
                ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=5000)
        dppl = (ppl - base_ppl) / base_ppl
        tag = ("baseline" if f == 1 else "REMOVE(x0)" if f == 0 else f"amplify x{f:g}")
        sweep.append({"factor": f, "ppl_delta": dppl, **rep})
        print(f"  {tag:12} | Δppl {dppl:+7.1%} | minkpp AUC {rep['minkpp']['auc']:.3f} "
              f"TPR@1% {rep['minkpp']['tpr1']:.3f} | zlib TPR@1% {rep['zlib']['tpr1']:.3f} "
              f"TPR@5% {rep['zlib']['tpr5']:.3f} | mink TPR@5% {rep['mink']['tpr5']:.3f}",
              flush=True)

    RESULTS.mkdir(exist_ok=True)
    tag = MODEL_ID.split("/")[-1].replace(".", "").lower()
    (RESULTS / f"mia_defend_{tag}.json").write_text(json.dumps(
        {"model": MODEL_ID, "split": SPLIT, "n_edges": n_edges, "sparsity": SPARSITY,
         "sweep": sweep, "env": env_info()}, indent=2))
    print(f"saved results/mia_defend_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
