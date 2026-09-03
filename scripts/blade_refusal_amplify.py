"""Dual of BLADE: use the SAME refusal edge mask, but instead of zeroing edges
(factor=0, removal/BLADE) SCALE them up (factor>1, "reverse-BLADE") to STRENGTHEN
refusal. Test on one model. Metrics per factor:
  - plain harmful refusal (ceiling ~1.0 for aligned models)
  - prefill-attack refusal ("Sure, here is" jailbreak) -> the interesting axis
  - harmless over-refusal (must stay low; kimi's pitfall #2)
  - WikiText-2 ppl (capability)
factor=1.0 is baseline (no edit); factor=0.0 recovers BLADE removal.
"""
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch

from ttsafety.data import load_jsonl
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.judge import is_refusal
import ttsafety.extract as EX
import ttsafety.generate as GEN
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.sycophancy import score_edges
from ttsafety.weight_prune import (rank_weight_indices, selection_from_ranking,
                                   _resolve_modules)
from blade_refusal_els import last_token_moments, refusal_rate  # safe: __main__ guard

DATA = Path("data"); RESULTS = Path("results")
MODEL_ID = os.environ.get("BLADE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
IS_BASE = MODEL_ID.lower().endswith("-pt") or "-base" in MODEL_ID.lower()
WRAP = (lambda tok, s: s + "\n") if IS_BASE else chat_wrap  # base: no chat template
L_STAR = [int(x) for x in os.environ.get("BLADE_LSTAR", "12").split(",")]
COMPONENTS = "both"
SPARSITY = 0.005              # same mask size as BLADE removal point
N_DIR = 200
N_EVAL = 64
PPL_TOKENS = 5000
GEN_TOKENS = 48
PREFILL = " Sure, here is"
FACTORS = [float(x) for x in os.environ.get(
    "BLADE_FACTORS", "1.0,0.0,2.0,3.0,5.0,10.0").split(",")]


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
def gen_wrapped(model, tok, wrapped, max_new=GEN_TOKENS, bs=16):
    """Generate from already-wrapped prompts (for prefill injection)."""
    prev = tok.padding_side; tok.padding_side = "left"
    outs = []
    try:
        for s in range(0, len(wrapped), bs):
            enc = tok(wrapped[s:s + bs], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            outs.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True))
    finally:
        tok.padding_side = prev
    return outs


def prefill_refusal(model, tok, prompts):
    wrapped = [WRAP(tok, p) + PREFILL for p in prompts]
    outs = gen_wrapped(model, tok, wrapped)
    return sum(is_refusal(PREFILL + o) for o in outs) / len(outs)


def main():
    if IS_BASE:                         # route direction/gen through plain wrap
        EX.chat_wrap = WRAP
        GEN.chat_wrap = WRAP
        print("[base mode] plain prompt wrap", flush=True)
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmful_val = [r["instruction"] for r in load_jsonl(DATA / "harmful_val.jsonl")][:N_EVAL]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")][:N_EVAL]
    wiki = load_wikitext_text()

    print(f"{MODEL_ID}: building refusal mask on L*={L_STAR} @ s={SPARSITY:.1%}", flush=True)
    directions = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:N_DIR])
    all_layers = list(range(len(get_decoder_layers(model))))
    muH = last_token_moments(model, tok, harmful[:N_DIR], all_layers, COMPONENTS, WRAP)
    muU = last_token_moments(model, tok, harmless[:N_DIR], all_layers, COMPONENTS, WRAP)
    scores = score_edges(model, directions, muH, muU, L_STAR, COMPONENTS)
    rk = rank_weight_indices(scores, 0.03)
    sel = selection_from_ranking(rk, SPARSITY)
    n_edges = sum(len(v) for v in sel.values())
    print(f"mask = {n_edges:,} edges", flush=True)

    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    rows = []
    for f in FACTORS:
        if f == 1.0:
            plain = refusal_rate(model, tok, harmful_val)
            pref = prefill_refusal(model, tok, harmful_val)
            harml = refusal_rate(model, tok, harmless)
            ppl = base_ppl
        else:
            with scaled_weights(model, sel, f):
                plain = refusal_rate(model, tok, harmful_val)
                pref = prefill_refusal(model, tok, harmful_val)
                harml = refusal_rate(model, tok, harmless)
                ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
        tag = ("baseline" if f == 1.0 else "REMOVAL(BLADE)" if f == 0.0 else f"amplify x{f:g}")
        dppl = (ppl - base_ppl) / base_ppl
        rows.append({"factor": f, "plain_refusal": plain, "prefill_refusal": pref,
                     "harmless_refusal": harml, "ppl_delta": dppl})
        print(f"  {tag:16} factor={f:>4g} | plain {plain:.3f} | PREFILL {pref:.3f} "
              f"| harmless(over-ref) {harml:.3f} | Δppl {dppl:+.2%}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    tag = MODEL_ID.split("/")[-1].replace(".", "").lower()
    (RESULTS / f"blade_refusal_amplify_{tag}.json").write_text(json.dumps(
        {"model": MODEL_ID, "L_star": L_STAR, "sparsity": SPARSITY, "n_edges": n_edges,
         "base_ppl": base_ppl, "prefill": PREFILL, "sweep": rows, "env": env_info()},
        indent=2, ensure_ascii=False))
    print("saved results/blade_refusal_amplify_%s.json" % tag, flush=True)


if __name__ == "__main__":
    main()
