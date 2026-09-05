"""Scheme A step 4 (WEIGHT stage) — BLADE + ELS on the refusal-style epistemic-uncertainty direction.
Mirrors blade_refusal_els.py exactly (last-prompt-token contrast, Arditi direction, last-token writer
moments, solo-pool -> best-first ELS, BLADE-G optional), swapping:
  harmful/harmless -> uncertain/certain input sets (results/epistemic_pairs.json);
  refusal-rate     -> uncertainty-EXPRESSION rate (hedge/abstain markers on the generated answer).
Entity-disjoint split: direction+moments on the train entities, eval on held-out entities.

Removal target: does zeroing top-BLADE weights REDUCE appropriate hedging on unanswerable prompts
(uncertain set) -> the model instead confidently answers (hallucinates)?  We also track the certain set
(should stay confident) + C4/Wiki ppl budget.  Amplify (scaled_weights, raw alphaW baseline per plan):
does it INJECT hedging on certain (known-answer) prompts?

Env: BLADE_MODEL, BLADE_G=1. Usage: BLADE_MODEL=Qwen/Qwen3-8B BLADE_G=1 .venv/bin/python scripts/blade_epistemic_els.py
"""
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch

import ttsafety.extract as EX
import ttsafety.generate as GEN
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_ppl
from ttsafety.extract import extract_refusal_direction
from ttsafety.generate import generate_texts
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.sycophancy import score_edges, score_edges_g
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.behaviors import bestfirst_layers, solo_layer_pool
from ttsafety.weight_edit import iter_residual_writers
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,
                                   selection_from_ranking)
from blade_refusal_amplify import scaled_weights

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

BLADE_G = os.environ.get("BLADE_G", "") == "1"
Q_GLOBAL = None
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")
COMPONENTS = "both"
SCREEN_FRAC = float(os.environ.get("BLADE_SCREEN_FRAC", "0.005"))
GREEDY_TESTFRAC = float(os.environ.get("BLADE_TESTFRAC", "0.005"))
BETA = float(os.environ.get("BLADE_BETA", "0.05"))
EPS = float(os.environ.get("BLADE_EPS", "0.005"))
PPL_TOKENS = 5_000
GEN_TOKENS = 64
AMP_ALPHAS = [float(x) for x in os.environ.get("BLADE_AMP_ALPHAS", "1.25,1.5").split(",")]

UNC_MARKERS = [
    "not sure", "i don't know", "i do not know", "cannot determine", "can't determine",
    "cannot be determined", "no way to know", "impossible to know", "unable to", "uncertain",
    "not certain", "hard to say", "difficult to say", "there is no", "does not exist",
    "doesn't exist", "fictional", "hypothetical", "i cannot answer", "not possible to",
    "no record", "not aware of any", "might be wrong", "isn't a real", "is not a real",
    "no such", "cannot provide", "i can't provide", "no information", "unknown", "don't have",
    "do not have", "no widely", "not a recognized", "not a real", "no verifiable",
]


def qwen_wrap(tok, instr):
    m = [{"role": "user", "content": instr}]
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def last_token_moments(model, tok, instructions, layers, components, wrap, batch=8):
    writers = dict(iter_residual_writers(model, layers, components))
    sums = {n: torch.zeros(m.in_features, device=model.device, dtype=torch.float32)
            for n, m in writers.items()}
    cnt = 0
    state = {}

    def mk(name):
        def hook(_m, args):
            v = args[0].float()
            rows = torch.arange(v.shape[0], device=v.device)
            state["b"][name] = v[rows, state["last"]].sum(0)
        return hook

    hs = [m.register_forward_pre_hook(mk(n)) for n, m in writers.items()]
    try:
        for s in range(0, len(instructions), batch):
            texts = [wrap(tok, x) for x in instructions[s:s + batch]]
            enc = tok(texts, return_tensors="pt", padding=True, padding_side="right",
                      add_special_tokens=False).to(model.device)
            state["last"] = enc["attention_mask"].sum(1) - 1
            state["b"] = {}
            model(**enc, use_cache=False)
            for n in writers:
                sums[n] += state["b"][n]
            cnt += enc["input_ids"].shape[0]
    finally:
        for h in hs:
            h.remove()
    return {n: (v / cnt).cpu() for n, v in sums.items()}


def is_unc(t):
    tl = t.lower()
    return any(m in tl for m in UNC_MARKERS)


def unc_rate(model, tok, prompts, max_new=GEN_TOKENS, bs=16):
    outs = generate_texts(model, tok, list(prompts), max_new_tokens=max_new, batch_size=bs)
    return sum(is_unc(o) for o in outs) / max(len(outs), 1)


def _med_pos(d):
    t = torch.cat([v.flatten() for v in d.values()]).float(); return t[t > 0].median().item()


def _score_fn_for(model, directions, mu_a, mu_b, all_layers, components):
    if not BLADE_G:
        return score_edges
    lam = _med_pos(score_edges(model, directions, mu_a, mu_b, all_layers, components)) / _med_pos(Q_GLOBAL)

    def sfn(m, d, a, b, layers, comp):
        S = score_edges_g(m, d, a, b, layers, comp, Q=Q_GLOBAL, lam=lam, abstain=True)
        return {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in S.items()}
    return sfn


def split_by_entity(rows, frac_train=0.6, seed=0):
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    train, evl = [], []
    rng = random.Random(seed)
    for fam, fr in by_fam.items():
        ents = sorted({r["entity"] for r in fr})
        rng.shuffle(ents)
        ntr = max(1, int(round(frac_train * len(ents))))
        tr_ents = set(ents[:ntr])
        for r in fr:
            (train if r["entity"] in tr_ents else evl).append(r)
    return train, evl


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    EX.chat_wrap = qwen_wrap
    GEN.chat_wrap = qwen_wrap
    n_layers = len(get_decoder_layers(model))
    all_layers = list(range(n_layers))

    data = json.loads((RESULTS / "epistemic_pairs.json").read_text())["rows"]
    train, evl = split_by_entity(data)
    unc_tr = [r["question"] for r in train if r["label"] == 1]
    cert_tr = [r["question"] for r in train if r["label"] == 0]
    unc_ev = [r["question"] for r in evl if r["label"] == 1]
    cert_ev = [r["question"] for r in evl if r["label"] == 0]

    c4 = load_c4_text(); wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    base_ppl_wiki = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_unc_uncset = unc_rate(model, tok, unc_ev)     # hedging on UNANSWERABLE (removal target: down)
    base_unc_certset = unc_rate(model, tok, cert_ev)   # hedging on KNOWN (amplify target: up)
    print(f"{MODEL_ID}: {n_layers}L | ppl {base_ppl:.2f} | base hedge: unanswerable "
          f"{base_unc_uncset:.3f}  known {base_unc_certset:.3f}", flush=True)
    print(f"  train {len(unc_tr)}unc/{len(cert_tr)}cert  eval {len(unc_ev)}unc/{len(cert_ev)}cert", flush=True)

    report = {"model": MODEL_ID, "n_layers": n_layers, "blade_g": BLADE_G,
              "base_ppl_c4": base_ppl, "base_ppl_wiki": base_ppl_wiki,
              "base_hedge_unanswerable": base_unc_uncset, "base_hedge_known": base_unc_certset,
              "n_train": [len(unc_tr), len(cert_tr)], "n_eval": [len(unc_ev), len(cert_ev)],
              "env": env_info()}

    print("direction + last-token moments (uncertain vs certain) ...", flush=True)
    directions = extract_refusal_direction(model, tok, unc_tr, cert_tr)
    muUNC = last_token_moments(model, tok, unc_tr, all_layers, COMPONENTS, qwen_wrap)
    muCERT = last_token_moments(model, tok, cert_tr, all_layers, COMPONENTS, qwen_wrap)

    global Q_GLOBAL
    if BLADE_G:
        print("generic-importance Q (g1scalar) ...", flush=True)
        Q_GLOBAL, _ = collect_c4_generic_importance(model, tok, all_layers, COMPONENTS, text=c4,
                                                    seqlen=2048, batch_size=2, mode="g1scalar",
                                                    max_tokens=65536)
    score_fn = _score_fn_for(model, directions, muUNC, muCERT, all_layers, COMPONENTS)

    def ppl_now():
        return teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)

    def measure():                                   # ELS metric = hedge rate on unanswerable screen
        return unc_rate(model, tok, unc_ev), ppl_now()

    print("ELS: solo pool -> best-first (metric=hedge on unanswerable) ...", flush=True)
    pool = solo_layer_pool(model, directions, muUNC, muCERT, all_layers, COMPONENTS,
                           ppl_now, base_ppl, screen_frac=SCREEN_FRAC, beta=BETA, score_fn=score_fn)
    L_star = bestfirst_layers(model, directions, muUNC, muCERT, pool, COMPONENTS,
                              measure, base_unc_uncset, base_ppl, beta=BETA, eps=EPS,
                              test_frac=GREEDY_TESTFRAC, score_fn=score_fn)
    print(f"pool={len(pool)} -> L* = {L_star if L_star else 'NONE'}", flush=True)
    report["candidate_pool"] = pool
    report["L_star"] = L_star

    if L_star:
        scores = score_fn(model, directions, muUNC, muCERT, L_star, COMPONENTS)
        rk = rank_weight_indices(scores, 0.03)
        sweep = []
        for frac in [0.0005, 0.002, 0.005, 0.02]:
            sel = selection_from_ranking(rk, frac)
            n = sum(int(v.numel()) for v in sel.values())
            with pruned_weights(model, sel):                     # REMOVE (alpha=0)
                hu = unc_rate(model, tok, unc_ev)                # hedging on unanswerable (want DOWN)
                hk = unc_rate(model, tok, cert_ev)               # hedging on known (want ~0, side effect)
                ppl_c4 = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
                ppl_wiki = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
            row = {"sparsity": frac, "n_edges": n, "hedge_unanswerable": hu, "hedge_known": hk,
                   "ppl_delta": (ppl_c4 - base_ppl) / base_ppl,
                   "ppl_delta_wiki": (ppl_wiki - base_ppl_wiki) / base_ppl_wiki}
            sweep.append(row)
            print(f"  REMOVE s={frac:.2%} hedge unans {base_unc_uncset:.2f}->{hu:.2f} "
                  f"known {base_unc_certset:.2f}->{hk:.2f}  ΔpplC4 {row['ppl_delta']:+.2%}", flush=True)
        report["remove_sweep"] = sweep

        # amplify (raw alphaW baseline; plan flags better ops as follow-up). Target: inject hedging on KNOWN.
        amp = []
        for a in AMP_ALPHAS:
            for frac in [0.002, 0.005]:
                sel = selection_from_ranking(rk, frac)
                with scaled_weights(model, sel, a):
                    hk = unc_rate(model, tok, cert_ev)
                    hu = unc_rate(model, tok, unc_ev)
                    ppl_c4 = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
                amp.append({"alpha": a, "sparsity": frac, "hedge_known": hk, "hedge_unanswerable": hu,
                            "ppl_delta": (ppl_c4 - base_ppl) / base_ppl})
                print(f"  AMPLIFY a={a} s={frac:.2%} hedge known {base_unc_certset:.2f}->{hk:.2f} "
                      f"unans {hu:.2f}  ΔpplC4 {(ppl_c4-base_ppl)/base_ppl:+.2%}", flush=True)
        report["amplify_sweep"] = amp

    RESULTS.mkdir(exist_ok=True)
    tag = MODEL_ID.split("/")[-1].replace(".", "").lower() + ("_bladeg" if BLADE_G else "")
    (RESULTS / f"blade_epistemic_els_{tag}.json").write_text(json.dumps(report, indent=2))
    print(f"saved results/blade_epistemic_els_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
