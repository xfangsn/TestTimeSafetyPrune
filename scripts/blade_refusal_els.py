"""Refusal cross-model via BLADE + Effective-Layer Selection.

Refusal uses a different contrast/metric than the A/B behaviors:
  * contrast  = harmful (AdvBench) vs harmless (Alpaca) prompts, LAST prompt token.
  * direction = mean-diff of last-token residual (Arditi-style), all layers.
  * moments   = mean last-token writer input, harmful vs harmless.
  * metric    = refusal RATE on generated answers to harmful prompts (keyword judge).
ELS then screens each layer (single-layer prune -> Δrefusal, Δppl) and full BLADE
runs on the effective layers L*. Works on Llama / Qwen (thinking off) / Gemma.

Env: BLADE_MODEL. Output: results/blade_refusal_els_<tag>.json
"""

import json
import os
from pathlib import Path

import torch

import ttsafety.extract as EX
import ttsafety.generate as GEN
from ttsafety.data import load_jsonl
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_ppl
from ttsafety.extract import extract_refusal_direction
from ttsafety.generate import generate_texts
from ttsafety.hooks import get_decoder_layers
from ttsafety.judge import is_refusal
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.sycophancy import score_edges, score_edges_g
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.behaviors import bestfirst_layers, solo_layer_pool
from ttsafety.weight_edit import iter_residual_writers
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,
                                   selection_from_ranking)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

BLADE_G = os.environ.get("BLADE_G", "") == "1"   # primary method: score_edges_g (g1scalar)
Q_GLOBAL = None                                   # generic-importance Q on all layers (set in main)


def _med_pos(d):
    t = torch.cat([v.flatten() for v in d.values()]).float(); return t[t > 0].median().item()


def _score_fn_for(model, directions, mu_a, mu_b, all_layers, components):
    """score_edges (base) or a BLADE-G closure with lambda fixed on all layers."""
    if not BLADE_G:
        return score_edges
    lam = _med_pos(score_edges(model, directions, mu_a, mu_b, all_layers, components)) / _med_pos(Q_GLOBAL)

    def sfn(m, d, a, b, layers, comp):
        S = score_edges_g(m, d, a, b, layers, comp, Q=Q_GLOBAL, lam=lam, abstain=True)
        return {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in S.items()}
    return sfn


MODEL_ID = os.environ.get("BLADE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
COMPONENTS = "both"
SCREEN_FRAC = float(os.environ.get("BLADE_SCREEN_FRAC", "0.005"))
GREEDY_TESTFRAC = float(os.environ.get("BLADE_TESTFRAC", "0.005"))
BETA = float(os.environ.get("BLADE_BETA", "0.05"))
EPS = float(os.environ.get("BLADE_EPS", "0.005"))
PPL_TOKENS = 5_000
N_DIR = 200          # prompts for direction + moments
N_SCREEN = 48        # harmful prompts for per-layer screen
GEN_TOKENS = 48


def qwen_wrap(tok, instr):
    m = [{"role": "user", "content": instr}]
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
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


def refusal_rate(model, tok, prompts, max_new=GEN_TOKENS, bs=16):
    outs = generate_texts(model, tok, prompts, max_new_tokens=max_new, batch_size=bs)
    return sum(is_refusal(o) for o in outs) / len(outs)


def main():
    mid = MODEL_ID.lower()
    is_qwen = "qwen" in mid
    is_base = mid.endswith("-pt") or "-base" in mid
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    wrap = chat_wrap
    if is_qwen:                       # disable thinking for direction/moments/gen
        wrap = qwen_wrap
        EX.chat_wrap = qwen_wrap
        GEN.chat_wrap = qwen_wrap
    if is_base:                       # pretrained: no chat template, plain prompt
        base_wrap = lambda tokenizer, s: s + "\n"
        wrap = base_wrap
        EX.chat_wrap = base_wrap
        GEN.chat_wrap = base_wrap
        print("[base mode] plain prompt wrap", flush=True)
    n_layers = len(get_decoder_layers(model))
    all_layers = list(range(n_layers))

    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmful_val = [r["instruction"] for r in load_jsonl(DATA / "harmful_val.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    c4 = load_c4_text()            # calibration for the ppl budget
    wiki = load_wikitext_text()    # held-out ppl for reporting
    base_ppl = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    base_ppl_wiki = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    screen = harmful_val[:N_SCREEN]
    base_ref = refusal_rate(model, tok, screen)
    print(f"{MODEL_ID}: {n_layers} layers | base ppl {base_ppl:.2f} "
          f"| base refusal {base_ref:.3f}", flush=True)

    report = {"model": MODEL_ID, "n_layers": n_layers, "base_ppl_c4": base_ppl,
              "base_ppl": base_ppl_wiki, "calibration": "c4", "report_ppl": "wikitext",
              "base_refusal": base_ref, "env": env_info()}
    if base_ref < 0.5:
        print("  baseline refusal < 0.5 — model barely refuses AdvBench; "
              "not localizable.", flush=True)
        report["note"] = "low baseline refusal"
        RESULTS.mkdir(exist_ok=True)
        tag = MODEL_ID.split("/")[-1].replace(".", "").lower() + ("_bladeg" if BLADE_G else "") + os.environ.get("BLADE_OUTSUFFIX", "")
        (RESULTS / f"blade_refusal_els_{tag}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False))
        return

    print("extracting refusal direction + last-token moments (all layers) ...",
          flush=True)
    directions = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:N_DIR])
    muH = last_token_moments(model, tok, harmful[:N_DIR], all_layers, COMPONENTS, wrap)
    muU = last_token_moments(model, tok, harmless[:N_DIR], all_layers, COMPONENTS, wrap)

    global Q_GLOBAL
    if BLADE_G:
        print("computing generic-importance Q (g1scalar) on all layers ...", flush=True)
        Q_GLOBAL, _qm = collect_c4_generic_importance(model, tok, all_layers, COMPONENTS, text=c4,
                                                      seqlen=2048, batch_size=2, mode="g1scalar",
                                                      max_tokens=65536)
    score_fn = _score_fn_for(model, directions, muH, muU, all_layers, COMPONENTS)

    # ELS: solo pool (candidate filter) -> best-first joint selection.
    # Metric = refusal rate on the screen set (lower = more removed).
    def ppl_now():                 # C4 = calibration signal
        return teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)

    def measure():
        return refusal_rate(model, tok, screen), ppl_now()

    print("ELS: solo pool -> best-first (metric=refusal on screen) ...", flush=True)
    pool = solo_layer_pool(model, directions, muH, muU, all_layers, COMPONENTS,
                           ppl_now, base_ppl, screen_frac=SCREEN_FRAC, beta=BETA, score_fn=score_fn)
    L_star = bestfirst_layers(model, directions, muH, muU, pool, COMPONENTS,
                              measure, base_ref, base_ppl, beta=BETA, eps=EPS,
                              test_frac=GREEDY_TESTFRAC, score_fn=score_fn)
    print(f"pool={len(pool)} within ppl budget -> L* = {L_star if L_star else 'NONE'}",
          flush=True)
    report["candidate_pool"] = pool
    report["L_star"] = L_star

    if L_star:
        scores = score_fn(model, directions, muH, muU, L_star, COMPONENTS)
        rk = rank_weight_indices(scores, 0.03)
        sweep = []
        for frac in [0.0005, 0.002, 0.005, 0.02]:
            sel = selection_from_ranking(rk, frac)
            n = sum(int(v.numel()) for v in sel.values())
            with pruned_weights(model, sel):
                ref = refusal_rate(model, tok, harmful_val)  # full 64
                ppl_c4 = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
                ppl_wiki = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
            sweep.append({"sparsity": frac, "refusal": ref, "n_edges": n,
                          "ppl_delta": (ppl_c4 - base_ppl) / base_ppl,               # C4 (budget)
                          "ppl_delta_wiki": (ppl_wiki - base_ppl_wiki) / base_ppl_wiki})  # report
            print(f"  s={frac:.3%} refusal {ref:.3f} ΔpplC4 "
                  f"{(ppl_c4 - base_ppl) / base_ppl:+.2%} ΔpplWiki "
                  f"{(ppl_wiki - base_ppl_wiki) / base_ppl_wiki:+.2%}", flush=True)
        report["Lstar_sweep"] = sweep
        within = [s for s in sweep if s["ppl_delta"] <= BETA]   # budget on C4
        best = min(within or sweep, key=lambda s: s["refusal"])
        report["best_within_budget"] = bool(within)
        flag = "" if within else "  [INFEASIBLE: no point within C4 budget]"
        print(f"best: refusal {base_ref:.2f}→{best['refusal']:.2f} "
              f"@ C4 {best['ppl_delta']:+.2%} / Wiki {best['ppl_delta_wiki']:+.2%} ppl{flag}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    tag = MODEL_ID.split("/")[-1].replace(".", "").lower() + ("_bladeg" if BLADE_G else "") + os.environ.get("BLADE_OUTSUFFIX", "")
    (RESULTS / f"blade_refusal_els_{tag}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved results/blade_refusal_els_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
