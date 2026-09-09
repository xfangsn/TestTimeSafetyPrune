"""Solo-vs-joint ablation for BLADE's selected layers -- REFUSAL + UNCERTAINTY.

Extends scripts/solo_vs_joint.py (6 A/B behaviors) with the two non-A/B
behaviors on Llama-3.2-3B: refusal (harmful-prompt refusal rate) and epistemic
uncertainty (hedge rate on unanswerable questions). Same protocol: prune EACH
ELS-selected layer's BLADE-G weights ALONE and compare to pruning ALL of L*
jointly at the SAME operating rho -- does one layer suffice, or do the layers
act synergistically?

Method is always BLADE-G (g1scalar), the primary method shown as "BLADE":
score_edges_g(..., Q, lam, abstain=True) with lam fixed on ALL layers.

Output: results/solo_vs_joint_llama_extra{OUT_TAG}.json, same per-behavior
schema as results/solo_vs_joint_llama.json plus a "metric" field.

Env: BLADE_MODEL, L_STAR (pin uncertainty L*), UNC_RUN_ELS=1 (run ELS instead),
     OUT_TAG (output filename suffix).
Usage: PYTHONPATH=src:scripts .venv/bin/python scripts/solo_vs_joint_extra.py
"""
import json
import os
from pathlib import Path

import torch

import ttsafety.extract as EX
import ttsafety.generate as GEN
from ttsafety.behaviors import bestfirst_layers, score_edges, solo_layer_pool
from ttsafety.data import load_jsonl
from ttsafety.eval import load_c4_text, teacher_forced_ppl
from ttsafety.extract import extract_refusal_direction
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.sycophancy import score_edges_g
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,
                                   selection_from_ranking)

# sibling scripts are imported by bare module name (PYTHONPATH=src:scripts)
from blade_refusal_els import last_token_moments as refusal_moments
from blade_refusal_els import refusal_rate
from blade_epistemic_els import last_token_moments as unc_moments
from blade_epistemic_els import qwen_wrap, unc_rate
from blade_epistemic_p0 import split_3way

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; RESULTS = ROOT / "results"

MODEL_ID = os.environ.get("BLADE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
COMPONENTS = "both"
BETA = 0.05           # ppl budget (C4, calibration signal)
PPL_TOKENS = 5_000
N_DIR = 200           # prompts for direction + moments
N_SCREEN = 48         # harmful prompts for the refusal measurement set
REF_GEN_TOKENS = 48
UNC_GEN_TOKENS = 64
UNC_RHOS = [0.0005, 0.002, 0.005, 0.02]          # remove sweep for the uncertainty operating rho
UNC_L_STAR = os.environ.get("L_STAR", "14,1")    # published Llama uncertainty selection
UNC_RUN_ELS = os.environ.get("UNC_RUN_ELS", "") == "1"
OUT_TAG = os.environ.get("OUT_TAG", "")

MODEL = None          # set in main(); pruned_rate/_lam close over them
TOK = None
Q_GLOBAL = None       # generic-importance Q (g1scalar) on ALL layers, computed ONCE


def _med_pos(d):
    t = torch.cat([v.flatten() for v in d.values()]).float(); return t[t > 0].median().item()


def _lam(directions, mu_a, mu_b, all_layers):
    """BLADE-G Lagrange multiplier, fixed on ALL layers (as in blade_*_els.py)."""
    return (_med_pos(score_edges(MODEL, directions, mu_a, mu_b, all_layers, COMPONENTS))
            / _med_pos(Q_GLOBAL))


def pruned_rate(directions, mu_a, mu_b, layers, rho, measure_fn, lam):
    """Mirror of solo_vs_joint.pruned_pick: BLADE-G scores on `layers`, finite-mask,
    zero the top-rho weights, then evaluate measure_fn() inside the prune context."""
    S = score_edges_g(MODEL, directions, mu_a, mu_b, layers, COMPONENTS,
                      Q=Q_GLOBAL, lam=lam, abstain=True)
    S = {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in S.items()}
    sel = selection_from_ranking(rank_weight_indices(S, max(rho, 0.01)), rho)
    with pruned_weights(MODEL, sel):
        return measure_fn()


def operating_rho(sweep, metric):
    """Sweep point that minimizes `metric` within the beta ppl budget (fallback: all)."""
    ok = [r for r in sweep if r["ppl_delta"] <= BETA] or sweep
    return min(ok, key=lambda r: r[metric])["sparsity"]


def _summary(beh, rec):
    solo = rec["solo"]
    print(f"{beh:16} rho={rec['rho']} base={rec['base']:.3f} joint={rec['joint']:.3f} "
          f"solo_min={min(solo.values()):.3f} L*={rec['L_star']}", flush=True)


# ---------------------------------------------------------------- refusal ----
def run_refusal(all_layers):
    """harmful vs harmless last-token contrast; metric = refusal rate on harmful_val."""
    EX.chat_wrap = GEN.chat_wrap = chat_wrap     # Llama: default chat template (as blade_refusal_els)
    tag = MODEL_ID.split("/")[-1].replace(".", "").lower() + "_bladeg"
    src = json.loads((RESULTS / f"blade_refusal_els_{tag}.json").read_text())
    L_star = src["L_star"]                        # BLADE-G ELS selection (currently [13])
    rho = operating_rho(src["Lstar_sweep"], "refusal")

    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmful_val = [r["instruction"] for r in load_jsonl(DATA / "harmful_val.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    screen = harmful_val[:N_SCREEN]               # measurement set

    directions = extract_refusal_direction(MODEL, TOK, harmful[:N_DIR], harmless[:N_DIR])
    mu_a = refusal_moments(MODEL, TOK, harmful[:N_DIR], all_layers, COMPONENTS, chat_wrap)
    mu_b = refusal_moments(MODEL, TOK, harmless[:N_DIR], all_layers, COMPONENTS, chat_wrap)
    lam = _lam(directions, mu_a, mu_b, all_layers)

    def measure():
        return refusal_rate(MODEL, TOK, screen, max_new=REF_GEN_TOKENS)

    base = measure()
    solo = {l: pruned_rate(directions, mu_a, mu_b, [l], rho, measure, lam) for l in L_star}
    joint = pruned_rate(directions, mu_a, mu_b, sorted(L_star), rho, measure, lam)
    return {"side": "refusal", "metric": "refusal_rate", "rho": rho, "L_star": L_star,
            "base": base, "base_refusal_src": src.get("base_refusal"),
            "solo": solo, "joint": joint}


# ------------------------------------------------------------ uncertainty ----
def run_uncertainty(all_layers, c4, base_ppl):
    """uncertain vs certain last-token contrast; metric = hedge rate on held-out unanswerable."""
    EX.chat_wrap = GEN.chat_wrap = qwen_wrap      # how the Llama uncertainty runs were done
    rows = json.loads((RESULTS / "epistemic_pairs_v2.json").read_text())["rows"]
    tr, sel, test = split_3way(rows)              # entity-disjoint 3-way split
    unc_tr = [r["question"] for r in tr if r["label"] == 1]
    cert_tr = [r["question"] for r in tr if r["label"] == 0]
    unc_sel = [r["question"] for r in sel if r["label"] == 1]     # ELS selection metric
    unc_test = [r["question"] for r in test if r["label"] == 1]   # untouched eval
    print(f"uncertainty split: dir_train {len(tr)} | els_select {len(sel)} | eval {len(test)} "
          f"({len(unc_test)} unanswerable)", flush=True)

    directions = extract_refusal_direction(MODEL, TOK, unc_tr, cert_tr)
    mu_a = unc_moments(MODEL, TOK, unc_tr, all_layers, COMPONENTS, qwen_wrap)
    mu_b = unc_moments(MODEL, TOK, cert_tr, all_layers, COMPONENTS, qwen_wrap)
    lam = _lam(directions, mu_a, mu_b, all_layers)

    def sfn(m, d, a, b, layers, comp):            # BLADE-G scorer for ELS
        S = score_edges_g(m, d, a, b, layers, comp, Q=Q_GLOBAL, lam=lam, abstain=True)
        return {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in S.items()}

    def ppl_now():
        return teacher_forced_ppl(MODEL, TOK, c4, max_tokens=PPL_TOKENS)

    def measure():                                # hedge rate on the untouched eval set
        return unc_rate(MODEL, TOK, unc_test, max_new=UNC_GEN_TOKENS)

    if UNC_RUN_ELS:                               # same call shape as blade_rho_sweep.els(0.005, 0.005)
        def els_measure():
            return unc_rate(MODEL, TOK, unc_sel, max_new=UNC_GEN_TOKENS), ppl_now()
        base_sel = els_measure()[0]
        pool = solo_layer_pool(MODEL, directions, mu_a, mu_b, all_layers, COMPONENTS,
                               ppl_now, base_ppl, screen_frac=0.005, beta=BETA, score_fn=sfn)
        L_star = bestfirst_layers(MODEL, directions, mu_a, mu_b, pool, COMPONENTS,
                                  els_measure, base_sel, base_ppl, beta=BETA, eps=0.005,
                                  test_frac=0.005, score_fn=sfn)
        print(f"ELS: pool={len(pool)} -> L*={L_star}", flush=True)
    else:
        L_star = [int(x) for x in UNC_L_STAR.split(",")]
        print(f"PINNED L*={L_star} (L_STAR env; ELS skipped)", flush=True)

    # operating rho: REMOVE (alpha=0) sweep on L*, minimize hedge within the ppl budget
    remove_sweep = []
    for frac in UNC_RHOS:
        def sweep_measure():
            return unc_rate(MODEL, TOK, unc_test, max_new=UNC_GEN_TOKENS), ppl_now()
        hedge, ppl = pruned_rate(directions, mu_a, mu_b, sorted(L_star), frac, sweep_measure, lam)
        remove_sweep.append({"sparsity": frac, "hedge": hedge,
                             "ppl_delta": (ppl - base_ppl) / base_ppl})
        print(f"  REMOVE s={frac:.2%} hedge {hedge:.3f} ΔpplC4 "
              f"{(ppl - base_ppl) / base_ppl:+.2%}", flush=True)
    rho = operating_rho(remove_sweep, "hedge")

    base = measure()
    solo = {l: pruned_rate(directions, mu_a, mu_b, [l], rho, measure, lam) for l in L_star}
    joint = pruned_rate(directions, mu_a, mu_b, sorted(L_star), rho, measure, lam)
    return {"side": "uncertainty", "metric": "hedge_rate", "rho": rho, "L_star": L_star,
            "base": base, "solo": solo, "joint": joint, "remove_sweep": remove_sweep}


def main():
    global MODEL, TOK, Q_GLOBAL
    MODEL, TOK = load_model(MODEL_ID)
    if TOK.pad_token is None:
        TOK.pad_token = TOK.eos_token
    all_layers = list(range(len(get_decoder_layers(MODEL))))
    c4 = load_c4_text()
    base_ppl = teacher_forced_ppl(MODEL, TOK, c4, max_tokens=PPL_TOKENS)
    print(f"{MODEL_ID}: {len(all_layers)} layers | base ppl {base_ppl:.2f}", flush=True)

    print("computing generic-importance Q (g1scalar) on all layers ...", flush=True)
    Q_GLOBAL, _ = collect_c4_generic_importance(MODEL, TOK, all_layers, COMPONENTS, text=c4,
                                                seqlen=2048, batch_size=2, mode="g1scalar",
                                                max_tokens=65536)

    out = {}
    out["refusal"] = run_refusal(all_layers)
    _summary("refusal", out["refusal"])
    out["uncertainty"] = run_uncertainty(all_layers, c4, base_ppl)
    _summary("uncertainty", out["uncertainty"])

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"solo_vs_joint_llama_extra{OUT_TAG}.json"
    p.write_text(json.dumps({"model": MODEL_ID, "beta": BETA, "results": out,
                             "env": env_info()}, indent=2))
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
