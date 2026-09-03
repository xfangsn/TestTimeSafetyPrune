"""Effective-Layer Selection (ELS): the data-driven, multi-layer step of BLADE
that picks which destination layers to prune, instead of a hand-picked window.

Two sub-steps:
  1. Solo screen (diagnostic/filter): prune each layer alone at SCREEN_FRAC and
     keep only layers whose solo prune stays within the ppl budget beta. This
     drops capability-critical layers (e.g. L0); it does NOT decide L*.
  2. Best-first greedy JOINT selection: repeatedly add the single layer that most
     reduces the joint behavior metric (pruning L*∪{l}), while ppl<=beta, until
     no layer improves by >= eps. Order-independent, captures synergy (layers
     useful only in combination). Hyperparameters: only beta (ppl budget) and
     eps (stop threshold). No top-k cap / fixed order / margin.

Full BLADE then sweeps sparsity on L*. If L* is empty the behavior is not
cleanly localizable in this model (entangled with capability).

Handles Llama (<|eot_id|>), Qwen (<|im_end|>, thinking off), Gemma (<end_of_turn>).
"""

import json
import os
from pathlib import Path

import torch

import ttsafety.behaviors as B
from ttsafety.behaviors import (bestfirst_layers, collect_span_input_moments,
                                extract_direction, fetch_ab, make_splits,
                                pick_rate, score_edges, solo_layer_pool)
from ttsafety.sycophancy import score_edges_g
from ttsafety.generic_importance import collect_c4_generic_importance

BLADE_G = os.environ.get("BLADE_G", "") == "1"   # primary method: score_edges_g (g1scalar)
Q_GLOBAL = None                                   # generic-importance Q on all layers (set in main)


def _med_pos(d):
    import torch
    t = torch.cat([v.flatten() for v in d.values()]).float(); return t[t > 0].median().item()


def _score_fn_for(model, directions, mu_a, mu_b, all_layers, components):
    """Return score_edges (base) or a BLADE-G closure with per-behaviour lambda (fixed on all layers)."""
    if not BLADE_G:
        return score_edges
    import torch
    c_all = score_edges(model, directions, mu_a, mu_b, all_layers, components)
    lam = _med_pos(c_all) / _med_pos(Q_GLOBAL)

    def sfn(m, d, a, b, layers, comp):
        S = score_edges_g(m, d, a, b, layers, comp, Q=Q_GLOBAL, lam=lam, abstain=True)
        return {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in S.items()}
    return sfn
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_ppl
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,
                                   selection_from_ranking)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-4B")
BEHAVIORS = os.environ.get(
    "BLADE_BEH", "self-awareness,wealth-seeking,power-seeking").split(",")
MIN_BIAS = float(os.environ.get("BLADE_MIN_BIAS", "0.10"))
COMPONENTS = "both"
SCREEN_FRAC = float(os.environ.get("BLADE_SCREEN_FRAC", "0.005"))   # solo-screen budget (pool filter)
GREEDY_TESTFRAC = float(os.environ.get("BLADE_TESTFRAC", "0.005"))  # test sparsity during best-first
BETA = float(os.environ.get("BLADE_BETA", "0.05"))  # max ppl cost (pool + selection + sweep)
EPS = float(os.environ.get("BLADE_EPS", "0.005"))   # min joint improvement to keep adding a layer
PPL_TOKENS = 5_000
SPARS = [float(x) for x in os.environ.get(
    "BLADE_SPARS", "0.0005,0.002,0.005,0.02").split(",")]  # final ρ sweep grid
RANK_MAXFRAC = min(0.10, max(0.03, max(SPARS)))  # ≥ max ρ, ≤ per_matrix_cap


def qwen_wrap(tok, instr):
    m = [{"role": "user", "content": instr}]
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)


def run_els(model, tok, name, is_qwen, eot, all_layers, c4, wiki, base_ppl, base_ppl_wiki):
    rows = fetch_ab(name, DATA / "behaviors")
    sp = make_splits(rows)
    rate_m, _ = pick_rate(model, tok, sp["val"], "matching")
    side = "matching" if rate_m >= 0.5 else "not_matching"
    other = "not_matching" if side == "matching" else "matching"
    base_pi, _ = pick_rate(model, tok, sp["val"], side)
    print(f"\n=== [{name}] pick(match)={rate_m:.3f} side={side} bias={base_pi:.3f} ===",
          flush=True)
    if abs(rate_m - 0.5) < MIN_BIAS:
        print("  skip (weak bias on this model)", flush=True)
        return {"behavior": name, "skipped": True, "baseline_bias": base_pi}

    directions = extract_direction(model, tok, sp["train"], side, eot=eot)
    mu_A = collect_span_input_moments(model, tok, sp["train"], side,
                                      all_layers, COMPONENTS, eot=eot)
    mu_B = collect_span_input_moments(model, tok, sp["train"], other,
                                      all_layers, COMPONENTS, eot=eot)
    score_fn = _score_fn_for(model, directions, mu_A, mu_B, all_layers, COMPONENTS)

    def ppl_now():                       # C4 = calibration signal for selection/budget
        return teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)

    def measure():
        return pick_rate(model, tok, sp["val"], side)[0], ppl_now()

    # ELS: solo screen -> candidate pool (diagnostic), then best-first joint selection
    pool = solo_layer_pool(model, directions, mu_A, mu_B, all_layers, COMPONENTS,
                           ppl_now, base_ppl, screen_frac=SCREEN_FRAC, beta=BETA, score_fn=score_fn)
    L_star = bestfirst_layers(model, directions, mu_A, mu_B, pool, COMPONENTS,
                              measure, base_pi, base_ppl, beta=BETA, eps=EPS,
                              test_frac=GREEDY_TESTFRAC, score_fn=score_fn)
    print(f"  pool={len(pool)} layers within ppl budget -> L* = "
          f"{L_star if L_star else 'NONE'}", flush=True)

    rec = {"behavior": name, "skipped": False, "side": side,
           "baseline_bias": base_pi, "candidate_pool": pool, "L_star": L_star}
    if L_star:
        scores = score_fn(model, directions, mu_A, mu_B, L_star, COMPONENTS)
        rk = rank_weight_indices(scores, RANK_MAXFRAC)
        sweep = []
        for frac in SPARS:
            sel = selection_from_ranking(rk, frac)
            n = sum(int(v.numel()) for v in sel.values())
            with pruned_weights(model, sel):
                pi, _ = pick_rate(model, tok, sp["val"], side)
                ppl_c4 = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
                ppl_wiki = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
            sweep.append({"sparsity": frac, "pick_rate": pi, "n_edges": n,
                          "ppl_delta": (ppl_c4 - base_ppl) / base_ppl,          # C4 (budget)
                          "ppl_delta_wiki": (ppl_wiki - base_ppl_wiki) / base_ppl_wiki})  # report
        rec["Lstar_sweep"] = sweep
        within = [s for s in sweep if s["ppl_delta"] <= BETA]   # budget on C4
        best = min(within or sweep, key=lambda s: s["pick_rate"])
        rec["best_within_budget"] = bool(within)
        flag = "" if within else "  [INFEASIBLE: no point within C4 budget]"
        print(f"  L* full BLADE best: π {best['pick_rate']:.3f} @ C4 {best['ppl_delta']:+.2%} / "
              f"WikiText {best['ppl_delta_wiki']:+.2%} ppl (s={best['sparsity']:.3%}){flag}", flush=True)
    return rec


def main():
    mid = MODEL_ID.lower()
    is_qwen = "qwen" in mid
    is_gemma = "gemma" in mid
    is_phi = "phi" in mid
    is_base = mid.endswith("-pt") or "-base" in mid  # pretrained (no chat template)
    eot = ("<|im_end|>" if is_qwen else "<end_of_turn>" if is_gemma
           else "<|end|>" if is_phi else "<|eot_id|>")
    if is_base:
        # base models have no chat template: use a plain prompt + the model's eos
        B.chat_wrap = lambda tokenizer, s: s + "\nAnswer:"
        eot = "<eos>"  # gemma/most SentencePiece bases
        print(f"[base mode] plain prompt wrap, eot={eot!r}", flush=True)
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if is_qwen:
        B.chat_wrap = qwen_wrap
    n_layers = len(get_decoder_layers(model))
    all_layers = list(range(n_layers))
    # calibrate the ppl budget on C4; report ppl change on held-out WikiText (Wanda-style)
    c4 = load_c4_text()
    wiki = load_wikitext_text()
    base_ppl_c4 = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    base_ppl_wiki = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    global Q_GLOBAL
    if BLADE_G:
        print('computing generic-importance Q (g1scalar) on all layers ...', flush=True)
        Q_GLOBAL, _qm = collect_c4_generic_importance(model, tok, all_layers, COMPONENTS, text=c4, seqlen=2048, batch_size=2, mode='g1scalar', max_tokens=65536)

    print(f"{MODEL_ID}: {n_layers} layers | base ppl C4 {base_ppl_c4:.3f} / "
          f"WikiText {base_ppl_wiki:.3f} | screen={SCREEN_FRAC:.3%}/layer | behaviors={BEHAVIORS}",
          flush=True)

    report = {"model": MODEL_ID, "n_layers": n_layers, "base_ppl_c4": base_ppl_c4,
              "base_ppl": base_ppl_wiki, "calibration": "c4", "report_ppl": "wikitext",
              "env": env_info(), "results": {}}
    for name in BEHAVIORS:
        report["results"][name] = run_els(model, tok, name, is_qwen, eot, all_layers,
                                          c4, wiki, base_ppl_c4, base_ppl_wiki)

    RESULTS.mkdir(exist_ok=True)
    tag = MODEL_ID.split("/")[-1].replace(".", "").lower()
    if os.environ.get("BLADE_BETA"):   # beta-sweep: dedicated file, avoid clobber
        tag = f"{tag}_beta{int(BETA*100)}"
    if BLADE_G:
        tag += "_bladeg"
    tag += os.environ.get("BLADE_OUTSUFFIX", "")   # dedicated output, avoid clobber
    (RESULTS / f"blade_els_{tag}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nsaved results/blade_els_{tag}.json", flush=True)
    print("\n=== summary ===", flush=True)
    for name, r in report["results"].items():
        if r.get("skipped"):
            print(f"{name:16s} bias {r['baseline_bias']:.2f}  (weak, skipped)")
        else:
            b = min(r["Lstar_sweep"], key=lambda s: s["pick_rate"]) if r.get("Lstar_sweep") else None
            bs = (f"π→{b['pick_rate']:.2f}@{b['ppl_delta']:+.1%}") if b else "no clean removal"
            print(f"{name:16s} bias {r['baseline_bias']:.2f}  L*={r['L_star']}  {bs}")


if __name__ == "__main__":
    main()
