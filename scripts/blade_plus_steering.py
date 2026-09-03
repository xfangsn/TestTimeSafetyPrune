"""BLADE + activation steering (CAA) on sycophancy: complementary or redundant?
Conditions: baseline | steer-only (base + CAA) | BLADE-only | BLADE+steer.
Steering vector = per-layer mean-diff of block output (matching - not_matching answer
spans) at a chosen mid layer; hook subtracts coef*v from that layer's residual for all
positions. Metric: A/B pick-rate (sycophancy) + held-out WikiText ppl. See
docs/plan-blade-plus-steering.md.
"""
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch

from ttsafety.behaviors import (collect_span_input_moments, extract_direction,
                                fetch_ab, make_splits, pick_rate, score_edges)
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,
                                   selection_from_ranking)

DATA = Path("data"); RESULTS = Path("results")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
EOT = "<|eot_id|>"; COMPONENTS = "both"; PPL_TOKENS = 5000; RHO = 0.005
COEFS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
LAYER_SCAN = [8, 12, 14, 18]


@contextmanager
def steering(model, layer_idx, vec, coef):
    """Subtract coef*vec from decoder layer_idx's residual output (all positions)."""
    if coef == 0.0 or vec is None:
        yield; return
    layer = get_decoder_layers(model)[layer_idx]
    v = vec.to(model.device)

    def hook(_m, _inp, out):
        if isinstance(out, tuple):
            out[0].add_(-coef * v.to(out[0].dtype))
            return out
        return out - coef * v.to(out.dtype)

    h = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def syco_L_star():
    """sycophancy L* at rho=0.005 from the C4-calibrated run (fallback: beta5)."""
    for suf in ("_beta5_c4", "_beta5"):
        p = RESULTS / f"blade_els_llama-32-3b-instruct{suf}.json"
        if p.exists():
            r = json.loads(p.read_text())["results"].get("sycophancy", {})
            if r.get("L_star"):
                return r["L_star"], suf
    return [12, 15], "fallback"


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    all_layers = list(range(len(get_decoder_layers(model))))
    wiki = load_wikitext_text()
    sp = make_splits(fetch_ab("sycophancy", DATA / "behaviors"))
    val, train = sp["val"], sp["train"]
    rate_m, _ = pick_rate(model, tok, val, "matching")
    side = "matching" if rate_m >= 0.5 else "not_matching"
    other = "not_matching" if side == "matching" else "matching"

    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_pick = pick_rate(model, tok, val, side)[0]
    print(f"side={side} base pick {base_pick:.3f} base ppl {base_ppl:.2f}", flush=True)

    # --- steering vectors (base model): per-layer mean-diff of block output ---
    dirs_base = extract_direction(model, tok, train, side, eot=EOT)

    def meas(v, coef, lidx, sel=None):
        cm_prune = pruned_weights(model, sel) if sel else _noop()
        with cm_prune:
            with steering(model, lidx, v, coef):
                pk = pick_rate(model, tok, val, side)[0]
                pl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
        return pk, (pl - base_ppl) / base_ppl

    # --- pick best steering layer on the base model (coef=1) ---
    scan = {}
    for l in LAYER_SCAN:
        pk, dp = meas(dirs_base[l], 1.0, l)
        scan[l] = {"pick": pk, "ppl_delta": dp}
        print(f"[layer-scan] L{l} c=1  pick {pk:.3f}  Δppl {dp:+.1%}", flush=True)
    ls = min(LAYER_SCAN, key=lambda l: scan[l]["pick"])   # deepest removal
    print(f"chosen steering layer = L{ls}", flush=True)

    # --- BLADE selection (sycophancy L* @ rho=0.005) ---
    L_star, src = syco_L_star()
    mu_a = collect_span_input_moments(model, tok, train, side, all_layers, COMPONENTS, eot=EOT)
    mu_b = collect_span_input_moments(model, tok, train, other, all_layers, COMPONENTS, eot=EOT)
    scores = score_edges(model, dirs_base, mu_a, mu_b, L_star, COMPONENTS)
    rk = rank_weight_indices(scores, max(RHO, 0.01))
    sel = selection_from_ranking(rk, RHO)
    n_edges = sum(int(v.numel()) for v in sel.values())
    with pruned_weights(model, sel):
        blade_pick = pick_rate(model, tok, val, side)[0]
        blade_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
        dirs_blade = extract_direction(model, tok, train, side, eot=EOT)   # V1 vector on edited model
    blade_dppl = (blade_ppl - base_ppl) / base_ppl
    print(f"BLADE-only (L*={L_star} src={src}) pick {blade_pick:.3f} Δppl {blade_dppl:+.1%} "
          f"({n_edges:,} edges)", flush=True)

    rows = {"baseline": {"pick": base_pick, "ppl_delta": 0.0},
            "blade_only": {"pick": blade_pick, "ppl_delta": blade_dppl},
            "steer_only": [], "blade_steer_v1": [], "blade_steer_v2": []}
    for c in COEFS:
        pk, dp = meas(dirs_base[ls], c, ls)
        rows["steer_only"].append({"coef": c, "pick": pk, "ppl_delta": dp})
        print(f"[steer-only]    c={c:<4g} pick {pk:.3f} Δppl {dp:+.1%}", flush=True)
    for c in COEFS:
        pk, dp = meas(dirs_blade[ls], c, ls, sel=sel)          # V1: vector recomputed on edited model
        rows["blade_steer_v1"].append({"coef": c, "pick": pk, "ppl_delta": dp})
        print(f"[BLADE+steer V1] c={c:<4g} pick {pk:.3f} Δppl {dp:+.1%}", flush=True)
    for c in COEFS:
        pk, dp = meas(dirs_base[ls], c, ls, sel=sel)           # V2: base vector on edited model
        rows["blade_steer_v2"].append({"coef": c, "pick": pk, "ppl_delta": dp})
        print(f"[BLADE+steer V2] c={c:<4g} pick {pk:.3f} Δppl {dp:+.1%}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "blade_plus_steering_sycophancy.json").write_text(json.dumps(
        {"model": MODEL_ID, "side": side, "steer_layer": ls, "layer_scan": scan,
         "L_star": L_star, "L_star_src": src, "rho": RHO, "n_edges": n_edges,
         "base_ppl": base_ppl, "rows": rows, "env": env_info()}, indent=2))
    print("saved results/blade_plus_steering_sycophancy.json", flush=True)


@contextmanager
def _noop():
    yield


if __name__ == "__main__":
    main()
