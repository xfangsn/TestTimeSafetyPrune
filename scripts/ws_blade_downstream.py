"""Held-out capability comparison for the BLADE vs Weight-Steering head-to-head:
evaluate the 6 zero-shot downstream tasks at the operating points where both
methods hit the OOD sycophancy floor (~0.25). This is the capability metric that
is NOT used in either method's selection (unlike WikiText ppl, which BLADE's ELS
budget optimizes -- so downstream is the non-circular, held-out check).

Points:
  base
  WS  scale in {0.5, 1.0}   (W_base - scale*task_vector, task_vector=syco-nonsyco FT)
  BLADE(from A/B, beta=5%, L*) rho in {0.002, 0.005}
"""
import json
from pathlib import Path

import torch

from ttsafety.behaviors import (collect_span_input_moments, extract_direction,
                                fetch_ab, make_splits, pick_rate, score_edges)
from ttsafety.downstream import TASKS, evaluate_task
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,
                                   selection_from_ranking)

DATA = Path("data"); RESULTS = Path("results"); FT = Path("data/ws_ft")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
EOT = "<|eot_id|>"; COMPONENTS = "both"
L_STAR = [12, 15, 10, 9, 3, 1, 4, 2, 16, 5, 6, 20]   # from ood_sycophancy_blade_ab.json (beta=5%)
BLADE_RHOS = [0.002, 0.005]
WS_SCALES = [0.5, 1.0]


def down(model, tok):
    r = {t: evaluate_task(model, tok, t)["acc"] for t in TASKS}
    r["mean"] = sum(r.values()) / len(TASKS)
    return r


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    all_layers = list(range(len(get_decoder_layers(model))))
    rows = []

    # --- base ---
    base_down = down(model, tok)
    print(f"BASE  down_mean {base_down['mean']:.4f}  {base_down}", flush=True)
    rows.append({"method": "base", "point": 0, **base_down})

    # --- BLADE: rebuild direction/moments, score L*, prune at rho ---
    sp = make_splits(fetch_ab("sycophancy", DATA / "behaviors"))
    val, train = sp["val"], sp["train"]
    rate_m, _ = pick_rate(model, tok, val, "matching")
    side = "matching" if rate_m >= 0.5 else "not_matching"
    other = "not_matching" if side == "matching" else "matching"
    directions = extract_direction(model, tok, train, side, eot=EOT)
    mu_a = collect_span_input_moments(model, tok, train, side, all_layers, COMPONENTS, eot=EOT)
    mu_b = collect_span_input_moments(model, tok, train, other, all_layers, COMPONENTS, eot=EOT)
    scores = score_edges(model, directions, mu_a, mu_b, L_STAR, COMPONENTS)
    rk = rank_weight_indices(scores, max(0.03, max(BLADE_RHOS)))
    for rho in BLADE_RHOS:
        sel = selection_from_ranking(rk, rho)
        n = sum(int(v.numel()) for v in sel.values())
        with pruned_weights(model, sel):
            d = down(model, tok)
        print(f"BLADE rho={rho:<6g} ({n:,} edges)  down_mean {d['mean']:.4f}  {d}", flush=True)
        rows.append({"method": "blade", "point": rho, "n_edges": n, **d})

    # --- WS: apply task vector, eval, restore ---
    base_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    syco = torch.load(FT / "llama32_syco.pt", map_location="cpu")
    nonsyco = torch.load(FT / "llama32_nonsyco.pt", map_location="cpu")
    tv = {k: (syco[k].float() - nonsyco[k].float()) for k in base_sd
          if k in syco and k in nonsyco and (syco[k].float() - nonsyco[k].float()).abs().sum() > 0}
    del syco, nonsyco
    for s in WS_SCALES:
        with torch.no_grad():
            for k, dv in tv.items():
                model.state_dict()[k].copy_((base_sd[k].float() - s * dv).to(base_sd[k].dtype))
        d = down(model, tok)
        print(f"WS scale={s:<4g}  down_mean {d['mean']:.4f}  {d}", flush=True)
        rows.append({"method": "weight-steering", "point": s, **d})
    with torch.no_grad():
        for k in tv:
            model.state_dict()[k].copy_(base_sd[k])

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "downstream_ws_vs_blade.json").write_text(json.dumps(
        {"model": MODEL_ID, "tasks": list(TASKS), "L_star": L_STAR,
         "rows": rows, "env": env_info()}, indent=2))
    print("saved results/downstream_ws_vs_blade.json", flush=True)


if __name__ == "__main__":
    main()
