"""Solo-vs-joint ablation for BLADE's selected layers on Llama-3.2-3B.

For each behavior, best-first ELS selected a layer set L*. Here we prune EACH
selected layer's BLADE weights ALONE (score_edges on [l], zero top-rho) and
compare to pruning ALL of L* jointly (score_edges on L*, zero top-rho) at the
SAME operating rho. Metric = A/B pick-rate (behavior side). Shows whether any
single selected layer suffices, or the layers act synergistically (joint << each
solo) -- the core justification for JOINT best-first selection.

Operating rho per behavior = the sweep point that minimizes pick-rate within the
beta=5% ppl budget (read from results/blade_els_llama-32-3b-instruct_beta5.json).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ttsafety.behaviors import (collect_span_input_moments, extract_direction,
                                fetch_ab, make_splits, pick_rate, score_edges)
from ttsafety.sycophancy import score_edges_g
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.eval import load_c4_text
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,
                                   selection_from_ranking)

DATA = Path("data"); RESULTS = Path("results"); FIG = Path("figures")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
EOT = "<|eot_id|>"; COMPONENTS = "both"; BETA = 0.05
# PRIMARY method is BLADE-G (g1scalar), shown as "BLADE" in the figure. Read its ELS selection.
SRC = RESULTS / "blade_els_llama-32-3b-instruct_beta5_bladeg.json"


def _med_pos(d):
    t = torch.cat([v.flatten() for v in d.values()]).float(); return t[t > 0].median().item()
BEHAVIORS = ["power-seeking", "deception", "self-rate-highly",
             "self-awareness", "wealth-seeking", "sycophancy"]


def operating_rho(sweep):
    ok = [r for r in sweep if r["ppl_delta"] <= BETA] or sweep
    best = min(ok, key=lambda r: r["pick_rate"])
    return best["sparsity"]


def main():
    src = json.loads(SRC.read_text())["results"]
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    all_layers = list(range(len(get_decoder_layers(model))))
    print("computing generic-importance Q (g1scalar) on all layers ...", flush=True)
    Q, _ = collect_c4_generic_importance(model, tok, all_layers, COMPONENTS, text=load_c4_text(),
                                         seqlen=2048, batch_size=2, mode="g1scalar", max_tokens=65536)
    out = {}

    for beh in BEHAVIORS:
        rec = src[beh]
        L_star = rec["L_star"]; side = rec["side"]   # BLADE-G (g1scalar) selection
        rho = operating_rho(rec["Lstar_sweep"])
        other = "not_matching" if side == "matching" else "matching"
        sp = make_splits(fetch_ab(beh, DATA / "behaviors"))
        val, train = sp["val"], sp["train"]
        directions = extract_direction(model, tok, train, side, eot=EOT)
        mu_a = collect_span_input_moments(model, tok, train, side, all_layers, COMPONENTS, eot=EOT)
        mu_b = collect_span_input_moments(model, tok, train, other, all_layers, COMPONENTS, eot=EOT)
        base = pick_rate(model, tok, val, side)[0]
        lam = _med_pos(score_edges(model, directions, mu_a, mu_b, all_layers, COMPONENTS)) / _med_pos(Q)

        def pruned_pick(cand, lam=lam):
            S = score_edges_g(model, directions, mu_a, mu_b, cand, COMPONENTS, Q=Q, lam=lam, abstain=True)
            S = {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in S.items()}
            sel = selection_from_ranking(rank_weight_indices(S, max(rho, 0.01)), rho)
            with pruned_weights(model, sel):
                return pick_rate(model, tok, val, side)[0]

        solo = {l: pruned_pick([l]) for l in L_star}
        joint = pruned_pick(sorted(L_star))
        out[beh] = {"side": side, "rho": rho, "L_star": L_star, "base": base,
                    "solo": solo, "joint": joint}
        print(f"{beh:16} rho={rho} base={base:.3f} joint={joint:.3f} "
              f"solo_min={min(solo.values()):.3f} L*={L_star}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "solo_vs_joint_llama.json").write_text(json.dumps(
        {"model": MODEL_ID, "beta": BETA, "results": out, "env": env_info()}, indent=2))

    # ---- figure: 2x3 panels, per behavior: solo bars + JOINT bar ----
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, beh in zip(axes.flat, BEHAVIORS):
        r = out[beh]
        layers = r["L_star"]                       # selection order
        vals = [r["solo"][l] for l in layers] + [r["joint"]]
        labels = [f"L{l}" for l in layers] + ["ALL\n(joint)"]
        colors = ["#9db4c0"] * len(layers) + ["#c1121f"]
        x = range(len(vals))
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.axhline(r["base"], ls="--", color="#333", lw=1, label=f"base {r['base']:.2f}")
        ax.axhline(0.5, ls=":", color="green", lw=1, label="chance 0.5")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{beh}  (ρ={r['rho']}, |L*|={len(layers)})", fontsize=10)
        ax.set_ylabel("A/B pick-rate")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Solo vs joint pruning of BLADE-selected layers (Llama-3.2-3B, best-first ELS β=5%)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = FIG / "solo_vs_joint_llama.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
