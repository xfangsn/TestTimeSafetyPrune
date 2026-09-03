"""Compare S0/S1/S2 layer-selection strategies across 3 models (academic style)."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots  # noqa

TAGS = {"Llama-3.2-3B": "llama-32-3b-instruct", "Qwen3-4B": "qwen3-4b",
        "Gemma-3-4B": "gemma-3-4b-it"}
# S1 (fixed drop-edges window) is NOT data-driven -> dropped. All below are data-driven.
STRAT = [("S0_solo_ELS", "S0 solo-ELS (diagnostic)", "#8a8f86", "s"),
         ("S2_greedy_joint", "S2 ranked-greedy", "#D55E00", "^"),
         ("S3_bestfirst", "S3 best-first (final ELS)", "#009E73", "o")]

rows = []  # (label, baseline, {strat: pick or None}, is_group_top)
BEH_ORDER = ["sycophancy", "corrigibility", "power-seeking", "self-awareness",
             "wealth-seeking", "deception", "self-rate-highly"]
tally = {k: {"best_or_tie": 0, "n": 0} for k, *_ in STRAT}
for model, tag in TAGS.items():
    d = json.load(open(f"results/blade_layer_select_{tag}.json"))["behaviors"]
    first = True
    for beh in BEH_ORDER:
        if beh not in d or d[beh].get("status") == "not-exhibited":
            continue
        r = d[beh]
        picks = {}
        for key, *_ in STRAT:
            res = r[key]["result"]
            picks[key] = res["pick"] if (res and res["pick"] < r["baseline"]) else None
        # tally best-or-tie (lower pick = better; None = worst)
        vals = {k: (picks[k] if picks[k] is not None else 1.0) for k, *_ in STRAT}
        lo = min(vals.values())
        for k, *_ in STRAT:
            tally[k]["n"] += 1
            if vals[k] <= lo + 0.015:
                tally[k]["best_or_tie"] += 1
        rows.append((f"{model[:5]} · {beh}", r["baseline"], picks, first))
        first = False

print("=== S2 best-or-tie tally (within 0.015) ===")
for k, name, *_ in STRAT:
    print(f"  {name:18s}: {tally[k]['best_or_tie']}/{tally[k]['n']}")

plt.rcParams["font.family"] = "serif"
with plt.style.context(["science", "no-latex"]):
    fig, ax = plt.subplots(figsize=(8.8, 0.5 * len(rows) + 2.2))
    plt.rc("font", size=12)
    ax.axvline(0.5, color="#9aa0a6", ls="--", lw=1.2, zorder=1)
    ax.text(0.5, len(rows) - 0.3, "chance", fontsize=9.5, color="#7a7f76",
            ha="center", style="italic")
    yt, yl = [], []
    DODGE = {"S0_solo_ELS": 0.24, "S2_greedy_joint": 0.0, "S3_bestfirst": -0.24}
    for i, (label, base, picks, is_top) in enumerate(rows):
        y = len(rows) - 1 - i
        yt.append(y); yl.append(label)
        ax.axhline(y, color="#e6e8e4", lw=0.8, zorder=0)  # row guide
        ax.scatter([base], [y], s=90, facecolors="white", edgecolors="#333",
                   lw=1.6, zorder=3, clip_on=False)  # baseline (center)
        for key, name, col, mk in STRAT:
            yy = y + DODGE[key]
            p = picks[key]
            if p is None:  # failed / no removal
                ax.scatter([base + 0.02], [yy], s=75, marker="x", color=col,
                           lw=2.2, zorder=4, clip_on=False)
            else:
                ax.plot([p, base], [yy, y], color=col, lw=0.9, alpha=0.5, zorder=2)
                ax.scatter([p], [yy], s=100, marker=mk, color=col, zorder=5,
                           clip_on=False)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=10.5)
    ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("post-prune pick-rate  (← lower = behavior more removed)", fontsize=12.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                      markeredgecolor="#333", markersize=9, label="baseline")]
    handles += [Line2D([0], [0], marker=mk, color="w", markerfacecolor=col,
                       markersize=9, label=name) for _, name, col, mk in STRAT]
    handles += [Line2D([0], [0], marker="x", color="#666", markersize=9, lw=0,
                       label="failed (no removal)")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9.5,
              ncol=1, bbox_to_anchor=(1.0, -0.02))
    ax.set_title("Data-driven layer selection across 3 models\n"
                 "S3 best-first greedy = final ELS (best-or-tied in 15/17)",
                 fontsize=12.5, pad=10)
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/blade_layer_select.pdf", bbox_inches="tight")
    fig.savefig("results/blade_layer_select.png", dpi=300, bbox_inches="tight")
    print("saved figures/blade_layer_select.pdf")
