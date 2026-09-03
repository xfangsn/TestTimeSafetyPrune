"""Variant of plot_solo_vs_joint: solo bars ordered by LAYER ID (ascending) instead of by pick-rate,
flatter, saved to figures/solo_vs_joint_llama_bylayer.* (keeps the original figure intact)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import scienceplots  # noqa: F401

RESULTS = Path("results"); FIG = Path("figures")
DATA = json.loads((RESULTS / "solo_vs_joint_llama.json").read_text())["results"]
ORDER = ["power-seeking", "deception", "self-rate-highly",
         "self-awareness", "wealth-seeking", "sycophancy"]

SOLO_CMAP = LinearSegmentedColormap.from_list("solo", ["#BCE3D6", "#3AA6A0", "#0E6E6E"])
SOLO_LEG = "#3AA6A0"
JOINT = "#EE6C4D"
BASE_C = "#3D405B"
CHANCE_C = "#B8860B"

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 17,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": False, "ytick.right": False,
    "axes.linewidth": 1.0, "figure.dpi": 150,
})

fig, axes = plt.subplots(2, 3, figsize=(11, 2.9), sharey=True)
for ax, beh in zip(axes.flat, ORDER):
    r = DATA[beh]
    # order solo bars by LAYER ID ascending
    items = sorted(r["solo"].items(), key=lambda kv: int(kv[0]))
    layers = [int(l) for l, _ in items]
    vals = [v for _, v in items]
    n = len(layers)
    xs = list(range(n))
    # gradient deepens with layer id (left=shallow, right=deep)
    cols = [SOLO_CMAP(0.15 + 0.85 * (i / max(1, n - 1))) for i in range(n)]
    ax.bar(xs, vals, color=cols, edgecolor="white", linewidth=0.8, width=0.74, zorder=3)
    xj = n + 0.4
    ax.bar([xj], [r["joint"]], color=JOINT, edgecolor="white", linewidth=1.0,
           width=0.74, zorder=4)
    ax.axhline(r["base"], ls=(0, (5, 2)), color=BASE_C, lw=1.3, zorder=2)
    ax.axhline(0.5, ls=(0, (1, 1.5)), color=CHANCE_C, lw=1.6, zorder=2)
    ax.set_xticks(xs + [xj])
    ax.set_xticklabels([str(l) for l in layers] + ["All"], fontsize=12)
    ax.get_xticklabels()[-1].set_color(JOINT)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_ylim(0, 1.0)
    ax.set_title(beh, pad=4, fontweight="bold")
    ax.margins(x=0.04)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, ls="-", lw=0.5, color="#DfDfDf", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False)

fig.supylabel("A/B pick-rate", fontsize=16, x=0.035)   # single shared y-axis title
for ax in axes[1, :]:
    ax.set_xlabel("selected layer id")

handles = [
    Patch(fc=SOLO_LEG, ec="white", lw=0.8, label="single layer"),
    Patch(fc=JOINT, ec="white", lw=1.0, label="all L*  (BLADE)"),
    Line2D([0], [0], ls=(0, (5, 2)), color=BASE_C, lw=1.3, label="baseline"),
    Line2D([0], [0], ls=(0, (1, 1.5)), color=CHANCE_C, lw=1.6, label="no behavioral preference"),
]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True,
           fancybox=True, edgecolor="grey", facecolor="white",
           bbox_to_anchor=(0.5, 1.08), fontsize=14)
fig.tight_layout(rect=[0.0, 0, 1, 0.95])
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"solo_vs_joint_llama_bylayer.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/solo_vs_joint_llama_bylayer.png / .pdf")
