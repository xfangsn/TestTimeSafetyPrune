"""Re-plot solo-vs-joint in the house style learned from SSW / FaPriST:
scienceplots base, muted bars w/ thin edges, one highlighted 'ALL' bar (ours),
baseline + chance as dashed/dotted refs, minimal text. Reads the saved JSON.
"""
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

# solo bars: mint -> deep-teal gradient; joint: coral accent (ours)
SOLO_CMAP = LinearSegmentedColormap.from_list("solo", ["#BCE3D6", "#3AA6A0", "#0E6E6E"])
SOLO_LEG = "#3AA6A0"
JOINT = "#EE6C4D"     # coral = ours (all L*)
BASE_C = "#3D405B"    # baseline (dark slate)
CHANCE_C = "#E1A730"  # chance / target (warm gold)

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": False, "ytick.right": False,
    "axes.linewidth": 1.0, "figure.dpi": 150,
})

fig, axes = plt.subplots(2, 3, figsize=(11, 3.6), sharey=True)
for ax, beh in zip(axes.flat, ORDER):
    r = DATA[beh]
    # sort solo layers by pick-rate descending -> staircase down to JOINT
    items = sorted(r["solo"].items(), key=lambda kv: -kv[1])
    layers = [int(l) for l, _ in items]
    vals = [v for _, v in items]
    n = len(layers)
    xs = list(range(n))
    # gradient deepens as the bar gets lower (more removed)
    cols = [SOLO_CMAP(0.15 + 0.85 * (i / max(1, n - 1))) for i in range(n)]
    ax.bar(xs, vals, color=cols, edgecolor="white", linewidth=0.8, width=0.74, zorder=3)
    xj = n + 0.4
    ax.bar([xj], [r["joint"]], color=JOINT, edgecolor="white", linewidth=1.0,
           width=0.74, zorder=4)
    ax.axhline(r["base"], ls=(0, (5, 2)), color=BASE_C, lw=1.3, zorder=2)
    ax.axhline(0.5, ls=(0, (1, 1.5)), color=CHANCE_C, lw=1.6, zorder=2)
    # minimal labels
    ax.set_xticks(xs + [xj])
    ax.set_xticklabels([str(l) for l in layers] + ["All"], fontsize=9)
    ax.get_xticklabels()[-1].set_color(JOINT)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_ylim(0, 1.0)
    ax.set_title(beh, pad=4, fontweight="bold")
    ax.margins(x=0.04)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, ls="-", lw=0.5, color="#DfDfDf", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False)

for ax in axes[:, 0]:
    ax.set_ylabel("A/B pick-rate")
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
           bbox_to_anchor=(0.5, 1.03), fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"solo_vs_joint_llama.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/solo_vs_joint_llama.png / .pdf")
