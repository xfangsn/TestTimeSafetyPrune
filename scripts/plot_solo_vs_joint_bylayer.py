"""Variant of plot_solo_vs_joint: solo bars ordered by LAYER ID (ascending) instead of by pick-rate,
flatter, saved to figures/solo_vs_joint_llama_bylayer.* (keeps the original figure intact).

Panels 1-6 are the A/B behaviors (metric = A/B pick-rate, chance 0.5); panels 7-8 are refusal
(refusal rate) and uncertainty (hedge rate) from results/solo_vs_joint_llama_extra.json -- these
have no A/B chance level, so the 0.5 line is drawn only on the first six."""
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
AB = ["power-seeking", "deception", "self-rate-highly",
      "self-awareness", "wealth-seeking", "sycophancy"]
EXTRA = {"refusal": "refusal (refusal rate)", "uncertainty": "uncertainty (hedge rate)"}

extra_p = RESULTS / "solo_vs_joint_llama_extra.json"
if extra_p.exists():
    DATA.update(json.loads(extra_p.read_text())["results"])
else:
    print(f"note: {extra_p} missing -- plotting the 6 A/B behaviors only")
ORDER = AB + [b for b in EXTRA if b in DATA]

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

# common x-span: bar width is meaningless, so every panel gets the same data-unit
# extent (widest panel = most solo layers + the joint bar), keeping bars equal-width.
XMAX = max(len(DATA[b]["solo"]) for b in ORDER) + 0.4 + 0.7
ncol = 4 if len(ORDER) > 6 else 3
figsize = (14.5, 2.9) if ncol == 4 else (11, 2.9)
fig, axes = plt.subplots(2, ncol, figsize=figsize, sharey=True)
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
    if beh in AB:                      # chance level is an A/B-only notion
        ax.axhline(0.5, ls=(0, (1, 1.5)), color=CHANCE_C, lw=1.6, zorder=2)
    for xp, v in list(zip(xs, vals)) + [(xj, r["joint"])]:
        if v < 0.02:              # zero-height bar: annotate so the result stays visible
            ax.text(xp, 0.035, f"{v:.2f}", ha="center", va="bottom", fontsize=10, color="#444")
    ax.set_xticks(xs + [xj])
    ax.set_xticklabels([str(l) for l in layers] + ["All"], fontsize=12)
    ax.get_xticklabels()[-1].set_color(JOINT)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(-0.7, XMAX)
    ax.set_ylim(0, 1.04)          # keep a baseline at 1.0 (refusal) off the axis edge
    ax.set_title(EXTRA.get(beh, beh), pad=4, fontweight="bold")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, ls="-", lw=0.5, color="#DfDfDf", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False)
for ax in axes.flat[len(ORDER):]:      # unused cells in the grid
    ax.set_visible(False)

fig.supylabel("behavior rate", fontsize=16, x=0.035)   # single shared y-axis title
for ax in axes[1, :]:
    if ax.get_visible():
        ax.set_xlabel("selected layer id")

handles = [
    Patch(fc=SOLO_LEG, ec="white", lw=0.8, label="single layer"),
    Patch(fc=JOINT, ec="white", lw=1.0, label="all L*  (BLADE)"),
    Line2D([0], [0], ls=(0, (5, 2)), color=BASE_C, lw=1.3, label="baseline"),
    Line2D([0], [0], ls=(0, (1, 1.5)), color=CHANCE_C, lw=1.6, label="no behavioral preference (A/B)"),
]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True,
           fancybox=True, edgecolor="grey", facecolor="white",
           bbox_to_anchor=(0.5, 1.08), fontsize=14)
fig.tight_layout(rect=[0.0, 0, 1, 0.95])
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"solo_vs_joint_llama_bylayer.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/solo_vs_joint_llama_bylayer.png / .pdf")
