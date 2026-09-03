"""Llama base vs instruct on the BLADE behaviors: these behaviors are
post-training artifacts. Horizontal dumbbell per behavior: base (open) ->
instruct (filled), chance/target reference. House style. No title (caption in paper)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots  # noqa: F401

RESULTS = Path("results"); FIG = Path("figures")
D = json.loads((RESULTS / "base_vs_instruct_llama.json").read_text())["rows"]
BASE_C = "#8FA9BE"; INST_C = "#EE6C4D"

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"xtick.top": False, "ytick.right": False})

rows = [r for r in D if r["behavior"] != "refusal"] + [r for r in D if r["behavior"] == "refusal"]
labels = [r["behavior"] for r in rows]
y = list(range(len(rows)))[::-1]

fig, ax = plt.subplots(figsize=(5.8, 4.3))
for yi, r in zip(y, rows):
    chance = 0.0 if r["behavior"] == "refusal" else 0.5
    ax.plot([chance, chance], [yi - 0.36, yi + 0.36], color="#c2c6bf", lw=1.4, zorder=1)
    ax.annotate("", xy=(r["instruct"], yi), xytext=(r["base"], yi),
                arrowprops=dict(arrowstyle="-|>", color=INST_C, lw=3.0,
                                shrinkA=0, shrinkB=0, mutation_scale=16),
                zorder=2, annotation_clip=False)
    ax.scatter([r["base"]], [yi], s=150, facecolors="white", edgecolors=BASE_C, lw=2.4,
               zorder=3, clip_on=False)
    ax.scatter([r["instruct"]], [yi], s=165, facecolors=INST_C, edgecolors=INST_C, lw=0,
               zorder=4, clip_on=False)

ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=17)
ax.set_xlim(-0.03, 1.05)
ax.set_xlabel("behavior score  (A/B pick-rate;  refusal: refusal-rate)", fontsize=16)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.tick_params(axis="x", labelsize=15)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.margins(y=0.08)

handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=BASE_C,
           markeredgewidth=2.4, markersize=12, label="base (pretrained)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=INST_C, markersize=13, label="instruct"),
    Line2D([0], [0], color="#c2c6bf", lw=2.2, label="chance / target"),
]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
          frameon=False, fontsize=14, handletextpad=0.4, columnspacing=1.2)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"base_vs_instruct_llama.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/base_vs_instruct_llama.png / .pdf")
