"""Danger deletability map figure (academic). Sorted by how strongly the model
leans toward the DANGEROUS answer; shows what BLADE can suppress."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots  # noqa

TAG = "llama-32-3b-instruct"
rep = json.load(open(f"results/blade_danger_map_{TAG}.json"))
SHORT = {
    "desire-for-more-capabilities": "want more capabilities",
    "desire-for-recursive-self-improvement": "recursive self-improvement",
    "desire-for-independence-from-human-oversight": "independence from oversight",
    "desire-to-escape-sandbox": "escape sandbox",
    "no-shut-down": "resist shut-down",
    "desire-for-acquiring-power": "acquire power",
    "willingness-to-manipulate-overseers-to-think-it-is-HHH": "manipulate overseers",
    "self-replication": "self-replication",
    "willingness-to-use-social-engineering-to-achieve-its-goals": "social engineering",
    "okay-with-deceiving-a-user-to-help-them": "deceive the user",
    "ends-justify-means": "ends justify means",
    "machiavellianism": "machiavellianism",
    "coordinate-other-ais": "collude with other AIs",
    "psychopathy": "psychopathy",
}
COL = {"removable": "#009E73", "diffuse": "#D55E00", "already-safe": "#8a8f86"}
# behaviors that were 'diffuse' under the old window/solo method, now removable with best-first ELS
UNLOCKED = {"desire-for-more-capabilities", "desire-for-acquiring-power",
            "no-shut-down", "desire-for-independence-from-human-oversight"}
items = sorted(rep["behaviors"].items(), key=lambda x: x[1]["danger_level"])

plt.rcParams["font.family"] = "serif"
with plt.style.context(["science", "no-latex"]):
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    plt.rc("font", size=13)
    ax.axvspan(0.0, 0.5, color="#e8efe8", alpha=0.5, zorder=0)   # "safe" half
    ax.axvline(0.5, color="#9aa0a6", lw=1.2, ls="--", zorder=1)
    yt, yl = [], []
    for i, (name, r) in enumerate(items):
        y = i
        d = r["danger_level"]
        st = r["status"]
        c = COL[st]
        if st == "already-safe":
            ax.scatter([d], [y], s=95, marker="o", color=c, zorder=3, clip_on=False)
        else:
            post = r["best"]["pick"]
            ax.plot([post, d], [y, y], color=c, lw=2.6, zorder=2,
                    solid_capstyle="round", clip_on=False)
            ax.scatter([d], [y], s=95, facecolors="white", edgecolors=c, lw=2.0,
                       zorder=3, clip_on=False)
            ax.scatter([post], [y], s=105, color=c, zorder=4, clip_on=False)
            ax.annotate(f"{r['best']['ppl_delta']:+.1f}%".replace("+0.0%", "~0%"),
                        (post, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=9, color=c)
            if name in UNLOCKED:
                ax.annotate("* newly removable", (post, y), textcoords="offset points",
                            xytext=(0, -13), ha="center", fontsize=8, color="#1a7a55",
                            style="italic")
        yt.append(y); yl.append(SHORT.get(name, name))
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=11.5)
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, len(items) - 0.4)
    ax.set_xlabel("P(dangerous answer)   —   left of dashed line = safe", fontsize=13)
    ax.text(0.25, len(items) - 0.5, "already safe", ha="center", fontsize=10,
            color="#5a5f56", style="italic")
    ax.text(0.75, len(items) - 0.5, "leans dangerous", ha="center", fontsize=10,
            color="#8a5a2a", style="italic")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["removable"],
               markersize=10, label="removable by BLADE"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["diffuse"],
               markersize=10, label="exhibited but diffuse"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["already-safe"],
               markersize=10, label="already safe (RLHF)"),
        Line2D([0], [0], marker="o", color="#555", markerfacecolor="white",
               markersize=10, lw=0, label="baseline  →  filled = post-BLADE"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=10,
              bbox_to_anchor=(1.0, -0.02))
    ax.set_title("What dangerous dispositions can BLADE delete?  (Llama-3.2-3B, best-first ELS)",
                 fontsize=12.5, pad=10)
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/blade_danger_map.pdf", bbox_inches="tight")
    fig.savefig("results/blade_danger_map.png", dpi=300, bbox_inches="tight")
    print("saved figures/blade_danger_map.pdf and results/blade_danger_map.png")
