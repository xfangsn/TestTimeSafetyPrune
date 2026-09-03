"""Two honest Pareto views of strengthening OOD refusal (HarmBench 200), all methods selected on
in-dist val (leakage-safe). LEFT: OOD refusal vs BENIGN over-refusal (XSTest) -- the selective-
safety trade-off. RIGHT: OOD refusal vs WikiText ppl change (symlog) -- the capability cost.
Eligible points (<=beta ppl AND benign<=base+5pp) solid; ineligible hollow; frozen c*/alpha* ringed.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

R = Path("results"); FIG = Path("figures")
D = json.loads((R / "blade_steering_baselines.json").read_text())
# BLADE overlay = ELS-selected L14 (rho=0.002, BLADE's own best-first layer selection),
# alpha selected on in-dist val; leakage-safe and faithful (layer not hand-picked)
B = json.loads((R / "blade_amplify_els_configs.json").read_text())
BETA = D["config"]["BETA"] * 100
base_ood = D["base"]["ood_refusal"]
base_benign = D["base"]["benign"] * 100
BENIGN_LIMIT = base_benign + D["config"]["L_BENIGN"] * 100

STEER = {"caa": ("CAA (Rimsky 2024)", "#3A7CA5", "o"),
         "arditi": ("Refusal-dir add (Arditi 2024)", "#B0779E", "^"),
         "actadd": ("ActAdd (Turner 2023)", "#C9A06A", "D"),
         "iti": ("ITI (Li 2023)", "#6A994E", "s")}

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"xtick.top": False, "ytick.right": False,
                     "xtick.labelsize": 12, "ytick.labelsize": 12})
fig, (axB, axP) = plt.subplots(1, 2, figsize=(12.4, 5.2))


def series(rows, xkey, scale):
    xs = [r[xkey] * scale for r in rows]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [rows[i] for i in order]


def draw(ax, rows, xkey, scale, color, marker, label, lw, ms, z):
    rows = series(rows, xkey, scale)
    xs = [r[xkey] * scale for r in rows]; ys = [r["ood_refusal"] for r in rows]
    ax.plot(xs, ys, "-", color=color, lw=lw, alpha=0.8, zorder=z, label=label)
    for x, y, r in zip(xs, ys, rows):
        elig = r.get("report_eligible_wiki_full_xstest", True)
        lo, hi = r["ci"]
        ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="none", ecolor=color,
                    elinewidth=1, alpha=0.4, zorder=z)
        ax.plot([x], [y], marker=marker, color=color, ms=ms, zorder=z + 1,
                markerfacecolor=(color if elig else "white"), markeredgecolor=color, markeredgewidth=1.6)
        if r.get("is_cstar") or r.get("is_astar"):
            ax.plot([x], [y], marker="o", ms=ms + 7, mfc="none", mec=color, mew=2.0, zorder=z + 2)


for ax, xkey, scale in [(axB, "benign_full", 100), (axP, "wiki_relppl", 100)]:
    for key, (label, color, marker) in STEER.items():
        draw(ax, D["methods"][key]["report"], xkey, scale, color, marker, label, 1.6, 7, 3)
    draw(ax, B["report"], xkey, scale, "#EE6C4D", "o", "BLADE amplify (ours)", 3.0, 10, 6)

# LEFT: benign axis
axB.scatter([base_benign], [base_ood], s=120, facecolors="white", edgecolors="#3D405B", lw=2, zorder=8)
axB.annotate("baseline", (base_benign, base_ood), xytext=(6, -12), textcoords="offset points", fontsize=10)
axB.axvline(BENIGN_LIMIT, ls="--", color="#c44", lw=1.2, alpha=0.6)
axB.text(BENIGN_LIMIT + 0.5, 0.02, f"benign budget\n(base+5pp)", fontsize=8.5, color="#c44")
axB.set_xlabel("XSTest over-refusal rate (%)  —  false refusals of SAFE prompts", fontsize=12.5)
axB.set_ylabel("OOD refusal rate (HarmBench)", fontsize=13.5)
axB.set_title("Selective safety: refuse attacks, not safe prompts", fontsize=12.5, fontweight="bold")

# RIGHT: ppl axis
axP.axhline(base_ood, ls=":", color="#888", lw=1)
axP.axvline(BETA, ls="--", color="#c44", lw=1.2, alpha=0.6)
axP.text(BETA * 1.05, 0.02, f"β={BETA:g}% ppl", fontsize=8.5, color="#c44")
axP.set_xscale("symlog", linthresh=0.5)
axP.set_xlabel("WikiText perplexity change (%, symlog)  —  capability cost", fontsize=12.5)
axP.set_title("Capability cost", fontsize=12.5, fontweight="bold")

for ax in (axB, axP):
    ax.set_ylim(0, 0.55)
    ax.spines[["top", "right"]].set_visible(False)
axB.legend(fontsize=9, frameon=True, edgecolor="grey", facecolor="white", loc="upper right")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"steering_baselines_pareto.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/steering_baselines_pareto.png / .pdf")
