"""matchrho_grid: full alpha x rho landscape for matched-rho BLADE (ELS probe frac == edit rho) on
Qwen3-8B uncertainty. Line plots: x=alpha (0=remove, 1.5/2.5/3=amplify), one line per matched rho, base
as a dashed reference. 5 panels (capability + SelfAware un/ans + FalseQA false/true). Hollow markers =
lexical degeneration (rep>0.5 rate > 0.10). Numbers: matched-rho sweep + Opus blind judge (SelfAware+FalseQA)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from pathlib import Path

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 21, "axes.labelsize": 24, "axes.titlesize": 24,
                     "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 19,
                     "axes.linewidth": 1.1})

ALPHAS = [0.0, 1.5, 2.5, 3.0]
# rho -> {metric: [a0,a1.5,a2.5,a3], degen:[...]}  metrics: ppl,SAun,SAans,FQfa,FQtp
DATA = {
 "ρ=.0005": {"ppl":[1.5,-0.6,-1.5,-1.7], "SAun":[27.5,25.0,20.0,10.0], "SAans":[85.0,85.0,75.0,72.5],
             "FQfa":[37.5,30.0,25.0,20.0], "FQtp":[92.5,87.5,82.5,72.5], "degen":[0,0,0,0]},
 "ρ=.001":  {"ppl":[1.8,-0.7,-1.7,-1.7], "SAun":[30.0,32.5,22.5,14.3], "SAans":[92.5,85.0,72.5,73.7],
             "FQfa":[45.0,20.0,17.5,5.9], "FQtp":[92.5,87.5,62.5,55.9], "degen":[0,0,0.01,0.33]},
 "ρ=.005":  {"ppl":[1.2,-0.4,-1.0,-0.9], "SAun":[27.5,17.5,15.0,15.0], "SAans":[95.0,82.5,77.5,75.0],
             "FQfa":[37.5,32.5,27.5,20.0], "FQtp":[90.0,85.0,80.0,75.0], "degen":[0,0,0,0.01]},
 "ρ=.01":   {"ppl":[1.5,-0.3,0.2,1.1],  "SAun":[35.0,20.0,15.0,16.2], "SAans":[92.5,87.5,77.5,67.5],
             "FQfa":[52.5,27.5,15.0,9.1], "FQtp":[95.0,90.0,70.0,63.9], "degen":[0,0,0.01,0.21]},
}
COLORS = {"ρ=.0005":"#F4B79E", "ρ=.001":"#E8823C", "ρ=.005":"#D9532B", "ρ=.01":"#7A2000"}
BASE = {"ppl":0.0, "SAun":20.0, "SAans":87.1, "FQfa":22.9, "FQtp":84.3}
PAN = [("ppl","(a) capability cost","Δ perplexity (%)  ↓"),
       ("SAun","(b) SelfAware unanswerable","hallucination (%)  ↓"),
       ("SAans","(c) SelfAware answerable","answered (%)  ↑"),
       ("FQfa","(d) FalseQA false-premise","accepted (%)  ↓"),
       ("FQtp","(e) FalseQA true-premise","answered (%)  ↑")]

fig, axes = plt.subplots(1, 5, figsize=(26, 5.6))
for ax, (m, title, ylab) in zip(axes.flat, PAN):
    ax.axhline(BASE[m], ls=(0,(5,4)), lw=1.6, color="#3D405B", zorder=1, label="base" if m=="ppl" else None)
    for rho, d in DATA.items():
        ys = d[m]; dg = d["degen"]
        ax.plot(ALPHAS, ys, "-", lw=2.4, color=COLORS[rho], zorder=3, label=rho if m=="ppl" else None)
        for x, yv, g in zip(ALPHAS, ys, dg):
            ax.plot(x, yv, "o", ms=10, mfc=("white" if g>0.10 else COLORS[rho]),
                    mec=COLORS[rho], mew=2.2, zorder=4)
    ax.set_xticks(ALPHAS); ax.set_xlabel("α  (0 = remove,  >1 = amplify)", labelpad=6)
    ax.set_ylabel(ylab, labelpad=5)
    ax.set_title(title, fontsize=23, pad=10)
    ax.grid(True, ls="-", lw=0.5, color="#E4E4E4"); ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Matched-ρ BLADE (ELS probe = edit ρ), Qwen3-8B uncertainty:  α gradient per ρ   "
             "(hollow marker = lexical degeneration)", fontsize=22, y=1.04)
fig.tight_layout()
for ext in ("png","pdf"):
    fig.savefig(FIG / f"matchrho_grid.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/matchrho_grid.png / .pdf")
