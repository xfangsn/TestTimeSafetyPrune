"""uncertainty_method_cmp_llama: same house style as uncertainty_method_cmp, on meta-llama/Llama-3.2-3B-Instruct.
Configs: base / ITI (c=1) / ITI (c=2) / BLADE (alpha=1.75) / BLADE (alpha=2.0). SA/FQ n=70, Opus-judged
(ACT taxonomy). Capability panel = C4 teacher-forced Delta-ppl (ITI from results/iti_ppl_llama-3.2-3b-instruct.json;
BLADE from the edited-model C4 ppl of each alldata run). SimpleQA panels omitted: Llama base already abstains
(~1.5% incorrect) so there is no hallucination headroom (see [[llama32-3b-transfer]]). Numbers are the
already-derived Llama figures for this exploratory cross-model panel (Llama is a poorer testbed than Qwen3-8B)."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import scienceplots  # noqa: F401

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 31, "axes.labelsize": 38, "axes.titlesize": 33,
                     "xtick.labelsize": 32, "ytick.labelsize": 34, "xtick.top": False,
                     "ytick.right": False, "axes.linewidth": 1.2})

# base / ITI c=1 / ITI c=2 / BLADE a=1.75 / BLADE a=2.0
METHODS = [("base", "#3D405B"),
           ("ITI (c=1)", "#3AA6A0"),
           ("ITI (c=2)", "#0E6E6E"),
           ("BLADE\n(ρ=.005, α=1.75)", "#E08A3C"),
           ("BLADE\n(ρ=.005, α=2.0)", "#D9532B")]
labs = [m[0] for m in METHODS]; cols = [m[1] for m in METHODS]
y = np.arange(len(METHODS)); H = 0.7

# Capability cost: C4 teacher-forced Delta-ppl (%)
PPL = [0.0, 8.4, 46.5, 3.5, 5.7]
# Behaviour rates (%), Opus-judged ACT taxonomy, SA/FQ n=70
SAUN = [47.1, 38.6, 28.6, 32.9, 37.1]   # SelfAware unanswerable: hallucination (lower better)
SAANS = [72.9, 65.2, 44.3, 40.0, 40.0]  # SelfAware answerable: answered (higher better)
FQFA = [29.0, 31.4, 21.4, 20.0, 12.9]   # FalseQA false-premise: accepted (lower better)
FQTP = [71.4, 67.1, 55.1, 72.9, 71.4]   # FalseQA true-premise: answered (higher better)

PAN = [("ppl", PPL, "capability\ncost", "Δ perplexity (%) ↓", True),
       ("sa", SAUN, "SelfAware\nunanswerable", "hallucination (%) ↓", True),
       ("sa", SAANS, "SelfAware\nanswerable", "answered (%) ↑", False),
       ("fq", FQFA, "FalseQA\nfalse-premise", "accepted (%) ↓", True),
       ("fq", FQTP, "FalseQA\ntrue-premise", "answered (%) ↑", False)]

fig, axes = plt.subplots(1, 5, figsize=(24.5, 9.8), sharey=True)
for k, (ax, (kind, vals, title, xlab, low)) in enumerate(zip(axes.flat, PAN)):
    b = ax.barh(y, vals, H, color=cols, edgecolor="white", linewidth=0.9, zorder=3)
    if kind == "ppl":
        hi = max(vals)
        for yi, v in zip(y, vals):
            ax.text(max(v, 0) + hi * 0.012, yi, f"{v:+.1f}", va="center", ha="left", fontsize=27)
        ax.set_xlim(0, hi * 1.28)
    else:
        ax.bar_label(b, fmt="%.0f", fontsize=27, padding=2)
        ax.set_xlim(0, max(vals) * (1.32 if low else 1.18))
    ax.set_yticks(y); ax.set_yticklabels(labs, linespacing=0.85); ax.invert_yaxis()
    ax.set_xlabel(xlab, labelpad=8)
    ax.text(0.0, -0.42, f"({chr(97+k)}) {title}", transform=ax.transAxes,
            ha="left", va="top", ma="left", fontsize=44, fontweight="normal", linespacing=0.9)
    ax.xaxis.grid(True, ls="-", lw=0.5, color="#DFDFDF", zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
p0 = axes[0].get_position()
y_top = p0.y1; y_bot = p0.y0 - 0.42 * p0.height
for a, b_ in [(0, 1), (2, 3)]:   # capability | SelfAware | FalseQA
    xd = (axes[a].get_position().x1 + axes[b_].get_position().x0) / 2
    fig.add_artist(mlines.Line2D([xd, xd], [y_bot, y_top], color="#9AA0A6", ls=(0, (5, 4)),
                                 lw=1.3, transform=fig.transFigure))
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"uncertainty_method_cmp_llama.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/uncertainty_method_cmp_llama.png / .pdf")
for lb, p, un, an, fa, tp in zip(labs, PPL, SAUN, SAANS, FQFA, FQTP):
    print(f"  {lb.replace(chr(10),' '):16s} ppl {p:+5.1f}  SAun {un:5.1f} SAans {an:5.1f} FQfa {fa:5.1f} FQtp {tp:5.1f}")
