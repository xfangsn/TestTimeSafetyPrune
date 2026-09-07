"""uncertainty_method_cmp_qwen: same house style as uncertainty_method_cmp_llama, on Qwen/Qwen3-4B.
Configs: base / ITI (c=2) / ITI (c=4) / BLADE (rho=.002, alpha=1.5) / BLADE (rho=.02, alpha=1.5).
SA/FQ n=70, Opus-judged (ACT taxonomy). Capability panel = C4 teacher-forced Delta-ppl:
ITI from results/iti_ppl_qwen3-4b.json (c2 +3.1%, c4 +29.9%); BLADE from the rho-sweep grid Delta-ppl
(r0.002_a1.5 -0.55%, r0.02_a1.5 -0.33%). Behaviour rates: results/ood_judged.json over q4b_cmp (base/ITI)
and q4b_cmp2 (BLADE alpha1.5). On Qwen3-4B BLADE alpha=1.5 matches/undercuts ITI's failure reduction while
preserving answering, at negative ppl (ITI pays +3% to +30%). SimpleQA omitted here (behaviour panels only)."""
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

# base / ITI c=2 / ITI c=4 / BLADE rho.002 a1.5 / BLADE rho.02 a1.5
METHODS = [("base", "#3D405B"),
           ("ITI (c=2)", "#3AA6A0"),
           ("ITI (c=4)", "#0E6E6E"),
           ("BLADE\n(ρ=.002, α=1.5)", "#E08A3C"),
           ("BLADE\n(ρ=.02, α=1.5)", "#D9532B")]
labs = [m[0] for m in METHODS]; cols = [m[1] for m in METHODS]
y = np.arange(len(METHODS)); H = 0.7

# Capability cost: C4 teacher-forced Delta-ppl (%)
PPL = [0.0, 3.1, 29.9, -0.6, -0.3]
# Behaviour rates (%), Opus-judged ACT taxonomy, SA/FQ n=70
SAUN = [27, 23, 20, 21, 17]   # SelfAware unanswerable: hallucination (lower better)
SAANS = [87, 81, 67, 81, 71]  # SelfAware answerable: answered (higher better)
FQFA = [36, 24, 20, 24, 20]   # FalseQA false-premise: accepted (lower better)
FQTP = [83, 71, 70, 76, 73]   # FalseQA true-premise: answered (higher better)

PAN = [("ppl", PPL, "capability\ncost", "Δ perplexity (%) ↓", True),
       ("sa", SAUN, "SelfAware\nunanswerable", "hallucination (%) ↓", True),
       ("sa", SAANS, "SelfAware\nanswerable", "answered (%) ↑", False),
       ("fq", FQFA, "FalseQA\nfalse-premise", "accepted (%) ↓", True),
       ("fq", FQTP, "FalseQA\ntrue-premise", "answered (%) ↑", False)]

fig, axes = plt.subplots(1, 5, figsize=(24.5, 9.1), sharey=True)
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
for a, b_ in [(0, 1), (2, 3)]:
    xd = (axes[a].get_position().x1 + axes[b_].get_position().x0) / 2
    fig.add_artist(mlines.Line2D([xd, xd], [y_bot, y_top], color="#9AA0A6", ls=(0, (5, 4)),
                                 lw=1.3, transform=fig.transFigure))
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"uncertainty_method_cmp_qwen.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/uncertainty_method_cmp_qwen.png / .pdf")
for lb, p, un, an, fa, tp in zip(labs, PPL, SAUN, SAANS, FQFA, FQTP):
    print(f"  {lb.replace(chr(10),' '):20s} ppl {p:+5.1f}  SAun {un:5.0f} SAans {an:5.0f} FQfa {fa:5.0f} FQtp {tp:5.0f}")
