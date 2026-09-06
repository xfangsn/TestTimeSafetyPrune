"""matchrho_cmp: matched-rho BLADE on Qwen3-8B uncertainty (ELS probe frac == final edit rho; re-select L*
per rho). Bars = base + remove(alpha=0) + amplify(alpha=2.5) at rho in {.0005,.001,.005,.01}. Numbers from
the matched-rho sweep (degen/ppl) + Opus blind judge (SelfAware+FalseQA). Style mirrors uncertainty_method_cmp.
Provenance: results/blade_rho_sweep_matchrho{rho}_qwen3-8b.json + scratchpad opus_judge_matchrho{,01}."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import scienceplots  # noqa: F401
from pathlib import Path

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 26, "axes.labelsize": 31, "axes.titlesize": 28,
                     "xtick.labelsize": 27, "ytick.labelsize": 27, "xtick.top": False,
                     "ytick.right": False, "axes.linewidth": 1.2})

# label, ppl%, SAun, SAans, FQfa, FQtp, color
NAVY = "#3D405B"
REM = ["#BBD5E8", "#7FB0D6", "#4A86C0", "#1F5FA0"]   # remove: blue gradient (rho .0005->.01)
AMP = ["#F4B79E", "#E88A63", "#D9532B", "#A63603"]   # amplify: coral gradient
ROWS = [
    ("base",            0.0, 20.0, 87.1, 22.9, 84.3, NAVY),
    ("remove ρ=.0005", +1.5, 27.5, 85.0, 37.5, 92.5, REM[0]),
    ("remove ρ=.001",  +1.8, 30.0, 92.5, 45.0, 92.5, REM[1]),
    ("remove ρ=.005",  +1.2, 27.5, 95.0, 37.5, 90.0, REM[2]),
    ("remove ρ=.01",   +1.5, 35.0, 92.5, 52.5, 95.0, REM[3]),
    ("amplify ρ=.0005", -1.5, 20.0, 75.0, 25.0, 82.5, AMP[0]),
    ("amplify ρ=.001",  -1.7, 22.5, 72.5, 17.5, 62.5, AMP[1]),
    ("amplify ρ=.005",  -1.0, 15.0, 77.5, 27.5, 80.0, AMP[2]),
    ("amplify ρ=.01",   +0.2, 15.0, 77.5, 15.0, 70.0, AMP[3]),
]
labs = [r[0] for r in ROWS]; cols = [r[6] for r in ROWS]
y = np.arange(len(ROWS)); H = 0.72
# panel: (col index in ROWS tuple, title, xlabel, lower_is_better)
PAN = [(1, "capability\ncost", "Δ perplexity (%) ↓", True),
       (2, "SelfAware\nunanswerable", "hallucination (%) ↓", True),
       (3, "SelfAware\nanswerable", "answered (%) ↑", False),
       (4, "FalseQA\nfalse-premise", "accepted (%) ↓", True),
       (5, "FalseQA\ntrue-premise", "answered (%) ↑", False)]

fig, axes = plt.subplots(1, 5, figsize=(25, 11.5), sharey=True)
for k, (ax, (ci, title, xlab, low)) in enumerate(zip(axes.flat, PAN)):
    vals = [r[ci] for r in ROWS]
    b = ax.barh(y, vals, H, color=cols, edgecolor="white", linewidth=0.9, zorder=3)
    if ci == 1:  # ppl: labels right of 0 (negatives are tiny)
        hi = max(vals)
        for yi, v in zip(y, vals):
            ax.text(max(v, 0) + hi * 0.012, yi, f"{v:+.1f}", va="center", ha="left", fontsize=22)
        ax.set_xlim(0, hi * 1.30)
    else:
        ax.bar_label(b, fmt="%.0f", fontsize=22, padding=2)
        ax.set_xlim(0, max(vals) * (1.30 if low else 1.16))
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis()
    ax.set_xlabel(xlab, labelpad=8)
    ax.text(0.0, -0.30, f"({chr(97+k)}) {title}", transform=ax.transAxes,
            ha="left", va="top", ma="left", fontsize=34, fontweight="normal", linespacing=0.9)
    ax.xaxis.grid(True, ls="-", lw=0.5, color="#DFDFDF", zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
# dashed dividers between panel groups (capability | SelfAware | FalseQA)
p0 = axes[0].get_position(); y_top = p0.y1; y_bot = p0.y0 - 0.30 * p0.height
for a, b_ in [(0, 1), (2, 3)]:
    xd = (axes[a].get_position().x1 + axes[b_].get_position().x0) / 2
    fig.add_artist(mlines.Line2D([xd, xd], [y_bot, y_top], color="#9AA0A6", ls=(0, (5, 4)),
                                 lw=1.3, transform=fig.transFigure))
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"matchrho_cmp.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/matchrho_cmp.png / .pdf")
