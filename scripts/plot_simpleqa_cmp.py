"""simpleqa_method_cmp: SimpleQA (Wei et al. 2024) outcome breakdown for the uncertainty_method_cmp
configs on Qwen3-8B (n=400, thinking-off, Opus-graded correct/incorrect/not-attempted). Stacked bars per
method show the intervention trading confident-wrong (incorrect) for abstention (not-attempted).
Separate figure; does NOT touch uncertainty_method_cmp. Numbers from scratchpad opus_grade_simpleqa."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from pathlib import Path

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 24, "axes.labelsize": 27, "xtick.labelsize": 24,
                     "ytick.labelsize": 26, "legend.fontsize": 23, "axes.linewidth": 1.2})

# method label, correct, not_attempted, incorrect  (n=400, Opus-graded)
ROWS = [
    ("base",              3.8,  7.2, 89.0),
    ("ITI (c=4)",         4.5, 21.2, 74.2),
    ("ITI (c=6)",         3.8, 33.2, 63.0),
    ("BLADE\n(ρ=.005, α=2.5)", 4.2, 24.0, 71.8),
]
labs = [r[0] for r in ROWS]
correct = np.array([r[1] for r in ROWS])
not_att = np.array([r[2] for r in ROWS])
incorr = np.array([r[3] for r in ROWS])
y = np.arange(len(ROWS)); H = 0.62
C_CORRECT = "#2E8B57"; C_NA = "#B8BCC4"; C_INC = "#D9532B"

fig, ax = plt.subplots(figsize=(15, 6.2))
b1 = ax.barh(y, correct, H, color=C_CORRECT, edgecolor="white", linewidth=1.0, zorder=3, label="correct")
b2 = ax.barh(y, not_att, H, left=correct, color=C_NA, edgecolor="white", linewidth=1.0, zorder=3, label="not attempted")
b3 = ax.barh(y, incorr, H, left=correct + not_att, color=C_INC, edgecolor="white", linewidth=1.0, zorder=3, label="incorrect (hallucination)")

for yi, (c, na, inc) in enumerate(zip(correct, not_att, incorr)):
    if c >= 3:  ax.text(c / 2, yi, f"{c:.0f}", va="center", ha="center", color="white", fontsize=20, fontweight="bold")
    ax.text(c + na / 2, yi, f"{na:.0f}", va="center", ha="center", color="#333", fontsize=21, fontweight="bold")
    ax.text(c + na + inc / 2, yi, f"{inc:.0f}", va="center", ha="center", color="white", fontsize=21, fontweight="bold")

ax.set_yticks(y); ax.set_yticklabels(labs, linespacing=0.9); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_xlabel("SimpleQA responses (%)", labelpad=8)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.46), ncol=3, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.grid(True, ls="-", lw=0.5, color="#E4E4E4", zorder=0); ax.set_axisbelow(True)
ax.set_title("SimpleQA (Qwen3-8B, n=400): interventions trade confident-wrong for abstention",
             fontsize=23, pad=12)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"simpleqa_method_cmp.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/simpleqa_method_cmp.png / .pdf")
