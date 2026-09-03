"""BLADE algorithm pipeline as a wide horizontal flow diagram. House palette,
scienceplots serif. Left->right: contrast -> represent -> score -> select layers
-> select weights -> reweight -> outcome."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import scienceplots  # noqa: F401

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
TEAL = "#3AA6A0"; TEAL_D = "#0E6E6E"; TEAL_F = "#E3F2EF"
CORAL = "#EE6C4D"; CORAL_D = "#B8442B"; CORAL_F = "#FCE7E0"
GRN = "#2A7F4F"; GRN_F = "#DCEFE4"; SLATE = "#3D405B"; GREY = "#8A8A8A"; GREY_F = "#EEEFF2"

plt.style.use(["science", "no-latex"])
fig, ax = plt.subplots(figsize=(15.5, 3.7))
ax.set_xlim(0, 100); ax.set_ylim(0, 22); ax.axis("off")

# stage: (x, w, fill, edge, title, [lines], title_color)
BW, GAP, Y, BH = 13.6, 1.7, 5.0, 11.0
stages = [
    (GREY_F, SLATE, "1 · Contrast", ["harmful vs harmless", "or  A vs B  prompts"], SLATE),
    (CORAL_F, CORAL_D, "2 · Represent", [r"$r_\ell$ = block-out  A$-$B (unit)",
                                          r"$\Delta\mu_W$ = writer-in  A$-$B"], CORAL_D),
    (CORAL_F, CORAL_D, "3 · Score edges", [r"$s_{ij}=[\,r_i\,W_{ij}\,\Delta\mu_j\,]_+$",
                                            "on o_proj, down_proj"], CORAL_D),
    (TEAL_F, TEAL_D, "4 · Select layers", ["solo pool  (Δppl≤β)", r"best-first $\to L^\star$"], TEAL_D),
    (TEAL_F, TEAL_D, "5 · Select weights", ["global top-ρ ranking", "per-matrix cap 0.10"], TEAL_D),
    (CORAL_F, CORAL_D, "6 · Reweight ×α", ["α=0  remove", "α>1  amplify"], CORAL_D),
]
x = 2.0
centers = []
for fill, edge, title, lines, tc in stages:
    ax.add_patch(FancyBboxPatch((x, Y), BW, BH, boxstyle="round,pad=0.15,rounding_size=1.0",
                                fc=fill, ec=edge, lw=1.8, zorder=3))
    ax.text(x + BW / 2, Y + BH - 2.2, title, ha="center", va="center", fontsize=14.5,
            fontweight="bold", color=tc, zorder=4)
    for k, ln in enumerate(lines):
        ax.text(x + BW / 2, Y + BH - 4.9 - k * 2.5, ln, ha="center", va="center",
                fontsize=12, color=SLATE, zorder=4)
    centers.append(x + BW / 2)
    x += BW + GAP

# arrows between stages
for i in range(len(stages) - 1):
    x0 = 2.0 + (i + 1) * BW + i * GAP
    ax.add_patch(FancyArrowPatch((x0, Y + BH / 2), (x0 + GAP, Y + BH / 2),
                                 arrowstyle="-|>", mutation_scale=22, lw=2.2,
                                 color=SLATE, zorder=5))

# outcome tag on the right
xo = 2.0 + 6 * BW + 5 * GAP + 0.2
ax.add_patch(FancyArrowPatch((xo - GAP - 0.2, Y + BH / 2), (xo, Y + BH / 2),
                             arrowstyle="-|>", mutation_scale=22, lw=2.2, color=GRN, zorder=5))
ax.add_patch(FancyBboxPatch((xo, Y + 1.2), 9.0, BH - 2.4, boxstyle="round,pad=0.15,rounding_size=1.0",
                            fc=GRN_F, ec=GRN, lw=1.8, zorder=3))
ax.text(xo + 4.5, Y + BH / 2 + 1.2, "behavior removed", ha="center", va="center",
        fontsize=13, fontweight="bold", color=GRN, zorder=4)
ax.text(xo + 4.5, Y + BH / 2 - 1.4, "capability preserved", ha="center", va="center",
        fontsize=13, fontweight="bold", color=GRN, zorder=4)

# banner
ax.text(2.0, 19.5, "BLADE", fontsize=20, fontweight="bold", color=SLATE, va="center")
ax.text(12.2, 19.5, "— Behavioral Localization via Activation-Difference Edges  "
        "(gradient-free · forward-only · ≈0.002% of weights)",
        fontsize=13, color=GREY, va="center")

# phase brackets under the boxes
def bracket(x0, x1, label, color):
    yb = 3.0
    ax.plot([x0, x0, x1, x1], [yb + 0.6, yb, yb, yb + 0.6], color=color, lw=1.4, zorder=2)
    ax.text((x0 + x1) / 2, yb - 1.3, label, ha="center", va="center", fontsize=11.5,
            color=color, fontstyle="italic")

b0 = 2.0
bracket(b0 + BW + GAP, b0 + 3 * BW + 2 * GAP, "activation-difference edges", CORAL_D)
bracket(b0 + 3 * BW + 3 * GAP, b0 + 5 * BW + 4 * GAP, "localization (ELS)", TEAL_D)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"blade_pipeline.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/blade_pipeline.png / .pdf")
