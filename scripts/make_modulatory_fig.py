"""Concept figure v2 (synthesized from codex + kimi design review):
BLADE scales a sparse MODULATORY gain by factor f -- behavior moves, capability doesn't.

Left  : mechanism. Teal driving trunk passes UNINTERRUPTED; a sparse coral
        modulatory side-input enters an op-amp (gain) triangle; BLADE sets its ×f.
Right : effect of f, everything relative to the f=1 (original) model.
        top strip = behavior expression (schematic curve anchored at f=1=original,
        real markers: star f=0 removed, dot f=1 baseline, open f≈1.3 amplify);
        bottom strip = capability Δ ≈ 0 band (separate axis, never crosses behavior).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrow, Polygon, FancyBboxPatch, Rectangle
import scienceplots  # noqa: F401

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
TEAL = "#3AA6A0"; TEAL_D = "#0E6E6E"; CORAL = "#EE6C4D"; CORAL_D = "#B8442B"
SLATE = "#3D405B"; GREY = "#8A8A8A"; UNSEL = "#D9DCE5"
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"xtick.top": False, "ytick.right": False})

fig = plt.figure(figsize=(11.6, 4.5))
gs = GridSpec(2, 2, width_ratios=[1.0, 1.12], height_ratios=[2.4, 0.85],
              hspace=0.12, wspace=0.22, figure=fig)
axM = fig.add_subplot(gs[:, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 1], sharex=axB)


def arrow(ax, x, y, dx, dy, c, w=0.09, hw=0.30, hl=0.26, z=3):
    ax.add_patch(FancyArrow(x, y, dx, dy, width=w, head_width=hw, head_length=hl,
                            fc=c, ec="none", length_includes_head=True, zorder=z))


# ============ LEFT: sparse multi-layer weight map ============
# rows = layers, cols = residual-writer weights within a layer. Almost all weights
# are driving (grey, untouched); a FEW layers each contribute a FEW modulatory
# weights (coral) that BLADE scales by ×α.
axM.set_xlim(0, 10); axM.set_ylim(0, 10); axM.axis("off")
NR, NC = 7, 15
gx0, gx1, gy0, gy1 = 2.15, 8.9, 2.55, 8.35
cw = (gx1 - gx0) / NC; ch = (gy1 - gy0) / NR
sel_cells = {1: [7], 3: [4, 11], 5: [9]}   # few selected layers, few coral weights each
mod_pts = []
for r in range(NR):
    cc = sel_cells.get(r, [])
    for c in range(NC):
        xx = gx0 + c * cw; yy = gy0 + r * ch
        is_mod = c in cc
        if is_mod:
            mod_pts.append((xx + cw / 2, yy + ch / 2))
        axM.add_patch(Rectangle((xx + 0.04, yy + 0.06), cw - 0.08, ch - 0.12,
                                fc=CORAL if is_mod else UNSEL, ec="white", lw=0.5,
                                zorder=(5 if is_mod else 3)))
# "layers" axis arrow
arrow(axM, 1.72, gy0, 0, gy1 - gy0, SLATE, w=0.025, hw=0.16, hl=0.22, z=2)
axM.text(1.4, (gy0 + gy1) / 2, "layers", color=SLATE, fontsize=12, rotation=90, va="center", ha="center")
axM.text((gx0 + gx1) / 2, gy0 - 0.42, "weights within a layer", color=GREY, fontsize=11, ha="center")
# inline legend swatches
axM.add_patch(Rectangle((gx0, 9.0), 0.34, 0.34, fc=UNSEL, ec="white", zorder=3))
axM.text(gx0 + 0.5, 9.17, "not selected", color="#6b7280", fontsize=10.5, va="center")
axM.add_patch(Rectangle((gx0 + 2.85, 9.0), 0.34, 0.34, fc=CORAL, ec="white", zorder=5))
axM.text(gx0 + 3.35, 9.17, "selected: modulatory ×α", color=CORAL_D, fontsize=10.5, va="center", fontweight="bold")
# ×α callout pointing at a coral cell
cx, cy = mod_pts[1]
axM.annotate("×α", (cx, cy), xytext=(cx + 1.7, cy + 1.15), fontsize=15, color=CORAL_D,
             fontweight="bold", ha="center", zorder=7,
             arrowprops=dict(arrowstyle="-|>", color=CORAL_D, lw=1.8))
# BLADE tag carries the sparsity claim
axM.add_patch(FancyBboxPatch((1.2, 0.55), 8.0, 0.66, boxstyle="round,pad=0.03,rounding_size=0.12",
                             fc=CORAL, ec="none", zorder=6))
axM.text(5.2, 0.88, "BLADE:  ×α  on ≈0.002% of weights", color="white",
         fontsize=12, ha="center", va="center", fontweight="bold", zorder=8)


# ================= RIGHT top: behavior =================
for ax in (axB, axC):
    ax.axvspan(0, 1, color="#F0F1F3", zorder=0)
    ax.axvspan(1, 1.5, color=CORAL, alpha=0.08, zorder=0)
    ax.axvline(1, color=SLATE, lw=1.1, ls=(0, (3, 3)), zorder=2)
    ax.set_xlim(-0.04, 1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False, colors=SLATE)

ORIG = 0.55
x1 = np.linspace(0, 1, 160); x2 = np.linspace(1, 1.5, 90)
y1 = 0.12 + (ORIG - 0.12) * x1 ** 1.5
y2 = ORIG + 0.30 * (1 - np.exp(-2.6 * (x2 - 1)))
axB.plot(x1, y1, color=CORAL, lw=2.6, zorder=4)
axB.plot(x2[x2 <= 1.3], y2[x2 <= 1.3], color=CORAL, lw=2.6, zorder=4)
axB.plot(x2[x2 >= 1.3], y2[x2 >= 1.3], color=CORAL, lw=2.6, ls=":", zorder=4)  # extrapolated
axB.axhline(ORIG, color=SLATE, lw=0.8, alpha=0.5, zorder=1)
# real anchors
axB.scatter([0], [0.12], marker="*", s=250, color=CORAL, edgecolor=CORAL_D, lw=0.8,
            zorder=7, clip_on=False)
axB.annotate("α=0: refusal 0.91→0.09", (0, 0.12), xytext=(12, 10), textcoords="offset points",
             color=CORAL_D, fontsize=10.5, fontweight="bold")
axB.scatter([1], [ORIG], s=55, facecolor="white", edgecolor=SLATE, lw=1.6, zorder=6)
axB.annotate("original", (1, ORIG), xytext=(7, -14), textcoords="offset points",
             color=SLATE, fontsize=10)
axB.scatter([1.3], [ORIG + 0.30 * (1 - np.exp(-2.6 * 0.3))], s=60, facecolor="white",
            edgecolor=CORAL_D, lw=1.6, zorder=6)
axB.annotate("α≈1.3: robustness ↑", (1.3, 0.80), xytext=(-4, 6), textcoords="offset points",
             color=CORAL_D, fontsize=10.5, ha="right")
axB.set_ylim(0, 1.0)
axB.set_yticks([0.12, ORIG, 0.85]); axB.set_yticklabels(["lower", "original", "higher"], fontsize=10.5)
axB.set_ylabel("behavior", color=SLATE, fontsize=12.5)
axB.tick_params(axis="x", labelbottom=False)
axB.text(0.5, 0.94, "attenuate", color=GREY, ha="center", fontsize=11)
axB.text(1.25, 0.94, "amplify", color=CORAL_D, ha="center", fontsize=11)

# ================= RIGHT bottom: capability strip =================
axC.fill_between([0, 1.5], -0.03, 0.03, color=TEAL, alpha=0.22, zorder=1)
axC.plot([0, 1.5], [0, 0], color=TEAL, lw=2.2, zorder=3)
axC.text(1.46, 0.05, "preserved", color=TEAL_D, ha="right", va="bottom", fontsize=10.5)
axC.set_ylim(-0.11, 0.11); axC.set_yticks([0]); axC.set_yticklabels(["≈0"], fontsize=10.5)
axC.set_ylabel("capability Δ", color=SLATE, fontsize=12)
axC.set_xticks([0, 1, 1.3]); axC.set_xticklabels(["0", "1", "1.3"], fontsize=11)
axC.set_xlabel("reweighting factor  α", color=SLATE, fontsize=12.5)

fig.suptitle("BLADE scales a sparse modulatory gain — behavior moves, capability doesn't",
             fontsize=15, fontweight="bold", y=1.01)
fig.text(0.5, -0.04, "Llama-3.2-3B · ≈0.002% of all weights (69k / 3.2B) · α=0: refusal 0.91→0.09, Δppl +0.3%, 6-task Δ≈0",
         ha="center", fontsize=10, color=GREY)
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"blade_modulatory_concept.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/blade_modulatory_concept.png / .pdf")
