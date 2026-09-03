"""Cross-model BLADE+ELS dumbbell chart (academic, legible).
baseline (open) -> within-budget post-BLADE (filled), per behavior x model.
Per-row chance tick (0.5 A/B pick-rate, 0.0 refusal-rate) unifies the mixed metric.
grey x = not exhibited; open square = diffuse (L* empty). Wong CVD-safe palette,
serif, no ieee shrink, ppl above marker, behavior labels vertical at left.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots  # noqa: F401

# Wong palette: high-contrast, colorblind-safe
MC = {"Llama-3.2-3B": "#0072B2", "Qwen3-4B": "#D55E00", "Gemma-3-4B": "#009E73"}

DATA = [
 ("refusal", 0.0, [
    ("Llama-3.2-3B", "ok", 1.00, 0.00, "~0%*"),
    ("Qwen3-4B", "ok", 0.98, 0.00, "+2%*"),
    ("Gemma-3-4B", "ok", 0.60, 0.00, "+3.6%")]),
 ("power-seeking", 0.5, [
    ("Llama-3.2-3B", "ok", 0.75, 0.43, "+4.6%"),
    ("Qwen3-4B", "ok", 0.63, 0.46, "+3.6%"),
    ("Gemma-3-4B", "ok", 0.65, 0.45, "-3.3%")]),
 ("wealth-seeking", 0.5, [
    ("Llama-3.2-3B", "ok", 0.62, 0.51, "+0.3%"),
    ("Qwen3-4B", "ok", 0.67, 0.45, "+4.0%"),
    ("Gemma-3-4B", "ok", 0.62, 0.36, "+5.3%")]),
 ("self-awareness", 0.5, [
    ("Llama-3.2-3B", "ok", 0.60, 0.33, "+1.3%"),
    ("Qwen3-4B", "none", 0.49, None, None),
    ("Gemma-3-4B", "none", 0.52, None, None)]),
 ("corrigibility", 0.5, [
    ("Llama-3.2-3B", "diffuse", 0.60, None, None),
    ("Qwen3-4B", "ok", 0.65, 0.48, "+2.4%"),
    ("Gemma-3-4B", "diffuse", 0.66, None, None)]),
]

plt.rcParams["font.family"] = "serif"
with plt.style.context(["science", "no-latex"]):
    fig, ax = plt.subplots(figsize=(8.8, 8.0))
    plt.rc("font", size=14)
    row_h, grp_gap = 1.0, 1.2
    y = 0.0
    yticks, ylabels, group_spans = [], [], []
    for beh, chance, rows in DATA:
        g_top = y
        for model, st, base, after, ppl in rows:
            c = MC[model]
            ax.plot([chance, chance], [y - 0.38, y + 0.38], color="#b0b4ae",
                    lw=1.0, zorder=1)
            if st == "ok":
                ax.plot([after, base], [y, y], color=c, lw=2.6, zorder=2,
                        clip_on=False, solid_capstyle="round")
                ax.scatter([base], [y], s=90, facecolors="white", edgecolors=c,
                           lw=2.0, zorder=3, clip_on=False)
                ax.scatter([after], [y], s=105, facecolors=c, edgecolors=c,
                           lw=0, zorder=4, clip_on=False)
                # ppl above the filled marker
                ax.annotate(ppl, (after, y), textcoords="offset points",
                            xytext=(0, 11), ha="center", va="bottom",
                            fontsize=11, color=c, fontweight="bold")
            elif st == "none":
                ax.scatter([base], [y], s=95, marker="x", color="#9aa0a6",
                           lw=2.2, zorder=3, clip_on=False)
                ax.annotate("not exhibited", (base, y), textcoords="offset points",
                            xytext=(13, 0), ha="left", va="center", fontsize=11.5,
                            color="#8a8f86", style="italic")
            else:
                ax.scatter([base], [y], s=92, marker="s", facecolors="white",
                           edgecolors="#C99700", lw=2.2, zorder=3, clip_on=False)
                ax.annotate("diffuse", (base, y), textcoords="offset points",
                            xytext=(13, 0), ha="left", va="center", fontsize=11.5,
                            color="#A67C00", style="italic")
            yticks.append(y)
            ylabels.append(model.split("-")[0])
            y += row_h
        group_spans.append((beh, g_top, y - row_h))
        y += grp_gap

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=12.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=12.5)
    ax.set_xlabel("behavior score   (A/B: pick-rate,  chance $=0.5$   |   refusal: refusal-rate,  target $=0$)",
                  fontsize=13.5)
    # vertical behavior group labels at far left
    for beh, top, bot in group_spans:
        ax.text(-0.19, (top + bot) / 2, beh, rotation=90, va="center", ha="center",
                fontsize=13.5, fontweight="bold",
                transform=ax.get_yaxis_transform(), clip_on=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.margins(y=0.04)

    mh = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10,
                 label=m) for m, c in MC.items()]
    sh = [Line2D([0], [0], marker="o", color="#444", markerfacecolor="white",
                 markersize=10, lw=0, label="baseline"),
          Line2D([0], [0], marker="o", color="#444", markerfacecolor="#444",
                 markersize=10, lw=0, label="post-BLADE"),
          Line2D([0], [0], color="#b0b4ae", lw=1.6, label="chance")]
    leg1 = ax.legend(handles=mh, loc="lower left", bbox_to_anchor=(-0.02, -0.145),
                     ncol=3, frameon=False, fontsize=11.5, handletextpad=0.4,
                     columnspacing=1.4)
    ax.add_artist(leg1)
    ax.legend(handles=sh, loc="lower right", bbox_to_anchor=(1.02, -0.145), ncol=3,
              frameon=False, fontsize=11.5, handletextpad=0.4, columnspacing=1.4)

    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/blade_crossmodel_dumbbell.pdf", bbox_inches="tight")
    fig.savefig("results/blade_crossmodel_dumbbell.png", dpi=300, bbox_inches="tight")
    print("saved figures/blade_crossmodel_dumbbell.pdf and results/blade_crossmodel_dumbbell.png")
