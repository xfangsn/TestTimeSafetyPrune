"""High-beta vs low-beta: what the extra budget buys, and what it costs.
For sycophancy and deception (the budget-sensitive behaviors) on Llama-3.2-3B,
plot removal depth (post-pick, x) against the REAL downstream cost (mean acc_norm
drop over 6 zero-shot tasks, y). Marker area = pruning rate (% params zeroed);
arrow = low-beta -> high-beta. Each point annotated with beta and WikiText ppl,
to show ppl overstates the true (downstream) cost.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

PER_LAYER, TOTAL = 34_603_008, 3_212_749_824
# config -> (behavior, beta, post-pick, |L*|, rho, colour)
CFG = {
    "sycophancy_b10":  ("sycophancy", 10,  0.307, 2,  0.002, "#0072B2"),
    "sycophancy_b100": ("sycophancy", 100, 0.213, 3,  0.005, "#0072B2"),
    "deception_b10":   ("deception",  10,  0.367, 9,  0.005, "#D55E00"),
    "deception_b100":  ("deception",  100, 0.287, 10, 0.020, "#D55E00"),
}
R = json.load(open("results/blade_downstream_beta_compare.json"))
base = R["base"]["mean_acc_norm"]


def prune_pct(nL, rho):
    return 100 * round(rho * nL * PER_LAYER) / TOTAL


plt.rcParams["font.family"] = "serif"
with plt.style.context(["science", "no-latex"]):
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    plt.rc("font", size=14)
    for beh in ("sycophancy", "deception"):
        pts = {}
        for cfg, (b, beta, post, nL, rho, col) in CFG.items():
            if b != beh:
                continue
            dacc = (R[cfg]["mean_acc_norm"] - base) * 100
            ppl = R[cfg]["ppl_delta_pct"]
            pr = prune_pct(nL, rho)
            pts[beta] = (post, dacc, ppl, pr, col)
        (x0, y0, p0, r0, c) = pts[10]
        (x1, y1, p1, r1, _) = pts[100]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.0,
                                    alpha=0.55, shrinkA=12, shrinkB=12))
        for (x, y, ppl, pr, beta) in [(x0, y0, p0, r0, 10), (x1, y1, p1, r1, 100)]:
            ax.scatter([x], [y], s=120 + pr * 900, facecolors=c, edgecolors="white",
                       lw=1.5, zorder=4, alpha=0.9)
            # beta=10 label above; beta=100 label to the upper-left (open space)
            xy = (0, 22) if beta == 10 else (-16, 14)
            ha = "center" if beta == 10 else "right"
            ax.annotate(f"$\\beta$={beta}%,  +{ppl:.0f}% ppl,  {pr:.3f}% pruned",
                        (x, y), textcoords="offset points", xytext=xy, ha=ha,
                        va="bottom", fontsize=10.5, color=c)
        ax.annotate(beh, (x1, y1), textcoords="offset points", xytext=(16, -2),
                    ha="left", va="center", fontsize=13, fontweight="bold", color=c)

    ax.axhline(0, ls="--", lw=1.2, color="#9aa0a6", zorder=0)
    ax.text(0.40, 0.05, "base (no removal)", fontsize=10.5, color="#8a8f86", ha="right")
    ax.axvline(0.5, ls=":", lw=1.1, color="#b0b4ae", zorder=0)
    ax.text(0.5, ax.get_ylim()[0], " chance", fontsize=10, color="#8a8f86", va="bottom")
    ax.set_xlabel("post-BLADE pick-rate  (← removed more)", fontsize=14)
    ax.set_ylabel("downstream mean acc_norm change (pp)\n(6 zero-shot tasks)", fontsize=13.5)
    ax.set_title("Sycophancy & deception: what more perplexity budget buys vs costs",
                 fontsize=14.5, fontweight="bold", pad=12)
    ax.invert_xaxis()
    ax.set_ylim(-4.0, 0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(x=0.22)
    fig.text(0.5, -0.01, "marker area = pruning rate (bigger = more weights zeroed);  "
             "arrow = $\\beta$ 10% to 100%.  Going deeper (higher $\\beta$) removes a "
             "little more, but the downstream cost grows faster than the removal.",
             ha="center", fontsize=10, style="italic", color="#555")
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/blade_beta_tradeoff.pdf", bbox_inches="tight")
    fig.savefig("results/blade_beta_tradeoff.png", dpi=300, bbox_inches="tight")
    print("saved figures/blade_beta_tradeoff.pdf")
