"""Amplify-direction comparison: BLADE vs BLADE-G (g0, g1scalar) for STRENGTHENING OOD refusal.
Two panels trace, per score, the frontier over alpha (marker style = rho): OOD refusal (x, higher is
better) vs the two collateral costs -- WikiText Δppl and XSTest over-refusal (y, lower is better).
Lower-right = more OOD refusal at less collateral. Reads results/blade_g_amplify.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots  # noqa: F401

RESULTS = Path("results"); FIG = Path("figures")
D = json.loads((RESULTS / "blade_g_amplify.json").read_text())
base = D["base"]
COL = {"BLADE": "#3D405B", "BLADE-G_g0": "#3AA6A0", "BLADE-G_g1scalar": "#EE6C4D"}
LBL = {"BLADE": "BLADE", "BLADE-G_g0": "BLADE-G (g0)", "BLADE-G_g1scalar": "BLADE-G (g1scalar)"}
RHO_MK = {0.002: "o", 0.005: "s"}

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 12, "axes.labelsize": 14, "axes.titlesize": 14,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "figure.dpi": 150})

cells = D["cells"]
scores = ["BLADE", "BLADE-G_g0", "BLADE-G_g1scalar"]
rhos = sorted({c["rho"] for c in cells})

def frontier(pts, ykey):
    """Non-dominated set: no other point has OOD>= and cost<=. Returns sorted by OOD."""
    fr = [p for p in pts if not any(
        (q["ood_refusal"] >= p["ood_refusal"] and q[ykey] <= p[ykey] and q is not p
         and (q["ood_refusal"] > p["ood_refusal"] or q[ykey] < p[ykey])) for q in pts)]
    return sorted(fr, key=lambda c: c["ood_refusal"])


fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, (ykey, ylab, ybase) in zip(axes, [("wiki_relppl", "WikiText $\\Delta$ppl", 0.0),
                                          ("benign", "XSTest over-refusal", base["benign"])]):
    sc = 100 if ykey == "wiki_relppl" else 1
    for s in scores:
        pts = [c for c in cells if c["score"] == s]
        # faint scatter of all (alpha, rho) points
        for rho in rhos:
            rp = [p for p in pts if p["rho"] == rho]
            ax.scatter([p["ood_refusal"] for p in rp], [p[ykey] * sc for p in rp],
                       s=26, color=COL[s], marker=RHO_MK[rho],
                       facecolors=(COL[s] if rho == rhos[0] else "none"), edgecolors=COL[s],
                       linewidths=1.2, alpha=0.55, zorder=2)
        fr = frontier(pts, ykey)   # Pareto frontier across all alpha & rho for this score
        ax.plot([p["ood_refusal"] for p in fr], [p[ykey] * sc for p in fr], "-",
                color=COL[s], lw=2.0, zorder=4, label=LBL[s])
    yb = ybase * sc
    ax.scatter([base["ood_refusal"]], [yb], marker="*", s=160, color="#E1A730",
               edgecolor="k", lw=0.6, zorder=5, label="base (unedited)")
    ax.set_xlabel("OOD refusal (HarmBench, prefill attack) $\\rightarrow$")
    ax.set_ylabel(ylab + " $\\leftarrow$")
    ax.grid(True, ls="-", lw=0.5, color="#E8E8E8", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

handles = [Line2D([0], [0], color=COL[s], lw=1.6, marker="o", ms=6, label=LBL[s]) for s in scores]
handles += [Line2D([0], [0], color="grey", lw=0, marker="o", ms=6, mfc="grey", label=r"$\rho=0.002$"),
            Line2D([0], [0], color="grey", lw=0, marker="s", ms=6, mfc="white", mec="grey", label=r"$\rho=0.005$"),
            Line2D([0], [0], color="#E1A730", lw=0, marker="*", ms=11, mec="k", label="base")]
fig.legend(handles=handles, loc="upper center", ncol=6, frameon=True, edgecolor="grey",
           facecolor="white", bbox_to_anchor=(0.5, 1.04), fontsize=10.5)
fig.suptitle("Strengthening OOD refusal: BLADE vs BLADE-G (points = $\\alpha\\in\\{1.3,1.5,2,2.5,3\\}$)",
             y=1.10, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"blade_g_amplify.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/blade_g_amplify.png / .pdf")
