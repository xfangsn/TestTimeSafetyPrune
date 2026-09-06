"""Capability cost of each method: C4 Δperplexity. solo_vs_joint palette, compact, large fonts,
subtitle BELOW. ITI at the alpha needed to cut hallucination (a6) is catastrophic; BLADE is ~flat."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

R = Path("results"); FIG = Path("figures"); FIG.mkdir(exist_ok=True)
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "xtick.top": False,
                     "ytick.right": False, "axes.linewidth": 1.0})

iti = json.load(open(R / "iti_ppl.json"))["alpha_ppl_delta"]
p0 = json.load(open(R / "epistemic_p0_qwen3-8b_bladeg.json"))
amp = json.load(open(R / "epistemic_amplify_v2_qwen3-8b_bladeg.json"))
rem = [r for r in p0["sweep"] if abs(r["sparsity"] - 0.005) < 1e-9][0]["ppl_delta_c4"] * 100
ampx2 = [c for c in amp["conditions"] if c["label"] == "alphaW_a2.0"][0]["ppl_delta_c4"] * 100

BARS = [("base", 0.0, "#3D405B"), ("ITI α=2", iti["iti_a2.0"] * 100, "#3AA6A0"),
        ("ITI α=6", iti["iti_a6.0"] * 100, "#0E6E6E"), ("BLADE amp.", ampx2, "#EE6C4D"),
        ("BLADE rem.", rem, "#B8860B")]
labs = [b[0] for b in BARS]; vals = [b[1] for b in BARS]; cols = [b[2] for b in BARS]
y = np.arange(len(BARS))

fig, ax = plt.subplots(figsize=(5.4, 3.6))
b = ax.barh(y, vals, 0.66, color=cols, edgecolor="white", linewidth=0.9, zorder=3)
ax.bar_label(b, fmt="%+.1f%%", fontsize=13, padding=3)
ax.axvline(0, color="#3D405B", lw=0.9)
ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis()
ax.set_xlabel("Δ perplexity (%, C4)")
ax.set_xlim(min(vals) - 5, max(vals) * 1.22)
ax.xaxis.grid(True, ls="-", lw=0.5, color="#DFDFDF", zorder=0); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, -0.06, "Capability cost (Qwen3-8B). ITI at the strength needed to reduce hallucination "
         "(α=6) is catastrophic; BLADE stays near-flat.", ha="center", fontsize=11.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"uncertainty_method_ppl.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/uncertainty_method_ppl.png / .pdf")
for l, v in zip(labs, vals):
    print(f"  {l:11s} {v:+.1f}%")
