"""uncertainty_method_ppl: capability cost (C4 Δperplexity) per method, matching uncertainty_method_cmp.
solo_vs_joint palette, compact, large fonts, subtitle BELOW. ITI at the strength that cuts hallucination
(α=6, +57%) is catastrophic; BLADE amplify ×4 achieves lower hallucination at +1.6%."""
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
amp = json.load(open(R / "amp_ppl.json"))["amp_ppl_delta"]
rem = [r for r in json.load(open(R / "epistemic_p0_qwen3-8b_bladeg.json"))["sweep"]
       if abs(r["sparsity"] - 0.005) < 1e-9][0]["ppl_delta_c4"] * 100

BARS = [("base", 0.0, "#3D405B"),
        ("ITI α=2", iti["iti_a2.0"] * 100, "#9AD5CD"), ("ITI α=4", iti["iti_a4.0"] * 100, "#3AA6A0"),
        ("ITI α=6", iti["iti_a6.0"] * 100, "#0E6E6E"),
        ("BLADE amp ×2", amp["ampW2.0"] * 100, "#F4A98F"), ("BLADE amp ×4", amp["ampW4.0"] * 100, "#EE6C4D"),
        ("BLADE remove", rem, "#B8860B")]
labs = [b[0] for b in BARS]; vals = [b[1] for b in BARS]; cols = [b[2] for b in BARS]
y = np.arange(len(BARS))

fig, ax = plt.subplots(figsize=(6.2, 4.0))
b = ax.barh(y, vals, 0.72, color=cols, edgecolor="white", linewidth=0.9, zorder=3)
ax.bar_label(b, fmt="%+.1f%%", fontsize=13, padding=3)
ax.axvline(0, color="#3D405B", lw=0.9)
ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis()
ax.set_xlabel("Δ perplexity (%, C4)")
ax.set_xlim(min(vals) - 3, max(vals) * 1.2)
ax.xaxis.grid(True, ls="-", lw=0.5, color="#DFDFDF", zorder=0); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, -0.05, "Capability cost (Qwen3-8B, C4). ITI at the strength needed to cut hallucination "
         "(α=6) is catastrophic (+57%); BLADE amplify ×4 cuts it more at +1.6%.", ha="center", fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"uncertainty_method_ppl.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/uncertainty_method_ppl.png / .pdf")
for l, v in zip(labs, vals):
    print(f"  {l:14s} {v:+.2f}%")
