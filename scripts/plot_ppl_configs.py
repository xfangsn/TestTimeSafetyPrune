"""Capability cost (WikiText perplexity change), horizontal bars, per config. Deterministic (no judge)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

R = Path("results"); FIG = Path("figures")
sb = json.loads((R / "blade_steering_baselines.json").read_text())
els = json.loads((R / "blade_amplify_els_configs.json").read_text())
sw = json.loads((R / "blade_config_sweep.json").read_text())


def rep(method, key, val):
    return next(r[key] for r in sb["methods"][method]["report"] if abs(r["coef"] - val) < 1e-9)


relppl = {
    "base": 0.0,
    "ActAdd_L14_c8": rep("actadd", "wiki_relppl", 8.0),
    "ActAdd_L14_c16": rep("actadd", "wiki_relppl", 16.0),
    "CAA_L20_c0.5": rep("caa", "wiki_relppl", 0.5),
    "CAA_L20_c0.7": rep("caa", "wiki_relppl", 0.7),
    "RefDir_L16_c0.35": rep("arditi", "wiki_relppl", 0.35),
    "RefDir_L16_c0.5": rep("arditi", "wiki_relppl", 0.5),
    "BLADE_L14_r002_a1.5": next(r["wiki_relppl"] for r in els["report"] if abs(r["alpha"] - 1.5) < 1e-9),
    "BLADE_L12_r005_a1.5": next(r["relppl"] for c in sw["configs"] if c["name"] == "L12 rho.005 both"
                               for r in c["sweep"] if abs(r["alpha"] - 1.5) < 1e-9),
}
ORDER = ["base", "ActAdd_L14_c8", "ActAdd_L14_c16", "CAA_L20_c0.5", "CAA_L20_c0.7",
         "RefDir_L16_c0.35", "RefDir_L16_c0.5", "BLADE_L14_r002_a1.5", "BLADE_L12_r005_a1.5"]
LABELS = {"base": "base", "ActAdd_L14_c8": "ActAdd (c=8)", "ActAdd_L14_c16": "ActAdd (c=16)",
          "CAA_L20_c0.5": "CAA (c=.5)", "CAA_L20_c0.7": "CAA (c=.7)",
          "RefDir_L16_c0.35": "RefDir (c=.35)", "RefDir_L16_c0.5": "RefDir (c=.5)",
          "BLADE_L14_r002_a1.5": "BLADE (ρ=.002)", "BLADE_L12_r005_a1.5": "BLADE (ρ=.005)"}
COLORS = {"base": "#8A8A8A", "ActAdd": "#C9A06A", "CAA": "#3A7CA5", "RefDir": "#B0779E", "BLADE": "#EE6C4D"}


def color(k):
    for pre, c in COLORS.items():
        if k.startswith(pre) or k == "base" and pre == "base":
            return c
    return "#8A8A8A"


vals = [relppl[k] * 100 for k in ORDER]
cols = [color(k) for k in ORDER]

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"xtick.top": False, "ytick.right": False,
                     "xtick.labelsize": 18, "ytick.labelsize": 21})
fig, ax = plt.subplots(figsize=(5.2, 5.8))
y = np.arange(len(ORDER))
b = ax.barh(y, vals, 0.62, color=cols)
ax.bar_label(b, fmt="%.1f%%", fontsize=15, padding=2)
ax.set_yticks(y); ax.set_yticklabels([LABELS[k] for k in ORDER])
ax.invert_yaxis()
ax.set_xlabel("Perplexity $\\Delta$ (%)", fontsize=23)
ax.set_xlim(0, max(vals) * 1.18)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"ppl_configs.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/ppl_configs.png / .pdf")
