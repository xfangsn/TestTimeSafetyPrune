"""XSTest over-refusal rate, horizontal bars: string-match vs LLM judge (Kimi K3)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

R = Path("results"); FIG = Path("figures")
J = json.loads((R / "xstest_llm_judge.json").read_text())
ORDER = ["base", "ActAdd_L14_c8", "ActAdd_L14_c16", "CAA_L20_c0.5", "CAA_L20_c0.7",
         "RefDir_L16_c0.35", "RefDir_L16_c0.5", "BLADE_L14_r002_a1.5", "BLADE_L12_r005_a1.5"]
LABELS = {"base": "base", "ActAdd_L14_c8": "ActAdd (c=8)", "ActAdd_L14_c16": "ActAdd (c=16)",
          "CAA_L20_c0.5": "CAA (c=.5)", "CAA_L20_c0.7": "CAA (c=.7)",
          "RefDir_L16_c0.35": "RefDir (c=.35)", "RefDir_L16_c0.5": "RefDir (c=.5)",
          "BLADE_L14_r002_a1.5": "BLADE (ρ=.002)", "BLADE_L12_r005_a1.5": "BLADE (ρ=.005)"}
sm = [J[k]["stringmatch"] * 100 for k in ORDER]
llm = [J[k]["llm_overrefusal"] * 100 for k in ORDER]

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"xtick.top": False, "ytick.right": False,
                     "xtick.labelsize": 18, "ytick.labelsize": 21})
fig, ax = plt.subplots(figsize=(5.2, 5.8))
y = np.arange(len(ORDER)); h = 0.4
b1 = ax.barh(y + h / 2, sm, h, label="string-match", color="#6E9FC4")
b2 = ax.barh(y - h / 2, llm, h, label="kimi k3", color="#E0A458")
ax.bar_label(b1, fmt="%.1f", fontsize=14, padding=2)
ax.bar_label(b2, fmt="%.1f", fontsize=14, padding=2)
ax.set_yticks(y); ax.set_yticklabels([LABELS[k] for k in ORDER])
ax.invert_yaxis()
ax.set_xlabel("XSTest over-refusal (%)", fontsize=23)
ax.set_xlim(0, max(max(sm), max(llm)) * 1.15)
ax.legend(fontsize=19, ncol=2, frameon=True, edgecolor="grey", facecolor="white", loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"xstest_judge_agreement.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/xstest_judge_agreement.png / .pdf")
