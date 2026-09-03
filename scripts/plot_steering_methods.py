"""Pareto: weight-amplify (BLADE) vs mainstream activation-steering methods for
strengthening OOD refusal (HarmBench) on Llama-3.2-3B. y=refusal, x=WikiText Δppl (log)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

R = Path("results"); FIG = Path("figures")
sm = json.loads((R / "blade_steering_methods.json").read_text())
amp = json.loads((R / "blade_amplify_steer_grid.json").read_text())["rows"]
FLOOR = 0.05  # % floor for log-x (base ~0)

STEER = {"caa_1L": ("CAA (best single layer)", "#8FA9BE", "--o"),
         "caa_multi": ("CAA (multi-layer)", "#3A7CA5", "--s"),
         "addunit_1L": ("ActAdd (unit-norm)", "#B0779E", "--^"),
         "clamp_1L": ("Refusal Direction", "#C9A06A", "--D")}
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"xtick.top": False, "ytick.right": False,
                     "xtick.labelsize": 15, "ytick.labelsize": 15})

fig, ax = plt.subplots(figsize=(5.4, 3.9))
for key, (lab, col, mk) in STEER.items():
    rows = sm["methods"][key]
    xs = [max(r["ppl_delta"] * 100, FLOOR) for r in rows]
    ys = [r["refusal"] for r in rows]
    ax.plot(xs, ys, mk, color=col, lw=1.8, ms=6, label=lab, alpha=0.9)
# BLADE amplify (weight-level) — highlighted
xs = [max(r["amplify_ppl_delta"] * 100, FLOOR) for r in amp]
ys = [r["amplify_refusal"] for r in amp]
ax.plot(xs, ys, "-o", color="#EE6C4D", lw=3.0, ms=9, label="BLADE amplify (weights ×α)", zorder=6)
# baseline
ax.scatter([FLOOR], [sm["base"]["refusal"]], s=130, facecolors="white", edgecolors="#3D405B",
           lw=2.2, zorder=5); ax.annotate("baseline", (FLOOR, sm["base"]["refusal"]),
           xytext=(6, 6), textcoords="offset points", fontsize=13)
ax.set_xscale("log")
ax.set_xlabel("WikiText perplexity change (%, log scale)", fontsize=15)
ax.set_ylabel("OOD refusal rate (HarmBench)", fontsize=15)
ax.set_ylim(0, 0.6)
# optimal = high refusal at low ppl cost -> top-left
ax.annotate("optimal", xy=(0.055, 0.575), xytext=(0.30, 0.50),
            fontsize=14, fontweight="bold", color="#3D405B",
            arrowprops=dict(arrowstyle="->", color="#3D405B", lw=1.6))
ax.legend(fontsize=12, frameon=True, edgecolor="grey", facecolor="white", loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"steering_methods_pareto.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/steering_methods_pareto.png / .pdf")
