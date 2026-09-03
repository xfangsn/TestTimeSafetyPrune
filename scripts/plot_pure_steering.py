"""Pure activation-steering (no BLADE): OOD jailbreak refusal vs WikiText ppl for
four mainstream methods on Llama-3.2-3B. Standalone (no weight-editing baseline)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

R = Path("results"); FIG = Path("figures")
sm = json.loads((R / "blade_steering_methods.json").read_text())
amp = json.loads((R / "blade_amplify_steer_grid.json").read_text())["rows"]
FLOOR = 0.05
STEER = {"caa_1L": ("CAA (best single layer)", "#8FA9BE", "--o"),
         "caa_multi": ("CAA (multi-layer)", "#3A7CA5", "--s"),
         "addunit_1L": ("ActAdd (unit-norm)", "#B0779E", "--^"),
         "clamp_1L": ("directional clamp (ours, ablation-style)", "#C9A06A", "--D")}
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"xtick.top": False, "ytick.right": False})

fig, ax = plt.subplots(figsize=(7.4, 5.0))
for key, (lab, col, mk) in STEER.items():
    rows = sm["methods"][key]
    ax.plot([max(r["ppl_delta"] * 100, FLOOR) for r in rows], [r["refusal"] for r in rows],
            mk, color=col, lw=1.8, ms=6, label=lab, alpha=0.9)
# our method: BLADE weight-amplify (highlighted)
ax.plot([max(r["amplify_ppl_delta"] * 100, FLOOR) for r in amp],
        [r["amplify_refusal"] for r in amp], "-o", color="#EE6C4D", lw=3.0, ms=9,
        label="BLADE amplify (ours, weights ×α)", zorder=6)
ax.scatter([FLOOR], [sm["base"]["refusal"]], s=140, facecolors="white", edgecolors="#3D405B",
           lw=2.2, zorder=5)
ax.annotate("baseline (no steering)", (FLOOR, sm["base"]["refusal"]), xytext=(8, 4),
            textcoords="offset points", fontsize=10)
ax.axvline(5, ls=":", color="#999", lw=1.2); ax.text(5.4, 0.02, "β=5% budget", fontsize=9, color="#777")
ax.set_xscale("log")
ax.set_xlabel("WikiText perplexity change (%, log scale)  —  capability cost", fontsize=12.5)
ax.set_ylabel("OOD jailbreak refusal (HarmBench)", fontsize=12.5)
ax.set_title("Strengthening OOD refusal: BLADE (ours) vs activation steering", fontsize=12.5, fontweight="bold")
ax.set_ylim(0, 0.6)
ax.legend(fontsize=9.5, frameon=True, edgecolor="grey", facecolor="white", loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"pure_steering.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/pure_steering.png / .pdf")
