"""Sycophancy-ppl Pareto for BLADE + activation steering. House style."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

RESULTS = Path("results"); FIG = Path("figures")
d = json.loads((RESULTS / "blade_plus_steering_sycophancy.json").read_text())
STEER = "#8FA9BE"; V2 = "#EE6C4D"; V1 = "#C7A0D8"; SLATE = "#3D405B"; GRN = "#2A7F4F"
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"xtick.top": False, "ytick.right": False})

def xy(rows):  # keep the low-ppl regime (<=10%)
    return ([r["ppl_delta"] * 100 for r in rows if r["ppl_delta"] <= 0.10],
            [r["pick"] for r in rows if r["ppl_delta"] <= 0.10])

fig, ax = plt.subplots(figsize=(6.6, 4.6))
xs, ys = xy(d["rows"]["steer_only"]); ax.plot(xs, ys, "-o", color=STEER, lw=2.2, ms=6, label="steer-only")
xs, ys = xy(d["rows"]["blade_steer_v2"]); ax.plot(xs, ys, "-o", color=V2, lw=2.4, ms=6, label="BLADE + steer (base vec)")
xs, ys = xy(d["rows"]["blade_steer_v1"]); ax.plot(xs, ys, "--s", color=V1, lw=1.8, ms=5, label="BLADE + steer (re-extracted vec)")
ax.scatter([0], [d["rows"]["baseline"]["pick"]], s=150, facecolors="white", edgecolors=SLATE, lw=2.2, zorder=5)
ax.annotate("baseline", (0, d["rows"]["baseline"]["pick"]), xytext=(8, 0), textcoords="offset points", fontsize=11, va="center")
b = d["rows"]["blade_only"]
ax.scatter([b["ppl_delta"] * 100], [b["pick"]], s=150, marker="*", color=GRN, edgecolors=GRN, zorder=5)
ax.annotate("BLADE-only", (b["ppl_delta"] * 100, b["pick"]), xytext=(8, -4), textcoords="offset points", fontsize=11, color=GRN, va="center")
ax.axhline(0.5, ls=":", color="#999", lw=1.4)
ax.text(ax.get_xlim()[1], 0.5, " chance", color="#777", fontsize=10, va="bottom", ha="right")
ax.set_xlabel("WikiText $\\Delta$ppl (%)", fontsize=13)
ax.set_ylabel("sycophancy A/B pick-rate", fontsize=13)
ax.set_ylim(0.45, 0.95)
ax.set_title("BLADE + activation steering (sycophancy, Llama-3.2-3B)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10.5, frameon=True, edgecolor="grey", facecolor="white", loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"blade_steering_pareto.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/blade_steering_pareto.png / .pdf")
