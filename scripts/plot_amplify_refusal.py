"""Amplify BLADE refusal weights (alpha>=1), with vs without activation steering:
OOD jailbreak refusal (HarmBench) and WikiText ppl change. House style."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

R = Path("results"); FIG = Path("figures")
rows = json.loads((R / "blade_amplify_steer_grid.json").read_text())["rows"]
a = [r["alpha"] for r in rows]
AMP = "#EE6C4D"; STEER = "#3AA6A0"; GREY = "#8A8A8A"
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"xtick.top": False, "ytick.right": False})

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.1))

ax[0].plot(a, [r["amplify_refusal"] for r in rows], "-o", color=AMP, lw=2.4, ms=7, label="amplify only")
ax[0].plot(a, [r["amplify_steer_refusal"] for r in rows], "--s", color=STEER, lw=2.2, ms=6,
           label="amplify + steering")
ax[0].set_xlabel("reweight factor α  (selected weights ×α)", fontsize=12)
ax[0].set_ylabel("OOD jailbreak refusal (HarmBench)", fontsize=12)
ax[0].set_title("Refusal robustness", fontsize=12.5, fontweight="bold")
ax[0].set_ylim(-0.02, 0.6); ax[0].legend(fontsize=10.5, frameon=True, edgecolor="grey", facecolor="white")

ax[1].plot(a, [r["amplify_ppl_delta"] * 100 for r in rows], "-o", color=AMP, lw=2.4, ms=7, label="amplify only")
ax[1].plot(a, [r["amplify_steer_ppl_delta"] * 100 for r in rows], "--s", color=STEER, lw=2.2, ms=6,
           label="amplify + steering")
ax[1].axhline(5, ls=":", color=GREY, lw=1.2); ax[1].text(a[0], 5.4, "β=5% budget", fontsize=9, color=GREY)
ax[1].set_xlabel("reweight factor α  (selected weights ×α)", fontsize=12)
ax[1].set_ylabel("WikiText Δppl (%)", fontsize=12)
ax[1].set_title("Capability cost", fontsize=12.5, fontweight="bold")
ax[1].legend(fontsize=10.5, frameon=True, edgecolor="grey", facecolor="white")

for x in ax:
    x.set_xticks(a); x.spines[["top", "right"]].set_visible(False)
fig.suptitle("Amplifying BLADE refusal weights ± activation steering (Llama-3.2-3B)",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"amplify_refusal_ood.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/amplify_refusal_ood.png / .pdf")
