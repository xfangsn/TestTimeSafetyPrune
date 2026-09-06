"""Size-scaling figure: scheme-A bidirectional control of closed-book parametric (un)certainty across
Qwen3 sizes (1.7B/4B/8B/14B, Hazel). (a) unanswerable abstention: base vs REMOVE(0.5%) vs AMPLIFY(αW×2)
— the controllable range at each size, known-question specificity annotated; (b) capability cost Δppl.
House style: scienceplots no-latex, Wong palette, png/pdf."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

R = Path("results"); FIG = Path("figures"); FIG.mkdir(exist_ok=True)
plt.style.use(["science", "no-latex"])
SIZES = [("1.7B", "17b"), ("4B", "4b"), ("8B", "8b"), ("14B", "14b")]

base, rem, amp, known, ppl = [], [], [], [], []
for nm, tag in SIZES:
    p = json.load(open(R / f"epistemic_p0_qwen3-{tag}_bladeg.json"))
    a = json.load(open(R / f"epistemic_amplify_v2_qwen3-{tag}_bladeg.json"))
    sw = [r for r in p["sweep"] if abs(r["sparsity"] - 0.005) < 1e-9][0]
    ax2 = [c for c in a["conditions"] if c["label"] == "alphaW_a2.0"][0]
    base.append(sw["hedge_unans_base"] * 100); rem.append(sw["hedge_unans_remove"] * 100)
    amp.append(ax2["abstain_unans"] * 100); known.append(ax2["abstain_known"] * 100)
    ppl.append(sw["ppl_delta_c4"] * 100)

x = np.arange(len(SIZES)); w = 0.26
fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.7))

axA.bar(x - w, rem, w, color="#D55E00", label="REMOVE (ρ=0.5%)")
axA.bar(x, base, w, color="#999999", label="base")
axA.bar(x + w, amp, w, color="#0072B2", label="AMPLIFY (αW×2)")
axA.set_xticks(x); axA.set_xticklabels([s[0] for s in SIZES])
axA.set_ylabel("abstention on unanswerable (%)"); axA.set_ylim(0, 105)
axA.set_title("(a) bidirectional control vs model size", fontsize=9)
axA.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
for xi, (r, a_) in enumerate(zip(rem, amp)):        # annotate the controllable range
    axA.annotate("", xy=(xi + w, a_), xytext=(xi - w, r),
                 arrowprops=dict(arrowstyle="<->", color="#555", lw=0.8, alpha=0.6))

axB.bar(x - w / 2, ppl, w, color="#D55E00", label="Δppl REMOVE")
axB.bar(x + w / 2, known, w, color="#0072B2", label="known-Q abstain (spec.)")
axB.axhline(0, color="#444", lw=0.6)
axB.set_xticks(x); axB.set_xticklabels([s[0] for s in SIZES])
axB.set_ylabel("%"); axB.set_title("(b) cost & specificity", fontsize=9)
axB.legend(fontsize=7, frameon=False, loc="upper left")

fig.suptitle("Scheme-A across Qwen3 sizes (closed-book parametric uncertainty; lexical hedge rate)",
             fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"epistemic_sizes.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/epistemic_sizes.png / .pdf")
for (nm, _), b, r, a_, k, pp in zip(SIZES, base, rem, amp, known, ppl):
    print(f"  {nm:5} base {b:.0f}  remove {r:.0f}  amplify {a_:.0f}  known {k:.0f}  Δppl {pp:+.1f}")
