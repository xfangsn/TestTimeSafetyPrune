"""Figure: scheme-A OOD transfer (FalseQA + SelfAware, kimi-judged) + capability cost (Δ perplexity).
House style: scienceplots no-latex, Wong CVD-safe palette, png(300dpi)+pdf to figures/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

R = Path("results"); FIG = Path("figures"); FIG.mkdir(exist_ok=True)
plt.style.use(["science", "no-latex"])

J = json.load(open(R / "ood_judged.json"))["endpoints"]
P0 = json.load(open(R / "epistemic_p0_qwen3-8b_bladeg.json"))
AMP = json.load(open(R / "epistemic_amplify_v2_qwen3-8b_bladeg.json"))

CONDS = ["base", "remove", "amplify", "ctrl_random", "ctrl_shuffledr"]
CLAB = ["base", "remove", "amplify", "random", "shuffled-r"]
# Wong palette
COL = {"base": "#999999", "remove": "#D55E00", "amplify": "#0072B2",
       "ctrl_random": "#CC9DB6", "ctrl_shuffledr": "#B7C9A8"}
cols = [COL[c] for c in CONDS]

fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))

# --- Panel A: OOD harmful commitment (lower = better) ---
axA = axes[0]
harm = {
    "SelfAware\nunanswerable\n(hallucinate)": [J[f"sa_unans_{c}"]["answer"] for c in CONDS],
    "FalseQA\nfalse-premise\n(accept)": [J[f"fq_false_{c}"]["accept"] for c in CONDS],
}
x = np.arange(len(harm)); w = 0.16
for i, c in enumerate(CONDS):
    axA.bar(x + (i - 2) * w, [harm[k][i] for k in harm], w, color=cols[i], label=CLAB[i])
axA.set_xticks(x); axA.set_xticklabels(list(harm), fontsize=7.5)
axA.set_ylabel("harmful-commitment rate")
axA.set_title("(a) OOD transfer  (lower = better)", fontsize=9)
axA.set_ylim(0, 0.6); axA.legend(fontsize=6.5, ncol=2, loc="upper right", frameon=False)

# --- Panel B: preservation (higher = better) ---
axB = axes[1]
pres = {
    "SelfAware\nanswerable\n(answer)": [J[f"sa_ans_{c}"]["answer"] for c in CONDS],
    "FalseQA\ntrue-premise\n(answer)": [J[f"fq_true_{c}"]["answer"] for c in CONDS],
}
for i, c in enumerate(CONDS):
    axB.bar(x + (i - 2) * w, [pres[k][i] for k in pres], w, color=cols[i])
axB.set_xticks(x); axB.set_xticklabels(list(pres), fontsize=7.5)
axB.set_ylabel("correct-answer rate")
axB.set_title("(b) OOD preservation  (higher = better)", fontsize=9)
axB.set_ylim(0, 1.0)

# --- Panel C: capability cost (Δ perplexity, C4) ---
axC = axes[2]
rem = [(f"remove {r['sparsity']*100:.1f}%", r["ppl_delta_c4"] * 100) for r in P0["sweep"]]
amp = [(c["label"].replace("alphaW_a", "ampl xW ").replace("suppress_r", "suppr "), c["ppl_delta_c4"] * 100)
       for c in AMP["conditions"]]
bars = rem + amp
labs = [b[0] for b in bars]; vals = [b[1] for b in bars]
bc = ["#D55E00"] * len(rem) + ["#0072B2"] * len(amp)
y = np.arange(len(bars))
axC.barh(y, vals, 0.62, color=bc)
axC.axvline(0, color="#444", lw=0.6)
axC.set_yticks(y); axC.set_yticklabels(labs, fontsize=6.8); axC.invert_yaxis()
axC.set_xlabel("Δ perplexity (%, C4)")
axC.set_title("(c) capability cost", fontsize=9)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"epistemic_ood.{ext}", dpi=300, bbox_inches="tight")
print(f"saved figures/epistemic_ood.png / .pdf")
