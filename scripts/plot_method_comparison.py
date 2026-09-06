"""uncertainty_method_cmp (Opus-judged): base vs ITI (2306.03341, α=2/4/6) vs our BLADE (amplify α×2/×4,
remove) on SelfAware + FalseQA (closed-book). solo_vs_joint palette, compact, large fonts, subtitle BELOW.
Shows BLADE's tunable frontier dominates ITI at aggressive strength. DoLa pending (native broken tf5.15)."""
import json, os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

R = Path("results"); FIG = Path("figures"); FIG.mkdir(exist_ok=True)
SCS = Path("/tmp/claude-1000/-home-xfang1999-Projects-TestTimeSafetyPrune/"
           "e16f646c-64a5-440b-bd68-985c068d25df/scratchpad")
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 14, "axes.labelsize": 17, "axes.titlesize": 16,
                     "xtick.labelsize": 15, "ytick.labelsize": 16, "xtick.top": False,
                     "ytick.right": False, "axes.linewidth": 1.0})

cell = defaultdict(list)
for src, jdir in [("method_cmp_qwen3-8b", "opus_judge"), ("method_cmp_fq_qwen3-8b", "opus_judge_fq"),
                  ("ood_tune_qwen3-8b", "opus_judge_tune")]:
    items = json.load(open(R / f"{src}.json"))["items"]
    mp = json.load(open(SCS / jdir / "map.json")); lab = {}
    for f in ("labels_A.json", "labels_B.json", "labels_C.json"):
        p = SCS / jdir / f
        if p.exists():
            lab.update(json.load(open(p)))
    for jid, m in mp.items():
        if jid in lab:
            it = items[m["idx"]]
            cell[(it["dataset"], it["gold"], m["cond"])].append(lab[jid]["act"])


def rate(ds, gold, cond, acts=("answer",)):
    xs = cell[(ds, gold, cond)]
    return 100 * sum(a in acts for a in xs) / len(xs) if xs else float("nan")


METHODS = [("base", "base", "#3D405B"),
           ("ITI α=2", "iti_a2.0", "#9AD5CD"), ("ITI α=4", "iti_a4.0", "#3AA6A0"),
           ("ITI α=6", "iti_a6.0", "#0E6E6E"),
           ("BLADE amp ×2", "amplify", "#F4A98F"), ("BLADE amp ×4", "blade_ampW4.0", "#EE6C4D"),
           ("BLADE remove", "remove", "#B8860B")]

# Δppl (%) per method config for the capability-cost panel
_iti = json.load(open(R / "iti_ppl.json"))["alpha_ppl_delta"]
_amp = json.load(open(R / "amp_ppl.json"))["amp_ppl_delta"]
_rem = [r for r in json.load(open(R / "epistemic_p0_qwen3-8b_bladeg.json"))["sweep"]
        if abs(r["sparsity"] - 0.005) < 1e-9][0]["ppl_delta_c4"] * 100
PPL = {"base": 0.0, "iti_a2.0": _iti["iti_a2.0"] * 100, "iti_a4.0": _iti["iti_a4.0"] * 100,
       "iti_a6.0": _iti["iti_a6.0"] * 100, "amplify": _amp["ampW2.0"] * 100,
       "blade_ampW4.0": _amp["ampW4.0"] * 100, "remove": _rem}
labs = [m[0] for m in METHODS]; cols = [m[2] for m in METHODS]; y = np.arange(len(METHODS)); H = 0.7
PAN = [("ppl", None, "capability cost", "Δ perplexity (%) ↓", True),
       ("selfaware", "unanswerable", "SelfAware unanswerable", "hallucination (%) ↓", True),
       ("selfaware", "answerable", "SelfAware answerable", "answered (%) ↑", False),
       ("falseqa", "false_premise", "FalseQA false-premise", "accepted (%) ↓", True),
       ("falseqa", "true_premise", "FalseQA true-premise", "answered (%) ↑", False)]
fig, axes = plt.subplots(1, 5, figsize=(20.5, 3.4), sharey=True)
for ax, (ds, gold, title, xlab, low) in zip(axes.flat, PAN):
    if ds == "ppl":
        vals = [PPL[c[1]] for c in METHODS]
        b = ax.barh(y, vals, H, color=cols, edgecolor="white", linewidth=0.9, zorder=3)
        ax.bar_label(b, fmt="%+.1f", fontsize=12, padding=2)
        ax.set_xlim(0, max(vals) * 1.25)
    else:
        vals = [rate(ds, gold, c[1]) for c in METHODS]
        b = ax.barh(y, vals, H, color=cols, edgecolor="white", linewidth=0.9, zorder=3)
        ax.bar_label(b, fmt="%.0f", fontsize=12, padding=2)
        ax.set_xlim(0, max([v for v in vals if v == v]) * (1.3 if low else 1.18))
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis()
    ax.set_title(title, fontweight="bold", pad=6)
    ax.set_xlabel(xlab)
    ax.xaxis.grid(True, ls="-", lw=0.5, color="#DFDFDF", zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
# dashed divider between SelfAware (panels 1-2) and FalseQA (panels 3-4)
import matplotlib.lines as mlines
for a, b_ in [(0, 1), (2, 3)]:   # capability | SelfAware | FalseQA
    xd = (axes[a].get_position().x1 + axes[b_].get_position().x0) / 2
    fig.add_artist(mlines.Line2D([xd, xd], [0.04, 0.96], color="#9AA0A6", ls=(0, (5, 4)),
                                 lw=1.3, transform=fig.transFigure))
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"uncertainty_method_cmp.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/uncertainty_method_cmp.png / .pdf")
for lb, m in zip(labs, METHODS):
    print(f"  {lb:14s} SAun {rate('selfaware','unanswerable',m[1]):5.1f} SAans {rate('selfaware','answerable',m[1]):5.1f}"
          f" FQfa {rate('falseqa','false_premise',m[1]):5.1f} FQtp {rate('falseqa','true_premise',m[1]):5.1f}")
