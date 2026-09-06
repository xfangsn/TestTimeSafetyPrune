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
plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "axes.titlesize": 15,
                     "xtick.labelsize": 12, "ytick.labelsize": 13, "xtick.top": False,
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
labs = [m[0] for m in METHODS]; cols = [m[2] for m in METHODS]; y = np.arange(len(METHODS)); H = 0.7
PAN = [("selfaware", "unanswerable", "SelfAware unanswerable\nhallucination (%) ↓", True),
       ("selfaware", "answerable", "SelfAware answerable\nanswered (%) ↑", False),
       ("falseqa", "false_premise", "FalseQA false-premise\naccepted (%) ↓", True),
       ("falseqa", "true_premise", "FalseQA true-premise\nanswered (%) ↑", False)]
fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0), sharey=True)
for ax, (ds, gold, title, low) in zip(axes.flat, PAN):
    vals = [rate(ds, gold, c[1]) for c in METHODS]
    b = ax.barh(y, vals, H, color=cols, edgecolor="white", linewidth=0.9, zorder=3)
    ax.bar_label(b, fmt="%.0f", fontsize=11, padding=2)
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis()
    top = max([v for v in vals if v == v])
    ax.set_xlim(0, top * (1.3 if low else 1.18))
    ax.set_title(title, fontweight="bold", pad=4)
    ax.xaxis.grid(True, ls="-", lw=0.5, color="#DFDFDF", zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"uncertainty_method_cmp.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/uncertainty_method_cmp.png / .pdf")
for lb, m in zip(labs, METHODS):
    print(f"  {lb:14s} SAun {rate('selfaware','unanswerable',m[1]):5.1f} SAans {rate('selfaware','answerable',m[1]):5.1f}"
          f" FQfa {rate('falseqa','false_premise',m[1]):5.1f} FQtp {rate('falseqa','true_premise',m[1]):5.1f}")
