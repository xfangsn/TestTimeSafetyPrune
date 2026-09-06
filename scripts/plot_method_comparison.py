"""Method comparison (Opus-judged): base vs ITI (2306.03341) vs our BLADE, on SelfAware + FalseQA
(closed-book). solo_vs_joint_llama palette, compact, large fonts, subtitle BELOW. DoLa pending."""
import json
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
plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16,
                     "xtick.labelsize": 13, "ytick.labelsize": 14, "xtick.top": False,
                     "ytick.right": False, "axes.linewidth": 1.0})

METHODS = [("base", "base", "#3D405B"), ("ITI α=2", "iti_a2.0", "#3AA6A0"),
           ("ITI α=6", "iti_a6.0", "#0E6E6E"), ("BLADE amp.", "amplify", "#EE6C4D"),
           ("BLADE rem.", "remove", "#B8860B")]
labs = [m[0] for m in METHODS]; cols = [m[2] for m in METHODS]


def load(srcname, judgedir):
    items = json.load(open(R / f"{srcname}.json"))["items"]
    mp = json.load(open(SCS / judgedir / "map.json"))
    lab = {}
    for f in ("labels_A.json", "labels_B.json"):
        p = SCS / judgedir / f
        if p.exists():
            lab.update(json.load(open(p)))
    cell = defaultdict(list)
    for jid, m in mp.items():
        if jid in lab:
            cell[(items[m["idx"]]["gold"], m["cond"])].append(lab[jid]["act"])
    return cell


def rate(cell, gold, cond, acts):
    xs = cell[(gold, cond)]
    return 100 * sum(a in acts for a in xs) / len(xs) if xs else float("nan")


SA = load("method_cmp_qwen3-8b", "opus_judge")
FQ = load("method_cmp_fq_qwen3-8b", "opus_judge_fq")
y = np.arange(len(METHODS)); H = 0.66

# panels: (cell, gold, acts, title, lower_better)
PAN = [
    (SA, "unanswerable", ["answer"], "SelfAware unanswerable\nhallucination (%) ↓", True),
    (SA, "answerable", ["answer"], "SelfAware answerable\nanswered (%) ↑", False),
    (FQ, "false_premise", ["answer"], "FalseQA false-premise\naccepted (%) ↓", True),
    (FQ, "true_premise", ["answer"], "FalseQA true-premise\nanswered (%) ↑", False),
]
fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.2), sharey=True)
for ax, (cell, gold, acts, title, low) in zip(axes.flat, PAN):
    vals = [rate(cell, gold, c[1], acts) for c in METHODS]
    b = ax.barh(y, vals, H, color=cols, edgecolor="white", linewidth=0.9, zorder=3)
    ax.bar_label(b, fmt="%.0f", fontsize=12, padding=2)
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis()
    ax.set_xlim(0, max(100, max([v for v in vals if v == v]) * 1.18) if not low
               else max([v for v in vals if v == v]) * 1.28)
    ax.set_title(title, fontweight="bold", pad=4)
    ax.xaxis.grid(True, ls="-", lw=0.5, color="#DFDFDF", zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

fig.text(0.5, -0.04, "Ours vs ITI on closed-book abstention (Qwen3-8B, Opus-judged). ITI cuts "
         "hallucination only at high α, which collapses answering; BLADE is bidirectional and preserves it.",
         ha="center", fontsize=12)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"uncertainty_method_cmp.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/uncertainty_method_cmp.png / .pdf")
for lab_, m in zip(labs, METHODS):
    print(f"  {lab_:11s} SA-un {rate(SA,'unanswerable',m[1],['answer']):5.1f}  SA-ans "
          f"{rate(SA,'answerable',m[1],['answer']):5.1f}  FQ-false-acc {rate(FQ,'false_premise',m[1],['answer']):5.1f}"
          f"  FQ-true-ans {rate(FQ,'true_premise',m[1],['answer']):5.1f}")
