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
plt.rcParams.update({"font.size": 16, "axes.labelsize": 19, "axes.titlesize": 18,
                     "xtick.labelsize": 17, "ytick.labelsize": 18, "xtick.top": False,
                     "ytick.right": False, "axes.linewidth": 1.0})

cell = defaultdict(list)
for src, jdir in [("method_cmp_qwen3-8b", "opus_judge"), ("method_cmp_fq_qwen3-8b", "opus_judge_fq"),
                  ("ood_tune_qwen3-8b", "opus_judge_tune"),
                  ("ood_ampextra_qwen3-8b", "opus_judge_amp56"),
                  ("blade_rho_sweep_qwen3-8b", "opus_judge_rho")]:
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


# Tuned, NON-degenerate operating points (from the fixed-L* ρ×α sweep). BLADE remove=α0, amp=α2.
METHODS = [("base", "base", "#3D405B"),
           ("ITI (α=2)", "iti_a2.0", "#9AD5CD"), ("ITI (α=4)", "iti_a4.0", "#3AA6A0"),
           ("ITI (α=6)", "iti_a6.0", "#0E6E6E"),
           ("BLADE (ρ=.01, α=2)", "r0.01_a2.0", "#D9532B")]

# Δppl (%) per method config for the capability-cost panel
_iti = json.load(open(R / "iti_ppl.json"))["alpha_ppl_delta"]
_grid = {g["cond"]: g["ppl_delta_c4"] * 100 for g in json.load(open(R / "blade_rho_sweep_qwen3-8b.json"))["grid"]}
PPL = {"base": 0.0, "iti_a2.0": _iti["iti_a2.0"] * 100, "iti_a4.0": _iti["iti_a4.0"] * 100,
       "iti_a6.0": _iti["iti_a6.0"] * 100, "r0.01_a2.0": _grid["r0.01_a2.0"]}
labs = [m[0] for m in METHODS]; cols = [m[2] for m in METHODS]; y = np.arange(len(METHODS)); H = 0.7
PAN = [("ppl", None, "capability cost", "Δ perplexity (%) ↓", True),
       ("selfaware", "unanswerable", "SelfAware unanswerable", "hallucination (%) ↓", True),
       ("selfaware", "answerable", "SelfAware answerable", "answered (%) ↑", False),
       ("falseqa", "false_premise", "FalseQA false-premise", "accepted (%) ↓", True),
       ("falseqa", "true_premise", "FalseQA true-premise", "answered (%) ↑", False)]
fig, axes = plt.subplots(1, 5, figsize=(21, 4.6), sharey=True)
for k, (ax, (ds, gold, title, xlab, low)) in enumerate(zip(axes.flat, PAN)):
    if ds == "ppl":
        vals = [PPL[c[1]] for c in METHODS]
        b = ax.barh(y, vals, H, color=cols, edgecolor="white", linewidth=0.9, zorder=3)
        ax.bar_label(b, fmt="%+.1f", fontsize=14, padding=2)
        ax.set_xlim(0, max(vals) * 1.28)
    else:
        vals = [rate(ds, gold, c[1]) for c in METHODS]
        b = ax.barh(y, vals, H, color=cols, edgecolor="white", linewidth=0.9, zorder=3)
        ax.bar_label(b, fmt="%.0f", fontsize=14, padding=2)
        ax.set_xlim(0, max([v for v in vals if v == v]) * (1.32 if low else 1.18))
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis()
    ax.set_xlabel(xlab, labelpad=8)                                   # metric = x-axis title
    ax.text(0.5, -0.42, f"({chr(97+k)}) {title}", transform=ax.transAxes,   # panel subtitle below
            ha="center", va="top", fontsize=16, fontweight="bold")
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
