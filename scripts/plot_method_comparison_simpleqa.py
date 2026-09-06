"""uncertainty_method_cmp_simpleqa: the uncertainty_method_cmp panels (capability + SelfAware un/ans +
FalseQA false/true) PLUS two SimpleQA panels (incorrect / correct-given-attempted) for the same 4 methods
(base, ITI c=4, ITI c=6, BLADE rho=.005 a2.5). Saved under a NEW name; does NOT overwrite
uncertainty_method_cmp. SimpleQA numbers: n=400, Qwen3-8B thinking-off, Opus-graded; BLADE uses the same
pinned edit L*=[23,31,18,2] as panels a-e. SelfAware/FalseQA/ppl loaded exactly as in plot_method_comparison.py."""
import json, os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import scienceplots  # noqa: F401

R = Path("results"); FIG = Path("figures"); FIG.mkdir(exist_ok=True)
SCS = Path("/tmp/claude-1000/-home-xfang1999-Projects-TestTimeSafetyPrune/"
           "e16f646c-64a5-440b-bd68-985c068d25df/scratchpad")
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 31, "axes.labelsize": 38, "axes.titlesize": 33,
                     "xtick.labelsize": 32, "ytick.labelsize": 34, "xtick.top": False,
                     "ytick.right": False, "axes.linewidth": 1.2})

cell = defaultdict(list)
for src, jdir in [("method_cmp_qwen3-8b", "opus_judge"), ("method_cmp_fq_qwen3-8b", "opus_judge_fq"),
                  ("ood_tune_qwen3-8b", "opus_judge_tune"),
                  ("ood_ampextra_qwen3-8b", "opus_judge_amp56"),
                  ("blade_rho_sweep_qwen3-8b", "opus_judge_rho"),
                  ("blade_rho_sweep_lowa_qwen3-8b", "opus_judge_lowa"),
                  ("blade_rho_sweep_lowa_qwen3-8b", "opus_judge_lowa_sa")]:
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
           ("ITI (c=4)", "iti_a4.0", "#3AA6A0"),
           ("ITI (c=6)", "iti_a6.0", "#0E6E6E"),
           ("BLADE\n(ρ=.005, α=2.5)", "r0.005_a2.5", "#D9532B")]

_iti = json.load(open(R / "iti_ppl.json"))["alpha_ppl_delta"]
_grid = {g["cond"]: g["ppl_delta_c4"] * 100 for g in json.load(open(R / "blade_rho_sweep_qwen3-8b.json"))["grid"]}
_grid.update({g["cond"]: g["ppl_delta_c4"] * 100 for g in json.load(open(R / "blade_rho_sweep_lowa_qwen3-8b.json"))["grid"]})
PPL = {"base": 0.0, "iti_a4.0": _iti["iti_a4.0"] * 100, "iti_a6.0": _iti["iti_a6.0"] * 100,
       "r0.005_a2.5": _grid["r0.005_a2.5"]}

# SimpleQA (n=400, Opus-graded correct/incorrect/not-attempted); keyed by method cond.
# BLADE uses the SAME edit as panels a-e (L*=[23,31,18,2], pinned; simpleqa_bladepin run).
SQ_INC = {"base": 89.0, "iti_a4.0": 74.2, "iti_a6.0": 63.0, "r0.005_a2.5": 82.8}
# correct-given-attempted = correct/(correct+incorrect); base 3.8/89.0, c4 4.5/74.2, c6 3.8/63.0, blade 3.2/82.8
SQ_CGA = {"base": 4.0, "iti_a4.0": 5.7, "iti_a6.0": 5.6, "r0.005_a2.5": 3.8}

labs = [m[0] for m in METHODS]; cols = [m[2] for m in METHODS]; y = np.arange(len(METHODS)); H = 0.7
# (kind, key, title, xlab, low)  kind in {ppl, judge, sq}
PAN = [("ppl", None, "capability\ncost", "Δ perplexity (%) ↓", True),
       ("judge", ("selfaware", "unanswerable"), "SelfAware\nunanswerable", "hallucination (%) ↓", True),
       ("judge", ("selfaware", "answerable"), "SelfAware\nanswerable", "answered (%) ↑", False),
       ("judge", ("falseqa", "false_premise"), "FalseQA\nfalse-premise", "accepted (%) ↓", True),
       ("judge", ("falseqa", "true_premise"), "FalseQA\ntrue-premise", "answered (%) ↑", False),
       ("sq", SQ_INC, "SimpleQA\nincorrect", "incorrect (%) ↓", True),
       ("sq", SQ_CGA, "SimpleQA\nattempted acc.", "corr. | att. (%) ↑", False)]
fig, axes = plt.subplots(1, 7, figsize=(34, 9.8), sharey=True)
for k, (ax, (kind, key, title, xlab, low)) in enumerate(zip(axes.flat, PAN)):
    if kind == "ppl":
        vals = [PPL[c[1]] for c in METHODS]
        ax.barh(y, vals, H, color=cols, edgecolor="white", linewidth=0.9, zorder=3)
        hi = max(vals)
        for yi, v in zip(y, vals):
            ax.text(max(v, 0) + hi * 0.012, yi, f"{v:+.1f}", va="center", ha="left", fontsize=27)
        ax.set_xlim(0, hi * 1.28)
    else:
        if kind == "judge":
            vals = [rate(key[0], key[1], c[1]) for c in METHODS]
        else:  # sq
            vals = [key[c[1]] for c in METHODS]
        b = ax.barh(y, vals, H, color=cols, edgecolor="white", linewidth=0.9, zorder=3)
        vmax = max([v for v in vals if v == v])
        ax.bar_label(b, fmt="%.1f" if vmax < 15 else "%.0f", fontsize=27, padding=2)
        ax.set_xlim(0, vmax * (1.32 if low else 1.22))
    ax.set_yticks(y); ax.set_yticklabels(labs, linespacing=0.85); ax.invert_yaxis()
    ax.set_xlabel(xlab, labelpad=8)
    ax.text(0.0, -0.42, f"({chr(97+k)}) {title}", transform=ax.transAxes,
            ha="left", va="top", ma="left", fontsize=44, fontweight="normal", linespacing=0.9)
    ax.xaxis.grid(True, ls="-", lw=0.5, color="#DFDFDF", zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
# dashed dividers between dataset groups: capability | SelfAware | FalseQA | SimpleQA
p0 = axes[0].get_position(); y_top = p0.y1; y_bot = p0.y0 - 0.42 * p0.height
for a, b_ in [(0, 1), (2, 3), (4, 5)]:
    xd = (axes[a].get_position().x1 + axes[b_].get_position().x0) / 2
    fig.add_artist(mlines.Line2D([xd, xd], [y_bot, y_top], color="#9AA0A6", ls=(0, (5, 4)),
                                 lw=1.3, transform=fig.transFigure))
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"uncertainty_method_cmp_simpleqa.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/uncertainty_method_cmp_simpleqa.png / .pdf")
