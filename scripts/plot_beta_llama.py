"""Single model (Llama-3.2-3B): how each behavior's BLADE outcome moves as the
perplexity budget beta grows (5% -> 10% -> 50% -> 100%). Two panels sharing x=beta
(small multiples, not dual-axis): (left) post-pick-rate, (right) Delta-ppl cost.
beta=100 is read from the beta200 file (proven identical -- beta saturates by 100%).
Categorical colour by behavior (fixed order), legend + direct labels. Wong palette.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

TAG = "llama-32-3b-instruct"
PER_LAYER = 34_603_008          # Llama o_proj+down_proj per decoder block
TOTAL = 3_212_749_824           # Llama-3.2-3B total params
BETAS = [5, 10, 50, 100]                 # x positions (100 read from beta200)
FILEBETA = {5: 5, 10: 10, 50: 50, 100: 200}
BEHS = ["power-seeking", "wealth-seeking", "corrigibility", "deception",
        "self-rate-highly", "self-awareness", "sycophancy"]
COL = {"power-seeking": "#0072B2", "wealth-seeking": "#D55E00",
       "corrigibility": "#009E73", "deception": "#CC79A7",
       "self-rate-highly": "#56B4E9", "self-awareness": "#E69F00",
       "sycophancy": "#111111"}


def load(beta_file):
    p = f"results/blade_els_{TAG}_beta{beta_file}.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))["results"]


def best(r):
    if not r or r.get("skipped") or not r.get("Lstar_sweep"):
        return None, None, None
    b = min(r["Lstar_sweep"], key=lambda s: s["pick_rate"])
    k = round(b["sparsity"] * len(r["L_star"]) * PER_LAYER)   # weights zeroed
    prune_pct = 100 * k / TOTAL                               # % of total params
    return b["pick_rate"], b["ppl_delta"] * 100, prune_pct


data = {b: load(FILEBETA[b]) for b in BETAS}
xs = list(range(len(BETAS)))

plt.rcParams["font.family"] = "serif"
with plt.style.context(["science", "no-latex"]):
    fig, (axP, axQ, axR) = plt.subplots(1, 3, figsize=(16.5, 6.2))
    plt.rc("font", size=14)
    for beh in BEHS:
        pk, pp, pr = [], [], []
        for b in BETAS:
            d = data[b]
            r = d.get(beh) if d else None
            p, q, rr = best(r)
            pk.append(p); pp.append(q); pr.append(rr)
        c = COL[beh]
        for ax, series, lab in ((axP, pk, beh), (axQ, pp, None), (axR, pr, beh)):
            xv = [x for x, v in zip(xs, series) if v is not None]
            yv = [v for v in series if v is not None]
            if not yv:
                continue
            ax.plot(xv, yv, "-o", color=c, lw=2.2, ms=8, clip_on=False,
                    label=lab if lab else None)
            if lab:
                ax.annotate(lab, (xv[-1], yv[-1]), textcoords="offset points",
                            xytext=(8, 0), va="center", ha="left", fontsize=10, color=c)

    axP.axhline(0.5, ls="--", lw=1.2, color="#9aa0a6", zorder=0)
    axP.text(0.02, 0.51, "chance", fontsize=10, color="#8a8f86")
    for ax in (axP, axQ, axR):
        ax.set_xticks(xs); ax.set_xticklabels([f"{b}%" for b in BETAS], fontsize=12.5)
        ax.set_xlabel(r"perplexity budget $\beta$", fontsize=13.5)
        ax.set_xlim(-0.15, len(BETAS) - 1 + 1.05)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axP.set_ylabel("post-BLADE MC pick-rate", fontsize=13.5)
    axP.set_ylim(0.15, 0.95)
    axP.set_title("(a) removal depth", fontsize=14, fontweight="bold")
    axQ.set_ylabel(r"$\Delta$ppl (%)", fontsize=13.5)
    axQ.set_title("(b) capability cost", fontsize=14, fontweight="bold")
    axQ.axhline(5, ls=":", lw=1.0, color="#b0b4ae")
    axR.set_ylabel("pruning rate (% of total params)", fontsize=13.5)
    axR.set_title("(c) weights removed", fontsize=14, fontweight="bold")

    fig.suptitle("Llama-3.2-3B: behavior removal vs perplexity budget "
                 r"$\beta$ (post-pick, ppl cost, pruning rate)",
                 fontsize=15.5, fontweight="bold", y=0.99)
    fig.subplots_adjust(wspace=0.30, bottom=0.12, top=0.88, left=0.055, right=0.93)
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/blade_beta_llama.pdf", bbox_inches="tight")
    fig.savefig("results/blade_beta_llama.png", dpi=300, bbox_inches="tight")
    print("saved figures/blade_beta_llama.pdf")
