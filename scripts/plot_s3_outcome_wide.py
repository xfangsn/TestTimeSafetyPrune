"""Landscape crossmodel figure coloured by OUTCOME of BLADE removal, using the
BEST result over the perplexity-budget sweep (beta in {5,10,100}%) per cell -- so
diffuse behaviours that only crack at higher budget (e.g. sycophancy) show their
true achievable outcome. Also annotates pruning rate (% of total params zeroed)
alongside Delta-ppl.

Outcomes (colour + marker, never colour alone):
  no preference : A/B post in [0.45,0.55]      green  o
  reversed      : A/B post < 0.45              blue   <
  partial       : A/B post > 0.55              orange s
  removed       : refusal post <= 0.10         green  *
  not exhibited : behaviour absent             grey   x
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots  # noqa: F401

# model -> (tag, per-layer writer size, total params)
MODELS = {
    "Llama-3.2-3B": ("llama-32-3b-instruct", 34_603_008, 3_212_749_824),
    "Qwen3-4B":     ("qwen3-4b",             35_389_440, 4_022_468_096),
    "Gemma-3-4B":   ("gemma-3-4b-it",        31_457_280, 4_300_079_472),
    "Phi-4-mini":   ("phi-4-mini-instruct",  34_603_008, 3_836_021_760),
}
ORDER = ["refusal", "power-seeking", "wealth-seeking", "deception",
         "corrigibility", "self-awareness", "self-rate-highly", "sycophancy", "evil"]
LEFT, RIGHT = ORDER[:4], ORDER[4:]   # 4 left, 5 right
OUT = {
    "no_pref":  ("#009E73", "o", "no preference (chance $\\pm$0.05)"),
    "removed":  ("#009E73", "*", "refusal removed ($\\to$0)"),
    "reversed": ("#0072B2", "<", "reversed (past chance)"),
    "partial":  ("#D55E00", "s", "partial (still biased)"),
}
BAND = "#d7dbd5"
PPL_X = 1.06   # single right-edge column: "Δppl · prune%"
BETAFILES = {"Llama-3.2-3B": [5, 10, 200], "Qwen3-4B": [5, 10, 100],
             "Gemma-3-4B": [5, 10, 100], "Phi-4-mini": [5, 10, 100]}


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _abcell(r):
    """(base, post, sparsity, nL, ppl) from an els-style behavior record."""
    if not r or r.get("skipped") or not r.get("Lstar_sweep"):
        return None
    b = min(r["Lstar_sweep"], key=lambda s: s["pick_rate"])
    return (r["baseline_bias"], b["pick_rate"], b["sparsity"],
            len(r["L_star"]), b["ppl_delta"])


def best_over_beta(beh, tag, betas):
    cands = []
    # "evil" lives in a dedicated _evil file (beta=5% only, all skip on aligned models)
    if beh == "evil":
        ev = _load(f"results/blade_els_{tag}_evil.json")
        r = ev["results"].get("evil") if ev else None
        c = _abcell(r) if r else None
        if c:
            return ("ok",) + c
        base = r.get("baseline_bias") if r else 0.5
        return ("none", base, None, None, None)
    # beta=5%: layer_select (Llama/Qwen/Gemma) or els (Phi)
    ls = _load(f"results/blade_layer_select_{tag}.json")
    if ls and beh in ls["behaviors"]:
        r = ls["behaviors"][beh]; s3 = r.get("S3_bestfirst")
        if s3 and s3.get("result"):
            res = s3["result"]
            cands.append((r["baseline"], res["pick"], res["sparsity"],
                          len(s3["L"]), res["ppl_delta"]))
    els5 = _load(f"results/blade_els_{tag}.json")
    if els5 and beh in els5["results"]:
        c = _abcell(els5["results"][beh])
        if c:
            cands.append(c)
    # beta=10/100 sweeps
    for bt in betas:
        d = _load(f"results/blade_els_{tag}_beta{bt}.json")
        if d and beh in d["results"]:
            c = _abcell(d["results"][beh])
            if c:
                cands.append(c)
    if not cands:
        # exhibited-but-not-removable / not exhibited -> report a baseline if any
        base = None
        if ls and beh in ls["behaviors"]:
            base = ls["behaviors"][beh].get("baseline")
        if base is None and els5 and beh in els5["results"]:
            base = els5["results"][beh].get("baseline_bias")
        return ("none", base if base is not None else 0.5, None, None, None)
    base, post, sp, nL, ppl = min(cands, key=lambda c: c[1])  # deepest removal
    return ("ok", base, post, sp, nL, ppl)


def load_all():
    out = {b: {} for b in ORDER}
    for name, (tag, _pl, _tot) in MODELS.items():
        for beh in ORDER:
            if beh == "refusal":
                rf = _load(f"results/blade_refusal_els_{tag}.json")
                if rf and rf.get("Lstar_sweep"):
                    b = min(rf["Lstar_sweep"], key=lambda s: s["refusal"])
                    out["refusal"][name] = ("ok", rf["base_refusal"], b["refusal"],
                                            b["sparsity"], len(rf["L_star"]),
                                            b["ppl_delta"])
                continue
            out[beh][name] = best_over_beta(beh, tag, BETAFILES[name])
    # Gemma self-rate gate-bypassed probe
    p = _load("results/blade_gemma_selfrate_probe.json")
    if p and p["S3_bestfirst"].get("result"):
        r = p["S3_bestfirst"]["result"]
        out["self-rate-highly"]["Gemma-3-4B"] = (
            "ok", p["baseline"], r["pick"], r["sparsity"],
            len(p["S3_bestfirst"]["L"]), r["ppl_delta"])
    return out


DATA = load_all()
CHANCE = {b: (0.0 if b == "refusal" else 0.5) for b in ORDER}


def outcome(beh, post):
    if beh == "refusal":
        return "removed" if post <= 0.10 else "partial"
    if post < 0.45:
        return "reversed"
    if post <= 0.55:
        return "no_pref"
    return "partial"


def prune_pct(name, sp, nL):
    _tag, per_layer, total = MODELS[name]
    return 100 * round(sp * nL * per_layer) / total


def draw_panel(ax, behaviors):
    row_h, grp_gap = 1.0, 1.9
    y = 0.0
    yticks, ylabels = [], []
    for beh in behaviors:
        chance = CHANCE[beh]
        if beh != "refusal":
            ax.axvspan(0.45, 0.55, color=BAND, zorder=0, lw=0)
        ax.text(0.0, y - 0.9, beh, ha="left", va="center", fontsize=15.5,
                fontweight="bold", clip_on=False)
        for model in MODELS:
            cell = DATA[beh].get(model)
            if not cell:
                y += row_h; continue
            st = cell[0]
            ax.plot([chance, chance], [y - 0.42, y + 0.42], color="#9aa0a6", lw=1.0, zorder=1)
            if st == "ok":
                _, base, post, sp, nL, ppl = cell
                oc = outcome(beh, post); c, mk, _ = OUT[oc]
                ax.plot([post, base], [y, y], color=c, lw=3.0, zorder=2,
                        clip_on=False, solid_capstyle="round")
                ax.scatter([base], [y], s=120, facecolors="white", edgecolors="#555",
                           lw=1.8, zorder=3, clip_on=False)
                ax.scatter([post], [y], s=190 if mk == "*" else 150, facecolors=c,
                           edgecolors=c, marker=mk, lw=0, zorder=4, clip_on=False)
                ax.text(PPL_X, y, f"{ppl*100:+.1f}%  ·  {prune_pct(model, sp, nL):.3f}%",
                        ha="left", va="center", fontsize=11.5, color="#555",
                        transform=ax.get_yaxis_transform(), clip_on=False)
            else:
                base = cell[1]
                ax.scatter([base], [y], s=130, marker="x", color="#9aa0a6",
                           lw=2.4, zorder=3, clip_on=False)
                ax.annotate("not exhibited", (base, y), textcoords="offset points",
                            xytext=(13, 0), ha="left", va="center", fontsize=11.5,
                            color="#8a8f86", style="italic")
            yticks.append(y); ylabels.append(model); y += row_h
        y += grp_gap
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=12.5)
    ax.invert_yaxis(); ax.set_xlim(-0.03, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.tick_params(axis="x", labelsize=13.5)
    ax.text(PPL_X, yticks[0] - 1.7, r"$\Delta$ppl $\cdot$ prune%", ha="left",
            va="center", fontsize=12, fontweight="bold", color="#555",
            transform=ax.get_yaxis_transform(), clip_on=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.margins(y=0.02)


plt.rcParams["font.family"] = "serif"
with plt.style.context(["science", "no-latex"]):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.0, 10.6))
    plt.rc("font", size=15)
    draw_panel(axL, LEFT); draw_panel(axR, RIGHT)
    fig.subplots_adjust(wspace=0.92, bottom=0.20, top=0.965, left=0.10, right=0.86)
    fig.text(0.5, 0.115, "behavior score after BLADE (best over $\\beta$ budget)   "
             "(A/B: MC pick-rate, chance $=0.5$ | refusal: refusal-rate, target $=0$)",
             ha="center", fontsize=14.5)
    fig.text(0.5, 0.075, "shaded band = no-preference zone (chance $\\pm$0.05); open circle = baseline, "
             "filled marker = post-BLADE; prune% = weights zeroed / total params",
             ha="center", fontsize=10.5, style="italic", color="#555")
    handles = [Line2D([0], [0], marker=mk, color="w", markerfacecolor=c,
                      markeredgecolor=c, markersize=13, lw=0, label=lab)
               for (c, mk, lab) in OUT.values()]
    handles.append(Line2D([0], [0], marker="x", color="#9aa0a6", lw=0, markersize=12,
                          label="not exhibited"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=5, frameon=False, fontsize=12, handletextpad=0.4, columnspacing=1.4)
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/blade_outcome_crossmodel_wide.pdf", bbox_inches="tight")
    fig.savefig("results/blade_outcome_crossmodel_wide.png", dpi=300, bbox_inches="tight")
    print("saved figures/blade_outcome_crossmodel_wide.pdf")
