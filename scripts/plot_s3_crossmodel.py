"""S3 BLADE (best-first ELS) across 3 models x all behaviors.

Loads real numbers from results/blade_layer_select_*.json (S3_bestfirst) and
results/blade_refusal_els_*.json (refusal). Dumbbell: baseline (open) ->
post-BLADE (filled), per behavior x model. Per-row chance tick unifies the
mixed metric (A/B pick-rate chance=0.5; refusal-rate target=0). grey x =
behavior not exhibited (|base-0.5|<0.10, S3 L* empty). ppl-delta annotated
above each post marker. Wong CVD-safe palette, serif.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots  # noqa: F401

MC = {"Llama-3.2-3B": "#0072B2", "Qwen3-4B": "#D55E00", "Gemma-3-4B": "#009E73",
      "Phi-4-mini": "#CC79A7"}
TAGS = {
    "Llama-3.2-3B": "llama-32-3b-instruct",
    "Qwen3-4B": "qwen3-4b",
    "Gemma-3-4B": "gemma-3-4b-it",
    "Phi-4-mini": "phi-4-mini-instruct",
}
# top -> bottom: cleanest removal first, diffuse last
ORDER = ["refusal", "power-seeking", "wealth-seeking", "deception",
         "corrigibility", "self-awareness", "self-rate-highly", "sycophancy"]
MIN_BIAS = 0.10   # exhibited/not gate used by the pipeline
NULL_BAND = 0.05  # below this -> genuinely at chance; between -> weak-but-ungated


def _classify_empty(base):
    ad = abs(base - 0.5)
    return "none" if ad < NULL_BAND else "weak" if ad < MIN_BIAS else "diffuse"


def _from_layer_select(tag, out, name):
    beh = json.load(open(f"results/blade_layer_select_{tag}.json"))["behaviors"]
    for b, r in beh.items():
        if b not in out:
            continue
        base = r["baseline"]
        s3 = r.get("S3_bestfirst")
        if s3 and s3.get("result"):
            res = s3["result"]
            out[b][name] = ("ok", base, res["pick"], f"{res['ppl_delta']*100:+.1f}%")
        else:
            out[b][name] = (_classify_empty(base), base, None, None)


def _from_els(tag, out, name):
    res = json.load(open(f"results/blade_els_{tag}.json"))["results"]
    for b, r in res.items():
        if b not in out:
            continue
        base = r["baseline_bias"]
        if r.get("skipped") or not r.get("L_star") or not r.get("Lstar_sweep"):
            out[b][name] = (_classify_empty(base), base, None, None)
        else:
            best = min(r["Lstar_sweep"], key=lambda s: s["pick_rate"])
            out[b][name] = ("ok", base, best["pick_rate"],
                            f"{best['ppl_delta']*100:+.1f}%")


def load_all():
    """behavior -> {model -> (state, base, post, ppl_str)}"""
    out = {b: {} for b in ORDER}
    for name, tag in TAGS.items():
        if os.path.exists(f"results/blade_layer_select_{tag}.json"):
            _from_layer_select(tag, out, name)
        elif os.path.exists(f"results/blade_els_{tag}.json"):
            _from_els(tag, out, name)
        rf_path = f"results/blade_refusal_els_{tag}.json"
        if os.path.exists(rf_path):
            rf = json.load(open(rf_path))
            sweep = rf.get("Lstar_sweep", [])
            if sweep:
                best = min(sweep, key=lambda s: s["refusal"])
                out["refusal"][name] = ("ok", rf["base_refusal"], best["refusal"],
                                        f"{best['ppl_delta']*100:+.1f}%")
    # Gemma self-rate-highly is below the 0.10 gate (base 0.58); a gate-bypassed
    # best-first probe shows it IS removable -> show it as a real dumbbell.
    pf = "results/blade_gemma_selfrate_probe.json"
    if os.path.exists(pf):
        p = json.load(open(pf))
        r = p["S3_bestfirst"]["result"]
        if r and r["pick"] < p["baseline"]:
            out["self-rate-highly"]["Gemma-3-4B"] = (
                "ok", p["baseline"], r["pick"], f"{r['ppl_delta']*100:+.1f}%†")
    return out


DATA = load_all()
CHANCE = {b: (0.0 if b == "refusal" else 0.5) for b in ORDER}

PPL_X = 1.09   # dedicated right-edge column for ppl deltas
plt.rcParams["font.family"] = "serif"
with plt.style.context(["science", "no-latex"]):
    fig, ax = plt.subplots(figsize=(11.2, 14.8))
    plt.rc("font", size=14)
    row_h, grp_gap = 1.0, 1.5
    y = 0.0
    yticks, ylabels, group_spans = [], [], []
    for beh in ORDER:
        chance = CHANCE[beh]
        g_top = y
        for model in MC:  # fixed model order per group
            st, base, after, ppl = DATA[beh].get(model, ("none", 0.5, None, None))
            c = MC[model]
            ax.plot([chance, chance], [y - 0.42, y + 0.42], color="#b0b4ae",
                    lw=1.0, zorder=1)
            if st == "ok":
                ax.plot([after, base], [y, y], color=c, lw=2.6, zorder=2,
                        clip_on=False, solid_capstyle="round")
                ax.scatter([base], [y], s=105, facecolors="white", edgecolors=c,
                           lw=2.0, zorder=3, clip_on=False)
                ax.scatter([after], [y], s=120, facecolors=c, edgecolors=c,
                           lw=0, zorder=4, clip_on=False)
                # ppl in a dedicated right-edge column (no collision with markers)
                ax.text(PPL_X, y, ppl, ha="left", va="center", fontsize=11,
                        color=c, fontweight="bold",
                        transform=ax.get_yaxis_transform(), clip_on=False)
            elif st == "none":
                ax.scatter([base], [y], s=100, marker="x", color="#9aa0a6",
                           lw=2.2, zorder=3, clip_on=False)
                ax.annotate("not exhibited", (base, y), textcoords="offset points",
                            xytext=(14, 0), ha="left", va="center", fontsize=10.5,
                            color="#8a8f86", style="italic")
            elif st == "weak":  # mild real preference, below the 0.10 removal gate
                ax.scatter([base], [y], s=100, marker="x", color=c,
                           lw=2.2, zorder=3, clip_on=False, alpha=0.85)
                ax.annotate(f"weak ({base:.2f}, below gate)", (base, y),
                            textcoords="offset points", xytext=(14, 0), ha="left",
                            va="center", fontsize=10.5, color=c, style="italic")
            else:  # diffuse: exhibited but S3 found no removable layer
                ax.scatter([base], [y], s=98, marker="s", facecolors="white",
                           edgecolors="#C99700", lw=2.2, zorder=3, clip_on=False)
                ax.annotate(r"L*=$\varnothing$ (diffuse)", (base, y),
                            textcoords="offset points", xytext=(14, 0), ha="left",
                            va="center", fontsize=10.5, color="#A67C00", style="italic")
            yticks.append(y)
            ylabels.append(model)  # full model version on the axis
            y += row_h
        group_spans.append((beh, g_top, y - row_h))
        y += grp_gap

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=12.5)
    ax.set_xlabel("behavior score   (A/B: MC pick-rate, chance $=0.5$   |   "
                  "refusal: refusal-rate, target $=0$)", fontsize=13)
    # behavior name = horizontal bold header in the far-left margin (fits long names)
    for beh, top, bot in group_spans:
        ax.text(-0.40, (top + bot) / 2, beh, rotation=0, va="center", ha="left",
                fontsize=12.5, fontweight="bold",
                transform=ax.get_yaxis_transform(), clip_on=False)
    # right-column header
    ax.text(PPL_X, group_spans[0][1] - 0.95, r"$\Delta$ppl", ha="left", va="center",
            fontsize=11.5, fontweight="bold", color="#444",
            transform=ax.get_yaxis_transform(), clip_on=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.margins(y=0.03)

    # model versions are on the y-axis; bottom legend only explains encoding.
    sh = [Line2D([0], [0], marker="o", color="#444", markerfacecolor="white",
                 markersize=11, lw=0, label="baseline"),
          Line2D([0], [0], marker="o", color="#444", markerfacecolor="#444",
                 markersize=11, lw=0, label="post-BLADE (S3)"),
          Line2D([0], [0], color="#b0b4ae", lw=1.8, label="chance / target"),
          Line2D([0], [0], marker="x", color="#9aa0a6", lw=0, markersize=10,
                 label="not exhibited ($<$0.05)"),
          Line2D([0], [0], marker="x", color="#555", lw=0, markersize=10,
                 label="weak, below 0.1 gate")]
    ax.legend(handles=sh, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=5,
              frameon=False, fontsize=10.5, handletextpad=0.4, columnspacing=1.3)
    fig.text(0.5, 0.045,
             "† Gemma self-rate-highly (base 0.58) is below the 0.10 exhibited-gate; "
             "shown gate-bypassed — best-first still removes it to 0.46.",
             ha="center", fontsize=9, style="italic", color="#555")

    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/blade_s3_crossmodel.pdf", bbox_inches="tight")
    fig.savefig("results/blade_s3_crossmodel.png", dpi=300, bbox_inches="tight")
    print("saved figures/blade_s3_crossmodel.pdf and results/blade_s3_crossmodel.png")
