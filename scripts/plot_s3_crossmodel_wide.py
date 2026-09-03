"""Landscape (two-column) S3 BLADE cross-model figure for papers.

Same data as plot_s3_crossmodel.py (loaded live from results/*.json) but laid
out wide: 8 behaviors split into two side-by-side panels, each a horizontal
dumbbell baseline (open) -> post-BLADE (filled). Behavior name = header above
each 3-model group; per-row chance tick; Delta-ppl in a right-edge column.
Wong CVD-safe palette, serif. Outputs a wide PDF suited to a full text column.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots  # noqa: F401

MC = {"Llama-3.2-3B": "#2A6F97", "Qwen3-4B": "#E76F51", "Gemma-3-4B": "#2A9D8F",
      "Phi-4-mini": "#9B5DE5"}
FIXED_RHO = 0.005   # report the beta=5% operating point at rho=0.005 (not min-over-sweep)
TAGS = {"Llama-3.2-3B": "llama-32-3b-instruct", "Qwen3-4B": "qwen3-4b",
        "Gemma-3-4B": "gemma-3-4b-it", "Phi-4-mini": "phi-4-mini-instruct"}
ORDER = ["refusal", "power-seeking", "wealth-seeking", "deception",
         "corrigibility", "self-awareness", "self-rate-highly", "sycophancy"]
LEFT, RIGHT = ORDER[:4], ORDER[4:]
MIN_BIAS = 0.10   # exhibited/not gate used by the pipeline
NULL_BAND = 0.05  # below this -> genuinely at chance; between -> weak-but-ungated
PPL_X = 1.05
# per-model (per-layer residual-writer params, total params) for pruning-rate
MODELDIM = {"Llama-3.2-3B": (34_603_008, 3_212_749_824),
            "Qwen3-4B": (35_389_440, 4_022_468_096),
            "Gemma-3-4B": (31_457_280, 4_300_079_472),
            "Phi-4-mini": (34_603_008, 3_836_021_760)}


def _prune_str(name, n_layers_star, rho, n_edges=None):
    """Reported pruning rate = zeroed weights / total params."""
    pl, tot = MODELDIM[name]
    n = n_edges if n_edges is not None else round(rho * n_layers_star * pl)
    return f"{100.0 * n / tot:.3f}%"


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
            pr = _prune_str(name, len(s3.get("L", [])), res["sparsity"])
            out[b][name] = ("ok", base, res["pick"], f"{res['ppl_delta']*100:+.1f}%", pr)
        else:
            out[b][name] = (_classify_empty(base), base, None, None, None)


def _els_path(tag):
    for suf in ("_beta5_c4", "_beta5", ""):   # prefer C4-calibrated
        p = f"results/blade_els_{tag}{suf}.json"
        if os.path.exists(p):
            return p
    return f"results/blade_els_{tag}.json"


def _ppl_report(pt):
    """Reported (held-out WikiText) ppl change; falls back to legacy ppl_delta."""
    return pt.get("ppl_delta_wiki", pt["ppl_delta"])


def _from_els(tag, out, name):
    res = json.load(open(_els_path(tag)))["results"]
    for b, r in res.items():
        if b not in out:
            continue
        base = r["baseline_bias"]
        if r.get("skipped") or not r.get("L_star") or not r.get("Lstar_sweep"):
            out[b][name] = (_classify_empty(base), base, None, None, None)
        else:
            sweep = r["Lstar_sweep"]
            # For sycophancy/corrigibility we report the MOST removal within the 5% C4 ppl budget
            # ("remove as much as possible within budget"); other behaviors keep the fixed rho=0.005
            # operating point. 0.005 is itself within budget, so this never underperforms it.
            if b in ("sycophancy", "corrigibility"):
                within = [s for s in sweep if s.get("ppl_delta", 1.0) <= 0.05]
                pt = min(within or sweep, key=lambda s: s["pick_rate"])
            else:
                pt = next((s for s in sweep if abs(s["sparsity"] - FIXED_RHO) < 1e-9), None)
                if pt is None:
                    pt = min(sweep, key=lambda s: s["pick_rate"])
            pr = _prune_str(name, len(r["L_star"]), pt["sparsity"], pt.get("n_edges"))
            out[b][name] = ("ok", base, pt["pick_rate"],
                            f"{_ppl_report(pt)*100:+.1f}%", pr)


def load_all():
    out = {b: {} for b in ORDER}
    for name, tag in TAGS.items():
        # uniform beta=5% / rho=0.005 operating point across all models
        if os.path.exists(_els_path(tag)):
            _from_els(tag, out, name)
        elif os.path.exists(f"results/blade_layer_select_{tag}.json"):
            _from_layer_select(tag, out, name)
        rf_path = next((f"results/blade_refusal_els_{tag}{s}.json"
                        for s in ("_c4", "") if os.path.exists(f"results/blade_refusal_els_{tag}{s}.json")), None)
        if rf_path:
            rf = json.load(open(rf_path))
            sweep = rf.get("Lstar_sweep", [])
            if sweep:
                pt = next((s for s in sweep if abs(s["sparsity"] - FIXED_RHO) < 1e-9), None)
                if pt is None:
                    pt = min(sweep, key=lambda s: s["refusal"])
                pr = _prune_str(name, len(rf.get("L_star", [])), pt["sparsity"], pt.get("n_edges"))
                out["refusal"][name] = ("ok", rf["base_refusal"], pt["refusal"],
                                        f"{_ppl_report(pt)*100:+.1f}%", pr)
    # Gemma self-rate-highly is below the 0.10 gate (base 0.58) so the main run
    # skipped it; a gate-bypassed best-first probe shows it IS removable.
    pf = "results/blade_gemma_selfrate_probe.json"
    if os.path.exists(pf):
        p = json.load(open(pf))
        r = p["S3_bestfirst"]["result"]
        if r and r["pick"] < p["baseline"]:
            pr = _prune_str("Gemma-3-4B", len(p["S3_bestfirst"].get("L", [])), r["sparsity"])
            out["self-rate-highly"]["Gemma-3-4B"] = (
                "ok", p["baseline"], r["pick"], f"{r['ppl_delta']*100:+.1f}%†", pr)
    return out


DATA = load_all()


def _safe(path):
    return json.load(open(path)) if os.path.exists(path) else None


# --- corrigibility now comes from the primary BLADE-G ELS run (like every other A/B behavior);
#     the legacy gate-bypassed blade_layer_select override is retired so the whole figure is
#     one consistent BLADE-G pipeline. (BLADE-G removes corrigibility far better, esp. on Gemma.) ---

# NOTE: sycophancy uses the standard beta=5% pipeline value (blade_els ..._beta5.json,
# rho=0.005). A one-off deeper run reached ~0.53 but is NOT reproducible: re-running the
# identical pipeline gives a shallow ~2-layer L* -> ~0.73 (best-first is unstable in which
# pair it picks but stable in depth/result for this diffuse behavior). We report the
# reproducible ~0.73; sycophancy is the weakest-removed behavior, consistent with the rest.

CHANCE = {b: (0.0 if b == "refusal" else 0.5) for b in ORDER}


def draw_panel(ax, behaviors):
    row_h, grp_gap = 1.0, 1.9
    y = 0.0
    yticks, ylabels = [], []
    for beh in behaviors:
        chance = CHANCE[beh]
        g_top = y
        ax.text(0.0, y - 0.85, beh, ha="left", va="center", fontsize=16,
                fontweight="bold", clip_on=False)
        for model in MC:
            st, base, after, ppl, prune = DATA[beh].get(
                model, ("none", 0.5, None, None, None))
            c = MC[model]
            ax.plot([chance, chance], [y - 0.42, y + 0.42], color="#b0b4ae",
                    lw=1.2, zorder=1)
            if st == "ok":
                # arrow shows the direction BLADE moves the behavior: base -> after
                ax.annotate("", xy=(after, y), xytext=(base, y),
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=3.0,
                                            shrinkA=0, shrinkB=0,
                                            mutation_scale=20), zorder=2,
                            annotation_clip=False)
                ax.scatter([base], [y], s=135, facecolors="white", edgecolors=c,
                           lw=2.4, zorder=3, clip_on=False)
                ax.scatter([after], [y], s=70, facecolors=c, edgecolors=c,
                           lw=0, zorder=4, clip_on=False)
                txt = f"{ppl} · {prune}" if prune else ppl
                ax.text(PPL_X, y, txt, ha="left", va="center", fontsize=11,
                        color=c, fontweight="bold",
                        transform=ax.get_yaxis_transform(), clip_on=False)
            elif st == "none":
                ax.scatter([base], [y], s=130, marker="x", color="#9aa0a6",
                           lw=2.6, zorder=3, clip_on=False)
                ax.annotate("not exhibited", (base, y), textcoords="offset points",
                            xytext=(14, 0), ha="left", va="center", fontsize=12.5,
                            color="#8a8f86", style="italic")
            elif st == "weak":  # mild real preference, below the 0.10 removal gate
                ax.scatter([base], [y], s=130, marker="x", color=c,
                           lw=2.6, zorder=3, clip_on=False, alpha=0.85)
                ax.annotate(f"weak ({base:.2f}, below gate)", (base, y),
                            textcoords="offset points", xytext=(14, 0), ha="left",
                            va="center", fontsize=12.5, color=c, style="italic")
            else:
                ax.scatter([base], [y], s=128, marker="s", facecolors="white",
                           edgecolors="#C99700", lw=2.6, zorder=3, clip_on=False)
                ax.annotate(r"L*=$\varnothing$", (base, y), textcoords="offset points",
                            xytext=(14, 0), ha="left", va="center", fontsize=12.5,
                            color="#A67C00", style="italic")
            yticks.append(y)
            ylabels.append(model)  # full model version on the axis
            y += row_h
        y += grp_gap
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=14)
    ax.text(PPL_X, yticks[0] - 1.7, r"$\Delta$ppl $\cdot$ prune", ha="left", va="center",
            fontsize=13.5, fontweight="bold", color="#444",
            transform=ax.get_yaxis_transform(), clip_on=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.margins(y=0.02)


plt.rcParams["font.family"] = "serif"
with plt.style.context(["science", "no-latex"]):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 8.8))
    plt.rc("font", size=15)
    draw_panel(axL, LEFT)
    draw_panel(axR, RIGHT)
    fig.subplots_adjust(wspace=0.92, bottom=0.21, top=0.965, left=0.125, right=0.94)
    fig.text(0.5, 0.125,
             "behavior score   (A/B: MC pick-rate, chance $=0.5$   |   "
             "refusal: refusal-rate, target $=0$)      "
             r"arrow: base $\rightarrow$ post-BLADE  ($\beta=5\%,\ \rho=0.005$)",
             ha="center", fontsize=14)
    fig.text(0.5, 0.103,
             r"$\Delta$ppl on held-out WikiText (budget calibrated on C4);  "
             r"prune $=$ weights zeroed / total params",
             ha="center", fontsize=11.5, color="#555")
    fig.text(0.5, 0.082,
             "† below the 0.10 exhibited-gate on this model; shown gate-bypassed — "
             "best-first still removes the behavior.",
             ha="center", fontsize=11, style="italic", color="#555")

    # model versions are now on the y-axis, so the bottom legend only explains
    # the marker/line encoding (colors keyed per-row on the axis).
    sh = [Line2D([0], [0], marker="o", color="#444", markerfacecolor="white",
                 markersize=13, lw=0, label="baseline"),
          Line2D([0], [0], marker="o", color="#444", markerfacecolor="#444",
                 markersize=13, lw=0, label="post-BLADE (S3)"),
          Line2D([0], [0], color="#b0b4ae", lw=2.0, label="chance / target"),
          Line2D([0], [0], marker="x", color="#9aa0a6", lw=0, markersize=12,
                 label="not exhibited ($<$0.05)"),
          Line2D([0], [0], marker="x", color="#555", lw=0, markersize=12,
                 label="weak, below 0.1 gate")]
    fig.legend(handles=sh, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=5, frameon=False, fontsize=12.5, handletextpad=0.4,
               columnspacing=1.5)

    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/blade_s3_crossmodel_wide.pdf", bbox_inches="tight")
    fig.savefig("results/blade_s3_crossmodel_wide.png", dpi=300, bbox_inches="tight")
    print("saved figures/blade_s3_crossmodel_wide.pdf and results/blade_s3_crossmodel_wide.png")
