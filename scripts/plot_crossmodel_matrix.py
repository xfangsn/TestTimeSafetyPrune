"""Cross-model BLADE+ELS matrix: 5 behaviors x 3 ~4B models.
Status categories: localized&removed / not-exhibited / diffuse (L*=empty).
Values are baseline -> best within-budget post-BLADE, from the ELS/refusal runs.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MODELS = ["Llama-3.2-3B", "Qwen3-4B", "Gemma-3-4B"]
BEHAV = ["refusal", "power-seeking", "wealth-seeking", "self-awareness", "corrigibility"]

# (status, base, after, ppl)  status: ok / none / diffuse
CELLS = {
 ("refusal","Llama-3.2-3B"):("ok",1.00,0.00,"~0%*"),
 ("refusal","Qwen3-4B"):("ok",0.98,0.00,"+2%*"),
 ("refusal","Gemma-3-4B"):("ok",0.60,0.00,"+3.6%"),
 ("power-seeking","Llama-3.2-3B"):("ok",0.75,0.43,"+4.6%"),
 ("power-seeking","Qwen3-4B"):("ok",0.63,0.46,"+3.6%"),
 ("power-seeking","Gemma-3-4B"):("ok",0.65,0.45,"-3.3%"),
 ("wealth-seeking","Llama-3.2-3B"):("ok",0.62,0.51,"+0.3%"),
 ("wealth-seeking","Qwen3-4B"):("ok",0.67,0.45,"+4.0%"),
 ("wealth-seeking","Gemma-3-4B"):("ok",0.62,0.36,"+5.3%"),
 ("self-awareness","Llama-3.2-3B"):("ok",0.60,0.33,"+1.3%"),
 ("self-awareness","Qwen3-4B"):("none",0.49,None,None),
 ("self-awareness","Gemma-3-4B"):("none",0.52,None,None),
 ("corrigibility","Llama-3.2-3B"):("diffuse",0.60,None,None),
 ("corrigibility","Qwen3-4B"):("ok",0.65,0.48,"+2.4%"),
 ("corrigibility","Gemma-3-4B"):("diffuse",0.66,None,None),
}
COL = {"ok":"#1baf7a","none":"#b4b8b2","diffuse":"#eda100"}
LABEL = {"ok":"localized & removed","none":"not exhibited (≈chance)","diffuse":"diffuse — L* empty"}

nb, nm = len(BEHAV), len(MODELS)
fig, ax = plt.subplots(figsize=(9.2, 6.4))
for i, b in enumerate(BEHAV):
    y = nb - 1 - i
    for j, m in enumerate(MODELS):
        st, base, after, ppl = CELLS[(b, m)]
        ax.add_patch(Rectangle((j, y), 1, 1, facecolor=COL[st], edgecolor="white",
                               linewidth=3, alpha=0.9))
        if st == "ok":
            txt = f"{base:.2f} → {after:.2f}\n{ppl} ppl"
        elif st == "none":
            txt = f"bias {base:.2f}\nnot exhibited"
        else:
            txt = f"bias {base:.2f}\ndiffuse"
        ax.text(j + 0.5, y + 0.5, txt, ha="center", va="center", fontsize=10.5,
                color="#0b0b0b", fontweight="600" if st == "ok" else "400")

ax.set_xlim(0, nm); ax.set_ylim(0, nb)
ax.set_xticks([j + 0.5 for j in range(nm)]); ax.set_xticklabels(MODELS, fontsize=12)
ax.set_yticks([nb - 1 - i + 0.5 for i in range(nb)]); ax.set_yticklabels(BEHAV, fontsize=12)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("BLADE + ELS across three ~4B instruct models\n"
             "baseline → within-budget post-prune (pick-rate; refusal = refusal-rate)",
             fontsize=13, pad=14)
handles = [Rectangle((0,0),1,1,facecolor=COL[k]) for k in ("ok","none","diffuse")]
ax.legend(handles, [LABEL[k] for k in ("ok","none","diffuse")],
          loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False, fontsize=10)
fig.text(0.5, -0.02, "* refusal cheapest via single effective layer; wide-L* sweep costs more (Llama +6.9%, Qwen +17.5%)",
         ha="center", fontsize=8.5, color="#54544f")
fig.tight_layout()
from pathlib import Path
figdir = Path("figures")
figdir.mkdir(exist_ok=True)
fig.savefig(figdir / "blade_crossmodel_matrix.pdf", bbox_inches="tight")  # vector, for paper
fig.savefig("results/blade_crossmodel_matrix.png", dpi=140, bbox_inches="tight")
print("saved figures/blade_crossmodel_matrix.pdf and results/blade_crossmodel_matrix.png")
