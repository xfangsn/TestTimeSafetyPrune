"""PNG preview / paper figure mirroring slide 1 of the editable deck:
Driving vs Modulatory weights; BLADE targets the modulatory ones."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow, Circle, Rectangle

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
BLUE="#1D4E89"; BLUE_F="#D5E5F7"; AMB="#C87A00"; AMB_F="#FBE7C6"
GRN="#2A7F4F"; GRN_F="#D9EFE1"; INK="#222222"; GREY="#8A8A8A"

fig, ax = plt.subplots(figsize=(13.33, 7.5)); ax.set_xlim(0,13.33); ax.set_ylim(0,7.5)
ax.axis("off"); ax.invert_yaxis()

def rbox(x,y,w,h,fc,ec,lw=1.5,rad=0.08):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.02,rounding_size={rad}",
                 fc=fc,ec=ec,lw=lw));
def t(x,y,s,sz,c=INK,b=False,ha="left",va="top"):
    ax.text(x,y,s,fontsize=sz,color=c,fontweight="bold" if b else "normal",ha=ha,va=va)

t(0.5,0.55,"Two kinds of weights in an LLM — BLADE edits the modulatory ones",20,INK,True)
t(0.5,1.15,"Control a behavior by zeroing a few gain/gate weights, without touching what the model can do",13,GREY)

# DRIVING lane
rbox(0.5,1.8,7.4,2.05,BLUE_F,BLUE)
t(0.75,2.15,"DRIVING weights  (W_drive)",13,BLUE,True)
for yy in [2.55,3.0,3.45]:
    ax.add_patch(Circle((1.05,yy),0.16,fc="white",ec=BLUE,lw=1.2))
ax.add_patch(FancyArrow(1.45,2.95,1.55,0,width=0.28,head_width=0.5,head_length=0.35,fc=BLUE,ec="none",length_includes_head=True))
rbox(3.3,2.45,1.75,1.0,"white",BLUE,1.3,0.06); t(4.17,2.98,"signal /\ncontent",11,BLUE,True,"center","center")
ax.add_patch(FancyArrow(5.35,2.95,1.05,0,width=0.24,head_width=0.44,head_length=0.3,fc=BLUE,ec="none",length_includes_head=True))
ax.add_patch(Circle((7.05,2.95),0.5,fc="white",ec=BLUE,lw=1.3)); t(7.05,2.95,"out",11,BLUE,True,"center","center")
t(0.75,3.62,"carry the signal  →  WHAT the model can do (capability)",12,INK)

# MODULATORY gate on the path
ax.add_patch(Rectangle((5.5,2.05),0.7,1.85,fc=AMB_F,ec=AMB,lw=1.6))
ax.add_patch(Circle((5.85,2.95),0.28,fc=AMB,ec=AMB)); t(5.85,2.95,"⦿",13,"white",True,"center","center")
rbox(8.15,1.8,4.65,2.05,AMB_F,AMB)
t(8.4,2.15,"MODULATORY weights  (W_mod)",13,AMB,True)
t(8.4,2.62,"a gain / gate on the path",12,AMB,True)
t(8.4,3.05,"→ WHETHER a behavior is expressed",11.5,INK)
t(8.4,3.42,"sparse — a few residual-writer edges",11,GREY)

# BLADE callout
rbox(3.75,4.35,5.65,1.05,GRN_F,GRN)
t(6.57,4.68,"BLADE  (gradient-free, forward-only)",12.5,GRN,True,"center","center")
t(6.57,5.05,"scores  s = [ r · W · Δμ ]₊  and zeros ~0.2–1% modulatory writer weights",11,INK,"center","center")
ax.add_patch(FancyArrow(5.85,4.32,0,-0.5,width=0.06,head_width=0.28,head_length=0.22,fc=GRN,ec="none",length_includes_head=True))

# contrast boxes
yb=5.75
rbox(0.5,yb,6.05,1.55,"white",BLUE,1.3)
t(0.72,yb+0.32,"Zero DRIVING weights",12.5,BLUE,True)
t(0.72,yb+0.72,"• dense, distributed   • carry content",11,INK)
t(0.72,yb+1.05,"• → capability collapses (ppl ↑↑)",11,INK)
t(0.72,yb+1.35,"• neuro analogy: driving synapse (sets firing)",10.5,GREY)
rbox(6.78,yb,6.05,1.55,"white",AMB,1.3)
t(7.0,yb+0.32,"Zero MODULATORY weights   ← BLADE",12.5,AMB,True)
t(7.0,yb+0.72,"• sparse, few edges   • gain / gating of a behavior",11,INK)
t(7.0,yb+1.05,"• → behavior OFF, capability intact (ppl ≈ 0)",11,INK)
t(7.0,yb+1.35,"• neuro analogy: modulatory synapse (tunes gain)",10.5,GREY)

fig.savefig(FIG/"blade_modulatory_preview.png",dpi=130,bbox_inches="tight")
print("saved figures/blade_modulatory_preview.png")
