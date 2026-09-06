"""Method comparison (hallucination reduction), Opus-judged: base vs ITI (Li et al. 2306.03341) vs our
BLADE edit, on SelfAware closed-book. DoLa (Chuang 2309.03883) to be added (native broken on tf5.15).
Horizontal bars, house style: (a) SelfAware unanswerable -> answer rate (=hallucination, lower better);
(b) SelfAware answerable -> answer rate (=preservation, higher better)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

R = Path("results"); FIG = Path("figures"); FIG.mkdir(exist_ok=True)
SC = Path("/tmp/claude-1000/-home-xfang1999-Projects-TestTimeSafetyPrune/"
          "e16f646c-64a5-440b-bd68-985c068d25df/scratchpad/opus_judge")
plt.style.use(["science", "no-latex"])

items = json.load(open(R / "method_cmp_qwen3-8b.json"))["items"]
mapping = json.load(open(SC / "map.json"))
labels = {}
for f in ("labels_A.json", "labels_B.json"):
    p = SC / f
    if p.exists():
        labels.update(json.load(open(p)))

# cell[(gold,cond)] -> list of acts
from collections import defaultdict
cell = defaultdict(list)
for jid, m in mapping.items():
    lab = labels.get(jid)
    if not lab:
        continue
    it = items[m["idx"]]
    cell[(it["gold"], m["cond"])].append(lab["act"])


def answer_rate(gold, cond):
    acts = cell[(gold, cond)]
    return 100 * sum(a == "answer" for a in acts) / len(acts) if acts else float("nan")


METHODS = [("base", "base"), ("ITI (α=2)", "iti_a2.0"), ("ITI (α=6)", "iti_a6.0"),
           ("BLADE amplify", "amplify"), ("BLADE remove", "remove")]
labs = [m[0] for m in METHODS]
hall = [answer_rate("unanswerable", m[1]) for m in METHODS]     # hallucination (lower better)
pres = [answer_rate("answerable", m[1]) for m in METHODS]       # preservation (higher better)
COL = ["#999999", "#6E9FC4", "#4E7FA6", "#0072B2", "#D55E00"]
y = np.arange(len(METHODS))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.0, 3.6))
b1 = axA.barh(y, hall, 0.62, color=COL); axA.bar_label(b1, fmt="%.0f", fontsize=9, padding=2)
axA.set_yticks(y); axA.set_yticklabels(labs); axA.invert_yaxis()
axA.set_xlabel("hallucination rate (%)"); axA.set_xlim(0, max([v for v in hall if v == v]) * 1.25)
axA.set_title("(a) SelfAware unanswerable  (lower = better)", fontsize=9)
b2 = axB.barh(y, pres, 0.62, color=COL); axB.bar_label(b2, fmt="%.0f", fontsize=9, padding=2)
axB.set_yticks(y); axB.set_yticklabels([]); axB.invert_yaxis()
axB.set_xlabel("answer rate (%)"); axB.set_xlim(0, 100)
axB.set_title("(b) SelfAware answerable  (higher = better)", fontsize=9)
fig.suptitle("Hallucination reduction: ours vs ITI  (Qwen3-8B, SelfAware, Opus-judged)", fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"method_comparison.{ext}", dpi=300, bbox_inches="tight")
print("saved figures/method_comparison.png / .pdf")
for m, h, p in zip(labs, hall, pres):
    print(f"  {m:16s} hallucinate {h:5.1f}%   answerable-answer {p:5.1f}%")
