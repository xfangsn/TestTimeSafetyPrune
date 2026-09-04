"""Analyze an amplify-alpha-sweep manip file: per behavior x mode (remove, amplify_a*), report the
blind semantic rate, paired Delta vs base (bootstrap 95% CI), and WikiText Delta-ppl. Answers: does a
larger alpha make the edit change the SEMANTIC behavior (not just keywords), without wrecking ppl?
Usage: python scripts/analyze_amp_sweep.py 14b_ampsweep"""
import json
import random
import sys
from pathlib import Path

TAG = sys.argv[1]
SC = Path(f"/tmp/claude-1000/-home-xfang1999-Projects-TestTimeSafetyPrune/"
          f"e16f646c-64a5-440b-bd68-985c068d25df/scratchpad/manip_{TAG}")
SHORT = {"uncertainty-estimation": "uncertainty", "backtracking": "backtracking",
         "adding-knowledge": "adding_knowledge"}
mf = json.load(open("results/manip_check_qwen3_%s.json" % TAG))
key = json.loads((SC / "key.json").read_text())
labels = {}
for f in sorted(SC.glob("labels_*.json")):
    labels.update(json.loads(f.read_text()))
ids = [i for i in key if i in labels]
ppl = mf.get("ppl", {})


def boot(xs, n=2000):
    if not xs:
        return (float("nan"),) * 3
    m = sum(xs) / len(xs); r = random.Random(0)
    bs = sorted(sum(r.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return m, bs[int(.025 * n)], bs[int(.975 * n)]


print(f"labeled {len(ids)}/{len(key)}\n")
base_by_task = {key[i]["task"]: labels[i] for i in ids if key[i]["mode"] == "base"}
modes = sorted({key[i]["mode"] for i in ids if key[i]["mode"] != "base"},
               key=lambda m: (m != "remove", m))
for beh in mf["behaviors"]:
    sh = SHORT[beh]
    b = sum(base_by_task[t][sh] for t in base_by_task) / len(base_by_task)
    print(f"== {beh} == base semantic rate {b:.2f} ({sum(base_by_task[t][sh] for t in base_by_task)}/{len(base_by_task)})")
    for mode in modes:
        sub = [i for i in ids if key[i]["mode"] == mode and key[i]["behavior"] == beh]
        if not sub:
            continue
        rate = sum(labels[i][sh] for i in sub) / len(sub)
        d = [labels[i][sh] - base_by_task[key[i]["task"]][sh] for i in sub if key[i]["task"] in base_by_task]
        m, lo, hi = boot(d); sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        pp = ppl.get(f"{beh}:{mode}")
        print(f"   {mode:14} rate {rate:.2f} ({sum(labels[i][sh] for i in sub)}/{len(sub)})  "
              f"Δ {m:+.2f} [{lo:+.2f},{hi:+.2f}] {sig}  ppl {pp:+.1%}" if pp is not None else
              f"   {mode:14} rate {rate:.2f}  Δ {m:+.2f} [{lo:+.2f},{hi:+.2f}] {sig}")
