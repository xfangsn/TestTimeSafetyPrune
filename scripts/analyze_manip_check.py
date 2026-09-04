"""P-1 analysis: from blind Opus semantic labels + the held-back key, compute
(1) lexicon precision/recall vs semantic labels on BASE traces (are the keywords a good proxy?),
(2) semantic-change from the edit (Opus-label rate base vs remove vs amplify, length-reported),
(3) double dissociation (edit-X should move dimension-X more than dimension-Y, and more than edit-Y
    moves dimension-X) with z-scored effects + bootstrap 95% CIs.
Reads scratchpad/manip/{key.json, labels_*.json} + results/manip_check_qwen3_17b.json."""
import json
import random
from pathlib import Path

import sys
TAG = sys.argv[1] if len(sys.argv)>1 else "17b"
SC = Path("/tmp/claude-1000/-home-xfang1999-Projects-TestTimeSafetyPrune/"
          "e16f646c-64a5-440b-bd68-985c068d25df/scratchpad/manip_%s" % TAG)
KW = {
    "uncertainty-estimation": [" maybe", " perhaps", "not sure", "i think", " possibly", " might ",
                               "could be", "i'm not", " unsure", " i guess", "not certain"],
    "backtracking": [" wait", " actually", "reconsider", " hmm", "scratch that",
                     "on second thought", " no,", "let me re", " but wait"],
}
PATTERNS = ["uncertainty-estimation", "backtracking"]
SHORT = {"uncertainty-estimation": "uncertainty", "backtracking": "backtracking"}


def kw_present(text, pat):
    t = text.lower()
    return int(any(k in t for k in KW[pat]))


def main():
    key = json.loads((SC / "key.json").read_text())
    labels = {}
    for f in sorted(SC.glob("labels_*.json")):
        labels.update(json.loads(f.read_text()))
    traces = {it["_id"]: it for it in
              [dict(x, _id=f"t{i:03d}") for i, x in
               enumerate(json.load(open("results/manip_check_qwen3_%s.json"%TAG))["items"])]}
    ids = [i for i in key if i in labels]
    print(f"labeled {len(ids)}/{len(key)} traces\n")

    # (1) lexicon precision/recall on BASE traces (semantic label = ground truth)
    print("== (1) lexicon vs semantic (BASE traces) ==")
    for pat in PATTERNS:
        base = [i for i in ids if key[i]["mode"] == "base"]
        tp = fp = fn = tn = 0
        for i in base:
            sem = labels[i][SHORT[pat]]; kw = kw_present(traces[i]["thinking"], pat)
            tp += kw and sem; fp += kw and not sem; fn += (not kw) and sem; tn += (not kw) and (not sem)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"  {pat:22} keyword→semantic  precision {prec:.2f}  recall {rec:.2f}  "
              f"(kw-free positives={fn}, kw negatives={fp}, n={len(base)})")

    # (2) semantic-label rate by condition + mean length
    print("\n== (2) semantic-label rate (Opus) by condition ==")
    def rate(mode, editbeh, dim):
        sub = [i for i in ids if key[i]["mode"] == mode and (mode == "base" or key[i]["behavior"] == editbeh)]
        if not sub:
            return float("nan"), 0, 0
        r = sum(labels[i][SHORT[dim]] for i in sub) / len(sub)
        wl = sum(key[i]["think_words"] for i in sub) / len(sub)
        return r, len(sub), wl
    for pat in PATTERNS:
        b, nb, lb = rate("base", None, pat)
        rm, nr, lr = rate("remove", pat, pat)
        am, na, la = rate("amplify", pat, pat)
        print(f"  {pat:22} base {b:.2f}(w{lb:.0f}) → remove {rm:.2f}(w{lr:.0f})  amplify {am:.2f}(w{la:.0f})")

    # (3) double dissociation: Δ(dim rate) from base for each edit, paired by task, z-scored + bootstrap
    print("\n== (3) double dissociation (Δ semantic rate vs base, remove edits) ==")
    base_by_task = {key[i]["task"]: labels[i] for i in ids if key[i]["mode"] == "base"}
    def deltas(editbeh, dim):
        out = []
        for i in ids:
            if key[i]["mode"] != "remove" or key[i]["behavior"] != editbeh:
                continue
            bt = base_by_task.get(key[i]["task"])
            if bt is None:
                continue
            out.append(labels[i][SHORT[dim]] - bt[SHORT[dim]])
        return out
    def boot(xs, n=2000):
        if not xs:
            return (float("nan"),) * 3
        m = sum(xs) / len(xs); r = random.Random(0)
        bs = sorted(sum(r.choice(xs) for _ in xs) / len(xs) for _ in range(n))
        return m, bs[int(.025 * n)], bs[int(.975 * n)]
    print("  edit \\ dim        Δuncertainty            Δbacktracking")
    for editbeh in PATTERNS:
        cells = []
        for dim in PATTERNS:
            m, lo, hi = boot(deltas(editbeh, dim))
            cells.append(f"{m:+.2f} [{lo:+.2f},{hi:+.2f}]")
        print(f"  {SHORT[editbeh]:14}  {cells[0]:22}  {cells[1]:22}")
    print("\nPASS if each remove-edit's own-dim Δ is negative & CI-separated from the other edit's "
          "effect on that dim AND from its own off-dim Δ.")


if __name__ == "__main__":
    main()
