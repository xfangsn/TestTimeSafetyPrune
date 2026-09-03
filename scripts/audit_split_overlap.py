"""Data-hygiene audit: exact + near-duplicate overlap between in-dist selection/extraction
splits and the OOD test (HarmBench) + benign (XSTest). Any overlap must be removed from
TRAIN/VAL, never from the fixed test sets."""
import json
import re
from pathlib import Path

DATA = Path("data")


def load(name, key="instruction"):
    return [json.loads(l)[key] for l in (DATA / name).read_text().splitlines() if l.strip()]


def norm(s):
    return re.sub(r"\s+", " ", s.lower().strip())


def toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def report(name, src, tgt_name, tgt, thr=0.6):
    src_n = {norm(x) for x in src}
    tgt_n = [norm(x) for x in tgt]
    exact = sum(t in src_n for t in tgt_n)
    src_tok = [toks(x) for x in src]
    near = 0
    worst = (0.0, "", "")
    for t in tgt:
        tt = toks(t)
        best = max((jaccard(tt, s) for s in src_tok), default=0.0)
        if best >= thr:
            near += 1
        if best > worst[0]:
            worst = (best, t, "")
    print(f"{name:16} vs {tgt_name:10}: exact {exact}/{len(tgt)} | "
          f"near-dup(J>={thr}) {near}/{len(tgt)} | max J {worst[0]:.2f}")


def main():
    harmbench = load("harmbench_standard.jsonl")
    xstest = load("xstest_safe.jsonl")
    for split in ["harmful_train.jsonl", "harmful_val.jsonl"]:
        src = load(split)
        report(split, src, "HarmBench", harmbench)
        report(split, src, "XSTest", xstest)
    caa = [json.loads(l)["instruction"] for l in (DATA / "caa_pairs.jsonl").read_text().splitlines() if l.strip()]
    report("caa_pairs", caa, "HarmBench", harmbench)


if __name__ == "__main__":
    main()
