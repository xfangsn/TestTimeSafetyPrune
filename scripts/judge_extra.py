"""Kimi-judge the extra steering points (CAA_L20_c0.7, ActAdd_L14_c16), merge into *_llm_judge.json."""
import json
import re
import subprocess
from pathlib import Path

RESULTS = Path("results"); KIMI = str(Path.home() / ".local/bin/kimi"); CHUNK = 50
KEYS = ["CAA_L20_c0.7", "ActAdd_L14_c16"]
from judge_xstest_llm import PREAMBLE as XP
from judge_harmbench_llm import PREAMBLE as HP


def run(preamble, items, item_fmt):
    labels = {}
    for s in range(0, len(items), CHUNK):
        body = "".join(item_fmt(s + 1 + i, it) for i, it in enumerate(items[s:s + CHUNK]))
        out = subprocess.run([KIMI, "-p", preamble + body, "--print", "--quiet", "-m", "kimi-code/k3"],
                             capture_output=True, text=True, timeout=300).stdout
        for m in re.finditer(r"(\d+)\s*[:\-]\s*(compliance|partial_refusal|full_refusal|REFUSAL|COMPLY)", out, re.I):
            labels[int(m.group(1))] = m.group(2)
    return labels


def main():
    xg = json.loads((RESULTS / "xstest_gens_for_judge.json").read_text())
    hg = json.loads((RESULTS / "harmbench_gens_for_judge.json").read_text())
    xm = json.loads((RESULTS / "xstest_llm_judge.json").read_text())
    hm = json.loads((RESULTS / "harmbench_llm_judge.json").read_text())
    for key in KEYS:
        xl = run(XP, xg[key]["items"], lambda i, it: f"{i}. Q: {it['prompt']}\n   A: {it['output'][:400].strip()}\n")
        xover = sum(v in ("full_refusal", "partial_refusal") for v in xl.values()) / max(1, len(xl))
        xm[key] = {"stringmatch": xg[key]["stringmatch_overrefusal"], "llm_overrefusal": xover, "n_parsed": len(xl)}
        hl = run(HP, hg[key]["items"], lambda i, it: f"{i}. REQUEST: {it['request']}\n   RESPONSE: {it['continuation'][:400].strip()}\n")
        href = sum(v.upper() == "REFUSAL" for v in hl.values()) / max(1, len(hl))
        hm[key] = {"stringmatch": hg[key]["stringmatch_refusal"], "llm_refusal": href, "n_parsed": len(hl)}
        print(f"{key}: XSTest {xm[key]['stringmatch']:.3f}/{xover:.3f} | OOD {hm[key]['stringmatch']:.3f}/{href:.3f}", flush=True)
    (RESULTS / "xstest_llm_judge.json").write_text(json.dumps(xm, indent=2))
    (RESULTS / "harmbench_llm_judge.json").write_text(json.dumps(hm, indent=2))
    print("merged", flush=True)


if __name__ == "__main__":
    main()
