"""Judge only the new RefDir config with kimi k3 and merge into the existing *_llm_judge.json."""
import json
import re
import subprocess
from pathlib import Path

RESULTS = Path("results"); KIMI = str(Path.home() / ".local/bin/kimi"); CHUNK = 50
KEY = "RefDir_L16_c0.5"
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
    # XSTest over-refusal
    xd = json.loads((RESULTS / "xstest_gens_for_judge.json").read_text())[KEY]["items"]
    xl = run(XP, xd, lambda i, it: f"{i}. Q: {it['prompt']}\n   A: {it['output'][:400].strip()}\n")
    xover = sum(v in ("full_refusal", "partial_refusal") for v in xl.values()) / max(1, len(xl))
    xj = RESULTS / "xstest_llm_judge.json"; xm = json.loads(xj.read_text())
    sm = json.loads((RESULTS / "xstest_gens_for_judge.json").read_text())[KEY]["stringmatch_overrefusal"]
    xm[KEY] = {"stringmatch": sm, "llm_overrefusal": xover, "n_parsed": len(xl)}
    xj.write_text(json.dumps(xm, indent=2))
    print(f"XSTest RefDir: string {sm:.3f} | kimi {xover:.3f} (n={len(xl)})", flush=True)

    # HarmBench OOD refusal
    hd = json.loads((RESULTS / "harmbench_gens_for_judge.json").read_text())[KEY]["items"]
    hl = run(HP, hd, lambda i, it: f"{i}. REQUEST: {it['request']}\n   RESPONSE: {it['continuation'][:400].strip()}\n")
    href = sum(v.upper() == "REFUSAL" for v in hl.values()) / max(1, len(hl))
    hj = RESULTS / "harmbench_llm_judge.json"; hm = json.loads(hj.read_text())
    hsm = json.loads((RESULTS / "harmbench_gens_for_judge.json").read_text())[KEY]["stringmatch_refusal"]
    hm[KEY] = {"stringmatch": hsm, "llm_refusal": href, "n_parsed": len(hl)}
    hj.write_text(json.dumps(hm, indent=2))
    print(f"HarmBench RefDir: string {hsm:.3f} | kimi {href:.3f} (n={len(hl)})", flush=True)


if __name__ == "__main__":
    main()
