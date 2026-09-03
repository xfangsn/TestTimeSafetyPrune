"""Per-item kimi-k3 REFUSAL/COMPLY labels on HarmBench for 4 configs, to find prompts where BLADE
refuses but the steering baselines comply. Saves results/harmbench_peritem_labels.json."""
import json
import re
import subprocess
from pathlib import Path

RESULTS = Path("results"); KIMI = str(Path.home() / ".local/bin/kimi"); CHUNK = 25
KEYS = ["ActAdd_L14_c8", "CAA_L20_c0.5", "RefDir_L16_c0.35", "BLADE_L14_r002_a1.5"]
from judge_harmbench_llm import PREAMBLE as HP

G = json.loads((RESULTS / "harmbench_gens_for_judge.json").read_text())


def judge(items):
    labels = {}
    for s in range(0, len(items), CHUNK):
        body = "".join(f"{s+1+i}. REQUEST: {it['request']}\n   RESPONSE: {it['continuation'][:400].strip()}\n"
                       for i, it in enumerate(items[s:s + CHUNK]))
        out = subprocess.run([KIMI, "-p", HP + body, "--print", "--quiet", "-m", "kimi-code/k3"],
                             capture_output=True, text=True, timeout=300).stdout
        for m in re.finditer(r"(\d+)\s*[:\-]\s*(REFUSAL|COMPLY)", out, re.I):
            labels[int(m.group(1))] = m.group(2).upper()
    return labels


def main():
    out = {}
    for k in KEYS:
        lab = judge(G[k]["items"])
        out[k] = {str(i): lab.get(i, "NA") for i in range(1, len(G[k]["items"]) + 1)}
        print(f"{k}: parsed {sum(v!='NA' for v in out[k].values())}/{len(G[k]['items'])} "
              f"| REFUSAL {sum(v=='REFUSAL' for v in out[k].values())}", flush=True)
    (RESULTS / "harmbench_peritem_labels.json").write_text(json.dumps(out, indent=1))
    print("saved results/harmbench_peritem_labels.json", flush=True)


if __name__ == "__main__":
    main()
