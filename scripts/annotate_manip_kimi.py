"""Blind semantic annotation of manipulation-check chunks via the kimi CLI (k3) — same reliable path
as the judge_*_llm scripts. For each chunk_i.json writes labels_i.json {id:{uncertainty,backtracking}}.
Usage: python scripts/annotate_manip_kimi.py 14b"""
import json
import re
import subprocess
import sys
from pathlib import Path

TAG = sys.argv[1] if len(sys.argv) > 1 else "17b"
SC = Path(f"/tmp/claude-1000/-home-xfang1999-Projects-TestTimeSafetyPrune/"
          f"e16f646c-64a5-440b-bd68-985c068d25df/scratchpad/manip_{TAG}")
KIMI = str(Path.home() / ".local/bin/kimi")

RUBRIC = (
    "You are a BLIND annotator of LLM reasoning traces. For EACH trace judge SEMANTICALLY (by meaning, "
    "NOT keyword presence) two INDEPENDENT binary properties:\n"
    "- uncertainty=1 if it genuinely expresses doubt/low confidence about its OWN reasoning or answer "
    "(might be wrong, hedges, not sure, weighs options without committing); else 0. Not for rhetorical "
    "maybe/I-think; yes for genuine doubt even without such words.\n"
    "- backtracking=1 if it genuinely REVISES/CORRECTS an earlier step (catches a mistake, changes a "
    "prior claim/approach, abandons a started line); else 0. Not mere continuation. Not for rhetorical "
    "wait/actually; yes for genuine self-correction even without them.\n"
    "Return ONLY a JSON object mapping each id to {\"uncertainty\":0 or 1,\"backtracking\":0 or 1}. "
    "No prose, no markdown fences.\n\nTRACES:\n")


def extract_json(s):
    m = re.search(r"\{.*\}", s, re.DOTALL)
    return json.loads(m.group(0)) if m else {}


def main():
    for cf in sorted(SC.glob("chunk_*.json")):
        i = cf.stem.split("_")[1]
        out = SC / f"labels_{i}.json"
        if out.exists():
            print(f"chunk {i}: labels exist, skip"); continue
        chunk = json.loads(cf.read_text())
        payload = RUBRIC + json.dumps(chunk, ensure_ascii=False)
        r = subprocess.run([KIMI, "-p", payload, "--print", "--quiet", "-m", "kimi-code/k3"],
                           capture_output=True, text=True, timeout=600)
        labels = extract_json(r.stdout)
        ids = {c["id"] for c in chunk}
        got = set(labels)
        print(f"chunk {i}: {len(got & ids)}/{len(ids)} ids labeled"
              + ("" if got >= ids else f"  MISSING {len(ids - got)}"))
        if got & ids:
            out.write_text(json.dumps({k: labels[k] for k in labels if k in ids}, indent=1))
    print("done")


if __name__ == "__main__":
    main()
