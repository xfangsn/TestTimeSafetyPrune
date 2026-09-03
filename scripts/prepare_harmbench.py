"""Download HarmBench standard text behaviors and dump to data/harmbench_standard.jsonl.

One-time held-out data preparation (plan: docs/plan-harmbench-heldout.md).
Source: official HarmBench GitHub CSV (HF walledai/HarmBench is gated).
Filter: FunctionalCategory == "standard" AND empty ContextString
(excludes copyright / contextual / multimodal behaviors).
"""
import csv
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    "main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
OUT = ROOT / "data" / "harmbench_standard.jsonl"
META = ROOT / "data" / "harmbench_standard.meta.json"


def main() -> None:
    raw = urllib.request.urlopen(CSV_URL, timeout=60).read()
    sha256 = hashlib.sha256(raw).hexdigest()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    kept = [
        r for r in rows
        if r["FunctionalCategory"] == "standard" and not r["ContextString"].strip()
    ]
    with OUT.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps({
                "instruction": r["Behavior"],
                "behavior_id": r["BehaviorID"],
                "semantic_category": r["SemanticCategory"],
            }, ensure_ascii=False) + "\n")
    meta = {
        "source_url": CSV_URL,
        "sha256_csv": sha256,
        "total_rows_in_csv": len(rows),
        "kept_standard_text_behaviors": len(kept),
        "filter": "FunctionalCategory == 'standard' and empty ContextString",
    }
    META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    sys.exit(main())
