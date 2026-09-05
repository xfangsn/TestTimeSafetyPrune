"""Blind + shuffle the P-1 manipulation-check traces for annotation, split into chunks.
Writes chunk files (anon id + thinking only) for blind annotators, and a held-back key
(id -> true behavior/mode). Annotators label SEMANTIC presence of each pattern per trace."""
import json
import random
from pathlib import Path

import os, sys
TAG = sys.argv[1] if len(sys.argv)>1 else "17b"
CHUNK_SIZE = int(os.environ.get("MANIP_CHUNK_SIZE", "9"))   # small chunks: kimi chokes on >~15K-char prompts
SC = Path("/tmp/claude-1000/-home-xfang1999-Projects-TestTimeSafetyPrune/"
          "e16f646c-64a5-440b-bd68-985c068d25df/scratchpad/manip_%s" % TAG)
SC.mkdir(parents=True, exist_ok=True)
for old in list(SC.glob("chunk_*.json")) + list(SC.glob("labels_*.json")):
    old.unlink()   # clear stale chunks/labels before re-chunking

d = json.load(open("results/manip_check_qwen3_%s.json"%TAG))
items = d["items"]
import math as _m
N_CHUNKS = max(1, _m.ceil(len(items) / CHUNK_SIZE))
for i, it in enumerate(items):
    it["_id"] = f"t{i:03d}"
key = {it["_id"]: {"behavior": it["behavior"], "mode": it["mode"], "task": it["task"],
                   "think_words": it["think_words"]} for it in items}
(SC / "key.json").write_text(json.dumps(key, indent=1))

order = list(range(len(items)))
random.Random(0).shuffle(order)
chunks = [[] for _ in range(N_CHUNKS)]
for j, idx in enumerate(order):
    it = items[idx]
    chunks[j % N_CHUNKS].append({"id": it["_id"], "thinking": it["thinking"]})
for i, ch in enumerate(chunks):
    (SC / f"chunk_{i}.json").write_text(json.dumps(ch, ensure_ascii=False, indent=1))
print(f"wrote {N_CHUNKS} chunks ({len(items)} traces) + key to {SC}")
print("chunk sizes:", [len(c) for c in chunks])
