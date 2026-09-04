#!/usr/bin/env bash
# Login-node prefetch of GSM8K + MATH-500 to offline jsonl ({question, answer}) for the calibration
# eval (compute nodes are air-gapped; jobs read $TTS_GSM8K_FILE / $TTS_MATH_FILE). Run on a login node.
set -euo pipefail
ENV_PREFIX="${ENV_PREFIX:-/usr/local/usrapps/jekml/xfang23/venv}"
SHARE_ROOT="${SHARE_ROOT:-/share/jekml/xfang23}"
DATA_DIR="${DATA_DIR:-${SHARE_ROOT}/ttsafety-data}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export HF_HOME="${SHARE_ROOT}/.cache/huggingface" HF_HUB_DISABLE_XET=1
mkdir -p "$DATA_DIR"
python - "$DATA_DIR" <<'PY'
import sys, json, pathlib
from datasets import load_dataset
d = pathlib.Path(sys.argv[1])
g = load_dataset("openai/gsm8k", "main", split="test")
with open(d / "gsm8k.jsonl", "w") as f:
    for r in g:
        f.write(json.dumps({"question": r["question"],
                            "answer": r["answer"].split("####")[-1].strip()}) + "\n")
print("gsm8k", len(g))
m = load_dataset("HuggingFaceH4/MATH-500", split="test")
with open(d / "math500.jsonl", "w") as f:
    for r in m:
        f.write(json.dumps({"question": r["problem"], "answer": r["answer"]}) + "\n")
print("math500", len(m))
print("wrote", d / "gsm8k.jsonl", d / "math500.jsonl")
PY
