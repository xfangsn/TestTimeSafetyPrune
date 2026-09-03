#!/usr/bin/env bash
# Login-node prefetch for the Hazel Qwen3 reasoning sweep (login nodes have internet; compute nodes
# do NOT). Downloads each Qwen3 model into the shared HF cache and materializes C4 + WikiText to plain
# text files that the air-gapped jobs read via $TTS_C4_FILE / $TTS_WIKI_FILE.
#
# Run ONCE on a Hazel login node:  bash scripts/prefetch_reasoning_hazel.sh
set -euo pipefail
ENV_PREFIX="${ENV_PREFIX:-/usr/local/usrapps/jekml/xfang23/venv}"
SHARE_ROOT="${SHARE_ROOT:-/share/jekml/xfang23}"
CACHE_ROOT="${CACHE_ROOT:-${SHARE_ROOT}/.cache}"
DATA_DIR="${DATA_DIR:-${SHARE_ROOT}/ttsafety-data}"
SIZES="${SIZES:-1.7b 4b 8b 14b}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export HF_HOME="${CACHE_ROOT}/huggingface" HF_HUB_DISABLE_XET=1
mkdir -p "$DATA_DIR" "$HF_HOME"

hf_model() { case "$1" in
  1.7b) echo "Qwen/Qwen3-1.7B" ;; 4b) echo "Qwen/Qwen3-4B" ;; 8b) echo "Qwen/Qwen3-8B" ;;
  14b) echo "Qwen/Qwen3-14B" ;; 32b) echo "Qwen/Qwen3-32B" ;; *) echo "" ;; esac; }

for sz in $SIZES; do
  M="$(hf_model "$sz")"; [[ -z "$M" ]] && continue
  echo "== prefetch $M =="
  python - "$M" <<'PY'
import sys
from huggingface_hub import snapshot_download
m = sys.argv[1]
snapshot_download(m, ignore_patterns=["*.gguf", "*.pth", "original/*"])
print("cached", m)
PY
done

echo "== materialize C4 + WikiText offline text =="
python - "$DATA_DIR" <<'PY'
import sys, pathlib
sys.path.insert(0, "src")
from ttsafety.eval import load_c4_text, load_wikitext_text
d = pathlib.Path(sys.argv[1])
(d / "c4.txt").write_text(load_c4_text(), encoding="utf-8")
(d / "wikitext.txt").write_text(load_wikitext_text(), encoding="utf-8")
print("wrote", d / "c4.txt", d / "wikitext.txt")
PY
echo "prefetch done -> models in $HF_HOME ; text in $DATA_DIR"
