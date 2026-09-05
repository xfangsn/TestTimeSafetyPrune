#!/usr/bin/env bash
# Scheme-A epistemic (un)certainty pipeline for ONE Qwen3 size: untouched-split REMOVE (paired McNemar)
# + powered AMPLIFY, both BLADE-G, thinking-off, closed-book. Portable ($PY + PYTHONPATH; C4/Wiki from
# $TTS_C4_FILE/$TTS_WIKI_FILE; dataset = results/epistemic_pairs_v2.json, tracked in git).
# Usage: MODEL=Qwen/Qwen3-4B TAG=qwen3_4b bash scripts/run_epistemic_qwen3.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="src:scripts:${PYTHONPATH:-}"
PY="${PY:-.venv/bin/python}"
MODEL="${MODEL:?set MODEL, e.g. Qwen/Qwen3-4B}"
TAG="${TAG:?set TAG, e.g. qwen3_4b}"
log(){ echo "[$TAG $(date +%T)] $*"; }
mkdir -p results logs

log "1/2 REMOVE (untouched split + McNemar) ..."
BLADE_MODEL="$MODEL" BLADE_G=1 $PY scripts/blade_epistemic_p0.py

log "2/2 AMPLIFY (powered, untouched split) ..."
BLADE_MODEL="$MODEL" BLADE_G=1 $PY scripts/blade_epistemic_amplify_v2.py

log "done -> results/epistemic_p0_${TAG}_bladeg.json + results/epistemic_amplify_v2_${TAG}_bladeg.json"
