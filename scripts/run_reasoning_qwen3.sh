#!/bin/bash
# Full Qwen3 reasoning-behavior pipeline for ONE model size, running BOTH BLADE-B and BLADE-G.
# Portable: uses $PY + PYTHONPATH (no hardcoded paths); messages come from scripts/steer_messages
# unless $STEER_REPO is set; C4/WikiText come from $TTS_C4_FILE/$TTS_WIKI_FILE on air-gapped nodes.
#
# Usage:  MODEL=Qwen/Qwen3-4B TAG=qwen3_4b bash scripts/run_reasoning_qwen3.sh
# Knobs (env): PY (python), N_TRACES (200), REMOVE_RHO (0.008), AMP_RHO (0.001), AMP_ALPHAS (1.25),
#              STEER_REPO, TTS_C4_FILE, TTS_WIKI_FILE, SKIP_TRACES/SKIP_DIRS/SKIP_ELS (reuse cached).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="src:scripts:${PYTHONPATH:-}"
PY="${PY:-.venv/bin/python}"
MODEL="${MODEL:?set MODEL, e.g. Qwen/Qwen3-4B}"
TAG="${TAG:?set TAG, e.g. qwen3_4b}"
N_TRACES="${N_TRACES:-200}"; REMOVE_RHO="${REMOVE_RHO:-0.008}"
AMP_RHO="${AMP_RHO:-0.001}"; AMP_ALPHAS="${AMP_ALPHAS:-1.25}"
log(){ echo "[$TAG $(date +%T)] $*"; }
mkdir -p results logs   # results/ is gitignored; jobs must create it

if [[ -z "${SKIP_TRACES:-}" ]]; then
  log "1/5 generate traces ..."
  $PY scripts/gen_qwen3_traces.py --model "$MODEL" --out "${TAG}_traces.json" --n "$N_TRACES"
fi
if [[ -z "${SKIP_DIRS:-}" ]]; then
  log "2/5 keyword-span directions ..."
  $PY scripts/qwen3_directions.py --model "$MODEL" --traces "${TAG}_traces.json" --out "${TAG}_dirs.pt"
fi
if [[ -z "${SKIP_ELS:-}" ]]; then
  log "3/5 ELS auto-select layers ..."
  $PY scripts/reasoning_els.py --dirs "${TAG}_dirs.pt" --model "$MODEL" --key "$TAG"
fi
log "4/5 intervention BLADE-B ..."
$PY scripts/blade_reasoning_full.py --dirs "${TAG}_dirs.pt" --els "reasoning_els_${TAG}.json" \
  --model "$MODEL" --out "blade_reasoning_${TAG}.json" \
  --remove-rho "$REMOVE_RHO" --amp-rho "$AMP_RHO" --amp-alphas "$AMP_ALPHAS"
log "5/5 intervention BLADE-G ..."
$PY scripts/blade_reasoning_full.py --dirs "${TAG}_dirs.pt" --els "reasoning_els_${TAG}.json" \
  --model "$MODEL" --out "blade_reasoning_${TAG}_bladeg.json" --blade-g \
  --remove-rho "$REMOVE_RHO" --amp-rho "$AMP_RHO" --amp-alphas "$AMP_ALPHAS"
log "ALL DONE -> results/blade_reasoning_${TAG}{,_bladeg}.json"
