#!/bin/bash
# Qwen3-8B reasoning-behavior BLADE run — READY TO FIRE (deferred until usage recovers).
# Disentangles size vs architecture: Qwen3-4B was POSITIVE at ELS layers; if Qwen3-8B (same arch,
# +size) is also positive, R1-Distill-Llama-8B's weakness is architecture-specific, not pure size.
# Uses the LESSON learned: go straight to ELS auto-selected layers + full intervention (the fixed-band
# probe misleads). Keyword-span directions (annotation-free); Opus only for optional final validation.
# Fire with:  setsid nohup bash scripts/run_qwen3_8b.sh > logs/qwen3_8b_chain.log 2>&1 &
set -e
cd /home/xfang1999/Projects/TestTimeSafetyPrune
export PYTHONPATH=src:scripts
PY=.venv/bin/python
M="Qwen/Qwen3-8B"
log(){ echo "[q3-8b $(date +%T)] $*"; }

log "1/4 generate traces (~16GB download first) ..."
$PY scripts/gen_qwen3_traces.py --model "$M" --out qwen3_8b_traces.json --n 200 \
  > logs/qwen3_8b_traces.log 2>&1
log "traces: $(grep -o 'closed </think>[^|]*| mean thinking words [0-9]*' logs/qwen3_8b_traces.log)"

log "2/4 keyword-span directions ..."
$PY scripts/qwen3_directions.py --model "$M" --traces qwen3_8b_traces.json --out qwen3_8b_dirs.pt \
  > logs/qwen3_8b_dirs.log 2>&1
log "dirs: $(grep -o 'span counts {[^}]*}' logs/qwen3_8b_dirs.log | tail -1)"

log "3/4 ELS auto-select layers ..."
$PY scripts/reasoning_els.py --dirs qwen3_8b_dirs.pt --model "$M" --key qwen3_8b \
  > logs/reasoning_els_qwen3_8b.log 2>&1
log "ELS: $(grep -o 'L\* = \[[^]]*\]' logs/reasoning_els_qwen3_8b.log | tr '\n' ' ')"

log "4/4 full intervention (4x4 selectivity + controls + steer + amplify) at ELS layers ..."
$PY scripts/blade_reasoning_full.py --dirs qwen3_8b_dirs.pt --els reasoning_els_qwen3_8b.json \
  --model "$M" --out blade_reasoning_qwen3_8b.json --remove-rho 0.008 --amp-rho 0.001 --amp-alphas 1.25 \
  > logs/blade_reasoning_qwen3_8b.log 2>&1
log "ALL DONE -> results/blade_reasoning_qwen3_8b.json"
