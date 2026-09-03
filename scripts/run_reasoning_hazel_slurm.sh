#!/usr/bin/env bash
# Submit the Qwen3 reasoning-behavior sweep (BLADE-B vs BLADE-G) on NCSU Hazel via Slurm.
# One sbatch job per model size; each runs the full portable pipeline (scripts/run_reasoning_qwen3.sh).
# Mirrors the SSW Hazel conventions (venv at ~/venv, caches on /share, typed gres, `module load cuda`).
#
# PREREQUISITE (run ONCE on a login node, has internet):
#   bash scripts/prefetch_reasoning_hazel.sh          # models + C4/WikiText offline files
#
# Usage (from a login node):
#   bash scripts/run_reasoning_hazel_slurm.sh                 # submit the default sweep
#   SIZES="1.7b 4b 8b 14b" bash scripts/run_reasoning_hazel_slurm.sh
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/usr/local/usrapps/jekml/xfang23/venv}"
SHARE_ROOT="${SHARE_ROOT:-/share/jekml/xfang23}"
CACHE_ROOT="${CACHE_ROOT:-${SHARE_ROOT}/.cache}"
LOG_DIR="${LOG_DIR:-${SHARE_ROOT}/jobs/ttsafety/logs}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${SHARE_ROOT}/ttsafety-data}"       # holds c4.txt / wikitext.txt (from prefetch)
# Hazel today (2026-09-03): a30 does not exist. Idle a10 (24GB) + l40s (48GB) live in gpu_partners,
# which for us is QOS short_gpu capped at 2h; the `gpu` partition (h100, infinite) is the big-model path.
PARTITION="${PARTITION:-gpu_partners}"
QOS="${QOS:-short_gpu}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"                      # short_gpu maxwall
SIZES="${SIZES:-1.7b 4b 8b 14b}"                          # 32b: needs h100 in the `gpu` partition
mkdir -p "$LOG_DIR"

# size -> "HF_MODEL gres mem ngpus"  (a10=24GB fits <=8B incl generation; l40s=48GB for 14B)
spec_for() {
  case "$1" in
    1.7b) echo "Qwen/Qwen3-1.7B gpu:a10:1  48G 1" ;;
    4b)   echo "Qwen/Qwen3-4B   gpu:a10:1  64G 1" ;;
    8b)   echo "Qwen/Qwen3-8B   gpu:a10:1  64G 1" ;;
    14b)  echo "Qwen/Qwen3-14B  gpu:l40s:1 96G 1" ;;
    32b)  echo "Qwen/Qwen3-32B  gpu:h100:1 128G 1" ;;   # submit with PARTITION=gpu QOS=gpu
    *) echo "" ;;
  esac
}

for sz in $SIZES; do
  read -r MODEL GRES MEM NGPU <<<"$(spec_for "$sz")"
  [[ -z "${MODEL:-}" ]] && { echo "skip unknown size $sz"; continue; }
  TAG="qwen3_${sz//./}"
  JOB="rsn_${TAG}"
  sbatch --job-name="$JOB" --partition="$PARTITION" --qos="$QOS" --gres="$GRES" \
         --cpus-per-task=8 --mem="$MEM" --time="$TIME_LIMIT" \
         --output="${LOG_DIR}/${JOB}.%j.out" --error="${LOG_DIR}/${JOB}.%j.out" \
         --wrap "set -euo pipefail
module load cuda
source '${ENV_PREFIX}/bin/activate' 2>/dev/null || export PATH='${ENV_PREFIX}/bin':\$PATH
export HF_HOME='${CACHE_ROOT}/huggingface' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TTS_C4_FILE='${DATA_DIR}/c4.txt' TTS_WIKI_FILE='${DATA_DIR}/wikitext.txt'
export PYTHONPATH='${REPO_DIR}/src:${REPO_DIR}/scripts'
cd '${REPO_DIR}'
MODEL='${MODEL}' TAG='${TAG}' PY=python bash scripts/run_reasoning_qwen3.sh"
  echo "submitted $JOB  ($MODEL, $GRES, $MEM)"
done
echo "watch: squeue -u \$USER ; tail -f ${LOG_DIR}/rsn_*.out"
