#!/usr/bin/env bash
# Submit the scheme-A epistemic (un)certainty size sweep on NCSU Hazel via Slurm — one sbatch per Qwen3
# size, each runs scripts/run_epistemic_qwen3.sh (REMOVE untouched-split+McNemar, powered AMPLIFY; BLADE-G).
# Mirrors scripts/run_reasoning_hazel_slurm.sh (venv on usrapps, caches on /share, typed gres, module cuda).
#
# PREREQUISITE (login node, once): Qwen3 models cached (the reasoning prefetch already did 1.7b/4b/8b/14b)
#   + C4/WikiText offline files at $DATA_DIR (scripts/prefetch_reasoning_hazel.sh).
# Usage (login node): bash scripts/run_epistemic_hazel_slurm.sh          # 1.7b 4b 8b 14b
#                     SIZES="8b 14b" bash scripts/run_epistemic_hazel_slurm.sh
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/usr/local/usrapps/jekml/xfang23/venv}"
SHARE_ROOT="${SHARE_ROOT:-/share/jekml/xfang23}"
CACHE_ROOT="${CACHE_ROOT:-${SHARE_ROOT}/.cache}"
LOG_DIR="${LOG_DIR:-${SHARE_ROOT}/jobs/ttsafety/logs}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${SHARE_ROOT}/ttsafety-data}"
SIZES="${SIZES:-1.7b 4b 8b 14b}"
mkdir -p "$LOG_DIR"

# size -> "HF_MODEL gres mem partition qos time"
spec_for() {
  case "$1" in
    1.7b) echo "Qwen/Qwen3-1.7B gpu:a10:1  48G  gpu_partners short_gpu 02:00:00" ;;
    4b)   echo "Qwen/Qwen3-4B   gpu:a10:1  64G  gpu_partners short_gpu 02:00:00" ;;
    8b)   echo "Qwen/Qwen3-8B   gpu:l40s:1 64G  gpu_partners short_gpu 02:00:00" ;;  # a10 24GB OOMs on 8B+ppl
    14b)  echo "Qwen/Qwen3-14B  gpu:h100:1 96G  gpu          gpu      08:00:00" ;;  # h100/infinite: 2 pipelines > 2h
    32b)  echo "Qwen/Qwen3-32B  gpu:h100:1 128G gpu          gpu      12:00:00" ;;
    *) echo "" ;;
  esac
}

JOBSCRIPT_DIR="${JOBSCRIPT_DIR:-${LOG_DIR}/jobscripts}"; mkdir -p "$JOBSCRIPT_DIR"
for sz in $SIZES; do
  read -r MODEL GRES MEM PART QOS TLIM <<<"$(spec_for "$sz")"
  [[ -z "${MODEL:-}" ]] && { echo "skip unknown size $sz"; continue; }
  TAG="qwen3_${sz//./}"
  JOB="epi_${TAG}"
  JS="${JOBSCRIPT_DIR}/${JOB}.sh"
  cat > "$JS" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB}
#SBATCH --partition=${PART}
#SBATCH --qos=${QOS}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=8
#SBATCH --mem=${MEM}
#SBATCH --time=${TLIM}
#SBATCH --output=${LOG_DIR}/${JOB}.%j.out
#SBATCH --error=${LOG_DIR}/${JOB}.%j.out
set -euo pipefail
module load cuda
export PATH="${ENV_PREFIX}/bin:\$PATH"
export HF_HOME="${CACHE_ROOT}/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TTS_C4_FILE="${DATA_DIR}/c4.txt" TTS_WIKI_FILE="${DATA_DIR}/wikitext.txt"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts"
cd "${REPO_DIR}"
MODEL="${MODEL}" TAG="${TAG}" PY=python bash scripts/run_epistemic_qwen3.sh
EOF
  sbatch "$JS"
  echo "submitted $JOB  ($MODEL, $GRES, $PART/$QOS $TLIM)  [$JS]"
done
echo "watch: squeue -u \$USER ; tail -f ${LOG_DIR}/epi_*.out"
