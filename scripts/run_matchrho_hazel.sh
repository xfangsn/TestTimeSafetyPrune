#!/usr/bin/env bash
# Matched-rho investigation (Qwen3-8B uncertainty): ELS probe fraction == final edit rho.
# For each rho, re-select L* with screen_frac=test_frac=rho, then apply alpha in {0,1.5,2.5,3} at that rho.
# One job per rho (each = 1 ELS + 4 alpha cells) to stay under the 2h short_gpu QOS limit.
# Usage (login node): bash scripts/run_matchrho_hazel.sh          # RHOS default below
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/usr/local/usrapps/jekml/xfang23/venv}"
SHARE_ROOT="${SHARE_ROOT:-/share/jekml/xfang23}"
CACHE_ROOT="${CACHE_ROOT:-${SHARE_ROOT}/.cache}"
LOG_DIR="${LOG_DIR:-${SHARE_ROOT}/jobs/ttsafety/logs}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${SHARE_ROOT}/ttsafety-data}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
RHOS="${RHOS:-0.0005 0.001 0.005 0.01}"
ALPHAS="${ALPHAS:-0,1.5,2.5,3}"
mkdir -p "$LOG_DIR"; JOBSCRIPT_DIR="${LOG_DIR}/jobscripts"; mkdir -p "$JOBSCRIPT_DIR"

for RHO in $RHOS; do
  JOB="matchrho_${RHO}_qwen3_8b"; JS="${JOBSCRIPT_DIR}/${JOB}.sh"
  cat > "$JS" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB}
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/${JOB}.%j.out
#SBATCH --error=${LOG_DIR}/${JOB}.%j.out
set -euo pipefail
module load cuda
export PATH="${ENV_PREFIX}/bin:\$PATH"
export HF_HOME="${CACHE_ROOT}/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TTS_C4_FILE="${DATA_DIR}/c4.txt" TTS_WIKI_FILE="${DATA_DIR}/wikitext.txt"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts"
cd "${REPO_DIR}"
BLADE_MODEL=${MODEL} MATCH_RHO=1 BETA=0.05 RHOS=${RHO} ALPHAS="${ALPHAS}" CAP=40 \\
  OUT_TAG=_matchrho${RHO} python scripts/blade_rho_sweep.py
echo "DONE matchrho ${RHO}"
EOF
  sbatch "$JS"; echo "submitted $JOB (rho=$RHO, alphas=$ALPHAS)"
done
echo "watch: squeue -u \$USER ; tail -f ${LOG_DIR}/matchrho_*_qwen3_8b.*.out"
