#!/usr/bin/env bash
# Full rho x alpha grid on Hazel, evaluated on ALL THREE datasets at the new n (SelfAware+FalseQA CAP=70
# + SimpleQA SQ_N=400). MATCH_RHO (ELS probe == edit rho, re-select L* per rho). NOT filtered by degen.
# One job per rho (each = 1 ELS + |ALPHAS| cells) to stay under the 2h short_gpu QOS.
# Usage (login node): bash scripts/run_grid3_hazel.sh
set -euo pipefail
ENV_PREFIX="${ENV_PREFIX:-/usr/local/usrapps/jekml/xfang23/venv}"
SHARE_ROOT="${SHARE_ROOT:-/share/jekml/xfang23}"
CACHE_ROOT="${CACHE_ROOT:-${SHARE_ROOT}/.cache}"
LOG_DIR="${LOG_DIR:-${SHARE_ROOT}/jobs/ttsafety/logs}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${SHARE_ROOT}/ttsafety-data}"
RHOS="${RHOS:-0.002 0.005 0.01 0.02}"
ALPHAS="${ALPHAS:-1.25,1.5,1.75,2.0,2.5,3.0}"
mkdir -p "$LOG_DIR"; JOBSCRIPT_DIR="${LOG_DIR}/jobscripts"; mkdir -p "$JOBSCRIPT_DIR"
for RHO in $RHOS; do
  JOB="grid3_${RHO}_qwen3_8b"; JS="${JOBSCRIPT_DIR}/${JOB}.sh"
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
export TTS_SIMPLEQA_FILE="${DATA_DIR}/simpleqa_test.csv"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts"
cd "${REPO_DIR}"
BLADE_MODEL=Qwen/Qwen3-8B MATCH_RHO=1 BETA=0.05 RHOS=${RHO} ALPHAS="${ALPHAS}" CAP=70 SQ_N=400 \\
  OUT_TAG=_grid3_${RHO} python scripts/blade_rho_sweep.py
echo "DONE grid3 ${RHO}"
EOF
  sbatch "$JS"; echo "submitted $JOB (rho=$RHO, alphas=$ALPHAS, CAP=70, SQ_N=400)"
done
echo "watch: squeue -u \$USER ; ls ${REPO_DIR}/results/blade_rho_sweep_grid3_*_qwen3-8b.json"
