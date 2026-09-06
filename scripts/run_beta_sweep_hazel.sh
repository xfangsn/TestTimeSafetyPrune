#!/usr/bin/env bash
# ELS-budget (β) sweep on Hazel: does a larger ppl budget → more/deeper layers in L* → a better
# hallucination/capability frontier than the β=0.05 operating point (L*=[23,31,18,2])?
# Fixed ρ=0.005, α∈{0,2.5} (remove + our amplify point); one job runs BETAS in sequence, each with its
# own OUT_TAG so blade_rho_sweep writes results/blade_rho_sweep_beta<β>_qwen3-8b.json (records its L*).
# Usage (login node): bash scripts/run_beta_sweep_hazel.sh          # BETAS="0.10 0.20"
#                     BETAS="0.10 0.20 0.35" bash scripts/run_beta_sweep_hazel.sh
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/usr/local/usrapps/jekml/xfang23/venv}"
SHARE_ROOT="${SHARE_ROOT:-/share/jekml/xfang23}"
CACHE_ROOT="${CACHE_ROOT:-${SHARE_ROOT}/.cache}"
LOG_DIR="${LOG_DIR:-${SHARE_ROOT}/jobs/ttsafety/logs}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${SHARE_ROOT}/ttsafety-data}"
BETAS="${BETAS:-0.10 0.20}"
RHOS="${RHOS:-0.005}"
ALPHAS="${ALPHAS:-0,2.5}"
CAP="${CAP:-40}"
mkdir -p "$LOG_DIR"

JOB="beta_sweep_qwen3_8b"
JOBSCRIPT_DIR="${LOG_DIR}/jobscripts"; mkdir -p "$JOBSCRIPT_DIR"
JS="${JOBSCRIPT_DIR}/${JOB}.sh"
cat > "$JS" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB}
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/${JOB}.%j.out
#SBATCH --error=${LOG_DIR}/${JOB}.%j.out
set -euo pipefail
module load cuda
export PATH="${ENV_PREFIX}/bin:\$PATH"
export HF_HOME="${CACHE_ROOT}/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TTS_C4_FILE="${DATA_DIR}/c4.txt" TTS_WIKI_FILE="${DATA_DIR}/wikitext.txt"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts"
cd "${REPO_DIR}"
for B in ${BETAS}; do
  echo "===== BETA=\$B ====="
  BLADE_MODEL=Qwen/Qwen3-8B BETA=\$B RHOS=${RHOS} ALPHAS=${ALPHAS} CAP=${CAP} \\
    OUT_TAG=_beta\${B} python scripts/blade_rho_sweep.py
done
echo "DONE beta sweep"
EOF
sbatch "$JS"
echo "submitted $JOB (BETAS=$BETAS, ρ=$RHOS, α=$ALPHAS)  [$JS]"
echo "watch: squeue -u \$USER ; tail -f ${LOG_DIR}/${JOB}.*.out"
