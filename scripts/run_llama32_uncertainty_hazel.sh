#!/usr/bin/env bash
# Uncertainty ITI + BLADE on Llama-3.2-3B-Instruct (mirrors the Qwen3-8B uncertainty_method_cmp pipeline).
# PHASE=1 (default): the figure config group only — ITI baseline (c=2/4/6) + ITI ppl + BLADE (rho=.005,
#   alpha in {0,2.5}, beta=.05). Validates the pipeline end-to-end on Llama before the hyperparameter search.
# PHASE=2: hyperparameter search — BLADE rho x alpha grid at beta=.05, plus beta sweep (0.10/0.20).
# 3B fits on a10 (24GB). Usage (login node): PHASE=1 bash scripts/run_llama32_uncertainty_hazel.sh
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/usr/local/usrapps/jekml/xfang23/venv}"
SHARE_ROOT="${SHARE_ROOT:-/share/jekml/xfang23}"
CACHE_ROOT="${CACHE_ROOT:-${SHARE_ROOT}/.cache}"
LOG_DIR="${LOG_DIR:-${SHARE_ROOT}/jobs/ttsafety/logs}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${SHARE_ROOT}/ttsafety-data}"
MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
PHASE="${PHASE:-1}"
mkdir -p "$LOG_DIR"
JOBSCRIPT_DIR="${LOG_DIR}/jobscripts"; mkdir -p "$JOBSCRIPT_DIR"

emit_header() {  # $1 = job name
  cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$1
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/$1.%j.out
#SBATCH --error=${LOG_DIR}/$1.%j.out
set -euo pipefail
module load cuda
export PATH="${ENV_PREFIX}/bin:\$PATH"
export HF_HOME="${CACHE_ROOT}/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TTS_C4_FILE="${DATA_DIR}/c4.txt" TTS_WIKI_FILE="${DATA_DIR}/wikitext.txt"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts"
cd "${REPO_DIR}"
export BLADE_MODEL="${MODEL}"
EOF
}

submit() { local js="$1"; sbatch "$js"; echo "submitted $(basename "$js")"; }

if [[ "$PHASE" == "1" ]]; then
  JOB="llama32_unc_p1"; JS="${JOBSCRIPT_DIR}/${JOB}.sh"
  { emit_header "$JOB"; cat <<EOF
echo "===== ITI baseline (c=2/4/6) ====="
SKIP_DOLA=1 ITI_ALPHAS="2,4,6" python scripts/baseline_dola_iti.py
echo "===== ITI ppl ====="
python scripts/measure_iti_ppl.py
echo "===== BLADE figure config (rho=.005, alpha 0/2.5, beta=.05) ====="
BETA=0.05 RHOS=0.005 ALPHAS="0,2.5" CAP=40 python scripts/blade_rho_sweep.py
echo "DONE phase 1"
EOF
  } > "$JS"
  submit "$JS"
elif [[ "$PHASE" == "2" ]]; then
  # rho x alpha grid at beta=.05
  JOB="llama32_unc_grid"; JS="${JOBSCRIPT_DIR}/${JOB}.sh"
  { emit_header "$JOB"; cat <<EOF
echo "===== BLADE rho x alpha grid (beta=.05) ====="
BETA=0.05 RHOS="0.002,0.005,0.01,0.02" ALPHAS="0,1.5,2.5,3" CAP=40 OUT_TAG=_grid python scripts/blade_rho_sweep.py
echo "DONE grid"
EOF
  } > "$JS"
  submit "$JS"
  # beta sweep (each redoes ELS): 0.10, 0.20 at rho=.005, alpha 0/2.5
  for B in 0.10 0.20; do
    JOB="llama32_unc_beta${B}"; JS="${JOBSCRIPT_DIR}/${JOB}.sh"
    { emit_header "$JOB"; cat <<EOF
echo "===== BLADE beta=${B} (rho=.005, alpha 0/2.5) ====="
BETA=${B} RHOS=0.005 ALPHAS="0,2.5" CAP=40 OUT_TAG=_beta${B} python scripts/blade_rho_sweep.py
echo "DONE beta=${B}"
EOF
    } > "$JS"
    submit "$JS"
  done
else
  echo "unknown PHASE=$PHASE"; exit 1
fi
echo "watch: squeue -u \$USER ; tail -f ${LOG_DIR}/llama32_unc_*.out"
