#!/usr/bin/env bash
# From an isolated snapshot root containing src/, scripts/, and inputs.json.
set -euo pipefail
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${RUN_DIR:-${SNAPSHOT_DIR}/output}"
ENV_PREFIX="${ENV_PREFIX:-/usr/local/usrapps/jekml/xfang23/venv}"
mkdir -p "$RUN_DIR/logs"

cat > "$RUN_DIR/fit.sbatch" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=bladeiti_fit
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=${RUN_DIR}/logs/fit.%j.out
set -euo pipefail
module load cuda
export PATH="${ENV_PREFIX}/bin:\$PATH"
export HF_HOME=/share/jekml/xfang23/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TTS_C4_FILE=/share/jekml/xfang23/ttsafety-data/c4.txt
export PYTHONPATH="${SNAPSHOT_DIR}/src:${SNAPSHOT_DIR}/scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${SNAPSHOT_DIR}"
python -u scripts/blade_plus_iti.py fit --inputs inputs.json --run-dir "${RUN_DIR}"
EOF

cat > "$RUN_DIR/generate.sbatch" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=bladeiti_combo
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=${RUN_DIR}/logs/combo.%A_%a.out
set -euo pipefail
module load cuda
export PATH="${ENV_PREFIX}/bin:\$PATH"
export HF_HOME=/share/jekml/xfang23/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TTS_C4_FILE=/share/jekml/xfang23/ttsafety-data/c4.txt
export PYTHONPATH="${SNAPSHOT_DIR}/src:${SNAPSHOT_DIR}/scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${SNAPSHOT_DIR}"
variants=(transfer refit)
python -u scripts/blade_plus_iti.py generate --inputs inputs.json --run-dir "${RUN_DIR}" --variant "\${variants[\$SLURM_ARRAY_TASK_ID]}" --batch-size 12
EOF

bash -n "$RUN_DIR/fit.sbatch" "$RUN_DIR/generate.sbatch"
FIT_JOB=$(sbatch --parsable "$RUN_DIR/fit.sbatch")
GEN_JOB=$(sbatch --parsable --dependency="afterok:${FIT_JOB}" "$RUN_DIR/generate.sbatch")
printf 'fit_job=%s\ngenerate_array=%s\nrun_dir=%s\n' "$FIT_JOB" "$GEN_JOB" "$RUN_DIR" | tee "$RUN_DIR/submission.txt"
