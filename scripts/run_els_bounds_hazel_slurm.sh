#!/usr/bin/env bash
# Git worktree must already contain committed benchmark source and inputs.
set -euo pipefail
: "${SNAPSHOT_DIR:?Set SNAPSHOT_DIR to the isolated Git worktree}"
INPUTS_FILE="${INPUTS_FILE:-${SNAPSHOT_DIR}/results/els_cache/20260906/inputs.json}"
RUN_DIR="${RUN_DIR:-${SNAPSHOT_DIR}/results/els_bounds/20260906/gpu}"
ENV_PREFIX="${ENV_PREFIX:-/usr/local/usrapps/jekml/xfang23/venv}"
BENCH_NODE="${BENCH_NODE:-gpu18}"
mkdir -p "$RUN_DIR/logs"
cat > "$RUN_DIR/benchmark.sbatch" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=els_bounds
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --nodelist=${BENCH_NODE}
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=${RUN_DIR}/logs/benchmark.%A_%a.out
set -euo pipefail
module load cuda
export PATH="${ENV_PREFIX}/bin:\$PATH"
export HF_HOME=/share/jekml/xfang23/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TTS_C4_FILE=/share/jekml/xfang23/ttsafety-data/c4.txt
export PYTHONPATH="${SNAPSHOT_DIR}/src:${SNAPSHOT_DIR}/scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${SNAPSHOT_DIR}"
modes=(baseline bounded)
mode="\${modes[\$SLURM_ARRAY_TASK_ID]}"
python -u scripts/benchmark_els_bounds.py --mode "\$mode" --inputs "${INPUTS_FILE}" --run-dir "${RUN_DIR}/\$mode"
EOF
bash -n "$RUN_DIR/benchmark.sbatch"
BENCH_JOB=$(sbatch --parsable "$RUN_DIR/benchmark.sbatch")
printf 'benchmark_array=%s\nrun_dir=%s\n' "$BENCH_JOB" "$RUN_DIR" | tee "$RUN_DIR/submission.txt"
