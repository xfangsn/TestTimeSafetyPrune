#!/usr/bin/env bash
# Submit from an isolated Git worktree (SNAPSHOT_DIR retains its legacy name).
set -euo pipefail
: "${SNAPSHOT_DIR:?Set SNAPSHOT_DIR to the isolated benchmark Git worktree}"
INPUTS_FILE="${INPUTS_FILE:-${SNAPSHOT_DIR}/results/els_cache/20260906/inputs.json}"
RUN_DIR="${RUN_DIR:-${SNAPSHOT_DIR}/output}"
ENV_PREFIX="${ENV_PREFIX:-/usr/local/usrapps/jekml/xfang23/venv}"
REFERENCE_RUN="${REFERENCE_RUN:-/share/jekml/xfang23/ttsafety_blade_iti/rho001_a2_20260906/output}"
PAIRED_REPEATS="${PAIRED_REPEATS:-2}"
mkdir -p "$RUN_DIR/logs"
cat > "$RUN_DIR/benchmark.sbatch" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=els_cache_bench
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=${RUN_DIR}/logs/benchmark.%j.out
set -euo pipefail
module load cuda
export PATH="${ENV_PREFIX}/bin:\$PATH"
export HF_HOME=/share/jekml/xfang23/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TTS_C4_FILE=/share/jekml/xfang23/ttsafety-data/c4.txt
export PYTHONPATH="${SNAPSHOT_DIR}/src:${SNAPSHOT_DIR}/scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${SNAPSHOT_DIR}"
python -u scripts/benchmark_els_cache.py --inputs "${INPUTS_FILE}" --run-dir "${RUN_DIR}" --reference-run "${REFERENCE_RUN}" --paired-repeats "${PAIRED_REPEATS}"
EOF
bash -n "$RUN_DIR/benchmark.sbatch"
BENCH_JOB=$(sbatch --parsable "$RUN_DIR/benchmark.sbatch")
printf 'benchmark_job=%s\nrun_dir=%s\n' "$BENCH_JOB" "$RUN_DIR" | tee "$RUN_DIR/submission.txt"
