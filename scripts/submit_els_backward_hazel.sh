#!/usr/bin/env bash
# Generate and submit the complete fit -> pools -> search slices -> final DAG.
set -euo pipefail
: "${SNAPSHOT_DIR:?Set SNAPSHOT_DIR to the isolated Git worktree on Hazel}"
: "${RUN_ROOT:?Set RUN_ROOT to an external persistent run directory}"
INPUTS_FILE="${INPUTS_FILE:-${SNAPSHOT_DIR}/results/els_cache/20260906/inputs.json}"
ENV_PREFIX="${ENV_PREFIX:-/usr/local/usrapps/jekml/xfang23/venv}"
PARTITION="${PARTITION:-gpu_partners}"
QOS="${QOS:-short_gpu}"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/sbatch"

write_job () {
  local path="$1" name="$2" array="$3" minutes="$4" body="$5"
  {
    printf '%s\n' '#!/usr/bin/env bash'
    printf '#SBATCH --job-name=%s\n' "$name"
    printf '#SBATCH --partition=%s\n#SBATCH --qos=%s\n' "$PARTITION" "$QOS"
    printf '%s\n' '#SBATCH --gres=gpu:l40s:1' '#SBATCH --cpus-per-task=8' '#SBATCH --mem=128G'
    printf '#SBATCH --time=%s\n' "$minutes"
    [[ -z "$array" ]] || printf '#SBATCH --array=%s\n' "$array"
    printf '#SBATCH --output=%s/logs/%s.%%A_%%a.out\n' "$RUN_ROOT" "$name"
    printf '%s\n' 'set -euo pipefail' 'module load cuda'
    printf 'export PATH=%q/bin:$PATH\n' "$ENV_PREFIX"
    printf '%s\n' 'export HF_HOME=/share/jekml/xfang23/.cache/huggingface' \
      'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1' \
      'export TTS_C4_FILE=/share/jekml/xfang23/ttsafety-data/c4.txt' \
      'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
    printf 'export PYTHONPATH=%q/src:%q/scripts\ncd %q\n' "$SNAPSHOT_DIR" "$SNAPSHOT_DIR" "$SNAPSHOT_DIR"
    printf '%s\n' "$body"
  } > "$path"
  bash -n "$path"
}

PY='python -u scripts/run_els_backward.py'
COMMON="--inputs \"${INPUTS_FILE}\" --run-root \"${RUN_ROOT}\""
write_job "$RUN_ROOT/sbatch/fit.sbatch" els_b_fit "" 02:00:00 "$PY fit $COMMON"
write_job "$RUN_ROOT/sbatch/pool.sbatch" els_b_pool '0-3%4' 02:00:00 \
  'RHOS=(.005 .01); BETAS=(.05 .10); i=$SLURM_ARRAY_TASK_ID; b=$((i%2)); r=$((i/2)); python -u scripts/run_els_backward.py pool '"$COMMON"' --rho "${RHOS[$r]}" --beta "${BETAS[$b]}"'

SEARCH_BODY='RHOS=(.005 .01); BETAS=(.05 .10); EPS=(.005 .025); DIRS=(forward backward); i=$SLURM_ARRAY_TASK_ID; d=$((i%2)); i=$((i/2)); e=$((i%2)); i=$((i/2)); b=$((i%2)); r=$((i/2)); python -u scripts/run_els_backward.py search '"$COMMON"' --rho "${RHOS[$r]}" --beta "${BETAS[$b]}" --eps "${EPS[$e]}" --direction "${DIRS[$d]}" --slice "${SLICE}" --slice-minutes 70'
write_job "$RUN_ROOT/sbatch/search.sbatch" els_b_search '0-15%4' 01:20:00 "$SEARCH_BODY"
FINAL_BODY='RHOS=(.005 .01); BETAS=(.05 .10); EPS=(.005 .025); DIRS=(forward backward); i=$SLURM_ARRAY_TASK_ID; d=$((i%2)); i=$((i/2)); e=$((i%2)); i=$((i/2)); b=$((i%2)); r=$((i/2)); python -u scripts/run_els_backward.py final '"$COMMON"' --rho "${RHOS[$r]}" --beta "${BETAS[$b]}" --eps "${EPS[$e]}" --direction "${DIRS[$d]}"'
write_job "$RUN_ROOT/sbatch/final.sbatch" els_b_final '0-15%4' 02:00:00 "$FINAL_BODY"

FIT_JOB=$(sbatch --parsable "$RUN_ROOT/sbatch/fit.sbatch")
POOL_JOB=$(sbatch --parsable --dependency="afterok:${FIT_JOB}" "$RUN_ROOT/sbatch/pool.sbatch")
PREV="$POOL_JOB"; SEARCH_JOBS=()
for SLICE in 0 1 2; do
  export SLICE
  # Expand only the slice number; the array variables remain literal in the file.
  sed "s/\${SLICE}/${SLICE}/g" "$RUN_ROOT/sbatch/search.sbatch" > "$RUN_ROOT/sbatch/search-${SLICE}.sbatch"
  JOB=$(sbatch --parsable --dependency="afterany:${PREV}" "$RUN_ROOT/sbatch/search-${SLICE}.sbatch")
  SEARCH_JOBS+=("$JOB"); PREV="$JOB"
done
FINAL_JOB=$(sbatch --parsable --dependency="afterany:${PREV}" "$RUN_ROOT/sbatch/final.sbatch")
printf 'fit_job=%s\npool_job=%s\nsearch_jobs=%s\nfinal_job=%s\nrun_root=%s\n' \
  "$FIT_JOB" "$POOL_JOB" "${SEARCH_JOBS[*]}" "$FINAL_JOB" "$RUN_ROOT" | tee "$RUN_ROOT/submission.txt"
