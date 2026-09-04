#!/usr/bin/env bash
# Submit the P0 uncertainty-calibration eval on Hazel: one job per (bench x mode), 14B on l40s.
# Prereq: bash scripts/prefetch_bench_hazel.sh (login node). Run from a login node.
# Env: BENCHES (default "math"), MODES (default "base remove random shuffle"), N (200), RHO (0.008).
set -euo pipefail
ENV=/usr/local/usrapps/jekml/xfang23/venv
SH=/share/jekml/xfang23; REPO=$SH/TestTimeSafetyPrune; LOG=$SH/jobs/ttsafety/logs
JS=$LOG/jobscripts; mkdir -p "$JS"
MODEL="Qwen/Qwen3-14B"; DIRS=qwen3_14b_dirs.pt; ELS=reasoning_els_qwen3_14b.json
BENCHES="${BENCHES:-math}"; MODES="${MODES:-base remove random shuffle}"; N="${N:-200}"; RHO="${RHO:-0.008}"

for bench in $BENCHES; do for mode in $MODES; do
  job=calib_14b_${bench}_${mode}
  cat > "$JS/$job.sh" <<JOB
#!/usr/bin/env bash
#SBATCH --job-name=$job
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=$LOG/$job.%j.out
#SBATCH --error=$LOG/$job.%j.out
set -euo pipefail
module load cuda
export PATH=$ENV/bin:\$PATH
export HF_HOME=$SH/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TTS_C4_FILE=$SH/ttsafety-data/c4.txt TTS_WIKI_FILE=$SH/ttsafety-data/wikitext.txt
export TTS_GSM8K_FILE=$SH/ttsafety-data/gsm8k.jsonl TTS_MATH_FILE=$SH/ttsafety-data/math500.jsonl
export PYTHONPATH=$REPO/src:$REPO/scripts
cd $REPO
python scripts/eval_calibration.py --model $MODEL --dirs $DIRS --els $ELS \
  --bench $bench --mode $mode --rho $RHO --n $N --out calib_14b_${bench}_${mode}.json
JOB
  sbatch "$JS/$job.sh"
  echo "submitted $job"
done; done
