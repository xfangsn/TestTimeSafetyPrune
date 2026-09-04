#!/usr/bin/env bash
# Double-insurance 14B submit on Hazel:
#   Path A: gpu partition H100 (infinite time), full BLADE-B+BLADE-G chain, TAG=qwen3_14b.
#   Path B: gpu_partners l40s (2h short_gpu), split into prep -> BLADE-B -> BLADE-G (deps), TAG=qwen3_14bL.
# Whichever finishes 14B first wins; cancel the other. Separate TAGs avoid output clobbering.
# Run from a Hazel login node:  bash scripts/submit_14b_hazel.sh
set -euo pipefail
ENV_PREFIX="${ENV_PREFIX:-/usr/local/usrapps/jekml/xfang23/venv}"
SHARE_ROOT="${SHARE_ROOT:-/share/jekml/xfang23}"
CACHE_ROOT="${CACHE_ROOT:-${SHARE_ROOT}/.cache}"
DATA_DIR="${DATA_DIR:-${SHARE_ROOT}/ttsafety-data}"
LOG_DIR="${LOG_DIR:-${SHARE_ROOT}/jobs/ttsafety/logs}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
JS="${LOG_DIR}/jobscripts"; mkdir -p "$JS"
MODEL="Qwen/Qwen3-14B"

emit() {  # emit <file> <jobname> <partition> <qos> <gres> <mem> <time> <extra-env> <tag> <only>
  local f="$1" job="$2" part="$3" qos="$4" gres="$5" mem="$6" tl="$7" env="$8" tag="$9" only="${10}"
  cat > "$f" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job}
#SBATCH --partition=${part}
#SBATCH --qos=${qos}
#SBATCH --gres=${gres}
#SBATCH --cpus-per-task=8
#SBATCH --mem=${mem}
#SBATCH --time=${tl}
#SBATCH --output=${LOG_DIR}/${job}.%j.out
#SBATCH --error=${LOG_DIR}/${job}.%j.out
set -euo pipefail
module load cuda
export PATH="${ENV_PREFIX}/bin:\$PATH"
export HF_HOME="${CACHE_ROOT}/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TTS_C4_FILE="${DATA_DIR}/c4.txt" TTS_WIKI_FILE="${DATA_DIR}/wikitext.txt"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts"
cd "${REPO_DIR}"
${env} MODEL="${MODEL}" TAG="${tag}" PY=python bash scripts/run_reasoning_qwen3.sh
EOF
}

# --- Path A: H100, infinite time, full chain ---
emit "$JS/rsn_q14b_h100.sh" rsn_q14b_h100 gpu gpu gpu:h100:1 96G 24:00:00 "ONLY=both" qwen3_14b both
A=$(sbatch --parsable "$JS/rsn_q14b_h100.sh"); echo "A(h100 full) job $A"

# --- Path B: l40s, 2h, prep -> bladeb -> bladeg (dependencies) ---
emit "$JS/rsn_q14bL_prep.sh"   rsn_q14bL_prep   gpu_partners short_gpu gpu:l40s:1 96G 02:00:00 "ONLY=prep" qwen3_14bL prep
B1=$(sbatch --parsable "$JS/rsn_q14bL_prep.sh"); echo "B1(l40s prep) job $B1"
emit "$JS/rsn_q14bL_bladeb.sh" rsn_q14bL_bladeb gpu_partners short_gpu gpu:l40s:1 96G 02:00:00 "ONLY=bladeb SKIP_TRACES=1 SKIP_DIRS=1 SKIP_ELS=1" qwen3_14bL bladeb
B2=$(sbatch --parsable --dependency=afterok:$B1 "$JS/rsn_q14bL_bladeb.sh"); echo "B2(l40s BLADE-B, after $B1) job $B2"
emit "$JS/rsn_q14bL_bladeg.sh" rsn_q14bL_bladeg gpu_partners short_gpu gpu:l40s:1 96G 02:00:00 "ONLY=bladeg SKIP_TRACES=1 SKIP_DIRS=1 SKIP_ELS=1" qwen3_14bL bladeg
B3=$(sbatch --parsable --dependency=afterok:$B1 "$JS/rsn_q14bL_bladeg.sh"); echo "B3(l40s BLADE-G, after $B1) job $B3"
echo "submitted. A=$A (h100 full, TAG qwen3_14b) ; B=$B1->$B2,$B3 (l40s split, TAG qwen3_14bL)"
