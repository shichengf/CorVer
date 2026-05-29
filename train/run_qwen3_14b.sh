#!/bin/bash
# Qwen3-14B CorVer launcher: lr 2.5e-5, beta=0 (no KL penalty), max_steps=100.
# Trains 100 GRPO steps + evals the step-50 / step-100 checkpoints on TriviaQA.
set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="$PROJECT_ROOT/train"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

export STEPWISE_REWARD_MODE="${STEPWISE_REWARD_MODE:-full}"
export JUDGE_TYPE="${JUDGE_TYPE:-string_match}"
export JUDGE_REWARD_WEIGHT="${JUDGE_REWARD_WEIGHT:-1.0}"
export INFIGRAM_LOCAL_INDEX_DIR="${INFIGRAM_LOCAL_INDEX_DIR:-./infigram_index}"
export PYTHONUNBUFFERED=1
export DISABLE_VERSION_CHECK=1

safe_run() {
    local TAG="$1"; shift
    local LOG="$LOG_DIR/${TAG}.log"
    echo ""; echo "##  [$(date '+%H:%M:%S')] $TAG START"
    "$@" 2>&1 | tee "$LOG" || echo "##  [$(date '+%H:%M:%S')] $TAG FAILED (continuing)"
    echo "##  [$(date '+%H:%M:%S')] $TAG DONE"
    return 0
}

BASE_MODEL="Qwen/Qwen3-14B"
CKPT_DIR="$TRAIN_DIR/output/qwen3_14b"

# Verify the self-filtered curriculum is in place
FILTERED="$PROJECT_ROOT/data/rl/stepwise_curriculum_qwen3_14b.json"
if [ ! -f "$FILTERED" ]; then
    echo "WARNING: $FILTERED not found."
    echo "Build it first with: python train/check_positive_signal.py --model Qwen/Qwen3-14B ..."
    echo "(see data/pipeline/ and check_positive_signal.py for details)."
fi

# ---- Train 50 steps ----
cd "$TRAIN_DIR"
if [ -d "$CKPT_DIR/checkpoint-100" ]; then
    echo "Qwen3-14B already trained 100 steps"
else
    safe_run "qwen3_14b_grpo" python main_stepwise.py --config script/grpo_qwen3_14b.yaml
fi
cd "$PROJECT_ROOT"

# ---- Eval step-25 + step-50 on TriviaQA ----
EVAL_DIR="$TRAIN_DIR/eval_results/qwen3_14b"
mkdir -p "$EVAL_DIR"
for STEP in 50 100; do
    CKPT="$CKPT_DIR/checkpoint-$STEP"
    [ -d "$CKPT" ] || { echo "skip step-$STEP (no ckpt)"; continue; }
    safe_run "qwen3_14b_eval_step${STEP}" python train/eval_triviaqa.py \
        --model_path "$BASE_MODEL" \
        --lora_path "$CKPT" --lora_rank 128 --max_tokens 2048 \
        --model_name "qwen3_14b_step${STEP}" \
        --num_samples 17944 \
        --output_dir "$EVAL_DIR"
done

echo ""
echo "##  Qwen3-14B DONE @ $(date)"
