#!/bin/bash
# Llama-3.2-3B-Instruct + lp (light prompt + anti-loop rule).
# Trains 100 steps + evals at step-100 on 5 closed-book QA datasets at max_tokens=1000.
set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="$PROJECT_ROOT/train"
LOG_DIR="$PROJECT_ROOT/logs"
EVAL_DIR="$TRAIN_DIR/eval_results/llama32_3b"
mkdir -p "$EVAL_DIR" "$LOG_DIR"

export STEPWISE_REWARD_MODE="${STEPWISE_REWARD_MODE:-full}"
export JUDGE_TYPE="${JUDGE_TYPE:-string_match}"
export JUDGE_REWARD_WEIGHT="${JUDGE_REWARD_WEIGHT:-1.0}"
export INFIGRAM_LOCAL_INDEX_DIR="${INFIGRAM_LOCAL_INDEX_DIR:-./infigram_index}"
export PYTHONUNBUFFERED=1
export DISABLE_VERSION_CHECK=1

cd "$PROJECT_ROOT"

safe_run() {
    local TAG="$1"; shift
    local LOG="$LOG_DIR/${TAG}.log"
    echo ""; echo "##  [$(date '+%H:%M:%S')] $TAG START"
    "$@" 2>&1 | tee "$LOG" || echo "##  [$(date '+%H:%M:%S')] $TAG FAILED (continuing)"
    echo "##  [$(date '+%H:%M:%S')] $TAG DONE"
    return 0
}

DATASETS=(
    "triviaqa:data/triviaqa_validation.json:17944"
    "nq:data/nq_open_validation.json:3610"
    "popqa:data/popqa_test.json:14267"
    "simpleqa:data/simpleqa.json:4326"
    "truthfulqa:data/truthfulqa.json:817"
)

BASE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
RUN_DIR="$TRAIN_DIR/output/llama32_3b"

# ---- Train 100 steps ----
echo ""
echo "############################################################"
echo "#  Llama-3.2-3B-Instruct lp"
echo "############################################################"
cd "$TRAIN_DIR"
if [ -d "$RUN_DIR/checkpoint-100" ]; then
    echo "llama32_3b already trained 100 steps"
else
    safe_run "llama32_3b_grpo" python main_stepwise.py --config script/grpo_llama32_3b.yaml
fi
cd "$PROJECT_ROOT"

CKPT="$RUN_DIR/checkpoint-100"
if [ ! -d "$CKPT" ]; then
    echo "ERROR: $CKPT not found, skipping eval"
    exit 1
fi

# ---- Eval step-100 on 5 datasets ----
for DS in "${DATASETS[@]}"; do
    DS_NAME=$(echo "$DS" | cut -d: -f1)
    DS_PATH=$(echo "$DS" | cut -d: -f2)
    DS_N=$(echo "$DS" | cut -d: -f3)
    OUT_NAME="llama32_3b_step100_${DS_NAME}"
    DONE_FILE="$EVAL_DIR/triviaqa_${OUT_NAME}_completions.json"
    if [ -f "$DONE_FILE" ]; then echo "  ⊘ $OUT_NAME already done"; continue; fi
    safe_run "llama32_3b_${OUT_NAME}" python train/eval_triviaqa.py \
        --model_path "$BASE_MODEL" \
        --lora_path "$CKPT" --lora_rank 128 \
        --model_name "$OUT_NAME" \
        --num_samples "$DS_N" \
        --max_tokens 1000 \
        --triviaqa_path "$DS_PATH" \
        --output_dir "$EVAL_DIR"
done

echo ""
echo "##  llama32_3b DONE @ $(date)"
