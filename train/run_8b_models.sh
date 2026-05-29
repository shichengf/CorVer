#!/bin/bash
# Llama-3.1-8B + Qwen3-8B -> GRPO 100 steps
# then eval step100 ONLY × 5 ds with light think prompt + anti-loop (auto via lora_path) + max=1000
# (NO frequency_penalty per user decision).
# Serial: Llama-8B train -> Llama-8B eval -> Qwen-8B train -> Qwen-8B eval.
set -u
set -o pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="$PROJECT_ROOT/train"
LOG_DIR="$PROJECT_ROOT/logs"
EVAL_DIR="$TRAIN_DIR/eval_results/8b_models"
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

eval_ckpt() {
    local MODEL_SHORT="$1"   # llama31_8b or qwen3_8b
    local BASE_HF="$2"
    local OUT_TAG="$3"       # llama31_8b or qwen3_8b
    local CKPT="$4"
    if [ ! -d "$CKPT" ]; then
        echo "skip eval: $CKPT does not exist"; return 0
    fi
    for DS in "${DATASETS[@]}"; do
        DS_NAME=$(echo "$DS" | cut -d: -f1)
        DS_PATH=$(echo "$DS" | cut -d: -f2)
        DS_N=$(echo "$DS" | cut -d: -f3)
        OUT_NAME="${OUT_TAG}_step100_${DS_NAME}"
        DONE_FILE="$EVAL_DIR/triviaqa_${OUT_NAME}_completions.json"
        if [ -f "$DONE_FILE" ]; then echo "  ⊘ $OUT_NAME already done"; continue; fi
        safe_run "eval_${OUT_NAME}" python train/eval_triviaqa.py \
            --model_path "$BASE_HF" \
            --lora_path "$CKPT" --lora_rank 128 \
            --model_name "$OUT_NAME" \
            --num_samples "$DS_N" \
            --max_tokens 1000 \
            --triviaqa_path "$DS_PATH" \
            --output_dir "$EVAL_DIR"
    done
}

# ============================================================
# RUN 1: Llama-3.1-8B-Instruct
# ============================================================
RUN1_DIR="$TRAIN_DIR/output/llama31_8b"
echo ""
echo "############################################################"
echo "#  RUN 1: llama31_8b (anti-loop training prompt)"
echo "############################################################"
cd "$TRAIN_DIR"
if [ -d "$RUN1_DIR/checkpoint-100" ]; then
    echo "llama31_8b already trained 100 steps"
else
    safe_run "llama31_8b_grpo" python main_stepwise.py --config script/grpo_llama31_8b.yaml
fi
cd "$PROJECT_ROOT"
eval_ckpt "llama31_8b" "meta-llama/Llama-3.1-8B-Instruct" "llama31_8b" "$RUN1_DIR/checkpoint-100"

# ============================================================
# RUN 2: Qwen3-8B
# ============================================================
RUN2_DIR="$TRAIN_DIR/output/qwen3_8b"
echo ""
echo "############################################################"
echo "#  RUN 2: qwen3_8b (anti-loop training prompt)"
echo "############################################################"
cd "$TRAIN_DIR"
if [ -d "$RUN2_DIR/checkpoint-100" ]; then
    echo "qwen3_8b already trained 100 steps"
else
    safe_run "qwen3_8b_grpo" python main_stepwise.py --config script/grpo_qwen3_8b.yaml
fi
cd "$PROJECT_ROOT"
eval_ckpt "qwen3_8b" "Qwen/Qwen3-8B" "qwen3_8b" "$RUN2_DIR/checkpoint-100"

echo ""
echo "##  8B 2 runs DONE @ $(date)"
