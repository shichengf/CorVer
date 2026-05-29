#!/bin/bash
# Llama-3.1-8B-Instruct Raw+RL ablations:
#   1. Vanilla GRPO (judge_only, no CorVer)
#   2. CorVer only (no judge)
# Each: GRPO ~5h + 3 ckpt full eval ~1.5h
# Total: ~13h serial

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="$PROJECT_ROOT/train"

export JUDGE_TYPE="${JUDGE_TYPE:-string_match}"
export JUDGE_REWARD_WEIGHT="${JUDGE_REWARD_WEIGHT:-1.0}"
export INFIGRAM_LOCAL_INDEX_DIR="${INFIGRAM_LOCAL_INDEX_DIR:-./infigram_index}"
export PYTHONUNBUFFERED=1
export DISABLE_VERSION_CHECK=1

LOG_BASE="$PROJECT_ROOT/logs/llama_ablation"
mkdir -p "$LOG_BASE"

run_ablation() {
    local NAME=$1   # judge_only | corver_only
    local MODE=$1   # same as name
    local GRPO_YAML="script/grpo_llama31_8b_${NAME}.yaml"
    local CKPT_DIR="$TRAIN_DIR/output/llama31_8b_${NAME}"
    local EVAL_DIR="$TRAIN_DIR/eval_results/llama31_8b_${NAME}"
    local BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

    echo ""
    echo "##########################################################"
    echo "##  Ablation: $NAME  START  $(date)"
    echo "##########################################################"

    # ----- 1. GRPO -----
    echo "===== [1/2] GRPO training (mode=$MODE, ~5h) ====="
    cd "$TRAIN_DIR"
    STEPWISE_REWARD_MODE="$MODE" python main_stepwise.py --config "$GRPO_YAML" 2>&1 | tee "$LOG_BASE/${NAME}_grpo.log"

    # ----- 2. Eval checkpoints (full 17944) -----
    echo ""
    echo "===== [2/2] Full eval checkpoints ($NAME) ====="
    cd "$PROJECT_ROOT"
    mkdir -p "$EVAL_DIR"
    for STEP in 100 150 200; do
        if [ ! -d "$CKPT_DIR/checkpoint-$STEP" ]; then
            echo "  skip step-$STEP (not found)"
            continue
        fi
        echo "  --- step-$STEP ---"
        python train/eval_triviaqa.py \
            --model_path "$BASE_MODEL" \
            --lora_path "$CKPT_DIR/checkpoint-$STEP" --lora_rank 128 \
            --model_name "llama31_8b_${NAME}_step${STEP}" \
            --num_samples 17944 \
            --output_dir "$EVAL_DIR" 2>&1 | tail -10
    done

    echo ""
    echo "##  Ablation: $NAME  DONE  $(date)"
}

run_ablation judge_only
run_ablation corver_only

echo ""
echo "##########################################################"
echo "##  ALL ABLATIONS DONE  $(date)"
echo "##########################################################"
