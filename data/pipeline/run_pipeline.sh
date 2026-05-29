#!/bin/bash
# End-to-end driver for the CorVer data pipeline.
#
# Builds the RL training pool from NQ-Open and WebQuestions. Steps 2/3 issue a
# large number of LLM API requests; budget several hours of wall-clock per pass.
#
# Required environment variables:
#   DATA_LLM_BASE_URL    OpenAI-compatible chat endpoint
#   DATA_LLM_API_KEY     API key for the above
#   DATA_LLM_MODEL       Primary refinement / CoT model id
#   SECONDARY_LLM_MODEL  Secondary verifier model id (used in step 3 only)
#   WIKIPEDIA_JSONL_DIR  Path to the Wikipedia JSONL dump (used in step 4)
#
# Optional:
#   QWEN25_MODEL_PATH    Local Qwen2.5-7B-Instruct snapshot for the phase-1 verifier
set -e
cd "$(dirname "$0")"

echo "========================================="
echo "CorVer data pipeline"
echo "========================================="

echo ""
echo ">>> Step 0: download raw datasets"
python step0_download_sources.py

echo ""
echo ">>> Step 1: dedup"
python step1_dedup.py

echo ""
echo ">>> Step 2: quality check + entity extraction"
python step2_refine.py

echo ""
echo ">>> Step 3: three-level difficulty classification"
echo "    (phase 1: local verifier, phase 2: secondary LLM, phase 3: primary LLM with CoT)"
python step3_difficulty_classify.py

echo ""
echo ">>> Step 4: Wikipedia entity grounding"
python step4_wiki_grounding.py

echo ""
echo ">>> Step 5: assemble final RL files"
python step5_build_final_data.py

echo ""
echo "========================================="
echo "Pipeline complete!"
echo "========================================="
echo ""
echo "Outputs (under data/final/):"
echo "  RL pool (by difficulty):  easy_rl.json / medium_rl.json / hard_rl.json"
echo "  RL + knowledge companion: easy_rl_withknowledge.json / medium_* / hard_*"
echo ""
echo "Next: build per-model self-filtered curricula via train/check_positive_signal.py."
