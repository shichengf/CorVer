"""Shared config for the data construction pipeline.

All paths and credentials are read from environment variables. See README for
the expected variables.
"""
import os

# Project layout (relative to this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.dirname(os.path.abspath(__file__)).replace("/pipeline", "")
RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERMEDIATE_DIR = os.path.join(DATA_DIR, "intermediate")
FINAL_DIR = os.path.join(DATA_DIR, "final")

# Wikipedia JSONL dump directory (set WIKIPEDIA_JSONL_DIR; default = ./data/wikipedia_jsonl)
WIKIPEDIA_JSONL_DIR = os.environ.get(
    "WIKIPEDIA_JSONL_DIR",
    os.path.join(DATA_DIR, "wikipedia_jsonl"),
)

# LLM API configuration. The pipeline calls an OpenAI-compatible chat-completions
# endpoint for question refinement, CoT distillation, and answer verification.
# Any OpenAI-compatible server works (vLLM, SGLang, TGI, hosted APIs, ...).
# Set DATA_LLM_BASE_URL and DATA_LLM_API_KEY before running the pipeline.
DATA_LLM_BASE_URL = os.environ.get("DATA_LLM_BASE_URL", "")
DATA_LLM_API_KEY = os.environ.get("DATA_LLM_API_KEY", "")
# Primary refinement / CoT model (e.g. a strong instruct model).
DATA_LLM_MODEL = os.environ.get("DATA_LLM_MODEL", "")
# Secondary verifier model for multi-model cross-checking in step3.
SECONDARY_LLM_MODEL = os.environ.get("SECONDARY_LLM_MODEL", "")

# Local vLLM verifier model (HF id by default; override with QWEN25_MODEL_PATH
# to point at a local snapshot).
QWEN25_MODEL = os.environ.get("QWEN25_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")

# Concurrency
API_MAX_CONCURRENT = 50
API_RETRY_MAX = 3
API_RETRY_DELAY = 2.0

# Data filtering hyperparameters
MAX_WIKI_LINKS_PER_ENTITY = 3
SEMHASH_THRESHOLD = 0.9
