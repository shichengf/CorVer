# CorVer

[![arXiv](https://img.shields.io/badge/arXiv-2605.29648-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.29648) [![Hugging Face Paper](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2605.29648) [![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Index-yellow)](https://huggingface.co/datasets/Shichengf/CorVer-infini-gram-index) [![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

**CorVer** (*Corpus Verify*) — a lightweight, plug-in-ready process reward for RL fine-tuning on factual QA. Sentence-level credit comes from a single Wikipedia co-occurrence lookup, no neural verifier.

**[arXiv](https://arxiv.org/abs/2605.29648)** · **[HF Paper](https://huggingface.co/papers/2605.29648)** · **[Wikipedia Index](https://huggingface.co/datasets/Shichengf/CorVer-infini-gram-index)** · **[Training Data](https://huggingface.co/datasets/Shichengf/CorVer-training-data)**

![CorVer pipeline](figures/figure_2.png)

*Each sentence is scored for Wikipedia co-occurrence via an Infini-gram index. The per-sentence reward is mapped to token-level returns through an alignment $\sigma$, then combined with response-level judge and format rewards in a GRPO update.*

## What is CorVer?

> CorVer replaces neural verifiers (NLI, LLM judges, retrieval-and-grade) with a corpus-grounded signal derived from Wikipedia co-occurrence counts. Each sentence costs one 0.5 B extractor pass and one indexed CNF lookup; sentence-level credit is mapped to token-level advantages through a simple alignment.

**Headline numbers** (paper Table 1): 30 (model × benchmark) cells across six instruction-tuned models (3 B – 14 B) and five QA benchmarks. CorVer beats the raw baseline in every cell (+4.1 pp average on TriviaQA) and outperforms four neural-verifier baselines in 18 / 20 cells at 4.8 – 8.4× lower training cost.

## Quick start

```bash
# 1. env
conda env create -f environment.yml && conda activate verl

# 2. prebuilt Wikipedia infini-gram index (Llama-2 tokenized)
huggingface-cli download Shichengf/CorVer-infini-gram-index \
    --repo-type dataset --local-dir ./infigram_index
export INFIGRAM_LOCAL_INDEX_DIR=./infigram_index

# 3. infini-gram engine + Llama-2 tokenizer (gated; request access on HF)
pip install infini-gram
huggingface-cli login

# 4. train Llama-3.1-8B-Instruct (1× A100 80GB, ~3 h)
bash train/run_8b_models.sh
```

The Llama-3.1-8B self-filtered curriculum is bundled at [`data/rl/stepwise_curriculum_llama31_8b.json`](data/rl/stepwise_curriculum_llama31_8b.json). For the other five models, grab the matching curriculum from the [training-data dataset](https://huggingface.co/datasets/Shichengf/CorVer-training-data) and drop it into `data/rl/`:

```bash
huggingface-cli download Shichengf/CorVer-training-data \
    --repo-type dataset --include "curricula/*" --local-dir ./_corver_data
mv ./_corver_data/curricula/*.json data/rl/ && rm -rf ./_corver_data
```

Or rebuild from scratch (see *Rebuilding data*).

## Models & launchers

One canonical config per model, all 100 GRPO steps (paper Appendix A.3):

| Model | YAML | Launcher |
|---|---|---|
| Llama-3.2-3B-Instruct | `train/script/grpo_llama32_3b.yaml` | `train/run_llama32_3b.sh` |
| Qwen3-4B | `train/script/grpo_qwen3_4b.yaml` | `train/run_qwen3_4b.sh` |
| Llama-3.1-8B-Instruct *(headline)* | `train/script/grpo_llama31_8b.yaml` | `train/run_8b_models.sh` |
| Qwen3-8B | `train/script/grpo_qwen3_8b.yaml` | `train/run_8b_models.sh` |
| OLMo-2-1124-13B-Instruct | `train/script/grpo_olmo2_13b.yaml` | `train/run_olmo2_13b.sh` |
| Qwen3-14B | `train/script/grpo_qwen3_14b.yaml` | `train/run_qwen3_14b.sh` |

Reward ablations live in `train/script/grpo_llama31_8b_{judge,corver}_only.yaml` (`train/run_llama31_8b_ablations.sh`).

## Rebuilding data

The end-to-end pipeline lives in `data/pipeline/`; run `bash data/pipeline/run_pipeline.sh`. It pulls NQ-Open + WebQuestions, dedups, refines / classifies via an OpenAI-compatible chat endpoint, grounds entities against Wikipedia, then assembles the per-difficulty RL pools. Set `DATA_LLM_BASE_URL`, `DATA_LLM_API_KEY`, `DATA_LLM_MODEL`, and `WIKIPEDIA_JSONL_DIR` first. The upstream [KnowRL training data](https://huggingface.co/datasets/zjunlp/KnowRL-Train-Data) is used as the baseline pool in the paper; CorVer itself does not depend on it.

Per-target self-filtering (`train/check_positive_signal.py`) produces `data/rl/stepwise_curriculum_<model>.json` from the merged pool, retaining prompts the raw target solves with `n_correct ∈ [1, G-1]` over G = 16 generations.

## Citation

```bibtex
@article{fan2026verifiable,
  title={Verifiable Rewards Beyond Math and Code: Lightweight Corpus-Grounded Process Supervision for Factual Question Answering},
  author={Fan, Shicheng and Hao, Haochang and Min, Dehai and Liu, Weihao and Yu, Philip S and Cheng, Lu},
  journal={arXiv preprint arXiv:2605.29648},
  year={2026}
}
```

## License

[MIT](LICENSE)
