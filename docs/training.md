# Training guide

The entry point is `python -m corver.train`; choose a YAML from `corver/configs/`. All commands in the repository documentation run from the repository root.

## Model configuration

All six configurations start from the official instruction-tuned base with a fresh LoRA adapter: rank 128, alpha 256, no intermediate SFT, seed 42, 100 optimizer steps, 16 completions per question, rollout temperature 1.0, and at most 1,000 new tokens.

| Configuration | Learning rate | KL | Correctness weight | Microbatch | Accumulation | Questions/step |
|---|---:|---:|---:|---:|---:|---:|
| `llama32_3b.yaml` | 1e-5 | .001 | 1 | 2 | 24 | 3 |
| `qwen3_4b.yaml` | 1e-5 | .01 | 1 | 2 | 32 | 4 |
| `llama31_8b.yaml` | 1e-5 | .003 | 2 | 2 | 24 | 3 |
| `qwen3_8b.yaml` | 1e-5 | .001 | 1 | 2 | 24 | 3 |
| `olmo2_13b.yaml` | 1e-5 | .001 | 1 | 1 | 48 | 3 |
| `qwen3_14b.yaml` | 1e-5 | .001 | 1 | 1 | 48 | 3 |

## Rewards and advantages

For each sentence, use the first extracted head/tail pair that passes validity checks and yields at least two distinct content terms. The relation is not queried. The index receives one AND query; a failed query receives zero reward, without trying a later pair. Counts map to rewards as follows:

| Count | Reward |
|---|---:|
| No valid query / query failure | 0 |
| 0 | −0.2 |
| 1–4 | −0.1 |
| 5–19 | 0 |
| ≥20 | +0.1 |

Answer correctness is +2 for a matching final answer and −1 otherwise, including refusals and empty answers. Matching uses the normalized bidirectional substring rule in `common/protocol.py`. Format reward is +1 for a closed, nonempty answer span and −1 otherwise, minus `0.5 * max(0, completion_length - 512) / 512`. Sentence and format weights are 1.

Eq. (4) uses each completion's mean token return to compute the group mean and sample standard deviation. Token advantages are `(token_return - group_mean) / (group_std + 1e-4)`, with padding excluded. The denominator is not a token-level standard deviation. There is no direct advantage clipping; epsilon prevents division by zero but does not bound advantage magnitude. Training logs include maximum absolute advantage, group standard deviation, and token alignment. GRPO ratio clipping and optimizer gradient clipping are distinct operations.

## Frozen reward resources

The index uses version 4, u16 tokens, `max_diff_tokens=1000`, and `max_clause_freq=500000`. This distance uses the fixed Llama-2 index tokenizer. Successful counts, including zero, are cached; failed queries are not.

The frozen `ZhishanQ/QuCo-extractor-0.5B` uses BF16, temperature .7, top-p .8, top-k 20, repetition penalty 1.1, up to 256 new tokens, 4,096-token context, stop IDs 151645/151643, and seed `42 + successful_request_count`. Its persistent vLLM worker uses CUDA Graphs, up to 64 sequences and 4,096 batched tokens. The policy uses eager vLLM. Extractor request counts are saved with checkpoints for resume.

Each YAML pins the base model, extractor, index tokenizer, and corpus index revisions. Local path overrides should point to those snapshots. Use `--index-tokenizer-path` or `CORVER_INDEX_TOKENIZER_PATH` for a local copy of the index tokenizer. Index/model weights are downloaded separately and keep their upstream licenses.

## Runtime settings

Runtime settings: single GPU, policy memory fraction .35 for 3B–8B or .55 for 13B–14B, extractor memory fraction .065, prompt limit 256, AdamW8bit, cosine schedule, warmup .03, weight decay .01, gradient norm limit 1, GRPO ratio clip .2, LoRA dropout 0, checkpoint every 10 updates, and two retained checkpoints. Top-p is 1 and top-k is disabled for policy rollouts. The fixed chat-template date is `07 Sep 2026`. Effective Trainer arguments are written with each run.
