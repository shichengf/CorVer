"""Five-dataset evaluator: SFT/RL model vs base model.

Uses alias-aware substring matching, no LLM judge required.

Usage:
    python train/eval_triviaqa.py --num_samples 500
"""
import argparse
import json
import os
import re
import time
import unicodedata
from typing import List, Dict

import torch

TRIVIAQA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "triviaqa_validation.json")
SFT_MODEL = os.environ.get("SFT_MODEL", "")
RL_MODEL = os.environ.get("RL_MODEL", "")
BASELINE_MODEL = os.environ.get("BASELINE_MODEL", "")
BASE_MODEL = os.environ.get("BASE_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

SYSTEM_PROMPT = (
    "Answer using this exact format:\n"
    "<think>brief reasoning</think>\n"
    "<answer>direct answer</answer>\n\n"
    "If you genuinely do not know, reply exactly: <think>unknown</think><answer>I don't know</answer>"
)


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def extract_answer(completion: str) -> str:
    # Match everything after <answer>, ignore whether </answer> closes.
    match = re.search(r"<answer>(.*)", completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def check_correct(predicted: str, gold: str, aliases: List[str]) -> str:
    if not predicted or predicted.lower().strip() in ["i don't know", "i do not know", ""]:
        return "NOT_ATTEMPTED"

    pred_norm = normalize(predicted)
    if not pred_norm:
        return "NOT_ATTEMPTED"

    # Check against gold + all aliases
    all_answers = [gold] + (aliases or [])
    for ans in all_answers:
        ans_norm = normalize(ans)
        if not ans_norm:
            continue
        if pred_norm == ans_norm:
            return "CORRECT"
        if ans_norm in pred_norm:
            return "CORRECT"
        if pred_norm in ans_norm:
            return "CORRECT"

    return "INCORRECT"


def generate_and_eval(model_path: str, name: str, samples: List[Dict], gpu_mem: float = 0.85,
                       lora_path: str = None, lora_rank: int = 128, max_tokens: int = 1024):
    """Generate completions and grade them.

    Args:
        model_path: HF model id or local path to base model.
        lora_path: optional LoRA adapter path. If set, vLLM loads the adapter
            on top of `model_path` (skipping PEFT merge). LoRA tokenizer is
            ignored — base model tokenizer is used.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"  Model: {model_path}")
    if lora_path:
        print(f"  LoRA:  {lora_path}")
    print(f"  Samples: {len(samples)}")
    print(f"{'='*60}")

    # Always use base model tokenizer (LoRA shares vocab with base).
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # For trained ckpts (lora_path set), append anti-loop rule to mitigate
    # repetition mode-collapse induced by per-token CorVer cumulative reward.
    sys_prompt = SYSTEM_PROMPT
    if lora_path:
        sys_prompt = sys_prompt + "\nDo not loop or repeat the same point or phrase."

    prompts = []
    for s in samples:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": s["question"]},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    llm_kwargs = dict(
        model=model_path,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_tokens + 512,
        trust_remote_code=True,
    )
    if lora_path:
        llm_kwargs.update(enable_lora=True, max_lora_rank=lora_rank)
    llm = LLM(**llm_kwargs)

    # Build stop tokens dynamically based on model type
    stop_tokens = []
    vocab = tokenizer.get_vocab()
    for tok in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]:
        if tok in vocab:
            stop_tokens.append(tok)
    if not stop_tokens:
        stop_tokens = [tokenizer.eos_token]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop=stop_tokens,
    )

    generate_kwargs = {}
    if lora_path:
        from vllm.lora.request import LoRARequest
        generate_kwargs["lora_request"] = LoRARequest("eval_lora", 1, lora_path)

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params, **generate_kwargs)
    gen_time = time.time() - t0
    completions = [o.outputs[0].text for o in outputs]

    del llm
    torch.cuda.empty_cache()

    correct = 0
    incorrect = 0
    not_attempted = 0
    format_correct = 0
    details = []

    for i, (s, comp) in enumerate(zip(samples, completions)):
        predicted = extract_answer(comp)
        grade = check_correct(predicted, s["answer"], s.get("aliases", []))

        if grade == "CORRECT":
            correct += 1
        elif grade == "INCORRECT":
            incorrect += 1
        else:
            not_attempted += 1

        text = comp if comp.startswith("<think>") else "<think>" + comp
        # Lenient: don't require </answer> close (truncated outputs from verbose
        # models still pass) — matches extract_answer's lenient regex.
        if re.search(r"<think>.*?</think>.*?<answer>", text, re.DOTALL):
            format_correct += 1

        details.append({
            "question": s["question"],
            "gold_answer": s["answer"],
            "aliases": s.get("aliases", []),
            "predicted": predicted,
            "completion": comp,
            "grade": grade,
        })

    total = len(samples)
    attempted = correct + incorrect
    precision = correct / attempted if attempted > 0 else 0
    recall = correct / total if total > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        "name": name,
        "model": model_path,
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": not_attempted,
        "correct_rate": correct / total,
        "incorrect_rate": incorrect / total,
        "not_attempted_rate": not_attempted / total,
        "f1": f1,
        "format_accuracy": format_correct / total,
        "generation_time": gen_time,
    }

    print(f"\nResults for {name}:")
    print(f"  Correct:       {correct}/{total} ({correct/total:.1%})")
    print(f"  Incorrect:     {incorrect}/{total} ({incorrect/total:.1%})")
    print(f"  Not Attempted: {not_attempted}/{total} ({not_attempted/total:.1%})")
    print(f"  F1:            {f1:.4f}")
    print(f"  Format Acc:    {format_correct}/{total} ({format_correct/total:.1%})")
    print(f"  Gen Time:      {gen_time:.1f}s")

    return results, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--gpu_mem", type=float, default=0.85)
    parser.add_argument("--output_dir", type=str, default="train/eval_results")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Evaluate a single model (overrides default model list)")
    parser.add_argument("--model_name", type=str, default="model",
                        help="Name for the single model (used in output filenames)")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Optional LoRA adapter path. Loaded on top of --model_path via vLLM (no PEFT merge needed).")
    parser.add_argument("--lora_rank", type=int, default=128,
                        help="LoRA rank (must match training, default 128). Only used when --lora_path is set.")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Max output tokens. Default 1024; 512 may be faster if responses are short.")
    parser.add_argument("--triviaqa_path", type=str, default=TRIVIAQA_PATH,
                        help="Override dataset path (file must use TriviaQA schema: question/answer/aliases).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.triviaqa_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    import random
    random.seed(42)
    samples = random.sample(all_data, min(args.num_samples, len(all_data)))
    print(f"Loaded {len(samples)} TriviaQA samples (random seed=42)")

    all_results = []

    if args.model_path:
        models = [(args.model_path, args.model_name)]
    else:
        models = [
            (BASELINE_MODEL, "baseline"),
            (RL_MODEL, "corver"),
            (SFT_MODEL, "sft"),
            (BASE_MODEL, "base"),
        ]

    for model_path, name in models:
        if not os.path.exists(model_path) and "/" not in model_path and not model_path.startswith("deepseek"):
            print(f"Skipping {name}: {model_path} not found")
            continue
        results, details = generate_and_eval(
            model_path, name, samples, args.gpu_mem,
            lora_path=args.lora_path, lora_rank=args.lora_rank, max_tokens=args.max_tokens,
        )
        all_results.append(results)
        with open(os.path.join(args.output_dir, f"triviaqa_{name}_completions.json"), "w") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)

    # Comparison
    print(f"\n{'='*90}")
    print("  TriviaQA Comparison")
    print(f"{'='*90}")
    header = f"{'Metric':<25}" + "".join(f"{r['name']:<18}" for r in all_results)
    print(header)
    print("-" * len(header))
    for metric in ["correct_rate", "incorrect_rate", "not_attempted_rate", "f1", "format_accuracy"]:
        row = f"{metric:<25}"
        for r in all_results:
            val = r[metric]
            row += f"{val:<18.4f}" if metric == "f1" else f"{val:<18.1%}"
        print(row)

    with open(os.path.join(args.output_dir, "triviaqa_eval_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
