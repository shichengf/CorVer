"""Re-generate ONLY truncated rows in an existing TriviaQA completions JSON.

Loads the existing completions file, finds rows where '</answer>' is missing,
re-runs vLLM generation on JUST those questions with a larger max_tokens,
re-grades them, and writes a new completions JSON + summary.

Usage:
  python train/regen_truncated.py \\
      --in_json train/eval_results/<dir>/triviaqa_<name>_completions.json \\
      --out_json <new path> \\
      --model_path meta-llama/Llama-3.1-8B-Instruct \\
      --lora_path train/output/.../checkpoint-50 --lora_rank 128 \\
      --max_tokens 4096
"""
import argparse
import json
import os
import re
import unicodedata
from typing import List, Dict


SYSTEM_PROMPT = (
    "Answer using this exact format:\n"
    "<think>brief reasoning</think>\n"
    "<answer>direct answer</answer>\n\n"
    "Do not loop or repeat the same point or phrase.\n"
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


def check_correct(predicted: str, gold: str, aliases) -> str:
    if not predicted or predicted.lower().strip() in ["i don't know", "i do not know", ""]:
        return "NOT_ATTEMPTED"
    pred_norm = normalize(predicted)
    if not pred_norm:
        return "NOT_ATTEMPTED"
    if isinstance(aliases, str):
        try:
            aliases = eval(aliases) if aliases.startswith('[') else [aliases]
        except Exception:
            aliases = []
    all_answers = [gold] + (aliases or [])
    for ans in all_answers:
        ans_norm = normalize(ans)
        if not ans_norm:
            continue
        if pred_norm == ans_norm or ans_norm in pred_norm or pred_norm in ans_norm:
            return "CORRECT"
    return "INCORRECT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--lora_rank", type=int, default=128)
    ap.add_argument("--max_tokens", type=int, default=8000)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    args = ap.parse_args()

    data = json.load(open(args.in_json))
    print(f"loaded {len(data)} records from {args.in_json}")

    # Only regen truncated rows that are NOT already CORRECT —
    # if a truncated answer was already graded CORRECT, the partial output is fine.
    trunc_indices = [
        i for i, r in enumerate(data)
        if "</answer>" not in r.get("completion", "") and r.get("grade") != "CORRECT"
    ]
    print(f"truncated AND not-CORRECT rows to regen: {len(trunc_indices)}")
    if not trunc_indices:
        print("nothing to do; copying input to output")
        json.dump(data, open(args.out_json, "w"), ensure_ascii=False, indent=2)
        return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm_kwargs = dict(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_mem,
        dtype="bfloat16",
        max_model_len=args.max_tokens + 512,
        trust_remote_code=True,
    )
    if args.lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.lora_rank
        llm_kwargs["max_loras"] = 1

    llm = LLM(**llm_kwargs)

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    prompts = []
    for i in trunc_indices:
        r = data[i]
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["question"]},
        ]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    gen_kwargs = {}
    if args.lora_path:
        from vllm.lora.request import LoRARequest
        gen_kwargs["lora_request"] = LoRARequest("adapter", 1, args.lora_path)

    outputs = llm.generate(prompts, sp, **gen_kwargs)

    fixed = 0
    closed_now = 0
    flipped_to_correct = 0
    for k, idx in enumerate(trunc_indices):
        r = data[idx]
        new_comp = outputs[k].outputs[0].text
        if "</answer>" in new_comp:
            closed_now += 1
        new_pred = extract_answer(new_comp)
        new_grade = check_correct(new_pred, r["gold_answer"], r.get("aliases", []))
        if new_grade != r["grade"]:
            fixed += 1
            if new_grade == "CORRECT":
                flipped_to_correct += 1
        r["completion"] = new_comp
        r["predicted"] = new_pred
        r["grade"] = new_grade

    json.dump(data, open(args.out_json, "w"), ensure_ascii=False, indent=2)
    print(f"saved -> {args.out_json}")
    print(f"regen rows: {len(trunc_indices)}")
    print(f"  closed </answer> now: {closed_now} ({closed_now/len(trunc_indices)*100:.1f}%)")
    print(f"  grade changed: {fixed}")
    print(f"  flipped to CORRECT: {flipped_to_correct}")

    from collections import Counter
    c = Counter(r["grade"] for r in data)
    n = len(data)
    print()
    print(f"NEW totals (N={n}):")
    print(f"  CORRECT       = {c['CORRECT']} ({c['CORRECT']/n*100:.2f}%)")
    print(f"  INCORRECT     = {c['INCORRECT']} ({c['INCORRECT']/n*100:.2f}%)")
    print(f"  NOT_ATTEMPTED = {c['NOT_ATTEMPTED']} ({c['NOT_ATTEMPTED']/n*100:.2f}%)")


if __name__ == "__main__":
    main()
