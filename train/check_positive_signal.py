"""Check positive signal density: for each training question,
generate N responses and see if at least 1 is correct (string match).

Usage:
    python train/check_positive_signal.py \
        --model <HF id or local path to base model> \
        --data data/rl/stepwise_curriculum_phase1.json \
        --num_gen 16 \
        --max_tokens 512
"""

import argparse
import json
import re
import unicodedata
from collections import Counter

from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "Answer this question using the following format strictly:\n\n"
    "1. First use <think> </think> tags for brief step-by-step reasoning.\n"
    "2. Then use <answer> </answer> tags for your final answer.\n\n"
    "Rules:\n"
    "- Keep your answer to 3-4 sentences maximum. Be concise and direct.\n"
    "- Each factual sentence must be self-contained.\n"
    "- Always use full entity names (e.g., \"Albert Einstein\", \"Berlin, Germany\"). "
    "Never use pronouns (he, she, it, they, this, that) to refer to key entities.\n"
    "- Do not repeat information or add unnecessary elaboration.\n"
    "- Do not include claims you are not confident about. "
    "If you genuinely do not know, reply exactly with: I don't know.\n"
    "- Avoid filler phrases like \"It is worth noting that\" or \"Interestingly\"."
)


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def check_correct(predicted: str, gold: str) -> bool:
    if not predicted or predicted.lower().strip() in [
        "i don't know", "i do not know", "i don\u2019t know", ""
    ]:
        return False
    pred_norm = normalize(predicted)
    if not pred_norm:
        return False
    for ans in gold.split(";"):
        ans_norm = normalize(ans.strip())
        if not ans_norm:
            continue
        if pred_norm == ans_norm or ans_norm in pred_norm or pred_norm in ans_norm:
            return True
    return False


def extract_answer(text: str) -> str:
    # Lenient: read to closing tag OR end-of-string so truncated completions still
    # extract their answer (avoids false 0/16 signals during filtering).
    m = re.search(r'<answer>(.*?)(?:</answer>|$)', text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="HF model id or local path to the base model used for self-filtering")
    parser.add_argument("--data", default="data/rl/stepwise_curriculum_phase1.json")
    parser.add_argument("--num_gen", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--sample", type=int, default=0, help="0=all")
    parser.add_argument("--out_dir", default="train/eval_results_stepwise",
                        help="Where to write the per-question and summary JSON files")
    parser.add_argument("--out_prefix", default="positive_signal",
                        help="Prefix for output files (use a model-specific name when running multiple models)")
    parser.add_argument("--filtered_out", default=None,
                        help="If set, write the filtered dataset to this JSON path. "
                             "By default keeps questions with 1<=n_correct<=num_gen-1 (excludes 0/N and N/N)."
                             " Override with --min_correct / --max_correct.")
    parser.add_argument("--min_correct", type=int, default=1,
                        help="Minimum n_correct (out of num_gen) for a question to be kept. Default 1 (drop fully-wrong).")
    parser.add_argument("--max_correct", type=int, default=None,
                        help="Maximum n_correct (out of num_gen) for a question to be kept. "
                             "Default = num_gen - 1 (drop fully-correct, which are reward-hack fuel).")
    args = parser.parse_args()

    # Load data
    with open(args.data) as f:
        data = json.load(f)
    if args.sample > 0:
        data = data[:args.sample]
    print(f"Questions: {len(data)}, num_gen: {args.num_gen}")

    # Load model
    llm = LLM(
        model=args.model,
        max_model_len=1024,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling = SamplingParams(
        n=args.num_gen,
        temperature=0.7,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    import os
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, f"{args.out_prefix}_checkpoint.jsonl")
    out_path = os.path.join(out_dir, f"{args.out_prefix}_check.json")

    # Resume from checkpoint if exists
    done_indices = set()
    per_question_results = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            for line in f:
                rec = json.loads(line)
                done_indices.add(rec["idx"])
                per_question_results[rec["idx"]] = rec["n_correct"]
        print(f"Resumed from checkpoint: {len(done_indices)} questions already done")

    # Build prompts for remaining questions
    todo = [(i, item) for i, item in enumerate(data) if i not in done_indices]
    print(f"Remaining: {len(todo)} questions to generate")

    if todo:
        prompts = []
        for i, item in todo:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        # Generate in batches to save intermediate results
        BATCH = 1000
        ckpt_f = open(checkpoint_path, "a")
        for batch_start in range(0, len(todo), BATCH):
            batch_end = min(batch_start + BATCH, len(todo))
            batch_prompts = prompts[batch_start:batch_end]
            batch_items = todo[batch_start:batch_end]

            print(f"Generating batch {batch_start//BATCH+1} "
                  f"({batch_start}-{batch_end}/{len(todo)})...")
            outputs = llm.generate(batch_prompts, sampling)

            for (idx, item), output in zip(batch_items, outputs):
                gold = item["answers"]
                n_correct = 0
                for completion in output.outputs:
                    pred = extract_answer(completion.text)
                    if check_correct(pred, gold):
                        n_correct += 1
                per_question_results[idx] = n_correct
                ckpt_f.write(json.dumps({"idx": idx, "n_correct": n_correct}) + "\n")
            ckpt_f.flush()
            print(f"  Checkpoint saved ({len(per_question_results)}/{len(data)} done)")

        ckpt_f.close()

    # Final stats
    total = len(data)
    correct_count_dist = Counter(per_question_results.values())
    at_least_1_correct = sum(1 for v in per_question_results.values() if v > 0)

    print(f"\n{'='*60}")
    print(f"Results: {total} questions × {args.num_gen} generations")
    print(f"{'='*60}")
    print(f"At least 1 correct: {at_least_1_correct}/{total} = {at_least_1_correct/total*100:.1f}%")
    print(f"All wrong (zero signal): {total - at_least_1_correct}/{total} = {(total-at_least_1_correct)/total*100:.1f}%")
    print(f"\nCorrect count distribution (out of {args.num_gen}):")
    for k in sorted(correct_count_dist.keys()):
        pct = correct_count_dist[k] / total * 100
        bar = "#" * int(pct)
        print(f"  {k:2d} correct: {correct_count_dist[k]:5d} ({pct:5.1f}%) {bar}")

    # Save final results
    results = {
        "total": total,
        "num_gen": args.num_gen,
        "at_least_1_correct": at_least_1_correct,
        "pct": round(at_least_1_correct / total * 100, 2),
        "distribution": {str(k): v for k, v in sorted(correct_count_dist.items())},
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Optionally save the filtered dataset (questions in [min_correct, max_correct] range)
    if args.filtered_out:
        os.makedirs(os.path.dirname(args.filtered_out) or ".", exist_ok=True)
        max_c = args.max_correct if args.max_correct is not None else (args.num_gen - 1)
        min_c = args.min_correct
        filtered = [
            data[i] for i in range(total)
            if min_c <= per_question_results.get(i, 0) <= max_c
        ]
        with open(args.filtered_out, "w") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(filtered)} filtered questions ({min_c}<=n_correct<={max_c}) to {args.filtered_out}")


if __name__ == "__main__":
    main()
