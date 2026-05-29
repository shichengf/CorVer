"""Re-grade existing triviaqa completions JSON files with current lenient regexes.

Reads triviaqa_*_completions.json from a directory, recomputes correct/incorrect/
not_attempted/format_accuracy using the same logic as eval_triviaqa.py, and writes
a new summary file alongside.

Usage:
    python train/regrade_completions.py --dir train/eval_results/llama32_3b
"""
import argparse
import json
import os
import re
import unicodedata
from typing import List


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def extract_answer(completion: str) -> str:
    text = completion if completion.startswith("<think>") else "<think>" + completion
    match = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def check_correct(predicted: str, gold: str, aliases: List[str]) -> str:
    if not predicted or predicted.lower().strip() in ["i don't know", "i do not know", ""]:
        return "NOT_ATTEMPTED"
    pred_norm = normalize(predicted)
    if not pred_norm:
        return "NOT_ATTEMPTED"
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


def regrade(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)

    correct = incorrect = not_attempted = 0
    format_correct = 0
    for r in data:
        comp = r["completion"]
        gold = r.get("gold_answer") or r.get("answer", "")
        aliases = r.get("aliases", [])
        predicted = extract_answer(comp)
        grade = check_correct(predicted, gold, aliases)
        if grade == "CORRECT": correct += 1
        elif grade == "INCORRECT": incorrect += 1
        else: not_attempted += 1
        text = comp if comp.startswith("<think>") else "<think>" + comp
        if re.search(r"<think>.*?</think>.*?<answer>", text, re.DOTALL):
            format_correct += 1
    total = len(data)
    f1 = (2 * correct) / (2 * correct + incorrect + not_attempted) if total else 0.0
    return {
        "name": os.path.basename(path).replace("triviaqa_", "").replace("_completions.json", ""),
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": not_attempted,
        "format_correct": format_correct,
        "correct_rate": correct / total,
        "incorrect_rate": incorrect / total,
        "not_attempted_rate": not_attempted / total,
        "format_accuracy": format_correct / total,
        "f1": f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory with triviaqa_*_completions.json")
    ap.add_argument("--out", default=None, help="Output summary path (default <dir>/triviaqa_eval_summary_regraded.json)")
    args = ap.parse_args()

    files = sorted([f for f in os.listdir(args.dir) if f.startswith("triviaqa_") and f.endswith("_completions.json")])
    print(f"Found {len(files)} completion files in {args.dir}")
    results = []
    for f in files:
        path = os.path.join(args.dir, f)
        r = regrade(path)
        results.append(r)
        print(f"  {r['name']:50s} | Correct {r['correct_rate']:6.1%} | Format {r['format_accuracy']:6.1%} | NA {r['not_attempted_rate']:6.1%}")

    out = args.out or os.path.join(args.dir, "triviaqa_eval_summary_regraded.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
