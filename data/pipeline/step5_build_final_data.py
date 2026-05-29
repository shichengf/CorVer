"""Step 5: build the final RL training pool from the wiki-grounded records.

Splits records by difficulty (if available) into easy / medium / hard RL files,
and an optional `*_withknowledge` companion that carries the matched Wikipedia
text alongside each record. When the input has no difficulty field, falls back
to a single-file output.
"""
import json
import os

from config import INTERMEDIATE_DIR, FINAL_DIR


def load_jsonl(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_rl_data(records: list) -> list:
    return [
        {"question": r["question"], "answers": r["answers"], "title": r["title"]}
        for r in records
    ]


def build_rl_with_knowledge(records_with_text: list) -> list:
    return [
        {
            "question": r["question"],
            "answers": r["answers"],
            "title": r["title"],
            "text": r.get("text", ""),
        }
        for r in records_with_text
    ]


def main():
    os.makedirs(FINAL_DIR, exist_ok=True)

    grounded_path = os.path.join(INTERMEDIATE_DIR, "step4_wiki_grounded.jsonl")
    grounded_with_text_path = os.path.join(INTERMEDIATE_DIR, "step4_wiki_grounded_with_text.jsonl")

    if not os.path.exists(grounded_path):
        print(f"ERROR: {grounded_path} not found. Run step4 first.")
        return

    grounded = load_jsonl(grounded_path)
    grounded_with_text = load_jsonl(grounded_with_text_path) if os.path.exists(grounded_with_text_path) else []

    print(f"Loaded {len(grounded)} grounded records")
    print(f"Loaded {len(grounded_with_text)} grounded records with text")

    has_difficulty = any("difficulty" in r for r in grounded)
    if has_difficulty:
        print("\nDifficulty field detected; emitting per-difficulty files.")
        _build_with_difficulty(grounded, grounded_with_text)
    else:
        print("\nNo difficulty field detected; emitting single-file outputs.")
        _build_single(grounded, grounded_with_text)


def _build_with_difficulty(grounded, grounded_with_text):
    difficulties = ["easy", "medium", "hard"]

    by_diff = {d: [] for d in difficulties}
    by_diff_text = {d: [] for d in difficulties}
    for r in grounded:
        d = r.get("difficulty", "unknown")
        if d in by_diff:
            by_diff[d].append(r)
    for r in grounded_with_text:
        d = r.get("difficulty", "unknown")
        if d in by_diff_text:
            by_diff_text[d].append(r)

    for d in difficulties:
        print(f"  {d}: {len(by_diff[d])} grounded, {len(by_diff_text[d])} with text")

    print("\n--- RL data (by difficulty) ---")
    for d in difficulties:
        rl_data = build_rl_data(by_diff[d])
        rl_path = os.path.join(FINAL_DIR, f"{d}_rl.json")
        with open(rl_path, "w", encoding="utf-8") as f:
            json.dump(rl_data, f, ensure_ascii=False, indent=2)
        print(f"  {d}_rl.json: {len(rl_data)} samples")

    print("\n--- RL + Knowledge data (by difficulty) ---")
    for d in difficulties:
        rl_k = build_rl_with_knowledge(by_diff_text[d])
        rl_k_path = os.path.join(FINAL_DIR, f"{d}_rl_withknowledge.json")
        with open(rl_k_path, "w", encoding="utf-8") as f:
            json.dump(rl_k, f, ensure_ascii=False, indent=2)
        print(f"  {d}_rl_withknowledge.json: {len(rl_k)} samples")

    total_rl = sum(len(by_diff[d]) for d in difficulties)
    print(f"\n=== Step 5 Summary (per difficulty) ===")
    for d in difficulties:
        print(f"  {d} RL: {len(by_diff[d])}")
    print(f"  Total RL: {total_rl}")


def _build_single(grounded, grounded_with_text):
    print("\n--- RL data ---")
    rl_data = build_rl_data(grounded)
    rl_path = os.path.join(FINAL_DIR, "rl_pool.json")
    with open(rl_path, "w", encoding="utf-8") as f:
        json.dump(rl_data, f, ensure_ascii=False, indent=2)
    print(f"  RL data: {len(rl_data)} samples -> {rl_path}")

    print("\n--- RL + Knowledge data ---")
    rl_knowledge = build_rl_with_knowledge(grounded_with_text)
    rl_knowledge_path = os.path.join(FINAL_DIR, "rl_pool_withknowledge.json")
    with open(rl_knowledge_path, "w", encoding="utf-8") as f:
        json.dump(rl_knowledge, f, ensure_ascii=False, indent=2)
    print(f"  RL+Knowledge data: {len(rl_knowledge)} samples -> {rl_knowledge_path}")

    print(f"\n=== Step 5 Summary ===")
    print(f"RL Data:        {len(rl_data)} samples")
    print(f"RL+Knowledge:   {len(rl_knowledge)} samples")


if __name__ == "__main__":
    main()
