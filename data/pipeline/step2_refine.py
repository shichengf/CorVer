"""Step 2: local rule-based quality filter + local entity extraction.

Rule-based filtering replaces the per-sample API call from earlier iterations
(orders of magnitude faster); the only API-dependent step is Step 3 (CoT distillation).
"""
import json
import os
import re
import sys
import time
from config import INTERMEDIATE_DIR

# Make the QuCo extractor importable from the train/ tree.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "train"))


TEMPORAL_KEYWORDS = {
    "now", "current", "currently", "today", "latest", "recent", "recently",
    "this year", "this month", "this week", "right now", "at present",
    "as of", "upcoming", "next year", "last year",
}

VAGUE_PATTERNS = [
    r"^who is (he|she|they|it|that|this)\b",
    r"^what (happened|is happening|will happen)\b.*\b(him|her|them|it)\b",
    r"^(who|what|where|when|how) (will|is going to) .{0,20}$",
]

MIN_QUESTION_LENGTH = 15
MAX_QUESTION_LENGTH = 500


def rule_based_quality_filter(question: str, answers: list) -> tuple:
    """Return (accept, reason) for the rule-based quality check."""
    q = question.strip()
    q_lower = q.lower()

    if len(q) < MIN_QUESTION_LENGTH:
        return False, "too_short"
    if len(q) > MAX_QUESTION_LENGTH:
        return False, "too_long"

    if not any(q_lower.startswith(w) for w in
               ["who", "what", "where", "when", "why", "how", "which",
                "is ", "are ", "was ", "were ", "did ", "does ", "do ",
                "can ", "could ", "has ", "have ", "had ", "name "]):
        if "?" not in q:
            return False, "not_question"

    for keyword in TEMPORAL_KEYWORDS:
        if keyword in q_lower:
            return False, f"temporal:{keyword}"

    for pattern in VAGUE_PATTERNS:
        if re.search(pattern, q_lower):
            return False, "vague"

    if not answers:
        return False, "no_answer"
    best = answers[0] if isinstance(answers, list) else str(answers)
    if not best.strip():
        return False, "empty_answer"

    return True, "ok"


def load_jsonl(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    input_path = os.path.join(INTERMEDIATE_DIR, "step1_deduped.jsonl")
    output_path = os.path.join(INTERMEDIATE_DIR, "step2_refined.jsonl")

    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} records from Step 1")

    print("\n--- Phase 1: Rule-based quality filter ---")
    accepted = []
    reject_reasons = {}

    for r in records:
        ok, reason = rule_based_quality_filter(r["question"], r["answers"])
        if ok:
            accepted.append(r)
        else:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    print(f"  Accepted: {len(accepted)} / {len(records)}")
    print(f"  Reject reasons:")
    for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")

    print("\n--- Phase 2: QuCo entity extraction ---")
    start = time.time()

    try:
        from corver_reward.entity_extractor import QuCoEntityExtractor
        extractor = QuCoEntityExtractor()

        questions = [r["question"] for r in accepted]

        batch_size = 64
        all_entities = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            entities_batch = extractor.extract_question_entities_batch(batch, batch_size=batch_size)
            all_entities.extend(entities_batch)
            if (i + batch_size) % (batch_size * 10) == 0:
                print(f"  Progress: {min(i + batch_size, len(questions))}/{len(questions)}")

        elapsed = time.time() - start
        print(f"  Entity extraction completed in {elapsed:.1f}s")

    except Exception as e:
        print(f"  WARNING: QuCo extractor failed: {e}")
        print(f"  Falling back to simple noun extraction...")
        all_entities = [simple_entity_extract(r["question"]) for r in accepted]

    # Drop records with no extracted entities.
    output_records = []
    no_entity_count = 0
    for r, entities in zip(accepted, all_entities):
        if entities and len(entities) > 0:
            output_records.append({
                "id": r["id"],
                "source": r["source"],
                "question": r["question"],
                "answers": r["answers"],
                "best_answer": r["best_answer"],
                "entities": entities,
            })
        else:
            no_entity_count += 1

    save_jsonl(output_records, output_path)

    print(f"\n=== Step 2 Summary ===")
    print(f"Input:              {len(records)}")
    print(f"After quality filter: {len(accepted)}")
    print(f"After entity filter:  {len(output_records)} (no entities: {no_entity_count})")
    print(f"Output: {output_path}")


def simple_entity_extract(question: str) -> list:
    """Fallback entity extractor: collect capitalized word runs."""
    words = question.split()
    entities = []
    current_entity = []

    for word in words:
        if not entities and word.lower() in {"who", "what", "where", "when", "which", "how", "is", "are", "was", "were", "did", "does", "do", "the", "a", "an", "in", "of", "for", "by", "from", "to", "on", "at", "with"}:
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
            continue

        clean = word.strip("?.,!;:'\"")
        if clean and clean[0].isupper():
            current_entity.append(clean)
        else:
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []

    if current_entity:
        entities.append(" ".join(current_entity))

    seen = set()
    unique = []
    for e in entities:
        if e.lower() not in seen and len(e) > 1:
            seen.add(e.lower())
            unique.append(e)

    return unique[:5]


if __name__ == "__main__":
    main()
