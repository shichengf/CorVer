"""Step 1: dedup NqOpen (exact + semantic) and cross-dedup against WebQuestions."""
import json
import os
import re
from collections import defaultdict
from config import RAW_DIR, INTERMEDIATE_DIR, SEMHASH_THRESHOLD


def normalize_question(q: str) -> str:
    """Normalize question text for exact dedup."""
    q = q.strip().lower()
    q = re.sub(r'\s+', ' ', q)
    q = q.rstrip('?').rstrip('.').rstrip('!').strip()
    return q


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


def exact_dedup(records: list) -> list:
    """Exact dedup based on normalized question text."""
    seen = set()
    deduped = []
    for r in records:
        key = normalize_question(r["question"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def semantic_dedup_semhash(records: list, threshold: float = 0.9) -> list:
    """Semantic dedup using semhash + potion-base-8M."""
    try:
        from semhash import SemHash
    except ImportError:
        print("WARNING: semhash not installed, skipping semantic dedup")
        print("  Install with: pip install semhash model2vec")
        return records

    print(f"  Running semantic dedup with threshold={threshold}...")

    record_dicts = [{"question": r["question"], "idx": i} for i, r in enumerate(records)]

    semhasher = SemHash.from_records(
        records=record_dicts,
        columns=["question"],
    )

    result = semhasher.self_deduplicate(threshold=threshold)
    deduplicated_indices = set()
    for item in result.selected:
        deduplicated_indices.add(item["idx"])

    deduped = [records[i] for i in sorted(deduplicated_indices)]
    print(f"  Semantic dedup: {len(records)} -> {len(deduped)} (removed {len(records) - len(deduped)}, ratio={result.duplicate_ratio:.4f})")
    return deduped


def cross_dedup(base_records: list, new_records: list, threshold: float = 0.9) -> list:
    """Remove entries in new_records that are duplicates of anything in base_records."""
    try:
        from semhash import SemHash
    except ImportError:
        print("WARNING: semhash not installed, falling back to exact dedup only")
        base_keys = {normalize_question(r["question"]) for r in base_records}
        return [r for r in new_records if normalize_question(r["question"]) not in base_keys]

    print(f"  Cross-dedup: {len(new_records)} new records against {len(base_records)} base records...")

    base_dicts = [{"question": r["question"], "idx": i} for i, r in enumerate(base_records)]

    semhasher = SemHash.from_records(
        records=base_dicts,
        columns=["question"],
    )

    new_dicts = [{"question": r["question"], "idx": i} for i, r in enumerate(new_records)]
    result = semhasher.deduplicate(records=new_dicts, threshold=threshold)

    unique_indices = set()
    for item in result.selected:
        unique_indices.add(item["idx"])

    unique = [new_records[i] for i in sorted(unique_indices)]
    print(f"  Cross-dedup: {len(new_records)} -> {len(unique)} (removed {len(new_records) - len(unique)})")
    return unique


def main():
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

    nq_path = os.path.join(RAW_DIR, "nqopen_train.jsonl")
    wq_path = os.path.join(RAW_DIR, "webquestions_all.jsonl")

    nq_records = load_jsonl(nq_path)
    wq_records = load_jsonl(wq_path)
    print(f"Loaded NqOpen: {len(nq_records)}, WebQuestions: {len(wq_records)}")

    print("\n--- NqOpen Exact Dedup ---")
    nq_exact = exact_dedup(nq_records)
    print(f"  NqOpen exact dedup: {len(nq_records)} -> {len(nq_exact)}")

    print("\n--- NqOpen Semantic Dedup ---")
    nq_deduped = semantic_dedup_semhash(nq_exact, threshold=SEMHASH_THRESHOLD)

    print("\n--- WebQuestions Exact Dedup ---")
    wq_exact = exact_dedup(wq_records)
    print(f"  WebQuestions exact dedup: {len(wq_records)} -> {len(wq_exact)}")

    # Cross-source dedup: NqOpen is the base; drop matches from WebQuestions.
    print("\n--- Cross-Source Dedup ---")
    wq_unique = cross_dedup(nq_deduped, wq_exact, threshold=SEMHASH_THRESHOLD)

    merged = nq_deduped + wq_unique

    for i, r in enumerate(merged):
        r["id"] = f"dedup_{i}"

    output_path = os.path.join(INTERMEDIATE_DIR, "step1_deduped.jsonl")
    save_jsonl(merged, output_path)

    print(f"\n=== Step 1 Summary ===")
    print(f"NqOpen original:    {len(nq_records)}")
    print(f"NqOpen after dedup: {len(nq_deduped)}")
    print(f"WebQuestions original: {len(wq_records)}")
    print(f"WebQuestions unique:   {len(wq_unique)}")
    print(f"Merged total:       {len(merged)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
