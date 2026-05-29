"""Step 4: Wikipedia entity grounding.

Links each QA record's entities to Wikipedia article titles using an inverted
index, and drops records whose entities cannot all be matched.
"""
import json
import os
import re
import time
from collections import defaultdict
from config import INTERMEDIATE_DIR, WIKIPEDIA_JSONL_DIR, MAX_WIKI_LINKS_PER_ENTITY


def build_title_index(jsonl_dir: str) -> dict:
    """Build a title index from the Wikipedia JSONL dump.

    Returns:
        title_lower_to_info: {title_lower: {"title": original_title, "file": jsonl_path, "line": line_no}}
    """
    print(f"Building Wikipedia title index from {jsonl_dir}...")
    start = time.time()

    title_index = {}
    files = sorted([f for f in os.listdir(jsonl_dir) if f.endswith('.jsonl')])
    total_articles = 0

    for fname in files:
        fpath = os.path.join(jsonl_dir, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f):
                try:
                    data = json.loads(line)
                    title = data.get("title", "")
                    if title:
                        title_lower = title.lower().strip()
                        if title_lower not in title_index:
                            title_index[title_lower] = {
                                "title": title,
                                "file": fpath,
                                "line": line_no,
                            }
                        total_articles += 1
                except json.JSONDecodeError:
                    continue

    elapsed = time.time() - start
    print(f"  Indexed {len(title_index)} unique titles from {total_articles} articles in {elapsed:.1f}s")
    return title_index


def build_word_index(title_index: dict) -> dict:
    """Build a word-level inverted index for fast containment matching."""
    print("Building word inverted index...")
    start = time.time()

    word_to_titles = defaultdict(set)
    for title_lower in title_index:
        words = title_lower.split()
        for word in words:
            if len(word) > 2:
                word_to_titles[word].add(title_lower)

    elapsed = time.time() - start
    print(f"  Built inverted index with {len(word_to_titles)} words in {elapsed:.1f}s")
    return word_to_titles


def match_entity_to_wiki(entity: str, title_index: dict, word_index: dict,
                         max_matches: int = MAX_WIKI_LINKS_PER_ENTITY) -> list:
    """Match an entity to Wikipedia titles via exact match then containment.

    Returns:
        list of matched original titles
    """
    entity_lower = entity.lower().strip()
    matches = []

    if entity_lower in title_index:
        matches.append(title_index[entity_lower]["title"])

    if len(matches) < max_matches:
        entity_words = [w for w in entity_lower.split() if len(w) > 2]
        if entity_words:
            candidates = word_index.get(entity_words[0], set()).copy()
            for word in entity_words[1:]:
                candidates &= word_index.get(word, set())

            for candidate_lower in candidates:
                if candidate_lower == entity_lower:
                    continue
                if entity_lower in candidate_lower or candidate_lower in entity_lower:
                    matches.append(title_index[candidate_lower]["title"])
                    if len(matches) >= max_matches:
                        break

    return matches[:max_matches]


def get_article_text(title: str, title_index: dict) -> str:
    """Return the full Wikipedia article text for a title."""
    title_lower = title.lower().strip()
    if title_lower not in title_index:
        return ""

    info = title_index[title_lower]
    try:
        with open(info["file"], 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == info["line"]:
                    data = json.loads(line)
                    return data.get("text", "")
    except (json.JSONDecodeError, IOError):
        pass
    return ""


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
    # Prefer the three-way difficulty-classified output; fall back to the older CoT-filtered file.
    classified_path = os.path.join(INTERMEDIATE_DIR, "step3_classified.jsonl")
    legacy_path = os.path.join(INTERMEDIATE_DIR, "step3_cot_filtered.jsonl")
    if os.path.exists(classified_path):
        input_path = classified_path
    else:
        input_path = legacy_path
    output_path = os.path.join(INTERMEDIATE_DIR, "step4_wiki_grounded.jsonl")
    output_with_text_path = os.path.join(INTERMEDIATE_DIR, "step4_wiki_grounded_with_text.jsonl")

    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found. Run step3 first.")
        return

    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} records from Step 3")

    title_index = build_title_index(WIKIPEDIA_JSONL_DIR)
    word_index = build_word_index(title_index)

    print("\n--- Matching entities to Wikipedia ---")
    grounded = []
    grounded_with_text = []
    no_entities = 0
    no_match = 0

    for i, r in enumerate(records):
        entities = r.get("entities", [])
        if not entities:
            no_entities += 1
            continue

        all_matched_titles = []
        all_matched = True

        for entity in entities:
            titles = match_entity_to_wiki(entity, title_index, word_index)
            if titles:
                all_matched_titles.extend(titles)
            else:
                # Drop the record if any entity fails to match.
                all_matched = False
                break

        if not all_matched or not all_matched_titles:
            no_match += 1
            continue

        seen = set()
        unique_titles = []
        for t in all_matched_titles:
            if t not in seen:
                seen.add(t)
                unique_titles.append(t)

        title_str = ";".join(unique_titles)

        grounded_record = {
            "question": r["question"],
            "answers": r.get("best_answer", ""),
            "title": title_str,
        }
        if "difficulty" in r:
            grounded_record["difficulty"] = r["difficulty"]
        grounded.append(grounded_record)

        texts = []
        for title in unique_titles:
            text = get_article_text(title, title_index)
            if text:
                texts.append(text)

        knowledge_text = "\n\n".join(texts)
        grounded_with_text_record = {
            **grounded_record,
            "text": knowledge_text,
            "cot_response": r.get("cot_response", ""),
            "entities": r.get("entities", []),
        }
        grounded_with_text.append(grounded_with_text_record)

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{len(records)}, grounded: {len(grounded)}")

    save_jsonl(grounded, output_path)
    save_jsonl(grounded_with_text, output_with_text_path)

    print(f"\n=== Step 4 Summary ===")
    print(f"Input records:     {len(records)}")
    print(f"No entities:       {no_entities}")
    print(f"No wiki match:     {no_match}")
    print(f"Grounded:          {len(grounded)}")
    print(f"Output:            {output_path}")
    print(f"Output with text:  {output_with_text_path}")


if __name__ == "__main__":
    main()
