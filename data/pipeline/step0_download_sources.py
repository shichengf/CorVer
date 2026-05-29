"""Step 0: download raw datasets (NqOpen + WebQuestions)."""
import json
import os
from datasets import load_dataset
from config import RAW_DIR


def download_nqopen(output_path: str):
    """Download the NqOpen training split."""
    print("Downloading NqOpen (google-research-datasets/nq_open)...")
    ds = load_dataset("google-research-datasets/nq_open", split="train")
    print(f"  NqOpen train: {len(ds)} samples")

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(ds):
            answers = item["answer"]
            record = {
                "id": f"nqopen_{i}",
                "source": "nqopen",
                "question": item["question"],
                "answers": answers,
                "best_answer": answers[0] if answers else "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} records to {output_path}")
    return count


def download_webquestions(output_path: str):
    """Download both train and test splits of WebQuestions."""
    print("Downloading WebQuestions (stanfordnlp/web_questions)...")
    ds_train = load_dataset("stanfordnlp/web_questions", split="train")
    ds_test = load_dataset("stanfordnlp/web_questions", split="test")
    print(f"  WebQuestions train: {len(ds_train)}, test: {len(ds_test)}")

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for split_name, ds in [("train", ds_train), ("test", ds_test)]:
            for i, item in enumerate(ds):
                answers = item["answers"]
                record = {
                    "id": f"webq_{split_name}_{i}",
                    "source": "webquestions",
                    "question": item["question"],
                    "answers": answers,
                    "best_answer": answers[0] if answers else "",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    print(f"  Saved {count} records to {output_path}")
    return count


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    nq_path = os.path.join(RAW_DIR, "nqopen_train.jsonl")
    wq_path = os.path.join(RAW_DIR, "webquestions_all.jsonl")

    nq_count = download_nqopen(nq_path)
    wq_count = download_webquestions(wq_path)

    merged_path = os.path.join(RAW_DIR, "merged_raw.jsonl")
    total = 0
    with open(merged_path, "w", encoding="utf-8") as out:
        for path in [nq_path, wq_path]:
            with open(path, "r", encoding="utf-8") as inp:
                for line in inp:
                    out.write(line)
                    total += 1

    print(f"\n=== Summary ===")
    print(f"NqOpen:       {nq_count}")
    print(f"WebQuestions: {wq_count}")
    print(f"Merged total: {total}")
    print(f"Merged file:  {merged_path}")


if __name__ == "__main__":
    main()
