"""Phase 3 supplement: retry the primary LLM and try additional models to expand the Hard set.

For items the secondary LLM got wrong, retry the primary LLM on previous failures and
optionally route them through several alternative models. Any model getting the answer
right promotes the item to Hard; if all fail it is discarded. The run is resumable.

Usage:
    source ~/.bashrc && conda activate verl
    cd data/pipeline
    python step3_multimodel_rerun.py
"""
import asyncio
import json
import os
import re
import string
import time
import unicodedata

from config import (
    INTERMEDIATE_DIR, DATA_LLM_API_KEY, DATA_LLM_BASE_URL, DATA_LLM_MODEL,
    API_MAX_CONCURRENT, API_RETRY_MAX, API_RETRY_DELAY,
)

MODELS = {
    "kimi": {"model": "kimi", "thinking_off": {"chat_template_kwargs": {"thinking": False}}},
    "glm": {"model": "glm-4.7", "thinking_off": {"chat_template_kwargs": {"enable_thinking": False}}},
    "minimax": {"model": "minimax-m2", "thinking_off": None},
}

PHASE2_PATH = os.path.join(INTERMEDIATE_DIR, "step3_phase2_gptoss.jsonl")
PHASE3_OLD_PATH = os.path.join(INTERMEDIATE_DIR, "step3_phase3_primary.jsonl")
MULTIMODEL_DIR = os.path.join(INTERMEDIATE_DIR, "step3_multimodel")
# Overwrites the original primary-LLM phase 3 file.
OUTPUT_PATH = os.path.join(INTERMEDIATE_DIR, "step3_phase3_primary.jsonl")
CLASSIFIED_PATH = os.path.join(INTERMEDIATE_DIR, "step3_classified.jsonl")

SIMPLE_QA_PROMPT = (
    "Answer the following question concisely with just the answer, "
    "no explanation needed.\n\nQuestion: {question}\nAnswer:"
)

COT_SYSTEM_PROMPT = """Answer this question using the following format strictly:

1. First use <think> </think> tags to show your step-by-step reasoning process.
2. Then use <answer> </answer> tags for your final answer. Use your factual knowledge to iteratively verify your reasoning and derive the answer. If you genuinely do not know the answer, you can reply exactly with: I don't know."""

JUDGE_PROMPT = """You are a factual answer judge. Given a question, the gold answer, and a model's predicted answer, determine if the predicted answer is CORRECT.

The predicted answer is considered CORRECT if it conveys the same factual information as the gold answer, even if worded differently. Minor differences in formatting, spelling, or phrasing are acceptable.

Output EXACTLY one word: CORRECT or INCORRECT

Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {predicted_answer}"""


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    for article in ["a ", "an ", "the "]:
        if text.startswith(article):
            text = text[len(article):]
    text = " ".join(text.split())
    return text


def check_correct(response: str, gold_answers) -> bool:
    resp_norm = normalize_answer(response)
    if not resp_norm or resp_norm in ["i dont know", "i dont know", "unknown"]:
        return False
    if isinstance(gold_answers, str):
        answers = [gold_answers]
    else:
        answers = gold_answers
    for ans in answers:
        ans_norm = normalize_answer(ans)
        if not ans_norm:
            continue
        if ans_norm == resp_norm or ans_norm in resp_norm or resp_norm in ans_norm:
            return True
    return False


def extract_answer_from_response(response: str) -> str:
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'</think>\s*(.*)', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(records, path):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


async def _api_call(session, semaphore, record, model_name, model_id,
                    thinking_off=None, system_prompt=None, temperature=0, timeout=120):
    import aiohttp

    headers = {
        "Authorization": f"Bearer {DATA_LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record["question"]},
        ]
    else:
        messages = [
            {"role": "user", "content": SIMPLE_QA_PROMPT.format(question=record["question"])},
        ]

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
    }
    if thinking_off:
        payload["extra_body"] = thinking_off

    for attempt in range(API_RETRY_MAX):
        async with semaphore:
            try:
                async with session.post(
                    f"{DATA_LLM_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        try:
                            content = data["choices"][0]["message"]["content"]
                        except (KeyError, IndexError):
                            if attempt < API_RETRY_MAX - 1:
                                await asyncio.sleep(API_RETRY_DELAY)
                                continue
                            return None
                        return content
                    elif resp.status == 429:
                        await asyncio.sleep(API_RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        if attempt < API_RETRY_MAX - 1:
                            await asyncio.sleep(API_RETRY_DELAY)
                            continue
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt < API_RETRY_MAX - 1:
                    await asyncio.sleep(API_RETRY_DELAY * (attempt + 1))
                    continue
                return None
    return None


async def _judge_call(session, semaphore, question, gold_answer, predicted_answer):
    import aiohttp

    prompt = JUDGE_PROMPT.format(
        question=question, gold_answer=gold_answer, predicted_answer=predicted_answer,
    )
    headers = {
        "Authorization": f"Bearer {DATA_LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DATA_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    for attempt in range(2):
        async with semaphore:
            try:
                async with session.post(
                    f"{DATA_LLM_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        try:
                            content = data["choices"][0]["message"]["content"]
                        except (KeyError, IndexError):
                            if attempt < 1:
                                await asyncio.sleep(API_RETRY_DELAY)
                                continue
                            return False
                        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                        return "CORRECT" in content.upper()
                    elif resp.status == 429:
                        await asyncio.sleep(API_RETRY_DELAY * (attempt + 1))
                        continue
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt < 1:
                    await asyncio.sleep(API_RETRY_DELAY)
                    continue
    return False


async def run_model_batch(records, model_name, model_id, thinking_off=None,
                          system_prompt=None, temperature=0, timeout=120):
    """Call one model on a batch of records and return {id: response}."""
    import aiohttp

    semaphore = asyncio.Semaphore(API_MAX_CONCURRENT)
    results = {}

    async with aiohttp.ClientSession() as session:
        tasks = []
        for r in records:
            tasks.append((r["id"], _api_call(
                session, semaphore, r, model_name, model_id,
                thinking_off=thinking_off, system_prompt=system_prompt,
                temperature=temperature, timeout=timeout,
            )))

        try:
            from tqdm.asyncio import tqdm_asyncio
            coros = [t[1] for t in tasks]
            responses = await tqdm_asyncio.gather(*coros, desc=f"{model_name}")
        except ImportError:
            coros = [t[1] for t in tasks]
            responses = await asyncio.gather(*coros)

        for (rid, _), response in zip(tasks, responses):
            results[rid] = response

    return results


async def run_judge_batch(items_to_judge):
    import aiohttp

    semaphore = asyncio.Semaphore(API_MAX_CONCURRENT)
    async with aiohttp.ClientSession() as session:
        tasks = [
            _judge_call(session, semaphore, item["question"], item["gold"], item["predicted"])
            for item in items_to_judge
        ]
        try:
            from tqdm.asyncio import tqdm_asyncio
            results = await tqdm_asyncio.gather(*tasks, desc="LLM Judge")
        except ImportError:
            results = await asyncio.gather(*tasks)
    return results


def check_model_response(response, gold_answers, is_cot=False):
    """Return (is_correct_local, predicted_text) for a single model response."""
    if not response:
        return False, ""

    if is_cot:
        predicted = extract_answer_from_response(response)
    else:
        predicted = response.strip()

    is_correct = check_correct(predicted, gold_answers)
    return is_correct, predicted


def main():
    print("=" * 60)
    print("Phase 3 supplement: primary LLM retry + alternative models")
    print("=" * 60)

    os.makedirs(MULTIMODEL_DIR, exist_ok=True)

    phase2_data = load_jsonl(PHASE2_PATH)
    hard_candidates = [r for r in phase2_data if not r.get("gptoss_correct")]
    print(f"Loaded {len(hard_candidates)} secondary-LLM failures")

    old_primary = {}
    if os.path.exists(PHASE3_OLD_PATH):
        for r in load_jsonl(PHASE3_OLD_PATH):
            old_primary[r["id"]] = r
    print(f"Existing primary-LLM results: {len(old_primary)}")
    primary_correct_ids = {rid for rid, r in old_primary.items() if r.get("primary_correct")}
    primary_failed_ids = {rid for rid, r in old_primary.items()
                        if r.get("error") or not r.get("cot_response")}
    print(f"  Correct: {len(primary_correct_ids)}")
    print(f"  Need retry due to API failure: {len(primary_failed_ids)}")

    # Step A: retry primary LLM on previous API failures (temperature=0).
    primary_retry_path = os.path.join(MULTIMODEL_DIR, "primary_retry.jsonl")
    primary_retry_ids = set()
    if os.path.exists(primary_retry_path):
        existing = load_jsonl(primary_retry_path)
        primary_retry_ids = {r["id"] for r in existing}

    to_retry = [r for r in hard_candidates
                if r["id"] in primary_failed_ids and r["id"] not in primary_retry_ids]

    if to_retry:
        print(f"\n--- Primary LLM retry on {len(to_retry)} failed items (temperature=0) ---")
        batch_size = 200
        for i in range(0, len(to_retry), batch_size):
            batch = to_retry[i:i + batch_size]
            print(f"  Batch {i // batch_size + 1}/{(len(to_retry) + batch_size - 1) // batch_size}")
            responses = asyncio.run(run_model_batch(
                batch, "primary-retry", DATA_LLM_MODEL,
                system_prompt=COT_SYSTEM_PROMPT, temperature=0, timeout=180,
            ))
            results = []
            for r in batch:
                resp = responses.get(r["id"])
                results.append({"id": r["id"], "response": resp})
            append_jsonl(results, primary_retry_path)

    # Step B: alternative models (skipped by default; flip SKIP_OTHER_MODELS to enable).
    SKIP_OTHER_MODELS = True
    for name, cfg in MODELS.items():
        if SKIP_OTHER_MODELS:
            print(f"\n--- Skipping {name} (SKIP_OTHER_MODELS=True) ---")
            continue
        model_path = os.path.join(MULTIMODEL_DIR, f"{name}.jsonl")
        existing_ids = set()
        if os.path.exists(model_path):
            existing_ids = {r["id"] for r in load_jsonl(model_path)}

        to_run = [r for r in hard_candidates if r["id"] not in existing_ids]

        if to_run:
            print(f"\n--- {name} ({cfg['model']}) simple QA ({len(to_run)} items) ---")
            batch_size = 200
            for i in range(0, len(to_run), batch_size):
                batch = to_run[i:i + batch_size]
                print(f"  Batch {i // batch_size + 1}/{(len(to_run) + batch_size - 1) // batch_size}")
                responses = asyncio.run(run_model_batch(
                    batch, name, cfg["model"],
                    thinking_off=cfg["thinking_off"], temperature=0, timeout=120,
                ))
                results = []
                for r in batch:
                    resp = responses.get(r["id"])
                    results.append({"id": r["id"], "response": resp})
                append_jsonl(results, model_path)

    # Step D: aggregate -- any model correct -> hard.
    print("\n" + "=" * 60)
    print("Aggregating multi-model results")
    print("=" * 60)

    id_to_record = {r["id"]: r for r in hard_candidates}

    all_responses = {}  # {id: {model_name: response}}
    for r in hard_candidates:
        all_responses[r["id"]] = {}

    for rid, r in old_primary.items():
        if rid in all_responses and r.get("cot_response"):
            all_responses[rid]["primary_orig"] = r["cot_response"]

    if os.path.exists(primary_retry_path):
        for r in load_jsonl(primary_retry_path):
            if r["id"] in all_responses and r.get("response"):
                all_responses[r["id"]]["primary_retry"] = r["response"]

    for name in MODELS:
        model_path = os.path.join(MULTIMODEL_DIR, f"{name}.jsonl")
        if os.path.exists(model_path):
            for r in load_jsonl(model_path):
                if r["id"] in all_responses and r.get("response"):
                    all_responses[r["id"]][name] = r["response"]

    correct_ids = set()
    correct_by_model = {}
    need_judge = []

    for rid, responses in all_responses.items():
        record = id_to_record[rid]
        gold = record.get("answers", [])

        found_correct = False
        for model_name, resp in responses.items():
            is_cot = "primary" in model_name
            is_correct, predicted = check_model_response(resp, gold, is_cot=is_cot)
            if is_correct:
                correct_ids.add(rid)
                correct_by_model[model_name] = correct_by_model.get(model_name, 0) + 1
                found_correct = True
                break

        if not found_correct:
            for model_name, resp in responses.items():
                is_cot = "primary" in model_name
                _, predicted = check_model_response(resp, gold, is_cot=is_cot)
                if predicted and predicted.lower() not in ["i dont know", "i don't know", "unknown", ""]:
                    need_judge.append({
                        "rid": rid,
                        "model": model_name,
                        "question": record["question"],
                        "gold": record.get("best_answer", str(gold)),
                        "predicted": predicted,
                    })

    print(f"\nLocal-match correct: {len(correct_ids)}")
    for m, c in sorted(correct_by_model.items(), key=lambda x: -x[1]):
        print(f"  {m}: {c}")

    if need_judge:
        # Judge each rid at most once, using the first model that produced an answer.
        seen_rids = set()
        unique_judge = []
        for item in need_judge:
            if item["rid"] not in seen_rids and item["rid"] not in correct_ids:
                seen_rids.add(item["rid"])
                unique_judge.append(item)

        print(f"\nLLM Judge queue: {len(unique_judge)}")

        batch_size = 200
        judge_correct = 0
        for i in range(0, len(unique_judge), batch_size):
            batch = unique_judge[i:i + batch_size]
            results = asyncio.run(run_judge_batch(batch))
            for item, is_correct in zip(batch, results):
                if is_correct:
                    correct_ids.add(item["rid"])
                    model = item["model"]
                    correct_by_model[model] = correct_by_model.get(model, 0) + 1
                    judge_correct += 1

        print(f"LLM Judge added correct: {judge_correct}")

    # Step E: write outputs.
    print(f"\n=== Final results ===")
    print(f"Hard (any model correct): {len(correct_ids)}")
    print(f"Discard (all wrong):      {len(hard_candidates) - len(correct_ids)}")
    for m, c in sorted(correct_by_model.items(), key=lambda x: -x[1]):
        print(f"  {m}: {c}")

    phase3_results = []
    for r in hard_candidates:
        rid = r["id"]
        is_correct = rid in correct_ids
        # Prefer the original primary-LLM CoT trace, then the retry.
        cot = None
        for key in ["primary_orig", "primary_retry"]:
            resp = all_responses.get(rid, {}).get(key)
            if resp and "<think>" in resp:
                cot = resp
                break
        phase3_results.append({
            **{k: r[k] for k in ["id", "source", "question", "answers", "best_answer", "entities"]
               if k in r},
            "primary_correct": is_correct,
            "cot_response": cot,
        })

    save_jsonl(phase3_results, OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")

    print("\n--- Rebuilding step3_classified.jsonl ---")

    phase1_data = load_jsonl(os.path.join(INTERMEDIATE_DIR, "step3_phase1_qwen25.jsonl"))
    BASE_FIELDS = {"id", "source", "question", "answers", "best_answer", "entities"}

    classified = []

    for r in phase1_data:
        if r.get("qwen25_correct"):
            record = {k: r[k] for k in BASE_FIELDS if k in r}
            record["difficulty"] = "easy"
            classified.append(record)

    for r in phase2_data:
        if r.get("gptoss_correct"):
            record = {k: r[k] for k in BASE_FIELDS if k in r}
            record["difficulty"] = "medium"
            classified.append(record)

    for r in phase3_results:
        if r.get("primary_correct"):
            record = {k: r[k] for k in BASE_FIELDS if k in r}
            record["difficulty"] = "hard"
            if r.get("cot_response"):
                record["cot_response"] = r["cot_response"]
            classified.append(record)

    save_jsonl(classified, CLASSIFIED_PATH)

    from collections import Counter
    counts = Counter(r["difficulty"] for r in classified)
    print(f"Easy:   {counts['easy']}")
    print(f"Medium: {counts['medium']}")
    print(f"Hard:   {counts['hard']}")
    print(f"Total:  {len(classified)}")
    print(f"\nSaved: {CLASSIFIED_PATH}")
    print("\nNext: rerun step4 and step5.")


if __name__ == "__main__":
    main()
