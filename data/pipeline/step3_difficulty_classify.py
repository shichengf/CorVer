"""Step 3: three-tier difficulty classification.

Cascades each Step 2 record through three models and labels it by the first
one that answers correctly:
  Phase 1: local vLLM verifier model -> easy
  Phase 2: secondary LLM (API)       -> medium
  Phase 3: primary LLM with CoT      -> hard
Records that no model answers correctly are discarded.

Each phase writes its own checkpoint so the run is resumable.
Follows Appendix A.1 of the paper: single first attempt, temperature=0.

Usage:
    source ~/.bashrc && conda activate verl
    cd data/pipeline
    python step3_difficulty_classify.py
"""
import asyncio
import json
import os
import re
import string
import time
import unicodedata

from config import (
    INTERMEDIATE_DIR, DATA_LLM_API_KEY, DATA_LLM_BASE_URL,
    DATA_LLM_MODEL, SECONDARY_LLM_MODEL, QWEN25_MODEL,
    API_MAX_CONCURRENT, API_RETRY_MAX, API_RETRY_DELAY,
)

INPUT_PATH = os.path.join(INTERMEDIATE_DIR, "step2_refined.jsonl")
PHASE1_PATH = os.path.join(INTERMEDIATE_DIR, "step3_phase1_local.jsonl")
PHASE2_PATH = os.path.join(INTERMEDIATE_DIR, "step3_phase2_secondary.jsonl")
PHASE3_PATH = os.path.join(INTERMEDIATE_DIR, "step3_phase3_primary.jsonl")
OUTPUT_PATH = os.path.join(INTERMEDIATE_DIR, "step3_classified.jsonl")

SIMPLE_QA_PROMPT = (
    "Answer the following question concisely with just the answer, "
    "no explanation needed.\n\n"
    "Question: {question}\n"
    "Answer:"
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
    """Lowercase, strip punctuation and leading articles, collapse whitespace."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    for article in ["a ", "an ", "the "]:
        if text.startswith(article):
            text = text[len(article):]
    text = " ".join(text.split())
    return text


def check_correct(response: str, gold_answers) -> bool:
    """Bidirectional substring match against any gold answer after normalization."""
    resp_norm = normalize_answer(response)
    if not resp_norm:
        return False
    if resp_norm in ["i dont know", "i don't know", "unknown"]:
        return False

    if isinstance(gold_answers, str):
        answers = [gold_answers]
    else:
        answers = gold_answers

    for ans in answers:
        ans_norm = normalize_answer(ans)
        if not ans_norm:
            continue
        if ans_norm == resp_norm:
            return True
        if ans_norm in resp_norm or resp_norm in ans_norm:
            return True
    return False


def extract_answer_from_response(response: str) -> str:
    """Extract the final answer from <answer> tags, falling back to post-</think> text."""
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'</think>\s*(.*)', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


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


def append_jsonl(records: list, path: str):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_phase1(records: list) -> list:
    """Run the local vLLM verifier and tag each record with qwen25_correct."""
    print("\n" + "=" * 60)
    print("Phase 1: local vLLM verifier")
    print("=" * 60)

    if os.path.exists(PHASE1_PATH):
        existing = load_jsonl(PHASE1_PATH)
        processed_ids = {r["id"] for r in existing}
        remaining = [r for r in records if r["id"] not in processed_ids]
        if not remaining:
            print(f"Phase 1 already complete, loaded {len(existing)} checkpoint records")
            return existing
        print(f"Found {len(existing)} checkpoint records, {len(remaining)} remaining")
    else:
        remaining = records
        existing = []

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN25_MODEL, trust_remote_code=True)

    prompts = []
    for r in remaining:
        messages = [{"role": "user", "content": SIMPLE_QA_PROMPT.format(question=r["question"])}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    print(f"Built {len(prompts)} prompts")
    print(f"Loading model: {QWEN25_MODEL}")

    llm = LLM(
        model=QWEN25_MODEL,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        max_model_len=2048,
    )
    params = SamplingParams(
        max_tokens=128,
        temperature=0,
        stop=["<|endoftext|>", "<|im_end|>"],
    )

    print(f"Running inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, params)

    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    results = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text.strip()
        is_correct = check_correct(response, remaining[i]["answers"])
        results.append({
            **remaining[i],
            "qwen25_response": response,
            "qwen25_correct": is_correct,
        })

    append_jsonl(results, PHASE1_PATH)
    all_results = existing + results

    correct = sum(1 for r in all_results if r["qwen25_correct"])
    total = len(all_results)
    print(f"\nPhase 1: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"  Easy: {correct}")
    print(f"  Forwarded to Phase 2: {total - correct}")

    return all_results


async def _api_simple_qa(session, semaphore, record, model):
    import aiohttp

    headers = {
        "Authorization": f"Bearer {DATA_LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": SIMPLE_QA_PROMPT.format(question=record["question"])},
        ],
        "temperature": 0,
    }

    for attempt in range(API_RETRY_MAX):
        async with semaphore:
            try:
                async with session.post(
                    f"{DATA_LLM_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        try:
                            content = data["choices"][0]["message"]["content"]
                        except (KeyError, IndexError):
                            if attempt < API_RETRY_MAX - 1:
                                await asyncio.sleep(API_RETRY_DELAY)
                                continue
                            return {**record, "api_response": None, "error": f"bad_json:{str(data)[:80]}"}
                        return {**record, "api_response": content}
                    elif resp.status == 429:
                        await asyncio.sleep(API_RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        if attempt < API_RETRY_MAX - 1:
                            await asyncio.sleep(API_RETRY_DELAY)
                            continue
                        return {**record, "api_response": None, "error": f"api_{resp.status}"}
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < API_RETRY_MAX - 1:
                    await asyncio.sleep(API_RETRY_DELAY * (attempt + 1))
                    continue
                return {**record, "api_response": None, "error": str(e)[:100]}

    return {**record, "api_response": None, "error": "max_retries"}


async def _api_cot_qa(session, semaphore, record):
    import aiohttp

    headers = {
        "Authorization": f"Bearer {DATA_LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DATA_LLM_MODEL,
        "messages": [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": record["question"]},
        ],
        "temperature": 0,
    }

    for attempt in range(API_RETRY_MAX):
        async with semaphore:
            try:
                async with session.post(
                    f"{DATA_LLM_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        try:
                            content = data["choices"][0]["message"]["content"]
                        except (KeyError, IndexError):
                            if attempt < API_RETRY_MAX - 1:
                                await asyncio.sleep(API_RETRY_DELAY)
                                continue
                            return {**record, "cot_response": None, "error": f"bad_json:{str(data)[:80]}"}
                        return {**record, "cot_response": content}
                    elif resp.status == 429:
                        await asyncio.sleep(API_RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        if attempt < API_RETRY_MAX - 1:
                            await asyncio.sleep(API_RETRY_DELAY)
                            continue
                        return {**record, "cot_response": None, "error": f"api_{resp.status}"}
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < API_RETRY_MAX - 1:
                    await asyncio.sleep(API_RETRY_DELAY * (attempt + 1))
                    continue
                return {**record, "cot_response": None, "error": str(e)[:100]}

    return {**record, "cot_response": None, "error": "max_retries"}


async def _judge_correctness(session, semaphore, question, gold_answer, predicted_answer):
    import aiohttp

    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
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


async def _batch_api_qa(records, model, desc="API QA"):
    import aiohttp

    semaphore = asyncio.Semaphore(API_MAX_CONCURRENT)
    async with aiohttp.ClientSession() as session:
        tasks = [_api_simple_qa(session, semaphore, r, model) for r in records]
        try:
            from tqdm.asyncio import tqdm_asyncio
            results = await tqdm_asyncio.gather(*tasks, desc=desc)
        except ImportError:
            results = []
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                result = await coro
                results.append(result)
                if (i + 1) % 100 == 0:
                    print(f"  {desc} Progress: {i+1}/{len(tasks)}")
    return results


async def _batch_cot_qa(records, desc="CoT"):
    import aiohttp

    semaphore = asyncio.Semaphore(API_MAX_CONCURRENT)
    async with aiohttp.ClientSession() as session:
        tasks = [_api_cot_qa(session, semaphore, r) for r in records]
        try:
            from tqdm.asyncio import tqdm_asyncio
            results = await tqdm_asyncio.gather(*tasks, desc=desc)
        except ImportError:
            results = []
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                result = await coro
                results.append(result)
                if (i + 1) % 100 == 0:
                    print(f"  {desc} Progress: {i+1}/{len(tasks)}")
    return results


async def _batch_judge(records_to_judge, desc="LLM Judge"):
    import aiohttp

    semaphore = asyncio.Semaphore(API_MAX_CONCURRENT)
    async with aiohttp.ClientSession() as session:
        tasks = [
            _judge_correctness(
                session, semaphore,
                r["question"], r["best_answer"], r["_predicted"]
            )
            for r in records_to_judge
        ]
        try:
            from tqdm.asyncio import tqdm_asyncio
            results = await tqdm_asyncio.gather(*tasks, desc=desc)
        except ImportError:
            results = await asyncio.gather(*tasks)
    return results


def run_phase2(phase1_results: list) -> list:
    """Classify Phase 1 failures with the secondary LLM."""
    print("\n" + "=" * 60)
    print("Phase 2: secondary LLM")
    print("=" * 60)

    to_process = [r for r in phase1_results if not r["qwen25_correct"]]
    print(f"Items failed by Phase 1: {len(to_process)}")

    if os.path.exists(PHASE2_PATH):
        existing = load_jsonl(PHASE2_PATH)
        processed_ids = {r["id"] for r in existing}
        remaining = [r for r in to_process if r["id"] not in processed_ids]
        if not remaining:
            print(f"Phase 2 already complete, loaded {len(existing)} checkpoint records")
            return existing
        print(f"Found {len(existing)} checkpoint records, {len(remaining)} remaining")
    else:
        remaining = to_process
        existing = []

    if remaining:
        batch_size = 200
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(remaining) + batch_size - 1) // batch_size
            print(f"\n--- Secondary LLM batch {batch_num}/{total_batches} ({len(batch)} records) ---")

            api_results = asyncio.run(_batch_api_qa(batch, SECONDARY_LLM_MODEL, "Secondary QA"))

            batch_results = []
            need_judge = []
            for r in api_results:
                response = r.get("api_response")
                if not response:
                    batch_results.append({**r, "gptoss_response": None, "gptoss_correct": False})
                    continue

                if check_correct(response, r["answers"]):
                    batch_results.append({**r, "gptoss_response": response, "gptoss_correct": True})
                else:
                    need_judge.append({**r, "gptoss_response": response, "_predicted": response})

            if need_judge:
                judge_results = asyncio.run(_batch_judge(need_judge, "Phase2 Judge"))
                for r, is_correct in zip(need_judge, judge_results):
                    r_clean = {k: v for k, v in r.items() if k != "_predicted"}
                    batch_results.append({**r_clean, "gptoss_correct": is_correct})
            else:
                pass

            clean_results = []
            for r in batch_results:
                clean = {k: v for k, v in r.items() if not k.startswith("_") and k != "api_response"}
                clean_results.append(clean)
            append_jsonl(clean_results, PHASE2_PATH)

            correct = sum(1 for r in clean_results if r["gptoss_correct"])
            print(f"  Batch: {correct}/{len(clean_results)} correct")

    all_results = load_jsonl(PHASE2_PATH)
    correct = sum(1 for r in all_results if r["gptoss_correct"])
    total = len(all_results)
    print(f"\nPhase 2: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"  Medium: {correct}")
    print(f"  Forwarded to Phase 3: {total - correct}")

    return all_results


def run_phase3(phase2_results: list) -> list:
    """Classify Phase 2 failures with the primary LLM using CoT."""
    print("\n" + "=" * 60)
    print("Phase 3: primary LLM with CoT")
    print("=" * 60)

    to_process = [r for r in phase2_results if not r["gptoss_correct"]]
    print(f"Items failed by Phase 2: {len(to_process)}")

    if os.path.exists(PHASE3_PATH):
        existing = load_jsonl(PHASE3_PATH)
        processed_ids = {r["id"] for r in existing}
        remaining = [r for r in to_process if r["id"] not in processed_ids]
        if not remaining:
            print(f"Phase 3 already complete, loaded {len(existing)} checkpoint records")
            return existing
        print(f"Found {len(existing)} checkpoint records, {len(remaining)} remaining")
    else:
        remaining = to_process
        existing = []

    if remaining:
        batch_size = 200
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(remaining) + batch_size - 1) // batch_size
            print(f"\n--- CoT batch {batch_num}/{total_batches} ({len(batch)} records) ---")

            cot_results = asyncio.run(_batch_cot_qa(batch, "CoT"))

            batch_results = []
            need_judge = []
            skipped = 0
            for r in cot_results:
                response = r.get("cot_response")
                if not response:
                    batch_results.append({**r, "primary_correct": False})
                    skipped += 1
                    continue

                predicted = extract_answer_from_response(response)
                gold_answers = r.get("answers", [])
                if isinstance(gold_answers, str):
                    gold_answers = [gold_answers]

                if check_correct(predicted, gold_answers):
                    batch_results.append({**r, "primary_correct": True})
                else:
                    need_judge.append({**r, "_predicted": predicted})

            if need_judge:
                judge_results = asyncio.run(_batch_judge(need_judge, "Phase3 Judge"))
                for r, is_correct in zip(need_judge, judge_results):
                    r_clean = {k: v for k, v in r.items() if k != "_predicted"}
                    batch_results.append({**r_clean, "primary_correct": is_correct})

            clean_results = []
            for r in batch_results:
                clean = {k: v for k, v in r.items() if not k.startswith("_")}
                clean_results.append(clean)
            append_jsonl(clean_results, PHASE3_PATH)

            correct = sum(1 for r in clean_results if r["primary_correct"])
            print(f"  Batch: {correct}/{len(clean_results)} correct, {skipped} skipped")

    all_results = load_jsonl(PHASE3_PATH)
    correct = sum(1 for r in all_results if r["primary_correct"])
    total = len(all_results)
    print(f"\nPhase 3: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"  Hard: {correct}")
    print(f"  Discard: {total - correct}")

    return all_results


def merge_results(phase1_results, phase2_results, phase3_results):
    """Combine the three phases and write step3_classified.jsonl."""
    print("\n" + "=" * 60)
    print("Merging results")
    print("=" * 60)

    BASE_FIELDS = {"id", "source", "question", "answers", "best_answer", "entities"}

    classified = []

    for r in phase1_results:
        if r["qwen25_correct"]:
            record = {k: r[k] for k in BASE_FIELDS if k in r}
            record["difficulty"] = "easy"
            classified.append(record)

    for r in phase2_results:
        if r["gptoss_correct"]:
            record = {k: r[k] for k in BASE_FIELDS if k in r}
            record["difficulty"] = "medium"
            classified.append(record)

    # Hard records keep their CoT trace for downstream use.
    for r in phase3_results:
        if r["primary_correct"]:
            record = {k: r[k] for k in BASE_FIELDS if k in r}
            record["difficulty"] = "hard"
            if r.get("cot_response"):
                record["cot_response"] = r["cot_response"]
            classified.append(record)

    save_jsonl(classified, OUTPUT_PATH)

    from collections import Counter
    counts = Counter(r["difficulty"] for r in classified)
    total = len(classified)
    discarded = len(phase3_results) - sum(1 for r in phase3_results if r["primary_correct"])

    print(f"\n=== Step 3 summary ===")
    print(f"Input: {len(phase1_results)}")
    print(f"Easy:    {counts['easy']:>6} ({counts['easy']/len(phase1_results)*100:.1f}%)")
    print(f"Medium:  {counts['medium']:>6} ({counts['medium']/len(phase1_results)*100:.1f}%)")
    print(f"Hard:    {counts['hard']:>6} ({counts['hard']/len(phase1_results)*100:.1f}%)")
    print(f"Discard: {discarded:>6} ({discarded/len(phase1_results)*100:.1f}%)")
    print(f"Kept total: {total}")
    print(f"Output: {OUTPUT_PATH}")

    return classified


def main():
    print("=" * 60)
    print("Step 3: three-tier difficulty classification")
    print("  Easy:   local vLLM verifier correct")
    print("  Medium: secondary LLM correct")
    print("  Hard:   primary LLM correct (CoT)")
    print("=" * 60)

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found. Run step2 first.")
        return

    records = load_jsonl(INPUT_PATH)
    print(f"Loaded {len(records)} Step 2 records")

    t0 = time.time()

    phase1_results = run_phase1(records)
    phase2_results = run_phase2(phase1_results)
    phase3_results = run_phase3(phase2_results)

    classified = merge_results(phase1_results, phase2_results, phase3_results)

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
