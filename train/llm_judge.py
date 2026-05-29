"""Optional LLM response-level judge for knowledge QA.

Calls any OpenAI-compatible chat-completions endpoint and grades each
completion on correctness / completeness / unsupported claims / conciseness.
The discrete label is mapped to a scalar reward.

API configuration is read from environment variables; no hardcoded defaults.
"""

import hashlib
import json
import logging
import os
import re
import time
import traceback
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward mapping
# ---------------------------------------------------------------------------

LABEL_TO_REWARD = {
    "GOOD": 2.0,
    "PARTIAL": 0.5,
    "BAD": -1.0,
    "REDUNDANT": -0.5,
    "NOT_ATTEMPTED": -1.0,
}

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """You are evaluating a knowledge QA answer. Judge whether the model answer contains the gold answer.

Question: {question}
Gold answer: {gold_answer}
Model answer: {model_answer}

IMPORTANT RULE: Your primary task is to check whether the model answer CONTAINS the essential information from the gold answer. If the gold answer information is present in the model answer, the label CANNOT be BAD regardless of other errors.

Step 1: Does the model answer contain or match the gold answer? (Yes/No)
Step 2: If Yes — are there extra unsupported or incorrect claims beyond the gold answer?
Step 3: Assign label based on these rules:

- GOOD: Contains the gold answer, concise, no significant extra errors.
- PARTIAL: Contains the gold answer BUT also has notable unsupported claims or errors elsewhere. OR only partially matches the gold answer.
- REDUNDANT: Contains the gold answer BUT has excessive redundancy, repetition, or off-topic expansion.
- BAD: Does NOT contain the gold answer. The core answer is wrong or completely missing.
- NOT_ATTEMPTED: Explicitly refuses to answer (e.g., "I don't know", empty answer).

Key point: An answer that contains the gold answer should NEVER be labeled BAD. Even if there are hallucinations elsewhere, use PARTIAL or REDUNDANT instead.

Output ONLY the label in tags: <LABEL>GOOD</LABEL> (or PARTIAL/BAD/REDUNDANT/NOT_ATTEMPTED)"""


# ---------------------------------------------------------------------------
# LLMJudge class
# ---------------------------------------------------------------------------

class LLMJudge:
    """Response-level judge backed by an OpenAI-compatible chat endpoint.

    Reads API configuration from environment variables:
    - JUDGE_LLM_API_KEY (required)
    - JUDGE_LLM_BASE_URL (required)
    - JUDGE_LLM_MODEL (required)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.api_key = os.environ.get("JUDGE_LLM_API_KEY")
        self.base_url = os.environ.get("JUDGE_LLM_BASE_URL")
        self.model = os.environ.get("JUDGE_LLM_MODEL")

        if not self.api_key:
            raise ValueError(
                "JUDGE_LLM_API_KEY environment variable is required. "
                "Set it before running training."
            )
        if not self.base_url:
            raise ValueError(
                "JUDGE_LLM_BASE_URL environment variable is required. "
                "Set it before running training."
            )
        if not self.model:
            raise ValueError(
                "JUDGE_LLM_MODEL environment variable is required. "
                "Set it before running training."
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"LLMJudge initialized: model={self.model}, base_url={self.base_url}")

        # Optional disk cache
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, "judge_cache.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r") as f:
                        self._cache = json.load(f)
                    logger.info(f"Loaded {len(self._cache)} cached judge results")
                except Exception:
                    self._cache = {}

    def _cache_key(self, question: str, answer: str) -> str:
        content = f"{question}|||{answer}"
        return hashlib.md5(content.encode()).hexdigest()

    def _save_cache(self):
        if not self._cache_dir:
            return
        cache_path = os.path.join(self._cache_dir, "judge_cache.json")
        try:
            with open(cache_path, "w") as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning(f"Failed to save judge cache: {e}")

    def judge_single(
        self,
        question: str,
        model_answer: str,
        gold_answer: str,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> Tuple[float, str, str]:
        """Judge a single response.

        Args:
            question: The question.
            model_answer: The model's answer (extracted from <answer> tags).
            gold_answer: The gold reference answer.
            max_retries: Max API retries.
            timeout: Request timeout in seconds.

        Returns:
            (reward, label, raw_response)
        """
        # Check cache
        key = self._cache_key(question, model_answer)
        if key in self._cache:
            label, reward = self._cache[key]
            return reward, label, f"[cached] {label}"

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer,
            model_answer=model_answer,
        )

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=128,
                    timeout=timeout,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("API returned empty content (None)")
                raw = content.strip()
                label = self._parse_label(raw)
                reward = LABEL_TO_REWARD.get(label, 0.0)

                # Cache result
                self._cache[key] = (label, reward)

                return reward, label, raw

            except Exception as e:
                wait = 2.0 * (attempt + 1)
                logger.warning(
                    f"Judge API error (attempt {attempt+1}/{max_retries}): {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        logger.error(f"Judge API failed after {max_retries} retries")
        return 0.0, "ERROR", "API_FAILURE"

    def _parse_label(self, raw: str) -> str:
        """Extract label from <LABEL>...</LABEL> tags or raw text."""
        # Try tag extraction first
        match = re.search(r'<LABEL>\s*(.*?)\s*</LABEL>', raw, re.IGNORECASE)
        if match:
            label = match.group(1).strip().upper()
            if label in LABEL_TO_REWARD:
                return label

        # Fallback: search for known labels in text
        upper = raw.upper()
        for label in ["NOT_ATTEMPTED", "REDUNDANT", "PARTIAL", "GOOD", "BAD"]:
            if label in upper:
                return label

        logger.warning(f"Could not parse judge label from: {raw!r}")
        return "PARTIAL"  # Conservative default

    def judge_batch(
        self,
        questions: List[str],
        completions: List[str],
        best_answers: List[str],
    ) -> Tuple[List[float], List[Dict]]:
        """Judge a batch of completions.

        Extracts <answer>...</answer> content from each completion before judging.

        Args:
            questions: List of questions.
            completions: List of full completions (may include <think>/<answer> tags).
            best_answers: List of gold reference answers.

        Returns:
            rewards: List[float] rewards.
            details: List[Dict] with per-sample judge info.
        """
        rewards = []
        details = []

        for i, (q, comp, gold) in enumerate(zip(questions, completions, best_answers)):
            # Extract answer from tags
            answer_match = re.search(r'<answer>(.*?)</answer>', comp, re.DOTALL)
            if answer_match:
                model_answer = answer_match.group(1).strip()
            else:
                model_answer = comp.strip()

            # Check for refusal
            if not model_answer or model_answer.isspace():
                reward, label, raw = 0.0, "NOT_ATTEMPTED", "empty_answer"
            else:
                refusal_patterns = ["i don't know", "i don\u2019t know", "i do not know"]
                is_refusal = any(p in model_answer.lower() for p in refusal_patterns)
                if is_refusal:
                    reward, label, raw = 0.0, "NOT_ATTEMPTED", "refusal_detected"
                else:
                    reward, label, raw = self.judge_single(q, model_answer, gold)

            rewards.append(reward)
            details.append({
                "sample_index": i,
                "question": q,
                "model_answer": model_answer[:200],
                "gold_answer": gold[:200],
                "label": label,
                "reward": reward,
                "raw_response": raw[:200] if raw else "",
            })

        # Save cache periodically
        self._save_cache()

        avg_reward = sum(rewards) / max(len(rewards), 1)
        label_counts = {}
        for d in details:
            label_counts[d["label"]] = label_counts.get(d["label"], 0) + 1
        logger.info(f"LLM judge: avg_reward={avg_reward:.3f}, labels={label_counts}")

        return rewards, details
