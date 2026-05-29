"""String-match judge for knowledge QA.

Drop-in replacement for LLMJudge. Uses exact/substring string matching
(same logic as eval_triviaqa.py) instead of LLM API calls.

Advantages: no API latency, deterministic, higher positive signal rate.
Disadvantage: no semantic understanding, can't detect hallucination.
"""

import logging
import re
import unicodedata
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# Reward mapping — same interface as the LLM judge
LABEL_TO_REWARD = {
    "GOOD": 2.0,
    "PARTIAL": 0.5,
    "BAD": -1.0,
    "NOT_ATTEMPTED": -1.0,
}


def _normalize(s: str) -> str:
    """Normalize text for comparison: lowercase, remove articles/punctuation."""
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _check_correct(predicted: str, gold: str) -> str:
    """Check if predicted answer matches gold answer.

    Returns: "GOOD", "BAD", or "NOT_ATTEMPTED"
    """
    if not predicted or predicted.lower().strip() in [
        "i don't know", "i do not know", "i don\u2019t know", ""
    ]:
        return "NOT_ATTEMPTED"

    pred_norm = _normalize(predicted)
    if not pred_norm:
        return "NOT_ATTEMPTED"

    # Gold might contain semicolon-separated aliases
    all_answers = [a.strip() for a in gold.split(";")]

    for ans in all_answers:
        ans_norm = _normalize(ans)
        if not ans_norm:
            continue
        # Exact match
        if pred_norm == ans_norm:
            return "GOOD"
        # Gold in predicted (predicted contains the answer)
        if ans_norm in pred_norm:
            return "GOOD"
        # Predicted in gold (predicted is a substring of the answer)
        if pred_norm in ans_norm:
            return "GOOD"

    return "BAD"


class StringMatchJudge:
    """String-match judge — drop-in replacement for LLMJudge.

    Same interface: judge_batch(questions, completions, best_answers) -> (rewards, details)
    """

    def judge_single(
        self,
        question: str,
        model_answer: str,
        gold_answer: str,
        **kwargs,
    ) -> Tuple[float, str, str]:
        label = _check_correct(model_answer, gold_answer)
        reward = LABEL_TO_REWARD.get(label, -1.0)
        return reward, label, f"string_match:{label}"

    def judge_batch(
        self,
        questions: List[str],
        completions: List[str],
        best_answers: List[str],
    ) -> Tuple[List[float], List[Dict]]:
        rewards = []
        details = []

        for i, (q, comp, gold) in enumerate(zip(questions, completions, best_answers)):
            # Extract answer from tags. Lenient: read to closing tag OR end-of-string
            # so truncated completions still get extracted (avoids false-negative reward).
            answer_match = re.search(r'<answer>(.*?)(?:</answer>|$)', comp, re.DOTALL)
            if answer_match:
                model_answer = answer_match.group(1).strip()
            else:
                model_answer = comp.strip()

            reward, label, raw = self.judge_single(q, model_answer, gold)
            rewards.append(reward)
            details.append({
                "sample_index": i,
                "question": q,
                "model_answer": model_answer[:200],
                "gold_answer": gold[:200],
                "label": label,
                "reward": reward,
                "raw_response": raw,
            })

        # Log stats
        from collections import Counter
        label_counts = Counter(d["label"] for d in details)
        avg_reward = sum(rewards) / max(len(rewards), 1)
        logger.info(f"StringMatch judge: avg_reward={avg_reward:.3f}, labels={dict(label_counts)}")

        return rewards, details
