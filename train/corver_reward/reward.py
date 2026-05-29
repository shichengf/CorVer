"""Core CorVer reward logic for GRPO training.

Implements binary CorVer scoring following the paper (Section 3.3):
- Sentence-level hallucination detection via entity co-occurrence
- Only ternary triplets (h, r, t) → check cooc(h, t) in corpus
- Zero co-occurrence signals hallucination (τ_cooc = 1)
- Reward is 0 (any hallucination detected) or 1 (clean)

Based on QuCo-RAG (https://github.com/ZhishanQ/QuCo-RAG).
"""

import logging
import re

from .entity_extractor import QuCoEntityExtractor

logger = logging.getLogger(__name__)


# Common English pronouns — triplets with these as head/tail are uninformative
PRONOUNS = frozenset({
    "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "theirs", "we", "us", "our", "ours",
    "i", "me", "my", "mine", "you", "your", "yours",
    "this", "that", "these", "those", "who", "whom", "which",
    "himself", "herself", "itself", "themselves", "myself", "yourself",
    "there", "here", "someone", "something", "anyone", "anything",
    "everyone", "everything", "nobody", "nothing",
})


def _is_pronoun(entity: str) -> bool:
    """Check if an entity string is a pronoun (case-insensitive)."""
    return entity.strip().lower() in PRONOUNS


def split_sentences(text: str) -> list:
    """Split text into sentences using regex-based heuristics.

    Avoids requiring spacy as a dependency.
    """
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def compute_corver_reward(
    response_str: str,
    extractor: QuCoEntityExtractor = None,
    client=None,
    ternary_tuple_threshold: int = 1,
) -> dict:
    """Compute binary CorVer reward for a generated response.

    Following the QuCo-RAG paper Section 3.3:
    - Extract knowledge triplets (h, r, t) from each sentence
    - Check co-occurrence cooc(h, t) in the pre-training corpus
    - If any cooc(h, t) < threshold → hallucination → reward = 0
    - Otherwise reward = 1

    Args:
        response_str: The model's generated response text.
        extractor: QuCoEntityExtractor instance.
        client: Infigram client instance (local or remote).
        ternary_tuple_threshold: Co-occurrence count threshold.
            Default 1 (paper Eq. 4: τ_cooc = 1).

    Returns:
        Dict with keys: score, num_sentences_checked, num_hallucinated, details.
    """
    if extractor is None or client is None:
        logger.error("Extractor or client not initialized")
        return {"score": 0.0}

    sentences = split_sentences(response_str)

    # Batch extract triplets for all sentences at once
    all_triplets = extractor.extract_triplets_batch(sentences)

    num_checked = 0
    num_hallucinated = 0
    sentence_details = []

    # Collect co-occurrence queries (ternary only, following paper Eq. 4)
    query_tasks = []  # (sentence_idx, query_str)
    for sent_idx, triplets in enumerate(all_triplets):
        if not triplets:
            continue
        for trp in triplets:
            if not isinstance(trp, list):
                continue
            # Only ternary triplets: check cooc(h, t)
            if len(trp) == 3:
                ent1, relation, ent2 = trp[0], trp[1], trp[2]
                if isinstance(ent1, str) and isinstance(ent2, str) and ent1.strip() and ent2.strip():
                    # Skip triplets where head or tail is a pronoun — uninformative for co-occurrence
                    if _is_pronoun(ent1) or _is_pronoun(ent2):
                        continue
                    query_tasks.append((sent_idx, f"{ent1} AND {ent2}"))

    # Batch all Infigram queries at once
    if query_tasks:
        query_strings = [q for _, q in query_tasks]
        batch_results = client.count_batch(query_strings)
    else:
        batch_results = []

    # Build a mapping: sentence_idx -> list of counts
    sent_query_results = {}
    for (sent_idx, _), (count, _) in zip(query_tasks, batch_results):
        if sent_idx not in sent_query_results:
            sent_query_results[sent_idx] = []
        sent_query_results[sent_idx].append(count)

    # Evaluate each sentence
    for sent_idx, triplets in enumerate(all_triplets):
        if not triplets:
            sentence_details.append({
                "sentence": sentences[sent_idx],
                "triplets": [],
                "hallucinated": False,
                "reason": "no_factual_content",
            })
            continue

        # Check if this sentence has any ternary queries
        counts = sent_query_results.get(sent_idx, [])
        if not counts:
            sentence_details.append({
                "sentence": sentences[sent_idx],
                "triplets": triplets,
                "hallucinated": False,
                "reason": "no_ternary_triplets",
            })
            continue

        num_checked += 1
        # Paper Eq. 4: hallucinated if min co-occurrence < τ_cooc
        sentence_hallucinated = False
        for count in counts:
            if count is not None and count < ternary_tuple_threshold:
                sentence_hallucinated = True
                break

        if sentence_hallucinated:
            num_hallucinated += 1

        sentence_details.append({
            "sentence": sentences[sent_idx],
            "triplets": triplets,
            "hallucinated": sentence_hallucinated,
        })

    # Per-sentence average: proportion of checked sentences that are clean
    if num_checked == 0:
        score = 0.0  # No checkable sentences → no information
    else:
        score = (num_checked - num_hallucinated) / num_checked

    return {
        "score": score,
        "num_sentences_checked": num_checked,
        "num_hallucinated": num_hallucinated,
        "details": sentence_details,
    }
