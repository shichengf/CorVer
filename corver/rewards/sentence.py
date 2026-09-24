"""First-pair corpus rewards, Eq. (2) and Appendix C.2."""

import re
from typing import List, Optional, Tuple, Dict


def sentence_corver_penalty(count: Optional[int]) -> float:
    """Map a single head-tail co-occurrence count to a penalty.

    Each sentence yields at most one main ternary triplet.
    Weak signal: small positive for supported facts, moderate penalty for unsupported.
    Missing or failed queries receive a neutral sentence reward.

    Args:
        count: Co-occurrence count for the sentence's main triplet.
               None means no valid ternary triplet was extracted.
    """
    if count is None:
        return 0.0  # no valid triplet, neutral
    if count == 0:
        return -0.2  # zero co-occurrence, moderate penalty
    if count < 5:
        return -0.1  # very low co-occurrence, light penalty
    if count < 20:
        return 0.0  # borderline, neutral
    return 0.1


PRONOUNS = frozenset(
    {
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
        "we",
        "us",
        "our",
        "ours",
        "i",
        "me",
        "my",
        "mine",
        "you",
        "your",
        "yours",
        "this",
        "that",
        "these",
        "those",
        "who",
        "whom",
        "which",
        "himself",
        "herself",
        "itself",
        "themselves",
        "myself",
        "yourself",
        "there",
        "here",
        "someone",
        "something",
        "anyone",
        "anything",
        "everyone",
        "everything",
        "nobody",
        "nothing",
    }
)


def _is_pronoun(entity: str) -> bool:
    return entity.strip().lower() in PRONOUNS


_QUERY_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "at",
        "on",
        "for",
        "by",
        "with",
        "to",
        "from",
        "and",
        "or",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "has",
        "had",
        "have",
        "do",
        "does",
        "did",
        "not",
        "no",
        "but",
        "if",
        "so",
        "as",
        "its",
        "it",
    }
)


def _extract_content_words(entity: str) -> List[str]:
    """Extract content words from an entity string.

    Keeps words that are likely meaningful for co-occurrence:
    - Capitalized words (proper nouns)
    - Non-stop words longer than 2 characters

    E.g.:
        "Congress of the United States" -> ["Congress", "United", "States"]
        "John Adams" -> ["John", "Adams"]
        "theory of general relativity" -> ["theory", "general", "relativity"]
    """
    words = entity.strip().replace(",", " ").split()
    # First try: capitalized non-stop words (proper nouns)
    proper = [
        w
        for w in words
        if len(w) > 1 and w[0].isupper() and w.lower() not in _QUERY_STOP_WORDS
    ]
    if proper:
        return proper
    # Fallback: all non-stop words > 2 chars
    content = [w for w in words if len(w) > 2 and w.lower() not in _QUERY_STOP_WORDS]
    return content if content else [entity.strip()]


def _build_word_level_query(head: str, tail: str) -> Optional[str]:
    """Build a word-level AND query from head and tail entities.

    Instead of querying exact entity substrings:
        "Congress of the United States AND John Adams"
    Query individual content words:
        "Congress AND United AND States AND John AND Adams"

    This is domain-agnostic and handles entity name variations naturally.
    """
    h_words = _extract_content_words(head)
    t_words = _extract_content_words(tail)

    # Deduplicate while preserving order
    all_words = []
    seen = set()
    for w in h_words + t_words:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            all_words.append(w)

    if len(all_words) < 2:
        return None  # need at least 2 words for meaningful co-occurrence

    return " AND ".join(all_words)


def compute_sentence_corver_penalties(
    sentences: List[str],
    extractor,
    client,
) -> Tuple[List[float], List[Dict]]:
    """Compute the CorVer corpus reward for each sentence.

    Each sentence yields at most one main ternary triplet.

    Args:
        sentences: List of sentence strings.
        extractor: Frozen triplet extractor.
        client: Infigram client instance.

    Returns:
        penalties: List[float] of per-sentence penalties.
        details: List[Dict] with debug info per sentence.
    """
    if not sentences:
        return [], []

    # Batch extract triplets
    all_triplets = extractor.extract_triplets_batch(sentences)

    if len(all_triplets) != len(sentences):
        raise ValueError("Extractor returned an incorrect number of rows")

    # For each sentence, find the first valid ternary triplet
    queries = []  # (sent_idx, query_str, triplet)
    sent_main_triplet = [None] * len(sentences)  # index -> triplet or None

    for sent_idx, triplets in enumerate(all_triplets):
        if not triplets:
            continue
        for trp in triplets:
            if not isinstance(trp, list) or len(trp) != 3:
                continue
            h, r, t = trp[0], trp[1], trp[2]
            if not (
                isinstance(h, str) and isinstance(t, str) and h.strip() and t.strip()
            ):
                continue
            if _is_pronoun(h) or _is_pronoun(t):
                continue
            # Found first valid ternary triplet — build word-level AND query
            query = _build_word_level_query(h, t)
            if query is None:
                continue
            sent_main_triplet[sent_idx] = trp
            queries.append((sent_idx, query, trp))
            break  # Never try another triplet after a query failure.

    # Batch query co-occurrence (single pass, word-level AND)
    if queries:
        query_strings = [q[1] for q in queries]
        batch_results = client.count_batch(query_strings)
    else:
        batch_results = []

    if len(batch_results) != len(queries):
        raise ValueError("Corpus client returned an incorrect number of results")

    # Build per-sentence count map
    sent_count = {}
    for (sent_idx, _, _), (count, _) in zip(queries, batch_results):
        sent_count[sent_idx] = count

    # Compute penalties and details
    penalties = []
    details = []
    for sent_idx, sent in enumerate(sentences):
        triplet = sent_main_triplet[sent_idx]
        count = sent_count.get(sent_idx, None)
        penalty = sentence_corver_penalty(count)
        penalties.append(penalty)
        # Find the query used for this sentence (for logging)
        query_used = None
        for si, qs, _ in queries:
            if si == sent_idx:
                query_used = qs
                break
        details.append(
            {
                "sentence": sent,
                "triplet": triplet,
                "query": query_used,
                "cooc_count": count,
                "sentence_reward": penalty,
            }
        )

    return penalties, details
