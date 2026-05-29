"""Sentence-level CorVer penalty + token alignment + per-token return construction.

Core module for stepwise GRPO: computes sentence-level CorVer penalties,
maps sentences to token spans, and constructs per-token returns.

This is NOT a retrieval system or RAG module. CorVer serves as a weak
corpus-grounded penalty signal — it punishes unsupported claims but
does not certify correctness from high co-occurrence.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CorVer sentence penalty — four-tier nonlinear mapping
# ---------------------------------------------------------------------------

def sentence_corver_penalty(count: Optional[int]) -> float:
    """Map a single head-tail co-occurrence count to a penalty.

    Each sentence yields at most one main ternary triplet.
    Weak signal: small positive for supported facts, moderate penalty for unsupported.
    This prevents the model from learning to "say nothing" to avoid penalties.

    Args:
        count: Co-occurrence count for the sentence's main triplet.
               None means no valid ternary triplet was extracted.
    """
    if count is None:
        return 0.0       # no valid triplet, neutral
    if count == 0:
        return -0.3      # zero co-occurrence, moderate penalty
    if count < 5:
        return -0.1      # very low co-occurrence, light penalty
    if count < 20:
        return 0.0       # borderline, neutral
    return 0.1            # sufficient co-occurrence, small positive (supported fact)


# ---------------------------------------------------------------------------
# Sentence splitting (reuse from corver_reward.reward)
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex heuristics.

    Replicates corver_reward.reward.split_sentences to avoid circular imports.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Pronoun set (reuse from corver_reward.reward)
# ---------------------------------------------------------------------------

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
    return entity.strip().lower() in PRONOUNS


# ---------------------------------------------------------------------------
# Word-level co-occurrence query construction
# ---------------------------------------------------------------------------

# Common English stop words — too frequent to be meaningful in co-occurrence
_QUERY_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "in", "at", "on", "for", "by", "with",
    "to", "from", "and", "or", "is", "was", "are", "were", "be",
    "been", "being", "has", "had", "have", "do", "does", "did",
    "not", "no", "but", "if", "so", "as", "its", "it",
})


def _extract_content_words(entity: str) -> List[str]:
    """Extract content words from an entity string.

    Keeps words that are likely meaningful for co-occurrence:
    - Capitalized words (proper nouns)
    - Non-stop words longer than 2 characters

    E.g.:
        "First Congress of the United States" -> ["Congress"]
        "John Adams" -> ["John", "Adams"]
        "theory of general relativity" -> ["theory", "general", "relativity"]
    """
    words = entity.strip().replace(",", " ").split()
    # First try: capitalized non-stop words (proper nouns)
    proper = [w for w in words
              if len(w) > 1 and w[0].isupper() and w.lower() not in _QUERY_STOP_WORDS]
    if proper:
        return proper
    # Fallback: all non-stop words > 2 chars
    content = [w for w in words
               if len(w) > 2 and w.lower() not in _QUERY_STOP_WORDS]
    return content if content else [entity.strip()]


def _build_word_level_query(head: str, tail: str) -> Optional[str]:
    """Build a word-level AND query from head and tail entities.

    Instead of querying exact entity substrings:
        "First Congress of the United States AND John Adams"
    Query individual content words:
        "Congress AND John AND Adams"

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


# ---------------------------------------------------------------------------
# Token-to-sentence alignment
# ---------------------------------------------------------------------------

def build_token_to_sentence_map(
    completion_ids: torch.Tensor,
    tokenizer,
    completion_mask: torch.Tensor,
) -> Tuple[torch.Tensor, List[str], float]:
    """Map each completion token to a sentence index.

    Args:
        completion_ids: (T,) token IDs for one completion.
        tokenizer: HuggingFace tokenizer.
        completion_mask: (T,) binary mask (1 = valid token).

    Returns:
        token_to_sentence: (T,) int tensor. -1 for unaligned / tag / padding.
        sentences: List of extracted sentences.
        alignment_rate: Fraction of valid tokens successfully aligned.
    """
    T = completion_ids.shape[0]
    token_to_sentence = torch.full((T,), -1, dtype=torch.long)

    # 1. Decode full completion text (skip special tokens)
    valid_ids = completion_ids[completion_mask.bool()].tolist()
    if not valid_ids:
        return token_to_sentence, [], 0.0

    full_text = tokenizer.decode(valid_ids, skip_special_tokens=True)
    if not full_text.strip():
        return token_to_sentence, [], 0.0

    # 2. Build per-token character spans by decoding each token individually
    token_char_spans = []  # (start, end) for each token position
    char_offset = 0
    for idx in range(T):
        if completion_mask[idx] == 0:
            token_char_spans.append((-1, -1))
            continue
        tok_text = tokenizer.decode([completion_ids[idx].item()], skip_special_tokens=False)
        token_char_spans.append((char_offset, char_offset + len(tok_text)))
        char_offset += len(tok_text)

    # 3. Decode again with skip_special_tokens=False to get raw text for offset matching
    raw_text = tokenizer.decode(valid_ids, skip_special_tokens=False)

    # 4. Strip tags from BOTH <think> and <answer> portions for sentence splitting.
    # <think> first, <answer> after; both sections contribute to CorVer scoring with equal weight.
    think_match = re.search(r'<think>(.*?)</think>', full_text, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', full_text, re.DOTALL)

    sections = []  # list of (text, start_in_full)
    if think_match:
        t = think_match.group(1).strip()
        if t:
            sections.append((t, full_text.find(t)))
    if answer_match:
        a = answer_match.group(1).strip()
        if a:
            anchor = think_match.end() if think_match else 0
            sections.append((a, full_text.find(a, anchor)))
    if not sections:
        sections = [(full_text, 0)]

    # 5. Split each section into sentences, keep ordered list + char spans
    sentences = []
    sentence_char_spans = []
    for sec_text, sec_start in sections:
        sec_sents = split_sentences(sec_text)
        if not sec_sents:
            continue
        search_start = sec_start
        for sent in sec_sents:
            idx = full_text.find(sent, search_start)
            sentences.append(sent)
            if idx >= 0:
                sentence_char_spans.append((idx, idx + len(sent)))
                search_start = idx + len(sent)
            else:
                sentence_char_spans.append((-1, -1))

    if not sentences:
        return token_to_sentence, [], 0.0

    # 7. Map raw_text char offsets to full_text char offsets
    #    This handles the difference between skip_special_tokens=True/False
    #    We need to map token char spans (based on raw decode) to full_text positions
    #    Simplified approach: re-decode with skip_special_tokens=True and track offsets
    #    Actually, let's use a direct approach: decode the valid tokens with skip_special_tokens=True
    #    and build spans from that.

    # Rebuild char spans using skip_special_tokens=True decoding
    token_char_spans_clean = []
    char_offset = 0
    for idx in range(T):
        if completion_mask[idx] == 0:
            token_char_spans_clean.append((-1, -1))
            continue
        tok_text = tokenizer.decode([completion_ids[idx].item()], skip_special_tokens=True)
        if tok_text:
            token_char_spans_clean.append((char_offset, char_offset + len(tok_text)))
            char_offset += len(tok_text)
        else:
            # Special token that gets removed
            token_char_spans_clean.append((-1, -1))

    # 8. Assign tokens to sentences using midpoint matching
    for tok_idx, (ts, te) in enumerate(token_char_spans_clean):
        if ts == -1:
            continue
        tok_mid = (ts + te) / 2.0
        for sent_idx, (ss, se) in enumerate(sentence_char_spans):
            if ss == -1:
                continue
            if ss <= tok_mid < se:
                token_to_sentence[tok_idx] = sent_idx
                break

    # 9. Compute alignment rate — denominator = tokens whose midpoint falls
    # within ANY of the scored sections (think + answer when both present).
    section_spans = []
    for sec_text, sec_start in sections:
        if sec_start >= 0:
            section_spans.append((sec_start, sec_start + len(sec_text)))
    if section_spans:
        section_token_count = 0
        for ts, te in token_char_spans_clean:
            if ts == -1:
                continue
            tok_mid = (ts + te) / 2.0
            for ss, se in section_spans:
                if ss <= tok_mid < se:
                    section_token_count += 1
                    break
        aligned = (token_to_sentence >= 0).sum().item()
        alignment_rate = aligned / max(section_token_count, 1)
    else:
        total_valid = completion_mask.sum().item()
        aligned = (token_to_sentence >= 0).sum().item()
        alignment_rate = aligned / max(total_valid, 1)

    return token_to_sentence, sentences, alignment_rate


# ---------------------------------------------------------------------------
# Sentence-level CorVer scoring (single completion)
# ---------------------------------------------------------------------------

def compute_sentence_corver_penalties(
    sentences: List[str],
    extractor,
    client,
) -> Tuple[List[float], List[Dict]]:
    """Compute CorVer penalty for each sentence.

    Each sentence yields at most one main ternary triplet.

    Args:
        sentences: List of sentence strings.
        extractor: QuCoEntityExtractor instance.
        client: Infigram client instance.

    Returns:
        penalties: List[float] of per-sentence penalties.
        details: List[Dict] with debug info per sentence.
    """
    if not sentences:
        return [], []

    # Batch extract triplets
    all_triplets = extractor.extract_triplets_batch(sentences)

    # For each sentence, find the first valid ternary triplet
    queries = []       # (sent_idx, query_str, triplet)
    sent_main_triplet = [None] * len(sentences)  # index -> triplet or None

    for sent_idx, triplets in enumerate(all_triplets):
        if not triplets:
            continue
        for trp in triplets:
            if not isinstance(trp, list) or len(trp) != 3:
                continue
            h, r, t = trp[0], trp[1], trp[2]
            if not (isinstance(h, str) and isinstance(t, str) and h.strip() and t.strip()):
                continue
            if _is_pronoun(h) or _is_pronoun(t):
                continue
            # Found first valid ternary triplet — build word-level AND query
            sent_main_triplet[sent_idx] = trp
            query = _build_word_level_query(h, t)
            if query:
                queries.append((sent_idx, query, trp))
            break  # only first

    # Batch query co-occurrence (single pass, word-level AND)
    if queries:
        query_strings = [q[1] for q in queries]
        batch_results = client.count_batch(query_strings)
    else:
        batch_results = []

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
        details.append({
            "sentence": sent,
            "triplet": triplet,
            "query": query_used,
            "cooc_count": count,
            "sentence_reward": penalty,
        })

    return penalties, details


# ---------------------------------------------------------------------------
# Per-token return construction (single completion)
# ---------------------------------------------------------------------------

ALIGNMENT_FALLBACK_THRESHOLD = float(os.environ.get("ALIGNMENT_FALLBACK_THRESHOLD", "0.5"))


def compute_per_token_returns(
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    tokenizer,
    extractor,
    client,
    judge_reward: float,
    format_reward: float,
    beta_judge: float = 0.3,
) -> Tuple[torch.Tensor, Dict]:
    """Compute per-token returns for a single completion.

    G_k(t) = r_corver(sentence(t)) + beta * R_judge + r_format

    Args:
        completion_ids: (T,) token IDs.
        completion_mask: (T,) binary mask.
        tokenizer: HuggingFace tokenizer.
        extractor: QuCoEntityExtractor.
        client: Infigram client.
        judge_reward: Response-level LLM-judge reward.
        format_reward: Format reward (+1 or -1).
        beta_judge: Weight for judge reward in per-token return.

    Returns:
        per_token_returns: (T,) float tensor.
        info: Dict with debug info.
    """
    T = completion_ids.shape[0]
    device = completion_ids.device

    # Global baseline (broadcast to all tokens)
    global_return = beta_judge * judge_reward + format_reward

    # Build token-sentence map
    token_to_sentence, sentences, alignment_rate = build_token_to_sentence_map(
        completion_ids, tokenizer, completion_mask
    )

    info = {
        "alignment_rate": alignment_rate,
        "num_sentences": len(sentences),
        "fallback_to_global": False,
        "sentence_details": [],
    }

    # Fallback: if alignment too poor, use global return only
    if alignment_rate < ALIGNMENT_FALLBACK_THRESHOLD or not sentences:
        per_token_returns = torch.full((T,), global_return, dtype=torch.float32, device=device)
        per_token_returns = per_token_returns * completion_mask.float()
        info["fallback_to_global"] = True
        return per_token_returns, info

    # Compute sentence-level CorVer penalties
    penalties, sentence_details = compute_sentence_corver_penalties(
        sentences, extractor, client
    )
    info["sentence_details"] = sentence_details

    # Add token span info to details
    for sent_idx, detail in enumerate(sentence_details):
        token_indices = (token_to_sentence == sent_idx).nonzero(as_tuple=True)[0]
        if len(token_indices) > 0:
            detail["token_span"] = [token_indices[0].item(), token_indices[-1].item()]
        else:
            detail["token_span"] = None

    # Construct per-token returns
    per_token_returns = torch.full((T,), global_return, dtype=torch.float32, device=device)

    for sent_idx, penalty in enumerate(penalties):
        mask = (token_to_sentence == sent_idx)
        if mask.any():
            # G_i = r_corver_i + beta * R_judge + r_format
            per_token_returns[mask] = penalty + global_return

    # Zero out padding positions
    per_token_returns = per_token_returns * completion_mask.float()

    return per_token_returns, info


# ---------------------------------------------------------------------------
# Batch version
# ---------------------------------------------------------------------------

def compute_batch_per_token_returns(
    completion_ids_batch: torch.Tensor,
    completion_mask_batch: torch.Tensor,
    tokenizer,
    extractor,
    client,
    judge_rewards: List[float],
    format_rewards: List[float],
    beta_judge: float = 0.3,
) -> Tuple[torch.Tensor, List[Dict]]:
    """Compute per-token returns for a batch of completions.

    Args:
        completion_ids_batch: (B, T) token IDs.
        completion_mask_batch: (B, T) binary mask.
        tokenizer: HuggingFace tokenizer.
        extractor: QuCoEntityExtractor.
        client: Infigram client.
        judge_rewards: List[float] of length B.
        format_rewards: List[float] of length B.
        beta_judge: Weight for judge in per-token return.

    Returns:
        per_token_returns: (B, T) float tensor.
        batch_info: List[Dict] with per-sample debug info.
    """
    B, T = completion_ids_batch.shape
    device = completion_ids_batch.device
    per_token_returns = torch.zeros((B, T), dtype=torch.float32, device=device)
    batch_info = []

    fallback_count = 0

    for i in range(B):
        returns_i, info_i = compute_per_token_returns(
            completion_ids=completion_ids_batch[i],
            completion_mask=completion_mask_batch[i],
            tokenizer=tokenizer,
            extractor=extractor,
            client=client,
            judge_reward=judge_rewards[i],
            format_reward=format_rewards[i],
            beta_judge=beta_judge,
        )
        per_token_returns[i] = returns_i
        batch_info.append(info_i)
        if info_i.get("fallback_to_global", False):
            fallback_count += 1

    # Log batch-level stats
    alignment_rates = [info["alignment_rate"] for info in batch_info]
    avg_alignment = sum(alignment_rates) / max(len(alignment_rates), 1)
    logger.info(
        f"Sentence reward batch: avg_alignment={avg_alignment:.3f}, "
        f"fallback={fallback_count}/{B}"
    )

    return per_token_returns, batch_info
