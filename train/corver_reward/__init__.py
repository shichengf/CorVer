"""CorVer: Corpus-grounded binary hallucination detection reward.

Standalone package for computing CorVer-based factuality rewards using
Infini-gram corpus verification. Works with TRL GRPOTrainer, verl, or standalone.
Returns binary reward: 1 (no hallucination) or 0 (hallucination detected).

Usage::

    # Standalone scoring
    from corver_reward import compute_score
    result = compute_score("Einstein was born in Ulm.")
    print(result["score"])  # 0 or 1

    # TRL GRPOTrainer
    from corver_reward import CorVerScorer
    scorer = CorVerScorer()
    rewards = scorer.corver_reward_func(prompts, completions)

    # Low-level access
    from corver_reward import get_extractor, get_client
    from corver_reward.reward import compute_corver_reward
    extractor = get_extractor()
    client = get_client()
    result = compute_corver_reward(response_str=text,
                                 extractor=extractor, client=client)

Dependencies: transformers, torch
"""

__version__ = "0.1.0"

import logging
import os
import threading

logger = logging.getLogger(__name__)

_extractor = None
_client = None
_init_lock = threading.Lock()


def get_extractor(model_name="ZhishanQ/QuCo-extractor-0.5B", device_map="auto"):
    """Lazy-initialize the QuCo entity extractor (singleton)."""
    global _extractor
    if _extractor is None:
        with _init_lock:
            if _extractor is None:
                from .entity_extractor import QuCoEntityExtractor
                _extractor = QuCoEntityExtractor(
                    model_name=model_name,
                    device_map=device_map,
                )
    return _extractor


def get_client(index="v4_rpj_llama_s4", max_diff_tokens=1000, max_clause_freq=500000,
               cache_file=None):
    """Lazy-initialize the Infini-gram client (singleton).

    Defaults to local suffix array engine (no network, millisecond latency).
    Set env INFIGRAM_USE_REMOTE=1 to fall back to the remote API.
    Configure the local index path with INFIGRAM_LOCAL_INDEX_DIR.

    Args:
        cache_file: Path to JSON file for persisting query cache across restarts.
            Only used in remote mode. Defaults to ./infigram_cache.json.
    """
    global _client
    if _client is None:
        with _init_lock:
            if _client is None:
                use_remote = os.environ.get("INFIGRAM_USE_REMOTE", "0") == "1"
                if use_remote:
                    from .infigram_client import InfigramClient
                    if cache_file is None:
                        cache_file = os.path.join(
                            os.path.dirname(__file__), "infigram_cache.json"
                        )
                    _client = InfigramClient(
                        index=index,
                        max_diff_tokens=max_diff_tokens,
                        max_clause_freq=max_clause_freq,
                        cache_file=cache_file,
                    )
                else:
                    from .infigram_local_client import InfigramLocalClient
                    local_index_dir = os.environ.get(
                        "INFIGRAM_LOCAL_INDEX_DIR",
                        "./infigram_index",
                    )
                    _client = InfigramLocalClient(
                        index_dir=local_index_dir,
                        max_diff_tokens=max_diff_tokens,
                        max_clause_freq=max_clause_freq,
                    )
    return _client


def reset():
    """Clear singleton instances. Useful for testing or reconfiguration."""
    global _extractor, _client
    with _init_lock:
        _extractor = None
        _client = None


from .reward import compute_corver_reward


def compute_score(
    response="",
    *,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    ternary_tuple_threshold=1,
    extractor_model="ZhishanQ/QuCo-extractor-0.5B",
    extractor_device_map="auto",
    infigram_index="v4_rpj_llama_s4",
    **kwargs,
) -> dict:
    """Binary CorVer reward scoring.

    Accepts both standalone and verl-compatible calling conventions::

        # Standalone
        compute_score("Einstein was born in Ulm.")

        # verl-compatible
        compute_score(solution_str=resp, ground_truth=gt, extra_info=info)

    Args:
        response: The model's generated response text.
        solution_str: verl alias for response.
        ground_truth: verl alias (ignored for scoring, kept for API compat).
        extra_info: verl dict (ignored, kept for API compat).
        ternary_tuple_threshold: Co-occurrence count threshold (default 1).
        extractor_model: HuggingFace model name for entity extractor.
        extractor_device_map: Device map for extractor model.
        infigram_index: Infini-gram corpus index name.
        **kwargs: Ignored (e.g., data_source from verl).

    Returns:
        Dict with keys: score (0 or 1), num_sentences_checked,
        num_hallucinated, details.
    """
    response_str = solution_str if solution_str is not None else response

    extractor = get_extractor(model_name=extractor_model, device_map=extractor_device_map)
    client = get_client(index=infigram_index)

    return compute_corver_reward(
        response_str=response_str,
        extractor=extractor,
        client=client,
        ternary_tuple_threshold=ternary_tuple_threshold,
    )


from .scorer import CorVerScorer

__all__ = [
    "compute_score",
    "compute_corver_reward",
    "CorVerScorer",
    "get_extractor",
    "get_client",
    "reset",
]
