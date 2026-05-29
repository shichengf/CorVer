"""CorVer binary reward scorer for TRL GRPOTrainer.

Wraps the CorVer sentence-level co-occurrence hallucination detection
into TRL's reward function interface. Returns binary reward: 1 (clean)
or 0 (hallucination detected).
"""

import logging
import os
import re
import traceback
from typing import List, Optional

from .logging_utils import save_json_output, get_timestamp, get_output_timestamp

logger = logging.getLogger(__name__)


class CorVerScorer:
    """CorVer binary reward scorer for TRL GRPOTrainer.

    Usage::

        scorer = CorVerScorer()
        rewards = scorer.corver_reward_func(prompts, completions)
    """

    def __init__(
        self,
        scale: float = None,
        log_dir: str = None,
        extractor_model: str = "ZhishanQ/QuCo-extractor-0.5B",
        extractor_device_map: str = "auto",
        infigram_index: str = "v4_rpj_llama_s4",
    ):
        """Initialize the CorVer scorer.

        Args:
            scale: Scale factor for final reward. Falls back to CORVER_REWARD_SCALE env var, then 1.0.
            log_dir: Directory to save JSON logs. None = no file logging.
            extractor_model: HuggingFace model name for entity extractor.
            extractor_device_map: Device map for extractor model.
            infigram_index: Infini-gram corpus index name.
        """
        self._extractor = None
        self._client = None
        self._initialized = False

        self.scale = scale if scale is not None else float(os.environ.get("CORVER_REWARD_SCALE", "1.0"))
        self.log_dir = log_dir
        self.extractor_model = extractor_model
        self.extractor_device_map = extractor_device_map
        self.infigram_index = infigram_index

    def _ensure_initialized(self):
        """Lazy initialization of QuCo extractor and Infini-gram client."""
        if self._initialized:
            return

        try:
            from . import get_extractor, get_client
            self._extractor = get_extractor(
                model_name=self.extractor_model,
                device_map=self.extractor_device_map,
            )
            self._client = get_client(index=self.infigram_index)
            self._initialized = True
            logger.info("CorVerScorer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CorVerScorer: {e}")
            self._initialized = True  # Don't retry on every call

    def _get_log_dir(self) -> Optional[str]:
        """Get the output directory for JSON logs."""
        if self.log_dir is None:
            return None
        timestamp = get_output_timestamp()
        out = os.path.join(self.log_dir, timestamp)
        os.makedirs(out, exist_ok=True)
        return out

    def corver_reward_func(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        """Compute binary CorVer reward for a batch of completions.

        Compatible with TRL GRPOTrainer's reward function interface.

        Args:
            prompts: List of input prompts.
            completions: List of model completions.
            **kwargs: Additional dataset columns (ignored).

        Returns:
            List of reward floats (0.0 or scale).
        """
        self._ensure_initialized()

        rewards = [0.0] * len(completions)
        log_entries = []
        log_output_dir = self._get_log_dir()

        if self._extractor is None or self._client is None:
            logger.error("QuCo extractor or client not available, returning 0 for all samples")
            for i in range(len(completions)):
                log_entries.append({
                    "sample_index": i,
                    "error": "CorVer initialization failed",
                    "reward": 0.0,
                    "timestamp": get_timestamp(),
                })
            if log_output_dir:
                save_json_output({"corver_results": log_entries}, "corver_reward", log_output_dir)
            return rewards

        from .reward import compute_corver_reward

        for i, completion in enumerate(completions):
            try:
                # Check for refusal in <answer> tag — refusal gets 0
                answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL | re.IGNORECASE)
                is_refusal = False
                if answer_match:
                    answer_text = answer_match.group(1).strip().lower()
                    refusal_patterns = ["i don't know", "i don\u2019t know", "i do not know"]
                    is_refusal = any(p in answer_text for p in refusal_patterns)

                if is_refusal:
                    score = 0.0
                    result = {"score": 0.0, "num_sentences_checked": 0, "num_hallucinated": 0}
                else:
                    # Extract <think> content for evaluation
                    think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
                    if think_match:
                        eval_text = think_match.group(1).strip()
                    else:
                        eval_text = completion.strip()

                    result = compute_corver_reward(
                        response_str=eval_text,
                        extractor=self._extractor,
                        client=self._client,
                    )
                    score = result.get("score", 0.0)

                rewards[i] = score * self.scale

                log_entries.append({
                    "sample_index": i,
                    "prompt": prompts[i] if i < len(prompts) else "",
                    "completion": completion,
                    "eval_text_source": "refusal" if is_refusal else ("think_tag" if not is_refusal and answer_match else "full_completion"),
                    "is_refusal": is_refusal,
                    "raw_score": score,
                    "scale": self.scale,
                    "reward": rewards[i],
                    "num_sentences_checked": result.get("num_sentences_checked", 0),
                    "num_hallucinated": result.get("num_hallucinated", 0),
                    "timestamp": get_timestamp(),
                })

            except Exception as e:
                logger.error(f"CorVer scoring error for sample {i}: {e}")
                rewards[i] = 0.0
                log_entries.append({
                    "sample_index": i,
                    "prompt": prompts[i] if i < len(prompts) else "",
                    "completion": completion,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "reward": 0.0,
                    "timestamp": get_timestamp(),
                })

        if log_output_dir:
            save_json_output({"corver_results": log_entries}, "corver_reward", log_output_dir)

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        logger.info(f"CorVer reward: avg={avg_reward:.4f}, scale={self.scale}")

        return rewards
