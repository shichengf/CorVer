"""Local Infini-gram engine client using direct mmap access.

Drop-in replacement for InfigramClient that uses a local suffix array
index instead of the remote API. Eliminates network latency and rate limits.
"""

import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

# Allow overriding the path to a local clone of github.com/liujch1998/infini-gram
# (only needed if the package isn't already pip-installed).
INFINI_GRAM_PKG = os.environ.get("INFINI_GRAM_PKG")
if INFINI_GRAM_PKG and INFINI_GRAM_PKG not in sys.path:
    sys.path.insert(0, INFINI_GRAM_PKG)


class InfigramLocalClient:
    """Local engine client compatible with InfigramClient.count() interface.

    Wraps InfiniGramEngine to provide the same (count, latency) return
    signature expected by corver_reward.reward.compute_corver_reward.
    """

    def __init__(
        self,
        index_dir="./infigram_index",
        max_diff_tokens=1000,
        max_clause_freq=500000,
        **kwargs,
    ):
        from infini_gram.engine import InfiniGramEngine
        from transformers import AutoTokenizer

        self.max_diff_tokens = max_diff_tokens
        self.max_clause_freq = max_clause_freq

        # The infigram index is tokenized with Llama-2's vocabulary, so reward
        # queries must use the same tokenizer. Point LLAMA2_TOKENIZER_ID at the
        # gated HF repo (after `huggingface-cli login`) or at a local directory
        # containing tokenizer.model + tokenizer_config.json.
        tokenizer_id = os.environ.get("LLAMA2_TOKENIZER_ID", "meta-llama/Llama-2-7b-hf")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            token=os.environ.get("HF_TOKEN"),
            use_fast=False,
            add_bos_token=False,
            add_eos_token=False,
        )

        self.engine = InfiniGramEngine(
            s3_names=[],
            index_dir=[index_dir],
            eos_token_id=self.tokenizer.eos_token_id,
            vocab_size=self.tokenizer.vocab_size,
            version=4,
            token_dtype="u16",
            max_clause_freq=max_clause_freq,
            max_diff_tokens=max_diff_tokens,
        )
        logger.info("InfigramLocalClient initialized with index at %s", index_dir)

    def _tokenize(self, text):
        """Tokenize text, stripping the leading space token (LLaMA quirk)."""
        ids = self.tokenizer.encode(text)
        if ids and ids[0] == 29871:
            ids = ids[1:]
        return ids

    def count(self, query):
        """Send a count query to the local engine.

        Args:
            query: The query string. Use 'AND'/'OR' for CNF co-occurrence.

        Returns:
            Tuple of (count, engine_latency_seconds).
            Returns (None, None) on error.
        """
        start = time.time()
        try:
            if " AND " in query or " OR " in query:
                clauses = query.split(" AND ")
                termss = [clause.split(" OR ") for clause in clauses]
                cnf = [[self._tokenize(term) for term in terms] for terms in termss]
                result = self.engine.count_cnf(
                    cnf=cnf,
                    max_clause_freq=self.max_clause_freq,
                    max_diff_tokens=self.max_diff_tokens,
                )
            else:
                input_ids = self._tokenize(query)
                result = self.engine.count(input_ids=input_ids)

            elapsed = time.time() - start

            if "error" in result:
                logger.warning("Local engine error: %s", result["error"])
                return None, None

            return result["count"], elapsed
        except Exception as e:
            logger.warning("Local engine exception: %s", e)
            return None, None

    def count_batch(self, queries):
        """Send multiple count queries.

        Args:
            queries: List of query strings.

        Returns:
            List of (count, engine_latency_seconds) tuples.
        """
        return [self.count(q) for q in queries]
