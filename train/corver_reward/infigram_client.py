"""Infini-gram API client for CorVer reward scoring.

Mirrors the `get_infini_gram_count()` function from QuCo-RAG
(https://github.com/ZhishanQ/QuCo-RAG) with retry logic, caching, and batch support.
"""

import json
import logging
import os
import threading
import time

import requests

logger = logging.getLogger(__name__)

API_ENDPOINT = "https://api.infini-gram.io/"
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2
REQUEST_TIMEOUT = 30


class InfigramClient:
    """HTTP client for the Infini-gram online API with query caching.

    The Infini-gram public API has strict rate limits and will return HTTP 403
    for rapid sequential requests. This client includes:
    - In-memory LRU cache to avoid duplicate queries
    - Optional disk persistence so cache survives restarts
    - Configurable rate limiting between requests
    - Aggressive backoff on 403/429 errors
    """

    def __init__(
        self,
        index: str = "v4_rpj_llama_s4",
        max_diff_tokens: int = 1000,
        max_clause_freq: int = 500000,
        max_workers: int = 1,
        min_request_interval: float = 2.0,
        cache_file: str = None,
    ):
        self.index = index
        self.max_diff_tokens = max_diff_tokens
        self.max_clause_freq = max_clause_freq
        self.max_workers = max_workers
        self.min_request_interval = min_request_interval
        self._last_request_time = 0.0

        # Query cache: query_string -> count (int or None)
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._cache_file = cache_file
        self._cache_dirty = False
        self._consecutive_api_calls = 0
        self._burst_limit = 30  # Pause after this many consecutive API calls
        self._burst_pause = 10  # Seconds to pause after burst

        # Load cache from disk if available
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached queries from {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load cache from {cache_file}: {e}")

    def _save_cache(self):
        """Persist cache to disk if a cache file is configured."""
        if not self._cache_file or not self._cache_dirty:
            return
        try:
            os.makedirs(os.path.dirname(self._cache_file) or ".", exist_ok=True)
            with open(self._cache_file, "w") as f:
                json.dump(self._cache, f)
            self._cache_dirty = False
        except Exception as e:
            logger.warning(f"Failed to save cache to {self._cache_file}: {e}")

    def count(self, query: str) -> tuple:
        """Send a count query to the Infini-gram API (with cache).

        Args:
            query: The query string. Use 'AND'/'OR' for CNF co-occurrence queries.

        Returns:
            Tuple of (count, engine_latency_seconds).
            Returns (None, None) on failure.
        """
        # Check cache first
        with self._cache_lock:
            if query in self._cache:
                cached = self._cache[query]
                return (cached, 0.0)

        payload = {
            "index": self.index,
            "query_type": "count",
            "query": query,
        }

        # Add CNF parameters when query contains AND/OR operators
        if "AND" in query or "OR" in query:
            payload["max_diff_tokens"] = self.max_diff_tokens
            payload["max_clause_freq"] = self.max_clause_freq

        last_error = None
        for attempt in range(MAX_RETRIES):
            # Rate limiting: wait between requests
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)

            # Burst limiting: pause after N consecutive calls
            self._consecutive_api_calls += 1
            if self._consecutive_api_calls >= self._burst_limit:
                logger.info(f"Infigram burst pause: {self._burst_pause}s after {self._burst_limit} calls")
                time.sleep(self._burst_pause)
                self._consecutive_api_calls = 0

            self._last_request_time = time.time()

            try:
                response = requests.post(
                    API_ENDPOINT,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )

                if response.status_code == 403:
                    # IP banned - back off aggressively
                    delay = INITIAL_RETRY_DELAY * (2 ** (attempt + 2))  # 8, 16, 32, 64, 128s
                    last_error = f"HTTP 403 forbidden on attempt {attempt + 1}/{MAX_RETRIES}, waiting {delay}s"
                    logger.warning(f"Infini-gram API: {last_error}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(delay)
                    continue

                if response.status_code == 429:
                    # Rate limited - back off with exponential delay
                    delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    last_error = f"HTTP 429 rate limited on attempt {attempt + 1}/{MAX_RETRIES}, waiting {delay}s"
                    logger.warning(f"Infini-gram API: {last_error}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(delay)
                    continue

                if response.status_code >= 500:
                    last_error = f"HTTP {response.status_code} on attempt {attempt + 1}/{MAX_RETRIES}"
                    logger.warning(f"Infini-gram API: {last_error}")
                    if attempt < MAX_RETRIES - 1:
                        delay = INITIAL_RETRY_DELAY * (attempt + 1)
                        time.sleep(delay)
                    continue

                response.raise_for_status()
                result = response.json()

                if "error" in result:
                    logger.error(f"Infini-gram API error: {result['error']}")
                    return None, None

                count = result.get("count")
                if count is None:
                    if isinstance(result, (int, float)):
                        count = result
                    else:
                        logger.warning(f"Unexpected API response format: {result}")
                        return None, None

                engine_latency = None
                if "latency" in result:
                    engine_latency = result["latency"] / 1000.0  # ms -> seconds

                # Cache the result
                with self._cache_lock:
                    self._cache[query] = count
                    self._cache_dirty = True

                return count, engine_latency

            except requests.exceptions.RequestException as e:
                last_error = f"HTTP request failed: {e}"
                logger.warning(f"Infini-gram API: {last_error}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
                continue
            except ValueError as e:
                last_error = f"Failed to parse JSON: {e}"
                logger.error(f"Infini-gram API: {last_error}")
                break

        logger.warning(f"Infini-gram API call failed after {MAX_RETRIES} retries. Last error: {last_error}")
        return None, None

    def count_batch(self, queries: list) -> list:
        """Send multiple count queries sequentially with caching.

        Queries that are already cached are returned immediately.
        Only uncached queries hit the API (serially with rate limiting).

        Args:
            queries: List of query strings.

        Returns:
            List of (count, engine_latency_seconds) tuples.
        """
        if not queries:
            return []

        results = []
        api_calls = 0
        cache_hits = 0
        for q in queries:
            # Check cache inside count() handles this, but we can
            # log stats at the batch level
            with self._cache_lock:
                if q in self._cache:
                    cache_hits += 1
                    results.append((self._cache[q], 0.0))
                    continue
            api_calls += 1
            results.append(self.count(q))

        if api_calls > 0:
            logger.info(
                f"Infigram batch: {len(queries)} queries, "
                f"{cache_hits} cached, {api_calls} API calls"
            )
            # Persist cache after each batch
            self._save_cache()

        return results
