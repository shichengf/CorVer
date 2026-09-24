"""Batched extraction, successful-count caching and token rewards."""

import atexit
from collections import OrderedDict
import json
from multiprocessing.connection import Client
import os
import sys
import tempfile
from pathlib import Path
import subprocess
import time

import torch


class CachedLocalCounts:
    def __init__(self, client, capacity=100000):
        self.client = client
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = self.misses = 0

    def count(self, query):
        key = (query, self.client.max_clause_freq, self.client.max_diff_tokens)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key], 0.0
        self.misses += 1
        value, elapsed = self.client.count(query)
        if value is not None:
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return value, elapsed

    def count_batch(self, queries):
        return [self.count(query) for query in queries]


class VLLMExtractorClient:
    def __init__(
        self,
        output,
        model_path,
        memory_utilization=0.065,
        initial_calls=0,
    ):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        # Short AF_UNIX pathname; PID plus randomness avoids collisions across jobs.
        self.socket_path = (
            Path(tempfile.gettempdir())
            / f"corver_{os.getpid()}_{os.urandom(4).hex()}.sock"
        )
        self.log = (self.output / "extractor_server.log").open("a")
        self.conn = None
        self.proc = None
        self.last_timing = {}
        self.calls = initial_calls
        env = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "OMP_NUM_THREADS": "4",
        }
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "corver.rewards.worker",
                "--socket",
                str(self.socket_path),
                "--model",
                str(model_path),
                "--output",
                str(self.output),
                "--memory-utilization",
                str(memory_utilization),
                "--initial-calls",
                str(initial_calls),
            ],
            env=env,
            cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        atexit.register(self.close)
        deadline = time.monotonic() + 600
        while not self.socket_path.exists():
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"CorVer vLLM server exited {self.proc.returncode}; see server log"
                )
            if time.monotonic() > deadline:
                self.close()
                raise TimeoutError("CorVer vLLM server startup timed out")
            time.sleep(1)
        self.conn = Client(str(self.socket_path), family="AF_UNIX")

    def extract_triplets_batch(self, sentences, batch_size=64):
        if not sentences:
            return []
        self.conn.send_bytes(
            json.dumps({"sentences": sentences}, ensure_ascii=False).encode()
        )
        if not self.conn.poll(600):
            raise TimeoutError("CorVer vLLM extraction timed out")
        result = json.loads(self.conn.recv_bytes())
        if "error" in result:
            raise RuntimeError(result["error"])
        if len(result["rows"]) != len(sentences):
            raise ValueError("Extractor dropped sentence rows")
        self.last_timing = result["timing"]
        self.calls = result["timing"]["request"] + 1
        return result["rows"]

    def close(self):
        if self.conn is not None:
            try:
                self.conn.send_bytes(b'{"stop":true}')
                self.conn.close()
            except (OSError, EOFError):
                pass
            self.conn = None
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        self.socket_path.unlink(missing_ok=True)
        if not self.log.closed:
            self.log.close()


def fast_batch_returns(
    completion_ids_batch,
    completion_mask_batch,
    tokenizer,
    extractor,
    client,
    judge_rewards,
    format_rewards,
    beta_judge=1.0,
):
    from corver.rewards.alignment import (
        map_generated_sentences as build_token_to_sentence_map,
    )
    from corver.rewards.sentence import compute_sentence_corver_penalties

    started = time.perf_counter()
    ids = completion_ids_batch.detach().cpu()
    masks = completion_mask_batch.detach().cpu()
    mappings, infos, ranges, all_sentences = [], [], [], []
    for row, mask in zip(ids, masks):
        mapping, sentences, rate = build_token_to_sentence_map(row, tokenizer, mask)
        fallback = not sentences
        mappings.append(mapping)
        infos.append(
            {
                "alignment_rate": rate,
                "num_sentences": len(sentences),
                "fallback_to_global": bool(fallback),
                "sentence_details": [],
            }
        )
        start = len(all_sentences)
        if not fallback:
            all_sentences.extend(sentences)
        ranges.append((start, len(all_sentences)))
    aligned = time.perf_counter()
    penalties, details = compute_sentence_corver_penalties(
        all_sentences, extractor, client
    )
    scored = time.perf_counter()
    results = []
    for i, (start, end) in enumerate(ranges):
        global_return = beta_judge * judge_rewards[i] + format_rewards[i]
        info = infos[i]
        mapping = mappings[i]
        if info["fallback_to_global"]:
            result = torch.full(masks[i].shape, global_return, dtype=torch.float32)
        else:
            info["sentence_details"] = details[start:end]
            for sent_id, detail in enumerate(info["sentence_details"]):
                tokens = (mapping == sent_id).nonzero(as_tuple=True)[0]
                detail["token_span"] = (
                    [tokens[0].item(), tokens[-1].item()] if len(tokens) else None
                )
            values = torch.tensor(
                [global_return] + [p + global_return for p in penalties[start:end]],
                dtype=torch.float32,
            )
            result = values[mapping + 1]
        results.append(result * masks[i].float())
    output = torch.stack(results).to(completion_ids_batch.device)
    timing = {
        "batch": len(ids),
        "sentences": len(all_sentences),
        "alignment_s": aligned - started,
        "reward_s": scored - aligned,
        "construct_s": time.perf_counter() - scored,
        "extractor": getattr(extractor, "last_timing", {}),
        "cache_hits": getattr(client, "hits", None),
        "cache_misses": getattr(client, "misses", None),
    }
    print("CORVER_REWARD_TIMING", json.dumps(timing), flush=True)
    return output, infos
