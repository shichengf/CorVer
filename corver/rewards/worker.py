"""Standalone vLLM/CUDA Graph reward worker, isolated from Unsloth patches."""

import argparse
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--initial-calls", type=int, default=0)
    ap.add_argument("--memory-utilization", type=float, default=0.065)
    args = ap.parse_args()
    os.umask(0o077)
    import torch

    torch.set_num_threads(4)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from corver.rewards.parsing import parse_case

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    template = json.loads(Path(__file__).with_name("prompts.json").read_text())[
        "entity_extraction_prompt"
    ]
    generation = dict(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.1,
        eos_token_id=[151645, 151643],
    )
    eos = generation["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.memory_utilization,
        max_model_len=4096,
        max_num_seqs=64,
        max_num_batched_tokens=4096,
        seed=42,
        enforce_eager=False,
        enable_prefix_caching=False,
        generation_config="vllm",
    )
    socket = Path(args.socket)
    calls = args.initial_calls
    listener = Listener(str(socket), family="AF_UNIX")
    Path(args.output, "extractor_ready.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "model": args.model,
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "default"),
                "memory_utilization": args.memory_utilization,
                "backend": "vLLM CUDA Graph",
                "sampling": generation,
                "seed_policy": "42 + reward batch index",
            },
            indent=2,
        )
    )
    try:
        with listener.accept() as conn:
            while True:
                try:
                    data = json.loads(conn.recv_bytes())
                except EOFError:
                    break
                if data.get("stop"):
                    break
                try:
                    started = time.perf_counter()
                    prompts = [
                        tok.apply_chat_template(
                            [{"role": "user", "content": template.format(s)}],
                            add_generation_prompt=True,
                            tokenize=True,
                        )
                        for s in data["sentences"]
                    ]
                    if any(len(ids) + 256 > 4096 for ids in prompts):
                        raise ValueError(
                            "Extraction prompt exceeds bound; refusing silent truncation"
                        )
                    params = SamplingParams(
                        temperature=(
                            generation["temperature"]
                            if generation.get("do_sample")
                            else 0.0
                        ),
                        top_p=generation.get("top_p", 1.0),
                        top_k=generation.get("top_k", -1),
                        repetition_penalty=generation.get("repetition_penalty", 1.0),
                        seed=42 + calls,
                        stop_token_ids=eos,
                        max_tokens=256,
                    )
                    outputs = llm.generate(
                        [{"prompt_token_ids": ids} for ids in prompts],
                        params,
                        use_tqdm=False,
                    )
                    rows = []
                    for output in outputs:
                        text = output.outputs[0].text.strip()
                        if text.startswith("entities:"):
                            text = text.replace("entities:", "", 1).strip()
                        try:
                            row = parse_case(text)
                        except Exception:
                            row = []
                        rows.append(row if isinstance(row, list) else [])
                    conn.send_bytes(
                        json.dumps(
                            {
                                "rows": rows,
                                "timing": {
                                    "seconds": time.perf_counter() - started,
                                    "sentences": len(rows),
                                    "request": calls,
                                    "tokens": sum(
                                        len(o.outputs[0].token_ids) for o in outputs
                                    ),
                                },
                            }
                        ).encode()
                    )
                    calls += 1
                except Exception:
                    conn.send_bytes(
                        json.dumps({"error": traceback.format_exc()}).encode()
                    )
                    raise
    finally:
        listener.close()
        socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
