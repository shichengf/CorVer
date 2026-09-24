"""Train the paper's full CorVer method. Run --dry-run for CPU preflight."""

import argparse
import json
import os
from pathlib import Path
from common.io import ROOT, read_config, repo_path, load_rows, freeze_run, save
from common.protocol import SYSTEM_PROMPT, format_reward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", help="Override the training JSON/JSONL path")
    parser.add_argument("--output", help="Override output directory")
    parser.add_argument(
        "--model-path", help="Local directory for the configured base model"
    )
    parser.add_argument("--index-dir", default=os.getenv("CORVER_INDEX_DIR"))
    parser.add_argument(
        "--index-tokenizer-path",
        default=os.getenv("CORVER_INDEX_TOKENIZER_PATH"),
        help="Local directory for the corpus index tokenizer",
    )
    parser.add_argument(
        "--extractor-path", help="Local directory for the pinned extractor"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/data, without GPU or downloads",
    )
    args = parser.parse_args()
    cfg = read_config(args.config)
    data = repo_path(args.data or cfg["data"]["path"])
    output = repo_path(args.output or cfg["output_dir"])
    rows = load_rows(data, cfg["data"]["expected_rows"])
    train = cfg["training"]
    loader_name = cfg["model"].get("loader", "fast_language_model")
    if loader_name not in {"fast_language_model", "fast_model"}:
        raise ValueError(f"Unsupported model loader: {loader_name}")
    if (
        train["per_device_train_batch_size"] * train["gradient_accumulation_steps"]
        != cfg["questions_per_step"] * train["num_generations"]
    ):
        raise ValueError(
            "Single-GPU batch must equal questions_per_step × num_generations"
        )
    if train["scale_rewards"] != "group":
        raise ValueError("Paper Eq. (4) requires group normalization")
    if args.dry_run:
        print(
            json.dumps(
                dict(config=cfg, rows=len(rows), data=str(data), output=str(output)),
                indent=2,
            )
        )
        return
    if not args.index_dir:
        parser.error(
            "Set --index-dir or CORVER_INDEX_DIR to the downloaded corpus index"
        )
    index_tokenizer = cfg["resources"].get("index_tokenizer")
    if not args.index_tokenizer_path and not index_tokenizer:
        parser.error(
            "Set --index-tokenizer-path or CORVER_INDEX_TOKENIZER_PATH "
            "to the local Llama-2 tokenizer"
        )
    index = Path(args.index_dir).expanduser().resolve()
    for name in ["metadata.0", "metaoff.0", "offset.0", "table.0", "tokenized.0"]:
        if not (index / name).is_file():
            raise FileNotFoundError(index / name)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output is not empty; use --resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock = (output / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze_run(output, cfg, data)
    if (output / "complete.json").exists():
        print("This run is already complete.")
        return
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import unsloth
    import torch
    from unsloth import FastLanguageModel, FastModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    from transformers import TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from huggingface_hub import snapshot_download
    from corver.rewards.batch import CachedLocalCounts, VLLMExtractorClient
    from corver.rewards.corpus import InfigramLocalClient
    from corver.training.trainer import make_stepwise_trainer
    from corver.training.token_boundary import install

    torch.set_num_threads(4)
    torch.manual_seed(train["seed"])
    checkpoint = get_last_checkpoint(str(output)) if args.resume else None
    if args.resume and checkpoint is None:
        raise ValueError("No complete Trainer checkpoint to resume")
    resource = cfg["resources"]
    base = args.model_path or snapshot_download(
        cfg["model"]["id"], revision=cfg["model"]["revision"]
    )
    extractor_path = args.extractor_path or snapshot_download(
        resource["extractor"]["id"], revision=resource["extractor"]["revision"]
    )
    tokenizer_path = args.index_tokenizer_path or snapshot_download(
        index_tokenizer["id"], revision=index_tokenizer["revision"]
    )
    os.environ["LLAMA2_TOKENIZER_ID"] = tokenizer_path
    client = CachedLocalCounts(InfigramLocalClient(index_dir=str(index)))
    count, _ = client.count("United States")
    if count is None or count <= 0:
        raise RuntimeError("Corpus index preflight failed")
    initial_calls = 0
    if checkpoint:
        state = json.loads((Path(checkpoint) / "corver_worker.json").read_text())
        initial_calls = state["calls"]
    extractor = VLLMExtractorClient(
        output,
        extractor_path,
        cfg["runtime"]["extractor_gpu_memory_utilization"],
        initial_calls=initial_calls,
    )
    try:
        model_loader = (
            FastModel if loader_name == "fast_model" else FastLanguageModel
        )
        model, tokenizer = model_loader.from_pretrained(
            model_name=str(base),
            fast_inference=True,
            load_in_4bit=False,
            max_lora_rank=cfg["lora"]["r"],
            max_seq_length=train["max_prompt_length"] + train["max_completion_length"],
            gpu_memory_utilization=cfg["runtime"]["policy_gpu_memory_utilization"],
            attn_implementation="sdpa",
            compilation_config=0,
            enforce_eager=True,
        )
        model = model_loader.get_peft_model(
            model,
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=0,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            use_gradient_checkpointing="unsloth",
            random_state=train["seed"],
        )
        adapter_b = [v for k, v in model.named_parameters() if "lora_B" in k]
        if not adapter_b or any(torch.count_nonzero(v).item() for v in adapter_b):
            raise RuntimeError("Expected a fresh zero-initialized LoRA adapter")
        save(
            output / "initialization.json",
            dict(fresh_zero_lora=True, base_revision=cfg["model"]["revision"]),
        )
        tokenizer.pad_token = tokenizer.eos_token
        dataset = Dataset.from_list(
            [
                dict(
                    question=r["question"],
                    best_answer=r["best_answer"],
                    prompt=tokenizer.apply_chat_template(
                        [
                            dict(role="system", content=SYSTEM_PROMPT),
                            dict(role="user", content=r["question"]),
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        date_string=cfg["runtime"]["prompt_date"],
                    ),
                )
                for r in rows
            ]
        )

        def response_format(completions, **kwargs):
            return [
                format_reward(t, len(tokenizer.encode(t, add_special_tokens=False)))
                for t in completions
            ]

        class WorkerState(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                save(
                    Path(args.output_dir)
                    / f"checkpoint-{state.global_step}"
                    / "corver_worker.json",
                    dict(calls=extractor.calls),
                )

        config = GRPOConfig(
            output_dir=str(output),
            **train,
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=cfg["runtime"]["policy_gpu_memory_utilization"],
            report_to="none",
            logging_steps=1,
            save_steps=10,
            save_total_limit=2,
            remove_unused_columns=False,
        )
        trainer = make_stepwise_trainer(
            GRPOTrainer,
            extractor=extractor,
            count_client=client,
            judge_weight=cfg["judge_weight"],
            model=model,
            reward_funcs=[response_format],
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[WorkerState()],
        )
        install(trainer)
        save(output / "effective_training_args.json", config.to_dict())
        save(
            output / "resources.json",
            dict(
                model=str(base),
                extractor=str(extractor_path),
                index=str(index),
                tokenizer=tokenizer_path,
            ),
        )
        trainer.train(resume_from_checkpoint=checkpoint)
        if not any(torch.count_nonzero(v).item() for v in adapter_b):
            raise RuntimeError("No adapter update; refusing successful completion")
        trainer.save_model(str(output))
        tokenizer.save_pretrained(str(output))
        save(
            output / "complete.json",
            dict(steps=trainer.state.global_step, extractor_calls=extractor.calls),
        )
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
