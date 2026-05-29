"""Stepwise GRPO training entry point — OLMo-2 variant.

Identical to main_stepwise.py except uses unsloth.FastModel (Auto dispatch)
instead of FastLanguageModel. FastLanguageModel has hardcoded Llama/Qwen/Mistral
dispatch and doesn't route OLMo-2; FastModel falls back to HF AutoModel which
picks up Olmo2ForCausalLM (transformers >= 4.54) + vLLM olmo2 executor.
"""

import unsloth  # Must be first — triggers GRPOTrainer patch
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from unsloth import FastModel
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

# Local imports
from reward_function import format_reward_func, get_reward_output_dir
from utils import create_swanlab_callback_from_yaml, setup_logging, get_default_config_path
from stepwise_grpo_trainer import make_stepwise_trainer

# Setup logging
logger = setup_logging()

# ---------------------------------------------------------------------------
# Stepwise reward mode
# ---------------------------------------------------------------------------
STEPWISE_REWARD_MODE = os.environ.get("STEPWISE_REWARD_MODE", "full")
BETA_JUDGE = float(os.environ.get("JUDGE_REWARD_WEIGHT", "1.0"))
PILOT_NUM_SAMPLES = int(os.environ.get("PILOT_NUM_SAMPLES", "0"))  # 0 = use all

logger.info(f"Stepwise mode: {STEPWISE_REWARD_MODE}, beta_judge: {BETA_JUDGE}")

# ---------------------------------------------------------------------------
# Dataset arguments (same as main.py)
# ---------------------------------------------------------------------------

@dataclass
class DatasetFieldMapping:
    question_field: str = "question"
    title_field: Optional[str] = "title"
    best_answer_field: Optional[str] = "answers"
    correct_answers_field: Optional[str] = None
    incorrect_answers_field: Optional[str] = None
    delimiter: str = ";"
    keep_all_fields: bool = False


@dataclass
class DatasetArguments:
    dataset_id_or_path: str = None
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None
    field_mapping: DatasetFieldMapping = field(default_factory=DatasetFieldMapping)


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def process_dataset_fields(example, field_mapping):
    processed = {}
    question = example.get(field_mapping.question_field, "")
    processed["question"] = question

    if field_mapping.title_field is not None:
        title = example.get(field_mapping.title_field, "")
        processed["title"] = title

    if field_mapping.best_answer_field is not None:
        best_answer = example.get(field_mapping.best_answer_field, "")
        processed["best_answer"] = best_answer if best_answer else ""

    if field_mapping.keep_all_fields:
        for key, value in example.items():
            if key not in processed:
                processed[key] = value

    return processed


# ---------------------------------------------------------------------------
# Anti-reward-hack generation prompt
# ---------------------------------------------------------------------------

STEPWISE_SYSTEM_PROMPT = (
    "Answer using this exact format:\n"
    "<think>brief reasoning</think>\n"
    "<answer>direct answer</answer>\n\n"
    "Rules:\n"
    "- Keep the entire response under 1000 tokens.\n"
    "- In <think> AND <answer>, always use full entity names; never pronouns "
    "(he, she, it, they, this, that, them, his, her, its, their). "
    "Repeat the entity name each sentence.\n"
    "- Do not loop or repeat the same point or phrase.\n"
    "- If you genuinely do not know, reply exactly: <think>unknown</think><answer>I don't know</answer>"
)


def generate_prompt(example, tokenizer):
    question = example["question"]

    prompt_prefix = [
        {"role": "system", "content": STEPWISE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    result = {
        "prompt": tokenizer.apply_chat_template(
            prompt_prefix,
            tokenize=False,
            add_generation_prompt=True,
        ),
    }

    for key in ["best_answer", "title", "question"]:
        if key in example:
            result[key] = example[key]

    return result


# ---------------------------------------------------------------------------
# Initialize CorVer and Judge
# ---------------------------------------------------------------------------

def init_corver_components():
    """Lazy-initialize QuCo extractor and Infigram client."""
    if STEPWISE_REWARD_MODE in ("full", "corver_only"):
        from corver_reward import get_extractor, get_client
        extractor = get_extractor()
        client = get_client()
        logger.info("CorVer components initialized (extractor + local Infigram)")
        return extractor, client
    return None, None


JUDGE_TYPE = os.environ.get("JUDGE_TYPE", "string_match")  # "string_match" or "llm"


def init_judge():
    """Initialize judge if needed."""
    if STEPWISE_REWARD_MODE in ("full", "judge_only"):
        if JUDGE_TYPE == "llm":
            from llm_judge import LLMJudge
            judge = LLMJudge(
                cache_dir=os.path.join(get_reward_output_dir(), "judge_cache")
            )
            logger.info("Using LLM judge")
        else:
            from string_match_judge import StringMatchJudge
            judge = StringMatchJudge()
            logger.info("Using string-match judge")
        return judge
    return None


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def grpo_function(
    model_args: ModelConfig,
    dataset_args: DatasetArguments,
    training_args: GRPOConfig,
    callbacks: List,
):
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Training parameters: {training_args}")
    logger.info(f"Stepwise reward mode: {STEPWISE_REWARD_MODE}")

    # Load model and tokenizer (FastModel = Auto dispatch for non-Llama/Qwen/Mistral models)
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        fast_inference=True,
        load_in_4bit=False,
        max_lora_rank=model_args.lora_r,
        max_seq_length=training_args.max_prompt_length + training_args.max_completion_length,
        gpu_memory_utilization=training_args.vllm_gpu_memory_utilization,
        attn_implementation=model_args.attn_implementation,
    )

    # Manually attach Unsloth's save_lora/load_lora to the base model.
    # FastModel's Auto dispatch path for OLMo-2 doesn't run the post-load hook
    # that normally attaches these (only fired for Llama/Qwen/Mistral path).
    # The Unsloth GRPOTrainer (UnslothGRPOTrainer.py:3027) calls
    # self.model.load_lora('grpo_trainer_lora_model', load_tensors=True)
    # to push LoRA into vLLM colocate inference; without it, training crashes.
    if not hasattr(model, "load_lora"):
        import functools
        from unsloth_zoo.vllm_utils import save_lora as _save_lora, load_lora as _load_lora
        model.save_lora = functools.partial(_save_lora, model)
        model.load_lora = functools.partial(_load_lora, model)

    # Configure PEFT model directly via peft library.
    # Bypass FastModel.get_peft_model: it dispatches non-Llama/Qwen/Mistral
    # models to FastVisionModel which rejects fast_inference (text-layers-only).
    # OLMo-2 is a text model — peft.LoraConfig + get_peft_model works fine.
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    # Keep LoRA in fp32 (PEFT default). The "expected Float / found BFloat16"
    # error at F.linear is fixed by wrapping the forward pass in
    # torch.autocast(bf16) inside StepwiseGRPOTrainer.compute_loss instead.
    # Casting LoRA params to bf16 destroys numerical stability:
    #   - bf16 epsilon (~7.8e-3) >> lr*grad (~3e-6) → updates are stochastically
    #     rounded to 0 or to ±epsilon, effective lr noise floor ~2600× higher
    #   - GRPO log-ratio (log_p_new - log_p_ref) suffers catastrophic cancellation
    #     in bf16 → artifactual KL → exploding loss in 5-10 steps.
    # Verified: lr=3e-6 + β=0.01 + bf16 LoRA still crashed at step 8 (KL→636).
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    try:
        dataset = load_dataset(
            'json',
            data_files=dataset_args.dataset_id_or_path,
            split=dataset_args.dataset_splits,
        )
        logger.info(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    field_mapping = dataset_args.field_mapping
    if field_mapping.question_field not in dataset.column_names:
        raise ValueError(
            f"Required question field '{field_mapping.question_field}' not found"
        )

    dataset = dataset.map(lambda x: process_dataset_fields(x, field_mapping))
    dataset = dataset.map(lambda x: generate_prompt(x, tokenizer))

    # Pilot: subset
    if PILOT_NUM_SAMPLES > 0 and PILOT_NUM_SAMPLES < len(dataset):
        dataset = dataset.select(range(PILOT_NUM_SAMPLES))
        logger.info(f"Pilot mode: using {PILOT_NUM_SAMPLES} samples")

    train_dataset = dataset
    rewards_output_dir = get_reward_output_dir()

    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Reward outputs: {os.path.abspath(rewards_output_dir)}")

    # Print samples
    for i in range(min(2, len(train_dataset))):
        logger.info(f"Sample {i}: Q={train_dataset[i]['question'][:80]}...")

    # Initialize components
    corver_extractor, corver_client = init_corver_components()
    llm_judge = init_judge()

    # Reward functions: only format_reward goes through TRL's standard interface
    reward_fns = [format_reward_func]
    logger.info(
        f"Standard reward_funcs: [format_reward]. "
        f"Stepwise rewards: CorVer={corver_extractor is not None}, "
        f"Judge={llm_judge is not None}"
    )

    # Honor curriculum order: when STEPWISE_SEQUENTIAL_SAMPLER=1, override the
    # default RandomSampler with SequentialSampler so the dataset is consumed
    # in file order. Required for any curriculum / sorted-by-difficulty dataset.
    SEQUENTIAL = os.environ.get("STEPWISE_SEQUENTIAL_SAMPLER", "0") == "1"

    # Create stepwise trainer
    trainer = make_stepwise_trainer(
        base_trainer_cls=GRPOTrainer,  # Already Unsloth-patched
        model=model,
        reward_funcs=reward_fns,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        callbacks=callbacks,
        llm_judge=llm_judge,
        corver_extractor=corver_extractor,
        corver_client=corver_client,
        beta_judge=BETA_JUDGE,
        stepwise_reward_mode=STEPWISE_REWARD_MODE,
        sentence_log_dir=os.path.join(rewards_output_dir, "sentence_logs"),
    )

    if SEQUENTIAL:
        from torch.utils.data import SequentialSampler
        trainer._get_train_sampler = lambda *a, **kw: SequentialSampler(trainer.train_dataset)
        logger.info("Sequential sampler ENABLED (curriculum order honored)")

    logger.info(
        f'*** Starting stepwise training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ***'
    )

    # Train
    train_result = trainer.train(
        resume_from_checkpoint=get_checkpoint(training_args)
        if training_args.resume_from_checkpoint
        else None
    )

    # Save
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training completed ***")

    trainer.model.config.use_cache = True
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")


def main():
    parser = TrlParser((ModelConfig, DatasetArguments, GRPOConfig))
    model_args, dataset_args, training_args = parser.parse_args_and_config(
        fail_with_unknown_args=False
    )

    config_file_path = get_default_config_path()
    swanlab_callback = create_swanlab_callback_from_yaml(config_file_path)

    if swanlab_callback:
        callbacks = [swanlab_callback]
    else:
        callbacks = None

    grpo_function(model_args, dataset_args, training_args, callbacks=callbacks)


if __name__ == "__main__":
    main()
