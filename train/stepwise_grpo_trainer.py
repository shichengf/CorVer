"""StepwiseGRPOTrainer: GRPO with sentence-level per-token advantages.

Minimal extension of TRL's GRPOTrainer (Unsloth-patched).
Overrides advantage construction to inject sentence-level CorVer penalties
mapped to per-token returns, while keeping all generation, optimization,
logging, checkpoint, and wandb logic intact.

The existing grpo_compute_loss already supports per-token advantages:
when advantages.dim() == 2 (B, T), it skips unsqueeze and uses them directly.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


class StepwiseGRPOTrainer:
    """Mixin that overrides _generate_and_score_completions for stepwise advantages.

    Usage: This class is combined with the Unsloth-patched GRPOTrainer at runtime.
    See make_stepwise_trainer() below.
    """

    def _stepwise_init(
        self,
        llm_judge=None,
        corver_extractor=None,
        corver_client=None,
        beta_judge: float = 0.3,
        stepwise_reward_mode: str = "full",
        sentence_log_dir: Optional[str] = None,
    ):
        """Initialize stepwise components. Call after __init__."""
        self.llm_judge = llm_judge
        self.corver_extractor = corver_extractor
        self.corver_client = corver_client
        self.beta_judge = beta_judge
        self.stepwise_reward_mode = stepwise_reward_mode
        self.sentence_log_dir = sentence_log_dir
        self._consecutive_high_fallback = 0

        logger.info(
            f"StepwiseGRPOTrainer: mode={stepwise_reward_mode}, "
            f"beta_judge={beta_judge}, "
            f"judge={'enabled' if llm_judge else 'disabled'}, "
            f"corver={'enabled' if corver_extractor else 'disabled'}"
        )

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Override: compute per-token advantages from sentence-level CorVer + judge.

        Flow:
        1. Call parent for generation, logps, format reward, scalar advantages
        2. Decode completions and extract questions/gold answers
        3. Compute LLM judge rewards (if enabled)
        4. Compute sentence-level CorVer penalties → per-token returns (if enabled)
        5. GRPO group normalization on per-token returns
        6. Replace scalar advantages with (B, T) per-token advantages
        """
        # Save inputs for accessing question/best_answer after super()
        self._stepwise_inputs = inputs

        # Step 1: Parent handles generation, old/ref logps, format reward
        output = super()._generate_and_score_completions(inputs)

        # If stepwise not configured, return as-is (standard GRPO)
        if not hasattr(self, 'stepwise_reward_mode'):
            return output

        device = output["completion_ids"].device
        completion_ids = output["completion_ids"]    # (B, T)
        completion_mask = output["completion_mask"]   # (B, T)
        B, T = completion_ids.shape

        # Step 2: Decode completions, get questions/gold answers
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        num_gen = self.num_generations
        questions = []
        best_answers = []
        for inp in inputs:
            q = inp.get("question", "")
            a = inp.get("best_answer", "")
            questions.extend([q] * num_gen)
            best_answers.extend([a] * num_gen)

        # Trim to match actual batch size (in case of process slicing)
        questions = questions[:B]
        best_answers = best_answers[:B]

        # Step 3: Compute format rewards (cheap — regex only)
        from reward_function import format_reward_func
        format_rewards = format_reward_func(completions=completions_text)

        # Step 4: Compute judge rewards
        if self.stepwise_reward_mode in ("full", "judge_only") and self.llm_judge is not None:
            judge_rewards, judge_details = self.llm_judge.judge_batch(
                questions, completions_text, best_answers
            )
            # Log judge metrics
            mode = "train" if self.model.training else "eval"
            avg_judge = sum(judge_rewards) / max(len(judge_rewards), 1)
            self._metrics[mode]["rewards/llm_judge/mean"].append(avg_judge)
        else:
            judge_rewards = [0.0] * B
            judge_details = []

        # Step 5: Compute per-token returns
        if self.stepwise_reward_mode in ("full", "corver_only") and self.corver_extractor is not None:
            from sentence_reward import compute_batch_per_token_returns
            per_token_returns, batch_info = compute_batch_per_token_returns(
                completion_ids_batch=completion_ids,
                completion_mask_batch=completion_mask,
                tokenizer=self.processing_class,
                extractor=self.corver_extractor,
                client=self.corver_client,
                judge_rewards=judge_rewards,
                format_rewards=format_rewards,
                beta_judge=self.beta_judge,
            )

            # Monitor alignment fallback rate
            fallback_count = sum(1 for info in batch_info if info.get("fallback_to_global", False))
            fallback_rate = fallback_count / max(B, 1)
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["stepwise/alignment_fallback_rate"].append(fallback_rate)
            self._metrics[mode]["stepwise/avg_alignment_rate"].append(
                sum(info["alignment_rate"] for info in batch_info) / max(B, 1)
            )

            # Log CorVer penalties
            all_corver_penalties = []
            for info in batch_info:
                for sd in info.get("sentence_details", []):
                    if sd.get("sentence_reward") is not None:
                        all_corver_penalties.append(sd["sentence_reward"])
            if all_corver_penalties:
                self._metrics[mode]["rewards/corver_sentence/mean"].append(
                    sum(all_corver_penalties) / len(all_corver_penalties)
                )

            # Warning for high fallback rate
            if fallback_rate > 0.5:
                self._consecutive_high_fallback += 1
                if self._consecutive_high_fallback >= 3:
                    logger.warning(
                        f"WARNING: {self._consecutive_high_fallback} consecutive batches with "
                        f">50% alignment fallback. CorVer signal may be ineffective."
                    )
            else:
                self._consecutive_high_fallback = 0

            # Save detailed logs if configured
            if self.sentence_log_dir:
                self._save_sentence_logs(batch_info, judge_details)

        else:
            # judge_only or no CorVer: broadcast judge + format to all tokens
            per_token_returns = torch.zeros((B, T), dtype=torch.float32, device=device)
            for i in range(B):
                ret = self.beta_judge * judge_rewards[i] + format_rewards[i]
                per_token_returns[i] = ret * completion_mask[i].float()

        # Step 6: GRPO group normalization on per-token returns
        # R_k = per-completion mean return (for centering)
        mask_float = completion_mask.float()
        R_k = (per_token_returns * mask_float).sum(-1) / mask_float.sum(-1).clamp(min=1)  # (B,)

        mean_group = R_k.view(-1, num_gen).mean(dim=1)
        mean_group = mean_group.repeat_interleave(num_gen, dim=0)  # (B,)

        if self.scale_rewards in ("group", "none"):
            std_group = R_k.view(-1, num_gen).std(dim=1)
            std_group = std_group.repeat_interleave(num_gen, dim=0)
        elif self.scale_rewards == "batch":
            std_group = R_k.std().expand_as(R_k)
        else:
            std_group = R_k.view(-1, num_gen).std(dim=1)
            std_group = std_group.repeat_interleave(num_gen, dim=0)

        is_std_zero = torch.isclose(std_group, torch.zeros_like(std_group))

        if self.scale_rewards != "none":
            # Per-token advantage with group normalization
            advantages = (per_token_returns - mean_group.unsqueeze(1)) / (std_group.unsqueeze(1) + 1e-4)
        else:
            advantages = per_token_returns - mean_group.unsqueeze(1)

        # Zero out padding positions
        advantages = advantages * mask_float

        # Step 7: Replace advantages in output
        output["advantages"] = advantages  # (B, T) instead of (B,)

        # Log advantage stats
        mode = "train" if self.model.training else "eval"
        valid_advantages = advantages[completion_mask.bool()]
        if valid_advantages.numel() > 0:
            self._metrics[mode]["stepwise/advantages_mean"].append(valid_advantages.mean().item())
            self._metrics[mode]["stepwise/advantages_std"].append(valid_advantages.std().item())
        self._metrics[mode]["reward"].append(mean_group.mean().item())
        self._metrics[mode]["reward_std"].append(std_group.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override to force the non-chunked loss path for per-token advantages.

        Requires sufficient GPU memory (~80GB A100) since it does a direct forward
        pass instead of Unsloth's memory-efficient accumulated path.

        The direct forward pass preserves gradients (unlike _get_per_token_logps_and_entropies
        which runs inside inference_mode). Uses logits_to_keep to minimize memory by only
        computing logits for completion tokens.
        """
        advantages = inputs.get("advantages")
        if advantages is not None and advantages.dim() == 2:
            if return_outputs:
                raise ValueError("StepwiseGRPOTrainer does not support return_outputs")

            prompt_ids = inputs["prompt_ids"]
            completion_ids = inputs["completion_ids"]
            prompt_mask = inputs["prompt_mask"]
            completion_mask = inputs["completion_mask"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)

            # Compute per-token logps WITH gradients via direct forward pass.
            # CRITICAL: must use left_pack_padding to match the attention pattern
            # used when computing old_logps and ref_logps. Without this, the logps
            # differ significantly → log_ratio explodes → KL/loss explosion.
            from unsloth_compiled_cache.UnslothGRPOTrainer import (
                left_pack_padding,
                calculate_pad_tokens_in_prompt,
                create_completion_attention_mask,
            )
            pad_token_id = self.processing_class.pad_token_id

            # Determine max_left_pad from pre-computed logprobs (computed on full batch).
            # CRITICAL: must match the max_left_pad used during _generate_and_score_completions,
            # NOT the micro-batch's max_left_pad (which can differ due to gradient accumulation).
            ref_logps = inputs.get("ref_per_token_logps")
            old_logps = inputs.get("old_per_token_logps")
            sampling_logps = inputs.get("sampling_per_token_logps")

            if old_logps is not None:
                max_left_pad = old_logps.size(1) - logits_to_keep
            elif ref_logps is not None:
                max_left_pad = ref_logps.size(1) - logits_to_keep
            else:
                left_pad_counts_tmp = calculate_pad_tokens_in_prompt(
                    input_ids, logits_to_keep, pad_token_id
                )
                max_left_pad = left_pad_counts_tmp.max().item()

            left_pad_counts = calculate_pad_tokens_in_prompt(
                input_ids, logits_to_keep, pad_token_id
            )
            packed_input_ids = left_pack_padding(input_ids, pad_token_id)
            packed_attention_mask = (packed_input_ids != pad_token_id).to(attention_mask.dtype)

            # Forward pass on left-packed input (with gradients).
            # autocast(bf16) lets fp32 LoRA params + bf16 base coexist:
            #   - matmul runs in bf16 (autocast policy for nn.Linear)
            #   - LoRA grads are computed in fp32 against fp32 params (correct precision)
            # Required for OLMo-2 path (FastModel + standard PEFT). For Llama/Qwen
            # via FastLanguageModel this is a no-op since Unsloth's fused LoRA
            # kernel handles dtype internally.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=packed_input_ids,
                    attention_mask=packed_attention_mask,
                    logits_to_keep=logits_to_keep + max_left_pad + 1,
                )
            # Shift by 1 for next-token prediction, keep completion + left_pad tokens
            logits = outputs.logits[:, :-1, :]  # (B, logits_to_keep + max_left_pad, V)
            log_probs = logits.log_softmax(dim=-1)

            # Gather log probs for the actual completion token IDs
            # completion_input_ids includes left-pad prompt tokens + completion tokens
            completion_input_ids = packed_input_ids[:, -(logits_to_keep + max_left_pad):]
            per_token_logps = log_probs.gather(
                2, completion_input_ids.unsqueeze(-1)
            ).squeeze(-1)  # (B, logits_to_keep + max_left_pad)

            # Create completion mask that zeros out left-pad positions
            completion_mask_padded = create_completion_attention_mask(
                completion_input_ids, left_pad_counts, max_left_pad, pad_token_id
            ).to(completion_mask.dtype)

            del logits, log_probs, outputs

            # All tensors now at (B, logits_to_keep + max_left_pad)
            T_padded = logits_to_keep + max_left_pad

            # Left-pad advantages to match
            if advantages.size(1) < T_padded:
                pad_size = T_padded - advantages.size(1)
                adv_pad = torch.zeros(
                    advantages.size(0), pad_size,
                    device=advantages.device, dtype=advantages.dtype
                )
                advantages = torch.cat([adv_pad, advantages], dim=1)

            # Use padded completion mask for loss
            completion_mask = completion_mask_padded

            import sys
            sys.path.insert(0, '.')
            from unsloth_compiled_cache.UnslothGRPOTrainer import grpo_compute_loss_slow

            loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1 = (
                grpo_compute_loss_slow(
                    ref_logps,
                    per_token_logps,
                    old_logps,
                    sampling_logps,
                    completion_input_ids,
                    completion_mask,
                    self.beta,
                    advantages,
                    loss_type=self.args.loss_type,
                    importance_sampling_level=self.importance_sampling_level,
                    epsilon_low=self.epsilon_low,
                    epsilon_high=self.epsilon_high,
                    max_completion_length=self.args.max_completion_length,
                    delta=self.args.delta,
                    temperature=self.args.temperature,
                    num_items_in_batch=inputs.get("num_items_in_batch"),
                    current_gradient_accumulation_steps=self.current_gradient_accumulation_steps,
                    num_processes=self.accelerator.num_processes,
                )
            )

            # Metrics logging
            mode = "train" if self.model.training else "eval"
            try:
                self._metrics[mode]["completion_length"].append(completion_length.item())
                self._metrics[mode]["kl"].append(mean_kl.item())
            except KeyError:
                self._metrics["completion_length"].append(completion_length.item())
                self._metrics["kl"].append(mean_kl.item())

            # Clip ratio metrics
            if coef_1 is not None and coef_1.dim() >= 2:
                mask_f = completion_mask.float()
                token_count = mask_f.sum().clamp(min=1.0)

                adv = advantages
                if adv.dim() == 1:
                    adv = adv.unsqueeze(1)
                # Align dimensions if needed
                if coef_1.shape[1] != adv.shape[1]:
                    min_T = min(coef_1.shape[1], adv.shape[1])
                    coef_1 = coef_1[:, -min_T:]
                    adv = adv[:, -min_T:]
                    mask_f = mask_f[:, -min_T:]
                    token_count = mask_f.sum().clamp(min=1.0)

                if self.loss_type in ("grpo", "bnpo", "dr_grpo", "dapo"):
                    is_clipped = (
                        ((coef_1 < 1 - self.epsilon_low) & (adv < 0)) |
                        ((coef_1 > 1 + self.epsilon_high) & (adv > 0))
                    )
                    clip_ratio = (is_clipped.float() * mask_f).sum() / token_count
                    self._metrics[mode]["clip_ratio"].append(
                        self.accelerator.gather(clip_ratio).mean().item()
                    )

            return loss

        # Fallback: scalar advantages — use parent's compute_loss
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def _save_sentence_logs(self, batch_info, judge_details):
        """Save per-sentence debug logs to JSON."""
        import json
        from corver_reward.logging_utils import get_output_timestamp

        timestamp = get_output_timestamp()
        log_dir = os.path.join(self.sentence_log_dir, timestamp)
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, "sentence_reward.json")
        log_data = {
            "batch_info": batch_info,
            "judge_details": judge_details,
        }

        # Merge-append
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    existing = json.load(f)
                if isinstance(existing, list):
                    existing.append(log_data)
                else:
                    existing = [existing, log_data]
                log_data = existing
            except Exception:
                log_data = [log_data]
        else:
            log_data = [log_data]

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2, default=str)


def make_stepwise_trainer(
    base_trainer_cls,
    model,
    reward_funcs,
    args,
    train_dataset,
    llm_judge=None,
    corver_extractor=None,
    corver_client=None,
    beta_judge: float = 0.3,
    stepwise_reward_mode: str = "full",
    sentence_log_dir: Optional[str] = None,
    **trainer_kwargs,
):
    """Factory: create a StepwiseGRPOTrainer from the (possibly patched) base class.

    This avoids class hierarchy issues with Unsloth's monkey-patching.

    Args:
        base_trainer_cls: GRPOTrainer class (may be Unsloth-patched).
        model: The language model.
        reward_funcs: List of reward functions (typically just format_reward).
        args: GRPOConfig.
        train_dataset: Training dataset.
        llm_judge: LLMJudge instance (or None).
        corver_extractor: QuCoEntityExtractor instance (or None).
        corver_client: Infigram client instance (or None).
        beta_judge: Weight for judge reward.
        stepwise_reward_mode: "full", "judge_only", or "corver_only".
        sentence_log_dir: Directory for sentence-level debug logs.
        **trainer_kwargs: Additional kwargs for GRPOTrainer.

    Returns:
        A trainer instance with stepwise advantage computation.
    """
    # Dynamically create subclass combining StepwiseGRPOTrainer mixin with the base
    StepwiseCls = type(
        "StepwiseGRPOTrainerImpl",
        (StepwiseGRPOTrainer, base_trainer_cls),
        {},
    )

    trainer = StepwiseCls(
        model=model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=train_dataset,
        **trainer_kwargs,
    )

    # Initialize stepwise components
    trainer._stepwise_init(
        llm_judge=llm_judge,
        corver_extractor=corver_extractor,
        corver_client=corver_client,
        beta_judge=beta_judge,
        stepwise_reward_mode=stepwise_reward_mode,
        sentence_log_dir=sentence_log_dir,
    )

    return trainer
