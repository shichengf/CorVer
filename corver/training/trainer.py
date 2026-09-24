"""Full CorVer GRPO: token rewards, Eq. (4), and aligned log-probability loss."""

import torch
from corver.rewards.batch import fast_batch_returns
from corver.training.loss import align_generation_metadata
from common.protocol import correctness_reward, format_reward


def token_advantages(returns, mask, group_size):
    if group_size < 2 or returns.shape != mask.shape or len(returns) % group_size:
        raise ValueError("Expected complete groups of aligned token returns")
    weights = mask.float()
    means = (returns * weights).sum(-1) / weights.sum(-1).clamp(min=1)
    groups = means.view(-1, group_size)
    mean = groups.mean(1).repeat_interleave(group_size)
    std = groups.std(1).repeat_interleave(group_size)
    advantages = (returns - mean[:, None]) / (std[:, None] + 1e-4)
    advantages = advantages * weights
    if not torch.isfinite(advantages).all():
        raise FloatingPointError("Non-finite token advantages")
    return advantages, means, std


class StepwiseGRPOTrainer:
    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        ids, mask = output["completion_ids"], output["completion_mask"]
        _, answers = align_generation_metadata(inputs, len(ids), self.num_generations)
        texts = self.processing_class.batch_decode(ids, skip_special_tokens=True)
        judge = [correctness_reward(t, a.split(";")) for t, a in zip(texts, answers)]
        formats = [format_reward(t, int(m.sum())) for t, m in zip(texts, mask)]
        returns, info = fast_batch_returns(
            ids,
            mask,
            self.processing_class,
            self.corver_extractor,
            self.corver_client,
            judge,
            formats,
            self.judge_weight,
        )
        advantages, means, std = token_advantages(returns, mask, self.num_generations)
        output["advantages"] = advantages
        mode = "train" if self.model.training else "eval"
        valid = advantages[mask.bool()]
        if valid.numel():
            self._metrics[mode]["corver/advantages_abs_max"].append(
                valid.abs().max().item()
            )
        self._metrics[mode]["corver/reward"].append(means.mean().item())
        self._metrics[mode]["corver/group_std"].append(std.mean().item())
        self._metrics[mode]["corver/zero_std_fraction"].append(
            torch.isclose(std, torch.zeros_like(std)).float().mean().item()
        )
        self._metrics[mode]["corver/alignment"].append(
            sum(x["alignment_rate"] for x in info) / len(info)
        )
        return output

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        **kwargs
    ):
        # Unsloth 2025.11's parent returns hidden states here. This mixin's loss
        # needs selected log-probabilities, aligned with the original completion.
        if compute_entropy:
            raise ValueError("Entropy calculation is not configured for CorVer")
        from corver.training.loss import selected_completion_logps

        with torch.no_grad():
            return (
                selected_completion_logps(
                    model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size or self.args.per_device_train_batch_size,
                    self.temperature,
                ),
                None,
            )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        advantages = inputs.get("advantages")
        if advantages is None or advantages.dim() != 2:
            return super().compute_loss(
                model, inputs, return_outputs, num_items_in_batch
            )
        if return_outputs:
            raise ValueError("StepwiseGRPOTrainer does not support return_outputs")
        from corver.training.loss import (
            selected_completion_logps,
            grpo_logprob_loss,
            pack_padding,
        )

        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat(
            [inputs["prompt_mask"], inputs["completion_mask"]], dim=1
        )
        # Match the parent's packing used for old/reference log-probabilities.
        packed_ids = pack_padding(input_ids, self.processing_class.pad_token_id)
        new = selected_completion_logps(
            model,
            packed_ids,
            attention_mask,
            inputs["completion_ids"].shape[1],
            input_ids.shape[0],
            self.temperature,
        )
        loss, length, kl, _, _, ratio = grpo_logprob_loss(
            inputs.get("ref_per_token_logps"),
            new,
            inputs.get("old_per_token_logps"),
            inputs.get("sampling_per_token_logps"),
            inputs["completion_ids"],
            inputs["completion_mask"],
            self.beta,
            advantages,
            loss_type=self.args.loss_type,
            importance_sampling_level=self.importance_sampling_level,
            epsilon_low=self.epsilon_low,
            epsilon_high=self.epsilon_high,
            max_completion_length=self.args.max_completion_length,
            delta=self.args.delta,
            num_items_in_batch=inputs.get("num_items_in_batch"),
            current_gradient_accumulation_steps=self.current_gradient_accumulation_steps,
            num_processes=self.accelerator.num_processes,
        )
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["completion_length"].append(length.item())
        self._metrics[mode]["kl"].append(kl.item())
        clipped = ((ratio < 1 - self.epsilon_low) & (advantages < 0)) | (
            (ratio > 1 + self.epsilon_high) & (advantages > 0)
        )
        mask = inputs["completion_mask"].float()
        self._metrics[mode]["clip_ratio"].append(
            ((clipped.float() * mask).sum() / mask.sum().clamp(min=1)).item()
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite stepwise loss")
        return loss


def make_stepwise_trainer(
    base_trainer_cls, *, extractor, count_client, judge_weight, **kwargs
):
    cls = type("CorVerGRPOTrainer", (StepwiseGRPOTrainer, base_trainer_cls), {})
    trainer = cls(**kwargs)
    trainer.corver_extractor = extractor
    trainer.corver_client = count_client
    trainer.judge_weight = judge_weight
    return trainer
