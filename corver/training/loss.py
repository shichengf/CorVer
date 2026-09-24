"""GRPO on selected token log-probabilities with sentence-level advantages."""

import torch
import os

def pack_padding(input_ids, pad_id):
    order = torch.argsort(input_ids != pad_id, dim=1, descending=True, stable=True)
    return input_ids.gather(1, order)

def selected_completion_logps(
    model, packed_ids, original_mask, completion_length, batch_size, temperature=1.0
):
    """Score packed tokens; return (B,T) in the original completion coordinates.

    Prompt padding varies by row. Selecting the last T positions of a packed row
    would include different tokens, so gather at each row's true prompt boundary.
    """
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
    prompt_width = packed_ids.shape[1] - completion_length
    prompt_lengths = original_mask[:, :prompt_width].sum(-1).long()
    if torch.any(prompt_lengths < 1):
        raise ValueError("Every completion needs a nonempty prompt")
    keep = completion_length + int((prompt_width - prompt_lengths).max()) + 1
    result = []
    for start in range(0, len(packed_ids), batch_size):
        ids = packed_ids[start : start + batch_size]
        valid_lengths = original_mask[start : start + batch_size].sum(-1)
        packed_mask = (
            torch.arange(ids.shape[1], device=ids.device)[None, :]
            < valid_lengths[:, None]
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
            logits = model(
                input_ids=ids,
                attention_mask=packed_mask,
                logits_to_keep=keep,
                use_cache=False,
            ).logits
        positions = (
            prompt_lengths[start : start + batch_size, None]
            - 1
            + torch.arange(completion_length, device=ids.device)
        )
        selected_positions = positions - (ids.shape[1] - logits.shape[1])
        rows = torch.arange(len(ids), device=ids.device)[:, None]
        selected_logits = logits[rows, selected_positions].float() / temperature
        targets = ids.gather(1, positions + 1)
        logps = selected_logits.gather(-1, targets[..., None]).squeeze(
            -1
        ) - selected_logits.logsumexp(-1)
        completion_mask = original_mask[start : start + batch_size, -completion_length:]
        result.append(torch.where(completion_mask.bool(), logps, 0.0))
        del logits, selected_logits
    return torch.cat(result, dim=0)

def align_generation_metadata(inputs, batch_size, num_generations):
    # TRL's RepeatSampler already repeats each input once per completion.
    if len(inputs) == batch_size:
        rows = inputs
    elif len(inputs) * num_generations == batch_size:
        rows = [row for row in inputs for _ in range(num_generations)]
    else:
        raise ValueError(
            f"Cannot align {len(inputs)} inputs to {batch_size} completions"
        )
    if num_generations < 2 or batch_size % num_generations:
        raise ValueError(
            "Stepwise normalization requires complete groups of at least two generations"
        )
    for start in range(0, batch_size, num_generations):
        head = rows[start]
        for row in rows[start + 1 : start + num_generations]:
            if any(
                row.get(key) != head.get(key)
                for key in ("question", "best_answer", "prompt")
            ):
                raise ValueError(
                    f"Mixed questions or references in generation group at row {start}"
                )
    return [r.get("question", "") for r in rows], [
        r.get("best_answer", "") for r in rows
    ]

def grpo_logprob_loss(
    ref, new, old, sampling, input_ids, mask, beta, advantages, **kwargs
):
    if new.ndim != 2 or new.shape != mask.shape or advantages.shape != new.shape:
        raise ValueError(
            "Expected matching (batch, tokens) logps, masks and advantages"
        )
    for name, value in [("ref", ref), ("old", old)]:
        if value is not None and value.shape != new.shape:
            raise ValueError(f"{name} logps are not aligned with current logps")
    new = new.float()
    mask = mask.float()
    active = mask.bool()
    lengths = mask.sum(-1).clamp(min=1)
    old = new.detach() if old is None else old.detach().float()
    log_ratio = torch.where(active, new - old, 0.0)
    level = kwargs.get("importance_sampling_level", "token")
    if level == "sequence":
        log_ratio = (log_ratio * mask).sum(-1, keepdim=True) / lengths[:, None]
    elif level != "token":
        raise ValueError(level)
    ratio = log_ratio.exp()
    clipped = ratio.clamp(
        1 - kwargs.get("epsilon_low", 0.2), 1 + kwargs.get("epsilon_high", 0.2)
    )
    delta = kwargs.get("delta")
    objective_ratio = ratio if delta is None else ratio.clamp(max=delta)
    token_loss = -torch.minimum(objective_ratio * advantages, clipped * advantages)
    if beta:
        if ref is None:
            raise ValueError("Reference logps required for KL regularization")
        difference = torch.where(active, ref.detach().float() - new, 0.0)
        kl = difference.exp() - difference - 1
        token_loss = token_loss + beta * kl
    else:
        kl = torch.zeros_like(new)
        # No additional vLLM importance-sampling correction.
    if kwargs.get("use_vllm", False):
        raise ValueError(
            "vLLM importance correction is not configured in this implementation"
        )
    masked = torch.where(active, token_loss, 0.0)
    loss_type = kwargs.get("loss_type", "grpo")
    accumulation = kwargs.get("current_gradient_accumulation_steps", 1)
    if loss_type == "grpo":
        loss = (masked.sum(-1) / lengths).mean() / accumulation
    elif loss_type == "bnpo":
        loss = masked.sum() / mask.sum().clamp(min=1) / accumulation
    elif loss_type == "dr_grpo":
        loss = (
            masked.sum()
            / (new.shape[0] * kwargs["max_completion_length"])
            / accumulation
        )
    elif loss_type == "dapo":
        loss = masked.sum() / (
            kwargs["num_items_in_batch"] / kwargs.get("num_processes", 1)
        )
    else:
        raise ValueError(loss_type)
    with torch.no_grad():
        mean_kl = ((kl * mask).sum(-1) / lengths).mean()
    empty = new.new_empty(0).detach()
    return loss, mask.sum(-1).mean(), mean_kl, empty, empty, ratio.detach()
