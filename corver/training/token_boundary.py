"""Keep vLLM rollout and GRPO forward-pass prompt tokens identical."""


def install(trainer):
    original = trainer.llm.generate
    captured = []

    def generate(prompts, *args, **kwargs):
        nonlocal captured
        if not all(isinstance(p, str) for p in prompts):
            raise TypeError("Expected rendered text prompts")
        ids = [
            trainer.processing_class.encode(p, add_special_tokens=False)
            for p in prompts
        ]
        # Match the trainer's left truncation rather than introducing a second BOS.
        ids = [row[-trainer.max_prompt_length :] for row in ids]
        output = original([dict(prompt_token_ids=row) for row in ids], *args, **kwargs)
        captured = [o.prompt_token_ids for o in output]
        if captured != ids:
            raise ValueError("vLLM altered explicit prompt tokens")
        return output

    trainer.llm.generate = generate
    original_score = trainer._generate_and_score_completions

    def score(inputs):
        result = original_score(inputs)
        expected = [
            row[mask.bool()].cpu().tolist()
            for row, mask in zip(result["prompt_ids"], result["prompt_mask"])
        ]
        if expected != captured:
            raise ValueError("Rollout/forward prompt token mismatch")
        return result

    trainer._generate_and_score_completions = score
