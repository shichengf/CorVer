"""QuCo entity extraction using ZhishanQ/QuCo-extractor-0.5B.

This module replicates the exact entity extraction approach from QuCo-RAG
(https://github.com/ZhishanQ/QuCo-RAG), including:
- The fine-tuned 0.5B extractor model (Qwen2.5-0.5B-Instruct based)
- The few-shot prompt templates
- The parse_case() bracket/quote-aware parser
"""

import json
import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def parse_case(case_str: str):
    """Parse the model's entity extraction output into a nested Python list.

    Handles nested brackets, smart/curly quotes, escaped characters,
    and commas inside entity names (e.g., "Mumbai, Maharashtra, India").

    This is the exact parse_case() from QuCo-RAG's generate_quco.py.
    """
    case_str = case_str.strip()
    if not case_str:
        return []

    stack = []
    current_token = []
    in_quotes = False
    escape_next = False
    expected_quote = None
    root = None

    quote_pairs = {
        '"': '"',
        '\u201c': '\u201d',
    }
    closing_quotes = {closing: opening for opening, closing in quote_pairs.items()}

    for ch in case_str:
        if in_quotes:
            if escape_next:
                current_token.append(ch)
                escape_next = False
                continue

            if ch == '\\':
                escape_next = True
                continue

            if expected_quote and ch == expected_quote:
                in_quotes = False
                expected_quote = None
                continue

            current_token.append(ch)
            continue

        if ch in quote_pairs:
            in_quotes = True
            expected_quote = quote_pairs[ch]
            continue

        if ch in closing_quotes:
            current_token.append(ch)
            continue

        if ch == '[':
            new_list = []
            if stack:
                stack[-1].append(new_list)
            stack.append(new_list)
            if root is None:
                root = stack[0]
            continue

        if ch == ']':
            token = ''.join(current_token).strip()
            if token:
                stack[-1].append(token)
            current_token = []
            if stack:
                finished = stack.pop()
                if not stack:
                    root = finished
            continue

        if ch == ',':
            if stack:
                current_list = stack[-1]
                if (
                    len(stack) >= 2
                    and len(current_list) >= 2
                    and current_token
                ):
                    current_token.append(ch)
                    continue

            token = ''.join(current_token).strip()
            if token and stack:
                stack[-1].append(token)
            current_token = []
            continue

        if ch.isspace():
            if current_token:
                current_token.append(' ')
            continue

        current_token.append(ch)

    if stack:
        token = ''.join(current_token).strip()
        if token:
            stack[-1].append(token)

    return root or []


class QuCoEntityExtractor:
    """Entity extractor using the QuCo-extractor-0.5B model.

    Uses the exact same model and prompts as the QuCo-RAG paper.
    The model is fine-tuned from Qwen2.5-0.5B-Instruct to extract
    knowledge triplets from sentences.
    """

    def __init__(
        self,
        model_name: str = "ZhishanQ/QuCo-extractor-0.5B",
        device_map: str = "auto",
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        # Load prompt templates
        template_path = os.path.join(os.path.dirname(__file__), "prompt_templates.json")
        with open(template_path, "r", encoding="utf-8") as f:
            templates = json.load(f)
        self.entity_extraction_prompt = templates["entity_extraction_prompt"]
        self.entity_extraction_prompt_for_question = templates["entity_extraction_prompt_for_question"]

        # Load model and tokenizer
        # Reload Qwen2 module to undo unsloth's monkey-patches on Qwen2Attention,
        # which break loading a separate Qwen2 model (apply_qkv missing).
        logger.info(f"Loading QuCo entity extractor: {model_name}")
        import importlib
        try:
            import transformers.models.qwen2.modeling_qwen2 as _qwen2_mod
            importlib.reload(_qwen2_mod)
            _Qwen2ForCausalLM = _qwen2_mod.Qwen2ForCausalLM
            logger.info("Reloaded Qwen2 module to bypass unsloth patches")
        except Exception as e:
            logger.warning(f"Could not reload Qwen2 module: {e}, falling back to Auto")
            _Qwen2ForCausalLM = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if _Qwen2ForCausalLM is not None:
            self.model = _Qwen2ForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        logger.info("QuCo entity extractor loaded successfully")

    def _generate(self, prompt: str) -> str:
        """Generate text using the model with chat template."""
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_length = inputs["input_ids"].shape[1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _generate_batch(self, prompts: list, batch_size: int = 32) -> list:
        """Generate text for multiple prompts using batched inference with left-padding."""
        all_results = []
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]

            # Tokenize each prompt via chat template
            token_ids_list = []
            for p in batch_prompts:
                msgs = [{"role": "user", "content": p}]
                ids = self.tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True,
                )
                token_ids_list.append(ids)

            # Left-pad to max length in this batch
            max_len = max(len(ids) for ids in token_ids_list)
            pad_id = self.tokenizer.pad_token_id
            padded_ids = []
            attention_masks = []
            for ids in token_ids_list:
                pad_len = max_len - len(ids)
                padded_ids.append([pad_id] * pad_len + ids)
                attention_masks.append([0] * pad_len + [1] * len(ids))

            input_ids = torch.tensor(padded_ids, device=self.model.device)
            attention_mask = torch.tensor(attention_masks, device=self.model.device)

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=pad_id,
            )

            for out in outputs:
                gen_tokens = out[max_len:]
                text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                all_results.append(text)

        return all_results

    def extract_triplets(self, sentence: str) -> list:
        """Extract knowledge triplets from a sentence.

        Returns a list of triplets, where each triplet is:
        - [head_entity, relation, tail_entity] for declarative sentences
        - [entity, relation] for questions
        - [] if no factual content
        """
        prompt = self.entity_extraction_prompt.format(sentence)
        response = self._generate(prompt)

        # Strip "entities:" prefix if present (model sometimes echoes it)
        response = response.strip()
        if response.startswith("entities:"):
            response = response.replace("entities:", "", 1).strip()

        try:
            result = parse_case(response)
        except Exception as e:
            logger.warning(f"Failed to parse extraction result: {e}")
            return []

        if not isinstance(result, list):
            return []
        return result

    def extract_triplets_batch(self, sentences: list, batch_size: int = 32) -> list:
        """Extract knowledge triplets from multiple sentences using batched inference."""
        if not sentences:
            return []
        prompts = [self.entity_extraction_prompt.format(s) for s in sentences]
        responses = self._generate_batch(prompts, batch_size=batch_size)
        results = []
        for response in responses:
            response = response.strip()
            if response.startswith("entities:"):
                response = response.replace("entities:", "", 1).strip()
            try:
                result = parse_case(response)
            except Exception as e:
                logger.warning(f"Failed to parse extraction result: {e}")
                result = []
            if not isinstance(result, list):
                result = []
            results.append(result)
        return results

    def extract_question_entities_batch(self, questions: list, batch_size: int = 32) -> list:
        """Extract key entities from multiple questions using batched inference."""
        if not questions:
            return []
        prompts = [self.entity_extraction_prompt_for_question.format(q) for q in questions]
        responses = self._generate_batch(prompts, batch_size=batch_size)
        all_entities = []
        for response in responses:
            response = response.strip()
            if response.startswith("entities:"):
                response = response.replace("entities:", "", 1).strip()
            try:
                result = parse_case(response)
            except Exception as e:
                logger.warning(f"Failed to parse question entity result: {e}")
                result = []
            if not isinstance(result, list):
                result = []
            entities = []
            for item in result:
                if isinstance(item, list) and len(item) > 0:
                    entities.append(item[0])
                elif isinstance(item, str):
                    entities.append(item)
            all_entities.append(list(dict.fromkeys(entities)))
        return all_entities

    def extract_question_entities(self, question: str) -> list:
        """Extract key entities from a question.

        Returns a list of entity name strings.
        """
        prompt = self.entity_extraction_prompt_for_question.format(question)
        response = self._generate(prompt)

        response = response.strip()
        if response.startswith("entities:"):
            response = response.replace("entities:", "", 1).strip()

        try:
            result = parse_case(response)
        except Exception as e:
            logger.warning(f"Failed to parse question entity result: {e}")
            return []

        if not isinstance(result, list):
            return []

        # Extract entity names from nested lists like [["entity1"], ["entity2"]]
        entities = []
        for item in result:
            if isinstance(item, list) and len(item) > 0:
                entities.append(item[0])
            elif isinstance(item, str):
                entities.append(item)

        # Deduplicate while preserving order
        return list(dict.fromkeys(entities))
