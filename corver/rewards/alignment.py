"""Character-accurate sentence rewards for the actual generated token sequence."""

import re
import torch


def decoded_token_spans(tokenizer, ids):
    text = tokenizer.decode(ids, skip_special_tokens=True)
    special = set(tokenizer.all_special_ids)
    visible = [(i, token) for i, token in enumerate(ids) if token not in special]
    spans = [(-1, -1)] * len(ids)
    # Fast-tokenizer offsets are valid only if re-encoding preserves the *exact*
    # generated token sequence. Never assume generated segmentations are canonical.
    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if encoded["input_ids"] == [token for _, token in visible]:
            for (i, _), (start, end) in zip(visible, encoded["offset_mapping"]):
                spans[i] = (start, end)
            return text, spans, "verified_fast_offsets"
    # Prefix decode preserves byte fragments, SentencePiece context and decoding
    # cleanup. A token completing a fragmented character inherits that character's
    # span. This path is slower but cannot silently add replacement-character drift.
    prefixes = [""]
    for i in range(len(ids)):
        prefixes.append(tokenizer.decode(ids[: i + 1], skip_special_tokens=True))
    stable = []
    for prefix in prefixes:
        n = 0
        for a, b in zip(prefix, text):
            if a != b:
                break
            n += 1
        stable.append(n)
    for i, token in enumerate(ids):
        if token in special:
            continue
        start = min(stable[i], stable[i + 1])
        end = stable[i + 1]
        if end == start:
            # Incomplete byte fragments share the next completed character span.
            end = next((n for n in stable[i + 2 :] if n > start), start)
        spans[i] = (start, end)
    return text, spans, "prefix_decode_offsets"


def scored_sections(text):
    sections = []
    first = re.search(r"</?think\b[^>]*>", text, re.I)
    depth = int(bool(first and first.group().startswith("</")))
    think_start = 0 if depth else None
    answer = None
    for tag in re.finditer(r"</?think\b[^>]*>|<answer\s*>", text, re.I):
        token = tag.group().lower()
        if token.startswith("</think"):
            depth = max(0, depth - 1)
            if depth == 0 and think_start is not None:
                sections.append((think_start, tag.start()))
                think_start = None
        elif token.startswith("<think"):
            if depth == 0:
                think_start = tag.end()
            depth += 1
        elif depth == 0 and answer is None:
            answer = tag
    if depth and think_start is not None:
        sections.append((think_start, len(text)))
    remaining = text
    if answer:
        start = answer.end()
        tail = remaining[start:]
        tag = re.search(r"<[^>]*>", tail)
        if tag and re.fullmatch(r"</answer\s*>", tag.group(), re.I):
            end = start + tag.start()
        else:
            boundary = re.search(r"<|\r|\n", tail)
            end = start + (boundary.start() if boundary else len(tail))
        sections.append((start, end))
    if not sections and text.strip():
        sections = [(0, len(text))]
    return sorted(sections)


def map_generated_sentences(completion_ids, tokenizer, completion_mask):
    ids_all = completion_ids.detach().cpu().tolist()
    mask_all = completion_mask.detach().cpu().bool().tolist()
    valid_positions = [i for i, flag in enumerate(mask_all) if flag]
    mapping = torch.full((len(ids_all),), -1, dtype=torch.long)
    if not valid_positions:
        return mapping, [], 0.0
    text, offsets, _ = decoded_token_spans(
        tokenizer, [ids_all[i] for i in valid_positions]
    )
    sections = scored_sections(text)
    sentences, sentence_spans = [], []
    for start, end in sections:
        cursor = start
        for part in re.split(r"(?<=[.!?])\s+", text[start:end]):
            sentence = part.strip()
            if sentence:
                pos = text.find(sentence, cursor, end)
                if pos < 0:
                    raise ValueError("Sentence not found inside scored section")
                sentences.append(sentence)
                sentence_spans.append((pos, pos + len(sentence)))
                cursor = pos + len(sentence)
    eligible = aligned = 0
    for original, (start, end) in zip(valid_positions, offsets):
        if start < 0 or end <= start:
            continue
        midpoint = (start + end) / 2
        if not any(a <= midpoint < b for a, b in sections):
            continue
        eligible += 1
        for sent_idx, (a, b) in enumerate(sentence_spans):
            if a <= midpoint < b:
                mapping[original] = sent_idx
                aligned += 1
                break
    return mapping, sentences, aligned / max(eligible, 1)
