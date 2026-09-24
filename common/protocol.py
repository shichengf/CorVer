"""Training prompt and response-level reward rules (paper Appendix A)."""

import re
import unicodedata

SYSTEM_PROMPT = (
    "Answer using this exact format:\n"
    "<think>brief reasoning</think>\n"
    "<answer>direct answer</answer>\n\n"
    "If you genuinely do not know, reply exactly: <think>unknown</think><answer>I don't know</answer>\n"
    "Do not loop or repeat the same point or phrase."
)


def extract_answer(completion):
    # Track nested reasoning instead of deleting it: deletion can accidentally
    # concatenate answer text on both sides of a reasoning span.
    first_think = re.search(r"</?think\b[^>]*>", completion, re.I)
    depth = int(bool(first_think and first_think.group().startswith("</")))
    answer_start = None
    for tag in re.finditer(r"</?think\b[^>]*>|<answer\s*>", completion, re.I):
        token = tag.group().lower()
        if token.startswith("</think"):
            depth = max(0, depth - 1)
        elif token.startswith("<think"):
            depth += 1
        elif depth == 0:
            answer_start = tag.end()
            break
    if answer_start is None:
        return "", "missing"
    tail = completion[answer_start:]
    # First tag bounds the answer: never consume later reasoning or answers.
    tag = re.search(r"<[^>]*>", tail)
    if tag and re.fullmatch(r"</answer\s*>", tag.group(), re.I):
        return tail[: tag.start()].strip(), "bounded"
    end = re.search(r"<|\r|\n", tail)
    return (tail[: end.start()] if end else tail).strip(), "fallback"


def normalize(s):
    s = unicodedata.normalize("NFKD", s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


REFUSALS = {
    normalize(s) for s in ["I don't know", "I do not know", "I have no comment"]
}


def answer_matches(answer, references):
    prediction = normalize(answer)
    if not prediction or prediction in REFUSALS:
        return False
    return any(
        normalize(ref)
        and (prediction in normalize(ref) or normalize(ref) in prediction)
        for ref in references
    )


def correctness_reward(completion, references):
    answer, _ = extract_answer(completion)
    return 2.0 if answer_matches(answer, references) else -1.0


def format_reward(completion, length):
    answer, mode = extract_answer(completion)
    return (1.0 if answer and mode == "bounded" else -1.0) - 0.5 * max(
        0, length - 512
    ) / 512
