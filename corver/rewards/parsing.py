"""Triplet parser adapted from QuCo-RAG; see docs/third_party.md."""


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
        "\u201c": "\u201d",
    }
    closing_quotes = {closing: opening for opening, closing in quote_pairs.items()}

    for ch in case_str:
        if in_quotes:
            if escape_next:
                current_token.append(ch)
                escape_next = False
                continue

            if ch == "\\":
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

        if ch == "[":
            new_list = []
            if stack:
                stack[-1].append(new_list)
            stack.append(new_list)
            if root is None:
                root = stack[0]
            continue

        if ch == "]":
            token = "".join(current_token).strip()
            if token:
                stack[-1].append(token)
            current_token = []
            if stack:
                finished = stack.pop()
                if not stack:
                    root = finished
            continue

        if ch == ",":
            if stack:
                current_list = stack[-1]
                if len(stack) >= 2 and len(current_list) >= 2 and current_token:
                    current_token.append(ch)
                    continue

            token = "".join(current_token).strip()
            if token and stack:
                stack[-1].append(token)
            current_token = []
            continue

        if ch.isspace():
            if current_token:
                current_token.append(" ")
            continue

        current_token.append(ch)

    if stack:
        token = "".join(current_token).strip()
        if token:
            stack[-1].append(token)

    return root or []
