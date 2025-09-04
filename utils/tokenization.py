from __future__ import annotations

from typing import Callable


def get_token_counter_for_model(model: str) -> Callable[[str], int]:
    """Return a function that counts tokens for a given model using tiktoken.
    Falls back to a rough heuristic if tiktoken is unavailable.
    """
    try:
        import tiktoken  # type: ignore
        # Most OpenAI embedding/chat models use cl100k_base
        enc = tiktoken.get_encoding("cl100k_base")

        def counter(text: str) -> int:
            return len(enc.encode(text or ""))

        return counter
    except Exception:
        from .chunker import approx_token_count

        def counter(text: str) -> int:
            return approx_token_count(text)

        return counter


def trim_to_token_limit(text: str, max_tokens: int, count_tokens: Callable[[str], int]) -> str:
    """Trim text so that token count <= max_tokens using binary search on length.
    If already within limit, returns text as-is.
    """
    if count_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        if count_tokens(candidate) <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best

