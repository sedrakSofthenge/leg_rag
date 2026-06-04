from __future__ import annotations

from typing import Optional

# Rewrites a colloquial question into a retrieval query rich in the legal
# terminology a statute would actually use, plus the related concepts needed to
# answer (e.g. for an attempted crime, the general rule on criminal attempt).
# This bridges the vocabulary gap between how users ask and how laws are written.
_SYS = (
    "You rewrite a user's question into a concise search query for retrieving passages "
    "from legal codes. Reply in the SAME language as the user's question. Use the precise "
    "legal terms and synonyms a statute would use, and add the related concepts required "
    "to fully answer the question (for an attempted crime, add the general rule on criminal "
    "attempt / unfinished crime; for speeding, add 'exceeding the established speed limit'; "
    "etc.). Output ONLY the rewritten query terms, one line, no preamble or explanation."
)


def expand_query(client, model: str, question: str) -> Optional[str]:
    """Return an expanded Armenian search query, or None on any failure."""
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYS},
                {"role": "user", "content": question},
            ],
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or None
    except Exception:
        return None
