from __future__ import annotations

import os
import re
from typing import List, Dict, Any


def simple_lexical_rerank(query: str, items: List[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
    """
    Simple lexical reranker: ranks by overlap of normalized words.
    Useful as an offline fallback.
    """
    qwords = set(_normalize(query).split())
    def score(text: str) -> int:
        return sum(1 for w in _normalize(text).split() if w in qwords)
    scored = [
        {**it, "_lexical_score": score(it.get(text_key, ""))}
        for it in items
    ]
    scored.sort(key=lambda x: x.get("_lexical_score", 0), reverse=True)
    return scored


def _normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^\w\s\u0531-\u058F]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


class Reranker:
    def __init__(self, enabled: bool = False, provider: str = "none", api_key: str | None = None, key_file: str | None = None) -> None:
        self.enabled = enabled
        self.provider = provider
        self._client = None
        if enabled and provider == "openai":
            try:
                from openai import OpenAI  # type: ignore
                key = api_key or _read_key_file(key_file) or os.environ.get("OPENAI_API_KEY")
                if not key:
                    # If no key, disable gracefully
                    self.enabled = False
                else:
                    self._client = OpenAI(api_key=key)
            except Exception:
                self.enabled = False

    def rerank(self, query: str, items: List[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
        if not self.enabled:
            return items
        if self.provider == "openai" and self._client is not None:
            # Basic LLM reranker: ask model to rank by relevance (kept concise for cost)
            # Fallback to lexical if any error occurs
            try:
                prompt = self._format_prompt(query, items, text_key)
                resp = self._client.chat.completions.create(
                    model="gpt-5",
                    temperature=0,
                    messages=[{"role": "system", "content": "Rank passages by relevance to the query. Return a list of indices 0..N-1 in descending order of relevance as JSON array."},
                              {"role": "user", "content": prompt}],
                )
                content = resp.choices[0].message.content
                import json
                order = json.loads(content)
                ordered = [items[i] for i in order if 0 <= i < len(items)]
                # Keep the rest in original order
                remaining = [it for i, it in enumerate(items) if i not in set(order)]
                return ordered + remaining
            except Exception:
                return simple_lexical_rerank(query, items, text_key)
        # default fallback
        return simple_lexical_rerank(query, items, text_key)

    def _format_prompt(self, query: str, items: List[Dict[str, Any]], text_key: str) -> str:
        lines = [f"Query: {query}", "Passages:"]
        for i, it in enumerate(items):
            text = it.get(text_key, "")
            lines.append(f"[{i}] {text[:500]}")
        lines.append("Return JSON array of indices.")
        return "\n".join(lines)


def _read_key_file(key_file: str | None) -> str | None:
    if not key_file:
        return None
    import os
    path = os.path.expanduser(key_file)
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            key = f.read().strip()
            return key or None
    except Exception:
        return None
