from __future__ import annotations

import re
from typing import List, Dict, Any


def simple_lexical_rerank(query: str, items: List[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
    """Rank by normalized word overlap. Cheap offline fallback used only for the
    non-hybrid (OpenAI) provider path; the hybrid (BGE-M3) pipeline already folds
    in lexical matching via the sparse vector + RRF fusion.
    """
    qwords = set(_normalize(query).split())

    def score(text: str) -> int:
        return sum(1 for w in _normalize(text).split() if w in qwords)

    scored = [{**it, "_lexical_score": score(it.get(text_key, ""))} for it in items]
    scored.sort(key=lambda x: x.get("_lexical_score", 0), reverse=True)
    return scored


def _normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^\w\sԱ-֏]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
