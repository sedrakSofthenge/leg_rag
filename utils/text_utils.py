from __future__ import annotations

import re

_NUM = re.compile(r"\d+")


def numeric_overlap_score(q: str, t: str) -> float:
    """Crude feature: count of numeric overlaps, with a small unit bonus for km/h.
    Helps tie-break for questions that include figures like km/h over-speed.
    """
    qnums = set(_NUM.findall(q or ""))
    tnums = set(_NUM.findall(t or ""))
    if not qnums or not tnums:
        return 0.0
    inter = len(qnums & tnums)
    unit_hits = 0.0
    low_t = (t or "").lower()
    if "km/h" in low_t or "km / h" in low_t or "kmh" in low_t:
        unit_hits += 0.5
    return inter + unit_hits


def normalize_speed_tokens(s: str) -> str:
    """Normalize common MT/locale variants to 'km/h' and collapse whitespace."""
    s = (s or "").replace("km / h", "km/h").replace("kmh", "km/h").replace("км/ч", "km/h")
    s = re.sub(r"\s{2,}", " ", s)
    return s

