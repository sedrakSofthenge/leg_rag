from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


_KMH = re.compile(r"(?i)(?:exceed(?:ing|s|ed)|over)\s*(?:the\s*speed\s*limit\s*by)?\s*(\d{1,3})\s*km\s*/?\s*h")
_DEVICE_HINT = re.compile(r"(?i)(speedometer|radar|camera|reading|device)")


def maybe_extract_over_speed_kmh(q: str) -> Optional[int]:
    """Return the numeric km/h excess if present in the question, else None."""
    if not q:
        return None
    m = _KMH.search(q)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def classify_speed(over_kmh: int, deduct_10: bool) -> Tuple[int, Dict[str, str]]:
    """Classify effective over-speed into legal brackets per Article 124.4.

    Returns (effective_over_kmh, details_dict).
    details contains keys like: fine, points, sanction, article.
    """
    eff = max(0, over_kmh - (10 if deduct_10 else 0))
    if 1 <= eff <= 10:
        return eff, {"fine": "per-km @ 1× minimum wage each", "points": "0", "article": "124.4(1)"}
    if 11 <= eff <= 30:
        return eff, {"fine": "20× minimum wage", "points": "2", "article": "124.4(2)"}
    if 31 <= eff <= 50:
        return eff, {"fine": "25× minimum wage", "points": "3", "article": "124.4(3)"}
    if 51 <= eff <= 80:
        return eff, {"fine": "29× minimum wage", "points": "4", "article": "124.4(4)"}
    if eff >= 81:
        return eff, {"sanction": "license deprivation for 1 year", "fine": "200× minimum wage if already deprived", "article": "124.4(5)"}
    return eff, {"fine": "—", "article": "124.4"}


def format_speed_answer(q: str) -> Optional[str]:
    """Produce a compact, user-facing line explaining the computed bracket.

    If no numeric over-speed found, return None.
    Prefers device-reading deduction when the question mentions a device context.
    """
    n = maybe_extract_over_speed_kmh(q)
    if n is None:
        return None
    eff_no_deduct, p1 = classify_speed(n, deduct_10=False)
    eff_deduct, p2 = classify_speed(n, deduct_10=True)
    prefer_deduct = bool(_DEVICE_HINT.search(q or ""))
    picked = p2 if prefer_deduct else p1
    eff = eff_deduct if prefer_deduct else eff_no_deduct

    if "sanction" in picked:
        core = f"{picked['sanction']}; if already deprived: {picked['fine']}."
    else:
        pts = f", penalty points: {picked.get('points','0')}" if 'points' in picked else ""
        core = f"Fine: {picked['fine']}{pts}."
    note = (
        "Per Article 124.4(6), subtract 10 km/h from device readings; "
        "if your number is a speedometer/camera reading, effective excess = "
        f"{eff} km/h → {picked['article']}."
    )
    return f"Computed speeding interpretation: {core} (Article {picked['article']}). {note}"

