import math
import hashlib
from typing import List, Dict, Any, Callable, Optional


def approx_token_count(text: str) -> int:
    # Rough heuristic: ~4 chars/token
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _chunk_id(seed: str, idx: int) -> str:
    h = hashlib.sha1(f"{seed}:{idx}".encode("utf-8")).hexdigest()
    return h


def build_chunks(
    segments: List[Dict[str, Any]],
    chunk_size_tokens: int = 1000,
    chunk_overlap_tokens: int = 200,
    seed: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Build chunks from clause segments. Do not split inside a clause.
    Overlap is implemented by re-adding trailing clauses until overlap size is met.
    Returns list of {id, text, metadata}.
    """
    chunks: List[Dict[str, Any]] = []
    cur_texts: List[str] = []
    cur_metas: List[Dict[str, Any]] = []
    cur_tokens = 0
    seed = seed or "seed"

    for seg in segments:
        seg_text = seg.get("text", "")
        seg_tok = approx_token_count(seg_text)
        if seg_tok > chunk_size_tokens:
            # If a single clause is huge, still make it a standalone chunk
            if cur_texts:
                chunks.append(_finalize_chunk(chunks, cur_texts, cur_metas, seed))
                cur_texts, cur_metas, cur_tokens = [], [], 0
            cur_texts = [seg_text]
            cur_metas = [seg]
            cur_tokens = seg_tok
            chunks.append(_finalize_chunk(chunks, cur_texts, cur_metas, seed))
            cur_texts, cur_metas, cur_tokens = [], [], 0
            continue

        if cur_tokens + seg_tok <= chunk_size_tokens:
            cur_texts.append(seg_text)
            cur_metas.append(seg)
            cur_tokens += seg_tok
        else:
            # finalize current chunk
            chunks.append(_finalize_chunk(chunks, cur_texts, cur_metas, seed))
            # start next chunk with overlap: include trailing clauses until overlap satisfied
            overlap_texts, overlap_metas, overlap_tok = _take_overlap(cur_texts, cur_metas, chunk_overlap_tokens)
            cur_texts, cur_metas, cur_tokens = overlap_texts[:], overlap_metas[:], overlap_tok
            # add current seg
            cur_texts.append(seg_text)
            cur_metas.append(seg)
            cur_tokens += seg_tok

    if cur_texts:
        chunks.append(_finalize_chunk(chunks, cur_texts, cur_metas, seed))

    return chunks


def _take_overlap(texts: List[str], metas: List[Dict[str, Any]], target_tokens: int):
    out_texts: List[str] = []
    out_metas: List[Dict[str, Any]] = []
    tok = 0
    # Accumulate from end backwards until target token count reached
    for t, m in zip(reversed(texts), reversed(metas)):
        t_tok = approx_token_count(t)
        if tok + t_tok > target_tokens and out_texts:
            break
        out_texts.insert(0, t)
        out_metas.insert(0, m)
        tok += t_tok
        if tok >= target_tokens:
            break
    return out_texts, out_metas, tok


def _finalize_chunk(chunks: List[Dict[str, Any]], texts: List[str], metas: List[Dict[str, Any]], seed: str) -> Dict[str, Any]:
    if not texts:
        return {}
    text = "\n".join([t.strip() for t in texts if t]).strip()
    if not text:
        return {}
    idx = len(chunks)
    cid = _chunk_id(seed, idx)
    # Derive combined metadata
    first = metas[0] if metas else {}
    last = metas[-1] if metas else {}
    meta = {
        "law_title": first.get("law_title"),
        "article": first.get("article"),
        "clause": first.get("clause"),
        "chapter": first.get("chapter"),
        "page_span": [first.get("page_start"), last.get("page_end")],
        "source_file": first.get("source_file"),
        "article_title": first.get("article_title"),
        "chapter_title": first.get("chapter_title"),
        # Keep a small list of clause ids included (for citation clarity)
        "clauses_included": [m.get("clause") for m in metas],
    }
    return {"id": cid, "text": text, "metadata": meta}


def split_text_to_token_windows(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 200,
    token_counter: Optional[Callable[[str], int]] = None,
) -> List[str]:
    """Split a long text into windows not exceeding max_tokens (approximate).
    Heuristic: prefer paragraph boundaries, then sentences, then hard split.
    """
    count = token_counter or approx_token_count
    if count(text) <= max_tokens:
        return [text]

    # Try paragraph-based packing
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) <= 1:
        paras = [p.strip() for p in text.split("\n") if p.strip()]

    windows: List[str] = []
    cur: List[str] = []
    cur_tok = 0
    for p in paras:
        t = count(p)
        if t > max_tokens:
            # sentence-level or hard split
            windows.extend(_split_hard(p, max_tokens, overlap_tokens, count))
            continue
        if cur_tok + t <= max_tokens:
            cur.append(p)
            cur_tok += t
        else:
            if cur:
                windows.append("\n\n".join(cur))
                # overlap from end
                ov_text = _take_overlap_text(cur, overlap_tokens, count)
                cur = [ov_text] if ov_text else []
                cur_tok = count(ov_text) if ov_text else 0
            cur.append(p)
            cur_tok += t
    if cur:
        windows.append("\n\n".join(cur))
    # Ensure none exceed max; if any do due to estimates, hard split
    final: List[str] = []
    for w in windows:
        if count(w) > max_tokens:
            final.extend(_split_hard(w, max_tokens, overlap_tokens, count))
        else:
            final.append(w)
    return final


def _take_overlap_text(parts: List[str], target_tokens: int, count: Callable[[str], int]) -> str:
    tok = 0
    out: List[str] = []
    for p in reversed(parts):
        t = count(p)
        if tok + t > target_tokens and out:
            break
        out.insert(0, p)
        tok += t
        if tok >= target_tokens:
            break
    return "\n\n".join(out).strip()


def _split_hard(text: str, max_tokens: int, overlap_tokens: int, count: Callable[[str], int]) -> List[str]:
    words = text.split()
    windows: List[str] = []
    cur: List[str] = []
    cur_tok = 0
    for w in words:
        t = count(w + " ")
        if cur_tok + t <= max_tokens:
            cur.append(w)
            cur_tok += t
        else:
            if cur:
                windows.append(" ".join(cur))
                # add overlap
                ov_words = []
                tok = 0
                for ww in reversed(cur):
                    tw = count(ww + " ")
                    if tok + tw > overlap_tokens and ov_words:
                        break
                    ov_words.insert(0, ww)
                    tok += tw
                cur = ov_words
                cur_tok = tok
            cur.append(w)
            cur_tok += t
    if cur:
        windows.append(" ".join(cur))
    return windows
