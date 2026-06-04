import hashlib
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Iterable


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_text_pages(pdf_path: str) -> List[str]:
    """Extract text per-page using pypdf. OCR is not performed here."""
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError(
            "pypdf is required for PDF parsing. Install with `pip install pypdf`."
        ) from e

    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


ARTICLE_RE = re.compile(r"(?m)^\s*Հոդված\s+([0-9]+(?:\.[0-9]+)*)\.?\s*(.*)$")
CHAPTER_RE = re.compile(r"(?m)^\s*ԳԼՈՒԽ\s+([0-9IVXLCDM]+)\.?\s*(.*)$")
CATEGORY_RE = re.compile(r"(?m)^\s*ԲԱԺԻՆ\s+([0-9IVXLCDM]+)\.?\s*(.*)$")

# English headers (some PDFs contain English overlays or translations)
ARTICLE_EN_RE = re.compile(r"(?m)^\s*Article\s+([0-9]+(?:\.[0-9]+)*)\.?\s*(.*)$", re.IGNORECASE)
CHAPTER_EN_RE = re.compile(r"(?m)^\s*(CHAPTER|SECTION|PART)\s+([0-9IVXLCDM]+)\.?\s*(.*)$", re.IGNORECASE)
CLAUSE_RE = re.compile(r"(?m)^\s*(\d+)[\).]\s+")


def _filename_to_title(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return re.sub(r"[_\-]+", " ", name).strip()


def segment_armenian_law(
    pages: List[str],
    source_file: str,
    noise_patterns: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Segment Armenian law text by chapters -> articles -> clauses.
    Returns list of dicts: {law_title, chapter, article, clause, page_start, page_end, text, source_file}
    Heuristics-based; may need tuning for specific corpora.
    """
    law_title = _filename_to_title(source_file)

    segments: List[Dict[str, Any]] = []
    current_chapter: Tuple[str, str] | None = None  # (id, title)
    current_article: Tuple[str, str, int] | None = None  # (id, title, page_start)
    buffer_lines: List[str] = []
    article_start_page = 0

    def flush_article(upto_page_idx: int):
        nonlocal buffer_lines, current_article, current_chapter
        if current_article is None:
            return
        a_id, a_title, a_start_page = current_article
        text_block = "\n".join(buffer_lines).strip()
        if not text_block:
            current_article = None
            buffer_lines = []
            return
        # Split into clauses by regex markers; keep article text if no clauses
        clauses = []
        positions = []
        for m in CLAUSE_RE.finditer(text_block):
            positions.append((m.start(), m.group(1)))
        if not positions:
            segments.append({
                "law_title": law_title,
                "chapter": current_chapter[0] if current_chapter else None,
                "chapter_title": current_chapter[1] if current_chapter else None,
                "article": a_id,
                "article_title": a_title.strip() if a_title else None,
                "clause": None,
                "page_start": a_start_page + 1,
                "page_end": upto_page_idx + 1,
                "text": text_block,
                "source_file": source_file,
            })
        else:
            # Add an artificial end position at the end of text
            positions.append((len(text_block), None))
            for i in range(len(positions) - 1):
                start_pos, clause_id = positions[i]
                end_pos, _ = positions[i + 1]
                clause_text = text_block[start_pos:end_pos].strip()
                segments.append({
                    "law_title": law_title,
                    "chapter": current_chapter[0] if current_chapter else None,
                    "chapter_title": current_chapter[1] if current_chapter else None,
                    "article": a_id,
                    "article_title": a_title.strip() if a_title else None,
                    "clause": clause_id,
                    "page_start": a_start_page + 1,
                    "page_end": upto_page_idx + 1,
                    "text": clause_text,
                    "source_file": source_file,
                })
        current_article = None
        buffer_lines = []

    # Compile noise patterns once
    _compiled_noise = []
    if noise_patterns:
        for pat in noise_patterns:
            try:
                _compiled_noise.append(re.compile(pat, re.IGNORECASE))
            except Exception:
                # Skip invalid regex
                pass

    def _clean_text(text: str) -> str:
        if not text:
            return ""
        # Remove known noisy overlays
        # Example: "Machine Translated by Google" often appears in some PDFs
        text = re.sub(r"Machine\s+Translated\s+by\s+Google", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Translated\s+by\s+Google", "", text, flags=re.IGNORECASE)
        # Apply user-provided noise patterns
        for rx in _compiled_noise:
            text = rx.sub("", text)
        # Collapse excessive spaces introduced by removal
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text

    # Identify repeated short header/footer lines across pages and mark for removal
    # Heuristic: lines appearing on >= 30% of pages and length <= 120
    line_counts: Dict[str, int] = {}
    total_pages = max(1, len(pages))
    for pg in pages:
        if not pg:
            continue
        seen_on_page = set()
        for raw_line in str(pg).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if len(line) > 120 or len(line) < 3:
                continue
            # Avoid overcounting the same line multiple times per page
            if line in seen_on_page:
                continue
            seen_on_page.add(line)
            line_counts[line] = line_counts.get(line, 0) + 1
    repeated_lines = {ln for ln, cnt in line_counts.items() if cnt >= max(3, int(0.3 * total_pages))}

    def _strip_repeated_lines(text: str) -> str:
        if not repeated_lines:
            return text
        kept = []
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if line in repeated_lines:
                continue
            kept.append(raw_line)
        return "\n".join(kept)

    for page_idx, page_text in enumerate(pages):
        if not page_text:
            page_text = ""
        else:
            page_text = _strip_repeated_lines(_clean_text(page_text))
        # Detect chapter on this page
        chm = CHAPTER_RE.search(page_text) or CATEGORY_RE.search(page_text)
        if not chm:
            chm = CHAPTER_EN_RE.search(page_text)
        if chm:
            if chm.re is CHAPTER_EN_RE:
                # English: groups: label, id, title
                current_chapter = (chm.group(2), (chm.group(3) or "").strip())
            else:
                current_chapter = (chm.group(1), (chm.group(2) or "").strip())

        # Detect ALL article starts on this page. Dense legal codes routinely pack
        # several short articles onto one page, so we must split on every header —
        # not just the last one (which would discard every article in between).
        article_matches = list(ARTICLE_RE.finditer(page_text))
        if not article_matches:
            article_matches = list(ARTICLE_EN_RE.finditer(page_text))
        if article_matches:
            # Text before the first header continues the article still open from a
            # previous page; append it, then close that article on this page.
            pre = page_text[: article_matches[0].start()]
            if pre.strip():
                buffer_lines.append(pre)
            flush_article(page_idx)
            # Each header opens a new article whose body runs to the next header
            # (or end of page). Every article but the last on this page is fully
            # contained here, so flush it immediately; the last stays open to
            # continue onto the following page.
            for i, m in enumerate(article_matches):
                a_id = m.group(1)
                a_title = (m.group(2) or "").strip()
                body_end = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(page_text)
                body = page_text[m.end(): body_end]
                current_article = (a_id, a_title, page_idx)
                buffer_lines = [body]
                if i + 1 < len(article_matches):
                    flush_article(page_idx)
        else:
            # Continue accumulating into current article buffer
            if current_article is None:
                # If no current article yet, accumulate page-wise fallback into a pseudo article
                # This ensures we still index text even if headers are not detected.
                # Start a synthetic article with id = None at this page boundary
                if not buffer_lines:
                    current_article = (None, None, page_idx)
                buffer_lines.append(page_text)
            else:
                buffer_lines.append(page_text)

    # Flush last buffered article at end
    flush_article(len(pages) - 1 if pages else 0)

    return segments
