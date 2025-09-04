import hashlib
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple


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
CLAUSE_RE = re.compile(r"(?m)^\s*(\d+)[\).]\s+")


def _filename_to_title(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return re.sub(r"[_\-]+", " ", name).strip()


def segment_armenian_law(pages: List[str], source_file: str) -> List[Dict[str, Any]]:
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

    for page_idx, page_text in enumerate(pages):
        if not page_text:
            page_text = ""
        # Detect chapter on this page
        chm = CHAPTER_RE.search(page_text) or CATEGORY_RE.search(page_text)
        if chm:
            current_chapter = (chm.group(1), (chm.group(2) or "").strip())

        # Detect article starts on this page. There could be multiple occurrences, but
        # in legal docs typically it's one article header per start.
        article_matches = list(ARTICLE_RE.finditer(page_text))
        if article_matches:
            # flush existing article up to previous page
            flush_article(page_idx - 1 if page_idx > 0 else 0)
            # Keep everything after the last article marker for the new article
            last = article_matches[-1]
            a_id = last.group(1)
            a_title = (last.group(2) or "").strip()
            current_article = (a_id, a_title, page_idx)
            # Capture rest of the page after the header
            rest = page_text[last.end():]
            buffer_lines = [rest]
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

