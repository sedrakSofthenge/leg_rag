from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re

import yaml

from utils import (
    compute_sha256,
    extract_text_pages,
    segment_armenian_law,
    build_chunks,
    split_text_to_token_windows,
    get_token_counter_for_model,
    trim_to_token_limit,
    Embeddings,
    BGEM3Embedder,
    BGE_DENSE_DIM,
    QdrantConnector,
    BGEReranker,
)

HEADER_FOOTER_PATTERNS = [
    r"^\s*Էջ\s*\d+\s*$",
    r"^\s*Երևան\s*,?\s*\d{4}\s*թ\.?\s*$",
    r"^\s*ՀՀ\s+.*?\s+ՕՐԵՆՔ\s*$",
]

def strip_boilerplate_page(text: str) -> str:
    lines = text.splitlines()
    kept = [ln for ln in lines if not any(re.search(p, ln) for p in HEADER_FOOTER_PATTERNS)]
    return "\n".join(kept)


@dataclass
class Cfg:
    paths: Dict[str, Any]
    qdrant: Dict[str, Any]
    embeddings: Dict[str, Any]
    ingest: Dict[str, Any]
    server: Dict[str, Any]
    reranker: Dict[str, Any] = field(default_factory=dict)


def load_cfg(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # simple env expansion for nested references like ${paths.vdb_path}
    # Here we only expand top-level strings containing ${...}
    def expand(val):
        if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
            key = val[2:-1]
            # support dotted paths in our cfg
            parts = key.split(".")
            ref = raw
            for p in parts:
                ref = ref[p]
            return ref
        return val
    for section in raw:
        if isinstance(raw[section], dict):
            for k, v in list(raw[section].items()):
                raw[section][k] = expand(v)
    return Cfg(
        paths=raw.get("paths", {}),
        qdrant=raw.get("qdrant", {}),
        embeddings=raw.get("embeddings", {}),
        ingest=raw.get("ingest", {}),
        server=raw.get("server", {}),
        reranker=raw.get("reranker", {}),
    )


class ManifestDB:
    def __init__(self, path: str) -> None:
        dirn = os.path.dirname(path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self._init()

    def _init(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                filepath TEXT PRIMARY KEY,
                sha256 TEXT,
                parsed INTEGER DEFAULT 0,
                segmented INTEGER DEFAULT 0,
                chunked INTEGER DEFAULT 0,
                embedded INTEGER DEFAULT 0,
                upserted INTEGER DEFAULT 0,
                law_title TEXT,
                updated_at REAL
            )
            """
        )
        self.conn.commit()

    def get(self, filepath: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM files WHERE filepath=?", (filepath,))
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def upsert(self, filepath: str, sha256: str, **kwargs):
        now = time.time()
        rec = self.get(filepath)
        fields = {
            "filepath": filepath,
            "sha256": sha256,
            "updated_at": now,
            **kwargs,
        }
        if rec is None:
            self.conn.execute(
                """
                INSERT INTO files (filepath, sha256, parsed, segmented, chunked, embedded, upserted, law_title, updated_at)
                VALUES (:filepath, :sha256, :parsed, :segmented, :chunked, :embedded, :upserted, :law_title, :updated_at)
                """,
                {
                    **fields,
                    "parsed": kwargs.get("parsed", 0),
                    "segmented": kwargs.get("segmented", 0),
                    "chunked": kwargs.get("chunked", 0),
                    "embedded": kwargs.get("embedded", 0),
                    "upserted": kwargs.get("upserted", 0),
                    "law_title": kwargs.get("law_title"),
                },
            )
        else:
            # Update
            self.conn.execute(
                """
                UPDATE files SET sha256=:sha256, parsed=:parsed, segmented=:segmented, chunked=:chunked,
                    embedded=:embedded, upserted=:upserted, law_title=:law_title, updated_at=:updated_at
                WHERE filepath=:filepath
                """,
                {
                    **fields,
                    # rec is a dict from self.get(); fall back to existing values by column name
                    "parsed": kwargs.get("parsed", rec.get("parsed", 0)),
                    "segmented": kwargs.get("segmented", rec.get("segmented", 0)),
                    "chunked": kwargs.get("chunked", rec.get("chunked", 0)),
                    "embedded": kwargs.get("embedded", rec.get("embedded", 0)),
                    "upserted": kwargs.get("upserted", rec.get("upserted", 0)),
                    "law_title": kwargs.get("law_title", rec.get("law_title")),
                },
            )
        self.conn.commit()


def build_embedder(cfg: Cfg):
    """Construct the configured embedder. Returns (embedder, vector_dim, hybrid).

    provider 'bge-m3' -> local BGE-M3 (dense 1024 + sparse), hybrid retrieval.
    provider 'openai'/'fake' -> dense-only Embeddings (legacy).
    """
    provider = cfg.embeddings.get("provider", "openai")
    if provider in ("bge-m3", "bge_m3", "bgem3"):
        embs = BGEM3Embedder(
            model_name=cfg.embeddings.get("model", "BAAI/bge-m3"),
            batch_size=int(cfg.embeddings.get("batch_size", 12)),
            max_length=int(cfg.embeddings.get("max_input_tokens", 8192)),
        )
        return embs, BGE_DENSE_DIM, True
    embs = Embeddings(
        provider=provider,
        model=cfg.embeddings.get("model", "text-embedding-3-large"),
        dim=int(cfg.embeddings.get("dim", 3072)),
        batch_size=int(cfg.embeddings.get("batch_size", 64)),
        key_file=cfg.embeddings.get("key_file"),
    )
    return embs, int(cfg.embeddings.get("dim", 3072)), False


def build_reranker(cfg: Cfg):
    """Construct the query-time cross-encoder reranker, or None if disabled.

    Query-time only — never used during ingest. The model loads lazily on first use.
    """
    rc = getattr(cfg, "reranker", {}) or {}
    if not rc.get("enabled", False):
        return None
    if rc.get("provider", "bge") in ("bge", "bge-reranker", "bgem3"):
        return BGEReranker(
            model_name=rc.get("model", "BAAI/bge-reranker-v2-m3"),
            top_n=int(rc.get("top_n", 40)),
            enabled=True,
        )
    return None


def ensure_dirs(paths: Dict[str, Any]):
    os.makedirs(paths["data_raw"], exist_ok=True)
    os.makedirs(paths["data_text"], exist_ok=True)
    os.makedirs(paths["vdb_path"], exist_ok=True)


def jsonl_write(path: str, items: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def ingest_one_pdf(
    pdf_path: str,
    cfg: Cfg,
    mdb: ManifestDB,
    qdrant: QdrantConnector,
    embs: Embeddings,
) -> None:
    sha = compute_sha256(pdf_path)
    file_prefix = sha[:12]
    rec = mdb.get(pdf_path)
    if rec and rec.get("sha256") == sha and rec.get("upserted"):
        print(f"Skip unchanged: {pdf_path}")
        return

    print(f"Extracting text: {pdf_path}")
    pages = extract_text_pages(pdf_path)
    # Strip headers/footers per page to avoid embedding boilerplate
    pages = [strip_boilerplate_page(p) for p in pages]
    mdb.upsert(pdf_path, sha, parsed=1)

    print("Segmenting into articles/clauses…")
    noise_pats = cfg.ingest.get("noise_patterns") or []
    segments = segment_armenian_law(pages, source_file=os.path.basename(pdf_path), noise_patterns=noise_pats)
    law_title = segments[0]["law_title"] if segments else os.path.basename(pdf_path)
    mdb.upsert(pdf_path, sha, parsed=1, segmented=1, law_title=law_title)

    print("Chunking…")
    seed = sha[:12]
    chunks = build_chunks(
        segments,
        chunk_size_tokens=cfg.ingest.get("chunk_size_tokens", 1000),
        chunk_overlap_tokens=cfg.ingest.get("chunk_overlap_tokens", 200),
        seed=seed,
    )
    mdb.upsert(pdf_path, sha, parsed=1, segmented=1, chunked=1, law_title=law_title)

    # Save JSONL for debugging/preview
    out_jsonl = os.path.join(cfg.paths["data_text"], os.path.basename(pdf_path) + ".jsonl")
    jsonl_items = [
        {
            "id": ch["id"],
            "text": ch["text"],
            "metadata": ch["metadata"],
        }
        for ch in chunks
    ]
    jsonl_write(out_jsonl, jsonl_items)

    print("Embedding and upserting to Qdrant…")
    max_inp = int(cfg.embeddings.get("max_input_tokens", 8000))
    # Use tiktoken counter when available for OpenAI
    token_counter = get_token_counter_for_model(cfg.embeddings.get("model", "text-embedding-3-large"))
    # Enforce embedding input size by splitting oversized chunks
    final_ids: List[str] = []
    final_texts: List[str] = []
    final_payloads: List[Dict[str, Any]] = []
    # Optional law_title overrides (e.g., Armenian -> English names)
    title_overrides = (cfg.ingest or {}).get("law_title_overrides", {})

    for c in chunks:
        text = c["text"]
        cid = c["id"]
        meta_base = dict(c["metadata"])  # shallow copy
        # Apply law_title override if configured
        lt = meta_base.get("law_title")
        if lt and lt in title_overrides:
            meta_base["law_title"] = title_overrides[lt]
        windows = split_text_to_token_windows(
            text,
            max_tokens=max_inp,
            overlap_tokens=cfg.ingest.get("chunk_overlap_tokens", 200),
            token_counter=token_counter,
        )
        if len(windows) == 1:
            safe_text = text
            if token_counter(safe_text) > max_inp:
                safe_text = trim_to_token_limit(safe_text, max_inp, token_counter)
            final_ids.append(f"{file_prefix}:{cid}")
            final_texts.append(safe_text)
            meta = {**meta_base, "sha256": sha, "source_file": os.path.basename(pdf_path), "text": safe_text}
            final_payloads.append(meta)
        else:
            for i, w in enumerate(windows):
                if token_counter(w) > max_inp:
                    w = trim_to_token_limit(w, max_inp, token_counter)
                # wid = f"{cid}-p{i+1}"
                # final_ids.append(wid)
                wid = f"{file_prefix}:{cid}-p{i+1}"
                final_ids.append(wid)
                final_texts.append(w)
                meta = {**meta_base, "sha256": sha, "source_file": os.path.basename(pdf_path), "text": w, "parent_id": cid, "subchunk_index": i+1, "subchunk_total": len(windows)}
                final_payloads.append(meta)

    if getattr(qdrant, "hybrid", False):
        # BGE-M3: dense + sparse in one pass, upsert both for hybrid retrieval.
        emb_out = embs.embed(final_texts)
        qdrant.upsert_hybrid(final_ids, emb_out["dense"], emb_out["sparse"], final_payloads)
    else:
        vectors = embs.embed_texts(final_texts)
        qdrant.upsert(final_ids, vectors, final_payloads)
    mdb.upsert(pdf_path, sha, parsed=1, segmented=1, chunked=1, embedded=1, upserted=1, law_title=law_title)
    print(f"Done: {pdf_path} ({len(chunks)} chunks)")


def ingest_all(pdf_dir: Optional[str], cfg_path: str) -> None:
    cfg = load_cfg(cfg_path)
    ensure_dirs(cfg.paths)
    # Fall back to the config's paths.data_raw when --pdf-dir isn't given.
    if not pdf_dir:
        pdf_dir = cfg.paths.get("data_raw", "data_raw")

    mdb = ManifestDB(cfg.paths["manifest_sqlite"])

    embs, vector_dim, hybrid = build_embedder(cfg)

    q = QdrantConnector(
        collection=cfg.qdrant.get("collection", "armenian_legal_chunks"),
        vector_size=vector_dim,
        distance=cfg.qdrant.get("distance", "cosine"),
        mode=cfg.qdrant.get("mode", "local"),
        url=cfg.qdrant.get("url"),
        path=cfg.qdrant.get("path", cfg.paths.get("vdb_path")),
        hybrid=hybrid,
    )

    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return

    for pdf in pdfs:
        try:
            ingest_one_pdf(pdf, cfg, mdb, q, embs)
        except Exception as e:
            print(f"Error ingesting {pdf}: {e}")


def main():
    ap = argparse.ArgumentParser(description="Ingest Armenian legal PDFs into Qdrant")
    ap.add_argument("--pdf-dir", default=None, help="Directory with PDFs (defaults to cfg paths.data_raw)")
    ap.add_argument("--cfg", default="cfg.yaml")
    args = ap.parse_args()
    ingest_all(args.pdf_dir, args.cfg)


if __name__ == "__main__":
    main()
