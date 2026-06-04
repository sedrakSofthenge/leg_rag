#!/usr/bin/env python3
# debug_db.py — quick Qdrant inspector for your legal RAG

from __future__ import annotations
import argparse
import os
from typing import Any, Dict, List, Optional

from ingest import load_cfg  # reuse your cfg loader
from utils import Embeddings  # reuse your embedding wrapper

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


# ---------- small helpers ----------
def connect_qdrant(cfg) -> QdrantClient:
    mode = cfg.qdrant.get("mode", "local")
    if mode == "http":
        url = cfg.qdrant.get("url", "http://localhost:6333")
        return QdrantClient(url=url)
    # default: local
    path = cfg.qdrant.get("path", cfg.paths.get("vdb_path", "vdb"))
    os.makedirs(path, exist_ok=True)
    return QdrantClient(path=path)

def make_filter(article: Optional[str], clause: Optional[str], law: Optional[str], extra: Dict[str, Any]|None=None) -> Optional[Filter]:
    must = []
    if article:
        must.append(FieldCondition(key="article", match=MatchValue(value=str(article))))
    if clause:
        must.append(FieldCondition(key="clause", match=MatchValue(value=str(clause))))
    if law:
        must.append(FieldCondition(key="law_title", match=MatchValue(value=law)))
    if extra:
        for k,v in extra.items():
            must.append(FieldCondition(key=k, match=MatchValue(value=v)))
    return Filter(must=must) if must else None

def citation(payload: Dict[str, Any]) -> str:
    law = payload.get("law_title") or payload.get("source_file") or "?"
    art = payload.get("article")
    cl  = payload.get("clause")
    pages = payload.get("page_span") or (payload.get("page_start"), payload.get("page_end"))
    ps = None
    if isinstance(pages, (list, tuple)) and len(pages) >= 2 and pages[0] is not None and pages[1] is not None:
        ps = f"pp. {pages[0]}–{pages[1]}"
    parts = [str(law)]
    if art: parts.append(f"Article {art}")
    if cl:  parts.append(f"Clause {cl}")
    if ps:  parts.append(ps)
    return " → ".join(parts)

def preview_text(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


# ---------- actions ----------
def action_scroll(client: QdrantClient, collection: str, flt: Optional[Filter], limit: int, offset: Optional[str]) -> List[Dict[str, Any]]:
    # Qdrant scroll returns (points, next_offset)
    points, next_offset = client.scroll(
        collection_name=collection,
        scroll_filter=flt,
        with_payload=True,
        with_vectors=False,
        limit=limit,
        offset=offset
    )
    out = []
    for p in points:
        out.append({"id": str(p.id), "payload": p.payload})
    return out, next_offset

def search_by_metadata(client: QdrantClient, collection: str, article: Optional[str], clause: Optional[str], law: Optional[str], limit: int):
    flt = make_filter(article, clause, law)
    pts, _ = client.scroll(collection_name=collection, scroll_filter=flt, with_payload=True, with_vectors=False, limit=limit)
    return [{"id": str(p.id), "payload": p.payload} for p in pts]

def keyword_scan_in_law(client: QdrantClient, collection: str, law: str, contains: str, limit: int):
    flt = make_filter(article=None, clause=None, law=law)
    pts, _ = client.scroll(collection_name=collection, scroll_filter=flt, with_payload=True, with_vectors=False, limit=10_000)
    hits = []
    needle = contains.lower()
    for p in pts:
        text = (p.payload.get("text") or "")
        if needle in text.lower():
            hits.append({"id": str(p.id), "payload": p.payload})
            if len(hits) >= limit:
                break
    return hits

def vector_search(client: QdrantClient, collection: str, emb: Embeddings, query: str, top_k: int, hnsw_ef: Optional[int], threshold: Optional[float]):
    vec = emb.embed_texts([query])[0]
    search_params = {"hnsw_ef": hnsw_ef} if hnsw_ef is not None else None
    res = client.search(
        collection_name=collection,
        query_vector=vec,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        score_threshold=threshold,
        search_params=search_params,
    )
    return [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in res]


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Qdrant debugger for Armenian Legal RAG")
    ap.add_argument("--cfg", default="cfg.yaml", help="Path to cfg.yaml")
    ap.add_argument("--collection", default=None, help="Override collection name")
    ap.add_argument("--article", default=None, help="Filter: article number (e.g., 124.4)")
    ap.add_argument("--clause", default=None, help="Filter: clause id")
    ap.add_argument("--law", default=None, help='Filter: law title (exact match)')
    ap.add_argument("--contains", default=None, help='Case-insensitive substring search within law payload text (requires --law)')
    ap.add_argument("--vector", default=None, help='Vector search text (embed & ANN search)')
    ap.add_argument("--top-k", type=int, default=10, help="Top K for vector search (or max returned for filters)")
    ap.add_argument("--limit", type=int, default=50, help="Limit for metadata/keyword scans")
    ap.add_argument("--hnsw-ef", type=int, default=256, help="ANN breadth for vector search")
    ap.add_argument("--threshold", type=float, default=None, help="Score threshold for vector search")
    ap.add_argument("--preview", type=int, default=220, help="Chars of text to print")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    coll = args.collection or cfg.qdrant.get("collection", "armenian_legal_chunks")
    client = connect_qdrant(cfg)

    # Build embedding only if needed
    emb = None
    if args.vector:
        emb = Embeddings(
            provider=cfg.embeddings.get("provider", "openai"),
            model=cfg.embeddings.get("model", "text-embedding-3-large"),
            dim=int(cfg.embeddings.get("dim", 3072)),
            batch_size=int(cfg.embeddings.get("batch_size", 64)),
            key_file=cfg.embeddings.get("key_file"),
        )

    printed_any = False

    # 1) Metadata filter path
    if args.article or args.clause or args.law:
        rows = search_by_metadata(client, coll, args.article, args.clause, args.law, args.limit)
        print(f"# Metadata filter results: {len(rows)}")
        for r in rows[:args.limit]:
            pl = r["payload"]
            print(f"- {citation(pl)}")
            print(f"  {preview_text(pl.get('text',''), args.preview)}")
        printed_any = True

    # 2) Keyword scan (within a law)
    if args.contains:
        if not args.law:
            print("[error] --contains requires --law (to bound the scan).")
        else:
            rows = keyword_scan_in_law(client, coll, args.law, args.contains, args.limit)
            print(f"# Keyword scan in law='{args.law}' contains='{args.contains}': {len(rows)}")
            for r in rows[:args.limit]:
                pl = r["payload"]
                print(f"- {citation(pl)}")
                print(f"  {preview_text(pl.get('text',''), args.preview)}")
            printed_any = True

    # 3) Vector search path
    if args.vector:
        rows = vector_search(client, coll, emb, args.vector, args.top_k, args.hnsw_ef, args.threshold)
        print(f"# Vector search top-{args.top_k} for: {args.vector!r}")
        for r in rows:
            pl = r["payload"]
            print(f"- score={r['score']:.3f} | {citation(pl)}")
            print(f"  {preview_text(pl.get('text',''), args.preview)}")
        printed_any = True

    if not printed_any:
        print("Nothing to do. Try one of:")
        print("  --article 124.4")
        print("  --law 'Administrative Offences Code' --contains 'km/h'")
        print("  --vector 'speeding penalty 85 km/h' --top-k 20")


if __name__ == "__main__":
    main()
