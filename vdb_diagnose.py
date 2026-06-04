from __future__ import annotations

import argparse
import math
import re
import statistics as stats
from typing import Any, Dict, List, Optional, Tuple

from utils import Embeddings, QdrantConnector
from ingest import load_cfg  # reuse pipeline loader to expand ${...}


def looks_english(text: str) -> bool:
    if not text:
        return False
    # Filter out Armenian characters
    if re.search(r"[\u0531-\u058F]", text):
        return False
    letters = re.findall(r"[A-Za-z]", text)
    if not letters:
        return False
    total = len(re.findall(r"\w", text)) or 1
    return (len(letters) / total) >= 0.5


def collect_samples(cfg: Dict[str, Any], max_samples: int, english_only: bool, collection_override: Optional[str] = None) -> List[Dict[str, Any]]:
    # Use qdrant_client directly for scrolling
    try:
        from qdrant_client import QdrantClient
    except Exception as e:
        raise RuntimeError("qdrant-client is required to diagnose the DB. pip install qdrant-client")

    mode = cfg["qdrant"].get("mode", "local")
    path = cfg["qdrant"].get("path")
    url = cfg["qdrant"].get("url")
    client = QdrantClient(path=path) if mode == "local" else QdrantClient(url=url)
    # Validate collection exists
    existing = [c.name for c in client.get_collections().collections]
    col = collection_override or cfg["qdrant"].get("collection")
    if col not in existing:
        raise SystemExit(
            f"Collection '{col}' not found. Existing: {existing}. Using {'path='+str(path) if mode=='local' else 'url='+str(url)}"
        )
    pts: List[Any] = []
    next_page = None
    while len(pts) < max_samples:
        limit = min(256, max_samples - len(pts))
        batch, next_page = client.scroll(
            collection_name=col,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            offset=next_page,
        )
        if not batch:
            break
        for p in batch:
            payload = p.payload or {}
            text = (payload.get("text") or "").strip()
            if not text:
                continue
            if english_only and not looks_english(text):
                continue
            pts.append({
                "id": str(p.id),
                "text": text,
                "payload": payload,
            })
        if next_page is None:
            break
    return pts


def evaluate_retrieval(cfg: Dict[str, Any], samples: List[Dict[str, Any]], top_k: int, provider_override: Optional[str] = None) -> Dict[str, Any]:
    provider = provider_override or cfg["embeddings"].get("provider", "openai")
    emb = Embeddings(
        provider=provider,
        model=cfg["embeddings"].get("model", "text-embedding-3-large"),
        dim=int(cfg["embeddings"].get("dim", 3072)),
        batch_size=int(cfg["embeddings"].get("batch_size", 64)),
        key_file=cfg["embeddings"].get("key_file"),
    )
    q = QdrantConnector(
        collection=cfg["qdrant"].get("collection"),
        vector_size=int(cfg["embeddings"].get("dim", 3072)),
        distance=cfg["qdrant"].get("distance", "cosine"),
        mode=cfg["qdrant"].get("mode", "local"),
        url=cfg["qdrant"].get("url"),
        path=cfg["qdrant"].get("path"),
    )

    texts = [s["text"] for s in samples]
    vecs = emb.embed_texts(texts)

    self_at_1 = 0
    self_rank: List[int] = []
    top1_scores: List[float] = []
    top1_same_law = 0
    topk_same_law_ratio: List[float] = []
    score_drops: List[float] = []

    for s, v in zip(samples, vecs):
        res = q.search(v, top_k=top_k)
        if not res:
            continue
        ids = [str(r["id"]) for r in res]
        scores = [float(r["score"]) for r in res]
        payloads = [r.get("payload", {}) for r in res]

        # self rank
        if s["id"] in ids:
            rnk = ids.index(s["id"]) + 1
            self_rank.append(rnk)
            if rnk == 1:
                self_at_1 += 1
        else:
            self_rank.append(top_k + 1)

        # score stats
        top1_scores.append(scores[0])
        if len(scores) >= 2:
            score_drops.append(scores[0] - scores[1])

        # law coherence
        law = (s["payload"] or {}).get("law_title")
        same_law = 0
        for p in payloads:
            if p and p.get("law_title") == law:
                same_law += 1
        top1_same_law += 1 if (payloads[0] or {}).get("law_title") == law else 0
        topk_same_law_ratio.append(same_law / len(payloads))

    n = len(samples) or 1
    out = {
        "samples_evaluated": n,
        "self_hit_at_1": round(self_at_1 / n, 3),
        "self_rank_median": int(stats.median(self_rank)) if self_rank else None,
        "top1_score_mean": round(stats.fmean(top1_scores), 4) if top1_scores else None,
        "top1_score_median": round(stats.median(top1_scores), 4) if top1_scores else None,
        "top1_minus_top2_mean": round(stats.fmean(score_drops), 4) if score_drops else None,
        "topk_same_law_mean": round(stats.fmean(topk_same_law_ratio), 3) if topk_same_law_ratio else None,
        "top1_same_law_rate": round(top1_same_law / n, 3),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="Diagnose Qdrant vector DB retrieval quality")
    ap.add_argument("--cfg", default="cfg.yaml", help="Path to cfg.yaml")
    ap.add_argument("--collection", default=None, help="Override collection name (optional)")
    ap.add_argument("--provider", default=None, help="Override embeddings provider (e.g., openai|fake)")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--english-only", action="store_true", default=False, help="Restrict diagnostics to English-looking chunks")
    args = ap.parse_args()

    cfg_obj = load_cfg(args.cfg)
    # Convert to plain dict for downstream use
    cfg = {
        "paths": cfg_obj.paths,
        "qdrant": cfg_obj.qdrant,
        "embeddings": cfg_obj.embeddings,
        "ingest": cfg_obj.ingest,
        "server": getattr(cfg_obj, "server", {}),
    }

    samples = collect_samples(cfg, max_samples=args.max_samples, english_only=args.english_only, collection_override=args.collection)
    if not samples:
        print("No suitable samples found. Consider increasing --max-samples or disabling --english-only.")
        return

    report = evaluate_retrieval(cfg, samples, top_k=args.top_k, provider_override=args.provider)
    print("Vector DB Diagnostics")
    print("- collection:", args.collection or cfg["qdrant"]["collection"])
    print("- samples:", report["samples_evaluated"])
    print("- self@1:", report["self_hit_at_1"], " median self-rank:", report["self_rank_median"])
    print("- top1 score mean/median:", report["top1_score_mean"], report["top1_score_median"])
    print("- top1-top2 drop mean:", report["top1_minus_top2_mean"])
    print("- same-law in top-k (mean):", report["topk_same_law_mean"], " top1 same-law rate:", report["top1_same_law_rate"])


if __name__ == "__main__":
    main()
