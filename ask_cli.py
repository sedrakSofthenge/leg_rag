from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

from ingest import load_cfg, build_embedder, build_reranker
from utils.query_expand import expand_query
from utils import (
    QdrantConnector,
    simple_lexical_rerank,
    normalize_speed_tokens,
    numeric_overlap_score,
    format_speed_answer,
)


def _is_armenian(text: str) -> bool:
    return any("\u0531" <= ch <= "\u058F" for ch in text)


def _make_citation(payload: Dict[str, Any], lang: str = "en") -> str:
    law = payload.get("law_title") or payload.get("source_file")
    art = payload.get("article")
    cl = payload.get("clause")
    pages = payload.get("page_span") or (payload.get("page_start"), payload.get("page_end"))
    pages_str = None
    if isinstance(pages, (list, tuple)):
        try:
            pages_str = f"pp. {int(pages[0])}-{int(pages[1])}"
        except Exception:
            pages_str = None
    parts = [str(law)]
    if art:
        parts.append((f"Հոդված {art}") if lang == "hy" else (f"Article {art}"))
    if cl:
        parts.append((f"կետ {cl}") if lang == "hy" else (f"Clause {cl}"))
    if pages_str:
        parts.append(pages_str)
    return " → ".join(parts)


def _compose_answer(question: str, lang: str, items: List[Dict[str, Any]]) -> str:
    top = items[:3]
    if lang == "hy":
        lines = ["Պատասխան (նախնական)"]
        for it in top:
            payload = it.get("payload", {})
            text = it.get("text") or payload.get("text") or "…"
            quote = (text[:240] + "…") if len(text) > 240 else text
            lines.append(f"- Մեջբերում: \"{quote}\"")
            lines.append(f"  Աղբյուր: [{_make_citation(payload, 'hy')}]")
        return "\n".join(lines)
    else:
        lines = ["Answer (template)"]
        for it in top:
            payload = it.get("payload", {})
            text = it.get("text") or payload.get("text") or "…"
            quote = (text[:240] + "…") if len(text) > 240 else text
            lines.append(f"- Quote: \"{quote}\"")
            lines.append(f"  Citation: [{_make_citation(payload, 'en')}]")
        return "\n".join(lines)


def run_interactive(cfg_path: str, top_k: int, lang: Optional[str]) -> None:
    cfg = load_cfg(cfg_path)
    emb, vector_dim, hybrid = build_embedder(cfg)
    q = QdrantConnector(
        collection=cfg.qdrant.get("collection", "armenian_legal_chunks"),
        vector_size=vector_dim,
        distance=cfg.qdrant.get("distance", "cosine"),
        mode=cfg.qdrant.get("mode", "local"),
        url=cfg.qdrant.get("url"),
        path=cfg.qdrant.get("path", cfg.paths.get("vdb_path")),
        hybrid=hybrid,
    )
    bge_reranker = build_reranker(cfg)  # query-time cross-encoder, None if disabled
    if bge_reranker is not None:
        print(f"[reranker] BGE cross-encoder enabled (top_n={bge_reranker.top_n}); loads on first query")

    # Optional LLM answer configuration
    server_cfg = cfg.server if hasattr(cfg, "server") else {}
    llm_cfg = server_cfg.get("llm_answer", {}) if isinstance(server_cfg, dict) else {}
    llm_enabled = bool(llm_cfg.get("enabled", False))
    llm_model = llm_cfg.get("model", "gpt-5")
    llm_temp = float(llm_cfg.get("temperature", 0.2))
    llm_max_ctx = int(llm_cfg.get("max_context_passages", 8))

    openai_client = None
    try:
        from openai import OpenAI  # type: ignore
        key = None
        key_file = cfg.embeddings.get("key_file")
        if key_file and os.path.exists(key_file):
            with open(key_file, "r", encoding="utf-8") as f:
                key = (f.read() or "").strip() or None
        key = key or os.environ.get("OPENAI_API_KEY")
        if key:
            openai_client = OpenAI(api_key=key)
        elif llm_enabled:
            # keep final-answer disabled if no key
            llm_enabled = False
    except Exception:
        if llm_enabled:
            llm_enabled = False

    print("Interactive Armenian Legal RAG. Type :q to quit, :help for help.")
    while True:
        try:
            query = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query in (":q", ":quit", ":exit"):
            break
        if query in (":help", ":h"):
            print("Commands: :q to quit. Ask any question in Armenian or English.")
            continue

        try:
            l = lang or ("hy" if _is_armenian(query) else "en")
            # Expand the colloquial question into legal terminology for retrieval.
            # The original `query` still drives the final answer; this only affects
            # which passages we fetch.
            search_query = query
            if openai_client is not None:
                rq = expand_query(openai_client, llm_model, query)
                if rq:
                    search_query = f"{query} {rq}"
                    print(f"[expanded] {rq[:200]}")

            if hybrid:
                qe = emb.embed_one(search_query)
                results = q.search_hybrid(
                    qe["dense"], qe["sparse"],
                    top_k=max(120, top_k),
                )
            else:
                results = q.search(
                    emb.embed_texts([search_query])[0],
                    top_k=max(120, top_k),     # pull more
                    hnsw_ef=256,               # better recall
                    score_threshold=None,      # don't drop candidates
                    with_payload=True,
                    with_vectors=False,
                )

            print("---- raw top retrieved ----")
            for r in results[:8]:
                s = r.get("score", 0.0)
                t = (normalize_speed_tokens(r.get("payload", {}).get("text",""))[:200]).replace("\n"," ")
                print(f"{s:.3f} :: {t}")

            # Attach text for reranking and display
            items = []

            for r in results:
                payload = r.get("payload", {})
                text = normalize_speed_tokens(payload.get("text") or "")
                items.append({"text": text, **r})

            # In hybrid (RRF) mode the fused order already blends dense + lexical;
            # the numeric/article boosts and lexical rerank are tuned for raw cosine
            # and would swamp the small RRF scores, so skip them.
            if not hybrid:
                # Light numeric and article label boosts
                q_low = query.lower()
                for it in items:
                    base = float(it.get("score") or 0.0)
                    numsig = numeric_overlap_score(query, it["text"])
                    # light blend: keep ANN as primary, numbers as tie-breaker
                    score = base + 0.15 * numsig
                    # exact article label booster for queries mentioning e.g., 124.4
                    art = (it.get("payload", {}) or {}).get("article")
                    if art and str(art).lower() in q_low:
                        score += 0.2
                    it["score"] = score

            # Cross-encoder rerank (query-time); lexical fallback only helps the
            # non-hybrid (OpenAI) path.
            if bge_reranker is not None:
                items = bge_reranker.rerank(query, items, text_key="text")
            elif not hybrid:
                items = simple_lexical_rerank(query, items, text_key="text")
            answer = None

            items = items[:top_k]

            print("---- top retrieved (after rerank/trim) ----")
            for it in items[:min(8, len(items))]:
                s = it.get("score") or it.get("payload", {}).get("score")
                snippet = (it.get("text") or it.get("payload", {}).get("text",""))[:200].replace("\n"," ")
                print(f"{(s or 0):.3f} :: {snippet}")


            if llm_enabled and openai_client:
                # Build compact context and ask the model
                ctx = []
                for it in items[:llm_max_ctx]:
                    p = it.get("payload", {})
                    text = (it.get("text") or p.get("text") or "")[:2000]
                    if not text:
                        continue
                    cite = _make_citation(p, l)
                    ctx.append({"text": text, "cite": cite})
                sys_prompt = (
                    "You are a precise legal assistant for Armenian law."
                    " Use only the provided passages. Quote verbatim Armenian when citing."
                )
                calc_line = format_speed_answer(query)
                if l == "hy":
                    user_prompt = (
                        f"Հարցը: {query}\n\n"
                        + (f"(Հաշվարկված մեկնաբանություն) {calc_line}\n" if calc_line else "")
                        + "Ապավինիր միայն հետևյալ հատվածներին. Յուրաքանչյուր մեջբերման կողքին ավելացրու հղումը [Օրենք → Հոդված → Կետ → էջեր].\n"
                        "Պատասխանը ներկայացրու հայերենով։\n\n"
                        + "\n\n".join([f"- [{_make_citation(it.get('payload', {}), 'hy')}]\n{it.get('text')}" for it in items[:llm_max_ctx]])
                    )
                else:
                    user_prompt = (
                        f"Question: {query}\n\n"
                        + (f"Computed interpretation: {calc_line}\n" if calc_line else "")
                        + "Rely only on the passages below. Provide: (1) a concise English answer, (2) one short English quote, (3) citations [Law → Article → Clause → pages].\n\n"
                        + "\n\n".join([f"- [{_make_citation(it.get('payload', {}), 'en')}]\n{it.get('text')}" for it in items[:llm_max_ctx]])
                    )
                try:
                    kwargs = {
                        "model": llm_model,
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    }
                    if llm_temp is not None and llm_temp != 1:
                        kwargs["temperature"] = llm_temp
                    try:
                        resp = openai_client.chat.completions.create(**kwargs)
                    except Exception as e:
                        msg = str(e)
                        if "temperature" in msg and "Unsupported value" in msg:
                            kwargs.pop("temperature", None)
                            resp = openai_client.chat.completions.create(**kwargs)
                        else:
                            raise
                    answer = resp.choices[0].message.content
                except Exception as e:
                    print(f"[warn] LLM failed, falling back to template: {e}")
            if not answer:
                # Prepend computed speeding interpretation if available
                calc_line = format_speed_answer(query)
                prefix = (calc_line + "\n\n") if calc_line else ""
                answer = prefix + _compose_answer(query, l, items)
            print(answer)
        except Exception as e:
            print(f"Error: {e}")


def main():
    ap = argparse.ArgumentParser(description="Interactive CLI for Armenian Legal RAG")
    ap.add_argument("--cfg", default="cfg.yaml")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--lang", choices=["hy", "en"], default=None)
    args = ap.parse_args()
    run_interactive(args.cfg, args.top_k, args.lang)


if __name__ == "__main__":
    main()
