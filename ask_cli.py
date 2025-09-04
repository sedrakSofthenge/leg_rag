from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

from ingest import load_cfg
from utils import Embeddings, QdrantConnector, Reranker, simple_lexical_rerank


def _is_armenian(text: str) -> bool:
    return any("\u0531" <= ch <= "\u058F" for ch in text)


def _make_citation(payload: Dict[str, Any]) -> str:
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
        parts.append(f"Հոդված {art}")
    if cl:
        parts.append(f"կետ {cl}")
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
            lines.append(f"  Աղբյուր: [{_make_citation(payload)}]")
        return "\n".join(lines)
    else:
        lines = ["Answer (template)"]
        for it in top:
            payload = it.get("payload", {})
            text = it.get("text") or payload.get("text") or "…"
            quote = (text[:240] + "…") if len(text) > 240 else text
            lines.append(f"- Armenian quote: \"{quote}\"")
            lines.append(f"  Citation: [{_make_citation(payload)}]")
        return "\n".join(lines)


def run_interactive(cfg_path: str, top_k: int, lang: Optional[str]) -> None:
    cfg = load_cfg(cfg_path)
    # print("running", cfg)  # optional debug

    emb = Embeddings(
        provider=cfg.embeddings.get("provider", "openai"),
        model=cfg.embeddings.get("model", "text-embedding-3-large"),
        dim=int(cfg.embeddings.get("dim", 3072)),
        batch_size=int(cfg.embeddings.get("batch_size", 64)),
        key_file=cfg.embeddings.get("key_file"),
    )
    q = QdrantConnector(
        collection=cfg.qdrant.get("collection", "armenian_legal_chunks"),
        vector_size=int(cfg.embeddings.get("dim", 3072)),
        distance=cfg.qdrant.get("distance", "cosine"),
        mode=cfg.qdrant.get("mode", "local"),
        url=cfg.qdrant.get("url"),
        path=cfg.qdrant.get("path", cfg.paths.get("vdb_path")),
    )
    rr = Reranker(enabled=bool(cfg.ingest.get("reranker_enabled", False)), provider="openai", key_file=cfg.embeddings.get("key_file"))

    # Optional LLM answer configuration
    server_cfg = cfg.server if hasattr(cfg, "server") else {}
    llm_cfg = server_cfg.get("llm_answer", {}) if isinstance(server_cfg, dict) else {}
    llm_enabled = bool(llm_cfg.get("enabled", False))
    llm_model = llm_cfg.get("model", "gpt-4o-mini")
    llm_temp = float(llm_cfg.get("temperature", 0.2))
    llm_max_ctx = int(llm_cfg.get("max_context_passages", 8))
    openai_client = None
    if llm_enabled:
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
            else:
                llm_enabled = False
        except Exception:
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
            qvec = emb.embed_texts([query])[0]
            results = q.search(qvec, top_k=top_k)
            # Attach text for reranking and display
            items = []
            for r in results:
                payload = r.get("payload", {})
                text = payload.get("text") or ""
                items.append({"text": text, **r})
            # Optional rerank
            if rr.enabled:
                items = rr.rerank(query, items, text_key="text")
            else:
                items = simple_lexical_rerank(query, items, text_key="text")
            answer = None
            if llm_enabled and openai_client:
                # Build compact context and ask the model
                ctx = []
                for it in items[:llm_max_ctx]:
                    p = it.get("payload", {})
                    text = (it.get("text") or p.get("text") or "")[:2000]
                    if not text:
                        continue
                    cite = _make_citation(p)
                    ctx.append({"text": text, "cite": cite})
                sys_prompt = (
                    "You are a precise legal assistant for Armenian law."
                    " Use only the provided passages. Quote verbatim Armenian when citing."
                )
                if l == "hy":
                    user_prompt = (
                        f"Հարցը: {query}\n\n"
                        "Ապավինիր միայն հետևյալ հատվածներին. Յուրաքանչյուր մեջբերման կողքին ավելացրու հղումը [Օրենք → Հոդված → Կետ → էջեր].\n"
                        "Պատասխանը ներկայացրու հայերենով։\n\n"
                        + "\n\n".join([f"- [{c['cite']}]\n{c['text']}" for c in ctx])
                    )
                else:
                    user_prompt = (
                        f"Question: {query}\n\n"
                        "Rely only on the passages below. Provide: (1) a brief English answer, (2) one short Armenian quote with an English gloss, (3) citations [Law → Article → Clause → pages].\n\n"
                        + "\n\n".join([f"- [{c['cite']}]\n{c['text']}" for c in ctx])
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
                answer = _compose_answer(query, l, items)
            print(answer)
        except Exception as e:
            print(f"Error: {e}")


def main():
    ap = argparse.ArgumentParser(description="Interactive CLI for Armenian Legal RAG")
    ap.add_argument("--cfg", default="cfg.yaml")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--lang", choices=["hy", "en"], default=None)
    args = ap.parse_args()
    run_interactive(args.cfg, args.top_k, args.lang)


if __name__ == "__main__":
    main()
