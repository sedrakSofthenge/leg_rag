from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI
from pydantic import BaseModel

from utils import Embeddings, QdrantConnector, Reranker, simple_lexical_rerank
from ingest import load_cfg, ingest_all


app = FastAPI(title="Armenian Legal RAG")


class AskRequest(BaseModel):
    question: str
    filters: Optional[Dict[str, Any]] = None
    answer_lang: Optional[str] = None  # "hy" or "en"
    top_k: Optional[int] = None


class PreviewRequest(BaseModel):
    question: str
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = None


CFG_PATH = os.environ.get("RAG_CFG", "cfg.yaml")
_cfg = load_cfg(CFG_PATH)

_emb = Embeddings(
    provider=_cfg.embeddings.get("provider", "openai"),
    model=_cfg.embeddings.get("model", "text-embedding-3-large"),
    dim=int(_cfg.embeddings.get("dim", 3072)),
    batch_size=int(_cfg.embeddings.get("batch_size", 64)),
    key_file=_cfg.embeddings.get("key_file"),
)

_q = QdrantConnector(
    collection=_cfg.qdrant.get("collection", "armenian_legal_chunks"),
    vector_size=int(_cfg.embeddings.get("dim", 3072)),
    distance=_cfg.qdrant.get("distance", "cosine"),
    mode=_cfg.qdrant.get("mode", "local"),
    url=_cfg.qdrant.get("url"),
    path=_cfg.qdrant.get("path", _cfg.paths.get("vdb_path")),
)

_reranker = Reranker(
    enabled=bool(_cfg.ingest.get("reranker_enabled", False)),
    provider="openai",
    key_file=_cfg.embeddings.get("key_file"),
)

_server_cfg = _cfg.server if hasattr(_cfg, "server") else {}
_llm_cfg = (_server_cfg or {}).get("llm_answer", {})
_llm_enabled = bool(_llm_cfg.get("enabled", False))
_llm_model = _llm_cfg.get("model", "gpt-4o-mini")
_llm_temp = float(_llm_cfg.get("temperature", 0.2))
_llm_max_ctx = int(_llm_cfg.get("max_context_passages", 8))

_openai_client = None
if _llm_enabled:
    try:
        from openai import OpenAI  # type: ignore
        # Reuse key from embeddings key_file if set
        key_file = _cfg.embeddings.get("key_file")
        key = None
        if key_file:
            try:
                with open(key_file, "r", encoding="utf-8") as f:
                    key = f.read().strip() or None
            except Exception:
                key = None
        key = key or os.environ.get("OPENAI_API_KEY")
        if key:
            _openai_client = OpenAI(api_key=key)
        else:
            _llm_enabled = False
    except Exception:
        _llm_enabled = False


def _is_armenian(text: str) -> bool:
    return any("\u0531" <= ch <= "\u058F" for ch in text)


def _make_citation(payload: Dict[str, Any]) -> str:
    law = payload.get("law_title") or payload.get("source_file")
    art = payload.get("article")
    cl = payload.get("clause")
    pages = payload.get("page_span") or payload.get("page_start"), payload.get("page_end")
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


def _generate_answer_offline(question: str, lang: str, items: List[Dict[str, Any]]) -> str:
    # Simple template answer with top 3 quotes + citations
    top = items[:3]
    if lang == "hy":
        lines = ["Պատասխան (նախնական, առանց LLM)"]
        for it in top:
            payload = it.get("payload", {})
            quote = (it.get("payload", {}).get("text") or it.get("text") or "")
            text = quote if quote else "…"
            lines.append(f"- Մեջբերում: \"{(text[:240] + '…') if len(text) > 240 else text}\"")
            lines.append(f"  Աղբյուր: [{_make_citation(payload)}]")
        return "\n".join(lines)
    else:
        lines = ["Answer (template, no LLM)"]
        for it in top:
            payload = it.get("payload", {})
            quote = (it.get("payload", {}).get("text") or it.get("text") or "")
            text = quote if quote else "…"
            lines.append(f"- Armenian quote: \"{(text[:240] + '…') if len(text) > 240 else text}\"")
            lines.append(f"  Citation: [{_make_citation(payload)}]")
        return "\n".join(lines)


def _attach_text_to_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Qdrant returns payload but not the text; we stored text in payload? We did not by default.
    # For preview, we include text from payload if present, else empty.
    out = []
    for r in results:
        payload = r.get("payload", {})
        text = payload.get("text") or payload.get("chunk_text") or ""
        out.append({"text": text, **r})
    return out


def _generate_answer_llm(question: str, lang: str, items: List[Dict[str, Any]]) -> Optional[str]:
    if not _openai_client:
        return None
    # Prepare compact context from top passages
    ctx = []
    for it in items[:_llm_max_ctx]:
        p = it.get("payload", {})
        text = p.get("text") or it.get("text") or ""
        cite = _make_citation(p)
        if text:
            ctx.append({"text": text[:2000], "cite": cite})
    sys_prompt = (
        "You are a precise legal assistant for Armenian law."
        " Use only the provided passages. Quote verbatim Armenian when citing."
    )
    if lang == "hy":
        user_prompt = (
            f"Հարցը: {question}\n\n"
            "Ապավինիր միայն հետևյալ հատվածներին. Յուրաքանչյուր մեջբերման կողքին ավելացրու հղումը [Օրենք → Հոդված → Կետ → էջեր].\n"
            "Պատասխանը ներկայացրու հայերենով։\n\n"
            + "\n\n".join([f"- [{c['cite']}]\n{c['text']}" for c in ctx])
        )
    else:
        user_prompt = (
            f"Question: {question}\n\n"
            "Rely only on the passages below. Provide: (1) a brief English answer, (2) one short Armenian quote with an English gloss, (3) citations [Law → Article → Clause → pages].\n\n"
            + "\n\n".join([f"- [{c['cite']}]\n{c['text']}" for c in ctx])
        )
    try:
        # Some models only accept default temperature; attempt graceful fallback
        kwargs = {"model": _llm_model, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]}
        if _llm_temp is not None and _llm_temp != 1:
            kwargs["temperature"] = _llm_temp
        try:
            resp = _openai_client.chat.completions.create(**kwargs)
        except Exception as e:
            msg = str(e)
            if "temperature" in msg and "Unsupported value" in msg:
                # Retry without temperature
                kwargs.pop("temperature", None)
                resp = _openai_client.chat.completions.create(**kwargs)
            else:
                raise
        return resp.choices[0].message.content
    except Exception:
        return None


@app.post("/ingest")
def api_ingest() -> Dict[str, Any]:
    ingest_all(_cfg.paths.get("data_raw", "data_raw"), CFG_PATH)
    return {"status": "ok"}


@app.post("/ask")
def api_ask(req: AskRequest) -> Dict[str, Any]:
    lang = req.answer_lang or ("hy" if _is_armenian(req.question) else "en")
    qvec = _emb.embed_texts([req.question])[0]
    top_k = int(req.top_k or _cfg.ingest.get("top_k", 40))
    raw = _q.search(qvec, top_k=top_k, filters=req.filters)

    # Rerank (optional)
    items = raw
    if _reranker.enabled:
        # prepare items with text in payload if available
        # Note: We didn't store text in payload to save space during upsert; for reranking offline,
        # we can't fetch the original text back from Qdrant without storing it. Ingest can be adjusted
        # to include a short text snippet if needed. Here we fallback to lexical rerank on available fields.
        items = _reranker.rerank(req.question, items, text_key="text")
    else:
        items = simple_lexical_rerank(req.question, items, text_key="text")

    # Compose answer
    answer = None
    if _llm_enabled:
        answer = _generate_answer_llm(req.question, lang, items)
    if not answer:
        answer = _generate_answer_offline(req.question, lang, items)
    return {
        "answer": answer,
        "answer_lang": lang,
        "top_k": top_k,
        "results": items,
    }


@app.post("/preview")
def api_preview(req: PreviewRequest) -> Dict[str, Any]:
    qvec = _emb.embed_texts([req.question])[0]
    top_k = int(req.top_k or _cfg.ingest.get("top_k", 40))
    raw = _q.search(qvec, top_k=top_k, filters=req.filters)
    return {"results": raw}
